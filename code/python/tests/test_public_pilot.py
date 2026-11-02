import importlib.util
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from phm_core.decision import DecisionConfig
from phm_runtime.maintenance import MaintenanceStore
from phm_runtime.store import StateStore
from phm_runtime.streaming import WindowQueue


def script(name, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(name, root / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def window(sequence=0):
    return {
        "entity": "asset",
        "task": "valve",
        "sequence": sequence,
        "values": [[1.0, 2.0]],
        "channels": ["PS1", "PS2"],
        "units": ["bar", "bar"],
        "sample_rate_hz": 100,
    }


def test_queue_lost_reply_restart_and_duplicate(tmp_path):
    queue = WindowQueue(tmp_path / "queue.db")
    store = StateStore(tmp_path / "runtime.db")
    queue.put(window())
    config = DecisionConfig(1, 0.8, required=1)

    def send(body):
        return store.consume(body["entity"], "v1", body["sequence"], 2, config, {}, {})

    def lost(body):
        send(body)
        raise ConnectionError("response lost")

    with pytest.raises(ConnectionError):
        queue.drain_one(lost)
    assert queue.snapshot()["pending"] == 1
    restarted = WindowQueue(queue.path)
    assert restarted.drain_one(send)["duplicate"]
    assert restarted.snapshot()["pending"] == 0
    assert len(store.events()) == 1
    restarted.put(window())
    assert restarted.snapshot()["acknowledged"] == 1


def test_queue_conflicts_backpressure_and_order(tmp_path):
    q = WindowQueue(tmp_path / "q.db", capacity=1)
    q.put(window(2))
    with pytest.raises(ValueError, match="冲突"):
        q.put({**window(2), "values": [[3, 4]]})
    with pytest.raises(ValueError, match="乱序"):
        q.put(window(1))
    with pytest.raises(BufferError):
        q.put(window(3))
    with pytest.raises(ValueError):
        q.drain_one(lambda b: {"status": "unknown"})
    assert q.snapshot()["pending"] == 1


def test_transaction_failure_rolls_back_alarm(tmp_path, monkeypatch):
    store = StateStore(tmp_path / "r.db")
    original = store.connect

    @contextmanager
    def failing():
        with original() as db:
            db.set_authorizer(
                lambda action, a, b, c, d: (
                    sqlite3.SQLITE_DENY
                    if action == sqlite3.SQLITE_INSERT and a == "predictions"
                    else sqlite3.SQLITE_OK
                )
            )
            yield db

    monkeypatch.setattr(store, "connect", failing)
    with pytest.raises(sqlite3.DatabaseError):
        store.consume("a", "v1", 0, 2, DecisionConfig(1, 0.8, required=1), {}, {})
    monkeypatch.setattr(store, "connect", original)
    assert store.events() == []
    assert store.history("a") == []


def test_sqlite_full_is_explicit_and_transactional(tmp_path):
    with sqlite3.connect(tmp_path / "full.db") as db:
        db.execute("CREATE TABLE sample(data BLOB)")
        count = db.execute("PRAGMA page_count").fetchone()[0]
        db.execute(f"PRAGMA max_page_count={count}")
        with pytest.raises(sqlite3.OperationalError, match="full"):
            with db:
                db.execute("INSERT INTO sample VALUES(?)", (bytes(100000),))
        assert db.execute("SELECT count(*) FROM sample").fetchone()[0] == 0


def test_maintenance_evidence_does_not_create_training_truth(tmp_path):
    store = StateStore(tmp_path / "runtime.db")
    m = MaintenanceStore(store)
    m.asset("asset", "Test", "component", None, {})
    for i in range(3):
        store.consume(
            "asset",
            "valve/v1/ae",
            i,
            2,
            DecisionConfig(1, 0.8, required=1),
            {},
            {"input_sha256": "public-test-hash"},
        )
    order = m.create_order(
        "asset", "Inspect", "tester", event_id=store.events()[0]["id"]
    )
    m.transition(order, "working", 1, "tester")
    m.transition(order, "completed", 2, "tester", notes="Inspected; cause unknown")
    m.transition(order, "closed", 3, "reviewer", verification="No confirmed fault")
    evidence = m.evidence(order)
    assert evidence["asset_event_identity_matches"]
    assert evidence["prediction_total"] == 3
    assert evidence["training_label_status"].startswith("not_exported")
    assert evidence["event"]["version"] == "valve/v1/ae"
    assert evidence["order"]["cause"] == ""
    json.dumps(evidence, allow_nan=False)


def test_holdout_excludes_seen_and_duplicate_assets(monkeypatch):
    module = script("prepare_care_holdout", monkeypatch)
    rows = [
        {
            "farm": "A",
            "asset_id": str(i // 2),
            "event_id": str(i),
            "event_label": "anomaly" if i < 4 else "normal",
        }
        for i in range(8)
    ]
    selected = module.select(rows, {("A", "0")})
    assert len({e["asset_id"] for e in selected}) == len(selected)
    assert all(e["asset_id"] != "0" for e in selected)


def test_cusum_is_causal_and_resets_on_missing(monkeypatch):
    module = script("study_care_holdout", monkeypatch)
    x = np.array([2.0, 2.0, np.nan, 2.0, 2.0])
    times = np.arange(5) / 6
    a = module.cusum(times, x, 1)
    b = module.cusum(times, x + np.array([0, 0, 0, 99, 99]), 1)
    np.testing.assert_allclose(a[:3], b[:3], equal_nan=True)
    assert a[3] == 1


def test_group_conformal_insufficient_groups_is_unbounded(monkeypatch):
    module = script("study_electrode_uncertainty", monkeypatch)
    assert module.group_radius(np.array([1, 2]), np.array(["a", "b"]), 0.2) is None
    assert module.group_radius(np.arange(6.0), np.arange(6), 0.2) == 5


def test_asset_acl_denies_global_and_other_entities(monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from phm_demo.security import install_access_control

    monkeypatch.setenv("MECHANISM_MODE", "service")
    for role in ("ADMIN", "OPERATOR", "READER"):
        monkeypatch.setenv(f"MECHANISM_{role}_USER", role.lower())
        monkeypatch.setenv(f"MECHANISM_{role}_PASSWORD", "test-only")
    monkeypatch.setenv(
        "MECHANISM_ASSET_ACL", json.dumps({"operator": ["a"], "reader": ["a"]})
    )
    app = FastAPI()
    install_access_control(app)

    @app.post("/api/runtime/predict")
    async def predict(request: Request):
        return await request.json()

    @app.get("/api/runtime/history")
    def history(entity: str):
        return {"entity": entity}

    client = TestClient(app)
    auth = ("operator", "test-only")
    assert client.post(
        "/api/runtime/predict", json={"entity": "a"}, auth=auth
    ).json() == {"entity": "a"}
    assert (
        client.post("/api/runtime/predict", json={"entity": "b"}, auth=auth).status_code
        == 403
    )
    assert client.get("/api/runtime/events", auth=auth).status_code == 403
    assert client.get("/api/runtime/history?entity=b", auth=auth).status_code == 403
    assert (
        client.get(
            "/api/runtime/history?entity=a", auth=("reader", "test-only")
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/runtime/predict", json={"entity": "a"}, auth=("reader", "test-only")
        ).status_code
        == 403
    )


def test_trust_rotation_accepts_overlap_and_rejects_revoked(tmp_path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    from phm_core.signing import sign_directory, verify_trust_set

    keys = []
    for i in range(2):
        key = Ed25519PrivateKey.generate()
        private = tmp_path / f"test-{i}.pem"
        public = tmp_path / f"test-{i}.pub"
        private.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.BestAvailableEncryption(b"test-only"),
            )
        )
        public.write_bytes(
            key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        keys.append((private, public))
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "data").write_text("public-test")
    sign_directory(artifact, keys[0][0], b"test-only")
    verify_trust_set(artifact, [str(p) for _, p in keys])
    with pytest.raises(ValueError, match="信任"):
        verify_trust_set(artifact, [str(keys[1][1])])
    with pytest.raises(ValueError):
        verify_trust_set(artifact, [])


def test_public_search_is_whitelisted_and_cited(tmp_path):
    from phm_runtime.knowledge import search

    (tmp_path / "OPERATIONS.md").write_text(
        "# 备份恢复\n签名验证与备份恢复。", encoding="utf-8"
    )
    (tmp_path / "private.txt").write_text("备份恢复 private-canary", encoding="utf-8")
    result = search(tmp_path, "备份恢复")
    assert result and result[0]["line_start"] == 1
    assert "private-canary" not in json.dumps(result)
    assert result[0]["source_url"].startswith("https://github.com/fatasia/")
    assert search(tmp_path, "completelyunrelated") == []


def test_resource_sampling_current_process():
    import os
    from phm_runtime.resources import resident_bytes

    value = resident_bytes(os.getpid())
    assert value is None or value > 0
