import json
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from phm_demo.observability import install_observability
from phm_demo.security import install_access_control
from phm_runtime.audit import audit
from phm_runtime.backups import create_backup
from phm_runtime.observability import (
    RequestMetrics,
    json_event,
    redact,
    storage_snapshot,
)
from phm_runtime.store import StateStore


def test_redaction_and_audit(tmp_path):
    data = {
        "password": "fixture-sensitive",
        "nested": [{"api_key": "fixture-sensitive"}],
        "message": "Bearer fixture-sensitive token=fixture-sensitive",
        "ok": 4,
    }
    result = redact(data)
    assert "fixture-sensitive" not in json.dumps(result)
    assert result["ok"] == 4
    store = StateStore(tmp_path / "state.sqlite")
    with store.connect() as db:
        audit(db, "test", "asset", data)
        assert (
            "fixture-sensitive"
            not in db.execute("SELECT detail FROM audit").fetchone()[0]
        )
    assert json.loads(json_event("test", **data))["event"] == "test"


def test_access_logs_exclude_raw_path_query_body_and_headers(caplog):
    app = FastAPI()
    install_observability(app)

    @app.post("/api/entities/{entity}")
    def save(entity: str):
        return {"ok": True}

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        response = TestClient(app).post(
            "/api/entities/private-asset?token=private-query",
            headers={"Authorization": "Bearer private-header"},
            json={"password": "private-body"},
        )
    assert response.status_code == 200 and response.headers["X-Request-ID"]
    records = [
        json.loads(r.message) for r in caplog.records if r.name == "uvicorn.error"
    ]
    assert records[-1]["route"] == "/api/entities/{entity}"
    assert not any(
        word in json.dumps(records)
        for word in ("private-asset", "private-query", "private-header", "private-body")
    )
    assert app.state.request_metrics.snapshot()["requests"] == 1


def test_metrics_are_bounded():
    metrics = RequestMetrics(window=3)
    for i in range(20):
        metrics.record(200, i)
    value = metrics.snapshot()
    assert value["requests"] == 20 and value["latency_window"] == 3
    assert value["status_counts"] == {200: 20}


def test_monitor_is_admin_only(monkeypatch, tmp_path):
    from phm_demo.runtime_api import router
    from phm_demo import runtime_api

    monkeypatch.setenv("MECHANISM_MODE", "service")
    for role in ("ADMIN", "READER"):
        monkeypatch.setenv(f"MECHANISM_{role}_USER", role.lower())
        monkeypatch.setenv(f"MECHANISM_{role}_PASSWORD", "fixture-only")
    store = StateStore(tmp_path / "state.sqlite")
    monkeypatch.setattr(runtime_api, "state_store", lambda: store)
    app = FastAPI()
    install_access_control(app)
    install_observability(app)
    app.include_router(router)
    client = TestClient(app)
    assert client.get("/api/runtime/monitor").status_code == 401
    assert (
        client.get("/api/runtime/monitor", auth=("reader", "fixture-only")).status_code
        == 403
    )
    response = client.get("/api/runtime/monitor", auth=("admin", "fixture-only"))
    assert (
        response.status_code == 200 and response.json()["storage"]["database_bytes"] > 0
    )


def test_repeated_backups_are_standalone_and_restore(tmp_path):
    store = StateStore(tmp_path / "source.sqlite")
    destination = tmp_path / "restore.sqlite"
    for i in range(12):
        store.bind("valve", f"v{i}", "ae", "shadow")
        store.backup(destination)
        with sqlite3.connect(destination) as db:
            assert db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
            assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert db.execute("SELECT run_id FROM bindings").fetchone()[0] == f"v{i}"
        assert not Path(str(destination) + "-wal").exists()
    with pytest.raises(ValueError):
        store.backup(store.path)
    assert storage_snapshot(store.path)["disk_free_bytes"] > 0


def test_backup_retention_and_cli(tmp_path):
    store = StateStore(tmp_path / "source.sqlite")
    store.bind("valve", "v1", "ae", "shadow")
    directory = tmp_path / "backups"
    directory.mkdir()
    unrelated = directory / "manual.sqlite"
    unrelated.write_bytes(b"untouched")
    for _ in range(4):
        result = create_backup(store, directory, keep=2)
    assert len(list(directory.glob("runtime-*.sqlite"))) == 2
    assert unrelated.read_bytes() == b"untouched"
    assert result["integrity"] == "ok"
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "runtime_ops.py"),
        "--database",
        str(store.path),
        "--backup-dir",
        str(directory),
        "--keep",
        "2",
    ]
    process = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["event"] == "backup_complete"
