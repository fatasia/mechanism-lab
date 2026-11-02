import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from phm_core.decision import DecisionConfig
from phm_runtime.jobs import JobStore
from phm_runtime.outbox import Outbox
from phm_runtime.store import StateStore


def alarm(store):
    return store.consume("a", "v1", 0, 2, DecisionConfig(1, 0.8, required=1), {}, {})


def test_notification_atomic_idempotent_and_restart(tmp_path):
    path = tmp_path / "state.db"
    store = StateStore(path)
    alarm(store)
    alarm(store)
    queue = Outbox(StateStore(path))
    assert queue.snapshot()["counts"] == {"pending": 1}
    with ThreadPoolExecutor(max_workers=4) as executor:
        claims = list(executor.map(lambda _: queue.claim(now=100), range(4)))
    item = next(x for x in claims if x)
    assert sum(x is not None for x in claims) == 1
    assert len(item["payload"]["notification_key"]) == 32
    expired = queue.claim(now=161)
    assert expired["payload"]["notification_key"] == item["payload"]["notification_key"]
    with pytest.raises(ValueError):
        queue.finish(item["id"], item["token"], True, now=162)
    queue.finish(expired["id"], expired["token"], True, now=162)
    assert queue.snapshot()["counts"] == {"delivered": 1}


def test_transaction_failure_does_not_leave_notification(tmp_path):
    store = StateStore(tmp_path / "state.db")
    # 事件和队列插入之后，预测JSON失败应整体回滚。
    with pytest.raises(ValueError):
        store.consume(
            "a",
            "v1",
            0,
            2,
            DecisionConfig(1, 0.8, required=1),
            {},
            {"invalid": float("nan")},
        )
    assert not store.events()
    assert Outbox(store).snapshot()["counts"] == {}


def test_retry_dead_letter_and_restore(tmp_path):
    store = StateStore(tmp_path / "state.db")
    alarm(store)
    queue = Outbox(store)
    for attempt in range(5):
        item = queue.claim(now=1000 * attempt)
        queue.finish(
            item["id"], item["token"], False, "transport_error", now=1000 * attempt + 1
        )
    assert queue.snapshot()["counts"] == {"dead": 1}
    store.backup(tmp_path / "backup.db")
    restored = Outbox(StateStore(tmp_path / "backup.db"))
    restored.retry(item["id"])
    assert restored.claim(now=10000)["attempts"] == 1


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("success", "complete"),
        ("cancel", "cancelled"),
        ("timeout", "timed_out"),
        ("failure", "failed"),
    ],
)
def test_worker_persists_after_controller_reconstruction(tmp_path, mode, expected):
    database = tmp_path / "state.db"
    jobs = JobStore(StateStore(database))
    command = [
        sys.executable,
        "-c",
        "import time;time.sleep(.5);print('ok')"
        if mode == "success"
        else "raise SystemExit(7)"
        if mode == "failure"
        else "import time;time.sleep(20)",
    ]
    job_id = jobs.create(
        "valve",
        command,
        tmp_path / "train.log",
        timeout=0.3 if mode == "timeout" else 60,
    )
    with pytest.raises(ValueError):
        jobs.create("pump", command, tmp_path / "other.log")
    if mode == "cancel":
        jobs.cancel(job_id)
    root = Path(__file__).resolve().parents[3]
    worker = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "phm_runtime.job_worker",
            "--database",
            str(database),
            "--job",
            job_id,
            "--root",
            str(root),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        out, err = worker.communicate(timeout=15)
        assert worker.returncode == 0, (out, err)
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()
    restored = JobStore(StateStore(database))
    assert restored.latest()["status"] == expected
    if mode == "success":
        assert "ok" in (tmp_path / "train.log").read_text()
    with pytest.raises(ValueError):
        restored.claim(job_id)


def test_stale_job_is_visible_not_silently_restarted(tmp_path):
    store = StateStore(tmp_path / "state.db")
    jobs = JobStore(store)
    jobs.create("valve", [], tmp_path / "train.log")
    with store.connect() as db:
        db.execute("UPDATE jobs SET heartbeat=?", (time.time() - 120,))
    assert jobs.latest()["stale"]
    with pytest.raises(ValueError):
        jobs.create("valve", [], tmp_path / "other.log")


def test_numpy_window_metadata_has_stable_fingerprint(tmp_path):
    from types import SimpleNamespace

    import numpy as np

    from phm_runtime.registry import ModelRegistry

    store = StateStore(tmp_path / "state.db")
    store.bind("valve", "fixture", "ae", "shadow")
    scorer = SimpleNamespace(
        manifest={
            "channels": ["x"],
            "units": ["bar"],
            "sampleRateHz": 10,
            "models": {"ae": {"shape": [None, 4, 1], "threshold": 1.0}},
        },
        score=lambda x, model: x.mean((1, 2)),
    )
    registry = ModelRegistry(tmp_path, store)
    registry.loaded[("valve", "fixture")] = (scorer, {})
    result = registry.predict(
        "valve", "a", 0, [[2]] * 4, ["x"], ["bar"], np.float32(10), np.arange(4) / 10
    )
    repeated = registry.predict(
        "valve", "a", 0, [[2]] * 4, ["x"], ["bar"], 10, [0, 0.1, 0.2, 0.3]
    )
    assert result["status"] == "scored" and repeated["duplicate"]


def test_real_webhook_retry_preserves_idempotency_key(tmp_path):
    import importlib.util
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    spec = importlib.util.spec_from_file_location(
        "deliver_notifications", Path(__file__).parents[1] / "deliver_notifications.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.headers["Idempotency-Key"], body))
            self.send_response(503 if len(received) == 1 else 204)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        store = StateStore(tmp_path / "state.db")
        alarm(store)
        queue = Outbox(store)
        url = f"http://127.0.0.1:{server.server_port}/events"
        assert module.deliver(queue, url) == "retry_pending_or_dead"
        with store.connect() as db:
            db.execute("UPDATE outbox SET available=0")
        assert module.deliver(queue, url) == "delivered"
        assert received[0] == received[1]
        assert queue.snapshot()["counts"] == {"delivered": 1}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
