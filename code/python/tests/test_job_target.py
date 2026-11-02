from fastapi.testclient import TestClient

from phm_demo import api
from phm_runtime.jobs import JobStore
from phm_runtime.store import StateStore


def test_old_task_cancel_does_not_cancel_new_task(monkeypatch, tmp_path):
    store = StateStore(tmp_path / "runtime.sqlite")
    monkeypatch.setattr(api, "state_store", lambda: store)
    jobs = JobStore(store)
    old = jobs.create("valve", [], tmp_path / "old.log")
    jobs.finish(old, "complete")
    current = jobs.create("pump", [], tmp_path / "current.log")
    client = TestClient(api.app)
    assert client.post("/api/job/cancel", json={"job_id": old}).status_code == 409
    assert jobs.latest()["cancel"] == 0
    assert client.post("/api/job/cancel").status_code == 422
    assert client.post("/api/job/cancel", json={"job_id": current}).status_code == 200
    assert jobs.latest()["cancel"] == 1
    assert client.get("/api/job").json()["cancel_requested"]


def test_duplicate_cancel_has_one_audit_entry(tmp_path):
    store = StateStore(tmp_path / "runtime.sqlite")
    jobs = JobStore(store)
    job_id = jobs.create("valve", [], tmp_path / "log")
    jobs.cancel(job_id)
    jobs.cancel(job_id)
    with store.connect() as db:
        assert (
            db.execute(
                "SELECT count(*) FROM audit WHERE action='job_cancel_requested'"
            ).fetchone()[0]
            == 1
        )
