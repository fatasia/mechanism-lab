import json
from zipfile import ZipFile

import numpy as np
import pytest

from phm_core.artifacts import new_run, publish_run, run_path
from phm_core.evaluation import evaluate
from phm_core.profiles import load_profile


def test_publication_is_atomic_and_old_version_survives(tmp_path):
    run, path = new_run(tmp_path, "valve")
    with pytest.raises(ValueError):
        publish_run(tmp_path, "valve", run)
    assert not (tmp_path / "valve/latest.json").exists()
    (path / "metrics.json").write_text("{}")
    (path / "model-bundle.zip").write_bytes(b"fixture")
    publish_run(tmp_path, "valve", run)
    second, other = new_run(tmp_path, "valve")
    assert run_path(tmp_path, "valve") == path
    with pytest.raises(FileNotFoundError):
        run_path(tmp_path, "valve", second)
    with pytest.raises(ValueError):
        run_path(tmp_path, "valve", "../../docs")
    assert run_path(tmp_path, "valve", run) == path
    assert other != path


def test_error_indices_and_threshold_ties():
    score = np.array([0.1, 0.5, 0.5, 0.9])
    healthy = np.array([True, True, False, False])
    result = evaluate(score, healthy, np.arange(4), 0.5)
    assert result["confusion"] == [[2, 0], [1, 1]]
    assert result["false_negative_ids"] == [2]
    assert result["false_positive_per_1000"] == 0
    assert result["recall"] == 0.5


def test_process_profile_does_not_upsample(tmp_path):
    archive = tmp_path / "source.zip"
    with ZipFile(archive, "w") as package:
        for name in ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"]:
            package.writestr(name + ".txt", "\t".join(["3"] * 6000) + "\n")
        for name in ["FS1", "FS2"]:
            package.writestr(name + ".txt", "\t".join(["2"] * 600) + "\n")
        package.writestr("profile.txt", "100 100 0 130 0\n100 100 0 130 0\n")
    # 矩阵必须至少两行，避免 loadtxt 折叠周期维度。
    with ZipFile(archive) as package:
        contents = {name: package.read(name) for name in package.namelist()}
    with ZipFile(archive, "w") as package:
        for name, value in contents.items():
            package.writestr(name, value if name == "profile.txt" else value * 2)
    x, labels, digest = load_profile(archive, "process-10hz", tmp_path / "cache")
    assert x.shape == (2, 600, 9)
    np.testing.assert_allclose(x[:, :, 0], 3, atol=1e-5)
    np.testing.assert_allclose(x[:, :, -1], 2)
    cached = load_profile(archive, "process-10hz", tmp_path / "cache")
    np.testing.assert_array_equal(cached[0], x)
    assert cached[2] == digest and labels.shape == (2, 5)


def test_api_version_and_failed_paths():
    from fastapi.testclient import TestClient

    from phm_demo.api import app

    client = TestClient(app)
    assert client.get("/api/replay/valve?run_id=missing-run").status_code == 404
    assert client.get("/api/replay/valve?run_id=..").status_code == 422
    assert (
        client.post("/api/train/valve", json={"profile": "unknown"}).status_code == 422
    )
    assert client.post("/api/train/valve", json={"epochs": 0}).status_code == 422
    assert client.post("/api/job/cancel").status_code == 422


def test_corrupted_bundle_is_rejected(tmp_path):
    from phm_core.inference import BundleScorer

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "models": {
                    "ae": {"shape": [None, 60, 5], "file": "ae.onnx", "sha256": "bad"}
                }
            }
        )
    )
    (tmp_path / "ae.onnx").write_bytes(b"not-a-model")
    with pytest.raises(ValueError, match="哈希"):
        BundleScorer(tmp_path)


def test_source_snapshot_excludes_non_source(tmp_path):
    from phm_core.artifacts import capture_source

    (tmp_path / "src/phm_core").mkdir(parents=True)
    for name in [
        "train_hydraulic.py",
        "pyproject.toml",
        "requirements.lock.txt",
        "src/phm_core/example.py",
    ]:
        (tmp_path / name).write_text("# fixture")
    (tmp_path / "private.pdf").write_bytes(b"not part of source")
    target = tmp_path / "source.zip"
    fingerprint = capture_source(tmp_path, target)
    with ZipFile(target) as archive:
        assert "private.pdf" not in archive.namelist()
        assert "source-manifest.json" in archive.namelist()
    assert len(fingerprint) == 64


def test_job_concurrency_and_cancel(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from phm_demo import api
    from phm_runtime.jobs import JobStore
    from phm_runtime.store import StateStore

    store = StateStore(tmp_path / "runtime.db")
    monkeypatch.setattr(api, "ROOT", tmp_path)
    monkeypatch.setattr(api, "state_store", lambda: store)
    monkeypatch.setattr(api.subprocess, "Popen", lambda *args, **kwargs: None)
    client = TestClient(api.app)
    response = client.post("/api/train/valve", json={"profile": "process-10hz"})
    assert response.status_code == 202
    assert client.post("/api/train/pump").status_code == 409
    assert (
        client.post(
            "/api/job/cancel", json={"job_id": response.json()["id"]}
        ).status_code
        == 200
    )
    jobs = JobStore(store)
    job_id = response.json()["id"]
    assert jobs.latest()["cancel"] == 1
    jobs.claim(job_id)
    assert not jobs.heartbeat(job_id)
    jobs.finish(job_id, "cancelled")
    assert client.get("/api/job").json()["status"] == "cancelled"
