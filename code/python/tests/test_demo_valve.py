import json

import numpy as np
from fastapi.testclient import TestClient

from phm_core.artifacts import new_run, publish_run
from phm_core.valve import fit_mechanism_reference, mechanism_features
from phm_demo import api


def test_cycle_exposes_independent_mechanism_evidence(monkeypatch, tmp_path):
    time = 8 + np.arange(400) / 100
    response = np.clip((time - 9.2) / 0.2, 0, 1)
    normal = np.stack([190 - 50 * response, 120 * response], axis=-1)
    delayed = np.roll(normal, 10, axis=0)
    delayed[:10] = normal[0]
    data = np.stack([normal, delayed]).astype("float32")
    reference = fit_mechanism_reference(mechanism_features(data[:1]), np.array([0]))
    reference["threshold"] = 2.0
    (tmp_path / "mechanism-reference.json").write_text(json.dumps(reference))
    monkeypatch.setattr(api, "artifact", lambda *args: tmp_path / "metrics.json")
    monkeypatch.setattr(
        api,
        "replay_data",
        lambda _: {
            "x": data,
            "reconstruction": data,
            "healthy": np.ones(2, dtype=bool),
        },
    )
    client = TestClient(api.app)
    baseline = client.get("/api/cycle/valve/0").json()
    result = client.get("/api/cycle/valve/1").json()
    assert not baseline["mechanism"]["flagged"]
    assert result["mechanism"]["flagged"] and result["healthy"]
    assert len(result["actual"]) == 400 and len(result["mechanism"]["items"]) == 9
    assert result["mechanism"]["items"][0]["lower"] is None
    assert result["mechanism"]["items"][6]["lower"] is not None
    assert all(item["unit"] == "s" for item in result["mechanism"]["items"])
    # 历史整周期版本没有机理参考文件，接口保持原有字段。
    (tmp_path / "mechanism-reference.json").unlink()
    assert "mechanism" not in client.get("/api/cycle/valve/0").json()


def test_valve_study_download_preserves_selected_version(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    client = TestClient(api.app)
    assert client.get("/api/valve-study").json() is None
    assert client.get("/api/valve-study/download?run_id=missing").status_code == 404
    versions = []
    for label in ("first", "second"):
        run_id, directory = new_run(tmp_path / "outputs", "valve-study")
        (directory / "metrics.json").write_text(json.dumps({"run_id": run_id}))
        (directory / "model-bundle.zip").write_bytes(label.encode())
        publish_run(tmp_path / "outputs", "valve-study", run_id)
        versions.append(run_id)
    assert client.get("/api/valve-study").json()["run_id"] == versions[-1]
    old = client.get(f"/api/valve-study/download?run_id={versions[0]}")
    assert old.status_code == 200 and old.content == b"first"
    assert client.get("/api/valve-study/download?run_id=..").status_code == 422


def test_valve_profile_is_not_used_for_other_components():
    client = TestClient(api.app)
    result = client.post(
        "/api/train/pump",
        json={"profile": "valve-switch-100hz", "architecture": "dense"},
    )
    assert result.status_code == 422
