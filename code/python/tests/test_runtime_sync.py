from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from phm_demo import runtime_api
from phm_demo.api import app
from phm_runtime.registry import ModelRegistry, apply_sync_requirement
from phm_runtime.store import StateStore


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    store = StateStore(tmp_path / "runtime.sqlite")
    store.bind("valve", "fixture", "ae", "shadow")
    calls = []
    manifest = {
        "channels": ["PS1", "PS2"],
        "units": ["bar", "bar"],
        "sampleRateHz": 100,
        "models": {"ae": {"shape": [None, 4, 2], "threshold": 1.0}},
        "sync": {"required": True, "expected_lead_s": 8.0, "tolerance_samples": 1},
    }

    def score(values, model):
        calls.append(values)
        return np.array([2.0])

    registry = ModelRegistry(tmp_path, store)
    registry.loaded[("valve", "fixture")] = (
        SimpleNamespace(manifest=manifest, score=score),
        {},
    )
    monkeypatch.setattr(runtime_api, "model_registry", lambda: registry)
    client = TestClient(app)
    body = {
        "task": "valve",
        "entity": "fixture",
        "sequence": 0,
        "values": [[2.0, 2.0]] * 4,
        "channels": ["PS1", "PS2"],
        "units": ["bar", "bar"],
        "sample_rate_hz": 100,
    }
    return client, body, calls, store, manifest


@pytest.mark.parametrize(
    "sync",
    [
        None,
        {},
        {"command_edge_s": 0, "window_start_s": 8},
        {"command_edge_s": 0, "window_start_s": 8.02, "channel_skew_s": 0},
        {"command_edge_s": 0, "window_start_s": 8, "channel_skew_s": 0.02},
    ],
)
def test_http_unsupported_never_scores_and_is_persisted(runtime, sync):
    client, body, calls, store, _ = runtime
    response = client.post("/api/runtime/predict", json={**body, "sync": sync})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "unscored" and result["score"] is None
    assert result["quality"]["status"] == "unsupported" and not result["alarm"]
    assert not calls and not store.events()
    assert store.history("fixture")[0]["quality"] == result["quality"]


@pytest.mark.parametrize("offset", [-0.01, 0, 0.01])
def test_http_sync_tolerance_and_idempotency(runtime, offset):
    client, body, calls, _, _ = runtime
    sync = {"command_edge_s": 0, "window_start_s": 8 + offset, "channel_skew_s": 0}
    response = client.post("/api/runtime/predict", json={**body, "sync": sync})
    assert response.status_code == 200 and response.json()["status"] == "scored"
    assert len(calls) == 1
    assert client.post("/api/runtime/predict", json={**body, "sync": sync}).json()[
        "duplicate"
    ]
    changed = {**sync, "channel_skew_s": 0.005}
    assert (
        client.post("/api/runtime/predict", json={**body, "sync": changed}).status_code
        == 409
    )


def test_request_cannot_override_model_lead(runtime):
    client, body, _, _, _ = runtime
    sync = {
        "command_edge_s": 0,
        "window_start_s": 8,
        "channel_skew_s": 0,
        "expected_lead_s": 8,
    }
    assert (
        client.post("/api/runtime/predict", json={**body, "sync": sync}).status_code
        == 422
    )
    sync.pop("expected_lead_s")
    sync["window_start_s"] = "NaN"
    assert (
        client.post("/api/runtime/predict", json={**body, "sync": sync}).status_code
        == 422
    )


def test_legacy_bundle_remains_compatible(runtime):
    client, body, calls, _, manifest = runtime
    del manifest["sync"]
    assert client.post("/api/runtime/predict", json=body).json()["status"] == "scored"
    assert calls


def test_invalid_sync_contract_fails_closed():
    with pytest.raises(ValueError, match="同步契约"):
        apply_sync_requirement(
            {"status": "valid", "issues": []},
            {"sampleRateHz": 100, "sync": {"required": True}},
            100,
            {},
        )


def test_registry_runtime_partition_does_not_need_training_metadata(
    tmp_path, monkeypatch
):
    import json

    directory = tmp_path / "runtime-release"
    (directory / "bundle").mkdir(parents=True)
    manifest = {"bundleKind": "valve-fusion-runtime", "models": {"ae": {}}}
    (directory / "bundle/manifest.json").write_text(json.dumps(manifest))
    (directory / "metrics.json").write_text('{"research_only": true}')
    np.savez(directory / "parity.npz", raw=np.ones((7, 4, 2)), ae=np.ones(7))
    calls = []

    def path(outputs, task, run_id):
        calls.append(task)
        if task == "valve":
            raise FileNotFoundError
        return directory

    monkeypatch.setattr("phm_runtime.registry.run_path", path)
    monkeypatch.setattr(
        "phm_runtime.registry.BundleScorer",
        lambda p: SimpleNamespace(
            manifest=manifest, score=lambda x, model: np.ones(len(x))
        ),
    )
    registry = ModelRegistry(tmp_path, StateStore(tmp_path / "state.sqlite"))
    assert registry.load("valve", "fixture")[1]["research_only"]
    assert calls == ["valve", "valve-runtime"]
