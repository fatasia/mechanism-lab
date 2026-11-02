from typing import ClassVar
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from phm_core.decision import DecisionConfig
from phm_demo import maintenance_api, runtime_api
from phm_demo.api import app
from phm_runtime.maintenance import MaintenanceStore
from phm_runtime.registry import ModelRegistry
from phm_runtime.store import StateStore


def test_stock_api_requires_idempotency_key(monkeypatch, tmp_path):
    store = MaintenanceStore(StateStore(tmp_path / "runtime.db"))
    monkeypatch.setattr(maintenance_api, "maintenance_store", lambda: store)
    client = TestClient(app)
    assert (
        client.put(
            "/api/maintenance/assets",
            json={"id": "a", "name": "Test asset", "kind": "component"},
        ).status_code
        == 200
    )
    assert (
        client.put(
            "/api/maintenance/parts", json={"id": "p", "name": "Test part"}
        ).status_code
        == 200
    )
    payload = {"delta": 5, "reason": "receive", "operator": "test"}
    assert (
        client.post("/api/maintenance/parts/p/stock", json=payload).status_code == 422
    )
    payload["request_id"] = str(uuid4())
    assert not client.post("/api/maintenance/parts/p/stock", json=payload).json()[
        "duplicate"
    ]
    assert client.post("/api/maintenance/parts/p/stock", json=payload).json()[
        "duplicate"
    ]
    payload["delta"] = 10
    assert (
        client.post("/api/maintenance/parts/p/stock", json=payload).status_code == 409
    )
    assert store.snapshot()["parts"][0]["quantity"] == 5


def test_peer_api_rejects_self_and_mismatched_cohort():
    client = TestClient(app)
    payload = {
        "target_entity": "a",
        "cohort": "same-profile",
        "values": [1.5],
        "scale_floor": 0.01,
        "peers": [
            {"entity": str(i), "cohort": "same-profile", "values": [1]}
            for i in range(5)
        ],
    }
    assert client.post("/api/runtime/peers", json=payload).json()["score"] == 50
    payload["peers"][0]["entity"] = "a"
    assert client.post("/api/runtime/peers", json=payload).status_code == 422
    payload["peers"][0] = {"entity": "x", "cohort": "different", "values": [1]}
    assert client.post("/api/runtime/peers", json=payload).status_code == 422


def test_missing_window_is_unscored_and_order_is_enforced(monkeypatch, tmp_path):
    store = StateStore(tmp_path / "runtime.db")
    registry = ModelRegistry(tmp_path, store)

    class FixtureScorer:
        manifest: ClassVar[dict] = {
            "channels": ["x"],
            "units": ["bar"],
            "sampleRateHz": 10,
            "models": {"ae": {"shape": [None, 4, 1], "threshold": 1.0}},
        }

        def score(self, values, model):
            return values.mean((1, 2))

    registry.loaded[("valve", "fixture")] = (FixtureScorer(), {})
    store.bind("valve", "fixture", "ae", "shadow")
    monkeypatch.setattr(runtime_api, "model_registry", lambda: registry)
    client = TestClient(app)
    payload = {
        "task": "valve",
        "entity": "fixture",
        "sequence": 0,
        "values": [[None], [1], [1], [1]],
        "channels": ["x"],
        "units": ["bar"],
        "sample_rate_hz": 10,
    }
    result = client.post("/api/runtime/predict", json=payload).json()
    assert result["status"] == "unscored" and result["score"] is None
    payload.update(sequence=2, values=[[2]] * 4)
    assert client.post("/api/runtime/predict", json=payload).json()["gap"]
    payload["sequence"] = 1
    assert client.post("/api/runtime/predict", json=payload).status_code == 409


def test_stale_binding_does_not_mutate_state(tmp_path):
    store = StateStore(tmp_path / "runtime.db")
    store.bind("valve", "v2", "ae", "shadow")
    with pytest.raises(ValueError, match="绑定"):
        store.consume(
            "entity", "valve/v1/ae", 1, 2, DecisionConfig(1, 0.8), {}, {"task": "valve"}
        )
    assert store.history("entity") == []
