from concurrent.futures import ThreadPoolExecutor

import pytest

from phm_core.decision import DecisionConfig
from phm_runtime.registry import promotion_gate
from phm_runtime.store import StateStore


def test_alarm_state_survives_restart_and_duplicate(tmp_path):
    path = tmp_path / "runtime.db"
    config = DecisionConfig(1, 0.8)
    for sequence in range(3):
        result = StateStore(path).consume(
            "asset/component", "v1", sequence, 2, config, {}, {}
        )
    assert result["alarm"] and result["transition"] == "opened"
    store = StateStore(path)
    repeated = store.consume("asset/component", "v1", 2, 2, config, {}, {})
    assert repeated["duplicate"] and len(store.events()) == 1
    with pytest.raises(ValueError, match="乱序"):
        store.consume("asset/component", "v1", -1, 2, config, {}, {})
    for sequence in range(3, 6):
        result = store.consume("asset/component", "v1", sequence, 0, config, {}, {})
    assert not result["alarm"] and store.events()[0]["reason"] == "recovery"


def test_entities_and_versions_do_not_share_state(tmp_path):
    store = StateStore(tmp_path / "runtime.db")
    config = DecisionConfig(1, 0.8)
    for sequence in range(3):
        store.consume("a", "v1", sequence, 2, config, {}, {})
    assert not store.consume("b", "v1", 0, 2, config, {}, {})["alarm"]
    assert not store.consume("a", "v2", 0, 2, config, {}, {})["alarm"]
    assert (
        store.consume("a", "v1", 6, None, config, {"status": "invalid"}, {})["status"]
        == "unscored"
    )
    assert store.events()[0]["reason"] == "data-gap"


def test_concurrent_retries_are_idempotent_and_backup_restores(tmp_path):
    store = StateStore(tmp_path / "runtime.db")
    config = DecisionConfig(1, 0.8, required=1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda _: store.consume("a", "v1", 0, 2, config, {}, {}), range(8))
        )
    assert sum(not r["duplicate"] for r in results) == 1
    assert len(store.events()) == 1
    store.feedback(store.events()[0]["id"], "tester", "unconfirmed")
    backup = tmp_path / "backup.db"
    store.backup(backup)
    assert StateStore(backup).events()[0]["operator"] == "tester"


def test_failed_quality_gate_is_not_promoted():
    report = {
        "models": {
            "ae": {
                "confusion": [[112, 0], [12, 140]],
                "recall": 140 / 152,
                "onnx_max_abs_error": 1e-7,
            }
        }
    }
    gate = promotion_gate(report, "ae")
    assert not gate["passed"] and "95%" in gate["reasons"][0]


def test_research_candidate_never_passes_active_gate():
    report = {
        "research_only": True,
        "models": {
            "ae": {
                "confusion": [[112, 0], [0, 152]],
                "recall": 1.0,
                "onnx_max_abs_error": 1e-7,
            }
        },
    }
    gate = promotion_gate(report, "ae")
    assert not gate["passed"] and "研究候选" in gate["reasons"][0]
    with pytest.raises(ValueError, match="不包含"):
        promotion_gate(report, "rf")


def test_duplicate_sequence_with_changed_payload_is_rejected(tmp_path):
    store = StateStore(tmp_path / "runtime.db")
    store.consume("a", "v1", 0, 2, DecisionConfig(1, 0.8), {}, {"input_sha256": "one"})
    with pytest.raises(ValueError, match="输入数据不同"):
        store.consume(
            "a", "v1", 0, 2, DecisionConfig(1, 0.8), {}, {"input_sha256": "two"}
        )


def test_model_change_closes_old_events_and_rollback_restarts_state(tmp_path):
    store = StateStore(tmp_path / "runtime.db")
    config = DecisionConfig(1, 0.8)
    store.bind("valve", "v1", "ae", "shadow")
    for sequence in range(3):
        store.consume("a", "valve/v1/ae", sequence, 2, config, {}, {})
    store.bind("valve", "v2", "ae", "shadow")
    assert store.events()[0]["reason"] == "model-change"
    store.bind("valve", "v1", "ae", "shadow")
    result = store.consume("a", "valve/v1/ae", 3, 2, config, {}, {})
    assert not result["alarm"]
