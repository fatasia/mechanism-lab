import numpy as np
import pytest

from phm_core.actuator import fit_phase_references, motion_features, split_records
from phm_core.monitoring import (
    TrendState,
    calibrated_threshold,
    compare_peers,
    distribution_drift,
    tail_probability,
    trend_step,
)
from phm_core.quality import assess_window


def test_actuator_partitions_keep_held_condition_out():
    metadata = [
        {"id": i, "load": load, "profile": profile, "repeat": repeat}
        for i, (load, profile, repeat) in enumerate(
            [
                ("20", "trap", 1),
                ("40", "trap", 7),
                ("20", "trap", 9),
                ("neg40", "trap", 1),
                ("20", "sin", 1),
            ]
        )
    ]
    split = split_records(metadata)
    assert split["test"].tolist() == [3]
    assert not set(split["train"]) & set(split["test"])
    assert split_records(metadata, "held-out-profile")["test"].tolist() == [4]


def test_motion_features_have_physical_error_and_current_units():
    setpoint = np.r_[
        np.zeros(100),
        np.linspace(0, 10, 100),
        np.ones(100) * 10,
        np.linspace(10, 0, 100),
    ]
    x = np.stack([setpoint, np.ones(400) * 2, np.ones(400) * 3], axis=1)[None].astype(
        "float32"
    )
    features = motion_features(x)
    assert features.shape == (1, 20)
    np.testing.assert_allclose(features[0, :4], [2, 2, 3, 3])
    with pytest.raises(ValueError):
        motion_features(np.zeros((1, 400, 3)))


def test_commissioning_reference_rejects_fault_samples():
    with pytest.raises(ValueError, match="健康"):
        fit_phase_references(np.zeros((1, 400, 3)), [0], [{"id": 0, "label": 1}], [0])


def test_finite_sample_calibration_and_ties():
    assert calibrated_threshold(np.arange(99), 0.01) == 98
    with pytest.raises(ValueError):
        calibrated_threshold(np.arange(20), 0.01)
    np.testing.assert_allclose(
        tail_probability([0, 4, 5], np.arange(5)), [1, 2 / 6, 1 / 6]
    )


def test_peer_reference_can_find_bad_initial_baseline():
    peers = np.array([[1], [1.01], [1.02], [0.99], [0.98]])
    result = compare_peers(np.array([1.5]), peers, 0.01)
    assert result["score"] > 10
    assert (
        compare_peers(np.array([1.5]), peers[:2], 0.01)["status"]
        == "insufficient_peers"
    )


def test_drift_and_fixed_reference_trend():
    reference = np.random.default_rng(42).normal(size=(200, 2))
    drift = distribution_drift(reference, reference + [2, 0])
    assert drift[0]["review"] and not drift[1]["review"]
    state = TrendState()
    for _ in range(20):
        state = trend_step(2, state, 0, 1)
    assert state.ewma > 1.5 and state.cusum == 30


def test_bad_input_and_missing_timestamps_do_not_get_normal_score():
    x = np.ones((10, 2))
    assert assess_window(x, (10, 2), 10, 10)["status"] == "valid"
    x[0, 0] = np.nan
    result = assess_window(x, (10, 2), 1, 10, np.arange(10), [(0, 2), (0, 2)])
    assert {item["code"] for item in result["issues"]} == {
        "sample_rate",
        "missing_or_nonfinite",
        "timestamp_gap_or_order",
    }
