import numpy as np
import pytest
import torch

from phm_core.deep import DenseAutoencoder
from phm_core.learning import fit_autoencoder
from phm_core.valve import (
    choose_waveform,
    fit_mechanism_reference,
    mechanism_features,
    mechanism_score,
)


def pressure_window(delay=0.0):
    t = 8 + np.arange(400) / 100
    inlet = np.clip((t - 9.2 - delay) / 0.2, 0, 1)
    outlet = np.clip((t - 9.1 - delay) / 0.2, 0, 1)
    return np.stack([190 - 50 * inlet, 120 * outlet], axis=1).astype("float32")[None]


def test_switch_features_follow_injected_delay():
    baseline = mechanism_features(pressure_window())
    delayed = mechanism_features(pressure_window(0.1))
    np.testing.assert_allclose(delayed[:, :6] - baseline[:, :6], 0.1, atol=0.011)
    np.testing.assert_allclose(delayed[:, 6:], baseline[:, 6:], atol=0.011)
    assert baseline.shape == (1, 9)


def test_pressure_gain_and_offset_do_not_become_timing_lag():
    x = pressure_window()
    np.testing.assert_allclose(
        mechanism_features(x * 1.2 + 4), mechanism_features(x), atol=0.011
    )


def test_missing_pressure_response_is_not_filled_as_healthy():
    normal = mechanism_features(np.repeat(pressure_window(), 4, axis=0))
    reference = fit_mechanism_reference(normal, np.arange(4))
    missing = mechanism_features(np.zeros((1, 400, 2), dtype="float32"))
    assert np.all(missing[:, :6] == 12)
    assert (
        mechanism_score(missing, reference)[0]
        > mechanism_score(normal, reference).max()
    )
    with pytest.raises(ValueError):
        mechanism_features(pressure_window(), rate=10)


def test_dense_autoencoder_shapes():
    assert DenseAutoencoder((400, 2))(torch.zeros(3, 400, 2)).shape == (3, 400, 2)
    assert DenseAutoencoder((9,))(torch.zeros(3, 9)).shape == (3, 9)


def test_selection_uses_validation_and_respects_false_alarm_budget():
    candidates = {
        "a": [{"recall": 0.5, "test_pr_auc": 0.8, "false_positive_per_1000": 10}],
        "b": [{"recall": 1.0, "test_pr_auc": 1.0, "false_positive_per_1000": 90}],
    }
    assert choose_waveform(candidates) == "a"
    with pytest.raises(ValueError):
        choose_waveform({"b": candidates["b"]})


def test_test_samples_do_not_change_training_or_scaling():
    values = np.random.default_rng(42).normal(size=(24, 12, 2)).astype("float32")
    modified = values.copy()
    modified[16:] += 100
    first = fit_autoencoder(values, np.arange(8), np.arange(8, 16), "dense", 42, 2)
    second = fit_autoencoder(modified, np.arange(8), np.arange(8, 16), "dense", 42, 2)
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])
    for a, b in zip(first[0].parameters(), second[0].parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_no_fault_slice_has_no_recall_claim():
    from phm_core.evaluation import evaluate

    result = evaluate(np.zeros(4), np.ones(4, dtype=bool), np.arange(4), 1)
    assert result["test_pr_auc"] is None
    assert result["recall"] is None and result["precision"] is None
    assert result["false_positive_per_1000"] == 0
