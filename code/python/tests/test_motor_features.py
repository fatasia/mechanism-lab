import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "motor_features", Path(__file__).resolve().parents[1] / "motor_features.py"
)
features = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(features)


def example():
    t = np.arange(features.WINDOW) / features.FS
    x = np.zeros((len(t), 9))
    x[:, 0] = t
    x[:, 1] = ((t * 20) % 1 < 0.2) * 4
    for channel in range(3):
        x[:, 3 + channel] = (1 + 0.5 * np.cos(2 * np.pi * 20 * 3.585 * t)) * np.sin(
            2 * np.pi * 1500 * t
        )
        x[:, 6 + channel] = np.sin(2 * np.pi * 40 * t - channel * 2 * np.pi / 3)
    return x


def test_phase_uses_pulses_not_filename():
    edges, quality = features.phase_reference(example()[:, 1])
    assert quality["status"] == "active"
    assert abs(quality["rpm"] - 1200) < 2
    assert len(edges) >= 38
    assert features.phase_reference(np.zeros(features.WINDOW))[0] is None


def test_features_balance_and_order_evidence():
    x = example()
    edges, _ = features.phase_reference(x[:, 1])
    vector, spectrum = features.window_features(x, edges)
    assert vector.shape == (len(features.FEATURE_NAMES),) == (114,)
    assert spectrum.shape == (6, 1600)
    assert np.isfinite(vector).all() and np.isfinite(spectrum).all()
    assert vector[84] < 1e-4 and vector[85] < 1e-4
    assert vector[87] > 0.8
    assert vector[90] < 0.05


def test_current_imbalance_changes_without_vibration_change():
    x = example()
    edges, _ = features.phase_reference(x[:, 1])
    original, _ = features.window_features(x, edges)
    x[:, 6] *= 0.5
    changed, _ = features.window_features(x, edges)
    np.testing.assert_array_equal(original[:42], changed[:42])
    assert changed[84] > original[84] + 0.1
    assert changed[85] > original[85] + 0.1
