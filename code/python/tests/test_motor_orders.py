import importlib.util
import sys
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).resolve().parents[1]
for name in ("motor_features", "extract_motor_orders"):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
study = loaded


def test_order_peak_remains_at_same_order_across_speed():
    peaks = []
    for period in (768, 256):
        t = np.arange(study.WINDOW)
        signal = np.sin(2 * np.pi * 9.25 * t / period)
        values = np.repeat(signal[:, None], 6, axis=1)
        spectrum = study.angular_spectrum(values, np.arange(0, study.WINDOW, period))
        assert spectrum.shape == (6, 512)
        assert np.isfinite(spectrum).all()
        peaks.append(study.ORDER_GRID[np.argmax(spectrum[0])])
    np.testing.assert_allclose(peaks, 9.25, atol=0.063)


def test_constant_signal_has_no_spurious_order_peak():
    spectrum = study.angular_spectrum(
        np.ones((study.WINDOW, 6)), np.arange(0, study.WINDOW, 512)
    )
    np.testing.assert_allclose(spectrum, -12, atol=1e-6)
