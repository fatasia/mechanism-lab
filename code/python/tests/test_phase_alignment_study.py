"""分相位表示的数值契约；合成信号不用于故障效果统计。"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "phase_alignment", Path(__file__).resolve().parents[1] / "study_phase_alignment.py"
)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_phase_shape_preserves_signal_amplitude_and_duration():
    x = np.zeros((1, 400, 3), dtype="float32")
    x[0, :100, 0] = np.linspace(0, 10, 100)
    x[0, 100:200, 0] = 10
    x[0, 200:300, 0] = np.linspace(10, 0, 100)
    x[0, :, 1] = .2
    x[0, :, 2] = -.5
    base = study.phase_features(x)
    assert base.shape == (1, 195)
    assert base[0, -3:].sum() == pytest.approx(16)
    y = x.copy()
    y[:, :, 1:] *= 2
    scaled = study.phase_features(y)
    np.testing.assert_allclose(scaled[:, :192], base[:, :192] * 2)
    np.testing.assert_allclose(scaled[:, -3:], base[:, -3:])
    assert (base[0, 32:64] < 0).all()


def test_phase_rejects_partial_motion_and_nonfinite_input():
    with pytest.raises(ValueError, match="阶段"):
        study.phase_features(np.zeros((1, 400, 3)))
    with pytest.raises(ValueError, match="有限"):
        study.phase_features(np.full((1, 400, 3), np.nan))
