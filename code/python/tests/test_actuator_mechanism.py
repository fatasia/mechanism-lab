import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "actuator_mechanism", Path(__file__).resolve().parents[1] / "actuator_mechanism.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def example():
    command = np.concatenate(
        [
            np.zeros(50),
            np.linspace(0, 100, 100),
            np.full(100, 100),
            np.linspace(100, 0, 100),
            np.zeros(50),
        ]
    )
    basis = study.motion_basis(np.stack([command, command]))
    coefficients = np.array(
        [[0, 0.002], [0.001, 0.02], [0.0001, 0.001], [0.01, 0.1], [0, 0]]
    )
    signals = basis @ coefficients + np.array([0.02, 2.0])
    return basis, signals


def test_reference_never_reads_unselected_fault_signals():
    basis, signals = example()
    first = study.fit_motion_reference(basis, signals, np.array([0]))
    changed = signals.copy()
    changed[1] += np.array([10, 50])
    second = study.fit_motion_reference(basis, changed, np.array([0]))
    assert first == second
    residual = study.motion_residual(basis, changed, first)
    assert np.sqrt(np.mean(residual[0] ** 2)) < 0.01
    np.testing.assert_allclose(residual[1].mean(0), [10, 50], atol=1e-4)


def test_saved_raw_coefficients_match_standardized_predictor():
    basis, signals = example()
    reference = study.fit_motion_reference(basis, signals, np.array([0]))
    expected = basis @ np.array(reference["raw_basis_coefficients"]) + np.array(
        reference["raw_basis_intercept"]
    )
    actual = signals - study.motion_residual(basis, signals, reference)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    scaled = study.motion_residual(basis, signals, reference, True)
    np.testing.assert_allclose(np.sqrt(np.mean(scaled[0] ** 2, axis=0)), 1, atol=1e-6)
