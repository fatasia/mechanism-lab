import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "audit_actuator_errors",
    Path(__file__).resolve().parents[1] / "audit_actuator_errors.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_grouped_counts_separate_wrong_fault_class_and_missed_fault():
    y = np.array([0, 0, 1, 2, 3])
    result = audit.grouped_counts(
        y, np.array([0, 3, 3, 0, 3]), np.array(["a", "a", "b", "b", "b"])
    )
    assert result["a"]["false_positives"] == 1
    assert result["b"]["false_negatives"] == 1
    assert result["b"]["wrong_fault_class"] == 1
    assert result["b"]["confusion"][1][3] == 1
    assert result["b"]["records"] == 3


def test_seed_agreement_preserves_record_denominator():
    y = np.array([0, 1, 2, 3])
    prediction = np.array([[1, 0, 3, 3], [2, 0, 2, 3], [3, 0, 3, 3]])
    result = audit.stable_errors(y, prediction)
    np.testing.assert_array_equal(
        result["all_seeds_false_positive"], [True, False, False, False]
    )
    np.testing.assert_array_equal(
        result["all_seeds_false_negative"], [False, True, False, False]
    )
    assert result["any_seed_wrong"].sum() == 3
    assert result["all_seeds_wrong"].sum() == 2
