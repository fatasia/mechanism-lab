import importlib.util
import sys
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).resolve().parents[1]
for name in ("audit_stiction", "study_stiction"):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
study = loaded


def test_shapes_are_invariant_to_consistent_positive_unit_conversion():
    rng = np.random.default_rng(23)
    values = rng.normal(size=(751, 8)).cumsum(0)
    original = study.shape_features(values, 10)
    transformed = values.copy()
    transformed[:, 0] = 5 * values[:, 0] + 100
    transformed[:, 1:3] = 2 * values[:, 1:3] - 80
    result = study.shape_features(transformed, 10)
    assert len(original) == 31
    np.testing.assert_allclose(
        list(original.values()), list(result.values()), atol=1e-10
    )


def test_many_records_from_one_loop_do_not_dominate_training():
    labels = np.array([0, 0, 0, 1, 1, 1, 1])
    groups = np.array(
        ["normal_a", "normal_a", "normal_b", "fault_a", "fault_a", "fault_a", "fault_b"]
    )
    weights = study.training_weights(labels, groups)
    totals = [weights[groups == g].sum() for g in np.unique(groups)]
    np.testing.assert_allclose(totals, totals[0])
    assert np.isclose(weights[labels == 0].sum(), weights[labels == 1].sum())


def test_unsupported_records_stay_in_coverage_denominator():
    result = study.summary(np.array([0, 1]), np.array([0.5, 0.9]), 3)
    assert result["records"] == 3
    assert result["supported_records"] == 2
    assert result["unsupported_records"] == 1
    assert result["false_positives"] == result["false_negatives"] == 0


def test_mixed_state_loop_uses_all_labels_for_class_balance():
    labels = np.array([0, 1, 1, 0])
    groups = np.array(["mixed", "mixed", "mixed", "normal"])
    weights = study.training_weights(labels, groups)
    assert np.isclose(weights[labels == 0].sum(), weights[labels == 1].sum())
