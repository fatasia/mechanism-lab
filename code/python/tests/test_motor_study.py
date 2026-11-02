import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "motor_study", Path(__file__).resolve().parents[1] / "study_motor.py"
)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_record_aggregation_and_unsupported_coverage():
    probabilities = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])
    output = study.record_probabilities(probabilities, np.array([0, 0, 1]), [0, 1, 2])
    np.testing.assert_allclose(output[:2], [[0.7, 0.3], [0.1, 0.9]])
    result = study.metrics(np.array([0, 1, 1]), output, 0, 2)
    assert result["unsupported_records"] == 1
    assert result["normal"] == 1 and result["fault"] == 1
    assert not study.source_passes(result)


def test_fault_misclassification_is_not_detection_miss():
    result = study.metrics(
        np.array([0, 1, 2]), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), 0, 3
    )
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["macro_f1"] == 0


def test_feature_arms_do_not_overlap_modalities():
    assert not set(study.ARMS["vibration_rf"]) & set(study.ARMS["current_rf"])
    assert set(study.ARMS["vibration_rf"]) | set(study.ARMS["current_rf"]) == set(
        study.ARMS["fusion_rf"]
    )
