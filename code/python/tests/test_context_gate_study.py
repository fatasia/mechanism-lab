"""研究脚本的判定边界；使用合成概率，不计入设备效果实验。"""

import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "context_gate", Path(__file__).resolve().parents[1] / "study_context_gate.py"
)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_gate_does_not_choose_normal_by_fault_subtype_argmax():
    probabilities = np.array([[.4, .2, .2, .2], [.2, .1, .6, .1]])
    np.testing.assert_array_equal(study.predict(probabilities, .5), [1, 2])
    np.testing.assert_array_equal(study.predict(probabilities, 1), [0, 0])


def test_gate_boundary_and_binary_detection_separate_from_fault_type():
    probabilities = np.array([[.5, .2, .2, .1], [.1, .6, .2, .1]])
    prediction = study.predict(probabilities, .5)
    np.testing.assert_array_equal(prediction, [0, 1])
    result = study.metrics(np.array([0, 2]), prediction)
    assert result["false_positives"] == result["false_negatives"] == 0
    assert result["fault_recall"] == 1
    assert result["macro_f1"] < 1
