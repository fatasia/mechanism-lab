import importlib.util
import sys
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).resolve().parents[1]
for name in ("study_valve_residual", "study_valve_fusion", "study_valve_multiseed"):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
study = loaded


def test_binary_report_counts_and_macro_f1():
    healthy = np.array([True, True, False, False, False])
    score = np.array([0.0, 2.0, 0.0, 2.0, 2.0])
    metric = study.report_metric(score, healthy, np.arange(5), 1.0)
    assert (metric["normal"], metric["fault"]) == (2, 3)
    assert (metric["false_positives"], metric["false_negatives"]) == (1, 1)
    assert metric["records"] == metric["supported_records"] == 5
    assert metric["macro_f1"] == (0.5 + 2 / 3) / 2


def test_threshold_ties_are_not_alerts_and_subset_is_respected():
    metric = study.report_metric(
        np.array([100.0, 1.0, 2.0]),
        np.array([True, True, False]),
        np.array([1, 2]),
        1.0,
    )
    assert metric["false_positives"] == metric["false_negatives"] == 0
    assert metric["records"] == 2
    assert metric["macro_f1"] == 1
