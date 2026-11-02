import importlib.util
import sys
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).resolve().parents[1]
for name in ("study_context_gate", "study_commissioning_budget"):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
study = loaded


def test_reference_budgets_share_identical_test_records():
    y = np.r_[np.zeros(10, dtype=int), np.ones(10, dtype=int)]
    repeats = np.tile(np.arange(1, 11), 2)
    tests = []
    for budget in (2, 4, 6):
        references, test = study.commissioning_split(np.arange(20), y, repeats, budget)
        assert len(references) == budget
        assert not set(references) & set(test)
        tests.append(test)
    np.testing.assert_array_equal(tests[0], tests[1])
    np.testing.assert_array_equal(tests[1], tests[2])
    np.testing.assert_array_equal(tests[0], np.arange(6, 20))


def test_local_scale_shrinkage_uses_reference_only():
    values = np.array([[-1], [1], [-10], [10], [999]], dtype="float32")
    contexts = np.array(["low", "low", "high", "high", "high"])
    healthy = np.array([True, True, True, True, False])
    references = np.arange(4)
    pooled = study.fit_commissioning_reference(
        values, contexts, healthy, references, "pooled"
    )
    first = study.fit_commissioning_reference(
        values, contexts, healthy, references, "shrunk"
    )
    assert pooled["low"]["scale"] == pooled["high"]["scale"]
    assert first["low"]["scale"][0] < first["high"]["scale"][0]
    values[4] = -999
    assert first == study.fit_commissioning_reference(
        values, contexts, healthy, references, "shrunk"
    )


def test_directional_error_features_keep_two_sided_current_changes():
    center = np.ones(20) * 2
    center[[1, 5, 9, 13]] = -2
    reference = {"a": {"center": center.tolist(), "scale": np.ones(20).tolist()}}
    lower = center.copy()
    lower[[0, 4, 8, 12, 16, 17]] = 1
    lower[[1, 5, 9, 13]] = -1
    lower[2] = 1
    higher = center.copy()
    higher[[0, 4, 8, 12, 16, 17]] = 3
    higher[[1, 5, 9, 13]] = -3
    higher[2] = 3
    values = np.stack([lower, higher])
    contexts = np.array(["a", "a"])
    directional = study.represent_residual(values, contexts, reference, "directional")
    np.testing.assert_array_equal(directional[:, 0], [0, 1])
    np.testing.assert_array_equal(directional[:, 1], [0, 1])
    np.testing.assert_array_equal(directional[:, 2], [1, 1])
    signed = study.represent_residual(values, contexts, reference, "signed")
    absolute = study.represent_residual(values, contexts, reference, "absolute")
    np.testing.assert_array_equal(absolute, np.abs(signed))
