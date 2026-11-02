import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "pneumatic_states", Path(__file__).resolve().parents[1] / "study_pneumatic_states.py"
)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)
alarm_summary = study.alarm_summary
chronological_split = study.chronological_split
fit_scores = study.fit_scores


def test_whole_day_partitions():
    days = np.repeat(["a", "b", "c", "d", "e", "f", "g"], 3)
    parts = chronological_split(days)
    assert set(days[parts["train"]]) == {"a", "b", "c", "d"}
    assert set(days[parts["calibration"]]) == {"e"}
    assert set(days[parts["test"]]) == {"f", "g"}
    assert len(set(parts["train"]) & set(parts["test"])) == 0


def test_unsupported_state_is_not_normal():
    x = np.arange(62).reshape(-1, 1)
    states = np.r_[np.zeros(60), [1, 1]]
    parts = {
        "train": np.arange(30),
        "calibration": np.arange(30, 60),
        "test": np.array([60, 61]),
    }
    scores, thresholds, support = fit_scores(x, states, parts, 42, True)
    assert not support["1.0"]["supported"]
    assert alarm_summary(scores, thresholds, parts["test"]) == {
        "segments": 2,
        "supported": 0,
        "unsupported": 2,
        "alarms": 0,
        "alarm_fraction": None,
    }


def test_test_values_do_not_change_fitted_threshold():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(70, 3))
    parts = {
        "train": np.arange(30),
        "calibration": np.arange(30, 60),
        "test": np.arange(60, 70),
    }
    _, before, _ = fit_scores(x, np.zeros(70), parts, 42, True)
    x[parts["test"]] *= 10000
    _, after, _ = fit_scores(x, np.zeros(70), parts, 42, True)
    np.testing.assert_array_equal(before, after)
