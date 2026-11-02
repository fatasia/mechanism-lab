import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "study_valve_stress", Path(__file__).resolve().parents[1] / "study_valve_stress.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_shift_holds_edges_and_only_changes_requested_channel():
    x = np.arange(12, dtype="float32").reshape(1, 6, 2)
    shifted = study.perturb(x, {"kind": "shift_samples", "value": 2, "channels": [0]})
    np.testing.assert_array_equal(shifted[0, :, 0], [0, 0, 0, 2, 4, 6])
    np.testing.assert_array_equal(shifted[:, :, 1], x[:, :, 1])
    assert x[0, 1, 0] == 2


def test_noise_is_seeded_and_grid_has_no_duplicate_names():
    x = np.zeros((2, 400, 2), dtype="float32")
    scenario = {"kind": "noise_std_bar", "value": 0.1, "channels": [1]}
    first = study.perturb(x, scenario)
    np.testing.assert_array_equal(first, study.perturb(x, scenario))
    assert not np.any(first[:, :, 0])
    assert np.any(first[:, :, 1])
    grid = study.scenarios()
    assert len(grid) == len({s["name"] for s in grid}) == 49
