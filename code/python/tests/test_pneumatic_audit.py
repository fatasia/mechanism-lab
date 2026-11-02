"""气动分段仅按已发生的状态边界，截断段不混入完整动作特征。"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "pneumatic_audit", Path(__file__).resolve().parents[1] / "audit_pneumatic.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_segment_bounds_short_states_and_physical_units():
    t = np.arange(201) / 100
    p, c = np.full(len(t), 6.0), np.full(len(t), 0.5)
    features, states, quality = audit.summarize(
        t, p, c, np.array([-0.1, 0.2, 0.24, 0.8, 1.5, 2.2]), np.arange(6)
    )
    np.testing.assert_array_equal(states, [2, 3])
    np.testing.assert_allclose(features[:, 1], 6)
    np.testing.assert_allclose(features[:, 7], 0.5)
    assert quality == {
        "truncated_intervals": 2,
        "short_intervals": 1,
        "terminal_state_excluded": 1,
    }


def test_raw_time_reversal_is_preserved_not_silently_sorted(tmp_path):
    path = tmp_path / "Dump_long_100Hz_line_pressure_20230920_105128.csv"
    path.write_text("Time;Value\n10:51:28.020;6\n10:51:28.010;7\n", encoding="utf-8")
    times, values = audit.read_signal(path)
    assert times[1] < times[0]
    np.testing.assert_array_equal(values, [6, 7])


def test_noninteger_plc_state_is_not_rounded():
    t = np.arange(201) / 100
    with pytest.raises(ValueError, match="整数"):
        audit.summarize(t, t, t, np.array([0.1, 0.8]), np.array([1.5, 2]))


def test_plc_filename_one_second_offset_is_explicit_and_unique():
    assert audit.pair_state_keys({"20231129_103036"}, {"20231129_103035"}) == {
        "20231129_103036": "20231129_103035"
    }
    with pytest.raises(ValueError, match="唯一"):
        audit.pair_state_keys(
            {"20231129_103036"}, {"20231129_103035", "20231129_103037"}
        )
    with pytest.raises(ValueError, match="唯一"):
        audit.pair_state_keys({"20231129_103036"}, {"20231129_103033"})
