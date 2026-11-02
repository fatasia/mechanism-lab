import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "motor_audit", Path(__file__).resolve().parents[1] / "audit_motor.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_record_grouping_preserves_missing_suffix():
    result = audit.record_metadata(
        "bearing_outer_H_and_inner_H_speed_circulation_20Nm_1000rpm.csv"
    )
    assert result["label"] == "bearing_outer_H_and_inner_H"
    assert result["condition"] == "speed/20Nm/1000rpm"
    assert result["run_suffix"] is None


def test_raw_time_column_and_clock_gate():
    values = np.ones((20, 9))
    values[:, 0] = np.arange(20) / 12800
    assert audit.array_audit(values)["valid"]
    values[10, 0] = values[9, 0]
    assert not audit.array_audit(values)["valid"]
    with pytest.raises(ValueError, match="9|8"):
        audit.array_audit(values[:, 1:])


def test_nonfinite_channel_rejected():
    values = np.ones((20, 9))
    values[:, 0] = np.arange(20) / 12800
    values[3, 8] = np.nan
    result = audit.array_audit(values)
    assert not result["valid"]
    json.dumps(result, allow_nan=False)
