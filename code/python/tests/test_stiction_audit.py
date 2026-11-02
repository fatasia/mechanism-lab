import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "audit_stiction", Path(__file__).resolve().parents[1] / "audit_stiction.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def sample():
    lines = [
        "PROCESSDATA\tloop_A Data\tyyyy-MM-dd HH:mm:ss",
        "8\t10\t2024-01-01 00:00:00\t2",
        "tagname\tMV\tPV\tSV\tMODE\tbOpOFL\tbPvOFL\tbSpOFL\tbAutOFL",
        "desc\tOP\tPV\tSP\tAut\ta\tb\tc\td",
        "fsMax=999\tStick1=1",
        "2024-01-01 00:00:00\t1\t2\t3\t1\t0\t0\t0\t0",
        "2024-01-01 00:00:10\t2\t3\t3\t1\t0\t0\t0\t0",
    ]
    return "\n".join(lines).encode()


def test_metadata_and_signals_are_separate():
    values, times, metadata = study.parse_record(sample())
    assert values.shape == (2, 8)
    assert metadata["loop"] == "loop_A"
    assert metadata["sample_interval_seconds"] == 10
    assert metadata["metadata_start_matches"]
    np.testing.assert_array_equal(np.diff(times), [10])
    assert not np.any(values == 999)


def test_malformed_row_is_not_silently_truncated():
    with pytest.raises(ValueError, match="列数"):
        study.parse_record(sample().replace(b"\t2\t3\t3\t1", b"\t2\t3\t1"))
