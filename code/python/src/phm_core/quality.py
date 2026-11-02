"""工业输入质量是评分前置条件，不能用填零或上一值伪装正常。"""

import numpy as np


def assess_window(
    values, expected_shape, sample_rate, expected_rate, timestamps=None, bounds=None
):
    values = np.asarray(values, dtype=float)
    issues = []
    if tuple(values.shape) != tuple(expected_shape):
        issues.append(
            {
                "code": "shape",
                "expected": list(expected_shape),
                "actual": list(values.shape),
            }
        )
    if sample_rate != expected_rate:
        issues.append(
            {"code": "sample_rate", "expected": expected_rate, "actual": sample_rate}
        )
    if not np.isfinite(values).all():
        issues.append(
            {"code": "missing_or_nonfinite", "count": int((~np.isfinite(values)).sum())}
        )
    if timestamps is not None:
        times = np.asarray(timestamps, dtype=float)
        if (
            len(times) != len(values)
            or len(times) < 2
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
            or np.any(np.abs(np.diff(times) - 1 / expected_rate) > 0.1 / expected_rate)
        ):
            issues.append({"code": "timestamp_gap_or_order"})
    if bounds is not None and values.ndim == 2 and values.shape[-1] == len(bounds):
        for channel, (lower, upper) in enumerate(bounds):
            if np.any(values[:, channel] < lower) or np.any(values[:, channel] > upper):
                issues.append({"code": "physical_range", "channel": channel})
    return {"status": "invalid" if issues else "valid", "issues": issues}
