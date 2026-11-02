"""局部质量筛选不通过时间排序伪造有效信号。"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).resolve().parents[1]
# 研究入口之间的依赖使用显式源码路径；不安装到运行SDK。
for name in ("audit_pneumatic", "audit_pneumatic_segments"):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
study = loaded


def test_valid_interval_keeps_original_values():
    t = np.arange(201) / 100
    values, status = study.interval_values(t, t * 2, 0.2, 0.8)
    assert status == "valid"
    np.testing.assert_array_equal(values, (t * 2)[20:80])


def test_time_reversal_or_excursion_is_not_sorted_away():
    t = np.arange(201) / 100
    t[35] = 0.9
    values, status = study.interval_values(t, np.arange(len(t)), 0.2, 0.8)
    assert values is None
    assert status == "nonmonotonic"
    # 同一记录里未受影响的其它完整动作仍可使用。
    _, status = study.interval_values(t, np.arange(len(t)), 1.0, 1.5)
    assert status == "valid"


def test_boundary_gap_and_truncation_are_not_filled():
    t = np.arange(201) / 100
    assert study.interval_values(t, t, -0.1, 0.5)[1] == "truncated"
    assert study.interval_values(t, t, 0.1, 0.12)[1] == "short"
    keep = (t < 0.2) | (t > 0.3)
    assert study.interval_values(t[keep], t[keep], 0.1, 0.5)[1] == "gap"
