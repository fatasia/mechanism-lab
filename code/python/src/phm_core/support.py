"""适用域门控：输入不满足模型前提时输出“不支持”，不默认判为正常。

阀门时序判据依赖独立命令边沿对齐。同步容差取自输入扰动研究：两路同步偏移
±1 个采样点（10 ms）决策不变，+2 点出现误报，±5 点出现大量误报或漏报
（outputs/valve-block-evidence 的 shift_tolerance）。
"""

import math

VALVE_SYNC_TOLERANCE_SAMPLES = 1
# 秒级时间戳相减的浮点误差不能把恰好等于容差的偏移误判为超限。
FLOAT_SLACK = 1e-6


def _number(value):
    return isinstance(value, int | float) and not isinstance(value, bool)


def valve_support(
    sample_rate_hz,
    expected_rate_hz,
    command_edge_s,
    window_start_s,
    expected_lead_s=0.0,
    channel_skew_s=0.0,
    tolerance_samples=VALVE_SYNC_TOLERANCE_SAMPLES,
):
    """返回 {status, reasons, offset_samples, skew_samples}。

    没有独立命令边沿时一律不支持：按压力波形自身对齐会把真实的响应迟滞一并对齐掉。
    """
    reasons = []
    numbers = (sample_rate_hz, expected_rate_hz, expected_lead_s, channel_skew_s)
    if not all(_number(v) and math.isfinite(v) for v in numbers):
        return {
            "status": "unsupported",
            "reasons": [{"code": "invalid_input"}],
            "offset_samples": None,
            "skew_samples": None,
        }
    if sample_rate_hz != expected_rate_hz:
        reasons.append(
            {
                "code": "sample_rate",
                "expected": expected_rate_hz,
                "actual": sample_rate_hz,
            }
        )
    offset = None
    if command_edge_s is None:
        reasons.append({"code": "no_independent_command_edge"})
    elif not (
        _number(command_edge_s)
        and _number(window_start_s)
        and math.isfinite(command_edge_s)
        and math.isfinite(window_start_s)
    ):
        reasons.append({"code": "invalid_input"})
    else:
        offset = (window_start_s - command_edge_s - expected_lead_s) * expected_rate_hz
        if abs(offset) > tolerance_samples + FLOAT_SLACK:
            reasons.append(
                {
                    "code": "sync_offset",
                    "offset_samples": offset,
                    "tolerance_samples": tolerance_samples,
                }
            )
    skew = channel_skew_s * expected_rate_hz
    if abs(skew) > tolerance_samples + FLOAT_SLACK:
        reasons.append(
            {
                "code": "channel_skew",
                "skew_samples": skew,
                "tolerance_samples": tolerance_samples,
            }
        )
    return {
        "status": "unsupported" if reasons else "supported",
        "reasons": reasons,
        "offset_samples": offset,
        "skew_samples": skew,
    }
