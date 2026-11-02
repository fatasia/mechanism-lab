"""准入证据的统计工具：聚类区间、所需样本量、方差分量。

窗口/周期不是独立样本；连续周期属于同一块（设备、运行或工况），区间按块计算。
"""

import math

import numpy as np
from scipy.stats import beta


def clopper_pearson(events: int, trials: int, confidence: float = 0.95):
    """精确二项区间 (下界, 上界)；trials 为 0 时返回完全不确定。"""
    if not 0 <= events <= trials:
        raise ValueError("events 须在 0..trials 内")
    if trials == 0:
        return 0.0, 1.0
    alpha = 1 - confidence
    lower = 0.0 if events == 0 else beta.ppf(alpha / 2, events, trials - events + 1)
    upper = (
        1.0
        if events == trials
        else beta.ppf(1 - alpha / 2, events + 1, trials - events)
    )
    return float(lower), float(upper)


def one_sided_upper(events: int, trials: int, confidence: float = 0.95) -> float:
    """单侧精确上界，例如“0 次误报”对应的误报率上限。"""
    if not 0 <= events <= trials:
        raise ValueError("events 须在 0..trials 内")
    if trials == 0 or events == trials:
        return 1.0
    return float(beta.ppf(confidence, events + 1, trials - events))


def one_sided_lower(events: int, trials: int, confidence: float = 0.95) -> float:
    """单侧精确下界，例如“全部检出”对应的召回下限。"""
    if not 0 <= events <= trials:
        raise ValueError("events 须在 0..trials 内")
    if trials == 0 or events == 0:
        return 0.0
    return float(beta.ppf(1 - confidence, events, trials - events + 1))


def blocks(ids) -> list[tuple[int, int]]:
    """把连续编号合并为块；块是真正可近似独立的评估单位。"""
    ordered = sorted(int(i) for i in ids)
    if not ordered:
        return []
    result, start, previous = [], ordered[0], ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            result.append((start, previous))
            start = value
        previous = value
    result.append((start, previous))
    return result


def block_summary(truth_ids, error_ids, confidence: float = 0.95) -> dict:
    """按块统计：一个块只要含一次错误就算该块出错（保守）。

    truth_ids 为某类别全部样本编号，error_ids 为其中判错的样本编号。
    """
    all_blocks = blocks(truth_ids)
    wrong = {int(i) for i in error_ids}
    wrong_blocks = sum(
        any(i in wrong for i in range(start, end + 1)) for start, end in all_blocks
    )
    return {
        "blocks": len(all_blocks),
        "samples": len(list(truth_ids)),
        "error_samples": len(wrong),
        "error_blocks": wrong_blocks,
        "error_block_interval": clopper_pearson(
            wrong_blocks, len(all_blocks), confidence
        ),
        "clean_block_one_sided_upper": one_sided_upper(
            wrong_blocks, len(all_blocks), confidence
        ),
    }


def events_needed_zero_miss(recall_floor: float = 0.95, confidence: float = 0.95):
    """零漏检时，使召回单侧下界达到 recall_floor 所需的独立故障事件数。"""
    if not 0 < recall_floor < 1 or not 0 < confidence < 1:
        raise ValueError("召回下限和置信度须在 (0,1)")
    return math.ceil(math.log(1 - confidence) / math.log(recall_floor))


def hours_needed_zero_alarm(rate_per_1000h: float, confidence: float = 0.95):
    """零误报时，使误报率单侧上界达到 rate_per_1000h 所需的正常设备小时（泊松）。"""
    if rate_per_1000h <= 0 or not 0 < confidence < 1:
        raise ValueError("速率须为正，置信度须在 (0,1)")
    return math.ceil(-math.log(1 - confidence) / rate_per_1000h * 1000)


def variance_components(scores: np.ndarray) -> dict:
    """scores[记录, 重复抽样]（可含 NaN 表示该抽样中此记录不参与）。

    between：记录均值间方差，反映记录自身的稳定偏差；
    within：同一记录跨参考抽样的方差，反映参考估计噪声。
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2:
        raise ValueError("scores 须为二维 [记录, 抽样]")
    # 有效抽样少于 2 次的记录无法估计方差，不参与分解。
    values = values[np.sum(~np.isnan(values), axis=1) >= 2]
    if values.shape[0] < 2:
        raise ValueError("至少需要两条各有两次有效抽样的记录")
    means = np.nanmean(values, axis=1)
    within = np.nanvar(values, axis=1, ddof=1)
    within_mean = float(np.nanmean(within))
    between = float(np.var(means, ddof=1))
    # 记录均值的方差里含有抽样噪声 within/n，需扣除后才是真实的记录间差异。
    counts = np.sum(~np.isnan(values), axis=1)
    corrected = max(between - float(np.nanmean(within / np.maximum(counts, 1))), 0.0)
    total = corrected + within_mean
    return {
        "between_record": corrected,
        "within_record": within_mean,
        "icc": 0.0 if total == 0 else corrected / total,
    }
