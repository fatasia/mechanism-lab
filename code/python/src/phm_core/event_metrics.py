"""事件级评估：告警段合并、按提前量匹配、以暴露小时计误报。

时间一律为浮点小时。召回的分母来自独立事件清单，而不是系统已报警的事件。
"""

import math

from phm_core.evidence import clopper_pearson


def merge_episodes(times, gap_hours):
    """相邻时间点间隔不超过 gap_hours 的合并为一段，返回 [(起, 止)]。"""
    if not math.isfinite(gap_hours) or gap_hours < 0:
        raise ValueError("合并间隔必须为非负有限数")
    ordered = sorted(set(float(t) for t in times))
    if not all(math.isfinite(t) for t in ordered):
        raise ValueError("时间必须为有限数")
    if not ordered:
        return []
    episodes, start, previous = [], ordered[0], ordered[0]
    for t in ordered[1:]:
        if t - previous > gap_hours:
            episodes.append((start, previous))
            start = t
        previous = t
    episodes.append((start, previous))
    return episodes


def match_warnings(
    warning_times,
    event_starts,
    exposure_hours,
    horizon_hours,
    min_lead_hours,
    merge_gap_hours,
):
    """告警必须在事件开始前至少 min_lead 小时、至多 horizon 小时内发出才算提前预警。

    两端均包含：min_lead <= event-start <= horizon；不足min_lead但严格在故障前
    的告警记为不确定。事件恰在告警时刻且min_lead=0时算命中。长告警段只按其
    首次发出时刻匹配，不利用后续持续报警追认提前预警。暴露分母的健康确认由调用方负责。
    """
    if not all(
        math.isfinite(x) for x in (min_lead_hours, horizon_hours, exposure_hours)
    ):
        raise ValueError("评价参数必须为有限数")
    if not 0 <= min_lead_hours < horizon_hours:
        raise ValueError("要求 0 <= min_lead < horizon")
    if exposure_hours <= 0:
        raise ValueError("暴露小时必须为正")
    episodes = merge_episodes(warning_times, merge_gap_hours)
    events = sorted(float(e) for e in event_starts)
    if not all(math.isfinite(e) for e in events) or len(set(events)) != len(events):
        raise ValueError("事件时刻必须为有限数且不重复")
    detected, leads = 0, []
    for event in events:
        candidates = [
            start
            for start, _ in episodes
            if event - horizon_hours <= start <= event - min_lead_hours
        ]
        if candidates:
            detected += 1
            leads.append(event - min(candidates))
    false_episodes, ambiguous = 0, 0
    for start, _ in episodes:
        if any(min_lead_hours <= e - start <= horizon_hours for e in events):
            continue
        if any(0 <= e - start < min_lead_hours for e in events):
            ambiguous += 1
            continue
        false_episodes += 1
    low, high = clopper_pearson(detected, len(events)) if events else (0.0, 1.0)
    return {
        "events": len(events),
        "detected": detected,
        "recall": detected / len(events) if events else None,
        "recall_interval_95": [low, high],
        "lead_hours": leads,
        "warning_episodes": len(episodes),
        "matched_episodes": len(episodes) - false_episodes - ambiguous,
        "false_episodes": false_episodes,
        "ambiguous_episodes": ambiguous,
        "exposure_hours": exposure_hours,
        "false_episodes_per_1000h": false_episodes / exposure_hours * 1000,
    }
