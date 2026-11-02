"""事件级评估：告警段合并、按提前量匹配、以暴露小时计误报。

时间一律为浮点小时。召回的分母来自独立事件清单，而不是系统已报警的事件。
"""

from phm_core.evidence import clopper_pearson


def merge_episodes(times, gap_hours):
    """相邻时间点间隔不超过 gap_hours 的合并为一段，返回 [(起, 止)]。"""
    ordered = sorted(float(t) for t in times)
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

    误报是没有任何事件落在其后 (min_lead, horizon] 内的告警段，按每 1000 暴露
    小时计；不确定的情形（事件在 min_lead 内才发生）不计为命中也不计为误报。
    """
    if not 0 <= min_lead_hours < horizon_hours:
        raise ValueError("要求 0 <= min_lead < horizon")
    if exposure_hours <= 0:
        raise ValueError("暴露小时必须为正")
    episodes = merge_episodes(warning_times, merge_gap_hours)
    events = sorted(float(e) for e in event_starts)
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
        after = [e for e in events if e > start]
        if any(start + min_lead_hours < e <= start + horizon_hours for e in after):
            continue
        if any(start < e <= start + min_lead_hours for e in after):
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
        "false_episodes": false_episodes,
        "ambiguous_episodes": ambiguous,
        "exposure_hours": exposure_hours,
        "false_episodes_per_1000h": false_episodes / exposure_hours * 1000,
    }
