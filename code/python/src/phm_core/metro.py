"""压缩机日志按原时间戳聚合，缺口和冻结记录不跨接成连续序列。"""

import csv
import hashlib
import io
from zipfile import ZipFile

import numpy as np

ANALOG = (
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
)
FEATURES = tuple(
    f"{name}_{stat}" for stat in ("mean", "std", "min") for name in ANALOG
) + (
    "loaded_fraction",
    "off_fraction",
    "panel_reservoir_gap",
    "reservoir_decay_bar_per_second",
)
FAILURES = (
    ("2020-04-18T00:00", "2020-04-18T23:59"),
    ("2020-05-29T23:30", "2020-05-30T06:00"),
    ("2020-06-05T10:00", "2020-06-07T14:30"),
    ("2020-07-15T14:30", "2020-07-15T19:00"),
)


def timestamp(value):
    return int(np.datetime64(value, "s").astype("int64"))


def aggregate_bucket(rows, times):
    values = np.asarray(rows)
    gaps = np.diff(times)
    if (
        len(rows) < 48
        or (len(gaps) and (gaps.max() > 30 or gaps.min() <= 0))
        or not np.isfinite(values).all()
    ):
        return None, "gap_or_missing"
    analog = values[:, :7]
    if np.all(np.ptp(analog, axis=0) == 0):
        return None, "frozen_analog"
    unloaded = values[:, 8] < 0.5
    t = np.asarray(times)[unloaded]
    pressure = values[unloaded, 4]
    if len(t) >= 5 and np.ptp(t) > 0:
        t = t - t.mean()
        decay = -float(t @ (pressure - pressure.mean()) / (t @ t))
    else:
        # 无卸载段时衰减不可估计；额外工况特征保留卸载占比，数值项记 0。
        decay = 0.0
    context = [
        values[:, 8].mean(),
        (values[:, 6] < 1).mean(),
        np.abs(values[:, 1] - values[:, 4]).mean(),
        decay,
    ]
    return np.r_[analog.mean(0), analog.std(0), analog.min(0), context].astype(
        "float32"
    ), None


def prepare(archive, destination):
    with open(archive, "rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if destination.exists():
        with np.load(destination) as data:
            if "source_sha256" in data and str(data["source_sha256"]) == source_hash:
                return {k: data[k] for k in data.files}
    features, ends, rows, times, periods = [], [], [], [], []
    invalid, bucket, previous, count = (
        {"gap_or_missing": 0, "frozen_analog": 0},
        None,
        None,
        0,
    )
    with ZipFile(archive) as package:
        name = next(n for n in package.namelist() if n.endswith(".csv"))
        reader = csv.reader(io.TextIOWrapper(package.open(name), newline=""))
        next(reader)
        for row in reader:
            if not row:
                continue
            now = timestamp(row[1])
            if previous is not None and len(periods) < 10000:
                periods.append(now - previous)
            current = now // 600
            if bucket is not None and current != bucket:
                feature, reason = aggregate_bucket(rows, times)
                if feature is not None:
                    features.append(feature)
                    ends.append((bucket + 1) * 600)
                else:
                    invalid[reason] += 1
                rows, times = [], []
            rows.append([float(v) for v in row[2:]])
            times.append(now)
            bucket, previous = current, now
            count += 1
    # 文件尾没有完整闭合的十分钟区间，不评分。
    result = {
        "source_sha256": np.array(source_hash),
        "x": np.asarray(features),
        "end": np.asarray(ends),
        "source_rows": np.array(count),
        "median_interval_seconds": np.array(np.median(periods)),
        "invalid_counts": np.array(
            [invalid[k] for k in ("gap_or_missing", "frozen_analog")]
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **result)
    return result


def event_report(scores, ends, threshold, horizon_hours=24):
    from phm_core.decision import DecisionConfig, DecisionState, step

    config = DecisionConfig(threshold, threshold - max(abs(threshold) * 0.2, 1e-8))
    state, events, previous = DecisionState(), [], None
    for score, end in zip(scores, ends):
        if previous is not None and end - previous != 600:
            if state.active:
                events[-1]["end"] = previous
            state = DecisionState()
        event, state = step(float(score), state, config)
        if event == "opened":
            events.append({"start": int(end), "end": None})
        elif event == "closed":
            events[-1]["end"] = int(end)
        previous = int(end)
    if events and events[-1]["end"] is None:
        events[-1]["end"] = previous
    matched, failures = set(), []
    for start, end in FAILURES:
        indices = [
            i
            for i, e in enumerate(events)
            if timestamp(start) - horizon_hours * 3600 <= e["start"] <= timestamp(end)
        ]
        matched.update(indices)
        failures.append(
            {
                "start": start,
                "end": end,
                "detected": bool(indices),
                "lead_hours": (timestamp(start) - events[indices[0]]["start"]) / 3600
                if indices
                else None,
            }
        )
    hours = len(ends) / 6
    return {
        "failures": failures,
        "detected_events": sum(f["detected"] for f in failures),
        "reported_events": len(FAILURES),
        "alarm_events": len(events),
        "unmatched_events": len(events) - len(matched),
        "logged_hours": hours,
        "unmatched_per_1000_logged_hours": (len(events) - len(matched)) / hours * 1000
        if hours
        else None,
        "events": events,
    }
