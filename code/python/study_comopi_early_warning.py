"""CoMoPI 包装机早期预警：整机留出、事件级、按暴露小时计误报。

标签是设备自身产生的部件故障报警 AL_53/AL_54，不是维修确认的故障真值；
结论只说明“传感器能否早于设备报警”，不能当作故障预测的现场验收。
参数在看结果前固定（见 PROTOCOL）。
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from phm_core.artifacts import capture_source, new_run
from phm_core.event_metrics import match_warnings, merge_episodes
from phm_core.evidence import clopper_pearson

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/comopi"
CHANNELS = [
    "AE",
    "BE",
    "AF",
    "BF",
    "APP",
    "BPP",
    "AP",
    "BP",
    "ALE",
    "BLE",
    "ALP",
    "BLP",
    "ADS",
    "BDS",
    "AES",
    "BES",
]
FAULT_CODES = ("AL_53", "AL_54")
PROTOCOL = {
    "reference_windows": 432,  # 每台机器最早的 72 传感器小时作为投运参考
    "event_merge_gap_h": 6,
    "horizon_h": 72,
    "min_lead_h": 1,
    "warning_merge_gap_h": 6,
    "ewma_alpha": 0.1,
    "segment_break_h": 1.0,
    "min_valid_channels": 8,
    "z_clip": 10.0,
    "healthy_margin_h": 72,
    "threshold_quantiles": [0.99, 0.995, 0.999],
    "min_prior_windows": 36,
    "mad_floor": 1e-3,
    "chance_seed": 20261102,
}
CHANCE_DRAWS = 500


def hours(text):
    return datetime.fromisoformat(text).timestamp() / 3600


def load_sensors():
    rows = {}
    with (DATA / "industrial_dataset_sensors_10m_agg.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            values = [
                float(row[c]) if row[c] not in ("", "nan") else np.nan for c in CHANNELS
            ]
            rows.setdefault(row["_serial"], []).append((hours(row["_time"]), values))
    result = {}
    for serial, items in rows.items():
        items.sort(key=lambda x: x[0])
        result[serial] = (
            np.array([t for t, _ in items]),
            np.array([v for _, v in items], dtype=float),
        )
    return result


def load_events():
    times = {}
    with (DATA / "industrial_dataset_alarm_10m_agg.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        reader = csv.reader(stream)
        head = next(reader)
        index = {c: i for i, c in enumerate(head)}
        for row in reader:
            if any(float(row[index[c]] or 0) > 0 for c in FAULT_CODES):
                times.setdefault(row[index["_serial"]], []).append(
                    hours(row[index["_time"]])
                )
    return {
        s: [start for start, _ in merge_episodes(t, PROTOCOL["event_merge_gap_h"])]
        for s, t in times.items()
    }


def machine_scores(times, values):
    """以机器自身最早的参考窗口做稳健标准化，再做按段重置的 EWMA。"""
    ref = values[: PROTOCOL["reference_windows"]]
    center = np.nanmedian(ref, axis=0)
    scale = np.maximum(
        1.4826 * np.nanmedian(np.abs(ref - center), axis=0), PROTOCOL["mad_floor"]
    )
    z = np.clip(np.abs(values - center) / scale, 0, PROTOCOL["z_clip"])
    valid = np.sum(~np.isnan(z), axis=1) >= PROTOCOL["min_valid_channels"]
    raw = np.where(
        valid,
        np.nanmean(np.where(np.isnan(z), 0, z), axis=1)
        * z.shape[1]
        / np.maximum(np.sum(~np.isnan(z), axis=1), 1),
        np.nan,
    )
    smooth = np.full(len(raw), np.nan)
    state = None
    alpha = PROTOCOL["ewma_alpha"]
    for i, value in enumerate(raw):
        if np.isnan(value):
            state = None
            continue
        broken = i > 0 and times[i] - times[i - 1] > PROTOCOL["segment_break_h"]
        state = (
            value if state is None or broken else alpha * value + (1 - alpha) * state
        )
        smooth[i] = state
    return smooth


def healthy_scores(times, scores, events):
    """远离任何故障事件（前后 healthy_margin 内无事件）的参考期后得分。"""
    margin = PROTOCOL["healthy_margin_h"]
    keep = np.arange(len(times)) >= PROTOCOL["reference_windows"]
    keep &= ~np.isnan(scores)
    for event in events:
        keep &= ~((times >= event - margin) & (times <= event + margin))
    return scores[keep]


def chance_level(times_post, n_episodes, events, exposure, observed_detected, seed):
    """同样数量的告警段随机落在有传感器数据的时刻，得到机会水平的召回。"""
    rng = np.random.default_rng(seed)
    if n_episodes == 0 or not events:
        return None
    detected, false = [], []
    for _ in range(CHANCE_DRAWS):
        picks = rng.choice(
            times_post, size=min(n_episodes, len(times_post)), replace=False
        )
        result = match_warnings(
            picks,
            events,
            exposure,
            PROTOCOL["horizon_h"],
            PROTOCOL["min_lead_h"],
            PROTOCOL["warning_merge_gap_h"],
        )
        detected.append(result["detected"])
        false.append(result["false_episodes"])
    detected = np.array(detected)
    return {
        "draws": CHANCE_DRAWS,
        "mean_detected": float(detected.mean()),
        "mean_false_episodes": float(np.mean(false)),
        "p_chance_detects_at_least_observed": float(
            np.mean(detected >= observed_detected)
        ),
    }


def evaluate_target(serial, data, events, quantile):
    times, _ = data[serial]
    scores = {s: machine_scores(*data[s]) for s in data}
    pool = np.concatenate(
        [
            healthy_scores(data[s][0], scores[s], events.get(s, []))
            for s in data
            if s != serial and len(data[s][0]) > PROTOCOL["reference_windows"] + 144
        ]
    )
    threshold = float(np.quantile(pool, quantile))
    start = PROTOCOL["reference_windows"]
    post = np.arange(len(times)) >= start
    warn = times[post & ~np.isnan(scores[serial]) & (scores[serial] > threshold)]
    exposure = float(np.sum(post & ~np.isnan(scores[serial])) / 6)
    evaluable, excluded = [], []
    for event in events.get(serial, []):
        prior = np.sum((times >= event - PROTOCOL["horizon_h"]) & (times < event))
        if event < times[start]:
            excluded.append({"time_h": event, "reason": "inside_reference_period"})
        elif prior < PROTOCOL["min_prior_windows"]:
            excluded.append(
                {"time_h": event, "reason": "insufficient_prior_sensor_data"}
            )
        else:
            evaluable.append(event)
    sensor_result = (
        match_warnings(
            warn,
            evaluable,
            exposure,
            PROTOCOL["horizon_h"],
            PROTOCOL["min_lead_h"],
            PROTOCOL["warning_merge_gap_h"],
        )
        if exposure > 0
        else None
    )
    # 朴素基线：每个已发生的故障事件之后发出一次预警，不使用传感器。
    baseline = (
        match_warnings(
            [e for e in events.get(serial, []) if e >= times[start]],
            evaluable,
            exposure,
            PROTOCOL["horizon_h"],
            PROTOCOL["min_lead_h"],
            PROTOCOL["warning_merge_gap_h"],
        )
        if exposure > 0
        else None
    )
    chance = (
        chance_level(
            times[post & ~np.isnan(scores[serial])],
            sensor_result["warning_episodes"],
            evaluable,
            exposure,
            sensor_result["detected"],
            PROTOCOL["chance_seed"] + int(quantile * 10000),
        )
        if sensor_result
        else None
    )
    return {
        "threshold": threshold,
        "evaluable_events": len(evaluable),
        "excluded_events": excluded,
        "sensor": sensor_result,
        "chance_level_same_episode_count": chance,
        "recent_fault_alarm_baseline": baseline,
    }


def pool(results, key):
    detected = sum(r[key]["detected"] for r in results if r[key])
    events = sum(r[key]["events"] for r in results if r[key])
    false = sum(r[key]["false_episodes"] for r in results if r[key])
    hours_total = sum(r[key]["exposure_hours"] for r in results if r[key])
    return {
        "events": events,
        "detected": detected,
        "recall": detected / events if events else None,
        "recall_interval_95_events_not_independent": list(
            clopper_pearson(detected, events)
        )
        if events
        else None,
        "false_episodes": false,
        "exposure_hours": hours_total,
        "false_episodes_per_1000h": false / hours_total * 1000 if hours_total else None,
    }


def run():
    data, events = load_sensors(), load_events()
    run_id, output = new_run(ROOT / "outputs", "comopi-early-warning")
    report = {
        "run_id": run_id,
        "protocol": PROTOCOL,
        "scope": "CoMoPI, 8 packaging machines; label = machine-generated AL_53/AL_54 alarms, not maintenance-confirmed faults; leave-one-machine-out threshold; exploratory",
        "source_hash": capture_source(
            ROOT / "code/python", output / "source.zip", "study_comopi_early_warning.py"
        ),
        "data_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(DATA.glob("*.csv"))
        },
        "events_per_machine": {s: len(v) for s, v in sorted(events.items())},
        "operating_points": {},
    }
    targets = [
        s for s in sorted(data) if len(data[s][0]) > PROTOCOL["reference_windows"] + 144
    ]
    for quantile in PROTOCOL["threshold_quantiles"]:
        results = {s: evaluate_target(s, data, events, quantile) for s in targets}
        report["operating_points"][str(quantile)] = {
            "per_machine": results,
            "pooled_sensor": pool(list(results.values()), "sensor"),
            "pooled_chance_expected_detected": sum(
                r["chance_level_same_episode_count"]["mean_detected"]
                for r in results.values()
                if r["chance_level_same_episode_count"]
            ),
            "pooled_recent_fault_alarm_baseline": pool(
                list(results.values()), "recent_fault_alarm_baseline"
            ),
        }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "output": str(output)}))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
