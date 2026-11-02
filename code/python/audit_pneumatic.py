"""气动过程数据配对与PLC状态切段审计；不把动作标签解释为故障标签。"""

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from phm_core.artifacts import capture_source, new_run

ROOT = Path(__file__).resolve().parents[2]
PREFIXES = {
    "pressure": "Dump_long_100Hz_line_pressure_",
    "current": "Dump_long_100Hz_hall_sensor_",
    "state": "Dump_long_PLCstate_dt",
}


def pair_state_keys(signal_keys, state_keys):
    """记录名可能相差1秒；仅接收同日、唯一且不复用的PLC匹配，并保留映射。"""
    mapping = {}
    for key in sorted(signal_keys):
        instant = datetime.strptime(key, "%Y%m%d_%H%M%S")
        candidates = [
            other
            for other in state_keys
            if other[:8] == key[:8]
            and abs(
                (datetime.strptime(other, "%Y%m%d_%H%M%S") - instant).total_seconds()
            )
            <= 1
        ]
        if len(candidates) != 1 or candidates[0] in mapping.values():
            raise ValueError(f"PLC文件名不能唯一配对: {key}")
        mapping[key] = candidates[0]
    if set(mapping.values()) != set(state_keys):
        raise ValueError("存在未配对PLC记录")
    return mapping


def read_signal(path):
    """文件名给出日期；CSV仅含时分秒，不把日期当作模型输入。"""
    date = datetime.strptime(path.stem[-15:-7], "%Y%m%d").date()
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter=";")
        next(reader)
        rows = [(datetime.fromisoformat(f"{date}T{t}"), float(v)) for t, v in reader]
    # 原资料未声明时区；以无时区日历秒做相对对齐，不套用运行主机时区。
    times = np.array([(t - datetime(1970, 1, 1)).total_seconds() for t, _ in rows])
    values = np.array([v for _, v in rows])
    if len(times) < 2 or not np.isfinite(values).all():
        raise ValueError(f"数据不完整: {path.name}")
    return times, values


def summarize(times, pressure, current, state_times, states):
    """只提取有完整左右边界的状态段，首尾截断和短段单独计数。"""
    features, labels = [], []
    truncated = short = 0
    if not np.equal(states, np.floor(states)).all():
        raise ValueError("PLC动作状态必须为整数")
    for i in range(len(state_times) - 1):
        start, end = state_times[i : i + 2]
        if start < times[0] or end > times[-1]:
            truncated += 1
            continue
        ids = np.flatnonzero((times >= start) & (times < end))
        if len(ids) < 5:
            short += 1
            continue
        row = [end - start]
        for signal in (pressure[ids], current[ids]):
            row.extend(
                [
                    signal.mean(),
                    signal.std(),
                    np.quantile(signal, 0.05),
                    np.quantile(signal, 0.95),
                    signal[-1] - signal[0],
                    np.sqrt(np.mean(np.diff(signal) ** 2)),
                ]
            )
        features.append(row)
        labels.append(int(states[i]))
    return (
        np.asarray(features, dtype="float32").reshape(-1, 13),
        np.asarray(labels),
        {
            "truncated_intervals": truncated,
            "short_intervals": short,
            "terminal_state_excluded": 1,
        },
    )


def run():
    directory = ROOT / "data/raw/pneumatic-process"
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    paths = {
        kind: {p.stem[len(prefix) :]: p for p in directory.glob(prefix + "*.csv")}
        for kind, prefix in PREFIXES.items()
    }
    keys = set(paths["pressure"])
    if set(paths["current"]) != keys:
        raise ValueError("压力、电流记录不能完整配对")
    state_keys = pair_state_keys(keys, set(paths["state"]))
    for item in source["files"]:
        path = directory / item["name"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"本地文件与来源清单不一致: {path.name}")
    run_id, out = new_run(ROOT / "outputs", "pneumatic-audit")
    report = {
        "run_id": run_id,
        "doi": source["doi"],
        "records": [],
        "source_hash": capture_source(
            ROOT / "code/python", out / "audit-source.zip", "audit_pneumatic.py"
        ),
        "label_scope": "PLC operation states, not fault ground truth",
        "features": ["duration_s"]
        + [
            f"{c}_{f}"
            for c in ("pressure_bar", "current_A")
            for f in ("mean", "std", "p05", "p95", "end_minus_start", "diff_rms")
        ],
    }
    all_features, all_states, all_records = [], [], []
    for index, key in enumerate(sorted(keys)):
        pt, pressure = read_signal(paths["pressure"][key])
        ct, current = read_signal(paths["current"][key])
        st, states = read_signal(paths["state"][state_keys[key]])
        time_issues = {
            kind: {
                "backward_steps": int(np.sum(np.diff(t) < 0)),
                "repeated_steps": int(np.sum(np.diff(t) == 0)),
            }
            for kind, t in (("pressure", pt), ("current", ct), ("state", st))
        }
        if any(
            v["backward_steps"] or v["repeated_steps"] for v in time_issues.values()
        ):
            report["records"].append(
                {
                    "id": index,
                    "key": key,
                    "state_key": state_keys[key],
                    "day": key[:8],
                    "segments": 0,
                    "status": "excluded_nonmonotonic_time",
                    "time_issues": time_issues,
                }
            )
            continue
        common, pi, ci = np.intersect1d(pt, ct, return_indices=True)
        if len(common) < 2:
            raise ValueError(f"信号缺少对齐样本: {key}")
        features, labels, quality = summarize(
            common, pressure[pi], current[ci], st, states
        )
        report["records"].append(
            {
                "id": index,
                "key": key,
                "state_key": state_keys[key],
                "day": key[:8],
                "pressure_samples": len(pt),
                "current_samples": len(ct),
                "aligned_samples": len(common),
                "max_sample_gap_s": float(np.diff(common).max()),
                "states": sorted(set(labels.tolist())),
                "segments": len(labels),
                **quality,
            }
        )
        all_features.extend(features)
        all_states.extend(labels)
        all_records.extend([index] * len(labels))
    np.savez_compressed(
        out / "segments.npz",
        features=all_features,
        states=all_states,
        record_ids=all_records,
    )
    report["summary"] = {
        "records": len(keys),
        "segments": len(all_states),
        "days": sorted({r["day"] for r in report["records"]}),
        "states": sorted({int(v) for v in all_states}),
        "excluded_nonmonotonic_records": sum(
            r.get("status") == "excluded_nonmonotonic_time" for r in report["records"]
        ),
    }
    (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, **report["summary"]}))


if __name__ == "__main__":
    run()
