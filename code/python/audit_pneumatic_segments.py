"""按完整PLC动作段筛除时钟异常，保留其它有效动作；不修补原始时间戳。"""

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from audit_pneumatic import PREFIXES, pair_state_keys, read_signal
from phm_core.artifacts import capture_source, new_run

ROOT = Path(__file__).resolve().parents[2]


def interval_values(times, values, start, end, max_gap=0.03):
    """不排序、不去重、不插值；边缘覆盖、连续索引和段内时钟同时有效才接收。"""
    if start < times[0] or end > times[-1]:
        return None, "truncated"
    ids = np.flatnonzero((times >= start) & (times < end))
    if len(ids) < 5:
        return None, "short"
    selected = times[ids]
    if np.any(np.diff(ids) != 1) or np.any(np.diff(selected) <= 0):
        return None, "nonmonotonic"
    if (
        max(selected[0] - start, end - selected[-1], float(np.diff(selected).max()))
        > max_gap
    ):
        return None, "gap"
    return values[ids], "valid"


def segment_features(pressure, current, duration):
    row = [duration]
    for signal in (pressure, current):
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
    return row


def run():
    directory = ROOT / "data/raw/pneumatic-process"
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    for item in source["files"]:
        if (
            hashlib.sha256((directory / item["name"]).read_bytes()).hexdigest()
            != item["sha256"]
        ):
            raise ValueError(f"文件摘要不一致: {item['name']}")
    paths = {
        k: {p.stem[len(prefix) :]: p for p in directory.glob(prefix + "*.csv")}
        for k, prefix in PREFIXES.items()
    }
    if set(paths["pressure"]) != set(paths["current"]):
        raise ValueError("压力电流记录键不匹配")
    mapping = pair_state_keys(set(paths["pressure"]), set(paths["state"]))
    run_id, out = new_run(ROOT / "outputs", "pneumatic-segment-audit")
    protocol = {
        "timestamp_policy": "preserve raw sequence; no sorting, interpolation or nominal-clock replacement",
        "segment_gate": "whole PLC interval, each signal >=5 samples, contiguous raw indices, strictly increasing timestamps, internal gaps and boundary distances <=0.03s",
        "synchronization": "same PLC interval summaries, not pointwise paired waveforms",
        "labels": "PLC action states; no fault labels established",
        "feature_scope": "descriptive pressure/current features; not response-delay or leak estimates",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    helper = ROOT / "code/python/audit_pneumatic.py"
    shutil.copyfile(helper, out / helper.name)
    report = {
        "run_id": run_id,
        "protocol": protocol,
        "records": [],
        "doi": source["doi"],
        "source_hash": capture_source(
            ROOT / "code/python",
            out / "audit-source.zip",
            "audit_pneumatic_segments.py",
        ),
        "dependency_source_sha256": {
            helper.name: hashlib.sha256(helper.read_bytes()).hexdigest()
        },
    }
    features, states, record_ids, intervals = [], [], [], []
    total = Counter()
    for record, key in enumerate(sorted(mapping)):
        pt, pv = read_signal(paths["pressure"][key])
        ct, cv = read_signal(paths["current"][key])
        st, sv = read_signal(paths["state"][mapping[key]])
        counts = Counter()
        if np.any(np.diff(st) <= 0) or not np.equal(sv, np.floor(sv)).all():
            counts["invalid_plc_record"] += 1
        else:
            for index in range(len(st) - 1):
                start, end = st[index : index + 2]
                pressure, ps = interval_values(pt, pv, start, end)
                current, cs = interval_values(ct, cv, start, end)
                if ps != "valid" or cs != "valid":
                    counts[f"pressure:{ps}/current:{cs}"] += 1
                    continue
                features.append(segment_features(pressure, current, end - start))
                states.append(int(sv[index]))
                record_ids.append(record)
                intervals.append([float(start), float(end)])
                counts["valid"] += 1
            counts["terminal_state_excluded"] += 1
        total.update(counts)
        report["records"].append(
            {
                "id": record,
                "key": key,
                "day": key[:8],
                "state_key": mapping[key],
                "counts": dict(counts),
            }
        )
    report["summary"] = {
        "records": len(mapping),
        "records_with_valid_segments": sum(
            r["counts"].get("valid", 0) > 0 for r in report["records"]
        ),
        "segments": len(states),
        "counts": dict(total),
        "state_counts": dict(Counter(states)),
    }
    np.savez_compressed(
        out / "segments.npz",
        features=np.asarray(features, dtype="float32"),
        states=states,
        record_ids=record_ids,
        intervals=intervals,
    )
    (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, **report["summary"]}))


if __name__ == "__main__":
    run()
