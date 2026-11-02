"""执行器误报来源分解：同一条健康记录在不同参考抽样下是否稳定误报。

只用已有 Cranfield 数据，不训练分类器。对每个工况反复随机抽取 k 条健康参考，
计算每条记录对该参考的绝对标准化偏差；阈值取其它工况健康记录的 q95（整工况留出）。
若某健康记录无论抽到哪些参考都超阈值，说明是记录自身偏差而不是参考估计噪声。
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from phm_core.actuator import CLASSES, load_records
from phm_core.artifacts import capture_source, new_run
from phm_core.evidence import (
    clopper_pearson,
    one_sided_upper,
    variance_components,
)

ROOT = Path(__file__).resolve().parents[2]
SCALE_FLOOR = np.array([0.01, 0.005])  # 跟随误差 mm、电流 A 的尺度下限，非传感器精度


def record_scores(windows, record_ids, reference_records, target_records):
    """返回 [目标记录, 2]：跟随误差、电流相对参考相位模板的平均绝对标准化偏差。"""
    reference = windows[np.isin(record_ids, reference_records)][:, :, 1:]
    center = np.median(reference, axis=0)
    scale = np.maximum(
        1.4826 * np.median(np.abs(reference - center), axis=(0, 1)), SCALE_FLOOR
    )
    scores = np.empty((len(target_records), 2))
    for row, record in enumerate(target_records):
        cycles = windows[record_ids == record][:, :, 1:]
        scores[row] = (np.abs(cycles - center) / scale).mean(axis=(0, 1))
    return scores


def contexts_of(metadata):
    result = {}
    for item in metadata:
        result.setdefault(f"{item['profile']}/{item['load']}", []).append(item["id"])
    return result


def run_draws(windows, record_ids, metadata, budget, draws, seed):
    """每次抽样为每个工况各抽 k 条健康参考；返回逐抽样的记录得分表。"""
    rng = np.random.default_rng(seed)
    contexts = contexts_of(metadata)
    label = {m["id"]: m["label"] for m in metadata}
    healthy = {c: [i for i in ids if label[i] == 0] for c, ids in contexts.items()}
    table = np.full((draws, len(metadata), 2), np.nan)
    in_reference = np.zeros((draws, len(metadata)), dtype=bool)
    for draw in range(draws):
        for context, ids in contexts.items():
            chosen = rng.choice(healthy[context], size=budget, replace=False)
            in_reference[draw, chosen] = True
            targets = [i for i in ids if i not in set(chosen.tolist())]
            table[draw, targets] = record_scores(windows, record_ids, chosen, targets)
    return table, in_reference


def evaluate(table, in_reference, metadata, quantile=0.95):
    """整工况留出阈值：阈值来自其它工况健康记录，只应用于被留出的工况。"""
    label = np.array([m["label"] for m in metadata])
    context = np.array([f"{m['profile']}/{m['load']}" for m in metadata])
    combined = table.max(axis=2)  # 误差或电流任一偏离即报警
    alarms = np.full(combined.shape, np.nan)
    for held in np.unique(context):
        source = (label == 0) & (context != held)
        for draw in range(combined.shape[0]):
            values = combined[draw, source]
            threshold = np.quantile(values[~np.isnan(values)], quantile)
            target = context == held
            alarms[draw, target] = combined[draw, target] > threshold
    return combined, alarms


def channel_breakdown(table, record_ids, metadata):
    """持续误报记录的分通道得分，对照同工况其它健康记录的中位数。"""
    context = {m["id"]: f"{m['profile']}/{m['load']}" for m in metadata}
    healthy = [m["id"] for m in metadata if m["label"] == 0]
    mean_scores = np.nanmean(table, axis=0)  # [记录, 通道]
    result = {}
    for record in record_ids:
        peers = [i for i in healthy if context[i] == context[record] and i != record]
        peer_median = np.median(mean_scores[peers], axis=0)
        result[int(record)] = {
            "key": metadata[record]["key"],
            "context": context[record],
            "error_score": float(mean_scores[record, 0]),
            "current_score": float(mean_scores[record, 1]),
            "peer_median_error_score": float(peer_median[0]),
            "peer_median_current_score": float(peer_median[1]),
        }
    return result


def summarise(combined, alarms, in_reference, metadata, budget):
    label = np.array([m["label"] for m in metadata])
    fault_file = np.array([m["file"] for m in metadata])
    healthy_ids = np.where(label == 0)[0]
    evaluated = ~in_reference
    per_record_rate = {}
    for record in healthy_ids:
        used = evaluated[:, record] & ~np.isnan(alarms[:, record])
        per_record_rate[int(record)] = (
            float(alarms[used, record].mean()) if used.any() else None
        )
    rates = np.array([v for v in per_record_rate.values() if v is not None])
    healthy_scores = combined[:, healthy_ids]
    components = variance_components(healthy_scores.T)
    draw_fpr = np.array(
        [
            np.nanmean(alarms[d, healthy_ids][evaluated[d, healthy_ids]])
            for d in range(alarms.shape[0])
        ]
    )
    fault_ids = np.where(label != 0)[0]
    by_file = {}
    for name in sorted(set(fault_file[fault_ids])):
        ids = fault_ids[fault_file[fault_ids] == name]
        by_file[name] = float(np.nanmean(alarms[:, ids]))
    persistent = [i for i, v in per_record_rate.items() if v is not None and v >= 0.8]
    return {
        "reference_budget": budget,
        "healthy_records": len(healthy_ids),
        "mean_false_positive_rate": float(np.nanmean(draw_fpr)),
        "draw_fpr_p05_p95": [float(np.quantile(draw_fpr, q)) for q in (0.05, 0.95)],
        "records_never_alarmed": int(np.sum(rates == 0)),
        "records_alarmed_in_at_least_80pct_of_draws": persistent,
        "variance_components": components,
        "fault_recall_by_file": by_file,
        "fault_recall_mean": float(np.nanmean(alarms[:, fault_ids])),
        "record_level_upper_bound_if_no_persistent": one_sided_upper(
            len(persistent), len(healthy_ids)
        ),
        "record_level_interval_persistent": clopper_pearson(
            len(persistent), len(healthy_ids)
        ),
        "per_record_alarm_rate": per_record_rate,
    }


def run(budgets, draws, seed):
    directory = ROOT / "data/raw/cranfield-actuator"
    windows, _, record_ids, metadata = load_records(directory)
    run_id, output = new_run(ROOT / "outputs", "actuator-false-alarm-decomposition")
    report = {
        "run_id": run_id,
        "classes": CLASSES,
        "protocol": {
            "question": "healthy false alarms: persistent record deviation vs reference estimation noise",
            "score": "mean absolute standardized deviation from the phase template of k healthy references, error or current channel, max of the two",
            "threshold": "q95 of healthy scores from OTHER contexts in the same draw (held-out-context), frozen before scoring the held-out context",
            "draws": draws,
            "seed": seed,
            "budgets": list(budgets),
            "scope": "historical Cranfield only; 60 healthy records from one Normal.mat; exploratory, not a blind test; no classifier, no promotion",
        },
        "source_hash": capture_source(
            ROOT / "code/python",
            output / "source.zip",
            "study_false_alarm_decomposition.py",
        ),
        "data_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob("*.mat"))
        },
        "budgets": {},
    }
    for budget in budgets:
        table, in_reference = run_draws(
            windows, record_ids, metadata, budget, draws, seed
        )
        combined, alarms = evaluate(table, in_reference, metadata)
        summary = summarise(combined, alarms, in_reference, metadata, budget)
        summary["persistent_channel_breakdown"] = channel_breakdown(
            table, summary["records_alarmed_in_at_least_80pct_of_draws"], metadata
        )
        report["budgets"][str(budget)] = summary
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    run(args.budgets, args.draws, args.seed)
