"""气动过程动作条件化的跨日告警负担对照；PLC动作标签不是故障标签。"""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from phm_core.artifacts import capture_source, new_run

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "outputs/pneumatic-segment-audit/runs/20261031T192446Z-75d7439a"


def chronological_split(days):
    """按整日划分；末两日测试，倒数第三日校准，其余日期训练。"""
    ordered = np.unique(days)
    if len(ordered) < 4:
        raise ValueError("至少需要4个独立日期")
    return {
        "train": np.flatnonzero(np.isin(days, ordered[:-3])),
        "calibration": np.flatnonzero(days == ordered[-3]),
        "test": np.flatnonzero(np.isin(days, ordered[-2:])),
    }


def alarm_summary(scores, thresholds, ids):
    supported = np.isfinite(thresholds[ids])
    selected = ids[supported]
    return {
        "segments": len(ids),
        "supported": int(supported.sum()),
        "unsupported": int((~supported).sum()),
        "alarms": int(np.sum(scores[selected] > thresholds[selected])),
        "alarm_fraction": float(np.mean(scores[selected] > thresholds[selected]))
        if len(selected)
        else None,
    }


def fit_scores(x, states, parts, seed, conditioned):
    """只用历史训练/校准日期；支持不足的动作返回NaN，不默认为正常。"""
    scores = np.full(len(x), np.nan)
    thresholds = np.full(len(x), np.nan)
    support = {}
    groups = np.unique(states) if conditioned else [None]
    for group in groups:
        mask = states == group if group is not None else np.ones(len(x), dtype=bool)
        train = parts["train"][mask[parts["train"]]]
        calibration = parts["calibration"][mask[parts["calibration"]]]
        supported = len(train) >= 20 and len(calibration) >= 20
        support[str(group)] = {
            "train": len(train),
            "calibration": len(calibration),
            "supported": supported,
        }
        if not supported:
            continue
        model = IsolationForest(
            n_estimators=200, max_samples="auto", random_state=seed, n_jobs=1
        )
        model.fit(x[train])
        selected = np.flatnonzero(mask)
        scores[selected] = -model.score_samples(x[selected])
        threshold = float(np.quantile(scores[calibration], 0.99, method="higher"))
        thresholds[selected] = threshold
        support[str(group)]["threshold"] = threshold
    return scores, thresholds, support


def run():
    run_id, out = new_run(ROOT / "outputs", "pneumatic-state-study")
    protocol = {
        "split": "whole chronological days: all except last3 train, third-last calibration, last2 test",
        "seeds": [42, 43, 44],
        "arms": ["pooled_iforest", "plc_state_iforest"],
        "threshold": "calibration q99 higher, strict >; per-state min20 train and min20 calibration segments",
        "labels": "PLC action states, NOT healthy/fault ground truth",
        "meaning": "alarm burden and coverage only; not false-positive rate, recall or fault accuracy",
        "selection": "none; no target threshold adaptation or model promotion",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    audit = json.loads((AUDIT / "audit.json").read_text(encoding="utf-8"))
    data = np.load(AUDIT / "segments.npz")
    x, states, records = data["features"], data["states"], data["record_ids"]
    record_days = {r["id"]: r["day"] for r in audit["records"]}
    days = np.array([record_days[int(r)] for r in records])
    parts = chronological_split(days)
    report = {
        "run_id": run_id,
        "protocol": protocol,
        "input_sha256": {
            name: hashlib.sha256((AUDIT / name).read_bytes()).hexdigest()
            for name in ("audit.json", "segments.npz")
        },
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", "study_pneumatic_states.py"
        ),
        "partitions": {
            key: {
                "ids": ids.tolist(),
                "days": np.unique(days[ids]).tolist(),
                "records": len(np.unique(records[ids])),
            }
            for key, ids in parts.items()
        },
        "arms": {},
    }
    saved_scores = {}
    for conditioned, arm in enumerate(protocol["arms"]):
        report["arms"][arm] = {}
        for seed in protocol["seeds"]:
            scores, thresholds, support = fit_scores(
                x, states, parts, seed, bool(conditioned)
            )
            saved_scores[(arm, seed)] = (scores, thresholds)
            result = {
                key: alarm_summary(scores, thresholds, ids)
                for key, ids in parts.items()
            }
            result["state_support"] = support
            result["test_by_day"] = {
                day: alarm_summary(
                    scores,
                    thresholds,
                    np.intersect1d(parts["test"], np.flatnonzero(days == day)),
                )
                for day in np.unique(days[parts["test"]])
            }
            result["test_by_state"] = {
                str(state): alarm_summary(
                    scores, thresholds, parts["test"][states[parts["test"]] == state]
                )
                for state in np.unique(states[parts["test"]])
            }
            np.savez_compressed(
                out / f"{arm}-{seed}.npz",
                scores=scores,
                thresholds=thresholds,
                states=states,
                record_ids=records,
                days=days,
            )
            report["arms"][arm][str(seed)] = result
    # 在共同可评分的测试段上另列对照，避免缺少稀有动作造成表面改进。
    report["common_support_test"] = {}
    for seed in protocol["seeds"]:
        common = parts["test"]
        for arm in protocol["arms"]:
            common = common[np.isfinite(saved_scores[(arm, seed)][1][common])]
        report["common_support_test"][str(seed)] = {
            arm: alarm_summary(*saved_scores[(arm, seed)], common)
            for arm in protocol["arms"]
        }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "results": {
                    arm: {seed: item["test"] for seed, item in runs.items()}
                    for arm, runs in report["arms"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
