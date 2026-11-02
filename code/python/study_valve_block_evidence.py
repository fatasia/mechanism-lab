"""阀门历史测试的块级置信区间与同步容差推导；只读已有运行，不训练、不改阈值。

测试集的 264 个周期实际是 11 个连续块；块（而不是周期）才是近似独立的评估单位，
因此“0 误报/0 漏报”只能支持块级的单侧上界，而不是周期级的 0%。
"""

import argparse
import hashlib
import json
from pathlib import Path

from phm_core.artifacts import capture_source, new_run
from phm_core.data import block_split
from phm_core.evidence import (
    block_summary,
    blocks,
    events_needed_zero_miss,
    hours_needed_zero_alarm,
)
from phm_core.profiles import load_profile

ROOT = Path(__file__).resolve().parents[2]


def derive_shift_tolerance(scenarios, channel_tag="ch01"):
    """两路同步偏移下，使结果相对未扰动不变的最大对称偏移（采样点）。

    正负方向都必须零决策变化；一旦某个 |k| 出现误报或漏报变化即停止。
    """
    by_shift = {}
    for name, item in scenarios.items():
        if name.startswith("shift_samples_") and name.endswith(f"_{channel_tag}"):
            by_shift[int(name.split("_")[2])] = item
    baseline = scenarios["unchanged"]
    tolerance = 0
    for size in sorted({abs(k) for k in by_shift}):
        sides = [by_shift[k] for k in (-size, size) if k in by_shift]
        if all(
            s["false_positives"] <= baseline["false_positives"]
            and s["false_negatives"] <= baseline["false_negatives"]
            and s["decision_changes"] == 0
            for s in sides
        ):
            tolerance = size
        else:
            break
    return {
        "tolerance_samples": tolerance,
        "tolerance_ms": tolerance * 10,
        "tested_shifts": sorted(by_shift),
        "per_shift": {
            str(k): {
                "false_positives": v["false_positives"],
                "false_negatives": v["false_negatives"],
            }
            for k, v in sorted(by_shift.items())
        },
    }


def run(multiseed: Path, stress: Path):
    run_id, output = new_run(ROOT / "outputs", "valve-block-evidence")
    study = json.loads((multiseed / "metrics.json").read_text(encoding="utf-8"))
    perturbation = json.loads((stress / "metrics.json").read_text(encoding="utf-8"))
    _, labels, digest = load_profile(
        ROOT / "data/raw/uci-hydraulic/dataset.zip",
        "valve-switch-100hz",
        ROOT / "data/prepared",
    )
    test_ids = block_split(len(labels))["test"]
    assert sorted(test_ids.tolist()) == sorted(study["split"]["test"])
    healthy = labels[test_ids, 1] == 100
    normal_ids, fault_ids = test_ids[healthy], test_ids[~healthy]
    arms = {}
    for arm, seeds in study["protocols"]["historical-test"]["target"].items():
        arms[arm] = {}
        for seed, result in seeds.items():
            detail = result["detail"]
            arms[arm][seed] = {
                "normal": block_summary(normal_ids, detail["false_positive_ids"]),
                "fault": block_summary(fault_ids, detail["false_negative_ids"]),
            }
    report = {
        "run_id": run_id,
        "scope": "re-analysis of already inspected historical UCI test split; clustered by contiguous cycle blocks; no new evidence about other machines",
        "sources": {
            "multiseed": multiseed.name,
            "stress": stress.name,
            "multiseed_metrics_sha256": hashlib.sha256(
                (multiseed / "metrics.json").read_bytes()
            ).hexdigest(),
            "stress_metrics_sha256": hashlib.sha256(
                (stress / "metrics.json").read_bytes()
            ).hexdigest(),
        },
        "source_hash": capture_source(
            ROOT / "code/python", output / "source.zip", "study_valve_block_evidence.py"
        ),
        "data_sha256": digest,
        "test_blocks": {
            "normal": blocks(normal_ids.tolist()),
            "fault": blocks(fault_ids.tolist()),
        },
        "arms": arms,
        "shift_tolerance": derive_shift_tolerance(perturbation["scenarios"]),
        "planning": {
            "fault_events_for_recall_lower_95pct_zero_miss": events_needed_zero_miss(
                0.95, 0.95
            ),
            "normal_hours_for_one_alarm_per_1000h_zero_alarm": hours_needed_zero_alarm(
                1.0, 0.95
            ),
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("multiseed", type=Path)
    parser.add_argument("stress", type=Path)
    args = parser.parse_args()
    run(args.multiseed, args.stress)
