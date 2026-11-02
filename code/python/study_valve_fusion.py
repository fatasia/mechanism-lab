"""冻结局部AE残差与换向机理作双证据确认；各支路独立健康校准。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from phm_core.artifacts import capture_source, new_run
from phm_core.data import block_split
from phm_core.evaluation import evaluate
from phm_core.profiles import load_profile
from phm_core.valve import fit_mechanism_reference, mechanism_features, mechanism_score

ROOT = Path(__file__).resolve().parents[2]
RESIDUAL_RUN = "20261031T191349Z-791f997f"


def conjunction_score(first, second, first_threshold, second_threshold):
    """大于1等价于两支路同时超各自门槛；不是概率或因果证明。"""
    if first_threshold <= 0 or second_threshold <= 0:
        raise ValueError("双证据门槛必须为正")
    return np.minimum(first / first_threshold, second / second_threshold)


def run(residual_run_id=RESIDUAL_RUN):
    parent = ROOT / "outputs/valve-residual-study/runs" / residual_run_id
    run_id, out = new_run(ROOT / "outputs", "valve-fusion-study")
    protocol = {
        "parent": residual_run_id,
        "branches": ["mean", "top10", "local200ms", "point_scaled_top10"],
        "fusion": "AND of AE residual and timing-rule branches; each q99 on independent normal calibration; strict greater-than",
        "selection": "validation branch q99; FPR<=1%, maximize recall then AP; no eligible means null",
        "status": "exploratory follow-up on historical corpus after residual study; not new blind evidence",
        "training": "AE remains frozen; timing center and scale use only healthy training",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    x, labels, digest = load_profile(
        ROOT / "data/raw/uci-hydraulic/dataset.zip",
        "valve-switch-100hz",
        ROOT / "data/prepared",
    )
    healthy = labels[:, 1] == 100
    split = block_split(len(x))
    normal = {k: ids[healthy[ids]] for k, ids in split.items()}
    features = mechanism_features(x)
    reference = fit_mechanism_reference(features, normal["train"])
    timing = mechanism_score(features, reference)
    scores = {}
    with np.load(parent / "scores.npz") as archive:
        for arm in protocol["branches"]:
            scores[arm] = archive[arm]
    val_timing = float(np.quantile(timing[normal["validation"]], 0.99))
    cal_timing = float(np.quantile(timing[normal["calibration"]], 0.99))
    validation = {}
    for arm, score in scores.items():
        val_ae = float(np.quantile(score[normal["validation"]], 0.99))
        combined = conjunction_score(score, timing, val_ae, val_timing)
        validation[arm] = evaluate(combined, healthy, split["validation"], 1.0)
    eligible = [a for a, v in validation.items() if v["false_positive_per_1000"] <= 10]
    selected = (
        max(
            eligible,
            key=lambda a: (validation[a]["recall"], validation[a]["test_pr_auc"]),
        )
        if eligible
        else None
    )
    (out / "selection.json").write_text(
        json.dumps({"selected": selected, "validation": validation}, indent=2),
        encoding="utf-8",
    )
    report = {
        "run_id": run_id,
        "protocol": protocol,
        "data_sha256": digest,
        "selected": selected,
        "source_hash": capture_source(
            ROOT / "code/python", out / "study-source.zip", "study_valve_fusion.py"
        ),
        "parent_scores_sha256": hashlib.sha256(
            (parent / "scores.npz").read_bytes()
        ).hexdigest(),
        "mechanism_reference": reference,
        "arms": {},
    }
    exported = {"timing": timing}
    for arm, score in scores.items():
        threshold = float(np.quantile(score[normal["calibration"]], 0.99))
        combined = conjunction_score(score, timing, threshold, cal_timing)
        exported[arm] = combined
        report["arms"][arm] = {
            "ae_threshold": threshold,
            "timing_threshold": cal_timing,
            "validation": validation[arm],
            "test": evaluate(combined, healthy, split["test"], 1.0),
            "stability": {},
        }
        for stability in [0, 1]:
            ids = split["test"][labels[split["test"], 4] == stability]
            report["arms"][arm]["stability"][str(stability)] = evaluate(
                combined, healthy, ids, 1.0
            )
    np.savez_compressed(out / "scores.npz", **exported)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"run_id": run_id, "selected": selected, "results": report["arms"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-run-id", default=RESIDUAL_RUN)
    run(parser.parse_args().residual_run_id)
