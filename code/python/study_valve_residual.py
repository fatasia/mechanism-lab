"""冻结阀门AE权重，比较局部残差聚合；验证选型先于历史测试报告。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from phm_core.artifacts import capture_source, new_run
from phm_core.data import block_split
from phm_core.deep import DenseAutoencoder
from phm_core.evaluation import evaluate
from phm_core.profiles import load_profile

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "20261031T052816Z-a59e08dc"


def residual_scores(residual, training_ids):
    """输入为通道标准化绝对残差；只由健康训练残差估计逐点尺度。"""
    training = residual[training_ids]
    # 健康重构难度不均：逐位置q90作尺度。全局中位数的10%为固定数值下限。
    point_scale = np.maximum(
        np.quantile(training, 0.90, axis=0), max(float(np.median(training)) * 0.1, 1e-6)
    )
    flat = residual.reshape(len(residual), -1)
    local = residual.reshape(len(residual), 20, 20, 2).mean(axis=(2, 3))
    whitened = (residual / point_scale).reshape(len(residual), -1)
    return {
        "mean": flat.mean(1),
        "top10": np.sort(flat, axis=1)[:, -80:].mean(1),
        "local200ms": local.max(1),
        "point_scaled_top10": np.sort(whitened, axis=1)[:, -80:].mean(1),
    }, point_scale


def run(baseline_run_id=BASELINE):
    torch.set_num_threads(2)
    baseline = ROOT / "outputs/valve/runs" / baseline_run_id
    run_id, out = new_run(ROOT / "outputs", "valve-residual-study")
    protocol = {
        "baseline": baseline_run_id,
        "weights_frozen": True,
        "arms": ["mean", "top10", "local200ms", "point_scaled_top10"],
        "point_scale": "healthy train pointwise q90, floor max(healthy train global median*0.1,1e-6)",
        "threshold": "each arm own independent healthy calibration q99",
        "selection": "validation healthy q99; validation normal FPR<=5%; max validation recall then AP; no eligible means null",
        "evaluation": "historical same-rig test, already seen; not new blind validation",
        "interpretation": "postprocessing of frozen deep AE, not a new training-loss optimization",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    x, labels, digest = load_profile(
        ROOT / "data/raw/uci-hydraulic/dataset.zip",
        "valve-switch-100hz",
        ROOT / "data/prepared",
    )
    split = block_split(len(x))
    healthy = labels[:, 1] == 100
    normal = {k: ids[healthy[ids]] for k, ids in split.items()}
    normalization = json.loads(
        (baseline / "preprocessing.json").read_text(encoding="utf-8")
    )
    z = ((x - normalization["mean"]) / normalization["scale"]).astype("float32")
    model = DenseAutoencoder(x.shape[1:])
    model.load_state_dict(
        torch.load(baseline / "ae.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        reconstructed = np.concatenate(
            [model(batch).numpy() for batch in torch.from_numpy(z).split(64)]
        )
    residual = np.abs(reconstructed - z)
    scores, point_scale = residual_scores(residual, normal["train"])
    with np.load(baseline / "scores.npz") as stored:
        # 冻结均值残差必须复现既有训练产物，否则不继续比较。
        np.testing.assert_allclose(scores["mean"], stored["ae"], atol=1e-5, rtol=1e-4)
    validation = {}
    for arm, score in scores.items():
        threshold = float(np.quantile(score[normal["validation"]], 0.99))
        validation[arm] = {
            "threshold": threshold,
            **evaluate(score, healthy, split["validation"], threshold),
        }
    eligible = [a for a, v in validation.items() if v["false_positive_per_1000"] <= 50]
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
        "source_hash": capture_source(
            ROOT / "code/python", out / "train-source.zip", "study_valve_residual.py"
        ),
        "baseline_weights_sha256": hashlib.sha256(
            (baseline / "ae.pt").read_bytes()
        ).hexdigest(),
        "selected": selected,
        "arms": {},
    }
    for arm, score in scores.items():
        threshold = float(np.quantile(score[normal["calibration"]], 0.99))
        report["arms"][arm] = {
            "threshold": threshold,
            "validation": validation[arm],
            "test": evaluate(score, healthy, split["test"], threshold),
            "severity": {},
        }
        for state in [100, 90, 80, 73]:
            ids = split["test"][labels[split["test"], 1] == state]
            report["arms"][arm]["severity"][str(state)] = {
                "cycles": len(ids),
                "alerts": int((score[ids] > threshold).sum()),
            }
    np.savez_compressed(out / "scores.npz", point_scale=point_scale, **scores)
    (out / "split.json").write_text(
        json.dumps({k: v.tolist() for k, v in split.items()}, indent=2),
        encoding="utf-8",
    )
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "selected": selected,
                "tests": {a: r["test"] for a, r in report["arms"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-id", default=BASELINE)
    run(parser.parse_args().baseline_run_id)
