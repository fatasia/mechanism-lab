"""预先固定的阀门消融：按验证集选择，随后统一报告历史测试集。"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from phm_core.artifacts import capture_source, new_run, publish_run, run_path
from phm_core.data import block_split
from phm_core.evaluation import evaluate
from phm_core.export import DeployAE
from phm_core.inference import cpu_session
from phm_core.learning import fit_autoencoder
from phm_core.profiles import load_profile
from phm_core.valve import (
    TIMING_NAMES,
    choose_waveform,
    fit_mechanism_reference,
    mechanism_features,
    mechanism_score,
)

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)
ARMS = {
    "roi_conv_stable": ("conv", "stable"),
    "roi_conv_all": ("conv", "all"),
    "roi_dense_all": ("dense", "all"),
    "timing_ae": ("dense", "features"),
}


class FeatureScore(torch.nn.Module):
    def __init__(self, model, mean, scale):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.from_numpy(mean))
        self.register_buffer("scale", torch.from_numpy(scale))

    def forward(self, x):
        z = (x - self.mean) / self.scale
        return (self.model(z) - z).abs().mean(1, keepdim=True)


def report_arm(score, healthy, labels, split):
    validation = split["validation"]
    cal = split["calibration"]
    val_threshold = float(np.quantile(score[validation[healthy[validation]]], 0.99))
    threshold = float(np.quantile(score[cal[healthy[cal]]], 0.99))
    result = {
        "threshold": threshold,
        "validation_threshold": val_threshold,
        "validation": evaluate(score, healthy, validation, val_threshold),
        "test": evaluate(score, healthy, split["test"], threshold),
        "slices": {},
    }
    for flag in [0, 1]:
        ids = split["test"][labels[split["test"], 4] == flag]
        result["slices"][f"stability_{flag}"] = evaluate(score, healthy, ids, threshold)
    for state in [90, 80, 73]:
        ids = split["test"][labels[split["test"], 1] == state]
        result["slices"][f"fault_{state}"] = {
            "cycles": len(ids),
            "detected": int((score[ids] > threshold).sum()),
        }
    return result


def run(baseline_id: str | None = None, epochs: int = 80):
    if epochs < 1:
        raise ValueError("epochs 必须为正数")
    baseline = run_path(ROOT / "outputs", "valve", baseline_id)
    baseline_metrics = json.loads(
        (baseline / "metrics.json").read_text(encoding="utf-8")
    )
    if baseline_metrics["profile"] != "process-10hz":
        raise ValueError("请指定原 process-10hz 整周期基线的 --baseline-run-id")
    x, labels, digest = load_profile(
        ROOT / "data/raw/uci-hydraulic/dataset.zip",
        "valve-switch-100hz",
        ROOT / "data/prepared",
    )
    healthy = labels[:, 1] == 100
    split = block_split(len(x))
    features = mechanism_features(x)
    normal = {k: ids[healthy[ids]] for k, ids in split.items()}
    stable = {k: ids[healthy[ids] & (labels[ids, 4] == 0)] for k, ids in split.items()}
    run_id, out = new_run(ROOT / "outputs", "valve-study")
    source = capture_source(
        ROOT / "code/python", out / "train-source.zip", "study_valve.py"
    )
    protocol = {
        "baseline_run_id": baseline.name,
        "seeds": SEEDS,
        "epochs": epochs,
        "split_seed": 42,
        "window": [8, 12],
        "sample_rate_hz": 100,
        "channels": ["PS1", "PS2"],
        "arms": ARMS,
        "selection": "waveform AE family: mean validation recall at validation healthy q99, mean AP tie-break; seed42 fixed deployment; no test-based choice",
        "calibration": "separate all-normal calibration q99, same policy across arms",
        "evaluation_status": "historical same-rig test already seen; exploratory, not blind",
        "feature_resolution_floor_seconds": 0.01,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    report = {
        "run_id": run_id,
        "source_hash": source,
        "data_sha256": digest,
        "protocol": protocol,
        "counts": {
            k: {
                "all": len(ids),
                "healthy": len(normal[k]),
                "stable_healthy": len(stable[k]),
            }
            for k, ids in split.items()
        },
        "arms": {},
    }
    scores = {}
    with np.load(baseline / "replay.npz") as data:
        old = data["ae"]
        scores["archived_full_cycle"] = old
        reference = report_arm(old, healthy, labels, split)
        reference["original_operating_point"] = evaluate(
            old, healthy, split["test"], baseline_metrics["models"]["ae"]["threshold"]
        )
        report["arms"]["archived_full_cycle"] = {"seeds": {"42": reference}}
    # 先拟合全部候选并只计算验证集选择依据；测试标签不参与拟合与选择。
    models = {}
    for arm, (architecture, policy) in ARMS.items():
        report["arms"][arm] = {"seeds": {}}
        for seed in SEEDS:
            eligible = stable if policy == "stable" else normal
            values = features if policy == "features" else x
            model, mean, scale, score, _reconstruction, history = fit_autoencoder(
                values,
                eligible["train"],
                eligible["validation"],
                architecture,
                seed,
                epochs,
                scale_floor=0.01 if policy == "features" else None,
            )
            key = f"{arm}_{seed}"
            scores[key] = score
            # 保留模型与配置，避免为部署另选“最好种子”。
            torch.save(
                {"weights": model.state_dict(), "mean": mean, "scale": scale},
                out / f"{key}.pt",
            )
            models[(arm, seed)] = (model, mean, scale)
            v = split["validation"]
            th = float(np.quantile(score[normal["validation"]], 0.99))
            report["arms"][arm]["seeds"][str(seed)] = {
                "validation": evaluate(score, healthy, v, th),
                "history": history,
            }
    wave = [key for key in ARMS if key != "timing_ae"]

    selected = choose_waveform(
        {
            arm: [report["arms"][arm]["seeds"][str(s)]["validation"] for s in SEEDS]
            for arm in wave
        }
    )
    report["selected_waveform"] = {
        "arm": selected,
        "architecture": ARMS[selected][0],
        "normal_policy": ARMS[selected][1],
        "seed": 42,
    }
    (out / "selection.json").write_text(
        json.dumps(report["selected_waveform"], indent=2), encoding="utf-8"
    )
    # 到这里模型族已经固定，之后才汇总候选的测试结果。
    for arm in ARMS:
        for seed in SEEDS:
            history = report["arms"][arm]["seeds"][str(seed)]["history"]
            report["arms"][arm]["seeds"][str(seed)] = {
                **report_arm(scores[f"{arm}_{seed}"], healthy, labels, split),
                "history": history,
            }
        test = [report["arms"][arm]["seeds"][str(seed)]["test"] for seed in SEEDS]
        report["arms"][arm]["summary"] = {
            m: {
                "mean": float(np.mean([v[m] for v in test])),
                "min": float(min(v[m] for v in test)),
                "max": float(max(v[m] for v in test)),
            }
            for m in [
                "test_pr_auc",
                "precision",
                "recall",
                "false_positive_per_1000",
                "false_positives",
                "false_negatives",
            ]
        }
    mechanism = fit_mechanism_reference(features, normal["train"])
    rule = mechanism_score(features, mechanism)
    scores["timing_rules"] = rule
    report["arms"]["timing_rules"] = {
        "seeds": {"42": report_arm(rule, healthy, labels, split)}
    }
    report["mechanism_reference"] = mechanism
    bundle = out / "bundle"
    bundle.mkdir()
    ids = split["test"][np.linspace(0, len(split["test"]) - 1, 64, dtype=int)]
    contracts = {}
    for arm in ARMS:
        model, mean, scale = models[(arm, 42)]
        is_feature = arm == "timing_ae"
        values = features if is_feature else x
        deploy = (
            FeatureScore(model, mean, scale)
            if is_feature
            else DeployAE(model, mean, scale)
        )
        graph = bundle / f"{arm}.onnx"
        torch.onnx.export(
            deploy.eval(),
            torch.from_numpy(values[:1]),
            str(graph),
            input_names=["values"],
            output_names=["score"],
            dynamic_axes={"values": {0: "batch"}, "score": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        session = cpu_session(graph)
        actual = session.run(None, {"values": values[ids]})[0].ravel()
        np.testing.assert_allclose(
            actual, scores[f"{arm}_42"][ids], atol=1e-5, rtol=1e-4
        )
        report["arms"][arm]["onnx_max_abs_error"] = float(
            np.max(np.abs(actual - scores[f"{arm}_42"][ids]))
        )
        contracts[arm] = {
            "input_kind": "mechanism_features" if is_feature else "pressure_window",
            "shape": [None, *values.shape[1:]],
            "threshold": report["arms"][arm]["seeds"]["42"]["threshold"],
            "high_is_anomalous": True,
            "seed": 42,
        }
    (bundle / "contracts.json").write_text(
        json.dumps(contracts, indent=2), encoding="utf-8"
    )
    (bundle / "mechanism-reference.json").write_text(
        json.dumps(mechanism, indent=2), encoding="utf-8"
    )
    (bundle / "feature-recipe.txt").write_text(
        "PS1 inlet drop and PS2 outlet rise; raw 100 Hz crop [8,12)s. Compute first 10/50/90% crossing against first/last 1s means; no crossing ->12s; amplitude floor 1 bar. Feature order: "
        + ",".join(TIMING_NAMES)
        + "\nSee phm_core.valve.mechanism_features; feature AE consumes seconds, not raw pressure. No labels or stability flag at inference.",
        encoding="utf-8",
    )
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / "scores.npz", healthy=healthy, features=features, **scores
    )
    shutil.copyfile(out / "metrics.json", bundle / "metrics.json")
    shutil.copyfile(out / "protocol.json", bundle / "protocol.json")
    shutil.make_archive(str(out / "model-bundle"), "zip", bundle)
    publish_run(ROOT / "outputs", "valve-study", run_id)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "selected_waveform": report["selected_waveform"],
                "results": {
                    a: v.get("summary", v["seeds"]["42"]["test"])
                    for a, v in report["arms"].items()
                },
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-id")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    run(args.baseline_run_id, args.epochs)
