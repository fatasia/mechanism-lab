"""按固定信号配置执行部件实验；完整周期块隔离切分。"""

import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from phm_core.artifacts import capture_source, new_run, publish_run
from phm_core.data import block_split
from phm_core.evaluation import evaluate
from phm_core.export import export_models
from phm_core.features import RobustBaseline, cycle_features
from phm_core.learning import fit_autoencoder
from phm_core.profiles import PROFILES, load_profile
from phm_core.valve import fit_mechanism_reference, mechanism_features, mechanism_score

ROOT = Path(__file__).resolve().parents[2]


def main(
    task: str = "valve",
    profile_name: str = "process-10hz",
    epochs: int = 40,
    architecture: str = "conv",
    normal_policy: str = "stable",
    seed: int = 42,
    selection_study_run: str | None = None,
) -> None:
    if epochs < 1:
        raise ValueError("epochs 必须为正数")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    profile = PROFILES[profile_name]
    channels = list(profile.channels)
    x, labels, data_hash = load_profile(
        ROOT / "data/raw/uci-hydraulic/dataset.zip",
        profile_name,
        ROOT / "data/prepared",
    )
    column, normal = {
        "cooler": (0, 100),
        "valve": (1, 100),
        "pump": (2, 0),
        "accumulator": (3, 130),
    }[task]
    healthy = labels[:, column] == normal
    stable = labels[:, 4] == 0
    # 随机分配完整块，块内周期保序；标签不进入网络输入。
    split = block_split(len(x))
    eligible = healthy & stable if normal_policy == "stable" else healthy
    train_ids = split["train"][eligible[split["train"]]]
    val_ids = split["validation"]
    normal_val_ids = val_ids[eligible[val_ids]]
    calibration_ids = split["calibration"][eligible[split["calibration"]]]
    test_ids = split["test"]
    run_id, out = new_run(ROOT / "outputs", task)
    source_hash = capture_source(ROOT / "code/python", out / "train-source.zip")
    (out / "split.json").write_text(
        json.dumps({k: v.tolist() for k, v in split.items()}), encoding="utf-8"
    )
    if len(train_ids) < 10 or min(len(normal_val_ids), len(calibration_ids)) < 5:
        raise ValueError(
            f"块切分健康样本不足：训练 {len(train_ids)}、验证 {len(normal_val_ids)}、校准 {len(calibration_ids)}"
        )
    model, mean, scale, ae, reconstruction, history = fit_autoencoder(
        x, train_ids, normal_val_ids, architecture, seed, epochs
    )
    torch.save(model.state_dict(), out / "ae.pt")
    features = cycle_features(x)
    baseline = RobustBaseline().fit(features[train_ids])
    forest = IsolationForest(n_estimators=100, random_state=42).fit(features[train_ids])
    classifier = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    ).fit(features[split["train"]], (~healthy[split["train"]]).astype("int64"))
    scores = {
        "mad": baseline.score(features),
        "iforest": -forest.decision_function(features),
        "ae": ae,
        "rf": classifier.predict_proba(features)[:, 1],
    }
    report = {
        "task": task,
        "run_id": run_id,
        "profile": profile_name,
        "sample_rate_hz": profile.rate,
        "units": list(profile.units),
        "cycle_count": len(x),
        "config": {
            "epochs": epochs,
            "learning_rate": 0.001,
            "batch_size": 32,
            "patience": 6,
            "calibration_quantile": 0.99,
        },
        "source_hash": source_hash,
        "split_hash": hashlib.sha256((out / "split.json").read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "metric_definition": "average_precision_score; not trapezoidal PR-AUC",
        "evaluation_status": "exploratory; same-rig block split; not an untouched external validation set",
        "seed": seed,
        "architecture": architecture,
        "normal_policy": normal_policy,
        "window_start_seconds": profile.start,
        "window_seconds": profile.end - profile.start,
        "protocol": "30-cycle blocks, 3-cycle boundary purge; separate train/validation/calibration/test; same rig, not cross-machine or future prediction",
        "scope": "Component-labelled normal; other component conditions vary. Not RUL.",
        "selection_study_run": selection_study_run,
        "source": "UCI Hydraulic Systems; DOI 10.24432/C5CW21; CC BY 4.0",
        "data_sha256": data_hash,
        "channels": channels,
        "train_healthy_cycles": len(train_ids),
        "calibration_healthy_cycles": len(calibration_ids),
        "validation_healthy_cycles": len(normal_val_ids),
        "split_counts": {k: len(v) for k, v in split.items()},
        "history": history,
        "models": {},
    }
    for name, score in scores.items():
        threshold = float(np.quantile(score[calibration_ids], 0.99))
        report["models"][name] = {
            "threshold": threshold,
            **evaluate(score, healthy, test_ids, threshold),
        }
        stable_test = test_ids[stable[test_ids]]
        report["models"][name]["stable_test"] = evaluate(
            score, healthy, stable_test, threshold
        )
    extra_scores = {}
    if profile_name == "valve-switch-100hz":
        timings = mechanism_features(x)
        reference = fit_mechanism_reference(timings, train_ids)
        rule = mechanism_score(timings, reference)
        reference["threshold"] = float(np.quantile(rule[calibration_ids], 0.99))
        (out / "mechanism-reference.json").write_text(
            json.dumps(reference, indent=2), encoding="utf-8"
        )
        report["mechanism_rule"] = evaluate(
            rule, healthy, test_ids, reference["threshold"]
        )
        extra_scores["mechanism_score"] = rule
    np.savez(out / "scores.npz", **scores, healthy=healthy, mean=mean, scale=scale)
    np.savez_compressed(
        out / "replay.npz",
        x=x,
        reconstruction=reconstruction,
        healthy=healthy,
        labels=labels,
        **scores,
        **extra_scores,
    )
    (out / "baseline.json").write_text(
        json.dumps(
            {"center": baseline.center.tolist(), "scale": baseline.scale.tolist()}
        ),
        encoding="utf-8",
    )
    export_models(
        out,
        model,
        forest,
        classifier,
        x,
        features.astype("float32"),
        mean,
        scale,
        scores,
        test_ids,
        report,
    )
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copyfile(out / "metrics.json", out / "bundle/metrics.json")
    shutil.make_archive(str(out / "model-bundle"), "zip", out / "bundle")
    publish_run(ROOT / "outputs", task, run_id)
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=["valve", "cooler", "pump", "accumulator"], default="valve"
    )
    parser.add_argument("--profile", choices=list(PROFILES), default="process-10hz")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--architecture", choices=["conv", "dense"], default="conv")
    parser.add_argument("--normal-policy", choices=["stable", "all"], default="stable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-study-run")
    args = parser.parse_args()
    main(
        args.task,
        args.profile,
        args.epochs,
        args.architecture,
        args.normal_policy,
        args.seed,
        args.selection_study_run,
    )
