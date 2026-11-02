"""电机多模态RF强基线：整工况留出、源工况选型、记录级评价。"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

from phm_core.artifacts import capture_source, new_run

ROOT = Path(__file__).resolve().parents[2]
ARMS = {
    "vibration_rf": np.r_[0:42, 87:114],
    "current_rf": np.r_[42:87],
    "fusion_no_orders_rf": np.r_[0:87],
    "fusion_rf": np.r_[0:114],
}


def record_probabilities(probabilities, record_ids, selected):
    """等权合并同一完整试验的有效窗口；无支持窗口时返回NaN而非健康。"""
    return np.asarray(
        [
            probabilities[record_ids == record].mean(axis=0)
            if np.any(record_ids == record)
            else np.full(probabilities.shape[1], np.nan)
            for record in selected
        ]
    )


def metrics(labels, probabilities, healthy, classes):
    supported = np.isfinite(probabilities).all(axis=1)
    y = labels[supported]
    predicted = probabilities[supported].argmax(axis=1)
    normal, fault = y == healthy, y != healthy
    return {
        "records": len(labels),
        "supported_records": int(supported.sum()),
        "unsupported_records": int((~supported).sum()),
        "normal": int(normal.sum()),
        "fault": int(fault.sum()),
        "false_positives": int(np.sum(normal & (predicted != healthy))),
        "false_negatives": int(np.sum(fault & (predicted == healthy))),
        "macro_f1": float(
            f1_score(
                y,
                predicted,
                labels=np.arange(classes),
                average="macro",
                zero_division=0,
            )
        )
        if len(y)
        else 0.0,
        "confusion": confusion_matrix(y, predicted, labels=np.arange(classes)).tolist(),
    }


def fit_predict(x, y, window_records, train, test, columns, seed):
    mask = np.isin(window_records, train)
    counts = np.bincount(window_records[mask], minlength=len(y))
    weights = 1 / counts[window_records[mask]]
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x[mask][:, columns], y[window_records[mask]], sample_weight=weights)
    target = np.isin(window_records, test)
    probabilities = np.zeros((target.sum(), int(y.max()) + 1))
    if target.any():
        probabilities[:, model.classes_] = model.predict_proba(x[target][:, columns])
    return model, record_probabilities(probabilities, window_records[target], test)


def source_passes(result):
    return bool(
        result["unsupported_records"] == 0
        and result["normal"] > 0
        and result["fault"] > 0
        and result["false_positives"] / result["normal"] <= 0.05
        and 1 - result["false_negatives"] / result["fault"] >= 0.95
        and result["macro_f1"] >= 0.90
    )


def run(feature_directory, task="multiclass"):
    manifest_path = feature_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arms = (
        ARMS
        if task == "multiclass"
        else {
            "all_features_binary_rf": np.arange(114),
            "mechanism_binary_rf": np.arange(84, 114),
        }
    )
    run_id, out = new_run(
        ROOT / "outputs",
        "motor-study" if task == "multiclass" else "motor-binary-study",
    )
    protocol = {
        "seeds": [42, 43, 44],
        "task": task,
        "arms": {name: columns.tolist() for name, columns in arms.items()},
        "models": "RandomForest200,max_depth14,min_leaf2,sqrt,balanced_subsample; equal total sample weight per recording",
        "target_protocols": {
            "held-out-3000rpm": "all nominal3000rpm records",
            "held-out-torque-profile": "all torque_circulation records",
        },
        "source_validation": "leave one complete source condition out; same-condition duplicate runs kept together",
        "selection_gate": "all3seeds source FPR<=5%,fault recall>=95%,task class macroF1>=90%,coverage100%; then max mean source macroF1",
        "evaluation": "record-average window probabilities then argmax; unsupported records count separately",
        "target_use": "no tuning, no calibration, no adaptation, no automatic promotion",
        "scope": "historical exploratory dataset; binary task tests fault screening, not24class identification; dimensionless arm retains only3current imbalance and27envelope order ratios; source gating precedes target evaluation",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    records = manifest["records"]
    if [r["id"] for r in records] != list(range(len(records))):
        raise ValueError("记录ID必须与审计顺序一致")
    labels = [
        r["label"]
        if task == "multiclass"
        else "health"
        if r["label"] == "health"
        else "fault"
        for r in records
    ]
    classes = sorted(set(labels))
    y = np.asarray([classes.index(label) for label in labels])
    healthy = classes.index("health")
    conditions = np.asarray([r["condition"] for r in records])
    windows, window_ids = [], []
    for record in records:
        path = feature_directory / record["cache"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["cache_sha256"]:
            raise ValueError("特征缓存哈希不一致")
        with np.load(path) as cache:
            windows.append(cache["features"])
            window_ids.extend([record["id"]] * len(cache["features"]))
    x, window_ids = np.concatenate(windows), np.asarray(window_ids)
    report = {
        "run_id": run_id,
        "protocol": protocol,
        "classes": classes,
        "healthy_class": healthy,
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", "study_motor.py"
        ),
        "feature_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "feature_run": manifest["run_id"],
        "protocols": {},
    }
    shutil.copyfile(manifest_path, out / "feature-manifest.json")
    start, fits = time.monotonic(), 0
    for name in protocol["target_protocols"]:
        target_mask = np.asarray(
            [
                r["nominal_rpm"] == 3000
                if name == "held-out-3000rpm"
                else r["profile"] == "torque"
                for r in records
            ]
        )
        source, target = np.flatnonzero(~target_mask), np.flatnonzero(target_mask)
        task = out / name
        task.mkdir()
        result = {
            "source_ids": source.tolist(),
            "target_test_ids": target.tolist(),
            "source_folds": {},
            "source_summary": {},
            "target": {},
        }
        for arm, columns in arms.items():
            result["source_folds"][arm] = []
            source_metrics = {}
            for seed in protocol["seeds"]:
                truth, predictions = [], []
                for fold, condition in enumerate(np.unique(conditions[source])):
                    train = source[conditions[source] != condition]
                    validation = source[conditions[source] == condition]
                    _, probabilities = fit_predict(
                        x, y, window_ids, train, validation, columns, seed
                    )
                    fits += 1
                    item = metrics(y[validation], probabilities, healthy, len(classes))
                    result["source_folds"][arm].append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "train_ids": train.tolist(),
                            "test_ids": validation.tolist(),
                            "metrics": item,
                        }
                    )
                    np.savez_compressed(
                        task / f"source-{arm}-{seed}-{fold}.npz",
                        record_ids=validation,
                        labels=y[validation],
                        probabilities=probabilities,
                    )
                    truth.extend(y[validation])
                    predictions.extend(probabilities)
                    (out / "progress.json").write_text(
                        json.dumps(
                            {
                                "fits": fits,
                                "protocol": name,
                                "arm": arm,
                                "seed": seed,
                                "stage": "source_validation",
                                "elapsed_s": time.monotonic() - start,
                            }
                        ),
                        encoding="utf-8",
                    )
                    print(name, arm, seed, condition, item["macro_f1"], flush=True)
                source_metrics[str(seed)] = metrics(
                    np.asarray(truth), np.asarray(predictions), healthy, len(classes)
                )
            result["source_summary"][arm] = source_metrics
        eligible = [
            arm
            for arm in arms
            if all(source_passes(v) for v in result["source_summary"][arm].values())
        ]
        result["selected"] = (
            max(
                eligible,
                key=lambda arm: np.mean(
                    [v["macro_f1"] for v in result["source_summary"][arm].values()]
                ),
            )
            if eligible
            else None
        )
        # 源工况结果和选择先落盘，之后才评估对应目标工况。
        (task / "source-selection.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        for arm, columns in arms.items():
            result["target"][arm] = {}
            for seed in protocol["seeds"]:
                model, probabilities = fit_predict(
                    x, y, window_ids, source, target, columns, seed
                )
                fits += 1
                result["target"][arm][str(seed)] = metrics(
                    y[target], probabilities, healthy, len(classes)
                )
                np.savez_compressed(
                    task / f"target-{arm}-{seed}.npz",
                    record_ids=target,
                    labels=y[target],
                    probabilities=probabilities,
                )
                joblib.dump(model, task / f"{arm}-{seed}.joblib", compress=3)
        report["protocols"][name] = result
        (out / "progress-metrics.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    report["fits"] = fits
    report["elapsed_s"] = time.monotonic() - start
    (out / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "fits": fits,
                "selected": {n: p["selected"] for n, p in report["protocols"].items()},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument(
        "--task", choices=["multiclass", "binary"], default="multiclass"
    )
    args = parser.parse_args()
    run(args.features.resolve(), args.task)
