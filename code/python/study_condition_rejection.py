"""完整记录级工况拒识：源校准固定阈值，报告拒识覆盖与全分母检出下界。"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from phm_core.actuator import aggregate, load_records, motion_features, split_records
from phm_core.artifacts import capture_source, new_run
from public_data import sha256

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = {
    "seeds": [42, 43, 44],
    "rejection_quantiles": [0.95, 0.99],
    "trees": 200,
    "commissioning_healthy_per_condition": 2,
    "scope": "record-level exploratory baseline, no target-label tuning; not comparable to earlier window-trained RF ranking",
    "rejection": "nearest source training record in robust-scaled feature space; threshold on source calibration only",
}


def rejection_metrics(y, predicted, accepted):
    y, predicted, accepted = (
        np.asarray(y),
        np.asarray(predicted),
        np.asarray(accepted, dtype=bool),
    )
    normal, fault = y == 0, y != 0

    def ratio(n, d):
        return float(n / d) if d else None

    return {
        "records": len(y),
        "accepted": int(accepted.sum()),
        "coverage": float(accepted.mean()),
        "normal": int(normal.sum()),
        "fault": int(fault.sum()),
        "accepted_normal": int((accepted & normal).sum()),
        "accepted_fault": int((accepted & fault).sum()),
        "false_positives": int((accepted & normal & (predicted != 0)).sum()),
        "accepted_normal_fpr": ratio(
            (accepted & normal & (predicted != 0)).sum(), (accepted & normal).sum()
        ),
        "accepted_fault_recall": ratio(
            (accepted & fault & (predicted != 0)).sum(), (accepted & fault).sum()
        ),
        "all_fault_detected_lower_bound": ratio(
            (accepted & fault & (predicted != 0)).sum(), fault.sum()
        ),
        "rejected_faults": int((~accepted & fault).sum()),
        "accepted_macro_f1": float(
            f1_score(
                y[accepted],
                predicted[accepted],
                labels=np.unique(y),
                average="macro",
                zero_division=0,
            )
        )
        if accepted.any()
        else None,
    }


def compare(
    x, y, contexts, train, calibration, target, references=None, infer_context=False
):
    if set(train) & set(calibration) or set(target) & (set(train) | set(calibration)):
        raise ValueError("完整记录分区重叠")
    if references is not None and set(references) & (
        set(target) | set(train) | set(calibration)
    ):
        raise ValueError("投运参考与评分分区重叠")
    transformed = x.copy()
    context_accuracy = None
    if infer_context:
        if references is None:
            raise ValueError("推断工况实验要求显式投运健康参考")
        healthy = np.r_[train[y[train] == 0], references]
        names = np.unique(contexts[healthy])
        centers = np.array(
            [np.median(x[healthy[contexts[healthy] == c]], axis=0) for c in names]
        )
        raw_scaler = RobustScaler().fit(x[train])
        prototype_model = NearestNeighbors(n_neighbors=1).fit(
            raw_scaler.transform(centers)
        )
        assigned = prototype_model.kneighbors(
            raw_scaler.transform(x), return_distance=False
        )[:, 0]
        transformed = x - centers[assigned]
        context_accuracy = float(np.mean(names[assigned[target]] == contexts[target]))
        # 不用目标真实工况选择基准；白名单检查推断工况，距离门控另行报告。
        contexts = names[assigned]
    elif references is not None:
        for context in np.unique(contexts):
            ids = (
                np.asarray(references)
                if context in contexts[references]
                else np.asarray(train)
            )
            ids = ids[(y[ids] == 0) & (contexts[ids] == context)]
            if not len(ids):
                raise ValueError("缺少工况健康参考")
            transformed[contexts == context] -= np.median(x[ids], axis=0)
    scaler = RobustScaler().fit(transformed[train])
    scaled = scaler.transform(transformed)
    neighbors = NearestNeighbors(n_neighbors=1).fit(scaled[train])
    distances = neighbors.kneighbors(scaled, return_distance=True)[0][:, 0]
    rows = []
    for seed in PROTOCOL["seeds"]:
        model = RandomForestClassifier(
            n_estimators=PROTOCOL["trees"],
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
        model.fit(transformed[train], y[train])
        predicted = model.predict(transformed[target])
        rows.append(
            {
                "seed": seed,
                "gate": "none",
                **rejection_metrics(
                    y[target], predicted, np.ones(len(target), dtype=bool)
                ),
            }
        )
        declared = (
            contexts[train]
            if references is None
            else np.r_[contexts[train], contexts[references]]
        )
        rows.append(
            {
                "seed": seed,
                "gate": "declared-context-only",
                **rejection_metrics(
                    y[target], predicted, np.isin(contexts[target], declared)
                ),
            }
        )
        for q in PROTOCOL["rejection_quantiles"]:
            threshold = float(np.quantile(distances[calibration], q, method="higher"))
            rows.append(
                {
                    "seed": seed,
                    "gate": "feature-distance",
                    "quantile": q,
                    "threshold": threshold,
                    **rejection_metrics(
                        y[target], predicted, distances[target] <= threshold
                    ),
                }
            )
    return {
        "train_ids": train.tolist(),
        "calibration_ids": calibration.tolist(),
        "test_ids": target.tolist(),
        "reference_ids": [] if references is None else references.tolist(),
        "inferred_context_accuracy": context_accuracy,
        "results": rows,
    }


def run(directory, infer_context=False):
    directory = directory.resolve()
    run_id, out = new_run(ROOT / "outputs", "condition-rejection")
    (out / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2))
    cases = {}
    raw, _, ids, records = load_records(ROOT / "data/raw/cranfield-actuator")
    x = aggregate(motion_features(raw), ids, np.arange(len(records))).astype(float)
    y = np.array([r["label"] for r in records])
    context = np.array([f"{r['profile']}/{r['load']}" for r in records])
    repeat = np.array([r["repeat"] for r in records])
    for name in ("held-out-load", "held-out-profile"):
        parts = split_records(records, name)
        source = np.concatenate(
            [parts[k] for k in ("train", "validation", "calibration")]
        )
        target_all = parts["test"]
        train = source[repeat[source] <= 6]
        calibration = source[(repeat[source] >= 7) & (repeat[source] <= 8)]
        known = source[repeat[source] >= 9]
        references = target_all[(y[target_all] == 0) & (repeat[target_all] <= 2)]
        target = target_all[(y[target_all] != 0) | (repeat[target_all] >= 9)]
        cases["actuator/" + name] = {
            "known_test": compare(x, y, context, train, calibration, known),
            "unknown_raw": compare(x, y, context, train, calibration, target),
            "two_healthy_references": compare(
                x, y, context, train, calibration, target, references
            ),
        }
        if infer_context:
            cases["actuator/" + name]["inferred_context_two_references"] = compare(
                x,
                y,
                context,
                train,
                calibration,
                target,
                references,
                infer_context=True,
            )
    manifest = json.loads((directory / "manifest.json").read_text())
    motor = []
    for record in manifest["records"]:
        path = directory / record["cache"].replace("\\", "/")
        if sha256(path) != record["cache_sha256"]:
            raise ValueError("电机特征缓存不一致")
        with np.load(path) as f:
            motor.append(f["features"].mean(axis=0))
    x = np.array(motor, dtype=float)
    recs = manifest["records"]
    labels = ["health"] + sorted({r["label"] for r in recs} - {"health"})
    y = np.array([labels.index(r["label"]) for r in recs])
    context = np.array([r["condition"] for r in recs])
    for name, mask in [
        ("3000rpm", np.array([r["nominal_rpm"] == 3000 for r in recs])),
        ("torque", np.array([r["profile"] == "torque" for r in recs])),
    ]:
        source = np.flatnonzero(~mask)
        calibration_contexts = sorted(set(context[source]))[-2:]
        calibration = source[np.isin(context[source], calibration_contexts)]
        train = source[~np.isin(context[source], calibration_contexts)]
        cases["motor/" + name] = {
            "unknown_raw": compare(
                x, y, context, train, calibration, np.flatnonzero(mask)
            ),
            "commissioning": "not evaluable: one healthy record per condition; two references plus independent normal test unavailable",
            "known_test": "not evaluable: one full recording per class/condition; no random window split used",
        }
    report = {
        "run_id": run_id,
        "protocol": PROTOCOL,
        "inferred_context_experiment": infer_context,
        "cases": cases,
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", Path(__file__).name
        ),
        "motor_manifest_sha256": sha256(directory / "manifest.json"),
        "actuator_inputs": {
            p.name: sha256(p)
            for p in (ROOT / "data/raw/cranfield-actuator").glob("*.mat")
        },
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"run_id": run_id, "output": str(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motor-features", type=Path, required=True)
    parser.add_argument("--infer-context", action="store_true")
    args = parser.parse_args()
    run(args.motor_features, args.infer_context)
