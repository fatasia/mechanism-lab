"""冻结的执行器工况外基准：完整记录隔离，强基线/深度网络/模态消融。"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from phm_core.actuator import (
    CLASSES,
    FEATURES,
    aggregate,
    align_motion,
    fit_phase_references,
    load_records,
    motion_features,
    split_records,
)
from phm_core.artifacts import capture_source, new_run, publish_run
from phm_core.diagnostic_models import (
    KernelDiagnostic,
    KernelTransform,
    fit_multiscale,
    infer_batches,
)
from phm_core.inference import cpu_session

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)
PROTOCOLS = ("held-out-load", "held-out-profile")
ARMS = (
    "mechanism_rf",
    "current_only_rf",
    "error_only_rf",
    "random_kernels",
    "multiscale_fusion",
)


def record_metrics(score, y, record_ids, ids):
    prediction = aggregate(score, record_ids, ids).argmax(1)
    truth = np.array([y[record_ids == record][0] for record in ids])
    return {
        "records": len(ids),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(
                truth, prediction, labels=range(4), average="macro", zero_division=0
            )
        ),
        "confusion": confusion_matrix(truth, prediction, labels=range(4)).tolist(),
        "false_positives": int(((truth == 0) & (prediction != 0)).sum()),
        "false_negatives": int(((truth != 0) & (prediction == 0)).sum()),
        "errors": [
            {"record": int(i), "truth": int(t), "predicted": int(p)}
            for i, t, p in zip(ids, truth, prediction)
            if t != p
        ],
    }


def export_graph(model, values, path):
    torch.onnx.export(
        model.eval(),
        torch.from_numpy(values[:1]),
        str(path),
        input_names=["values"],
        output_names=["scores"],
        dynamic_axes={"values": {0: "batch"}, "scores": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def run(epochs=60, reference_alignment=False, record_holdout=False):
    torch.set_num_threads(2)
    raw, y, record_ids, metadata = load_records(ROOT / "data/raw/cranfield-actuator")
    x = raw
    task = (
        "actuator-release-study"
        if record_holdout
        else "actuator-transfer-study"
        if reference_alignment
        else "actuator-study"
    )
    protocols = ("record-holdout",) if record_holdout else PROTOCOLS
    run_id, out = new_run(ROOT / "outputs", task)
    bundle = out / "bundle"
    bundle.mkdir()
    source_hash = capture_source(
        ROOT / "code/python", out / "train-source.zip", "study_actuator.py"
    )
    data_source = json.loads(
        (ROOT / "data/raw/cranfield-actuator/source.json").read_text()
    )
    protocol = {
        "version": 1,
        "seeds": SEEDS,
        "epochs": epochs,
        "arms": ARMS,
        "protocols": protocols,
        "sample_rate_hz": 25,
        "window_seconds": 16,
        "group": "original 80s MAT variable, never split its five cycles",
        "partition": "held-out negative40kg or sinusoidal profile; other conditions repeats1-6 train,7-8 validation,9-10 calibration",
        "selection": "per protocol: mean record-level validation macroF1 across seeds, RF/kernel/CNN complexity tie-break; deployment seed42 fixed",
        "evaluation": "single physical rig, seeded faults; not cross-device or life prediction",
        "rf": {
            "trees": 300,
            "max_depth": 10,
            "min_leaf": 2,
            "class_weight": "balanced",
        },
        "ridge_alpha": 10,
        "channels": ["position_setpoint_mm", "following_error_mm", "motor_current_A"],
        "data_source": data_source,
        "reference_alignment": reference_alignment,
        "record_holdout": record_holdout,
        "record_holdout_split": "all six observed contexts: repeats1-5 train,6-7 validation,8 calibration,9-10 test; same corpus already inspected; not new blind data",
        "commissioning": "when enabled: target normal repetitions1-2 are reference only and excluded from test; source templates use train normal only; supplied load/profile context required",
        "adaptation_status": "commissioned-context transfer, not zero-shot; protocol added after initial transfer failures were observed"
        if reference_alignment
        else "zero-shot held-out condition",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (out / "records.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = {
        "run_id": run_id,
        "source_hash": source_hash,
        "dataset": "Cranfield linear actuator",
        "records": len(metadata),
        "cycles": len(x),
        "classes": CLASSES,
        "protocol": protocol,
        "protocols": {},
        "contracts": {},
    }
    for name in protocols:
        split = split_records(metadata, name)
        x = raw
        if reference_alignment:
            commissioning = [
                i
                for i in split["test"]
                if metadata[i]["label"] == 0 and metadata[i]["repeat"] <= 2
            ]
            split["test"] = np.array(
                [i for i in split["test"] if i not in commissioning]
            )
            split["commissioning"] = np.asarray(commissioning)
            reference_ids = {
                i for i in split["train"] if metadata[i]["label"] == 0
            } | set(commissioning)
            references = fit_phase_references(raw, record_ids, metadata, reference_ids)
            (bundle / f"phase-reference-{name}.json").write_text(
                json.dumps(references), encoding="utf-8"
            )
            contexts = [
                metadata[i]["profile"] + "/" + metadata[i]["load"] for i in record_ids
            ]
            x = align_motion(raw, contexts, references)
        features = motion_features(x)
        windows = {
            k: np.flatnonzero(np.isin(record_ids, ids)) for k, ids in split.items()
        }
        (out / f"split-{name}.json").write_text(
            json.dumps({k: v.tolist() for k, v in split.items()}, indent=2),
            encoding="utf-8",
        )
        group = {
            "counts": {k: len(ids) for k, ids in split.items()},
            "arms": {arm: {"seeds": {}} for arm in ARMS},
        }
        scores, exports = {}, {}
        for seed in SEEDS:
            train, valid = windows["train"], windows["validation"]
            # 三种特征路线只改变可用模态；工况/故障标签不作为推理输入。
            for arm in ARMS[:3]:
                columns = (
                    np.arange(len(FEATURES))
                    if arm == "mechanism_rf"
                    else np.array(
                        [
                            i
                            for i, f in enumerate(FEATURES)
                            if ("current" if arm == "current_only_rf" else "error") in f
                        ]
                    )
                )
                model = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=2,
                )
                model.fit(features[train][:, columns], y[train])
                score = model.predict_proba(features[:, columns]).astype("float32")
                scores[(arm, seed)] = score
                if seed == 42:
                    exports[arm] = (model, features[:, columns], columns)
            mean, scale = x[train].mean((0, 1)), x[train].std((0, 1))
            transform = KernelTransform(mean, scale, seed).eval()
            kernel = infer_batches(transform, x)
            scaler = StandardScaler().fit(kernel[train])
            ridge = RidgeClassifier(alpha=10, class_weight="balanced").fit(
                scaler.transform(kernel[train]), y[train]
            )
            deployed = KernelDiagnostic(
                transform, scaler.mean_, scaler.scale_, ridge.coef_, ridge.intercept_
            ).eval()
            scores[("random_kernels", seed)] = infer_batches(deployed, x)
            if seed == 42:
                exports["random_kernels"] = (deployed, x, None)
            model, history = fit_multiscale(x, y, train, valid, seed, epochs)
            scores[("multiscale_fusion", seed)] = infer_batches(model, x)
            group["arms"]["multiscale_fusion"]["seeds"].setdefault(str(seed), {})[
                "history"
            ] = history
            if seed == 42:
                exports["multiscale_fusion"] = (model, x, None)
            for arm in ARMS:
                entry = group["arms"][arm]["seeds"].setdefault(str(seed), {})
                entry["validation"] = record_metrics(
                    scores[(arm, seed)], y, record_ids, split["validation"]
                )
            print(
                json.dumps(
                    {
                        "protocol": name,
                        "seed": seed,
                        "validation": {
                            a: group["arms"][a]["seeds"][str(seed)]["validation"][
                                "macro_f1"
                            ]
                            for a in ARMS
                        },
                    }
                ),
                flush=True,
            )
        candidates = ("mechanism_rf", "random_kernels", "multiscale_fusion")
        selected = max(
            candidates,
            key=lambda a: (
                np.mean(
                    [
                        group["arms"][a]["seeds"][str(s)]["validation"]["macro_f1"]
                        for s in SEEDS
                    ]
                ),
                -candidates.index(a),
            ),
        )
        group["selected"] = {"arm": selected, "seed": 42}
        (out / f"selection-{name}.json").write_text(
            json.dumps(group["selected"]), encoding="utf-8"
        )
        # 选定之后统一展开留出工况，部署模型不使用测试集重训。
        for arm in ARMS:
            for seed in SEEDS:
                group["arms"][arm]["seeds"][str(seed)]["test"] = record_metrics(
                    scores[(arm, seed)], y, record_ids, split["test"]
                )
            f1s = [
                group["arms"][arm]["seeds"][str(s)]["test"]["macro_f1"] for s in SEEDS
            ]
            group["arms"][arm]["test_f1"] = {
                "mean": float(np.mean(f1s)),
                "min": float(min(f1s)),
                "max": float(max(f1s)),
            }
            model, values, columns = exports[arm]
            path = bundle / f"{name}-{arm}.onnx"
            if columns is not None:
                graph = convert_sklearn(
                    model,
                    initial_types=[("values", FloatTensorType([None, len(columns)]))],
                    options={id(model): {"zipmap": False}},
                    target_opset=17,
                )
                path.write_bytes(graph.SerializeToString())
                output_index = 1
            else:
                export_graph(model, values, path)
                output_index = 0
            session = cpu_session(path)
            ids = windows["test"][:64]
            actual = session.run(None, {"values": values[ids]})[output_index]
            np.testing.assert_allclose(
                actual, scores[(arm, 42)][ids], atol=1e-4, rtol=1e-4
            )
            error = float(np.abs(actual - scores[(arm, 42)][ids]).max())
            group["arms"][arm]["onnx_max_abs_error"] = error
            report["contracts"][f"{name}/{arm}"] = {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "shape": [None, *values.shape[1:]],
                "input_kind": "motion_features"
                if columns is not None
                else "aligned_motion"
                if reference_alignment
                else "raw",
                "reference_alignment": reference_alignment,
                "feature_columns": columns.tolist() if columns is not None else None,
                "output_index": output_index,
                "score_kind": "class_probability"
                if columns is not None
                else "uncalibrated_class_score",
            }
        np.savez_compressed(
            out / f"scores-{name}.npz",
            **{f"{a}_{s}": v for (a, s), v in scores.items()},
        )
        report["protocols"][name] = group
    np.savez_compressed(
        out / "replay.npz",
        x=raw,
        y=y,
        record_ids=record_ids,
        features=motion_features(raw),
    )
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for filename in ("metrics.json", "protocol.json"):
        shutil.copyfile(out / filename, bundle / filename)
    (bundle / "features.json").write_text(
        json.dumps(
            {
                "names": FEATURES,
                "recipe": "phm_core.actuator.motion_features; 25Hz 16s complete bidirectional cycle; columns setpoint_mm,error_mm,current_A",
            },
            indent=2,
        )
    )
    shutil.make_archive(str(out / "model-bundle"), "zip", bundle)
    publish_run(ROOT / "outputs", task, run_id)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "results": {
                    k: {a: v["test_f1"] for a, v in g["arms"].items()}
                    for k, g in report["protocols"].items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--reference-alignment", action="store_true")
    parser.add_argument("--record-holdout", action="store_true")
    args = parser.parse_args()
    run(args.epochs, args.reference_alignment, args.record_holdout)
