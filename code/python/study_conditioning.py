"""阀门融合与执行器工况参考消融；历史语料探索，不更改生产模型。"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

from phm_core.actuator import aggregate, load_records, motion_features, split_records
from phm_core.artifacts import capture_source, new_run, publish_run, run_path
from phm_core.conditioning import fit_reference, transform_reference
from phm_core.data import block_split
from phm_core.evaluation import evaluate
from phm_core.inference import cpu_session
from phm_core.profiles import load_profile
from phm_core.valve import mechanism_features, pressure_features

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)


def forest(seed):
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=2,
    )


def export(model, values, path):
    graph = convert_sklearn(
        model,
        initial_types=[("features", FloatTensorType([None, values.shape[1]]))],
        options={id(model): {"zipmap": False}},
        target_opset=17,
    )
    path.write_bytes(graph.SerializeToString())
    expected = model.predict_proba(values)
    actual = cpu_session(path).run(None, {"features": values})[1]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    return float(np.abs(actual - expected).max())


def classify(probabilities, truth):
    predicted = probabilities.argmax(1)
    return {
        "records": len(truth),
        "normal": int((truth == 0).sum()),
        "faults": int((truth != 0).sum()),
        "false_positives": int(((truth == 0) & (predicted != 0)).sum()),
        "false_negatives": int(((truth != 0) & (predicted == 0)).sum()),
        "macro_f1": float(
            f1_score(
                truth, predicted, labels=range(4), average="macro", zero_division=0
            )
        ),
        "confusion": confusion_matrix(truth, predicted, labels=range(4)).tolist(),
    }


def run():
    run_id, out = new_run(ROOT / "outputs", "conditioning-study")
    bundle = out / "bundle"
    bundle.mkdir()
    protocol = {
        "seeds": SEEDS,
        "status": "exploratory historical corpora, not blind",
        "valve": "supervised RF uses train fault labels; q99 independent healthy calibration; baseline AE frozen",
        "actuator": "record mean of 5 cycles; source healthy train reference; target healthy repeats1-2 commissioning only excluded for BOTH arms; source-only validation selection",
        "arms": [
            "valve_timing_rf",
            "valve_pressure_rf",
            "valve_pressure_ae_rf",
            "actuator_raw_rf",
            "actuator_conditioned_rf",
        ],
        "deployment": "research only; no automatic activation; ONNX consumes documented features, not raw waveform",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    report = {
        "run_id": run_id,
        "protocol": protocol,
        "source_hash": capture_source(
            ROOT / "code/python", out / "train-source.zip", "study_conditioning.py"
        ),
        "valve": {},
        "actuator": {},
    }
    raw, labels, digest = load_profile(
        ROOT / "data/raw/uci-hydraulic/dataset.zip",
        "valve-switch-100hz",
        ROOT / "data/prepared",
    )
    healthy = labels[:, 1] == 100
    split = block_split(len(raw))
    baseline = run_path(ROOT / "outputs", "valve-study", "20261031T052200Z-1b2c5760")
    with np.load(baseline / "scores.npz") as data:
        ae, rules = data["roi_dense_all_42"], data["timing_rules"]
    report["valve_data_sha256"] = digest
    report["valve_baseline"] = baseline.name
    saved = {}
    for name, score in {"frozen_ae": ae, "timing_rules": rules}.items():
        threshold = float(
            np.quantile(
                score[split["calibration"]][healthy[split["calibration"]]], 0.99
            )
        )
        report["valve"][name] = {
            "42": {
                "threshold": threshold,
                "test": evaluate(score, healthy, split["test"], threshold),
            }
        }
    for name, features in {
        "timing_rf": mechanism_features(raw),
        "pressure_rf": pressure_features(raw),
        "pressure_ae_rf": np.column_stack([pressure_features(raw), ae]).astype(
            "float32"
        ),
    }.items():
        report["valve"][name] = {}
        for seed in SEEDS:
            model = forest(seed).fit(features[split["train"]], ~healthy[split["train"]])
            score = model.predict_proba(features)[:, 1]
            val = split["validation"]
            cal = split["calibration"]
            threshold = float(np.quantile(score[cal][healthy[cal]], 0.99))
            val_th = float(np.quantile(score[val][healthy[val]], 0.99))
            report["valve"][name][str(seed)] = {
                "threshold": threshold,
                "validation": evaluate(score, healthy, val, val_th),
                "test": evaluate(score, healthy, split["test"], threshold),
                "mild": evaluate(
                    score,
                    healthy,
                    split["test"][labels[split["test"], 1] == 90],
                    threshold,
                ),
            }
            saved[f"valve_{name}_{seed}"] = score
            if seed == 42:
                report["valve"][name]["42"]["onnx_max_abs_error"] = export(
                    model, features[split["test"][:64]], bundle / f"valve_{name}.onnx"
                )
    eligible = [
        name
        for name in ("timing_rf", "pressure_rf", "pressure_ae_rf")
        if all(
            v["validation"]["false_positive_per_1000"] <= 10
            for v in report["valve"][name].values()
        )
    ]
    report["valve_selected"] = (
        max(
            eligible,
            key=lambda n: np.mean(
                [v["validation"]["recall"] for v in report["valve"][n].values()]
            ),
        )
        if eligible
        else None
    )
    (out / "valve-split.json").write_text(
        json.dumps({k: v.tolist() for k, v in split.items()}), encoding="utf-8"
    )
    windows, _y, records, metadata = load_records(ROOT / "data/raw/cranfield-actuator")
    features = aggregate(motion_features(windows), records, np.arange(len(metadata)))
    truth = np.array([m["label"] for m in metadata])
    contexts = np.array([f"{m['profile']}/{m['load']}" for m in metadata])
    report["actuator_source"] = json.loads(
        (ROOT / "data/raw/cranfield-actuator/source.json").read_text(encoding="utf-8")
    )
    (bundle / "records.json").write_text(json.dumps(metadata), encoding="utf-8")
    for name in ("held-out-load", "held-out-profile", "record-holdout"):
        parts = split_records(metadata, name)
        commissioning = [
            int(i)
            for i in parts["test"]
            if name != "record-holdout" and truth[i] == 0 and metadata[i]["repeat"] <= 2
        ]
        parts["test"] = np.array([i for i in parts["test"] if i not in commissioning])
        reference_ids = np.concatenate(
            [parts["train"][truth[parts["train"]] == 0], commissioning]
        ).astype(int)
        reference = fit_reference(features, contexts, truth == 0, reference_ids)
        aligned = transform_reference(features, contexts, reference)
        result = {
            "commissioning": commissioning,
            "partitions": {k: v.tolist() for k, v in parts.items()},
            "arms": {},
        }
        (bundle / f"{name}-reference.json").write_text(
            json.dumps(reference, indent=2), encoding="utf-8"
        )
        for arm, values in {"raw_rf": features, "conditioned_rf": aligned}.items():
            result["arms"][arm] = {}
            for seed in SEEDS:
                model = forest(seed).fit(values[parts["train"]], truth[parts["train"]])
                probabilities = model.predict_proba(values)
                item = {
                    p: classify(probabilities[parts[p]], truth[parts[p]])
                    for p in ("validation", "test")
                }
                item["errors"] = [
                    {
                        "record": int(i),
                        "truth": int(truth[i]),
                        "predicted": int(probabilities[i].argmax()),
                    }
                    for i in parts["test"]
                    if truth[i] != probabilities[i].argmax()
                ]
                saved[f"actuator_{name}_{arm}_{seed}"] = probabilities
                if seed == 42:
                    item["onnx_max_abs_error"] = export(
                        model, values[parts["test"][:64]], bundle / f"{name}-{arm}.onnx"
                    )
                result["arms"][arm][str(seed)] = item
        result["selected"] = max(
            result["arms"],
            key=lambda a: np.mean(
                [s["validation"]["macro_f1"] for s in result["arms"][a].values()]
            ),
        )
        report["actuator"][name] = result
    (bundle / "feature-contract.json").write_text(
        json.dumps(
            {
                "valve": "phm_core.valve.mechanism_features / pressure_features; pressure_ae appends frozen baseline AE score",
                "actuator": "25Hz raw [5,400,3] -> motion_features -> five-cycle mean -> optional transform_reference; explicit profile/load required",
                "warning": "feature-input ONNX; preprocessing SDK required; not schema2 raw-window package",
                "protocol": protocol,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(out / "scores.npz", **saved)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for filename in ("metrics.json", "protocol.json"):
        shutil.copyfile(out / filename, bundle / filename)
    shutil.copyfile(
        ROOT / "code/THIRD_PARTY_NOTICES.md", bundle / "THIRD_PARTY_NOTICES.md"
    )
    (bundle / "integrity.json").write_text(
        json.dumps(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(bundle.iterdir())
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.make_archive(str(out / "model-bundle"), "zip", bundle)
    publish_run(ROOT / "outputs", "conditioning-study", run_id)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "valve": report["valve"],
                "actuator": report["actuator"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
