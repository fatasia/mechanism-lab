"""固定三种子集成、独立记录级校准及候选集合；不挑最好种子。"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from phm_core.actuator import (
    CLASSES,
    aggregate,
    load_records,
    motion_features,
    split_records,
)
from phm_core.artifacts import capture_source, new_run, publish_run
from phm_core.diagnostic_models import fit_multiscale, infer_batches
from phm_core.inference import cpu_session
from phm_core.monitoring import calibrated_threshold

ROOT = Path(__file__).resolve().parents[2]


class RecordEnsemble(torch.nn.Module):
    def __init__(self, models, temperatures, cutoff=1.0):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.register_buffer(
            "temperatures", torch.tensor(temperatures, dtype=torch.float32)
        )
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))

    def forward(self, records):
        windows = records.reshape(-1, 400, 3)
        probabilities = torch.stack(
            [
                torch.softmax(
                    model(windows).reshape(-1, 5, 4).mean(1) / self.temperatures[i],
                    dim=1,
                )
                for i, model in enumerate(self.models)
            ]
        ).mean(0)
        return probabilities, (1 - probabilities) <= self.cutoff


def metrics(probabilities, labels, ids, cutoff):
    prediction = probabilities[ids].argmax(1)
    truth = labels[ids]
    candidate = 1 - probabilities[ids] <= cutoff
    return {
        "records": len(ids),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "accuracy": float(accuracy_score(truth, prediction)),
        "log_loss": float(log_loss(truth, probabilities[ids], labels=range(4))),
        "confusion": confusion_matrix(truth, prediction, labels=range(4)).tolist(),
        "false_positives": int(((truth == 0) & (prediction != 0)).sum()),
        "false_negatives": int(((truth != 0) & (prediction == 0)).sum()),
        "candidate_set_coverage": float(candidate[np.arange(len(ids)), truth].mean()),
        "singleton_fraction": float((candidate.sum(1) == 1).mean()),
        "empty_fraction": float((candidate.sum(1) == 0).mean()),
        "errors": [
            {"record": int(i), "truth": int(t), "predicted": int(p)}
            for i, t, p in zip(ids, truth, prediction)
            if t != p
        ],
    }


def run():
    x, y, record_ids, metadata = load_records(ROOT / "data/raw/cranfield-actuator")
    split = split_records(metadata, "record-holdout")
    windows = {k: np.flatnonzero(np.isin(record_ids, ids)) for k, ids in split.items()}
    run_id, out = new_run(ROOT / "outputs", "actuator-ensemble-study")
    bundle = out / "bundle"
    bundle.mkdir()
    protocol = {
        "seeds": [42, 43, 44],
        "epochs": 60,
        "reference_alignment": False,
        "record_holdout": True,
        "partition": "all known contexts; repeats1-5 train,6-7 validation,8 calibration,9-10 test",
        "selection": "uniform fixed3seed ensemble; no seed or mixture weight selected by test",
        "temperature_grid": [0.25, 0.5, 1, 2, 4],
        "temperature_selection": "per-seed record-level calibration NLL only",
        "prediction_set_alpha": 0.1,
        "prediction_set_scope": "record-level marginal calibration; exchangeability not established for new devices or conditions",
        "input": "80s complete record [batch,2000,3]; five16s cycles,25Hz; setpoint_mm,error_mm,current_A",
        "scope": "same inspected corpus, known six load/profile contexts, not new blind or cross-device validation",
        "data_source": json.loads(
            (ROOT / "data/raw/cranfield-actuator/source.json").read_text()
        ),
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    source = capture_source(
        ROOT / "code/python", out / "train-source.zip", "study_actuator_ensemble.py"
    )
    labels = np.array([m["label"] for m in metadata])
    models, temperatures, histories = [], [], {}
    for seed in (42, 43, 44):
        model, history = fit_multiscale(
            x, y, windows["train"], windows["validation"], seed, 60
        )
        torch.save(model.state_dict(), out / f"model-{seed}.pt")
        logits = aggregate(
            infer_batches(model, x), record_ids, np.arange(len(metadata))
        )
        cal = split["calibration"]
        candidates = {}
        for temperature in protocol["temperature_grid"]:
            probability = torch.softmax(
                torch.from_numpy(logits[cal]) / temperature, 1
            ).numpy()
            candidates[temperature] = log_loss(
                labels[cal], probability, labels=range(4)
            )
        temperature = min(candidates, key=candidates.get)
        models.append(model)
        temperatures.append(temperature)
        histories[str(seed)] = history
        print(
            json.dumps({"seed": seed, "calibration_temperature": temperature}),
            flush=True,
        )
    records = np.stack(
        [x[record_ids == i].reshape(2000, 3) for i in range(len(metadata))]
    )
    deployed = RecordEnsemble(models, temperatures).eval()
    with torch.no_grad():
        probabilities = np.concatenate(
            [
                deployed(torch.from_numpy(batch))[0].numpy()
                for batch in np.array_split(records, 25)
            ]
        )
    cal = split["calibration"]
    cutoff = calibrated_threshold(1 - probabilities[cal, labels[cal]], 0.1)
    deployed.cutoff.fill_(cutoff)
    validation = metrics(probabilities, labels, split["validation"], cutoff)
    test = metrics(probabilities, labels, split["test"], cutoff)
    graph = bundle / "record-ensemble.onnx"
    torch.onnx.export(
        deployed,
        torch.from_numpy(records[:1]),
        str(graph),
        input_names=["values"],
        output_names=["probabilities", "candidate_set"],
        dynamic_axes={
            "values": {0: "batch"},
            "probabilities": {0: "batch"},
            "candidate_set": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    session = cpu_session(graph)
    max_error = 0.0
    for size in (1, 7, 64):
        ids = split["test"][:size]
        actual, candidate = session.run(None, {"values": records[ids]})
        np.testing.assert_allclose(actual, probabilities[ids], atol=1e-4, rtol=1e-4)
        np.testing.assert_array_equal(candidate, 1 - probabilities[ids] <= cutoff)
        max_error = max(max_error, float(np.abs(actual - probabilities[ids]).max()))
    arm = {
        "seeds": {"42": {"validation": validation, "test": test}},
        "test_f1": {
            "mean": test["macro_f1"],
            "min": test["macro_f1"],
            "max": test["macro_f1"],
        },
        "ensemble_size": 3,
        "onnx_max_abs_error": max_error,
    }
    report = {
        "run_id": run_id,
        "source_hash": source,
        "dataset": "Cranfield linear actuator",
        "records": len(metadata),
        "cycles": len(x),
        "classes": CLASSES,
        "protocol": protocol,
        "calibration": {
            "temperatures": temperatures,
            "cutoff": cutoff,
            "records": len(cal),
        },
        "protocols": {
            "record-holdout": {
                "counts": {k: len(v) for k, v in split.items()},
                "selected": {
                    "arm": "multiscale_ensemble",
                    "seed": "fixed-three-seed-ensemble",
                },
                "arms": {"multiscale_ensemble": arm},
            }
        },
        "contracts": {
            "record-holdout/multiscale_ensemble": {
                "file": graph.name,
                "sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
                "shape": [None, 2000, 3],
                "input_kind": "full_record",
                "output_index": 0,
                "score_kind": "temperature_calibrated_class_probability",
                "candidate_set_output_index": 1,
            }
        },
    }
    for name, obj in (
        ("metrics.json", report),
        ("records.json", metadata),
        ("split-record-holdout.json", {k: v.tolist() for k, v in split.items()}),
        ("histories.json", histories),
    ):
        (out / name).write_text(json.dumps(obj, indent=2))
    np.savez_compressed(
        out / "replay.npz",
        x=x,
        y=y,
        record_ids=record_ids,
        features=motion_features(x),
        probabilities=probabilities,
    )
    for name in ("protocol.json", "metrics.json"):
        shutil.copyfile(out / name, bundle / name)
    shutil.make_archive(str(out / "model-bundle"), "zip", bundle)
    publish_run(ROOT / "outputs", "actuator-ensemble-study", run_id)
    print(
        json.dumps({"run_id": run_id, "test": test, "onnx_max_abs_error": max_error}),
        flush=True,
    )


if __name__ == "__main__":
    run()
