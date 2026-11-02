"""把已完成实验重新包装为统一部署契约；传统模型的时域特征也嵌入 ONNX。"""

import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import onnx
import torch
from onnx.compose import merge_models

from phm_core.artifacts import capture_source, new_run, publish_run, run_path
from phm_core.portable import PortableScorer

ROOT = Path(__file__).resolve().parents[2]


class CycleFeatureGraph(torch.nn.Module):
    def forward(self, x):
        return torch.cat(
            [
                x.mean(1),
                x.std(1, unbiased=False),
                x.square().mean(1).sqrt(),
                x.amax(1) - x.amin(1),
            ],
            dim=1,
        )


def add_raw_frontend(source, raw, target):
    frontend = target.with_name("features.onnx")
    torch.onnx.export(
        CycleFeatureGraph(),
        torch.from_numpy(raw[:1]),
        str(frontend),
        input_names=["values"],
        output_names=["features"],
        dynamic_axes={"values": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    a, b = onnx.load(frontend), onnx.load(source)
    a.ir_version = b.ir_version
    merged = merge_models(
        a, b, [("features", b.graph.input[0].name)], prefix1="pre/", prefix2="model/"
    )
    onnx.checker.check_model(merged)
    onnx.save(merged, target)
    frontend.unlink()


def package_one(
    out,
    model_id,
    source_graph,
    input_spec,
    outputs,
    origin,
    expected,
    raw,
    max_error,
    embed_features=False,
):
    directory = out / "models" / model_id
    directory.mkdir(parents=True)
    target = directory / "model.onnx"
    if embed_features:
        add_raw_frontend(source_graph, raw, target)
        input_spec["name"] = "pre/values"
    else:
        shutil.copyfile(source_graph, target)
    manifest = {
        "schemaVersion": 2,
        "modelId": model_id,
        "version": out.name,
        "origin": origin,
        "dataSource": {
            "title": "UCI Hydraulic Systems"
            if origin.get("task") in ("valve", "cooler", "pump", "accumulator")
            else "Cranfield linear actuator",
            "url": "https://doi.org/10.24432/C5CW21"
            if origin.get("task") in ("valve", "cooler", "pump", "accumulator")
            else "https://doi.org/10.17862/cranfield.rd.5097649",
            "license": "CC BY 4.0",
        },
        "graph": {
            "file": "model.onnx",
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
        "input": {
            **input_spec,
            "layout": "BTC",
            "dtype": "float32",
            "preprocessingEmbedded": True,
        },
        "outputs": outputs,
        "execution": {"provider": "CPUExecutionProvider", "maxBatch": 8},
        "scope": "public dataset validation; external industrial equipment acceptance pending",
        "validation": {"referenceWindows": len(raw), "tolerance": max_error},
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    scorer = PortableScorer(directory)
    observed = {role: [] for role in expected}
    for batch in np.array_split(raw, max(1, int(np.ceil(len(raw) / 8)))):
        result = scorer.score(
            batch,
            input_spec["channels"],
            input_spec["units"],
            input_spec["sampleRateHz"],
        )
        for role in expected:
            observed[role].append(result[role])
    errors = {}
    for role, reference in expected.items():
        actual = np.concatenate(observed[role])
        if reference.dtype == bool:
            np.testing.assert_array_equal(actual, reference)
            errors[role] = 0
        else:
            actual = actual.reshape(reference.shape)
            np.testing.assert_allclose(
                actual, reference, atol=max_error, rtol=max_error
            )
            errors[role] = float(np.abs(actual - reference).max())
    manifest["validation"]["maxAbsErrors"] = errors
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    filename = out / f"{model_id}.zip"
    with ZipFile(filename, "w", ZIP_DEFLATED) as archive:
        archive.write(target, "model.onnx")
        archive.write(directory / "manifest.json", "manifest.json")
        archive.write(ROOT / "code/THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md")
        if model_id.startswith("minirocket"):
            archive.write(
                ROOT / "code/licenses/aeon-BSD-3-Clause.txt",
                "licenses/aeon-BSD-3-Clause.txt",
            )
    return {"id": model_id, "file": filename.name, "manifest": manifest}


def run():
    run_id, out = new_run(ROOT / "outputs", "deployments")
    catalog = {
        "run_id": run_id,
        "source_hash": capture_source(
            ROOT / "code/python", out / "train-source.zip", "package_deployments.py"
        ),
        "models": [],
    }
    for task in ("valve", "cooler", "pump", "accumulator"):
        directory = run_path(ROOT / "outputs", task)
        old = json.loads((directory / "manifest.json").read_text())
        with np.load(directory / "parity.npz") as data:
            for name, spec in old["models"].items():
                inputs = {
                    "name": spec["inputName"],
                    "shape": old["models"]["ae"]["shape"],
                    "channels": old["channels"],
                    "units": old["units"],
                    "sampleRateHz": old["sampleRateHz"],
                }
                output = {
                    "index": spec["outputIndex"],
                    "role": "class-probability" if name == "rf" else "anomaly-score",
                    "transform": spec["transform"],
                    "threshold": spec["threshold"],
                }
                catalog["models"].append(
                    package_one(
                        out,
                        f"{task}-{name}",
                        directory / spec["file"],
                        inputs,
                        [output],
                        {"task": task, "runId": directory.name, "model": name},
                        {output["role"]: data[name]},
                        data["raw"],
                        1e-4,
                        embed_features=name != "ae",
                    )
                )
    directory = run_path(ROOT / "outputs", "minirocket-export")
    old = json.loads((directory / "metrics.json").read_text())
    for protocol, spec in old["models"].items():
        reference = ROOT / "outputs/reference-study/runs" / old["reference_run"]
        with np.load(reference / f"{protocol}-minirocket-parameters.npz") as data:
            inputs = {
                "name": "values",
                "shape": [None, 400, 3],
                "channels": ["position_setpoint", "following_error", "motor_current"],
                "units": ["mm", "mm", "A"],
                "sampleRateHz": 25,
            }
            catalog["models"].append(
                package_one(
                    out,
                    f"minirocket-{protocol}",
                    directory / "bundle" / spec["file"],
                    inputs,
                    [
                        {
                            "index": 0,
                            "role": "uncalibrated-class-score",
                            "classes": spec["classes"],
                        }
                    ],
                    {
                        "task": "minirocket-export",
                        "runId": directory.name,
                        "protocol": protocol,
                    },
                    {"uncalibrated-class-score": data["parity_scores"]},
                    data["parity_raw"],
                    1e-3,
                )
            )
    directory = run_path(ROOT / "outputs", "actuator-ensemble-study")
    old = json.loads((directory / "metrics.json").read_text())
    spec = old["contracts"]["record-holdout/multiscale_ensemble"]
    split = json.loads((directory / "split-record-holdout.json").read_text())
    with np.load(directory / "replay.npz") as data:
        ids = np.array(split["test"][:64])
        records = np.stack(
            [data["x"][data["record_ids"] == i].reshape(2000, 3) for i in ids]
        )
        expected = data["probabilities"][ids]
        inputs = {
            "name": "values",
            "shape": [None, 2000, 3],
            "channels": ["position_setpoint", "following_error", "motor_current"],
            "units": ["mm", "mm", "A"],
            "sampleRateHz": 25,
        }
        catalog["models"].append(
            package_one(
                out,
                "actuator-ensemble",
                directory / "bundle" / spec["file"],
                inputs,
                [
                    {
                        "index": 0,
                        "role": "class-probability",
                        "classes": old["classes"],
                    },
                    {"index": 1, "role": "prediction-set", "classes": old["classes"]},
                ],
                {"task": "actuator-ensemble-study", "runId": directory.name},
                {
                    "class-probability": expected,
                    "prediction-set": 1 - expected <= old["calibration"]["cutoff"],
                },
                records,
                1e-4,
            )
        )
    (out / "metrics.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    with ZipFile(out / "model-bundle.zip", "w", ZIP_DEFLATED) as archive:
        archive.write(out / "metrics.json", "catalog.json")
        for entry in catalog["models"]:
            archive.write(out / entry["file"], entry["file"])
    publish_run(ROOT / "outputs", "deployments", run_id)
    print(
        json.dumps(
            {"run_id": run_id, "raw_input_bundles_verified": len(catalog["models"])}
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
