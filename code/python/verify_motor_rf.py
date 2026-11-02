"""在指定独立NumPy/ORT环境验证电机候选，不以推理环境加载训练pickle。"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from phm_core.artifacts import capture_source, new_run

ROOT = Path(__file__).resolve().parents[2]


def run(export_directory, study_directory, feature_directory, interpreter):
    exported = json.loads(
        (export_directory / "metrics.json").read_text(encoding="utf-8")
    )
    feature_manifest = json.loads(
        (feature_directory / "manifest.json").read_text(encoding="utf-8")
    )
    arrays, ids = [], []
    for record in feature_manifest["records"]:
        path = feature_directory / record["cache"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["cache_sha256"]:
            raise ValueError("原特征缓存哈希变化")
        with np.load(path) as cache:
            arrays.append(cache["features"])
            ids.extend([record["id"]] * len(cache["features"]))
    requested = np.array([r["id"] for r in feature_manifest["records"]] + [-99])
    run_id, out = new_run(ROOT / "outputs", "motor-rf-verification")
    np.savez_compressed(
        out / "input.npz",
        features=np.concatenate(arrays),
        window_records=ids,
        requested_record_ids=requested,
    )
    environment = subprocess.run(
        [str(interpreter), "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (out / "inference-environment.txt").write_text(environment, encoding="utf-8")
    if any(
        line.lower().startswith(("torch==", "scikit-learn==", "phm-"))
        for line in environment.splitlines()
    ):
        raise ValueError("本检查要求推理环境不含训练/平台包")
    results = []
    for model in exported["models"]:
        name = f"{model['protocol']}-{model['seed']}"
        bundle = export_directory / name
        output = out / f"{name}.npz"
        process = subprocess.run(
            [
                str(interpreter),
                str(bundle / "infer_motor_rf.py"),
                "--bundle",
                str(bundle),
                "--input",
                str(out / "input.npz"),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        (out / f"{name}.log").write_text(
            process.stdout + process.stderr, encoding="utf-8"
        )
        with (
            np.load(output) as actual,
            np.load(
                study_directory
                / model["protocol"]
                / f"target-{model['arm']}-{model['seed']}.npz"
            ) as expected,
        ):
            positions = np.array(
                [
                    np.flatnonzero(actual["record_ids"] == identifier)[0]
                    for identifier in expected["record_ids"]
                ]
            )
            probabilities = actual["probabilities"][positions]
            maximum = float(np.max(np.abs(probabilities - expected["probabilities"])))
            mismatches = int(
                np.sum(
                    actual["labels"][positions] != expected["probabilities"].argmax(1)
                )
            )
            assert maximum <= 1e-5 and mismatches == 0
            # -99只是接口覆盖测试，没有观测窗口，不能生成正常标签。
            assert actual["labels"][-1] == -1 and not actual["supported"][-1]
            assert np.isnan(actual["probabilities"][-1]).all()
            results.append(
                {
                    "model": name,
                    "target_records": len(positions),
                    "max_abs_error_to_original_target": maximum,
                    "target_label_mismatches": mismatches,
                    "unsupported_sentinel_passed": True,
                }
            )
    report = {
        "run_id": run_id,
        "export_run": exported["run_id"],
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", "verify_motor_rf.py"
        ),
        "status": "passed",
        "models": results,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path)
    parser.add_argument("study", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("interpreter", type=Path)
    args = parser.parse_args()
    run(
        args.export.resolve(),
        args.study.resolve(),
        args.features.resolve(),
        args.interpreter.resolve(),
    )
