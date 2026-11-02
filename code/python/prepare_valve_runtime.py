"""将冻结双证据候选封装为运行层兼容的新版本；不训练、不更改latest或绑定。"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from phm_core.artifacts import new_run
from phm_core.inference import BundleScorer
from phm_core.profiles import load_profile

ROOT = Path(__file__).resolve().parents[2]


def prepare(export_run_id):
    from phm_core.artifacts import RUN_PATTERN
    import re

    if not re.fullmatch(RUN_PATTERN, export_run_id):
        raise ValueError("导出版本格式不正确")
    source = ROOT / "outputs/valve-fusion-export/runs" / export_run_id
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    model = source / "model.onnx"
    if hashlib.sha256(model.read_bytes()).hexdigest() != original["sha256"]:
        raise ValueError("候选图哈希不匹配")
    if original["shape"] != [None, 400, 2] or original["sampleRateHz"] != 100:
        raise ValueError("只支持已验证的100Hz双压力阀门候选")
    values, _, digest = load_profile(
        ROOT / "data/raw/uci-hydraulic/dataset.zip",
        "valve-switch-100hz",
        ROOT / "data/prepared",
    )
    baseline = ROOT / "outputs/valve/runs" / original["baseline_run"] / "metrics.json"
    if json.loads(baseline.read_text(encoding="utf-8"))["data_sha256"] != digest:
        raise ValueError("验证输入与原始训练数据摘要不一致")
    run_id, directory = new_run(ROOT / "outputs", "valve-runtime")
    bundle = directory / "bundle"
    bundle.mkdir()
    shutil.copyfile(model, bundle / "ae.onnx")
    manifest = {
        "schemaVersion": 1,
        "bundleKind": "valve-fusion-runtime",
        "task": "valve",
        "channels": ["PS1", "PS2"],
        "units": ["bar", "bar"],
        "sampleRateHz": 100,
        "windowSeconds": 4,
        "windowStartSeconds": 8,
        "sync": {"required": True, "expected_lead_s": 8.0, "tolerance_samples": 1},
        "models": {
            "ae": {
                "file": "ae.onnx",
                "sha256": original["sha256"],
                "inputName": "values",
                "shape": [None, 400, 2],
                "outputIndex": 0,
                "transform": "identity",
                "threshold": 1.0,
            }
        },
        "origin": {"export_run_id": export_run_id, "data_sha256": digest},
        "scope": "frozen historical research candidate; shadow only",
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    scorer = BundleScorer(bundle)
    actual = np.concatenate(
        [scorer.score(batch) for batch in np.array_split(values, 35)]
    )
    with np.load(source / "parity.npz") as parity:
        expected = parity["expected"]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    np.testing.assert_array_equal(actual > 1, expected > 1)
    error = float(np.max(np.abs(actual - expected)))
    np.savez_compressed(directory / "parity.npz", raw=values[:7], ae=actual[:7])
    metrics = {
        "run_id": run_id,
        "task": "valve",
        "research_only": True,
        "data_sha256": digest,
        "source_export": export_run_id,
        "models": {
            "ae": {**original["test"], "threshold": 1.0, "onnx_max_abs_error": error}
        },
    }
    (directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    shutil.make_archive(str(directory / "model-bundle"), "zip", bundle)
    (directory / "ready.json").write_text(
        json.dumps({"run_id": run_id}), encoding="utf-8"
    )
    return {
        "run_id": run_id,
        "path": str(directory),
        "verified_windows": len(actual),
        "max_abs_error": error,
        "latest_updated": False,
        "bindings_updated": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-run-id", required=True)
    print(json.dumps(prepare(parser.parse_args().export_run_id)))
