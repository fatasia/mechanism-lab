"""重新加载每个模型包，核对哈希、输入、输出和演示端点。"""

import hashlib
import json
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fastapi.testclient import TestClient

from phm_core.artifacts import run_path
from phm_core.inference import BundleScorer, cpu_session
from phm_demo.api import app

root = Path(__file__).resolve().parents[2]
client = TestClient(app)
verified = []
for task in ["valve", "cooler", "pump", "accumulator"]:
    directory = run_path(root / "outputs", task)
    manifest = json.loads((directory / "manifest.json").read_text())
    with np.load(directory / "parity.npz") as samples:
        scorer = BundleScorer(directory / "bundle")
        decision_mismatches = {}
        for name, item in manifest["models"].items():
            path = directory / item["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
            session = cpu_session(path)
            results = session.run(
                None,
                {item["inputName"]: samples["raw" if name == "ae" else "features"]},
            )
            scores = results[item["outputIndex"]]
            if item["transform"] == "negate":
                scores = -scores
            if item["transform"] == "column1":
                scores = scores[:, 1]
            np.testing.assert_allclose(
                scores.ravel(), samples[name], atol=1e-5, rtol=1e-4
            )
            verified.append(f"{task}/{name}")
            threshold = manifest["models"][name]["threshold"]
            decision_mismatches[name] = int(
                np.sum((scores.ravel() > threshold) != (samples[name] > threshold))
            )
            for size in [1, 7, len(samples["raw"])]:
                np.testing.assert_allclose(
                    scorer.score(samples["raw"][:size], name),
                    samples[name][:size],
                    atol=1e-5,
                    rtol=1e-4,
                )
        if scorer.baseline is not None:
            np.testing.assert_allclose(
                scorer.score(samples["raw"], "mad"),
                samples["mad"],
                atol=1e-5,
                rtol=1e-4,
            )
        timings = []
        for _ in range(20):
            start = time.perf_counter()
            scorer.score(samples["raw"][:1], "ae")
            timings.append((time.perf_counter() - start) * 1000)
        (directory / "runtime-verification.json").write_text(
            json.dumps(
                {
                    "ae_single_window_ms_p50": float(np.median(timings)),
                    "ae_single_window_ms_p95": float(np.quantile(timings, 0.95)),
                    "provider": "CPUExecutionProvider",
                    "batches_tested": [1, 7, len(samples["raw"])],
                    "decision_mismatches_on_parity_samples": decision_mismatches,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    replay = client.get(f"/api/replay/{task}").json()
    assert len(replay["scores"]) == 2205
    assert (
        len(client.get(f"/api/cycle/{task}/0").json()["actual"])
        == manifest["models"]["ae"]["shape"][1]
    )
    assert replay["run_id"] == manifest.get("run_id", "legacy")
    assert set(replay["partitions"]) == {
        "train",
        "validation",
        "calibration",
        "test",
        "excluded",
    }
    assert client.get(f"/api/cycle/{task}/2205").status_code == 422
    assert client.get(f"/api/download/{task}").status_code == 200
    with ZipFile(directory / "model-bundle.zip") as package:
        assert set(package.namelist()) <= {
            "ae.onnx",
            "rf.onnx",
            "iforest.onnx",
            "manifest.json",
            "preprocessing.json",
            "metrics.json",
            "baseline.json",
            "mechanism-reference.json",
        }
print(
    json.dumps(
        {
            "onnx_verified": verified,
            "api_tasks_verified": 4,
            "private_material_in_model_bundles": False,
        },
        indent=2,
    )
)
