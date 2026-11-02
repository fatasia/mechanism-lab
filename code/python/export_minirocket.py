"""逐项对照 aeon 的留出分数后导出 MiniRocket；分类决策也必须一致。"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from phm_core.artifacts import capture_source, new_run, publish_run
from phm_core.inference import cpu_session
from phm_core.minirocket_onnx import MiniRocketONNX

ROOT = Path(__file__).resolve().parents[2]


def run(reference_run):
    reference = ROOT / "outputs/reference-study/runs" / reference_run
    run_id, out = new_run(ROOT / "outputs", "minirocket-export")
    bundle = out / "bundle"
    bundle.mkdir()
    report = {
        "run_id": run_id,
        "reference_run": reference_run,
        "source_hash": capture_source(
            ROOT / "code/python", out / "train-source.zip", "export_minirocket.py"
        ),
        "models": {},
    }
    torch.set_num_threads(2)
    for protocol in ("record-holdout", "held-out-load", "held-out-profile"):
        with np.load(reference / f"{protocol}-minirocket-parameters.npz") as data:
            parameters = {key: data[key] for key in data.files}
        model = MiniRocketONNX(parameters).eval()
        values, expected = parameters["parity_raw"], parameters["parity_scores"]
        with torch.no_grad():
            native = np.concatenate(
                [
                    model(torch.from_numpy(batch)).numpy()
                    for batch in np.array_split(values, 8)
                ]
            )
        np.testing.assert_allclose(native, expected, atol=1e-3, rtol=1e-3)
        np.testing.assert_array_equal(native.argmax(1), expected.argmax(1))
        path = bundle / f"{protocol}.onnx"
        torch.onnx.export(
            model,
            torch.from_numpy(values[:1]),
            str(path),
            input_names=["values"],
            output_names=["class_scores"],
            dynamic_axes={"values": {0: "batch"}, "class_scores": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        session = cpu_session(path)
        actual = np.concatenate(
            [
                session.run(None, {"values": batch})[0]
                for batch in np.array_split(values, 8)
            ]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
        np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
        timings = []
        for _ in range(20):
            start = time.perf_counter()
            session.run(None, {"values": values[:1]})
            timings.append((time.perf_counter() - start) * 1000)
        report["models"][protocol] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": [None, 400, 3],
            "classes": ["normal", "backlash", "lack_lubrication", "spalling"],
            "max_abs_error_vs_aeon": float(np.abs(actual - expected).max()),
            "decision_mismatches": int((actual.argmax(1) != expected.argmax(1)).sum()),
            "parity_windows": len(values),
            "latency_ms_p50": float(np.median(timings)),
            "latency_ms_p95": float(np.quantile(timings, 0.95)),
        }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    shutil.copyfile(out / "metrics.json", bundle / "manifest.json")
    (bundle / "NOTICE.txt").write_text(
        "MiniRocket: Dempster et al. 2021; fitted with aeon1.6.0 (BSD-3-Clause). Deployment bridge maps public learned numeric parameters to basic ONNX operators. Input: 25Hz16s,setpoint_mm,error_mm,current_A. No aeon/numba dependency at inference."
    )
    shutil.copyfile(
        ROOT / "code/licenses/aeon-BSD-3-Clause.txt", bundle / "aeon-BSD-3-Clause.txt"
    )
    shutil.make_archive(str(out / "model-bundle"), "zip", bundle)
    publish_run(ROOT / "outputs", "minirocket-export", run_id)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_run")
    run(parser.parse_args().reference_run)
