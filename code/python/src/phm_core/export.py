"""导出只使用标准算子；ONNX 与原模型输出逐样本比较。"""

import hashlib
import json
import shutil
from importlib.metadata import version

import numpy as np
import torch
from skl2onnx import to_onnx

from phm_core.inference import cpu_session


class DeployAE(torch.nn.Module):
    def __init__(self, model, mean, scale):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.from_numpy(mean))
        self.register_buffer("scale", torch.from_numpy(scale))

    def forward(self, values):
        x = (values - self.mean) / self.scale
        return (self.model(x) - x).abs().mean(dim=(1, 2)).unsqueeze(1)


def export_models(
    out, model, forest, classifier, x, features, mean, scale, scores, test_ids, report
):
    deploy = DeployAE(model, mean, scale).eval()
    torch.onnx.export(
        deploy,
        torch.from_numpy(x[:1]),
        str(out / "ae.onnx"),
        input_names=["values"],
        output_names=["score"],
        dynamic_axes={"values": {0: "batch"}, "score": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    for name, estimator in [("iforest", forest), ("rf", classifier)]:
        options = {id(estimator): {"zipmap": False}} if name == "rf" else None
        graph = to_onnx(
            estimator,
            features[:1],
            options=options,
            target_opset={"": 17, "ai.onnx.ml": 3},
        )
        (out / f"{name}.onnx").write_bytes(graph.SerializeToString())
    manifest = {
        "schemaVersion": 1,
        "task": report["task"],
        "models": {},
        "source": report["source"],
        "data_sha256": report["data_sha256"],
    }
    # 在整个测试集等距取样，避免只验证开头一个工况。
    ids = test_ids[np.linspace(0, len(test_ids) - 1, min(64, len(test_ids)), dtype=int)]
    for name in ["ae", "iforest", "rf"]:
        session = cpu_session(out / f"{name}.onnx")
        result = session.run(
            None,
            {session.get_inputs()[0].name: x[ids] if name == "ae" else features[ids]},
        )
        actual = (
            result[0].ravel()
            if name == "ae"
            else (-result[1].ravel() if name == "iforest" else result[1][:, 1])
        )
        np.testing.assert_allclose(actual, scores[name][ids], atol=1e-5, rtol=1e-4)
        report["models"][name]["onnx_max_abs_error"] = float(
            np.max(np.abs(actual - scores[name][ids]))
        )
        manifest["models"][name] = {
            "file": f"{name}.onnx",
            "sha256": hashlib.sha256((out / f"{name}.onnx").read_bytes()).hexdigest(),
            "inputName": session.get_inputs()[0].name,
            "shape": [None, *x.shape[1:]]
            if name == "ae"
            else [None, features.shape[1]],
            "normalizationEmbedded": name == "ae",
            "outputIndex": 0 if name == "ae" else 1,
            "transform": "identity"
            if name == "ae"
            else ("negate" if name == "iforest" else "column1"),
            "threshold": report["models"][name]["threshold"],
            "scoreKind": "class-probability" if name == "rf" else "anomaly-score",
        }
    manifest["channels"] = report["channels"]
    manifest["units"] = report["units"]
    manifest["sampleRateHz"] = report["sample_rate_hz"]
    manifest["windowSeconds"] = report.get("window_seconds", 60)
    manifest["windowStartSeconds"] = report.get("window_start_seconds", 0)
    manifest["run_id"] = report["run_id"]
    manifest["profile"] = report["profile"]
    manifest["source_hash"] = report["source_hash"]
    manifest["split_hash"] = report["split_hash"]
    manifest["versions"] = {
        name: version(name)
        for name in [
            "numpy",
            "torch",
            "scikit-learn",
            "skl2onnx",
            "onnx",
            "onnxruntime",
        ]
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    preprocessing = {
        "signal_profile": report["profile"],
        "source_rate_hz": {
            channel: 100
            if channel.startswith("PS") or channel == "EPS1"
            else report["sample_rate_hz"]
            for channel in report["channels"]
        },
        "resampling": "scipy.signal.resample_poly(up=1,down=10,padtype=line) for 100 Hz channels; no upsampling"
        if report["sample_rate_hz"] == 10
        else "native sampling; fixed window crop; no resampling",
        "window_start_seconds": report.get("window_start_seconds", 0),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "features": [
            f"{stat}:{channel}"
            for stat in ["mean", "std", "rms", "peak_to_peak"]
            for channel in report["channels"]
        ],
    }
    (out / "preprocessing.json").write_text(
        json.dumps(preprocessing, indent=2), encoding="utf-8"
    )
    np.savez(
        out / "parity.npz",
        raw=x[ids],
        features=features[ids],
        **{k: v[ids] for k, v in scores.items()},
    )
    bundle = out / "bundle"
    bundle.mkdir(exist_ok=True)
    for filename in [
        "ae.onnx",
        "iforest.onnx",
        "rf.onnx",
        "manifest.json",
        "preprocessing.json",
        "baseline.json",
    ]:
        shutil.copyfile(out / filename, bundle / filename)
    if (out / "mechanism-reference.json").exists():
        shutil.copyfile(
            out / "mechanism-reference.json", bundle / "mechanism-reference.json"
        )
    shutil.make_archive(str(out / "model-bundle"), "zip", bundle)
