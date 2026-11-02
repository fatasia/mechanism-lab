"""导出冻结AE局部残差与换向机理双证据图，含原始波形预处理。"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from phm_core.artifacts import capture_source, new_run
from phm_core.deep import DenseAutoencoder
from phm_core.inference import cpu_session
from phm_core.profiles import load_profile

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "20261031T052816Z-a59e08dc"
FUSION = "20261031T191451Z-1892de96"


class FusionGraph(torch.nn.Module):
    """标准ONNX算子：归一化、AE、TopK、压力首次越阈与最小双证据分数。"""

    def __init__(self, model, normalization, reference, ae_threshold, timing_threshold):
        super().__init__()
        self.model = model
        for name, value in {
            "mean": normalization["mean"],
            "scale": normalization["scale"],
            "center": reference["center"],
            "spread": reference["scale"],
        }.items():
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
        self.ae_threshold = ae_threshold
        self.timing_threshold = timing_threshold

    def timing_features(self, values):
        before = values[:, :100].mean(1)
        after = values[:, -100:].mean(1)
        direction = values.new_tensor([-1.0, 1.0])
        amplitude = (after - before) * direction
        progress = (
            (values - before.unsqueeze(1))
            * direction
            / amplitude.clamp(min=1.0).unsqueeze(1)
        )
        times = []
        for channel in range(2):
            for fraction in [0.1, 0.5, 0.9]:
                crossing = progress[:, :, channel] >= fraction
                index = crossing.to(torch.int64).argmax(1)
                index = torch.where(crossing.any(1), index, torch.full_like(index, 400))
                # NumPy机理配方先以双精度算秒及差值，再转float32；保持相同顺序。
                times.append(8.0 + index.to(torch.float64) / 100.0)
        return torch.stack(
            times + [times[2] - times[0], times[5] - times[3], times[1] - times[4]],
            dim=1,
        ).to(torch.float32)

    def forward(self, values):
        z = (values - self.mean) / self.scale
        residual = (self.model(z) - z).abs().flatten(1)
        ae = residual.topk(80, dim=1).values.mean(1) / self.ae_threshold
        deviation = (self.timing_features(values) - self.center) / self.spread
        evidence = torch.cat(
            [deviation[:, :6].clamp(min=0), deviation[:, 6:].abs()], dim=1
        )
        timing = evidence.max(1).values / self.timing_threshold
        return torch.minimum(ae, timing).unsqueeze(1)


def run(baseline_run_id=BASELINE, fusion_run_id=FUSION):
    torch.set_num_threads(2)
    parent = ROOT / "outputs/valve-fusion-study/runs" / fusion_run_id
    report = json.loads((parent / "metrics.json").read_text(encoding="utf-8"))
    if report["selected"] != "top10":
        raise ValueError("导出入口只对应已选定top10双证据候选")
    baseline = ROOT / "outputs/valve/runs" / baseline_run_id
    normalization = json.loads(
        (baseline / "preprocessing.json").read_text(encoding="utf-8")
    )
    model = DenseAutoencoder((400, 2))
    model.load_state_dict(
        torch.load(baseline / "ae.pt", map_location="cpu", weights_only=True)
    )
    arm = report["arms"]["top10"]
    graph = FusionGraph(
        model,
        normalization,
        report["mechanism_reference"],
        arm["ae_threshold"],
        arm["timing_threshold"],
    ).eval()
    x, _, digest = load_profile(
        ROOT / "data/raw/uci-hydraulic/dataset.zip",
        "valve-switch-100hz",
        ROOT / "data/prepared",
    )
    if digest != report["data_sha256"]:
        raise ValueError("导出验证输入与原实验数据摘要不一致")
    run_id, out = new_run(ROOT / "outputs", "valve-fusion-export")
    source_hash = capture_source(
        ROOT / "code/python", out / "export-source.zip", "export_valve_fusion.py"
    )
    torch.onnx.export(
        graph,
        torch.from_numpy(x[:1]),
        str(out / "model.onnx"),
        input_names=["values"],
        output_names=["score"],
        dynamic_axes={"values": {0: "batch"}, "score": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    session = cpu_session(out / "model.onnx")
    actual = np.concatenate(
        [
            session.run(None, {"values": batch})[0].ravel()
            for batch in np.array_split(x, 35)
        ]
    )
    with np.load(parent / "scores.npz") as scores:
        expected = scores["top10"]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    np.testing.assert_array_equal(actual > 1, expected > 1)
    manifest = {
        "schemaVersion": 1,
        "run_id": run_id,
        "source_run": fusion_run_id,
        "baseline_run": baseline_run_id,
        "task": "valve",
        "status": "research candidate; historical same-rig evaluation, no site acceptance",
        "file": "model.onnx",
        "sha256": hashlib.sha256((out / "model.onnx").read_bytes()).hexdigest(),
        "inputName": "values",
        "shape": [None, 400, 2],
        "dtype": "float32",
        "requiresFiniteInput": True,
        "channels": ["PS1", "PS2"],
        "units": ["bar", "bar"],
        "sampleRateHz": 100,
        "windowStartSeconds": 8,
        "windowSeconds": 4,
        "outputName": "score",
        "threshold": 1.0,
        "decision": "score > threshold",
        "normalizationEmbedded": True,
        "mechanismEmbedded": True,
        "validated_records": len(x),
        "max_abs_error": float(np.max(np.abs(actual - expected))),
        "decision_mismatches": int(np.sum((actual > 1) != (expected > 1))),
        "test": arm["test"],
        "source_hash": source_hash,
        "inferenceSourceSha256": hashlib.sha256(
            (ROOT / "code/python/infer_valve_fusion.py").read_bytes()
        ).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    np.savez_compressed(out / "parity.npz", expected=expected, actual=actual)
    bundle = out / "bundle"
    bundle.mkdir()
    for name in ("model.onnx", "manifest.json"):
        shutil.copyfile(out / name, bundle / name)
    shutil.copyfile(
        ROOT / "code/THIRD_PARTY_NOTICES.md", bundle / "THIRD_PARTY_NOTICES.md"
    )
    shutil.copyfile(ROOT / "code/python/infer_valve_fusion.py", bundle / "infer.py")
    (bundle / "README.txt").write_text(
        "Input: finite float32[N,400,2], raw PS1/PS2 bar, 100Hz, crop [8,12)s. Output: score[N,1], score>1 abnormal. Normalization, timing features, frozen AE and branch thresholds embedded. Research candidate only. Labels and stability flags are not inference inputs.\nInstall numpy and onnxruntime. Run: python infer.py --bundle . --input windows.npy --output scores.npz\nNonfinite inputs and scores must be rejected, not classified as normal.",
        encoding="utf-8",
    )
    shutil.make_archive(str(out / "model-bundle"), "zip", bundle)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-id", default=BASELINE)
    parser.add_argument("--fusion-run-id", default=FUSION)
    args = parser.parse_args()
    run(args.baseline_run_id, args.fusion_run_id)
