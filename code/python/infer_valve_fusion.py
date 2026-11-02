"""独立阀门候选推理；只依赖NumPy和ONNX Runtime，不依赖平台或训练包。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def predict(bundle: Path, values: np.ndarray):
    """values为PS1/PS2原始bar窗口；缺失值不能按正常处理。"""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    model = bundle / "model.onnx"
    if hashlib.sha256(model.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("模型校验和与清单不一致")
    x = np.asarray(values, dtype="float32")
    if (
        x.ndim != 3
        or x.shape[1:] != (400, 2)
        or len(x) == 0
        or not np.isfinite(x).all()
    ):
        raise ValueError(
            "输入须为有限的float32[N,400,2]，PS1/PS2 bar，100Hz的[8,12)s窗口"
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(model), options, providers=["CPUExecutionProvider"]
    )
    score = session.run(["score"], {"values": x})[0].ravel()
    if not np.isfinite(score).all():
        raise ValueError("模型返回非有限分数，拒绝生成正常/故障判定")
    return {"score": score, "anomalous": score > manifest["threshold"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--input", type=Path, required=True, help="float32[N,400,2]的.npy文件"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="分数与异常布尔值的.npz文件"
    )
    args = parser.parse_args()
    result = predict(args.bundle, np.load(args.input, allow_pickle=False))
    np.savez_compressed(args.output, **result)
    print(
        json.dumps(
            {
                "windows": len(result["score"]),
                "anomalous": int(result["anomalous"].sum()),
            }
        )
    )
