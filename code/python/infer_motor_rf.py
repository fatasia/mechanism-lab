"""独立电机RF推理：只依赖NumPy/ONNX Runtime；输入为约定的114维工程特征。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def aggregate(probabilities, window_records, requested):
    """记录等权合并有效窗口；无有效窗口不生成健康标签。"""
    result = np.full((len(requested), probabilities.shape[1]), np.nan, dtype="float32")
    counts = np.zeros(len(requested), dtype="int64")
    for i, record in enumerate(requested):
        mask = window_records == record
        counts[i] = int(mask.sum())
        if counts[i]:
            result[i] = probabilities[mask].mean(0)
    supported = counts > 0
    labels = np.full(len(requested), -1, dtype="int64")
    labels[supported] = result[supported].argmax(1)
    return {
        "record_ids": requested,
        "probabilities": result,
        "labels": labels,
        "supported": supported,
        "active_windows": counts,
    }


def predict(bundle, features, window_records, requested=None):
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    model = bundle / "model.onnx"
    if hashlib.sha256(model.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("模型与清单哈希不一致")
    x, ids = np.asarray(features, dtype="float32"), np.asarray(window_records)
    if x.ndim != 2 or x.shape[1] != 114 or not np.isfinite(x).all():
        raise ValueError("输入需要有限[N,114]工程特征，顺序见manifest.feature_names")
    if ids.shape != (len(x),) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("每个特征窗口需要一个整数record_id")
    requested = np.unique(ids) if requested is None else np.asarray(requested)
    if (
        requested.ndim != 1
        or not np.issubdtype(requested.dtype, np.integer)
        or len(np.unique(requested)) != len(requested)
    ):
        raise ValueError("requested_record_ids需要不重复整数")
    if len(x):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(
            str(model), options, providers=["CPUExecutionProvider"]
        )
        probabilities = session.run([manifest["probability_output"]], {"features": x})[
            0
        ]
    else:
        probabilities = np.empty((0, len(manifest["classes"])), dtype="float32")
    return aggregate(probabilities, ids, requested)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="npz:features,window_records,optional requested_record_ids",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        result = predict(
            args.bundle,
            data["features"],
            data["window_records"],
            data.get("requested_record_ids"),
        )
    np.savez_compressed(args.output, **result)
    print(
        json.dumps(
            {
                "records": len(result["record_ids"]),
                "supported": int(result["supported"].sum()),
                "unsupported": int((~result["supported"]).sum()),
            }
        )
    )
