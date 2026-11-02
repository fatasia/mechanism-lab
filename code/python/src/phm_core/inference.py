"""从发布目录推理，不导入训练器、PyTorch 或平台 API。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from phm_core.features import cycle_features


def cpu_session(path: Path) -> ort.InferenceSession:
    """小窗口 CPU 推理固定单线程，避免多个模型各自抢占整台机器。"""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


class BundleScorer:
    def __init__(self, directory: Path):
        self.manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        self.sessions = {}
        for name, item in self.manifest["models"].items():
            path = directory / item["file"]
            if (
                path.parent.resolve() != directory.resolve()
                or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
            ):
                raise ValueError("模型文件路径或哈希不匹配")
            self.sessions[name] = cpu_session(path)
        self.baseline = (
            json.loads((directory / "baseline.json").read_text(encoding="utf-8"))
            if (directory / "baseline.json").exists()
            else None
        )

    def score(self, values: np.ndarray, model: str = "ae") -> np.ndarray:
        """values 必须按 manifest 通道顺序排列，单位已由调用方核对。"""
        shape = self.manifest["models"]["ae"]["shape"][1:]
        values = np.asarray(values, dtype="float32")
        if (
            values.ndim != 3
            or len(values) == 0
            or list(values.shape[1:]) != shape
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"输入须为有限数值的 [batch,{shape[0]},{shape[1]}]")
        if model == "mad":
            if self.baseline is None:
                raise ValueError("旧模型包没有稳健统计参数")
            center, scale = (
                np.asarray(self.baseline["center"]),
                np.asarray(self.baseline["scale"]),
            )
            active = scale > 0
            return np.max(
                np.abs(cycle_features(values)[:, active] - center[active])
                / scale[active],
                axis=1,
            )
        item = self.manifest["models"][model]
        inputs = values if model == "ae" else cycle_features(values)
        result = self.sessions[model].run(None, {item["inputName"]: inputs})[
            item["outputIndex"]
        ]
        if item["transform"] == "negate":
            result = -result
        elif item["transform"] == "column1":
            result = result[:, 1]
        return result.ravel()
