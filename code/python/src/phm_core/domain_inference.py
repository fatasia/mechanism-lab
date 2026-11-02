"""视觉与寿命模型的专用契约；仅依赖 NumPy/ORT，不加载训练库。"""

import hashlib
import json
from pathlib import Path

import numpy as np

from phm_core.inference import cpu_session


class DomainScorer:
    def __init__(self, directory: Path):
        self.manifest = json.loads((directory / "manifest.json").read_text())
        file = directory / self.manifest["file"]
        if (
            file.parent.resolve() != directory.resolve()
            or hashlib.sha256(file.read_bytes()).hexdigest() != self.manifest["sha256"]
        ):
            raise ValueError("模型文件路径或哈希不满足部署契约")
        self.session = cpu_session(file)

    def score(self, values):
        contract = self.manifest["input"]
        values = np.asarray(values, dtype="float32")
        if (
            values.ndim != len(contract["shape"])
            or list(values.shape[1:]) != contract["shape"][1:]
            or not 1 <= len(values) <= self.manifest["maxBatch"]
            or not np.isfinite(values).all()
        ):
            raise ValueError("输入形状、批量大小或有限值不满足部署契约")
        if self.manifest["task"] == "visual-anomaly" and (
            values.min() < 0 or values.max() > 1
        ):
            raise ValueError("视觉输入需要 RGB、BCHW、[0,1]；归一化已在图内")
        outputs = self.session.run(None, {contract["name"]: values})
        if self.manifest["task"] == "remaining-useful-life":
            rule = self.manifest["postprocess"]
            prediction = np.clip(outputs[0].ravel(), *rule["clip_prediction"])
            radius = rule["interval_radius_cycles"]
            return {
                "remaining_cycles": prediction,
                "lower_cycles": np.maximum(0, prediction - radius),
                "upper_cycles": prediction + radius,
            }
        return {
            "score": outputs[0],
            "heatmap": outputs[1],
            "flagged": outputs[0] > self.manifest["threshold"],
        }
