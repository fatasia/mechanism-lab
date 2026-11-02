"""统一的原始窗口 ONNX 包推理，不依赖训练库、Web 或具体平台。"""

import hashlib
import json
from pathlib import Path

import numpy as np

from phm_core.inference import cpu_session


class PortableScorer:
    def __init__(self, directory: Path):
        self.manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest["schemaVersion"] != 2:
            raise ValueError("需要 schemaVersion2 的原始窗口模型包")
        graph = directory / self.manifest["graph"]["file"]
        if (
            graph.parent.resolve() != directory.resolve()
            or hashlib.sha256(graph.read_bytes()).hexdigest()
            != self.manifest["graph"]["sha256"]
        ):
            raise ValueError("模型包路径或校验和不一致")
        self.session = cpu_session(graph)

    def score(self, values, channels, units, sample_rate_hz):
        contract = self.manifest["input"]
        values = np.asarray(values, dtype="float32")
        if (
            values.ndim != 3
            or list(values.shape[1:]) != contract["shape"][1:]
            or len(values) > self.manifest["execution"]["maxBatch"]
            or len(values) == 0
            or not np.isfinite(values).all()
        ):
            raise ValueError("输入形状、有限值或批量大小不满足模型契约")
        if (
            channels != contract["channels"]
            or units != contract["units"]
            or sample_rate_hz != contract["sampleRateHz"]
        ):
            raise ValueError("通道顺序、单位或采样率不满足模型契约")
        outputs = self.session.run(None, {contract["name"]: values})
        result = {}
        for spec in self.manifest["outputs"]:
            score = outputs[spec["index"]]
            if spec.get("transform") == "negate":
                score = -score
            elif spec.get("transform") == "column1":
                score = score[:, 1]
            result[spec["role"]] = score
        return result
