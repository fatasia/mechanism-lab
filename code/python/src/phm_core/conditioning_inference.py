"""专项候选的离线原始窗口推理；只依赖 NumPy/ORT，不修改运行模型绑定。"""

import hashlib
import json
from pathlib import Path

import numpy as np

from phm_core.actuator import motion_features
from phm_core.conditioning import transform_reference
from phm_core.inference import cpu_session
from phm_core.valve import mechanism_features


class ConditioningScorer:
    def __init__(self, bundle: Path):
        self.bundle = Path(bundle)
        inventory = json.loads((self.bundle / "integrity.json").read_text())
        for name, digest in inventory.items():
            if (
                Path(name).name != name
                or hashlib.sha256((self.bundle / name).read_bytes()).hexdigest()
                != digest
            ):
                raise ValueError("专项模型包完整性校验失败")
        self.report = json.loads((self.bundle / "metrics.json").read_text())
        self.sessions = {}

    def _probabilities(self, name, features):
        if name not in self.sessions:
            self.sessions[name] = cpu_session(self.bundle / name)
        return self.sessions[name].run(None, {"features": features})[1]

    def valve(self, values):
        """PS1/PS2，100Hz，固定[8,12)s；分数与独立健康校准阈值。"""
        score = self._probabilities(
            "valve_timing_rf.onnx",
            mechanism_features(np.asarray(values, dtype="float32")),
        )[:, 1]
        threshold = self.report["valve"]["timing_rf"]["42"]["threshold"]
        return {"score": score, "anomalous": score > threshold, "threshold": threshold}

    def actuator(self, values, contexts, protocol="held-out-profile", conditioned=True):
        """完整80秒记录，25Hz，三通道顺序同原数据；显式工况不从标签推断。"""
        if protocol not in ("held-out-load", "held-out-profile", "record-holdout"):
            raise ValueError("未知评价协议")
        values = np.asarray(values, dtype="float32")
        if values.ndim != 3 or values.shape[1:] != (2000, 3) or len(values) == 0:
            raise ValueError("执行器需要[batch,2000,3]完整记录")
        features = (
            motion_features(values.reshape(-1, 400, 3)).reshape(-1, 5, 20).mean(1)
        )
        reference = json.loads((self.bundle / f"{protocol}-reference.json").read_text())
        # 原始特征对照也执行适用域检查；未投运工况不允许被分类为健康。
        transformed = transform_reference(features, contexts, reference)
        arm = "conditioned_rf" if conditioned else "raw_rf"
        score = self._probabilities(
            f"{protocol}-{arm}.onnx", transformed if conditioned else features
        )
        return {"scores": score, "predicted": score.argmax(1)}
