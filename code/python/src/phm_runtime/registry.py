"""模型先加载核验再切换绑定，旧版本可直接回退；训练发布不等于生产激活。"""

import hashlib
import json
import os
from pathlib import Path
from threading import RLock

import numpy as np

from phm_core.artifacts import run_path
from phm_core.decision import DecisionConfig
from phm_core.inference import BundleScorer
from phm_core.quality import assess_window
from phm_core.support import VALVE_SYNC_TOLERANCE_SAMPLES, valve_support
from phm_runtime.store import StateStore


def promotion_gate(metrics, model):
    if model not in metrics.get("models", {}):
        raise ValueError("模型包不包含所选评分器")
    metric = metrics["models"][model]
    matrix = metric["confusion"]
    normal = sum(matrix[0])
    fpr = matrix[0][1] / normal if normal else None
    reasons = []
    if metrics.get("research_only"):
        reasons.append("研究候选只允许影子运行，尚未通过现场准入")
    if metric["recall"] is None or metric["recall"] < 0.95:
        reasons.append("固定阈值测试召回低于 95%")
    if fpr is None or fpr > 0.01:
        reasons.append("正常周期误报超过 1% 或缺少正常测试样本")
    if metric.get("onnx_max_abs_error", float("inf")) > 1e-4:
        reasons.append("缺少通过的 ONNX 数值检查")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "scope": "offline artifact gate; not site acceptance",
    }


def apply_sync_requirement(quality, manifest, rate, sync):
    """模型包声明需要独立同步时，不满足就判为“不支持”，不评分、不当作正常。

    只有清单里写了 sync.required 才生效：同步敏感性的实测证据来自冻结的阀门双确认候选，
    没有对其它模型做过同样的扰动，不把结论强加给它们。
    """
    spec = manifest.get("sync")
    if spec is None:
        return quality
    if not isinstance(spec, dict):
        raise ValueError("同步契约格式不正确")
    if not spec.get("required"):
        return quality
    sync = {} if sync is None else sync
    if not isinstance(sync, dict):
        raise ValueError("同步契约格式不正确")
    lead = spec.get("expected_lead_s")
    tolerance = spec.get("tolerance_samples", VALVE_SYNC_TOLERANCE_SAMPLES)
    if (
        not isinstance(lead, (int, float))
        or isinstance(lead, bool)
        or not np.isfinite(lead)
        or not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or not 0 <= tolerance <= VALVE_SYNC_TOLERANCE_SAMPLES
    ):
        raise ValueError("同步契约须声明有效的预期偏移和不超过1点的容差")
    support = valve_support(
        rate,
        manifest["sampleRateHz"],
        sync.get("command_edge_s"),
        sync.get("window_start_s"),
        lead,
        sync.get("channel_skew_s"),
        tolerance,
    )
    if support["status"] == "supported":
        return quality
    issues = [{**r, "code": f"sync_{r['code']}"} for r in support["reasons"]]
    status = "invalid" if quality["status"] == "invalid" else "unsupported"
    return {"status": status, "issues": [*quality["issues"], *issues]}


class ModelRegistry:
    def __init__(self, outputs: Path, store: StateStore):
        self.outputs, self.store = outputs, store
        self.loaded = {}
        self.lock = RLock()

    def load(self, task, run_id):
        key = (task, run_id)
        with self.lock:
            if key in self.loaded:
                return self.loaded[key]
        try:
            directory = run_path(self.outputs, task, run_id)
        except FileNotFoundError:
            if task != "valve":
                raise
            # 运行封装与完整训练实验分区，避免稀疏元数据进入六页面的训练结果列表。
            directory = run_path(self.outputs, "valve-runtime", run_id)
            manifest = json.loads(
                (directory / "bundle/manifest.json").read_text(encoding="utf-8")
            )
            if manifest.get("bundleKind") != "valve-fusion-runtime":
                raise ValueError("未知的阀门运行包种类")
        trusted_key = os.environ.get("MECHANISM_TRUSTED_MODEL_KEY")
        trusted_keys = os.environ.get("MECHANISM_TRUSTED_MODEL_KEYS")
        if trusted_keys is not None:
            from phm_core.signing import verify_trust_set

            verify_trust_set(directory, json.loads(trusted_keys))
        elif trusted_key:
            from phm_core.signing import verify_directory

            verify_directory(directory, Path(trusted_key))
        scorer = BundleScorer(directory / "bundle")
        with np.load(directory / "parity.npz") as parity:
            for name in scorer.manifest["models"]:
                actual = scorer.score(parity["raw"][:7], name)
                np.testing.assert_allclose(
                    actual, parity[name][:7], atol=1e-4, rtol=1e-4
                )
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        with self.lock:
            self.loaded[key] = (scorer, metrics)
            # 有界缓存；已经运行的请求仍持有自己的模型引用。
            if len(self.loaded) > 8:
                self.loaded.pop(next(iter(self.loaded)))
            return self.loaded[key]

    def activate(self, task, run_id, model="ae", mode="shadow"):
        _, metrics = self.load(task, run_id)
        gate = promotion_gate(metrics, model)
        if mode == "active" and not gate["passed"]:
            raise ValueError("；".join(gate["reasons"]))
        self.store.bind(task, run_id, model, mode)
        return {"binding": self.store.binding(task), "gate": gate}

    def predict(
        self,
        task,
        entity,
        sequence,
        values,
        channels,
        units,
        rate,
        timestamps=None,
        sync=None,
    ):
        binding = self.store.binding(task)
        if not binding:
            raise ValueError("先选择一个影子或已通过离线门槛的模型版本")
        scorer, _ = self.load(task, binding["run_id"])
        manifest = scorer.manifest
        if channels != manifest["channels"] or units != manifest["units"]:
            raise ValueError("通道顺序或单位与模型契约不一致")
        x = np.asarray(values, dtype="float32")
        quality = assess_window(
            x,
            manifest["models"]["ae"]["shape"][1:],
            rate,
            manifest["sampleRateHz"],
            timestamps=timestamps,
            bounds=manifest.get("quality", {}).get("physical_ranges"),
        )
        quality = apply_sync_requirement(quality, manifest, rate, sync)
        score = (
            float(scorer.score(x[None], binding["model"])[0])
            if quality["status"] == "valid"
            else None
        )
        if score is not None and not np.isfinite(score):
            quality = {"status": "invalid", "issues": [{"code": "model_nonfinite"}]}
            score = None
        threshold = manifest["models"].get(binding["model"], {}).get("threshold")
        if threshold is None:
            baseline = json.loads(
                (
                    run_path(self.outputs, task, binding["run_id"]) / "metrics.json"
                ).read_text()
            )
            threshold = baseline["models"][binding["model"]]["threshold"]
        config = DecisionConfig(threshold, threshold - max(abs(threshold) * 0.2, 1e-8))
        fingerprint_parts = {
            "shape": x.shape,
            "rate": float(rate),
            "timestamps": None
            if timestamps is None
            else np.asarray(timestamps, dtype="float64").tolist(),
        }
        # 只在提供同步信息时纳入指纹，没有同步声明的旧窗口指纹保持不变。
        if sync:
            fingerprint_parts["sync"] = {k: sync[k] for k in sorted(sync)}
        fingerprint = hashlib.sha256(
            x.tobytes() + json.dumps(fingerprint_parts, sort_keys=True).encode()
        ).hexdigest()
        return self.store.consume(
            entity,
            f"{task}/{binding['run_id']}/{binding['model']}",
            sequence,
            score,
            config,
            quality,
            {
                "mode": binding["mode"],
                "task": task,
                "threshold": threshold,
                "input_sha256": fingerprint,
            },
        )
