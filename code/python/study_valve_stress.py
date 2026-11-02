"""冻结阀门ONNX的输入扰动敏感性；合成扰动不是新增故障或现场精度。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from phm_core.artifacts import capture_source, new_run
from phm_core.data import block_split
from phm_core.evaluation import evaluate
from phm_core.profiles import load_profile

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "outputs/valve-fusion-export/runs/20261031T191856Z-a74b1b5d"


def scenarios():
    """固定网格，不根据扰动结果选门槛；数值不是传感器规格容差。"""
    result = [{"name": "unchanged", "kind": "identity", "value": 0}]
    for kind, values in {
        "offset_bar": [-2, -0.5, 0.5, 2],
        "gain_fraction": [-0.05, -0.01, 0.01, 0.05],
        "shift_samples": [-5, -2, -1, 1, 2, 5],
        "noise_std_bar": [0.1, 0.5],
    }.items():
        for channels in ([0], [1], [0, 1]):
            for value in values:
                result.append(
                    {
                        "name": f"{kind}_{value}_ch{''.join(map(str, channels))}",
                        "kind": kind,
                        "value": value,
                        "channels": channels,
                    }
                )
    return result


def perturb(values, scenario, seed=20261101):
    output = values.copy()
    kind, value = scenario["kind"], scenario["value"]
    if kind == "identity":
        return output
    channels = scenario["channels"]
    if kind == "offset_bar":
        output[:, :, channels] += value
    elif kind == "gain_fraction":
        output[:, :, channels] *= 1 + value
    elif kind == "shift_samples":
        # 正值延迟、负值提前，边界保持端点；绝不环绕拼接窗口尾部。
        indices = np.clip(
            np.arange(values.shape[1]) - int(value), 0, values.shape[1] - 1
        )
        output[:, :, channels] = values[:, indices, :][:, :, channels]
    elif kind == "noise_std_bar":
        noise = np.random.default_rng(seed).normal(0, value, values.shape)
        output[:, :, channels] += noise[:, :, channels]
    else:
        raise ValueError(f"未知扰动类型: {kind}")
    return output


def run(bundle=BUNDLE):
    run_id, out = new_run(ROOT / "outputs", "valve-stress-study")
    protocol = {
        "purpose": "synthetic input sensitivity, not field accuracy or new fault data",
        "model_frozen": bundle.name,
        "threshold_frozen": 1.0,
        "labels": "unchanged historical labels assumed for perturbation comparison only",
        "grid": scenarios(),
        "noise_seed": 20261101,
        "shift_boundary": "edge hold, positive delay, no wrapping; 1 sample = 10ms",
        "selection": "none; no tuning, retraining or online promotion",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    model_hash = hashlib.sha256((bundle / "model.onnx").read_bytes()).hexdigest()
    if model_hash != manifest["sha256"]:
        raise ValueError("冻结模型哈希不一致")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(bundle / "model.onnx"), options, providers=["CPUExecutionProvider"]
    )
    values, labels, digest = load_profile(
        ROOT / "data/raw/uci-hydraulic/dataset.zip",
        "valve-switch-100hz",
        ROOT / "data/prepared",
    )
    original_ids = block_split(len(values))["test"]
    values = values[original_ids].astype("float32")
    healthy = labels[original_ids, 1] == 100
    ids = np.arange(len(values))
    baseline = session.run(["score"], {"values": values})[0].ravel()
    report = {
        "run_id": run_id,
        "protocol": protocol,
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", "study_valve_stress.py"
        ),
        "data_sha256": digest,
        "model_sha256": model_hash,
        "original_cycle_ids": original_ids.tolist(),
        "scenarios": {},
    }
    score_archive = {"original_cycle_ids": original_ids, "healthy": healthy}
    for scenario in protocol["grid"]:
        x = perturb(values, scenario, protocol["noise_seed"])
        score = session.run(["score"], {"values": x})[0].ravel()
        item = evaluate(score, healthy, ids, 1.0)
        for key in ("false_positive_ids", "false_negative_ids"):
            item[key] = original_ids[item[key]].tolist()
        item.update(
            decision_changes=int(np.sum((score > 1) != (baseline > 1))),
            max_abs_score_change=float(np.max(np.abs(score - baseline))),
            score_quantiles=np.quantile(score, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        )
        report["scenarios"][scenario["name"]] = item
        score_archive[scenario["name"]] = score
    np.savez_compressed(out / "scores.npz", **score_archive)
    (out / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "scenarios": len(protocol["grid"]),
                "changed_scenarios": sum(
                    v["decision_changes"] > 0 for v in report["scenarios"].values()
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    run(parser.parse_args().bundle.resolve())
