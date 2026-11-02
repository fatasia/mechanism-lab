"""伺服电机电流：整定类故障与松动检出，按转速留出，正常基线随转速插值。

协议在看结果前冻结（见 PROTOCOL）：
- 评估单位是工况点（状态 × 转速），每个工况点只录过一次，段不是独立样本；
- 只用正常数据训练；阈值取训练转速内部留一插值得分的分位数，不看被留出的转速；
- 文件判为异常：超阈值段占比不低于 file_fraction；
- 同时报告与转速无关的全局基线，以及示波器量化步长的混杂审计。
单台无负载电机、实验室条件；不代表其它设备或现场。
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from phm_core.artifacts import capture_source, new_run
from phm_core.current_signature import (
    FEATURES,
    ORDER_FEATURES,
    VIBRATION_BANDS,
    VIBRATION_FEATURES,
    GlobalBaseline,
    SpeedConditionedBaseline,
    fixed_segments,
    order_features,
    quantization_step,
    read_rigol,
    read_wav,
    segment_features,
    to_segments,
)
from phm_core.evidence import clopper_pearson, one_sided_upper

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/servo-dc"
SENSORS = {
    "current": {
        "pattern": re.compile(r"^(.+)_no_reversal_speed(\d{3})_current\.bin$"),
        "glob": "*_current.bin",
        "manifest": "source.json",
        "task": "servo-tuning-study",
        "audit": "quantization_step",
    },
    "vibration": {
        "pattern": re.compile(r"^(.+)_no_reversal_speed(\d{3})_vibration\.wav$"),
        "glob": "*_vibration.wav",
        "manifest": "source-vibration.json",
        "task": "servo-vibration-study",
        "audit": "clipped_fraction",
    },
    # 第二个特征族：转速同步谐波。看到频带族零检出后才加入，依据是松动的物理机理，
    # 不是从结果里挑的；只用基频 >= 3 Hz（转速 >= 10%）的文件。
    "vibration_orders": {
        "pattern": re.compile(r"^(.+)_no_reversal_speed(\d{3})_vibration\.wav$"),
        "glob": "*_vibration.wav",
        "manifest": "source-vibration.json",
        "task": "servo-vibration-orders-study",
        "audit": "clipped_fraction",
        "min_speed_percent": 10,
    },
}
RATED_RPM = 2000.0
FAULTS = ("loose_foundation", "suboptimal_control", "suboptimal_control_rt")
PROTOCOL = {
    "target_rate_hz": 20_000.0,
    "segment_seconds": 1.0,
    "threshold_quantiles": [0.95, 0.99, 0.999],
    "file_fraction": 0.5,
    "min_training_speeds": 5,
    "labels": {
        "suboptimal_control": "speed-regulator gain -98%",
        "suboptimal_control_rt": "current-regulator gain -95%",
        "loose_foundation": "mounting bolts loosened by hand, degree unmeasured",
    },
}


def load_features(sensor="current"):
    """逐文件提取特征和混杂审计量；顺序固定，文件名即状态与转速。

    电流：降采样到 20 kHz，审计量是示波器量化步长。
    振动：44.1 kHz 原采样率取第 0 通道，审计量是削波占比。
    """
    spec = SENSORS[sensor]
    features, audit = {}, {}
    for path in sorted(DATA.glob(spec["glob"])):
        match = spec["pattern"].match(path.name)
        if match is None:
            continue
        key = (match[1], int(match[2]))
        if key[1] < spec.get("min_speed_percent", 0):
            continue
        if sensor == "current":
            values, interval = read_rigol(path)
            segments, rate = to_segments(
                values,
                interval,
                PROTOCOL["target_rate_hz"],
                PROTOCOL["segment_seconds"],
            )
            features[key] = segment_features(segments, rate)
            audit[key] = quantization_step(values)
        else:
            channels, rate, clipped = read_wav(path)
            segments = fixed_segments(channels[0], rate, PROTOCOL["segment_seconds"])
            if sensor == "vibration_orders":
                rotation_hz = key[1] / 100.0 * RATED_RPM / 60.0
                features[key] = order_features(segments, rate, rotation_hz)
            else:
                features[key] = segment_features(segments, rate, VIBRATION_BANDS)
            audit[key] = clipped
    # 各文件时长略有差异；统一截取共有的前 N 段，使各转速的段数相同。
    common = min(len(v) for v in features.values())
    return {k: v[:common] for k, v in features.items()}, audit


def decide(scores, threshold):
    fraction = float(np.mean(scores > threshold))
    return fraction, fraction >= PROTOCOL["file_fraction"]


def global_inner_threshold(speeds, features, quantile):
    """全局基线同样用留一转速得分定阈值，避免对照被训练内偏乐观的阈值占便宜。"""
    scores = []
    for held in range(len(speeds)):
        keep = [i for i in range(len(speeds)) if i != held]
        scores.append(GlobalBaseline(features[keep]).score(features[held]))
    return float(np.quantile(np.concatenate(scores), quantile))


def evaluate(features, quantile):
    normal_speeds = sorted(s for (op, s) in features if op == "normal")
    rows = []
    for held in normal_speeds:
        train = [s for s in normal_speeds if s != held]
        if len(train) < PROTOCOL["min_training_speeds"]:
            continue
        train_features = np.stack([features[("normal", s)] for s in train])
        conditioned = SpeedConditionedBaseline(train, train_features)
        threshold = float(
            np.quantile(conditioned.inner_scores(train_features), quantile)
        )
        global_baseline = GlobalBaseline(train_features)
        global_threshold = global_inner_threshold(train, train_features, quantile)
        for operation in ("normal", *FAULTS):
            if (operation, held) not in features:
                continue
            f = features[(operation, held)]
            c_fraction, c_flag = decide(conditioned.score(f, held), threshold)
            g_fraction, g_flag = decide(global_baseline.score(f), global_threshold)
            rows.append(
                {
                    "operation": operation,
                    "speed_percent": held,
                    "conditioned_fraction": c_fraction,
                    "conditioned_flagged": c_flag,
                    "global_fraction": g_fraction,
                    "global_flagged": g_flag,
                }
            )
    return rows


def summarise(rows):
    result = {}
    for operation in ("normal", *FAULTS):
        selected = [r for r in rows if r["operation"] == operation]
        entry = {"files": len(selected)}
        for arm in ("conditioned", "global"):
            flagged = sum(r[f"{arm}_flagged"] for r in selected)
            entry[arm] = {
                "flagged_files": flagged,
                "rate": flagged / len(selected) if selected else None,
                "interval_95": list(clopper_pearson(flagged, len(selected)))
                if selected
                else None,
                "mean_segment_fraction": float(
                    np.mean([r[f"{arm}_fraction"] for r in selected])
                )
                if selected
                else None,
            }
        if operation == "normal" and selected:
            entry["false_alarm_one_sided_upper_95"] = one_sided_upper(
                entry["conditioned"]["flagged_files"], len(selected)
            )
        result[operation] = entry
    return result


def by_speed_band(rows, arm="conditioned"):
    """低速（<=5%）电流量化严重、纹波淹没，单独列出，避免平均掩盖。"""
    bands = {"low_<=5": (0, 5), "mid_10-50": (10, 50), "high_>=60": (60, 100)}
    out = {}
    for name, (low, high) in bands.items():
        out[name] = {}
        for operation in ("normal", *FAULTS):
            selected = [
                r
                for r in rows
                if r["operation"] == operation and low <= r["speed_percent"] <= high
            ]
            out[name][operation] = {
                "files": len(selected),
                "flagged": sum(r[f"{arm}_flagged"] for r in selected),
            }
    return out


def quantization_audit(steps):
    """量化步长（≈垂直量程）按状态汇总；与状态相关就会造成假检出。"""
    table = {}
    for (operation, speed), step in sorted(steps.items()):
        table.setdefault(operation, {})[speed] = step
    summary = {
        op: {
            "min": min(v.values()),
            "max": max(v.values()),
            "distinct": len({round(x, 6) for x in v.values()}),
        }
        for op, v in table.items()
    }
    medians = {op: float(np.median(list(v.values()))) for op, v in table.items()}
    return {"per_file": table, "summary": summary, "median_by_operation": medians}


def run(sensor="current"):
    spec = SENSORS[sensor]
    features, audit = load_features(sensor)
    expected = json.loads((DATA / spec["manifest"]).read_text(encoding="utf-8"))[
        "files"
    ]
    # 清单是全部已校验文件；速度过滤会合法地少读一部分。
    skipped = sum(
        1
        for item in expected
        if int(re.search(r"speed(\d{3})", item["file"])[1])
        < spec.get("min_speed_percent", 0)
    )
    if len(expected) - skipped != len(features):
        raise ValueError(
            f"文件数不一致: 清单 {len(expected)} 减去按速度排除的 {skipped} 与已读取 {len(features)}"
        )
    run_id, output = new_run(ROOT / "outputs", spec["task"])
    report = {
        "run_id": run_id,
        "sensor": sensor,
        "protocol": PROTOCOL,
        "features": list(
            FEATURES
            if sensor == "current"
            else ORDER_FEATURES
            if sensor == "vibration_orders"
            else VIBRATION_FEATURES
        ),
        "scope": "single unloaded brushed DC servo motor, one recording per operating point, no-reversal only; leave-one-speed-out; exploratory",
        "source_hash": capture_source(
            ROOT / "code/python", output / "source.zip", "study_servo_tuning.py"
        ),
        "source_manifest_sha256": hashlib.sha256(
            (DATA / spec["manifest"]).read_bytes()
        ).hexdigest(),
        "confound_audit": {"measure": spec["audit"], **quantization_audit(audit)},
        "operating_points": {},
    }
    for quantile in PROTOCOL["threshold_quantiles"]:
        rows = evaluate(features, quantile)
        report["operating_points"][str(quantile)] = {
            "rows": rows,
            "summary": summarise(rows),
            "by_speed_band_conditioned": by_speed_band(rows),
        }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", choices=sorted(SENSORS), default="current")
    run(parser.parse_args().sensor)
