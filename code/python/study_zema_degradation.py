"""ZeMA 机电缸退化趋势：整轴留出，跨设备选特征，评价趋势性与持续越界位置。

数据没有故障时刻标签，数值是未校准的相对量；这里不报告剩余寿命，也不把试验结束当作
维修确认的故障。只回答：在没见过的那台设备上，健康指数能否稳定地、
不早于参考期地、早于振动剧增地给出退化起点。
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

from phm_core.artifacts import capture_source, new_run
from phm_core.degradation import (
    FEATURES,
    cycle_features,
    health_index,
    persistent_crossing,
    reference_threshold,
    reference_z,
    rolling_median,
    select_consistent,
)

ROOT = Path(__file__).resolve().parents[2]
SENSORS = (
    "microphone",
    "acc_plain_bearing",
    "acc_piston_rod",
    "acc_ball_bearing",
    "axial_force",
    "pressure",
    "velocity",
    "active_current",
    "motor_current_1",
    "motor_current_2",
    "motor_current_3",
)
PROTOCOL = {
    "reference_fraction": 0.1,
    "selection_min_abs_spearman": 0.5,
    "selection_smooth_cycles": 51,
    "index_smooth_cycles": 101,
    "threshold_sigmas": 6.0,
    "threshold_floor": 0.5,
    "persistence_cycles": 100,
    "early_flag_fraction": 0.2,
}


def load_axis(path: Path):
    """返回 [周期, 传感器*特征] 的特征矩阵和特征名；逐传感器读取以控制内存。"""
    columns, names = [], []
    with h5py.File(path, "r") as stream:
        data = stream["Sensor_Data"]
        if data.shape[0] != len(SENSORS) or data.shape[1] != 2000:
            raise ValueError(f"数据形状与文档不一致: {data.shape}")
        for sensor, name in enumerate(SENSORS):
            block = np.asarray(data[sensor, :, :], dtype=float)
            if not np.isfinite(block).all():
                raise ValueError(f"{path.name} 的 {name} 含非有限值")
            columns.append(cycle_features(block))
            names.extend(f"{name}.{f}" for f in FEATURES)
    return np.concatenate(columns, axis=1), names


def describe(index, count, protocol=PROTOCOL):
    threshold = reference_threshold(
        index, count, protocol["threshold_sigmas"], protocol["threshold_floor"]
    )
    cycle, fraction = persistent_crossing(
        index, threshold, count, protocol["persistence_cycles"]
    )
    rho = spearmanr(np.arange(len(index)), index).statistic
    return {
        "threshold": threshold,
        "trend_spearman": None if np.isnan(rho) else float(rho),
        "first_persistent_crossing_cycle": cycle,
        "first_persistent_crossing_life_fraction": fraction,
        "crossing_before_early_flag": fraction is not None
        and fraction < protocol["early_flag_fraction"],
        "life_cycles": len(index),
    }


def evaluate_target(target, zs, counts, names):
    sources = [k for k in zs if k != target]
    indices, signs = select_consistent(
        [zs[k] for k in sources],
        PROTOCOL["selection_min_abs_spearman"],
        PROTOCOL["selection_smooth_cycles"],
    )
    z, count = zs[target], counts[target]
    smooth = PROTOCOL["index_smooth_cycles"]
    methods = {}
    methods["cross_axis_selected"] = describe(
        health_index(z, indices, signs, smooth), count
    )
    methods["cross_axis_selected"]["features"] = [
        f"{names[i]}({'+' if s > 0 else '-'})" for i, s in zip(indices, signs)
    ]
    all_features = rolling_median(np.mean(np.abs(z), axis=1), smooth)
    methods["all_features_no_selection"] = describe(all_features, count)
    rod = names.index("acc_piston_rod.std")
    methods["single_acc_piston_rod_std"] = describe(
        rolling_median(z[:, rod], smooth), count
    )
    return {"sources": sources, "methods": methods}


def load_axis11(path: Path):
    """axis11 按传感器分组、已换算为 SI；通道顺序与前三台一致，特征在各自参考期内标准化，单位不影响。"""
    order = (
        ("Sound_Pressure", None),
        ("Acceleration", 0),
        ("Acceleration", 1),
        ("Acceleration", 2),
        ("Force", None),
        ("Pressure", None),
        ("Velocity", None),
        ("Active_Current", None),
        ("Motor_Current", 0),
        ("Motor_Current", 1),
        ("Motor_Current", 2),
    )
    columns, names = [], []
    with h5py.File(path, "r") as stream:
        for (group, channel), name in zip(order, SENSORS, strict=True):
            data = stream[f"ZeMA_DAQ/{group}/qudt:value"]
            block = np.asarray(
                data[:, :] if channel is None else data[channel], dtype=float
            )
            if block.shape[0] != 2000 or not np.isfinite(block).all():
                raise ValueError(f"axis11 的 {name} 形状或数值异常: {block.shape}")
            columns.append(cycle_features(block))
            names.extend(f"{name}.{f}" for f in FEATURES)
    return np.concatenate(columns, axis=1), names


def run(files, axis11=None):
    run_id, output = new_run(ROOT / "outputs", "zema-degradation")
    features, zs, counts = {}, {}, {}
    names = None
    for path in files:
        axis = path.stem.split("_")[0]
        features[axis], names = load_axis(path)
        zs[axis], _, counts[axis] = reference_z(
            features[axis], PROTOCOL["reference_fraction"]
        )
    original = sorted(zs)
    extra = None
    if axis11 is not None:
        features["axis11"], names11 = load_axis11(axis11)
        if names11 != names:
            raise ValueError("axis11 特征名与前三台不一致")
        zs["axis11"], _, counts["axis11"] = reference_z(
            features["axis11"], PROTOCOL["reference_fraction"]
        )
        extra = {
            # 协议在评估前已冻结：只用原三台选特征，axis11 不参与任何选择。
            "frozen_selection_from_original_three": evaluate_target(
                "axis11", {k: zs[k] for k in (*original, "axis11")}, counts, names
            ),
            "four_axis_leave_one_out": {
                axis: evaluate_target(axis, zs, counts, names) for axis in sorted(zs)
            },
        }
    report = {
        "run_id": run_id,
        "protocol": PROTOCOL,
        "scope": "ZeMA EMC, 3 axes (+axis11 when given), relative units, no failure labels; leave-one-axis-out; exploratory, cannot support RUL or field accuracy",
        "source_hash": capture_source(
            ROOT / "code/python", output / "source.zip", "study_zema_degradation.py"
        ),
        "data_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*files, *([axis11] if axis11 else [])]
        },
        "cycles": {a: len(f) for a, f in features.items()},
        "targets": {
            axis: evaluate_target(axis, {k: zs[k] for k in original}, counts, names)
            for axis in original
        },
        "axis11": extra,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        type=Path,
        nargs="*",
        default=[ROOT / f"data/raw/zema-emc/axis{n}_2kHz.h5" for n in (3, 5, 7)],
    )
    parser.add_argument("--axis11", type=Path, default=None)
    args = parser.parse_args()
    run(args.files, args.axis11)
