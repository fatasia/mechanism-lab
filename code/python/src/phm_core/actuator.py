"""执行器原始记录、工况隔离及跟随误差/电流的相位特征。"""

import re
from pathlib import Path

import numpy as np

CLASSES = ("normal", "backlash", "lack_lubrication", "spalling")
FEATURES = tuple(
    f"{phase}_{feature}"
    for phase in ("all", "forward", "reverse", "dwell")
    for feature in ("error_rms_mm", "error_bias_mm", "current_rms_A", "current_p95_A")
) + ("error_peak_mm", "error_p95_mm", "current_peak_A", "current_diff_rms_A")


def load_records(directory: Path):
    """完整 80 秒记录分成 5 个动作周期，同一记录的周期始终在同一分区。"""
    from scipy.io import loadmat

    windows, labels, records, metadata = [], [], [], []
    for path in sorted(directory.glob("*.mat")):
        label = next(
            i
            for i, prefix in enumerate(
                ("Normal", "Backlash", "LackLubrication", "Spalling")
            )
            if path.stem.startswith(prefix)
        )
        for key, values in loadmat(path, simplify_cells=True).items():
            if key.startswith("__"):
                continue
            match = re.search(r"(neg40|40|20)kg(\d+)$", key)
            if match is None or values.shape != (2000, 3):
                raise ValueError(f"原始记录契约不一致: {path.name}/{key}")
            record_id = len(metadata)
            metadata.append(
                {
                    "id": record_id,
                    "file": path.name,
                    "key": key,
                    "load": match[1],
                    "repeat": int(match[2]),
                    "profile": "sin" if "sin" in key else "trap",
                    "label": label,
                }
            )
            windows.extend(values.reshape(5, 400, 3).astype("float32"))
            labels.extend([label] * 5)
            records.extend([record_id] * 5)
    return np.asarray(windows), np.asarray(labels), np.asarray(records), metadata


def split_records(metadata, protocol="held-out-load"):
    partitions = {k: [] for k in ("train", "validation", "calibration", "test")}
    for record in metadata:
        if protocol == "record-holdout":
            name = (
                "train"
                if record["repeat"] <= 5
                else "validation"
                if record["repeat"] <= 7
                else "calibration"
                if record["repeat"] == 8
                else "test"
            )
            partitions[name].append(record["id"])
            continue
        held_out = (
            record["load"] == "neg40"
            if protocol == "held-out-load"
            else record["profile"] == "sin"
        )
        name = (
            "test"
            if held_out
            else "train"
            if record["repeat"] <= 6
            else "validation"
            if record["repeat"] <= 8
            else "calibration"
        )
        partitions[name].append(record["id"])
    return {k: np.asarray(v, dtype=int) for k, v in partitions.items()}


def motion_features(windows: np.ndarray) -> np.ndarray:
    """输入 [批次,400,3]：指令位置 mm、跟随误差 mm、电流 A，25 Hz。"""
    values = np.asarray(windows, dtype="float32")
    if (
        values.ndim != 3
        or values.shape[1:] != (400, 3)
        or not np.isfinite(values).all()
    ):
        raise ValueError("执行器输入需要有限的 [batch,400,3] 数组")
    velocity = np.diff(values[:, :, 0], axis=1, prepend=values[:, :1, 0]) * 25
    error, current = values[:, :, 1], np.abs(values[:, :, 2])
    masks = (
        np.ones_like(velocity, dtype=bool),
        velocity > 0.05,
        velocity < -0.05,
        np.abs(velocity) <= 0.05,
    )
    features = []
    for mask in masks:
        count = mask.sum(1)
        if np.any(count == 0):
            raise ValueError("完整动作周期须包含正向、反向和停留阶段")
        features.extend(
            [
                np.sqrt((error**2 * mask).sum(1) / count),
                (error * mask).sum(1) / count,
                np.sqrt((current**2 * mask).sum(1) / count),
                np.array(
                    [np.quantile(row[flag], 0.95) for row, flag in zip(current, mask)]
                ),
            ]
        )
    features.extend(
        [
            np.abs(error).max(1),
            np.quantile(np.abs(error), 0.95, axis=1),
            current.max(1),
            np.sqrt(np.mean(np.diff(current, axis=1) ** 2, axis=1)),
        ]
    )
    return np.stack(features, axis=1).astype("float32")


def aggregate(scores, record_ids, selected_records):
    return np.stack(
        [scores[record_ids == record].mean(axis=0) for record in selected_records]
    )


def fit_phase_references(windows, record_ids, metadata, reference_records):
    """仅接收明确指定的健康投运记录；每个工况独立拟合相位模板。"""
    by_id = {m["id"]: m for m in metadata}
    if any(i not in by_id or by_id[i]["label"] != 0 for i in reference_records):
        raise ValueError("投运参考只能包含已标注健康的原始记录")
    references = {}
    for context in sorted({(m["profile"], m["load"]) for m in metadata}):
        records = [
            m["id"]
            for m in metadata
            if (m["profile"], m["load"]) == context and m["id"] in reference_records
        ]
        if not records:
            raise ValueError(f"缺少工况健康投运参考: {context}")
        values = windows[np.isin(record_ids, records)]
        center = np.median(values[:, :, 1:], axis=0)
        # 以下是特征尺度下限，不是传感器精度。参考波形不含故障样本。
        scale = np.maximum(
            1.4826 * np.median(np.abs(values[:, :, 1:] - center), axis=(0, 1)),
            [0.01, 0.005],
        )
        references["/".join(context)] = {
            "center": center.tolist(),
            "scale": scale.tolist(),
            "records": records,
        }
    return references


def align_motion(windows, contexts, references):
    """指令位置保留 mm；误差/电流变为相位模板的绝对标准化偏差。"""
    result = np.asarray(windows, dtype="float32").copy()
    for i, context in enumerate(contexts):
        reference = references[context]
        result[i, :, 1:] = np.abs(
            result[i, :, 1:] - np.asarray(reference["center"])
        ) / np.asarray(reference["scale"])
    return result
