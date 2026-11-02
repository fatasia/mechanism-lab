"""PS1 下降、PS2 上升的换向响应；不是阀芯位移或控制指令的替代测量。"""

import numpy as np

TIMING_NAMES = (
    "PS1_t10_s",
    "PS1_t50_s",
    "PS1_t90_s",
    "PS2_t10_s",
    "PS2_t50_s",
    "PS2_t90_s",
    "PS1_transition_s",
    "PS2_transition_s",
    "handover_gap_s",
)


def mechanism_features(
    windows: np.ndarray, rate: int = 100, start: float = 8
) -> np.ndarray:
    """固定 8～12 s 窗口；前后各 1 s 均值定义压力变化进程。

    交点为首次越过 10/50/90% 的采样时刻，精度限于采样间隔。
    无交点时记作窗口末尾，保留为未完成响应而非正常值。
    """
    if (
        windows.ndim != 3
        or windows.shape[1:] != (400, 2)
        or not np.isfinite(windows).all()
        or rate != 100
        or start != 8
    ):
        raise ValueError("换向特征需要 [batch,400,2] 的 PS1/PS2 100 Hz 压力窗口")
    before = windows[:, :100].mean(axis=1)
    after = windows[:, -100:].mean(axis=1)
    amplitude = (after - before) * np.array([-1, 1])
    # 1 bar 是算法有效幅值下限，不宣称为传感器精度；低于下限的波形仍得到迟滞/未完成证据。
    progress = (
        (windows - before[:, None, :])
        * np.array([-1, 1])
        / np.maximum(amplitude[:, None, :], 1)
    )
    times = []
    for channel in range(2):
        for fraction in [0.1, 0.5, 0.9]:
            crossing = progress[:, :, channel] >= fraction
            index = np.where(
                crossing.any(axis=1), crossing.argmax(axis=1), windows.shape[1]
            )
            times.append(start + index / rate)
    features = np.stack(
        times + [times[2] - times[0], times[5] - times[3], times[1] - times[4]], axis=1
    )
    return features.astype("float32")


def fit_mechanism_reference(features: np.ndarray, normal_ids: np.ndarray) -> dict:
    center = np.median(features[normal_ids], axis=0)
    scale = np.maximum(
        1.4826 * np.median(np.abs(features[normal_ids] - center), axis=0), 0.01
    )
    return {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "names": list(TIMING_NAMES),
        "units": ["s"] * len(TIMING_NAMES),
        "resolution_floor_seconds": 0.01,
    }


def mechanism_score(features: np.ndarray, reference: dict) -> np.ndarray:
    """换向到达时间延迟取单侧，其余时长/交接关系取双侧。"""
    deviation = (features - np.asarray(reference["center"])) / np.asarray(
        reference["scale"]
    )
    deviation[:, :6] = np.maximum(deviation[:, :6], 0)
    deviation[:, 6:] = np.abs(deviation[:, 6:])
    return deviation.max(axis=1)


def choose_waveform(validation: dict[str, list[dict]]) -> str:
    """只接收验证结果；平均正常误报须 <= 5%，再比较召回与 AP。"""
    eligible = {
        key: values
        for key, values in validation.items()
        if all(v["recall"] is not None and v["test_pr_auc"] is not None for v in values)
        and np.mean([v["false_positive_per_1000"] for v in values]) <= 50
    }
    if not eligible:
        raise ValueError("没有候选满足验证集误报预算")
    return max(
        eligible,
        key=lambda key: (
            np.mean([v["recall"] for v in eligible[key]]),
            np.mean([v["test_pr_auc"] for v in eligible[key]]),
        ),
    )


def pressure_features(windows: np.ndarray) -> np.ndarray:
    """时序加压力幅值/稳态波动；保留供压信息，不用归一化消除真实迟滞。"""
    timing = mechanism_features(windows)
    before = windows[:, :100].mean(axis=1)
    after = windows[:, -100:].mean(axis=1)
    return np.concatenate(
        [
            timing,
            before,
            after,
            after - before,
            windows[:, -100:].std(axis=1),
            (windows[:, :, 0] - windows[:, :, 1]).mean(axis=1, keepdims=True),
        ],
        axis=1,
    ).astype("float32")
