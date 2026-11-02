"""退化趋势：周期特征、投运参考标准化、跨设备特征选择与持续越界位置。

没有故障时刻标签时只能评价趋势性与越界位置，不能计算剩余寿命误差。
"""

import numpy as np
from scipy.stats import spearmanr

FEATURES = ("std", "mean", "diff_rms", "p95_dev")


def cycle_features(x: np.ndarray) -> np.ndarray:
    """x 为 [采样点, 周期]；返回 [周期, 4]：标准差、均值、差分均方根、95% 偏离。"""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 4:
        raise ValueError("需要 [采样点>=4, 周期] 的二维数组")
    median = np.median(x, axis=0)
    return np.stack(
        [
            x.std(axis=0),
            x.mean(axis=0),
            np.sqrt(np.mean(np.diff(x, axis=0) ** 2, axis=0)),
            np.quantile(np.abs(x - median), 0.95, axis=0),
        ],
        axis=1,
    )


def reference_z(features: np.ndarray, reference_fraction: float = 0.1):
    """以最早的 reference_fraction 周期为投运参考做稳健标准化。"""
    if not 0 < reference_fraction < 1:
        raise ValueError("reference_fraction 须在 (0,1)")
    count = max(int(len(features) * reference_fraction), 20)
    if count >= len(features):
        raise ValueError("周期数不足以划分参考期")
    ref = features[:count]
    center = np.median(ref, axis=0)
    spread = 1.4826 * np.median(np.abs(ref - center), axis=0)
    # 常数特征没有可标准化的尺度，置零而不是放大数值噪声。
    floor = 1e-9 * (np.abs(center) + 1.0)
    usable = spread > floor
    z = np.zeros_like(features, dtype=float)
    z[:, usable] = (features[:, usable] - center[usable]) / spread[usable]
    return z, usable, count


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window < 1 or window % 2 == 0:
        raise ValueError("window 须为正奇数")
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    stacked = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(stacked, axis=1)


def select_consistent(sources, min_rho=0.5, smooth=51):
    """所有源设备上趋势方向一致且 |rho| 都不小于阈值的特征；返回 (索引, 符号)。"""
    if len(sources) < 2:
        raise ValueError("跨设备选择至少需要两台源设备")
    indices, signs = [], []
    for k in range(sources[0].shape[1]):
        rhos = []
        for z in sources:
            if not np.any(z[:, k]):
                rhos.append(0.0)
                continue
            t = np.arange(len(z))
            rho = spearmanr(t, rolling_median(z[:, k], smooth)).statistic
            rhos.append(0.0 if np.isnan(rho) else float(rho))
        if (
            all(abs(r) >= min_rho for r in rhos)
            and len({np.sign(r) for r in rhos}) == 1
        ):
            indices.append(k)
            signs.append(float(np.sign(rhos[0])))
    return np.array(indices, dtype=int), np.array(signs)


def health_index(z, indices, signs, smooth=101):
    if len(indices) == 0:
        return np.zeros(len(z))
    return rolling_median((z[:, indices] * signs).mean(axis=1), smooth)


def reference_threshold(index, reference_count, sigmas=6.0, floor=0.5):
    ref = index[:reference_count]
    center = np.median(ref)
    spread = max(1.4826 * np.median(np.abs(ref - center)), floor / sigmas)
    return float(center + sigmas * spread)


def persistent_crossing(index, threshold, reference_count, hold=100):
    """参考期之后首次连续 hold 个周期都高于阈值的位置，返回 (周期, 寿命占比)。"""
    above = index > threshold
    for start in range(reference_count, len(index) - hold + 1):
        if above[start : start + hold].all():
            return start, start / len(index)
    return None, None
