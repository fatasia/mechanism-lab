"""按寿命单元分组的窗口特征与区间校准；输出周期，不换算成天数。"""

import numpy as np

from phm_core.monitoring import calibrated_threshold


def life_features(values):
    # 高量级、近常量传感器在 float32 方差中存在消减误差；仅聚合使用 float64。
    values = np.asarray(values, dtype="float64")
    t = np.arange(values.shape[1], dtype="float64")
    t -= t.mean()
    slope = np.einsum("btc,t->bc", values, t) / (t @ t)
    return np.concatenate(
        [values[:, -1], values.mean(1), values.std(1), slope], axis=1
    ).astype("float32")


def regression_metrics(truth, prediction):
    error = np.asarray(prediction) - truth
    return {
        "rmse_cycles": float(np.sqrt(np.mean(error**2))),
        "mae_cycles": float(np.mean(np.abs(error))),
        "late_fraction": float(np.mean(error > 0)),
    }


def residual_interval(truth, prediction, alpha=0.1):
    """每校准寿命单元一个预先固定的观测点，避免重叠窗口充当独立样本。"""
    return calibrated_threshold(np.abs(np.asarray(truth) - prediction), alpha)
