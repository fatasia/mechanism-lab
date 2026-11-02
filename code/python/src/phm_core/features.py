"""完整周期的时域特征；输入布局为周期、时间、通道。"""

import numpy as np
from numpy.typing import NDArray


def cycle_features(values: NDArray[np.floating]) -> NDArray[np.floating]:
    """返回每通道的均值、标准差、RMS、峰峰值，按特征组排列。"""
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("输入须为有限数值的 [cycle,time,channel] 张量")
    return np.concatenate(
        [
            values.mean(axis=1),
            values.std(axis=1),
            np.sqrt(np.mean(values**2, axis=1)),
            np.ptp(values, axis=1),
        ],
        axis=1,
    )


class RobustBaseline:
    """训练健康样本的固定尺度；常量特征不用于评分。"""

    def fit(self, features: NDArray[np.floating]) -> "RobustBaseline":
        self.center = np.median(features, axis=0)
        self.scale = 1.4826 * np.median(np.abs(features - self.center), axis=0)
        self.active = self.scale > 0
        if not self.active.any():
            raise ValueError("健康样本中没有可评分的变化特征")
        return self

    def score(self, features: NDArray[np.floating]) -> NDArray[np.floating]:
        deviation = np.abs(features[:, self.active] - self.center[self.active])
        return np.max(deviation / self.scale[self.active], axis=1)
