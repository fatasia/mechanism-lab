"""旋转机械的独立频谱特征；不依赖演示数据或平台。"""

import numpy as np
from scipy.signal import hilbert, periodogram


def spectrum(
    values: np.ndarray, sample_rate: float, envelope: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    if (
        sample_rate <= 0
        or values.ndim != 1
        or len(values) < 4
        or not np.isfinite(values).all()
    ):
        raise ValueError("频谱需要有限一维信号与正采样率")
    centered = values - values.mean()
    signal = np.abs(hilbert(centered)) if envelope else centered
    return periodogram(signal, fs=sample_rate, window="hann", scaling="density")


def bearing_frequencies(
    speed_hz: float, balls: int, diameter_ratio: float, angle_rad: float = 0
) -> dict[str, float]:
    """理想无滑移运动学参考；几何参数必须来自实物规格。"""
    if speed_hz <= 0 or balls < 1 or not 0 < diameter_ratio < 1:
        raise ValueError("轴转频、滚动体数或直径比不合法")
    ratio = diameter_ratio * np.cos(angle_rad)
    return {
        "BPFO": balls * speed_hz / 2 * (1 - ratio),
        "BPFI": balls * speed_hz / 2 * (1 + ratio),
        "FTF": speed_hz / 2 * (1 - ratio),
        "BSF": speed_hz / (2 * diameter_ratio) * (1 - ratio**2),
    }


def compare_duration(duration: float, reference: float, tolerance: float) -> dict:
    """动作时间只能支持变慢证据，不能独自区分泄漏和负载变化。"""
    if reference <= 0 or duration < 0 or tolerance < 0:
        raise ValueError("时间与相对容差不合法")
    relative = duration / reference - 1
    return {
        "relative_change": relative,
        "abnormal": relative > tolerance,
        "evidence": "动作时间延长" if relative > tolerance else "动作时间在参考范围",
        "candidates": ["负载变化", "摩擦增加", "气源或泄漏"]
        if relative > tolerance
        else [],
    }
