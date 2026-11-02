"""电枢电流波形的读取、分段特征与随转速插值的正常基线。

电流特征强烈依赖转速设定值（受控输入，现场已知），因此正常行为按转速插值，
而不是用一个与转速无关的全局基线。
"""

import struct
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

# Hz，左闭右开；300 Hz 是 6 脉冲整流纹波，不是故障。
BANDS = (
    (0.0, 5.0),
    (5.0, 20.0),
    (20.0, 50.0),
    (50.0, 100.0),
    (100.0, 200.0),
    (200.0, 280.0),
    (280.0, 320.0),
    (320.0, 580.0),
    (580.0, 620.0),
    (620.0, 880.0),
    (880.0, 920.0),
    (920.0, 2000.0),
    (2000.0, 5000.0),
    (5000.0, 8000.0),
)
TIME_FEATURES = ("mean", "std", "peak_to_peak", "crest", "kurtosis")


def feature_names(bands=BANDS):
    return TIME_FEATURES + tuple(f"band_{int(a)}_{int(b)}Hz" for a, b in bands)


FEATURES = feature_names(BANDS)
# 振动：低频段覆盖转速基频及其谐波（最高转速约 33 Hz），其余按倍频程；松动先出现在
# 低频谐波和宽带。频带按物理意义预先给定，不是看结果选的。
VIBRATION_BANDS = (
    (0.0, 10.0),
    (10.0, 30.0),
    (30.0, 60.0),
    (60.0, 120.0),
    (120.0, 250.0),
    (250.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 2000.0),
    (2000.0, 4000.0),
    (4000.0, 8000.0),
    (8000.0, 16000.0),
)
VIBRATION_FEATURES = feature_names(VIBRATION_BANDS)
# 次谐波 0.5×、1×、1.5×、2×…6×：松动典型地抬高次谐波和多阶谐波。
ORDERS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
ORDER_FEATURES = tuple(f"order_{o:g}x" for o in ORDERS)


def read_rigol(path: Path):
    """读取单通道、单缓冲的 Rigol 原生 BIN，返回 (float32 电压, 采样间隔 s)。"""
    with path.open("rb") as stream:
        head = stream.read(12)
        if len(head) != 12 or head[:4] not in (b"RG01", b"AG01"):
            raise ValueError(f"不是 Rigol BIN: {path.name}")
        _, waveforms = struct.unpack_from("<ii", head, 4)
        if waveforms != 1:
            raise ValueError(f"只支持单波形文件: {path.name}")
        header = stream.read(140)
        size, _, buffers, points = struct.unpack_from("<iiii", header)
        interval = struct.unpack_from("<d", header, 32)[0]
        if buffers != 1 or points < 1 or interval <= 0 or size < 140:
            raise ValueError(f"波形头不合法: {path.name}")
        stream.seek(12 + size)
        _, _, width, length = struct.unpack("<ihhi", stream.read(12))
        if width != 4 or length != points * 4:
            raise ValueError(f"期望 float32 缓冲: {path.name}")
        values = np.fromfile(stream, dtype="<f4", count=points)
    if len(values) != points:
        raise ValueError(f"文件被截断: {path.name}")
    return values, float(interval)


def quantization_step(values, sample=2_000_000):
    """相邻不同取值的最小间隔，反映示波器垂直量程；量程随状态变化会造成假相关。"""
    unique = np.unique(np.asarray(values[:sample], dtype=np.float64))
    if len(unique) < 2:
        return 0.0
    return float(np.min(np.diff(unique)))


def to_segments(values, interval, target_rate=20_000.0, segment_seconds=1.0):
    """抗混叠降采样后切成整秒段，丢弃不足一段的尾部。"""
    rate = 1.0 / interval
    factor = round(rate / target_rate)
    if factor < 1 or abs(rate / factor - target_rate) > 0.01 * target_rate:
        raise ValueError(f"采样率 {rate:.0f} Hz 无法整数倍降到 {target_rate:.0f} Hz")
    down = resample_poly(np.asarray(values, dtype=np.float64), 1, factor)
    length = int(target_rate * segment_seconds)
    count = len(down) // length
    if count < 1:
        raise ValueError("信号短于一个分析段")
    return down[: count * length].reshape(count, length), target_rate


def read_wav(path: Path):
    """读取 16 位 WAV，返回 ([通道, 采样点] float64 归一化到 ±1, 采样率 Hz, 削波占比)。"""
    from scipy.io import wavfile

    rate, data = wavfile.read(path)
    if data.dtype != np.int16:
        raise ValueError(f"期望 16 位 PCM: {path.name}")
    data = data.reshape(len(data), -1).T
    clipped = float(np.mean((data == 32767) | (data == -32768)))
    return data.astype(np.float64) / 32768.0, float(rate), clipped


def fixed_segments(values, rate, segment_seconds=1.0):
    """不降采样，直接切成整秒段；丢弃不足一段的尾部。"""
    length = int(rate * segment_seconds)
    count = len(values) // length
    if count < 1:
        raise ValueError("信号短于一个分析段")
    return np.asarray(values[: count * length], dtype=np.float64).reshape(count, length)


def order_features(segments, rate, rotation_hz, orders=ORDERS, half_width_hz=1.0):
    """转速同步谐波/次谐波：基频 k 倍附近的峰值功率（log10），松动的经典指标。

    rotation_hz 由受控的转速设定值得出（现场已知），分辨率 1 Hz，所以只适用于
    基频不低于 3 Hz 的工况。
    """
    x = np.asarray(segments, dtype=np.float64)
    if rotation_hz < 3.0:
        raise ValueError("基频低于 3 Hz，1 Hz 分辨率无法分辨各阶")
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("需要有限的二维分段数组")
    centered = x - x.mean(axis=1, keepdims=True)
    window = np.hanning(x.shape[1])
    spectrum = np.abs(np.fft.rfft(centered * window, axis=1)) ** 2
    spectrum /= np.sum(window**2) * rate
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / rate)
    columns = []
    for order in orders:
        near = np.abs(freqs - order * rotation_hz) <= half_width_hz
        if not near.any() or order * rotation_hz + half_width_hz > rate / 2:
            raise ValueError(f"{order} 阶超出可分析范围")
        columns.append(np.log10(spectrum[:, near].max(axis=1) + 1e-18))
    return np.column_stack(columns)


def segment_features(segments, rate, band_edges=BANDS):
    """[段, 特征]：时域 5 项 + 14 个频带的 log10 功率。"""
    x = np.asarray(segments, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 64 or not np.isfinite(x).all():
        raise ValueError("需要有限的 [段, 采样点>=64] 数组")
    mean = x.mean(axis=1)
    centered = x - mean[:, None]
    std = centered.std(axis=1)
    safe = np.maximum(std, 1e-12)
    kurtosis = np.mean(centered**4, axis=1) / safe**4
    crest = np.abs(centered).max(axis=1) / safe
    window = np.hanning(x.shape[1])
    spectrum = np.abs(np.fft.rfft(centered * window, axis=1)) ** 2
    spectrum /= np.sum(window**2) * rate
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / rate)
    df = freqs[1] - freqs[0]
    bands = []
    for low, high in band_edges:
        selected = (freqs >= low) & (freqs < high)
        if not selected.any():
            raise ValueError(f"频带 {low}-{high} Hz 内没有频点")
        bands.append(np.log10(spectrum[:, selected].sum(axis=1) * df + 1e-18))
    return np.column_stack(
        [
            mean,
            std,
            centered.max(axis=1) - centered.min(axis=1),
            crest,
            kurtosis,
            *bands,
        ]
    )


class SpeedConditionedBaseline:
    """正常特征随转速的分段线性插值（对数转速轴）；越界处取最近端点。

    尺度取留一转速插值残差的稳健离散度，包含“在没见过的转速上插值”的误差。
    """

    def __init__(self, speeds, features):
        speeds = np.asarray(speeds, dtype=float)
        if features.ndim != 3 or features.shape[0] != len(speeds) or len(speeds) < 3:
            raise ValueError("需要至少 3 个转速、形状为 [转速, 段, 特征] 的正常特征")
        if len(set(speeds.tolist())) != len(speeds) or np.any(speeds <= 0):
            raise ValueError("转速须为互不相同的正数")
        order = np.argsort(speeds)
        self.speeds = speeds[order]
        self.means = features[order].mean(axis=1)
        self.scale = self._inner_scale(features[order])

    def _interpolate(self, speeds, means, speed):
        return np.array(
            [
                np.interp(np.log(speed), np.log(speeds), means[:, k])
                for k in range(means.shape[1])
            ]
        )

    def _inner_scale(self, features):
        residuals = []
        for held in range(len(self.speeds)):
            keep = np.arange(len(self.speeds)) != held
            predicted = self._interpolate(
                self.speeds[keep], features[keep].mean(axis=1), self.speeds[held]
            )
            residuals.append(features[held] - predicted)
        pooled = np.concatenate(residuals)
        center = np.median(pooled, axis=0)
        spread = 1.4826 * np.median(np.abs(pooled - center), axis=0)
        # 下限避免常数特征放大数值噪声，不是传感器精度。
        return np.maximum(spread, 1e-6 * (np.abs(self.means).mean(axis=0) + 1.0))

    def inner_scores(self, features):
        """训练转速的留一插值得分，用于冻结阈值。"""
        scores = []
        for held in range(len(self.speeds)):
            keep = np.arange(len(self.speeds)) != held
            predicted = self._interpolate(
                self.speeds[keep], features[keep].mean(axis=1), self.speeds[held]
            )
            scores.append(self._score(features[held], predicted))
        return np.concatenate(scores)

    def _score(self, features, predicted):
        z = (features - predicted) / self.scale
        return np.sqrt(np.mean(z**2, axis=1))

    def score(self, features, speed):
        predicted = self._interpolate(self.speeds, self.means, speed)
        return self._score(features, predicted)


class GlobalBaseline:
    """与转速无关的对照：训练正常段的全局稳健标准化。"""

    def __init__(self, features):
        pooled = features.reshape(-1, features.shape[-1])
        self.center = np.median(pooled, axis=0)
        spread = 1.4826 * np.median(np.abs(pooled - self.center), axis=0)
        self.scale = np.maximum(spread, 1e-6 * (np.abs(self.center) + 1.0))

    def score(self, features):
        z = (features - self.center) / self.scale
        return np.sqrt(np.mean(z**2, axis=1))
