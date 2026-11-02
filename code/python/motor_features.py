"""同步电机窗口的统计、相序和角域包络特征；独立于API及业务状态。"""

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

FS = 12800
WINDOW = FS * 2
ORDERS = (3.585, 5.415, 2.357)
CHANNELS = (
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "current_a",
    "current_b",
    "current_c",
)
STAT_NAMES = (
    "mean",
    "std",
    "rms",
    "peak_to_peak",
    "kurtosis",
    "crest",
    "skew",
    "spectral_centroid",
    "spectral_spread",
    "spectral_entropy",
    "power_0_50",
    "power_50_200",
    "power_200_1000",
    "power_1000_6400",
)
FEATURE_NAMES = [f"{channel}/{stat}" for channel in CHANNELS for stat in STAT_NAMES]
FEATURE_NAMES += [
    "current/rms_imbalance",
    "current/sequence_imbalance",
    "current/zero_sequence",
]
FEATURE_NAMES += [
    f"vibration_{axis}/envelope_order_{order}_h{harmonic}"
    for axis in ("x", "y", "z")
    for order in ORDERS
    for harmonic in (1, 2, 3)
]


def phase_reference(keyphase):
    """2V上升沿、暂按每转1脉冲；停止/脉冲不足单列为不支持。"""
    edges = np.flatnonzero((keyphase[:-1] < 2) & (keyphase[1:] >= 2)) + 1
    if len(edges) < 8:
        return None, {"status": "insufficient_pulses", "pulses": len(edges)}
    intervals = np.diff(edges) / FS
    rpm = 60 / np.median(intervals)
    if rpm < 300 or np.min(intervals) < 0.01:
        return None, {
            "status": "unsupported_speed",
            "pulses": len(edges),
            "rpm": float(rpm),
        }
    return edges, {
        "status": "active",
        "pulses": len(edges),
        "rpm": float(rpm),
        "period_cv": float(intervals.std() / intervals.mean()),
    }


def current_balance(fft, frequencies, rms):
    """相序方向无关的不平衡指标；相量取相同主频，不能直接据此确认电气根因。"""
    band = np.flatnonzero((frequencies >= 5) & (frequencies <= 200))
    peak = band[np.argmax(np.sum(np.abs(fft[band]) ** 2, axis=1))]
    a, b, c = fft[peak]
    rotate = np.exp(2j * np.pi / 3)
    first = abs((a + rotate * b + rotate**2 * c) / 3)
    second = abs((a + rotate**2 * b + rotate * c) / 3)
    denominator = max(first, second, 1e-12)
    return [
        float(rms.std() / max(rms.mean(), 1e-12)),
        float(min(first, second) / denominator),
        float(abs((a + b + c) / 3) / denominator),
    ]


def envelope_orders(vibration, edges):
    """先固定500–5000Hz带通/Hilbert，再按键相等角度采样；不以标称RPM替代变速。"""
    sos = butter(4, [500, 5000], btype="bandpass", fs=FS, output="sos")
    envelope = np.abs(hilbert(sosfiltfilt(sos, vibration, axis=0), axis=0))
    revolutions = len(edges) - 1
    angles = np.arange(revolutions * 256) / 256
    positions = np.interp(angles, np.arange(len(edges)), edges)
    result = []
    for channel in range(3):
        angular = np.interp(positions, np.arange(len(vibration)), envelope[:, channel])
        angular -= angular.mean()
        energy = np.abs(np.fft.rfft(angular * np.hanning(len(angular)))) ** 2
        orders = np.fft.rfftfreq(len(angular), d=1 / 256)
        total = max(float(energy[(orders >= 0.5) & (orders <= 30)].sum()), 1e-20)
        for order in ORDERS:
            for harmonic in (1, 2, 3):
                mask = np.abs(orders - order * harmonic) <= 0.15
                result.append(float(energy[mask].sum() / total))
    return result


def window_features(values, edges):
    """输入2秒×9列原始CSV值；输出114维特征和6×1600的全带宽对数功率谱。"""
    if values.shape != (WINDOW, 9) or not np.isfinite(values).all():
        raise ValueError("电机窗口需要有限的25600×9输入")
    signals = values[:, 3:9] * 10  # 论文灵敏度0.1 V/g、0.1 V/A。
    centered = signals - signals.mean(axis=0)
    taper = np.hanning(WINDOW)
    fft = np.fft.rfft(centered * taper[:, None], axis=0)
    frequencies = np.fft.rfftfreq(WINDOW, d=1 / FS)
    power = abs(fft) ** 2 / (FS * np.sum(taper**2))
    power[1:-1] *= 2
    std = centered.std(axis=0)
    rms = np.sqrt(np.mean(signals**2, axis=0))
    result = []
    for channel in range(6):
        scale = max(float(std[channel]), 1e-12)
        standardized = centered[:, channel] / scale
        weights = power[:, channel] / max(float(power[:, channel].sum()), 1e-20)
        centroid = float(weights @ frequencies)
        spread = float(np.sqrt(weights @ ((frequencies - centroid) ** 2)))
        result.extend(
            [
                float(signals[:, channel].mean()),
                float(std[channel]),
                float(rms[channel]),
                float(np.ptp(signals[:, channel])),
                float(np.mean(standardized**4)),
                float(np.max(np.abs(centered[:, channel])) / max(rms[channel], 1e-12)),
                float(np.mean(standardized**3)),
                centroid,
                spread,
                float(
                    -np.sum(weights * np.log(np.maximum(weights, 1e-20)))
                    / np.log(len(weights))
                ),
            ]
        )
        for low, high in ((0, 50), (50, 200), (200, 1000), (1000, 6400.1)):
            result.append(
                float(weights[(frequencies >= low) & (frequencies < high)].sum())
            )
    result.extend(current_balance(fft[:, 3:6], frequencies, rms[3:6]))
    result.extend(envelope_orders(signals[:, :3], edges))
    # 固定4Hz频带平均后取log10；保留0–6400Hz，末端单个Nyquist点不进入CNN。
    spectrum = np.log10(
        np.maximum(power[:-1].reshape(1600, 8, 6).mean(axis=1).T, 1e-12)
    )
    return np.asarray(result, dtype="float32"), spectrum.astype("float32")
