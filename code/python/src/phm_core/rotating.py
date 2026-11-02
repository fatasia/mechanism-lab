"""带通包络谱与轴承特征频率证据，输出候选而非确定失效根因。"""

import numpy as np
from scipy.signal import butter, hilbert, periodogram, sosfiltfilt

ORDERS_6205 = {
    "inner_ring": 5.4152,
    "outer_ring": 3.5848,
    "rolling_element": 4.7135,
    "cage": 0.39828,
}


def envelope_evidence(values, sample_rate, rpm, orders, band=(1200, 5000)):
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or len(values) < sample_rate
        or not np.isfinite(values).all()
        or rpm <= 0
        or not 0 < band[0] < band[1] < sample_rate / 2
    ):
        raise ValueError("包络分析需要至少 1 秒有限信号、实际转速及有效带通范围")
    centered = values - values.mean()
    filtered = sosfiltfilt(
        butter(4, band, btype="bandpass", fs=sample_rate, output="sos"), centered
    )
    envelope = np.abs(hilbert(filtered))
    f, density = periodogram(envelope - envelope.mean(), fs=sample_rate, window="hann")
    candidates = []
    for name, order in orders.items():
        target = order * rpm / 60
        peaks = []
        for harmonic in (1, 2, 3):
            frequency = target * harmonic
            bins = np.abs(f - frequency) <= max(2, frequency * 0.02)
            background = (np.abs(f - frequency) < max(15, frequency * 0.15)) & ~bins
            if not bins.any() or not background.any():
                continue
            index = np.flatnonzero(bins)[density[bins].argmax()]
            ratio = density[index] / max(float(np.median(density[background])), 1e-18)
            peaks.append(
                {
                    "harmonic": harmonic,
                    "expected_hz": frequency,
                    "observed_hz": float(f[index]),
                    "contrast": float(ratio),
                }
            )
        candidates.append(
            {
                "candidate": name,
                "order": order,
                "frequency_hz": target,
                "evidence_score": float(sum(np.log1p(p["contrast"]) for p in peaks)),
                "peaks": peaks,
            }
        )
    candidates.sort(key=lambda x: x["evidence_score"], reverse=True)
    shown = f <= 1000
    return {
        "rms": float(np.sqrt(np.mean(centered**2))),
        "crest_factor": float(
            np.abs(centered).max() / max(np.sqrt(np.mean(centered**2)), 1e-12)
        ),
        "band_hz": band,
        "rpm": rpm,
        "candidates": candidates,
        "frequency_hz": f[shown].tolist(),
        "envelope_psd": density[shown].tolist(),
    }
