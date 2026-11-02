import struct

import numpy as np
import pytest

from phm_core.current_signature import (
    FEATURES,
    ORDER_FEATURES,
    ORDERS,
    VIBRATION_BANDS,
    VIBRATION_FEATURES,
    GlobalBaseline,
    SpeedConditionedBaseline,
    fixed_segments,
    order_features,
    quantization_step,
    read_rigol,
    read_wav,
    segment_features,
    to_segments,
)


def write_rigol(path, values, interval=1e-6, width=4, waveforms=1):
    header = b"RG01" + struct.pack("<ii", 0, waveforms)
    wave = bytearray(140)
    struct.pack_into("<iiii", wave, 0, 140, 0, 1, len(values))
    struct.pack_into("<d", wave, 32, interval)
    data = np.asarray(values, dtype="<f4").tobytes()
    buffer = struct.pack("<ihhi", 12, 0, width, len(data))
    path.write_bytes(header + bytes(wave) + buffer + data)


def test_read_rigol_roundtrip_and_validation(tmp_path):
    values = np.random.default_rng(0).normal(size=1000).astype("float32")
    path = tmp_path / "a.bin"
    write_rigol(path, values)
    read, interval = read_rigol(path)
    np.testing.assert_array_equal(read, values)
    assert interval == pytest.approx(1e-6)
    path.write_bytes(path.read_bytes()[:-40])
    with pytest.raises(ValueError):
        read_rigol(path)
    bad = tmp_path / "b.bin"
    bad.write_bytes(b"XXXX" + b"\0" * 200)
    with pytest.raises(ValueError):
        read_rigol(bad)
    multi = tmp_path / "c.bin"
    write_rigol(multi, values, waveforms=2)
    with pytest.raises(ValueError):
        read_rigol(multi)


def test_quantization_step_reflects_vertical_scale():
    coarse = np.round(np.random.default_rng(1).normal(size=5000) / 0.04) * 0.04
    fine = np.round(np.random.default_rng(1).normal(size=5000) / 0.01) * 0.01
    assert quantization_step(coarse) == pytest.approx(0.04, rel=0.05)
    assert quantization_step(fine) == pytest.approx(0.01, rel=0.05)
    assert quantization_step(np.ones(10)) == 0.0


def test_segments_drop_partial_tail_and_reject_bad_rate():
    x = np.zeros(1_000_000 * 3 + 500_000)
    segments, rate = to_segments(x, 1e-6)
    assert segments.shape == (3, 20_000) and rate == 20_000
    with pytest.raises(ValueError):
        to_segments(np.zeros(1000), 3.3e-6)
    with pytest.raises(ValueError):
        to_segments(np.zeros(1000), 1e-6)


def test_wav_reader_segments_and_clipping(tmp_path):
    from scipy.io import wavfile

    rate = 8000
    t = np.arange(rate * 3) / rate
    tone = (0.5 * np.sin(2 * np.pi * 50 * t) * 32767).astype("int16")
    tone[:10] = 32767
    path = tmp_path / "a.wav"
    wavfile.write(path, rate, np.column_stack([tone, tone]))
    data, got_rate, clipped = read_wav(path)
    assert data.shape == (2, rate * 3) and got_rate == rate
    assert clipped == pytest.approx(20 / (2 * rate * 3))
    segments = fixed_segments(data[0], got_rate)
    assert segments.shape == (3, rate)
    with pytest.raises(ValueError):
        fixed_segments(data[0][:100], got_rate)
    wavfile.write(tmp_path / "f.wav", rate, tone.astype("float32"))
    with pytest.raises(ValueError):
        read_wav(tmp_path / "f.wav")


def test_vibration_bands_put_a_low_frequency_tone_in_the_right_band():
    rate = 44_100.0
    t = np.arange(44_100) / rate
    x = np.stack([0.3 * np.sin(2 * np.pi * 20 * t)])
    features = segment_features(x, rate, VIBRATION_BANDS)
    names = list(VIBRATION_FEATURES)
    assert features.shape[1] == len(VIBRATION_FEATURES)
    low = features[0, names.index("band_10_30Hz")]
    assert low > features[0, names.index("band_250_500Hz")] + 6


def test_order_features_find_harmonic_and_reject_low_speed():
    rate = 44_100.0
    t = np.arange(44_100) / rate
    f_rot = 16.0
    x = np.stack(
        [np.sin(2 * np.pi * f_rot * t) + 0.5 * np.sin(2 * np.pi * 2 * f_rot * t)]
    )
    f = order_features(x, rate, f_rot)
    names = list(ORDER_FEATURES)
    assert f.shape == (1, len(ORDERS))
    one, two = f[0, names.index("order_1x")], f[0, names.index("order_2x")]
    other = f[0, names.index("order_5x")]
    assert one > two > other + 5
    with pytest.raises(ValueError):
        order_features(x, rate, 2.0)
    with pytest.raises(ValueError):
        order_features(np.full((1, 100), np.nan), rate, 16.0)


def test_features_locate_300hz_line_in_its_band():
    rate = 20_000.0
    t = np.arange(20_000) / rate
    x = np.stack([1.0 + 0.5 * np.sin(2 * np.pi * 300 * t)] * 2)
    f = segment_features(x, rate)
    names = list(FEATURES)
    line = f[0, names.index("band_280_320Hz")]
    neighbour = f[0, names.index("band_320_580Hz")]
    assert line > neighbour + 6
    assert f[0, 0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        segment_features(np.full((1, 100), np.nan), rate)


def synthetic_normal(speeds, seed=0, segments=10):
    rng = np.random.default_rng(seed)
    out = []
    for s in speeds:
        base = np.array([np.log(s), 2 * np.log(s), 0.5])
        out.append(base + rng.normal(0, 0.05, (segments, 3)))
    return np.stack(out)


def test_speed_conditioned_baseline_separates_fault_that_global_baseline_hides():
    speeds = np.array([5, 10, 20, 40, 60, 80, 100], dtype=float)
    normal = synthetic_normal(speeds)
    conditioned = SpeedConditionedBaseline(speeds, normal)
    global_baseline = GlobalBaseline(normal)
    held = synthetic_normal([30.0], seed=5)[0]
    # 故障落在随转速大幅变化的特征上：相对条件基线是 8 倍噪声，相对全局尺度很小。
    fault = held + np.array([0.0, -0.4, 0.0])
    conditioned_ratio = (
        conditioned.score(fault, 30.0).mean() / conditioned.score(held, 30.0).mean()
    )
    global_ratio = (
        global_baseline.score(fault).mean() / global_baseline.score(held).mean()
    )
    assert conditioned_ratio > 3
    assert global_ratio < 1.5


def test_inner_scores_cover_every_training_speed_and_validation():
    speeds = np.array([5, 10, 20, 40], dtype=float)
    baseline = SpeedConditionedBaseline(speeds, synthetic_normal(speeds))
    assert baseline.inner_scores(synthetic_normal(speeds)).shape == (40,)
    with pytest.raises(ValueError):
        SpeedConditionedBaseline(speeds[:2], synthetic_normal(speeds[:2]))
    with pytest.raises(ValueError):
        SpeedConditionedBaseline(
            np.array([5, 5, 10, 20.0]), synthetic_normal([5, 5, 10, 20])
        )
