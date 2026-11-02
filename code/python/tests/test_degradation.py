import numpy as np
import pytest

from phm_core.degradation import (
    cycle_features,
    health_index,
    persistent_crossing,
    reference_threshold,
    reference_z,
    rolling_median,
    select_consistent,
)


def synthetic(n=1200, onset=800, seed=0, sign=1.0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (n, 3))
    trend = np.zeros(n)
    if onset < n:
        trend[onset:] = np.linspace(0, 8, n - onset)
    base[:, 0] += sign * trend  # 退化特征
    base[:, 1] += rng.normal(0, 0.1, n)  # 纯噪声
    base[:, 2] = 5.0  # 常数
    return base


def test_cycle_features_shape_and_validation():
    x = np.random.default_rng(1).normal(size=(2000, 30))
    assert cycle_features(x).shape == (30, 4)
    with pytest.raises(ValueError):
        cycle_features(np.ones(10))


def test_reference_z_zeroes_constant_features_instead_of_amplifying():
    z, usable, count = reference_z(synthetic())
    assert not usable[2] and np.all(z[:, 2] == 0)
    assert count == 120


def test_rolling_median_rejects_even_windows_and_removes_spikes():
    values = np.zeros(21)
    values[10] = 100
    assert rolling_median(values, 5)[10] == 0
    with pytest.raises(ValueError):
        rolling_median(values, 4)


def test_selection_keeps_consistent_trend_and_drops_noise_and_disagreeing_axes():
    zs = [reference_z(synthetic(seed=s))[0] for s in (1, 2)]
    indices, signs = select_consistent(zs)
    assert indices.tolist() == [0] and signs.tolist() == [1.0]
    opposite = [zs[0], reference_z(synthetic(seed=3, sign=-1.0))[0]]
    assert len(select_consistent(opposite)[0]) == 0
    with pytest.raises(ValueError):
        select_consistent(zs[:1])


def test_persistent_crossing_locates_onset_not_the_reference_period():
    train = [reference_z(synthetic(seed=s))[0] for s in (1, 2)]
    indices, signs = select_consistent(train)
    z, _, count = reference_z(synthetic(seed=7))
    index = health_index(z, indices, signs)
    threshold = reference_threshold(index, count)
    start, fraction = persistent_crossing(index, threshold, count)
    assert start is not None and 780 <= start <= 1000
    assert 0.6 < fraction < 0.85


def test_no_crossing_for_healthy_run():
    healthy = synthetic(onset=10_000)
    z, _, count = reference_z(healthy)
    index = health_index(z, np.array([0, 1]), np.array([1.0, 1.0]))
    threshold = reference_threshold(index, count)
    assert persistent_crossing(index, threshold, count) == (None, None)
