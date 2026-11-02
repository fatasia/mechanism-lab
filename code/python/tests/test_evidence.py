import math

import numpy as np
import pytest

from phm_core.evidence import (
    block_summary,
    blocks,
    clopper_pearson,
    events_needed_zero_miss,
    hours_needed_zero_alarm,
    one_sided_lower,
    one_sided_upper,
    variance_components,
)


def test_zero_events_upper_bound_matches_rule_of_three():
    assert one_sided_upper(0, 59, 0.95) == pytest.approx(1 - 0.05 ** (1 / 59))
    assert one_sided_upper(0, 3, 0.95) > 0.5


def test_all_detected_lower_bound_needs_59_events_for_95_percent():
    assert one_sided_lower(59, 59, 0.95) >= 0.95
    assert one_sided_lower(58, 58, 0.95) < 0.95
    assert events_needed_zero_miss(0.95, 0.95) == 59


def test_three_thousand_hours_for_one_per_thousand():
    assert hours_needed_zero_alarm(1.0, 0.95) == math.ceil(-math.log(0.05) * 1000)


def test_clopper_pearson_edges_and_validation():
    assert clopper_pearson(0, 0) == (0.0, 1.0)
    low, high = clopper_pearson(8, 8)
    assert high == 1.0 and 0.6 < low < 0.7
    with pytest.raises(ValueError):
        clopper_pearson(3, 2)


def test_blocks_merge_consecutive_ids():
    assert blocks([5, 3, 4, 10, 11, 20]) == [(3, 5), (10, 11), (20, 20)]
    assert blocks([]) == []


def test_block_summary_counts_blocks_not_cycles():
    ids = list(range(24)) + list(range(30, 54)) + list(range(60, 84))
    result = block_summary(ids, error_ids=[31, 32, 33])
    assert result["blocks"] == 3 and result["samples"] == 72
    assert result["error_samples"] == 3 and result["error_blocks"] == 1
    assert result["error_block_interval"][1] < 1.0


def test_variance_components_separates_persistent_from_noise():
    rng = np.random.default_rng(0)
    persistent = rng.normal(0, 1, (30, 1)) + rng.normal(0, 0.05, (30, 40))
    noisy = rng.normal(0, 1, (30, 40))
    assert variance_components(persistent)["icc"] > 0.9
    assert variance_components(noisy)["icc"] < 0.2
    with pytest.raises(ValueError):
        variance_components(np.ones((1, 4)))


def test_variance_components_skips_records_with_too_few_draws():
    rng = np.random.default_rng(1)
    values = rng.normal(0, 1, (10, 20))
    values[0, 1:] = np.nan
    values[1, :] = np.nan
    result = variance_components(values)
    assert all(np.isfinite(v) for v in result.values())
