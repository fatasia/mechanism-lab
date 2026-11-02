import pytest

from phm_core.event_metrics import match_warnings, merge_episodes


def test_merge_episodes_by_gap():
    assert merge_episodes([1, 2, 3, 10, 11, 30], 2) == [(1, 3), (10, 11), (30, 30)]
    assert merge_episodes([], 2) == []


def test_early_warning_counts_lead_and_denominator_is_event_list():
    result = match_warnings(
        warning_times=[90],
        event_starts=[100, 500],
        exposure_hours=1000,
        horizon_hours=72,
        min_lead_hours=1,
        merge_gap_hours=6,
    )
    assert result["events"] == 2 and result["detected"] == 1
    assert result["lead_hours"] == [10]
    assert result["recall"] == 0.5
    assert result["false_episodes"] == 0


def test_warning_too_late_or_too_early_is_not_a_hit():
    late = match_warnings([99.5], [100], 100, 72, 1, 6)
    assert late["detected"] == 0 and late["ambiguous_episodes"] == 1
    assert late["false_episodes"] == 0
    early = match_warnings([10], [100], 100, 72, 1, 6)
    assert early["detected"] == 0 and early["false_episodes"] == 1


def test_false_episodes_are_normalised_by_exposure_hours():
    result = match_warnings([5, 400, 800], [], 2000, 72, 1, 6)
    assert result["false_episodes"] == 3
    assert result["false_episodes_per_1000h"] == pytest.approx(1.5)
    assert result["recall"] is None and result["recall_interval_95"] == [0.0, 1.0]


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        match_warnings([1], [2], 10, 5, 5, 1)
    with pytest.raises(ValueError):
        match_warnings([1], [2], 0, 72, 1, 1)


@pytest.mark.parametrize(
    "lead,hit,ambiguous",
    [(0, 0, 1), (1.999, 0, 1), (2, 1, 0), (8, 1, 0), (8.001, 0, 0)],
)
def test_inclusive_boundaries_have_disjoint_episode_categories(lead, hit, ambiguous):
    result = match_warnings([10], [10 + lead], 100, 8, 2, 0)
    assert result["detected"] == hit
    assert result["ambiguous_episodes"] == ambiguous
    assert (
        result["matched_episodes"]
        + result["ambiguous_episodes"]
        + result["false_episodes"]
        == 1
    )


def test_zero_lead_and_invalid_inputs():
    assert match_warnings([10], [10], 100, 8, 0, 0)["matched_episodes"] == 1
    for kwargs in [
        dict(warning_times=[float("nan")]),
        dict(event_starts=[10, 10]),
        dict(merge_gap_hours=-1),
        dict(exposure_hours=float("inf")),
    ]:
        args = dict(
            warning_times=[1],
            event_starts=[10],
            exposure_hours=100,
            horizon_hours=8,
            min_lead_hours=2,
            merge_gap_hours=0,
        )
        args.update(kwargs)
        with pytest.raises(ValueError):
            match_warnings(**args)
