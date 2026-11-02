import math

from phm_core.support import valve_support


def codes(result):
    return [r["code"] for r in result["reasons"]]


def test_supported_when_aligned_within_one_sample():
    result = valve_support(100, 100, command_edge_s=10.00, window_start_s=10.01)
    assert result["status"] == "supported" and result["reasons"] == []
    assert math.isclose(result["offset_samples"], 1.0, abs_tol=1e-9)


def test_exactly_one_sample_is_supported_despite_float_error():
    # (1000.57 - 1000.56) * 100 在二进制浮点下是 1.0000000000104592。
    result = valve_support(100, 100, 1000.56, 1000.57)
    assert result["status"] == "supported"
    assert valve_support(100, 100, 1000.56, 1000.5801)["status"] == "unsupported"


def test_no_command_edge_is_unsupported_never_self_aligned():
    result = valve_support(100, 100, command_edge_s=None, window_start_s=10.0)
    assert result["status"] == "unsupported"
    assert codes(result) == ["no_independent_command_edge"]


def test_offset_beyond_tolerance_rejected_in_both_directions():
    for start in (10.03, 9.97):
        result = valve_support(100, 100, 10.0, start)
        assert result["status"] == "unsupported" and codes(result) == ["sync_offset"]


def test_expected_lead_is_removed_before_comparison():
    result = valve_support(100, 100, 10.0, 10.5, expected_lead_s=0.5)
    assert result["status"] == "supported"


def test_channel_skew_and_sample_rate_reported_together():
    result = valve_support(50, 100, 10.0, 10.0, channel_skew_s=0.05)
    assert set(codes(result)) == {"sample_rate", "channel_skew"}


def test_missing_or_non_numeric_window_start_is_unsupported_not_an_exception():
    assert valve_support(100, 100, 10.0, None)["status"] == "unsupported"
    assert valve_support(100, 100, 10.0, "10.0")["status"] == "unsupported"
    assert valve_support(100, 100, True, 10.0)["status"] == "unsupported"


def test_nonfinite_input_is_unsupported_not_an_exception():
    for bad in (float("nan"), float("inf")):
        assert valve_support(100, 100, bad, 10.0)["status"] == "unsupported"
        assert valve_support(bad, 100, 10.0, 10.0)["status"] == "unsupported"
    assert valve_support(100, 100, 10.0, float("nan"))["status"] == "unsupported"
