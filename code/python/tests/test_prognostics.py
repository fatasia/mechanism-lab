import numpy as np
import pytest
import torch

from phm_core.prognostics import life_features, regression_metrics, residual_interval


def test_high_magnitude_constant_channels_have_zero_deviation():
    values = np.ones((2, 30, 24), dtype="float32") * 520.17
    features = life_features(values)
    np.testing.assert_allclose(features[:, 48:], 0, atol=1e-12)
    assert features.dtype == np.float32


def test_life_features_match_double_precision_deployment_aggregation():
    # 同一个输入用于 NumPy 训练和 Torch→ONNX 前端，防止树分支被消减误差改变。
    values = np.random.default_rng(42).normal(1000, 0.01, (3, 30, 24)).astype("float32")
    x = torch.from_numpy(values).double()
    t = torch.arange(30, dtype=torch.float64) - 14.5
    reference = (
        torch.cat(
            [
                x[:, -1],
                x.mean(1),
                x.std(1, unbiased=False),
                (x * t[None, :, None]).sum(1) / t.square().sum(),
            ],
            1,
        )
        .float()
        .numpy()
    )
    np.testing.assert_allclose(life_features(values), reference, atol=1e-8)


def test_rul_reports_cycles_and_needs_enough_independent_calibration_units():
    assert (
        regression_metrics(np.array([10, 20]), np.array([12, 18]))["rmse_cycles"] == 2
    )
    assert residual_interval(np.arange(15), np.zeros(15)) == 14
    with pytest.raises(ValueError):
        residual_interval(np.arange(5), np.zeros(5))
