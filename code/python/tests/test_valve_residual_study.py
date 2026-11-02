"""冻结AE后处理与可移植机理图的边界测试。"""

import importlib.util
from pathlib import Path

import numpy as np
import torch

from phm_core.valve import mechanism_features


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / f"{name}.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_local_residual_is_not_diluted_and_scale_uses_training_only():
    study = module("study_valve_residual")
    residual = np.zeros((2, 400, 2), dtype="float32")
    residual[1, 40:60] = 1
    scores, scale = study.residual_scores(residual, [0])
    assert scores["mean"][1] == np.float32(0.05)
    assert scores["top10"][1] == 0.5
    assert scores["local200ms"][1] == 1
    residual[1] *= 100
    _, changed = study.residual_scores(residual, [0])
    np.testing.assert_array_equal(scale, changed)


def test_conjunction_requires_both_strict_thresholds():
    study = module("study_valve_fusion")
    score = study.conjunction_score(np.array([2, 3, 3]), np.array([5, 4, 5]), 2, 4)
    np.testing.assert_array_equal(score > 1, [False, False, True])


def test_torch_timing_matches_numpy_time_subtraction_precision():
    exporter = module("export_valve_fusion")
    graph = exporter.FusionGraph(
        torch.nn.Identity(),
        {"mean": [0, 0], "scale": [1, 1]},
        {"center": [0] * 9, "scale": [1] * 9},
        1,
        1,
    )
    x = np.empty((1, 400, 2), dtype="float32")
    x[0, :, 0] = np.linspace(200, 100, 400)
    x[0, :, 1] = np.linspace(0, 100, 400)
    np.testing.assert_array_equal(
        graph.timing_features(torch.from_numpy(x)).numpy(), mechanism_features(x)
    )
