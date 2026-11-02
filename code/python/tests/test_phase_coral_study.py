"""源域矩匹配与目标数据隔离的训练契约。"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

DIRECTORY = Path(__file__).resolve().parents[1]
for name in ("study_phase_alignment", "study_phase_coral"):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
study = loaded


def test_alignment_matches_same_class_domains_only():
    z = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    labels = torch.zeros(4, dtype=torch.long)
    domains = torch.tensor([0, 0, 1, 1])
    assert study.alignment_loss(z, labels, domains).item() == 0
    shifted = z.clone()
    shifted[2:] += 2
    assert study.alignment_loss(shifted, labels, domains).item() > 0
    assert study.alignment_loss(shifted, domains, domains).item() == 0


def test_target_values_do_not_change_training_weights():
    torch.set_num_threads(2)
    x = np.random.default_rng(42).normal(size=(48, 192)).astype("float32")
    labels = np.tile(np.arange(4), 12)
    domains = np.repeat(np.arange(3), 16)
    train, validation, test = np.arange(32), np.arange(32, 40), np.arange(40, 48)
    first, history, weights = study.fit(
        x, labels, domains, train, validation, test, 42, 0.1, epochs=2
    )
    changed = x.copy()
    changed[test] *= 100
    _, _, other = study.fit(
        changed, labels, domains, train, validation, test, 42, 0.1, epochs=2
    )
    assert len(history) == 2
    np.testing.assert_allclose(first.sum(1), 1, atol=1e-6)
    for key in weights:
        torch.testing.assert_close(weights[key], other[key], atol=0, rtol=0)
