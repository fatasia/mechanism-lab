import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

DIRECTORY = Path(__file__).resolve().parents[1]
for name in ("study_motor", "study_motor_cnn"):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
study = loaded


def test_training_seed_and_normalization_are_reproducible():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        x = np.random.default_rng(2).normal(size=(16, 6, 1600)).astype("float32")
        records = np.repeat(np.arange(8), 2)
        labels = records % 4
        first, history, epoch = study.train(x, labels, records, 4, 42, epochs=2)
        second, _, _ = study.train(x, labels, records, 4, 42, epochs=2)
        assert len(history) == epoch == 2
        for key, tensor in first.state_dict().items():
            torch.testing.assert_close(tensor, second.state_dict()[key], atol=0, rtol=0)
        np.testing.assert_allclose(
            first.mean.numpy().reshape(6), x.mean((0, 2)), atol=1e-7
        )
        probabilities = study.predict(first, x)
        assert probabilities.shape == (16, 4)
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
        assert study.predict(first, x[:0]).shape == (0, 4)
    finally:
        torch.set_num_threads(threads)
