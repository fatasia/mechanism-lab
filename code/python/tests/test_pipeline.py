import numpy as np
import torch
from fastapi.testclient import TestClient

from phm_core.data import block_split
from phm_core.deep import ConvAutoencoder
from phm_core.signal import bearing_frequencies, spectrum
from phm_core.tcn import TCN
from phm_demo.api import app


def test_block_isolation():
    split = block_split(2205)
    groups = [set(ids // 30) for ids in split.values()]
    for i, a in enumerate(groups):
        for b in groups[i + 1 :]:
            assert a.isdisjoint(b)
    for ids in split.values():
        assert np.all((ids % 30 >= 3) & (ids % 30 < 27))
    assert all(
        np.array_equal(ids, block_split(2205)[name]) for name, ids in split.items()
    )


def test_ae_shapes():
    model = ConvAutoencoder(5).eval()
    for length in [59, 60]:
        x = torch.randn(2, length, 5)
        assert model(x).shape == x.shape


def test_api_contract():
    client = TestClient(app)
    assert client.get("/api/summary").status_code == 200
    assert client.get("/api/cycle/unknown/0").status_code == 422
    assert client.get("/api/cycle/valve/0?channel=9").status_code == 422
    assert client.get("/api/job").json()["status"] in [
        "idle",
        "running",
        "complete",
        "failed",
    ]


def test_tcn_no_future_leakage():
    model = TCN(2).eval()
    x = torch.randn(1, 40, 2)
    changed = x.clone()
    changed[:, 20:] += 100
    torch.testing.assert_close(model(x)[:, :20], model(changed)[:, :20])


def test_signal_frequency():
    t = np.arange(1000) / 1000
    f, p = spectrum(np.sin(2 * np.pi * 80 * t), 1000)
    assert f[np.argmax(p)] == 80
    assert bearing_frequencies(10, 8, 0.2)["BPFO"] == 32
