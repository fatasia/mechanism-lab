import numpy as np
import pytest

from phm_core.conditioning import fit_reference, transform_reference
from phm_core.valve import pressure_features


def test_reference_isolated_from_non_reference_records():
    x = np.array([[1.0, 2.0], [2.0, 4.0], [11.0, 12.0], [12.0, 14.0], [100.0, 100.0]])
    context = np.array(["a", "a", "b", "b", "b"])
    healthy = np.array([True, True, True, True, False])
    reference = fit_reference(x, context, healthy, np.arange(4))
    changed = x.copy()
    changed[-1] *= 100
    assert reference == fit_reference(changed, context, healthy, np.arange(4))
    aligned = transform_reference(x, context, reference)
    np.testing.assert_allclose(aligned[:2], aligned[2:4])
    assert aligned[-1].max() > 10


def test_missing_context_is_not_healthy_and_bad_reference_rejected():
    x = np.array([[1.0], [2.0], [3.0]])
    ctx = np.array(["a", "a", "b"])
    healthy = np.array([True, True, False])
    reference = fit_reference(x, ctx, healthy, [0, 1])
    with pytest.raises(ValueError, match="未投运"):
        transform_reference(x, ctx, reference)
    for ids in ([], [0, 0], [0, 2], [-1]):
        with pytest.raises(ValueError):
            fit_reference(x, ctx, healthy, ids)


def test_pressure_features_keep_amplitude_and_finite_shape():
    x = np.zeros((2, 400, 2), dtype="float32")
    x[:, :200, 0] = 190
    x[:, 200:, 0] = 140
    x[:, 200:, 1] = 120
    base = pressure_features(x)
    gain = pressure_features(x * 2)
    assert base.shape == (2, 18)
    assert np.isfinite(base).all()
    np.testing.assert_allclose(base[:, :9], gain[:, :9])
    np.testing.assert_allclose(base[:, 9:] * 2, gain[:, 9:])


def test_conditioning_benchmark_api_uses_versioned_artifact(monkeypatch, tmp_path):
    import json

    from fastapi.testclient import TestClient

    from phm_core.artifacts import new_run, publish_run
    from phm_demo import benchmarks_api
    from phm_demo.api import app

    run_id, directory = new_run(tmp_path / "outputs", "conditioning-study")
    (directory / "metrics.json").write_text(json.dumps({"run_id": run_id}))
    (directory / "model-bundle.zip").write_bytes(b"bundle")
    publish_run(tmp_path / "outputs", "conditioning-study", run_id)
    monkeypatch.setattr(benchmarks_api, "ROOT", tmp_path)
    client = TestClient(app)
    assert client.get("/api/benchmarks/conditioning-study").json()["run_id"] == run_id
    result = client.get(f"/api/benchmarks/conditioning-study/download?run_id={run_id}")
    assert result.status_code == 200 and result.content == b"bundle"
    assert (
        client.get(
            "/api/benchmarks/conditioning-study/download?run_id=../bad"
        ).status_code
        == 422
    )
