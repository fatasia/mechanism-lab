import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.ensemble import RandomForestClassifier

DIRECTORY = Path(__file__).resolve().parents[1]
for name in ("infer_motor_rf", "motor_features", "export_motor_rf"):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
study = loaded


def test_onnx_embeds_column_selection_from_full_feature_vector():
    x = np.random.default_rng(5).normal(size=(30, 114)).astype("float32")
    columns = [7, 2, 113]
    labels = (x[:, 7] > 0).astype("int64")
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42).fit(
        x[:, columns], labels
    )
    graph = study.graph_for(model, columns)
    session = ort.InferenceSession(
        graph.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    actual = session.run([graph.graph.output[1].name], {"features": x})[0]
    np.testing.assert_allclose(actual, model.predict_proba(x[:, columns]), atol=1e-6)


def test_record_aggregation_and_unsupported_label():
    result = study.aggregate(
        np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]]),
        np.array([7, 7, 9]),
        np.array([9, 7, 99]),
    )
    np.testing.assert_array_equal(result["labels"], [0, 1, -1])
    np.testing.assert_array_equal(result["supported"], [True, True, False])
    np.testing.assert_allclose(result["probabilities"][1], [0.3, 0.7])
    assert np.isnan(result["probabilities"][2]).all()


def test_empty_supported_windows_remain_unsupported():
    result = study.aggregate(
        np.empty((0, 24)), np.array([], dtype="int64"), np.array([1, 2])
    )
    np.testing.assert_array_equal(result["labels"], [-1, -1])
    assert not result["supported"].any()
