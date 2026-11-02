import hashlib
import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from phm_core.portable import PortableScorer


def model_package(path):
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["values"], ["score"], axes=[1], keepdims=0)],
        "fixture",
        [helper.make_tensor_value_info("values", TensorProto.FLOAT, ["batch", 4, 1])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, ["batch", 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path / "model.onnx")
    manifest = {
        "schemaVersion": 2,
        "graph": {
            "file": "model.onnx",
            "sha256": hashlib.sha256((path / "model.onnx").read_bytes()).hexdigest(),
        },
        "input": {
            "name": "values",
            "shape": [None, 4, 1],
            "channels": ["x"],
            "units": ["bar"],
            "sampleRateHz": 10,
        },
        "outputs": [{"index": 0, "role": "anomaly-score"}],
        "execution": {"maxBatch": 8},
    }
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_portable_scores_raw_windows_and_checks_contract(tmp_path):
    model_package(tmp_path)
    scorer = PortableScorer(tmp_path)
    values = np.arange(8, dtype="float32").reshape(2, 4, 1)
    result = scorer.score(values, ["x"], ["bar"], 10)
    np.testing.assert_allclose(result["anomaly-score"].ravel(), [1.5, 5.5])
    with pytest.raises(ValueError, match="通道"):
        scorer.score(values, ["other"], ["bar"], 10)
    with pytest.raises(ValueError, match="批量"):
        scorer.score(np.zeros((9, 4, 1)), ["x"], ["bar"], 10)
    with pytest.raises(ValueError):
        scorer.score(np.full((1, 4, 1), np.nan), ["x"], ["bar"], 10)


def test_modified_graph_is_rejected_before_inference(tmp_path):
    model_package(tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"modified")
    with pytest.raises(ValueError, match="校验"):
        PortableScorer(tmp_path)
