"""导出源工况所选RF候选，包含列选择；工程特征变换仍在ONNX之外。"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from infer_motor_rf import aggregate
from motor_features import FEATURE_NAMES
from phm_core.artifacts import capture_source, new_run

ROOT = Path(__file__).resolve().parents[2]


def graph_for(model, columns):
    graph = convert_sklearn(
        model,
        initial_types=[("features", FloatTensorType([None, len(columns)]))],
        target_opset=17,
        options={id(model): {"zipmap": False}},
    )
    for node in graph.graph.node:
        for i, name in enumerate(node.input):
            if name == "features":
                node.input[i] = "selected_features"
    graph.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 114
    graph.graph.initializer.append(
        numpy_helper.from_array(
            np.asarray(columns, dtype="int64"), name="feature_columns"
        )
    )
    graph.graph.node.insert(
        0,
        helper.make_node(
            "Gather", ["features", "feature_columns"], ["selected_features"], axis=1
        ),
    )
    onnx.checker.check_model(graph)
    return graph


def run(study_directory, feature_directory):
    study = json.loads((study_directory / "metrics.json").read_text(encoding="utf-8"))
    manifest_path = feature_directory / "manifest.json"
    if (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        != study["feature_manifest_sha256"]
    ):
        raise ValueError("特征清单不属于原训练实验")
    features = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts, ids = [], []
    for record in features["records"]:
        cache = feature_directory / record["cache"]
        if hashlib.sha256(cache.read_bytes()).hexdigest() != record["cache_sha256"]:
            raise ValueError("特征缓存哈希变化")
        with np.load(cache) as packet:
            parts.append(packet["features"])
            ids.extend([record["id"]] * len(packet["features"]))
    x, ids = np.concatenate(parts), np.asarray(ids)
    requested = np.array([r["id"] for r in features["records"]])
    run_id, out = new_run(ROOT / "outputs", "motor-rf-export")
    source_hash = capture_source(
        ROOT / "code/python", out / "source.zip", "export_motor_rf.py"
    )
    results = []
    for protocol, result in study["protocols"].items():
        arm = result["selected"]
        if arm is None:
            continue
        columns = study["protocol"]["arms"][arm]
        for seed in study["protocol"]["seeds"]:
            model_path = study_directory / protocol / f"{arm}-{seed}.joblib"
            # 仅加载本工作区刚完成训练的自有产物；部署包不包含pickle/joblib。
            model = joblib.load(model_path)
            directory = out / f"{protocol}-{seed}"
            directory.mkdir()
            graph = graph_for(model, columns)
            path = directory / "model.onnx"
            onnx.save(graph, path)
            output_name = graph.graph.output[1].name
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            session = ort.InferenceSession(
                str(path), options, providers=["CPUExecutionProvider"]
            )
            actual = session.run([output_name], {"features": x})[0]
            expected = model.predict_proba(x[:, columns])
            maximum = float(np.max(np.abs(actual - expected)))
            window_mismatches = int(np.sum(actual.argmax(1) != expected.argmax(1)))
            a, b = (
                aggregate(actual, ids, requested),
                aggregate(expected, ids, requested),
            )
            record_mismatches = int(np.sum(a["labels"] != b["labels"]))
            if maximum > 1e-5 or window_mismatches or record_mismatches:
                raise ValueError(
                    f"ONNX一致性未通过: {protocol}/{seed}: {maximum}, {window_mismatches}, {record_mismatches}"
                )
            metadata = {
                "run_id": run_id,
                "origin": study["run_id"],
                "protocol": protocol,
                "arm": arm,
                "seed": seed,
                "status": "research candidate only; selected by source conditions; target3000rpm failed generalization; no online promotion",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "training_model_sha256": hashlib.sha256(
                    model_path.read_bytes()
                ).hexdigest(),
                "feature_manifest_sha256": study["feature_manifest_sha256"],
                "shape": [None, 114],
                "dtype": "float32",
                "feature_names": FEATURE_NAMES,
                "feature_columns_embedded": columns,
                "preprocessing_embedded": False,
                "raw_input_not_accepted": True,
                "classes": study["classes"],
                "healthy_class": study["healthy_class"],
                "probability_output": output_name,
                "aggregation": "mean supported-window probabilities per complete record,then argmax; absent record label=-1 supported=false",
                "target_result": result["target"][arm][str(seed)],
                "parity": {
                    "windows": len(x),
                    "records": len(requested),
                    "max_abs_error": maximum,
                    "window_label_mismatches": window_mismatches,
                    "record_label_mismatches": record_mismatches,
                },
            }
            for name in ("infer_motor_rf.py", "motor_features.py"):
                source = ROOT / "code/python" / name
                shutil.copyfile(source, directory / name)
                metadata[f"{name}_sha256"] = hashlib.sha256(
                    source.read_bytes()
                ).hexdigest()
            shutil.copyfile(
                ROOT / "code/THIRD_PARTY_NOTICES.md",
                directory / "THIRD_PARTY_NOTICES.md",
            )
            (directory / "manifest.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
            )
            (directory / "README.md").write_text(
                "# 电机RF研究候选\n\n输入为manifest中固定顺序的114维工程特征，不是原始波形。ONNX仅含列选择与分类器；FFT、相序及角域包络变换不在图中。motor_features.py提供原变换（另需SciPy），不得混用其它尺度或特征定义。\n\n推理只需NumPy、ONNX Runtime，无平台、Torch或scikit-learn依赖：\n\n```\npython infer_motor_rf.py --bundle . --input windows.npz --output result.npz\n```\n\n输入npz含features[N,114]和window_records[N]整数记录ID，可选requested_record_ids。输出按完整记录平均概率判类；不支持记录label=-1，不视为健康。原评价单位是记录，不应把窗口分类直接作为告警。\n\n本包没有通过现场准入。原3000rpm留出迁移失败；torque-profile结果也只有6条健康记录。数据来源MCC5-THU Motor，DOI10.17632/6s3dggj9mw.1，作者及许可见THIRD_PARTY_NOTICES.md。\n",
                encoding="utf-8",
            )
            with ZipFile(directory.with_suffix(".zip"), "w", ZIP_DEFLATED) as archive:
                for file in sorted(directory.iterdir()):
                    archive.write(file, file.name)
            results.append(metadata)
            print(protocol, seed, maximum, flush=True)
    (out / "metrics.json").write_text(
        json.dumps(
            {"run_id": run_id, "source_hash": source_hash, "models": results}, indent=2
        ),
        encoding="utf-8",
    )
    print(run_id, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    parser.add_argument("features", type=Path)
    args = parser.parse_args()
    run(args.study.resolve(), args.features.resolve())
