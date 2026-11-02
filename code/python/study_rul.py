"""C-MAPSS FD001 剩余寿命对照；训练/选型/区间校准严格按发动机隔离。"""

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import onnx
import torch
from onnx.compose import merge_models
from skl2onnx import to_onnx
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from phm_core.artifacts import capture_source, new_run, publish_run
from phm_core.inference import cpu_session
from phm_core.prognostics import life_features, regression_metrics, residual_interval

ROOT = Path(__file__).resolve().parents[2]
WINDOW = 30


class FeatureGraph(torch.nn.Module):
    def __init__(self):
        super().__init__()
        t = torch.arange(WINDOW, dtype=torch.float64)
        self.register_buffer("t", t - t.mean())

    def forward(self, x):
        x = x.double()
        slope = (x * self.t[None, :, None]).sum(1) / self.t.square().sum()
        return torch.cat(
            [x[:, -1], x.mean(1), x.std(1, unbiased=False), slope], 1
        ).float()


def read_data():
    with ZipFile(ROOT / "data/raw/cmapss/CMAPSSData.zip") as archive:
        train = np.loadtxt(archive.open("train_FD001.txt"))
        test = np.loadtxt(archive.open("test_FD001.txt"))
        truth = np.loadtxt(archive.open("RUL_FD001.txt"))
    return train, test, truth


def examples(data, units, training=False):
    raw, truth, ids, cycles = [], [], [], []
    for unit in units:
        rows = data[data[:, 0] == unit]
        # 验证和校准每个单元仅取一次截断，截断位置不依赖传感器或预测误差。
        ends = (
            range(WINDOW, len(rows) + 1, 5)
            if training
            else [
                max(
                    WINDOW,
                    int(
                        len(rows)
                        * np.random.default_rng(1000 + int(unit)).uniform(0.4, 0.9)
                    ),
                )
            ]
        )
        for end in ends:
            raw.append(rows[end - WINDOW : end, 2:])
            truth.append(len(rows) - end)
            ids.append(int(unit))
            cycles.append(int(rows[end - 1, 1]))
    return np.asarray(raw, dtype="float32"), np.array(truth), ids, cycles


def run():
    torch.set_num_threads(4)
    train, test, truth = read_data()
    units = np.random.default_rng(42).permutation(np.arange(1, 101))
    splits = {
        "train": units[:70].tolist(),
        "validation": units[70:85].tolist(),
        "calibration": units[85:].tolist(),
    }
    run_id, out = new_run(ROOT / "outputs", "rul-study")
    protocol = {
        "data_source": json.loads((ROOT / "data/raw/cmapss/source.json").read_text()),
        "dataset": "NASA C-MAPSS FD001, simulated turbofan",
        "source": "https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data",
        "domain": "synthetic aircraft engines, not lithium manufacturing equipment",
        "window_cycles": WINDOW,
        "input": "three operating settings + 21 sensors in publisher order",
        "target": "min(remaining cycles,125) during training; raw and capped test metrics both reported",
        "selection": "validation endpoint RMSE, then fixed seed42; test never used for selection",
        "calibration": "15 separate engine units, one seeded truncation per unit; absolute residual 90% split conformal rank",
        "scope": "unit exchangeability assumption; no calendar life or cross-machine assurance",
        "splits": splits,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    source = capture_source(
        ROOT / "code/python", out / "train-source.zip", "study_rul.py"
    )
    batches = {
        name: examples(train, ids, name == "train") for name, ids in splits.items()
    }
    test_x = np.stack(
        [test[test[:, 0] == i][-WINDOW:, 2:] for i in range(1, 101)]
    ).astype("float32")
    features = {k: life_features(v[0]) for k, v in batches.items()}
    test_features = life_features(test_x)
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=100)),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=4,
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(features["train"], np.minimum(batches["train"][1], 125))
        valid = np.clip(model.predict(features["validation"]), 0, 125)
        results[name] = {
            "validation": regression_metrics(batches["validation"][1], valid)
        }
    selected = min(results, key=lambda name: results[name]["validation"]["rmse_cycles"])
    # 在打开正式 test 预测前冻结选择和训练协议。
    (out / "selection.json").write_text(
        json.dumps({"selected": selected, "validation": results}, indent=2)
    )
    records = {}
    for name, model in models.items():
        prediction = np.clip(model.predict(test_features), 0, 125)
        calibration = np.clip(model.predict(features["calibration"]), 0, 125)
        radius = residual_interval(batches["calibration"][1], calibration)
        lower, upper = np.maximum(0, prediction - radius), prediction + radius
        results[name].update(
            {
                "test": regression_metrics(truth, prediction),
                "capped_test": regression_metrics(np.minimum(truth, 125), prediction),
                "interval_radius_cycles": radius,
                "interval_coverage": float(
                    ((truth >= lower) & (truth <= upper)).mean()
                ),
                "mean_interval_width_cycles": float((upper - lower).mean()),
                "calibration_units": 15,
            }
        )
        records[name] = [
            {
                "unit": i + 1,
                "observed_cycles": int((test[:, 0] == i + 1).sum()),
                "truth": float(truth[i]),
                "prediction": float(prediction[i]),
                "lower": float(lower[i]),
                "upper": float(upper[i]),
            }
            for i in range(100)
        ]
    # 将原始窗口特征与回归器合并；限幅和区间是部署契约中的后处理。
    model = models[selected]
    downstream = to_onnx(model, features["train"][:1], target_opset=17)
    torch.onnx.export(
        FeatureGraph().eval(),
        torch.from_numpy(test_x[:1]),
        str(out / "features.onnx"),
        input_names=["values"],
        output_names=["features"],
        dynamic_axes={"values": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    frontend = onnx.load(out / "features.onnx")
    frontend.ir_version = downstream.ir_version
    graph = merge_models(
        frontend,
        downstream,
        [("features", downstream.graph.input[0].name)],
        prefix1="pre/",
        prefix2="reg/",
    )
    onnx.checker.check_model(graph)
    onnx.save(graph, out / "model.onnx")
    (out / "features.onnx").unlink()
    session = cpu_session(out / "model.onnx")
    actual = np.concatenate(
        [
            session.run(None, {"pre/values": x})[0].ravel()
            for x in np.array_split(test_x[:64], 8)
        ]
    )
    reference = model.predict(test_features[:64])
    np.testing.assert_allclose(actual, reference, atol=1e-3, rtol=1e-4)
    manifest = {
        "schemaVersion": 1,
        "task": "remaining-useful-life",
        "version": run_id,
        "file": "model.onnx",
        "sha256": hashlib.sha256((out / "model.onnx").read_bytes()).hexdigest(),
        "input": {
            "name": "pre/values",
            "layout": "BTC",
            "shape": [None, 30, 24],
            "channels": ["setting1", "setting2", "setting3"]
            + [f"sensor{i}" for i in range(1, 22)],
            "units": "publisher native units; one observation per operational cycle",
        },
        "postprocess": {
            "clip_prediction": [0, 125],
            "interval_radius_cycles": results[selected]["interval_radius_cycles"],
            "lower_bound": 0,
        },
        "output": {"index": 0, "unit": "cycles"},
        "maxBatch": 8,
        "parity_max_abs_error": float(np.max(np.abs(actual - reference))),
        "scope": protocol["domain"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    report = {
        "run_id": run_id,
        "source_hash": source,
        "protocol": protocol,
        "selected": selected,
        "models": results,
        "records": records,
        "manifest": manifest,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(out / "parity.npz", raw=test_x[:64], reference=reference)
    with ZipFile(out / "model-bundle.zip", "w", ZIP_DEFLATED) as archive:
        for name in ("model.onnx", "manifest.json", "protocol.json", "selection.json"):
            archive.write(out / name, name)
    publish_run(ROOT / "outputs", "rul-study", run_id)
    print(
        json.dumps({"run_id": run_id, "selected": selected, "models": results}),
        flush=True,
    )


if __name__ == "__main__":
    run()
