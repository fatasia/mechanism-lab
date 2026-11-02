"""极片厚度回归：配对样本/工艺组合整组留出，禁止把密度、质量、厚度派生值入模。"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from phm_core.artifacts import capture_source, new_run
from public_data import sha256

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = {
    "version": 4,
    "target": "Thickness_um",
    "process_fields": [
        "Web_speed",
        "Calendering_Speed",
        "Roll_Gap",
        "Coat_Weight",
        "Calendering_State",
    ],
    "forbidden_inputs": ["Thickness", "Density", "Mass", "Sample_ID"],
    "protocols": ["leave-original-electrode-out", "leave-process-combination-out"],
    "arms": ["process_only", "process_plus_spectrum"],
    "models": ["ridge_alpha1", "rf200_leaf2_seed42"],
    "selection": "all prespecified; no winning model selected from target results; materials evaluated separately",
    "scope": "quality regression, not equipment PHM or failure lead time",
}


def encode(record):
    meta = record["metadata"]
    process = [float(meta.get(k, 0)) for k in PROTOCOL["process_fields"][:-1]]
    process.append(float(meta["Calendering_State"] == "after-calendering"))
    frequency, magnitude = (
        np.asarray(record["fft_frequency"]),
        np.asarray(record["fft_magnitude"]),
    )
    if (
        frequency.shape != magnitude.shape
        or not np.isfinite(magnitude).all()
        or np.any(magnitude < 0)
    ):
        raise ValueError("无效频谱")
    total = max(magnitude.sum(), 1e-12)
    spectral = [
        float(magnitude.mean()),
        float(magnitude.std()),
        float(frequency @ magnitude / total),
        float(frequency[magnitude.argmax()]),
    ]
    spectral += [
        float(magnitude[(frequency >= lo) & (frequency < hi)].sum() / total)
        for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 16))
    ]
    return process, spectral


def paired_gain(rows):
    base = {(r["sample"], r["state"]): r for r in rows if r["arm"] == "process_only"}
    gains = {}
    for row in rows:
        if row["arm"] == "process_plus_spectrum":
            reference = base[(row["sample"], row["state"])]
            gains.setdefault(row["held_group"], []).append(
                abs(reference["predicted"] - row["actual"])
                - abs(row["predicted"] - row["actual"])
            )
    values = np.array([np.mean(v) for v in gains.values()])
    rng = np.random.default_rng(20261102)
    means = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return {
        "cluster_equal_weight_mae_gain_um": float(values.mean()),
        "bootstrap_interval_95": np.quantile(means, [0.025, 0.975]).tolist(),
        "independent_groups": len(values),
    }


def run():
    run_id, out = new_run(ROOT / "outputs", "electrode-quality")
    (out / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2))
    directory = ROOT / "data/raw/electrode-ultrasound-v4"
    source = json.loads((directory / "source.json").read_text())
    records = []
    for entry in source["files"]:
        path = directory / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("输入哈希不符")
        records.append(json.loads(path.read_text()))
    results = []
    predictions = []
    for material in ("GA", "NMC622"):
        subset = [
            r for r in records if r["metadata"]["Sample_ID"].startswith(material + "_")
        ]
        keys = [r["metadata"]["Sample_ID"] for r in subset]
        x = np.array([a + b for a, b in map(encode, subset)])
        y = np.array([r["metadata"]["Thickness"] for r in subset], dtype=float)
        # 同一极片的前后两份记录共用工艺组合，状态不进入分组键。
        process_groups = [json.dumps(encode(r)[0][:-1]) for r in subset]
        for key in set(keys):
            paired = [r for r in subset if r["metadata"]["Sample_ID"] == key]
            if (
                len(paired) != 2
                or len({json.dumps(encode(r)[0][:-1]) for r in paired}) != 1
            ):
                raise ValueError("配对样本或工艺元数据不一致，不允许拆散到不同折")
        for protocol, groups in [
            (PROTOCOL["protocols"][0], keys),
            (PROTOCOL["protocols"][1], process_groups),
        ]:
            if len(set(groups)) < 3:
                raise ValueError("独立工艺组太少")
            for arm, columns in [
                ("process_only", slice(0, 5)),
                ("process_plus_spectrum", slice(None)),
            ]:
                for method in PROTOCOL["models"]:
                    predicted = np.zeros(len(y))
                    for train, test in LeaveOneGroupOut().split(x, y, groups):
                        model = (
                            make_pipeline(StandardScaler(), Ridge(alpha=1))
                            if method.startswith("ridge")
                            else RandomForestRegressor(
                                n_estimators=200,
                                min_samples_leaf=2,
                                random_state=42,
                                n_jobs=1,
                            )
                        )
                        model.fit(x[train, columns], y[train])
                        predicted[test] = model.predict(x[test, columns])
                        for i in test:
                            predictions.append(
                                {
                                    "material": material,
                                    "protocol": protocol,
                                    "arm": arm,
                                    "model": method,
                                    "sample": keys[i],
                                    "state": subset[i]["metadata"]["Calendering_State"],
                                    "held_group": groups[i],
                                    "actual": float(y[i]),
                                    "predicted": float(predicted[i]),
                                }
                            )
                    results.append(
                        {
                            "material": material,
                            "protocol": protocol,
                            "arm": arm,
                            "model": method,
                            "records": len(y),
                            "electrodes": len(set(keys)),
                            "groups": len(set(groups)),
                            "mae_um": float(mean_absolute_error(y, predicted)),
                            "rmse_um": float(np.sqrt(mean_squared_error(y, predicted))),
                        }
                    )
    gains = []
    for material in ("GA", "NMC622"):
        for protocol in PROTOCOL["protocols"]:
            for method in PROTOCOL["models"]:
                rows = [
                    r
                    for r in predictions
                    if r["material"] == material
                    and r["protocol"] == protocol
                    and r["model"] == method
                ]
                gains.append(
                    {
                        "material": material,
                        "protocol": protocol,
                        "model": method,
                        **paired_gain(rows),
                    }
                )
    report = {
        "run_id": run_id,
        "protocol": PROTOCOL,
        "results": results,
        "predictions": predictions,
        "paired_gains": gains,
        "source_manifest_sha256": sha256(directory / "source.json"),
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", Path(__file__).name
        ),
        "limitations": "one laboratory; no equipment故障标签; recipe parameters may encode planned thickness; no production tolerance provided",
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"run_id": run_id, "output": str(out)}), flush=True)


if __name__ == "__main__":
    run()
