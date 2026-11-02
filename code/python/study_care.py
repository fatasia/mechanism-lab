"""CARE v6预注册12例试点：只用Avg信号，训练状态可筛选，预测状态/事件标签不入模。"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from phm_core.artifacts import capture_source, new_run
from phm_core.event_metrics import match_warnings
from public_data import sha256

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = {
    "record": 15846963,
    "cases": "frozen download protocol; 2 normal +2 anomaly per farm, distinct assets",
    "features": "only *_avg; no status, id, train_test, event or fault description in model",
    "training": "first70% publisher train normal status0/2; last30% train normal for calibration; no target refit",
    "quantiles": [0.99, 0.995, 0.999],
    "methods": ["mad", "ewma_mad", "pca_residual"],
    "min_lead_h": 1,
    "merge_gap_h": 6,
    "chance_draws": 200,
    "seed": 42,
    "scope": "12-case pilot, custom event metrics NOT official CARE score, NOT full95 dataset or battery-line validation",
}


def hour(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp() / 3600


def load(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream, delimiter=";")
        header = next(row for row in reader if row)
        columns = [i for i, name in enumerate(header) if name.endswith("_avg")]
        positions = [
            header.index(k)
            for k in ("time_stamp", "train_test", "status_type_id", "asset_id")
        ]
        values, times, parts, status, assets = [], [], [], [], set()
        for row in reader:
            if not row:
                continue
            times.append(hour(row[positions[0]]))
            parts.append(row[positions[1]])
            status.append(int(float(row[positions[2]])))
            assets.add(row[positions[3]])
            values.append([float(row[i]) if row[i] else np.nan for i in columns])
    times, x = np.array(times), np.array(values, dtype=float)
    if not len(columns) or len(assets) != 1 or np.any(np.diff(times) <= 0):
        raise ValueError("CARE身份/时间/信号审计失败")
    train = np.array(parts) == "train"
    if not train.any() or train.all() or not np.all(train[: train.sum()]):
        raise ValueError("不支持的训练/预测分区")
    return times, x, train, np.isin(status, [0, 2]), sorted(assets)[0]


def smooth(times, values):
    result, state = np.full(len(values), np.nan), None
    for i, value in enumerate(values):
        if not np.isfinite(value):
            state = None
            continue
        state = (
            value
            if state is None or (i and times[i] - times[i - 1] > 1)
            else 0.1 * value + 0.9 * state
        )
        result[i] = state
    return result


def run():
    run_id, out = new_run(ROOT / "outputs", "care-early-warning")
    (out / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2))
    directory = ROOT / "data/raw/care-v6"
    source = json.loads((directory / "source.json").read_text())
    cases = []
    for entry in source["files"]:
        path = directory / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("CARE输入摘要不一致")
        times, x, train, normal, asset = load(path)
        cut = int(train.sum() * 0.7)
        fit = train & normal & (np.arange(len(train)) < cut)
        calibration = train & normal & (np.arange(len(train)) >= cut)
        center = np.nanmedian(x[fit], axis=0)
        scale = 1.4826 * np.nanmedian(np.abs(x[fit] - center), axis=0)
        usable = np.isfinite(center) & np.isfinite(scale) & (scale > 1e-8)
        if usable.sum() < 2 or calibration.sum() < 100:
            raise ValueError("训练支持不足")
        z = (x[:, usable] - center[usable]) / scale[usable]
        supported = np.mean(np.isfinite(z), axis=1) >= 0.8
        z = np.clip(np.nan_to_num(z, nan=0, posinf=100, neginf=-100), -100, 100)
        raw = np.abs(z).mean(axis=1)
        raw[~supported] = np.nan
        pca = PCA(
            n_components=min(8, z.shape[1] - 1),
            svd_solver="randomized",
            random_state=42,
        ).fit(z[fit & supported])
        residual = np.sqrt(
            np.mean((z - pca.inverse_transform(pca.transform(z))) ** 2, axis=1)
        )
        residual[~supported] = np.nan
        scores = {
            "mad": raw,
            "ewma_mad": smooth(times, raw),
            "pca_residual": smooth(times, residual),
        }
        # 留出的正常事件区间才提供正常暴露分母；故障案例中其它时间不冒充确认健康。
        start, end = hour(entry["event_start"]), hour(entry["event_end"])
        evaluated = ~train & supported & (times >= start) & (times <= end)
        observed = ~train & supported
        rows = []
        for method, score in scores.items():
            for q in PROTOCOL["quantiles"]:
                threshold = float(
                    np.quantile(score[calibration & np.isfinite(score)], q)
                )
                alarm = observed & (score > threshold)
                fault = entry["event_label"] == "anomaly"
                result = match_warnings(
                    times[alarm if fault else alarm & evaluated],
                    [end] if fault else [],
                    max(
                        float(observed.sum() / 6 if fault else evaluated.sum() / 6),
                        1 / 6,
                    ),
                    max(end - start, 2),
                    1,
                    6,
                )
                rng = np.random.default_rng(42)
                hits = []
                if fault:
                    prediction_times = times[observed]
                    flags = alarm[observed]
                    for offset in rng.integers(
                        0, len(flags), size=PROTOCOL["chance_draws"]
                    ):
                        rotated = np.roll(flags, int(offset))
                        hits.append(
                            match_warnings(
                                prediction_times[rotated],
                                [end],
                                observed.sum() / 6,
                                max(end - start, 2),
                                1,
                                6,
                            )["detected"]
                        )
                rows.append(
                    {
                        "method": method,
                        "quantile": q,
                        "threshold": threshold,
                        **result,
                        "normal_hours": float(evaluated.sum() / 6)
                        if not fault
                        else None,
                        "chance_circular_shift_mean_detected": float(np.mean(hits))
                        if hits
                        else None,
                    }
                )
        cases.append(
            {
                "file": entry["file"],
                "farm": entry["farm"],
                "asset_id": asset,
                "event_label": entry["event_label"],
                "train_rows": int(train.sum()),
                "fit_rows": int(fit.sum()),
                "calibration_rows": int(calibration.sum()),
                "observed_prediction_hours": float(observed.sum() / 6),
                "unscored_prediction_hours": float((~train & ~supported).sum() / 6),
                "average_features": int(usable.sum()),
                "results": rows,
            }
        )
        print(json.dumps({"care_case": entry["file"], "done": len(cases)}), flush=True)
    report = {
        "run_id": run_id,
        "protocol": PROTOCOL,
        "cases": cases,
        "source_manifest_sha256": sha256(directory / "source.json"),
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", Path(__file__).name
        ),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"run_id": run_id, "output": str(out)}), flush=True)


if __name__ == "__main__":
    run()
