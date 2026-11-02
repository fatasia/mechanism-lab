"""未见资产最终评估：一次性账本、源校准、固定方法、不按测试结果选型。"""

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from phm_core.artifacts import capture_source, new_run
from phm_core.event_metrics import match_warnings
from public_data import sha256
from study_care import hour, load, smooth

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = {
    "methods": ["pca", "regime_pca", "cusum"],
    "quantile": 0.995,
    "fit_fraction": 0.7,
    "components": 8,
    "regimes": 4,
    "seed": 42,
    "chance_draws": 1000,
    "normal_budget_per_1000h": 1.0,
    "scope": "unseen assets relative to pilot, but per-asset normal training is provided; not zero-shot",
    "comparison": "all fixed methods, no test-selected winner; equal source quantile is NOT equal target false alarm rate",
}


def cusum(times, values, drift):
    result = np.full(len(values), np.nan)
    state = 0.0
    for i, value in enumerate(values):
        if not np.isfinite(value):
            state = 0.0
            continue
        if i and times[i] - times[i - 1] > 1:
            state = 0.0
        state = max(0.0, state + min(value, 100) - drift)
        result[i] = state
    return result


def score_case(times, x, train, normal):
    split = int(train.sum() * PROTOCOL["fit_fraction"])
    fit = train & normal & (np.arange(len(train)) < split)
    calibration = train & normal & (np.arange(len(train)) >= split)
    center = np.nanmedian(x[fit], axis=0)
    scale = 1.4826 * np.nanmedian(abs(x[fit] - center), axis=0)
    usable = np.isfinite(center) & np.isfinite(scale) & (scale > 1e-8)
    if usable.sum() < 2 or calibration.sum() < 100:
        raise ValueError("正常训练支持不足")
    z = (x[:, usable] - center[usable]) / scale[usable]
    supported = np.mean(np.isfinite(z), axis=1) >= 0.8
    z = np.clip(np.nan_to_num(z, nan=0, posinf=100, neginf=-100), -100, 100)
    fit &= supported
    calibration &= supported
    pca = PCA(
        n_components=min(PROTOCOL["components"], z.shape[1] - 1),
        svd_solver="randomized",
        random_state=42,
    ).fit(z[fit])
    latent = pca.transform(z)
    residual = np.sqrt(np.mean((z - pca.inverse_transform(latent)) ** 2, axis=1))
    # 工况簇仅由源正常训练的前两主成分定义，不使用目标状态码或标签。
    model = MiniBatchKMeans(n_clusters=4, random_state=42, n_init=3).fit(
        latent[fit, :2]
    )
    groups = model.predict(latent[:, :2])
    denominators = np.array(
        [
            max(float(np.median(residual[fit & (groups == g)])), 1e-6)
            if (fit & (groups == g)).any()
            else max(float(np.median(residual[fit])), 1e-6)
            for g in range(4)
        ]
    )
    normalized = residual / denominators[groups]
    residual[~supported] = np.nan
    normalized[~supported] = np.nan
    drift = float(np.quantile(normalized[fit], 0.95))
    scores = {
        "pca": smooth(times, residual),
        "regime_pca": smooth(times, normalized),
        "cusum": cusum(times, normalized, drift),
    }
    thresholds = {
        k: float(np.quantile(v[calibration], 0.995)) for k, v in scores.items()
    }
    return scores, thresholds, supported


def run():
    directory = ROOT / "data/raw/care-holdout-v6"
    source = json.loads((directory / "source.json").read_text())
    pilot = json.loads((ROOT / "data/raw/care-v6/source.json").read_text())
    seen = {(r["farm"], str(r["asset_id"])) for r in pilot["files"]}
    identities = [(r["farm"], str(r["asset_id"])) for r in source["files"]]
    if seen.intersection(identities) or len(set(identities)) != len(identities):
        raise ValueError("最终测试设备不独立")
    ledger = directory / "evaluation-started.json"
    if ledger.exists():
        raise ValueError("最终测试已打开；保留账本，后续只可标记为重复评价")
    run_id, out = new_run(ROOT / "outputs", "care-holdout")
    ledger.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "protocol": PROTOCOL,
                "source_sha256": sha256(directory / "source.json"),
            },
            indent=2,
        )
    )
    (out / "protocol.json").write_text(ledger.read_text())
    cases = []
    nulls = {k: np.zeros(PROTOCOL["chance_draws"]) for k in PROTOCOL["methods"]}
    for index, entry in enumerate(source["files"]):
        path = directory / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("输入哈希不符")
        times, x, train, normal, asset = load(path)
        if str(asset) != str(entry["asset_id"]):
            raise ValueError("设备身份不符")
        scores, thresholds, supported = score_case(times, x, train, normal)
        start, end = hour(entry["event_start"]), hour(entry["event_end"])
        fault = entry["event_label"] == "anomaly"
        observed = ~train & supported
        evaluated = observed if fault else observed & (times >= start) & (times <= end)
        exposure = float(evaluated.sum() / 6)
        if exposure <= 0:
            raise ValueError("无可评价时间")
        results = []
        for method, values in scores.items():
            alarm = evaluated & (values > thresholds[method])
            metrics = match_warnings(
                times[alarm],
                [end] if fault else [],
                exposure,
                max(end - start, 2),
                1,
                6,
            )
            hits = []
            if fault:
                stamps = times[observed]
                flags = alarm[observed]
                rng = np.random.default_rng(42 + index)
                for offset in rng.integers(
                    0, len(flags), size=PROTOCOL["chance_draws"]
                ):
                    hits.append(
                        match_warnings(
                            stamps[np.roll(flags, int(offset))],
                            [end],
                            exposure,
                            max(end - start, 2),
                            1,
                            6,
                        )["detected"]
                    )
                nulls[method] += hits
            results.append(
                {
                    "method": method,
                    "threshold": thresholds[method],
                    **metrics,
                    "normal_hours": None if fault else exposure,
                    "chance_mean_detected": float(np.mean(hits)) if hits else None,
                }
            )
        cases.append(
            {
                "file": entry["file"],
                "farm": entry["farm"],
                "asset_id": asset,
                "event_label": entry["event_label"],
                "unscored_prediction_hours": float((~train & ~supported).sum() / 6),
                "results": results,
            }
        )
        (out / "progress.json").write_text(json.dumps({"cases_completed": len(cases)}))
        print(json.dumps({"case": entry["file"], "done": len(cases)}), flush=True)
    summary = []
    for method in PROTOCOL["methods"]:
        rows = [
            (c, next(r for r in c["results"] if r["method"] == method)) for c in cases
        ]
        fault_rows = [r for c, r in rows if c["event_label"] == "anomaly"]
        normal_rows = [r for c, r in rows if c["event_label"] == "normal"]
        hits = sum(r["detected"] for r in fault_rows)
        hours = sum(r["normal_hours"] for r in normal_rows)
        alerts = sum(r["false_episodes"] for r in normal_rows)
        p = float(
            (1 + np.count_nonzero(nulls[method] >= hits)) / (1 + len(nulls[method]))
        )
        summary.append(
            {
                "method": method,
                "detected": hits,
                "fault_assets": len(fault_rows),
                "normal_hours": hours,
                "normal_alerts": alerts,
                "normal_alerts_per_1000h": alerts / hours * 1000,
                "within_prespecified_normal_budget": alerts / hours * 1000 <= 1,
                "chance_expected_hits": float(nulls[method].mean()),
                "randomization_p_one_sided": p,
                "p_bonferroni_three_methods": min(1, p * 3),
                "leads_h": [v for r in fault_rows for v in r["lead_hours"]],
            }
        )
    report = {
        "run_id": run_id,
        "protocol": PROTOCOL,
        "cases": cases,
        "summary": summary,
        "source_sha256": sha256(directory / "source.json"),
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", Path(__file__).name
        ),
        "limitations": "12 asset case-control sample; circular shifts preserve point count, not necessarily time episode count across gaps; post-maintenance benefits unmeasured; not official CARE score",
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(out), "summary": summary}), flush=True)


if __name__ == "__main__":
    run()
