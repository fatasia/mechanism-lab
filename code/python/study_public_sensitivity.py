"""预先固定全网格：CoMoPI事件窗口和ZeMA阈值/持续周期，不按目标结果选优。"""

import itertools
import json
from pathlib import Path

import numpy as np

import study_comopi_early_warning as comopi
import study_zema_degradation as zema
from phm_core.artifacts import capture_source, new_run
from phm_core.degradation import (
    health_index,
    persistent_crossing,
    reference_threshold,
    reference_z,
    select_consistent,
)
from public_data import sha256

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = {
    "comopi_horizon_h": [24, 72, 168],
    "comopi_merge_gap_h": [1, 6, 12],
    "comopi_quantiles": [0.99, 0.995, 0.999],
    "chance_draws": 100,
    "zema_sigmas": [3, 6, 9],
    "zema_hold": [25, 100, 250],
    "zema_smoothing": ["historical_centered", "causal_trailing"],
    "scope": "exploratory full grid; no best parameter selected; observed hours != healthy hours",
}


def trailing_median(values, width):
    padded = np.pad(values, (width - 1, 0), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, width), axis=1)


def run():
    run_id, out = new_run(ROOT / "outputs", "public-sensitivity")
    (out / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2))
    data, events = comopi.load_sensors(), comopi.load_events()
    scores = {s: comopi.machine_scores(*data[s]) for s in data}
    targets = [
        s
        for s in sorted(data)
        if len(data[s][0]) > comopi.PROTOCOL["reference_windows"] + 144
    ]
    results = []
    for horizon, gap, quantile in itertools.product(
        PROTOCOL["comopi_horizon_h"],
        PROTOCOL["comopi_merge_gap_h"],
        PROTOCOL["comopi_quantiles"],
    ):
        protocol = {**comopi.PROTOCOL, "horizon_h": horizon, "warning_merge_gap_h": gap}
        machines = {
            s: comopi.evaluate_target(
                s, data, events, quantile, protocol, scores, PROTOCOL["chance_draws"]
            )
            for s in targets
        }
        results.append(
            {
                "horizon_h": horizon,
                "merge_gap_h": gap,
                "quantile": quantile,
                "pooled": comopi.pool(list(machines.values()), "sensor"),
                "machines": machines,
            }
        )
    print("CoMoPI 27 cells completed", flush=True)
    zs, counts, inputs = {}, {}, {}
    cache = ROOT / "data/prepared/zema-sensitivity-features"
    cache.mkdir(parents=True, exist_ok=True)
    for axis in (3, 5, 7, 11):
        path = ROOT / (
            f"data/raw/zema-emc/axis{axis}_2kHz.h5"
            if axis != 11
            else "data/raw/zema-axis11-si/axis11_2kHz_ZeMA_PTB_SI.h5"
        )
        digest = sha256(path)
        cached = cache / f"{digest}.npz"
        if cached.exists():
            with np.load(cached) as f:
                features, names = f["features"], f["names"].tolist()
        else:
            features, names = (
                zema.load_axis(path) if axis != 11 else zema.load_axis11(path)
            )
            np.savez_compressed(cached, features=features, names=np.asarray(names))
        zs[str(axis)], _, counts[str(axis)] = reference_z(features)
        inputs[str(axis)] = digest
        print(f"ZeMA axis{axis} loaded", flush=True)
    trends = []
    for target in sorted(zs):
        indices, signs = select_consistent([v for k, v in zs.items() if k != target])
        raw = (
            (zs[target][:, indices] * signs).mean(axis=1)
            if len(indices)
            else np.zeros(len(zs[target]))
        )
        for smoothing in PROTOCOL["zema_smoothing"]:
            index = (
                health_index(zs[target], indices, signs)
                if smoothing == "historical_centered"
                else trailing_median(raw, 101)
            )
            for sigma, hold in itertools.product(
                PROTOCOL["zema_sigmas"], PROTOCOL["zema_hold"]
            ):
                threshold = reference_threshold(index, counts[target], sigma)
                start, fraction = persistent_crossing(
                    index, threshold, counts[target], hold
                )
                confirmed = None if start is None else start + hold - 1
                trends.append(
                    {
                        "target": target,
                        "smoothing": smoothing,
                        "sigmas": sigma,
                        "hold": hold,
                        "selected_features": len(indices),
                        "threshold": threshold,
                        "onset_cycle": start,
                        "onset_fraction": fraction,
                        "confirmation_cycle": confirmed,
                        "confirmation_fraction": None
                        if confirmed is None
                        else confirmed / len(index),
                    }
                )
    report = {
        "run_id": run_id,
        "protocol": PROTOCOL,
        "comopi": results,
        "zema": trends,
        "zema_input_sha256": inputs,
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", Path(__file__).name
        ),
        "helper_sha256": {
            p: sha256(ROOT / "code/python" / p)
            for p in (
                "study_comopi_early_warning.py",
                "study_zema_degradation.py",
                "public_data.py",
            )
        },
        "causality": "historical centered median uses future samples; onset is retrospective; causal confirmation uses trailing samples and full hold",
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"run_id": run_id, "output": str(out)}), flush=True)


if __name__ == "__main__":
    run()
