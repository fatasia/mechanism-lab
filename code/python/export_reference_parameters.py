"""在隔离的 aeon 环境中导出数值参数，不把 pickle 带入部署运行时。"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from phm_core.actuator import load_records, split_records

ROOT = Path(__file__).resolve().parents[2]


def run(run_id):
    directory = ROOT / "outputs/reference-study/runs" / run_id
    x, _, record_ids, metadata = load_records(ROOT / "data/raw/cranfield-actuator")
    for protocol in ("record-holdout", "held-out-load", "held-out-profile"):
        artifact = joblib.load(directory / f"{protocol}-MiniRocket.joblib")
        split = split_records(metadata, protocol)
        train = np.flatnonzero(np.isin(record_ids, split["train"]))
        test = np.flatnonzero(np.isin(record_ids, split["test"]))[:64]
        mean, scale = x[train].mean((0, 1)), x[train].std((0, 1))
        transform, scaler, classifier = (
            artifact["transform"],
            artifact["scaler"],
            artifact["classifier"],
        )
        normalized = ((x[test] - mean) / scale).transpose(0, 2, 1).copy()
        scores = classifier.decision_function(
            scaler.transform(transform.transform(normalized))
        )
        counts, channels, dilations, features, biases = transform.parameters
        np.savez_compressed(
            directory / f"{protocol}-minirocket-parameters.npz",
            channel_counts=counts,
            channel_indices=channels,
            dilations=dilations,
            features_per_dilation=features,
            biases=biases,
            mean=mean,
            scale=scale,
            feature_mean=scaler.mean_,
            feature_scale=scaler.scale_,
            coef=classifier.coef_,
            intercept=classifier.intercept_,
            parity_raw=x[test],
            parity_scores=scores,
        )
    print(json.dumps({"run_id": run_id, "parameter_files": 3}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    run(parser.parse_args().run_id)
