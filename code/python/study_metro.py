"""在役压缩机的时间外回测，维修表外告警记为未确认，不伪标为误报。"""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from skl2onnx import to_onnx
from sklearn.ensemble import IsolationForest

from phm_core.artifacts import capture_source, new_run, publish_run
from phm_core.inference import cpu_session
from phm_core.learning import fit_autoencoder
from phm_core.metro import FEATURES, event_report, prepare, timestamp
from phm_core.monitoring import calibrated_threshold

ROOT = Path(__file__).resolve().parents[2]


class SequenceScore(torch.nn.Module):
    def __init__(self, model, mean, scale):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.from_numpy(mean))
        self.register_buffer("scale", torch.from_numpy(scale))

    def forward(self, x):
        z = (x - self.mean) / self.scale
        return (self.model(z) - z).abs().mean((1, 2)).unsqueeze(1)


def run():
    data = prepare(
        ROOT / "data/raw/metropt3/dataset.zip", ROOT / "data/prepared/metro-v1.npz"
    )
    x, ends = data["x"], data["end"]
    train = np.flatnonzero(ends < timestamp("2020-02-21"))
    valid = np.flatnonzero(
        (ends >= timestamp("2020-02-21")) & (ends < timestamp("2020-03-01"))
    )
    calibration = np.flatnonzero(
        (ends >= timestamp("2020-03-01")) & (ends < timestamp("2020-03-15"))
    )
    test = np.flatnonzero(ends >= timestamp("2020-03-15"))
    active = x[train].std(0) > 1e-8
    x = x[:, active].astype("float32")
    run_id, out = new_run(ROOT / "outputs", "metro-study")
    bundle = out / "bundle"
    bundle.mkdir()
    protocol = {
        "training_end": "2020-02-21",
        "validation_end": "2020-03-01",
        "calibration_end": "2020-03-15",
        "test_start": "2020-03-15",
        "window_seconds": 600,
        "sequence_windows": 6,
        "seeds": [42, 43, 44],
        "epochs": 60,
        "threshold": "separate reference calibration upper rank at alpha=.01",
        "event_policy": "3 of 5 windows enter; 3 below hysteresis exit; 24h pre-report matching",
        "reference_assumption": "no published failure report during reference period does not establish healthy ground truth",
        "unmatched": "unconfirmed, not automatically false alarms",
        "features": [f for f, flag in zip(FEATURES, active) if flag],
        "excluded_inputs": ["LPS", "Oil_level", "Pressure_switch"],
        "selection": "no fault-labelled validation data; report all candidates, no production promotion",
        "time_zone": "source timestamps have no verified zone; comparisons use source clock",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    source = capture_source(
        ROOT / "code/python", out / "train-source.zip", "study_metro.py"
    )
    report = {
        "run_id": run_id,
        "source_hash": source,
        "protocol": protocol,
        "source_rows": int(data["source_rows"]),
        "median_interval_seconds": float(data["median_interval_seconds"]),
        "quality_excluded": data["invalid_counts"].tolist(),
        "counts": {
            "train": len(train),
            "validation": len(valid),
            "calibration": len(calibration),
            "test": len(test),
        },
        "models": {},
    }
    center = np.median(x[train], axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(x[train] - center), axis=0), 1e-4)
    baseline = np.abs((x - center) / scale).max(1)
    threshold = calibrated_threshold(baseline[calibration])
    report["models"]["robust"] = {
        "threshold": threshold,
        **event_report(baseline[test], ends[test], threshold),
    }
    scores = {"robust": baseline}
    sequence_ids = np.array(
        [i for i in range(5, len(x)) if ends[i] - ends[i - 5] == 3000]
    )
    sequence = np.stack([x[i - 5 : i + 1] for i in sequence_ids])
    seq_train = np.flatnonzero(
        np.isin(sequence_ids, train)
        & (ends[sequence_ids] - 3000 >= timestamp("2020-02-01"))
    )
    seq_valid = np.flatnonzero(
        np.isin(sequence_ids, valid)
        & (ends[sequence_ids] - 3000 >= timestamp("2020-02-21"))
    )
    seq_cal = np.flatnonzero(
        np.isin(sequence_ids, calibration)
        & (ends[sequence_ids] - 3000 >= timestamp("2020-03-01"))
    )
    seq_test = np.flatnonzero(
        np.isin(sequence_ids, test)
        & (ends[sequence_ids] - 3000 >= timestamp("2020-03-15"))
    )
    for seed in (42, 43, 44):
        forest = IsolationForest(
            n_estimators=300, max_samples=256, random_state=seed, n_jobs=2
        ).fit(x[train])
        score = -forest.decision_function(x)
        threshold = calibrated_threshold(score[calibration])
        report["models"][f"iforest_{seed}"] = {
            "threshold": threshold,
            **event_report(score[test], ends[test], threshold),
        }
        scores[f"iforest_{seed}"] = score
        model, mean, std, score, _, history = fit_autoencoder(
            sequence, seq_train, seq_valid, "dense", seed, 60
        )
        threshold = calibrated_threshold(score[seq_cal])
        report["models"][f"sequence_ae_{seed}"] = {
            "threshold": threshold,
            "history": history,
            **event_report(score[seq_test], ends[sequence_ids[seq_test]], threshold),
        }
        scores[f"sequence_ae_{seed}"] = score
        if seed == 42:
            graph = to_onnx(forest, x[:1], target_opset={"": 17, "ai.onnx.ml": 3})
            (bundle / "iforest.onnx").write_bytes(graph.SerializeToString())
            reference = -forest.decision_function(x[test[:64]])
            actual = (
                -cpu_session(bundle / "iforest.onnx")
                .run(None, {"X": x[test[:64]]})[1]
                .ravel()
            )
            np.testing.assert_allclose(actual, reference, atol=1e-5)
            deploy = SequenceScore(model, mean, std).eval()
            torch.onnx.export(
                deploy,
                torch.from_numpy(sequence[:1]),
                str(bundle / "sequence-ae.onnx"),
                input_names=["values"],
                output_names=["score"],
                dynamic_axes={"values": {0: "batch"}, "score": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
            actual = (
                cpu_session(bundle / "sequence-ae.onnx")
                .run(None, {"values": sequence[seq_test[:64]]})[0]
                .ravel()
            )
            np.testing.assert_allclose(
                actual, score[seq_test[:64]], atol=1e-4, rtol=1e-4
            )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "reported_events_detected": {
                        k: v["detected_events"] for k, v in report["models"].items()
                    },
                }
            ),
            flush=True,
        )
    np.savez_compressed(
        out / "replay.npz", x=x, end=ends, sequence_ids=sequence_ids, **scores
    )
    report["timeline"] = {
        "end": ends[test][::6].tolist(),
        "reservoir_bar": data["x"][test, 4][::6].tolist(),
        "ae_scores": scores["sequence_ae_42"][seq_test][::6].tolist(),
        "ae_end": ends[sequence_ids[seq_test]][::6].tolist(),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    for name in ("metrics.json", "protocol.json"):
        shutil.copyfile(out / name, bundle / name)
    shutil.make_archive(str(out / "model-bundle"), "zip", bundle)
    publish_run(ROOT / "outputs", "metro-study", run_id)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "models": {
                    k: {
                        n: v[n]
                        for n in ("detected_events", "alarm_events", "unmatched_events")
                    }
                    for k, v in report["models"].items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
