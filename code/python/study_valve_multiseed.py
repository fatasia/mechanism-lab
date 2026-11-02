"""阀门局部残差/机理融合的三种子重训；历史语料探索，不更新在线模型。"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from phm_core.artifacts import capture_source, new_run
from phm_core.data import block_split
from phm_core.evaluation import evaluate
from phm_core.learning import fit_autoencoder
from phm_core.profiles import load_profile
from phm_core.valve import fit_mechanism_reference, mechanism_features, mechanism_score
from study_valve_fusion import conjunction_score
from study_valve_residual import residual_scores

ROOT = Path(__file__).resolve().parents[2]


def report_metric(score, healthy, ids, threshold):
    result = evaluate(score, healthy, ids, threshold)
    tn, fp = result["confusion"][0]
    fn, tp = result["confusion"][1]

    def f1(n):
        return 2 * n / (2 * n + fp + fn) if 2 * n + fp + fn else 0

    return {
        "normal": tn + fp,
        "fault": fn + tp,
        "false_positives": fp,
        "false_negatives": fn,
        "macro_f1": (f1(tn) + f1(tp)) / 2,
        "records": len(ids),
        "supported_records": len(ids),
        "threshold": threshold,
        "detail": result,
    }


def run(baseline_run_id="20261031T052816Z-a59e08dc"):
    run_id, out = new_run(ROOT / "outputs", "valve-multiseed-study")
    protocol = {
        "reference_baseline": baseline_run_id,
        "seeds": [42, 43, 44],
        "epochs_max": 80,
        "training": "same DenseAE, Huber loss, AdamW lr0.001, batch32, early stopping6; healthy training/validation only",
        "arms": [
            "mean_ae",
            "top10_ae",
            "local200ms_ae",
            "point_scaled_top10_ae",
            "timing_rule",
            "mean_and_timing",
            "top10_and_timing",
            "local200ms_and_timing",
            "point_scaled_top10_and_timing",
        ],
        "selection": "per seed validation branch q99; fusion FPR<=1%, max recall then AP; independent healthy calibration q99 for historical test",
        "scope": "same historical UCI block split, already inspected; training variability only, not independent-machine or new blind evidence",
        "fixed_candidate": "top10_and_timing retained as predeclared comparison, not selected by test",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    helper_hashes = {}
    for name in ("study_valve_residual.py", "study_valve_fusion.py"):
        path = ROOT / "code/python" / name
        shutil.copyfile(path, out / name)
        helper_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    x, labels, digest = load_profile(
        ROOT / "data/raw/uci-hydraulic/dataset.zip",
        "valve-switch-100hz",
        ROOT / "data/prepared",
    )
    split = block_split(len(x))
    healthy = labels[:, 1] == 100
    normal = {name: ids[healthy[ids]] for name, ids in split.items()}
    timing_features = mechanism_features(x)
    reference = fit_mechanism_reference(timing_features, normal["train"])
    timing = mechanism_score(timing_features, reference)
    report = {
        "run_id": run_id,
        "protocol": protocol,
        "classes": ["正常", "阀门故障"],
        "data_sha256": digest,
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", "study_valve_multiseed.py"
        ),
        "helper_sha256": helper_hashes,
        "split": {k: v.tolist() for k, v in split.items()},
        "mechanism_reference": reference,
        "selected_by_seed": {},
        "protocols": {
            name: {"target": {arm: {} for arm in protocol["arms"]}}
            for name in ("validation-selection", "historical-test")
        },
        "training": {},
    }
    for seed in protocol["seeds"]:
        model, mean, scale, mean_score, _, history = fit_autoencoder(
            x, normal["train"], normal["validation"], "dense", seed, 80
        )
        directory = out / str(seed)
        directory.mkdir()
        torch.save(model.state_dict(), directory / "ae.pt")
        z = ((x - mean) / scale).astype("float32")
        with torch.no_grad():
            reconstruction = np.concatenate(
                [model(batch).numpy() for batch in torch.from_numpy(z).split(64)]
            )
        scores, point_scale = residual_scores(
            np.abs(reconstruction - z), normal["train"]
        )
        np.testing.assert_allclose(scores["mean"], mean_score, atol=1e-6, rtol=1e-5)
        if seed == 42:
            with np.load(
                ROOT / "outputs/valve/runs" / baseline_run_id / "scores.npz"
            ) as stored:
                np.testing.assert_allclose(
                    mean_score, stored["ae"], atol=1e-5, rtol=1e-4
                )
        (directory / "preprocessing.json").write_text(
            json.dumps({"mean": mean.tolist(), "scale": scale.tolist()}),
            encoding="utf-8",
        )
        report["training"][str(seed)] = {
            "history": history,
            "weights_sha256": hashlib.sha256(
                (directory / "ae.pt").read_bytes()
            ).hexdigest(),
        }
        validation = {}
        val_timing = float(np.quantile(timing[normal["validation"]], 0.99))
        for arm, values in scores.items():
            threshold = float(np.quantile(values[normal["validation"]], 0.99))
            combined = conjunction_score(values, timing, threshold, val_timing)
            validation[arm + "_and_timing"] = report_metric(
                combined, healthy, split["validation"], 1
            )
        eligible = [
            arm
            for arm, result in validation.items()
            if result["false_positives"] / result["normal"] <= 0.01
        ]
        selected = (
            max(
                eligible,
                key=lambda arm: (
                    validation[arm]["detail"]["recall"],
                    validation[arm]["detail"]["test_pr_auc"],
                ),
            )
            if eligible
            else None
        )
        report["selected_by_seed"][str(seed)] = selected
        (directory / "selection.json").write_text(
            json.dumps({"selected": selected, "validation": validation}, indent=2),
            encoding="utf-8",
        )
        score_archive = {"timing": timing, "point_scale": point_scale, **scores}
        for part, title, calibrator in (
            ("validation", "validation-selection", "validation"),
            ("test", "historical-test", "calibration"),
        ):
            timing_threshold = float(np.quantile(timing[normal[calibrator]], 0.99))
            target = report["protocols"][title]["target"]
            target["timing_rule"][str(seed)] = report_metric(
                timing, healthy, split[part], timing_threshold
            )
            for arm, values in scores.items():
                threshold = float(np.quantile(values[normal[calibrator]], 0.99))
                combined = conjunction_score(
                    values, timing, threshold, timing_threshold
                )
                target[arm + "_ae"][str(seed)] = report_metric(
                    values, healthy, split[part], threshold
                )
                target[arm + "_and_timing"][str(seed)] = report_metric(
                    combined, healthy, split[part], 1
                )
                target[arm + "_and_timing"][str(seed)]["branch_thresholds"] = {
                    "ae": threshold,
                    "timing": timing_threshold,
                }
                target[arm + "_and_timing"][str(seed)]["severity"] = {
                    str(state): {
                        "cycles": int(np.sum(labels[split[part], 1] == state)),
                        "alerts": int(
                            np.sum(
                                (labels[split[part], 1] == state)
                                & (combined[split[part]] > 1)
                            )
                        ),
                    }
                    for state in (100, 90, 80, 73)
                }
                score_archive[f"{part}_{arm}_and_timing"] = combined
        np.savez_compressed(directory / "scores.npz", **score_archive)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "selected": selected,
                    "test": {
                        arm: {
                            k: v
                            for k, v in item[str(seed)].items()
                            if k in ("false_positives", "false_negatives", "macro_f1")
                        }
                        for arm, item in report["protocols"]["historical-test"][
                            "target"
                        ].items()
                    },
                }
            ),
            flush=True,
        )
    (out / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(run_id, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-id", default="20261031T052816Z-a59e08dc")
    run(parser.parse_args().baseline_run_id)
