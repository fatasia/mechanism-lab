"""已查看极片数据的探索性消融/学习曲线/分组校准区间；不是新盲测。"""

import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from phm_core.artifacts import capture_source, new_run
from public_data import sha256
from study_electrode import encode

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = {
    "split": "leave process combination out, before/after same group",
    "model": "StandardScaler+Ridge(alpha=1)",
    "arms": ["process", "spectrum", "combined", "process_plus_four_spectral_moments"],
    "learning_fractions": [0.25, 0.5, 1.0],
    "seeds": [42, 43, 44],
    "interval_alpha": 0.2,
    "interval": "split conformal, calibration unit=process group, group maximum residual; insufficient groups=>unbounded",
    "ood": "source fit standardized process distance, max calibration nearest distance; exploratory, no guarantee",
    "status": "all data previously inspected; exploratory not final independent test",
}


def group_radius(errors, groups, alpha):
    scores = sorted(
        float(np.max(errors[np.asarray(groups) == g])) for g in sorted(set(groups))
    )
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    return None if k > len(scores) else scores[k - 1]


def model():
    return make_pipeline(StandardScaler(), Ridge(alpha=1))


def run():
    run_id, out = new_run(ROOT / "outputs", "electrode-uncertainty")
    (out / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2))
    source = ROOT / "data/raw/electrode-ultrasound-v4"
    records = []
    for entry in json.loads((source / "source.json").read_text())["files"]:
        path = source / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("极片输入变更")
        records.append(json.loads(path.read_text()))
    results = []
    for material in ("GA", "NMC622"):
        rows = [
            r for r in records if r["metadata"]["Sample_ID"].startswith(material + "_")
        ]
        x = np.array([a + b for a, b in map(encode, rows)])
        y = np.array([r["metadata"]["Thickness"] for r in rows])
        groups = np.array([json.dumps(encode(r)[0][:-1]) for r in rows])
        all_groups = sorted(set(groups))
        arms = {
            "process": slice(0, 5),
            "spectrum": slice(5, None),
            "combined": slice(None),
            "process_plus_four_spectral_moments": slice(0, 9),
        }
        ablations = {k: [] for k in arms}
        curves = {
            str((f, s)): []
            for f in PROTOCOL["learning_fractions"]
            for s in PROTOCOL["seeds"]
        }
        intervals = []
        for target in all_groups:
            test = groups == target
            train = ~test
            source_groups = [g for g in all_groups if g != target]
            for arm, columns in arms.items():
                predictor = model().fit(x[train, columns], y[train])
                ablations[arm].append(
                    float(np.mean(abs(predictor.predict(x[test, columns]) - y[test])))
                )
            for seed in PROTOCOL["seeds"]:
                ordered = np.random.default_rng(seed).permutation(source_groups)
                for fraction in PROTOCOL["learning_fractions"]:
                    selected = ordered[: max(1, math.ceil(len(ordered) * fraction))]
                    fit = np.isin(groups, selected)
                    predicted = model().fit(x[fit], y[fit]).predict(x[test])
                    curves[str((fraction, seed))].append(
                        float(np.mean(abs(predicted - y[test])))
                    )
            ordered = np.random.default_rng(42).permutation(source_groups)
            cut = max(1, int(len(ordered) * 0.67))
            fit = np.isin(groups, ordered[:cut])
            cal = np.isin(groups, ordered[cut:])
            predictor = model().fit(x[fit], y[fit])
            radius = group_radius(
                abs(predictor.predict(x[cal]) - y[cal]), groups[cal], 0.2
            )
            error = abs(predictor.predict(x[test]) - y[test])
            scaler = StandardScaler().fit(x[fit, :4])
            z = scaler.transform(x[:, :4])

            def distance(mask):
                return np.sqrt(
                    ((z[mask, None, :] - z[fit][None, :, :]) ** 2).sum(axis=2)
                ).min(axis=1)

            threshold = float(distance(cal).max())
            accepted = distance(test) <= threshold
            intervals.append(
                {
                    "held_group_index": all_groups.index(target),
                    "fit_groups": cut,
                    "calibration_groups": len(ordered) - cut,
                    "radius_um": radius,
                    "bounded": radius is not None,
                    "group_covered": True
                    if radius is None
                    else bool(np.all(error <= radius)),
                    "mae_um": float(error.mean()),
                    "accepted": int(accepted.sum()),
                    "records": int(test.sum()),
                    "accepted_absolute_error_sum": float(error[accepted].sum()),
                    "ood_threshold": threshold,
                }
            )
        accepted = sum(r["accepted"] for r in intervals)
        results.append(
            {
                "material": material,
                "groups": len(all_groups),
                "ablation_group_equal_mae_um": {
                    k: float(np.mean(v)) for k, v in ablations.items()
                },
                "learning_curve_group_equal_mae_um": {
                    k: float(np.mean(v)) for k, v in curves.items()
                },
                "interval_folds": intervals,
                "bounded_intervals": sum(r["bounded"] for r in intervals),
                "group_coverage": float(
                    np.mean([r["group_covered"] for r in intervals])
                ),
                "ood_record_coverage": accepted / len(y),
                "accepted_mae_um": sum(
                    r["accepted_absolute_error_sum"] for r in intervals
                )
                / accepted
                if accepted
                else None,
            }
        )
    report = {
        "run_id": run_id,
        "protocol": PROTOCOL,
        "results": results,
        "source_manifest_sha256": sha256(source / "source.json"),
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", Path(__file__).name
        ),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(out), "results": results}), flush=True)


if __name__ == "__main__":
    run()
