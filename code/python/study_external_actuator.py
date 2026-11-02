"""新公开设备留出基准：训练仅正常，不把设备身份当作已泛化故障类别。"""

import io
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.ensemble import IsolationForest

from phm_core.artifacts import capture_source, new_run, publish_run
from phm_core.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[2]
CHANNELS = ("Accel_1", "Accel_2", "Accel_3", "Angle", "PG_1", "PG_2", "PG_3")


def run():
    run_id, out = new_run(ROOT / "outputs", "external-actuator-study")
    protocol = {
        "status": "new dataset protocol frozen before model evaluation; no tuning after test",
        "features": "mean,std,p95-p05,difference_RMS per 7 raw channels; source counts, physical conversion unknown",
        "excluded_columns": [
            "F_Names",
            "F_Values",
            "Prediction",
            "Time",
            "device identity",
        ],
        "split": "healthy devices1/2/5/6 leave one out; other healthy per-device sequence first60%train,next20%validation,last20%calibration; held-out healthy and all gear/seal records test",
        "models": "train-normal median/MAD maximum and IsolationForest300,max_samples256,seeds42/43/44",
        "threshold": "independent source-healthy calibration q99; no target labels or reference; all arms reported, no promotion",
        "limitations": "gear and seal each occur on one distinct device; fault-vs-identity confounded; folds reuse fault records, not independent replication; no physical frequency or early-warning claims",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    source = json.loads((ROOT / "data/raw/vtnsi-actuator/source.json").read_text())
    archive = ROOT / "data/raw/vtnsi-actuator" / f"{source['commit']}.zip"
    features, metadata = [], []
    with ZipFile(archive) as z:
        files = [n for n in z.namelist() if "/Data/" in n and n.endswith(".txt")]
        for name in sorted(files):
            text = z.read(name).decode("utf-8-sig")
            headers = text.splitlines()[0].split("\t")
            x = np.loadtxt(
                io.StringIO(text),
                delimiter="\t",
                skiprows=1,
                usecols=[headers.index(c) for c in CHANNELS],
            )
            if len(x) < 2 or not np.isfinite(x).all():
                raise ValueError(f"非有限/不足记录: {name}")
            features.append(
                np.concatenate(
                    [
                        x.mean(0),
                        x.std(0),
                        np.quantile(x, 0.95, axis=0) - np.quantile(x, 0.05, axis=0),
                        np.sqrt(np.mean(np.diff(x, axis=0) ** 2, axis=0)),
                    ]
                )
            )
            relative = name.split("/", 1)[1]
            device = relative.split("/")[1]
            metadata.append(
                {
                    "path": relative,
                    "device": device,
                    "healthy": device in ("Act_1", "Act_2", "Act_5", "Act_6"),
                    "sequence": int(Path(name).stem.split("_")[2]),
                    "rows": len(x),
                }
            )
    x = np.asarray(features)
    healthy = np.array([m["healthy"] for m in metadata])
    report = {
        "run_id": run_id,
        "protocol": protocol,
        "source": source,
        "source_hash": capture_source(
            ROOT / "code/python", out / "train-source.zip", "study_external_actuator.py"
        ),
        "records": metadata,
        "folds": {},
    }
    saved = {}
    for device in ("Act_1", "Act_2", "Act_5", "Act_6"):
        parts = {name: [] for name in ("train", "validation", "calibration", "test")}
        for other in ("Act_1", "Act_2", "Act_5", "Act_6"):
            ids = sorted(
                [i for i, m in enumerate(metadata) if m["device"] == other],
                key=lambda i: metadata[i]["sequence"],
            )
            if other == device:
                parts["test"].extend(ids)
            else:
                a, b = int(0.6 * len(ids)), int(0.8 * len(ids))
                for p, subset in zip(
                    ("train", "validation", "calibration"), (ids[:a], ids[a:b], ids[b:])
                ):
                    parts[p].extend(subset)
        parts["test"].extend(np.flatnonzero(~healthy).tolist())
        ids = {p: np.asarray(v, dtype=int) for p, v in parts.items()}
        center = np.median(x[ids["train"]], axis=0)
        scale = np.maximum(
            1.4826 * np.median(np.abs(x[ids["train"]] - center), axis=0), 1e-6
        )
        scores = {"robust": np.abs((x - center) / scale).max(1)}
        for seed in (42, 43, 44):
            model = IsolationForest(
                n_estimators=300, max_samples=256, random_state=seed, n_jobs=2
            ).fit(x[ids["train"]])
            scores[f"iforest_{seed}"] = -model.score_samples(x)
        results = {}
        for name, score in scores.items():
            threshold = float(np.quantile(score[ids["calibration"]], 0.99))
            results[name] = {
                "threshold": threshold,
                "test": evaluate(score, healthy, ids["test"], threshold),
            }
            saved[f"{device}_{name}"] = score
        report["folds"][device] = {"partitions": parts, "models": results}
    np.savez_compressed(out / "scores.npz", features=x, healthy=healthy, **saved)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    bundle = out / "bundle"
    bundle.mkdir()
    for file in ("metrics.json", "protocol.json"):
        shutil.copyfile(out / file, bundle / file)
    shutil.copyfile(
        ROOT / "data/raw/vtnsi-actuator/LICENSE", bundle / "DATA-REPOSITORY-LICENSE.txt"
    )
    (bundle / "README.txt").write_text(
        "Evaluation reports only. No deployable model or raw data included.",
        encoding="utf-8",
    )
    shutil.make_archive(str(out / "model-bundle"), "zip", bundle)
    publish_run(ROOT / "outputs", "external-actuator-study", run_id)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "folds": {k: v["models"] for k, v in report["folds"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
