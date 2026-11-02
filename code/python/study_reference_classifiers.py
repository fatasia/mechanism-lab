"""隔离环境中的原版 MiniRocket/MultiRocket 强基线，生产环境不被降级。"""

import json
import shutil
import time
from pathlib import Path

import aeon
import joblib
import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from phm_core.actuator import aggregate, load_records, split_records
from phm_core.artifacts import capture_source, new_run, publish_run

ROOT = Path(__file__).resolve().parents[2]


def metric(scores, labels, records, ids):
    prediction = aggregate(scores, records, ids).argmax(1)
    truth = np.array([labels[records == i][0] for i in ids])
    return {
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "records": len(ids),
        "errors": [int(i) for i, t, p in zip(ids, truth, prediction) if t != p],
    }


def run():
    x, y, records, metadata = load_records(ROOT / "data/raw/cranfield-actuator")
    run_id, out = new_run(ROOT / "outputs", "reference-study")
    protocol = {
        "aeon": aeon.__version__,
        "models": {
            "MiniRocket": {"n_kernels": 10000},
            "MultiRocket": {"n_kernels": 6250, "normalise": False},
        },
        "ridge_alpha": 10,
        "class_weight": "balanced",
        "n_jobs": 2,
        "seeds": [42, 43, 44],
        "protocols": ["record-holdout", "held-out-load", "held-out-profile"],
        "scaling": "channel z-score fitted only on source train; transform feature scaling train only",
        "aggregation": "five cycle class scores mean per original record",
        "status": "same previously inspected corpus; exploratory external algorithm reference, not new blind test",
        "runtime": "reference environment aeon/numba, transform not exported to ONNX",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    source = capture_source(
        ROOT / "code/python", out / "train-source.zip", "study_reference_classifiers.py"
    )
    report = {
        "run_id": run_id,
        "source_hash": source,
        "protocol": protocol,
        "results": {},
    }
    for split_name in protocol["protocols"]:
        split = split_records(metadata, split_name)
        train = np.flatnonzero(np.isin(records, split["train"]))
        normalized = (
            ((x - x[train].mean((0, 1))) / x[train].std((0, 1)))
            .transpose(0, 2, 1)
            .copy()
        )
        report["results"][split_name] = {}
        for name, cls in (("MiniRocket", MiniRocket), ("MultiRocket", MultiRocket)):
            entries = []
            for seed in protocol["seeds"]:
                start = time.perf_counter()
                transform = cls(**protocol["models"][name], random_state=seed, n_jobs=2)
                transform.fit(normalized[train])
                features = np.asarray(transform.transform(normalized), dtype="float32")
                scaler = StandardScaler().fit(features[train])
                classifier = RidgeClassifier(alpha=10, class_weight="balanced").fit(
                    scaler.transform(features[train]), y[train]
                )
                scores = classifier.decision_function(scaler.transform(features))
                entry = {
                    "seed": seed,
                    "features": features.shape[1],
                    "seconds": time.perf_counter() - start,
                    "validation": metric(scores, y, records, split["validation"]),
                    "test": metric(scores, y, records, split["test"]),
                }
                entries.append(entry)
                if seed == 42:
                    # 仅保存本机拟合对象；不将 pickle 放进对外模型包或运行服务。
                    joblib.dump(
                        {
                            "transform": transform,
                            "scaler": scaler,
                            "classifier": classifier,
                        },
                        out / f"{split_name}-{name}.joblib",
                    )
                print(
                    json.dumps({"protocol": split_name, "model": name, **entry}),
                    flush=True,
                )
                del features, scores, transform, scaler, classifier
            report["results"][split_name][name] = entries
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    bundle = out / "bundle"
    bundle.mkdir()
    for name in ("metrics.json", "protocol.json"):
        shutil.copyfile(out / name, bundle / name)
    (bundle / "README.txt").write_text(
        "Reference algorithm benchmark only. No ONNX transform in this archive. Fitted pickle objects are local and are not distributed."
    )
    shutil.make_archive(str(out / "model-bundle"), "zip", bundle)
    publish_run(ROOT / "outputs", "reference-study", run_id)
    print(json.dumps({"run_id": run_id}), flush=True)


if __name__ == "__main__":
    run()
