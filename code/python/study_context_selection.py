"""源域整工况留出选型；目标历史测试仅报告，不作为晋升依据。"""

import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

from phm_core.actuator import aggregate, load_records, motion_features, split_records
from phm_core.artifacts import capture_source, new_run, publish_run
from phm_core.conditioning import fit_reference, transform_reference

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("raw", "centered", "absolute_residual")


def evaluate(values, labels, train, test, seed):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=2,
    )
    model.fit(values[train], labels[train])
    prediction = model.predict(values[test])
    truth = labels[test]
    normal, fault = truth == 0, truth != 0
    return {
        "records": len(test),
        "normal": int(normal.sum()),
        "fault": int(fault.sum()),
        "false_positives": int((normal & (prediction != 0)).sum()),
        "false_negatives": int((fault & (prediction == 0)).sum()),
        "macro_f1": float(
            f1_score(
                truth, prediction, labels=range(4), average="macro", zero_division=0
            )
        ),
        "confusion": confusion_matrix(truth, prediction, labels=range(4)).tolist(),
    }


def run():
    run_id, out = new_run(ROOT / "outputs", "context-selection-study")
    protocol = {
        "arms": ARMS,
        "seeds": [42, 43, 44],
        "status": "exploratory on previously seen Cranfield corpus",
        "selection": "each source context held out, normal repeats1-2 commissioning only; train other contexts repeats<=6; select mean source held-context macroF1 with pooled normal FPR<=5%; if none eligible: no promotion",
        "target": "target healthy repeats1-2 reference excluded from evaluation; no target labels used to choose arm",
        "absolute_residual": "absolute deviation from commissioned healthy center; hypothesis removes fault-direction sign reversal; all positive and negative outcomes retained",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    raw, _labels, record_ids, meta = load_records(ROOT / "data/raw/cranfield-actuator")
    x = aggregate(motion_features(raw), record_ids, np.arange(len(meta)))
    y = np.array([m["label"] for m in meta])
    ctx = np.array([f"{m['profile']}/{m['load']}" for m in meta])
    repeats = np.array([m["repeat"] for m in meta])
    report = {
        "run_id": run_id,
        "protocol": protocol,
        "source_hash": capture_source(
            ROOT / "code/python", out / "train-source.zip", "study_context_selection.py"
        ),
        "protocols": {},
    }
    for name in ("held-out-load", "held-out-profile"):
        parts = split_records(meta, name)
        source = np.concatenate(
            [parts[k] for k in ("train", "validation", "calibration")]
        )
        folds = {arm: [] for arm in ARMS}
        for held in sorted(set(ctx[source])):
            train = source[(ctx[source] != held) & (repeats[source] <= 6)]
            reference = source[
                (ctx[source] == held) & (y[source] == 0) & (repeats[source] <= 2)
            ]
            test = source[(ctx[source] == held) & ~np.isin(source, reference)]
            active = np.concatenate([train, test, reference])
            ref_ids = np.concatenate([train[y[train] == 0], reference])
            fitted = fit_reference(x, ctx, y == 0, ref_ids)
            # 不变换外部目标工况，防止源选择阶段读取目标参考。
            aligned = np.zeros_like(x)
            aligned[active] = transform_reference(x[active], ctx[active], fitted)
            for arm, values in zip(ARMS, (x, aligned, np.abs(aligned))):
                for seed in protocol["seeds"]:
                    folds[arm].append(
                        {
                            "context": held,
                            "seed": seed,
                            "train_ids": train.tolist(),
                            "test_ids": test.tolist(),
                            "reference_ids": reference.tolist(),
                            "metrics": evaluate(values, y, train, test, seed),
                        }
                    )
        summaries = {
            arm: {
                "macro_f1": float(np.mean([v["metrics"]["macro_f1"] for v in rows])),
                "normal_fpr": sum(v["metrics"]["false_positives"] for v in rows)
                / sum(v["metrics"]["normal"] for v in rows),
            }
            for arm, rows in folds.items()
        }
        eligible = [a for a in ARMS if summaries[a]["normal_fpr"] <= 0.05]
        selected = (
            max(eligible, key=lambda a: summaries[a]["macro_f1"]) if eligible else None
        )
        result = {
            "source_folds": folds,
            "source_summary": summaries,
            "selected": selected,
            "target": {},
        }
        # 将源选择先落盘，再计算目标测试；失败也保留选型证据。
        (out / f"{name}-selection.json").write_text(
            json.dumps({"summary": summaries, "selected": selected}, indent=2),
            encoding="utf-8",
        )
        reference = parts["test"][
            (y[parts["test"]] == 0) & (repeats[parts["test"]] <= 2)
        ]
        test = parts["test"][~np.isin(parts["test"], reference)]
        train = parts["train"]
        refs = np.concatenate([train[y[train] == 0], reference])
        fitted = fit_reference(x, ctx, y == 0, refs)
        aligned = transform_reference(x, ctx, fitted)
        result["target_test_ids"] = test.tolist()
        for arm, values in zip(ARMS, (x, aligned, np.abs(aligned))):
            result["target"][arm] = {
                str(seed): evaluate(values, y, train, test, seed)
                for seed in protocol["seeds"]
            }
        report["protocols"][name] = result
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    bundle = out / "bundle"
    bundle.mkdir()
    for name in ("metrics.json", "protocol.json"):
        shutil.copyfile(out / name, bundle / name)
    (bundle / "README.txt").write_text(
        "Selection study reports only; no production model promoted.", encoding="utf-8"
    )
    shutil.make_archive(str(out / "model-bundle"), "zip", bundle)
    publish_run(ROOT / "outputs", "context-selection-study", run_id)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "results": {
                    n: {
                        k: v
                        for k, v in p.items()
                        if k not in ("source_folds", "target_test_ids")
                    }
                    for n, p in report["protocols"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
