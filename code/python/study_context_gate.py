"""先在源工况选择正常/故障门槛，再报告历史目标工况；不自动发布模型。"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

from phm_core.actuator import aggregate, load_records, motion_features, split_records
from phm_core.artifacts import capture_source, new_run
from phm_core.conditioning import fit_reference, transform_reference

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("raw", "centered", "absolute_residual")


def metrics(y, prediction):
    normal, fault = y == 0, y != 0
    fp = int(np.sum(normal & (prediction != 0)))
    fn = int(np.sum(fault & (prediction == 0)))
    return {
        "normal": int(normal.sum()),
        "fault": int(fault.sum()),
        "false_positives": fp,
        "false_negatives": fn,
        "normal_fpr": fp / int(normal.sum()),
        "fault_recall": 1 - fn / int(fault.sum()),
        "macro_f1": float(
            f1_score(y, prediction, labels=range(4), average="macro", zero_division=0)
        ),
        "confusion": confusion_matrix(y, prediction, labels=range(4)).tolist(),
    }


def predict(probabilities, threshold):
    """故障概率未过门槛返回正常；过门槛才选择具体故障类型。"""
    fault = np.argmax(probabilities[:, 1:], axis=1) + 1
    return np.where(1 - probabilities[:, 0] > threshold, fault, 0)


def probabilities(x, y, train, test, seed):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=2,
    )
    model.fit(x[train], y[train])
    return model.predict_proba(x[test])


def run():
    run_id, out = new_run(ROOT / "outputs", "context-gate-study")
    protocol = {
        "seeds": [42, 43, 44],
        "arms": ARMS,
        "threshold_grid": np.linspace(0, 1, 101).tolist(),
        "source_selection": "whole-source-context out-of-fold; normal repeats1-2 commissioning excluded; train other contexts repeats<=6; threshold maximizes pooled macroF1 subject to FPR<=5%; candidate also requires recall>=95% for every seed",
        "target": "historical target; healthy repeats1-2 commissioning excluded; source-only threshold unchanged",
        "scope": "supervised source calibration; probabilities not claimed calibrated; repeated seeds are not independent records; no production promotion",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    raw, _, records, meta = load_records(ROOT / "data/raw/cranfield-actuator")
    x = aggregate(motion_features(raw), records, np.arange(len(meta)))
    y = np.array([m["label"] for m in meta])
    ctx = np.array([f"{m['profile']}/{m['load']}" for m in meta])
    repeats = np.array([m["repeat"] for m in meta])
    report = {
        "run_id": run_id,
        "protocol": protocol,
        "source_hash": capture_source(
            ROOT / "code/python", out / "train-source.zip", "study_context_gate.py"
        ),
        "protocols": {},
    }
    for name in ("held-out-load", "held-out-profile"):
        parts = split_records(meta, name)
        source = np.concatenate(
            [parts[k] for k in ("train", "validation", "calibration")]
        )
        folds, outputs = [], {a: {s: [] for s in protocol["seeds"]} for a in ARMS}
        test_ids = []
        for held in sorted(set(ctx[source])):
            train = source[(ctx[source] != held) & (repeats[source] <= 6)]
            reference = source[
                (ctx[source] == held) & (y[source] == 0) & (repeats[source] <= 2)
            ]
            test = source[(ctx[source] == held) & ~np.isin(source, reference)]
            active = np.concatenate([train, test, reference])
            refs = np.concatenate([train[y[train] == 0], reference])
            fitted = fit_reference(x, ctx, y == 0, refs)
            aligned = np.zeros_like(x)
            aligned[active] = transform_reference(x[active], ctx[active], fitted)
            test_ids.extend(test.tolist())
            folds.append(
                {
                    "held_context": held,
                    "train_ids": train.tolist(),
                    "test_ids": test.tolist(),
                    "reference_ids": reference.tolist(),
                }
            )
            for arm, values in zip(ARMS, (x, aligned, np.abs(aligned))):
                for seed in protocol["seeds"]:
                    outputs[arm][seed].append(
                        probabilities(values, y, train, test, seed)
                    )
        selections = {}
        truth = y[test_ids]
        for arm in ARMS:
            selections[arm] = {}
            for seed in protocol["seeds"]:
                probs = np.concatenate(outputs[arm][seed])
                curve = [
                    {"threshold": t, **metrics(truth, predict(probs, t))}
                    for t in protocol["threshold_grid"]
                ]
                feasible = [r for r in curve if r["normal_fpr"] <= 0.05]
                selected = max(feasible, key=lambda r: (r["macro_f1"], -r["threshold"]))
                selections[arm][str(seed)] = {
                    "selected": selected,
                    "eligible": selected["fault_recall"] >= 0.95,
                    "curve": curve,
                }
                np.savez_compressed(
                    out / f"{name}-{arm}-{seed}-source.npz",
                    ids=test_ids,
                    truth=truth,
                    probabilities=probs,
                )
        # 源选择完整落盘后，目标标签才进入指标计算。
        (out / f"{name}-selection.json").write_text(
            json.dumps({"folds": folds, "arms": selections}, indent=2), encoding="utf-8"
        )
        reference = parts["test"][
            (y[parts["test"]] == 0) & (repeats[parts["test"]] <= 2)
        ]
        test = parts["test"][~np.isin(parts["test"], reference)]
        train = parts["train"]
        fitted = fit_reference(
            x, ctx, y == 0, np.concatenate([train[y[train] == 0], reference])
        )
        aligned = transform_reference(x, ctx, fitted)
        target = {}
        for arm, values in zip(ARMS, (x, aligned, np.abs(aligned))):
            target[arm] = {}
            for seed in protocol["seeds"]:
                probs = probabilities(values, y, train, test, seed)
                threshold = selections[arm][str(seed)]["selected"]["threshold"]
                target[arm][str(seed)] = {
                    "threshold": threshold,
                    "source_eligible": selections[arm][str(seed)]["eligible"],
                    "argmax": metrics(y[test], np.argmax(probs, axis=1)),
                    "gated": metrics(y[test], predict(probs, threshold)),
                }
                np.savez_compressed(
                    out / f"{name}-{arm}-{seed}-target.npz",
                    ids=test,
                    truth=y[test],
                    probabilities=probs,
                )
        report["protocols"][name] = {
            "target": target,
            "source": {
                a: {
                    s: {k: v for k, v in r.items() if k != "curve"}
                    for s, r in rows.items()
                }
                for a, rows in selections.items()
            },
            "target_reference_ids": reference.tolist(),
            "target_test_ids": test.tolist(),
        }
        print(json.dumps({"protocol": name, "target": target}), flush=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "report": str(out / "metrics.json")}))


if __name__ == "__main__":
    run()
