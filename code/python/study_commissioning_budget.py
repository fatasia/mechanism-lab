"""固定测试集比较投运健康样本量与工况方差缩放，不用目标故障调参。"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from phm_core.actuator import aggregate, load_records, motion_features, split_records
from phm_core.artifacts import capture_source, new_run
from phm_core.conditioning import transform_reference
from study_context_gate import metrics, probabilities

ROOT = Path(__file__).resolve().parents[2]
CONFIGURATIONS = {
    f"{mode}_{budget}": (mode, budget, "absolute")
    for budget in (2, 4, 6)
    for mode in ("pooled", "shrunk")
}


def commissioning_split(candidates, labels, repeats, budget):
    references = candidates[(labels[candidates] == 0) & (repeats[candidates] <= budget)]
    # 所有预算共用正常7–10和全部故障；未使用的正常1–6保留，不挪到某个配置的测试集。
    test = candidates[(labels[candidates] != 0) | (repeats[candidates] >= 7)]
    return references, test


def fit_commissioning_reference(values, contexts, healthy, references, mode):
    if not np.all(healthy[references]):
        raise ValueError("投运参考包含非健康记录")
    centers = {
        context: np.median(values[references[contexts[references] == context]], axis=0)
        for context in np.unique(contexts[references])
    }
    residuals = np.stack([values[i] - centers[contexts[i]] for i in references])
    pooled = np.maximum(1.4826 * np.median(np.abs(residuals), axis=0), 1e-5)
    result = {}
    for context, center in centers.items():
        selected = references[contexts[references] == context]
        local = 1.4826 * np.median(np.abs(values[selected] - center), axis=0)
        weight = len(selected) / (len(selected) + 4)
        scale = (
            pooled
            if mode == "pooled"
            else np.maximum(np.sqrt(weight * local**2 + (1 - weight) * pooled**2), 1e-5)
        )
        result[context] = {
            "center": center.tolist(),
            "scale": scale.tolist(),
            "reference_ids": selected.tolist(),
            "local_mad": local.tolist(),
            "pooled_mad": pooled.tolist(),
            "local_weight": weight if mode == "shrunk" else 0,
        }
    return result


def represent_residual(values, contexts, reference, representation):
    residual = transform_reference(values, contexts, reference)
    if representation == "signed":
        return residual
    result = np.abs(residual)
    if representation == "directional":
        # 误差幅值增大与电流双向变化分开；这是待检验假设，不是故障通用规律。
        amplitude = [0, 4, 8, 12, 16, 17]
        bias = [1, 5, 9, 13]
        result[:, amplitude] = np.maximum(residual[:, amplitude], 0)
        center = np.asarray([reference[c]["center"] for c in contexts])
        scale = np.asarray([reference[c]["scale"] for c in contexts])
        result[:, bias] = np.maximum(
            (np.abs(values[:, bias]) - np.abs(center[:, bias])) / scale[:, bias], 0
        )
    return result


def transformed(
    values, contexts, labels, active, references, mode, representation="absolute"
):
    reference = fit_commissioning_reference(
        values, contexts, labels == 0, references, mode
    )
    output = np.zeros_like(values)
    output[active] = represent_residual(
        values[active], contexts[active], reference, representation
    )
    return output, reference


def run(directional_study=False):
    configurations = (
        {
            f"{representation}_6": ("pooled", 6, representation)
            for representation in ("absolute", "signed", "directional")
        }
        if directional_study
        else CONFIGURATIONS
    )
    run_id, out = new_run(
        ROOT / "outputs",
        "actuator-direction-study"
        if directional_study
        else "commissioning-budget-study",
    )
    protocol = {
        "configurations": configurations,
        "seeds": [42, 43, 44],
        "features": "same20phase-aware error/current descriptors, mean5cycles perrecord,absolute commissioned residual",
        "representation": "absolute baseline; optionaldirectional study fixes6healthy references and pooledMAD: signed standardized residual vs absolute vs positive error amplitude/bias-magnitude excess with two-sided current deviations; historical-error motivated hypothesis, not causal proof",
        "scale": "pooled robustMAD vs fixed n/(n+4) localMAD-squared shrinkage toward pooledMAD-squared, numericalfloor1e-5",
        "source": "other contexts repeats<=6 train; heldcontext healthy repeats1..budget commissioning; fixedheldcontext healthy7..10 and allfault records validation",
        "target": "source training only; target healthy repeats1..budget reference; fixedtarget healthy7..10 and allfault records test; unusedhealthy1..6 never move into test",
        "selection": "all3seeds pooledsource FPR<=5%,faultrecall>=95%,4classmacroF1>=90%; max meanF1 then lower commissioning budget; selection saved before target evaluation",
        "classifier": "RF300,max_depth10,min_leaf2,class_weightbalanced",
        "scope": "commissioning sample-efficiency study on historical Cranfield; target healthy reference required; reference sizes2/4/6 do not establish industrial tail error bounds; no promotion",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    raw, _, ids, metadata = load_records(ROOT / "data/raw/cranfield-actuator")
    x = aggregate(motion_features(raw), ids, np.arange(len(metadata)))
    y = np.array([r["label"] for r in metadata])
    contexts = np.array([f"{r['profile']}/{r['load']}" for r in metadata])
    repeats = np.array([r["repeat"] for r in metadata])
    helper = ROOT / "code/python/study_context_gate.py"
    shutil.copyfile(helper, out / helper.name)
    report = {
        "run_id": run_id,
        "protocol": protocol,
        "classes": ["normal", "backlash", "lack_lubrication", "spalling"],
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", "study_commissioning_budget.py"
        ),
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "data_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "data/raw/cranfield-actuator").glob("*.mat"))
        },
        "records": metadata,
        "protocols": {},
    }
    for name in ("held-out-load", "held-out-profile"):
        task = out / name
        task.mkdir()
        parts = split_records(metadata, name)
        source = np.concatenate(
            [parts[k] for k in ("train", "validation", "calibration")]
        )
        result = {"source_folds": {}, "source_summary": {}, "target": {}}
        for arm, (mode, budget, representation) in configurations.items():
            collected = {str(seed): [[], []] for seed in protocol["seeds"]}
            result["source_folds"][arm] = []
            for fold, held in enumerate(sorted(set(contexts[source]))):
                train = source[(contexts[source] != held) & (repeats[source] <= 6)]
                commissioning, test = commissioning_split(
                    source[contexts[source] == held], y, repeats, budget
                )
                references = np.concatenate([train[y[train] == 0], commissioning])
                active = np.concatenate([train, test, commissioning])
                values, reference = transformed(
                    x, contexts, y, active, references, mode, representation
                )
                result["source_folds"][arm].append(
                    {
                        "context": held,
                        "train_ids": train.tolist(),
                        "test_ids": test.tolist(),
                        "reference_ids": references.tolist(),
                        "reference": reference,
                    }
                )
                for seed in protocol["seeds"]:
                    probability = probabilities(values, y, train, test, seed)
                    truth, predicted = collected[str(seed)]
                    truth.extend(y[test])
                    predicted.extend(probability.argmax(1))
                    np.savez_compressed(
                        task / f"source-{arm}-{seed}-{fold}.npz",
                        record_ids=test,
                        probabilities=probability,
                    )
            result["source_summary"][arm] = {
                seed: metrics(np.asarray(truth), np.asarray(predicted))
                for seed, (truth, predicted) in collected.items()
            }
            print(name, arm, "source complete", flush=True)
        eligible = [
            arm
            for arm, seeds in result["source_summary"].items()
            if all(
                m["normal_fpr"] <= 0.05
                and m["fault_recall"] >= 0.95
                and m["macro_f1"] >= 0.9
                for m in seeds.values()
            )
        ]
        result["selected"] = (
            max(
                eligible,
                key=lambda arm: (
                    np.mean(
                        [m["macro_f1"] for m in result["source_summary"][arm].values()]
                    ),
                    -configurations[arm][1],
                ),
            )
            if eligible
            else None
        )
        (task / "source-selection.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        train = parts["train"]
        result["target_references"] = {}
        for arm, (mode, budget, representation) in configurations.items():
            commissioning, test = commissioning_split(parts["test"], y, repeats, budget)
            references = np.concatenate([train[y[train] == 0], commissioning])
            values, reference = transformed(
                x,
                contexts,
                y,
                np.concatenate([train, test, commissioning]),
                references,
                mode,
                representation,
            )
            result["target_test_ids"] = test.tolist()
            result["target_references"][arm] = {
                "ids": commissioning.tolist(),
                "reference": reference,
            }
            result["target"][arm] = {}
            for seed in protocol["seeds"]:
                probability = probabilities(values, y, train, test, seed)
                result["target"][arm][str(seed)] = {
                    "records": len(test),
                    "supported_records": len(test),
                    **metrics(y[test], probability.argmax(1)),
                }
                np.savez_compressed(
                    task / f"target-{arm}-{seed}.npz",
                    record_ids=test,
                    probabilities=probability,
                )
        report["protocols"][name] = result
    (out / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "selected": {
                    name: p["selected"] for name, p in report["protocols"].items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directional-study", action="store_true")
    run(parser.parse_args().directional_study)
