"""执行器健康运动回归消融；目标健康投运参考与目标测试记录严格分开。"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from actuator_mechanism import (
    BASIS_NAMES,
    fit_motion_reference,
    motion_basis,
    motion_residual,
)
from phm_core.actuator import aggregate, load_records, motion_features, split_records
from phm_core.artifacts import capture_source, new_run
from phm_core.conditioning import fit_reference, transform_reference
from study_context_gate import metrics, probabilities

ROOT = Path(__file__).resolve().parents[2]
ARMS = (
    "raw_features",
    "commissioned_absolute",
    "mechanical_residual",
    "mechanical_standardized",
)


def representations(
    raw, basis, record_ids, contexts, labels, original, active, references
):
    if not np.all(labels[references] == 0) or not np.all(np.isin(references, active)):
        raise ValueError("投运参考必须是本分区内明确健康的记录")
    result = {"raw_features": original}
    fitted = fit_reference(original, contexts, labels == 0, references)
    centered = np.zeros_like(original)
    centered[active] = np.abs(
        transform_reference(original[active], contexts[active], fitted)
    )
    result["commissioned_absolute"] = centered
    residuals = {False: np.zeros_like(raw), True: np.zeros_like(raw)}
    saved = {}
    for context in sorted(set(contexts[active])):
        reference_ids = references[contexts[references] == context]
        selected_records = active[contexts[active] == context]
        selected_windows = np.flatnonzero(np.isin(record_ids, selected_records))
        ref_windows = np.flatnonzero(np.isin(record_ids, reference_ids))
        reference = fit_motion_reference(basis, raw[:, :, 1:], ref_windows)
        saved[context] = {"reference_ids": reference_ids.tolist(), **reference}
        for standardized, values in residuals.items():
            values[selected_windows, :, 0] = raw[selected_windows, :, 0]
            values[selected_windows, :, 1:] = motion_residual(
                basis[selected_windows],
                raw[selected_windows, :, 1:],
                reference,
                standardized,
            )
    selected_windows = np.flatnonzero(np.isin(record_ids, active))
    for standardized, arm in (
        (False, "mechanical_residual"),
        (True, "mechanical_standardized"),
    ):
        features = np.zeros_like(original)
        features[active] = aggregate(
            motion_features(residuals[standardized][selected_windows]),
            record_ids[selected_windows],
            active,
        )
        result[arm] = features
    return result, saved


def qualifies(result):
    return (
        result["normal_fpr"] <= 0.05
        and result["fault_recall"] >= 0.95
        and result["macro_f1"] >= 0.9
    )


def run():
    run_id, out = new_run(ROOT / "outputs", "actuator-mechanism-study")
    protocol = {
        "arms": ARMS,
        "seeds": [42, 43, 44],
        "reference": "per context healthy current/error ridge regression with fixedalpha0.001 on command position,SG11/poly3 velocity/acceleration,direction threshold0.05mm/s,quadratic velocity; not identified physical constants",
        "phase_features": "same20 motion descriptors from current/error residual; standardized arm divides by healthy reference residualRMS floor1e-6 numerical units",
        "source": "leave complete source context out; other contexts repeats<=6 training; heldcontext healthy repeats1-2 commissioning only; remaining held records validation",
        "target": "source training records only; target healthy repeats1-2 commissioning reference excluded from test in all arms",
        "selection": "all3seeds pooled source FPR<=5%,fault recall>=95%,4class macroF1>=90%; highest mean source macroF1,otherwise null; source selection saved before target scoring",
        "classifier": "sameRF300,max_depth10,min_leaf2,class_weightbalanced,seeds42/43/44",
        "scope": "historical Cranfield data after earlier failed transfer studies; percontext healthy commissioning required; no online promotion and no source-only zero-shot claim",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    raw, _, record_ids, metadata = load_records(ROOT / "data/raw/cranfield-actuator")
    basis = motion_basis(raw[:, :, 0])
    original = aggregate(motion_features(raw), record_ids, np.arange(len(metadata)))
    y = np.array([r["label"] for r in metadata])
    contexts = np.array([f"{r['profile']}/{r['load']}" for r in metadata])
    repeats = np.array([r["repeat"] for r in metadata])
    report = {
        "run_id": run_id,
        "protocol": protocol,
        "classes": ["normal", "backlash", "lack_lubrication", "spalling"],
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", "study_actuator_mechanism.py"
        ),
        "helper_sha256": {},
        "basis_names": BASIS_NAMES,
        "records": metadata,
        "protocols": {},
        "data_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "data/raw/cranfield-actuator").glob("*.mat"))
        },
    }
    for name in ("actuator_mechanism.py", "study_context_gate.py"):
        helper = ROOT / "code/python" / name
        shutil.copyfile(helper, out / name)
        report["helper_sha256"][name] = hashlib.sha256(helper.read_bytes()).hexdigest()
    for name in ("held-out-load", "held-out-profile"):
        task = out / name
        task.mkdir()
        parts = split_records(metadata, name)
        source = np.concatenate(
            [parts[k] for k in ("train", "validation", "calibration")]
        )
        collected = {
            arm: {str(seed): [[], []] for seed in protocol["seeds"]} for arm in ARMS
        }
        result = {"source_folds": [], "source_summary": {}, "target": {}}
        for fold, held in enumerate(sorted(set(contexts[source]))):
            train = source[(contexts[source] != held) & (repeats[source] <= 6)]
            commissioning = source[
                (contexts[source] == held) & (y[source] == 0) & (repeats[source] <= 2)
            ]
            test = source[(contexts[source] == held) & ~np.isin(source, commissioning)]
            references = np.concatenate([train[y[train] == 0], commissioning])
            active = np.concatenate([train, test, commissioning])
            representations_by_arm, fitted = representations(
                raw, basis, record_ids, contexts, y, original, active, references
            )
            result["source_folds"].append(
                {
                    "context": held,
                    "train_ids": train.tolist(),
                    "test_ids": test.tolist(),
                    "reference_ids": references.tolist(),
                    "references": fitted,
                }
            )
            for arm, values in representations_by_arm.items():
                for seed in protocol["seeds"]:
                    probability = probabilities(values, y, train, test, seed)
                    truth, prediction = collected[arm][str(seed)]
                    truth.extend(y[test])
                    prediction.extend(probability.argmax(1))
                    np.savez_compressed(
                        task / f"source-{fold}-{arm}-{seed}.npz",
                        record_ids=test,
                        probabilities=probability,
                    )
            print(name, held, "source fold complete", flush=True)
        for arm in ARMS:
            result["source_summary"][arm] = {
                seed: metrics(np.asarray(truth), np.asarray(predicted))
                for seed, (truth, predicted) in collected[arm].items()
            }
        eligible = [
            arm
            for arm in ARMS
            if all(qualifies(m) for m in result["source_summary"][arm].values())
        ]
        result["selected"] = (
            max(
                eligible,
                key=lambda arm: np.mean(
                    [m["macro_f1"] for m in result["source_summary"][arm].values()]
                ),
            )
            if eligible
            else None
        )
        (task / "source-selection.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        train = parts["train"]
        commissioning = parts["test"][
            (y[parts["test"]] == 0) & (repeats[parts["test"]] <= 2)
        ]
        test = parts["test"][~np.isin(parts["test"], commissioning)]
        references = np.concatenate([train[y[train] == 0], commissioning])
        active = np.concatenate([train, test, commissioning])
        representations_by_arm, fitted = representations(
            raw, basis, record_ids, contexts, y, original, active, references
        )
        result.update(
            target_test_ids=test.tolist(),
            target_reference_ids=commissioning.tolist(),
            target_references=fitted,
        )
        for arm, values in representations_by_arm.items():
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
    run()
