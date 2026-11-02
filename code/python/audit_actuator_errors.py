"""按完整记录、工况和原始故障文件核对执行器错误；只诊断，不训练或重选模型。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from phm_core.actuator import CLASSES, load_records
from phm_core.artifacts import capture_source, new_run

ROOT = Path(__file__).resolve().parents[2]


def grouped_counts(truth, predicted, groups):
    """故障类别错分与漏报分开统计，不把识别成另一故障类别算成未检出。"""
    result = {}
    for group in sorted(set(groups)):
        selected = groups == group
        y, prediction = truth[selected], predicted[selected]
        confusion = np.zeros((4, 4), dtype=int)
        np.add.at(confusion, (y, prediction), 1)
        result[group] = {
            "records": len(y),
            "normal": int(np.sum(y == 0)),
            "fault": int(np.sum(y != 0)),
            "false_positives": int(np.sum((y == 0) & (prediction != 0))),
            "false_negatives": int(np.sum((y != 0) & (prediction == 0))),
            "wrong_fault_class": int(
                np.sum((y != 0) & (prediction != 0) & (y != prediction))
            ),
            "confusion": confusion.tolist(),
        }
    return result


def stable_errors(truth, predictions):
    """同一批记录的种子一致性，不累计种子作为更多独立样本。"""
    return {
        "all_seeds_false_positive": ((truth == 0) & np.all(predictions != 0, axis=0)),
        "all_seeds_false_negative": ((truth != 0) & np.all(predictions == 0, axis=0)),
        "any_seed_wrong": np.any(predictions != truth, axis=0),
        "all_seeds_wrong": np.all(predictions != truth, axis=0),
    }


def audit(studies):
    _, _, _, metadata = load_records(ROOT / "data/raw/cranfield-actuator")
    run_id, output = new_run(ROOT / "outputs", "actuator-error-audit")
    report = {
        "run_id": run_id,
        "classes": CLASSES,
        "scope": "post-hoc diagnosis of saved historical target predictions; no model fitting, threshold changes, or promotion; filenames are experiment identifiers, not assumed severity order",
        "source_hash": capture_source(
            ROOT / "code/python", output / "source.zip", "audit_actuator_errors.py"
        ),
        "data_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "data/raw/cranfield-actuator").glob("*.mat"))
        },
        "studies": {},
    }
    rows = []
    for directory in studies:
        path = directory / "metrics.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        item = {
            "metrics_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "protocols": {},
        }
        for protocol, study in original["protocols"].items():
            arms = {}
            for arm, seeds in study["target"].items():
                by_seed, prediction_rows, probability_rows = {}, [], []
                ids = None
                for seed in sorted(seeds, key=int):
                    predictions_path = directory / protocol / f"target-{arm}-{seed}.npz"
                    with np.load(predictions_path) as data:
                        current_ids, probability = (
                            data["record_ids"],
                            data["probabilities"],
                        )
                    if ids is None:
                        ids = current_ids
                    np.testing.assert_array_equal(ids, current_ids)
                    assert probability.shape == (len(ids), 4)
                    assert np.isfinite(probability).all()
                    np.testing.assert_allclose(probability.sum(1), 1, atol=1e-6)
                    prediction = probability.argmax(1)
                    truth = np.array([metadata[i]["label"] for i in ids])
                    contexts = np.array(
                        [f"{metadata[i]['profile']}/{metadata[i]['load']}" for i in ids]
                    )
                    files = np.array([metadata[i]["file"] for i in ids])
                    # 重新计算总数，与原报告核对后再生成细分表。
                    counts = grouped_counts(
                        truth, prediction, np.repeat("all", len(ids))
                    )["all"]
                    for key in (
                        "normal",
                        "fault",
                        "false_positives",
                        "false_negatives",
                    ):
                        assert counts[key] == seeds[seed][key], (
                            protocol,
                            arm,
                            seed,
                            key,
                        )
                    by_seed[seed] = {
                        "prediction_sha256": hashlib.sha256(
                            predictions_path.read_bytes()
                        ).hexdigest(),
                        "by_context": grouped_counts(truth, prediction, contexts),
                        "by_file": grouped_counts(truth, prediction, files),
                    }
                    prediction_rows.append(prediction)
                    probability_rows.append(probability[:, 0])
                stable = stable_errors(truth, np.stack(prediction_rows))
                arms[arm] = {
                    "seeds": by_seed,
                    "stable_record_ids": {
                        key: ids[mask].tolist() for key, mask in stable.items()
                    },
                }
                for index, record_id in enumerate(ids):
                    row = {
                        "study": original["run_id"],
                        "protocol": protocol,
                        "arm": arm,
                        **metadata[record_id],
                    }
                    for seed, predicted, normal_probability in zip(
                        sorted(seeds, key=int), prediction_rows, probability_rows
                    ):
                        row[f"prediction_{seed}"] = int(predicted[index])
                        row[f"normal_probability_{seed}"] = float(
                            normal_probability[index]
                        )
                    rows.append(row)
            item["protocols"][protocol] = arms
        report["studies"][original["run_id"]] = item
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (output / "records.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {"run_id": run_id, "studies": len(studies), "comparison_rows": len(rows)}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("studies", type=Path, nargs="+")
    audit(parser.parse_args().studies)
