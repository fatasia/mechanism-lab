"""电机全带宽频谱CNN对照；整工况留出，早停和训练轮数只由源域决定。"""

import argparse
import hashlib
import json
import shutil
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from phm_core.artifacts import capture_source, new_run
from study_motor import metrics, record_probabilities, source_passes

ROOT = Path(__file__).resolve().parents[2]


class SpectralNetwork(nn.Module):
    """六通道谱输入；归一化统计仅来自训练窗口，保存在模型内。"""

    def __init__(self, mean, std, classes):
        super().__init__()
        self.register_buffer(
            "mean", torch.as_tensor(mean, dtype=torch.float32).reshape(1, 6, 1)
        )
        self.register_buffer(
            "std", torch.as_tensor(std, dtype=torch.float32).reshape(1, 6, 1)
        )
        self.network = nn.Sequential(
            nn.Conv1d(6, 16, 9, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, 7, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=4, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, classes),
        )

    def forward(self, values):
        return self.network((values - self.mean) / self.std)


def predict(model, x):
    model.eval()
    if not len(x):
        return np.empty((0, model.network[-1].out_features), dtype="float32")
    with torch.no_grad():
        return np.concatenate(
            [
                model(torch.from_numpy(x[start : start + 128])).softmax(1).numpy()
                for start in range(0, len(x), 128)
            ]
        )


def train(x, labels, records, classes, seed, epochs=60, validation=None):
    torch.manual_seed(seed)
    mean = x.mean(axis=(0, 2), dtype=np.float64).astype("float32")
    std = np.maximum(x.std(axis=(0, 2), dtype=np.float64), 1e-6).astype("float32")
    model = SpectralNetwork(mean, std, classes)
    record_ids, counts = np.unique(records, return_counts=True)
    counts_by_record = dict(zip(record_ids, counts))
    weights = torch.as_tensor(
        [1 / counts_by_record[r] for r in records], dtype=torch.double
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights, num_samples=min(2048, len(x)), replacement=True, generator=generator
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.as_tensor(labels, dtype=torch.long)),
        batch_size=128,
        sampler=sampler,
    )
    record_labels = np.array(
        [labels[np.flatnonzero(records == r)[0]] for r in record_ids]
    )
    class_counts = np.bincount(record_labels, minlength=classes)
    class_weights = 1 / np.maximum(class_counts, 1)
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(
            class_weights / class_weights.mean(), dtype=torch.float32
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history, best, best_epoch, best_state, stale = [], float("inf"), 0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for values, target in loader:
            optimizer.zero_grad()
            loss = criterion(model(values), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if validation is not None:
            vx, vy, vr, selected = validation
            probabilities = record_probabilities(predict(model, vx), vr, selected)
            supported = np.isfinite(probabilities).all(axis=1)
            if not supported.any():
                raise ValueError("源验证工况没有可支持记录")
            score = float(
                -np.log(
                    np.maximum(probabilities[supported, vy[supported]], 1e-12)
                ).mean()
            )
            row["validation_record_cross_entropy"] = score
            if score < best - 1e-4:
                best, best_epoch, best_state, stale = (
                    score,
                    epoch,
                    deepcopy(model.state_dict()),
                    0,
                )
            else:
                stale += 1
        history.append(row)
        if validation is not None and stale >= 8:
            break
    if validation is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    return model, history, best_epoch


def run(feature_directory, representation="spectra"):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    path = feature_directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    arm, description = {
        "spectra": (
            "spectral_cnn",
            "6x1600 log power, full frequency range, no target normalization",
        ),
        "order_spectra": (
            "order_cnn",
            "6x512 log angular power, 0-64 orders, keyphase angular resampling; no target normalization",
        ),
        "envelope_order_spectra": (
            "envelope_order_cnn",
            "3 vibration envelope +3 current angular spectra,6x512,0-64 orders; no target normalization",
        ),
    }[representation]
    run_id, out = new_run(
        ROOT / "outputs",
        "motor-cnn-study" if representation == "spectra" else "motor-order-cnn-study",
    )
    protocol = {
        "seeds": [42, 43, 44],
        "arm": arm,
        "source_validation": "leave one complete source condition out, record-level CE early stopping",
        "target_protocols": ["held-out-3000rpm", "held-out-torque-profile"],
        "epochs": {
            "max": 60,
            "patience": 8,
            "min_delta": 0.0001,
            "final": "rounded median best epoch of source folds for same seed",
        },
        "optimizer": "AdamW lr0.001 weight_decay0.0001,batch128,up to2048 record-balanced window draws per epoch,class-balanced CE",
        "threads": 4,
        "input": description,
        "scope": "representation CNN comparison, not raw-waveform architecture-matched ablation; historical target inspected in earlier representation study; no online promotion",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    helper = ROOT / "code/python/study_motor.py"
    shutil.copyfile(helper, out / helper.name)
    records = manifest["records"]
    classes = sorted({r["label"] for r in records})
    y = np.array([classes.index(r["label"]) for r in records])
    healthy = classes.index("health")
    conditions = np.array([r["condition"] for r in records])
    packets, window_records = [], []
    for record in records:
        cache_path = feature_directory / record["cache"]
        if (
            hashlib.sha256(cache_path.read_bytes()).hexdigest()
            != record["cache_sha256"]
        ):
            raise ValueError("频谱缓存哈希不一致")
        with np.load(cache_path) as cache:
            packets.append(cache[representation])
            window_records.extend([record["id"]] * len(cache[representation]))
    x, window_records = np.concatenate(packets), np.asarray(window_records)
    del packets
    report = {
        "run_id": run_id,
        "protocol": protocol,
        "classes": classes,
        "healthy_class": healthy,
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", "study_motor_cnn.py"
        ),
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "feature_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "feature_run": manifest["run_id"],
        "protocols": {},
    }
    shutil.copyfile(path, out / "feature-manifest.json")
    start, fits = time.monotonic(), 0
    for name in protocol["target_protocols"]:
        target_mask = np.array(
            [
                r["nominal_rpm"] == 3000
                if name == "held-out-3000rpm"
                else r["profile"] == "torque"
                for r in records
            ]
        )
        source, target = np.flatnonzero(~target_mask), np.flatnonzero(target_mask)
        task = out / name
        task.mkdir()
        result = {
            "source_ids": source.tolist(),
            "target_test_ids": target.tolist(),
            "source_folds": {arm: []},
            "source_summary": {arm: {}},
            "target": {arm: {}},
        }
        chosen_epochs = {}
        for seed in protocol["seeds"]:
            epochs, truth, predictions = [], [], []
            for fold, condition in enumerate(np.unique(conditions[source])):
                training = source[conditions[source] != condition]
                validation = source[conditions[source] == condition]
                train_mask, val_mask = (
                    np.isin(window_records, training),
                    np.isin(window_records, validation),
                )
                model, history, best_epoch = train(
                    x[train_mask],
                    y[window_records[train_mask]],
                    window_records[train_mask],
                    len(classes),
                    seed,
                    validation=(
                        x[val_mask],
                        y[validation],
                        window_records[val_mask],
                        validation,
                    ),
                )
                probabilities = record_probabilities(
                    predict(model, x[val_mask]), window_records[val_mask], validation
                )
                item = metrics(y[validation], probabilities, healthy, len(classes))
                result["source_folds"][arm].append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "train_ids": training.tolist(),
                        "test_ids": validation.tolist(),
                        "best_epoch": best_epoch,
                        "history": history,
                        "metrics": item,
                    }
                )
                np.savez_compressed(
                    task / f"source-{seed}-{fold}.npz",
                    record_ids=validation,
                    labels=y[validation],
                    probabilities=probabilities,
                )
                torch.save(model.state_dict(), task / f"source-{seed}-{fold}.pt")
                epochs.append(best_epoch)
                truth.extend(y[validation])
                predictions.extend(probabilities)
                fits += 1
                (out / "progress.json").write_text(
                    json.dumps(
                        {
                            "fits": fits,
                            "protocol": name,
                            "seed": seed,
                            "fold": condition,
                            "elapsed_s": time.monotonic() - start,
                        }
                    ),
                    encoding="utf-8",
                )
                print(name, seed, condition, best_epoch, item["macro_f1"], flush=True)
            chosen_epochs[str(seed)] = max(1, round(float(np.median(epochs))))
            result["source_summary"][arm][str(seed)] = metrics(
                np.asarray(truth), np.asarray(predictions), healthy, len(classes)
            )
        result["selected"] = (
            arm
            if all(source_passes(v) for v in result["source_summary"][arm].values())
            else None
        )
        result["final_epochs"] = chosen_epochs
        (task / "source-selection.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        train_mask, test_mask = (
            np.isin(window_records, source),
            np.isin(window_records, target),
        )
        for seed in protocol["seeds"]:
            model, history, _ = train(
                x[train_mask],
                y[window_records[train_mask]],
                window_records[train_mask],
                len(classes),
                seed,
                epochs=chosen_epochs[str(seed)],
            )
            probabilities = record_probabilities(
                predict(model, x[test_mask]), window_records[test_mask], target
            )
            result["target"][arm][str(seed)] = metrics(
                y[target], probabilities, healthy, len(classes)
            )
            torch.save(model.state_dict(), task / f"target-{seed}.pt")
            (task / f"target-{seed}-history.json").write_text(
                json.dumps(history), encoding="utf-8"
            )
            np.savez_compressed(
                task / f"target-{seed}.npz",
                record_ids=target,
                labels=y[target],
                probabilities=probabilities,
            )
            fits += 1
        report["protocols"][name] = result
        (out / "progress-metrics.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    report.update(fits=fits, elapsed_s=time.monotonic() - start)
    (out / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "fits": fits,
                "selected": {n: p["selected"] for n, p in report["protocols"].items()},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument(
        "--representation",
        choices=["spectra", "order_spectra", "envelope_order_spectra"],
        default="spectra",
    )
    args = parser.parse_args()
    run(args.features.resolve(), args.representation)
