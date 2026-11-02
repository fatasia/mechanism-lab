"""源工况类条件矩匹配与同结构ERM对照；不读目标故障样本作训练或选型。"""

import hashlib
import json
import math
import shutil
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn

from phm_core.actuator import aggregate, load_records, split_records
from phm_core.artifacts import capture_source, new_run
from phm_core.conditioning import fit_reference, transform_reference
from study_phase_alignment import phase_features

ROOT = Path(__file__).resolve().parents[2]
ARMS = {"phase_erm": 0.0, "phase_conditional_alignment": 0.1}


class PhaseNetwork(nn.Module):
    """三阶段各两通道，32个阶段内位置；不输入负载标签或故障标签。"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(6, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )
        self.head = nn.Linear(64, 4)

    def forward(self, values):
        embedding = self.encoder(values.reshape(-1, 6, 32))
        return self.head(embedding), embedding


def alignment_loss(embedding, labels, domains):
    """仅匹配同类源工况的均值/协方差，避免按不同类别比例对齐。非原论文原样复现。"""
    losses = []
    for label in torch.unique(labels):
        groups = []
        for domain in torch.unique(domains):
            z = embedding[(labels == label) & (domains == domain)]
            if len(z) >= 2:
                mean = z.mean(0)
                centered = z - mean
                groups.append((mean, centered.T @ centered / (len(z) - 1)))
        for first in range(len(groups)):
            for second in range(first + 1, len(groups)):
                a, b = groups[first], groups[second]
                losses.append(
                    (a[0] - b[0]).square().mean() + 0.25 * (a[1] - b[1]).square().mean()
                )
    return torch.stack(losses).mean() if losses else embedding.sum() * 0


def fit(values, labels, domains, train, validation, test, seed, weight, epochs=100):
    if (
        set(train) & set(validation)
        or set(train) & set(test)
        or set(validation) & set(test)
    ):
        raise ValueError("训练、早停验证与评估记录必须隔离")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PhaseNetwork()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    x = torch.from_numpy(values)
    y = torch.from_numpy(labels).long()
    d = torch.from_numpy(domains).long()
    cells = [
        train[(labels[train] == label) & (domains[train] == domain)]
        for label in np.unique(labels[train])
        for domain in np.unique(domains[train])
    ]
    cells = [cell for cell in cells if len(cell)]
    steps = math.ceil(len(train) / (2 * len(cells)))
    best, stale, checkpoint = float("inf"), 0, None
    history = []
    for epoch in range(epochs):
        model.train()
        objectives = []
        for _ in range(steps):
            ids = np.concatenate(
                [rng.choice(cell, 2, replace=len(cell) < 2) for cell in cells]
            )
            optimizer.zero_grad()
            logits, z = model(x[ids])
            ce = nn.functional.cross_entropy(logits, y[ids])
            penalty = alignment_loss(z, y[ids], d[ids]) if weight else z.sum() * 0
            loss = ce + weight * penalty
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            objectives.append([loss.item(), ce.item(), penalty.item()])
        model.eval()
        with torch.no_grad():
            logits, _ = model(x[validation])
            losses = nn.functional.cross_entropy(
                logits, y[validation], reduction="none"
            )
            valid = (
                torch.stack(
                    [
                        losses[y[validation] == label].mean()
                        for label in torch.unique(y[validation])
                    ]
                )
                .mean()
                .item()
            )
        history.append(
            {
                "epoch": epoch + 1,
                "objective": np.mean(objectives, axis=0).tolist(),
                "validation_balanced_ce": valid,
            }
        )
        if valid < best:
            best, stale, checkpoint = valid, 0, deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        probs = model(x[test])[0].softmax(1).numpy()
    return probs, history, checkpoint


def metrics(truth, probs):
    prediction = probs.argmax(1)
    normal, fault = truth == 0, truth != 0
    return {
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
    torch.set_num_threads(2)
    run_id, out = new_run(ROOT / "outputs", "phase-coral-study")
    protocol = {
        "arms": ARMS,
        "seeds": [42, 43, 44],
        "epochs_max": 100,
        "patience": 10,
        "input": "phase32 signed error/current; per-context healthy centering and pooled MAD; clip to[-20,20]; no phase duration channels",
        "sampling": "2 records per source context/class cell each batch; identical sampler and architecture in both arms",
        "early_stop": "other source contexts repeats7-8, balanced CE; never held source-context evaluation labels",
        "selection": "whole source-context leave-out; each seed pooled FPR<=5% and recall>=95%; rank mean macroF1 only among eligible arms",
        "target": "healthy repeats1-2 commissioning excluded; historical target labels not used for selection",
        "references": [
            "https://arxiv.org/abs/1607.01719",
            "https://openreview.net/forum?id=lQdXeXDoWtI",
        ],
        "scope": "source-only class-conditional moment alignment, not exact Deep CORAL reproduction; historical same-rig corpus",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    helper = ROOT / "code/python/study_phase_alignment.py"
    shutil.copyfile(helper, out / helper.name)
    report = {
        "run_id": run_id,
        "protocol": protocol,
        "protocols": {},
        "source_hash": capture_source(
            ROOT / "code/python", out / "train-source.zip", "study_phase_coral.py"
        ),
        "dependency_source_sha256": {
            helper.name: hashlib.sha256(helper.read_bytes()).hexdigest()
        },
    }
    raw, _, record_ids, meta = load_records(ROOT / "data/raw/cranfield-actuator")
    x = aggregate(phase_features(raw), record_ids, np.arange(len(meta)))[:, :192]
    y = np.array([m["label"] for m in meta])
    ctx = np.array([f"{m['profile']}/{m['load']}" for m in meta])
    domains = np.unique(ctx, return_inverse=True)[1]
    repeats = np.array([m["repeat"] for m in meta])
    started = time.monotonic()
    completed = 0

    def experiment(name, arm, seed, train, validation, test, reference):
        nonlocal completed
        fitted = fit_reference(
            x, ctx, y == 0, np.concatenate([train[y[train] == 0], reference])
        )
        active = np.concatenate([train, validation, test, reference])
        values = np.zeros_like(x)
        values[active] = np.clip(
            transform_reference(x[active], ctx[active], fitted), -20, 20
        )
        probs, history, weights = fit(
            values, y, domains, train, validation, test, seed, ARMS[arm]
        )
        prefix = f"{name}-{arm}-{seed}"
        torch.save(weights, out / f"{prefix}.pt")
        np.savez_compressed(
            out / f"{prefix}.npz", ids=test, truth=y[test], probabilities=probs
        )
        (out / f"{prefix}.json").write_text(
            json.dumps(
                {
                    "train_ids": train.tolist(),
                    "validation_ids": validation.tolist(),
                    "test_ids": test.tolist(),
                    "commissioning_ids": reference.tolist(),
                    "reference": fitted,
                    "history": history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        completed += 1
        progress = {
            "run_id": run_id,
            "completed_fits": completed,
            "last": prefix,
            "epochs": len(history),
            "elapsed_seconds": time.monotonic() - started,
        }
        (out / "progress.json").write_text(
            json.dumps(progress, indent=2), encoding="utf-8"
        )
        print(json.dumps(progress), flush=True)
        return metrics(y[test], probs)

    for name in ("held-out-load", "held-out-profile"):
        parts = split_records(meta, name)
        source = np.concatenate(
            [parts[k] for k in ("train", "validation", "calibration")]
        )
        folds = {a: {str(s): [] for s in protocol["seeds"]} for a in ARMS}
        for index, held in enumerate(sorted(set(ctx[source]))):
            train = source[(ctx[source] != held) & (repeats[source] <= 6)]
            validation = source[
                (ctx[source] != held) & (repeats[source] >= 7) & (repeats[source] <= 8)
            ]
            reference = source[
                (ctx[source] == held) & (y[source] == 0) & (repeats[source] <= 2)
            ]
            test = source[(ctx[source] == held) & ~np.isin(source, reference)]
            for arm in ARMS:
                for seed in protocol["seeds"]:
                    folds[arm][str(seed)].append(
                        experiment(
                            f"{name}-source{index}",
                            arm,
                            seed,
                            train,
                            validation,
                            test,
                            reference,
                        )
                    )
        summary = {}
        for arm, seed_rows in folds.items():
            seed_metrics = {
                s: {
                    "normal_fpr": sum(r["false_positives"] for r in rows)
                    / sum(r["normal"] for r in rows),
                    "fault_recall": 1
                    - sum(r["false_negatives"] for r in rows)
                    / sum(r["fault"] for r in rows),
                    "macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
                }
                for s, rows in seed_rows.items()
            }
            summary[arm] = {
                "seeds": seed_metrics,
                "eligible": all(
                    r["normal_fpr"] <= 0.05 and r["fault_recall"] >= 0.95
                    for r in seed_metrics.values()
                ),
                "macro_f1": float(
                    np.mean([r["macro_f1"] for r in seed_metrics.values()])
                ),
            }
        eligible = [a for a in ARMS if summary[a]["eligible"]]
        selected = (
            max(eligible, key=lambda a: summary[a]["macro_f1"]) if eligible else None
        )
        (out / f"{name}-selection.json").write_text(
            json.dumps({"source": summary, "selected": selected}, indent=2),
            encoding="utf-8",
        )
        reference = parts["test"][
            (y[parts["test"]] == 0) & (repeats[parts["test"]] <= 2)
        ]
        test = parts["test"][~np.isin(parts["test"], reference)]
        target = {
            a: {
                str(s): experiment(
                    f"{name}-target",
                    a,
                    s,
                    parts["train"],
                    parts["validation"],
                    test,
                    reference,
                )
                for s in protocol["seeds"]
            }
            for a in ARMS
        }
        report["protocols"][name] = {
            "selected": selected,
            "source_summary": summary,
            "target": target,
        }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps({"run_id": run_id, "status": "complete", "fits": completed}),
        flush=True,
    )


if __name__ == "__main__":
    run()
