"""现场阀门黏滞的整回路留出基线；控制形状特征，不输入回路身份或已有黏滞分数。"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from audit_stiction import DATA, parse_record
from phm_core.artifacts import capture_source, new_run

ROOT = Path(__file__).resolve().parents[2]


def correlation(a, b):
    a, b = a - np.mean(a), b - np.mean(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else 0.0


def shape_features(values, dt):
    """单记录内的无量纲形状；PV与SP共用尺度，时序差分保留实际采样间隔。"""
    signals = values[:, :2]
    scale = np.quantile(signals, 0.75, axis=0) - np.quantile(signals, 0.25, axis=0)
    scale = np.where(scale > 0, scale, signals.std(0))
    scale = np.where(scale > 0, scale, 1.0)
    centered = (signals - np.median(signals, axis=0)) / scale
    result = {}
    for channel, name in enumerate(("op", "pv")):
        x = centered[:, channel]
        difference = np.diff(x)
        power = abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))) ** 2
        power = power[1:]
        probabilities = power / power.sum() if power.sum() > 0 else np.zeros_like(power)
        result.update(
            {
                f"{name}/std": float(x.std()),
                f"{name}/range": float(np.ptp(x)),
                f"{name}/derivative_abs": float(np.abs(difference).mean() / dt),
                f"{name}/derivative_std": float(difference.std() / dt),
                f"{name}/plateau_fraction": float(np.mean(np.abs(difference) <= 0.01)),
                f"{name}/spectral_entropy": float(
                    -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-20)))
                    / np.log(len(probabilities))
                ),
                f"{name}/spectral_peak_fraction": float(probabilities.max()),
            }
        )
        for seconds in (60, 120, 300):
            lag = max(1, round(seconds / dt))
            result[f"{name}/autocorrelation_{seconds}s"] = correlation(
                x[:-lag], x[lag:]
            )
    op, pv = centered.T
    dop, dpv = np.diff(op), np.diff(pv)
    lags = np.arange(-300, 301, 60)
    correlations = []
    for seconds in lags:
        lag = round(seconds / dt)
        correlations.append(
            correlation(op[:-lag], pv[lag:])
            if lag > 0
            else correlation(op[-lag:], pv[:lag])
            if lag < 0
            else correlation(op, pv)
        )
    best = int(np.argmax(np.abs(correlations)))
    error = (values[:, 1] - values[:, 2]) / scale[1]
    result.update(
        {
            "relation/op_pv_correlation": correlation(op, pv),
            "relation/derivative_correlation": correlation(dop, dpv),
            "relation/lag_correlation": correlations[best],
            "relation/lag_seconds": float(lags[best]),
            "relation/signed_path_area": float(np.mean(op[:-1] * dpv - pv[:-1] * dop)),
            "relation/absolute_path_area": float(
                np.mean(np.abs(op[:-1] * dpv - pv[:-1] * dop))
            ),
            "relation/moving_op_flat_pv": float(
                np.mean((np.abs(dop) > 0.01) & (np.abs(dpv) <= 0.01))
            ),
            "relation/flat_op_moving_pv": float(
                np.mean((np.abs(dop) <= 0.01) & (np.abs(dpv) > 0.01))
            ),
            "relation/error_rms": float(np.sqrt(np.mean(error**2))),
            "relation/error_op_correlation": correlation(error, op),
            "relation/setpoint_range": float(np.ptp(values[:, 2]) / scale[1]),
        }
    )
    return result


def training_weights(labels, groups):
    """同一回路所有记录总权重相同，再平衡两类回路数，避免23条单回路支配训练。"""
    unique, counts = np.unique(groups, return_counts=True)
    counts_by_group = dict(zip(unique, counts))
    weights = np.array([1 / counts_by_group[g] for g in groups])
    # 同回路未来可能同时出现两种标签；类质量由全部记录累计，不能用该回路首条标签代替。
    classes = np.bincount(labels, weights=weights, minlength=2)
    weights /= classes[labels]
    return weights * len(weights) / weights.sum()


def summary(labels, scores, total, weights=None):
    predicted = scores > 0.5
    normal, fault = labels == 0, labels == 1
    tn, fp, fn, tp = confusion_matrix(
        labels, predicted, labels=[0, 1], sample_weight=weights
    ).ravel()
    if weights is not None:
        return {
            "normal_fpr": float(fp / (tn + fp)),
            "fault_recall": float(tp / (tp + fn)),
            "macro_f1": float(
                f1_score(labels, predicted, average="macro", sample_weight=weights)
            ),
        }
    return {
        "records": total,
        "supported_records": len(labels),
        "unsupported_records": total - len(labels),
        "normal": int(normal.sum()),
        "fault": int(fault.sum()),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def run(audit_path):
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    run_id, out = new_run(ROOT / "outputs", "stiction-study")
    protocol = {
        "split": "leave one complete loop out; all overlapping dates and repeated records from loop stay in same fold",
        "quality": "exclude whole record for audit failure or any nonzero offline flag; no row sorting, interpolation or stitching",
        "representations": "20 univariate shape descriptors vs11 OP/PV/SP relationship descriptors vs31 combined; no file name,loop identity,Stick1 or fs metadata as feature",
        "models": "RF200,max_depth5,min_leaf2,max_features=sqrt,seeds42/43/44; weighted StandardScaler+LogisticRegressionC1,max_iter2000,seed42",
        "threshold": "fixed probability>0.5; no target tuning, no model selection",
        "weights": "equal total training weight per loop,then equal per class; record and equal-loop metrics both reported",
        "scope": "small observational plant dataset with label-loop confounding; publisher folder labels, not independently verified maintenance truth; not UCI pressure valve validation or field acceptance",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    records = [
        r
        for r in audit["records"]
        if r["valid"] and not any(r["offline_nonzero_counts"])
    ]
    features = []
    for record in records:
        content = (DATA / record["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError("原始黏滞记录哈希变化")
        values, _, metadata = parse_record(content)
        features.append(shape_features(values, metadata["sample_interval_seconds"]))
    names = list(features[0])
    x = np.array([[f[name] for name in names] for f in features])
    y, groups = (
        np.array([r["label"] for r in records]),
        np.array([r["loop"] for r in records]),
    )
    ids = np.array([r["id"] for r in records])
    np.savez_compressed(out / "features.npz", features=x, record_ids=ids)
    helper = ROOT / "code/python/audit_stiction.py"
    shutil.copyfile(helper, out / helper.name)
    report = {
        "run_id": run_id,
        "protocol": protocol,
        "classes": ["非黏滞", "黏滞"],
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", "study_stiction.py"
        ),
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "audit_run": audit["run_id"],
        "feature_names": names,
        "supported_ids": ids.tolist(),
        "excluded_ids": [r["id"] for r in audit["records"] if r not in records],
        "protocols": {"held-out-loop": {"target": {}}},
        "folds": {},
        "loop_weighted": {},
    }
    arms = {
        "statistics_rf": (np.arange(20), [42, 43, 44]),
        "relationships_rf": (np.arange(20, 31), [42, 43, 44]),
        "combined_rf": (np.arange(31), [42, 43, 44]),
        "combined_logistic": (np.arange(31), [42]),
    }
    counts = {g: np.sum(groups == g) for g in np.unique(groups)}
    evaluation_weights = np.array([1 / counts[g] for g in groups])
    for arm, (columns, seeds) in arms.items():
        report["protocols"]["held-out-loop"]["target"][arm] = {}
        report["folds"][arm], report["loop_weighted"][arm] = {}, {}
        for seed in seeds:
            scores, folds = np.empty(len(x)), []
            for group in np.unique(groups):
                test, train = groups == group, groups != group
                weights = training_weights(y[train], groups[train])
                a, b = x[train][:, columns], x[test][:, columns]
                if arm.endswith("logistic"):
                    scaler = StandardScaler().fit(a, sample_weight=weights)
                    a, b = scaler.transform(a), scaler.transform(b)
                    model = LogisticRegression(C=1, max_iter=2000, random_state=seed)
                else:
                    model = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=5,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        random_state=seed,
                        n_jobs=1,
                    )
                model.fit(a, y[train], sample_weight=weights)
                scores[test] = model.predict_proba(b)[:, list(model.classes_).index(1)]
                folds.append(
                    {
                        "loop": group,
                        "train_ids": ids[train].tolist(),
                        "test_ids": ids[test].tolist(),
                    }
                )
            result = summary(y, scores, len(audit["records"]))
            report["protocols"]["held-out-loop"]["target"][arm][str(seed)] = result
            report["loop_weighted"][arm][str(seed)] = summary(
                y, scores, len(audit["records"]), evaluation_weights
            )
            report["folds"][arm][str(seed)] = folds
            np.savez_compressed(
                out / f"{arm}-{seed}.npz",
                record_ids=ids,
                labels=y,
                probabilities=scores,
                loops=groups,
            )
            print(arm, seed, json.dumps(result), flush=True)
    (out / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(run_id, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    run(parser.parse_args().audit.resolve())
