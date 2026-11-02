"""同周期内一步预测与持久性基线对照，独立记录，不默认部署。"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from phm_core.artifacts import new_run, run_path
from phm_core.profiles import load_profile
from phm_core.tcn import TCN

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    torch.manual_seed(42)
    torch.set_num_threads(2)
    parent = run_path(ROOT / "outputs", "valve")
    run_id, out = new_run(ROOT / "outputs", "tcn")
    # 预测实验不用离线零相位降采样信号，避免滤波读取未来原始值。
    raw, labels, data_hash = load_profile(
        ROOT / "data/raw/uci-hydraulic/dataset.zip",
        "thermal-1hz",
        ROOT / "data/prepared",
    )
    healthy = labels[:, 1] == 100
    split = json.loads((parent / "split.json").read_text())
    ids = {k: np.array(v)[healthy[v] & (labels[v, 4] == 0)] for k, v in split.items()}
    mean, scale = (
        raw[ids["train"]].mean(axis=(0, 1)),
        raw[ids["train"]].std(axis=(0, 1)),
    )
    x = torch.from_numpy((raw - mean) / scale)
    model = TCN(raw.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best, history = float("inf"), []
    for epoch in range(30):
        for batch in x[ids["train"]][torch.randperm(len(ids["train"]))].split(32):
            optimizer.zero_grad()
            # 舍去前 14 步：每个计分点具有完整的因果感受野。
            loss = torch.nn.functional.huber_loss(
                model(batch[:, :-1])[:, 14:], batch[:, 15:]
            )
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            batch = x[ids["validation"]]
            val = torch.nn.functional.huber_loss(
                model(batch[:, :-1])[:, 14:], batch[:, 15:]
            ).item()
        history.append(val)
        if val < best:
            best = val
            torch.save(model.state_dict(), out / "tcn.pt")
    model.load_state_dict(torch.load(out / "tcn.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        scores = {
            "tcn": (model(x[:, :-1])[:, 14:] - x[:, 15:])
            .abs()
            .mean(dim=(1, 2))
            .numpy(),
            "persistence": (x[:, 14:-1] - x[:, 15:]).abs().mean(dim=(1, 2)).numpy(),
        }
    report = {
        "task": "valve within-cycle one-step prediction",
        "parent_split_run_id": parent.name,
        "profile": "thermal-1hz",
        "sample_rate_hz": 1,
        "data_sha256": data_hash,
        "scope": "native 1 Hz samples; within-cycle one-step prediction; no offline resampling; not an external validation set",
        "run_id": run_id,
        "not_deployed": True,
        "history": history,
        "models": {},
    }
    test = np.array(split["test"])
    for name, score in scores.items():
        threshold = float(np.quantile(score[ids["calibration"]], 0.99))
        report["models"][name] = {
            "pr_auc": float(average_precision_score(~healthy[test], score[test])),
            "threshold": threshold,
            "normal_false_positive_fraction": float(
                (score[test[healthy[test]]] > threshold).mean()
            ),
        }
    (out / "tcn-experiment.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["models"], indent=2))


if __name__ == "__main__":
    main()
