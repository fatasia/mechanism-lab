"""用实际发布的阀门 ONNX 验证并发、重试和数据库恢复；隔离验收数据。"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from phm_core.artifacts import run_path
from phm_runtime.registry import ModelRegistry
from phm_runtime.store import StateStore

ROOT = Path(__file__).resolve().parents[2]


def run():
    out = ROOT / "work/validation" / datetime.now(UTC).strftime("load-%Y%m%dT%H%M%S%f")
    out.mkdir(parents=True)
    store = StateStore(out / "runtime.sqlite")
    registry = ModelRegistry(ROOT / "outputs", store)
    directory = run_path(ROOT / "outputs", "valve")
    registry.activate("valve", directory.name, "ae", "shadow")
    scorer, _ = registry.load("valve", directory.name)
    spec = scorer.manifest
    with np.load(directory / "parity.npz") as data:
        raw = data["raw"].copy()

    def entity_work(entity):
        elapsed = []
        for sequence in range(500):
            start = time.perf_counter()
            result = registry.predict(
                "valve",
                entity,
                sequence,
                raw[sequence % len(raw)],
                spec["channels"],
                spec["units"],
                spec["sampleRateHz"],
            )
            elapsed.append((time.perf_counter() - start) * 1000)
            assert result["status"] == "scored" and not result["duplicate"]
        retry = registry.predict(
            "valve",
            entity,
            499,
            raw[499 % len(raw)],
            spec["channels"],
            spec["units"],
            spec["sampleRateHz"],
        )
        assert retry["duplicate"]
        return elapsed

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        latency = np.concatenate(
            list(pool.map(entity_work, [f"acceptance/{i}" for i in range(4)]))
        )
    seconds = time.perf_counter() - start
    backup = out / "backup.sqlite"
    store.backup(backup)
    restored = StateStore(backup)
    with restored.connect() as db:
        count = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        states = db.execute("SELECT COUNT(*) FROM states").fetchone()[0]
    assert count == 2000 and states == 4 and integrity == "ok"
    for i in range(4):
        assert restored.history(f"acceptance/{i}", 1)[0]["sequence"] == 499
    report = {
        "scope": "local runtime + actual ONNX + SQLite, not network/API or multi-day field soak",
        "run_id": directory.name,
        "requests": 2000,
        "workers": 4,
        "seconds": seconds,
        "requests_per_second": 2000 / seconds,
        "latency_ms": {f"p{q}": float(np.percentile(latency, q)) for q in (50, 95, 99)},
        "retry_duplicates": 4,
        "stored_predictions": count,
        "stored_states": states,
        "restored_integrity": integrity,
        "native_memory": "not measured",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(out / "report.json"), **report}), flush=True)


if __name__ == "__main__":
    run()
