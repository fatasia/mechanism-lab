"""演示只读取已发布的实验目录，缓存键包含不可变版本。"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from phm_core.artifacts import run_path

ROOT = Path(__file__).resolve().parents[4]
TASKS = ("valve", "cooler", "pump", "accumulator")


def resolve(task: str, run_id: str | None = None) -> Path:
    return run_path(ROOT / "outputs", task, run_id)


def metadata(path: Path) -> dict:
    report = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    report.setdefault("run_id", "legacy")
    report.setdefault("profile", "thermal-1hz")
    report.setdefault("units", manifest["units"])
    report.setdefault("sample_rate_hz", manifest["sampleRateHz"])
    report.setdefault("cycle_count", 2205)
    return report


@lru_cache(maxsize=2)
def replay_data(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / "replay.npz") as data:
        return {key: data[key] for key in data.files}


def list_runs(task: str) -> list[dict]:
    base = ROOT / "outputs" / task
    paths = sorted(
        (
            p
            for p in (base / "runs").glob("*/metrics.json")
            if (p.parent / "ready.json").exists()
        ),
        reverse=True,
    )
    if (base / "metrics.json").exists():
        paths.append(base / "metrics.json")
    return [metadata(path.parent) for path in paths]
