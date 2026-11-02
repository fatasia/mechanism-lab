"""已完成实验不可变；仅用原子指针发布新的默认版本。"""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

RUN_PATTERN = r"[A-Za-z0-9_-]+"


def capture_source(
    root: Path, target: Path, entrypoint: str = "train_hydraulic.py"
) -> str:
    """实验保存实际训练源文件，而不只保存一个无法找回的指纹。"""
    files = [
        root / entrypoint,
        root / "pyproject.toml",
        root / "requirements.lock.txt",
        *(
            [root / "requirements.reference.lock.txt"]
            if (root / "requirements.reference.lock.txt").exists()
            else []
        ),
        *sorted((root / "src/phm_core").glob("*.py")),
    ]
    inventory = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr(
            "source-manifest.json", json.dumps(inventory, sort_keys=True, indent=2)
        )
    return hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()


def new_run(outputs: Path, task: str) -> tuple[str, Path]:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    path = outputs / task / "runs" / run_id
    path.mkdir(parents=True)
    return run_id, path


def publish_run(outputs: Path, task: str, run_id: str) -> None:
    path = outputs / task / "runs" / run_id
    if not (path / "metrics.json").exists() or not (path / "model-bundle.zip").exists():
        raise ValueError("实验产物不完整，不能发布")
    (path / "ready.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    pointer = outputs / task / "latest.json"
    temporary = pointer.with_name(f"latest-{run_id}.tmp")
    temporary.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    temporary.replace(pointer)


def run_path(outputs: Path, task: str, run_id: str | None = None) -> Path:
    if run_id is not None and not re.fullmatch(RUN_PATTERN, run_id):
        raise ValueError("实验版本格式不正确")
    base = outputs / task
    if run_id is None and (base / "latest.json").exists():
        run_id = json.loads((base / "latest.json").read_text(encoding="utf-8"))[
            "run_id"
        ]
    # 旧版结果仅供回看，不再覆盖。
    path = base if run_id in (None, "legacy") else base / "runs" / run_id
    if not (path / "metrics.json").exists() or (
        run_id not in (None, "legacy") and not (path / "ready.json").exists()
    ):
        raise FileNotFoundError("该实验尚未完成或不存在")
    return path
