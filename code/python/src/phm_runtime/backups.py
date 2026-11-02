"""独立SQLite快照、校验清单与限定目录的保留策略。"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def create_backup(store, directory: Path, keep=96):
    if keep < 1:
        raise ValueError("至少保留一份备份")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    name = (
        "runtime-"
        + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + uuid4().hex[:8]
    )
    path = directory / (name + ".sqlite")
    store.backup(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    record = {
        "at": datetime.now(UTC).isoformat(),
        "file": path.name,
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "integrity": "ok",
        "journal_mode": "delete",
    }
    path.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    # 仅清理本工具创建且有对应清单的普通快照；不遍历/删除其它目录。
    snapshots = [
        p
        for p in sorted(directory.glob("runtime-*.sqlite"), reverse=True)
        if not p.is_symlink()
        and p.resolve().parent == directory
        and p.with_suffix(".json").is_file()
        and not p.with_suffix(".json").is_symlink()
    ]
    for old in snapshots[keep:]:
        metadata = json.loads(old.with_suffix(".json").read_text(encoding="utf-8"))
        if (
            metadata.get("file") == old.name
            and metadata.get("journal_mode") == "delete"
        ):
            old.unlink()
            old.with_suffix(".json").unlink()
    return {**record, "path": str(path)}
