"""固定发布提交下载六台液压执行器数据；只读归档审计，不执行外部源码。"""

import hashlib
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "f9926eb53e1c324d7ffafc4536f2494b7508d7ed"
URL = f"https://codeload.github.com/vtnsi/actuator_dataset/zip/{COMMIT}"


def run():
    target = ROOT / "data/raw/vtnsi-actuator"
    target.mkdir(parents=True, exist_ok=True)
    archive = target / f"{COMMIT}.zip"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        subprocess.run(
            [
                "curl.exe",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--max-time",
                "900",
                URL,
                "--output",
                str(partial),
            ],
            check=True,
        )
        with ZipFile(partial) as source:
            error = source.testzip()
            if error:
                raise ValueError(f"归档CRC失败: {error}")
        partial.replace(archive)
    inventory, groups = [], {}
    with ZipFile(archive) as source:
        for entry in source.infolist():
            relative = entry.filename.split("/", 1)[-1]
            if relative in ("LICENSE", "README.md"):
                (target / relative).write_bytes(source.read(entry))
            if not relative.startswith("Data/") or entry.is_dir():
                continue
            content = source.read(entry)
            device = relative.split("/")[1]
            # 文件名前缀只用于库存审计，不自动视为经过确认的故障标签。
            prefix = Path(relative).name.split("_")[0]
            key = f"{device}/{prefix}"
            groups[key] = groups.get(key, 0) + 1
            inventory.append(
                {
                    "path": relative,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "url": URL,
        "repository": "https://github.com/vtnsi/actuator_dataset",
        "commit": COMMIT,
        "sha256": digest,
        "bytes": archive.stat().st_size,
        "license": "repository MIT; LICENSE retained, raw data not redistributed",
        "data_files": len(inventory),
        "device_prefix_counts": groups,
        "inventory": inventory,
        "status": "downloaded and audited; no model trained or test accessed for scoring",
        "limitation": "README says each actuator corresponds to a condition: audit device/fault confounding before cross-device claims",
    }
    (target / "source.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "inventory"}, indent=2))


if __name__ == "__main__":
    run()
