"""只解压已校验电机归档中的CSV；检查路径、条目类型和所需磁盘空间。"""

import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[2]


def safe_member(name):
    path = PurePosixPath(name)
    return (
        not path.is_absolute()
        and not PureWindowsPath(name).drive
        and ".." not in path.parts
        and "\\" not in name
        and path.parts[0] == "MCC5-THU Motor"
    )


def run():
    directory = ROOT / "data/raw/mcc5-thu-motor"
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    archive = directory / source["file"]
    if archive.stat().st_size != source["bytes"]:
        raise ValueError("归档大小已改变")
    listing = subprocess.run(
        ["tar", "-tvf", str(archive)], capture_output=True, check=True
    ).stdout.decode("utf-8", errors="replace")
    (directory / "archive-list.txt").write_text(listing, encoding="utf-8")
    entries = []
    for line in listing.splitlines():
        fields = line.split(maxsplit=8)
        mode, name = fields[0][0], fields[8]
        if mode not in ("-", "d") or not safe_member(name):
            raise ValueError(f"归档包含不支持条目: {name}")
        if mode == "-" and name.endswith(".csv"):
            entries.append({"path": name, "bytes": int(fields[4])})
    if not entries or len({entry["path"] for entry in entries}) != len(entries):
        raise ValueError("CSV清单为空或路径重复")
    total = sum(entry["bytes"] for entry in entries)
    if shutil.disk_usage(directory).free < total + 2 * 1024**3:
        raise ValueError("磁盘空间不足以解压CSV并保留2GiB余量")
    destination = directory / "extracted"
    destination.mkdir(exist_ok=True)
    selection = directory / "csv-selection.txt"
    selection.write_text(
        "\n".join(entry["path"] for entry in entries), encoding="utf-8"
    )
    print(
        json.dumps({"csv_files": len(entries), "uncompressed_bytes": total}), flush=True
    )
    subprocess.run(
        ["tar", "-xf", str(archive), "-C", str(destination), "-T", str(selection)],
        check=True,
    )
    for entry in entries:
        if (destination / entry["path"]).stat().st_size != entry["bytes"]:
            raise ValueError(f"解压文件大小不一致: {entry['path']}")
    (directory / "extracted.json").write_text(
        json.dumps(
            {"archive_sha256": source["sha256"], "files": entries, "bytes": total},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("CSV extraction complete", flush=True)


if __name__ == "__main__":
    run()
