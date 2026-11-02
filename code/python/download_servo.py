"""下载伺服电机（Mendeley g28trvywnx）无反转电流或振动波形，逐文件核对发布方SHA-256。

Mendeley 的文件夹层级与状态名没有直接对应，所以用发布方 qc_per_file.csv 中的
SHA-256 把文件映射到（状态, 转速）。哈希对不上即报错，不信任文件夹名或文件名。
"""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "data/raw/servo-dc"
API = "https://data.mendeley.com/public-api/datasets/g28trvywnx"
VERSION = 13
# 本地后缀、发布方 qc 表中的传感器名、发布方文件名后缀、清单文件名。
SENSORS = {
    "current": ("current", "_current.bin", "source.json"),
    "vibration": ("sound_vibrometer", "_sound_vibrometer.wav", "source-vibration.json"),
}
OPERATIONS = (
    "normal",
    "loose_foundation",
    "suboptimal_control",
    "suboptimal_control_rt",
)


def curl(args, **kwargs):
    return subprocess.run(
        [shutil.which("curl.exe") or "curl", "--silent", "--location", *args],
        check=False,
        **kwargs,
    )


def read_json(url):
    for attempt in range(6):
        result = curl(["--fail", "--max-time", "60", url], capture_output=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
        time.sleep(2**attempt)
    raise RuntimeError(f"读取失败: {url}")


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publisher_files():
    """数据集详情接口已含全部文件及发布方哈希，无需按文件夹分页。"""
    files = read_json(API).get("files", [])
    if not files:
        raise RuntimeError("发布方文件清单为空")
    return files


def plan(sensor="current", reversal="no"):
    """返回 [(状态, 转速, 发布方哈希, 下载地址, 大小)]；以 qc 表为准做映射。"""
    with (DIRECTORY / "qc_per_file.csv").open(newline="", encoding="utf-8") as stream:
        qc = {
            row["sha256"]: row
            for row in csv.DictReader(stream)
            if row["sensor"] == SENSORS[sensor][0] and row["reversal"] == reversal
        }
    items = []
    for entry in publisher_files():
        details = entry["content_details"]
        row = qc.get(details["sha256_hash"])
        if row is None or not entry["filename"].endswith(SENSORS[sensor][1]):
            continue
        items.append(
            (
                row["condition"].rsplit("_", 2)[0]
                if row["condition"].endswith(("_no_reversal", "_with_reversal"))
                else row["condition"],
                int(float(row["speed_percent"])),
                details["sha256_hash"],
                details["download_url"],
                details["size"],
            )
        )
    return sorted(items)


def local_path(item, sensor="current"):
    suffix = ".bin" if sensor == "current" else ".wav"
    return DIRECTORY / f"{item[0]}_no_reversal_speed{item[1]:03d}_{sensor}{suffix}"


def fetch(item, sensor="current"):
    _, _, expected, url, size = item
    target = local_path(item, sensor)
    if target.exists() and sha256_of(target) == expected:
        return {"file": target.name, "sha256": expected, "bytes": size, "cached": True}
    partial = target.with_suffix(".partial")
    for attempt in range(8):
        curl(["--continue-at", "-", "--max-time", "900", "--output", str(partial), url])
        if partial.exists() and partial.stat().st_size == size:
            break
        if partial.exists() and partial.stat().st_size > size:
            partial.unlink()
        time.sleep(min(30, 2**attempt))
    else:
        raise RuntimeError(f"下载未完成: {target.name}")
    if sha256_of(partial) != expected:
        partial.unlink()
        raise ValueError(f"发布方SHA-256不一致: {target.name}")
    partial.replace(target)
    print(json.dumps({"verified": target.name}), flush=True)
    return {"file": target.name, "sha256": expected, "bytes": size, "cached": False}


def run(workers, speeds, sensor="current"):
    """只下载 speeds 指定的转速；其它转速若本地已有则一并校验登记，不会再去下载。"""
    items = [
        i
        for i in plan(sensor)
        if speeds is None or i[1] in speeds or local_path(i, sensor).exists()
    ]
    if not items:
        raise RuntimeError("没有匹配的文件；检查 qc_per_file.csv 与版本号")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        inventory = list(pool.map(lambda i: fetch(i, sensor), items))
    (DIRECTORY / SENSORS[sensor][2]).write_text(
        json.dumps(
            {
                "dataset": "servo-dc",
                "sensor": sensor,
                "doi": "10.17632/g28trvywnx",
                "source": "https://data.mendeley.com/datasets/g28trvywnx",
                "license": "CC-BY-4.0",
                "version": VERSION,
                "scope": "无反转；每个工况点只录一次，非独立重复，须按工况点分组切分",
                "files": inventory,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"servo-dc/{sensor}: {len(inventory)} files verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--speeds", type=int, nargs="*", default=None)
    parser.add_argument("--sensor", choices=sorted(SENSORS), default="current")
    arguments = parser.parse_args()
    run(arguments.workers, arguments.speeds, arguments.sensor)
