"""下载Zenodo公开工业数据，逐文件核对发布方MD5，记录许可证与来源。"""

import argparse
import json
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from download_benchmarks import fetch

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "zema-emc": {
        "record": 3929385,
        "files": (
            "documentation.pdf",
            "axis3_2kHz.h5",
            "axis5_2kHz.h5",
            "axis7_2kHz.h5",
        ),
        "scope": "3台机电缸(axis3/5/7)寿命试验周期，11传感器，2kHz；仅取回行程1秒",
    },
    "zema-axis11-si": {
        "record": 5185953,
        "files": ("axis11_2kHz_ZeMA_PTB_SI.h5",),
        "scope": "第4台机电缸(axis11)，SI单位，ZeMA与PTB两套测量系统；用于核对退化结论",
    },
    "pronto": {
        "record": 1341583,
        "files": ("README.txt", "PRONTO benchmark case study.zip"),
        "scope": "多相流装置：过程量、报警、高频超声、操作日志、视频；含/不含诱发故障",
    },
    "comopi": {
        "record": 7572501,
        "files": (
            "readme.md",
            "industrial_dataset_sensors_10m_agg.csv",
            "industrial_dataset_alarm_10m_agg.csv",
            "industrial_dataset_warnings_10m_agg.csv",
        ),
        "scope": "8台包装机10分钟聚合传感器与设备报警/警告；非维修确认故障真值",
    },
}


def read_json(url):
    result = subprocess.run(
        [
            shutil.which("curl.exe") or "curl",
            "--fail",
            "--silent",
            "--max-time",
            "60",
            url,
        ],
        check=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


CHUNK = 8 << 20


def fetch_chunk(url, start, end):
    """Zenodo 对不带 Range 的整文件请求会挂起，只能分块取回。"""
    for attempt in range(6):
        result = subprocess.run(
            [
                shutil.which("curl.exe") or "curl",
                "--fail",
                "--location",
                "--silent",
                "--max-time",
                "240",
                "--range",
                f"{start}-{end}",
                "--output",
                "-",
                url,
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and len(result.stdout) == end - start + 1:
            return result.stdout
        time.sleep(2**attempt)
    raise RuntimeError(f"分块下载失败: {start}-{end}")


def download_ranged(directory, items, workers=8):
    """多连接分块下载；已完成块记录在侧车文件，中断后可继续。MD5 对不上即报错。"""
    lock = threading.Lock()
    jobs = []
    for item in items:
        path = directory / item["name"]
        if path.exists():
            continue
        partial = path.with_suffix(path.suffix + ".partial")
        done_path = path.with_suffix(path.suffix + ".done.json")
        done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
        if not partial.exists() or partial.stat().st_size != item["size"]:
            with partial.open("wb") as stream:
                stream.truncate(item["size"])
            done = set()
        for start in range(0, item["size"], CHUNK):
            if start not in done:
                jobs.append((item, partial, done_path, done, start))

    def work(job):
        item, partial, done_path, done, start = job
        end = min(start + CHUNK, item["size"]) - 1
        data = fetch_chunk(item["url"], start, end)
        with lock:
            with partial.open("r+b") as stream:
                stream.seek(start)
                stream.write(data)
            done.add(start)
            done_path.write_text(json.dumps(sorted(done)))
        return item["name"]

    finished = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(work, jobs):
            finished += 1
            if finished % 20 == 0:
                print(
                    json.dumps({"chunks_done": finished, "of": len(jobs)}), flush=True
                )
    for item in items:
        path = directory / item["name"]
        partial = path.with_suffix(path.suffix + ".partial")
        if not path.exists():
            partial.replace(path)
            path.with_suffix(path.suffix + ".done.json").unlink(missing_ok=True)
    # 已存在的文件由 fetch 校验发布方 MD5 并记录 SHA-256。
    return [fetch(directory, item) for item in items]


def download(name):
    spec = DATASETS[name]
    record = read_json(f"https://zenodo.org/api/records/{spec['record']}")
    license_id = record["metadata"]["license"]["id"]
    if license_id != "cc-by-4.0":
        raise ValueError(f"许可证不是CC BY 4.0，停止: {license_id}")
    directory = ROOT / "data/raw" / name
    directory.mkdir(parents=True, exist_ok=True)
    published = {item["key"]: item for item in record["files"]}
    items = [
        {
            "name": key,
            "url": published[key]["links"]["self"],
            "size": published[key]["size"],
            "md5": published[key]["checksum"].removeprefix("md5:"),
        }
        for key in spec["files"]
    ]
    inventory = download_ranged(directory, items)
    (directory / "source.json").write_text(
        json.dumps(
            {
                "dataset": name,
                "doi": record["doi"],
                "source": f"https://zenodo.org/records/{spec['record']}",
                "license": "CC-BY-4.0",
                "title": record["metadata"]["title"],
                "scope": spec["scope"],
                "files": inventory,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"{name}: {len(inventory)} files verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASETS))
    download(parser.parse_args().dataset)
