"""下载气动装配测试机CSV，逐文件核对发布方SHA256，不加载pickle。"""

import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from download_benchmarks import fetch

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://data.mendeley.com/public-api/datasets/ypzswhhzh9"


def read_json(url):
    """与文件下载共用curl传输，避免发布站拒绝urllib客户端。"""
    result = subprocess.run(
        [
            shutil.which("curl.exe") or "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            "60",
            url,
        ],
        check=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def run():
    directory = ROOT / "data/raw/pneumatic-process"
    directory.mkdir(parents=True, exist_ok=True)
    folders = read_json(f"{BASE}/folders/3")
    folder = next(item["id"] for item in folders if item["name"] == "Csv")
    files = []
    offset = 0
    while True:
        batch = read_json(
            f"{BASE}/files?folder_id={folder}&version=3&$start={offset}&$limit=1000"
        )
        files.extend(batch)
        if len(batch) < 1000:
            break
        offset += len(batch)
    (directory / "publisher-files.json").write_text(
        json.dumps(files, indent=2), encoding="utf-8"
    )

    def download(item):
        details = item["content_details"]
        for attempt in range(3):
            try:
                result = fetch(
                    directory,
                    {
                        "name": item["filename"],
                        "url": details["download_url"],
                        "size": details["size"],
                    },
                )
                break
            except subprocess.CalledProcessError as error:
                if error.returncode not in (28, 35, 52, 56) or attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))
        if result["sha256"] != details["sha256_hash"]:
            raise ValueError(f"发布方SHA256不一致: {item['filename']}")
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        inventory = list(executor.map(download, files))
    report = {
        "doi": "10.17632/ypzswhhzh9.3",
        "source": "https://data.mendeley.com/datasets/ypzswhhzh9/3",
        "license": "CC-BY-4.0",
        "authors": [
            "Žiga Stržinar",
            "Boštjan Pregelj",
            "Janko Petrovčič",
            "Igor Škrjanc",
            "Gregor Dolanc",
        ],
        "scope": "Pressure/current and PLC action states; fault labels not established",
        "files": inventory,
    }
    (directory / "source.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"verified_files": len(inventory), "doi": report["doi"]}))


if __name__ == "__main__":
    run()
