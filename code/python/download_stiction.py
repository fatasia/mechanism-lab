"""下载发布方CC BY 4.0阀门黏滞现场数据；只接收文本并核对SHA-256。"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from download_benchmarks import fetch
from download_pneumatic import read_json

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://data.mendeley.com/public-api/datasets/dc9fp3nppf"


def run():
    directory = ROOT / "data/raw/plant-stiction"
    directory.mkdir(parents=True, exist_ok=True)
    folders = read_json(f"{BASE}/folders/1")
    files = []
    for folder in folders:
        if folder["name"] not in ("non-stiction_data", "stiction_data"):
            continue
        offset = 0
        while True:
            batch = read_json(
                f"{BASE}/files?folder_id={folder['id']}&version=1&$start={offset}&$limit=1000"
            )
            files.extend({**item, "label_folder": folder["name"]} for item in batch)
            if len(batch) < 1000:
                break
            offset += len(batch)
    (directory / "publisher-files.json").write_text(
        json.dumps(files, indent=2), encoding="utf-8"
    )

    def download(item):
        name = item["filename"]
        if Path(name).name != name or Path(name).suffix != ".txt":
            raise ValueError(f"预期单文件名文本: {name}")
        target = directory / item["label_folder"]
        target.mkdir(exist_ok=True)
        details = item["content_details"]
        result = fetch(
            target,
            {"name": name, "url": details["download_url"], "size": details["size"]},
        )
        if result["sha256"] != details["sha256_hash"]:
            raise ValueError(f"发布方SHA256不一致: {name}")
        return {**result, "label_folder": item["label_folder"]}

    with ThreadPoolExecutor(max_workers=2) as pool:
        inventory = list(pool.map(download, files))
    source = {
        "doi": "10.17632/dc9fp3nppf.1",
        "source": "https://data.mendeley.com/datasets/dc9fp3nppf/1",
        "license": "CC-BY-4.0",
        "authors": ["Tianci Xue", "Chao Shang"],
        "paper": "10.1016/j.conengprac.2025.106588",
        "scope": "plant control-loop stiction; folder labels require audit; filename Stick1 score is not an input feature or ground truth",
        "files": inventory,
    }
    (directory / "source.json").write_text(
        json.dumps(source, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({"verified_files": len(inventory), "doi": source["doi"]}), flush=True
    )


if __name__ == "__main__":
    run()
