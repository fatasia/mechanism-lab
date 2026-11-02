"""下载发布方CC BY 4.0发动机故障数据，用于独立跨负载方法验证。"""

import json
from pathlib import Path
from zipfile import ZipFile

from download_benchmarks import fetch

ROOT = Path(__file__).resolve().parents[2]


def run():
    directory = ROOT / "data/raw/marine-engine"
    directory.mkdir(parents=True, exist_ok=True)
    metadata = fetch(
        directory,
        {
            "name": "Marine_Engine_Fault_Data_v1.zip",
            "url": "https://zenodo.org/records/19857425/files/Marine_Engine_Fault_Data_v1.zip?download=1",
            "size": 26538633,
            "md5": "f4d246c1bb46e05b26e56221acc2606c",
        },
    )
    with ZipFile(directory / metadata["name"]) as archive:
        invalid = archive.testzip()
        if invalid:
            raise ValueError(f"归档CRC不一致: {invalid}")
        inventory = [
            {"path": item.filename, "bytes": item.file_size}
            for item in archive.infolist()
        ]
    report = {
        "doi": "10.5281/zenodo.19857425",
        "license": "CC-BY-4.0",
        "authors": [
            "Ahmad BahooToroody",
            "Oleksiy Bondarenko",
            "Mohammad Mahdi Abaei",
            "Yoichi Niki",
            "Enrico Zio",
        ],
        "domain": "Matsui MU323DGSC marine engine; not lithium equipment or a valve benchmark",
        "file": metadata,
        "inventory": inventory,
    }
    (directory / "source.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
