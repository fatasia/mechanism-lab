"""从 VisA 原始未压缩 tar 流中读取 candle 子集，不下载其余 11 类。"""

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"


def extract(archive, destination):
    files = []
    for entry in archive:
        if entry.name.split("/")[0] != "candle":
            break  # 原始 tar 首类是 candle；读到下一类便结束。
        if not entry.isfile():
            continue
        target = destination / entry.name
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError("tar 路径超出数据目录")
        target.parent.mkdir(parents=True, exist_ok=True)
        content = archive.extractfile(entry).read()
        target.write_bytes(content)
        files.append(
            {
                "file": entry.name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return files


def run():
    destination = ROOT / "data/raw/visa"
    destination.mkdir(parents=True, exist_ok=True)
    split = destination / "1cls.csv"
    if not split.exists():
        with urlopen(
            "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv",
            timeout=60,
        ) as response:
            split.write_bytes(response.read())
    # 已有完整首类的下载前缀也可读取；不把前缀当成完整原始归档。
    prefix = destination / "VisA_20220922.tar.partial"
    if prefix.exists():
        with tarfile.open(prefix) as archive:
            files = extract(archive, destination)
    else:
        with (ROOT / "work/visa-subset-download.log").open("w") as log:
            process = subprocess.Popen(
                ["curl.exe", "--fail", "--location", "--max-time", "600", URL],
                stdout=subprocess.PIPE,
                stderr=log,
            )
            try:
                with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                    files = extract(archive, destination)
            finally:
                process.stdout.close()
                process.terminate()
                process.wait()
    if len([f for f in files if "/Images/" in f["file"]]) != 1100:
        raise ValueError("candle 图像数与发布方的 1000 正常 + 100 异常不一致")
    (destination / "candle-source.json").write_text(
        json.dumps(
            {
                "source": URL,
                "landing": "https://github.com/amazon-science/spot-diff",
                "license": "CC BY 4.0",
                "transformation": "first candle category only, no relabelling; remaining archive not consumed",
                "files": files,
            },
            indent=2,
        )
    )
    print(json.dumps({"category": "candle", "verified_files": len(files)}), flush=True)


if __name__ == "__main__":
    run()
