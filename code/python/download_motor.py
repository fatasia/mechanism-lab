"""下载发布方CC BY 4.0多模态电机数据；断点续传后核对发布方SHA256。"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = "https://data.mendeley.com/public-files/datasets/6s3dggj9mw/files/70773d79-2dd6-46ee-ac7e-e24e4b1a57a7/file_downloaded"
SIZE = 9559624423
SHA256 = "594323c48c30ce8e4a6a41bc345cd8e9bc8bdcb9483f8f23665c4dbc55aea149"


def run():
    directory = ROOT / "data/raw/mcc5-thu-motor"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "MCC5-THU-Motor.7z"
    if not path.exists():
        partial = path.with_suffix(".7z.partial")
        subprocess.run(
            [
                shutil.which("curl.exe") or "curl",
                "--fail",
                "--location",
                "--continue-at",
                "-",
                "--retry",
                "3",
                "--retry-delay",
                "5",
                "--retry-all-errors",
                "--max-time",
                "14400",
                "--output",
                str(partial),
                URL,
            ],
            check=True,
        )
        if partial.stat().st_size != SIZE:
            raise ValueError("电机数据下载字节数与发布记录不符")
        partial.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise ValueError("电机数据发布方SHA256不一致")
    source = {
        "doi": "10.17632/6s3dggj9mw.1",
        "license": "CC-BY-4.0",
        "source": "https://data.mendeley.com/datasets/6s3dggj9mw/1",
        "authors": [
            "Shijin Chen",
            "Zeyi Liu",
            "Chenyang Li",
            "Dongliang Zou",
            "Xiao He",
            "Donghua Zhou",
        ],
        "file": path.name,
        "url": URL,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "scope": "induction motor vibration/current/torque/key-phase; not a servo or valve dataset",
    }
    (directory / "source.json").write_text(
        json.dumps(source, indent=2), encoding="utf-8"
    )
    print(json.dumps(source, indent=2))


if __name__ == "__main__":
    run()
