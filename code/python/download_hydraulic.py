"""从 UCI 官方源下载并审计标签；数据不写入代码目录。"""

import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
URL = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"


def main() -> None:
    target = ROOT / "data/raw/uci-hydraulic"
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "dataset.zip"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        with urlopen(URL, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(archive)
    with ZipFile(archive) as package:
        profile = np.loadtxt(package.open("profile.txt"))
        audit = {
            "source": URL,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "cycles": len(profile),
            "healthy_stable_cycles": int(
                np.all(profile == [100, 100, 0, 130, 0], axis=1).sum()
            ),
            "label_counts": [
                {
                    str(k): int(v)
                    for k, v in zip(*np.unique(profile[:, col], return_counts=True))
                }
                for col in range(5)
            ],
            "members": package.namelist(),
        }
    output = ROOT / "outputs/data-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
