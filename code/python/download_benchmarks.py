"""下载原发布方的执行器、轴承及真实压缩机数据，记录校验和与来源。"""

import argparse
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
CRAN = "https://cran-test-dspace.koha-ptfs.co.uk/server/api/core"
ITEM = "eff8a9f5-861e-49f1-9e0b-67d5c7393616"
METRO = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"


def read_json(url):
    with urlopen(url, timeout=60) as response:
        return json.load(response)


def fetch(directory, item):
    path = directory / item["name"]
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".partial")
        curl = shutil.which("curl.exe") or shutil.which("curl")
        if curl:
            subprocess.run(
                [
                    curl,
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--max-time",
                    "900",
                    "--output",
                    str(partial),
                    item["url"],
                ],
                check=True,
            )
        else:
            with (
                urlopen(item["url"], timeout=180) as response,
                partial.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
        if item.get("size") and partial.stat().st_size != item["size"]:
            raise ValueError(f"下载长度不一致: {path.name}")
        partial.replace(path)
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
            md5.update(chunk)
    if item.get("md5") and md5.hexdigest() != item["md5"]:
        raise ValueError(f"发布方校验和不一致: {path.name}")
    result = {
        **item,
        "sha256": sha.hexdigest(),
        "bytes": path.stat().st_size,
    }
    print(
        json.dumps({"downloaded": path.name, "bytes": path.stat().st_size}), flush=True
    )
    return result


def download(dataset):
    directory = ROOT / "data/raw" / dataset
    directory.mkdir(parents=True, exist_ok=True)
    if dataset == "cranfield-actuator":
        bundles = read_json(f"{CRAN}/items/{ITEM}/bundles?size=100")["_embedded"][
            "bundles"
        ]
        original = next(b for b in bundles if b["name"] == "ORIGINAL")
        bits = read_json(original["_links"]["bitstreams"]["href"] + "?size=100")
        items = [
            {
                "name": b["name"],
                "url": b["_links"]["content"]["href"],
                "size": b["sizeBytes"],
                "md5": b["checkSum"]["value"],
            }
            for b in bits["_embedded"]["bitstreams"]
        ]
        source, license_name = (
            "https://doi.org/10.17862/cranfield.rd.5097649",
            "CC BY 4.0",
        )
    elif dataset == "metropt3":
        items = [{"name": "dataset.zip", "url": METRO}]
        source, license_name = "https://doi.org/10.24432/C5VW3R", "CC BY 4.0"
    elif dataset == "visa":
        items = [
            {
                "name": "VisA_20220922.tar",
                "url": "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
            },
            {
                "name": "1cls.csv",
                "url": "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv",
            },
        ]
        source, license_name = (
            "https://github.com/amazon-science/spot-diff",
            "CC BY 4.0 (data)",
        )
    elif dataset == "cmapss":
        items = [
            {
                "name": "CMAPSSData.zip",
                "url": "https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
            }
        ]
        source, license_name = (
            "https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data",
            "Public access; catalog does not specify a redistribution license",
        )
    elif dataset == "resnet18-weights":
        items = [
            {
                "name": "resnet18-f37072fd.pth",
                "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            }
        ]
        source, license_name = (
            "https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html",
            "Torchvision BSD-3-Clause; ImageNet-pretrained weights",
        )
    else:
        ids = [
            97,
            98,
            99,
            100,
            105,
            106,
            107,
            108,
            118,
            119,
            120,
            121,
            130,
            131,
            132,
            133,
        ]
        items = [
            {
                "name": f"{i}.mat",
                "url": f"https://engineering.case.edu/sites/default/files/{i}.mat",
            }
            for i in ids
        ]
        source, license_name = (
            "https://engineering.case.edu/bearingdatacenter/download-data-file",
            "No explicit redistribution license found; local research only",
        )
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(lambda item: fetch(directory, item), items))
    (directory / "source.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "source": source,
                "license": license_name,
                "files": files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{dataset}: {len(files)} files verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=[
            "cranfield-actuator",
            "metropt3",
            "cwru",
            "visa",
            "cmapss",
            "resnet18-weights",
        ],
    )
    download(parser.parse_args().dataset)
