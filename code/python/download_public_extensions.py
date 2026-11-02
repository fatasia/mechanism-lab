"""固定公开版本：CARE v6预注册子集与极片超声v4。原始数据不随源码分发。"""

import argparse
import csv
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from public_data import RemoteZipFile, curl_bytes, read_json, sha256

ROOT = Path(__file__).resolve().parents[2]


def electrode():
    directory = ROOT / "data/raw/electrode-ultrasound-v4"
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "publisher.json"
    meta = (
        json.loads(meta_path.read_text())
        if meta_path.exists()
        else read_json("https://data.mendeley.com/public-api/datasets/c62yn37d9h")
    )
    if meta["version"] != 4 or meta["data_licence"]["short_name"] != "CC BY 4.0":
        raise ValueError("发布版本或许可已改变，须重新审计，不自动切换")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    files = [
        f
        for f in meta["files"]
        if f["filename"] in ("before-calendering.json", "after-calendering.json")
    ]

    def fetch(f):
        path = directory / (f["id"] + ".json")
        expected = f["content_details"]["sha256_hash"]
        if not path.exists() or sha256(path) != expected:
            path.write_bytes(curl_bytes(f["content_details"]["download_url"]))
        if sha256(path) != expected:
            raise ValueError("极片文件SHA256不符")
        return {"file": path.name, "publisher_name": f["filename"], "sha256": expected}

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(fetch, files))
    (directory / "source.json").write_text(
        json.dumps(
            {"doi": "10.17632/c62yn37d9h.4", "license": "CC-BY-4.0", "files": records},
            indent=2,
        )
    )
    print(json.dumps({"electrode_files": len(records)}), flush=True)


def care(per_class=2):
    directory = ROOT / "data/raw/care-v6"
    directory.mkdir(parents=True, exist_ok=True)
    meta = read_json("https://zenodo.org/api/records/15846963")
    if meta["metadata"]["license"]["id"] != "cc-by-sa-4.0":
        raise ValueError("CARE许可变更")
    (directory / "publisher.json").write_text(json.dumps(meta, indent=2))
    item = meta["files"][0]
    remote = RemoteZipFile(
        item["links"]["self"], item["size"], ROOT / "data/raw/care-range-cache-v6"
    )
    with zipfile.ZipFile(remote) as archive:
        info = {i.filename: i for i in archive.infolist()}
        selected = []
        for farm in "ABC":
            prefix = f"CARE_To_Compare/Wind Farm {farm}/"
            raw = archive.read(prefix + "event_info.csv")
            (directory / f"events-{farm}.csv").write_bytes(raw)
            events = list(
                csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter=";")
            )
            used = set()
            for label in ("anomaly", "normal"):
                count = 0
                for event in sorted(events, key=lambda e: int(e["event_id"])):
                    if event["event_label"] != label:
                        continue
                    if "asset_id" not in event:
                        member = prefix + "datasets/" + event["event_id"] + ".csv"
                        with archive.open(member) as stream:
                            header = (
                                stream.readline().decode("utf-8-sig").strip().split(";")
                            )
                            first = stream.readline().decode().strip().split(";")
                        event["asset_id"] = first[header.index("asset_id")]
                    if event["asset_id"] in used:
                        continue
                    used.add(event["asset_id"])
                    selected.append(
                        {
                            "farm": farm,
                            **event,
                            "member": prefix + "datasets/" + event["event_id"] + ".csv",
                        }
                    )
                    count += 1
                    if count == per_class:
                        break
                if count != per_class:
                    raise ValueError(f"{farm}/{label}不够独立资产，不能随意改取样规则")
        protocol = {
            "selection": "smallest event_id; anomaly then normal; unique assets within each farm; no target features inspected",
            "per_class_per_farm": per_class,
            "cases": selected,
        }
        protocol_path = directory / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("已有取样协议不同")
        protocol_path.write_text(json.dumps(protocol, indent=2))
        # 并行预取选中成员覆盖的块，避免全量5.5GB下载；不宣称整包MD5已校验。
        starts = set()
        for event in selected:
            member = info[event["member"]]
            first = member.header_offset // remote.block_size * remote.block_size
            end = (
                member.header_offset
                + 30
                + len(member.filename.encode())
                + len(member.extra)
                + member.compress_size
                + 1024
            )
            starts.update(range(first, min(end, remote.size), remote.block_size))

        def prefetch(start):
            reader = RemoteZipFile(remote.url, remote.size, remote.cache_dir)
            reader.seek(start)
            reader.read(min(reader.block_size, reader.size - start))

        with ThreadPoolExecutor(max_workers=8) as pool:
            for n, _ in enumerate(pool.map(prefetch, sorted(starts)), 1):
                if n % 50 == 0:
                    print(
                        json.dumps({"care_blocks": n, "total": len(starts)}), flush=True
                    )
        records = []
        for event in selected:
            path = directory / (event["farm"] + "-" + event["event_id"] + ".csv")
            # ZipFile.read验证成员CRC；文件名只用协议中数值ID，不按归档路径解压。
            path.write_bytes(archive.read(event["member"]))
            records.append(
                {
                    **event,
                    "file": path.name,
                    "sha256": sha256(path),
                    "zip_crc32": info[event["member"]].CRC,
                }
            )
        (directory / "source.json").write_text(
            json.dumps(
                {
                    "record": 15846963,
                    "license": "CC-BY-SA-4.0",
                    "verification": "selected ZIP member CRC32 + local SHA256; whole ZIP MD5 not verified",
                    "files": records,
                },
                indent=2,
            )
        )
        print(
            json.dumps(
                {
                    "care_cases": len(records),
                    "compressed_subset_bytes": len(starts) * remote.block_size,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("care", "electrode"))
    args = parser.parse_args()
    care() if args.dataset == "care" else electrode()
