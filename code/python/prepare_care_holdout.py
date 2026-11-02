"""冻结未看过设备的测试；只下载所选成员，复用原有Range缓存。"""

import csv
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from public_data import RemoteZipFile, sha256

ROOT = Path(__file__).resolve().parents[2]


def select(events, excluded, per_class=2):
    used = set(excluded)
    chosen = []
    for label in ("anomaly", "normal"):
        candidates = sorted(events, key=lambda e: int(e["event_id"]))
        count = 0
        for entry in candidates:
            identity = (entry["farm"], str(entry["asset_id"]))
            if entry["event_label"] != label or identity in used:
                continue
            chosen.append(entry)
            used.add(identity)
            count += 1
            if count == per_class:
                break
    return chosen


def run():
    original = ROOT / "data/raw/care-v6"
    destination = ROOT / "data/raw/care-holdout-v6"
    destination.mkdir(parents=True, exist_ok=True)
    pilot = json.loads((original / "source.json").read_text())
    excluded = {(r["farm"], str(r["asset_id"])) for r in pilot["files"]}
    meta = json.loads((original / "publisher.json").read_text())
    item = meta["files"][0]
    remote = RemoteZipFile(
        item["links"]["self"], item["size"], ROOT / "data/raw/care-range-cache-v6"
    )
    with zipfile.ZipFile(remote) as archive:
        protocol_path = destination / "protocol.json"
        if protocol_path.exists():
            protocol = json.loads(protocol_path.read_text())
            if protocol["pilot_manifest_sha256"] != sha256(original / "source.json"):
                raise ValueError("开发集身份已改变")
        else:
            selected = []
            availability = {}
            for farm in "ABC":
                prefix = f"CARE_To_Compare/Wind Farm {farm}/"
                entries = list(
                    csv.DictReader(
                        io.StringIO(
                            archive.read(prefix + "event_info.csv").decode("utf-8-sig")
                        ),
                        delimiter=";",
                    )
                )
                # 只读取身份表或成员首行；不读取候选时序评分、不按结果筛选。
                for e in entries:
                    e["farm"] = farm
                    e["member"] = prefix + "datasets/" + e["event_id"] + ".csv"
                    if "asset_id" not in e:
                        with archive.open(e["member"]) as stream:
                            header = (
                                stream.readline().decode("utf-8-sig").strip().split(";")
                            )
                            first = stream.readline().decode().strip().split(";")
                        e["asset_id"] = first[header.index("asset_id")]
                availability[farm] = {
                    label: len(
                        {
                            str(e["asset_id"])
                            for e in entries
                            if e["event_label"] == label
                            and (farm, str(e["asset_id"])) not in excluded
                        }
                    )
                    for label in ("anomaly", "normal")
                }
                selected += select(entries, excluded)
            if not all(
                any(e["event_label"] == label for e in selected)
                for label in ("anomaly", "normal")
            ):
                raise ValueError("无独立正常或故障资产；不能进行最终评估")
            protocol = {
                "record": 15846963,
                "pilot_manifest_sha256": sha256(original / "source.json"),
                "excluded_assets": sorted([list(x) for x in excluded]),
                "rule": "per farm: smallest event id, up to 2 anomaly then up to 2 normal; unique unseen assets; identity-only selection; fewer cases when assets unavailable",
                "identity_audit": availability,
                "sampling_change_before_features": "initial 2 per class per farm infeasible by identity audit; keep up to 2, never reuse pilot assets",
                "cases": selected,
                "evaluation": "fixed methods PCA residual, regime-normalized PCA, causal CUSUM; quantile .995, min lead 1h, gap6h; all candidates published, no target selection",
                "test_policy": "single final evaluation; after viewing, this set becomes used and cannot validate further tuning",
            }
            protocol_path.write_text(json.dumps(protocol, indent=2))
        starts = set()
        for entry in protocol["cases"]:
            if (entry["farm"], str(entry["asset_id"])) in excluded:
                raise ValueError("与开发集设备重叠")
            info = archive.getinfo(entry["member"])
            start = info.header_offset // remote.block_size * remote.block_size
            end = info.header_offset + info.compress_size + 2048
            starts.update(range(start, min(end, remote.size), remote.block_size))
        missing = [s for s in starts if not (remote.cache_dir / f"{s}.bin").exists()]
        print(
            json.dumps(
                {
                    "frozen_cases": len(protocol["cases"]),
                    "additional_cache_upper_bound_MiB": len(missing),
                }
            ),
            flush=True,
        )

        def fetch(start):
            reader = RemoteZipFile(remote.url, remote.size, remote.cache_dir)
            reader.seek(start)
            reader.read(min(reader.block_size, remote.size - start))

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(fetch, sorted(starts)))
        records = []
        for entry in protocol["cases"]:
            path = destination / f"{entry['farm']}-{entry['event_id']}.csv"
            path.write_bytes(archive.read(entry["member"]))
            records.append(
                {
                    **entry,
                    "file": path.name,
                    "sha256": sha256(path),
                    "zip_crc32": archive.getinfo(entry["member"]).CRC,
                }
            )
        (destination / "source.json").write_text(
            json.dumps(
                {
                    "record": 15846963,
                    "license": "CC-BY-SA-4.0",
                    "protocol_sha256": sha256(protocol_path),
                    "files": records,
                },
                indent=2,
            )
        )
        print(json.dumps({"holdout_downloaded": len(records)}), flush=True)


if __name__ == "__main__":
    run()
