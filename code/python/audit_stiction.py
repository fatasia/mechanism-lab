"""现场黏滞记录审计；文件夹标签、回路身份、时间重叠与离线标志分开保留。"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from phm_core.artifacts import capture_source, new_run

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/plant-stiction"


def parse_record(content):
    try:
        text, encoding = content.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        text, encoding = content.decode("gb18030"), "gb18030"
    lines = text.splitlines()
    heading = lines[0].split("\t")
    metadata, tags, descriptions = [line.split("\t") for line in lines[1:4]]
    if heading[0] != "PROCESSDATA" or len(metadata) != 4 or tags[0] != "tagname":
        raise ValueError("非预期PROCESSDATA格式")
    columns, interval, expected_rows = (
        int(metadata[0]),
        float(metadata[1]),
        int(metadata[3]),
    )
    if columns != 8 or len(tags) != 9 or len(descriptions) != 9:
        raise ValueError("预期8路信号与时间列")
    rows = [line.split("\t") for line in lines[5:] if line.strip()]
    if any(len(row) != 9 for row in rows):
        raise ValueError("记录列数不一致")
    timestamps = np.array([r[0] for r in rows], dtype="datetime64[s]")
    values = np.asarray([r[1:] for r in rows], dtype=np.float64)
    times = timestamps.astype("int64")
    return (
        values,
        times,
        {
            "loop": heading[1].removesuffix(" Data"),
            "encoding": encoding,
            "tags": tags[1:],
            "descriptions": descriptions[1:],
            "sample_interval_seconds": interval,
            "declared_rows": expected_rows,
            "rows": len(rows),
            "start": str(timestamps[0]),
            "end": str(timestamps[-1]),
            "metadata_start_matches": timestamps[0] == np.datetime64(metadata[2]),
        },
    )


def run():
    source = json.loads((DATA / "source.json").read_text(encoding="utf-8"))
    run_id, out = new_run(ROOT / "outputs", "stiction-audit")
    records, duplicates, groups = [], defaultdict(list), defaultdict(list)
    for identifier, item in enumerate(source["files"]):
        relative = Path(item["label_folder"]) / item["name"]
        content = (DATA / relative).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"数据哈希改变: {relative}")
        values, times, metadata = parse_record(content)
        delta = np.diff(times)
        finite = np.isfinite(values).all()
        row = {
            "id": identifier,
            "path": relative.as_posix(),
            "sha256": digest,
            "label": int(item["label_folder"] == "stiction_data"),
            **metadata,
            "metadata_start_matches": bool(metadata["metadata_start_matches"]),
            "finite": bool(finite),
            "nonpositive_intervals": int(np.sum(delta <= 0)),
            "off_nominal_intervals": int(
                np.sum(delta != metadata["sample_interval_seconds"])
            ),
            "mode_values": np.unique(values[:, 3]).tolist(),
            "offline_nonzero_counts": np.count_nonzero(values[:, 4:], axis=0).tolist(),
            "constant_channels": np.flatnonzero(np.ptp(values, axis=0) == 0).tolist()
            if finite
            else [],
            "valid": bool(
                finite
                and np.all(delta == metadata["sample_interval_seconds"])
                and metadata["rows"] == metadata["declared_rows"]
                and metadata["metadata_start_matches"]
            ),
        }
        records.append(row)
        duplicates[digest].append(identifier)
        groups[row["loop"]].append(row)
    overlaps = []
    for loop, items in groups.items():
        ordered = sorted(items, key=lambda r: r["start"])
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                if right["start"] <= left["end"]:
                    overlaps.append({"loop": loop, "ids": [left["id"], right["id"]]})
    report = {
        "run_id": run_id,
        "source": source["source"],
        "doi": source["doi"],
        "license": source["license"],
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", "audit_stiction.py"
        ),
        "records": records,
        "label_counts": dict(Counter(r["label"] for r in records)),
        "loops": {
            loop: {
                "ids": [r["id"] for r in items],
                "label_counts": dict(Counter(r["label"] for r in items)),
            }
            for loop, items in groups.items()
        },
        "duplicate_content": [v for v in duplicates.values() if len(v) > 1],
        "time_overlaps": overlaps,
        "scope": "folder labels are publisher labels, not independently verified maintenance truth; MV-tag description is OP controller output, not measured valve position; filename Stick1 and header fs fields excluded from future features",
    }
    (out / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "records": len(records),
                "valid": sum(r["valid"] for r in records),
                "loops": len(groups),
                "label_counts": report["label_counts"],
                "overlaps": len(overlaps),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
