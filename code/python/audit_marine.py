"""审计发动机公开归档的时序、标签、缺测与索引；不训练模型、不改写数据。"""

import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from phm_core.artifacts import capture_source, new_run

ROOT = Path(__file__).resolve().parents[2]


def run():
    directory = ROOT / "data/raw/marine-engine"
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    archive_path = directory / source["file"]["name"]
    if (
        hashlib.sha256(archive_path.read_bytes()).hexdigest()
        != source["file"]["sha256"]
    ):
        raise ValueError("发动机归档摘要不一致")
    run_id, out = new_run(ROOT / "outputs", "marine-audit")
    report = {
        "run_id": run_id,
        "doi": source["doi"],
        "files": [],
        "source_hash": capture_source(
            ROOT / "code/python", out / "audit-source.zip", "audit_marine.py"
        ),
        "scope": "induced marine engine fault experiments; not lithium equipment, valve transfer, or natural RUL ground truth",
    }
    with ZipFile(archive_path) as archive:
        prefix = "Marine_Engine_Fault_Data/"
        index = list(
            csv.DictReader(
                io.StringIO(archive.read(prefix + "dataset_index.csv").decode("utf-8"))
            )
        )
        for entry in index:
            data = archive.read(prefix + entry["file_name"])
            reader = csv.reader(io.StringIO(data.decode("utf-8")))
            names, symbols, units = next(reader), next(reader), next(reader)
            values = np.asarray(
                [[float(v) if v else np.nan for v in row] for row in reader]
            )
            if values.shape != (int(entry["data_rows"]), int(entry["columns"])):
                raise ValueError(f"索引尺寸不匹配: {entry['file_name']}")
            if len(names) != len(set(names)):
                raise ValueError("完整变量名存在重复")
            time_column = "Time_rel" if entry["schema_type"] == "scenario" else "Time"
            time = values[:, names.index(time_column)]
            item = {
                "file": entry["file_name"],
                "sha256": hashlib.sha256(data).hexdigest(),
                "rows": len(values),
                "columns": len(names),
                "channels": [
                    {"name": name, "symbol": symbol, "published_unit": unit}
                    for name, symbol, unit in zip(names, symbols, units)
                ],
                "scenario": entry["scenario"],
                "load": entry["nominal_load"],
                "backward_time_steps": int((np.diff(time) < 0).sum()),
                "duplicate_time_steps": int((np.diff(time) == 0).sum()),
                "nonfinite_time": int((~np.isfinite(time)).sum()),
                "duration_s": float(time[-1] - time[0]),
                "median_time_step_s": float(np.median(np.diff(time))),
                "missing_columns": {
                    name: int(np.isnan(values[:, i]).sum())
                    for i, name in enumerate(names)
                    if np.isnan(values[:, i]).any()
                },
                "infinite_values": int(np.isinf(values).sum()),
                "raw_voltage_channels": [
                    name for name, unit in zip(names, units) if unit == "V"
                ],
            }
            if "Anomaly State" in names:
                labels = values[:, names.index("Anomaly State")]
                if not np.isin(labels, [0, 1]).all():
                    raise ValueError("异常状态并非完整二值标签")
                boundaries = np.flatnonzero(np.diff(labels) != 0) + 1
                item.update(
                    normal_rows=int((labels == 0).sum()),
                    fault_rows=int((labels == 1).sum()),
                    transitions=[
                        {
                            "row": int(i),
                            "time_s": float(time[i]),
                            "state": int(labels[i]),
                        }
                        for i in boundaries
                    ],
                )
            report["files"].append(item)
    report["summary"] = {
        "files": len(report["files"]),
        "rows": sum(v["rows"] for v in report["files"]),
        "backward_time_steps": sum(v["backward_time_steps"] for v in report["files"]),
        "duplicate_time_steps": sum(v["duplicate_time_steps"] for v in report["files"]),
        "all_fault_scenarios": sum(v.get("normal_rows") == 0 for v in report["files"]),
    }
    (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, **report["summary"]}))


if __name__ == "__main__":
    run()
