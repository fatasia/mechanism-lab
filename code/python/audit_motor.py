"""MCC5-THU原始记录审计：逐文件哈希、9列格式、真实采样时钟和试验分组。"""

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from phm_core.artifacts import capture_source, new_run

ROOT = Path(__file__).resolve().parents[2]
PATTERN = re.compile(
    r"(.+)_(speed|torque)_circulation_(\d+)Nm_(\d+)rpm(?:_(.*))?\.csv$"
)


def record_metadata(name):
    match = PATTERN.fullmatch(Path(name).name)
    if match is None:
        raise ValueError(f"无法解析试验记录名: {name}")
    label, profile, torque, rpm, run = match.groups()
    return {
        "label": label,
        "profile": profile,
        "nominal_torque_nm": int(torque),
        "nominal_rpm": int(rpm),
        "run_suffix": run,
        "condition": f"{profile}/{torque}Nm/{rpm}rpm",
    }


def array_audit(values):
    if values.ndim != 2 or values.shape[1] != 9:
        raise ValueError(f"需要时间+8信号列，实际{values.shape}")
    finite = np.isfinite(values).all(axis=0)
    dt = np.diff(values[:, 0])

    def nullable(value):
        return float(value) if np.isfinite(value) else None

    return {
        "rows": len(values),
        "columns": values.shape[1],
        "finite_columns": finite.tolist(),
        "time_start_s": nullable(values[0, 0]),
        "time_end_s": nullable(values[-1, 0]),
        "nonincreasing_time_steps": int(np.sum(dt <= 0)),
        "sample_interval_max_error_s": nullable(np.max(np.abs(dt - 1 / 12800))),
        "min": [nullable(v) for v in values.min(axis=0)],
        "max": [nullable(v) for v in values.max(axis=0)],
        "mean": [nullable(v) for v in values.mean(axis=0)],
        "std": [nullable(v) for v in values.std(axis=0)],
        "valid": bool(
            finite.all() and np.all(dt > 0) and np.max(np.abs(dt - 1 / 12800)) <= 1e-7
        ),
    }


def run():
    directory = ROOT / "data/raw/mcc5-thu-motor"
    extracted = json.loads((directory / "extracted.json").read_text(encoding="utf-8"))
    run_id, out = new_run(ROOT / "outputs", "motor-audit")
    report = {
        "run_id": run_id,
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", "audit_motor.py"
        ),
        "archive_sha256": extracted["archive_sha256"],
        "data_doi": "10.17632/6s3dggj9mw.1",
        "paper_doi": "10.1016/j.dib.2026.112583",
        "column_order": [
            "time_s",
            "keyphase_voltage",
            "torque_voltage",
            "vibration_x_voltage",
            "vibration_y_voltage",
            "vibration_z_voltage",
            "current_a_voltage",
            "current_b_voltage",
            "current_c_voltage",
        ],
        "physical_conversion": "Paper: torque 100mV/Nm, vibration 100mV/g, current 100mV/A; multiply raw V by10. Nominal filename loads are not a calibration factor.",
        "dataset_description_difference": "Archive has288 CSV rather than paper282; actual files contain sampling time plus8 channels, unlike mirror README8 without timestamp.",
        "records": [],
    }
    start = time.monotonic()
    for index, entry in enumerate(extracted["files"]):
        path = directory / "extracted" / entry["path"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        values = np.loadtxt(path, delimiter=",")
        report["records"].append(
            {
                "id": index,
                "path": entry["path"],
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
                **record_metadata(path.name),
                **array_audit(values),
            }
        )
        (out / "progress.json").write_text(
            json.dumps(
                {
                    "completed": index + 1,
                    "total": len(extracted["files"]),
                    "elapsed_s": time.monotonic() - start,
                }
            ),
            encoding="utf-8",
        )
        print(f"{index + 1}/{len(extracted['files'])} {path.name}", flush=True)
    report["label_counts"] = dict(Counter(r["label"] for r in report["records"]))
    report["condition_counts"] = dict(
        Counter(r["condition"] for r in report["records"])
    )
    report["duplicate_content"] = [
        key
        for key, count in Counter(r["sha256"] for r in report["records"]).items()
        if count > 1
    ]
    report["elapsed_s"] = time.monotonic() - start
    (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "records": len(report["records"]),
                "valid": sum(r["valid"] for r in report["records"]),
                "duplicate_content": report["duplicate_content"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
