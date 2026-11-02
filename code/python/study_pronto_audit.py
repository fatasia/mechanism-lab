"""PRONTO 数据可用性审计：类别与采集日期、剂量与运行时间是否混杂。

只读压缩包头部信息，不解压、不训练。结论回答一件事：这份公开数据能不能用来
验证“故障 vs 正常”检出或多模态融合的准确率。
"""

import collections
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from phm_core.artifacts import capture_source, new_run

ROOT = Path(__file__).resolve().parents[2]
ZIP = ROOT / "data/raw/pronto/PRONTO benchmark case study.zip"
BASE = "PRONTO benchmark case study/"
LAYOUT = re.compile(r"(C\d [^/]+)/(Test\d+)/([^/]+)/(.+)")
NON_FAULT = "C0 Normal and Slugging conditions"


def burst_header(archive, path):
    """突发文件头部的日期与起始分钟；空文件返回 None，并作为异常记录。"""
    with archive.open(BASE + path) as stream:
        lines = []
        for _ in range(3):
            raw = stream.readline()
            if not raw:
                return None
            lines.append(raw.decode("utf-8", "replace").strip().split(",")[0])
    day, clock = lines[1], lines[2]
    month, date, year = (int(v) for v in re.match(r"(\d+)/(\d+)/(\d+)", day).groups())
    hours, minutes = (int(v) for v in clock.split(":"))
    return {
        "date": f"{year:04d}-{date:02d}-{month:02d}",
        "minute_of_day": hours * 60 + minutes,
    }


def inventory(archive):
    tests = collections.defaultdict(lambda: {"hf": [], "ut": [], "process": []})
    for info in archive.infolist():
        match = LAYOUT.match(info.filename[len(BASE) :])
        if match is None or info.is_dir():
            continue
        cls, test, kind, name = match.groups()
        entry = tests[(cls, test)]
        entry["class"] = cls
        if kind == "High Frequency Pressure":
            entry["hf"].append(name)
        elif kind == "Ultrasonic Sensor":
            entry["ut"].append(name)
        elif kind == "Process Data":
            entry["process"].append(name)
    return tests


def setpoint_of(name, test):
    tail = re.sub(rf".*{test}_", "", name)[:-4]
    return float(tail) if re.fullmatch(r"\d+(\.\d+)?", tail) else None


def audit_test(archive, cls, test, entry):
    bursts, anomalies = [], []
    for name in sorted(entry["hf"]):
        header = burst_header(archive, f"{cls}/{test}/High Frequency Pressure/{name}")
        if header is None:
            anomalies.append({"file": name, "problem": "empty or truncated header"})
            continue
        bursts.append({"file": name, "setpoint": setpoint_of(name, test), **header})
    numeric = [b for b in bursts if b["setpoint"] is not None]
    order = None
    if len(numeric) >= 4 and len({b["date"] for b in numeric}) == 1:
        rho = spearmanr(
            [b["setpoint"] for b in numeric], [b["minute_of_day"] for b in numeric]
        )
        order = {
            "spearman_setpoint_vs_start_time": None
            if np.isnan(rho.statistic)
            else float(rho.statistic),
            "levels": len(numeric),
        }
    return {
        "class": cls,
        "hf_files": len(entry["hf"]),
        "ultrasonic_files": len(entry["ut"]),
        "process_files": sorted(set(entry["process"])),
        "dates": sorted({b["date"] for b in bursts}),
        "has_zero_setpoint": any(b["setpoint"] == 0 for b in numeric),
        "dose_time_order": order,
        "bursts": bursts,
        "anomalies": anomalies,
    }


def run():
    with zipfile.ZipFile(ZIP) as archive:
        tests = inventory(archive)
        per_test = {
            test: audit_test(archive, cls, test, entry)
            for (cls, test), entry in sorted(
                tests.items(), key=lambda kv: int(kv[0][1][4:])
            )
        }
    by_date = collections.defaultdict(set)
    for result in per_test.values():
        for date in result["dates"]:
            by_date[date].add(result["class"])
    fault_dates = {d for d, classes in by_date.items() if classes - {NON_FAULT}}
    normal_dates = {d for d, classes in by_date.items() if NON_FAULT in classes}
    run_id, output = new_run(ROOT / "outputs", "pronto-audit")
    report = {
        "run_id": run_id,
        "scope": "availability audit of PRONTO multiphase-flow benchmark; headers only, no modelling",
        "source_hash": capture_source(
            ROOT / "code/python", output / "source.zip", "study_pronto_audit.py"
        ),
        "zip_sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
        "tests": per_test,
        "classes_by_date": {d: sorted(c) for d, c in sorted(by_date.items())},
        "confounds": {
            "dates_with_fault_data": sorted(fault_dates),
            "dates_with_non_fault_data": sorted(normal_dates),
            "any_date_has_both": bool(fault_dates & normal_dates),
            "tests_without_hf_files": [
                t for t, r in per_test.items() if r["hf_files"] == 0
            ],
            "tests_with_zero_setpoint": [
                t for t, r in per_test.items() if r["has_zero_setpoint"]
            ],
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "output": str(output)}))


if __name__ == "__main__":
    run()
