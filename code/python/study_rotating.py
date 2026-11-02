"""在原发布方的 CWRU 12k 故障记录上检查频率证据，并审计重复信号。"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from phm_core.artifacts import capture_source, new_run, publish_run
from phm_core.rotating import ORDERS_6205, envelope_evidence

ROOT = Path(__file__).resolve().parents[2]


def run():
    run_id, out = new_run(ROOT / "outputs", "rotating-study")
    protocol = {
        "sample_rate_hz": 12000,
        "recording_slice": "first 12000 drive-end samples, fixed before analysis",
        "band_hz": [1200, 5000],
        "harmonics": [1, 2, 3],
        "orders": ORDERS_6205,
        "geometry_source": "https://engineering.case.edu/bearingdatacenter/bearing-information",
        "ranking": "sum log1p peak/local-median contrast at three harmonics; includes cage distractor",
        "scope": "12 seeded-fault recordings only; healthy sample rate not independently resolved, no healthy false-alarm claim",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    source = capture_source(
        ROOT / "code/python", out / "train-source.zip", "study_rotating.py"
    )
    examples, seen, duplicates = [], {}, []
    for file in sorted((ROOT / "data/raw/cwru").glob("*.mat")):
        data = loadmat(file, simplify_cells=True)
        signal = next(v for k, v in data.items() if k.endswith("DE_time"))
        fingerprint = hashlib.sha256(np.asarray(signal).tobytes()).hexdigest()
        if fingerprint in seen:
            duplicates.append([seen[fingerprint], file.name])
        seen[fingerprint] = file.name
        code = int(file.stem)
        if code < 105:
            continue
        rpm = float(next(v for k, v in data.items() if k.endswith("RPM")))
        expected = (
            "inner_ring"
            if code < 118
            else "rolling_element"
            if code < 130
            else "outer_ring"
        )
        result = envelope_evidence(signal[:12000], 12000, rpm, ORDERS_6205)
        examples.append(
            {
                "file": file.name,
                "expected": expected,
                "top_candidate": result["candidates"][0]["candidate"],
                "raw": signal[:12000:12].tolist(),
                "raw_sample_rate_hz": 1000,
                "raw_display": "every12th point for display only; spectrum computed on raw12k",
                **result,
            }
        )
    report = {
        "run_id": run_id,
        "source_hash": source,
        "protocol": protocol,
        "duplicate_drive_end_signals": duplicates,
        "records": len(examples),
        "top_candidate_matches": sum(
            e["expected"] == e["top_candidate"] for e in examples
        ),
        "examples": examples,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    # 研究包为确定性 DSP 结果与契约，不把 FFT/滤波包装为已导出的 ONNX。
    from zipfile import ZipFile

    with ZipFile(out / "model-bundle.zip", "w") as package:
        package.write(out / "protocol.json", "protocol.json")
        package.write(out / "metrics.json", "metrics.json")
    publish_run(ROOT / "outputs", "rotating-study", run_id)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "records": report["records"],
                "top_candidate_matches": report["top_candidate_matches"],
                "duplicates": duplicates,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
