"""从已审计电机记录生成不跨记录窗口的特征缓存；标签只保存，不参与变换。"""

import argparse
import hashlib
import io
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

from motor_features import FEATURE_NAMES, FS, WINDOW, phase_reference, window_features
from phm_core.artifacts import capture_source, new_run

ROOT = Path(__file__).resolve().parents[2]


def run(audit_path):
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["duplicate_content"] or any(not r["valid"] for r in audit["records"]):
        raise ValueError("原始审计存在无效或重复记录，需先处理")
    run_id, out = new_run(ROOT / "outputs", "motor-features")
    (out / "records").mkdir()
    helper = ROOT / "code/python/motor_features.py"
    shutil.copyfile(helper, out / helper.name)
    protocol = {
        "sample_rate_hz": FS,
        "window_samples": WINDOW,
        "stride_samples": WINDOW,
        "feature_names": FEATURE_NAMES,
        "quality_gate": "2V rising edges, >=8 pulses, median >=300rpm under1pulse/revolution assumption, no inter-pulse interval below10ms",
        "phase_assumption": "one pulse per revolution, check observed max rpm against nominal; no filename speed used in feature calculation",
        "spectrum": "6x1600, Hann periodogram mean over8 adjacent0.5Hz bins, log10 floor1e-12; no Nyquist endpoint",
        "envelope": "500-5000Hz zero-phase bandpass, Hilbert,256angular samples/revolution, fixed3bearing orders x3harmonics; offline2s window",
        "labels": "record filename class, no label-dependent preprocessing or target adaptation",
    }
    report = {
        "run_id": run_id,
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "audit_path": str(audit_path.relative_to(ROOT)),
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", "extract_motor_features.py"
        ),
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "protocol": protocol,
        "records": [],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    start = time.monotonic()
    for record in audit["records"]:
        path = ROOT / "data/raw/mcc5-thu-motor/extracted" / record["path"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"原始数据已改变: {path.name}")
        values = np.loadtxt(io.BytesIO(raw), delimiter=",")
        del raw
        vectors, spectra, starts, speeds, cvs, qualities = [], [], [], [], [], []
        for index in range(0, len(values) - WINDOW + 1, WINDOW):
            window = values[index : index + WINDOW]
            edges, quality = phase_reference(window[:, 1])
            qualities.append({"start_s": float(window[0, 0]), **quality})
            if edges is None:
                continue
            vector, spectrum = window_features(window, edges)
            vectors.append(vector)
            spectra.append(spectrum)
            starts.append(window[0, 0])
            speeds.append(quality["rpm"])
            cvs.append(quality["period_cv"])
        destination = out / "records" / f"{record['id']:04d}.npz"
        np.savez_compressed(
            destination,
            features=np.asarray(vectors, dtype="float32").reshape(
                -1, len(FEATURE_NAMES)
            ),
            spectra=np.asarray(spectra, dtype="float32").reshape(-1, 6, 1600),
            starts_s=starts,
            rpm=speeds,
            period_cv=cvs,
        )
        report["records"].append(
            {
                **{
                    key: record[key]
                    for key in (
                        "id",
                        "label",
                        "profile",
                        "nominal_torque_nm",
                        "nominal_rpm",
                        "condition",
                        "path",
                        "sha256",
                    )
                },
                "cache": str(destination.relative_to(out)),
                "cache_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "active_windows": len(vectors),
                "quality_counts": dict(Counter(q["status"] for q in qualities)),
                "windows": qualities,
                "unused_tail_samples": len(values) % WINDOW,
            }
        )
        (out / "progress.json").write_text(
            json.dumps(
                {
                    "completed": len(report["records"]),
                    "total": len(audit["records"]),
                    "elapsed_s": time.monotonic() - start,
                }
            ),
            encoding="utf-8",
        )
        print(f"{record['id']} {path.name}: {len(vectors)} active windows", flush=True)
    report["elapsed_s"] = time.monotonic() - start
    (out / "manifest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "records": len(report["records"]),
                "active_windows": sum(r["active_windows"] for r in report["records"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    run(parser.parse_args().audit.resolve())
