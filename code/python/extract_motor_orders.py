"""按键相生成角域谱缓存；同既有2秒窗口和支持集，不使用标签或标称RPM变换。"""

import argparse
import hashlib
import io
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from motor_features import FS, WINDOW, phase_reference
from phm_core.artifacts import capture_source, new_run

ROOT = Path(__file__).resolve().parents[2]
ORDER_GRID = (np.arange(512) + 0.5) / 8


def angular_spectrum(signals, edges):
    """时间域抗混叠后等角度重采样，输出0–64阶的固定512点对数功率谱。"""
    revolutions = len(edges) - 1
    # 每转256点；以窗口最慢转速限制时间域带宽，再重采样，避免直接插值下采样混叠。
    cutoff = min(0.4 * FS, 0.4 * 256 * FS / np.max(np.diff(edges)))
    filtered = sosfiltfilt(butter(6, cutoff, fs=FS, output="sos"), signals, axis=0)
    positions = np.interp(
        np.arange(revolutions * 256) / 256, np.arange(len(edges)), edges
    )
    angular = np.stack(
        [
            np.interp(positions, np.arange(len(signals)), filtered[:, c])
            for c in range(6)
        ],
        axis=1,
    )
    angular -= angular.mean(0)
    taper = np.hanning(len(angular))
    power = np.abs(np.fft.rfft(angular * taper[:, None], axis=0)) ** 2 / (
        256 * np.sum(taper**2)
    )
    power[1:-1] *= 2
    orders = np.fft.rfftfreq(len(angular), d=1 / 256)
    fixed = np.stack([np.interp(ORDER_GRID, orders, power[:, c]) for c in range(6)])
    return np.log10(np.maximum(fixed, 1e-12)).astype("float32")


def representations(window, edges):
    signals = window[:, 3:9] * 10
    raw = angular_spectrum(signals, edges)
    envelope = np.abs(
        hilbert(
            sosfiltfilt(
                butter(4, [500, 5000], fs=FS, btype="bandpass", output="sos"),
                signals[:, :3],
                axis=0,
            ),
            axis=0,
        )
    )
    mixed = angular_spectrum(np.concatenate([envelope, signals[:, 3:]], axis=1), edges)
    return raw, mixed


def run(feature_directory):
    parent_path = feature_directory / "manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    run_id, out = new_run(ROOT / "outputs", "motor-order-features")
    (out / "records").mkdir()
    helper = ROOT / "code/python/motor_features.py"
    shutil.copyfile(helper, out / helper.name)
    protocol = {
        "parent_feature_run": parent["run_id"],
        "window": "same supported nonoverlap 2s windows as parent; no new quality selection",
        "angular": "keyphase 1pulse/rev,256samples/rev,antialias lowpass6 cutoff0.4*256*slowest_rev_hz; all filters zero phase offline",
        "grid_orders": ORDER_GRID.tolist(),
        "power": "Hann one-sided PSD per order, linear interpolation to512bin centers over0-64 orders,log10floor1e-12; interpolated grid not additional spectral resolution",
        "representations": ["order_spectra", "envelope_order_spectra"],
        "envelope": "vibration only500-5000Hz butter4+Hilbert before angular antialias; current retains raw waveform",
        "study_status": "predeclared representation comparison after observing fixed-frequency target degradation; not a fresh blind test",
    }
    report = {
        "run_id": run_id,
        "protocol": protocol,
        "parent_manifest_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "source_hash": capture_source(
            ROOT / "code/python", out / "source.zip", "extract_motor_orders.py"
        ),
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "records": [],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    start = time.monotonic()
    for record in parent["records"]:
        path = ROOT / "data/raw/mcc5-thu-motor/extracted" / record["path"]
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError("原始记录哈希不一致")
        values = np.loadtxt(io.BytesIO(content), delimiter=",")
        del content
        expected_path = feature_directory / record["cache"]
        if (
            hashlib.sha256(expected_path.read_bytes()).hexdigest()
            != record["cache_sha256"]
        ):
            raise ValueError("父窗口缓存哈希不一致")
        with np.load(expected_path) as cache:
            expected_starts = cache["starts_s"]
        raw, envelope, starts = [], [], []
        for index in range(0, len(values) - WINDOW + 1, WINDOW):
            window = values[index : index + WINDOW]
            edges, _ = phase_reference(window[:, 1])
            if edges is None:
                continue
            a, b = representations(window, edges)
            raw.append(a)
            envelope.append(b)
            starts.append(window[0, 0])
        np.testing.assert_array_equal(starts, expected_starts)
        destination = out / record["cache"]
        np.savez_compressed(
            destination,
            order_spectra=np.asarray(raw, dtype="float32").reshape(-1, 6, 512),
            envelope_order_spectra=np.asarray(envelope, dtype="float32").reshape(
                -1, 6, 512
            ),
            starts_s=starts,
        )
        report["records"].append(
            {
                **record,
                "cache_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
        (out / "progress.json").write_text(
            json.dumps(
                {
                    "completed": len(report["records"]),
                    "total": len(parent["records"]),
                    "elapsed_s": time.monotonic() - start,
                }
            ),
            encoding="utf-8",
        )
        print(record["id"], len(starts), flush=True)
    report["elapsed_s"] = time.monotonic() - start
    (out / "manifest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "records": len(report["records"])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    run(parser.parse_args().features.resolve())
