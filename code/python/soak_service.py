"""隔离HTTP服务长稳与恢复演练；报告持续写入，不把启动当成通过。"""

import argparse
import math
import hashlib
import json
import os
import socket
import sqlite3
import sys
import time
import urllib.request
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from phm_core.artifacts import run_path
from phm_runtime.store import StateStore
from phm_runtime.observability import storage_snapshot
from service_process import start_service, stop_service

ROOT = Path(__file__).resolve().parents[2]


def request(url, payload=None, timeout=30):
    req = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def run(seconds, interval, port):
    if port == 8765:
        raise ValueError("长稳测试不得占用主演示端口")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    out = (
        ROOT
        / "work/validation"
        / datetime.now(UTC).strftime("http-soak-%Y%m%dT%H%M%S%f")
    )
    out.mkdir(parents=True)
    source_files = [
        Path(__file__).resolve(),
        ROOT / "code/python/service_process.py",
        ROOT / "code/python/serve_demo.py",
        *sorted((ROOT / "code/python/src").rglob("*.py")),
    ]
    source_hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_files
    }
    (out / "source-manifest.json").write_text(
        json.dumps(source_hashes, indent=2), encoding="utf-8"
    )
    with ZipFile(out / "source.zip", "w", ZIP_DEFLATED) as archive:
        for path in source_files:
            archive.write(path, str(path.relative_to(ROOT)))
        archive.writestr("source-manifest.json", json.dumps(source_hashes, indent=2))
    database = out / "runtime.sqlite"
    env = {
        **os.environ,
        "MECHANISM_RUNTIME_DB": str(database),
        "MECHANISM_MODE": "local",
    }
    url = f"http://127.0.0.1:{port}"
    directory = run_path(ROOT / "outputs", "valve")
    manifest = json.loads((directory / "bundle/manifest.json").read_text())
    with np.load(directory / "parity.npz") as archive:
        raw = archive["raw"].copy()
    metrics = {
        "started": datetime.now(UTC).isoformat(),
        "required_seconds": seconds,
        "interval_seconds": interval,
        "status": "running",
        "requests": 0,
        "retries": 0,
        "restarts": 0,
        "restores": 0,
        "startup_probes": [],
        "scope": "single client loopback HTTP, real valve ONNX, isolated SQLite; not site operation or memory acceptance",
    }
    started = time.monotonic()
    latencies = deque(maxlen=10000)
    metrics["latency_scope"] = (
        "last 10000 requests; bounded reservoir, not lifetime percentile"
    )
    metrics["max_wal_bytes"] = 0
    metrics["max_backup_bytes"] = 0
    log = (out / "server.log").open("ab")
    server = None

    def start():
        proc, readiness = start_service(
            [
                sys.executable,
                "-u",
                str(ROOT / "code/python/serve_demo.py"),
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env=env,
            log=log,
            probe=lambda: request(url + "/api/health", timeout=1),
        )
        metrics["startup_probes"].append(readiness)
        return proc

    def save():
        metrics["elapsed_seconds"] = time.monotonic() - started
        if database.exists():
            metrics["storage"] = storage_snapshot(database)
            metrics["max_wal_bytes"] = max(
                metrics["max_wal_bytes"], metrics["storage"]["wal_bytes"]
            )
        if latencies:
            metrics["latency_ms"] = {
                f"p{p}": float(np.percentile(latencies, p)) for p in (50, 95, 99)
            }
        tmp = out / "progress.tmp"
        tmp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        tmp.replace(out / "progress.json")

    try:
        server = start()
        request(
            url + "/api/runtime/activate/valve",
            {"run_id": directory.name, "model": "ae", "mode": "shadow"},
        )
        while time.monotonic() - started < seconds:
            if (out / "stop.request").exists():
                metrics["status"] = "cancelled"
                break
            i = metrics["requests"]
            payload = {
                "task": "valve",
                "entity": f"soak-{i % 4}",
                "sequence": i // 4,
                "values": raw[i % len(raw)].tolist(),
                "channels": manifest["channels"],
                "units": manifest["units"],
                "sample_rate_hz": manifest["sampleRateHz"],
            }
            tick = time.perf_counter()
            result = request(url + "/api/runtime/predict", payload)
            if result["status"] != "scored" or result["duplicate"]:
                raise AssertionError("窗口未正确评分")
            latencies.append((time.perf_counter() - tick) * 1000)
            metrics["requests"] += 1
            if i % 10 == 0:
                if not request(url + "/api/runtime/predict", payload)["duplicate"]:
                    raise AssertionError("重复窗口未去重")
                metrics["retries"] += 1
            # 每100窗口真实重启API并检查已有输入去重，备份再恢复到独立文件。
            if metrics["requests"] % 100 == 0:
                if any(
                    hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
                    for name, digest in source_hashes.items()
                ):
                    raise RuntimeError("测试源码发生变化，停止本轮验收")
                recovery_started = time.monotonic()
                stop_service(server)
                server = start()
                if not request(url + "/api/runtime/predict", payload)["duplicate"]:
                    raise AssertionError("重启后状态丢失")
                metrics["restarts"] += 1
                metrics["max_restart_seconds"] = max(
                    metrics.get("max_restart_seconds", 0),
                    time.monotonic() - recovery_started,
                )
                backup = out / "restore-latest.sqlite"
                StateStore(database).backup(backup)
                with sqlite3.connect(backup) as db:
                    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                    assert (
                        db.execute("SELECT count(*) FROM predictions").fetchone()[0]
                        == metrics["requests"]
                    )
                metrics["restores"] += 1
                metrics["max_backup_bytes"] = max(
                    metrics["max_backup_bytes"], backup.stat().st_size
                )
                if (
                    backup.with_name(backup.name + "-wal").exists()
                    and backup.with_name(backup.name + "-wal").stat().st_size
                ):
                    raise AssertionError("备份仍依赖WAL sidecar")
            save()
            time.sleep(max(0, min(interval, seconds - (time.monotonic() - started))))
        if metrics["status"] == "running":
            metrics["status"] = "passed"
    except Exception as error:
        metrics.update(status="failed", error=type(error).__name__)
        raise
    finally:
        save()
        if server is not None and server.poll() is None:
            stop_service(server)
        log.close()
        print(str(out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    if (
        not all(math.isfinite(v) for v in (args.seconds, args.interval))
        or args.seconds <= 0
        or args.interval < 0
    ):
        parser.error("seconds须为正，interval不可负")
    run(args.seconds, args.interval, args.port)
