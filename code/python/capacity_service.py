"""独立Windows HTTP并发/内存长测，不重启服务，不使用主业务库。"""

import argparse
import ctypes
import hashlib
import json
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from http_benchmark import HttpSession
from phm_core.artifacts import run_path
from service_process import start_service, stop_service
from soak_service import request

ROOT = Path(__file__).resolve().parents[2]


def memory_usage(pid):
    """读取监听服务进程的工作集/私有提交；不是venv启动器的内存。"""

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
                "PrivateUsage",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), ctypes.sizeof(counters)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "working_set_bytes": counters.WorkingSetSize,
            "private_bytes": counters.PrivateUsage,
            "peak_working_set_bytes": counters.PeakWorkingSetSize,
        }
    finally:
        kernel.CloseHandle(handle)


def run(seconds, workers, interval, port, blas_threads=1, connections="keep-alive"):
    if os.name != "nt":
        raise ValueError("本入口验证Windows进程内存")
    if port in (8765, 8767):
        raise ValueError("不能占用主服务或72小时长测端口")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    out = (
        ROOT
        / "work/validation"
        / datetime.now(UTC).strftime("capacity-%Y%m%dT%H%M%S%f")
    )
    out.mkdir(parents=True)
    files = [
        Path(__file__).resolve(),
        ROOT / "code/python/soak_service.py",
        ROOT / "code/python/http_benchmark.py",
        ROOT / "code/python/service_process.py",
        *sorted((ROOT / "code/python/src").rglob("*.py")),
    ]
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    (out / "source-manifest.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )
    with ZipFile(out / "source.zip", "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(ROOT)))
        archive.writestr("source-manifest.json", json.dumps(hashes, indent=2))
    directory = run_path(ROOT / "outputs", "valve")
    manifest = json.loads(
        (directory / "bundle/manifest.json").read_text(encoding="utf-8")
    )
    with np.load(directory / "parity.npz") as archive:
        raw = archive["raw"].copy()
    env = {
        **os.environ,
        "MECHANISM_RUNTIME_DB": str(out / "runtime.sqlite"),
        "MECHANISM_MODE": "local",
    }
    if blas_threads:
        env.update(
            {
                name: str(blas_threads)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            }
        )
    url = f"http://127.0.0.1:{port}"
    log = (out / "server.log").open("ab")
    server, readiness = start_service(
        [
            sys.executable,
            "-u",
            "-m",
            "uvicorn",
            "phm_demo.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env=env,
        log=log,
        probe=lambda: request(url + "/api/health", timeout=1),
    )
    metrics = {
        "status": "starting",
        "seconds_required": seconds,
        "workers": workers,
        "blas_threads": blas_threads or "inherited",
        "interval_per_worker": interval,
        "connections": connections,
        "startup_probes": readiness,
        "scope": "Windows loopback, concurrent clients, one non-restarted API process, real ONNX and isolated SQLite; co-resident other work",
        "limits": {
            "p95_ms": 250,
            "p99_ms": 1000,
            "peak_private_bytes": 512 * 1024**2,
            "private_growth_bytes": 32 * 1024**2,
        },
    }
    latencies = []
    samples = []
    count = 0
    duplicates = 0
    connection_counts = {}
    worker_stages = {}
    client_errors = []
    lock = threading.Lock()
    stop = threading.Event()
    started = None

    def save():
        with lock:
            local_count = count
            local_duplicates = duplicates
            timings = np.asarray(latencies)
        metrics.update(
            requests=local_count,
            duplicates=local_duplicates,
            elapsed_seconds=0 if started is None else time.monotonic() - started,
        )
        if len(timings):
            metrics["latency_ms"] = {
                f"p{q}": float(np.percentile(timings, q)) for q in (50, 95, 99)
            }
        temporary = out / "progress.tmp"
        temporary.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        temporary.replace(out / "progress.json")

    def client_loop(worker, send):
        nonlocal count, duplicates
        sequence = 0
        while not stop.is_set() and time.monotonic() - started < seconds:
            tick = time.perf_counter()
            payload = {
                "task": "valve",
                "entity": f"capacity-{worker}",
                "sequence": sequence,
                "values": raw[(sequence + worker) % len(raw)].tolist(),
                "channels": manifest["channels"],
                "units": manifest["units"],
                "sample_rate_hz": manifest["sampleRateHz"],
            }
            worker_stages[worker] = {"stage": "new_prediction", "sequence": sequence}
            result = send("/api/runtime/predict", payload)
            if result["status"] != "scored" or result["duplicate"]:
                raise AssertionError("并发评分返回异常")
            elapsed = (time.perf_counter() - tick) * 1000
            with lock:
                count += 1
                latencies.append(elapsed)
            if sequence % 10 == 0:
                worker_stages[worker] = {
                    "stage": "duplicate_check",
                    "sequence": sequence,
                }
                if not send("/api/runtime/predict", payload)["duplicate"]:
                    raise AssertionError("重试去重失败")
                with lock:
                    duplicates += 1
            sequence += 1
            stop.wait(max(0, interval - (time.perf_counter() - tick)))

    def client(worker):
        session = HttpSession(port)
        calls = 0

        def send(path, payload):
            nonlocal calls
            calls += 1
            return (
                session.request(path, payload)
                if connections == "keep-alive"
                else request(url + path, payload)
            )

        try:
            client_loop(worker, send)
        except Exception as error:
            client_errors.append(
                {"worker": worker, **worker_stages[worker], "error": repr(error)}
            )
            raise
        finally:
            session.close()
            connection_counts[str(worker)] = (
                session.connection.connections if connections == "keep-alive" else calls
            )

    try:
        pid_file = out / "listener-pids.json"
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                f"$listeners = @(Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort {port} -State Listen | Select-Object -ExpandProperty OwningProcess -Unique); ConvertTo-Json -InputObject $listeners | Set-Content -LiteralPath '{pid_file}' -Encoding UTF8",
            ],
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=True,
            capture_output=True,
        )
        pids = json.loads(pid_file.read_text(encoding="utf-8-sig"))
        if len(pids) != 1:
            raise RuntimeError("隔离端口未对应唯一监听进程")
        pid = int(pids[0])
        metrics["pid"] = pid
        request(
            url + "/api/runtime/activate/valve",
            {"run_id": directory.name, "model": "ae", "mode": "shadow"},
        )
        started = time.monotonic()
        metrics["status"] = "running"
        save()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(client, i) for i in range(workers)]
            try:
                while time.monotonic() - started < seconds:
                    if (out / "stop.request").exists():
                        metrics["status"] = "cancelled"
                        break
                    for future in futures:
                        if future.done():
                            future.result()
                    sample = {
                        "elapsed_seconds": time.monotonic() - started,
                        **memory_usage(pid),
                    }
                    samples.append(sample)
                    with (out / "memory.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(sample) + "\n")
                    save()
                    stop.wait(min(5, max(0, seconds - (time.monotonic() - started))))
            finally:
                stop.set()
            for future in futures:
                future.result()
        if metrics["status"] != "cancelled":
            with sqlite3.connect(out / "runtime.sqlite") as db:
                assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                assert (
                    db.execute("SELECT count(*) FROM predictions").fetchone()[0]
                    == count
                )
            if any(
                hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
                for name, digest in hashes.items()
            ):
                raise RuntimeError("测试源码发生变化")
            save()
            # 前/后各5分钟中位数，短检用首/末样本；短检不能替代长时内存增长验证。
            first = [
                s["private_bytes"]
                for s in samples
                if s["elapsed_seconds"] <= min(300, seconds / 4)
            ]
            last = [
                s["private_bytes"]
                for s in samples
                if s["elapsed_seconds"] >= seconds - min(300, seconds / 4)
            ] or [samples[-1]["private_bytes"]]
            memory = {
                "peak_private_bytes": max(s["private_bytes"] for s in samples),
                "private_growth_bytes": float(np.median(last) - np.median(first)),
            }
            metrics["memory"] = memory
            metrics["checks"] = {
                "p95": metrics["latency_ms"]["p95"] <= 250,
                "p99": metrics["latency_ms"]["p99"] <= 1000,
                "private_peak": memory["peak_private_bytes"]
                <= metrics["limits"]["peak_private_bytes"],
                "private_growth": memory["private_growth_bytes"]
                <= metrics["limits"]["private_growth_bytes"],
                "state_count_integrity": True,
            }
            metrics["status"] = (
                "passed" if all(metrics["checks"].values()) else "failed_limits"
            )
    except Exception as error:
        metrics.update(status="failed", error=repr(error))
        raise
    finally:
        stop.set()
        metrics["worker_tcp_connections"] = connection_counts
        metrics["client_errors"] = client_errors
        if started is not None:
            with sqlite3.connect(out / "runtime.sqlite") as db:
                metrics["database_predictions"] = db.execute(
                    "SELECT count(*) FROM predictions"
                ).fetchone()[0]
                metrics["database_minus_acknowledged"] = (
                    metrics["database_predictions"] - count
                )
        save()
        np.savez_compressed(out / "latencies.npz", latency_ms=latencies)
        if server.poll() is None:
            stop_service(server)
        log.close()
        print(str(out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument(
        "--connections", choices=("keep-alive", "close"), default="keep-alive"
    )
    parser.add_argument(
        "--blas-threads", type=int, default=1, help="0保留环境值用于旧配置对照"
    )
    args = parser.parse_args()
    if (
        args.seconds <= 0
        or args.workers < 1
        or args.interval < 0
        or args.blas_threads < 0
    ):
        parser.error("时长/并发须为正，间隔不可负")
    run(
        args.seconds,
        args.workers,
        args.interval,
        args.port,
        args.blas_threads,
        args.connections,
    )
