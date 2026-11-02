"""真实训练任务跨 API 重启与取消检查；使用隔离服务和状态库。"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

from soak_service import request

ROOT = Path(__file__).resolve().parents[2]


def run():
    port = 8766
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    out = (
        ROOT
        / "work/validation"
        / datetime.now(UTC).strftime("job-restart-%Y%m%dT%H%M%S%f")
    )
    out.mkdir(parents=True)
    env = {
        **os.environ,
        "MECHANISM_RUNTIME_DB": str(out / "runtime.sqlite"),
        "MECHANISM_MODE": "local",
    }
    url = f"http://127.0.0.1:{port}"
    before = (ROOT / "outputs/valve/latest.json").read_bytes()
    log = (out / "server.log").open("ab")
    server = None
    submitted = False

    def start():
        p = subprocess.Popen(
            [
                sys.executable,
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
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for _ in range(100):
            if p.poll() is not None:
                raise RuntimeError("QA服务启动失败")
            try:
                request(url + "/api/health")
                return p
            except urllib.error.URLError:
                time.sleep(0.1)
        p.terminate()
        p.wait()
        raise TimeoutError("QA服务启动超时")

    try:
        server = start()
        result = request(
            url + "/api/train/valve",
            {
                "profile": "valve-switch-100hz",
                "architecture": "dense",
                "normal_policy": "all",
                "epochs": 200,
            },
        )
        submitted = True
        time.sleep(1)
        first = request(url + "/api/job")
        server.terminate()
        server.wait(timeout=15)
        server = start()
        restarted = request(url + "/api/job")
        assert restarted["id"] == result["id"] and restarted["status"] == "running"
        request(url + "/api/job/cancel", {"job_id": result["id"]})
        for _ in range(100):
            final = request(url + "/api/job")
            if final["status"] != "running":
                break
            time.sleep(0.1)
        assert final["status"] == "cancelled", final
        assert before == (ROOT / "outputs/valve/latest.json").read_bytes()
        report = {
            "job_id": result["id"],
            "before_restart": first["worker_status"],
            "after_restart": restarted["worker_status"],
            "final": final["status"],
            "model_pointer_unchanged": True,
        }
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        if submitted:
            from phm_runtime.jobs import JobStore
            from phm_runtime.store import StateStore

            jobs = JobStore(StateStore(out / "runtime.sqlite"))
            job = jobs.latest()
            if job and job["status"] in ("running", "queued"):
                jobs.cancel(job["id"])
        if server is not None and server.poll() is None:
            server.terminate()
            server.wait(timeout=15)
        log.close()


if __name__ == "__main__":
    run()
