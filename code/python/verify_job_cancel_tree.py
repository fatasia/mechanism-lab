"""验证取消训练后实际Python子进程不再写入；仅使用隔离测试任务。"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from phm_runtime.jobs import JobStore
from phm_runtime.store import StateStore

ROOT = Path(__file__).resolve().parents[2]


def run(mode="cancel"):
    out = (
        ROOT
        / "work/validation"
        / datetime.now(UTC).strftime("cancel-tree-%Y%m%dT%H%M%S")
    )
    out.mkdir(parents=True)
    pid_file, marker = out / "training.pid", out / "unexpected-completion.txt"
    child = f"import os,time;from pathlib import Path;Path({str(pid_file)!r}).write_text(str(os.getpid()));time.sleep(4);Path({str(marker)!r}).write_text('completed after cancel')"
    command = [
        sys.executable,
        "-c",
        f"import subprocess,sys;subprocess.run([sys.executable,'-c',{child!r}],check=True)",
    ]
    database = out / "runtime.sqlite"
    jobs = JobStore(StateStore(database))
    job_id = jobs.create(
        "valve", command, out / "training.log", timeout=1 if mode == "timeout" else 30
    )
    with (out / "worker.log").open("wb") as log:
        worker = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "phm_runtime.job_worker",
                "--database",
                str(database),
                "--job",
                job_id,
                "--root",
                str(ROOT),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.monotonic() + 15
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not pid_file.exists():
            raise TimeoutError("测试训练未启动，检查worker.log")
        if mode == "cancel":
            jobs.cancel(job_id)
        worker.wait(timeout=15)
    time.sleep(5)
    expected = "timed_out" if mode == "timeout" else "cancelled"
    report = {
        "status": "passed"
        if not marker.exists()
        and jobs.latest()["status"] == expected
        and worker.returncode == 0
        else "failed",
        "mode": mode,
        "job_status": jobs.latest()["status"],
        "worker_exit_code": worker.returncode,
        "training_pid": int(pid_file.read_text()),
        "wrote_after_cancel": marker.exists(),
        "scope": "isolated nested process lifecycle test, not model training",
        "source_sha256": {
            name: hashlib.sha256(
                (ROOT / "code/python/src/phm_runtime" / name).read_bytes()
            ).hexdigest()
            for name in ("job_worker.py", "processes.py", "jobs.py")
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"directory": str(out), **report}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cancel", "timeout"), default="cancel")
    run(parser.parse_args().mode)
