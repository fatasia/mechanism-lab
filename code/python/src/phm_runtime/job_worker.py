"""独立工作进程：API 重启后继续训练，取消只终止自己启动的子进程。"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from phm_runtime.jobs import JobStore
from phm_runtime.processes import spawn_process, stop_process
from phm_runtime.store import StateStore


def run(database, job_id, root):
    jobs = JobStore(StateStore(Path(database)))
    job = jobs.claim(job_id)
    process = None
    started = time.monotonic()
    try:
        if not jobs.heartbeat(job_id):
            jobs.finish(job_id, "cancelled", detail="cancelled_before_launch")
            return
        path = Path(job["log_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as log:
            process = spawn_process(
                json.loads(job["command"]),
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            while process.poll() is None:
                keep_running = jobs.heartbeat(job_id)
                expired = time.monotonic() - started >= job["timeout"]
                if not keep_running or expired:
                    stop_process(process)
                    jobs.finish(
                        job_id,
                        "timed_out" if expired else "cancelled",
                        process.returncode,
                    )
                    return
                time.sleep(0.2)
            jobs.finish(
                job_id,
                "complete" if process.returncode == 0 else "failed",
                process.returncode,
            )
    except Exception:
        if process is not None and process.poll() is None:
            stop_process(process)
        jobs.finish(job_id, "failed", detail="worker_error; inspect worker log")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    run(args.database, args.job, args.root)
