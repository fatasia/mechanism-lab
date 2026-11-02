"""干净Git克隆验收：复用依赖环境，但代码/数据/训练/服务/备份全部在新克隆中。"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


def run(destination):
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("验收目标已存在；不覆盖工作区")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--no-local", str(ROOT), str(destination)],
        check=True,
        capture_output=True,
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(destination / "code/python/src"),
        "PYTHONIOENCODING": "utf-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MECHANISM_MODE": "local",
    }
    for key in (
        "MECHANISM_TRUSTED_MODEL_KEY",
        "MECHANISM_RUNTIME_DB",
        "MECHANISM_SIGNING_PASSWORD",
    ):
        env.pop(key, None)
    logs = destination / "work/clean-verification"
    logs.mkdir(parents=True)

    def command(name, args, cwd=destination, timeout=240):
        result = subprocess.run(
            args, cwd=cwd, env=env, capture_output=True, timeout=timeout
        )
        (logs / (name + ".log")).write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"干净克隆步骤失败: {name}，见{logs}")
        print(json.dumps({"clean_step": name, "passed": True}), flush=True)
        return result.stdout

    origin = json.loads(
        command(
            "module-origin",
            [
                sys.executable,
                "-c",
                "import phm_core,phm_demo,phm_runtime,json; print(json.dumps([p.__file__ for p in [phm_core,phm_demo,phm_runtime]]))",
            ],
        )
    )
    if not all(Path(p).resolve().is_relative_to(destination) for p in origin):
        raise ValueError("导入了原工作区代码")
    if (destination / "data").exists() or (destination / "outputs").exists():
        raise ValueError("干净源码克隆意外包含数据或产物")
    command("pytest", [sys.executable, "-m", "pytest", "code/python/tests", "-q"])
    frontend = destination / "code/ts/demo"
    command(
        "npm-ci", [shutil.which("npm.cmd") or "npm", "ci", "--prefer-offline"], frontend
    )
    command(
        "frontend-build", [shutil.which("npm.cmd") or "npm", "run", "build"], frontend
    )
    command("public-download", [sys.executable, "code/python/download_hydraulic.py"])
    command(
        "smoke-train",
        [
            sys.executable,
            "code/python/train_hydraulic.py",
            "--task",
            "valve",
            "--profile",
            "valve-switch-100hz",
            "--architecture",
            "dense",
            "--normal-policy",
            "all",
            "--epochs",
            "2",
        ],
    )
    run_id = json.loads((destination / "outputs/valve/latest.json").read_text())[
        "run_id"
    ]
    artifact = destination / "outputs/valve/runs" / run_id
    for path in (artifact / "manifest.json", artifact / "bundle/manifest.json"):
        data = json.loads(path.read_text())
        data["sync"] = {
            "required": True,
            "expected_lead_s": 8.0,
            "tolerance_samples": 1,
        }
        path.write_text(json.dumps(data, indent=2))
    for path in (artifact / "metrics.json", artifact / "bundle/metrics.json"):
        data = json.loads(path.read_text())
        data["research_only"] = True
        path.write_text(json.dumps(data, indent=2))
    shutil.make_archive(str(artifact / "model-bundle"), "zip", artifact / "bundle")
    private, public = logs / "test-only-private.pem", logs / "test-only-public.pem"
    env["MECHANISM_SIGNING_PASSWORD"] = os.urandom(32).hex()
    command(
        "keygen",
        [
            sys.executable,
            "code/python/sign_models.py",
            "keygen",
            "--private",
            str(private),
            "--public",
            str(public),
        ],
    )
    command(
        "sign",
        [
            sys.executable,
            "code/python/sign_models.py",
            "sign",
            "--directory",
            str(artifact),
            "--key",
            str(private),
        ],
    )
    env.pop("MECHANISM_SIGNING_PASSWORD")
    command(
        "verify-signature",
        [
            sys.executable,
            "code/python/sign_models.py",
            "verify",
            "--directory",
            str(artifact),
            "--key",
            str(public),
        ],
    )
    env["MECHANISM_TRUSTED_MODEL_KEY"] = str(public)
    env["MECHANISM_RUNTIME_DB"] = str(destination / "work/runtime.sqlite")
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    server_log = logs / "http-server.log"
    with server_log.open("wb") as stream:
        server = subprocess.Popen(
            [sys.executable, "code/python/serve_demo.py", "--port", str(port)],
            cwd=destination,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        url = f"http://127.0.0.1:{port}"

        def request(path, body=None):
            payload = None if body is None else json.dumps(body).encode()
            with urlopen(
                Request(
                    url + path,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                ),
                timeout=15,
            ) as response:
                return json.loads(response.read())

        try:
            for _ in range(40):
                try:
                    request("/api/health")
                    break
                except (URLError, OSError):
                    if server.poll() is not None:
                        raise RuntimeError("验收服务提前退出")
                    time.sleep(0.25)
            else:
                raise TimeoutError("验收服务未就绪")
            replay = request("/api/replay/valve")
            if len(replay["scores"]) != 2205:
                raise ValueError("回放窗口不全")
            request("/api/runtime/activate/valve", {"run_id": run_id, "mode": "shadow"})
            import numpy as np

            with np.load(artifact / "parity.npz") as arrays:
                values = arrays["raw"][0].tolist()
            body = {
                "task": "valve",
                "entity": "clean-smoke",
                "sequence": 0,
                "values": values,
                "channels": ["PS1", "PS2"],
                "units": ["bar", "bar"],
                "sample_rate_hz": 100,
            }
            unsupported = request("/api/runtime/predict", body)
            supported = request(
                "/api/runtime/predict?token=smoke-query-canary",
                {
                    **body,
                    "sequence": 1,
                    "sync": {
                        "command_edge_s": 1000,
                        "window_start_s": 1008,
                        "channel_skew_s": 0,
                    },
                },
            )
            assert (
                unsupported["status"] == "unscored" and supported["status"] == "scored"
            )
            monitor = request("/api/runtime/monitor")
            command(
                "backup",
                [
                    sys.executable,
                    "code/python/runtime_ops.py",
                    "--database",
                    env["MECHANISM_RUNTIME_DB"],
                    "--backup-dir",
                    str(logs / "backups"),
                ],
            )
            backup = sorted((logs / "backups").glob("*.sqlite"))[-1]
            restored = logs / "restored.sqlite"
            shutil.copyfile(backup, restored)
            count = command(
                "restore-check",
                [
                    sys.executable,
                    "-c",
                    "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); assert c.execute('pragma integrity_check').fetchone()[0]=='ok'; print(c.execute('select count(*) from predictions').fetchone()[0])",
                    str(restored),
                ],
            )
            assert int(count) == 2
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    assert "smoke-query-canary" not in server_log.read_text(encoding="utf-8")
    report = {
        "status": "passed",
        "clone": str(destination),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=destination, text=True
        ).strip(),
        "scope": "clean code/data/artifacts; reused installed dependencies, not clean OS installation; two-epoch smoke NOT model performance",
        "module_origins": origin,
        "replay_windows": len(replay["scores"]),
        "restored_predictions": int(count),
        "sync_missing": unsupported["status"],
        "sync_aligned": supported["status"],
        "http_requests": monitor["http"]["requests"],
        "query_not_logged": True,
    }
    (logs / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    run(parser.parse_args().destination)
