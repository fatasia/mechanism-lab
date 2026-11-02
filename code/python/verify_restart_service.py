"""真实Windows API进程树重启检查，隔离数据库，不修改长测脚本。"""

import json
import os
import socket
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from phm_core.artifacts import run_path
from service_process import start_service, stop_service
from soak_service import request

ROOT = Path(__file__).resolve().parents[2]


def run():
    port = 8769
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", port))
    out = ROOT / "work/validation" / datetime.now(UTC).strftime("restart-%Y%m%dT%H%M%S")
    out.mkdir(parents=True)
    env = {
        **os.environ,
        "MECHANISM_RUNTIME_DB": str(out / "runtime.sqlite"),
        "MECHANISM_MODE": "local",
    }
    directory = run_path(ROOT / "outputs", "valve")
    manifest = json.loads((directory / "bundle/manifest.json").read_text())
    with np.load(directory / "parity.npz") as data:
        values = data["raw"][0].tolist()
    payload = {
        "task": "valve",
        "entity": "restart-check",
        "sequence": 0,
        "values": values,
        "channels": manifest["channels"],
        "units": manifest["units"],
        "sample_rate_hz": manifest["sampleRateHz"],
    }
    url = f"http://127.0.0.1:{port}"
    report = {"status": "running", "cycles": []}

    def probe():
        with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
            json.load(response)

    with (out / "server.log").open("ab") as log:
        try:
            for index in range(20):
                process, readiness = start_service(
                    [
                        sys.executable,
                        str(ROOT / "code/python/serve_demo.py"),
                        "--port",
                        str(port),
                    ],
                    ROOT,
                    env,
                    log,
                    probe,
                )
                try:
                    if index == 0:
                        request(
                            url + "/api/runtime/activate/valve",
                            {"run_id": directory.name, "model": "ae", "mode": "shadow"},
                        )
                    result = request(url + "/api/runtime/predict", payload)
                    if (
                        result["duplicate"] != (index > 0)
                        or result["status"] != "scored"
                    ):
                        raise AssertionError("重启后推理/去重状态不符")
                finally:
                    stop_service(process)
                with socket.socket() as connection:
                    connection.settimeout(1)
                    if connection.connect_ex(("127.0.0.1", port)) == 0:
                        raise AssertionError("停止后监听进程仍存在")
                report["cycles"].append(
                    {
                        "index": index,
                        **readiness,
                        "duplicate": result["duplicate"],
                        "port_closed": True,
                    }
                )
                print(index, readiness, flush=True)
            report["status"] = "passed"
        except Exception as error:
            report.update(status="failed", error=type(error).__name__)
            raise
        finally:
            (out / "report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print(str(out), flush=True)


if __name__ == "__main__":
    run()
