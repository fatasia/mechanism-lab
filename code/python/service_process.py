"""隔离验收服务的有界启动与进程树清理；不重试业务推理。"""

import subprocess
import time
import urllib.error
from collections import Counter

from phm_runtime.processes import spawn_process, stop_process


def wait_ready(process, probe, timeout=30, interval=0.1):
    deadline = time.monotonic() + timeout
    errors = Counter()
    attempts = 0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("隔离服务启动退出，检查server.log")
        attempts += 1
        try:
            probe()
            return {"attempts": attempts, "transport_errors": dict(errors)}
        except urllib.error.HTTPError:
            raise  # HTTP错误不是未监听，不吞掉应用故障。
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            errors[type(error).__name__] += 1
            time.sleep(min(interval, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"隔离服务启动超时，探测{attempts}次：{dict(errors)}")


def stop_service(process):
    stop_process(process)


def start_service(command, cwd, env, log, probe, timeout=30):
    process = spawn_process(
        command,
        cwd=cwd,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        readiness = wait_ready(process, probe, timeout)
        return process, readiness
    except BaseException:
        stop_service(process)
        raise
