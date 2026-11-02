"""显式配置的 webhook 工作者；默认不发送，接收方按 Idempotency-Key 去重。"""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from phm_runtime.outbox import Outbox
from phm_runtime.store import StateStore


def deliver(outbox, url):
    parsed = urlsplit(url)
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")
    ):
        raise ValueError("远端通知须HTTPS，本地回环允许HTTP")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("通知URL不允许内嵌凭据、查询参数或片段")
    item = outbox.claim()
    if item is None:
        return "idle"
    request = urllib.request.Request(
        url,
        data=json.dumps(item["payload"]).encode(),
        headers={
            "Content-Type": "application/json",
            "Idempotency-Key": item["payload"]["notification_key"],
        },
    )

    # 禁止重定向将事件数据转发到未经配置的主机。
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    try:
        with urllib.request.build_opener(NoRedirect()).open(
            request, timeout=10
        ) as response:
            success = 200 <= response.status < 300
        error = None if success else "http_status"
    except (urllib.error.URLError, TimeoutError, OSError):
        success, error = False, "transport_error"
    outbox.finish(item["id"], item["token"], success, error)
    return "delivered" if success else "retry_pending_or_dead"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    print(deliver(Outbox(StateStore(args.database)), args.url))
