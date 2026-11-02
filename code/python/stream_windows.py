"""公开数据窗口JSONL→本地HTTP；按行入队并立即发送，失败保留供重启重试。"""

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from phm_runtime.streaming import WindowQueue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    url = urlsplit(args.url)
    if (
        url.scheme != "http"
        or url.hostname not in ("127.0.0.1", "localhost", "::1")
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        parser.error("此回放适配器只连接本机HTTP，不接受凭据或远端地址")
    queue = WindowQueue(args.queue)

    def send(body):
        request = Request(
            args.url.rstrip("/") + "/api/runtime/predict",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read())

    while queue.drain_one(send) is not None:
        pass
    if args.input:
        with args.input.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                queue.put(json.loads(line))
                queue.drain_one(send)
    print(json.dumps(queue.snapshot()))


if __name__ == "__main__":
    main()
