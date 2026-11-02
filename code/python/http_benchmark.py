"""本地HTTP验收客户端：每个工作者复用一条连接，不自动重试业务请求。"""

import http.client
import json
import urllib.error


class CountedConnection(http.client.HTTPConnection):
    def __init__(self, host, port, timeout=30):
        super().__init__(host, port, timeout=timeout)
        self.connections = 0

    def connect(self):
        super().connect()
        self.connections += 1


class HttpSession:
    def __init__(self, port):
        self.connection = CountedConnection("127.0.0.1", port)

    def request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        try:
            self.connection.request(
                "GET" if payload is None else "POST",
                path,
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = self.connection.getresponse()
            data = response.read()
            if not 200 <= response.status < 300:
                raise urllib.error.HTTPError(
                    path, response.status, response.reason, response.headers, None
                )
            return json.loads(data)
        except Exception:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()
