import importlib.util
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

SPEC = importlib.util.spec_from_file_location(
    "http_benchmark", Path(__file__).resolve().parents[1] / "http_benchmark.py"
)
http = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(http)


def test_keep_alive_and_no_hidden_application_retry():
    paths = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            paths.append(self.path)
            self.send_response(500 if self.path == "/fail" else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    session = http.HttpSession(server.server_port)
    try:
        assert session.request("/a") == {}
        assert session.request("/b") == {}
        assert session.connection.connections == 1
        with pytest.raises(urllib.error.HTTPError):
            session.request("/fail")
        assert paths == ["/a", "/b", "/fail"]
        assert session.request("/c") == {}
        assert session.connection.connections == 2
    finally:
        session.close()
        server.shutdown()
        server.server_close()
        thread.join()
