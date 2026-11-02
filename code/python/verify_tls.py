"""本机临时测试证书TLS握手，不生成或发布生产证书。"""

import ipaddress
import json
import os
import socket
import ssl
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from service_process import start_service, stop_service

ROOT = Path(__file__).resolve().parents[2]


def run():
    with tempfile.TemporaryDirectory(prefix="mechanism-test-tls-") as temporary:
        directory = Path(temporary)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name(
            [x509.NameAttribute(NameOID.COMMON_NAME, "Mechanism test only")]
        )
        now = datetime.now(timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(hours=1))
            .add_extension(
                x509.SubjectAlternativeName(
                    [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
                ),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        cert_path, key_path = directory / "test.crt", directory / "test.key"
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        context = ssl.create_default_context(cafile=str(cert_path))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"https://127.0.0.1:{port}/api/health"

        def probe():
            with urlopen(url, context=context, timeout=2) as response:
                return json.loads(response.read())

        with (directory / "server.log").open("wb") as log:
            server, _ = start_service(
                [
                    sys.executable,
                    str(ROOT / "code/python/serve_demo.py"),
                    "--port",
                    str(port),
                    "--ssl-certfile",
                    str(cert_path),
                    "--ssl-keyfile",
                    str(key_path),
                ],
                ROOT,
                {
                    **os.environ,
                    "MECHANISM_MODE": "local",
                    "MECHANISM_RUNTIME_DB": str(directory / "runtime.db"),
                },
                log,
                probe,
            )
            try:
                assert probe()["mode"] == "local"
            finally:
                stop_service(server)
        print(
            json.dumps(
                {
                    "tls_handshake": "passed",
                    "hostname_and_ca_verified": True,
                    "certificate": "temporary test-only, removed",
                    "scope": "loopback TLS; not site PKI provisioning",
                }
            )
        )


if __name__ == "__main__":
    run()
