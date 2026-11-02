"""生成加密签名密钥，或签名/验证完整模型运行目录。密钥不随源码或模型分发。"""

import argparse
import json
import os
from pathlib import Path

from phm_core.signing import sign_directory, verify_directory


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("keygen")
    generate.add_argument("--private", type=Path, required=True)
    generate.add_argument("--public", type=Path, required=True)
    for name in ("sign", "verify"):
        action = commands.add_parser(name)
        action.add_argument("--directory", type=Path, required=True)
        action.add_argument("--key", type=Path, required=True)
    args = parser.parse_args()
    password = os.environ.get("MECHANISM_SIGNING_PASSWORD", "").encode()
    if args.command != "verify" and not password:
        parser.error("请从环境设置MECHANISM_SIGNING_PASSWORD，不通过命令参数传递口令")
    if args.command == "keygen":
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        if args.private.resolve() == args.public.resolve():
            parser.error("私钥与公钥必须使用不同路径")
        if args.private.exists() or args.public.exists():
            parser.error("密钥路径已存在，不覆盖已有信任根")
        key = Ed25519PrivateKey.generate()
        private = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.BestAvailableEncryption(password),
        )
        public = key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        # 独占创建，避免覆盖已经分发的信任根。
        with args.private.open("xb") as stream:
            stream.write(private)
        with args.public.open("xb") as stream:
            stream.write(public)
        print(json.dumps({"generated": True, "algorithm": "Ed25519"}))
    elif args.command == "sign":
        result = sign_directory(args.directory, args.key, password)
        print(json.dumps({"signed": True, "files": len(result["files"])}))
    else:
        print(json.dumps(verify_directory(args.directory, args.key)))


if __name__ == "__main__":
    main()
