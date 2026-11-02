import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from phm_core.signing import sign_directory, verify_directory


@pytest.fixture
def package(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "manifest.json").write_text('{"fixture": true}')
    (directory / "weights.onnx").write_bytes(b"fixture-not-an-actual-model")
    key = Ed25519PrivateKey.generate()
    private = tmp_path / "private.pem"
    public = tmp_path / "public.pem"
    password = b"test-only-passphrase"
    private.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.BestAvailableEncryption(password),
        )
    )
    public.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    return directory, private, public, password


def test_signature_and_wrong_trust_key(package, tmp_path):
    directory, private, public, password = package
    sign_directory(directory, private, password)
    assert verify_directory(directory, public)["files"] == 2
    wrong = tmp_path / "wrong.pem"
    wrong.write_bytes(
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    with pytest.raises(ValueError, match="签名无效"):
        verify_directory(directory, wrong)


@pytest.mark.parametrize(
    "change", ["weights", "manifest", "extra", "signature", "missing"]
)
def test_tampering_rejected(package, change):
    directory, private, public, password = package
    sign_directory(directory, private, password)
    if change == "weights":
        (directory / "weights.onnx").write_bytes(b"changed")
    elif change == "manifest":
        (directory / "manifest.json").write_text("{}")
    elif change == "extra":
        (directory / "extra.txt").write_text("unexpected")
    elif change == "signature":
        record = json.loads((directory / "signature.json").read_text())
        record["files"]["weights.onnx"] = "forged"
        (directory / "signature.json").write_text(json.dumps(record))
    else:
        (directory / "signature.json").unlink()
    with pytest.raises(ValueError):
        verify_directory(directory, public)


def test_registry_verifies_before_loading(package, tmp_path, monkeypatch):
    from phm_runtime.registry import ModelRegistry
    from phm_runtime.store import StateStore

    directory, _, public, _ = package
    monkeypatch.setenv("MECHANISM_TRUSTED_MODEL_KEY", str(public))
    monkeypatch.setattr("phm_runtime.registry.run_path", lambda *args: directory)
    registry = ModelRegistry(tmp_path, StateStore(tmp_path / "runtime.sqlite"))
    with pytest.raises(ValueError, match="缺少模型目录签名"):
        registry.load("valve", "fixture")
