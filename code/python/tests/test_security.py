import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from phm_demo.security import install_access_control
from phm_runtime.audit import actor


def protected_app(monkeypatch):
    monkeypatch.setenv("MECHANISM_MODE", "service")
    for role in ("ADMIN", "OPERATOR", "READER"):
        monkeypatch.setenv(f"MECHANISM_{role}_USER", role.lower())
        monkeypatch.setenv(f"MECHANISM_{role}_PASSWORD", "fixture-only")
    app = FastAPI()
    install_access_control(app)

    @app.get("/api/value")
    def get_value():
        return {"value": 1}

    @app.post("/api/value")
    def post_value():
        return {"saved": True, "actor": actor.get()}

    @app.post("/api/runtime/activate/valve")
    def activate():
        return {"activated": True}

    return TestClient(app)


def test_service_requires_credentials(monkeypatch):
    monkeypatch.setenv("MECHANISM_MODE", "service")
    monkeypatch.delenv("MECHANISM_ADMIN_USER", raising=False)
    with pytest.raises(ValueError, match="管理员"):
        install_access_control(FastAPI())


def test_roles_and_origin_are_enforced(monkeypatch):
    client = protected_app(monkeypatch)
    assert client.get("/api/value").status_code == 401
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/value", auth=("reader", "fixture-only")).status_code == 200
    assert client.post("/api/value", auth=("reader", "fixture-only")).status_code == 403
    assert (
        client.post("/api/value", auth=("operator", "fixture-only")).status_code == 200
    )
    assert (
        client.post("/api/value", auth=("operator", "fixture-only")).json()["actor"]
        == "operator"
    )
    assert (
        client.post(
            "/api/runtime/activate/valve", auth=("operator", "fixture-only")
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/runtime/activate/valve", auth=("admin", "fixture-only")
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/value",
            auth=("admin", "fixture-only"),
            headers={"Origin": "https://different.example"},
        ).status_code
        == 403
    )


def test_local_mode_rejects_remote_peer(monkeypatch):
    monkeypatch.setenv("MECHANISM_MODE", "local")
    app = FastAPI()
    install_access_control(app)
    assert TestClient(app, client=("192.0.2.1", 1200)).get("/").status_code == 403
