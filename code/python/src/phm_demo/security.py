"""本地模式限回环地址；服务模式启用环境凭据和最小角色权限。"""

import base64
import os
from hmac import compare_digest
from urllib.parse import urlsplit

from fastapi.responses import JSONResponse

from phm_runtime.audit import actor


def install_access_control(app):
    mode = os.environ.get("MECHANISM_MODE", "local")
    if mode not in ("local", "service"):
        raise ValueError("MECHANISM_MODE 只支持 local 或 service")
    accounts = []
    if mode == "service":
        for role in ("admin", "operator", "reader"):
            user = os.environ.get(f"MECHANISM_{role.upper()}_USER")
            password = os.environ.get(f"MECHANISM_{role.upper()}_PASSWORD")
            if user and password:
                accounts.append((user, password, role))
        if not any(role == "admin" for _, _, role in accounts):
            raise ValueError("服务模式必须从环境提供管理员凭据")

    @app.middleware("http")
    async def access(request, call_next):
        if request.url.path == "/api/health":
            return JSONResponse(
                {"service": "机理实验室", "version": "0.4.0", "mode": mode}
            )
        if mode == "local":
            principal = "local"
            if request.client and request.client.host not in (
                "127.0.0.1",
                "::1",
                "localhost",
                "testclient",
            ):
                return JSONResponse(
                    {"detail": "本地模式只接受回环地址"}, status_code=403
                )
            role = "admin"
        else:
            role = None
            authorization = request.headers.get("authorization", "")
            if authorization.startswith("Basic "):
                try:
                    user, password = (
                        base64.b64decode(authorization[6:], validate=True)
                        .decode("utf-8")
                        .split(":", 1)
                    )
                    for expected_user, expected_password, candidate in accounts:
                        if compare_digest(
                            user.encode(), expected_user.encode()
                        ) and compare_digest(
                            password.encode(), expected_password.encode()
                        ):
                            role = candidate
                            principal = expected_user
                except (ValueError, UnicodeDecodeError):
                    pass
            if role is None:
                return JSONResponse(
                    {"detail": "需要服务访问凭据"},
                    status_code=401,
                    headers={
                        "WWW-Authenticate": 'Basic realm="Mechanism Lab", charset="UTF-8"'
                    },
                )
        write = request.method not in ("GET", "HEAD", "OPTIONS")
        admin = request.url.path.startswith(
            (
                "/api/train/",
                "/api/job/cancel",
                "/api/runtime/activate/",
                "/api/runtime/backup",
                "/api/runtime/monitor",
                "/api/maintenance/people",
            )
        )
        if (admin and role != "admin") or (write and role == "reader"):
            return JSONResponse({"detail": "当前角色不允许此操作"}, status_code=403)
        origin = request.headers.get("origin")
        if write and origin and urlsplit(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "拒绝跨来源写入"}, status_code=403)
        request.state.role = role
        token = actor.set(principal)
        try:
            return await call_next(request)
        finally:
            actor.reset(token)
