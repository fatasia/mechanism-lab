"""请求监控使用固定字段；动态设备标识与查询内容不进入访问日志。"""

import logging
import time
from uuid import uuid4

from phm_runtime.observability import RequestMetrics, json_event

logger = logging.getLogger("uvicorn.error")


def install_observability(app):
    app.state.request_metrics = RequestMetrics()

    @app.middleware("http")
    async def observe(request, call_next):
        started = time.perf_counter()
        request_id = uuid4().hex
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            app.state.request_metrics.record(status, elapsed)
            route = request.scope.get("route")
            logger.info(
                json_event(
                    "http_request",
                    request_id=request_id,
                    method=request.method,
                    route=getattr(route, "path", "unmatched"),
                    status=status,
                    duration_ms=round(elapsed, 3),
                )
            )
