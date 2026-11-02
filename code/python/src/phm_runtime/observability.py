"""有界请求指标和脱敏JSON日志；不记录请求体、查询字符串、认证头或异常文本。"""

import json
import re
import shutil
import time
from collections import Counter, deque
from datetime import UTC, datetime
from threading import Lock

SENSITIVE = re.compile(
    r"password|passwd|secret|token|authorization|credential|api.?key|private.?key", re.I
)
INLINE_SECRET = re.compile(
    r"(?i)(?:Bearer\s+\S+|Basic\s+[A-Za-z0-9+/=]+|(?:password|secret|token|api[_-]?key)\s*[:=]\s*[^\s,;]+)"
)


def redact(value):
    if isinstance(value, dict):
        return {
            str(k): "[REDACTED]" if SENSITIVE.search(str(k)) else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return INLINE_SECRET.sub("[REDACTED]", value)
    return value


def json_event(event, **fields):
    return json.dumps(
        {"at": datetime.now(UTC).isoformat(), "event": event, **redact(fields)},
        ensure_ascii=False,
        allow_nan=False,
    )


class RequestMetrics:
    def __init__(self, window=512):
        self.lock = Lock()
        self.started = time.monotonic()
        self.counts = Counter()
        self.latencies = deque(maxlen=window)

    def record(self, status, duration_ms):
        with self.lock:
            self.counts[int(status)] += 1
            self.latencies.append(float(duration_ms))

    def snapshot(self):
        with self.lock:
            values = sorted(self.latencies)
            return {
                "uptime_seconds": round(time.monotonic() - self.started, 3),
                "requests": sum(self.counts.values()),
                "status_counts": dict(self.counts),
                "latency_window": len(values),
                "latency_window_limit": self.latencies.maxlen,
                "p95_ms": values[min(len(values) - 1, int(0.95 * (len(values) - 1)))]
                if values
                else None,
            }


def storage_snapshot(database):
    usage = shutil.disk_usage(database.parent)
    return {
        "disk_free_bytes": usage.free,
        "disk_total_bytes": usage.total,
        "database_bytes": database.stat().st_size if database.exists() else 0,
        "wal_bytes": (database.parent / (database.name + "-wal")).stat().st_size
        if (database.parent / (database.name + "-wal")).exists()
        else 0,
    }
