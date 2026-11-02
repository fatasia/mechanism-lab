"""有界持久化窗口队列：统一JSON契约，按序重试；不假设底层消息恰好一次。"""

import hashlib
import json
import sqlite3
from pathlib import Path


class WindowQueue:
    def __init__(self, path: Path, capacity=1000):
        if capacity < 1:
            raise ValueError("队列容量必须为正")
        self.path, self.capacity = path, capacity
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS windows(id INTEGER PRIMARY KEY, entity TEXT, task TEXT, sequence INTEGER, sha256 TEXT, body TEXT, receipt TEXT, UNIQUE(entity,task,sequence))"
            )

    def put(self, body):
        # 与HTTP共用结构契约；单位/通道/采样率的模型相关门控仍由在线接口处理。
        from phm_demo.runtime_api import Window

        value = Window.model_validate(body).model_dump(mode="json", exclude_none=True)
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with sqlite3.connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT id,sha256 FROM windows WHERE entity=? AND task=? AND sequence=?",
                (value["entity"], value["task"], value["sequence"]),
            ).fetchone()
            if row:
                if row[1] != digest:
                    raise ValueError("同序号输入冲突")
                return row[0]
            # 序号回退禁止进入队列；允许跳号，服务将记录数据间断。
            last = db.execute(
                "SELECT MAX(sequence) FROM windows WHERE entity=? AND task=?",
                (value["entity"], value["task"]),
            ).fetchone()[0]
            if last is not None and value["sequence"] <= last:
                raise ValueError("输入乱序")
            if (
                db.execute(
                    "SELECT count(*) FROM windows WHERE receipt IS NULL"
                ).fetchone()[0]
                >= self.capacity
            ):
                raise BufferError("持久队列已满；上游必须背压，不能静默丢弃")
            return db.execute(
                "INSERT INTO windows(entity,task,sequence,sha256,body) VALUES(?,?,?,?,?)",
                (value["entity"], value["task"], value["sequence"], digest, encoded),
            ).lastrowid

    def drain_one(self, send):
        # 单消费者。HTTP失败或响应丢失时不确认，重启后重发原始序号。
        with sqlite3.connect(self.path) as db:
            row = db.execute(
                "SELECT id,body FROM windows WHERE receipt IS NULL ORDER BY id LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        reply = send(json.loads(row[1]))
        if reply.get("status") not in ("scored", "unscored"):
            raise ValueError("服务器未返回明确评分状态")
        receipt = json.dumps(reply, allow_nan=False)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "UPDATE windows SET receipt=?,body=NULL WHERE id=? AND receipt IS NULL",
                (receipt, row[0]),
            )
        return reply

    def snapshot(self):
        with sqlite3.connect(self.path) as db:
            queued = db.execute(
                "SELECT count(*) FROM windows WHERE receipt IS NULL"
            ).fetchone()[0]
            acknowledged = db.execute(
                "SELECT count(*) FROM windows WHERE receipt IS NOT NULL"
            ).fetchone()[0]
        return {
            "pending": queued,
            "acknowledged": acknowledged,
            "capacity": self.capacity,
            "bytes": self.path.stat().st_size,
        }
