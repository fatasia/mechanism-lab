"""单机训练队列：持久状态、工作者心跳与协作取消，不依赖 Web 进程。"""

import json
import time
from uuid import uuid4

from phm_runtime.audit import audit


class JobStore:
    def __init__(self, store):
        self.store = store
        with store.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, task TEXT NOT NULL, command TEXT NOT NULL,
                status TEXT NOT NULL, created REAL NOT NULL, heartbeat REAL NOT NULL,
                timeout REAL NOT NULL, cancel INTEGER DEFAULT 0, exit_code INTEGER,
                detail TEXT, log_path TEXT NOT NULL)""")

    def create(self, task, command, log_path, timeout=7200):
        now = time.time()
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM jobs WHERE status IN ('queued','running')"
            ).fetchone():
                raise ValueError("已有训练任务运行中")
            job_id = uuid4().hex
            db.execute(
                "INSERT INTO jobs(id,task,command,status,created,heartbeat,timeout,log_path) VALUES(?,?,?,'queued',?,?,?,?)",
                (job_id, task, json.dumps(command), now, now, timeout, str(log_path)),
            )
            audit(db, "job_create", job_id, {"task": task, "timeout": timeout})
            return job_id

    def claim(self, job_id):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            updated = db.execute(
                "UPDATE jobs SET status='running',heartbeat=? WHERE id=? AND status='queued'",
                (time.time(), job_id),
            ).rowcount
            if not updated:
                raise ValueError("任务不存在或已被领取")
            return dict(
                db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            )

    def heartbeat(self, job_id):
        with self.store.connect() as db:
            row = db.execute(
                "SELECT status,cancel FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None or row["status"] != "running":
                return False
            db.execute("UPDATE jobs SET heartbeat=? WHERE id=?", (time.time(), job_id))
            return not bool(row["cancel"])

    def finish(self, job_id, status, exit_code=None, detail=None):
        if status not in ("complete", "failed", "cancelled", "timed_out"):
            raise ValueError("非法任务完成状态")
        with self.store.connect() as db:
            db.execute(
                "UPDATE jobs SET status=?,exit_code=?,detail=?,heartbeat=? WHERE id=? AND status IN ('queued','running')",
                (status, exit_code, detail, time.time(), job_id),
            )

    def cancel(self, job_id):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status,cancel FROM jobs WHERE id=?",
                (job_id,),
            ).fetchone()
            if row is None or row["status"] not in ("queued", "running"):
                raise ValueError("没有可停止的训练任务")
            if row["cancel"]:
                return
            db.execute("UPDATE jobs SET cancel=1 WHERE id=?", (job_id,))
            audit(db, "job_cancel_requested", job_id, {})

    def latest(self):
        with self.store.connect() as db:
            row = db.execute(
                "SELECT * FROM jobs ORDER BY created DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            # 心跳中断不自动启动第二个训练进程，避免未确认的工作者继续发布。
            result["stale"] = (
                result["status"] in ("queued", "running")
                and time.time() - result["heartbeat"] > 60
            )
            return result
