"""事务告警通知队列；至少一次投递，接收方按通知 id 幂等。"""

import json
import time
from uuid import uuid4

from phm_runtime.audit import audit

SCHEMA = """CREATE TABLE IF NOT EXISTS outbox(
    id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL UNIQUE,
    payload TEXT NOT NULL, status TEXT DEFAULT 'pending', attempts INTEGER DEFAULT 0,
    available REAL DEFAULT 0, lease_until REAL, token TEXT, last_error TEXT);
"""


def enqueue(db, event_id, payload):
    db.execute(
        "INSERT INTO outbox(event_id,payload) VALUES(?,?)",
        (
            event_id,
            json.dumps({**payload, "notification_key": uuid4().hex}, allow_nan=False),
        ),
    )


class Outbox:
    def __init__(self, store):
        self.store = store

    def claim(self, now=None, lease_seconds=60):
        now = time.time() if now is None else now
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM outbox WHERE (status='pending' AND available<=?) OR (status='leased' AND lease_until<=?) ORDER BY id LIMIT 1",
                (now, now),
            ).fetchone()
            if row is None:
                return None
            token = uuid4().hex
            db.execute(
                "UPDATE outbox SET status='leased',attempts=attempts+1,lease_until=?,token=? WHERE id=?",
                (now + lease_seconds, token, row["id"]),
            )
            return {
                "id": row["id"],
                "token": token,
                "attempts": row["attempts"] + 1,
                "payload": json.loads(row["payload"]),
            }

    def finish(self, notification_id, token, success, error=None, now=None):
        now = time.time() if now is None else now
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM outbox WHERE id=? AND token=? AND status='leased' AND lease_until>?",
                (notification_id, token, now),
            ).fetchone()
            if row is None:
                raise ValueError("通知租约已失效")
            status = (
                "delivered"
                if success
                else "dead"
                if row["attempts"] >= 5
                else "pending"
            )
            db.execute(
                "UPDATE outbox SET status=?,available=?,lease_until=NULL,token=NULL,last_error=? WHERE id=?",
                (status, now + min(300, 2 ** row["attempts"]), error, notification_id),
            )

    def retry(self, notification_id):
        with self.store.connect() as db:
            changed = db.execute(
                "UPDATE outbox SET status='pending',attempts=0,available=0,last_error=NULL WHERE id=? AND status='dead'",
                (notification_id,),
            ).rowcount
            if not changed:
                raise ValueError("仅失败队列可以人工重试")
            audit(db, "notification_retry", notification_id, {})

    def snapshot(self):
        with self.store.connect() as db:
            counts = {
                r["status"]: r["n"]
                for r in db.execute(
                    "SELECT status,count(*) n FROM outbox GROUP BY status"
                )
            }
            rows = [
                dict(r)
                for r in db.execute(
                    "SELECT id,event_id,status,attempts,last_error FROM outbox ORDER BY id DESC LIMIT 100"
                )
            ]
            return {"counts": counts, "recent": rows}
