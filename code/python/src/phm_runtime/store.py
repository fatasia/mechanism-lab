"""SQLite 事务保存推理状态与事件，重复窗口不重复触发告警。"""

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from phm_core.decision import DecisionState, step
from phm_runtime.audit import audit
from phm_runtime.outbox import SCHEMA as OUTBOX_SCHEMA
from phm_runtime.outbox import enqueue

SCHEMA = """
CREATE TABLE IF NOT EXISTS bindings(task TEXT PRIMARY KEY,run_id TEXT NOT NULL,model TEXT NOT NULL,mode TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS states(entity TEXT,version TEXT,sequence INTEGER,state TEXT,PRIMARY KEY(entity,version));
CREATE TABLE IF NOT EXISTS predictions(entity TEXT,version TEXT,sequence INTEGER,result TEXT,PRIMARY KEY(entity,version,sequence));
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,entity TEXT,version TEXT,start INTEGER,end INTEGER,reason TEXT,status TEXT DEFAULT 'new',operator TEXT,verdict TEXT);
CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY,at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),action TEXT,entity TEXT,detail TEXT);
"""


class StateStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(SCHEMA)
            db.executescript(OUTBOX_SCHEMA)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def binding(self, task):
        with self.connect() as db:
            row = db.execute("SELECT * FROM bindings WHERE task=?", (task,)).fetchone()
            return dict(row) if row else None

    def bind(self, task, run_id, model, mode):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM bindings WHERE task=?", (task,)).fetchone()
            if old and (old["run_id"], old["model"]) != (run_id, model):
                version = f"{task}/{old['run_id']}/{old['model']}"
                db.execute(
                    "UPDATE events SET end=COALESCE((SELECT sequence FROM states WHERE states.entity=events.entity AND states.version=events.version),start),reason='model-change' WHERE version=? AND end IS NULL",
                    (version,),
                )
                db.execute(
                    "DELETE FROM states WHERE version IN (?,?)",
                    (version, f"{task}/{run_id}/{model}"),
                )
            db.execute(
                "INSERT INTO bindings VALUES(?,?,?,?) ON CONFLICT(task) DO UPDATE SET run_id=excluded.run_id,model=excluded.model,mode=excluded.mode",
                (task, run_id, model, mode),
            )
            audit(
                db, "activate", task, {"run_id": run_id, "model": model, "mode": mode}
            )

    def consume(self, entity, version, sequence, score, config, quality, context):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if context.get("task"):
                binding = db.execute(
                    "SELECT * FROM bindings WHERE task=?", (context["task"],)
                ).fetchone()
                if (
                    binding is None
                    or version
                    != f"{binding['task']}/{binding['run_id']}/{binding['model']}"
                ):
                    raise ValueError("运行绑定已切换，此窗口未写入状态；请按新版本重发")
            cached = db.execute(
                "SELECT result FROM predictions WHERE entity=? AND version=? AND sequence=?",
                (entity, version, sequence),
            ).fetchone()
            if cached:
                previous_result = json.loads(cached[0])
                if (
                    context.get("input_sha256")
                    and previous_result.get("input_sha256") != context["input_sha256"]
                ):
                    raise ValueError("窗口序号已使用，但输入数据不同")
                return {**previous_result, "duplicate": True}
            row = db.execute(
                "SELECT sequence,state FROM states WHERE entity=? AND version=?",
                (entity, version),
            ).fetchone()
            if row and sequence <= row["sequence"]:
                raise ValueError("窗口乱序；只允许重发已记录窗口或提交更晚的窗口")
            state = (
                DecisionState(**json.loads(row["state"])) if row else DecisionState()
            )
            gap = row is not None and sequence != row["sequence"] + 1
            if gap or score is None:
                db.execute(
                    "UPDATE events SET end=?,reason='data-gap' WHERE entity=? AND version=? AND end IS NULL",
                    (row["sequence"] if row else sequence, entity, version),
                )
                state = DecisionState()
            event, next_state = step(score, state, config)
            if event == "opened":
                opened = db.execute(
                    "INSERT INTO events(entity,version,start,reason) VALUES(?,?,?,'active')",
                    (entity, version, sequence),
                )
                enqueue(
                    db,
                    opened.lastrowid,
                    {
                        "event_id": opened.lastrowid,
                        "entity": entity,
                        "version": version,
                        "sequence": sequence,
                        "score": score,
                        "mode": context.get("mode", "shadow"),
                    },
                )
            elif event == "closed":
                db.execute(
                    "UPDATE events SET end=?,reason='recovery' WHERE entity=? AND version=? AND end IS NULL",
                    (sequence, entity, version),
                )
            result = {
                **context,
                "entity": entity,
                "version": version,
                "sequence": sequence,
                "score": score,
                "quality": quality,
                "status": "scored" if score is not None else "unscored",
                "alarm": next_state.active,
                "transition": event,
                "gap": gap,
                "duplicate": False,
            }
            db.execute(
                "INSERT OR REPLACE INTO states VALUES(?,?,?,?)",
                (entity, version, sequence, json.dumps(asdict(next_state))),
            )
            db.execute(
                "INSERT INTO predictions VALUES(?,?,?,?)",
                (entity, version, sequence, json.dumps(result, allow_nan=False)),
            )
            return result

    def events(self):
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM events ORDER BY id DESC LIMIT 1000"
                )
            ]

    def feedback(self, event_id, operator, verdict):
        with self.connect() as db:
            changed = db.execute(
                "UPDATE events SET status='reviewed',operator=?,verdict=? WHERE id=?",
                (operator, verdict, event_id),
            ).rowcount
            if not changed:
                raise ValueError("告警不存在")
            audit(db, "feedback", event_id, {"operator": operator, "verdict": verdict})

    def history(self, entity, limit=200):
        with self.connect() as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT result FROM predictions WHERE entity=? ORDER BY rowid DESC LIMIT ?",
                    (entity, limit),
                )
            ]

    def backup(self, destination):
        destination = Path(destination)
        if destination.resolve() == self.path.resolve():
            raise ValueError("备份路径不能是运行库本身")
        destination.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 30

        def progress(status, remaining, total):
            if time.monotonic() > deadline:
                raise TimeoutError("SQLite备份超时")

        target = sqlite3.connect(destination, timeout=10)
        try:
            with self.connect() as source:
                source.backup(target, pages=256, progress=progress)
            # 在线备份可能继承 WAL 模式。将快照落回主文件，不能依赖长期增长的 sidecar。
            if target.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]:
                raise ValueError("备份库仍被读取，无法完成检查点")
            if target.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
                raise ValueError("备份库无法切换为独立文件")
            if target.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("备份完整性检查失败")
        finally:
            target.close()
