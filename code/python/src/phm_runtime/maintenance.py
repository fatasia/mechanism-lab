"""本地运维工作流：设备、维修、备件与点检；与诊断算法分层。"""

import json

from phm_runtime.audit import audit

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY,name TEXT NOT NULL,kind TEXT NOT NULL,parent_id TEXT REFERENCES assets(id),details TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS work_orders(id INTEGER PRIMARY KEY,asset_id TEXT REFERENCES assets(id),title TEXT NOT NULL,kind TEXT,operator TEXT,status TEXT DEFAULT 'open',revision INTEGER DEFAULT 1,event_id INTEGER REFERENCES events(id),notes TEXT DEFAULT '',cause TEXT DEFAULT '',verification TEXT DEFAULT '',created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),closed_at TEXT);
CREATE TABLE IF NOT EXISTS cases(id INTEGER PRIMARY KEY,order_id INTEGER UNIQUE REFERENCES work_orders(id),asset_id TEXT REFERENCES assets(id),cause TEXT,action TEXT,verification TEXT,confirmed_by TEXT);
CREATE TABLE IF NOT EXISTS parts(id TEXT PRIMARY KEY,name TEXT NOT NULL,quantity INTEGER DEFAULT 0 CHECK(quantity>=0));
CREATE TABLE IF NOT EXISTS stock_moves(id INTEGER PRIMARY KEY,part_id TEXT REFERENCES parts(id),order_id INTEGER REFERENCES work_orders(id),delta INTEGER,reason TEXT,operator TEXT,at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
CREATE TABLE IF NOT EXISTS people(id TEXT PRIMARY KEY,name TEXT,skills TEXT,shift TEXT);
CREATE TABLE IF NOT EXISTS plans(id INTEGER PRIMARY KEY,asset_id TEXT REFERENCES assets(id),name TEXT,interval_days INTEGER CHECK(interval_days>0),steps TEXT,next_due TEXT);
CREATE TABLE IF NOT EXISTS checks(id INTEGER PRIMARY KEY,plan_id INTEGER REFERENCES plans(id),operator TEXT,result TEXT,notes TEXT,at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
"""


class MaintenanceStore:
    def __init__(self, state_store):
        self.store = state_store
        with self.store.connect() as db:
            db.executescript(SCHEMA)
            columns = {
                row["name"] for row in db.execute("PRAGMA table_info(stock_moves)")
            }
            if "request_id" not in columns:
                db.execute("ALTER TABLE stock_moves ADD COLUMN request_id TEXT")
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS stock_request_id ON stock_moves(request_id) WHERE request_id IS NOT NULL"
            )
            check_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(checks)")
            }
            if "order_id" not in check_columns:
                db.execute(
                    "ALTER TABLE checks ADD COLUMN order_id INTEGER REFERENCES work_orders(id)"
                )

    def snapshot(self):
        with self.store.connect() as db:
            result = {
                table: [
                    dict(row)
                    for row in db.execute(
                        f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 1000"
                    )
                ]
                for table in (
                    "assets",
                    "work_orders",
                    "cases",
                    "parts",
                    "stock_moves",
                    "people",
                    "plans",
                    "checks",
                    "audit",
                )
            }
            result["performance"] = [
                dict(row)
                for row in db.execute(
                    "SELECT operator,COUNT(*) closed_orders,AVG((julianday(closed_at)-julianday(created_at))*24) order_cycle_hours FROM work_orders WHERE status='closed' GROUP BY operator"
                )
            ]
            return result

    def asset(self, asset_id, name, kind, parent_id=None, details=None):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            parents = {
                r["id"]: r["parent_id"]
                for r in db.execute("SELECT id,parent_id FROM assets")
            }
            current = parent_id
            while current is not None:
                if current == asset_id:
                    raise ValueError("设备层级不能形成环")
                if current not in parents:
                    raise ValueError("父设备不存在")
                current = parents[current]
            db.execute(
                "INSERT INTO assets VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,kind=excluded.kind,parent_id=excluded.parent_id,details=excluded.details",
                (
                    asset_id,
                    name,
                    kind,
                    parent_id,
                    json.dumps(details or {}, ensure_ascii=False),
                ),
            )
            audit(
                db,
                "asset.save",
                asset_id,
                {"name": name, "kind": kind, "parent_id": parent_id},
            )

    def create_order(self, asset_id, title, operator, kind="repair", event_id=None):
        with self.store.connect() as db:
            cursor = db.execute(
                "INSERT INTO work_orders(asset_id,title,operator,kind,event_id) VALUES(?,?,?,?,?)",
                (asset_id, title, operator, kind, event_id),
            )
            audit(
                db,
                "order.create",
                cursor.lastrowid,
                {"asset_id": asset_id, "operator": operator},
            )
            return cursor.lastrowid

    def transition(
        self, order_id, status, revision, operator, notes="", cause="", verification=""
    ):
        allowed = {
            "open": ("working", "cancelled"),
            "working": ("completed", "cancelled"),
            "completed": ("closed", "working"),
            "closed": (),
            "cancelled": (),
        }
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM work_orders WHERE id=?", (order_id,)
            ).fetchone()
            if not row or row["revision"] != revision:
                raise ValueError("工单不存在或版本已变更，请刷新")
            if status not in allowed[row["status"]]:
                raise ValueError("工单状态流转不允许")
            if status == "completed" and not notes.strip():
                raise ValueError("完成工单必须填写实际处理内容")
            if status == "closed" and not verification.strip():
                raise ValueError("关闭工单必须填写复核结果")
            db.execute(
                "UPDATE work_orders SET status=?,revision=revision+1,operator=?,notes=?,cause=?,verification=?,closed_at=CASE WHEN ?='closed' THEN strftime('%Y-%m-%dT%H:%M:%fZ','now') ELSE NULL END WHERE id=?",
                (
                    status,
                    operator,
                    notes or row["notes"],
                    cause or row["cause"],
                    verification,
                    status,
                    order_id,
                ),
            )
            if status == "closed":
                db.execute(
                    "INSERT INTO cases(order_id,asset_id,cause,action,verification,confirmed_by) VALUES(?,?,?,?,?,?)",
                    (
                        order_id,
                        row["asset_id"],
                        cause or row["cause"],
                        notes or row["notes"],
                        verification,
                        operator,
                    ),
                )
            audit(
                db,
                "order.transition",
                order_id,
                {"from": row["status"], "to": status, "operator": operator},
            )

    def evidence(self, order_id):
        """只读证据包：按实际事件实体/模型版本取数，不把关闭工单等同故障真值。"""
        with self.store.connect() as db:
            row = db.execute(
                "SELECT * FROM work_orders WHERE id=?", (order_id,)
            ).fetchone()
            if row is None:
                raise ValueError("工单不存在")
            order = dict(row)
            event = db.execute(
                "SELECT * FROM events WHERE id=?", (row["event_id"],)
            ).fetchone()
            predictions = []
            total = 0
            if event:
                end = event["end"]
                parameters = (
                    event["entity"],
                    event["version"],
                    max(0, event["start"] - 10),
                    end,
                )
                condition = "entity=? AND version=? AND sequence>=? AND (? IS NULL OR sequence<=?)"
                parameters = (*parameters, end)
                total = db.execute(
                    "SELECT count(*) FROM predictions WHERE " + condition, parameters
                ).fetchone()[0]
                predictions = [
                    json.loads(r[0])
                    for r in db.execute(
                        "SELECT result FROM predictions WHERE "
                        + condition
                        + " ORDER BY sequence LIMIT 1000",
                        parameters,
                    )
                ]
            return {
                "order": order,
                "event": dict(event) if event else None,
                "predictions": predictions,
                "prediction_total": total,
                "truncated": total > len(predictions),
                "training_label_status": "not_exported_requires_independent_review",
                "asset_event_identity_matches": None
                if not event
                else order["asset_id"] == event["entity"],
            }

    def part(self, part_id, name):
        with self.store.connect() as db:
            db.execute(
                "INSERT INTO parts(id,name) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name",
                (part_id, name),
            )
            audit(db, "part.save", part_id, {"name": name})

    def stock(self, part_id, delta, reason, operator, order_id=None, request_id=None):
        if (
            delta == 0
            or (reason == "issue" and delta > 0)
            or (reason in ("receive", "return") and delta < 0)
        ):
            raise ValueError("备件数量与出入库方向不一致")
        if reason in ("issue", "return") and order_id is None:
            raise ValueError("领用和退库必须关联工单")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if request_id is not None:
                cached = db.execute(
                    "SELECT * FROM stock_moves WHERE request_id=?", (request_id,)
                ).fetchone()
                if cached:
                    if (
                        cached["part_id"],
                        cached["order_id"],
                        cached["delta"],
                        cached["reason"],
                        cached["operator"],
                    ) != (part_id, order_id, delta, reason, operator):
                        raise ValueError("请求编号已使用，但库存操作内容不同")
                    return {"duplicate": True}
            if reason == "return":
                outstanding = db.execute(
                    "SELECT COALESCE(-SUM(delta),0) FROM stock_moves WHERE part_id=? AND order_id=? AND reason IN ('issue','return')",
                    (part_id, order_id),
                ).fetchone()[0]
                if delta > outstanding:
                    raise ValueError("退库数量超过该工单尚未退回的领用数量")
            count = db.execute(
                "UPDATE parts SET quantity=quantity+? WHERE id=? AND quantity+?>=0",
                (delta, part_id, delta),
            ).rowcount
            if not count:
                raise ValueError("备件不存在或库存不足")
            db.execute(
                "INSERT INTO stock_moves(part_id,order_id,delta,reason,operator,request_id) VALUES(?,?,?,?,?,?)",
                (part_id, order_id, delta, reason, operator, request_id),
            )
            audit(
                db,
                "stock.move",
                part_id,
                {
                    "delta": delta,
                    "reason": reason,
                    "operator": operator,
                    "order_id": order_id,
                },
            )
            return {"duplicate": False}

    def person(self, person_id, name, skills, shift):
        with self.store.connect() as db:
            db.execute(
                "INSERT INTO people VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,skills=excluded.skills,shift=excluded.shift",
                (person_id, name, json.dumps(skills, ensure_ascii=False), shift),
            )
            audit(db, "person.save", person_id, {"name": name})

    def plan(self, asset_id, name, interval_days, steps, next_due):
        if (
            not steps
            or any(not step.strip() for step in steps)
            or len(set(steps)) != len(steps)
        ):
            raise ValueError("点检计划需要至少一项检查内容")
        with self.store.connect() as db:
            cursor = db.execute(
                "INSERT INTO plans(asset_id,name,interval_days,steps,next_due) VALUES(?,?,?,?,?)",
                (
                    asset_id,
                    name,
                    interval_days,
                    json.dumps(steps, ensure_ascii=False),
                    next_due,
                ),
            )
            audit(
                db,
                "plan.create",
                cursor.lastrowid,
                {"asset_id": asset_id, "steps": steps},
            )
            return cursor.lastrowid

    def check(self, plan_id, operator, results, notes):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM plans WHERE id=?", (plan_id,)).fetchone()
            if row is None or set(results) != set(json.loads(row["steps"])):
                raise ValueError("点检结果须覆盖计划中的每一项")
            failed = [name for name, passed in results.items() if not passed]
            if failed and not notes.strip():
                raise ValueError("点检未通过时需要填写执行记录")
            order_id = None
            if failed:
                order_id = db.execute(
                    "INSERT INTO work_orders(asset_id,title,operator,kind,notes) VALUES(?,?,?,'inspection',?)",
                    (row["asset_id"], row["name"] + "：点检未通过", operator, notes),
                ).lastrowid
                audit(
                    db,
                    "order.from-check",
                    order_id,
                    {"plan_id": plan_id, "failed_items": failed},
                )
            db.execute(
                "INSERT INTO checks(plan_id,operator,result,notes,order_id) VALUES(?,?,?,?,?)",
                (
                    plan_id,
                    operator,
                    json.dumps(results, ensure_ascii=False),
                    notes,
                    order_id,
                ),
            )
            db.execute(
                "UPDATE plans SET next_due=date('now','+'||interval_days||' days') WHERE id=?",
                (plan_id,),
            )
            audit(db, "plan.check", plan_id, {"operator": operator, "results": results})
            return {"order_id": order_id}
