"""独立推理、持久告警、版本激活/回退和质量/漂移工具。"""

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from phm_core.monitoring import compare_peers, distribution_drift
from phm_demo.repository import ROOT, TASKS, metadata, resolve
from phm_demo.services import execute, model_registry, state_store
from phm_runtime.outbox import Outbox
from phm_runtime.observability import storage_snapshot
from phm_runtime.registry import promotion_gate

router = APIRouter(prefix="/api/runtime")
Task = Literal["valve", "cooler", "pump", "accumulator"]


class Activation(BaseModel):
    run_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    model: Literal["ae", "iforest", "rf", "mad"] = "ae"
    mode: Literal["shadow", "active"] = "shadow"


class Synchronization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command_edge_s: FiniteFloat | None = None
    window_start_s: FiniteFloat | None = None
    channel_skew_s: FiniteFloat | None = None


class Window(BaseModel):
    task: Task
    entity: str = Field(min_length=1, max_length=200)
    sequence: int = Field(ge=0)
    values: list[list[float | None]]
    channels: list[str]
    units: list[str]
    sample_rate_hz: float = Field(gt=0)
    timestamps: list[float] | None = None
    sync: Synchronization | None = None


class Feedback(BaseModel):
    operator: str = Field(min_length=1)
    verdict: Literal["confirmed_fault", "false_alarm", "unconfirmed"]


class Peer(BaseModel):
    entity: str = Field(min_length=1)
    cohort: str = Field(min_length=1)
    values: list[float]


class Peers(BaseModel):
    target_entity: str
    cohort: str
    values: list[float]
    peers: list[Peer]
    scale_floor: float = Field(gt=0)


class Drift(BaseModel):
    reference: list[list[float]]
    current: list[list[float]]


@router.get("/monitor")
def monitor(request: Request):
    store = state_store()
    with store.connect() as database:
        queue = {
            r["status"]: r["n"]
            for r in database.execute(
                "SELECT status,count(*) n FROM outbox GROUP BY status"
            )
        }
    return {
        "http": request.app.state.request_metrics.snapshot(),
        "storage": storage_snapshot(store.path),
        "notification_counts": queue,
        "scope": "single process; bounded latency window; no site acceptance",
    }


@router.get("/status")
def status():
    gates = {}
    for task in TASKS:
        try:
            gates[task] = promotion_gate(metadata(resolve(task)), "ae")
        except FileNotFoundError:
            gates[task] = None
    return {
        "bindings": {task: state_store().binding(task) for task in TASKS},
        "gates": gates,
        "events": state_store().events(),
        "notifications": Outbox(state_store()).snapshot(),
    }


@router.post("/notifications/{notification_id}/retry")
def retry_notification(notification_id: int):
    execute(Outbox(state_store()).retry, notification_id)
    return {"status": "pending"}


@router.post("/activate/{task}")
def activate(task: Task, body: Activation):
    return execute(model_registry().activate, task, body.run_id, body.model, body.mode)


@router.post("/predict")
def predict(body: Window):
    return execute(
        model_registry().predict,
        body.task,
        body.entity,
        body.sequence,
        body.values,
        body.channels,
        body.units,
        body.sample_rate_hz,
        body.timestamps,
        body.sync.model_dump() if body.sync is not None else None,
    )


@router.get("/history")
def history(entity: str, limit: int = Query(200, ge=1, le=1000)):
    return state_store().history(entity, limit)


@router.get("/events")
def events():
    return state_store().events()


@router.post("/events/{event_id}/feedback")
def feedback(event_id: int, body: Feedback):
    execute(state_store().feedback, event_id, body.operator, body.verdict)
    return {"status": "reviewed"}


@router.post("/peers")
def peers(body: Peers):
    from fastapi import HTTPException

    entities = [p.entity for p in body.peers]
    if (
        body.target_entity in entities
        or len(set(entities)) != len(entities)
        or any(p.cohort != body.cohort for p in body.peers)
    ):
        raise HTTPException(422, "同类群须排除自身、实体不重复且工况/规格分组一致")
    return execute(
        compare_peers, body.values, [p.values for p in body.peers], body.scale_floor
    )


@router.post("/drift")
def drift(body: Drift):
    return execute(distribution_drift, body.reference, body.current)


@router.get("/backup")
def backup():
    directory = ROOT / "work/backups"
    directory.mkdir(exist_ok=True)
    path = directory / f"runtime-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}.sqlite"
    state_store().backup(path)
    return FileResponse(path, filename=path.name)
