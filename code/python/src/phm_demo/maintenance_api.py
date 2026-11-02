"""设备、维修、备件、点检与人员的本地演示接口。"""

from datetime import date
from typing import Literal
from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel, Field

from phm_demo.services import execute, maintenance_store

router = APIRouter(prefix="/api/maintenance")


class Asset(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: Literal["line", "equipment", "station", "component"]
    parent_id: str | None = None
    details: dict = Field(default_factory=dict)


class Order(BaseModel):
    asset_id: str
    title: str = Field(min_length=1)
    operator: str = Field(min_length=1)
    kind: Literal["repair", "inspection", "service"] = "repair"
    event_id: int | None = None


class Transition(BaseModel):
    status: Literal["working", "completed", "closed", "cancelled"]
    revision: int = Field(ge=1)
    operator: str = Field(min_length=1)
    notes: str = ""
    cause: str = ""
    verification: str = ""


class Part(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)


class Stock(BaseModel):
    delta: int
    reason: Literal["receive", "issue", "return"]
    operator: str = Field(min_length=1)
    order_id: int | None = None
    request_id: UUID


class Person(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    skills: list[str]
    shift: str


class Plan(BaseModel):
    asset_id: str
    name: str = Field(min_length=1)
    interval_days: int = Field(gt=0)
    steps: list[str]
    next_due: date


class Check(BaseModel):
    operator: str = Field(min_length=1)
    results: dict[str, bool]
    notes: str = ""


@router.get("")
def snapshot():
    return maintenance_store().snapshot()


@router.put("/assets")
def asset(body: Asset):
    execute(
        maintenance_store().asset,
        body.id,
        body.name,
        body.kind,
        body.parent_id,
        body.details,
    )
    return {"status": "saved"}


@router.post("/orders", status_code=201)
def create_order(body: Order):
    return {"id": execute(maintenance_store().create_order, **body.model_dump())}


@router.post("/orders/{order_id}/transition")
def transition(order_id: int, body: Transition):
    execute(maintenance_store().transition, order_id, **body.model_dump())
    return {"status": body.status}


@router.put("/parts")
def part(body: Part):
    execute(maintenance_store().part, body.id, body.name)
    return {"status": "saved"}


@router.post("/parts/{part_id}/stock")
def stock(part_id: str, body: Stock):
    payload = body.model_dump()
    payload["request_id"] = str(body.request_id)
    result = execute(maintenance_store().stock, part_id, **payload)
    return {"status": "saved", **result}


@router.put("/people")
def person(body: Person):
    execute(maintenance_store().person, body.id, body.name, body.skills, body.shift)
    return {"status": "saved"}


@router.post("/plans", status_code=201)
def plan(body: Plan):
    return {"id": execute(maintenance_store().plan, **body.model_dump(mode="json"))}


@router.post("/plans/{plan_id}/check")
def check(plan_id: int, body: Check):
    result = execute(maintenance_store().check, plan_id, **body.model_dump())
    return {"status": "recorded", **result}
