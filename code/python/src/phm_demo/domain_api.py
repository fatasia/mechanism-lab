"""视觉与寿命的无状态推理；与持久告警服务分开。"""

from functools import lru_cache
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from phm_core.domain_inference import DomainScorer
from phm_demo.benchmarks_api import directory

router = APIRouter(prefix="/api/domain")


class Input(BaseModel):
    run_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    values: list
    channels: list[str] | None = None
    layout: Literal["BTC", "BCHW"]


@lru_cache(maxsize=4)
def scorer(path):
    return DomainScorer(path)


@router.post("/{family}")
def predict(family: Literal["vision", "rul"], body: Input):
    model = scorer(
        directory("vision-study" if family == "vision" else "rul-study", body.run_id)
    )
    contract = model.manifest["input"]
    if body.layout != contract["layout"] or (
        family == "rul" and body.channels != contract["channels"]
    ):
        raise HTTPException(422, "输入布局或传感器通道顺序与模型契约不一致")
    try:
        values = model.score(body.values)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return {
        "status": "scored",
        "family": family,
        "version": body.run_id,
        "scope": model.manifest["scope"],
        "outputs": {k: np.asarray(v).tolist() for k, v in values.items()},
    }
