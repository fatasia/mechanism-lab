"""可供外部调用的执行器诊断接口，模型版本和输入适用域显式返回。"""

import json
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from phm_core.quality import assess_window
from phm_demo.benchmarks_api import directory, graph

router = APIRouter(prefix="/api/diagnostics")


class DiagnosticInput(BaseModel):
    run_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    values: list[list[float | None]]
    sample_rate_hz: float = Field(gt=0)
    channels: list[str]
    units: list[str]
    profile: Literal["sin", "trap"]
    load_kgf: Literal["20", "40", "neg40"]


@router.post("/{family}")
def predict(family: Literal["actuator-ensemble", "minirocket"], body: DiagnosticInput):
    kind = (
        "actuator-ensemble-study"
        if family == "actuator-ensemble"
        else "minirocket-export"
    )
    path = directory(kind, body.run_id)
    report = json.loads((path / "metrics.json").read_text())
    if family == "actuator-ensemble":
        contract = report["contracts"]["record-holdout/multiscale_ensemble"]
        length, score_kind = 2000, "temperature_calibrated_class_probability"
    else:
        contract = report["models"]["record-holdout"]
        length, score_kind = 400, "uncalibrated_class_score"
    if body.channels != [
        "position_setpoint",
        "following_error",
        "motor_current",
    ] or body.units != ["mm", "mm", "A"]:
        raise HTTPException(422, "需要按契约传入指令位置 mm、跟随误差 mm、电流 A")
    values = np.asarray(body.values, dtype="float32")
    quality = assess_window(values, (length, 3), body.sample_rate_hz, 25)
    result = {
        "version": body.run_id,
        "family": family,
        "quality": quality,
        "scope": "Cranfield single rig, six observed load/profile contexts; external equipment not validated",
        "status": "unscored",
        "score_kind": score_kind,
        "scores": None,
        "candidate_set": None,
        "classes": report.get("classes", contract.get("classes")),
    }
    if quality["status"] == "valid":
        session = graph(path / "bundle" / contract["file"], contract["sha256"])
        outputs = session.run(None, {"values": values[None]})
        result.update(status="scored", scores=outputs[0][0].tolist())
        if family == "actuator-ensemble":
            result["candidate_set"] = outputs[1][0].tolist()
            result["decision_status"] = (
                "singleton"
                if outputs[1][0].sum() == 1
                else "ambiguous"
                if outputs[1][0].sum() > 1
                else "abstain"
            )
    return result
