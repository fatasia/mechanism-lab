"""公开基准结果、版本化下载及执行器逐记录推理。"""

import hashlib
import json
from functools import lru_cache
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from phm_core.actuator import CLASSES, FEATURES, align_motion, motion_features
from phm_core.artifacts import run_path
from phm_core.inference import cpu_session
from phm_demo.repository import ROOT

router = APIRouter(prefix="/api/benchmarks")
Kind = Literal[
    "actuator-study",
    "actuator-transfer-study",
    "actuator-release-study",
    "actuator-ensemble-study",
    "rotating-study",
    "metro-study",
    "reference-study",
    "minirocket-export",
    "deployments",
    "rul-study",
    "vision-study",
    "conditioning-study",
    "external-actuator-study",
    "context-selection-study",
]
VERSION = r"^[A-Za-z0-9_-]+$"


def directory(kind, run_id=None):
    try:
        return run_path(ROOT / "outputs", kind, run_id)
    except FileNotFoundError as error:
        raise HTTPException(404, "该实验尚未完成") from error


@router.get("/vision/image/{image_id}")
def visual_image(
    image_id: int,
    run_id: str = Query(pattern=VERSION),
    layer: Literal["raw", "heat"] = "raw",
):
    path = directory("vision-study", run_id)
    report = json.loads((path / "metrics.json").read_text())
    if not 0 <= image_id < len(report["records"]):
        raise HTTPException(404, "该版本没有这张图像")
    name = f"{image_id}.jpg" if layer == "raw" else f"{image_id}-heat.png"
    return FileResponse(path / "previews" / name)


@router.get("/{kind}")
def benchmark(kind: Kind):
    try:
        path = run_path(ROOT / "outputs", kind)
    except FileNotFoundError:
        return None
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


@router.get("/{kind}/download")
def download(kind: Kind, run_id: str = Query(pattern=VERSION)):
    return FileResponse(
        directory(kind, run_id) / "model-bundle.zip",
        filename=f"mechanism-lab-{kind}-{run_id}.zip",
    )


@router.get("/deployments/package/{model_id}")
def deployment_package(model_id: str, run_id: str = Query(pattern=VERSION)):
    path = directory("deployments", run_id)
    catalog = json.loads((path / "metrics.json").read_text())
    entry = next((m for m in catalog["models"] if m["id"] == model_id), None)
    if entry is None:
        raise HTTPException(404, "该部署版本没有这个模型")
    return FileResponse(
        path / entry["file"], filename=f"mechanism-lab-{model_id}-{run_id}.zip"
    )


@lru_cache(maxsize=10)
def graph(path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise HTTPException(409, "诊断模型校验和不一致")
    return cpu_session(path)


@router.get("/{kind}/record/{record}")
def actuator_record(
    kind: Literal[
        "actuator-study",
        "actuator-transfer-study",
        "actuator-release-study",
        "actuator-ensemble-study",
    ],
    record: int,
    protocol: Literal["held-out-load", "held-out-profile", "record-holdout"]
    | None = None,
    cycle: int = Query(0, ge=0, le=4),
    arm: Literal[
        "mechanism_rf",
        "current_only_rf",
        "error_only_rf",
        "random_kernels",
        "multiscale_fusion",
        "multiscale_ensemble",
    ]
    | None = None,
    run_id: str | None = Query(None, pattern=VERSION),
):
    path = directory(kind, run_id)
    report = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    protocol = protocol or next(iter(report["protocols"]))
    if protocol not in report["protocols"]:
        raise HTTPException(422, "该版本不包含所选评估协议")
    metadata = json.loads((path / "records.json").read_text())
    if not 0 <= record < len(metadata):
        raise HTTPException(422, "记录索引超出范围")
    selected = arm or report["protocols"][protocol]["selected"]["arm"]
    if selected not in report["protocols"][protocol]["arms"]:
        raise HTTPException(422, "该版本不包含所选方法")
    contract = report["contracts"][f"{protocol}/{selected}"]
    with np.load(path / "replay.npz") as data:
        raw = data["x"][data["record_ids"] == record]
    values = raw
    if report["protocol"]["reference_alignment"]:
        references = json.loads(
            (path / "bundle" / f"phase-reference-{protocol}.json").read_text()
        )
        context = metadata[record]["profile"] + "/" + metadata[record]["load"]
        values = align_motion(raw, [context] * 5, references)
    model_input = (
        motion_features(values)[:, contract["feature_columns"]]
        if contract["input_kind"] == "motion_features"
        else values
    )
    if contract["input_kind"] == "full_record":
        model_input = raw.reshape(1, 2000, 3)
    session = graph(path / "bundle" / contract["file"], contract["sha256"])
    outputs = session.run(None, {"values": model_input})
    score = outputs[contract["output_index"]]
    split = json.loads((path / f"split-{protocol}.json").read_text())
    return {
        "record": metadata[record],
        "partition": next(k for k, ids in split.items() if record in ids),
        "run_id": report["run_id"],
        "model": selected,
        "classes": CLASSES,
        "score_kind": contract["score_kind"],
        "scores": score.mean(0).tolist(),
        "predicted": int(score.mean(0).argmax()),
        "cycle": cycle,
        "channels": ["指令位置 / mm", "跟随误差 / mm", "电流 / A"],
        "sample_rate_hz": 25,
        "values": raw[cycle].tolist(),
        "mechanism": dict(
            zip(FEATURES, motion_features(raw[cycle : cycle + 1])[0].tolist())
        ),
        "reference_alignment": report["protocol"]["reference_alignment"],
        "candidate_set": outputs[contract["candidate_set_output_index"]][0].tolist()
        if "candidate_set_output_index" in contract
        else None,
    }
