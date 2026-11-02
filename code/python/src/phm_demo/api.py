"""HTTP 只负责产物与任务调度；算法计算不依赖 Web 框架。"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from phm_core.artifacts import run_path
from phm_core.decision import DecisionConfig, DecisionState, step
from phm_core.valve import mechanism_features, mechanism_score
from phm_demo.benchmarks_api import router as benchmarks_router
from phm_demo.diagnostics_api import router as diagnostics_router
from phm_demo.domain_api import router as domain_router
from phm_demo.maintenance_api import router as maintenance_router
from phm_demo.observability import install_observability
from phm_demo.repository import ROOT, TASKS, list_runs, metadata, replay_data, resolve
from phm_demo.runtime_api import router as runtime_router
from phm_demo.security import install_access_control
from phm_demo.services import execute, state_store
from phm_runtime.jobs import JobStore

Task = Literal["valve", "cooler", "pump", "accumulator"]
Model = Literal["ae", "iforest", "mad", "rf"]
app = FastAPI(title="机理实验室", version="0.4.0")
install_access_control(app)
install_observability(app)
app.include_router(benchmarks_router)
app.include_router(diagnostics_router)
app.include_router(domain_router)
app.include_router(runtime_router)
app.include_router(maintenance_router)


def artifact(task: str, name: str, run_id: str | None = None) -> Path:
    try:
        path = resolve(task, run_id) / name
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(404, str(error)) from error
    if not path.exists():
        raise HTTPException(404, "该部件尚无实验结果，请先运行训练")
    return path


@app.get("/api/summary")
def summary():
    result = []
    for task in TASKS:
        try:
            result.append(metadata(resolve(task)))
        except FileNotFoundError:
            continue
    return result


@app.get("/api/runs/{task}")
def runs(task: Task):
    return list_runs(task)


@app.get("/api/replay/{task}")
def replay(
    task: Task,
    model: Model = "ae",
    run_id: str | None = Query(None, pattern=r"^[A-Za-z0-9_-]+$"),
):
    directory = artifact(task, "metrics.json", run_id).parent
    metrics = metadata(directory)
    splits = json.loads((directory / "split.json").read_text(encoding="utf-8"))
    threshold = metrics["models"][model]["threshold"]
    events, state = [], DecisionState()
    config = DecisionConfig(
        enter=threshold, exit=threshold - max(abs(threshold) * 0.2, 1e-8)
    )
    data = replay_data(directory)
    scores = data[model]
    # 回放告警用于演示持续性，实验周期不等于自然退化时间线。
    for i, score in enumerate(scores):
        if i % 30 == 0:
            if state.active:
                events[-1]["end"] = i - 1
                events[-1]["reason"] = "block-end"
            state = DecisionState()
        event, state = step(float(score), state, config)
        if event == "opened":
            events.append(
                {"start": i, "end": None, "score": float(score), "reason": "data-end"}
            )
        elif event == "closed":
            events[-1]["end"] = i
            events[-1]["reason"] = "recovery"
        if state.active:
            events[-1]["score"] = max(events[-1]["score"], float(score))
    if events and events[-1]["end"] is None:
        events[-1]["end"] = len(scores) - 1
    partitions = np.full(len(scores), "excluded", dtype="<U12")
    for name, ids in splits.items():
        partitions[ids] = name
    return {
        "run_id": metrics["run_id"],
        "partitions": partitions.tolist(),
        "scores": scores.tolist(),
        "threshold": threshold,
        "healthy": data["healthy"].tolist(),
        "events": events,
        "splits": splits,
    }


@app.get("/api/cycle/{task}/{cycle}")
def cycle(
    task: Task,
    cycle: int,
    channel: int = Query(0, ge=0),
    run_id: str | None = Query(None, pattern=r"^[A-Za-z0-9_-]+$"),
):
    directory = artifact(task, "metrics.json", run_id).parent
    data = replay_data(directory)
    if channel >= data["x"].shape[2]:
        raise HTTPException(422, "通道索引超出范围")
    if not 0 <= cycle < len(data["x"]):
        raise HTTPException(422, "周期索引超出范围")
    result = {
        "actual": data["x"][cycle, :, channel].tolist(),
        "reconstruction": data["reconstruction"][cycle, :, channel].tolist(),
        "healthy": bool(data["healthy"][cycle]),
    }
    reference_file = directory / "mechanism-reference.json"
    if reference_file.exists():
        reference = json.loads(reference_file.read_text(encoding="utf-8"))
        features = mechanism_features(data["x"][cycle : cycle + 1])
        score = float(mechanism_score(features, reference)[0])
        result["mechanism"] = {
            "score": score,
            "threshold": reference["threshold"],
            "flagged": score > reference["threshold"],
            "items": [
                {
                    "name": name,
                    "value": float(features[0, i]),
                    "center": reference["center"][i],
                    "upper": reference["center"][i]
                    + reference["threshold"] * reference["scale"][i],
                    "lower": None
                    if i < 6
                    else reference["center"][i]
                    - reference["threshold"] * reference["scale"][i],
                    "unit": "s",
                }
                for i, name in enumerate(reference["names"])
            ],
        }
    return result


@app.get("/api/download/{task}")
def download(task: Task, run_id: str | None = Query(None, pattern=r"^[A-Za-z0-9_-]+$")):
    return FileResponse(
        artifact(task, "model-bundle.zip", run_id),
        filename=f"mechanism-lab-{task}-{run_id or 'latest'}-models.zip",
    )


@app.get("/api/job")
def job():
    current = JobStore(state_store()).latest()
    if current is None:
        return {"status": "idle", "task": None, "log": ""}
    path = Path(current["log_path"])
    log = ""
    if path.exists():
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 4000))
            log = stream.read().decode("utf-8", errors="replace")
    status = {"queued": "running", "timed_out": "failed"}.get(
        current["status"], current["status"]
    )
    if current["stale"]:
        log += "\n工作者心跳已中断；未自动重启第二个训练进程，请检查原工作者。"
    return {
        "status": status,
        "task": current["task"],
        "log": log,
        "id": current["id"],
        "worker_status": current["status"],
        "stale": current["stale"],
        "cancel_requested": bool(current["cancel"]),
    }


class TrainOptions(BaseModel):
    profile: Literal["thermal-1hz", "process-10hz", "valve-switch-100hz"] = (
        "process-10hz"
    )
    epochs: int = Field(40, ge=1, le=200)
    architecture: Literal["conv", "dense"] = "conv"
    normal_policy: Literal["stable", "all"] = "stable"


@app.post("/api/train/{task}", status_code=202)
def train(task: Task, options: TrainOptions | None = None):
    options = options if options is not None else TrainOptions()
    if options.profile == "valve-switch-100hz" and task != "valve":
        raise HTTPException(422, "换向窗口配置仅用于阀门研究")
    command = [
        sys.executable,
        str(ROOT / "code/python/train_hydraulic.py"),
        "--task",
        task,
        "--profile",
        options.profile,
        "--epochs",
        str(options.epochs),
        "--architecture",
        options.architecture,
        "--normal-policy",
        options.normal_policy,
    ]
    directory = ROOT / "work/jobs" / uuid4().hex
    directory.mkdir(parents=True)
    jobs = JobStore(state_store())
    job_id = execute(jobs.create, task, command, directory / "training.log")
    try:
        with (directory / "worker.log").open("wb") as log:
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "phm_runtime.job_worker",
                    "--database",
                    str(state_store().path.resolve()),
                    "--job",
                    job_id,
                    "--root",
                    str(ROOT),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
    except OSError:
        jobs.finish(job_id, "failed", detail="worker_spawn_failed")
        raise
    return {"status": "running", "task": task, "id": job_id}


class CancelOptions(BaseModel):
    job_id: str = Field(pattern=r"^[a-f0-9]{32}$")


@app.post("/api/job/cancel")
def cancel(options: CancelOptions):
    jobs = JobStore(state_store())
    execute(jobs.cancel, options.job_id)
    return {"status": "cancelling", "id": options.job_id}


@app.get("/api/valve-study")
def valve_study():
    try:
        path = run_path(ROOT / "outputs", "valve-study")
    except FileNotFoundError:
        return None
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


@app.get("/api/valve-study/download")
def valve_study_download(run_id: str = Query(pattern=r"^[A-Za-z0-9_-]+$")):
    try:
        path = run_path(ROOT / "outputs", "valve-study", run_id)
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    return FileResponse(
        path / "model-bundle.zip", filename=f"mechanism-lab-valve-study-{run_id}.zip"
    )


dist = ROOT / "code/ts/demo/dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="demo")
