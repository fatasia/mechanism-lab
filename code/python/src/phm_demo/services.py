"""Web 层依赖工厂；部署、状态和运维库本身不依赖 FastAPI。"""

import os
import sqlite3
from functools import lru_cache

from fastapi import HTTPException

from phm_demo.repository import ROOT
from phm_runtime.maintenance import MaintenanceStore
from phm_runtime.registry import ModelRegistry
from phm_runtime.store import StateStore


@lru_cache(maxsize=1)
def state_store():
    from pathlib import Path

    return StateStore(
        Path(os.environ.get("MECHANISM_RUNTIME_DB", str(ROOT / "work/runtime.sqlite")))
    )


@lru_cache(maxsize=1)
def model_registry():
    return ModelRegistry(ROOT / "outputs", state_store())


def maintenance_store():
    return MaintenanceStore(state_store())


def execute(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except (ValueError, sqlite3.IntegrityError) as error:
        raise HTTPException(409, str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
