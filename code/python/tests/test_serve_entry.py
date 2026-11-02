"""启动配置必须在数值库导入之前生效；不改变已运行服务的环境。"""

import importlib.util
import os
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "serve_demo", Path(__file__).resolve().parents[1] / "serve_demo.py"
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def test_thread_configuration_is_explicit(monkeypatch):
    names = (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for name in names:
        monkeypatch.setenv(name, "24")
    entry.configure_threads(1)
    assert all(os.environ[name] == "1" for name in names)


def test_invalid_thread_count_is_rejected():
    with pytest.raises(ValueError, match="正"):
        entry.configure_threads(0)
