import importlib.util
import urllib.error
from pathlib import Path
from unittest.mock import Mock

import pytest

SPEC = importlib.util.spec_from_file_location(
    "service_process", Path(__file__).resolve().parents[1] / "service_process.py"
)
service = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(service)


def test_reset_during_startup_is_retried_and_counted():
    process = Mock()
    process.poll.return_value = None
    probe = Mock(
        side_effect=[ConnectionResetError(), urllib.error.URLError("refused"), {}]
    )
    result = service.wait_ready(process, probe, interval=0)
    assert result == {
        "attempts": 3,
        "transport_errors": {"ConnectionResetError": 1, "URLError": 1},
    }


def test_application_error_is_not_hidden():
    process = Mock()
    process.poll.return_value = None
    probe = Mock(side_effect=urllib.error.HTTPError("local", 500, "failed", {}, None))
    with pytest.raises(urllib.error.HTTPError):
        service.wait_ready(process, probe)
    assert probe.call_count == 1


def test_failed_start_cleans_owned_process(monkeypatch):
    process = Mock()
    process.poll.return_value = 1
    monkeypatch.setattr(service, "spawn_process", Mock(return_value=process))
    stop = Mock()
    monkeypatch.setattr(service, "stop_service", stop)
    with pytest.raises(RuntimeError, match="退出"):
        service.start_service([], None, {}, None, lambda: None)
    stop.assert_called_once_with(process)


def test_startup_deadline_is_bounded():
    process = Mock()
    process.poll.return_value = None
    with pytest.raises(TimeoutError):
        service.wait_ready(
            process,
            Mock(side_effect=ConnectionResetError()),
            timeout=0.01,
            interval=0.001,
        )
