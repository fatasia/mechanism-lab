import pytest

from phm_runtime.maintenance import MaintenanceStore
from phm_runtime.store import StateStore


def create_store(tmp_path):
    store = MaintenanceStore(StateStore(tmp_path / "runtime.db"))
    store.asset("line", "Test line", "line")
    store.asset("motor", "Test motor", "component", "line")
    return store


def test_asset_hierarchy_prevents_cycles(tmp_path):
    store = create_store(tmp_path)
    with pytest.raises(ValueError, match="环"):
        store.asset("line", "Test line", "line", "motor")


def test_workflow_requires_processing_and_verification(tmp_path):
    store = create_store(tmp_path)
    order = store.create_order("motor", "Check vibration", "tester")
    with pytest.raises(ValueError):
        store.transition(order, "closed", 1, "tester")
    store.transition(order, "working", 1, "tester")
    with pytest.raises(ValueError):
        store.transition(order, "completed", 2, "tester")
    store.transition(
        order, "completed", 2, "tester", "Tightened mounting", "loose foundation"
    )
    store.transition(order, "closed", 3, "reviewer", verification="Recheck passed")
    snapshot = store.snapshot()
    assert snapshot["cases"][0]["action"] == "Tightened mounting"
    assert snapshot["cases"][0]["confirmed_by"] == "reviewer"
    with pytest.raises(ValueError, match="版本"):
        store.transition(order, "working", 1, "tester")


def test_stock_issue_return_and_failed_transaction(tmp_path):
    store = create_store(tmp_path)
    order = store.create_order("motor", "Bearing replacement", "tester")
    store.part("6205", "Bearing")
    store.stock("6205", 5, "receive", "tester")
    store.stock("6205", -3, "issue", "tester", order)
    with pytest.raises(ValueError):
        store.stock("6205", -4, "issue", "tester", order)
    with pytest.raises(ValueError):
        store.stock("6205", 4, "return", "tester", order)
    store.stock("6205", 1, "return", "tester", order)
    assert store.snapshot()["parts"][0]["quantity"] == 3
    assert len(store.snapshot()["stock_moves"]) == 3


def test_inspection_is_complete_and_rescheduled(tmp_path):
    store = create_store(tmp_path)
    plan = store.plan("motor", "Weekly", 7, ["noise", "mounting"], "2026-10-31")
    with pytest.raises(ValueError):
        store.check(plan, "tester", {"noise": True}, "")
    store.check(
        plan, "tester", {"noise": True, "mounting": False}, "Mounting needs repair"
    )
    assert len(store.snapshot()["checks"]) == 1
    assert store.snapshot()["checks"][0]["order_id"] is not None
    assert store.snapshot()["work_orders"][0]["kind"] == "inspection"
