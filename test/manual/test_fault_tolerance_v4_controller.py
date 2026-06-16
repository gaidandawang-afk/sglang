import importlib.util
import sys
from pathlib import Path

import pytest


def load_controller():
    path = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "sglang"
        / "srt"
        / "fault_tolerance"
        / "controller.py"
    )
    spec = importlib.util.spec_from_file_location("ft_controller", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FaultToleranceManager = load_controller().FaultToleranceManager


def make_manager(dp_size=4, enabled=True, mooncake=True, strategy="pause"):
    return FaultToleranceManager(
        enabled=enabled,
        dp_size=dp_size,
        on_error_strategy=strategy,
        recovery_timeout_sec=60,
        moe_a2a_backend="none",
        elastic_ep_backend="mooncake" if mooncake else None,
    )


def test_status_running_shape():
    mgr = make_manager(dp_size=2)
    status = mgr.status_response()
    assert status["enabled"] is True
    assert status["instance_state"] == "running"
    assert status["admission_open"] is True
    assert status["active_mask"] == [True, True]
    assert status["last_fault"] is None


def test_recoverable_fault_retry_lifecycle():
    mgr = make_manager(dp_size=2)
    mgr.pause_all_active("recoverable fault", rank=1)
    assert mgr.status_response()["instance_state"] == "paused"
    assert mgr.status_response()["active_mask"] == [False, False]
    assert mgr.live_mask() == [True, True]
    assert mgr.validate_retry() is None
    assert mgr.begin_retry()["success"] is True
    assert mgr.status_response()["instance_state"] == "recovering"
    assert mgr.commit_retry()["success"] is True
    assert mgr.status_response()["instance_state"] == "running"


def test_retry_rejects_running_and_dead_rank():
    mgr = make_manager(dp_size=2)
    assert mgr.validate_retry() == "already_running"
    mgr.record_fault(1, "rank 1 exited")
    error = mgr.validate_retry()
    assert error is not None
    assert "scale_down" in error


def test_scale_down_lifecycle():
    mgr = make_manager(dp_size=3)
    mgr.record_fault(1, "rank 1 exited")
    assert mgr.begin_scale_down([1])["success"] is True
    assert mgr.status_response()["active_mask"] == [False, False, False]
    assert mgr.live_mask() == [True, False, True]
    assert mgr.commit_scale_down()["success"] is True
    status = mgr.status_response()
    assert status["instance_state"] == "degraded_running"
    assert [item["state"] for item in status["ranks"]] == [
        "healthy",
        "dead",
        "healthy",
    ]


def test_scale_down_validation_edges():
    mgr = make_manager(dp_size=2)
    assert mgr.validate_scale_down([]) is not None
    assert mgr.validate_scale_down([9]) is not None
    assert mgr.validate_scale_down([0, 1]) is not None
    mgr = make_manager(dp_size=2, mooncake=False)
    assert "mooncake" in mgr.validate_scale_down([1])


def test_apply_rejected_while_recovering():
    mgr = make_manager(dp_size=2)
    mgr.pause_all_active("recoverable fault", rank=0)
    assert mgr.begin_retry()["success"] is True
    assert "already in progress" in mgr.validate_retry()
    assert "already in progress" in mgr.validate_scale_down([1])


def test_last_active_non_recoverable_fault_fails_instance():
    mgr = make_manager(dp_size=1)
    mgr.record_fault(0, "rank 0 exited")
    status = mgr.status_response()
    assert status["instance_state"] == "failed"
    assert status["active_mask"] == [False]


def test_continue_recoverable_fault_keeps_healthy_ranks_active():
    mgr = make_manager(dp_size=4, strategy="continue")
    mgr.pause_all_active("recoverable fault", rank=2)
    status = mgr.status_response()
    assert status["instance_state"] == "degraded_running"
    assert status["admission_open"] is True
    assert [item["state"] for item in status["ranks"]] == [
        "healthy",
        "healthy",
        "paused",
        "healthy",
    ]
    assert status["active_mask"] == [True, True, False, True]
    assert mgr.live_mask() == [True, True, True, True]
    assert mgr.paused_ranks() == [2]
    assert mgr.validate_retry() is None


def test_continue_retry_only_recovers_paused_rank():
    mgr = make_manager(dp_size=3, strategy="continue")
    mgr.pause_all_active("recoverable fault", rank=1)
    assert mgr.begin_retry()["success"] is True
    status = mgr.status_response()
    assert status["instance_state"] == "recovering"
    assert status["admission_open"] is True
    assert [item["state"] for item in status["ranks"]] == [
        "healthy",
        "paused",
        "healthy",
    ]
    assert mgr.paused_ranks() == [1]
    assert mgr.commit_retry()["success"] is True
    assert [item["state"] for item in mgr.status_response()["ranks"]] == [
        "healthy",
        "healthy",
        "healthy",
    ]


def test_continue_non_recoverable_fault_keeps_healthy_ranks_active():
    mgr = make_manager(dp_size=4, strategy="continue")
    mgr.record_fault(1, "rank 1 exited")
    status = mgr.status_response()
    assert status["instance_state"] == "degraded_running"
    assert status["admission_open"] is True
    assert [item["state"] for item in status["ranks"]] == [
        "healthy",
        "dead",
        "healthy",
        "healthy",
    ]
    assert status["active_mask"] == [True, False, True, True]
    assert mgr.live_mask() == [True, False, True, True]
    assert "scale_down" in mgr.validate_retry()


def test_continue_routed_paused_or_dead_rank_rejected():
    mgr = make_manager(dp_size=3, strategy="continue")
    mgr.pause_all_active("recoverable fault", rank=1)
    with pytest.raises(ValueError):
        mgr.validate_routed_rank(1)
    mgr.record_fault(2, "rank 2 exited")
    with pytest.raises(ValueError):
        mgr.validate_routed_rank(2)
    mgr.validate_routed_rank(0)


def test_retry_failure_rolls_back_recovery_state():
    mgr = make_manager(dp_size=3, strategy="continue")
    mgr.pause_all_active("recoverable fault", rank=1)
    assert mgr.begin_retry()["success"] is True
    result = mgr.rollback_recovery("retry failed")
    assert result["success"] is False
    status = mgr.status_response()
    assert status["instance_state"] == "degraded_running"
    assert [item["state"] for item in status["ranks"]] == [
        "healthy",
        "paused",
        "healthy",
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
