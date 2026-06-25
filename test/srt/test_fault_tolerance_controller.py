import importlib.util
import sys
from pathlib import Path


_CONTROLLER_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "srt"
    / "fault_tolerance"
    / "controller.py"
)
_SPEC = importlib.util.spec_from_file_location("ft_controller", _CONTROLLER_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

FaultToleranceManager = _MODULE.FaultToleranceManager
RankState = _MODULE.RankState


def make_manager(dp_size=4, strategy="pause"):
    return FaultToleranceManager(
        dp_size=dp_size,
        strategy=strategy,
    )


def test_pause_kill_then_retry_allows_dead_rank():
    manager = make_manager()

    targets = manager.record_kill(1)
    assert targets == [0, 2, 3]

    manager.finish_pause_collection(acked={0, 2, 3}, timed_out=set())
    assert manager.validate_apply("retry", None) is None

    active_mask, resume_targets, pending = manager.begin_recover("retry")
    assert active_mask == [True, False, True, True]
    assert resume_targets == [0, 2, 3]
    assert pending == []

    body = manager.commit_recover()
    assert [rank["state"] for rank in body["ranks"]] == [
        RankState.HEALTHY.value,
        RankState.DEAD.value,
        RankState.HEALTHY.value,
        RankState.HEALTHY.value,
    ]


def test_scale_down_marks_requested_rank_before_recover():
    manager = make_manager()
    manager.begin_exception_pause()
    manager.finish_pause_collection(acked={0, 1, 2, 3}, timed_out=set())

    assert manager.validate_apply("scale_down", [2]) is None
    active_mask, resume_targets, pending = manager.begin_recover("scale_down", [2])

    assert active_mask == [True, True, False, True]
    assert resume_targets == [0, 1, 2, 3]
    assert pending == [2]


def test_apply_rejected_without_paused_rank():
    manager = make_manager()
    assert manager.validate_apply("retry", None) == "no_paused_rank"


def test_scale_down_cannot_isolate_all_remaining_ranks():
    manager = make_manager(dp_size=2)
    manager.record_kill(1)
    manager.finish_pause_collection(acked={0}, timed_out=set())

    assert (
        manager.validate_apply("scale_down", [0])
        == "cannot_isolate_all_active_ranks"
    )


def test_continue_kill_marks_dead_without_pause():
    manager = make_manager(strategy="continue")

    assert manager.record_kill(2) == []
    assert manager.rank_states[2] == RankState.DEAD
    assert not manager.ft_operation_in_progress
