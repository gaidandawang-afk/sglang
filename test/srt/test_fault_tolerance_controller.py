import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


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
is_ft_supported_config = _MODULE.is_ft_supported_config


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

    ep_active_mask, dp_route_mask, resume_targets, pending = manager.begin_recover(
        "retry"
    )
    assert ep_active_mask == [True, False, True, True]
    assert dp_route_mask == [True, False, True, True]
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
    ep_active_mask, dp_route_mask, resume_targets, pending = manager.begin_recover(
        "scale_down", [2], shutdown_scale_down_ranks=True
    )

    assert ep_active_mask == [True, True, False, True]
    assert dp_route_mask == [True, True, False, True]
    assert resume_targets == [0, 1, 2, 3]
    assert pending == [2]


def test_apply_rejected_without_paused_rank():
    manager = make_manager()
    assert manager.validate_apply("retry", None) == "no_paused_rank"


def test_retry_rejects_ranks_parameter():
    manager = make_manager()
    manager.begin_exception_pause()
    manager.finish_pause_collection(acked={0, 1, 2, 3}, timed_out=set())

    assert manager.validate_apply("retry", []) == "retry_does_not_accept_ranks"


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
    assert manager.physical_rank_states[2] == RankState.DEAD
    assert manager.rank_states[2] == RankState.DEAD
    assert not manager.ft_operation_in_progress


def test_active_mask_recovers_dead_rank_without_resuming_paused_rank():
    manager = make_manager()
    manager.record_kill(3)
    manager.finish_pause_collection(acked={0, 1, 2}, timed_out=set())

    manager.record_inactive_mask([True, False, True, True])

    assert manager.rank_states == [
        RankState.PAUSED,
        RankState.DEAD,
        RankState.PAUSED,
        RankState.HEALTHY,
    ]


def test_tpgt1_kill_uses_global_rank_and_separate_ep_dp_masks():
    manager = FaultToleranceManager(
        dp_size=2,
        strategy="pause",
        global_rank_count=4,
        ranks_per_dp=2,
    )

    assert manager.dp_members == [[0, 1], [2, 3]]
    assert manager.dp_rank_for_global_rank(2) == 1

    targets = manager.record_kill(2)

    assert targets == [0, 1, 3]
    assert manager.physical_rank_states[2] == RankState.DEAD
    assert manager.rank_states == [RankState.HEALTHY, RankState.DEAD]
    assert manager.ep_active_mask() == [True, True, False, True]
    assert manager.dp_active_mask() == [True, False]


def test_tpgt1_scale_down_non_shutdown_preserves_healthy_sibling_expert():
    manager = FaultToleranceManager(
        dp_size=2,
        strategy="pause",
        global_rank_count=4,
        ranks_per_dp=2,
    )
    manager.record_kill(2)
    manager.finish_pause_collection(acked={0, 1, 3}, timed_out=set())

    ep_active_mask, dp_route_mask, resume_targets, pending = manager.begin_recover(
        "scale_down", [1], shutdown_scale_down_ranks=False
    )

    assert ep_active_mask == [True, True, False, True]
    assert dp_route_mask == [True, False]
    assert resume_targets == [0, 1, 3]
    assert pending == [1]
    manager.commit_recover(pending, shutdown_scale_down_ranks=False)
    assert manager.physical_rank_states[3] == RankState.HEALTHY
    assert manager.ep_active_mask() == [True, True, False, True]


def test_tpgt1_scale_down_live_targets_expand_dp_to_surviving_global_ranks():
    manager = FaultToleranceManager(
        dp_size=2,
        strategy="pause",
        global_rank_count=4,
        ranks_per_dp=2,
    )
    manager.record_kill(2)

    assert manager.expand_dp_ranks([1]) == [2, 3]
    assert manager.live_global_ranks_for_dp_ranks([1]) == [3]


def test_tpgt1_scale_down_commit_defaults_to_non_shutdown():
    manager = FaultToleranceManager(
        dp_size=2,
        strategy="pause",
        global_rank_count=4,
        ranks_per_dp=2,
    )
    manager.record_kill(2)
    manager.finish_pause_collection(acked={0, 1, 3}, timed_out=set())

    manager.commit_recover([1])

    assert manager.physical_rank_states[2] == RankState.DEAD
    assert manager.physical_rank_states[3] == RankState.HEALTHY
    assert manager.ep_active_mask() == [True, True, False, True]
    assert manager.dp_active_mask() == [True, False]


def test_tpgt1_scale_down_shutdown_expands_dp_to_all_global_ranks():
    manager = FaultToleranceManager(
        dp_size=2,
        strategy="pause",
        global_rank_count=4,
        ranks_per_dp=2,
    )
    manager.begin_exception_pause()
    manager.finish_pause_collection(acked={0, 1, 2, 3}, timed_out=set())

    ep_active_mask, dp_route_mask, resume_targets, pending = manager.begin_recover(
        "scale_down", [1], shutdown_scale_down_ranks=True
    )

    assert manager.expand_dp_ranks([1]) == [2, 3]
    assert ep_active_mask == [True, True, False, False]
    assert dp_route_mask == [True, False]
    assert resume_targets == [0, 1, 2, 3]
    assert pending == [1]
    manager.commit_recover(pending, shutdown_scale_down_ranks=True)
    assert manager.physical_rank_states[2] == RankState.DEAD
    assert manager.physical_rank_states[3] == RankState.DEAD


def test_tpgt1_active_dp_mask_restores_rejoined_physical_members():
    manager = FaultToleranceManager(
        dp_size=2,
        strategy="continue",
        global_rank_count=4,
        ranks_per_dp=2,
    )
    manager.record_kill(2)

    manager.record_inactive_mask([True, True])

    assert manager.physical_rank_states == [
        RankState.HEALTHY,
        RankState.HEALTHY,
        RankState.HEALTHY,
        RankState.HEALTHY,
    ]
    assert manager.rank_states == [RankState.HEALTHY, RankState.HEALTHY]
    assert manager.dp_active_mask() == [True, True]


def test_kill_while_other_ranks_are_paused_does_not_leave_operation_stuck():
    manager = FaultToleranceManager(dp_size=3, strategy="pause")
    manager.record_kill(2)
    manager.finish_pause_collection(acked={0, 1}, timed_out=set())

    assert manager.record_kill(1) == []
    assert not manager.ft_operation_in_progress
    assert manager.live_ranks() == [0]


def test_fault_tolerance_rejects_multinode_even_with_mooncake():
    supported, reason = is_ft_supported_config(
        SimpleNamespace(
            pp_size=1,
            nnodes=4,
            elastic_ep_backend="mooncake",
            disaggregation_mode="null",
            device="cuda",
            tokenizer_worker_num=1,
            use_ray=False,
        )
    )

    assert not supported
    assert reason == "ft_requires_single_node"
