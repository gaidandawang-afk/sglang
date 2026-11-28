import sys
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest

from sglang.srt.managers import scheduler_ft_mixin as ft
from sglang.srt.managers.scheduler_components.pause_state import PauseState
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture
def scheduler(monkeypatch):
    parallel = SimpleNamespace(
        dp_rank=0,
        launch_world_rank=0,
        attn_tp_size=1,
        attn_cp_size=1,
        attn_tp_rank=0,
        attn_cp_rank=0,
    )
    features = SimpleNamespace(
        enable_fault_tolerance=True,
        fault_tolerance_on_error_strategy="pause",
        fault_tolerance_pause_timeout=30,
    )
    monkeypatch.setattr(ft, "get_parallel", lambda: parallel)
    monkeypatch.setattr(ft, "get_exec", lambda: SimpleNamespace(features=features))
    monkeypatch.setattr(
        ft, "is_fault_tolerance_enabled", lambda: features.enable_fault_tolerance
    )
    monkeypatch.setattr(ft.time, "monotonic", lambda: 100.0)
    scheduler = ft.SchedulerFaultToleranceMixin()
    scheduler.init_fault_tolerance()
    scheduler.pause_state = PauseState()
    scheduler.ipc_channels = SimpleNamespace(send_to_tokenizer=Mock())
    scheduler.tp_worker = SimpleNamespace(model_runner=Mock())
    scheduler.tp_worker.model_runner.finish_fault_tolerance_scale_down.return_value = (
        None
    )
    scheduler.cur_batch_for_debug = None
    scheduler.last_batch = None
    scheduler.running_batch = SimpleNamespace(reqs=[])
    scheduler.chunked_req = None
    scheduler.tree_cache = object()
    return scheduler


def test_normal_exit_does_not_restart(scheduler):
    dispatch = Mock()
    scheduler._run_event_loop_fault_tolerance(dispatch)
    dispatch.assert_called_once_with(scheduler)
    scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()


@pytest.mark.parametrize("strategy", ["pause", "continue"])
def test_fault_reenters_loop_with_expected_state(scheduler, monkeypatch, strategy):
    ft.get_exec().features.fault_tolerance_on_error_strategy = strategy
    queue = deque([(SimpleNamespace(reqs=[]), None)])
    scheduler.result_queue = queue
    calls = []

    def dispatch(current):
        calls.append(current)
        if len(calls) == 1:
            raise RuntimeError("forward failed")
        assert current.pause_state.paused is (strategy == "pause")
        assert current._ft_pause_deadline == (130.0 if strategy == "pause" else None)
        if strategy == "pause":
            assert current._ft_result_queue is queue
        else:
            assert current._ft_result_queue is None
            assert not queue

    scheduler._run_event_loop_fault_tolerance(dispatch)
    assert calls == [scheduler, scheduler]
    report = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
    assert (report.rank, report.message) == (0, "forward failed")


def test_fault_handler_failure_propagates(scheduler, monkeypatch):
    monkeypatch.setattr(
        scheduler,
        "_ft_abort_inflight_window",
        Mock(side_effect=ValueError("abort failed")),
    )
    dispatch = Mock(side_effect=RuntimeError("forward failed"))
    with pytest.raises(ValueError, match="abort failed"):
        scheduler._run_event_loop_fault_tolerance(dispatch)
    dispatch.assert_called_once()


@pytest.mark.parametrize("error", [SystemExit, KeyboardInterrupt])
def test_shutdown_exceptions_propagate(scheduler, error):
    with pytest.raises(error):
        scheduler._run_event_loop_fault_tolerance(Mock(side_effect=error))
    scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()


def test_inflight_requests_are_aborted_and_released_once(scheduler, monkeypatch):
    def request(rid, finished=False, owns_kv=False):
        return SimpleNamespace(
            rid=rid,
            finished=lambda: finished,
            kv=SimpleNamespace(holds_kv=owns_kv, holds_mamba=False),
        )

    active = request("active")
    retained = request("retained", finished=True, owns_kv=True)
    completed = request("completed", finished=True)
    chunked = request("chunked")
    batch = SimpleNamespace(reqs=[active, retained, completed])
    scheduler.cur_batch_for_debug = scheduler.last_batch = scheduler.running_batch = (
        batch
    )
    scheduler._ft_result_queue = deque([(batch, None)])
    scheduler.chunked_req = chunked
    release = create_autospec(ft.release_kv_cache)
    monkeypatch.setattr(ft, "release_kv_cache", release)

    scheduler._ft_abort_inflight_window()
    sent = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args_list
    assert [call.args[0].rid for call in sent] == ["active", "retained", "chunked"]
    scheduler._ft_discard_inflight_window()
    assert [call.args[0].rid for call in release.call_args_list] == [
        "active",
        "retained",
        "chunked",
    ]
    assert all(call.kwargs == {"checkpoint": False} for call in release.call_args_list)
    assert scheduler.running_batch.reqs == []
    assert scheduler.cur_batch_for_debug is scheduler.last_batch is None
    assert scheduler.chunked_req is scheduler._ft_result_queue is None


@pytest.mark.parametrize(
    "command,mask", [("retry", None), ("scale_down", [True, False])]
)
@pytest.mark.parametrize("attn_tp_rank", [0, 1])
def test_recovery_command_resumes_and_only_leader_acks(
    scheduler, command, mask, attn_tp_rank
):
    ft.get_parallel().attn_tp_rank = attn_tp_rank
    scheduler.pause_state.pause()
    scheduler._ft_pause_deadline = 130.0
    runner = scheduler.tp_worker.model_runner
    if command == "scale_down":
        prepare = ft.FaultToleranceCommandReqInput(
            request_id="recover-1",
            command="scale_down_prepare",
            target_ranks=[0],
            active_mask=mask,
        )
        ready = scheduler.handle_fault_tolerance_command(prepare)
        runner.finish_fault_tolerance_scale_down.assert_not_called()
        assert scheduler.pause_state.paused
        assert scheduler._ft_pause_deadline == 130.0
        if attn_tp_rank == 0:
            assert (ready.request_id, ready.command) == (
                "recover-1",
                "scale_down_prepare",
            )
        else:
            assert ready is None
    req = ft.FaultToleranceCommandReqInput(
        request_id="recover-1", command=command, target_ranks=[0]
    )
    result = scheduler.handle_fault_tolerance_command(req)
    runner.update_fault_tolerance_active_ranks.assert_called_once_with(mask)
    if command == "scale_down":
        runner.finish_fault_tolerance_scale_down.assert_called_once_with()
    assert not scheduler.pause_state.paused
    assert scheduler._ft_pause_deadline is None
    if attn_tp_rank == 0:
        assert (result.request_id, result.command, result.rank) == (
            "recover-1",
            command,
            0,
        )
    else:
        assert result is None


def test_command_for_another_rank_does_nothing(scheduler):
    req = ft.FaultToleranceCommandReqInput(
        request_id="recover-1", command="retry", target_ranks=[1]
    )
    assert scheduler.handle_fault_tolerance_command(req) is None
    scheduler.tp_worker.model_runner.update_fault_tolerance_active_ranks.assert_not_called()


def test_joiner_acks_its_global_dp_slot(scheduler):
    parallel = ft.get_parallel()
    parallel.launch_world_rank = 4
    parallel.attn_tp_size = 2
    req = ft.FaultToleranceCommandReqInput(
        request_id="retry-joiner", command="retry", target_ranks=[2]
    )
    result = scheduler.handle_fault_tolerance_command(req)
    assert (result.request_id, result.rank) == ("retry-joiner", 2)


def test_scale_down_reports_shorter_world_before_ack(scheduler):
    scheduler.pause_state.pause()
    scheduler.tp_worker.model_runner.finish_fault_tolerance_scale_down.return_value = 2
    prepare = ft.FaultToleranceCommandReqInput(
        request_id="scale-down",
        command="scale_down_prepare",
        target_ranks=[0, 1],
        active_mask=[True, True, False],
    )
    scheduler.handle_fault_tolerance_command(prepare)
    req = ft.FaultToleranceCommandReqInput(
        request_id="scale-down", command="scale_down", target_ranks=[0, 1]
    )
    result = scheduler.handle_fault_tolerance_command(req)
    report = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
    assert (report.success, report.effective_ep_size, report.slot_count) == (True, 2, 0)
    assert (result.request_id, result.rank) == ("scale-down", 0)


def test_scale_down_replaces_mask_without_running_eplb_for_previous_round(scheduler):
    runner = scheduler.tp_worker.model_runner
    first = ft.FaultToleranceCommandReqInput(
        request_id="first",
        command="scale_down_prepare",
        target_ranks=[0],
        active_mask=[True, True, False],
    )
    assert scheduler.handle_fault_tolerance_command(first) is None
    runner.update_fault_tolerance_active_ranks.assert_not_called()

    scheduler.pause_state.pause()
    scheduler.handle_fault_tolerance_command(first)
    second = ft.FaultToleranceCommandReqInput(
        request_id="second",
        command="scale_down_prepare",
        target_ranks=[0],
        active_mask=[True, False, False],
    )
    scheduler.handle_fault_tolerance_command(second)
    assert [
        call.args[0]
        for call in runner.update_fault_tolerance_active_ranks.call_args_list
    ] == [
        first.active_mask,
        second.active_mask,
    ]

    execute_first = ft.FaultToleranceCommandReqInput(
        request_id="first", command="scale_down", target_ranks=[0]
    )
    assert scheduler.handle_fault_tolerance_command(execute_first) is None
    assert scheduler.pause_state.paused
    runner.finish_fault_tolerance_scale_down.assert_not_called()
    execute_second = ft.FaultToleranceCommandReqInput(
        request_id="second", command="scale_down", target_ranks=[0]
    )
    result = scheduler.handle_fault_tolerance_command(execute_second)
    assert (result.request_id, result.command) == ("second", "scale_down")
    runner.finish_fault_tolerance_scale_down.assert_called_once_with()
    assert not scheduler.pause_state.paused


def test_pause_timeout_notifies_once(scheduler, monkeypatch):
    notify = Mock()
    monkeypatch.setattr(ft, "notify_node_main_process_failure", notify)
    scheduler._ft_pause_deadline = 101.0
    scheduler._check_ft_pause_deadline()
    notify.assert_not_called()
    scheduler._ft_pause_deadline = 100.0
    scheduler._check_ft_pause_deadline()
    scheduler._check_ft_pause_deadline()
    notify.assert_called_once_with()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
