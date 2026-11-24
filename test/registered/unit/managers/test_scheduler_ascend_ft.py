import contextlib
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from sglang.srt.managers import scheduler_ascend_ft_mixin as ft
from sglang.srt.managers.io_struct import AscendFTCommandReq
from sglang.srt.managers.scheduler_components.pause_state import PauseState
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture
def scheduler(monkeypatch):
    scheduler = ft.SchedulerAscendFTMixin()
    scheduler.init_ascend_ft(None)
    scheduler.pause_state = PauseState()
    scheduler.cur_batch_for_debug = object()
    scheduler.send_control_output = Mock()
    monkeypatch.setattr(ft, "get_parallel", lambda: SimpleNamespace(dp_rank=1))
    monkeypatch.setattr(
        ft,
        "get_exec",
        lambda: SimpleNamespace(
            features=SimpleNamespace(fault_tolerance_pause_timeout=30)
        ),
    )
    return scheduler


def test_python_fault_reports_rank_and_fail_stops_an_unattended_pause(
    scheduler, monkeypatch
):
    clock = Mock(return_value=10.0)
    fail_stop = Mock()
    monkeypatch.setattr(ft, "time", SimpleNamespace(monotonic=clock))
    monkeypatch.setattr(ft, "notify_node_main_process_failure", fail_stop)

    dispatch = Mock(side_effect=[ValueError("scheduler fault"), None])
    scheduler.run_ascend_ft_loop(dispatch)
    ack = scheduler.send_control_output.call_args.args[0]
    assert (ack.op, ack.rank, ack.error) == ("fault", 1, "scheduler fault")
    assert scheduler.pause_state.paused
    assert scheduler.cur_batch_for_debug is None
    clock.return_value = 39.0
    scheduler.check_ascend_ft_pause_deadline()
    fail_stop.assert_not_called()
    clock.return_value = 40.0
    scheduler.check_ascend_ft_pause_deadline()
    scheduler.check_ascend_ft_pause_deadline()
    fail_stop.assert_called_once_with()


def test_planned_pause_has_no_fault_deadline(scheduler):
    scheduler.ascend_ft_communication = SimpleNamespace(pause_requested=False)
    req = AscendFTCommandReq(rid="planned", op="pause", target_ranks=[1, 2])
    assert scheduler.handle_ascend_ft_command(req) is None
    assert scheduler.ascend_ft_communication.pause_requested
    scheduler.run_ascend_ft_loop(Mock(side_effect=[ft.AscendFTPause(), None]))
    ack = scheduler.send_control_output.call_args.args[0]
    assert (ack.rid, ack.op, ack.rank, ack.error) == ("planned", "pause", 1, None)
    assert scheduler.pause_state.paused
    assert scheduler.ascend_ft_pause_deadline is None


@pytest.mark.parametrize(
    ("membership_changed", "reset_collective"),
    [(False, False), (False, True), (True, False)],
)
def test_recovery_completes_device_and_stream_steps_before_ack(
    scheduler, membership_changed, reset_collective
):
    trace = Mock()
    group = object()
    rebuild_group = object()
    scheduler.pause_state.pause()
    scheduler.ascend_ft_pause_deadline = 100.0
    scheduler.ascend_ft_communication = trace.comm
    trace.comm.group = group
    trace.comm.rebuild.return_value = rebuild_group
    scheduler.tp_worker = SimpleNamespace(model_runner=trace.runner)
    scheduler._discard_ascend_ft_attempts = trace.discard
    scheduler.schedule_stream = trace.schedule_stream
    scheduler.forward_stream = trace.forward_stream
    scheduler.forward_stream_ctx = contextlib.nullcontext()

    req = AscendFTCommandReq(
        rid="operation",
        op="recover",
        target_ranks=[1, 2],
        membership_changed=membership_changed,
        reset_collective=reset_collective,
    )
    ack = scheduler.handle_ascend_ft_command(req)
    expected = [call.runner.recover_ascend_ft_device()]
    if membership_changed or reset_collective:
        expected.append(call.comm.rebuild((1, 2), "operation"))
    if membership_changed:
        expected.append(
            call.runner.eplb_manager.rebalance_for_ascend_ft((1, 2), rebuild_group)
        )
    expected.append(call.discard())
    if membership_changed:
        expected.extend(
            [
                call.runner.install_ascend_ft_topology((1, 2)),
                call.schedule_stream.synchronize(),
                call.forward_stream.wait_stream(trace.schedule_stream),
                call.runner.run_ascend_ft_health_gate((1, 2), rebuild_group),
            ]
        )
    assert trace.mock_calls == expected
    assert ack.error is None
    assert scheduler.pause_state.paused
    assert scheduler.ascend_ft_pause_deadline is None


def test_recovery_health_gate_failure_is_reported_without_resume(scheduler):
    scheduler.pause_state.pause()
    scheduler.ascend_ft_communication = Mock()
    scheduler.tp_worker = SimpleNamespace(model_runner=Mock())
    scheduler.tp_worker.model_runner.run_ascend_ft_health_gate.side_effect = (
        RuntimeError("MC2 unhealthy")
    )
    scheduler._discard_ascend_ft_attempts = Mock()
    scheduler.schedule_stream = Mock()
    scheduler.forward_stream = Mock()
    scheduler.forward_stream_ctx = contextlib.nullcontext()
    ack = scheduler.handle_ascend_ft_command(
        AscendFTCommandReq(
            rid="shrink", op="recover", target_ranks=[1], membership_changed=True
        )
    )
    assert (ack.op, ack.rank, ack.error) == ("recover", 1, "MC2 unhealthy")
    assert scheduler.pause_state.paused
