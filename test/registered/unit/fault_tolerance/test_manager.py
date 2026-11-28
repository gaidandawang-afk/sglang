import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sglang.srt.arg_groups import parallel_hook
from sglang.srt.fault_tolerance import manager as ft_manager
from sglang.srt.fault_tolerance.ft_state import FaultToleranceState
from sglang.srt.fault_tolerance.manager import FaultToleranceManager
from sglang.srt.fault_tolerance.protocol import parse_apply_request
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    FaultToleranceCommandReqOutput,
    FaultToleranceDPExitOutput,
    FaultToleranceRankFaultOutput,
    FaultToleranceWatchdogHeartbeatOutput,
    RouteUpdateAckOutput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"attn_dp_size": 1}, "--attn-dp-size"),
        ({"attn_dp_size": 1, "ep_join_mode": "scale"}, None),
        ({"elastic_ep_backend": None}, "--elastic-ep-backend"),
        ({"pp_size": 2}, "--pp-size"),
        ({"enable_eplb": False}, "--enable-eplb"),
        ({}, None),
    ],
)
def test_fault_tolerance_startup_requirements(monkeypatch, overrides, error):
    cfg = SimpleNamespace(
        enable_fault_tolerance=True,
        attn_dp_size=2,
        ep_join_mode=None,
        disaggregation_mode="null",
        fault_tolerance_on_error_strategy="pause",
        fault_tolerance_timeout=60,
        fault_tolerance_pause_timeout=300,
        elastic_ep_backend="mooncake",
        pp_size=1,
        enable_eplb=True,
    )
    vars(cfg).update(overrides)
    monkeypatch.setattr(parallel_hook, "resolving_view", lambda _: cfg)
    declare_resolution = Mock()
    monkeypatch.setattr(parallel_hook, "declare_resolution", declare_resolution)
    if error is not None:
        with pytest.raises(AssertionError, match=error):
            parallel_hook.handle_fault_tolerance(cfg)
        declare_resolution.assert_not_called()
    else:
        parallel_hook.handle_fault_tolerance(cfg)
        assert declare_resolution.call_args.kwargs == {
            "enable_dp_attention_local_control_broadcast": True
        }


def test_protocol_and_state_contract(monkeypatch):
    monkeypatch.setattr(
        ft_manager,
        "get_exec",
        lambda: SimpleNamespace(
            features=SimpleNamespace(fault_tolerance_on_error_strategy="pause")
        ),
    )
    request = parse_apply_request(
        b'{"instruction":"scale_down","request_id":"scale-1","params":{"removed_dp_ranks":[1]}}'
    )
    assert request.params.removed_dp_ranks == [1]
    with pytest.raises(ValueError, match="Invalid instruction"):
        parse_apply_request(b'{"instruction":"recover"}')
    state = FaultToleranceState(
        dp_size=2, strategy="pause", max_world_size=8, attn_replica_size=2
    )
    assert state.expand_dp_mask_to_global_rank_mask([True, False, False, False]) == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    manager = FaultToleranceManager(
        server_args=SimpleNamespace(
            dp_size=1,
            attn_dp_size=2,
            num_dp_ranks=2,
            max_world_size=8,
            attn_tp_size=2,
            attn_cp_size=1,
        ),
        send_to_scheduler=AsyncMock(),
    )
    assert manager.state.dp_size == 2
    assert manager.status()[1]["total_engines"] == 2
    assert manager._route_dp_mask == [True, True, False, False]
    update = manager.observe_dp_exit(FaultToleranceDPExitOutput(dp_rank=1))
    assert update.status == [True, False, False, False]
    assert manager.admission_error(0) is None
    assert "not active" in manager.admission_error(1)
    assert manager.status()[1]["engines"][1]["status"] == "dead"
    assert (
        manager.observe_active_ranks(
            ActiveRanksOutput(status=[True, True, False, False])
        )
        is None
    )
    assert (
        manager.observe_active_ranks(
            ActiveRanksOutput(
                status=[True, False, False, False], recovered_ep_ranks=[2]
            )
        )
        is None
    )
    update = manager.observe_active_ranks(
        ActiveRanksOutput(status=[True, True, False, False], recovered_ep_ranks=[3])
    )
    assert update.status == [True, True, False, False]
    assert manager.status()[1]["engines"][1]["status"] == "healthy"

    manager.add_elastic_workers(slot_offset=2, slot_count=1)
    assert manager.state.expected_dp_ranks() == [0, 1, 2]
    assert manager._route_dp_mask == [True, True, True, False]
    assert manager.status()[1]["engines"][-1] == {"id": 2, "status": "healthy"}
    manager.observe_dp_exit(FaultToleranceDPExitOutput(dp_rank=2))
    manager.add_elastic_workers(slot_offset=2, slot_count=1)
    assert manager.status()[1]["engines"][-1] == {"id": 2, "status": "dead"}
    assert manager._route_dp_mask == [True, True, False, False]

    manager._finish_submitted_apply("request-1", None)
    status = manager.status()[1]
    assert status["last_ft_request_id"] == "request-1"
    assert "ft_error" not in status
    assert "last_ft_request_id" not in status["engines"][0]
    manager._finish_submitted_apply("request-2", "failed")
    assert manager.status()[1]["ft_error"] == "failed"


@pytest.fixture
def ft_manager_with_four_dps(monkeypatch):
    monkeypatch.setattr(
        ft_manager,
        "get_exec",
        lambda: SimpleNamespace(
            features=SimpleNamespace(
                fault_tolerance_on_error_strategy="pause",
                fault_tolerance_timeout=0,
            )
        ),
    )
    return FaultToleranceManager(
        server_args=SimpleNamespace(
            num_dp_ranks=4, max_world_size=8, attn_tp_size=2, attn_cp_size=1
        ),
        send_to_scheduler=AsyncMock(),
    )


@pytest.mark.parametrize("instruction", ["retry", "scale_down"])
@pytest.mark.parametrize("request_id", [{}, {"request_id": ""}, {"request_id": 0}])
def test_apply_requires_client_request_id(instruction, request_id):
    body = {"instruction": instruction, **request_id}
    if instruction == "scale_down":
        body["params"] = {"removed_dp_ranks": [3]}
    with pytest.raises(ValueError, match="must be a non-empty string"):
        parse_apply_request(json.dumps(body).encode())


def test_async_apply_logs_busy_requests_without_changing_status(
    ft_manager_with_four_dps, caplog
):
    manager = ft_manager_with_four_dps

    async def run():
        manager.bind_event_loop(asyncio.get_running_loop())
        manager.state.observe_rank_fault(0)
        release = asyncio.Event()
        error = None

        async def recover(timeout):
            await release.wait()
            return error

        manager._apply_retry = AsyncMock(side_effect=recover)
        assert "ft_state" not in manager.status()[1]
        assert "last_ft_request_id" not in manager.status()[1]

        def submit(request_id):
            request = parse_apply_request(
                json.dumps({"instruction": "retry", "request_id": request_id}).encode()
            )
            code, accepted = manager.submit(request)
            assert code == 202
            assert accepted["request_id"] == request_id

        submit("A")
        await asyncio.sleep(0)
        running = manager.status()[1]
        assert running["last_ft_request_id"] == "A"
        assert running["ft_state"] == "recovering"

        for request_id in ("B", "C"):
            submit(request_id)
        await asyncio.sleep(0)
        manager._apply_retry.assert_awaited_once()
        assert manager.status()[1] == running
        for request_id in ("B", "C"):
            assert (
                f"FT recovery request {request_id} rejected: another recovery A is in progress"
                in caplog.messages
            )

        release.set()
        await asyncio.gather(*manager.asyncio_tasks)
        assert manager.status()[1]["last_ft_request_id"] == "A"
        assert "ft_state" not in manager.status()[1]

        release.clear()
        error = "retry_failed"
        submit("D")
        await asyncio.sleep(0)
        running = manager.status()[1]
        assert running["last_ft_request_id"] == "D"
        assert running["ft_state"] == "recovering"
        assert "ft_error" not in running
        release.set()
        await asyncio.gather(*manager.asyncio_tasks)
        body = manager.status()[1]
        assert body["last_ft_request_id"] == "D"
        assert body["ft_state"] == "failed"
        assert body["ft_error"] == "retry_failed"

    asyncio.run(run())


def test_new_scheduler_faults_survive_apply_completion(ft_manager_with_four_dps):
    manager = ft_manager_with_four_dps

    async def send(req):
        if isinstance(req, ActiveRanksOutput):
            assert manager.status()[1]["ft_state"] == "recovering"
            manager.handle_rank_fault(
                FaultToleranceRankFaultOutput(rank=2, message="new fault")
            )
            manager.handle_route_update_ack(
                RouteUpdateAckOutput(request_id=req.request_id)
            )
            return
        for rank in req.target_ranks:
            ack = FaultToleranceCommandReqOutput(
                command=req.command, rank=rank, request_id=req.request_id
            )
            manager.handle_command_output(ack)
            if rank == 0:
                manager.handle_rank_fault(
                    FaultToleranceRankFaultOutput(rank=0, message="new fault")
                )
                manager.handle_command_output(ack)

    async def run():
        manager.bind_event_loop(asyncio.get_running_loop())
        manager.state.observe_rank_fault(0)
        manager.state.observe_rank_fault(1)
        manager.send_to_scheduler.side_effect = send
        request = parse_apply_request(b'{"instruction":"retry","request_id":"retry-1"}')
        assert manager.submit(request)[0] == 202
        await asyncio.gather(*manager.asyncio_tasks)
        assert manager.state.in_flight_request_id is None
        assert manager.state.unhealthy_dp_ranks == {0, 2}
        assert manager.state.cluster_paused
        assert manager.admission_error(None) == "fault_tolerance_paused"

    asyncio.run(run())


def test_scale_down_requires_paused_survivors_and_rejects_known_exit(
    ft_manager_with_four_dps,
):
    manager = ft_manager_with_four_dps
    request = parse_apply_request(
        b'{"request_id":"remove-3","instruction":"scale_down",'
        b'"params":{"removed_dp_ranks":[3]}}'
    )
    manager.state.observe_rank_fault(0)
    manager.state.observe_rank_fault(1)
    asyncio.run(manager._run_submitted_apply(request))
    assert manager.status()[1]["ft_error"] == "scale_down_requires_remaining_dp_paused"

    manager.observe_dp_exit(FaultToleranceDPExitOutput(dp_rank=2))
    asyncio.run(manager._run_submitted_apply(request))
    assert (
        manager.status()[1]["ft_error"]
        == "scale_down_requires_remaining_processes_alive"
    )
    manager.send_to_scheduler.assert_not_awaited()
    manager.state.strategy = "continue"
    assert manager.submit(request)[0] == 400


def test_scale_down_waits_for_mask_acks_before_execution(ft_manager_with_four_dps):
    manager = ft_manager_with_four_dps
    first = parse_apply_request(
        b'{"request_id":"remove-3","instruction":"scale_down",'
        b'"params":{"removed_dp_ranks":[3]}}'
    )
    second = parse_apply_request(
        b'{"request_id":"remove-2-and-3","instruction":"scale_down",'
        b'"params":{"removed_dp_ranks":[2,3]}}'
    )
    for rank in (0, 1, 2):
        manager.state.observe_rank_fault(rank)
    sent = []

    def ack(command, rank, *, phase=None, request_id=None):
        output = FaultToleranceCommandReqOutput(
            command=phase or command.command,
            rank=rank,
            request_id=request_id or command.request_id,
        )
        manager.handle_command_output(msgpack_decode(msgpack_encode(output)))

    async def send(req):
        sent.append(req)
        if isinstance(req, ActiveRanksOutput):
            manager.handle_route_update_ack(
                RouteUpdateAckOutput(request_id=req.request_id)
            )
            return
        if len(sent) > 1:
            pending = manager._pending_commands[req.request_id]
            ack(req, 0, request_id=sent[0].request_id)
            wrong_phase = (
                "scale_down"
                if req.command == "scale_down_prepare"
                else "scale_down_prepare"
            )
            ack(req, 1, phase=wrong_phase)
            assert pending.remaining_ranks == {0, 1}
        for rank in (0, 1, 0):
            ack(req, rank)

    async def run():
        manager.bind_event_loop(asyncio.get_running_loop())
        manager.send_to_scheduler.side_effect = send
        await manager._run_submitted_apply(first)
        assert len(sent) == 1
        assert sent[0].command == "scale_down_prepare"
        assert sent[0].target_ranks == [0, 1, 2]
        assert sent[0].active_mask == [True] * 6 + [False] * 2
        assert manager.status()[1]["ft_error"] == "scale_down_prepare_timeout"
        assert manager.state.cluster_paused
        assert manager.state.expected_dp_ranks() == [0, 1, 2, 3]

        await manager._run_submitted_apply(second)
        _, prepare, execute, route = sent
        assert prepare.command == "scale_down_prepare"
        assert prepare.target_ranks == execute.target_ranks == [0, 1]
        assert prepare.active_mask == [True] * 4 + [False] * 4
        assert execute.command == "scale_down" and execute.active_mask is None
        assert prepare.request_id == execute.request_id != sent[0].request_id
        assert route.status == [True, True, False, False]
        assert manager.state.expected_dp_ranks() == [0, 1]
        assert not manager.state.cluster_paused
        assert manager.status()[1]["last_ft_request_id"] == second.request_id
        assert "ft_error" not in manager.status()[1]
        manager.handle_rank_fault(
            FaultToleranceRankFaultOutput(rank=2, message="late fault")
        )
        assert not manager.state.cluster_paused

    asyncio.run(run())


def test_watchdog_heartbeat_only_changes_display(ft_manager_with_four_dps, monkeypatch):
    manager = ft_manager_with_four_dps
    now = 0.0
    monkeypatch.setattr(ft_manager.time, "monotonic", lambda: now)
    first = FaultToleranceWatchdogHeartbeatOutput(first_global_rank=0, dp_ranks=[0, 1])
    second = FaultToleranceWatchdogHeartbeatOutput(first_global_rank=3, dp_ranks=[1, 2])
    dispatch = manager.init_request_dispatcher()
    for event in (first, second):
        dispatch(msgpack_decode(msgpack_encode(event)))
    now = ft_manager.WATCHDOG_HEARTBEAT_TIMEOUT
    dispatch(first)
    assert [dp["status"] for dp in manager.status()[1]["engines"]] == [
        "healthy",
        "unreachable",
        "unreachable",
        "healthy",
    ]
    assert manager.admission_error(1) is None
    assert not manager.state.has_unresolved_expected_dp_fault()
    assert manager._route_dp_mask == [True] * 4
    manager.send_to_scheduler.assert_not_awaited()

    manager.state.observe_rank_fault(1)
    manager.observe_dp_exit(FaultToleranceDPExitOutput(dp_rank=2))
    assert [dp["status"] for dp in manager.status()[1]["engines"]] == [
        "healthy",
        "unhealthy",
        "dead",
        "healthy",
    ]
    dispatch(second)
    assert manager.status()[1]["engines"][2]["status"] == "dead"
    assert manager.state.unreachable_dp_mask == [False] * 4
    assert manager.state.cluster_paused
    assert manager._route_dp_mask == [True, True, False, True]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
