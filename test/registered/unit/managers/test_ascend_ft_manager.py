import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.fault_tolerance.protocol import (
    PauseRequest,
    RetryRequest,
    ScaleDownRequest,
    parse_apply_request,
)
from sglang.srt.managers import ascend_ft
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    AscendFTAckOutput,
    AscendFTCommandReq,
    FaultToleranceDPCShutdownReqInput,
    ProcessActiveRanksOutput,
    RouteUpdateAckOutput,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Tokenizer:
    def __init__(self):
        self.is_pause = False
        self.is_pause_cond = asyncio.Condition()
        self.events = []
        self.route = asyncio.Queue()
        self.fail_recovery = False

    def auto_create_handle_loop(self):
        pass

    async def _async_dispatch_to_scheduler(self, message):
        assert self.is_pause
        if isinstance(message, ActiveRanksOutput):
            self.events.append(("route", message.status))
            self.route.put_nowait(message)
        else:
            assert isinstance(message, AscendFTCommandReq)
            self.events.append((message.op, message.target_ranks))
            for rank in message.target_ranks:
                self.manager.handle_ack(
                    AscendFTAckOutput(
                        rid=message.rid,
                        op=message.op,
                        rank=rank,
                        error=(
                            "migration failed"
                            if self.fail_recovery and message.op == "recover"
                            else None
                        ),
                    )
                )

    async def ascend_ft_replay(self, removed_dp_ranks):
        assert self.is_pause
        self.events.append(("replay", removed_dp_ranks))


def make_tokenizer(monkeypatch):
    monkeypatch.setattr(
        ascend_ft, "get_parallel", lambda: SimpleNamespace(num_dp_ranks=4)
    )
    monkeypatch.setattr(
        ascend_ft,
        "get_exec",
        lambda: SimpleNamespace(features=SimpleNamespace(fault_tolerance_timeout=1)),
    )
    tokenizer = Tokenizer()
    tokenizer.manager = ascend_ft.AscendFTManager(tokenizer)
    tokenizer.manager.min_survivors = 2
    return tokenizer


def test_scale_down_finishes_only_after_shutdown_route_ack_and_resume(monkeypatch):
    async def run():
        tokenizer = make_tokenizer(monkeypatch)
        manager = tokenizer.manager

        accepted = manager.submit(PauseRequest(instruction="pause", request_id="hold"))
        assert accepted == (
            202,
            {"message": ascend_ft.FT_REQUEST_ACCEPTED_MESSAGE, "request_id": "hold"},
        )
        await manager.task
        assert manager.status()["last_ft_request_id"] == "hold"
        assert manager.admission_error(None) == "fault_tolerance_paused"

        manager._watchdog_leases[0] = (0, (0, 1, 2, 3))
        manager._watchdog_sockets[0] = object()

        async def shutdown(socket, request):
            assert isinstance(request, FaultToleranceDPCShutdownReqInput)
            tokenizer.events.append(("shutdown", request.target_dp_ranks))
            manager.observe_process_active_ranks(
                ProcessActiveRanksOutput(
                    ranks=request.target_dp_ranks,
                    active=False,
                )
            )

        monkeypatch.setattr(ascend_ft, "async_sock_send", shutdown)
        request = ScaleDownRequest(
            instruction="scale_down",
            request_id="retire-0",
            params={"removed_dp_ranks": [0]},
        )
        assert manager.submit(request)[1]["request_id"] == "retire-0"
        assert manager.submit(request)[0] == 409
        route = await asyncio.wait_for(tokenizer.route.get(), 1)
        manager.handle_route_update_ack(RouteUpdateAckOutput(request_id="stale"))
        await asyncio.sleep(0)
        assert manager.status()["last_ft_request_id"] == "hold"
        assert manager.active_ranks == (0, 1, 2, 3)
        assert tokenizer.is_pause
        assert manager.admission_error(0) == "routed_dp_rank=0 is not active"
        manager.handle_route_update_ack(
            RouteUpdateAckOutput(request_id=route.request_id)
        )
        await manager.task
        assert tokenizer.events == [
            ("pause", [0, 1, 2, 3]),
            ("shutdown", [0]),
            ("pause", [1, 2, 3]),
            ("recover", [1, 2, 3]),
            ("route", [False, True, True, True]),
            ("replay", [0]),
            ("resume", [1, 2, 3]),
        ]
        assert manager.status() == {
            "schema_version": 1,
            "total_engines": 4,
            "engines": [
                {"id": 0, "status": "dead"},
                {"id": 1, "status": "healthy"},
                {"id": 2, "status": "healthy"},
                {"id": 3, "status": "healthy"},
            ],
            "last_ft_request_id": "retire-0",
        }
        assert not tokenizer.is_pause

    asyncio.run(run())


def test_retry_semantics_and_failed_recovery_fail_stop(monkeypatch):
    async def run():
        tokenizer = make_tokenizer(monkeypatch)
        manager = tokenizer.manager
        failstop = Mock()
        monkeypatch.setattr(manager, "_failstop", failstop)
        assert (
            manager.submit(RetryRequest(instruction="retry", request_id="idle"))[0]
            == 202
        )
        await manager.task
        assert (
            manager.status()["ft_error"]
            == "retry_requires_unresolved_expected_dp_fault"
        )
        assert not tokenizer.is_pause

        manager.handle_ack(AscendFTAckOutput(op="fault", rank=1, error="MC2 timeout"))
        assert manager.status()["engines"][1]["status"] == "unhealthy"
        tokenizer.fail_recovery = True
        assert (
            manager.submit(RetryRequest(instruction="retry", request_id="failure"))[0]
            == 202
        )
        await manager.task
        failstop.assert_called_once()
        assert tokenizer.is_pause
        assert manager.status()["last_ft_request_id"] == "idle"
        assert tokenizer.events == [
            ("pause", [0, 1, 2, 3]),
            ("recover", [0, 1, 2, 3]),
        ]

    asyncio.run(run())


def test_lease_expiry_blocks_admission_and_retry(monkeypatch):
    async def run():
        tokenizer = make_tokenizer(monkeypatch)
        manager = tokenizer.manager
        manager._watchdog_leases[0] = (0, (1, 3))
        manager._sweep_expired_watchdog_leases(61)
        assert manager.admission_error(2) == "fault_tolerance_paused"
        assert [e["status"] for e in manager.status()["engines"]] == [
            "healthy",
            "dead",
            "healthy",
            "dead",
        ]
        manager.submit(RetryRequest(instruction="retry", request_id="still-down"))
        await manager.task
        assert manager.status()["last_ft_request_id"] == "still-down"
        assert (
            manager.status()["ft_error"]
            == "retry_requires_all_expected_processes_alive"
        )
        assert tokenizer.is_pause

    asyncio.run(run())


@pytest.mark.parametrize(
    "payload",
    [
        b'{"instruction":"retry","request_id":7}',
        b'{"instruction":"recover"}',
        b'{"instruction":"scale_down","params":{"removed_dp_ranks":[true]}}',
        b'{"instruction":"scale_down","params":{"removed_dp_ranks":[-1]}}',
        b'{"instruction":"scale_down","params":{}}',
    ],
)
def test_protocol_rejects_invalid_v1_envelopes(payload):
    with pytest.raises(ValueError):
        parse_apply_request(payload)
