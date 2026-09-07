import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from sglang.srt.fault_tolerance.ft_state import FaultToleranceState
from sglang.srt.fault_tolerance.manager import FaultToleranceManager
from sglang.srt.fault_tolerance.protocol import parse_apply_request
from sglang.srt.managers.io_struct import ActiveRanksOutput, ProcessActiveRanksOutput


def make_manager(*, dp_size=2, ranks_per_dp=1, strategy="pause"):
    return FaultToleranceManager(
        server_args=SimpleNamespace(
            dp_size=dp_size,
            tp_size=dp_size * ranks_per_dp,
            fault_tolerance_on_error_strategy=strategy,
            fault_tolerance_timeout=1,
        ),
        zmq_context=Mock(),
        send_to_scheduler=AsyncMock(),
    )


class TestFaultTolerance(unittest.IsolatedAsyncioTestCase):
    def test_protocol_and_state_contract(self):
        request = parse_apply_request(
            b'{"instruction":"scale_down","params":{"removed_dp_ranks":[1]}}'
        )
        self.assertEqual(request.params.removed_dp_ranks, [1])
        with self.assertRaisesRegex(ValueError, "Invalid instruction"):
            parse_apply_request(b'{"instruction":"recover"}')

        state = FaultToleranceState(dp_size=2, strategy="pause", global_rank_count=4)
        state.observe_process_active_ranks([2], active=False)
        self.assertEqual(state.process_alive_dp_mask(), [True, False])
        self.assertEqual(state.status_response()["engines"][1]["status"], "dead")
        self.assertEqual(
            state.expand_dp_mask_to_global_rank_mask([True, False]),
            [True, True, False, False],
        )

    async def test_retry_uses_expected_topology(self):
        manager = make_manager(dp_size=4)
        manager.state.expected_dp_mask = [True, True, False, True]
        manager._send_command_collect = AsyncMock()
        manager._publish_route_dp_mask = AsyncMock()

        self.assertIsNone(await manager._apply_retry(1))
        manager._send_command_collect.assert_awaited_once_with(
            command="retry", target_ranks=[0, 1, 3], timeout_sec=1
        )
        manager._publish_route_dp_mask.assert_awaited_once_with(
            [True, True, False, True], 1
        )

    async def test_scale_down_orders_shutdown_command_and_route(self):
        manager = make_manager(dp_size=4, ranks_per_dp=2)
        events = []
        manager._shutdown_dp_processes = AsyncMock(
            side_effect=lambda *_: events.append("shutdown")
        )
        manager._send_command_collect = AsyncMock(
            side_effect=lambda **_: events.append("command")
        )
        manager._publish_route_dp_mask = AsyncMock(
            side_effect=lambda *_: events.append("route")
        )

        self.assertIsNone(await manager._apply_scale_down([2], 1))
        self.assertEqual(events, ["shutdown", "command", "route"])
        manager._send_command_collect.assert_awaited_once_with(
            command="scale_down",
            target_ranks=[0, 1, 3],
            timeout_sec=1,
            active_global_rank_mask=[
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
            ],
        )
        self.assertEqual(manager.state.expected_dp_mask, [True, True, False, True])

    async def test_continue_routes_only_ready_processes(self):
        manager = make_manager(dp_size=2, ranks_per_dp=2, strategy="continue")

        down = manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[2, 3], active=False)
        )
        self.assertEqual(down.status, [True, False])
        self.assertEqual(manager.state.expected_dp_mask, [True, True])

        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[2, 3], active=True)
        )
        up = manager.observe_active_ranks(ActiveRanksOutput(status=[True, True]))
        self.assertEqual(up.status, [True, True])
        manager.send_to_scheduler.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
