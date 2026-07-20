import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from sglang.srt.fault_tolerance.controller import RankState
from sglang.srt.fault_tolerance.manager import FaultToleranceManager
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    FaultToleranceCommandReqOutput,
    FaultToleranceRankFaultOutput,
    ProcessActiveRanksOutput,
    RecoveredDPRanksOutput,
)


def make_manager(*, dp_size=2, strategy="pause"):
    server_args = SimpleNamespace(
        dp_size=dp_size,
        fault_tolerance_on_error_strategy=strategy,
        fault_tolerance_timeout=1,
    )
    manager = FaultToleranceManager(
        server_args=server_args,
        send_to_scheduler=SimpleNamespace(send_pyobj=AsyncMock()),
    )
    manager.bind_event_loop(asyncio.get_running_loop())
    return manager


class TestFaultToleranceManager(unittest.IsolatedAsyncioTestCase):
    async def test_process_and_native_masks_publish_only_effective_changes(self):
        manager = make_manager(strategy="continue")

        process_down = manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=False)
        )
        self.assertEqual(process_down.status, [True, False])
        self.assertIsNone(
            manager.observe_active_ranks(ActiveRanksOutput(status=[True, False]))
        )
        self.assertIsNone(
            manager.observe_process_active_ranks(
                ProcessActiveRanksOutput(ranks=[1], active=True)
            )
        )

        native_up = manager.observe_active_ranks(ActiveRanksOutput(status=[True, True]))
        self.assertEqual(native_up.status, [True, True])

    async def test_process_and_native_duplicate_fault_pause_once(self):
        manager = make_manager()
        manager._pause_schedulers = AsyncMock()

        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=False)
        )
        manager.observe_active_ranks(ActiveRanksOutput(status=[True, False]))
        await asyncio.sleep(0)

        manager._pause_schedulers.assert_awaited_once_with([0])

    async def test_process_down_shrinks_inflight_pause_targets(self):
        manager = make_manager(dp_size=4)

        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=False)
        )
        await asyncio.sleep(0)

        self.assertEqual(len(manager._pending_commands), 1)
        request_id, pending = next(iter(manager._pending_commands.items()))
        self.assertEqual(pending.command, "pause")
        self.assertEqual(pending.target_ranks, {0, 2, 3})

        for rank in (0, 3):
            manager.handle_command_output(
                FaultToleranceCommandReqOutput(
                    request_id=request_id,
                    rank=rank,
                    success=True,
                    message="paused",
                )
            )

        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[2], active=False)
        )
        await asyncio.gather(*list(manager.asyncio_tasks))

        self.assertEqual(manager.state.paused_dp_ranks, {0, 3})
        self.assertEqual(
            manager.state.rank_states,
            [RankState.PAUSED, RankState.DEAD, RankState.DEAD, RankState.PAUSED],
        )
        self.assertEqual(manager._pending_commands, {})

    async def test_rejoin_requires_process_native_and_recovery_sources(self):
        manager = make_manager(strategy="continue")
        manager.state.disabled_dp_ranks.add(1)
        manager.observe_active_ranks(ActiveRanksOutput(status=[True, False]))
        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=False)
        )

        self.assertIsNone(
            manager.observe_process_active_ranks(
                ProcessActiveRanksOutput(ranks=[1], active=True)
            )
        )
        self.assertEqual(manager.state.disabled_dp_ranks, {1})

        self.assertIsNone(
            manager.observe_active_ranks(ActiveRanksOutput(status=[True, True]))
        )
        self.assertEqual(manager.state.disabled_dp_ranks, {1})

        rejoined = manager.observe_recovered_dp_ranks(
            RecoveredDPRanksOutput(ranks=[1])
        )
        self.assertEqual(rejoined.status, [True, True])
        self.assertEqual(manager.state.disabled_dp_ranks, set())

    async def test_exception_does_not_change_availability_sources(self):
        manager = make_manager()
        manager._handle_exception_pause = AsyncMock()
        before = (
            list(manager.state.process_active_ranks),
            list(manager.state.mooncake_active_ranks),
            set(manager.state.disabled_dp_ranks),
        )

        event = FaultToleranceRankFaultOutput(
            rank=1,
            fault_type="exception",
            message="boom",
        )
        manager.handle_rank_fault(event)
        await asyncio.sleep(0)

        manager._handle_exception_pause.assert_awaited_once_with(event)
        self.assertEqual(
            before,
            (
                manager.state.process_active_ranks,
                manager.state.mooncake_active_ranks,
                manager.state.disabled_dp_ranks,
            ),
        )

    async def test_retry_resumes_only_runtime_paused_ranks(self):
        manager = make_manager()
        manager.state.begin_exception_pause()
        manager.state.finish_pause({0, 1})
        manager.state.process_active_ranks[1] = False
        manager.state.refresh_rank_states()
        manager._publish_active_ranks = AsyncMock()
        manager._send_command_collect = AsyncMock(return_value=({0}, set()))

        status, response = await manager.apply({"fault_tolerance_instruction": "retry"})

        self.assertEqual(status, 200)
        self.assertEqual(response["resumed_ranks"], [0])
        self.assertEqual(
            manager.state.rank_states,
            [RankState.HEALTHY, RankState.DEAD],
        )
        self.assertEqual(
            manager._send_command_collect.await_args.kwargs["target_ranks"],
            [0],
        )

    async def test_scale_down_is_logical_isolation(self):
        manager = make_manager()
        manager.state.begin_exception_pause()
        manager.state.finish_pause({0, 1})
        manager._publish_active_ranks = AsyncMock()
        manager._send_command_collect = AsyncMock(return_value=({0, 1}, set()))

        status, response = await manager.apply(
            {
                "fault_tolerance_instruction": "scale_down",
                "fault_tolerance_params": {"ranks": [1]},
            }
        )

        self.assertEqual(status, 200)
        self.assertEqual(response["resumed_ranks"], [0, 1])
        self.assertEqual(manager.state.paused_dp_ranks, set())
        manager._send_command_collect.assert_awaited_once_with(
            command="resume",
            target_ranks=[0, 1],
            timeout_sec=1,
        )
        self.assertEqual(manager.state.disabled_dp_ranks, {1})
        self.assertEqual(manager.state.process_active_ranks, [True, True])
        self.assertEqual(
            manager.state.rank_states,
            [RankState.HEALTHY, RankState.DEAD],
        )

    async def test_resume_timeout_enters_failstop_without_partial_commit(self):
        manager = make_manager()
        manager.state.begin_exception_pause()
        manager.state.finish_pause({0, 1})
        manager._publish_active_ranks = AsyncMock()
        manager._send_command_collect = AsyncMock(
            side_effect=TimeoutError("resume timed out")
        )
        manager._failstop = Mock(side_effect=RuntimeError("failstop"))

        with self.assertRaisesRegex(RuntimeError, "failstop"):
            await manager.apply({"fault_tolerance_instruction": "retry"})

        manager._failstop.assert_called_once()
        self.assertTrue(manager.state.ft_operation_in_progress)
        self.assertEqual(manager.state.paused_dp_ranks, {0, 1})

    async def test_scale_down_rejects_removed_shutdown_parameter(self):
        manager = make_manager()
        manager.state.begin_exception_pause()
        manager.state.finish_pause({0, 1})

        status, response = await manager.apply(
            {
                "fault_tolerance_instruction": "scale_down",
                "fault_tolerance_params": {"ranks": [1], "shutdown": True},
            }
        )

        self.assertEqual(status, 400)
        self.assertEqual(response["message"], "scale_down_does_not_accept_shutdown")


if __name__ == "__main__":
    unittest.main()
