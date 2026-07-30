import asyncio
import unittest
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.fault_tolerance.manager import FaultToleranceManager
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    FaultToleranceCommandReqOutput,
    FaultToleranceRankFaultOutput,
    ProcessActiveRanksOutput,
    WatchdogHeartbeatOutput,
)


def make_manager(*, dp_size=2, strategy="pause"):
    server_args = SimpleNamespace(
        dp_size=dp_size,
        fault_tolerance_on_error_strategy=strategy,
        fault_tolerance_timeout=1,
        fault_tolerance_pause_timeout=1,
    )
    manager = FaultToleranceManager(
        server_args=server_args,
        send_to_scheduler=AsyncMock(),
    )
    manager.bind_event_loop(asyncio.get_running_loop())
    return manager


class TestFaultToleranceManager(unittest.IsolatedAsyncioTestCase):
    async def _stop_watchdog_lease_sweep(self, manager):
        await asyncio.sleep(0)
        task = manager._watchdog_lease_task
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def test_unattended_pause_failstops(self):
        manager = make_manager()
        manager.server_args.fault_tolerance_pause_timeout = 0.01
        manager._send_command_collect = AsyncMock(return_value={0, 1})
        manager._failstop = Mock()

        await manager._pause_schedulers([0, 1])
        await asyncio.sleep(0.03)

        manager._failstop.assert_called_once()
        self.assertIn("pause unattended", manager._failstop.call_args.args[0])

    async def test_valid_retry_cancels_paused_failstop(self):
        manager = make_manager()
        manager.state.finish_pause({0, 1})
        manager._arm_paused_failstop()
        handle = manager._paused_failstop_handle
        manager._send_command_collect = AsyncMock(return_value={0, 1})

        status, _ = await manager.apply(
            {"fault_tolerance_instruction": "retry"}
        )

        self.assertEqual(status, 200)
        self.assertTrue(handle.cancelled())
        self.assertIsNone(manager._paused_failstop_handle)

    async def test_failed_retry_rearms_paused_failstop(self):
        manager = make_manager()
        manager.state.finish_pause({0, 1})
        manager.state.process_active_ranks[1] = False
        manager._arm_paused_failstop()
        original_handle = manager._paused_failstop_handle
        manager._publish_active_ranks = AsyncMock(
            side_effect=RuntimeError("route update failed")
        )

        status, _ = await manager.apply(
            {"fault_tolerance_instruction": "retry"}
        )

        self.assertEqual(status, 503)
        self.assertTrue(original_handle.cancelled())
        self.assertIsNot(manager._paused_failstop_handle, original_handle)
        manager._cancel_paused_failstop()

    async def test_arming_paused_failstop_is_idempotent(self):
        manager = make_manager()
        manager.state.finish_pause({0, 1})

        manager._arm_paused_failstop()
        handle = manager._paused_failstop_handle
        manager._arm_paused_failstop()

        self.assertIs(manager._paused_failstop_handle, handle)
        manager._cancel_paused_failstop()

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
            manager.state.status_response()["ranks"],
            [
                {"rank": 0, "state": "paused"},
                {"rank": 1, "state": "dead"},
                {"rank": 2, "state": "dead"},
                {"rank": 3, "state": "paused"},
            ],
        )
        self.assertEqual(manager._pending_commands, {})

    async def test_runtime_rejoin_does_not_clear_disabled_without_recover(self):
        manager = make_manager(strategy="continue")
        manager._publish_active_ranks = AsyncMock()
        manager._send_command_collect = AsyncMock()
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

        status, response = await manager.apply(
            {
                "fault_tolerance_instruction": "recover",
                "fault_tolerance_params": {"ranks": [1]},
            }
        )

        self.assertEqual(status, 200)
        self.assertEqual(response["ranks"][1]["state"], "healthy")
        self.assertEqual(manager.state.disabled_dp_ranks, set())
        manager._publish_active_ranks.assert_awaited_once_with([True, True], 1)
        manager._send_command_collect.assert_not_awaited()

    async def test_recover_runtime_inactive_rank_defers_route_publish(self):
        manager = make_manager(strategy="continue")
        manager._publish_active_ranks = AsyncMock()
        manager.state.disabled_dp_ranks.add(1)
        process_down = manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=False)
        )
        self.assertEqual(process_down.status, [True, False])

        status, response = await manager.apply(
            {
                "fault_tolerance_instruction": "recover",
                "fault_tolerance_params": {"ranks": [1]},
            }
        )

        self.assertEqual(status, 200)
        self.assertEqual(response["ranks"][1]["state"], "dead")
        manager._publish_active_ranks.assert_not_awaited()

        process_up = manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=True)
        )
        self.assertEqual(process_up.status, [True, True])
        self.assertEqual(
            manager.state.status_response()["ranks"][1]["state"], "healthy"
        )

    async def test_recover_while_paused_does_not_resume_schedulers(self):
        manager = make_manager()
        manager.state.disabled_dp_ranks.add(1)
        manager.state.paused_dp_ranks = {0, 1}
        manager.state._last_published_effective_active_mask = [True, False]
        manager._publish_active_ranks = AsyncMock()
        manager._send_command_collect = AsyncMock()

        status, response = await manager.apply(
            {
                "fault_tolerance_instruction": "recover",
                "fault_tolerance_params": {"ranks": [1]},
            }
        )

        self.assertEqual(status, 200)
        self.assertEqual(response["ranks"][1]["state"], "paused")
        self.assertEqual(manager.state.paused_dp_ranks, {0, 1})
        manager._send_command_collect.assert_not_awaited()

    async def test_recover_is_rejected_while_ft_operation_is_in_progress(self):
        manager = make_manager()
        manager.state.disabled_dp_ranks.add(1)
        manager.state.ft_operation_in_progress = True

        status, response = await manager.apply(
            {
                "fault_tolerance_instruction": "recover",
                "fault_tolerance_params": {"ranks": [1]},
            }
        )

        self.assertEqual(status, 409)
        self.assertEqual(response["message"], "ft_operation_in_progress")

    async def test_recover_validation_rejects_invalid_rank_requests(self):
        manager = make_manager()
        cases = [
            ({}, "recover_requires_non_empty_ranks"),
            (
                {"fault_tolerance_params": {"ranks": []}},
                "recover_requires_non_empty_ranks",
            ),
            (
                {"fault_tolerance_params": {"ranks": "1"}},
                "recover_requires_non_empty_ranks",
            ),
            ({"fault_tolerance_params": {"ranks": [2]}}, "unknown_rank"),
            (
                {"fault_tolerance_params": {"ranks": [1]}},
                "recover_requires_disabled_ranks",
            ),
        ]

        for extra, expected_message in cases:
            with self.subTest(expected_message=expected_message, extra=extra):
                request = {"fault_tolerance_instruction": "recover", **extra}
                status, response = await manager.apply(request)
                self.assertEqual(status, 400)
                self.assertEqual(response["message"], expected_message)

    async def test_recover_route_failure_does_not_restore_disabled_state(self):
        manager = make_manager(strategy="continue")
        manager.state.disabled_dp_ranks.add(1)
        manager.state._last_published_effective_active_mask = [True, False]
        manager._publish_active_ranks = AsyncMock(side_effect=TimeoutError("route ack"))
        manager._send_command_collect = AsyncMock()

        status, response = await manager.apply(
            {
                "fault_tolerance_instruction": "recover",
                "fault_tolerance_params": {"ranks": [1]},
            }
        )

        self.assertEqual(status, 503)
        self.assertFalse(response["success"])
        self.assertEqual(manager.state.disabled_dp_ranks, set())
        self.assertFalse(manager.state.ft_operation_in_progress)
        manager._send_command_collect.assert_not_awaited()

    async def test_exception_does_not_change_availability_sources(self):
        manager = make_manager()
        manager._pause_schedulers = AsyncMock()
        before = (
            list(manager.state.process_active_ranks),
            list(manager.state.mooncake_active_ranks),
            set(manager.state.disabled_dp_ranks),
        )

        event = FaultToleranceRankFaultOutput(
            rank=1,
            message="boom",
        )
        manager.handle_rank_fault(event)
        await asyncio.sleep(0)

        manager._pause_schedulers.assert_awaited_once_with([0, 1])
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
        manager._publish_active_ranks = AsyncMock()
        manager._send_command_collect = AsyncMock(return_value={0})

        status, response = await manager.apply({"fault_tolerance_instruction": "retry"})

        self.assertEqual(status, 200)
        self.assertEqual(response["resumed_ranks"], [0])
        self.assertEqual(
            response["ranks"],
            [
                {"rank": 0, "state": "healthy"},
                {"rank": 1, "state": "dead"},
            ],
        )
        self.assertEqual(
            manager._send_command_collect.await_args.kwargs["target_ranks"],
            [0],
        )
        self.assertEqual(manager.state.paused_dp_ranks, set())

    async def test_scale_down_is_logical_isolation(self):
        manager = make_manager()
        manager.state.begin_exception_pause()
        manager.state.finish_pause({0, 1})
        manager._publish_active_ranks = AsyncMock()
        manager._send_command_collect = AsyncMock(return_value={0, 1})

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
            response["ranks"],
            [
                {"rank": 0, "state": "healthy"},
                {"rank": 1, "state": "disabled"},
            ],
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

    async def test_scale_down_route_failure_preserves_paused_ranks(self):
        manager = make_manager()
        manager.state.begin_exception_pause()
        manager.state.finish_pause({0, 1})
        manager._publish_active_ranks = AsyncMock(side_effect=TimeoutError("route ack"))
        manager._send_command_collect = AsyncMock()

        status, response = await manager.apply(
            {
                "fault_tolerance_instruction": "scale_down",
                "fault_tolerance_params": {"ranks": [1]},
            }
        )

        self.assertEqual(status, 503)
        self.assertFalse(response["success"])
        self.assertEqual(manager.state.paused_dp_ranks, {0, 1})
        manager._send_command_collect.assert_not_awaited()

    async def test_watchdog_heartbeat_registers_and_refreshes_without_process_up(self):
        manager = make_manager(strategy="continue")
        manager.state.process_active_ranks[1] = False

        try:
            with patch(
                "sglang.srt.fault_tolerance.manager.time.monotonic",
                return_value=10.0,
            ):
                manager.observe_watchdog_heartbeat(
                    WatchdogHeartbeatOutput(node_rank=3, ranks=[1])
                )

            self.assertEqual(manager._watchdog_leases, {3: (10.0, (1,))})
            self.assertEqual(manager.state.process_active_ranks, [True, False])
            self.assertEqual(manager.state.mooncake_active_ranks, [True, True])

            with patch(
                "sglang.srt.fault_tolerance.manager.time.monotonic",
                return_value=12.0,
            ):
                manager.observe_watchdog_heartbeat(
                    WatchdogHeartbeatOutput(node_rank=3, ranks=[0])
                )

            self.assertEqual(manager._watchdog_leases, {3: (12.0, (1,))})
            self.assertEqual(manager.state.process_active_ranks, [True, False])
            manager.send_to_scheduler.assert_not_awaited()
        finally:
            await self._stop_watchdog_lease_sweep(manager)

    async def test_watchdog_lease_timeout_and_late_reregistration_do_not_mark_up(self):
        manager = make_manager(strategy="continue")

        try:
            with patch(
                "sglang.srt.fault_tolerance.manager.time.monotonic",
                return_value=10.0,
            ):
                manager.observe_watchdog_heartbeat(
                    WatchdogHeartbeatOutput(node_rank=3, ranks=[1])
                )

            await manager._sweep_expired_watchdog_leases(now=14.99)
            self.assertIn(3, manager._watchdog_leases)
            manager.send_to_scheduler.assert_not_awaited()

            await manager._sweep_expired_watchdog_leases(now=15.0)
            self.assertNotIn(3, manager._watchdog_leases)
            self.assertEqual(manager.state.process_active_ranks, [True, False])
            self.assertEqual(manager.state.mooncake_active_ranks, [True, True])
            manager.send_to_scheduler.assert_awaited_once()

            await manager._sweep_expired_watchdog_leases(now=20.0)
            manager.send_to_scheduler.assert_awaited_once()

            with patch(
                "sglang.srt.fault_tolerance.manager.time.monotonic",
                return_value=21.0,
            ):
                manager.observe_watchdog_heartbeat(
                    WatchdogHeartbeatOutput(node_rank=3, ranks=[1])
                )

            self.assertEqual(manager._watchdog_leases, {3: (21.0, (1,))})
            self.assertEqual(manager.state.process_active_ranks, [True, False])
            manager.send_to_scheduler.assert_awaited_once()
        finally:
            await self._stop_watchdog_lease_sweep(manager)

    async def test_watchdog_sweep_unions_duplicate_ranks_into_one_down_update(self):
        manager = make_manager(dp_size=4, strategy="continue")
        observe_process_active_ranks = manager.observe_process_active_ranks
        manager.observe_process_active_ranks = Mock(
            wraps=observe_process_active_ranks
        )

        try:
            with patch(
                "sglang.srt.fault_tolerance.manager.time.monotonic",
                return_value=10.0,
            ):
                manager.observe_watchdog_heartbeat(
                    WatchdogHeartbeatOutput(node_rank=3, ranks=[1, 2])
                )
                manager.observe_watchdog_heartbeat(
                    WatchdogHeartbeatOutput(node_rank=4, ranks=[2, 3])
                )

            await manager._sweep_expired_watchdog_leases(now=15.0)

            manager.observe_process_active_ranks.assert_called_once()
            process_down = manager.observe_process_active_ranks.call_args.args[0]
            self.assertEqual(process_down.ranks, [1, 2, 3])
            self.assertFalse(process_down.active)
            self.assertEqual(
                manager.state.process_active_ranks,
                [True, False, False, False],
            )
            self.assertEqual(
                manager.state.mooncake_active_ranks,
                [True, True, True, True],
            )
            manager.send_to_scheduler.assert_awaited_once()

            await manager._sweep_expired_watchdog_leases(now=20.0)
            manager.observe_process_active_ranks.assert_called_once()
            manager.send_to_scheduler.assert_awaited_once()
        finally:
            await self._stop_watchdog_lease_sweep(manager)

    async def test_watchdog_lease_timeout_schedules_pause_for_runtime_ranks(self):
        manager = make_manager(dp_size=3)
        manager._pause_schedulers = AsyncMock()

        try:
            with patch(
                "sglang.srt.fault_tolerance.manager.time.monotonic",
                return_value=10.0,
            ):
                manager.observe_watchdog_heartbeat(
                    WatchdogHeartbeatOutput(node_rank=3, ranks=[1])
                )

            await manager._sweep_expired_watchdog_leases(now=15.0)
            await asyncio.sleep(0)

            manager._pause_schedulers.assert_awaited_once_with([0, 2])
            self.assertEqual(
                manager.state.process_active_ranks,
                [True, False, True],
            )
            self.assertEqual(
                manager.state.mooncake_active_ranks,
                [True, True, True],
            )
        finally:
            await self._stop_watchdog_lease_sweep(manager)

    async def test_fatal_task_wrapper_reuses_failstop(self):
        manager = make_manager()
        manager._failstop = Mock(side_effect=RuntimeError("failstop"))

        async def fail():
            raise ValueError("unexpected")

        with self.assertRaisesRegex(RuntimeError, "failstop"):
            await manager._fatal_task_wrapper(fail())

        manager._failstop.assert_called_once()
        self.assertIn("FaultToleranceManager hit an exception", manager._failstop.call_args.args[0])

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
