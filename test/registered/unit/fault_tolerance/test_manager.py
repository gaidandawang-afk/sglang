import asyncio
import unittest
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from sglang.srt.fault_tolerance.manager import (
    WATCHDOG_LEASE_TIMEOUT_SEC,
    FaultToleranceManager,
)
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    FaultToleranceCommandReqOutput,
    FaultToleranceRankFaultOutput,
    ProcessActiveRanksOutput,
    WatchdogHeartbeatOutput,
)


def make_request(obj):
    return SimpleNamespace(
        instruction=obj["instruction"],
        params=SimpleNamespace(**obj.get("params", {})),
        request_id=obj.get("request_id", ""),
    )


def make_manager(*, dp_size=2, ranks_per_dp=1, strategy="pause"):
    manager = FaultToleranceManager(
        server_args=SimpleNamespace(
            dp_size=dp_size,
            tp_size=dp_size * ranks_per_dp,
            fault_tolerance_on_error_strategy=strategy,
            fault_tolerance_timeout=1,
        ),
        send_to_scheduler=AsyncMock(),
        send_to_dpc=AsyncMock(),
    )
    manager.bind_event_loop(asyncio.get_running_loop())
    return manager


async def submit_and_finish(manager, obj):
    status, response = manager.submit(make_request(obj))
    tasks = list(manager.asyncio_tasks)
    if tasks:
        await asyncio.gather(*tasks)
    return status, response


class TestFaultToleranceManager(unittest.IsolatedAsyncioTestCase):
    async def _stop_watchdog(self, manager):
        await asyncio.sleep(0)
        task = manager._watchdog_lease_task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def test_retry_is_maskless_and_uses_expected_topology(self):
        manager = make_manager(dp_size=4)
        manager.state.expected_dp_mask = [True, True, False, True]
        manager.state.process_alive_global_rank_mask[2] = False
        manager.state.unhealthy_dp_ranks.add(0)
        manager._send_command_collect = AsyncMock()
        manager._publish_route_dp_mask = AsyncMock()

        status, response = await submit_and_finish(
            manager,
            {
                "instruction": "retry",
                "params": {"timeout": 99},
                "request_id": "retry-1",
            },
        )

        self.assertEqual(status, 202)
        self.assertEqual(response["request_id"], "retry-1")
        self.assertEqual(manager.status()[1]["engines"][2]["status"], "dead")
        manager._send_command_collect.assert_awaited_once_with(
            command="retry",
            target_ranks=[0, 1, 3],
            timeout_sec=1,
        )
        manager._publish_route_dp_mask.assert_awaited_once_with(
            [True, True, False, True], 1
        )

    async def test_operations_require_unresolved_expected_dp_fault(self):
        cases = [
            (
                {"instruction": "retry", "params": {}, "request_id": "retry-healthy"},
                "retry_requires_unresolved_expected_dp_fault",
            ),
            (
                {
                    "instruction": "scale_down",
                    "params": {"removed_dp_ranks": [1]},
                    "request_id": "scale-down-healthy",
                },
                "scale_down_requires_unresolved_expected_dp_fault",
            ),
        ]

        for request, expected_error in cases:
            with self.subTest(instruction=request["instruction"]):
                manager = make_manager()
                status, _ = await submit_and_finish(manager, request)

                self.assertEqual(status, 202)
                for engine in manager.status()[1]["engines"]:
                    self.assertEqual(engine["ft_error"], expected_error)

    async def test_retry_rejects_process_loss(self):
        manager = make_manager()
        manager.state.observe_process_active_ranks([1], active=False)

        status, _ = await submit_and_finish(
            manager,
            {"instruction": "retry", "params": {}, "request_id": "retry-2"},
        )

        self.assertEqual(status, 202)
        for engine in manager.status()[1]["engines"]:
            self.assertEqual(engine["last_ft_request_id"], "retry-2")
            self.assertEqual(
                engine["ft_error"], "retry_requires_all_expected_processes_alive"
            )
        self.assertFalse(manager.state.ft_operation_in_progress)

    async def test_scale_down_has_one_scheduler_phase_after_shutdown(self):
        manager = make_manager(dp_size=4)
        manager.state.unhealthy_dp_ranks.add(0)
        order = []

        async def shutdown(ranks, timeout):
            order.append("shutdown")
            manager.state.observe_process_active_ranks(ranks, active=False)

        async def command(**kwargs):
            order.append(kwargs["command"])
            return set(kwargs["target_ranks"])

        async def publish(mask, timeout):
            order.append("route")

        manager._shutdown_dp_processes = shutdown
        manager._send_command_collect = command
        manager._publish_route_dp_mask = publish

        status, _ = await submit_and_finish(
            manager,
            {
                "instruction": "scale_down",
                "params": {"removed_dp_ranks": [2]},
                "request_id": "scale-1",
            },
        )

        self.assertEqual(status, 202)
        self.assertEqual(order, ["shutdown", "scale_down", "route"])
        self.assertEqual(manager.state.expected_dp_mask, [True, True, False, True])
        self.assertEqual(manager.status()[1]["engines"][2]["status"], "dead")

    async def test_scale_down_sends_sparse_global_mask(self):
        manager = make_manager(dp_size=4, ranks_per_dp=2)
        manager.state.unhealthy_dp_ranks.add(0)
        manager._shutdown_dp_processes = AsyncMock()
        manager._send_command_collect = AsyncMock()
        manager._publish_route_dp_mask = AsyncMock()

        await submit_and_finish(
            manager,
            {
                "instruction": "scale_down",
                "params": {"removed_dp_ranks": [2]},
            },
        )

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

    async def test_pause_runtime_ready_automatically_reopens_route(self):
        manager = make_manager()
        manager.state.finish_scale_down([1])
        manager._route_dp_mask = [True, False]
        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=False)
        )
        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=True)
        )

        self.assertEqual(manager.status()[1]["engines"][1]["status"], "dead")
        self.assertEqual(manager.state.pending_recovery_global_ranks, {1})

        update = manager.observe_active_ranks(ActiveRanksOutput(status=[True, True]))

        self.assertEqual(update.status, [True, True])
        self.assertEqual(manager.state.expected_dp_mask, [True, True])
        self.assertEqual(manager.state.pending_recovery_global_ranks, set())
        self.assertEqual(manager.status()[1]["engines"][1]["status"], "healthy")

    async def test_submit_uses_vllm_accepted_response_and_rejects_concurrency(self):
        manager = make_manager()
        manager.state.unhealthy_dp_ranks.add(0)
        command_started = asyncio.Event()
        release_command = asyncio.Event()

        async def command(**kwargs):
            command_started.set()
            await release_command.wait()
            return set(kwargs["target_ranks"])

        manager._send_command_collect = command
        manager._publish_route_dp_mask = AsyncMock()

        status, response = manager.submit(
            make_request({"instruction": "retry", "request_id": "request-1"})
        )

        self.assertEqual(status, 202)
        self.assertEqual(
            response,
            {
                "message": (
                    "Request accepted; poll /fault_tolerance/status for updates."
                ),
                "request_id": "request-1",
            },
        )
        self.assertTrue(manager.state.ft_operation_in_progress)
        await command_started.wait()

        busy_status, busy_response = manager.submit(
            make_request({"instruction": "retry", "request_id": "request-2"})
        )
        self.assertEqual(busy_status, 409)
        self.assertEqual(busy_response["message"], "ft_operation_in_progress")

        tasks = list(manager.asyncio_tasks)
        release_command.set()
        await asyncio.gather(*tasks)
        self.assertFalse(manager.state.ft_operation_in_progress)

    async def test_success_records_request_id_and_clears_previous_aggregate_error(self):
        manager = make_manager()
        manager._last_ft_request_id = "old-request"
        manager._ft_error = "old-error"
        manager.state.unhealthy_dp_ranks.add(0)
        manager._send_command_collect = AsyncMock()
        manager._publish_route_dp_mask = AsyncMock()

        await submit_and_finish(
            manager,
            {"instruction": "retry", "request_id": "new-request"},
        )

        for engine in manager.status()[1]["engines"]:
            self.assertEqual(engine["last_ft_request_id"], "new-request")
            self.assertNotIn("ft_error", engine)

    async def test_lower_execution_failure_keeps_failstop_behavior(self):
        manager = make_manager()
        manager.state.unhealthy_dp_ranks.add(0)
        manager._send_command_collect = AsyncMock(
            side_effect=RuntimeError("ack failed")
        )
        manager._failstop = Mock(side_effect=RuntimeError("failstop"))

        status, _ = manager.submit(
            make_request({"instruction": "retry", "request_id": "request-fail"})
        )
        tasks = list(manager.asyncio_tasks)

        self.assertEqual(status, 202)
        with self.assertRaisesRegex(RuntimeError, "failstop"):
            await asyncio.gather(*tasks)
        manager._failstop.assert_called_once()
        self.assertTrue(manager.state.ft_operation_in_progress)
        self.assertIsNone(manager._ft_error)

    async def test_scale_down_rejects_rank_above_dp_size(self):
        manager = make_manager()
        request = make_request(
            {
                "instruction": "scale_down",
                "params": {"removed_dp_ranks": [4]},
            }
        )
        status, response = manager.submit(request)
        self.assertEqual(status, 400)
        self.assertEqual(
            response["message"],
            "'removed_dp_ranks' contains a rank out of range.",
        )

    async def test_runtime_ready_before_process_up_still_auto_recovers(self):
        manager = make_manager()
        manager.state.finish_scale_down([1])
        manager._route_dp_mask = [True, False]
        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=False)
        )

        self.assertIsNone(
            manager.observe_active_ranks(ActiveRanksOutput(status=[True, True]))
        )
        self.assertEqual(manager.state.expected_dp_mask, [True, False])
        self.assertEqual(manager.status()[1]["engines"][1]["status"], "dead")

        update = manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=True)
        )

        self.assertEqual(update.status, [True, True])
        self.assertEqual(manager.state.expected_dp_mask, [True, True])
        self.assertEqual(manager.status()[1]["engines"][1]["status"], "healthy")

    async def test_process_up_waits_for_runtime_ready_before_reopening_route(self):
        manager = make_manager()
        manager.state.expected_dp_mask[1] = False
        manager._route_dp_mask = [True, False]
        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=False)
        )
        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[1], active=True)
        )
        self.assertEqual(manager.status()[1]["engines"][1]["status"], "dead")

        update = manager.observe_active_ranks(ActiveRanksOutput(status=[True, True]))
        self.assertEqual(manager.status()[1]["engines"][1]["status"], "healthy")
        self.assertEqual(update.status, [True, True])

    async def test_continue_observations_keep_expected_topology_without_ft_operation(
        self,
    ):
        manager = make_manager(dp_size=2, ranks_per_dp=2, strategy="continue")

        update = manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[2, 3], active=False)
        )
        self.assertEqual(manager.state.expected_dp_mask, [True, True])
        self.assertEqual(update.status, [True, False])
        self.assertEqual(manager.status()[1]["engines"][1]["status"], "dead")
        self.assertIsNone(manager.admission_error(None))

        # A process-up event cannot reopen the route using stale runtime state.
        self.assertIsNone(
            manager.observe_process_active_ranks(
                ProcessActiveRanksOutput(ranks=[2, 3], active=True)
            )
        )
        self.assertEqual(manager.status()[1]["engines"][1]["status"], "dead")

        # A fresh runtime-ready observation completes the rejoin.
        update = manager.observe_active_ranks(ActiveRanksOutput(status=[True, True]))

        self.assertEqual(manager.state.expected_dp_mask, [True, True])
        self.assertEqual(update.status, [True, True])
        self.assertEqual(manager.status()[1]["engines"][1]["status"], "healthy")
        self.assertFalse(manager.state.ft_operation_in_progress)
        manager.send_to_scheduler.assert_not_awaited()
        manager.send_to_dpc.assert_not_awaited()

    async def test_continue_runtime_ready_before_process_up_reopens_route_last(self):
        manager = make_manager(dp_size=2, ranks_per_dp=2, strategy="continue")

        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[2, 3], active=False)
        )
        manager.observe_active_ranks(ActiveRanksOutput(status=[True, False]))
        self.assertIsNone(
            manager.observe_active_ranks(ActiveRanksOutput(status=[True, True]))
        )
        self.assertEqual(manager.status()[1]["engines"][1]["status"], "dead")

        update = manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[2, 3], active=True)
        )

        self.assertEqual(manager.state.expected_dp_mask, [True, True])
        self.assertEqual(update.status, [True, True])
        self.assertEqual(manager.status()[1]["engines"][1]["status"], "healthy")

    async def test_continue_runtime_observation_updates_route_without_topology_change(
        self,
    ):
        manager = make_manager(strategy="continue")

        update = manager.observe_active_ranks(ActiveRanksOutput(status=[True, False]))
        self.assertEqual(manager.state.expected_dp_mask, [True, True])
        self.assertEqual(update.status, [True, False])
        self.assertEqual(manager.status()[1]["engines"][1]["status"], "dead")

        update = manager.observe_active_ranks(ActiveRanksOutput(status=[True, True]))
        self.assertEqual(manager.state.expected_dp_mask, [True, True])
        self.assertEqual(update.status, [True, True])
        self.assertEqual(manager.status()[1]["engines"][1]["status"], "healthy")

    async def test_continue_route_intersects_runtime_process_and_expected(self):
        manager = make_manager(dp_size=2, ranks_per_dp=2, strategy="continue")
        manager.state.expected_dp_mask[1] = False
        # DP1 (global ranks 2,3) is fully down and not yet runtime-recovered.
        manager.state.observe_process_active_ranks([2, 3], active=False)

        update = manager.observe_active_ranks(ActiveRanksOutput(status=[True, False]))

        # DP1 cannot auto-recover until process and runtime state are ready.
        self.assertEqual(manager.state.expected_dp_mask, [True, False])
        self.assertEqual(update.status, [True, False])

    async def test_continue_runtime_recovery_auto_recovers_dead_dp(self):
        manager = make_manager(dp_size=2, ranks_per_dp=2, strategy="continue")
        # Scale-down DP1 then kill it; it stays dead until it rejoins and its
        # data plane recovers, at which point "continue" re-admits it automatically.
        manager.state.finish_scale_down([1])
        manager._route_dp_mask = [True, False]
        manager.state.observe_process_active_ranks([2, 3], active=False)
        manager.state.observe_process_active_ranks([2, 3], active=True)
        self.assertEqual(manager.state.expected_dp_mask, [True, False])
        self.assertEqual(manager.state.pending_recovery_global_ranks, {2, 3})

        update = manager.observe_active_ranks(ActiveRanksOutput(status=[True, True]))

        # Runtime readiness clears pending ranks and recovers DP1 automatically.
        self.assertEqual(manager.state.pending_recovery_global_ranks, set())
        self.assertEqual(manager.state.expected_dp_mask, [True, True])
        self.assertEqual(update.status, [True, True])
        self.assertEqual(manager.status()[1]["engines"][1]["status"], "healthy")

    async def test_watchdog_lease_expiry_marks_global_processes_down(self):
        manager = make_manager(dp_size=2, ranks_per_dp=2, strategy="continue")
        manager.observe_watchdog_heartbeat(
            WatchdogHeartbeatOutput(node_rank=1, ranks=[2, 3])
        )
        last_seen, ranks = manager._watchdog_leases[1]

        await manager._sweep_expired_watchdog_leases(
            now=last_seen + WATCHDOG_LEASE_TIMEOUT_SEC + 1
        )

        self.assertEqual(ranks, (2, 3))
        self.assertEqual(
            manager.state.process_alive_global_rank_mask,
            [True, True, False, False],
        )
        manager.send_to_scheduler.assert_awaited_once()
        await self._stop_watchdog(manager)

    async def test_shutdown_completion_comes_from_process_down_events(self):
        manager = make_manager(dp_size=2, ranks_per_dp=2)
        manager._watchdog_leases = {
            0: (0.0, (0, 1)),
            1: (0.0, (2, 3)),
        }

        task = asyncio.create_task(manager._shutdown_dp_processes({1}, 1))
        await asyncio.sleep(0)
        manager.send_to_dpc.assert_awaited_once()
        nodes, request = manager.send_to_dpc.await_args.args
        self.assertEqual(nodes, [1])
        self.assertEqual(request.target_dp_ranks, [1])

        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[2, 3], active=False)
        )
        await task
        self.assertIsNone(manager._shutdown_waiter)

    async def test_command_waits_for_target_dp_acks(self):
        manager = make_manager()
        task = asyncio.create_task(
            manager._send_command_collect(
                command="retry", target_ranks=[0, 1], timeout_sec=1
            )
        )
        await asyncio.sleep(0)
        request = manager.send_to_scheduler.await_args.args[0]
        for rank in (0, 1):
            manager.handle_command_output(
                FaultToleranceCommandReqOutput(
                    request_id=request.request_id,
                    rank=rank,
                )
            )
        self.assertIsNone(await task)

    async def test_rank_fault_only_updates_unhealthy_state(self):
        manager = make_manager()
        manager.handle_rank_fault(FaultToleranceRankFaultOutput(rank=1, message="boom"))

        self.assertEqual(manager.state.unhealthy_dp_ranks, {1})
        self.assertFalse(hasattr(manager, "_paused_failstop_handle"))

    async def test_admission_reports_scaled_down_rank(self):
        manager = make_manager()
        manager._route_dp_mask = [True, False]

        self.assertIsNone(manager.admission_error(0))
        self.assertEqual(manager.admission_error(1), "routed_dp_rank=1 is not active")


if __name__ == "__main__":
    unittest.main()
