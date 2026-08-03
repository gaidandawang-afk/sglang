from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.fault_tolerance.controller import (
    FaultToleranceState,
    ft_failure,
)
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    ActiveRanksUpdateReqOutput,
    FaultToleranceCommandReqInput,
    FaultToleranceCommandReqOutput,
    FaultToleranceRankFaultOutput,
    ProcessActiveRanksOutput,
    WatchdogHeartbeatOutput,
)
from sglang.srt.utils import kill_process_tree
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)

WATCHDOG_LEASE_SWEEP_INTERVAL_SEC = 1
WATCHDOG_LEASE_TIMEOUT_SEC = 5


@dataclasses.dataclass
class PendingFTCommand:
    command: str
    target_ranks: set[int]
    future: asyncio.Future
    acked: set[int] = dataclasses.field(default_factory=set)
    failed: Dict[int, str] = dataclasses.field(default_factory=dict)

    def finish_if_ready(self):
        if self.future.done():
            return
        if self.acked.union(self.failed) >= self.target_ranks:
            self.future.set_result(None)


class FaultToleranceManager:
    def __init__(self, *, server_args, send_to_scheduler):
        self.server_args = server_args
        self.send_to_scheduler = send_to_scheduler
        self.state = FaultToleranceState(
            dp_size=server_args.dp_size,
            strategy=server_args.fault_tolerance_on_error_strategy,
        )
        self.event_loop = None
        self.asyncio_tasks = None
        self._pending_commands: Dict[str, PendingFTCommand] = {}
        self._pending_active_rank_updates: Dict[str, asyncio.Future] = {}
        self._watchdog_leases: Dict[int, Tuple[float, Tuple[int, ...]]] = {}
        self._watchdog_lease_task: Optional[asyncio.Task] = None
        self._paused_failstop_handle: Optional[asyncio.TimerHandle] = None

    def bind_event_loop(self, loop) -> None:
        if self.event_loop is loop:
            return
        if self.event_loop is not None:
            raise RuntimeError(
                "fault tolerance manager is already bound to an event loop"
            )
        self.event_loop = loop
        self.asyncio_tasks = set()

    def status(self) -> tuple[int, dict]:
        return 200, self.state.status_response()

    def _parse_apply_args(self, obj: Dict[str, Any]) -> Tuple[str, List[int], int]:
        instruction = obj["instruction"]
        if instruction not in ("retry", "scale_down", "recover"):
            raise ValueError(f"unsupported instruction: {instruction}")

        params = obj["params"]
        timeout = params.get("timeout", self.server_args.fault_tolerance_timeout)
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        ranks = params.get("ranks", [])
        if any(rank < 0 or rank >= self.state.dp_size for rank in ranks):
            raise ValueError("rank out of range")
        return instruction, ranks, timeout

    async def apply(self, obj: Dict[str, Any]) -> tuple[int, dict]:
        try:
            instruction, ranks, timeout = self._parse_apply_args(obj)
        except Exception as exc:
            return 400, ft_failure(f"invalid params: {exc}")
        if self.state.ft_operation_in_progress:
            return 409, ft_failure("ft_operation_in_progress")

        try:
            if instruction == "retry":
                return await self._apply_retry(timeout)
            if instruction == "scale_down":
                return await self._apply_scale_down(ranks, timeout)
            return await self._apply_recover(ranks, timeout)
        except Exception as exc:
            self._failstop(f"fault tolerance apply {instruction} failed: {exc}")

    async def _publish_pending_route(self, timeout: int) -> Optional[tuple[int, dict]]:
        st = self.state
        pending = st.get_unpublished_effective_active_mask()
        if pending is None:
            return None
        try:
            await self._publish_active_ranks(pending, timeout)
            st.mark_effective_active_mask_published(pending)
            return None
        except Exception as exc:
            logger.exception("Fault tolerance apply failed: %s", exc)
            body = st.commit_recover(resumed_ranks=set(), clear_paused=False)
            body.update(
                {
                    "success": False,
                    "message": f"fault_tolerance_apply_failed: {exc}",
                }
            )
            return 503, body

    async def _apply_retry(self, timeout: int) -> tuple[int, dict]:
        st = self.state
        if not st.has_paused_rank():
            return 400, ft_failure("no_paused_rank")

        self._cancel_paused_failstop()
        st.ft_operation_in_progress = True
        error = await self._publish_pending_route(timeout)
        if error is not None:
            self._arm_paused_failstop()
            return error
        acked = await self._send_command_collect(
            command="resume",
            target_ranks=st.resume_targets(),
            timeout_sec=timeout,
        )
        return 200, st.commit_recover(resumed_ranks=acked, clear_paused=True)

    async def _apply_scale_down(
        self, ranks: List[int], timeout: int
    ) -> tuple[int, dict]:
        st = self.state
        if not st.has_paused_rank():
            return 400, ft_failure("no_paused_rank")
        requested = set(ranks)
        disabled = st.disabled_dp_ranks | requested
        if not any(
            is_active and rank not in disabled
            for rank, is_active in enumerate(st.runtime_active_mask())
        ):
            return 400, ft_failure("cannot_isolate_all_active_ranks")

        self._cancel_paused_failstop()
        st.ft_operation_in_progress = True
        st.disabled_dp_ranks |= requested
        error = await self._publish_pending_route(timeout)
        if error is not None:
            self._arm_paused_failstop()
            return error
        acked = await self._send_command_collect(
            command="resume",
            target_ranks=st.resume_targets(),
            timeout_sec=timeout,
        )
        return 200, st.commit_recover(resumed_ranks=acked, clear_paused=True)

    async def _apply_recover(self, ranks: List[int], timeout: int) -> tuple[int, dict]:
        st = self.state
        requested = set(ranks)
        if not requested.issubset(st.disabled_dp_ranks):
            return 400, ft_failure("recover_requires_disabled_ranks")

        st.ft_operation_in_progress = True
        st.disabled_dp_ranks -= requested
        error = await self._publish_pending_route(timeout)
        if error is not None:
            return error
        return 200, st.commit_recover(resumed_ranks=set(), clear_paused=False)

    def validate_routed_rank(self, rank: int) -> None:
        if not self.state.is_rank_routable(rank):
            raise ValueError(f"routed_dp_rank={rank} is not active")

    def should_reject_admission(self) -> bool:
        return self.state.should_reject_admission()

    def observe_active_ranks(
        self, ranks: ActiveRanksOutput
    ) -> Optional[ActiveRanksOutput]:
        targets = self.state.observe_mooncake_active_ranks(ranks.status)
        if targets:
            self._create_task(self._pause_schedulers(targets))
        active_mask = self.state.get_unpublished_effective_active_mask()
        if active_mask is None:
            return None
        self.state.mark_effective_active_mask_published(active_mask)
        return ActiveRanksOutput(status=active_mask)

    def observe_process_active_ranks(
        self, ranks: ProcessActiveRanksOutput
    ) -> Optional[ActiveRanksOutput]:
        targets = self.state.observe_process_active_ranks(
            ranks.ranks, active=ranks.active
        )
        if not ranks.active:
            self._drop_process_inactive_pause_targets(set(ranks.ranks))
        if targets:
            self._create_task(self._pause_schedulers(targets))
        active_mask = self.state.get_unpublished_effective_active_mask()
        if active_mask is None:
            return None
        self.state.mark_effective_active_mask_published(active_mask)
        return ActiveRanksOutput(status=active_mask)

    def observe_watchdog_heartbeat(self, heartbeat: WatchdogHeartbeatOutput) -> None:
        now = time.monotonic()
        existing = self._watchdog_leases.get(heartbeat.node_rank)
        if existing is None:
            dp_ranks = tuple(sorted(set(heartbeat.ranks)))
        else:
            dp_ranks = existing[1]
        self._watchdog_leases[heartbeat.node_rank] = (now, dp_ranks)

        if self._watchdog_lease_task is None or self._watchdog_lease_task.done():
            self._watchdog_lease_task = self._create_task(
                self._watchdog_lease_sweep_loop()
            )

    async def _watchdog_lease_sweep_loop(self) -> None:
        try:
            while self._watchdog_leases:
                await asyncio.sleep(WATCHDOG_LEASE_SWEEP_INTERVAL_SEC)
                await self._sweep_expired_watchdog_leases()
        finally:
            self._watchdog_lease_task = None

    async def _sweep_expired_watchdog_leases(
        self, now: Optional[float] = None
    ) -> None:
        now = time.monotonic() if now is None else now
        expired_nodes = [
            node_rank
            for node_rank, (last_seen, _) in self._watchdog_leases.items()
            if now - last_seen >= WATCHDOG_LEASE_TIMEOUT_SEC
        ]
        if not expired_nodes:
            return

        inactive_ranks = set()
        for node_rank in expired_nodes:
            _, dp_ranks = self._watchdog_leases.pop(node_rank)
            inactive_ranks.update(dp_ranks)

        logger.warning("FT watchdog lease expired: nodes=%s dp_ranks=%s", sorted(expired_nodes), sorted(inactive_ranks))
        if not inactive_ranks:
            return

        active_ranks = self.observe_process_active_ranks(
            ProcessActiveRanksOutput(
                ranks=sorted(inactive_ranks),
                active=False,
            )
        )
        if active_ranks is not None:
            await self.send_to_scheduler(active_ranks)

    def handle_command_output(self, output: FaultToleranceCommandReqOutput) -> None:
        pending = self._pending_commands.get(output.request_id)
        if pending is None:
            logger.warning("Unknown fault tolerance command ack: rank=%s", output.rank)
            return
        if output.success:
            pending.acked.add(output.rank)
        else:
            pending.failed[output.rank] = output.message
        pending.finish_if_ready()

    def _drop_process_inactive_pause_targets(self, inactive_ranks: set[int]) -> None:
        if not inactive_ranks:
            return

        for request_id, pending in self._pending_commands.items():
            if pending.command != "pause":
                continue
            dropped_ranks = pending.target_ranks & inactive_ranks
            if not dropped_ranks:
                continue
            pending.target_ranks.difference_update(dropped_ranks)
            logger.info("FT pause targets became runtime-inactive: dropped=%s", sorted(dropped_ranks))
            pending.finish_if_ready()

    def handle_active_ranks_update_output(
        self, output: ActiveRanksUpdateReqOutput
    ) -> None:
        future = self._pending_active_rank_updates.get(output.request_id)
        if future is None:
            logger.warning("Unknown active-ranks update ack: success=%s", output.success)
            return
        if future.done():
            return
        if output.success:
            future.set_result(None)
        else:
            future.set_exception(RuntimeError(output.message))

    def handle_rank_fault(self, event: FaultToleranceRankFaultOutput) -> None:
        if self.state.strategy == "continue":
            logger.warning("FT continue observed scheduler exception on rank %s: %s", event.rank, event.message)
            return
        targets = self.state.begin_exception_pause()
        if not targets:
            logger.warning("Ignoring FT exception with no healthy pause target: rank=%s", event.rank)
            return
        self._create_task(self._pause_schedulers(targets))

    def _create_task(self, coro):
        if self.event_loop is None:
            coro.close()
            raise RuntimeError("fault tolerance manager has no bound event loop")
        task = self.event_loop.create_task(self._fatal_task_wrapper(coro))
        self.asyncio_tasks.add(task)
        task.add_done_callback(self.asyncio_tasks.discard)
        return task

    async def _fatal_task_wrapper(self, coro):
        try:
            await coro
        except Exception:
            self._failstop(f"FaultToleranceManager hit an exception: {get_exception_traceback()}")

    async def _pause_schedulers(self, targets: List[int]):
        acked = await self._send_command_collect(
            command="pause",
            target_ranks=targets,
            timeout_sec=self.server_args.fault_tolerance_timeout,
        )
        self.state.finish_pause(acked)
        self._arm_paused_failstop()

    def _arm_paused_failstop(self) -> None:
        if (
            not self.state.has_paused_rank()
            or self._paused_failstop_handle is not None
        ):
            return
        timeout_sec = float(self.server_args.fault_tolerance_pause_timeout)
        self._paused_failstop_handle = self.event_loop.call_later(
            timeout_sec, self._failstop_if_still_paused, timeout_sec
        )
        logger.info(
            "Fault tolerance paused fail-stop armed: timeout_sec=%s paused_ranks=%s",
            timeout_sec,
            sorted(self.state.paused_dp_ranks),
        )

    def _cancel_paused_failstop(self) -> None:
        if self._paused_failstop_handle is None:
            return
        self._paused_failstop_handle.cancel()
        self._paused_failstop_handle = None

    def _failstop_if_still_paused(self, timeout_sec: float) -> None:
        self._paused_failstop_handle = None
        if not self.state.has_paused_rank():
            return
        self._failstop(
            "Fault tolerance pause unattended: "
            f"timeout_sec={timeout_sec} "
            f"paused_ranks={sorted(self.state.paused_dp_ranks)}"
        )

    async def _publish_active_ranks(
        self, active_mask: List[bool], timeout_sec: int
    ) -> None:
        request_id = uuid.uuid4().hex
        future = self.event_loop.create_future()
        self._pending_active_rank_updates[request_id] = future
        try:
            await self.send_to_scheduler(
                ActiveRanksOutput(status=active_mask, request_id=request_id)
            )
            await asyncio.wait_for(future, timeout=timeout_sec)
        finally:
            self._pending_active_rank_updates.pop(request_id, None)

    async def _send_command_collect(
        self,
        *,
        command: str,
        target_ranks: List[int],
        timeout_sec: int,
    ) -> set[int]:
        target_set = set(target_ranks)
        if not target_set:
            return set()

        request_id = uuid.uuid4().hex
        pending = PendingFTCommand(
            command=command,
            target_ranks=target_set,
            future=self.event_loop.create_future(),
        )
        self._pending_commands[request_id] = pending
        req = FaultToleranceCommandReqInput(
            request_id=request_id,
            command=command,
            target_ranks=sorted(target_set),
        )
        logger.info("FT command dispatch: command=%s targets=%s", command, req.target_ranks)
        await self.send_to_scheduler(req)
        try:
            await asyncio.wait_for(pending.future, timeout=timeout_sec)
        finally:
            self._pending_commands.pop(request_id, None)

        if pending.failed:
            raise RuntimeError(
                f"fault tolerance command {command} failed: {pending.failed}"
            )
        logger.info("FT command complete: command=%s acked=%s", command, sorted(pending.acked))
        return pending.acked

    @staticmethod
    def _failstop(message: str) -> None:
        logger.error(message)
        kill_process_tree(os.getpid(), include_parent=True)
        raise RuntimeError(message)
