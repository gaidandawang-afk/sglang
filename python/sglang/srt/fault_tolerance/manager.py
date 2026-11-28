from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import time
import traceback
import uuid
from typing import Dict, List, Optional

from sglang.srt.fault_tolerance.constants import (
    FT_OPERATION_RETRY,
    FT_OPERATION_SCALE_DOWN,
    FT_OPERATION_SCALE_DOWN_PREPARE,
)
from sglang.srt.fault_tolerance.ft_state import FaultToleranceState
from sglang.srt.fault_tolerance.protocol import FaultToleranceApplyRequest
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    FaultToleranceCommandReqInput,
    FaultToleranceCommandReqOutput,
    FaultToleranceDPExitOutput,
    FaultToleranceRankFaultOutput,
    FaultToleranceWatchdogHeartbeatOutput,
    RouteUpdateAckOutput,
)
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import TypeBasedDispatcher

logger = logging.getLogger(__name__)

WATCHDOG_HEARTBEAT_TIMEOUT = 60.0

FT_REQUEST_ACCEPTED_MESSAGE = (
    "Request accepted; poll /v1/fault_tolerance/status for updates."
)


@dataclasses.dataclass
class PendingFTCommand:
    command: str
    remaining_ranks: set[int]
    future: asyncio.Future

    def acknowledge(self, rank: int) -> None:
        self.remaining_ranks.discard(rank)
        if not self.remaining_ranks and not self.future.done():
            self.future.set_result(None)


class FaultToleranceManager:
    def __init__(self, *, server_args, send_to_scheduler):
        self.send_to_scheduler = send_to_scheduler
        self.state = FaultToleranceState(
            dp_size=server_args.num_dp_ranks,
            strategy=get_exec().features.fault_tolerance_on_error_strategy,
            max_world_size=server_args.max_world_size,
            attn_replica_size=server_args.attn_tp_size * server_args.attn_cp_size,
        )
        self.event_loop = None
        self.asyncio_tasks = None
        self._pending_commands: Dict[str, PendingFTCommand] = {}
        self._pending_active_rank_updates: Dict[str, asyncio.Future] = {}
        self._recovered_dp_ranks: set[int] = set()
        self._watchdog_heartbeats: dict[int, tuple[float, list[int]]] = {}
        self._route_dp_mask = self.state.expected_dp_mask.copy()
        self._last_ft_request_id: Optional[str] = None
        self._ft_error: Optional[str] = None

    def bind_event_loop(self, loop) -> None:
        if self.event_loop is loop:
            return
        if self.event_loop is not None:
            raise RuntimeError(
                "fault tolerance manager is already bound to an event loop"
            )
        self.event_loop = loop
        self.asyncio_tasks = set()

    def init_request_dispatcher(self) -> TypeBasedDispatcher:
        return TypeBasedDispatcher(
            [
                (RouteUpdateAckOutput, self.handle_route_update_ack),
                (FaultToleranceCommandReqOutput, self.handle_command_output),
                (FaultToleranceRankFaultOutput, self.handle_rank_fault),
                (
                    FaultToleranceWatchdogHeartbeatOutput,
                    self.observe_watchdog_heartbeat,
                ),
            ]
        )

    def status(self) -> tuple[int, dict]:
        now = time.monotonic()
        mask = self.state.unreachable_dp_mask
        mask[:] = [False] * len(mask)
        for last_seen, dp_ranks in self._watchdog_heartbeats.values():
            if now - last_seen >= WATCHDOG_HEARTBEAT_TIMEOUT:
                for rank in dp_ranks:
                    mask[rank] = True
        body = self.state.status_response()
        if self.state.in_flight_request_id is not None:
            body["last_ft_request_id"] = self.state.in_flight_request_id
            body["ft_state"] = "recovering"
        elif self._last_ft_request_id is not None:
            body["last_ft_request_id"] = self._last_ft_request_id
            if self._ft_error is not None:
                body["ft_state"] = "failed"
                body["ft_error"] = self._ft_error
        return 200, body

    def add_elastic_workers(self, slot_offset: int, slot_count: int) -> None:
        st = self.state
        end = slot_offset + slot_count
        for rank in range(max(slot_offset, st.dp_size), end):
            st.expected_dp_mask[rank] = True
            st.runtime_active_dp_mask[rank] = True
            self._route_dp_mask[rank] = True
        st.dp_size = max(st.dp_size, end)

    def submit(self, request: FaultToleranceApplyRequest) -> tuple[int, dict]:
        if request.instruction == FT_OPERATION_SCALE_DOWN:
            if self.state.strategy != "pause":
                return 400, {"message": "scale_down requires the FT pause strategy."}
            if any(
                rank >= self.state.dp_size for rank in request.params.removed_dp_ranks
            ):
                return 400, {
                    "message": "'removed_dp_ranks' contains a rank out of range."
                }
        self._create_task(self._run_submitted_apply(request))
        return 202, {
            "message": FT_REQUEST_ACCEPTED_MESSAGE,
            "request_id": request.request_id,
        }

    async def _run_submitted_apply(self, request: FaultToleranceApplyRequest) -> None:
        request_id = request.request_id
        if self.state.in_flight_request_id is not None:
            logger.warning(
                "FT recovery request %s rejected: another recovery %s is in progress",
                request_id,
                self.state.in_flight_request_id,
            )
            return
        self.state.in_flight_request_id = request_id
        timeout = get_exec().features.fault_tolerance_timeout

        if not self.state.has_unresolved_expected_dp_fault():
            self._finish_submitted_apply(
                request.request_id,
                f"{request.instruction}_requires_unresolved_expected_dp_fault",
            )
            return

        if request.instruction == FT_OPERATION_RETRY:
            error = await self._apply_retry(timeout)
        elif request.instruction == FT_OPERATION_SCALE_DOWN:
            error = await self._apply_scale_down(
                request.params.removed_dp_ranks, timeout
            )

        self._finish_submitted_apply(request.request_id, error)
        update = self._update_route_after_observation()
        if update is not None:
            await self.send_to_scheduler(update)

    def _finish_submitted_apply(self, request_id: str, error: Optional[str]) -> None:
        self._last_ft_request_id = request_id
        self._ft_error = error
        self.state.in_flight_request_id = None

    async def _apply_retry(self, timeout: int) -> Optional[str]:
        st = self.state
        if st.exited_dp_ranks.intersection(st.expected_dp_ranks()):
            return "retry_requires_all_expected_processes_alive"

        await self._send_command_collect(
            command=FT_OPERATION_RETRY,
            target_ranks=st.expected_dp_ranks(),
            timeout_sec=timeout,
        )
        await self._publish_route_dp_mask(st.expected_dp_mask, timeout)
        return None

    async def _apply_scale_down(self, ranks: List[int], timeout: int) -> Optional[str]:
        st = self.state
        requested = set(ranks)
        if not requested:
            return "scale_down_requires_ranks"
        if any(not st.expected_dp_mask[rank] for rank in requested):
            return "scale_down_requires_expected_ranks"

        # DP ranks that remain active after the scale-down.
        candidate_dp_mask = [
            expected and rank not in requested
            for rank, expected in enumerate(st.expected_dp_mask)
        ]
        target_ranks = [rank for rank, active in enumerate(candidate_dp_mask) if active]
        if not target_ranks:
            return "cannot_scale_down_all_expected_ranks"
        if st.exited_dp_ranks.intersection(target_ranks):
            return "scale_down_requires_remaining_processes_alive"
        if not st.unhealthy_dp_ranks.issuperset(target_ranks):
            return "scale_down_requires_remaining_dp_paused"

        # The caller stops the removed DP's old processes before applying.
        request_id = uuid.uuid4().hex
        try:
            await self._send_command_collect(
                command=FT_OPERATION_SCALE_DOWN_PREPARE,
                target_ranks=target_ranks,
                timeout_sec=timeout,
                active_global_rank_mask=st.expand_dp_mask_to_global_rank_mask(
                    candidate_dp_mask
                ),
                request_id=request_id,
            )
        except TimeoutError:
            return "scale_down_prepare_timeout"
        if st.exited_dp_ranks.intersection(target_ranks):
            return "scale_down_requires_remaining_processes_alive"

        await self._send_command_collect(
            command=FT_OPERATION_SCALE_DOWN,
            target_ranks=target_ranks,
            timeout_sec=timeout,
            request_id=request_id,
        )
        await self._publish_route_dp_mask(candidate_dp_mask, timeout)
        for rank in requested:
            st.expected_dp_mask[rank] = False
        st.unhealthy_dp_ranks.difference_update(requested)
        return None

    def admission_error(self, routed_dp_rank: Optional[int]) -> Optional[str]:
        if routed_dp_rank is not None and not self._route_dp_mask[routed_dp_rank]:
            return f"routed_dp_rank={routed_dp_rank} is not active"
        if self.state.is_global_admission_blocked(self._route_dp_mask):
            return "fault_tolerance_paused"
        return None

    def _route_dp_update(
        self, route_dp_mask: List[bool]
    ) -> Optional[ActiveRanksOutput]:
        # Cache a changed route mask and return the update for TokenizerManager.
        if route_dp_mask == self._route_dp_mask:
            return None
        self._route_dp_mask = list(route_dp_mask)
        return ActiveRanksOutput(status=route_dp_mask)

    def _update_route_after_observation(self) -> Optional[ActiveRanksOutput]:
        st = self.state
        if st.in_flight_request_id is None:
            for dp_rank in self._recovered_dp_ranks:
                if st.runtime_active_dp_mask[dp_rank]:
                    st.exited_dp_ranks.discard(dp_rank)
                    st.expected_dp_mask[dp_rank] = True
            self._recovered_dp_ranks.clear()

        return self._route_dp_update(
            [
                expected
                and dp_rank not in st.exited_dp_ranks
                and (st.strategy != "continue" or st.runtime_active_dp_mask[dp_rank])
                for dp_rank, expected in enumerate(st.expected_dp_mask)
            ]
        )

    def observe_active_ranks(
        self, ranks: ActiveRanksOutput
    ) -> Optional[ActiveRanksOutput]:
        st = self.state
        st.runtime_active_dp_mask = list(ranks.status)
        for rank in ranks.recovered_ep_ranks or ():
            dp_rank = rank // st.global_ranks_per_dp
            if dp_rank < st.dp_size and ranks.status[dp_rank]:
                self._recovered_dp_ranks.add(dp_rank)
        return self._update_route_after_observation()

    def observe_dp_exit(
        self, event: FaultToleranceDPExitOutput
    ) -> Optional[ActiveRanksOutput]:
        self.state.exited_dp_ranks.add(event.dp_rank)
        self._recovered_dp_ranks.discard(event.dp_rank)
        return self._update_route_after_observation()

    def handle_command_output(self, output: FaultToleranceCommandReqOutput) -> None:
        pending = self._pending_commands.get(output.request_id)
        if (
            pending is None
            or pending.command != output.command
            or output.rank not in pending.remaining_ranks
        ):
            return
        if output.command != FT_OPERATION_SCALE_DOWN_PREPARE:
            # The scheduler has resumed. A later fault must keep this DP paused.
            self.state.unhealthy_dp_ranks.discard(output.rank)
        pending.acknowledge(output.rank)

    def handle_route_update_ack(self, output: RouteUpdateAckOutput) -> None:
        future = self._pending_active_rank_updates.get(output.request_id)
        if future is None or future.done():
            return
        future.set_result(None)

    def handle_rank_fault(self, event: FaultToleranceRankFaultOutput) -> None:
        self.state.observe_rank_fault(event.rank)
        logger.warning(
            "FT observed scheduler exception on rank %s: %s", event.rank, event.message
        )

    def observe_watchdog_heartbeat(
        self, event: FaultToleranceWatchdogHeartbeatOutput
    ) -> None:
        self._watchdog_heartbeats[event.first_global_rank] = (
            time.monotonic(),
            event.dp_ranks,
        )

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
            self._failstop(
                f"FaultToleranceManager hit an exception: {traceback.format_exc()}"
            )

    async def _publish_route_dp_mask(
        self, route_dp_mask: List[bool], timeout_sec: int
    ) -> None:
        request_id = uuid.uuid4().hex
        future = self.event_loop.create_future()
        self._pending_active_rank_updates[request_id] = future
        try:
            await self.send_to_scheduler(
                ActiveRanksOutput(status=list(route_dp_mask), request_id=request_id)
            )
            await asyncio.wait_for(future, timeout=timeout_sec)
            self._route_dp_mask = list(route_dp_mask)
        finally:
            self._pending_active_rank_updates.pop(request_id, None)

    async def _send_command_collect(
        self,
        *,
        command: str,
        target_ranks: List[int],
        timeout_sec: int,
        active_global_rank_mask: Optional[List[bool]] = None,
        request_id: Optional[str] = None,
    ) -> None:
        targets = set(target_ranks)
        if not targets:
            return
        request_id = request_id or uuid.uuid4().hex
        pending = PendingFTCommand(
            command=command,
            remaining_ranks=targets,
            future=self.event_loop.create_future(),
        )
        self._pending_commands[request_id] = pending
        req = FaultToleranceCommandReqInput(
            request_id=request_id,
            command=command,
            target_ranks=sorted(targets),
            # This command is consumed by per-scheduler Elastic EP control, so
            # its active_mask remains a physical scheduler/global-rank mask.
            active_mask=active_global_rank_mask,
        )
        try:
            await self.send_to_scheduler(req)
            await asyncio.wait_for(pending.future, timeout=timeout_sec)
        finally:
            self._pending_commands.pop(request_id, None)

    @staticmethod
    def _failstop(message: str) -> None:
        logger.error(message)
        kill_process_tree(os.getpid(), include_parent=True)
        raise RuntimeError(message)
