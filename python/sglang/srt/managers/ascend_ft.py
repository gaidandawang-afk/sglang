"""Coordinate fixed-world Ascend FT from the tokenizer process."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from uuid import uuid4

import zmq

from sglang.srt.fault_tolerance.ft_state import FaultToleranceState
from sglang.srt.fault_tolerance.protocol import FaultToleranceApplyRequest, PauseRequest
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    AscendFTAckOutput,
    AscendFTCommandReq,
    FaultToleranceDPCShutdownReqInput,
    ProcessActiveRanksOutput,
    RouteUpdateAckOutput,
    WatchdogHeartbeatOutput,
    async_sock_send,
)
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils.network import get_zmq_socket
from sglang.srt.utils.process import kill_process_tree

logger = logging.getLogger(__name__)

WATCHDOG_LEASE_SWEEP_INTERVAL_SEC = 1
WATCHDOG_LEASE_TIMEOUT_SEC = 60
FT_REQUEST_ACCEPTED_MESSAGE = (
    "Request accepted; poll /v1/fault_tolerance/status for updates."
)


@dataclass
class PendingCommand:
    rid: str
    op: str
    remaining: set[int]
    future: asyncio.Future


class AscendFTManager:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.state = FaultToleranceState(get_parallel().num_dp_ranks)
        self.min_survivors = self.state.dp_size
        self._route_dp_mask = [True] * self.state.dp_size
        self._last_ft_request_id: str | None = None
        self._ft_error: str | None = None
        self._operator_paused = False
        self.task: asyncio.Task | None = None
        self._command: PendingCommand | None = None
        self._route_update: tuple[str, asyncio.Future] | None = None
        self._shutdown_waiter: tuple[set[int], asyncio.Future] | None = None
        self._watchdog_endpoints: dict[int, str] = {}
        self._watchdog_sockets: dict[int, zmq.Socket] = {}
        self._watchdog_leases: dict[int, tuple[float, tuple[int, ...]]] = {}
        self._watchdog_lease_task: asyncio.Task | None = None

    @property
    def active_ranks(self) -> tuple[int, ...]:
        return self.state.expected_dp_ranks()

    def dispatchers(self):
        return [
            (AscendFTAckOutput, self.handle_ack),
            (ProcessActiveRanksOutput, self.observe_process_active_ranks),
            (RouteUpdateAckOutput, self.handle_route_update_ack),
            (WatchdogHeartbeatOutput, self.observe_watchdog_heartbeat),
        ]

    def status(self) -> dict:
        body = self.state.status_response()
        if self._last_ft_request_id is not None:
            body["last_ft_request_id"] = self._last_ft_request_id
        if self._ft_error is not None:
            body["ft_error"] = self._ft_error
        return body

    def submit(
        self, request: FaultToleranceApplyRequest | PauseRequest
    ) -> tuple[int, dict]:
        if request.instruction == "scale_down" and any(
            rank >= self.state.dp_size for rank in request.params.removed_dp_ranks
        ):
            return 400, {"message": "'removed_dp_ranks' contains a rank out of range."}
        if self.state.ft_operation_in_progress:
            return 409, {"message": "ft_operation_in_progress"}

        self.tokenizer.auto_create_handle_loop()
        self.state.ft_operation_in_progress = True
        self.tokenizer.is_pause = True
        self.task = asyncio.create_task(self._run_submitted_apply(request))
        return 202, {
            "message": FT_REQUEST_ACCEPTED_MESSAGE,
            "request_id": request.request_id,
        }

    def admission_error(self, routed_dp_rank: int | None) -> str | None:
        if routed_dp_rank is not None and 0 <= routed_dp_rank < self.state.dp_size:
            if (
                not self._route_dp_mask[routed_dp_rank]
                or not self.state.process_alive_global_rank_mask[routed_dp_rank]
            ):
                return f"routed_dp_rank={routed_dp_rank} is not active"
        if self.state.ft_operation_in_progress or self.state.cluster_paused:
            return "fault_tolerance_paused"
        return None

    async def _run_submitted_apply(
        self, request: FaultToleranceApplyRequest | PauseRequest
    ) -> None:
        try:
            timeout = get_exec().features.fault_tolerance_timeout
            error = await self._apply(request, timeout)
            async with self.tokenizer.is_pause_cond:
                self._last_ft_request_id = request.request_id
                self._ft_error = error
                self.state.ft_operation_in_progress = False
                self.tokenizer.is_pause = self.state.cluster_paused
                if not self.tokenizer.is_pause:
                    self.tokenizer.is_pause_cond.notify_all()
        except Exception:
            self._failstop("Ascend FT did not complete its control transaction.")

    async def _apply(
        self, request: FaultToleranceApplyRequest | PauseRequest, timeout: int
    ) -> str | None:
        st = self.state
        instruction = request.instruction
        if instruction == "pause":
            if not all(
                st.process_alive_global_rank_mask[rank] for rank in self.active_ranks
            ):
                return "pause_requires_all_expected_processes_alive"
            await self._send_command("pause", self.active_ranks, timeout)
            self._operator_paused = st.cluster_paused = True
            return None
        if not (self._operator_paused or st.has_unresolved_expected_dp_fault()):
            return f"{instruction}_requires_unresolved_expected_dp_fault"

        removed: set[int] = set()
        if instruction == "scale_down":
            removed = set(request.params.removed_dp_ranks)
            if not removed:
                return "scale_down_requires_ranks"
            if not removed <= set(self.active_ranks):
                return "scale_down_requires_expected_ranks"
            if len(removed) == len(self.active_ranks):
                return "cannot_scale_down_all_expected_ranks"

        survivors = tuple(rank for rank in self.active_ranks if rank not in removed)
        if not all(st.process_alive_global_rank_mask[rank] for rank in survivors):
            return (
                "retry_requires_all_expected_processes_alive"
                if instruction == "retry"
                else "scale_down_requires_live_survivors"
            )
        if removed and len(survivors) < self.min_survivors:
            return "scale_down_requires_surviving_expert_capacity"

        if removed:
            await self._shutdown_dp_processes(removed, timeout)
        await self._send_command("pause", survivors, timeout)
        reset_collective = any(
            not st.runtime_active_dp_mask[rank] for rank in survivors
        )
        for rank in survivors:
            st.runtime_active_dp_mask[rank] = False
        await self._send_command(
            "recover",
            survivors,
            timeout,
            membership_changed=bool(removed),
            reset_collective=reset_collective,
        )
        self._require_survivors_ready(survivors)
        route = [rank in survivors for rank in range(st.dp_size)]
        await self._publish_route_dp_mask(route, timeout)
        st.expected_dp_mask = route
        await self.tokenizer.ascend_ft_replay(sorted(removed))
        await self._send_command("resume", survivors, timeout)
        self._require_survivors_ready(survivors)
        st.unhealthy_dp_ranks.clear()
        st.pending_recovery_global_ranks.intersection_update(survivors)
        self._operator_paused = st.cluster_paused = False
        return None

    def _require_survivors_ready(self, survivors: tuple[int, ...]) -> None:
        if not all(self.state.serving_ready(rank) for rank in survivors):
            raise RuntimeError(
                "A surviving DP lost process or runtime readiness during recovery."
            )

    async def _send_command(
        self,
        op,
        ranks: tuple[int, ...],
        timeout: int,
        *,
        membership_changed: bool = False,
        reset_collective: bool = False,
    ) -> None:
        rid = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._command = PendingCommand(rid, op, set(ranks), future)
        try:
            await self.tokenizer._async_dispatch_to_scheduler(
                AscendFTCommandReq(
                    rid=rid,
                    op=op,
                    target_ranks=list(ranks),
                    membership_changed=membership_changed,
                    reset_collective=reset_collective,
                )
            )
            await asyncio.wait_for(future, timeout)
        finally:
            self._command = None

    def handle_ack(self, ack: AscendFTAckOutput) -> None:
        if ack.op == "fault":
            self.state.observe_rank_fault(ack.rank)
            if self.state.cluster_paused:
                self.tokenizer.is_pause = True
            logger.warning(
                "Ascend FT scheduler fault on DP %s: %s", ack.rank, ack.error
            )
            return
        cmd = self._command
        if (
            cmd is None
            or cmd.future.done()
            or (ack.rid, ack.op) != (cmd.rid, cmd.op)
            or ack.rank not in cmd.remaining
        ):
            return
        if ack.error is not None:
            cmd.future.set_exception(RuntimeError(f"DP rank {ack.rank}: {ack.error}"))
            return
        if ack.op == "recover":
            self.state.observe_recovery_ack(ack.rank)
        cmd.remaining.remove(ack.rank)
        if not cmd.remaining:
            cmd.future.set_result(None)

    async def _publish_route_dp_mask(self, route: list[bool], timeout: int) -> None:
        rid = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._route_update = (rid, future)
        try:
            await self.tokenizer._async_dispatch_to_scheduler(
                ActiveRanksOutput(
                    status=route,
                    request_id=rid,
                )
            )
            await asyncio.wait_for(future, timeout)
            self._route_dp_mask = route.copy()
        finally:
            self._route_update = None

    def handle_route_update_ack(self, ack: RouteUpdateAckOutput) -> None:
        if self._route_update is None:
            return
        rid, future = self._route_update
        if rid == ack.request_id and not future.done():
            future.set_result(None)

    def observe_process_active_ranks(self, msg: ProcessActiveRanksOutput) -> None:
        self.state.observe_process_active_ranks(msg.ranks, active=msg.active)
        if self.state.cluster_paused:
            self.tokenizer.is_pause = True
        if self._shutdown_waiter is not None:
            ranks, future = self._shutdown_waiter
            if not future.done() and not any(
                self.state.process_alive_global_rank_mask[rank] for rank in ranks
            ):
                future.set_result(None)

    def observe_watchdog_heartbeat(self, heartbeat: WatchdogHeartbeatOutput) -> None:
        node = heartbeat.node_rank
        self._watchdog_leases[node] = (time.monotonic(), tuple(heartbeat.ranks))
        if self._watchdog_lease_task is None or self._watchdog_lease_task.done():
            self._watchdog_lease_task = asyncio.create_task(
                self._watchdog_lease_sweep_loop()
            )
        endpoint = heartbeat.control_endpoint
        if endpoint and self._watchdog_endpoints.get(node) != endpoint:
            new_socket = get_zmq_socket(
                self.tokenizer._zmq_context, zmq.PUSH, endpoint, False
            )
            old_socket = self._watchdog_sockets.get(node)
            self._watchdog_sockets[node] = new_socket
            self._watchdog_endpoints[node] = endpoint
            if old_socket is not None:
                old_socket.close(linger=0)

    async def _watchdog_lease_sweep_loop(self) -> None:
        try:
            while self._watchdog_leases:
                await asyncio.sleep(WATCHDOG_LEASE_SWEEP_INTERVAL_SEC)
                self._sweep_expired_watchdog_leases(time.monotonic())
        except Exception:
            self._failstop("Ascend FT could not update process liveness.")

    def _sweep_expired_watchdog_leases(self, now: float) -> None:
        for node, (last_seen, ranks) in list(self._watchdog_leases.items()):
            if now - last_seen >= WATCHDOG_LEASE_TIMEOUT_SEC:
                del self._watchdog_leases[node]
                logger.warning(
                    "Ascend FT watchdog lease expired: node=%s ranks=%s", node, ranks
                )
                self.observe_process_active_ranks(
                    ProcessActiveRanksOutput(ranks=list(ranks), active=False)
                )

    async def _shutdown_dp_processes(self, ranks: set[int], timeout: int) -> None:
        pending = {
            rank for rank in ranks if self.state.process_alive_global_rank_mask[rank]
        }
        if not pending:
            return
        future = asyncio.get_running_loop().create_future()
        self._shutdown_waiter = (ranks, future)
        try:
            request = FaultToleranceDPCShutdownReqInput(target_dp_ranks=sorted(ranks))
            sockets = [
                self._watchdog_sockets[node]
                for node, (_, owned) in self._watchdog_leases.items()
                if pending.intersection(owned)
            ]
            async with asyncio.timeout(timeout):
                for socket in sockets:
                    await async_sock_send(socket, request)
                await future
        finally:
            self._shutdown_waiter = None

    @staticmethod
    def _failstop(message: str) -> None:
        logger.exception(message)
        kill_process_tree(os.getpid(), include_parent=True)
        raise RuntimeError(message)
