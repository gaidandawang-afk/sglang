from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import sys
import uuid
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple

import fastapi

from sglang.srt.environ import envs
from sglang.srt.fault_tolerance.controller import (
    FaultToleranceState,
    RankState,
    ft_error_status,
    ft_failure,
)
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    ActiveRanksUpdateReqOutput,
    FaultToleranceCommandReqInput,
    FaultToleranceCommandReqOutput,
    FaultToleranceRankFaultOutput,
)
from sglang.srt.utils import kill_process_tree
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PendingFTCommand:
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
        self._parked_scheduler_ranks: List[int] = []
        self._scheduler_park_lock: Optional[asyncio.Lock] = None
        self._last_dispatched_active_mask: Optional[List[bool]] = [
            True
        ] * server_args.dp_size

    def bind_event_loop(self, loop) -> None:
        if self.event_loop is loop:
            return
        if self.event_loop is not None:
            raise RuntimeError(
                "fault tolerance manager is already bound to an event loop"
            )
        self.event_loop = loop
        self.asyncio_tasks = set()
        self._scheduler_park_lock = asyncio.Lock()

    def status(self) -> tuple[int, dict]:
        return 200, self.state.status_response()

    async def apply(self, obj: Dict[str, Any]) -> tuple[int, dict]:
        instruction = obj.get("fault_tolerance_instruction")
        params = obj.get("fault_tolerance_params") or {}
        timeout = obj.get(
            "fault_tolerance_timeout", self.server_args.fault_tolerance_timeout
        )
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            return 400, ft_failure("invalid_fault_tolerance_timeout")
        if timeout <= 0:
            return 400, ft_failure("invalid_fault_tolerance_timeout")

        ranks = None
        if instruction == "scale_down":
            ranks = params.get("ranks")
            if not isinstance(ranks, list):
                return 400, ft_failure("scale_down_requires_non_empty_ranks")
            try:
                ranks = [int(rank) for rank in ranks]
            except (TypeError, ValueError):
                return 400, ft_failure("unknown_rank")
        elif instruction == "retry":
            if params:
                return 400, ft_failure("retry_does_not_accept_params")
            ranks = None

        error = self.state.validate_apply(instruction, ranks)
        if error:
            return ft_error_status(error), ft_failure(error)

        live_scale_down_targets = []
        if instruction == "scale_down":
            live_scale_down_targets = [
                rank
                for rank in ranks or []
                if self.state.rank_states[rank] != RankState.DEAD
            ]
        active_mask, resume_targets, pending_scale_down_ranks = (
            self.state.begin_recover(instruction, ranks)
        )
        shutdown_live_targets = (
            instruction == "scale_down"
            and envs.SGLANG_FT_SCALE_DOWN_SHUTDOWN_LIVE_RANK.get()
        )
        if shutdown_live_targets:
            resume_targets = sorted(set(resume_targets) - set(live_scale_down_targets))
        else:
            resume_targets = sorted(set(resume_targets + live_scale_down_targets))
        logger.info(
            "Fault tolerance apply plan: instruction=%s active_mask=%s "
            "resume_targets=%s pending_scale_down=%s shutdown_live_targets=%s "
            "live_scale_down_targets=%s",
            instruction,
            active_mask,
            resume_targets,
            pending_scale_down_ranks,
            shutdown_live_targets,
            live_scale_down_targets,
        )
        try:
            await self._apply_active_mask(
                active_mask,
                timeout,
                update_routing=False,
            )
            resume_targets = sorted(
                set(resume_targets) | set(self._parked_scheduler_ranks)
            )
            if shutdown_live_targets:
                resume_targets = sorted(
                    set(resume_targets) - set(live_scale_down_targets)
                )
            await self._send_command_collect(
                command="resume",
                target_ranks=resume_targets,
                timeout_sec=timeout,
            )
            self._parked_scheduler_ranks.clear()
            await self._publish_active_ranks(active_mask, timeout)
            if shutdown_live_targets and live_scale_down_targets:
                await self._send_command_collect(
                    command="shutdown",
                    target_ranks=live_scale_down_targets,
                    timeout_sec=timeout,
                )
        except Exception as exc:
            logger.exception("Fault tolerance apply failed; exiting: %s", exc)
            os._exit(1)

        response = self.state.commit_recover(pending_scale_down_ranks)
        logger.info(
            "Fault tolerance apply committed: instruction=%s pending_scale_down=%s",
            instruction,
            pending_scale_down_ranks,
        )
        return 200, response

    def validate_routed_rank(self, rank: int) -> None:
        if not self.state.is_rank_healthy(rank):
            raise ValueError(f"routed_dp_rank={rank} is not healthy")

    def should_reject_admission(self) -> bool:
        return self.state.should_reject_admission()

    def is_rank_healthy(self, rank: int) -> bool:
        return self.state.is_rank_healthy(rank)

    async def before_request(self) -> None:
        await self._resume_parked_schedulers_before_request()

    def observe_active_ranks(self, ranks: ActiveRanksOutput) -> None:
        targets = self.state.record_inactive_mask(ranks.status)
        if targets:
            self._create_task(self._pause_after_inactive(targets))

    def handle_command_output(self, output: FaultToleranceCommandReqOutput) -> None:
        pending = self._pending_commands.get(output.request_id)
        if pending is None:
            logger.warning(
                "Unknown fault tolerance command ack: request_id=%s rank=%s",
                output.request_id,
                output.rank,
            )
            return
        if output.success:
            pending.acked.add(output.rank)
        else:
            pending.failed[output.rank] = output.message
        pending.finish_if_ready()

    def handle_active_ranks_update_output(
        self, output: ActiveRanksUpdateReqOutput
    ) -> None:
        future = self._pending_active_rank_updates.get(output.request_id)
        if future is None:
            logger.warning(
                "Unknown active-ranks update ack: request_id=%s success=%s",
                output.request_id,
                output.success,
            )
            return
        if future.done():
            return
        if output.success:
            future.set_result(None)
        else:
            future.set_exception(RuntimeError(output.message))

    def handle_rank_fault(self, event: FaultToleranceRankFaultOutput) -> None:
        if event.fault_type == "exception":
            if self.state.strategy == "continue":
                logger.warning(
                    "FT continue observed scheduler exception on rank %s: %s",
                    event.rank,
                    event.message,
                )
                return
            self._create_task(self._handle_exception_pause(event))
            return
        if event.fault_type == "kill":
            self._create_task(self._handle_kill(event))

    def handle_process_exit(self, rank: int) -> bool:
        if self.event_loop is None:
            return False

        event = FaultToleranceRankFaultOutput(rank=rank, fault_type="kill")

        def schedule():
            self._create_task(self._handle_kill(event))

        self.event_loop.call_soon_threadsafe(schedule)
        return True

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
            traceback = get_exception_traceback()
            logger.error("FaultToleranceManager hit an exception: %s", traceback)
            kill_process_tree(os.getpid(), include_parent=True)
            sys.exit(1)

    async def _handle_exception_pause(self, event: FaultToleranceRankFaultOutput):
        if self.state.ft_operation_in_progress:
            logger.warning(
                "Ignoring duplicate FT exception while another operation is active: "
                "rank=%s message=%s",
                event.rank,
                event.message,
            )
            return

        targets = self.state.begin_exception_pause()
        await self._pause_schedulers(targets)
        if RankState.DEAD in self.state.rank_states:
            await self._apply_current_active_mask()

    async def _handle_kill(self, event: FaultToleranceRankFaultOutput):
        targets = self.state.record_kill(event.rank)
        if targets:
            await self._pause_schedulers(targets)
            await self._apply_current_active_mask()
        else:
            self.send_to_scheduler.send_pyobj(
                ActiveRanksOutput(status=self.state.active_mask())
            )

    async def _pause_after_inactive(self, targets: List[int]):
        await self._pause_schedulers(targets)
        await self._apply_current_active_mask()

    async def _pause_schedulers(self, targets: List[int]):
        acked, timed_out = await self._send_command_collect(
            command="pause",
            target_ranks=targets,
            timeout_sec=self.server_args.fault_tolerance_timeout,
            tolerate_timeout=True,
        )
        self.state.finish_pause_collection(acked, timed_out)

    async def _apply_current_active_mask(self):
        await self._apply_active_mask(
            self.state.active_mask(),
            self.server_args.fault_tolerance_timeout,
        )

    async def _apply_active_mask(
        self,
        active_mask: List[bool],
        timeout: int,
        update_routing: bool = True,
    ):
        if not any(active_mask):
            raise RuntimeError("fault tolerance active mask has no live rank")
        previous_mask = self._last_dispatched_active_mask
        if previous_mask is None or list(active_mask) != previous_mask:
            await self._park_schedulers_for_topology_change(
                target_ranks=self.state.live_ranks(),
            )
        await self._send_command_collect(
            command="apply_active_mask",
            target_ranks=self.state.live_ranks(),
            timeout_sec=timeout,
            active_mask=active_mask,
        )
        self._last_dispatched_active_mask = list(active_mask)
        if update_routing:
            await self._publish_active_ranks(active_mask, timeout)

    async def _publish_active_ranks(
        self, active_mask: List[bool], timeout_sec: int
    ) -> None:
        if self.server_args.dp_size <= 1:
            self.send_to_scheduler.send_pyobj(ActiveRanksOutput(status=active_mask))
            return

        request_id = uuid.uuid4().hex
        future = self.event_loop.create_future()
        self._pending_active_rank_updates[request_id] = future
        try:
            await self.send_to_scheduler.send_pyobj(
                ActiveRanksOutput(status=active_mask, request_id=request_id)
            )
            await asyncio.wait_for(future, timeout=timeout_sec)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"active-ranks routing update timed out: request_id={request_id}"
            ) from exc
        finally:
            self._pending_active_rank_updates.pop(request_id, None)

    async def _send_command_collect(
        self,
        *,
        command: str,
        target_ranks: List[int],
        timeout_sec: int,
        active_mask: Optional[List[bool]] = None,
        tolerate_timeout: bool = False,
    ) -> Tuple[set[int], set[int]]:
        target_set = set(target_ranks)
        if not target_set:
            return set(), set()

        request_id = uuid.uuid4().hex
        pending = PendingFTCommand(
            target_ranks=target_set,
            future=self.event_loop.create_future(),
        )
        self._pending_commands[request_id] = pending
        req = FaultToleranceCommandReqInput(
            request_id=request_id,
            command=command,
            target_ranks=sorted(target_set),
            active_mask=active_mask,
        )
        logger.info(
            "FT command dispatch: id=%s command=%s targets=%s",
            request_id,
            command,
            req.target_ranks,
        )
        await self.send_to_scheduler.send_pyobj(req)
        try:
            await asyncio.wait_for(pending.future, timeout=timeout_sec)
        except asyncio.TimeoutError:
            logger.warning(
                "Fault tolerance command timeout: id=%s command=%s acked=%s "
                "pending=%s tolerate_timeout=%s",
                request_id,
                command,
                sorted(pending.acked),
                sorted(target_set - pending.acked),
                tolerate_timeout,
            )
            if not tolerate_timeout:
                raise
        finally:
            self._pending_commands.pop(request_id, None)

        if pending.failed:
            raise RuntimeError(
                f"fault tolerance command {command} failed: {pending.failed}"
            )
        timed_out = target_set - pending.acked
        logger.info(
            "FT command complete: id=%s command=%s timed_out=%s",
            request_id,
            command,
            sorted(timed_out),
        )
        return pending.acked, timed_out

    def _should_park_schedulers(self) -> bool:
        return self.server_args.dp_size > 1

    async def _resume_parked_schedulers_before_request(self) -> None:
        if not self._should_park_schedulers():
            return
        async with self._scheduler_park_lock:
            if not self._parked_scheduler_ranks:
                return
            healthy_rank_set = set(self.state.healthy_ranks())
            target_ranks = [
                rank
                for rank in self._parked_scheduler_ranks
                if rank in healthy_rank_set
            ]
            if not target_ranks:
                self._parked_scheduler_ranks.clear()
                return
            await self._send_command_collect(
                command="resume",
                target_ranks=target_ranks,
                timeout_sec=self.server_args.fault_tolerance_timeout,
            )
            self._parked_scheduler_ranks.clear()

    async def _park_schedulers_for_topology_change(
        self,
        *,
        target_ranks: List[int],
        is_stream: bool = False,
    ) -> None:
        if not self._should_park_schedulers():
            return
        async with self._scheduler_park_lock:
            if self._parked_scheduler_ranks:
                return
            if not target_ranks:
                return
            try:
                await self._send_command_collect(
                    command="park_idle",
                    target_ranks=target_ranks,
                    timeout_sec=self.server_args.fault_tolerance_timeout,
                )
                self._parked_scheduler_ranks = target_ranks
            except Exception as exc:
                logger.exception("Failed to park FT schedulers")
                if not is_stream:
                    raise fastapi.HTTPException(
                        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                        detail=f"failed to park fault-tolerance schedulers: {exc}",
                    ) from exc
