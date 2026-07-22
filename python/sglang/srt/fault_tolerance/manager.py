from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

from sglang.srt.fault_tolerance.controller import (
    FaultToleranceState,
    ft_error_status,
    ft_failure,
)
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    ActiveRanksUpdateReqOutput,
    FaultToleranceCommandReqInput,
    FaultToleranceCommandReqOutput,
    FaultToleranceRankFaultOutput,
    ProcessActiveRanksOutput,
)
from sglang.srt.utils import kill_process_tree
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


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
        if instruction in ("scale_down", "recover"):
            ranks = params.get("ranks")
            if not isinstance(ranks, list):
                return 400, ft_failure(f"{instruction}_requires_non_empty_ranks")
            try:
                ranks = [int(rank) for rank in ranks]
            except (TypeError, ValueError):
                return 400, ft_failure("unknown_rank")
            if instruction == "scale_down" and "shutdown" in params:
                return 400, ft_failure("scale_down_does_not_accept_shutdown")
        elif instruction == "retry":
            if params:
                return 400, ft_failure("retry_does_not_accept_params")
            ranks = None

        error = self.state.validate_apply(instruction, ranks)
        if error:
            return ft_error_status(error), ft_failure(error)

        resume_targets = self.state.begin_recover(instruction, ranks)
        logger.info(
            "Fault tolerance apply plan: instruction=%s active_mask=%s "
            "resume_targets=%s ranks=%s",
            instruction,
            self.state.effective_active_mask(),
            resume_targets,
            ranks,
        )
        try:
            pending_route = self.state.pending_effective_active_update()
            if pending_route is not None:
                await self._publish_active_ranks(pending_route, timeout)
                self.state.mark_effective_active_published(pending_route)
        except Exception as exc:
            logger.exception("Fault tolerance apply failed: %s", exc)
            response = self.state.commit_recover(resumed_ranks=set())
            response.update(
                {
                    "success": False,
                    "message": f"fault_tolerance_apply_failed: {exc}",
                }
            )
            return 503, response

        acked = set()
        if instruction != "recover":
            resume_targets = set(resume_targets)
            try:
                acked = await self._send_command_collect(
                    command="resume",
                    target_ranks=sorted(resume_targets),
                    timeout_sec=timeout,
                )
            except Exception as exc:
                self._failstop(
                    f"fault tolerance resume failed for ranks "
                    f"{sorted(resume_targets)}: {exc}"
                )

        response = self.state.commit_recover(
            resumed_ranks=acked,
            isolated_ranks=(ranks if instruction == "scale_down" else None),
        )
        logger.info(
            "Fault tolerance apply committed: instruction=%s ranks=%s",
            instruction,
            ranks,
        )
        return 200, response

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
        active_mask = self.state.take_effective_active_update()
        if active_mask is None:
            return None
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
        active_mask = self.state.take_effective_active_update()
        if active_mask is None:
            return None
        return ActiveRanksOutput(status=active_mask)

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
            logger.info(
                "FT pause targets became runtime-inactive: id=%s "
                "dropped=%s remaining=%s",
                request_id,
                sorted(dropped_ranks),
                sorted(pending.target_ranks),
            )
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
        if self.state.strategy == "continue":
            logger.warning(
                "FT continue observed scheduler exception on rank %s: %s",
                event.rank,
                event.message,
            )
            return
        self._create_task(self._handle_exception_pause(event))

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
        targets = self.state.begin_exception_pause()
        if not targets:
            logger.warning(
                "Ignoring FT exception with no healthy pause target: "
                "rank=%s message=%s",
                event.rank,
                event.message,
            )
            return

        await self._pause_schedulers(targets)

    async def _pause_schedulers(self, targets: List[int]):
        acked = await self._send_command_collect(
            command="pause",
            target_ranks=targets,
            timeout_sec=self.server_args.fault_tolerance_timeout,
        )
        self.state.finish_pause(acked)

    async def _publish_active_ranks(
        self, active_mask: List[bool], timeout_sec: int
    ) -> None:
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
                "Fault tolerance command timeout: id=%s command=%s acked=%s pending=%s",
                request_id,
                command,
                sorted(pending.acked),
                sorted(pending.target_ranks - pending.acked),
            )
            raise
        finally:
            self._pending_commands.pop(request_id, None)

        if pending.failed:
            raise RuntimeError(
                f"fault tolerance command {command} failed: {pending.failed}"
            )
        logger.info(
            "FT command complete: id=%s command=%s acked=%s",
            request_id,
            command,
            sorted(pending.acked),
        )
        return pending.acked

    @staticmethod
    def _failstop(message: str) -> None:
        logger.error(message)
        kill_process_tree(os.getpid(), include_parent=True)
        raise RuntimeError(message)
