"""Serialize Ascend FT operations in the process that owns request streams."""

import asyncio
import logging
from uuid import uuid4

from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    AscendFTAckOutput,
    AscendFTCommandReq,
)
from sglang.srt.runtime_context import get_device, get_parallel

logger = logging.getLogger(__name__)


class AscendFTManager:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.world_size = get_parallel().num_dp_ranks
        self.active_ranks = tuple(range(self.world_size))
        self.min_survivors = self.world_size
        self.phase = "running"
        self.error = None
        self.request_id = None
        self.task = None
        self.acks = asyncio.Queue()
        self.pending_command = None

    def status(self) -> dict:
        return {
            "phase": self.phase,
            "active_dp_ranks": list(self.active_ranks),
            "min_survivors": self.min_survivors,
            "request_id": self.request_id,
            "error": self.error,
        }

    def apply(self, instruction: str, removed_dp_ranks: list[int]) -> dict:
        if self.task is not None and not self.task.done():
            raise RuntimeError("An Ascend FT operation is already running.")
        if self.phase == "failed":
            raise RuntimeError("Ascend FT recovery failed. Restart the engine.")
        if instruction == "scale_down":
            removed = set(removed_dp_ranks)
            if not removed or not removed < set(self.active_ranks):
                raise ValueError("Remove a nonempty proper subset of active DP ranks.")
            if len(self.active_ranks) - len(removed) < self.min_survivors:
                raise ValueError("Surviving DP ranks do not have enough expert slots.")
        elif instruction not in ("pause", "retry") or removed_dp_ranks:
            raise ValueError("Use pause, retry, or scale_down with removed_dp_ranks.")

        self.tokenizer.auto_create_handle_loop()
        self.request_id = uuid4().hex
        self.phase = "pausing"
        self.error = None
        self.tokenizer.is_pause = True
        self.task = asyncio.create_task(self._run(instruction, removed_dp_ranks))
        return self.status()

    def handle_ack(self, ack: AscendFTAckOutput) -> None:
        if ack.op == "fault" and ack.rank in self.active_ranks:
            self.tokenizer.is_pause = True
            if self.phase == "running":
                self.phase = "faulted"
                self.error = ack.error
            elif self.phase == "resuming":
                self.error = ack.error
        elif (ack.rid, ack.op) == self.pending_command:
            self.acks.put_nowait(ack)

    async def _send(self, op, active, *, membership_changed=False):
        wire_id = uuid4().hex
        self.pending_command = (wire_id, op)
        try:
            await self.tokenizer._async_dispatch_to_scheduler(
                AscendFTCommandReq(
                    rid=wire_id,
                    op=op,
                    target_ranks=list(active),
                    membership_changed=membership_changed,
                )
            )
            waiting = set(active)

            async def wait_for_workers():
                while waiting:
                    ack = await self.acks.get()
                    if ack.rid != wire_id or ack.rank not in waiting:
                        continue
                    if ack.error:
                        raise RuntimeError(f"DP rank {ack.rank}: {ack.error}")
                    waiting.remove(ack.rank)

            await asyncio.wait_for(wait_for_workers(), get_device().watchdog_timeout)
        finally:
            self.pending_command = None

    async def _run(self, instruction, removed_dp_ranks):
        active = tuple(
            rank for rank in self.active_ranks if rank not in removed_dp_ranks
        )
        try:
            await self._send("pause", active)
            self.phase = "paused"
            if instruction == "pause":
                return
            self.phase = "recovering"
            await self._send(
                "recover", active, membership_changed=active != self.active_ranks
            )
            self.active_ranks = active
            await self.tokenizer._async_dispatch_to_scheduler(
                ActiveRanksOutput(
                    status=[rank in active for rank in range(self.world_size)]
                )
            )
            await self.tokenizer.ascend_ft_replay(removed_dp_ranks)
            self.error = None
            self.phase = "resuming"
            await self._send("resume", active)
            async with self.tokenizer.is_pause_cond:
                if self.error is not None:
                    raise RuntimeError(self.error)
                self.phase = "running"
                self.tokenizer.is_pause = False
                self.tokenizer.is_pause_cond.notify_all()
        except Exception as exc:
            # A timeout before recovery is retryable with a smaller member set.
            # Recovery may have already moved expert weights on some workers.
            self.phase = "faulted" if self.phase == "pausing" else "failed"
            self.error = str(exc) or type(exc).__name__
            logger.exception("Ascend FT %s failed", instruction)
