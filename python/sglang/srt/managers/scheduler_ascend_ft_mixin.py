from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from sglang.srt.managers.ascend_ft_communication import (
    AscendFTCommunication,
    AscendFTPause,
)
from sglang.srt.managers.io_struct import AscendFTAckOutput, AscendFTCommandReq
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils.process import notify_node_main_process_failure

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerAscendFTMixin:
    def init_ascend_ft(self: Scheduler, endpoint: str | None) -> None:
        self.ascend_ft_pending_pause: AscendFTCommandReq | None = None
        self.ascend_ft_pause_deadline: float | None = None
        self.ascend_ft_communication = None
        if endpoint is not None:
            self.ascend_ft_communication = AscendFTCommunication(
                endpoint,
                rank=get_parallel().dp_rank,
                world_size=get_parallel().num_dp_ranks,
                timeout=get_exec().features.fault_tolerance_gloo_timeout,
            )

    def run_ascend_ft_loop(self: Scheduler, dispatch) -> None:
        while True:
            try:
                dispatch(self)
                return
            except AscendFTPause:
                self._enter_ascend_ft_pause()
            except Exception as exc:
                if self.pause_state.paused:
                    # A fault in the paused loop itself cannot re-enter FT.
                    notify_node_main_process_failure()
                    raise
                logger.exception("Ascend FT paused after a scheduler error")
                self._enter_ascend_ft_pause(error=str(exc) or type(exc).__name__)

    def check_ascend_ft_pause_deadline(self: Scheduler) -> None:
        deadline = self.ascend_ft_pause_deadline
        if deadline is None or time.monotonic() < deadline:
            return
        self.ascend_ft_pause_deadline = None
        logger.error("Ascend FT pause unattended on DP rank %d", get_parallel().dp_rank)
        notify_node_main_process_failure()

    def _enter_ascend_ft_pause(self: Scheduler, error: str | None = None) -> None:
        self.pause_state.pause()
        self.pause_state.start_window()
        # A paused worker is no longer executing the interrupted forward.
        self.cur_batch_for_debug = None
        if error is not None:
            self.ascend_ft_pause_deadline = (
                time.monotonic() + get_exec().features.fault_tolerance_pause_timeout
            )
            self.send_control_output(
                AscendFTAckOutput(op="fault", rank=get_parallel().dp_rank, error=error),
                None,
            )
        req = self.ascend_ft_pending_pause
        if req is not None:
            self.ascend_ft_pending_pause = None
            self.send_control_output(
                AscendFTAckOutput(rid=req.rid, op="pause", rank=get_parallel().dp_rank),
                req,
            )

    def handle_ascend_ft_command(
        self: Scheduler, req: AscendFTCommandReq
    ) -> AscendFTAckOutput | None:
        comm = self.ascend_ft_communication
        rank = get_parallel().dp_rank
        if comm is None or rank not in req.target_ranks:
            return None
        ack = AscendFTAckOutput(rid=req.rid, op=req.op, rank=rank)
        if req.op == "pause":
            if self.pause_state.paused:
                return ack
            self.ascend_ft_pending_pause = req
            comm.pause_requested = True
            return None

        if not self.pause_state.paused:
            ack.error = "The DP rank has not reached its pause boundary."
            return ack
        try:
            if req.op == "recover":
                self.ascend_ft_pause_deadline = None
                active = tuple(req.target_ranks)
                runner = self.tp_worker.model_runner
                runner.recover_ascend_ft_device()
                group = (
                    comm.rebuild(active, req.rid)
                    if req.membership_changed or req.reset_collective
                    else comm.group
                )
                if req.membership_changed:
                    runner.eplb_manager.rebalance_for_ascend_ft(active, group)
                self._discard_ascend_ft_attempts()
                if req.membership_changed:
                    runner.install_ascend_ft_topology(active)
                    self.schedule_stream.synchronize()
                    self.forward_stream.wait_stream(self.schedule_stream)
                    with self.forward_stream_ctx:
                        runner.run_ascend_ft_health_gate(active, group)
            else:
                comm.pause_requested = False
                self.pause_state.resume()
        except Exception as exc:
            logger.exception("Ascend FT %s failed on DP rank %d", req.op, rank)
            ack.error = str(exc) or type(exc).__name__
        return ack

    def _discard_ascend_ft_attempts(self: Scheduler) -> None:
        # The tokenizer owns replay. Retire the old local attempts without
        # emitting abort outputs, and reset KV including the interrupted step.
        for reqs in self.request_lists().values():
            if reqs is not None:
                old_reqs = list(reqs)
                for req in old_reqs:
                    self.output_streamer.release_detokenize_state(req)
                reqs._leave(old_reqs)
        self.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
        self.last_batch = self.cur_batch_for_debug = None
        self._pending_chunked_abort_req = None
        self.return_health_check_ipcs.clear()
        if not self.flush_cache(empty_cache=False):
            raise RuntimeError("The Ascend FT KV cache did not reset.")
