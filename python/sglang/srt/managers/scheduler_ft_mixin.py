from __future__ import annotations

import logging
import time
from http import HTTPStatus
from typing import TYPE_CHECKING, Callable, Deque, Dict, Optional

from sglang.srt.elastic_ep.topology import physical_ep_rank_to_dp_rank
from sglang.srt.fault_tolerance.constants import (
    FT_OPERATION_RETRY,
    FT_OPERATION_SCALE_DOWN,
    FT_OPERATION_SCALE_DOWN_PREPARE,
)
from sglang.srt.fault_tolerance.utils import is_fault_tolerance_enabled
from sglang.srt.managers.io_struct import (
    AbortReq,
    ElasticScaleUpdateReq,
    FaultToleranceCommandReqInput,
    FaultToleranceCommandReqOutput,
    FaultToleranceRankFaultOutput,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import notify_node_main_process_failure

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def _ft_dp_rank() -> int:
    parallel = get_parallel()
    return physical_ep_rank_to_dp_rank(
        parallel.launch_world_rank, parallel.attn_tp_size * parallel.attn_cp_size
    )


class SchedulerFaultToleranceMixin:
    def init_fault_tolerance(self: Scheduler) -> None:
        self._ft_pause_deadline: Optional[float] = None
        self._ft_result_queue: Optional[Deque] = None
        self._ft_scale_down_request_id: Optional[str] = None

    def _run_event_loop_fault_tolerance(
        self: Scheduler, dispatch: Callable[[Scheduler], None]
    ) -> None:
        while True:
            try:
                dispatch(self)
                return
            except Exception as exc:
                if self._ft_result_queue is None:
                    self._ft_result_queue = getattr(self, "result_queue", None)
                self._ft_abort_inflight_window()
                if get_exec().features.fault_tolerance_on_error_strategy == "continue":
                    self._ft_discard_inflight_window()
                else:
                    self.pause_state.pause()
                    self._ft_pause_deadline = (
                        time.monotonic()
                        + get_exec().features.fault_tolerance_pause_timeout
                    )
                self.ipc_channels.send_to_tokenizer.send_output(
                    FaultToleranceRankFaultOutput(
                        rank=_ft_dp_rank(),
                        message=str(exc),
                    )
                )

    def _ft_inflight_reqs(self: Scheduler) -> Dict[str, Req]:
        window_batches = [
            self.cur_batch_for_debug,
            self.last_batch,
            self.running_batch,
        ]
        if self._ft_result_queue is not None:
            window_batches.extend(batch for batch, _ in self._ft_result_queue)

        def should_discard(req):
            owns_state = req.kv.holds_kv or req.kv.holds_mamba
            return not req.finished() or owns_state

        discarded_by_rid = {}
        for batch in window_batches:
            if batch is None:
                continue
            for req in batch.reqs:
                if should_discard(req):
                    discarded_by_rid.setdefault(req.rid, req)
        if self.chunked_req is not None and should_discard(self.chunked_req):
            discarded_by_rid.setdefault(self.chunked_req.rid, self.chunked_req)
        return discarded_by_rid

    def _ft_abort_inflight_window(self: Scheduler) -> None:
        for req in self._ft_inflight_reqs().values():
            abort_reason = FINISH_ABORT(
                message="Request discarded during fault tolerance recovery.",
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                err_type="SchedulerFault",
            )
            req.finished_reason = abort_reason
            self.ipc_channels.send_to_tokenizer.send_output(
                AbortReq(
                    finished_reason=abort_reason.to_json(),
                    rid=req.rid,
                ),
                req,
            )

    def _ft_discard_inflight_window(self: Scheduler) -> None:
        discarded_reqs = self._ft_inflight_reqs()
        for req in discarded_reqs.values():
            release_kv_cache(req, self.tree_cache, checkpoint=False)

        self.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
        if self.chunked_req is not None and self.chunked_req.rid in discarded_reqs:
            self.chunked_req = None
        if self._ft_result_queue is not None:
            self._ft_result_queue.clear()
        self._ft_result_queue = None
        self.cur_batch_for_debug = None
        self.last_batch = None
        logger.warning("FT discarded %d in-flight request(s)", len(discarded_reqs))

    def handle_fault_tolerance_command(
        self: Scheduler, recv_req: FaultToleranceCommandReqInput
    ) -> Optional[FaultToleranceCommandReqOutput]:
        rank = _ft_dp_rank()
        if rank not in recv_req.target_ranks:
            return None

        parallel = get_parallel()
        is_leader = parallel.attn_tp_rank == 0 and parallel.attn_cp_rank == 0
        ack = (
            FaultToleranceCommandReqOutput(
                request_id=recv_req.request_id, command=recv_req.command, rank=rank
            )
            if is_leader
            else None
        )
        model_runner = self.tp_worker.model_runner
        if recv_req.command == FT_OPERATION_SCALE_DOWN_PREPARE:
            if not self.pause_state.paused:
                return None
            assert recv_req.active_mask is not None
            model_runner.update_fault_tolerance_active_ranks(recv_req.active_mask)
            self._ft_scale_down_request_id = recv_req.request_id
            return ack

        if recv_req.command == FT_OPERATION_RETRY:
            model_runner.update_fault_tolerance_active_ranks(None)
            new_ep_size = None
        elif recv_req.command == FT_OPERATION_SCALE_DOWN:
            if (
                not self.pause_state.paused
                or self._ft_scale_down_request_id != recv_req.request_id
            ):
                return None
            new_ep_size = model_runner.finish_fault_tolerance_scale_down()
        else:
            logger.warning("FT unknown command: %s", recv_req.command)
            return None

        self._ft_scale_down_request_id = None
        self._ft_discard_inflight_window()
        self.pause_state.resume()
        self._ft_pause_deadline = None

        if is_leader and new_ep_size is not None and rank == min(recv_req.target_ranks):
            self.ipc_channels.send_to_tokenizer.send_output(
                ElasticScaleUpdateReq(success=True, effective_ep_size=new_ep_size)
            )
        return ack

    def _check_ft_pause_deadline(self: Scheduler) -> None:
        if not is_fault_tolerance_enabled():
            return
        deadline = self._ft_pause_deadline
        if deadline is None or time.monotonic() < deadline:
            return
        self._ft_pause_deadline = None
        logger.error(
            "Fault tolerance pause unattended: timeout_sec=%s dp_rank=%s",
            get_exec().features.fault_tolerance_pause_timeout,
            get_parallel().dp_rank,
        )
        notify_node_main_process_failure()
