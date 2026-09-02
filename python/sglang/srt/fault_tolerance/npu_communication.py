from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import logging
from typing import Any, Sequence

import torch.distributed as dist
from torch.distributed import PrefixStore, TCPStore

logger = logging.getLogger(__name__)


def _parse_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    address = endpoint.removeprefix("tcp://")
    host, port = address.rsplit(":", 1)
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host, int(port)


def create_npu_ft_store(endpoint: str) -> TCPStore:
    host, port = _parse_tcp_endpoint(endpoint)
    return TCPStore(host_name=host, port=port, is_master=True, wait_for_workers=False)


@dataclass
class NPUFaultToleranceCommunication:
    store: TCPStore
    original_rank: int
    original_world_size: int
    gloo_timeout_sec: float
    control_group: Any
    active_original_ranks: tuple[int, ...]
    generation: int = 0

    def rebuild_survivor_control_group(self, active_mask: Sequence[bool]) -> None:
        if len(active_mask) != self.original_world_size:
            raise ValueError(
                "NPU FT active mask size does not match the original world: "
                f"mask={len(active_mask)} world={self.original_world_size}"
            )
        active_ranks = tuple(rank for rank, active in enumerate(active_mask) if active)
        if self.original_rank not in active_ranks:
            raise ValueError(
                f"NPU FT local original rank {self.original_rank} is not active"
            )

        from sglang.srt.distributed.parallel_state import get_moe_ep_group
        from sglang.srt.eplb.process_group_context import (
            EPLBProcessGroupContext,
            set_eplb_process_group_context,
        )
        from sglang.srt.utils import init_custom_process_group

        compact_rank = active_ranks.index(self.original_rank)
        generation = self.generation + 1
        membership = "".join("1" if active else "0" for active in active_mask)
        group_name = f"npu_ft_control_{membership}_{generation}"
        store = PrefixStore(group_name, self.store)
        timeout = timedelta(seconds=self.gloo_timeout_sec)
        logger.info(
            "NPU FT control group phase=begin original_rank=%s compact_rank=%s "
            "active_original_ranks=%s generation=%s",
            self.original_rank,
            compact_rank,
            active_ranks,
            generation,
        )
        group = init_custom_process_group(
            backend="gloo",
            store=store,
            timeout=timeout,
            world_size=len(active_ranks),
            rank=compact_rank,
            group_name=group_name,
        )
        device_group = get_moe_ep_group().device_group
        dist.barrier(group=group)
        self.control_group = group
        self.active_original_ranks = active_ranks
        self.generation = generation
        set_eplb_process_group_context(
            EPLBProcessGroupContext(
                control_group=group,
                device_group=device_group,
                active_original_ranks=active_ranks,
                control_group_uses_cpu=True,
            )
        )
        logger.info(
            "NPU FT control group phase=complete original_rank=%s compact_rank=%s "
            "active_original_ranks=%s generation=%s",
            self.original_rank,
            compact_rank,
            active_ranks,
            generation,
        )


_communication: NPUFaultToleranceCommunication | None = None


def init_npu_ft_communication(
    endpoint: str,
    *,
    original_rank: int,
    original_world_size: int,
    gloo_timeout_sec: float,
    control_group,
) -> NPUFaultToleranceCommunication:
    global _communication
    host, port = _parse_tcp_endpoint(endpoint)
    _communication = NPUFaultToleranceCommunication(
        store=TCPStore(
            host_name=host,
            port=port,
            is_master=False,
            wait_for_workers=False,
        ),
        original_rank=original_rank,
        original_world_size=original_world_size,
        gloo_timeout_sec=gloo_timeout_sec,
        control_group=control_group,
        active_original_ranks=tuple(range(original_world_size)),
    )
    return _communication


def get_npu_ft_communication() -> NPUFaultToleranceCommunication | None:
    return _communication


def all_gather_into_tensor_with_timeout(
    output_tensor,
    input_tensor,
    *,
    group,
    timeout_sec: float,
) -> None:
    try:
        work = dist.all_gather_into_tensor(
            output_tensor,
            input_tensor,
            group=group,
            async_op=True,
        )
        completed = work.wait(timeout=timedelta(seconds=timeout_sec))
    except Exception as exc:
        raise RuntimeError("NPU FT survivor Gloo collective failed") from exc
    if completed is False:
        raise RuntimeError(
            f"NPU FT survivor Gloo collective timed out after {timeout_sec:g}s"
        )
