from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch.distributed import PrefixStore, TCPStore


def _parse_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    address = endpoint.removeprefix("tcp://")
    host, port = address.rsplit(":", 1)
    return host, int(port)


def create_npu_ft_store(endpoint: str) -> TCPStore:
    host, port = _parse_tcp_endpoint(endpoint)
    return TCPStore(host_name=host, port=port, is_master=True, wait_for_workers=False)


@dataclass
class NpuFTCommunication:
    store: TCPStore
    original_rank: int
    timeout_sec: float
    mlp_sync_group: Any
    active_original_ranks: tuple[int, ...]
    generation: int = 0

    def rebuild_mlp_sync_group(self, active_mask: Sequence[bool]) -> None:
        active_ranks = tuple(rank for rank, active in enumerate(active_mask) if active)
        compact_rank = active_ranks.index(self.original_rank)
        generation = self.generation + 1
        membership = "".join("1" if active else "0" for active in active_mask)
        prefix = f"npu-ft/{membership}/{generation}"
        timeout = timedelta(seconds=self.timeout_sec)

        from sglang.srt.utils import init_custom_process_group

        mlp_sync_group = init_custom_process_group(
            backend="gloo",
            store=PrefixStore(f"{prefix}/mlp-sync", self.store),
            timeout=timeout,
            world_size=len(active_ranks),
            rank=compact_rank,
            group_name=f"npu_ft_mlp_sync_{membership}_{generation}",
        )
        dist.barrier(group=mlp_sync_group)

        self.mlp_sync_group = mlp_sync_group
        self.active_original_ranks = active_ranks
        self.generation = generation

_communication: NpuFTCommunication | None = None


def all_gather_into_tensor_with_timeout(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
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
        raise RuntimeError(
            "NPU MC2 MLP-sync collective failed; entering the FT control loop"
        ) from exc
    if completed is False:
        raise RuntimeError(
            "NPU MC2 MLP-sync collective timed out after "
            f"{timeout_sec:g}s; entering the FT control loop"
        )


def init_npu_ft_communication(
    endpoint: str,
    *,
    original_rank: int,
    original_world_size: int,
    timeout_sec: float,
    mlp_sync_group,
) -> NpuFTCommunication:
    global _communication
    host, port = _parse_tcp_endpoint(endpoint)
    _communication = NpuFTCommunication(
        store=TCPStore(
            host_name=host,
            port=port,
            is_master=False,
            wait_for_workers=False,
        ),
        original_rank=original_rank,
        timeout_sec=timeout_sec,
        mlp_sync_group=mlp_sync_group,
        active_original_ranks=tuple(range(original_world_size)),
    )
    return _communication


def get_npu_ft_communication() -> NpuFTCommunication | None:
    return _communication
