"""Fixed-world MC2 routing for Ascend fault tolerance."""

from __future__ import annotations

import torch


class NpuMC2ElasticInfo:
    """MC2 uses compact expert IDs and the original communicator."""

    def __init__(self, ep_size: int, local_experts: int, device: str):
        self.ep_size = ep_size
        self.local_experts = local_experts
        self.tensor = torch.empty(4 + 2 * ep_size, dtype=torch.int32, device=device)
        self.install(tuple(range(ep_size)))

    def install(self, active_ranks: tuple[int, ...]) -> None:
        count = len(active_ranks)
        values = [int(count != self.ep_size), count, 0, count * self.local_experts]
        values += [-1] * (2 * self.ep_size)
        for compact_rank, rank in enumerate(active_ranks):
            values[4 + rank] = compact_rank
            values[4 + self.ep_size + compact_rank] = rank
        self.tensor.copy_(torch.tensor(values, dtype=torch.int32))
        self.active_ranks = active_ranks

    @property
    def shrunk(self) -> bool:
        return len(self.active_ranks) < self.ep_size

    def compact_expert_ids(self, ids: torch.Tensor) -> torch.Tensor:
        if not self.shrunk:
            return ids
        capacity = self.ep_size * self.local_experts
        rank = ids.clamp(0, capacity - 1) // self.local_experts
        compact = self.tensor[4 : 4 + self.ep_size][rank.long()].to(ids.dtype)
        return (compact * self.local_experts + ids % self.local_experts).masked_fill(
            (ids < 0) | (ids >= capacity) | (compact < 0), -1
        )


# One target model per scheduler process in the initial Ascend FT topology.
_mc2_elastic_info: NpuMC2ElasticInfo | None = None


def init_mc2_elastic_info(ep_size: int, local_experts: int, device: str) -> None:
    global _mc2_elastic_info
    _mc2_elastic_info = NpuMC2ElasticInfo(ep_size, local_experts, device)


def get_mc2_elastic_info() -> NpuMC2ElasticInfo | None:
    return _mc2_elastic_info
