from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


MC2_ELASTIC_INFO_HEADER_SIZE = 4


def build_mc2_elastic_info_values(
    active_ranks: Sequence[bool] | torch.Tensor,
    *,
    original_ep_size: int,
    num_local_physical_experts: int,
) -> list[int]:
    values = (
        active_ranks.detach().cpu().flatten().tolist()
        if isinstance(active_ranks, torch.Tensor)
        else list(active_ranks)
    )
    if original_ep_size <= 0:
        raise ValueError("original_ep_size must be positive")
    if num_local_physical_experts <= 0:
        raise ValueError("num_local_physical_experts must be positive")
    if len(values) < original_ep_size:
        raise ValueError(
            "active-rank mask is smaller than the original EP world: "
            f"mask_size={len(values)} original_ep_size={original_ep_size}"
        )
    if any(values[original_ep_size:]):
        raise ValueError(
            "active-rank mask activates a rank outside the original EP world"
        )

    active_original_ranks = [
        rank for rank, is_active in enumerate(values[:original_ep_size]) if is_active
    ]
    effective_ep_size = len(active_original_ranks)
    if effective_ep_size == 0:
        raise ValueError("MC2 scale-down requires at least one active EP rank")

    payload = [-1] * (MC2_ELASTIC_INFO_HEADER_SIZE + 2 * original_ep_size)
    payload[:MC2_ELASTIC_INFO_HEADER_SIZE] = [
        int(effective_ep_size < original_ep_size),
        effective_ep_size,
        0,
        effective_ep_size * num_local_physical_experts,
    ]
    for effective_rank, original_rank in enumerate(active_original_ranks):
        payload[4 + original_rank] = effective_rank
        payload[4 + original_ep_size + effective_rank] = original_rank
    return payload


def build_mc2_elastic_info(
    active_ranks: Sequence[bool] | torch.Tensor,
    *,
    original_ep_size: int,
    num_local_physical_experts: int,
) -> torch.Tensor:
    return torch.tensor(
        build_mc2_elastic_info_values(
            active_ranks,
            original_ep_size=original_ep_size,
            num_local_physical_experts=num_local_physical_experts,
        ),
        dtype=torch.int32,
    )


def compact_mc2_physical_expert_ids(
    physical_expert_ids: torch.Tensor,
    *,
    elastic_info: torch.Tensor,
    original_ep_size: int,
    num_local_physical_experts: int,
) -> torch.Tensor:
    original_num_physical_experts = original_ep_size * num_local_physical_experts
    valid = (physical_expert_ids >= 0) & (
        physical_expert_ids < original_num_physical_experts
    )
    physical_ids = physical_expert_ids.masked_fill(~valid, 0)
    original_rank = torch.div(
        physical_ids, num_local_physical_experts, rounding_mode="floor"
    )
    local_expert_id = physical_ids % num_local_physical_experts
    effective_rank = elastic_info[4 : 4 + original_ep_size][original_rank.long()].to(
        dtype=physical_expert_ids.dtype
    )
    compact_ids = effective_rank * num_local_physical_experts + local_expert_id
    return compact_ids.masked_fill(~valid | (effective_rank < 0), -1)


@dataclass
class NpuMC2ElasticInfo:
    tensor: torch.Tensor
    original_ep_size: int
    num_local_physical_experts: int

    @classmethod
    def create(
        cls,
        active_ranks: Sequence[bool] | torch.Tensor,
        *,
        original_ep_size: int,
        num_physical_experts: int,
        device: torch.device | str,
    ) -> "NpuMC2ElasticInfo":
        if original_ep_size <= 0:
            raise ValueError("original_ep_size must be positive")
        if num_physical_experts <= 0:
            raise ValueError("num_physical_experts must be positive")
        if num_physical_experts % original_ep_size != 0:
            raise ValueError(
                "num_physical_experts must be divisible by original_ep_size"
            )
        num_local_physical_experts = num_physical_experts // original_ep_size
        tensor = build_mc2_elastic_info(
            active_ranks,
            original_ep_size=original_ep_size,
            num_local_physical_experts=num_local_physical_experts,
        ).to(device=device)
        return cls(tensor, original_ep_size, num_local_physical_experts)

    def update(self, active_ranks: Sequence[bool] | torch.Tensor) -> None:
        updated = build_mc2_elastic_info(
            active_ranks,
            original_ep_size=self.original_ep_size,
            num_local_physical_experts=self.num_local_physical_experts,
        ).to(device=self.tensor.device)
        self.tensor.copy_(updated)
