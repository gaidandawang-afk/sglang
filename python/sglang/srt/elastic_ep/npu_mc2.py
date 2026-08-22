from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

MC2_ELASTIC_INFO_HEADER_SIZE = 4


def build_mc2_elastic_info(
    active_ranks: Sequence[bool] | torch.Tensor,
    *,
    original_ep_size: int,
    num_local_physical_experts: int,
) -> torch.Tensor:
    mask = torch.as_tensor(active_ranks, dtype=torch.bool, device="cpu")
    active_original_ranks = torch.nonzero(mask, as_tuple=False).flatten()
    effective_ep_size = active_original_ranks.numel()
    payload = torch.full(
        (MC2_ELASTIC_INFO_HEADER_SIZE + 2 * original_ep_size,),
        -1,
        dtype=torch.int32,
    )
    payload[:MC2_ELASTIC_INFO_HEADER_SIZE] = torch.tensor(
        [
            int(effective_ep_size < original_ep_size),
            effective_ep_size,
            0,
            effective_ep_size * num_local_physical_experts,
        ],
        dtype=torch.int32,
    )
    original_to_effective = payload[4 : 4 + original_ep_size]
    effective_to_original = payload[4 + original_ep_size :]
    original_to_effective[active_original_ranks] = torch.arange(
        effective_ep_size, dtype=torch.int32
    )
    effective_to_original[:effective_ep_size] = active_original_ranks
    return payload


def compact_mc2_physical_expert_ids(
    physical_expert_ids: torch.Tensor,
    *,
    elastic_info: torch.Tensor,
    original_ep_size: int,
    num_local_physical_experts: int,
) -> torch.Tensor:
    valid = physical_expert_ids >= 0
    physical_ids = physical_expert_ids.masked_fill(~valid, 0)
    original_rank = torch.div(
        physical_ids, num_local_physical_experts, rounding_mode="floor"
    )
    local_expert_id = physical_ids % num_local_physical_experts
    effective_rank = elastic_info[4 : 4 + original_ep_size][original_rank.long()]
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
        num_local_physical_experts = num_physical_experts // original_ep_size
        tensor = build_mc2_elastic_info(
            active_ranks,
            original_ep_size=original_ep_size,
            num_local_physical_experts=num_local_physical_experts,
        ).to(device=device)
        return cls(tensor, original_ep_size, num_local_physical_experts)

    def update(self, active_ranks: Sequence[bool] | torch.Tensor) -> None:
        self.tensor.copy_(
            build_mc2_elastic_info(
                active_ranks,
                original_ep_size=self.original_ep_size,
                num_local_physical_experts=self.num_local_physical_experts,
            )
        )
