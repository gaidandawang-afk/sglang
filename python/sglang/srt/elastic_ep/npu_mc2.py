from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


MC2_ELASTIC_INFO_HEADER_SIZE = 4


def _to_host_values(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    return list(values)


def build_mc2_elastic_info_values(
    active_ranks: Sequence[bool] | torch.Tensor,
    *,
    original_ep_size: int,
    num_local_physical_experts: int,
) -> list[int]:
    values = _to_host_values(active_ranks)
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


def validate_mc2_scale_down_routing(
    active_ranks: Sequence[bool] | torch.Tensor,
    *,
    original_ep_size: int,
    num_local_physical_experts: int,
    ep_dispatch_algorithm: str | None,
    logical_to_all_physical_map,
    logical_to_rank_dispatch_physical_map,
) -> dict[str, int]:
    active = [bool(value) for value in _to_host_values(active_ranks)]
    if len(active) != original_ep_size:
        raise RuntimeError(
            "MC2 active-rank mask size mismatch: "
            f"expected={original_ep_size} actual={len(active)}"
        )

    original_num_physical_experts = original_ep_size * num_local_physical_experts
    candidate_map = _to_host_values(logical_to_all_physical_map)
    static_map = (
        _to_host_values(logical_to_rank_dispatch_physical_map)
        if logical_to_rank_dispatch_physical_map is not None
        else None
    )
    errors = []
    logical_expert_count = 0
    candidate_count = 0
    for layer_id, layer_candidates in enumerate(candidate_map):
        if static_map is not None and layer_id >= len(static_map):
            errors.append(f"static map is missing layer {layer_id}")
            continue
        for logical_expert_id, candidates in enumerate(layer_candidates):
            logical_expert_count += 1
            valid_candidates = [
                candidate for candidate in candidates if candidate != -1
            ]
            candidate_count += len(valid_candidates)
            invalid_candidates = [
                candidate
                for candidate in valid_candidates
                if not 0 <= candidate < original_num_physical_experts
            ]
            active_candidates = [
                candidate
                for candidate in valid_candidates
                if 0 <= candidate < original_num_physical_experts
                and active[candidate // num_local_physical_experts]
            ]
            location = f"layer={layer_id} logical_expert={logical_expert_id}"
            if invalid_candidates:
                errors.append(
                    f"{location} has out-of-range candidates {invalid_candidates[:4]}"
                )
            if not active_candidates:
                errors.append(f"{location} has no active physical expert")

            if ep_dispatch_algorithm in ("dynamic", "fake", "lp"):
                dead_candidates = [
                    candidate
                    for candidate in valid_candidates
                    if 0 <= candidate < original_num_physical_experts
                    and not active[candidate // num_local_physical_experts]
                ]
                if dead_candidates:
                    errors.append(
                        f"{location} references dead candidates {dead_candidates[:4]}"
                    )
            elif ep_dispatch_algorithm == "static":
                if static_map is None or logical_expert_id >= len(static_map[layer_id]):
                    errors.append(f"{location} is missing a static dispatch entry")
                else:
                    selected = static_map[layer_id][logical_expert_id]
                    if not 0 <= selected < original_num_physical_experts:
                        errors.append(
                            f"{location} has out-of-range static id {selected}"
                        )
                    elif not active[selected // num_local_physical_experts]:
                        errors.append(
                            f"{location} static id {selected} is on a dead rank"
                        )

            if len(errors) >= 8:
                break
        if len(errors) >= 8:
            break

    if errors:
        raise RuntimeError(
            "MC2 scale-down routing validation failed: " + "; ".join(errors)
        )
    return {
        "logical_expert_count": logical_expert_count,
        "candidate_count": candidate_count,
    }


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
