from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

MC2_ELASTIC_INFO_HEADER_SIZE = 4
logger = logging.getLogger(__name__)


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


def validate_mc2_scale_down_routing(
    active_ranks: Sequence[bool] | torch.Tensor,
    *,
    original_ep_size: int,
    num_local_physical_experts: int,
    ep_dispatch_algorithm: Optional[str],
    logical_to_all_physical_map: torch.Tensor,
    logical_to_all_physical_map_num_valid: torch.Tensor,
    logical_to_rank_dispatch_physical_map: Optional[torch.Tensor],
) -> dict[str, object]:
    active = torch.as_tensor(active_ranks, dtype=torch.bool, device="cpu")
    if active.numel() != original_ep_size:
        raise RuntimeError(
            "MC2 active-rank mask size mismatch: "
            f"expected={original_ep_size} actual={active.numel()}"
        )

    all_map = logical_to_all_physical_map.detach().to(device="cpu", dtype=torch.long)
    num_valid = logical_to_all_physical_map_num_valid.detach().to(
        device="cpu", dtype=torch.long
    )
    if all_map.shape[:-1] != num_valid.shape:
        raise RuntimeError(
            "MC2 routing metadata shape mismatch: "
            f"all_map={tuple(all_map.shape)} num_valid={tuple(num_valid.shape)}"
        )

    candidate_slot = torch.arange(all_map.shape[-1]).view(
        *((1,) * (all_map.ndim - 1)), -1
    ) < num_valid.unsqueeze(-1)
    original_num_physical_experts = original_ep_size * num_local_physical_experts
    invalid_candidate = candidate_slot & (
        (all_map < 0) | (all_map >= original_num_physical_experts)
    )
    safe_candidate_ids = all_map.clamp(0, original_num_physical_experts - 1)
    candidate_ranks = torch.div(
        safe_candidate_ids,
        num_local_physical_experts,
        rounding_mode="floor",
    )
    live_candidate = candidate_slot & ~invalid_candidate & active[candidate_ranks]
    dead_candidate = candidate_slot & ~invalid_candidate & ~active[candidate_ranks]
    missing_live_expert = ~live_candidate.any(dim=-1)

    static_dead_reference = torch.zeros_like(missing_live_expert)
    static_invalid_reference = torch.zeros_like(missing_live_expert)
    if logical_to_rank_dispatch_physical_map is not None:
        rank_map = logical_to_rank_dispatch_physical_map.detach().to(
            device="cpu", dtype=torch.long
        )
        if rank_map.shape != missing_live_expert.shape:
            raise RuntimeError(
                "MC2 static routing metadata shape mismatch: "
                f"rank_map={tuple(rank_map.shape)} "
                f"logical_map={tuple(missing_live_expert.shape)}"
            )
        static_invalid_reference = (rank_map < 0) | (
            rank_map >= original_num_physical_experts
        )
        safe_rank_map = rank_map.clamp(0, original_num_physical_experts - 1)
        static_owner = torch.div(
            safe_rank_map,
            num_local_physical_experts,
            rounding_mode="floor",
        )
        static_dead_reference = ~static_invalid_reference & ~active[static_owner]

    def _sample(mask: torch.Tensor) -> list[list[int]]:
        return torch.nonzero(mask, as_tuple=False)[:8].tolist()

    summary = {
        "dead_candidate_count": int(dead_candidate.sum().item()),
        "dead_candidate_sample": _sample(dead_candidate),
        "invalid_candidate_count": int(invalid_candidate.sum().item()),
        "invalid_candidate_sample": _sample(invalid_candidate),
        "missing_live_expert_count": int(missing_live_expert.sum().item()),
        "missing_live_expert_sample": _sample(missing_live_expert),
        "static_dead_reference_count": int(static_dead_reference.sum().item()),
        "static_dead_reference_sample": _sample(static_dead_reference),
        "static_invalid_reference_count": int(static_invalid_reference.sum().item()),
        "static_invalid_reference_sample": _sample(static_invalid_reference),
    }

    errors = []
    if summary["invalid_candidate_count"]:
        errors.append("candidate map contains out-of-range physical expert ids")
    if summary["missing_live_expert_count"]:
        errors.append("logical experts have no live physical replica")
    if ep_dispatch_algorithm == "static" and (
        summary["static_dead_reference_count"]
        or summary["static_invalid_reference_count"]
    ):
        errors.append("static dispatch map references dead or invalid ranks")
    if (
        ep_dispatch_algorithm in ("dynamic", "fake", "lp")
        and summary["dead_candidate_count"]
    ):
        errors.append("dynamic dispatch candidates still include dead ranks")
    if errors:
        raise RuntimeError(
            "MC2 scale-down routing validation failed: "
            f"errors={errors} summary={summary}"
        )
    return summary


def validate_mc2_dispatch_expert_ids(
    physical_expert_ids: torch.Tensor,
    *,
    expert_weights: Optional[torch.Tensor],
    elastic_info: torch.Tensor,
    original_ep_size: int,
    num_local_physical_experts: int,
    ids_are_compacted: bool,
) -> None:
    ids = physical_expert_ids.detach().to(device="cpu", dtype=torch.long)
    selected = (
        expert_weights.detach().to(device="cpu").ne(0)
        if expert_weights is not None
        else torch.ones_like(ids, dtype=torch.bool)
    )
    if selected.shape != ids.shape:
        raise RuntimeError(
            "MC2 dispatch diagnostic shape mismatch: "
            f"expert_ids={tuple(ids.shape)} weights={tuple(selected.shape)}"
        )

    info = elastic_info.detach().to(device="cpu", dtype=torch.long)
    original_num_physical_experts = original_ep_size * num_local_physical_experts
    effective_num_physical_experts = int(info[3].item())
    mapped_ids = ids.clone()
    invalid = selected & (ids < 0)
    if ids_are_compacted:
        invalid |= selected & (ids >= effective_num_physical_experts)
        namespace = "effective"
    else:
        original_valid = (ids >= 0) & (ids < original_num_physical_experts)
        safe_ids = ids.clamp(0, original_num_physical_experts - 1)
        original_rank = torch.div(
            safe_ids,
            num_local_physical_experts,
            rounding_mode="floor",
        )
        local_expert_id = safe_ids % num_local_physical_experts
        effective_rank = info[4 : 4 + original_ep_size][original_rank]
        mapped_ids = effective_rank * num_local_physical_experts + local_expert_id
        invalid |= selected & (~original_valid | (effective_rank < 0))
        namespace = "original"

    invalid_sample = torch.nonzero(invalid, as_tuple=False)[:8]
    sample_coords = invalid_sample.tolist()
    sample_input_ids = (
        [int(ids[tuple(coord)].item()) for coord in invalid_sample]
        if invalid_sample.numel()
        else []
    )
    sample_mapped_ids = (
        [int(mapped_ids[tuple(coord)].item()) for coord in invalid_sample]
        if invalid_sample.numel()
        else []
    )
    logger.info(
        "NPU FT MC2 dispatch step=validate_expert_ids phase=%s namespace=%s "
        "selected_count=%s invalid_count=%s invalid_sample=%s input_ids=%s "
        "mapped_ids=%s elastic_info=%s",
        "failed" if invalid.any() else "complete",
        namespace,
        int(selected.sum().item()),
        int(invalid.sum().item()),
        sample_coords,
        sample_input_ids,
        sample_mapped_ids,
        info.tolist(),
    )
    if invalid.any():
        raise RuntimeError(
            "MC2 dispatch expert ids reference a dead or invalid expert before "
            f"kernel launch: namespace={namespace} invalid_sample={sample_coords} "
            f"input_ids={sample_input_ids} mapped_ids={sample_mapped_ids}"
        )


@dataclass
class NpuMC2ElasticInfo:
    tensor: torch.Tensor
    original_ep_size: int
    num_local_physical_experts: int
    _validate_next_dispatch: bool = False
    _log_next_dispatch_mapping: bool = False

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

    def arm_dispatch_validation(self) -> None:
        self._validate_next_dispatch = True

    def consume_dispatch_validation(self) -> bool:
        validate = self._validate_next_dispatch
        self._validate_next_dispatch = False
        return validate

    def arm_dispatch_mapping_log(self) -> None:
        self._log_next_dispatch_mapping = True

    def consume_dispatch_mapping_log(self) -> bool:
        should_log = self._log_next_dispatch_mapping
        self._log_next_dispatch_mapping = False
        return should_log

    @property
    def dispatch_validation_pending(self) -> bool:
        return self._validate_next_dispatch
