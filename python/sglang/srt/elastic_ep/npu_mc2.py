"""Fixed-address MC2 elastic metadata for Ascend fault-tolerant EP.

The original EP rank namespace and tensor storage are intentionally stable.
Only tensor contents change after a scale-down so an already captured NPU
graph can keep replaying the same dispatch/combine nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


MC2_ELASTIC_INFO_HEADER_SIZE = 4
MC2_ELASTIC_INFO_RANK_TABLE_COUNT = 2


def compact_mc2_physical_expert_ids(
    physical_expert_ids: torch.Tensor,
    *,
    elastic_info: torch.Tensor,
    original_ep_size: int,
    num_local_physical_experts: int,
) -> torch.Tensor:
    """Translate storage IDs into the compact expert namespace used by MC2.

    EPLB keeps physical expert IDs in the immutable original-rank namespace so
    weight slots on surviving ranks do not move when a rank disappears.  The
    elastic MC2 kernels, however, lay out their status/window space by compact
    survivor rank.  Apply the original-to-effective rank table while retaining
    the expert's local slot.  Before scale-down the table is the identity, so
    this operation can be captured once and replayed after in-place metadata
    updates.

    ``-1`` is preserved for callers that use it as a padded/invalid expert ID.
    An ID belonging to an inactive original rank also maps to ``-1``; a
    converged EPLB dispatch map must never select such an ID.
    """

    if original_ep_size <= 0:
        raise ValueError("original_ep_size must be positive")
    if num_local_physical_experts <= 0:
        raise ValueError("num_local_physical_experts must be positive")
    expected_elastic_info_size = (
        MC2_ELASTIC_INFO_HEADER_SIZE
        + MC2_ELASTIC_INFO_RANK_TABLE_COUNT * original_ep_size
    )
    if elastic_info.numel() < expected_elastic_info_size:
        raise ValueError(
            "elastic_info is too small for the original EP rank tables: "
            f"expected at least {expected_elastic_info_size}, "
            f"got {elastic_info.numel()}"
        )

    valid = (physical_expert_ids >= 0) & (
        physical_expert_ids < original_ep_size * num_local_physical_experts
    )
    safe_physical_ids = physical_expert_ids.masked_fill(~valid, 0)
    original_rank = torch.div(
        safe_physical_ids,
        num_local_physical_experts,
        rounding_mode="floor",
    )
    local_expert_id = safe_physical_ids % num_local_physical_experts
    original_to_effective = elastic_info[
        MC2_ELASTIC_INFO_HEADER_SIZE : MC2_ELASTIC_INFO_HEADER_SIZE + original_ep_size
    ]
    effective_rank = original_to_effective[original_rank.long()].to(
        physical_expert_ids.dtype
    )
    compact_ids = effective_rank * num_local_physical_experts + local_expert_id
    routable = valid & (effective_rank >= 0)
    return compact_ids.masked_fill(~routable, -1)


def build_mc2_elastic_info_values(
    active_ranks: Sequence[bool] | torch.Tensor,
    *,
    original_ep_size: int,
    num_local_physical_experts: int,
    shared_expert_rank_num: int = 0,
) -> torch.Tensor:
    """Build the CPU int32 payload consumed by MC2 dispatch and combine.

    Layout::
      [scaled_down, effective_ep_size, shared_expert_rank_num,
       effective_physical_expert_num,
       original_ep_rank -> effective_ep_rank,
       effective_ep_rank -> original_ep_rank (padded with -1)]
    """

    if original_ep_size <= 0:
        raise ValueError("original_ep_size must be positive")
    if num_local_physical_experts <= 0:
        raise ValueError("num_local_physical_experts must be positive")
    if shared_expert_rank_num < 0:
        raise ValueError("shared_expert_rank_num must be non-negative")

    if isinstance(active_ranks, torch.Tensor):
        if active_ranks.ndim != 1:
            raise ValueError("active_ranks must be one-dimensional")
        mask_values = active_ranks.detach().to(device="cpu").tolist()
    else:
        mask_values = list(active_ranks)

    if len(mask_values) != original_ep_size:
        raise ValueError(
            "active_ranks length must match original_ep_size "
            f"({original_ep_size}), got {len(mask_values)}"
        )
    if any(
        not isinstance(value, (bool, int)) or value not in (0, 1)
        for value in mask_values
    ):
        raise ValueError("active_ranks must contain only boolean or 0/1 values")

    surviving_original_ranks = [
        rank for rank, is_active in enumerate(mask_values) if bool(is_active)
    ]
    effective_ep_size = len(surviving_original_ranks)
    if effective_ep_size == 0:
        raise ValueError("active_ranks must keep at least one EP rank active")

    payload_size = (
        MC2_ELASTIC_INFO_HEADER_SIZE
        + MC2_ELASTIC_INFO_RANK_TABLE_COUNT * original_ep_size
    )
    payload = torch.full((payload_size,), -1, dtype=torch.int32, device="cpu")
    payload[:MC2_ELASTIC_INFO_HEADER_SIZE] = torch.tensor(
        [
            int(effective_ep_size < original_ep_size),
            effective_ep_size,
            shared_expert_rank_num,
            num_local_physical_experts * effective_ep_size,
        ],
        dtype=torch.int32,
    )

    original_to_effective = payload[
        MC2_ELASTIC_INFO_HEADER_SIZE : MC2_ELASTIC_INFO_HEADER_SIZE + original_ep_size
    ]
    effective_to_original = payload[
        MC2_ELASTIC_INFO_HEADER_SIZE
        + original_ep_size : MC2_ELASTIC_INFO_HEADER_SIZE
        + 2 * original_ep_size
    ]
    for effective_rank, original_rank in enumerate(surviving_original_ranks):
        original_to_effective[original_rank] = effective_rank
        effective_to_original[effective_rank] = original_rank

    return payload


@dataclass
class NpuMC2ElasticInfo:
    """A fixed-storage materialization of the current ElasticEPState mask."""

    tensor: torch.Tensor
    original_ep_size: int
    num_local_physical_experts: int
    shared_expert_rank_num: int
    _data_ptr: int

    @classmethod
    def create(
        cls,
        active_ranks: Sequence[bool] | torch.Tensor,
        *,
        original_ep_size: int,
        num_physical_experts: int,
        device: torch.device | str,
        shared_expert_rank_num: int = 0,
    ) -> "NpuMC2ElasticInfo":
        num_local_physical_experts, remainder = divmod(
            num_physical_experts, original_ep_size
        )
        if remainder != 0:
            raise ValueError(
                "num_physical_experts must be divisible by original_ep_size "
                f"({num_physical_experts} vs {original_ep_size})"
            )
        values = build_mc2_elastic_info_values(
            active_ranks,
            original_ep_size=original_ep_size,
            num_local_physical_experts=num_local_physical_experts,
            shared_expert_rank_num=shared_expert_rank_num,
        )
        tensor = values.to(device=device).contiguous()
        tensor.requires_grad_(False)
        return cls(
            tensor=tensor,
            original_ep_size=original_ep_size,
            num_local_physical_experts=num_local_physical_experts,
            shared_expert_rank_num=shared_expert_rank_num,
            _data_ptr=tensor.data_ptr(),
        )

    @property
    def data_ptr(self) -> int:
        return self._data_ptr

    def update(self, active_ranks: Sequence[bool] | torch.Tensor) -> None:
        """Update contents without replacing storage captured by the NPU graph."""

        if self.tensor.data_ptr() != self._data_ptr:
            raise RuntimeError("MC2 elastic_info storage changed before update")
        values = build_mc2_elastic_info_values(
            active_ranks,
            original_ep_size=self.original_ep_size,
            num_local_physical_experts=self.num_local_physical_experts,
            shared_expert_rank_num=self.shared_expert_rank_num,
        )
        self.tensor.copy_(values)
        if self.tensor.data_ptr() != self._data_ptr:
            raise RuntimeError("MC2 elastic_info update replaced captured storage")

    def validate_logical_expert_capacity(
        self,
        active_ranks: Sequence[bool] | torch.Tensor,
        *,
        num_logical_experts: int,
    ) -> None:
        """Ensure the fixed per-rank slots can still hold every logical expert."""

        if num_logical_experts <= 0:
            raise ValueError("num_logical_experts must be positive")
        values = build_mc2_elastic_info_values(
            active_ranks,
            original_ep_size=self.original_ep_size,
            num_local_physical_experts=self.num_local_physical_experts,
            shared_expert_rank_num=self.shared_expert_rank_num,
        )
        effective_ep_size = int(values[1].item())
        effective_physical_experts = int(values[3].item())
        if effective_physical_experts >= num_logical_experts:
            return

        required_local_experts = (
            num_logical_experts + effective_ep_size - 1
        ) // effective_ep_size
        required_total_experts = required_local_experts * self.original_ep_size
        required_redundant_experts = required_total_experts - num_logical_experts
        raise RuntimeError(
            "NPU MC2 scale-down has insufficient preallocated expert slots: "
            f"survivors={effective_ep_size}, "
            f"available_physical_experts={effective_physical_experts}, "
            f"logical_experts={num_logical_experts}. Restart with "
            "--ep-num-redundant-experts at least "
            f"{required_redundant_experts} for this survivor count."
        )
