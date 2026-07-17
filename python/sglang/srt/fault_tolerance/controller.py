from __future__ import annotations

from enum import Enum
from http import HTTPStatus
from typing import Any, Dict, Iterable, List, Optional, Tuple


class RankState(str, Enum):
    HEALTHY = "healthy"
    PAUSED = "paused"
    DEAD = "dead"


def ft_failure(message: str) -> Dict[str, Any]:
    return {"success": False, "message": message}


def ft_error_status(error: str) -> int:
    if error == "ft_operation_in_progress":
        return HTTPStatus.CONFLICT
    return HTTPStatus.BAD_REQUEST


def is_mooncake_active_rank_backend(server_args) -> bool:
    return getattr(server_args, "elastic_ep_backend", None) == "mooncake"


def is_ft_supported_config(server_args) -> Tuple[bool, str]:
    if getattr(server_args, "dp_size", 1) <= 1:
        return False, "ft_requires_dp_gt1"
    if getattr(server_args, "pp_size", 1) != 1:
        return False, "ft_requires_pp1"
    if not is_mooncake_active_rank_backend(server_args):
        return False, "ft_requires_mooncake_active_rank_backend"
    if getattr(server_args, "disaggregation_mode", "null") != "null":
        return False, "ft_unsupported_with_pd"
    if getattr(server_args, "device", None) == "npu":
        return False, "ft_unsupported_with_npu"
    if getattr(server_args, "tokenizer_worker_num", 1) > 1:
        return False, "ft_unsupported_with_multi_tokenizer"
    if getattr(server_args, "use_ray", False):
        return False, "ft_unsupported_with_ray_engine"
    if getattr(server_args, "enable_dp_attention", False):
        attention_block_count = getattr(server_args, "dp_size", 1) * getattr(
            server_args, "attn_cp_size", 1
        )
        if getattr(server_args, "tp_size", 1) % attention_block_count != 0:
            return False, "ft_requires_tp_divisible_by_dp_and_attn_cp"
    return True, ""


class FaultToleranceState:
    def __init__(
        self,
        *,
        dp_size: int,
        strategy: str,
    ):
        self.dp_size = dp_size
        self.strategy = strategy
        self.process_active_ranks = [True] * dp_size
        self.mooncake_active_ranks = [True] * dp_size
        self.disabled_dp_ranks: set[int] = set()
        self.paused_dp_ranks: set[int] = set()
        self.rank_states = [RankState.HEALTHY] * dp_size
        self.ft_operation_in_progress = False
        self._last_effective_active_ranks = [True] * dp_size

    def status_response(self) -> Dict[str, Any]:
        self.refresh_rank_states()
        return {
            "ranks": [
                {"rank": rank, "state": state.value}
                for rank, state in enumerate(self.rank_states)
            ]
        }

    def has_paused_rank(self) -> bool:
        return bool(self.paused_dp_ranks)

    def should_reject_admission(self) -> bool:
        if not any(self.effective_active_mask()):
            return True
        return self.strategy == "pause" and (
            self.ft_operation_in_progress or self.has_paused_rank()
        )

    def runtime_active_mask(self) -> List[bool]:
        return [
            self.process_active_ranks[rank] and self.mooncake_active_ranks[rank]
            for rank in range(self.dp_size)
        ]

    def effective_active_mask(self) -> List[bool]:
        return [
            is_active and rank not in self.disabled_dp_ranks
            for rank, is_active in enumerate(self.runtime_active_mask())
        ]

    def pause_targets(self) -> List[int]:
        return [
            rank
            for rank, is_active in enumerate(self.runtime_active_mask())
            if is_active
        ]

    def resume_targets(self) -> List[int]:
        runtime_active = self.runtime_active_mask()
        return sorted(rank for rank in self.paused_dp_ranks if runtime_active[rank])

    def refresh_rank_states(self) -> None:
        effective_active = self.effective_active_mask()
        self.rank_states = [
            (
                RankState.DEAD
                if not effective_active[rank]
                else (
                    RankState.PAUSED
                    if rank in self.paused_dp_ranks
                    else RankState.HEALTHY
                )
            )
            for rank in range(self.dp_size)
        ]

    def is_rank_routable(self, rank: int) -> bool:
        return 0 <= rank < self.dp_size and self.effective_active_mask()[rank]

    def _normalize_mask(self, mask: List[bool]) -> List[bool]:
        return [rank < len(mask) and bool(mask[rank]) for rank in range(self.dp_size)]

    def _begin_availability_pause(self, old_effective: List[bool]) -> List[int]:
        new_effective = self.effective_active_mask()
        has_falling_edge = any(
            old_active and not new_active
            for old_active, new_active in zip(old_effective, new_effective)
        )
        if (
            self.strategy != "pause"
            or not has_falling_edge
            or self.ft_operation_in_progress
            or self.has_paused_rank()
        ):
            return []
        targets = self.pause_targets()
        if not targets:
            return []
        self.ft_operation_in_progress = True
        return targets

    def observe_process_active_ranks(
        self, ranks: Iterable[int], *, active: bool
    ) -> List[int]:
        old_effective = self.effective_active_mask()
        for rank in ranks:
            if 0 <= rank < self.dp_size:
                self.process_active_ranks[rank] = active
        return self._begin_availability_pause(old_effective)

    def observe_mooncake_active_ranks(self, new_mask: List[bool]) -> List[int]:
        old_effective = self.effective_active_mask()
        self.mooncake_active_ranks = self._normalize_mask(new_mask)
        return self._begin_availability_pause(old_effective)

    def observe_recovered_dp_ranks(self, ranks: Iterable[int]) -> List[int]:
        old_effective = self.effective_active_mask()
        for rank in ranks:
            if 0 <= rank < self.dp_size:
                self.disabled_dp_ranks.discard(rank)
        return self._begin_availability_pause(old_effective)

    def pending_effective_active_update(self) -> Optional[List[bool]]:
        effective_active = self.effective_active_mask()
        if effective_active == self._last_effective_active_ranks:
            return None
        return effective_active

    def mark_effective_active_published(self, effective_active: List[bool]) -> None:
        self._last_effective_active_ranks = list(effective_active)

    def take_effective_active_update(self) -> Optional[List[bool]]:
        effective_active = self.pending_effective_active_update()
        if effective_active is None:
            return None
        self._last_effective_active_ranks = effective_active
        return effective_active

    def begin_exception_pause(self) -> List[int]:
        if self.ft_operation_in_progress or self.has_paused_rank():
            return []
        targets = self.pause_targets()
        if not targets:
            return []
        self.ft_operation_in_progress = True
        return targets

    def finish_pause(self, paused_ranks: Iterable[int]) -> None:
        self.paused_dp_ranks.update(
            rank for rank in paused_ranks if 0 <= rank < self.dp_size
        )
        self.ft_operation_in_progress = False
        self.refresh_rank_states()

    def validate_apply(
        self, instruction: str, ranks: Optional[List[int]]
    ) -> Optional[str]:
        if self.ft_operation_in_progress:
            return "ft_operation_in_progress"
        if not self.has_paused_rank():
            return "no_paused_rank"
        if instruction not in ("retry", "scale_down"):
            return "unknown_instruction"
        if instruction == "retry":
            if ranks is not None:
                return "retry_does_not_accept_ranks"
            return None
        return self.validate_scale_down_ranks(ranks)

    def validate_scale_down_ranks(self, ranks: Optional[List[int]]) -> Optional[str]:
        if not ranks:
            return "scale_down_requires_non_empty_ranks"
        requested = set(ranks)
        if any(rank < 0 or rank >= self.dp_size for rank in requested):
            return "unknown_rank"
        disabled = self.disabled_dp_ranks | requested
        runtime_active = self.runtime_active_mask()
        if not any(
            is_active and rank not in disabled
            for rank, is_active in enumerate(runtime_active)
        ):
            return "cannot_isolate_all_active_ranks"
        return None

    def begin_recover(
        self, instruction: str, scale_down_ranks: Optional[List[int]] = None
    ) -> Tuple[List[bool], List[int]]:
        self.ft_operation_in_progress = True
        if instruction == "scale_down":
            self.disabled_dp_ranks.update(scale_down_ranks or [])
        return self.effective_active_mask(), self.resume_targets()

    def commit_recover(
        self,
        resumed_ranks: Optional[Iterable[int]] = None,
        isolated_ranks: Optional[Iterable[int]] = None,
    ) -> Dict[str, Any]:
        resumable = (
            self.paused_dp_ranks if resumed_ranks is None else set(resumed_ranks)
        )
        committed_resumed_ranks = sorted(self.paused_dp_ranks & set(resumable))
        self.paused_dp_ranks.difference_update(committed_resumed_ranks)
        self.paused_dp_ranks.difference_update(isolated_ranks or [])
        self.ft_operation_in_progress = False
        self.refresh_rank_states()
        body = self.status_response()
        body.update(
            {
                "success": True,
                "message": "fault_tolerance_apply_committed",
                "resumed_ranks": committed_resumed_ranks,
            }
        )
        return body
