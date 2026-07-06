from __future__ import annotations

from enum import Enum
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple


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
    return True, ""


class FaultToleranceManager:
    def __init__(
        self,
        *,
        dp_size: int,
        strategy: str,
    ):
        self.dp_size = dp_size
        self.strategy = strategy
        self.rank_states = [RankState.HEALTHY] * dp_size
        self.ft_operation_in_progress = False

    def status_response(self) -> Dict[str, Any]:
        return {
            "ranks": [
                {"rank": rank, "state": state.value}
                for rank, state in enumerate(self.rank_states)
            ]
        }

    def has_paused_rank(self) -> bool:
        return RankState.PAUSED in self.rank_states

    def should_reject_admission(self) -> bool:
        return (
            self.strategy == "pause"
            and (self.ft_operation_in_progress or self.has_paused_rank())
        )

    def is_rank_healthy(self, rank: int) -> bool:
        return (
            0 <= rank < self.dp_size
            and self.rank_states[rank] == RankState.HEALTHY
        )

    def healthy_ranks(self) -> List[int]:
        return [
            rank
            for rank, state in enumerate(self.rank_states)
            if state == RankState.HEALTHY
        ]

    def paused_ranks(self) -> List[int]:
        return [
            rank
            for rank, state in enumerate(self.rank_states)
            if state == RankState.PAUSED
        ]

    def live_ranks(self) -> List[int]:
        return [
            rank
            for rank, state in enumerate(self.rank_states)
            if state != RankState.DEAD
        ]

    def active_mask(self, excluded_ranks=()) -> List[bool]:
        excluded = set(excluded_ranks)
        return [
            state != RankState.DEAD and rank not in excluded
            for rank, state in enumerate(self.rank_states)
        ]

    def begin_exception_pause(self) -> List[int]:
        self.ft_operation_in_progress = True
        return self.healthy_ranks()

    def finish_pause_collection(self, acked: set[int], timed_out: set[int]) -> None:
        for rank in acked:
            if (
                0 <= rank < self.dp_size
                and self.rank_states[rank] != RankState.DEAD
            ):
                self.rank_states[rank] = RankState.PAUSED
        for rank in timed_out:
            if 0 <= rank < self.dp_size:
                self.rank_states[rank] = RankState.DEAD
        self.ft_operation_in_progress = False

    def record_kill(self, rank: int) -> List[int]:
        self.ft_operation_in_progress = True
        if 0 <= rank < self.dp_size:
            self.rank_states[rank] = RankState.DEAD

        if self.strategy == "pause":
            return self.healthy_ranks()

        self.ft_operation_in_progress = False
        return []

    def record_inactive_mask(self, new_mask: List[bool]) -> List[int]:
        newly_inactive = []
        for rank, is_active in enumerate(new_mask[: self.dp_size]):
            if not is_active and self.rank_states[rank] != RankState.DEAD:
                self.rank_states[rank] = RankState.DEAD
                newly_inactive.append(rank)
            elif is_active and self.rank_states[rank] == RankState.DEAD:
                self.rank_states[rank] = RankState.HEALTHY

        if self.strategy == "pause" and newly_inactive:
            self.ft_operation_in_progress = True
            return self.healthy_ranks()

        return []

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
        if not any(
            state != RankState.DEAD and rank not in requested
            for rank, state in enumerate(self.rank_states)
        ):
            return "cannot_isolate_all_active_ranks"
        return None

    def begin_recover(
        self, instruction: str, scale_down_ranks: Optional[List[int]] = None
    ) -> Tuple[List[bool], List[int], List[int]]:
        self.ft_operation_in_progress = True
        pending_scale_down_ranks: List[int] = []
        if instruction == "scale_down":
            pending_scale_down_ranks = sorted(set(scale_down_ranks or []))
        return (
            self.active_mask(pending_scale_down_ranks),
            self.paused_ranks(),
            pending_scale_down_ranks,
        )

    def commit_recover(
        self, pending_scale_down_ranks: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        pending = set(pending_scale_down_ranks or [])
        for rank in pending:
            if 0 <= rank < self.dp_size:
                self.rank_states[rank] = RankState.DEAD
        resumed_ranks = []
        for rank, state in enumerate(self.rank_states):
            if state == RankState.PAUSED:
                self.rank_states[rank] = RankState.HEALTHY
                resumed_ranks.append(rank)
        self.ft_operation_in_progress = False
        body = self.status_response()
        body.update(
            {
                "success": True,
                "message": "fault_tolerance_apply_committed",
                "resumed_ranks": resumed_ranks,
            }
        )
        return body
