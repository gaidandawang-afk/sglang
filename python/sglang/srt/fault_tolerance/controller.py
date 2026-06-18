from __future__ import annotations

import dataclasses
from enum import Enum
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple


class RankState(str, Enum):
    HEALTHY = "healthy"
    PAUSED = "paused"
    DEAD = "dead"


@dataclasses.dataclass
class RankRuntimeState:
    rank: int
    state: RankState = RankState.HEALTHY
    last_message: str = ""


def ft_failure(message: str) -> Dict[str, Any]:
    return {"success": False, "message": message}


def ft_error_status(error: str) -> int:
    if error == "fault_tolerance_disabled":
        return HTTPStatus.SERVICE_UNAVAILABLE
    if error == "ft_operation_in_progress":
        return HTTPStatus.CONFLICT
    return HTTPStatus.BAD_REQUEST


def is_mooncake_active_rank_backend(server_args) -> bool:
    return getattr(server_args, "elastic_ep_backend", None) == "mooncake"


def is_ft_supported_config(server_args) -> Tuple[bool, str]:
    if getattr(server_args, "pp_size", 1) != 1:
        return False, "ft_requires_pp1"
    if getattr(server_args, "nnodes", 1) != 1:
        return False, "ft_requires_single_node"
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
        enabled: bool,
        dp_size: int,
        strategy: str,
        is_mooncake_backend: bool,
    ):
        self.enabled = enabled
        self.dp_size = dp_size
        self.strategy = strategy
        self.is_mooncake_backend = is_mooncake_backend
        self.ranks = [RankRuntimeState(rank=i) for i in range(dp_size)]
        self.ft_operation_in_progress = False
        self.last_fault_type: str = ""
        self.last_fault_message: str = ""

    def status_response(self) -> Dict[str, Any]:
        return {
            "ranks": [
                {"rank": item.rank, "state": item.state.value}
                for item in self.ranks
            ]
        }

    def has_paused_rank(self) -> bool:
        return any(item.state == RankState.PAUSED for item in self.ranks)

    def has_healthy_rank(self) -> bool:
        return any(item.state == RankState.HEALTHY for item in self.ranks)

    def fault_handling_ready(self) -> bool:
        return self.has_paused_rank() and not self.ft_operation_in_progress

    def should_reject_admission(self) -> bool:
        return (
            self.enabled
            and self.strategy == "pause"
            and (self.ft_operation_in_progress or self.has_paused_rank())
        )

    def is_rank_healthy(self, rank: int) -> bool:
        return 0 <= rank < self.dp_size and self.ranks[rank].state == RankState.HEALTHY

    def healthy_ranks(self) -> List[int]:
        return [
            item.rank for item in self.ranks if item.state == RankState.HEALTHY
        ]

    def paused_ranks(self) -> List[int]:
        return [item.rank for item in self.ranks if item.state == RankState.PAUSED]

    def non_dead_ranks(self) -> List[int]:
        return [item.rank for item in self.ranks if item.state != RankState.DEAD]

    def begin_exception_pause(self, rank: int, message: str) -> List[int]:
        self.ft_operation_in_progress = True
        self.last_fault_type = "exception"
        self.last_fault_message = message
        if 0 <= rank < self.dp_size:
            self.ranks[rank].last_message = message
        return self.healthy_ranks()

    def finish_pause_collection(self, acked: set[int], timed_out: set[int]) -> None:
        for rank in acked:
            if 0 <= rank < self.dp_size and self.ranks[rank].state != RankState.DEAD:
                self.ranks[rank].state = RankState.PAUSED
        for rank in timed_out:
            if 0 <= rank < self.dp_size:
                self.ranks[rank].state = RankState.DEAD
        self.ft_operation_in_progress = False

    def record_kill(self, rank: int, message: str = "") -> List[int]:
        self.ft_operation_in_progress = True
        self.last_fault_type = "kill"
        self.last_fault_message = message
        if 0 <= rank < self.dp_size:
            self.ranks[rank].state = RankState.DEAD
            self.ranks[rank].last_message = message

        if self.strategy == "pause":
            return self.healthy_ranks()

        self.ft_operation_in_progress = False
        return []

    def record_inactive_mask(self, new_mask: List[bool]) -> List[int]:
        newly_inactive = []
        for rank, is_active in enumerate(new_mask[: self.dp_size]):
            if not is_active and self.ranks[rank].state != RankState.DEAD:
                self.ranks[rank].state = RankState.DEAD
                self.ranks[rank].last_message = "inactive_rank"
                newly_inactive.append(rank)

        if self.strategy == "pause" and newly_inactive:
            self.ft_operation_in_progress = True
            return self.healthy_ranks()

        return []

    def validate_apply(
        self, instruction: str, ranks: Optional[List[int]]
    ) -> Optional[str]:
        if not self.enabled:
            return "fault_tolerance_disabled"
        if self.ft_operation_in_progress:
            return "ft_operation_in_progress"
        if not self.has_paused_rank():
            return "no_paused_rank"
        if instruction not in ("retry", "scale_down"):
            return "unknown_instruction"
        if instruction == "retry":
            if ranks:
                return "retry_does_not_accept_ranks"
            return None
        return self.validate_scale_down_ranks(ranks)

    def validate_scale_down_ranks(self, ranks: Optional[List[int]]) -> Optional[str]:
        if not ranks:
            return "scale_down_requires_non_empty_ranks"
        requested = set(ranks)
        if any(rank < 0 or rank >= self.dp_size for rank in requested):
            return "unknown_rank"
        remaining = [
            item.rank
            for item in self.ranks
            if item.state != RankState.DEAD and item.rank not in requested
        ]
        if not remaining:
            return "cannot_isolate_all_active_ranks"
        return None

    def begin_recover(
        self, instruction: str, scale_down_ranks: Optional[List[int]] = None
    ) -> Tuple[List[bool], List[int]]:
        self.ft_operation_in_progress = True
        if instruction == "scale_down":
            for rank in set(scale_down_ranks or []):
                self.ranks[rank].state = RankState.DEAD
        active_mask = [item.state != RankState.DEAD for item in self.ranks]
        resume_targets = self.paused_ranks()
        return active_mask, resume_targets

    def commit_recover(self) -> Dict[str, Any]:
        resumed_ranks = []
        for item in self.ranks:
            if item.state == RankState.PAUSED:
                item.state = RankState.HEALTHY
                resumed_ranks.append(item.rank)
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

    def abort_operation(self, message: str = "") -> None:
        self.ft_operation_in_progress = False
        if message:
            self.last_fault_message = message
