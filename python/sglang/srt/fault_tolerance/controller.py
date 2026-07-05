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
        dp_size: int,
        strategy: str,
        global_rank_count: Optional[int] = None,
        attention_tp_size: int = 1,
    ):
        self.dp_size = dp_size
        self.strategy = strategy
        if dp_size <= 0:
            raise ValueError("dp_size must be positive")
        if global_rank_count is None:
            global_rank_count = dp_size
        if global_rank_count <= 0:
            raise ValueError("global_rank_count must be positive")
        if attention_tp_size <= 0:
            raise ValueError("attention_tp_size must be positive")
        if global_rank_count < dp_size:
            raise ValueError("global_rank_count must be >= dp_size")

        self.global_rank_count = global_rank_count
        self.attention_tp_size = attention_tp_size
        self.dp_members: List[List[int]] = [[] for _ in range(dp_size)]
        self.global_rank_to_dp_rank: List[int] = [0] * global_rank_count
        for global_rank in range(global_rank_count):
            dp_rank = min(global_rank // attention_tp_size, dp_size - 1)
            self.global_rank_to_dp_rank[global_rank] = dp_rank
            self.dp_members[dp_rank].append(global_rank)
        if any(not members for members in self.dp_members):
            raise ValueError("each DP rank must own at least one global rank")

        self.physical_rank_states = [RankState.HEALTHY] * global_rank_count
        self._dp_route_forced_inactive: set[int] = set()
        self.rank_states = [RankState.HEALTHY] * dp_size
        self.ft_operation_in_progress = False
        self._refresh_dp_rank_states()

    def _refresh_dp_rank_states(self) -> None:
        for dp_rank, members in enumerate(self.dp_members):
            member_states = [self.physical_rank_states[rank] for rank in members]
            if dp_rank in self._dp_route_forced_inactive or any(
                state == RankState.DEAD for state in member_states
            ):
                self.rank_states[dp_rank] = RankState.DEAD
            elif any(state == RankState.PAUSED for state in member_states):
                self.rank_states[dp_rank] = RankState.PAUSED
            else:
                self.rank_states[dp_rank] = RankState.HEALTHY

    def dp_rank_for_global_rank(self, global_rank: int) -> Optional[int]:
        if 0 <= global_rank < self.global_rank_count:
            return self.global_rank_to_dp_rank[global_rank]
        return None

    def expand_dp_ranks(self, dp_ranks: List[int]) -> List[int]:
        expanded: List[int] = []
        for dp_rank in sorted(set(dp_ranks)):
            if 0 <= dp_rank < self.dp_size:
                expanded.extend(self.dp_members[dp_rank])
        return expanded

    def live_global_ranks_for_dp_ranks(self, dp_ranks: List[int]) -> List[int]:
        return [
            global_rank
            for global_rank in self.expand_dp_ranks(dp_ranks)
            if self.physical_rank_states[global_rank] != RankState.DEAD
        ]

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
            for rank, state in enumerate(self.physical_rank_states)
            if state == RankState.HEALTHY
        ]

    def paused_ranks(self) -> List[int]:
        return [
            rank
            for rank, state in enumerate(self.physical_rank_states)
            if state == RankState.PAUSED
        ]

    def live_ranks(self) -> List[int]:
        return [
            rank
            for rank, state in enumerate(self.physical_rank_states)
            if state != RankState.DEAD
        ]

    def ep_active_mask(self, excluded_global_ranks=()) -> List[bool]:
        excluded = set(excluded_global_ranks)
        return [
            state != RankState.DEAD and rank not in excluded
            for rank, state in enumerate(self.physical_rank_states)
        ]

    def dp_active_mask(self, excluded_dp_ranks=()) -> List[bool]:
        excluded = set(excluded_dp_ranks)
        self._refresh_dp_rank_states()
        return [
            state != RankState.DEAD and rank not in excluded
            for rank, state in enumerate(self.rank_states)
        ]

    def active_mask(self, excluded_ranks=()) -> List[bool]:
        return self.ep_active_mask(excluded_ranks)

    def begin_exception_pause(self) -> List[int]:
        self.ft_operation_in_progress = True
        return self.healthy_ranks()

    def finish_pause_collection(self, acked: set[int], timed_out: set[int]) -> None:
        for rank in acked:
            if (
                0 <= rank < self.global_rank_count
                and self.physical_rank_states[rank] != RankState.DEAD
            ):
                self.physical_rank_states[rank] = RankState.PAUSED
        for rank in timed_out:
            if 0 <= rank < self.global_rank_count:
                self.physical_rank_states[rank] = RankState.DEAD
        self._refresh_dp_rank_states()
        self.ft_operation_in_progress = False

    def record_kill(self, global_rank: int) -> List[int]:
        self.ft_operation_in_progress = True
        if 0 <= global_rank < self.global_rank_count:
            self.physical_rank_states[global_rank] = RankState.DEAD
            dp_rank = self.dp_rank_for_global_rank(global_rank)
            if dp_rank is not None:
                self._dp_route_forced_inactive.add(dp_rank)
        self._refresh_dp_rank_states()

        if self.strategy == "pause":
            return self.healthy_ranks()

        self.ft_operation_in_progress = False
        return []

    def record_inactive_mask(self, new_mask: List[bool]) -> List[int]:
        newly_inactive = []
        for rank, is_active in enumerate(new_mask[: self.dp_size]):
            if not is_active and self.rank_states[rank] != RankState.DEAD:
                self._dp_route_forced_inactive.add(rank)
                newly_inactive.append(rank)
            elif is_active:
                self._dp_route_forced_inactive.discard(rank)
                if len(self.dp_members[rank]) == 1:
                    global_rank = self.dp_members[rank][0]
                    if self.physical_rank_states[global_rank] == RankState.DEAD:
                        self.physical_rank_states[global_rank] = RankState.HEALTHY
        self._refresh_dp_rank_states()

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
        self,
        instruction: str,
        scale_down_ranks: Optional[List[int]] = None,
        shutdown_scale_down_ranks: bool = False,
    ) -> Tuple[List[bool], List[bool], List[int], List[int]]:
        self.ft_operation_in_progress = True
        pending_scale_down_ranks: List[int] = []
        excluded_global_ranks: List[int] = []
        if instruction == "scale_down":
            pending_scale_down_ranks = sorted(set(scale_down_ranks or []))
            if shutdown_scale_down_ranks:
                excluded_global_ranks = self.expand_dp_ranks(pending_scale_down_ranks)
        return (
            self.ep_active_mask(excluded_global_ranks),
            self.dp_active_mask(pending_scale_down_ranks),
            self.paused_ranks(),
            pending_scale_down_ranks,
        )

    def commit_recover(
        self,
        pending_scale_down_ranks: Optional[List[int]] = None,
        shutdown_scale_down_ranks: bool = False,
    ) -> Dict[str, Any]:
        pending = set(pending_scale_down_ranks or [])
        for rank in pending:
            if 0 <= rank < self.dp_size:
                self._dp_route_forced_inactive.add(rank)
        if shutdown_scale_down_ranks:
            for global_rank in self.expand_dp_ranks(list(pending)):
                self.physical_rank_states[global_rank] = RankState.DEAD
        resumed_ranks = []
        for rank, state in enumerate(self.physical_rank_states):
            if state == RankState.PAUSED:
                self.physical_rank_states[rank] = RankState.HEALTHY
                resumed_ranks.append(rank)
        self._refresh_dp_rank_states()
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
