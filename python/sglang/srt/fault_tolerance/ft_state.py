from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from sglang.srt.elastic_ep.topology import physical_ep_size_to_dp_size


class RankState(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEAD = "dead"


class FaultToleranceState:
    def __init__(
        self,
        *,
        dp_size: int,
        strategy: str,
        max_world_size: int,
        attn_replica_size: int,
    ):
        self.dp_size = dp_size
        self.strategy = strategy
        self.max_world_size = max_world_size
        self.global_ranks_per_dp = attn_replica_size
        max_dp_size = physical_ep_size_to_dp_size(max_world_size, attn_replica_size)
        self.expected_dp_mask = [rank < dp_size for rank in range(max_dp_size)]
        self.runtime_active_dp_mask = self.expected_dp_mask.copy()
        self.unreachable_dp_mask = [False] * max_dp_size
        self.exited_dp_ranks: set[int] = set()
        self.unhealthy_dp_ranks: set[int] = set()
        self.in_flight_request_id: Optional[str] = None

    @property
    def cluster_paused(self) -> bool:
        return bool(self.unhealthy_dp_ranks)

    def expand_dp_mask_to_global_rank_mask(self, dp_mask: List[bool]) -> List[bool]:
        return [
            dp_mask[rank // self.global_ranks_per_dp]
            for rank in range(self.max_world_size)
        ]

    def expected_dp_ranks(self) -> List[int]:
        return [rank for rank, expected in enumerate(self.expected_dp_mask) if expected]

    def _rank_state(self, rank: int) -> RankState:
        if not self.expected_dp_mask[rank] or rank in self.exited_dp_ranks:
            return RankState.DEAD
        if rank in self.unhealthy_dp_ranks:
            return RankState.UNHEALTHY
        return RankState.HEALTHY

    def status_response(self) -> Dict[str, Any]:
        engines = []
        for rank in range(self.dp_size):
            state = self._rank_state(rank)
            status = (
                "unreachable"
                if state is RankState.HEALTHY and self.unreachable_dp_mask[rank]
                else state.value
            )
            engines.append({"id": rank, "status": status})
        return {
            "schema_version": 1,
            "total_engines": self.dp_size,
            "engines": engines,
        }

    def has_unresolved_expected_dp_fault(self) -> bool:
        """Return whether an expected DP is unhealthy or has lost a process."""
        return bool(
            self.unhealthy_dp_ranks
            or self.exited_dp_ranks.intersection(self.expected_dp_ranks())
        )

    def is_global_admission_blocked(self, route_dp_mask: List[bool]) -> bool:
        return (
            not any(route_dp_mask)
            or self.in_flight_request_id is not None
            or self.cluster_paused
        )

    def observe_rank_fault(self, rank: int) -> None:
        if self.strategy == "pause" and self.expected_dp_mask[rank]:
            self.unhealthy_dp_ranks.add(rank)
