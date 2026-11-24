"""DP state for the fixed-world Ascend fault-tolerance controller."""


class FaultToleranceState:
    def __init__(self, dp_size: int):
        self.dp_size = dp_size
        self.expected_dp_mask = [True] * dp_size
        self.process_alive_global_rank_mask = [True] * dp_size
        self.runtime_active_dp_mask = [True] * dp_size
        self.pending_recovery_global_ranks: set[int] = set()
        self.unhealthy_dp_ranks: set[int] = set()
        self.ft_operation_in_progress = False
        self.cluster_paused = False

    def expected_dp_ranks(self) -> tuple[int, ...]:
        return tuple(
            rank for rank, expected in enumerate(self.expected_dp_mask) if expected
        )

    def status_response(self) -> dict:
        engines = []
        for rank, expected in enumerate(self.expected_dp_mask):
            if not expected or not self.process_alive_global_rank_mask[rank]:
                status = "dead"
            elif rank in self.unhealthy_dp_ranks:
                status = "unhealthy"
            else:
                status = "healthy"
            engines.append({"id": rank, "status": status})
        return {"schema_version": 1, "total_engines": self.dp_size, "engines": engines}

    def has_unresolved_expected_dp_fault(self) -> bool:
        return bool(self.unhealthy_dp_ranks) or any(
            not self.process_alive_global_rank_mask[rank]
            or rank in self.pending_recovery_global_ranks
            for rank in self.expected_dp_ranks()
        )

    def observe_process_active_ranks(self, ranks, *, active: bool) -> None:
        for rank in ranks:
            self.process_alive_global_rank_mask[rank] = active
            if not active and self.expected_dp_mask[rank]:
                self.runtime_active_dp_mask[rank] = False
                self.pending_recovery_global_ranks.add(rank)
                self.cluster_paused = True

    def observe_rank_fault(self, rank: int) -> None:
        if self.expected_dp_mask[rank]:
            self.runtime_active_dp_mask[rank] = False
            self.unhealthy_dp_ranks.add(rank)
            self.cluster_paused = True

    def observe_recovery_ack(self, rank: int) -> None:
        if self.process_alive_global_rank_mask[rank]:
            self.pending_recovery_global_ranks.discard(rank)
            self.runtime_active_dp_mask[rank] = True

    def serving_ready(self, rank: int) -> bool:
        return (
            self.expected_dp_mask[rank]
            and self.process_alive_global_rank_mask[rank]
            and self.runtime_active_dp_mask[rank]
            and rank not in self.pending_recovery_global_ranks
        )
