from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class RankState(str, Enum):
    HEALTHY = "healthy"
    PAUSED = "paused"
    DEAD = "dead"


@dataclass
class RankStatus:
    rank: int
    state: RankState

    def to_api(self) -> Dict[str, Any]:
        return {"rank": self.rank, "state": self.state.value}


class FaultToleranceManager:
    """Main-process FT state machine for the v3 management API.

    This class intentionally owns only control-plane state. Runtime mutations
    such as aborting requests, updating DP routing, or applying a Mooncake mask
    are performed by TokenizerManager/Scheduler call sites.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        dp_size: int,
        on_error_strategy: str,
        recovery_timeout_sec: int,
        moe_a2a_backend: str,
        elastic_ep_backend: Optional[str],
    ) -> None:
        self.enabled = enabled
        self.dp_size = dp_size
        self.on_error_strategy = on_error_strategy
        self.recovery_timeout_sec = recovery_timeout_sec
        self.moe_a2a_backend = moe_a2a_backend
        self.elastic_ep_backend = elastic_ep_backend
        self._lock = threading.Lock()
        self._ranks = [
            RankStatus(rank=rank, state=RankState.HEALTHY)
            for rank in range(dp_size)
        ]
        self._admission_open = True
        self._last_message = ""

    @property
    def is_mooncake_backend(self) -> bool:
        return (
            self.moe_a2a_backend == "mooncake"
            or self.elastic_ep_backend == "mooncake"
        )

    def status_response(self) -> Dict[str, Any]:
        with self._lock:
            return {"ranks": [rank.to_api() for rank in self._ranks]}

    def admission_open(self) -> bool:
        if not self.enabled:
            return True
        with self._lock:
            return self._admission_open

    def unavailable_error(self) -> Dict[str, Any]:
        message = self._last_message or "SGLang engine is paused by fault tolerance."
        return {
            "error": {
                "type": "fault_tolerance_unavailable",
                "message": message,
            }
        }

    def active_mask(self) -> List[bool]:
        with self._lock:
            return [rank.state != RankState.DEAD for rank in self._ranks]

    def active_ranks(self) -> List[int]:
        with self._lock:
            return [
                rank.rank
                for rank in self._ranks
                if rank.state != RankState.DEAD
            ]

    def paused_ranks(self) -> List[int]:
        with self._lock:
            return [
                rank.rank
                for rank in self._ranks
                if rank.state == RankState.PAUSED
            ]

    def validate_routed_rank(self, rank: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            if rank < 0 or rank >= self.dp_size:
                return
            if self._ranks[rank].state == RankState.DEAD:
                raise ValueError(
                    f"routed_dp_rank={rank} is isolated by fault tolerance"
                )

    def record_fault(self, rank: int, message: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            if not 0 <= rank < self.dp_size:
                logger.warning("Ignoring FT fault for unknown rank %s", rank)
                return
            self._ranks[rank].state = RankState.DEAD
            for item in self._ranks:
                if item.state == RankState.HEALTHY:
                    item.state = RankState.PAUSED
            self._admission_open = False
            self._last_message = message or "SGLang engine is paused by fault tolerance."

    def pause_all_active(self, message: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            for item in self._ranks:
                if item.state == RankState.HEALTHY:
                    item.state = RankState.PAUSED
            self._admission_open = False
            self._last_message = message or "SGLang engine is paused by fault tolerance."

    def apply_retry(self) -> Dict[str, Any]:
        if not self.enabled:
            return self._failure("fault tolerance is not enabled")
        with self._lock:
            for item in self._ranks:
                if item.state == RankState.PAUSED:
                    item.state = RankState.HEALTHY
            self._admission_open = True
            self._last_message = ""
            return self._success_locked("retry succeeded")

    def begin_retry(self) -> Dict[str, Any]:
        if not self.enabled:
            return self._failure("fault tolerance is not enabled")
        with self._lock:
            for item in self._ranks:
                if item.state != RankState.DEAD:
                    item.state = RankState.PAUSED
            self._admission_open = False
            self._last_message = "SGLang engine is recovering by fault tolerance."
            return self._success_locked("retry started")

    def validate_retry(self) -> Optional[str]:
        """Validate that retry is applicable: no DEAD ranks exist.

        Retry only handles recoverable faults where ranks are merely PAUSED.
        DEAD ranks indicate process exit or kill faults that require scale_down.
        """
        if not self.enabled:
            return "fault tolerance is not enabled"
        with self._lock:
            dead_ranks = [
                item.rank
                for item in self._ranks
                if item.state == RankState.DEAD
            ]
            if dead_ranks:
                return (
                    f"retry cannot be performed when ranks {dead_ranks} are dead; "
                    "use scale_down for process/rank failures"
                )
            return None

    def validate_scale_down(self, ranks: Sequence[int]) -> Optional[str]:
        if not ranks:
            return "scale_down requires non-empty fault_tolerance_params.ranks"
        unknown = [rank for rank in ranks if rank < 0 or rank >= self.dp_size]
        if unknown:
            return f"unknown rank(s): {unknown}"
        active = set(self.active_ranks())
        remaining = active - set(ranks)
        if not remaining:
            return "scale_down cannot isolate all active ranks"
        return None

    def begin_scale_down(self, ranks: Sequence[int]) -> Dict[str, Any]:
        if not self.enabled:
            return self._failure("fault tolerance is not enabled")
        if not self.is_mooncake_backend:
            return self._failure("scale_down requires mooncake backend")
        error = self.validate_scale_down(ranks)
        if error is not None:
            return self._failure(error)

        with self._lock:
            for rank in ranks:
                self._ranks[rank].state = RankState.DEAD
            for item in self._ranks:
                if item.state != RankState.DEAD:
                    item.state = RankState.PAUSED
            self._admission_open = False
            self._last_message = "SGLang engine is recovering by fault tolerance."
            return self._success_locked("scale_down started")

    def commit_scale_down(self) -> Dict[str, Any]:
        if not self.enabled:
            return self._failure("fault tolerance is not enabled")
        with self._lock:
            for item in self._ranks:
                if item.state != RankState.DEAD:
                    item.state = RankState.HEALTHY
            self._admission_open = True
            self._last_message = ""
            return self._success_locked("scale_down succeeded")

    def fail_recovery(self, message: str) -> Dict[str, Any]:
        with self._lock:
            for item in self._ranks:
                if item.state == RankState.HEALTHY:
                    item.state = RankState.PAUSED
            self._admission_open = False
            self._last_message = message
            return self._failure_locked(message)

    def _success_locked(self, message: str) -> Dict[str, Any]:
        return {
            "success": True,
            "message": message,
            "ranks": [rank.to_api() for rank in self._ranks],
        }

    def _failure(self, message: str) -> Dict[str, Any]:
        with self._lock:
            return self._failure_locked(message)

    def _failure_locked(self, message: str) -> Dict[str, Any]:
        return {
            "success": False,
            "message": message,
            "ranks": [rank.to_api() for rank in self._ranks],
        }
