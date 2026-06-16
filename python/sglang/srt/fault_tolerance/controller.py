from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class RankState(str, Enum):
    HEALTHY = "healthy"
    PAUSED = "paused"
    DEAD = "dead"


class InstanceState(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    RECOVERING = "recovering"
    DEGRADED_RUNNING = "degraded_running"
    FAILED = "failed"


@dataclass
class RankStatus:
    rank: int
    state: RankState

    def to_api(self) -> Dict[str, Any]:
        return {"rank": self.rank, "state": self.state.value}


@dataclass
class FaultStatus:
    rank: Optional[int]
    type: str
    message: str

    def to_api(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "type": self.type,
            "message": self.message,
        }


class FaultToleranceManager:
    """Main-process FT state machine for the v4 management API."""

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
        self._instance_state = InstanceState.RUNNING
        self._admission_open = True
        self._last_fault: Optional[FaultStatus] = None
        self._last_message = ""
        self._recovery_snapshot: Optional[
            Tuple[List[RankState], InstanceState, bool, Optional[FaultStatus], str]
        ] = None

    @property
    def is_mooncake_backend(self) -> bool:
        return (
            self.moe_a2a_backend == "mooncake"
            or self.elastic_ep_backend == "mooncake"
        )

    def status_response(self) -> Dict[str, Any]:
        with self._lock:
            active_mask = self._active_mask_locked()
            return {
                "enabled": self.enabled,
                "instance_state": self._instance_state.value,
                "admission_open": self._admission_open,
                "ranks": [rank.to_api() for rank in self._ranks],
                "active_mask": active_mask,
                "last_fault": (
                    self._last_fault.to_api()
                    if self._last_fault is not None
                    else None
                ),
            }

    def admission_open(self) -> bool:
        if not self.enabled:
            return True
        with self._lock:
            return self._admission_open

    def recovery_in_progress(self) -> bool:
        with self._lock:
            return self._instance_state == InstanceState.RECOVERING

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
            return self._active_mask_locked()

    def live_mask(self) -> List[bool]:
        with self._lock:
            return self._live_mask_locked()

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
                rank.rank for rank in self._ranks if rank.state == RankState.PAUSED
            ]

    def dead_ranks(self) -> List[int]:
        with self._lock:
            return [
                rank.rank for rank in self._ranks if rank.state == RankState.DEAD
            ]

    def validate_routed_rank(self, rank: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            if 0 <= rank < self.dp_size and self._ranks[rank].state != RankState.HEALTHY:
                raise ValueError(
                    f"routed_dp_rank={rank} is unavailable by fault tolerance"
                )

    def pause_all_active(
        self,
        message: str,
        *,
        rank: Optional[int] = None,
        fault_type: str = "recoverable",
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self.on_error_strategy == "continue" and rank is not None:
                if 0 <= rank < self.dp_size and self._ranks[rank].state != RankState.DEAD:
                    self._ranks[rank].state = RankState.PAUSED
                self._instance_state = (
                    InstanceState.DEGRADED_RUNNING
                    if any(item.state == RankState.HEALTHY for item in self._ranks)
                    else InstanceState.PAUSED
                )
                self._admission_open = self._instance_state == InstanceState.DEGRADED_RUNNING
            else:
                for item in self._ranks:
                    if item.state == RankState.HEALTHY:
                        item.state = RankState.PAUSED
                self._instance_state = InstanceState.PAUSED
                self._admission_open = False
            self._last_fault = FaultStatus(
                rank=rank,
                type=fault_type,
                message=message or "scheduler rank fault",
            )
            self._last_message = message or "SGLang engine is degraded by fault tolerance."

    def record_fault(self, rank: int, message: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            if not 0 <= rank < self.dp_size:
                logger.warning("Ignoring FT fault for unknown rank %s", rank)
                return
            self._ranks[rank].state = RankState.DEAD
            if self.on_error_strategy == "continue":
                self._instance_state = (
                    InstanceState.DEGRADED_RUNNING
                    if any(item.state == RankState.HEALTHY for item in self._ranks)
                    else InstanceState.FAILED
                )
                self._admission_open = self._instance_state == InstanceState.DEGRADED_RUNNING
            else:
                for item in self._ranks:
                    if item.state == RankState.HEALTHY:
                        item.state = RankState.PAUSED
                self._instance_state = (
                    InstanceState.PAUSED
                    if any(item.state != RankState.DEAD for item in self._ranks)
                    else InstanceState.FAILED
                )
                self._admission_open = False
            self._last_fault = FaultStatus(
                rank=rank,
                type="non_recoverable",
                message=message or "scheduler process exited",
            )
            self._last_message = message or "SGLang engine is degraded by fault tolerance."

    def validate_retry(self) -> Optional[str]:
        if not self.enabled:
            return "fault tolerance is not enabled"
        if self.on_error_strategy == "continue":
            return "retry is not supported when fault_tolerance_on_error_strategy=continue"
        with self._lock:
            if self._instance_state == InstanceState.RECOVERING:
                return "fault tolerance recovery is already in progress"
            if self._instance_state == InstanceState.FAILED:
                return "fault tolerance instance is failed"
            dead_ranks = [
                item.rank for item in self._ranks if item.state == RankState.DEAD
            ]
            if dead_ranks:
                return (
                    f"retry cannot be performed when ranks {dead_ranks} are dead; "
                    "use scale_down for process/rank failures"
                )
            if not any(item.state == RankState.PAUSED for item in self._ranks):
                return "already_running"
            return None

    def begin_retry(self) -> Dict[str, Any]:
        error = self.validate_retry()
        if error is not None:
            return self._failure(error)
        with self._lock:
            self._snapshot_recovery_locked()
            if self.on_error_strategy != "continue":
                for item in self._ranks:
                    if item.state != RankState.DEAD:
                        item.state = RankState.PAUSED
            self._instance_state = InstanceState.RECOVERING
            self._admission_open = self.on_error_strategy == "continue"
            self._last_message = "SGLang engine is recovering by fault tolerance."
            return self._success_locked("retry started")

    def commit_retry(self) -> Dict[str, Any]:
        if not self.enabled:
            return self._failure("fault tolerance is not enabled")
        with self._lock:
            for item in self._ranks:
                if item.state == RankState.PAUSED:
                    item.state = RankState.HEALTHY
            self._instance_state = InstanceState.RUNNING
            self._admission_open = True
            self._last_message = ""
            self._recovery_snapshot = None
            return self._success_locked("retry succeeded")

    def validate_scale_down(self, ranks: Sequence[int]) -> Optional[str]:
        if not self.enabled:
            return "fault tolerance is not enabled"
        if self.on_error_strategy == "continue":
            return (
                "scale_down is not supported when "
                "fault_tolerance_on_error_strategy=continue"
            )
        if not self.is_mooncake_backend:
            return "scale_down requires mooncake backend"
        if not ranks:
            return "scale_down requires non-empty fault_tolerance_params.ranks"
        requested = set(ranks)
        unknown = [rank for rank in requested if rank < 0 or rank >= self.dp_size]
        if unknown:
            return f"unknown rank(s): {unknown}"
        with self._lock:
            if self._instance_state == InstanceState.RECOVERING:
                return "fault tolerance recovery is already in progress"
            if self._instance_state == InstanceState.FAILED:
                return "fault tolerance instance is failed"
            active = {
                item.rank for item in self._ranks if item.state != RankState.DEAD
            }
            if not active - requested:
                return "scale_down cannot isolate all active ranks"
        return None

    def begin_scale_down(self, ranks: Sequence[int]) -> Dict[str, Any]:
        error = self.validate_scale_down(ranks)
        if error is not None:
            return self._failure(error)
        with self._lock:
            self._snapshot_recovery_locked()
            for rank in set(ranks):
                self._ranks[rank].state = RankState.DEAD
            if self.on_error_strategy != "continue":
                for item in self._ranks:
                    if item.state != RankState.DEAD:
                        item.state = RankState.PAUSED
            self._instance_state = InstanceState.RECOVERING
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
            self._instance_state = (
                InstanceState.DEGRADED_RUNNING
                if any(item.state == RankState.DEAD for item in self._ranks)
                else InstanceState.RUNNING
            )
            self._admission_open = True
            self._last_message = ""
            self._recovery_snapshot = None
            return self._success_locked("scale_down succeeded")

    def rollback_recovery(self, message: str) -> Dict[str, Any]:
        with self._lock:
            if self._recovery_snapshot is None:
                self._instance_state = InstanceState.FAILED
                self._admission_open = False
                self._last_message = message
                return self._failure_locked(message)
            states, instance_state, admission_open, last_fault, last_message = (
                self._recovery_snapshot
            )
            for item, state in zip(self._ranks, states):
                item.state = state
            self._instance_state = instance_state
            self._admission_open = admission_open
            self._last_fault = last_fault
            self._last_message = last_message or message
            self._recovery_snapshot = None
            return self._failure_locked(message)

    def fail_recovery(self, message: str) -> Dict[str, Any]:
        with self._lock:
            self._instance_state = InstanceState.FAILED
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

    def _active_mask_locked(self) -> List[bool]:
        return [rank.state == RankState.HEALTHY for rank in self._ranks]

    def _live_mask_locked(self) -> List[bool]:
        return [rank.state != RankState.DEAD for rank in self._ranks]

    def _snapshot_recovery_locked(self) -> None:
        self._recovery_snapshot = (
            [item.state for item in self._ranks],
            self._instance_state,
            self._admission_open,
            self._last_fault,
            self._last_message,
        )
