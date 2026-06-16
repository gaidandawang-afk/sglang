from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class RankState(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
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
        self._pending_scale_down_ranks = set()

    @property
    def is_mooncake_backend(self) -> bool:
        return (
            self.moe_a2a_backend == "mooncake"
            or self.elastic_ep_backend == "mooncake"
        )

    def status_response(self) -> Dict[str, Any]:
        with self._lock:
            active_mask = self._service_active_mask_locked()
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
            return self._service_active_mask_locked()

    def active_ranks(self) -> List[int]:
        with self._lock:
            return [
                rank.rank
                for rank in self._ranks
                if rank.state == RankState.HEALTHY
            ]

    def alive_mask(self) -> List[bool]:
        with self._lock:
            return self._alive_mask_locked()

    def alive_ranks(self) -> List[int]:
        with self._lock:
            return [
                rank.rank
                for rank in self._ranks
                if rank.state != RankState.DEAD
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
                    f"routed_dp_rank={rank} is isolated by fault tolerance"
                )

    def sync_active_ranks(self, active_mask: Sequence[bool], message: str = "") -> None:
        if not self.enabled or len(active_mask) != self.dp_size:
            return
        with self._lock:
            changed = False
            for rank, is_active in enumerate(active_mask):
                if is_active or self._ranks[rank].state == RankState.DEAD:
                    continue
                if self._ranks[rank].state != RankState.UNHEALTHY:
                    self._ranks[rank].state = RankState.UNHEALTHY
                    changed = True
            if not changed:
                return

            self._last_fault = FaultStatus(
                rank=None,
                type="active_ranks",
                message=message or "Mooncake active_ranks isolated rank(s)",
            )
            if self.on_error_strategy == "pause":
                for item in self._ranks:
                    if item.state == RankState.HEALTHY:
                        item.state = RankState.PAUSED
                self._instance_state = InstanceState.PAUSED
                self._admission_open = False
                self._last_message = (
                    message or "SGLang engine is paused by fault tolerance."
                )
            else:
                self._set_degraded_or_failed_locked(
                    message or "Mooncake active_ranks isolated rank(s)"
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
            if rank is not None and 0 <= rank < self.dp_size:
                if self._ranks[rank].state != RankState.DEAD:
                    self._ranks[rank].state = RankState.UNHEALTHY
            self._last_fault = FaultStatus(
                rank=rank,
                type=fault_type,
                message=message or "scheduler rank fault",
            )
            if self.on_error_strategy == "pause":
                for item in self._ranks:
                    if item.state == RankState.HEALTHY:
                        item.state = RankState.PAUSED
                self._instance_state = InstanceState.PAUSED
                self._admission_open = False
                self._last_message = (
                    message or "SGLang engine is paused by fault tolerance."
                )
            else:
                self._set_degraded_or_failed_locked(message or "scheduler rank fault")

    def record_fault(self, rank: int, message: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            if not 0 <= rank < self.dp_size:
                logger.warning("Ignoring FT fault for unknown rank %s", rank)
                return
            self._ranks[rank].state = RankState.DEAD
            self._last_fault = FaultStatus(
                rank=rank,
                type="non_recoverable",
                message=message or "scheduler process exited",
            )
            if self.on_error_strategy == "pause":
                for item in self._ranks:
                    if item.state == RankState.HEALTHY:
                        item.state = RankState.PAUSED
                self._instance_state = (
                    InstanceState.PAUSED
                    if any(item.state != RankState.DEAD for item in self._ranks)
                    else InstanceState.FAILED
                )
                self._admission_open = False
                self._last_message = (
                    message or "SGLang engine is paused by fault tolerance."
                )
            else:
                self._set_degraded_or_failed_locked(
                    message or "scheduler process exited"
                )

    def validate_retry(self) -> Optional[str]:
        if not self.enabled:
            return "fault tolerance is not enabled"
        with self._lock:
            if self._instance_state == InstanceState.RECOVERING:
                return "fault tolerance recovery is already in progress"
            if self._instance_state in (
                InstanceState.RUNNING,
                InstanceState.DEGRADED_RUNNING,
            ):
                return "already_running"
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
            return None

    def begin_retry(self) -> Dict[str, Any]:
        error = self.validate_retry()
        if error is not None:
            return self._failure(error)
        with self._lock:
            self._pending_scale_down_ranks.clear()
            for item in self._ranks:
                if item.state != RankState.DEAD:
                    item.state = RankState.PAUSED
            self._instance_state = InstanceState.RECOVERING
            self._admission_open = False
            self._last_message = "SGLang engine is recovering by fault tolerance."
            return self._success_locked("retry started")

    def commit_retry(self) -> Dict[str, Any]:
        if not self.enabled:
            return self._failure("fault tolerance is not enabled")
        with self._lock:
            for item in self._ranks:
                if item.state == RankState.PAUSED:
                    item.state = RankState.HEALTHY
            self._pending_scale_down_ranks.clear()
            self._instance_state = InstanceState.RUNNING
            self._admission_open = True
            self._last_message = ""
            return self._success_locked("retry succeeded")

    def validate_scale_down(self, ranks: Sequence[int]) -> Optional[str]:
        if not self.enabled:
            return "fault tolerance is not enabled"
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
            self._pending_scale_down_ranks = set(ranks)
            for rank in self._pending_scale_down_ranks:
                if self._ranks[rank].state not in (
                    RankState.UNHEALTHY,
                    RankState.DEAD,
                ):
                    self._ranks[rank].state = RankState.DEAD
            for item in self._ranks:
                if (
                    item.rank not in self._pending_scale_down_ranks
                    and item.state != RankState.DEAD
                ):
                    item.state = RankState.PAUSED
            self._instance_state = InstanceState.RECOVERING
            self._admission_open = False
            self._last_message = "SGLang engine is recovering by fault tolerance."
            return self._success_locked("scale_down started")

    def commit_scale_down(self) -> Dict[str, Any]:
        if not self.enabled:
            return self._failure("fault tolerance is not enabled")
        with self._lock:
            pending = set(self._pending_scale_down_ranks)
            for item in self._ranks:
                if item.rank not in pending and item.state != RankState.DEAD:
                    item.state = RankState.HEALTHY
            self._pending_scale_down_ranks.clear()
            self._set_running_or_degraded_locked()
            self._last_message = ""
            return self._success_locked("scale_down succeeded")

    def fail_recovery(self, message: str) -> Dict[str, Any]:
        with self._lock:
            self._pending_scale_down_ranks.clear()
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

    def recovery_active_mask(self) -> List[bool]:
        with self._lock:
            if self._pending_scale_down_ranks:
                return [
                    item.rank not in self._pending_scale_down_ranks
                    and item.state != RankState.DEAD
                    for item in self._ranks
                ]
            return self._alive_mask_locked()

    def _service_active_mask_locked(self) -> List[bool]:
        return [rank.state == RankState.HEALTHY for rank in self._ranks]

    def _alive_mask_locked(self) -> List[bool]:
        return [rank.state != RankState.DEAD for rank in self._ranks]

    def _set_running_or_degraded_locked(self) -> None:
        if any(item.state == RankState.HEALTHY for item in self._ranks):
            self._instance_state = (
                InstanceState.DEGRADED_RUNNING
                if any(item.state != RankState.HEALTHY for item in self._ranks)
                else InstanceState.RUNNING
            )
            self._admission_open = True
        else:
            self._instance_state = InstanceState.FAILED
            self._admission_open = False

    def _set_degraded_or_failed_locked(self, message: str) -> None:
        if any(item.state == RankState.HEALTHY for item in self._ranks):
            self._instance_state = InstanceState.DEGRADED_RUNNING
            self._admission_open = True
            self._last_message = message
        else:
            self._instance_state = InstanceState.FAILED
            self._admission_open = False
            self._last_message = message
