from __future__ import annotations

import dataclasses
import time
import traceback as traceback_lib
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class FaultToleranceState(str, Enum):
    RUNNING = "RUNNING"
    FAULT_DETECTED = "FAULT_DETECTED"
    PAUSING = "PAUSING"
    ABORTING_COMM = "ABORTING_COMM"
    COMM_ABORTED = "COMM_ABORTED"
    PAUSED = "PAUSED"
    RECOVERING = "RECOVERING"
    WAITING_OPERATOR = "WAITING_OPERATOR"
    TERMINATING = "TERMINATING"


class ComponentState(str, Enum):
    HEALTHY = "HEALTHY"
    FAULTED = "FAULTED"
    PAUSED = "PAUSED"
    COMM_ABORTING = "COMM_ABORTING"
    COMM_ABORTED = "COMM_ABORTED"
    RECOVERING = "RECOVERING"
    UNRESPONSIVE = "UNRESPONSIVE"
    EXITED = "EXITED"
    WAITING_OPERATOR = "WAITING_OPERATOR"


@dataclass
class FaultEvent:
    event_id: str
    timestamp: float
    origin: str
    scheduler_id: Optional[int]
    rank: Optional[int]
    fault_type: str
    exception_type: Optional[str]
    message: str
    traceback: Optional[str]
    requires_hard_pause: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        origin: str,
        scheduler_id: Optional[int],
        rank: Optional[int],
        fault_type: str,
        message: str,
        exception_type: Optional[str] = None,
        traceback: Optional[str] = None,
        requires_hard_pause: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "FaultEvent":
        return cls(
            event_id=uuid.uuid4().hex,
            timestamp=time.time(),
            origin=origin,
            scheduler_id=scheduler_id,
            rank=rank,
            fault_type=fault_type,
            exception_type=exception_type,
            message=message,
            traceback=traceback,
            requires_hard_pause=requires_hard_pause,
            metadata=metadata or {},
        )

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        origin: str,
        scheduler_id: Optional[int],
        rank: Optional[int],
        requires_hard_pause: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "FaultEvent":
        return cls.create(
            origin=origin,
            scheduler_id=scheduler_id,
            rank=rank,
            fault_type="exception",
            exception_type=type(exc).__name__,
            message=str(exc),
            traceback="".join(
                traceback_lib.format_exception(type(exc), exc, exc.__traceback__)
            ),
            requires_hard_pause=requires_hard_pause,
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class SentinelStatus:
    scheduler_id: int
    pid: int
    state: ComponentState
    last_heartbeat_ts: float
    last_command_id: Optional[str] = None
    last_fault_event_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        ret = dataclasses.asdict(self)
        ret["state"] = self.state.value
        return ret


@dataclass
class TopologySnapshot:
    world_size: int
    rank: int
    local_rank: int
    backend: str
    tp_size: int
    pp_size: int
    dp_size: int
    ep_size: int
    attn_cp_size: int
    moe_dp_size: int
    moe_ep_size: int
    dist_init_method: str
    nccl_port: int

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
