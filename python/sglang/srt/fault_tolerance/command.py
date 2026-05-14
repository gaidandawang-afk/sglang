from __future__ import annotations

import dataclasses
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from sglang.srt.fault_tolerance.state import ComponentState


class SentinelCommandType(str, Enum):
    PAUSE = "pause"
    HARD_ABORT_COMM = "hard_abort_comm"
    PREPARE_RETRY = "prepare_retry"
    RETRY_REINIT = "retry_reinit"
    HEALTH_CHECK = "health_check"
    RESUME = "resume"
    TERMINATE = "terminate"


@dataclass
class SentinelCommand:
    command_id: str
    command: SentinelCommandType
    epoch: int
    timeout_sec: int
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        command: SentinelCommandType,
        *,
        epoch: int,
        timeout_sec: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> "SentinelCommand":
        return cls(
            command_id=uuid.uuid4().hex,
            command=command,
            epoch=epoch,
            timeout_sec=timeout_sec,
            params=params or {},
        )


@dataclass
class SentinelCommandResult:
    command_id: str
    scheduler_id: int
    success: bool
    state: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class SentinelHeartbeat:
    scheduler_id: int
    pid: int
    state: ComponentState
    epoch: int
    timestamp: float = field(default_factory=time.time)
    last_command_id: Optional[str] = None
    last_fault_event_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
