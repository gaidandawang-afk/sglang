from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Any, Optional

import zmq

from sglang.srt.fault_tolerance.command import (
    SentinelCommand,
    SentinelCommandResult,
    SentinelCommandType,
    SentinelHeartbeat,
)
from sglang.srt.fault_tolerance.distributed_recovery import (
    DistributedRecoveryManager,
)
from sglang.srt.fault_tolerance.state import ComponentState, FaultEvent

logger = logging.getLogger(__name__)


class FaultSentinel:
    """Scheduler-local out-of-band control thread."""

    def __init__(
        self,
        scheduler,
        manager_addr: Optional[str],
        scheduler_id: int,
        heartbeat_interval_sec: float,
        heartbeat_timeout_sec: float,
    ):
        self.scheduler = scheduler
        self.manager_addr = manager_addr
        self.scheduler_id = scheduler_id
        self.pid = os.getpid()
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self.heartbeat_timeout_sec = heartbeat_timeout_sec
        self.state = ComponentState.HEALTHY
        self.epoch = 0
        self.last_command_id: Optional[str] = None
        self.last_fault_event_id: Optional[str] = None

        self.recovery = DistributedRecoveryManager(scheduler)
        self._outbound: "queue.Queue[Any]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_main_loop_heartbeat = time.monotonic()
        self._heartbeat_stall_reported = False

    def start(self) -> None:
        if self.manager_addr is None or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._control_loop,
            daemon=True,
            name=f"fault-sentinel-{self.scheduler_id}",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def feed_main_loop_heartbeat(self) -> None:
        self._last_main_loop_heartbeat = time.monotonic()
        self._heartbeat_stall_reported = False

    def report_fault_from_main_loop(self, event: FaultEvent) -> None:
        self.state = ComponentState.FAULTED
        self.last_fault_event_id = event.event_id
        self._outbound.put(event)

    def _control_loop(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, f"scheduler-{self.scheduler_id}".encode())
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.connect(self.manager_addr)
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            next_heartbeat = 0.0

            while not self._stop_event.is_set():
                now = time.monotonic()
                if now >= next_heartbeat:
                    self._send_heartbeat(socket)
                    next_heartbeat = now + self.heartbeat_interval_sec
                    self._maybe_report_main_loop_stall()

                self._drain_outbound(socket)

                events = dict(poller.poll(100))
                if socket in events:
                    cmd = socket.recv_pyobj()
                    result = self.handle_command(cmd)
                    socket.send_pyobj(result)
        except Exception:
            logger.exception("FaultSentinel control loop crashed")
        finally:
            socket.close(linger=0)
            context.term()

    def _drain_outbound(self, socket) -> None:
        while True:
            try:
                message = self._outbound.get_nowait()
            except queue.Empty:
                return
            socket.send_pyobj(message)

    def _send_heartbeat(self, socket) -> None:
        socket.send_pyobj(
            SentinelHeartbeat(
                scheduler_id=self.scheduler_id,
                pid=self.pid,
                state=self.state,
                epoch=self.epoch,
                last_command_id=self.last_command_id,
                last_fault_event_id=self.last_fault_event_id,
                details={
                    "tp_rank": getattr(self.scheduler, "tp_rank", None),
                    "pp_rank": getattr(self.scheduler, "pp_rank", None),
                    "dp_rank": getattr(self.scheduler, "dp_rank", None),
                },
            )
        )

    def _maybe_report_main_loop_stall(self) -> None:
        if self._heartbeat_stall_reported:
            return
        if time.monotonic() - self._last_main_loop_heartbeat < self.heartbeat_timeout_sec:
            return
        self._heartbeat_stall_reported = True
        self.state = ComponentState.UNRESPONSIVE
        event = FaultEvent.create(
            origin="fault_sentinel",
            scheduler_id=self.scheduler_id,
            rank=getattr(self.scheduler, "tp_rank", None),
            fault_type="heartbeat_stall",
            exception_type=None,
            message="Scheduler main loop heartbeat timed out.",
            traceback=None,
            requires_hard_pause=True,
            metadata={"heartbeat_timeout_sec": self.heartbeat_timeout_sec},
        )
        self.last_fault_event_id = event.event_id
        self._outbound.put(event)

    def handle_command(self, cmd: SentinelCommand) -> SentinelCommandResult:
        self.last_command_id = cmd.command_id
        self.epoch = max(self.epoch, cmd.epoch)
        try:
            if cmd.command == SentinelCommandType.PAUSE:
                self.scheduler._engine_paused = True
                self.state = ComponentState.PAUSED
                return self._result(cmd, True, "Scheduler paused.")

            if cmd.command == SentinelCommandType.HARD_ABORT_COMM:
                self.scheduler._engine_paused = True
                self.state = ComponentState.COMM_ABORTING
                result = self.recovery.abort_communicators(cmd.timeout_sec)
                self.state = (
                    ComponentState.COMM_ABORTED
                    if result.success
                    else ComponentState.WAITING_OPERATOR
                )
                return self._result(
                    cmd, result.success, result.message, result.to_dict()
                )

            if cmd.command == SentinelCommandType.PREPARE_RETRY:
                self.state = ComponentState.RECOVERING
                return self._request_on_main_loop(cmd)

            if cmd.command in (
                SentinelCommandType.RETRY_REINIT,
                SentinelCommandType.HEALTH_CHECK,
                SentinelCommandType.RESUME,
                SentinelCommandType.TERMINATE,
            ):
                return self._request_on_main_loop(cmd)

            return self._result(cmd, False, f"Unsupported command {cmd.command}.")
        except Exception as exc:
            logger.exception("FaultSentinel command failed")
            self.state = ComponentState.WAITING_OPERATOR
            return self._result(cmd, False, str(exc))

    def _request_on_main_loop(self, cmd: SentinelCommand) -> SentinelCommandResult:
        if not hasattr(self.scheduler, "fault_tolerance_submit_command"):
            return self._result(cmd, False, "Scheduler has no recovery command queue.")
        result = self.scheduler.fault_tolerance_submit_command(cmd)
        try:
            self.state = ComponentState(result.state)
        except ValueError:
            pass
        return result

    def _result(
        self,
        cmd: SentinelCommand,
        success: bool,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> SentinelCommandResult:
        return SentinelCommandResult(
            command_id=cmd.command_id,
            scheduler_id=self.scheduler_id,
            success=success,
            state=self.state.value,
            message=message,
            details=details or {},
        )
