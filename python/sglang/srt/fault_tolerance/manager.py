from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import pickle
import queue
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional

import zmq

from sglang.srt.fault_tolerance.command import (
    SentinelCommand,
    SentinelCommandResult,
    SentinelCommandType,
    SentinelHeartbeat,
)
from sglang.srt.fault_tolerance.exceptions import FaultToleranceDisabledError
from sglang.srt.fault_tolerance.state import (
    ComponentState,
    FaultEvent,
    FaultToleranceState,
    SentinelStatus,
)
from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
)

logger = logging.getLogger(__name__)


class SentinelManager:
    """Main-process fault tolerance state machine and sentinel command router."""

    def __init__(
        self,
        server_args,
        tokenizer_manager=None,
        port_args=None,
        terminate_callback: Optional[Callable[[str], None]] = None,
    ):
        self.server_args = server_args
        self.tokenizer_manager = tokenizer_manager
        self.port_args = port_args
        self.terminate_callback = terminate_callback
        self.enabled = bool(getattr(server_args, "enable_fault_tolerance", False))

        self.state = (
            FaultToleranceState.RUNNING if self.enabled else FaultToleranceState.PAUSED
        )
        self.epoch = 0
        self.accepting_requests = self.enabled
        self.sentinels: Dict[int, SentinelStatus] = {}
        self.components: Dict[str, Dict[str, Any]] = {}
        self.last_fault: Optional[FaultEvent] = None
        self.last_apply_result: Optional[Dict[str, Any]] = None

        self._lock = threading.RLock()
        self._result_cond = threading.Condition(self._lock)
        self._identities: Dict[int, bytes] = {}
        self._results: Dict[str, List[SentinelCommandResult]] = defaultdict(list)
        self._outgoing: "queue.Queue[tuple[SentinelCommand, Optional[List[int]]]]" = (
            queue.Queue()
        )
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._reported_fault_event_ids: set[str] = set()

    def set_tokenizer_manager(self, tokenizer_manager) -> None:
        self.tokenizer_manager = tokenizer_manager

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        if self.port_args is None:
            logger.warning("Fault tolerance is enabled but PortArgs is unavailable.")
            return
        endpoint = getattr(self.port_args, "fault_tolerance_ipc_name", None)
        if not endpoint:
            logger.warning("Fault tolerance is enabled but no control endpoint exists.")
            return

        self._thread = threading.Thread(
            target=self._router_loop,
            args=(endpoint,),
            daemon=True,
            name="sentinel-manager",
        )
        self._thread.start()
        self._ready_event.wait(timeout=5)
        logger.info("Fault tolerance SentinelManager started at %s", endpoint)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _router_loop(self, endpoint: str) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.bind(endpoint)
            self._ready_event.set()
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)

            while not self._stop_event.is_set():
                self._drain_outgoing(socket)
                events = dict(poller.poll(100))
                if socket in events:
                    identity, payload = socket.recv_multipart()[:2]
                    message = pickle.loads(payload)
                    self._handle_message(identity, message)
        except Exception:
            logger.exception("SentinelManager router loop crashed")
        finally:
            socket.close(linger=0)
            context.term()

    def _drain_outgoing(self, socket) -> None:
        while True:
            try:
                cmd, scheduler_ids = self._outgoing.get_nowait()
            except queue.Empty:
                return

            with self._lock:
                target_ids = list(scheduler_ids or self._identities.keys())
                if not target_ids:
                    self._result_cond.notify_all()
                    continue
                for scheduler_id in target_ids:
                    identity = self._identities.get(scheduler_id)
                    if identity is None:
                        self._record_command_result_locked(
                            SentinelCommandResult(
                                command_id=cmd.command_id,
                                scheduler_id=scheduler_id,
                                success=False,
                                state=ComponentState.UNRESPONSIVE.value,
                                message="Sentinel has not registered.",
                            )
                        )
                        continue
                    status = self.sentinels.get(scheduler_id)
                    if status is not None:
                        status.last_command_id = cmd.command_id
                    socket.send_multipart([identity, pickle.dumps(cmd)])

    def _handle_message(self, identity: bytes, message: Any) -> None:
        if isinstance(message, SentinelHeartbeat):
            with self._lock:
                self._identities[message.scheduler_id] = identity
                self.sentinels[message.scheduler_id] = SentinelStatus(
                    scheduler_id=message.scheduler_id,
                    pid=message.pid,
                    state=message.state,
                    last_heartbeat_ts=message.timestamp,
                    last_command_id=message.last_command_id,
                    last_fault_event_id=message.last_fault_event_id,
                    details=message.details,
                )
            return

        if isinstance(message, SentinelCommandResult):
            with self._lock:
                self._record_command_result_locked(message)
            return

        if isinstance(message, FaultEvent):
            self.report_fault_sync(message)
            return

        logger.warning("Unknown fault tolerance message: %s", type(message))

    def _record_command_result_locked(self, result: SentinelCommandResult) -> None:
        self._results[result.command_id].append(result)
        status = self.sentinels.get(result.scheduler_id)
        if status is not None:
            status.last_command_id = result.command_id
            with contextlib.suppress(ValueError):
                status.state = ComponentState(result.state)
            status.details["last_command_message"] = result.message
            status.details["last_command_success"] = result.success
        self._result_cond.notify_all()

    def report_fault_sync(self, event: FaultEvent) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.last_fault = event
            self._reported_fault_event_ids.add(event.event_id)
            if event.scheduler_id is not None:
                status = self.sentinels.get(event.scheduler_id)
                if status is not None:
                    status.state = ComponentState.FAULTED
                    status.last_fault_event_id = event.event_id

            if self.state == FaultToleranceState.RUNNING:
                self.epoch += 1
                self.state = FaultToleranceState.FAULT_DETECTED
                self._freeze_admission_locked()
                if (
                    event.requires_hard_pause
                    or self.server_args.fault_tolerance_hard_pause_on_fault
                ):
                    self.state = FaultToleranceState.ABORTING_COMM
                    cmd = SentinelCommand.create(
                        SentinelCommandType.HARD_ABORT_COMM,
                        epoch=self.epoch,
                        timeout_sec=(
                            self.server_args.fault_tolerance_comm_abort_timeout_sec
                        ),
                        params={"reason": "fault", "event_id": event.event_id},
                    )
                    self._outgoing.put((cmd, None))

        logger.error(
            "Fault tolerance captured fault %s from %s: %s",
            event.event_id,
            event.origin,
            event.message,
        )

    def mark_component_exited(
        self, name: str, pid: Optional[int], exitcode: Optional[int]
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.components[name] = {
                "state": ComponentState.EXITED.value,
                "pid": pid,
                "exitcode": exitcode,
                "timestamp": time.time(),
            }
            if self.state == FaultToleranceState.RUNNING:
                self.epoch += 1
                self.state = FaultToleranceState.WAITING_OPERATOR
                self._freeze_admission_locked()

    def mark_unresponsive(self, scheduler_id: int, reason: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            status = self.sentinels.get(scheduler_id)
            if status is not None:
                status.state = ComponentState.UNRESPONSIVE
                status.details["unresponsive_reason"] = reason
            if self.state == FaultToleranceState.RUNNING:
                self.epoch += 1
                self.state = FaultToleranceState.WAITING_OPERATOR
                self._freeze_admission_locked()

    def get_status(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "state": "DISABLED"}
        with self._lock:
            return {
                "enabled": True,
                "state": self.state.value,
                "epoch": self.epoch,
                "accepting_requests": self.accepting_requests,
                "topology": self._topology_snapshot_locked(),
                "sentinels": {
                    scheduler_id: status.to_dict()
                    for scheduler_id, status in self.sentinels.items()
                },
                "components": dict(self.components),
                "last_fault": self.last_fault.to_dict() if self.last_fault else None,
                "last_apply_result": self.last_apply_result,
            }

    async def apply(
        self,
        instruction: str,
        timeout: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise FaultToleranceDisabledError("Fault tolerance is disabled.")
        params = params or {}
        timeout = timeout or self.server_args.fault_tolerance_recovery_timeout_sec
        instruction = instruction.lower()

        if instruction == "pause":
            result = await self.pause(
                hard=bool(params.get("hard", True)),
                mode=params.get(
                    "mode", self.server_args.fault_tolerance_default_pause_mode
                ),
                timeout=timeout,
                reason=params.get("reason", "operator"),
            )
        elif instruction == "retry":
            result = await self.retry(timeout=timeout, params=params)
        elif instruction == "terminate":
            result = await self.terminate(reason=params.get("reason", "operator"))
        else:
            result = {
                "success": False,
                "message": f"Unsupported fault tolerance instruction: {instruction}",
                "state": self.state.value,
            }

        with self._lock:
            self.last_apply_result = result
        return result

    async def pause(
        self, hard: bool, mode: str, timeout: int, reason: str
    ) -> Dict[str, Any]:
        with self._lock:
            self.epoch += 1
            self.state = FaultToleranceState.PAUSING
            self._freeze_admission_locked()

        await self._best_effort_tokenizer_pause(mode)

        if hard:
            with self._lock:
                self.state = FaultToleranceState.ABORTING_COMM
            results = self._issue_command(
                SentinelCommandType.HARD_ABORT_COMM,
                timeout_sec=timeout,
                params={"reason": reason},
            )
            success = self._all_success(results)
            with self._lock:
                self.state = (
                    FaultToleranceState.COMM_ABORTED
                    if success
                    else FaultToleranceState.WAITING_OPERATOR
                )
        else:
            results = self._issue_command(
                SentinelCommandType.PAUSE,
                timeout_sec=timeout,
                params={"reason": reason, "mode": mode},
            )
            success = self._all_success(results)
            with self._lock:
                self.state = (
                    FaultToleranceState.PAUSED
                    if success
                    else FaultToleranceState.WAITING_OPERATOR
                )

        return self._apply_result(success, f"pause hard={hard}", results)

    async def retry(self, timeout: int, params: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self.epoch += 1
            self.state = FaultToleranceState.RECOVERING
            self._freeze_admission_locked()

        all_results: List[SentinelCommandResult] = []
        prepare = self._issue_command(
            SentinelCommandType.PREPARE_RETRY,
            timeout_sec=timeout,
            params=params,
        )
        all_results.extend(prepare)

        if params.get("reinit_distributed", True):
            abort = self._issue_command(
                SentinelCommandType.HARD_ABORT_COMM,
                timeout_sec=self.server_args.fault_tolerance_comm_abort_timeout_sec,
                params={"reason": "retry"},
            )
            all_results.extend(abort)

        reinit = self._issue_command(
            SentinelCommandType.RETRY_REINIT,
            timeout_sec=timeout,
            params=params,
        )
        all_results.extend(reinit)

        health = self._issue_command(
            SentinelCommandType.HEALTH_CHECK,
            timeout_sec=timeout,
            params=params,
        )
        all_results.extend(health)

        success = self._all_success(all_results)
        if success:
            resume = self._issue_command(
                SentinelCommandType.RESUME,
                timeout_sec=timeout,
                params=params,
            )
            all_results.extend(resume)
            success = self._all_success(all_results)

        if success:
            await self._best_effort_tokenizer_continue(params)
            with self._lock:
                self.state = FaultToleranceState.RUNNING
                self._open_admission_locked()
            message = "retry succeeded"
        else:
            with self._lock:
                self.state = FaultToleranceState.WAITING_OPERATOR
            message = "retry failed"
            if self.server_args.shutdown_on_fault_tolerance_failure:
                await self.terminate("retry failed")

        return self._apply_result(success, message, all_results)

    async def terminate(self, reason: str) -> Dict[str, Any]:
        with self._lock:
            self.state = FaultToleranceState.TERMINATING
            self._freeze_admission_locked()

        results = self._issue_command(
            SentinelCommandType.TERMINATE,
            timeout_sec=self.server_args.fault_tolerance_sentinel_cmd_timeout_sec,
            params={"reason": reason},
        )
        if self.terminate_callback is not None:
            self.terminate_callback(reason)
        return self._apply_result(True, f"terminate requested: {reason}", results)

    def _issue_command(
        self,
        command_type: SentinelCommandType,
        *,
        timeout_sec: int,
        params: Optional[Dict[str, Any]] = None,
        scheduler_ids: Optional[Iterable[int]] = None,
    ) -> List[SentinelCommandResult]:
        with self._lock:
            expected = self._target_scheduler_ids_locked(scheduler_ids)
            cmd = SentinelCommand.create(
                command_type,
                epoch=self.epoch,
                timeout_sec=timeout_sec,
                params=params or {},
            )
            registered = []
            for scheduler_id in expected:
                if scheduler_id in self._identities:
                    registered.append(scheduler_id)
                else:
                    self._record_command_result_locked(
                        SentinelCommandResult(
                            command_id=cmd.command_id,
                            scheduler_id=scheduler_id,
                            success=False,
                            state=ComponentState.UNRESPONSIVE.value,
                            message="Sentinel has not registered.",
                        )
                    )

            if registered:
                self._outgoing.put((cmd, registered))
            deadline = time.monotonic() + timeout_sec
            while time.monotonic() < deadline:
                received = {
                    result.scheduler_id for result in self._results[cmd.command_id]
                }
                if set(expected).issubset(received):
                    break
                self._result_cond.wait(timeout=0.1)

            received = {result.scheduler_id for result in self._results[cmd.command_id]}
            missing = [
                scheduler_id
                for scheduler_id in registered
                if scheduler_id not in received
            ]
            for scheduler_id in missing:
                self._record_command_result_locked(
                    SentinelCommandResult(
                        command_id=cmd.command_id,
                        scheduler_id=scheduler_id,
                        success=False,
                        state=ComponentState.UNRESPONSIVE.value,
                        message=f"Command {command_type.value} timed out.",
                    )
                )
            return list(self._results.pop(cmd.command_id, []))

    def _target_scheduler_ids_locked(
        self, scheduler_ids: Optional[Iterable[int]]
    ) -> List[int]:
        if scheduler_ids is not None:
            return list(scheduler_ids)
        return self._expected_scheduler_ids_locked()

    def _expected_scheduler_ids_locked(self) -> List[int]:
        dp_size = int(getattr(self.server_args, "dp_size", 1) or 1)
        pp_size = int(getattr(self.server_args, "pp_size", 1) or 1)
        tp_size = int(getattr(self.server_args, "tp_size", 1) or 1)
        attn_cp_size = int(getattr(self.server_args, "attn_cp_size", 1) or 1)
        total = max(1, dp_size) * max(1, pp_size) * max(1, tp_size) * max(
            1, attn_cp_size
        )
        return list(range(total))

    def _freeze_admission_locked(self) -> None:
        self.accepting_requests = False
        tokenizer_manager = self.tokenizer_manager
        if tokenizer_manager is not None and hasattr(tokenizer_manager, "is_pause"):
            tokenizer_manager.is_pause = True

    def _open_admission_locked(self) -> None:
        self.accepting_requests = True
        tokenizer_manager = self.tokenizer_manager
        if tokenizer_manager is not None and hasattr(tokenizer_manager, "is_pause"):
            tokenizer_manager.is_pause = False
            cond = getattr(tokenizer_manager, "is_pause_cond", None)
            if cond is not None:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._notify_pause_cond(cond))
                except RuntimeError:
                    pass

    async def _notify_pause_cond(self, cond) -> None:
        async with cond:
            cond.notify_all()

    async def _best_effort_tokenizer_pause(self, mode: str) -> None:
        tokenizer_manager = self.tokenizer_manager
        if tokenizer_manager is None or not hasattr(tokenizer_manager, "pause_generation"):
            return
        try:
            await tokenizer_manager.pause_generation(PauseGenerationReqInput(mode=mode))
        except Exception:
            logger.exception("Tokenizer pause path failed during fault tolerance pause")

    async def _best_effort_tokenizer_continue(self, params: Dict[str, Any]) -> None:
        tokenizer_manager = self.tokenizer_manager
        if tokenizer_manager is None or not hasattr(
            tokenizer_manager, "continue_generation"
        ):
            return
        try:
            await tokenizer_manager.continue_generation(
                ContinueGenerationReqInput(
                    torch_empty_cache=params.get("torch_empty_cache", True)
                )
            )
        except Exception:
            logger.exception("Tokenizer continue path failed during fault tolerance retry")

    def _topology_snapshot_locked(self) -> Dict[str, Any]:
        return {
            "tp_size": getattr(self.server_args, "tp_size", None),
            "pp_size": getattr(self.server_args, "pp_size", None),
            "dp_size": getattr(self.server_args, "dp_size", None),
            "ep_size": getattr(self.server_args, "ep_size", None),
            "attn_cp_size": getattr(self.server_args, "attn_cp_size", None),
            "moe_dp_size": getattr(self.server_args, "moe_dp_size", None),
            "nnodes": getattr(self.server_args, "nnodes", None),
            "node_rank": getattr(self.server_args, "node_rank", None),
        }

    def _apply_result(
        self, success: bool, message: str, results: List[SentinelCommandResult]
    ) -> Dict[str, Any]:
        return {
            "success": success,
            "message": message,
            "state": self.state.value,
            "epoch": self.epoch,
            "results": [result.to_dict() for result in results],
        }

    @staticmethod
    def _all_success(results: List[SentinelCommandResult]) -> bool:
        return bool(results) and all(result.success for result in results)


def default_terminate_callback(reason: str) -> None:
    del reason
    os.kill(os.getpid(), 15)
