from __future__ import annotations

import logging
import threading
from multiprocessing import Process

import zmq
from sglang.srt.managers.io_struct import (
    FaultToleranceDPExitOutput,
    FaultToleranceWatchdogHeartbeatOutput,
    sock_send,
)
from sglang.srt.utils.network import (
    get_zmq_socket,
    is_zmq_endpoint_ipv6,
    zmq_connect,
)

logger = logging.getLogger(__name__)

FT_WATCHDOG_POLL_INTERVAL = 3.0


class DPCFaultToleranceWatchdog:
    """Report child exits and watchdog reachability."""

    def __init__(
        self,
        *,
        context: zmq.Context,
        tokenizer_endpoint: str,
        first_global_rank: int,
        processes: list[Process],
        process_dp_ranks: list[int],
    ) -> None:
        self._context = context
        self._tokenizer_endpoint = tokenizer_endpoint
        self._heartbeat = FaultToleranceWatchdogHeartbeatOutput(
            first_global_rank=first_global_rank,
            dp_ranks=sorted(set(process_dp_ranks)),
        )
        self._children = list(zip(processes, process_dp_ranks))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="ft-dpc-watchdog"
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=FT_WATCHDOG_POLL_INTERVAL * 2)

    def _monitor_loop(self) -> None:
        pending = list(self._children)
        with (
            get_zmq_socket(
                self._context, zmq.PUSH, self._tokenizer_endpoint, False
            ) as sender,
            self._context.socket(zmq.PUSH) as heartbeat_sender,
        ):
            # Heartbeats are best effort and must not queue behind exit reports.
            heartbeat_sender.setsockopt(zmq.LINGER, 0)
            heartbeat_sender.setsockopt(zmq.SNDHWM, 1)
            heartbeat_sender.setsockopt(zmq.IMMEDIATE, 1)
            if is_zmq_endpoint_ipv6(self._tokenizer_endpoint):
                heartbeat_sender.setsockopt(zmq.IPV6, 1)
            zmq_connect(heartbeat_sender, self._tokenizer_endpoint)
            while not self._stop_event.wait(FT_WATCHDOG_POLL_INTERVAL):
                for proc, dp_rank in pending[:]:
                    exitcode = proc.exitcode
                    if exitcode is None:
                        continue
                    logger.warning(
                        "Scheduler for DP rank %s exited (pid=%s, exitcode=%s)",
                        dp_rank,
                        proc.pid,
                        exitcode,
                    )
                    sock_send(sender, FaultToleranceDPExitOutput(dp_rank=dp_rank))
                    pending.remove((proc, dp_rank))
                try:
                    sock_send(heartbeat_sender, self._heartbeat, flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
