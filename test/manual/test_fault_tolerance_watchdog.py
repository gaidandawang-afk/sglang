"""
Test suite for SGLang fault tolerance watchdog (Commit 2).

Covers event-based watchdog with connection.wait:
- Sentinel-based DP=1 monitoring
- DPC event channel for DP>1
- Fallback psutil polling
"""

import multiprocessing as mp
import os
import sys
import time
from multiprocessing import connection
from unittest import mock

import pytest

# Ensure sglang is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.fault_tolerance.controller import (
    FaultToleranceManager,
    RankState,
)


def make_manager(dp_size=4, enabled=True):
    return FaultToleranceManager(
        enabled=enabled,
        dp_size=dp_size,
        on_error_strategy="continue",
        recovery_timeout_sec=300,
        moe_a2a_backend="mooncake",
        elastic_ep_backend=None,
    )


class TestSentinelWait:
    """Verify that connection.wait detects process exit correctly."""

    def test_connection_wait_detects_exit(self):
        """A child process that exits should trigger connection.wait."""
        proc = mp.Process(target=lambda: None)
        proc.start()
        proc.join()

        ready = connection.wait([proc.sentinel], timeout=5.0)
        assert proc.sentinel in ready

    def test_connection_wait_ignores_running(self):
        """A running child should NOT trigger connection.wait within timeout."""
        ready_event = mp.Event()
        proc = mp.Process(target=ready_event.wait)
        proc.start()
        try:
            ready = connection.wait([proc.sentinel], timeout=1.0)
            assert not ready
        finally:
            ready_event.set()
            proc.join(timeout=5)

    def test_multi_sentinel(self):
        """Multiple sentinels: only the exited one triggers."""
        procs = [mp.Process(target=lambda: None) for _ in range(3)]
        for p in procs:
            p.start()
        procs[1].join()

        ready = connection.wait([p.sentinel for p in procs], timeout=5.0)
        assert procs[1].sentinel in ready
        assert procs[0].sentinel not in ready
        assert procs[2].sentinel not in ready

        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)


class TestWatchdogPipeEventChannel:
    """DPC-to-main-process event channel through multiprocessing.Pipe."""

    def test_pipe_send_recv_rank_death_event(self):
        """Round-trip: send rank death event through a pipe."""
        reader, writer = mp.Pipe(duplex=False)

        event = {
            "rank": 2,
            "pid": 12345,
            "exitcode": -9,
            "message": "scheduler rank 2 process exited with code -9",
        }
        writer.send(event)

        assert reader.poll(1.0)
        received = reader.recv()
        assert received["rank"] == 2
        assert received["pid"] == 12345
        assert received["exitcode"] == -9

    def test_pipe_eof_when_closed(self):
        """When writer closes, reader gets EOFError."""
        reader, writer = mp.Pipe(duplex=False)
        writer.close()

        if reader.poll(1.0):
            with pytest.raises(EOFError):
                reader.recv()


class TestFallbackPidPolling:
    """Fallback path: psutil-based polling still works."""

    def test_is_pid_dead_running_process(self):
        """A running process should not be detected as dead."""
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        assert not TokenizerManager._is_fault_tolerance_pid_dead(os.getpid())

    def test_is_pid_dead_nonexistent(self):
        """A nonexistent PID should return True."""
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        dead_pid = 99999999
        assert TokenizerManager._is_fault_tolerance_pid_dead(dead_pid)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

