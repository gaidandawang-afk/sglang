from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from multiprocessing import Pipe, Process, connection
from typing import Callable, List, Optional

import psutil

from sglang.srt.utils.common import pyspy_dump_schedulers

logger = logging.getLogger(__name__)


class Watchdog:
    @staticmethod
    def create(
        debug_name: str,
        watchdog_timeout: Optional[float],
        soft: bool = False,
        test_stuck_time: float = 0,
    ) -> Watchdog:
        if watchdog_timeout is None:
            assert (
                test_stuck_time == 0
            ), f"stuck tester can be enabled only if soft watchdog is enabled."
            return _WatchdogNoop()
        return _WatchdogReal(
            debug_name=debug_name,
            watchdog_timeout=watchdog_timeout,
            soft=soft,
            test_stuck_time=test_stuck_time,
        )

    def feed(self):
        pass

    @contextmanager
    def disable(self):
        yield


class _WatchdogReal(Watchdog):
    def __init__(
        self,
        debug_name: str,
        watchdog_timeout: float,
        soft: bool = False,
        test_stuck_time: float = 0,
    ):
        self._counter = 0
        self._active = True
        self._test_stuck_time = test_stuck_time
        self._test_stuck_triggered = False
        self._raw = WatchdogRaw(
            debug_name=debug_name,
            get_counter=lambda: self._counter,
            is_active=lambda: self._active,
            watchdog_timeout=watchdog_timeout,
            soft=soft,
        )
        logger.info(f"Watchdog {self._raw.debug_name} initialized.")
        if self._test_stuck_time > 0:
            logger.info(
                f"Watchdog {self._raw.debug_name} is configured to use {test_stuck_time=}."
            )

    def feed(self):
        # Only trigger the test stuck behavior once to avoid blocking server
        # startup health checks while still testing watchdog timeout detection
        if self._test_stuck_time > 0 and not self._test_stuck_triggered:
            self._test_stuck_triggered = True
            logger.info(
                f"Watchdog {self._raw.debug_name} start deliberately stuck for {self._test_stuck_time}s"
            )
            time.sleep(self._test_stuck_time)
            logger.info(
                f"Watchdog {self._raw.debug_name} end deliberately stuck for {self._test_stuck_time}s"
            )

        self._counter += 1

    @contextmanager
    def disable(self):
        assert self._active
        self._active = False
        try:
            yield
        finally:
            assert not self._active
            self._active = True


class _WatchdogNoop(Watchdog):
    pass


class WatchdogRaw:
    def __init__(
        self,
        debug_name: str,
        get_counter: Callable[[], int],
        is_active: Callable[[], bool],
        watchdog_timeout: float,
        soft: bool = False,
        dump_info: Optional[Callable[[], str]] = None,
    ):
        self.debug_name = debug_name
        self.get_counter = get_counter
        self.is_active = is_active
        self.watchdog_timeout = watchdog_timeout
        self.soft = soft
        self.dump_info = dump_info

        self.parent_process = psutil.Process().parent()
        t = threading.Thread(target=self._watchdog_thread, daemon=True)
        t.start()

    def _watchdog_thread(self):
        try:
            while True:
                self._watchdog_once()
        except Exception as e:
            logger.error(
                f"{self.debug_name} watchdog thread crashed: {e}", exc_info=True
            )

    def _watchdog_once(self):
        watchdog_last_counter = 0
        watchdog_last_time = time.perf_counter()

        while True:
            current = time.perf_counter()
            if self.is_active():
                current_counter = self.get_counter()
                if watchdog_last_counter == current_counter:
                    if current > watchdog_last_time + self.watchdog_timeout:
                        break
                else:
                    watchdog_last_counter = current_counter
                    watchdog_last_time = current
            time.sleep(self.watchdog_timeout / 2)

        if self.dump_info is not None and (info_msg := self.dump_info()):
            logger.error(f"{self.debug_name} debug info:\n{info_msg}")

        pyspy_dump_schedulers()
        logger.error(
            f"{self.debug_name} watchdog timeout "
            f"({self.watchdog_timeout=}, {self.soft=})"
        )
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)

        if not self.soft:
            # Wait for some time so that the parent process can print the error.
            time.sleep(5)
            self.parent_process.send_signal(signal.SIGQUIT)


class SubprocessWatchdog:
    """Monitors subprocess sentinels and triggers SIGQUIT when a crash is detected.

    When a subprocess crashes (e.g., NCCL timeout causing C++ std::terminate()),
    Python exception handlers never run, leaving the main process as a zombie
    service. This watchdog waits for multiprocessing exit signals in a daemon
    thread and sends SIGQUIT to trigger proper cleanup.

    See: https://github.com/sgl-project/sglang/issues/18421

    An optional ``on_exit`` callback is invoked before the default SIGQUIT path.
    """

    def __init__(
        self,
        processes: List[Process],
        process_names: Optional[List[str]] = None,
        interval: float = 1.0,
        on_exit: Optional[Callable[[int, Process, str], None]] = None,
    ):
        self._processes = processes
        self._names = process_names or [f"process_{i}" for i in range(len(processes))]
        self._interval = interval
        self._on_exit = on_exit
        self._stop_event = threading.Event()
        self._stop_reader, self._stop_writer = Pipe(duplex=False)
        self._reported = set()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None or not self._processes:
            return
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="subprocess-watchdog"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            try:
                self._stop_writer.send_bytes(b"\0")
            except (BrokenPipeError, EOFError, OSError):
                pass
            self._thread.join(timeout=self._interval * 2)
            self._thread = None

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def _monitor_loop(self) -> None:
        try:
            sentinel_to_process = {
                proc.sentinel: (index, proc, name)
                for index, (proc, name) in enumerate(
                    zip(self._processes, self._names)
                )
            }
            remaining = set(sentinel_to_process)
            while remaining and not self._stop_event.is_set():
                ready = connection.wait([self._stop_reader, *remaining])
                if self._stop_reader in ready:
                    return
                for sentinel in ready:
                    remaining.discard(sentinel)
                    index, proc, name = sentinel_to_process[sentinel]
                    proc.join(timeout=0)
                    if self._handle_process_exit(index, proc, name):
                        return
        except Exception as e:
            logger.error(f"SubprocessWatchdog thread crashed: {e}", exc_info=True)

    def _handle_process_exit(self, index: int, proc: Process, name: str) -> bool:
        if index in self._reported:
            return False
        if proc.exitcode == 0:
            self._reported.add(index)
            return False

        if self._on_exit is not None:
            try:
                self._on_exit(index, proc, name)
            except Exception:
                logger.exception(
                    "Subprocess watchdog on-exit callback failed for %s", name
                )

        self._reported.add(index)
        logger.error(
            f"Subprocess {name} (pid={proc.pid}) crashed "
            f"with exit code {proc.exitcode}. "
            f"Triggering SIGQUIT for cleanup..."
        )
        os.kill(os.getpid(), signal.SIGQUIT)
        return True

    def _check_processes(self) -> bool:
        for index, (proc, name) in enumerate(zip(self._processes, self._names)):
            if index in self._reported or proc.is_alive() or proc.exitcode == 0:
                continue
            if self._handle_process_exit(index, proc, name):
                return True
        return False
