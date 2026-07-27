# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for SubprocessWatchdog in watchdog.py"""

import multiprocessing as mp
import os
import signal
import threading
import time
import unittest.mock

from sglang.srt.utils.watchdog import SubprocessWatchdog
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="stage-a-test-cpu")


def healthy_worker():
    time.sleep(10)


def crashing_worker():
    os._exit(1)


def slow_crash_worker(delay: float = 0.5):
    time.sleep(delay)
    os._exit(42)


def noop_worker():
    pass


def crash_after_event(event, exit_code: int):
    event.wait()
    os._exit(exit_code)


class TestSubprocessWatchdog(CustomTestCase):
    def setUp(self):
        self.sigquit_triggered = threading.Event()
        self._procs = []
        self._monitor = None

        original_kill = os.kill

        def mock_kill(pid, sig):
            if sig == signal.SIGQUIT:
                self.sigquit_triggered.set()
            else:
                original_kill(pid, sig)

        self._patcher = unittest.mock.patch("os.kill", side_effect=mock_kill)
        self._patcher.start()

    def tearDown(self):
        if self._monitor is not None:
            self._monitor.stop()
        self._patcher.stop()
        for p in self._procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)

    def _spawn(self, target, args=()):
        proc = mp.Process(target=target, args=args)
        proc.start()
        self._procs.append(proc)
        return proc

    def _watch(self, procs, names=None, **kwargs):
        if not isinstance(procs, list):
            procs = [procs]
        kwargs.setdefault("interval", 0.01)
        self._monitor = SubprocessWatchdog(
            processes=procs,
            process_names=names,
            **kwargs,
        )
        self._monitor.start()
        return self._monitor

    def test_healthy_processes_no_sigquit(self):
        proc = self._spawn(healthy_worker)
        self._watch(proc)
        time.sleep(0.5)
        self.assertFalse(self.sigquit_triggered.is_set())

    def test_crashed_process_triggers_sigquit(self):
        proc = self._spawn(slow_crash_worker, args=(0.2,))
        self._watch(proc)
        self.assertTrue(
            self.sigquit_triggered.wait(timeout=5.0),
            "SIGQUIT was not triggered within timeout",
        )

    def test_immediate_crash_detection(self):
        proc = self._spawn(crashing_worker)
        self._watch(proc)
        self.assertTrue(
            self.sigquit_triggered.wait(timeout=5.0),
            "Immediate crash was not detected",
        )

    def test_multiple_processes_one_crashes(self):
        healthy = self._spawn(healthy_worker)
        crashing = self._spawn(slow_crash_worker, args=(0.2,))
        self._watch([healthy, crashing], names=["healthy", "crashing"])
        self.assertTrue(
            self.sigquit_triggered.wait(timeout=5.0),
            "Crash was not detected when one of multiple processes crashed",
        )

    def test_empty_processes_list(self):
        self._watch([])
        time.sleep(0.3)
        self.assertFalse(self.sigquit_triggered.is_set())

    def test_normal_exit_no_sigquit(self):
        proc = self._spawn(noop_worker)
        proc.join(timeout=2)
        self._watch(proc)
        time.sleep(0.3)
        self.assertFalse(
            self.sigquit_triggered.is_set(),
            "SIGQUIT should not be triggered for normal exit (exitcode=0)",
        )

    def test_normal_exit_does_not_call_callback(self):
        proc = self._spawn(noop_worker)
        proc.join(timeout=2)
        exit_seen = threading.Event()
        self._monitor = SubprocessWatchdog(
            processes=[proc],
            on_exit=lambda index, process, name: exit_seen.set(),
        )
        self._monitor.start()

        self.assertFalse(exit_seen.wait(timeout=0.3))
        self.assertFalse(self.sigquit_triggered.is_set())

    def test_normal_exit_can_call_callback_without_sigquit(self):
        proc = self._spawn(noop_worker)
        proc.join(timeout=2)
        exits = []
        exit_seen = threading.Event()

        def record_exit(index, process, name):
            exits.append((index, process.pid, name))
            exit_seen.set()

        self._monitor = SubprocessWatchdog(
            processes=[proc],
            on_exit=record_exit,
            report_clean_exit=True,
            interval=0.01,
        )
        self._monitor.start()

        self.assertTrue(exit_seen.wait(timeout=1.0))
        self.assertEqual(exits, [(0, proc.pid, "process_0")])
        self.assertFalse(self.sigquit_triggered.is_set())

    def test_callback_runs_before_default_sigquit(self):
        exits = []
        exit_seen = threading.Event()
        process = self._spawn(slow_crash_worker, args=(0.1,))

        def record_exit(index, process, name):
            exits.append((index, process.pid, name, self.sigquit_triggered.is_set()))
            exit_seen.set()

        self._monitor = SubprocessWatchdog(
            processes=[process],
            on_exit=record_exit,
        )
        self._monitor.start()

        self.assertTrue(exit_seen.wait(timeout=5.0))
        self.assertEqual(exits, [(0, process.pid, "process_0", False)])
        self.assertTrue(self.sigquit_triggered.wait(timeout=5.0))

    def test_callback_can_isolate_crash_without_default_sigquit(self):
        exits = []
        exit_seen = threading.Event()
        process = self._spawn(slow_crash_worker, args=(0.1,))

        def record_exit(index, process, name):
            exits.append((index, process.pid, name))
            exit_seen.set()

        self._monitor = SubprocessWatchdog(
            processes=[process],
            on_exit=record_exit,
            fail_stop_on_exit=False,
        )
        self._monitor.start()

        self.assertTrue(exit_seen.wait(timeout=5.0))
        self.assertEqual(exits, [(0, process.pid, "process_0")])
        self.assertFalse(self.sigquit_triggered.is_set())

    def test_poll_runs_after_exit_callback_and_continues(self):
        events = []
        exit_seen = threading.Event()
        poll_after_exit = threading.Event()
        process = self._spawn(slow_crash_worker, args=(0.05,))

        def record_exit(index, process, name):
            events.append(("exit", index))
            exit_seen.set()

        def record_poll():
            events.append(("poll", None))
            if exit_seen.is_set():
                poll_after_exit.set()

        self._watch(
            process,
            on_exit=record_exit,
            fail_stop_on_exit=False,
            on_poll=record_poll,
        )

        self.assertTrue(poll_after_exit.wait(timeout=5.0))
        time.sleep(0.05)
        self._monitor.stop()

        exit_index = events.index(("exit", 0))
        self.assertEqual(events.count(("exit", 0)), 1)
        self.assertEqual(events[exit_index + 1], ("poll", None))
        self.assertGreater(
            sum(event == ("poll", None) for event in events[exit_index + 1 :]),
            1,
        )
        self.assertFalse(self.sigquit_triggered.is_set())

    def test_poll_continues_after_all_processes_exit_normally(self):
        process = self._spawn(noop_worker)
        process.join(timeout=2)
        exit_seen = threading.Event()
        three_polls_seen = threading.Event()
        poll_count = 0

        def record_poll():
            nonlocal poll_count
            poll_count += 1
            if poll_count >= 3:
                three_polls_seen.set()

        self._watch(
            process,
            on_exit=lambda index, process, name: exit_seen.set(),
            fail_stop_on_exit=False,
            on_poll=record_poll,
        )

        self.assertTrue(three_polls_seen.wait(timeout=5.0))
        self.assertFalse(exit_seen.is_set())
        self.assertFalse(self.sigquit_triggered.is_set())

    def test_non_fail_stop_reports_multiple_crashes_once_and_keeps_polling(self):
        release_second = mp.Event()
        first = self._spawn(slow_crash_worker, args=(0.05,))
        second = self._spawn(crash_after_event, args=(release_second, 43))
        exits = []
        both_exits_seen = threading.Event()
        poll_after_both = threading.Event()

        def record_exit(index, process, name):
            exits.append((index, process.pid, name))
            if index == 0:
                release_second.set()
            if len(exits) == 2:
                both_exits_seen.set()

        def record_poll():
            if both_exits_seen.is_set():
                poll_after_both.set()

        self._watch(
            [first, second],
            names=["first", "second"],
            on_exit=record_exit,
            fail_stop_on_exit=False,
            on_poll=record_poll,
        )

        self.assertTrue(both_exits_seen.wait(timeout=5.0))
        self.assertTrue(poll_after_both.wait(timeout=5.0))
        time.sleep(0.05)
        self._monitor.stop()

        self.assertEqual(
            exits,
            [
                (0, first.pid, "first"),
                (1, second.pid, "second"),
            ],
        )
        self.assertFalse(self.sigquit_triggered.is_set())

    def test_on_thread_stop_runs_once_in_watchdog_thread(self):
        process = self._spawn(healthy_worker)
        poll_seen = threading.Event()
        stop_seen = threading.Event()
        stop_threads = []

        def record_stop():
            stop_threads.append(threading.get_ident())
            stop_seen.set()

        self._watch(
            process,
            on_poll=poll_seen.set,
            on_thread_stop=record_stop,
        )

        self.assertTrue(poll_seen.wait(timeout=5.0))
        watchdog_thread_id = self._monitor._thread.ident
        self._monitor.stop()
        self.assertTrue(stop_seen.wait(timeout=1.0))
        self._monitor.stop()

        self.assertEqual(stop_threads, [watchdog_thread_id])


if __name__ == "__main__":
    import unittest

    unittest.main()
