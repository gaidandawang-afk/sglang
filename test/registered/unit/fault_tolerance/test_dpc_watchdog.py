import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import zmq

from sglang.srt.fault_tolerance import dpc_watchdog as module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_watchdog(processes):
    return module.DPCFaultToleranceWatchdog(
        context=Mock(),
        tokenizer_endpoint="tcp://127.0.0.1:12345",
        node_rank=0,
        processes=processes,
        process_dp_ranks=[i // 2 for i in range(len(processes))],
        process_global_ranks=list(range(len(processes))),
    )


def process(exitcode, alive=False):
    return Mock(exitcode=exitcode, pid=123, is_alive=Mock(return_value=alive))


class TestDPCFaultToleranceWatchdog(unittest.TestCase):
    def test_reports_clean_and_failed_exits_once(self):
        watchdog = make_watchdog(
            [process(0), process(1), process(None), process(None, alive=True)]
        )
        with patch.object(module, "sock_send") as send:
            watchdog._check_processes()
            watchdog._check_processes()
        self.assertEqual(
            [call.args[1].ranks for call in send.call_args_list], [[0], [1]]
        )
        self.assertTrue(all(not call.args[1].active for call in send.call_args_list))

    def test_control_continues_after_all_exits_and_sockets_stay_on_thread(self):
        for processes in ([process(0)], []):
            with self.subTest(processes=processes):
                watchdog = make_watchdog(processes)
                receiver = Mock()
                sender = watchdog._context.socket.return_value
                thread_ids = []
                heartbeats = []
                polled = threading.Event()

                def record(*args, **kwargs):
                    thread_ids.append(threading.get_ident())

                def send(sock, message, **kwargs):
                    record()
                    if isinstance(message, module.WatchdogHeartbeatOutput):
                        heartbeats.append(message)
                        if len(heartbeats) >= 2:
                            polled.set()

                sender.connect.side_effect = record
                sender.close.side_effect = receiver.close.side_effect = record
                with (
                    patch.object(module, "FT_WATCHDOG_POLL_INTERVAL", 0.001),
                    patch.object(module, "get_local_ip_auto", return_value="127.0.0.1"),
                    patch.object(
                        module, "get_zmq_socket_on_host", return_value=(12346, receiver)
                    ),
                    patch.object(module, "sock_recv", side_effect=zmq.Again),
                    patch.object(module, "sock_send", side_effect=send),
                ):
                    watchdog.start()
                    worker = watchdog._thread
                    try:
                        self.assertEqual(
                            watchdog.heartbeat().control_endpoint,
                            "tcp://127.0.0.1:12346",
                        )
                        watchdog.start()
                        self.assertIs(watchdog._thread, worker)
                        self.assertTrue(polled.wait(2))
                    finally:
                        watchdog.stop()
                        worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertEqual(set(thread_ids), {worker.ident})
                sender.close.assert_called_once_with(linger=0)
                receiver.close.assert_called_once_with(linger=0)

    def test_shutdown_kills_only_live_members_of_target_dp(self):
        processes = [
            process(None, True),
            process(None, True),
            process(0),
            process(None, True),
        ]
        watchdog = make_watchdog(processes)
        watchdog._shutdown_dp(SimpleNamespace(target_dp_ranks=[1]))
        for proc in processes[:3]:
            proc.kill.assert_not_called()
        processes[3].kill.assert_called_once_with()

    def test_startup_error_reaches_caller_and_closes_partial_sockets(self):
        watchdog = make_watchdog([process(None, True)])
        receiver = Mock()
        sender = watchdog._context.socket.return_value
        sender.connect.side_effect = RuntimeError("connect failed")
        with (
            patch.object(module, "get_local_ip_auto", return_value="127.0.0.1"),
            patch.object(
                module, "get_zmq_socket_on_host", return_value=(12346, receiver)
            ),
            self.assertLogs(module.logger, level="ERROR"),
        ):
            try:
                with self.assertRaisesRegex(RuntimeError, "connect failed"):
                    watchdog.start()
            finally:
                watchdog.stop()
        receiver.close.assert_called_once_with(linger=0)
        sender.close.assert_called_once_with(linger=0)

    def test_stop_retains_thread_while_exit_send_is_blocked(self):
        watchdog = make_watchdog([process(1)])
        worker = watchdog._thread = Mock()
        worker.is_alive.return_value = True
        watchdog._ready.set()
        watchdog.stop()
        self.assertTrue(watchdog._stop_event.is_set())
        self.assertIs(watchdog._thread, worker)
        watchdog.start()
        worker.start.assert_not_called()

    def test_exit_report_failure_stops_heartbeats_and_cleans_up(self):
        watchdog = make_watchdog([process(1)])
        with (
            patch.object(watchdog, "_on_thread_start"),
            patch.object(watchdog._stop_event, "wait", return_value=False),
            patch.object(module, "sock_send", side_effect=zmq.Again),
            patch.object(watchdog, "_poll_control") as poll,
            patch.object(watchdog, "_close_sockets") as close,
            self.assertLogs(module.logger, level="ERROR"),
        ):
            watchdog._monitor_loop()
        poll.assert_not_called()
        close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
