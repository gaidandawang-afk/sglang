import ast
import logging
import sys
import threading
import unittest
from collections import deque
from http import HTTPStatus
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional, Tuple
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[4]


class Struct:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FaultToleranceCommandReqInput(Struct):
    pass


class FaultToleranceCommandReqOutput(Struct):
    pass


class FaultToleranceDPCShutdownReqInput(Struct):
    pass


class FaultToleranceRankFaultOutput(Struct):
    pass


class ProcessActiveRanksOutput(Struct):
    pass


class WatchdogHeartbeatOutput(Struct):
    pass


class AbortReq(Struct):
    pass


class FinishAbort:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_json(self):
        return self.__dict__


class FakeTensor:
    def __init__(self, value):
        self.value = list(value)

    def copy_(self, other):
        self.value = list(other.value)
        return self

    def detach(self):
        return self

    def cpu(self):
        return self


class FakeReq:
    def __init__(self, rid, *, origin_input_ids=None, output_ids=None, committed=0):
        self.rid = rid
        self.finished_reason = None
        self.origin_input_ids = origin_input_ids or []
        self.output_ids = output_ids or []
        self.kv_committed_len = committed

    def finished(self):
        return self.finished_reason is not None


class FakeBatch:
    def __init__(self, reqs, batch_is_full=True):
        self.reqs = list(reqs)
        self.batch_is_full = batch_is_full


def load_class_methods(path, class_name, method_names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    ]
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in method_names}


class Sender:
    def __init__(self):
        self.sent = []
        self.received = []
        self.options = []
        self.endpoint = None
        self.closed = False

    def send_pyobj(self, value, flags=0):
        self.sent.append(value)

    def send_output(self, value, *args):
        self.sent.append((value, *args))

    def recv_pyobj(self, flags=0):
        if not self.received:
            raise RuntimeError
        return self.received.pop(0)

    def setsockopt(self, option, value):
        self.options.append((option, value))

    def connect(self, endpoint):
        self.endpoint = endpoint

    def close(self, linger=None):
        self.closed = True


class FakeContext:
    def __init__(self, sender):
        self.sender = sender

    def socket(self, socket_type):
        return self.sender


class TestSchedulerFaultToleranceControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scheduler = load_class_methods(
            REPO_ROOT / "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            {
                "_run_event_loop_fault_tolerance",
                "handle_fault_tolerance_command",
                "_check_ft_pause_deadline",
                "_ft_discard_inflight_window",
                "_process_next_overlap_result",
            },
            {
                "AbortReq": AbortReq,
                "FINISH_ABORT": FinishAbort,
                "FaultToleranceCommandReqInput": FaultToleranceCommandReqInput,
                "FaultToleranceCommandReqOutput": FaultToleranceCommandReqOutput,
                "FaultToleranceRankFaultOutput": FaultToleranceRankFaultOutput,
                "FT_OPERATION_RETRY": "retry",
                "FT_OPERATION_SCALE_DOWN": "scale_down",
                "HTTPStatus": HTTPStatus,
                "Optional": Optional,
                "ScheduleBatch": FakeBatch,
                "Tuple": Tuple,
                "logger": logging.getLogger(__name__),
                "notify_node_main_process_failure": Mock(),
                "release_kv_cache": lambda *args, **kwargs: None,
                "time": SimpleNamespace(monotonic=lambda: 100.0),
            },
        )
        cls.run_ft_loop = staticmethod(scheduler["_run_event_loop_fault_tolerance"])
        cls.handle_command = staticmethod(scheduler["handle_fault_tolerance_command"])
        cls.check_deadline = staticmethod(scheduler["_check_ft_pause_deadline"])
        cls.discard = staticmethod(scheduler["_ft_discard_inflight_window"])
        cls.process_overlap = staticmethod(scheduler["_process_next_overlap_result"])

        cls.ft_watchdog_globals = {
            "FaultToleranceDPCShutdownReqInput": FaultToleranceDPCShutdownReqInput,
            "ProcessActiveRanksOutput": ProcessActiveRanksOutput,
            "WatchdogHeartbeatOutput": WatchdogHeartbeatOutput,
            "FT_WATCHDOG_SEND_TIMEOUT_MS": 60_000,
            "NetworkAddress": lambda host, port: SimpleNamespace(
                to_tcp=lambda: f"tcp://{host}:{port}"
            ),
            "get_local_ip_auto": lambda fallback: "127.0.0.1",
            "logger": logging.getLogger(__name__),
            "sock_recv": lambda socket, flags=0: socket.recv_pyobj(flags),
            "sock_send": lambda socket, value, flags=0: socket.send_pyobj(value, flags),
            "zmq": SimpleNamespace(
                Context=None,
                PULL="pull",
                PUSH="push",
                LINGER="linger",
                SNDHWM="sndhwm",
                IMMEDIATE="immediate",
                SNDTIMEO="sndtimeo",
                IPV6="ipv6",
                NOBLOCK=1,
                Again=RuntimeError,
            ),
        }
        ft_watchdog = load_class_methods(
            REPO_ROOT / "python/sglang/srt/fault_tolerance/dpc_watchdog.py",
            "DPCFaultToleranceWatchdog",
            {
                "_on_thread_start",
                "heartbeat",
                "_report_process_exit",
                "_poll_control",
                "_shutdown_dp",
                "_close_sockets",
            },
            cls.ft_watchdog_globals,
        )
        cls.ft_watchdog_methods = ft_watchdog

    def make_scheduler(self, *, leader=True):
        model_runner = SimpleNamespace(apply_fault_tolerance_scale_down=Mock())
        return SimpleNamespace(
            ps=SimpleNamespace(
                dp_rank=1,
                attn_tp_rank=0 if leader else 1,
                attn_cp_rank=0,
            ),
            tp_worker=SimpleNamespace(model_runner=model_runner),
            _engine_paused=True,
            _ft_pause_deadline=130.0,
        )

    def test_retry_restores_last_mask_without_replacing_tensors(self):
        state = SimpleNamespace(
            active_ranks=FakeTensor([1, 0]),
            active_ranks_cpu=FakeTensor([1, 0]),
            last_active_ranks=FakeTensor([1, 1]),
        )
        manager = SimpleNamespace(instance=lambda: state)
        module = ModuleType("sglang.srt.elastic_ep.elastic_ep")
        module.ElasticEPStateManager = manager
        modules = {
            "sglang": ModuleType("sglang"),
            "sglang.srt": ModuleType("sglang.srt"),
            "sglang.srt.elastic_ep": ModuleType("sglang.srt.elastic_ep"),
            "sglang.srt.elastic_ep.elastic_ep": module,
        }
        scheduler = self.make_scheduler()
        request = FaultToleranceCommandReqInput(
            request_id="r", command="retry", target_ranks=[1], active_mask=None
        )

        with patch.dict(sys.modules, modules):
            output = self.handle_command(scheduler, request)

        self.assertEqual(state.active_ranks.value, [1, 1])
        self.assertEqual(state.active_ranks_cpu.value, [1, 1])
        self.assertFalse(scheduler._engine_paused)
        self.assertIsNone(scheduler._ft_pause_deadline)
        self.assertEqual((output.request_id, output.rank), ("r", 1))

    def test_scale_down_is_one_command_and_unpauses(self):
        scheduler = self.make_scheduler()
        request = FaultToleranceCommandReqInput(
            request_id="s",
            command="scale_down",
            target_ranks=[1],
            active_mask=[True, False],
        )

        output = self.handle_command(scheduler, request)

        apply_scale_down = (
            scheduler.tp_worker.model_runner.apply_fault_tolerance_scale_down
        )
        apply_scale_down.assert_called_once_with([True, False])
        self.assertEqual((output.request_id, output.rank), ("s", 1))
        self.assertFalse(scheduler._engine_paused)

    def test_nonleader_executes_without_ack(self):
        scheduler = self.make_scheduler(leader=False)
        request = FaultToleranceCommandReqInput(
            request_id="s",
            command="scale_down",
            target_ranks=[1],
            active_mask=[True, False],
        )
        self.assertIsNone(self.handle_command(scheduler, request))
        apply_scale_down = (
            scheduler.tp_worker.model_runner.apply_fault_tolerance_scale_down
        )
        apply_scale_down.assert_called_once()

    def test_exception_self_pause_starts_deadline_before_reporting(self):
        events = []

        def dispatch(_):
            if not events:
                events.append("fault")
                raise RuntimeError("boom")
            raise KeyboardInterrupt()

        self.run_ft_loop.__globals__["dispatch_event_loop"] = dispatch
        sender = SimpleNamespace(send_output=lambda *_: events.append("report"))
        scheduler = SimpleNamespace(
            _ft_discard_inflight_window=lambda exc: True,
            ipc_channels=SimpleNamespace(send_to_tokenizer=sender),
            ps=SimpleNamespace(dp_rank=0),
            server_args=SimpleNamespace(
                fault_tolerance_on_error_strategy="pause",
                fault_tolerance_pause_timeout=30,
            ),
            _engine_paused=False,
            _ft_pause_deadline=None,
        )
        with self.assertRaises(KeyboardInterrupt):
            self.run_ft_loop(scheduler)
        self.assertTrue(scheduler._engine_paused)
        self.assertEqual(scheduler._ft_pause_deadline, 130.0)
        self.assertEqual(events, ["fault", "report"])

    def test_pause_deadline_notifies_node_main_once(self):
        notify = self.check_deadline.__globals__["notify_node_main_process_failure"]
        notify.reset_mock()
        scheduler = SimpleNamespace(
            _ft_pause_deadline=100.0,
            server_args=SimpleNamespace(fault_tolerance_pause_timeout=30),
            ps=SimpleNamespace(dp_rank=0),
        )
        self.check_deadline(scheduler)
        self.check_deadline(scheduler)
        notify.assert_called_once_with()

    def test_exception_discards_overlap_window_once_per_request(self):
        shared = FakeReq("shared")
        current = FakeReq(
            "current", origin_input_ids=list(range(10)), output_ids=[10], committed=12
        )
        previous = FakeBatch([shared])
        running = FakeBatch([shared, current])
        sender = Sender()
        released = []
        self.discard.__globals__["release_kv_cache"] = lambda req, *args, **kwargs: (
            released.append(req.rid)
        )
        scheduler = SimpleNamespace(
            cur_batch_for_debug=FakeBatch([shared, current]),
            last_batch=previous,
            result_queue=deque([(previous, object())]),
            running_batch=running,
            chunked_req=current,
            tree_cache=object(),
            ipc_channels=SimpleNamespace(send_to_tokenizer=sender),
        )

        self.assertTrue(self.discard(scheduler, RuntimeError("boom")))

        self.assertCountEqual(released, ["shared", "current"])
        self.assertEqual(current.kv_committed_len, 11)
        self.assertEqual(scheduler.running_batch.reqs, [])
        self.assertEqual(scheduler.result_queue, deque())

    def make_ft_watchdog(self):
        sender = Sender()
        receiver = Sender()
        context = FakeContext(sender)
        watchdog = SimpleNamespace(
            _context=context,
            _tokenizer_endpoint="tcp://node0:1",
            _node_rank=1,
            _process_dp_ranks=[0, 1, 1],
            _process_global_ranks=[0, 2, 3],
            _processes=[],
            _shutdown_receiver=None,
            _heartbeat_sender=None,
            _ready=threading.Event(),
            _start_error=None,
            control_endpoint=None,
        )
        self.ft_watchdog_methods["_on_thread_start"].__globals__[
            "get_zmq_socket_on_host"
        ] = lambda context, socket_type, host: (2, receiver)
        self.ft_watchdog_methods["_on_thread_start"](watchdog)
        watchdog.heartbeat = lambda: self.ft_watchdog_methods["heartbeat"](watchdog)
        watchdog._shutdown_dp = lambda request: self.ft_watchdog_methods[
            "_shutdown_dp"
        ](watchdog, request)
        return watchdog, sender, receiver

    def test_watchdog_reports_global_rank_and_endpoint(self):
        watchdog, sender, _ = self.make_ft_watchdog()
        proc = SimpleNamespace(pid=123)
        self.ft_watchdog_methods["_report_process_exit"](
            watchdog, 1, proc, "scheduler"
        )
        self.ft_watchdog_methods["_poll_control"](watchdog)

        down, heartbeat = sender.sent
        self.assertIn(("sndtimeo", 60_000), sender.options)
        self.assertEqual(down.ranks, [2])
        self.assertEqual(heartbeat.ranks, [0, 2, 3])
        self.assertEqual(heartbeat.control_endpoint, "tcp://127.0.0.1:2")
        self.assertTrue(watchdog._ready.is_set())

    def test_watchdog_shutdown_kills_every_local_member_of_target_dp(self):
        watchdog, sender, receiver = self.make_ft_watchdog()
        watchdog._processes = [
            SimpleNamespace(is_alive=lambda: True, kill=Mock()) for _ in range(3)
        ]
        receiver.received.append(
            FaultToleranceDPCShutdownReqInput(target_dp_ranks=[1])
        )

        self.ft_watchdog_methods["_poll_control"](watchdog)

        watchdog._processes[0].kill.assert_not_called()
        watchdog._processes[1].kill.assert_called_once()
        watchdog._processes[2].kill.assert_called_once()
        self.assertIsInstance(sender.sent[-1], WatchdogHeartbeatOutput)


if __name__ == "__main__":
    unittest.main()
