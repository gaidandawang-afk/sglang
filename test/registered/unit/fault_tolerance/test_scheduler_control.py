import ast
from collections import deque
from http import HTTPStatus
import logging
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple
from unittest.mock import Mock

REPO_ROOT = Path(__file__).resolve().parents[4]


class Struct:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FaultToleranceCommandReqInput(Struct):
    pass


class FaultToleranceCommandReqOutput(Struct):
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


class FakeReq:
    def __init__(
        self,
        rid,
        *,
        origin_input_ids=None,
        output_ids=None,
        kv_committed_len=0,
        kv_allocated_len=0,
    ):
        self.rid = rid
        self.finished_reason = None
        self.origin_input_ids = origin_input_ids or []
        self.output_ids = output_ids or []
        self.kv_committed_len = kv_committed_len
        self.kv_allocated_len = kv_allocated_len

    def finished(self):
        return self.finished_reason is not None


class FakeBatch:
    def __init__(self, reqs, batch_is_full=True):
        self.reqs = list(reqs)
        self.batch_is_full = batch_is_full


def load_class_methods(path: Path, class_name: str, method_names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = [
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    ]
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in method_names}


def load_functions(path: Path, function_names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in function_names
    ]
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in function_names}


class GatherGroup:
    def __init__(self, label, events, extra_results=None):
        self.label = label
        self.events = events
        self.extra_results = extra_results or []

    def all_gather_object(self, value):
        self.events.append(self.label)
        return [value, *self.extra_results]


class Sender:
    def __init__(self):
        self.sent = []
        self.send_flags = []
        self.socket_options = []
        self.connected_endpoint = None
        self.fail_all_sends = False
        self.fail_nonblocking_sends = False
        self.closed = False

    def send_pyobj(self, value, flags=0):
        if self.fail_all_sends:
            raise RuntimeError("send failed")
        if flags and self.fail_nonblocking_sends:
            raise FakeAgain()
        self.sent.append(value)
        self.send_flags.append(flags)

    def send_output(self, value, *args):
        self.sent.append((value, *args))

    def setsockopt(self, option, value):
        self.socket_options.append((option, value))

    def connect(self, endpoint):
        self.connected_endpoint = endpoint

    def close(self, linger=None):
        self.closed = True


class FakeAgain(Exception):
    pass


class FakeContext:
    def __init__(self):
        self.terminated = False
        self.socket_types = []

    def socket(self, socket_type):
        self.socket_types.append(socket_type)
        return DPC_RUNTIME.sender

    def term(self):
        self.terminated = True


DPC_RUNTIME = SimpleNamespace(sender=None, context=None, context_count=0)


def make_context():
    DPC_RUNTIME.context = FakeContext()
    DPC_RUNTIME.context_count += 1
    return DPC_RUNTIME.context


class TestSchedulerFaultToleranceControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scheduler_methods = load_class_methods(
            REPO_ROOT / "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            {
                "handle_fault_tolerance_command",
                "_update_ft_pause_from_mlp_sync",
                "_complete_ft_pause",
                "_check_ft_pause_deadline",
                "_ft_discard_inflight_window",
                "_process_next_overlap_result",
            },
            {
                "AbortReq": AbortReq,
                "FINISH_ABORT": FinishAbort,
                "FaultToleranceCommandReqInput": FaultToleranceCommandReqInput,
                "FaultToleranceCommandReqOutput": FaultToleranceCommandReqOutput,
                "FT_ACTION_PAUSE_READY": 1,
                "HTTPStatus": HTTPStatus,
                "Optional": Optional,
                "ScheduleBatch": FakeBatch,
                "Tuple": Tuple,
                "logger": logging.getLogger(__name__),
                "notify_node_main_process_failure": Mock(),
                "os": os,
                "release_kv_cache": lambda *args, **kwargs: None,
                "time": SimpleNamespace(monotonic=lambda: 100.0),
            },
        )
        cls.handle_command = staticmethod(
            scheduler_methods["handle_fault_tolerance_command"]
        )
        cls.update_pause_from_mlp_sync = staticmethod(
            scheduler_methods["_update_ft_pause_from_mlp_sync"]
        )
        cls.complete_pause = staticmethod(scheduler_methods["_complete_ft_pause"])
        cls.check_pause_deadline = staticmethod(
            scheduler_methods["_check_ft_pause_deadline"]
        )
        cls.discard_inflight = staticmethod(
            scheduler_methods["_ft_discard_inflight_window"]
        )
        cls.process_next_overlap = staticmethod(
            scheduler_methods["_process_next_overlap_result"]
        )

        dpc_methods = load_class_methods(
            REPO_ROOT / "python/sglang/srt/managers/data_parallel_controller.py",
            "DataParallelController",
            {
                "send_fault_tolerance_command",
                "_get_watchdog_sender",
                "_handle_scheduler_process_exit",
                "_watchdog_heartbeat",
                "_report_initial_watchdog_heartbeat",
                "_report_watchdog_heartbeat",
                "_close_watchdog_sender",
                "_report_process_active_ranks",
            },
            {
                "FaultToleranceCommandReqInput": FaultToleranceCommandReqInput,
                "ProcessActiveRanksOutput": ProcessActiveRanksOutput,
                "WatchdogHeartbeatOutput": WatchdogHeartbeatOutput,
                "FT_WATCHDOG_SEND_TIMEOUT_MS": 1000,
                "logger": logging.getLogger(__name__),
                "sock_send": lambda socket, value, flags=0: socket.send_pyobj(
                    value, flags=flags
                ),
                "zmq": SimpleNamespace(
                    Context=make_context,
                    PUSH="push",
                    LINGER="linger",
                    SNDHWM="sndhwm",
                    IMMEDIATE="immediate",
                    SNDTIMEO="sndtimeo",
                    IPV6="ipv6",
                    NOBLOCK=1,
                    Again=FakeAgain,
                ),
            },
        )
        cls.send_dpc_command = staticmethod(dpc_methods["send_fault_tolerance_command"])
        cls.get_watchdog_sender = staticmethod(
            dpc_methods["_get_watchdog_sender"]
        )
        cls.handle_process_exit = staticmethod(
            dpc_methods["_handle_scheduler_process_exit"]
        )
        cls.watchdog_heartbeat = staticmethod(
            dpc_methods["_watchdog_heartbeat"]
        )
        cls.report_initial_watchdog_heartbeat = staticmethod(
            dpc_methods["_report_initial_watchdog_heartbeat"]
        )
        cls.report_watchdog_heartbeat = staticmethod(
            dpc_methods["_report_watchdog_heartbeat"]
        )
        cls.close_watchdog_sender = staticmethod(
            dpc_methods["_close_watchdog_sender"]
        )
        cls.report_process_active = staticmethod(
            dpc_methods["_report_process_active_ranks"]
        )

        common_functions = load_functions(
            REPO_ROOT / "python/sglang/srt/utils/common.py",
            {"notify_node_main_process_failure"},
            {},
        )
        cls.notify_node_main_failure = staticmethod(
            common_functions["notify_node_main_process_failure"]
        )

    def setUp(self):
        DPC_RUNTIME.sender = Sender()
        DPC_RUNTIME.context = None
        DPC_RUNTIME.context_count = 0
        self.check_pause_deadline.__globals__[
            "notify_node_main_process_failure"
        ].reset_mock()

    def make_watchdog_dpc(
        self,
        scheduler_process_dp_ranks,
        tokenizer_ipc_name="tcp://node0:1",
    ):
        dpc = SimpleNamespace(
            scheduler_process_dp_ranks=scheduler_process_dp_ranks,
            server_args=SimpleNamespace(node_rank=3),
            port_args=SimpleNamespace(tokenizer_ipc_name=tokenizer_ipc_name),
            send_to_tokenizer=Sender(),
            _watchdog_context=None,
            _watchdog_sender=None,
        )
        dpc._get_watchdog_sender = lambda: self.get_watchdog_sender(dpc)
        dpc._watchdog_heartbeat = lambda: self.watchdog_heartbeat(dpc)
        return dpc

    def make_scheduler(self, *, attn_tp_rank=0, remote_failure=False):
        events = []
        sender = Sender()
        scheduler = SimpleNamespace(
            tp_rank=0,
            tp_size=4,
            attn_tp_size=2,
            attn_cp_size=2,
            ps=SimpleNamespace(
                dp_rank=1,
                attn_tp_rank=attn_tp_rank,
                attn_cp_rank=0,
            ),
            server_args=SimpleNamespace(
                enable_dp_attention=True,
                fault_tolerance_pause_timeout=30,
            ),
            _engine_paused=False,
            _ft_pending_pause=None,
            _ft_pause_deadline=None,
            _ft_rank=lambda: 1,
            send_to_tokenizer=sender,
            ipc_channels=SimpleNamespace(send_to_tokenizer=sender),
        )
        cp_remote = {"tp_rank": 2, "success": True, "message": "paused"}
        tp_remote = [
            {
                "tp_rank": 1,
                "success": not remote_failure,
                "message": "boom" if remote_failure else "paused",
            },
            {"tp_rank": 3, "success": True, "message": "paused"},
        ]
        scheduler.attn_cp_group = GatherGroup("attn_cp", events, [cp_remote])
        scheduler.attn_tp_group = GatherGroup("attn_tp", events, [tp_remote])
        scheduler.tp_group = None
        return scheduler, events

    def test_pause_waits_for_all_target_dp_ranks_before_ack(self):
        scheduler, events = self.make_scheduler()
        request = FaultToleranceCommandReqInput(
            request_id="request",
            command="pause",
            target_ranks=[0, 1, 2],
        )

        output = self.handle_command(scheduler, request)

        self.assertIsNone(output)
        self.assertEqual(events, [])
        self.assertIs(scheduler._ft_pending_pause, request)
        self.assertFalse(scheduler._engine_paused)
        self.assertEqual(scheduler.send_to_tokenizer.sent, [])

        self.update_pause_from_mlp_sync(scheduler, [1, 1, 0])
        self.assertFalse(scheduler._engine_paused)

        self.update_pause_from_mlp_sync(scheduler, [1, 1, 1])
        self.assertTrue(scheduler._engine_paused)
        self.assertIs(scheduler._ft_pending_pause, request)
        self.assertEqual(scheduler.send_to_tokenizer.sent, [])

        self.complete_pause(scheduler)
        self.assertIsNone(scheduler._ft_pending_pause)
        self.assertEqual(scheduler._ft_pause_deadline, 130.0)
        self.assertEqual(len(scheduler.send_to_tokenizer.sent), 1)
        ack, original_request = scheduler.send_to_tokenizer.sent[0]
        self.assertEqual(ack.request_id, "request")
        self.assertEqual(ack.rank, 1)
        self.assertTrue(ack.success)
        self.assertIs(original_request, request)

    def test_nonleader_executes_without_ack(self):
        scheduler, _ = self.make_scheduler(attn_tp_rank=1)
        request = FaultToleranceCommandReqInput(
            request_id="request",
            command="resume",
            target_ranks=[1],
        )

        self.assertIsNone(self.handle_command(scheduler, request))
        self.assertFalse(scheduler._engine_paused)
        self.assertIsNone(scheduler._ft_pause_deadline)

    def test_leader_returns_local_resume_ack(self):
        scheduler, _ = self.make_scheduler(remote_failure=True)
        request = FaultToleranceCommandReqInput(
            request_id="request",
            command="resume",
            target_ranks=[1],
        )

        output = self.handle_command(scheduler, request)

        self.assertTrue(output.success)
        self.assertEqual(output.message, "resumed")

    def test_resume_clears_pause_deadline(self):
        scheduler, _ = self.make_scheduler()
        scheduler._engine_paused = True
        scheduler._ft_pause_deadline = 130.0
        request = FaultToleranceCommandReqInput(
            request_id="request",
            command="resume",
            target_ranks=[1],
        )

        self.handle_command(scheduler, request)

        self.assertFalse(scheduler._engine_paused)
        self.assertIsNone(scheduler._ft_pause_deadline)

    def test_pause_deadline_notifies_node_main_once(self):
        scheduler, _ = self.make_scheduler()
        scheduler._ft_pause_deadline = 101.0
        notify = self.check_pause_deadline.__globals__[
            "notify_node_main_process_failure"
        ]

        self.check_pause_deadline(scheduler)
        notify.assert_not_called()

        scheduler._ft_pause_deadline = 100.0
        self.check_pause_deadline(scheduler)
        self.check_pause_deadline(scheduler)

        notify.assert_called_once_with()
        self.assertIsNone(scheduler._ft_pause_deadline)

    def test_notify_node_main_process_failure_signals_scheduler_grandparent(self):
        signals = []
        node_main = SimpleNamespace(send_signal=signals.append)
        dpc = SimpleNamespace(parent=lambda: node_main)
        scheduler = SimpleNamespace(parent=lambda: dpc)
        notify = self.notify_node_main_failure
        notify.__globals__.update(
            psutil=SimpleNamespace(Process=lambda: scheduler),
            signal=SimpleNamespace(SIGQUIT="SIGQUIT"),
        )

        notify()

        self.assertEqual(signals, ["SIGQUIT"])

    def test_exception_discards_the_overlap_window_once_per_request(self):
        shared = FakeReq("shared")
        previous_only = FakeReq("previous")
        current_only = FakeReq(
            "current",
            origin_input_ids=list(range(10)),
            output_ids=[10],
            kv_committed_len=12,
            kv_allocated_len=12,
        )
        waiting = FakeReq("waiting")
        previous = FakeBatch([shared, previous_only])
        current = FakeBatch([shared, current_only])
        running = FakeBatch([shared, previous_only, current_only])
        sender = Sender()
        released = []
        release_options = {}
        scheduler = SimpleNamespace(
            cur_batch_for_debug=current,
            last_batch=previous,
            result_queue=deque([(previous, object())]),
            running_batch=running,
            chunked_req=current_only,
            waiting_queue=[waiting],
            tree_cache=object(),
            ipc_channels=SimpleNamespace(send_to_tokenizer=sender),
        )

        def record_release(req, *args, **kwargs):
            released.append(req.rid)
            release_options[req.rid] = kwargs

        self.discard_inflight.__globals__["release_kv_cache"] = record_release

        self.assertTrue(self.discard_inflight(scheduler, RuntimeError("boom")))

        self.assertCountEqual(released, ["shared", "previous", "current"])
        self.assertTrue(
            all(
                options["allow_non_spec_overallocated"]
                for options in release_options.values()
            )
        )
        self.assertEqual(current_only.kv_committed_len, 11)
        self.assertEqual(len(sender.sent), 3)
        self.assertEqual(
            {entry[0].rid for entry in sender.sent},
            {"shared", "previous", "current"},
        )
        for abort, req in sender.sent:
            self.assertEqual(
                abort.finished_reason["status_code"], HTTPStatus.SERVICE_UNAVAILABLE
            )
            self.assertEqual(abort.finished_reason["err_type"], "SchedulerFault")
            self.assertIs(
                req.finished_reason.status_code, HTTPStatus.SERVICE_UNAVAILABLE
            )
        self.assertEqual(scheduler.running_batch.reqs, [])
        self.assertEqual(scheduler.waiting_queue, [waiting])
        self.assertEqual(scheduler.result_queue, deque())
        self.assertIsNone(scheduler.cur_batch_for_debug)
        self.assertIsNone(scheduler.last_batch)
        self.assertIsNone(scheduler.chunked_req)

    def test_overlap_result_is_only_popped_after_successful_processing(self):
        batch = FakeBatch([])
        result = object()
        scheduler = SimpleNamespace(
            result_queue=deque([(batch, result)]),
            process_batch_result=lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        with self.assertRaisesRegex(RuntimeError, "boom"):
            self.process_next_overlap(scheduler)
        self.assertEqual(list(scheduler.result_queue), [(batch, result)])

        scheduler.process_batch_result = lambda *_: None
        self.process_next_overlap(scheduler)
        self.assertEqual(scheduler.result_queue, deque())

    def test_dpc_forwards_command_to_target_dp_leader(self):
        workers = [Sender(), Sender()]
        dpc = SimpleNamespace(workers=workers)
        request = FaultToleranceCommandReqInput(
            request_id="request",
            command="resume",
            target_ranks=[1],
        )

        self.send_dpc_command(dpc, request)

        self.assertEqual(workers[0].sent, [])
        self.assertEqual(workers[1].sent, [request])

    def test_watchdog_heartbeat_and_process_exit_share_persistent_sender(self):
        dpc = self.make_watchdog_dpc([2, 2, 3])
        process = SimpleNamespace(pid=123)

        self.assertIsNone(self.handle_process_exit(dpc, 0, process, "scheduler"))
        self.assertIsNone(self.report_watchdog_heartbeat(dpc))

        self.assertEqual(DPC_RUNTIME.context_count, 1)
        self.assertIs(dpc._watchdog_sender, DPC_RUNTIME.sender)
        self.assertIs(dpc._watchdog_context, DPC_RUNTIME.context)
        self.assertEqual(len(DPC_RUNTIME.sender.sent), 2)

        process_down, heartbeat = DPC_RUNTIME.sender.sent
        self.assertIsInstance(process_down, ProcessActiveRanksOutput)
        self.assertEqual(process_down.ranks, [2])
        self.assertFalse(process_down.active)
        self.assertIsInstance(heartbeat, WatchdogHeartbeatOutput)
        self.assertEqual(heartbeat.node_rank, 3)
        self.assertEqual(heartbeat.ranks, [2, 3])
        self.assertEqual(DPC_RUNTIME.sender.send_flags, [0, 1])
        self.assertEqual(
            DPC_RUNTIME.sender.socket_options,
            [
                ("linger", 0),
                ("sndhwm", 1),
                ("immediate", 1),
                ("sndtimeo", 1000),
            ],
        )
        self.assertEqual(
            DPC_RUNTIME.sender.connected_endpoint,
            "tcp://node0:1",
        )
        self.assertFalse(DPC_RUNTIME.sender.closed)
        self.assertFalse(DPC_RUNTIME.context.terminated)

        self.close_watchdog_sender(dpc)

        self.assertTrue(DPC_RUNTIME.sender.closed)
        self.assertTrue(DPC_RUNTIME.context.terminated)
        self.assertIsNone(dpc._watchdog_sender)
        self.assertIsNone(dpc._watchdog_context)

    def test_initial_watchdog_heartbeat_uses_main_thread_sender(self):
        dpc = self.make_watchdog_dpc([2, 2, 3])

        self.assertIsNone(self.report_initial_watchdog_heartbeat(dpc))

        self.assertEqual(len(dpc.send_to_tokenizer.sent), 1)
        heartbeat = dpc.send_to_tokenizer.sent[0]
        self.assertIsInstance(heartbeat, WatchdogHeartbeatOutput)
        self.assertEqual(heartbeat.node_rank, 3)
        self.assertEqual(heartbeat.ranks, [2, 3])
        self.assertEqual(DPC_RUNTIME.context_count, 0)

    def test_dpc_registers_lease_before_starting_watchdog(self):
        path = (
            REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "managers"
            / "data_parallel_controller.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dpc_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "DataParallelController"
        )
        init = next(
            node
            for node in dpc_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        calls = sorted(
            (
                node.lineno,
                node.func.attr,
            )
            for node in ast.walk(init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        )

        initial_heartbeat_line = next(
            line
            for line, name in calls
            if name == "_report_initial_watchdog_heartbeat"
        )
        watchdog_start_line = next(
            line
            for line, name in calls
            if name == "start"
            and line > initial_heartbeat_line
        )
        self.assertLess(initial_heartbeat_line, watchdog_start_line)

        watchdog_constructor = next(
            node
            for node in ast.walk(init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SubprocessWatchdog"
        )
        keywords = {
            keyword.arg: keyword.value
            for keyword in watchdog_constructor.keywords
        }
        self.assertTrue(ast.literal_eval(keywords["report_clean_exit"]))

    def test_watchdog_heartbeat_drops_nonblocking_send_failure(self):
        dpc = self.make_watchdog_dpc([1])
        DPC_RUNTIME.sender.fail_nonblocking_sends = True

        self.assertIsNone(self.report_watchdog_heartbeat(dpc))

        self.assertEqual(DPC_RUNTIME.sender.sent, [])
        self.assertFalse(DPC_RUNTIME.sender.closed)

    def test_watchdog_heartbeat_unexpected_failure_stops_watchdog(self):
        dpc = self.make_watchdog_dpc([1])
        DPC_RUNTIME.sender.fail_all_sends = True

        with self.assertRaisesRegex(RuntimeError, "send failed"):
            self.report_watchdog_heartbeat(dpc)

    def test_process_exit_send_failure_stops_watchdog_heartbeat(self):
        dpc = self.make_watchdog_dpc([1])
        DPC_RUNTIME.sender.fail_all_sends = True

        with self.assertRaisesRegex(RuntimeError, "send failed"):
            self.handle_process_exit(
                dpc,
                0,
                SimpleNamespace(pid=123),
                "scheduler",
            )

    def test_watchdog_sender_enables_ipv6_before_connect(self):
        dpc = self.make_watchdog_dpc(
            [1],
            tokenizer_ipc_name="tcp://[::1]:2000",
        )

        self.get_watchdog_sender(dpc)

        self.assertIn(("ipv6", 1), DPC_RUNTIME.sender.socket_options)
        self.assertEqual(
            DPC_RUNTIME.sender.connected_endpoint,
            "tcp://[::1]:2000",
        )

    def test_rejoined_dpc_reports_all_local_dp_ranks_active(self):
        sender = Sender()
        dpc = SimpleNamespace(
            scheduler_process_dp_ranks=[2, 2, 3],
            send_to_tokenizer=sender,
        )

        self.report_process_active(dpc, active=True)

        self.assertEqual(len(sender.sent), 1)
        self.assertEqual(sender.sent[0].ranks, [2, 3])
        self.assertTrue(sender.sent[0].active)


if __name__ == "__main__":
    unittest.main()
