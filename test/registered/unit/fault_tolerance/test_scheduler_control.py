import ast
import logging
import os
import signal
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple
from unittest.mock import patch


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


class RecoveredDPRanksOutput(Struct):
    pass


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
        self.closed = False

    def send_pyobj(self, value):
        self.sent.append(value)

    def send_output(self, value):
        self.sent.append(value)

    def close(self, linger=None):
        self.closed = True


class FakeContext:
    def __init__(self):
        self.terminated = False

    def term(self):
        self.terminated = True


DPC_RUNTIME = SimpleNamespace(sender=None, context=None, sleeps=[])


def make_context():
    DPC_RUNTIME.context = FakeContext()
    return DPC_RUNTIME.context


def get_zmq_socket(context, socket_type, endpoint, bind):
    return DPC_RUNTIME.sender


class TestSchedulerFaultToleranceControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scheduler_methods = load_class_methods(
            REPO_ROOT / "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            {
                "handle_fault_tolerance_command",
                "_aggregate_ft_command_result",
                "_report_recovered_dp_ranks",
            },
            {
                "FaultToleranceCommandReqInput": FaultToleranceCommandReqInput,
                "FaultToleranceCommandReqOutput": FaultToleranceCommandReqOutput,
                "Optional": Optional,
                "RecoveredDPRanksOutput": RecoveredDPRanksOutput,
                "Tuple": Tuple,
                "logger": logging.getLogger(__name__),
                "os": os,
                "signal": signal,
            },
        )
        cls.handle_command = staticmethod(
            scheduler_methods["handle_fault_tolerance_command"]
        )
        cls.aggregate_result = staticmethod(
            scheduler_methods["_aggregate_ft_command_result"]
        )
        cls.report_recovered = staticmethod(
            scheduler_methods["_report_recovered_dp_ranks"]
        )

        dpc_methods = load_class_methods(
            REPO_ROOT / "python/sglang/srt/managers/data_parallel_controller.py",
            "DataParallelController",
            {
                "send_fault_tolerance_command",
                "_handle_scheduler_process_exit",
                "_report_process_active_ranks",
            },
            {
                "FaultToleranceCommandReqInput": FaultToleranceCommandReqInput,
                "ProcessActiveRanksOutput": ProcessActiveRanksOutput,
                "FT_PROCESS_EXIT_GRACE_PERIOD": 2,
                "logger": logging.getLogger(__name__),
                "zmq": SimpleNamespace(Context=make_context, PUSH=object()),
                "get_zmq_socket": get_zmq_socket,
                "time": SimpleNamespace(
                    sleep=lambda seconds: DPC_RUNTIME.sleeps.append(seconds)
                ),
            },
        )
        cls.send_dpc_command = staticmethod(
            dpc_methods["send_fault_tolerance_command"]
        )
        cls.handle_process_exit = staticmethod(
            dpc_methods["_handle_scheduler_process_exit"]
        )
        cls.report_process_active = staticmethod(
            dpc_methods["_report_process_active_ranks"]
        )

    def setUp(self):
        DPC_RUNTIME.sender = Sender()
        DPC_RUNTIME.context = None
        DPC_RUNTIME.sleeps = []

    def make_scheduler(self, *, attn_tp_rank=0, remote_failure=False):
        events = []
        scheduler = SimpleNamespace(
            tp_rank=0,
            tp_size=4,
            attn_tp_rank=attn_tp_rank,
            attn_tp_size=2,
            attn_cp_rank=0,
            attn_cp_size=2,
            server_args=SimpleNamespace(enable_dp_attention=True),
            _engine_paused=False,
            _ft_rank=lambda: 1,
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
        scheduler._aggregate_ft_command_result = lambda success, message: (
            self.aggregate_result(scheduler, success, message)
        )
        return scheduler, events

    def test_pause_is_aggregated_before_leader_ack(self):
        scheduler, events = self.make_scheduler()
        request = FaultToleranceCommandReqInput(
            request_id="request",
            command="pause",
            target_ranks=[1],
        )

        output = self.handle_command(scheduler, request)

        self.assertEqual(events, ["attn_cp", "attn_tp"])
        self.assertTrue(scheduler._engine_paused)
        self.assertTrue(output.success)
        self.assertEqual(output.rank, 1)

    def test_nonleader_executes_without_ack(self):
        scheduler, _ = self.make_scheduler(attn_tp_rank=1)
        request = FaultToleranceCommandReqInput(
            request_id="request",
            command="resume",
            target_ranks=[1],
        )

        self.assertIsNone(self.handle_command(scheduler, request))
        self.assertFalse(scheduler._engine_paused)

    def test_remote_command_failure_is_returned_by_leader(self):
        scheduler, _ = self.make_scheduler(remote_failure=True)
        request = FaultToleranceCommandReqInput(
            request_id="request",
            command="pause",
            target_ranks=[1],
        )

        output = self.handle_command(scheduler, request)

        self.assertFalse(output.success)
        self.assertIn("tp_rank=1: boom", output.message)

    def test_shutdown_terminates_only_the_target_leader_without_collective(self):
        scheduler, events = self.make_scheduler()
        scheduler._aggregate_ft_command_result = lambda *_: self.fail(
            "shutdown must not enter a collective"
        )
        request = FaultToleranceCommandReqInput(
            request_id="request",
            command="shutdown",
            target_ranks=[1],
        )

        with patch("os.kill") as kill:
            self.assertIsNone(self.handle_command(scheduler, request))

        kill.assert_called_once_with(os.getpid(), signal.SIGTERM)
        self.assertEqual(events, [])

    def test_shutdown_is_ignored_by_nonleader_dp_members(self):
        scheduler, events = self.make_scheduler(attn_tp_rank=1)
        request = FaultToleranceCommandReqInput(
            request_id="request",
            command="shutdown",
            target_ranks=[1],
        )

        with patch("os.kill") as kill:
            self.assertIsNone(self.handle_command(scheduler, request))

        kill.assert_not_called()
        self.assertEqual(events, [])

    def test_successful_tp_recovery_is_reported_as_explicit_dp_ranks(self):
        sender = Sender()
        scheduler = SimpleNamespace(
            tp_size=16,
            dp_size=4,
            server_args=SimpleNamespace(enable_fault_tolerance=True),
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    take_recovered_ep_ranks=lambda: [4, 5, 7, 8]
                )
            ),
            send_to_tokenizer=sender,
        )

        self.report_recovered(scheduler)

        self.assertEqual(len(sender.sent), 1)
        self.assertEqual(sender.sent[0].ranks, [1, 2])

    def test_dpc_forwards_shutdown_to_dp_leader(self):
        workers = [Sender(), Sender()]
        dpc = SimpleNamespace(workers=workers)
        request = FaultToleranceCommandReqInput(
            request_id="request",
            command="shutdown",
            target_ranks=[1],
        )

        self.send_dpc_command(dpc, request)

        self.assertEqual(workers[0].sent, [])
        self.assertEqual(workers[1].sent, [request])

    def test_local_process_exit_reports_all_local_dp_ranks_then_waits(self):
        dpc = SimpleNamespace(
            scheduler_process_infos=[
                SimpleNamespace(global_rank=8, dp_rank=2),
                SimpleNamespace(global_rank=9, dp_rank=2),
            ],
            local_dp_ranks=[2, 3],
            port_args=SimpleNamespace(tokenizer_ipc_name="tcp://node0:1"),
        )
        process = SimpleNamespace(pid=123)

        self.assertIsNone(self.handle_process_exit(dpc, 0, process, "scheduler"))

        self.assertEqual(len(DPC_RUNTIME.sender.sent), 1)
        output = DPC_RUNTIME.sender.sent[0]
        self.assertEqual(output.ranks, [2, 3])
        self.assertFalse(output.active)
        self.assertEqual(DPC_RUNTIME.sleeps, [2])
        self.assertTrue(DPC_RUNTIME.sender.closed)
        self.assertTrue(DPC_RUNTIME.context.terminated)

    def test_rejoined_dpc_reports_all_local_dp_ranks_active(self):
        sender = Sender()
        dpc = SimpleNamespace(
            local_dp_ranks=[2, 3],
            _process_state_sender=sender,
        )

        self.report_process_active(dpc, active=True)

        self.assertEqual(len(sender.sent), 1)
        self.assertEqual(sender.sent[0].ranks, [2, 3])
        self.assertTrue(sender.sent[0].active)


if __name__ == "__main__":
    unittest.main()
