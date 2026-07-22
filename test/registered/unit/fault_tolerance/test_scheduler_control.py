import ast
from collections import deque
from http import HTTPStatus
import logging
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

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

    def send_output(self, value, *args):
        self.sent.append((value, *args))

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
                "_update_ft_pause_from_mlp_sync",
                "_complete_ft_pause",
                "_aggregate_ft_command_result",
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
                "os": os,
                "release_kv_cache": lambda *args, **kwargs: None,
            },
        )
        cls.handle_command = staticmethod(
            scheduler_methods["handle_fault_tolerance_command"]
        )
        cls.aggregate_result = staticmethod(
            scheduler_methods["_aggregate_ft_command_result"]
        )
        cls.update_pause_from_mlp_sync = staticmethod(
            scheduler_methods["_update_ft_pause_from_mlp_sync"]
        )
        cls.complete_pause = staticmethod(scheduler_methods["_complete_ft_pause"])
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
        cls.send_dpc_command = staticmethod(dpc_methods["send_fault_tolerance_command"])
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
            _ft_pending_pause=None,
            _ft_rank=lambda: 1,
            send_to_tokenizer=Sender(),
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

    def test_remote_resume_failure_is_returned_by_leader(self):
        scheduler, _ = self.make_scheduler(remote_failure=True)
        request = FaultToleranceCommandReqInput(
            request_id="request",
            command="resume",
            target_ranks=[1],
        )

        output = self.handle_command(scheduler, request)

        self.assertFalse(output.success)
        self.assertIn("tp_rank=1: boom", output.message)

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
            cur_batch=current,
            last_batch=previous,
            result_queue=deque([(previous, object())]),
            running_batch=running,
            chunked_req=current_only,
            waiting_queue=[waiting],
            tree_cache=object(),
            send_to_tokenizer=sender,
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
        self.assertIsNone(scheduler.cur_batch)
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

    def test_local_process_exit_reports_only_affected_dp_rank_then_waits(self):
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
        self.assertEqual(output.ranks, [2])
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
