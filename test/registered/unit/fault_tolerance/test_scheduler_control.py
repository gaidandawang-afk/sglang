import ast
from collections import deque
from http import HTTPStatus
import logging
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Optional, Tuple
import unittest
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
        self.options = []
        self.endpoint = None
        self.closed = False

    def send_pyobj(self, value, flags=0):
        self.sent.append(value)

    def send_output(self, value, *args):
        self.sent.append((value, *args))

    def setsockopt(self, option, value):
        self.options.append((option, value))

    def connect(self, endpoint):
        self.endpoint = endpoint

    def close(self, linger=None):
        self.closed = True


class FakeContext:
    def __init__(self, sender):
        self.sender = sender
        self.terminated = False

    def socket(self, socket_type):
        return self.sender

    def term(self):
        self.terminated = True


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
                "HTTPStatus": HTTPStatus,
                "Optional": Optional,
                "ScheduleBatch": FakeBatch,
                "Tuple": Tuple,
                "logger": logging.getLogger(__name__),
                "notify_node_main_process_failure": Mock(),
                "release_kv_cache": lambda *args, **kwargs: None,
                "time": SimpleNamespace(monotonic=lambda: 100.0, sleep=Mock()),
                "_is_npu": False,
            },
        )
        cls.run_ft_loop = staticmethod(scheduler["_run_event_loop_fault_tolerance"])
        cls.handle_command = staticmethod(scheduler["handle_fault_tolerance_command"])
        cls.check_deadline = staticmethod(scheduler["_check_ft_pause_deadline"])
        cls.discard = staticmethod(scheduler["_ft_discard_inflight_window"])
        cls.process_overlap = staticmethod(scheduler["_process_next_overlap_result"])

        server_args = load_class_methods(
            REPO_ROOT / "python/sglang/srt/server_args.py",
            "ServerArgs",
            {"_handle_fault_tolerance"},
            {
                "is_npu": lambda: True,
                "logger": logging.getLogger(__name__),
                "os": os,
            },
        )
        cls.handle_fault_tolerance_args = staticmethod(
            server_args["_handle_fault_tolerance"]
        )

        cls.dpc_globals = {
            "FaultToleranceCommandReqInput": FaultToleranceCommandReqInput,
            "FaultToleranceDPCShutdownReqInput": FaultToleranceDPCShutdownReqInput,
            "ProcessActiveRanksOutput": ProcessActiveRanksOutput,
            "WatchdogHeartbeatOutput": WatchdogHeartbeatOutput,
            "FT_WATCHDOG_SEND_TIMEOUT_MS": 1000,
            "logger": logging.getLogger(__name__),
            "sock_send": lambda socket, value, flags=0: socket.send_pyobj(value, flags),
            "zmq": SimpleNamespace(
                Context=None,
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
        dpc = load_class_methods(
            REPO_ROOT / "python/sglang/srt/managers/data_parallel_controller.py",
            "DataParallelController",
            {
                "send_fault_tolerance_command",
                "shutdown_dp",
                "_get_watchdog_sender",
                "_handle_scheduler_process_exit",
                "_watchdog_heartbeat",
                "_report_initial_watchdog_heartbeat",
                "_report_watchdog_heartbeat",
                "_close_watchdog_sender",
                "_report_process_active_ranks",
            },
            cls.dpc_globals,
        )
        cls.dpc_methods = dpc

        cls.fake_torch = SimpleNamespace(
            device=lambda device_type, device_id: (device_type, device_id),
            npu=SimpleNamespace(set_device=Mock(), synchronize=Mock()),
        )
        model_runner = load_class_methods(
            REPO_ROOT / "python/sglang/srt/model_executor/model_runner.py",
            "ModelRunner",
            {"recover_npu_device_for_fault_tolerance_scale_down"},
            {
                "logger": logging.getLogger(__name__),
                "torch": cls.fake_torch,
            },
        )
        cls.recover_npu = staticmethod(
            model_runner["recover_npu_device_for_fault_tolerance_scale_down"]
        )

    def make_scheduler(self, *, leader=True):
        model_runner = SimpleNamespace(
            apply_fault_tolerance_scale_down=Mock(),
            recover_npu_device_for_fault_tolerance_scale_down=Mock(),
        )
        return SimpleNamespace(
            ps=SimpleNamespace(
                dp_rank=1,
                attn_tp_rank=0 if leader else 1,
                attn_cp_rank=0,
            ),
            tp_worker=SimpleNamespace(model_runner=model_runner),
            server_args=SimpleNamespace(elastic_ep_backend="mc2"),
            _engine_paused=True,
            _ft_pause_deadline=130.0,
            _ft_pending_discard_reason=None,
            _ft_discard_inflight_window=Mock(return_value=True),
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
        self.assertEqual(output.message, "retried")

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
        self.assertEqual(output.message, "scaled down")
        self.assertFalse(scheduler._engine_paused)

    def test_npu_scale_down_recovers_discards_then_applies(self):
        events = []
        scheduler = self.make_scheduler()
        model_runner = scheduler.tp_worker.model_runner
        model_runner.recover_npu_device_for_fault_tolerance_scale_down.side_effect = (
            lambda: events.append("recover_device")
        )
        model_runner.apply_fault_tolerance_scale_down.side_effect = lambda mask: (
            events.append(("scale_down", mask))
        )
        scheduler._ft_pending_discard_reason = "mlp-sync failed"
        scheduler._ft_discard_inflight_window.side_effect = lambda reason: (
            events.append(("discard", reason)) or True
        )
        request = FaultToleranceCommandReqInput(
            request_id="s",
            command="scale_down",
            target_ranks=[1],
            active_mask=[True, False],
        )

        self.handle_command.__globals__["_is_npu"] = True
        try:
            output = self.handle_command(scheduler, request)
        finally:
            self.handle_command.__globals__["_is_npu"] = False

        self.assertEqual(
            events,
            [
                "recover_device",
                ("discard", "mlp-sync failed"),
                ("scale_down", [True, False]),
            ],
        )
        self.assertIsNone(scheduler._ft_pending_discard_reason)
        self.assertEqual(output.message, "scaled down")

    def test_retry_does_not_recover_device(self):
        state = SimpleNamespace(
            active_ranks=FakeTensor([1, 0]),
            active_ranks_cpu=FakeTensor([1, 0]),
            last_active_ranks=FakeTensor([1, 1]),
        )
        module = ModuleType("sglang.srt.elastic_ep.elastic_ep")
        module.ElasticEPStateManager = SimpleNamespace(instance=lambda: state)
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

        self.handle_command.__globals__["_is_npu"] = True
        try:
            with patch.dict(sys.modules, modules):
                self.handle_command(scheduler, request)
        finally:
            self.handle_command.__globals__["_is_npu"] = False

        recover = scheduler.tp_worker.model_runner.recover_npu_device_for_fault_tolerance_scale_down
        recover.assert_not_called()

    def test_npu_scale_down_returns_failed_ack_when_discard_validation_fails(self):
        scheduler = self.make_scheduler()
        scheduler._ft_pending_discard_reason = "mlp-sync failed"
        scheduler._ft_discard_inflight_window.return_value = False
        request = FaultToleranceCommandReqInput(
            request_id="s",
            command="scale_down",
            target_ranks=[1],
            active_mask=[True, False],
        )

        self.handle_command.__globals__["_is_npu"] = True
        try:
            output = self.handle_command(scheduler, request)
        finally:
            self.handle_command.__globals__["_is_npu"] = False

        self.assertFalse(output.success)
        self.assertTrue(scheduler._engine_paused)
        apply_scale_down = (
            scheduler.tp_worker.model_runner.apply_fault_tolerance_scale_down
        )
        apply_scale_down.assert_not_called()

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
            _ft_discard_inflight_window=lambda exc: events.append("discarded") or True,
            ipc_channels=SimpleNamespace(send_to_tokenizer=sender),
            ps=SimpleNamespace(dp_rank=0),
            request_receiver=SimpleNamespace(
                recv_requests=lambda: (_ for _ in ()).throw(KeyboardInterrupt())
            ),
            _check_ft_pause_deadline=Mock(),
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
        self.assertEqual(events, ["fault", "discarded", "report"])

    def test_gpu_continue_discards_and_reenters_normal_loop(self):
        events = []

        def dispatch(_):
            if not events:
                events.append("fault")
                raise RuntimeError("boom")
            raise KeyboardInterrupt()

        self.run_ft_loop.__globals__["dispatch_event_loop"] = dispatch
        self.run_ft_loop.__globals__["_is_npu"] = False
        scheduler = SimpleNamespace(
            _ft_discard_inflight_window=lambda exc: events.append("discarded") or True,
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(
                    send_output=lambda *_: events.append("report")
                )
            ),
            ps=SimpleNamespace(dp_rank=0),
            server_args=SimpleNamespace(
                fault_tolerance_on_error_strategy="continue",
                fault_tolerance_pause_timeout=30,
            ),
            _engine_paused=False,
            _ft_pause_deadline=None,
        )

        with self.assertRaises(KeyboardInterrupt):
            self.run_ft_loop(scheduler)

        self.assertEqual(events, ["fault", "discarded", "report"])

    def test_npu_mc2_defers_discard_until_scale_down(self):
        events = []

        def dispatch(_):
            if events:
                raise KeyboardInterrupt()
            events.append("fault")
            raise RuntimeError("mlp-sync failed")

        self.run_ft_loop.__globals__["dispatch_event_loop"] = dispatch
        self.run_ft_loop.__globals__["_is_npu"] = True
        scheduler = SimpleNamespace(
            _ft_discard_inflight_window=lambda exc: events.append("discarded") or True,
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(
                    send_output=lambda *_: events.append("report")
                )
            ),
            ps=SimpleNamespace(dp_rank=0),
            request_receiver=SimpleNamespace(
                recv_requests=lambda: (_ for _ in ()).throw(KeyboardInterrupt())
            ),
            _check_ft_pause_deadline=Mock(),
            server_args=SimpleNamespace(
                elastic_ep_backend="mc2",
                fault_tolerance_on_error_strategy="pause",
                fault_tolerance_pause_timeout=30,
            ),
            _engine_paused=False,
            _ft_pause_deadline=None,
            _ft_pending_discard_reason=None,
        )

        try:
            with self.assertRaises(KeyboardInterrupt):
                self.run_ft_loop(scheduler)
        finally:
            self.run_ft_loop.__globals__["_is_npu"] = False

        self.assertEqual(events, ["fault", "report"])
        self.assertEqual(scheduler._ft_pending_discard_reason, "mlp-sync failed")

    def test_npu_continue_strategy_is_normalized_during_startup(self):
        module = ModuleType("sglang.srt.fault_tolerance.controller")
        module.is_ft_supported_config = lambda server_args: (True, "")
        server_args = SimpleNamespace(
            enable_fault_tolerance=True,
            fault_tolerance_on_error_strategy="continue",
            elastic_ep_backend="mc2",
            fault_tolerance_communication_abort_timeout=10,
        )

        with patch.dict(
            sys.modules,
            {"sglang.srt.fault_tolerance.controller": module},
        ):
            with patch.dict(os.environ, {}, clear=True):
                self.handle_fault_tolerance_args(server_args)
                self.assertEqual(os.environ["TASK_QUEUE_ENABLE"], "0")
                self.assertEqual(os.environ["HCCL_EVENT_TIMEOUT"], "10")
                self.assertEqual(os.environ["HCCL_EXEC_TIMEOUT"], "9")
                self.assertEqual(os.environ["ACL_DEVICE_SYNC_TIMEOUT"], "10")
                self.assertEqual(os.environ["ACL_STREAM_TIMEOUT"], "10000")

        self.assertEqual(server_args.fault_tolerance_on_error_strategy, "pause")

    def test_npu_scale_down_restarts_without_artificial_delay(self):
        calls = []
        npu = SimpleNamespace(
            current_device=lambda: 3,
            stop_device=lambda device_id: calls.append(("stop", device_id)) or 0,
            restart_device=lambda device_id: calls.append(("restart", device_id)) or 0,
        )
        torch_npu = ModuleType("torch_npu")
        torch_npu.npu = npu
        torch_npu.distributed = SimpleNamespace(
            reinit_process_group=lambda *args: calls.append(("reinit", *args))
        )
        runner = SimpleNamespace(gpu_id=3, ps=SimpleNamespace(dp_rank=3))
        self.fake_torch.npu.set_device = lambda device: calls.append(("set", device))
        self.fake_torch.npu.synchronize = lambda: calls.append(("synchronize",))

        with patch.dict(sys.modules, {"torch_npu": torch_npu}):
            self.recover_npu(runner)

        self.assertEqual(
            calls,
            [
                ("set", ("npu", 3)),
                ("stop", 3),
                ("restart", 3),
                ("reinit", None, False),
                ("synchronize",),
            ],
        )

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
            ps=SimpleNamespace(dp_rank=1),
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

    def test_failed_discard_keeps_inflight_window_for_diagnosis(self):
        req = FakeReq("current")
        running = FakeBatch([req])
        self.discard.__globals__["release_kv_cache"] = Mock(
            side_effect=RuntimeError("release failed")
        )
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(dp_rank=1),
            cur_batch_for_debug=running,
            last_batch=running,
            result_queue=deque(),
            running_batch=running,
            chunked_req=None,
            tree_cache=object(),
            ipc_channels=SimpleNamespace(send_to_tokenizer=Sender()),
        )

        self.assertFalse(self.discard(scheduler, RuntimeError("boom")))
        self.assertIs(scheduler.running_batch, running)
        self.assertEqual(scheduler.running_batch.reqs, [req])

    def test_discard_fails_when_post_release_pool_invariant_fails(self):
        running = FakeBatch([])
        checker = SimpleNamespace(
            _check_all_pools=Mock(return_value=(True, ["missing one page"]))
        )
        observer = SimpleNamespace(get_pool_stats=Mock(return_value=object()))
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(dp_rank=1),
            cur_batch_for_debug=running,
            last_batch=running,
            result_queue=deque(),
            running_batch=running,
            chunked_req=None,
            tree_cache=object(),
            ipc_channels=SimpleNamespace(send_to_tokenizer=Sender()),
            invariant_checker=checker,
            pool_stats_observer=observer,
        )

        self.assertFalse(self.discard(scheduler, RuntimeError("boom")))
        checker._check_all_pools.assert_called_once()

    def make_dpc(self):
        sender = Sender()
        context = FakeContext(sender)
        self.dpc_globals["zmq"].Context = lambda: context
        dpc = SimpleNamespace(
            workers=[Sender(), Sender()],
            scheduler_procs=[],
            scheduler_process_dp_ranks=[0, 1, 1],
            scheduler_process_global_ranks=[0, 2, 3],
            server_args=SimpleNamespace(node_rank=1),
            port_args=SimpleNamespace(tokenizer_ipc_name="tcp://node0:1"),
            send_to_tokenizer=Sender(),
            ft_control_endpoint="tcp://node1:2",
            _watchdog_context=None,
            _watchdog_sender=None,
        )
        dpc._get_watchdog_sender = lambda: self.dpc_methods["_get_watchdog_sender"](dpc)
        dpc._watchdog_heartbeat = lambda: self.dpc_methods["_watchdog_heartbeat"](dpc)
        return dpc, sender

    def test_watchdog_reports_global_rank_and_endpoint(self):
        dpc, sender = self.make_dpc()
        proc = SimpleNamespace(pid=123)
        self.dpc_methods["_handle_scheduler_process_exit"](dpc, 1, proc, "scheduler")
        self.dpc_methods["_report_watchdog_heartbeat"](dpc)

        down, heartbeat = sender.sent
        self.assertEqual(down.ranks, [2])
        self.assertEqual(heartbeat.ranks, [0, 2, 3])
        self.assertEqual(heartbeat.control_endpoint, "tcp://node1:2")

    def test_shutdown_kills_every_local_member_of_target_dp(self):
        dpc, _ = self.make_dpc()
        dpc.scheduler_procs = [
            SimpleNamespace(is_alive=lambda: True, kill=Mock()) for _ in range(3)
        ]
        request = FaultToleranceDPCShutdownReqInput(target_dp_ranks=[1])

        self.dpc_methods["shutdown_dp"](dpc, request)

        dpc.scheduler_procs[0].kill.assert_not_called()
        dpc.scheduler_procs[1].kill.assert_called_once()
        dpc.scheduler_procs[2].kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
