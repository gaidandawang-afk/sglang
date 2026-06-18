import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("numpy")
pytest.importorskip("psutil")
pytest.importorskip("setproctitle")
pytest.importorskip("torch")
pytest.importorskip("zmq")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.modules.setdefault(
    "resource",
    SimpleNamespace(
        RLIMIT_NOFILE=0,
        RLIMIT_STACK=1,
        getrlimit=mock.Mock(return_value=(0, 0)),
        setrlimit=mock.Mock(),
    ),
)

from sglang.srt.managers.data_parallel_controller import DataParallelController
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    FaultToleranceRankFaultOutput,
)


class FakeSocket:
    def __init__(self):
        self.sent = []

    def send_pyobj(self, obj):
        self.sent.append(obj)


def make_controller(*, enable_fault_tolerance: bool):
    controller = object.__new__(DataParallelController)
    controller.server_args = SimpleNamespace(
        elastic_ep_backend="mooncake",
        enable_fault_tolerance=enable_fault_tolerance,
    )
    controller.workers = [FakeSocket(), FakeSocket()]
    controller.scheduler_procs = []
    controller.status = [True, True]
    controller._status_lock = threading.Lock()
    controller._scheduler_exit_events = queue.Queue()
    controller.send_to_tokenizer = FakeSocket()
    return controller


def test_noft_scheduler_exit_updates_routing_and_broadcasts_active_mask():
    controller = make_controller(enable_fault_tolerance=False)
    proc = SimpleNamespace(pid=1234, exitcode=-9)

    assert controller._handle_scheduler_process_exit(1, proc, "scheduler_dp_1")
    assert controller._current_status() == [True, False]

    controller._drain_scheduler_exit_events()

    assert controller.send_to_tokenizer.sent == []
    assert len(controller.workers[0].sent) == 1
    assert controller.workers[1].sent == []
    active_ranks = controller.workers[0].sent[0]
    assert isinstance(active_ranks, ActiveRanksOutput)
    assert active_ranks.status == [True, False]


def test_ft_scheduler_exit_reports_kill_without_cross_thread_zmq_send():
    controller = make_controller(enable_fault_tolerance=True)
    proc = SimpleNamespace(pid=1234, exitcode=-9)

    assert controller._handle_scheduler_process_exit(1, proc, "scheduler_dp_1")
    assert controller._current_status() == [True, False]
    assert controller.send_to_tokenizer.sent == []

    controller._drain_scheduler_exit_events()

    assert controller.workers[0].sent == []
    assert controller.workers[1].sent == []
    assert len(controller.send_to_tokenizer.sent) == 1
    event = controller.send_to_tokenizer.sent[0]
    assert isinstance(event, FaultToleranceRankFaultOutput)
    assert event.rank == 1
    assert event.fault_type == "kill"
