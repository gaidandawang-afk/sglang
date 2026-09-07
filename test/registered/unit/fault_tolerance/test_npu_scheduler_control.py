import ast
import logging
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


class Struct:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def load_command_handler():
    path = REPO_ROOT / "python/sglang/srt/managers/scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    method = next(
        node
        for node in scheduler.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "handle_fault_tolerance_command"
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {
        "FT_OPERATION_RETRY": "retry",
        "FT_OPERATION_SCALE_DOWN": "scale_down",
        "FaultToleranceCommandReqInput": Struct,
        "FaultToleranceCommandReqOutput": Struct,
        "Optional": Optional,
        "_is_npu": True,
        "logger": logging.getLogger(__name__),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["handle_fault_tolerance_command"]


HANDLE_COMMAND = load_command_handler()


def make_scheduler(events):
    def call(name, result=None):
        return Mock(side_effect=lambda *_: events.append(name) or result)

    model_runner = SimpleNamespace(
        recover_npu_device_for_fault_tolerance=call("recover"),
        rebuild_npu_fault_tolerance_survivor_control_group=call("rebuild"),
        update_fault_tolerance_active_ranks=call("update"),
        update_npu_fault_tolerance_mc2=call("mc2"),
        run_npu_fault_tolerance_dummy_batch=call("dummy"),
        synchronize_npu_fault_tolerance_health_gate=call("device_sync"),
    )
    return SimpleNamespace(
        ps=SimpleNamespace(dp_rank=1, attn_tp_rank=0, attn_cp_rank=0),
        tp_worker=SimpleNamespace(model_runner=model_runner),
        server_args=SimpleNamespace(elastic_ep_backend="mc2"),
        _engine_paused=True,
        _ft_pause_deadline=30,
        _ft_discard_inflight_window=call("discard", True),
        schedule_stream=SimpleNamespace(synchronize=call("schedule_sync")),
        forward_stream=SimpleNamespace(wait_stream=call("stream_wait")),
        forward_stream_ctx=nullcontext(),
    )


def test_retry_only_adds_runtime_recovery_to_generic_flow():
    events = []
    scheduler = make_scheduler(events)

    output = HANDLE_COMMAND(
        scheduler,
        Struct(request_id="r", command="retry", target_ranks=[1]),
    )

    assert events == ["recover", "update", "discard"]
    assert (output.request_id, output.rank) == ("r", 1)


def test_scale_down_preserves_npu_recovery_order():
    events = []
    scheduler = make_scheduler(events)

    HANDLE_COMMAND(
        scheduler,
        Struct(
            request_id="s",
            command="scale_down",
            target_ranks=[1],
            active_mask=[True, False],
        ),
    )

    assert events == [
        "recover",
        "rebuild",
        "update",
        "discard",
        "mc2",
        "schedule_sync",
        "stream_wait",
        "dummy",
        "device_sync",
    ]
    assert scheduler._engine_paused is False
    assert scheduler._ft_pause_deadline is None


def test_recovery_failure_keeps_scheduler_paused_without_discard():
    events = []
    scheduler = make_scheduler(events)
    scheduler.tp_worker.model_runner.recover_npu_device_for_fault_tolerance.side_effect = RuntimeError(
        "recovery failed"
    )

    with pytest.raises(RuntimeError, match="recovery failed"):
        HANDLE_COMMAND(
            scheduler,
            Struct(request_id="r", command="retry", target_ranks=[1]),
        )

    assert scheduler._engine_paused is True
    assert scheduler._ft_pause_deadline == 30
    assert events == []
