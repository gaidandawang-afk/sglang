import ast
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict


MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/eplb/expert_distribution.py"
)


class FakeTensor:
    def __init__(self, value=0.25):
        self.value = value
        self.cpu_called = False
        self.to_device = None

    def to(self, device):
        self.to_device = device
        return self

    def cpu(self):
        self.cpu_called = True
        return self

    def item(self):
        return self.value


class SurvivorContext:
    control_group = object()
    control_group_uses_cpu = True

    def is_control_group_root(self, original_rank):
        return original_rank == 1


def load_method(class_name, method_name, namespace):
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(MODULE_PATH), "exec"), namespace)
    return namespace[method_name]


def test_utilization_reduce_uses_cpu_survivor_group_and_original_root():
    calls = []
    context = SurvivorContext()
    tensor = FakeTensor()
    distributed = SimpleNamespace(
        ReduceOp=SimpleNamespace(SUM="sum"),
        reduce=lambda value, **kwargs: calls.append((value, kwargs)),
    )
    torch = SimpleNamespace(Tensor=FakeTensor, distributed=distributed)
    method = load_method(
        "_UtilizationRateAccumulatorMixin",
        "_append_utilization_rate",
        {
            "Any": Any,
            "Dict": Dict,
            "compute_gpu_physical_count": lambda value, num_gpu: value,
            "get_eplb_process_group_context": lambda: context,
            "torch": torch,
        },
    )
    accumulator = SimpleNamespace(
        _expert_location_metadata=SimpleNamespace(ep_size=4),
        _server_args=SimpleNamespace(device="npu"),
        _rank=2,
    )

    method(accumulator, 1, tensor, {})

    assert tensor.cpu_called
    assert calls == [
        (
            tensor,
            {"dst": 0, "op": "sum", "group": context.control_group},
        )
    ]


def test_stat_dump_all_reduce_uses_cpu_survivor_group():
    calls = []
    context = SurvivorContext()
    tensor = FakeTensor()
    distributed = SimpleNamespace(
        ReduceOp=SimpleNamespace(SUM="sum"),
        all_reduce=lambda value, **kwargs: calls.append((value, kwargs)),
    )
    torch = SimpleNamespace(Tensor=FakeTensor, distributed=distributed)
    method = load_method(
        "_StatAccumulator",
        "dump",
        {
            "_OutputMode": str,
            "_convert_global_physical_count_to_logical_count": lambda *args, **kwargs: tensor,
            "get_eplb_process_group_context": lambda: context,
            "torch": torch,
        },
    )
    accumulator = SimpleNamespace(
        _expert_location_metadata=SimpleNamespace(
            num_layers=2,
            num_logical_experts=4,
            physical_to_logical_map=object(),
        ),
        _global_physical_count_of_buffered_step=SimpleNamespace(
            get_all=lambda: object()
        ),
        _first_dump=False,
        _rank=2,
        _get_global_average_utilization_rate=lambda: None,
    )

    output = method(accumulator, "object")

    assert tensor.cpu_called
    assert calls == [
        (tensor, {"op": "sum", "group": context.control_group})
    ]
    assert output["logical_count"] is tensor


def test_average_utilization_broadcast_uses_compact_group_root_zero():
    calls = []
    context = SurvivorContext()
    tensor = FakeTensor()
    distributed = SimpleNamespace(
        broadcast=lambda value, **kwargs: calls.append((value, kwargs))
    )
    torch = SimpleNamespace(
        Tensor=FakeTensor,
        distributed=distributed,
        empty=lambda *args, **kwargs: tensor,
        float32="float32",
    )
    method = load_method(
        "_StatAccumulator",
        "_get_global_average_utilization_rate",
        {
            "get_eplb_process_group_context": lambda: context,
            "math": math,
            "torch": torch,
        },
    )
    accumulator = SimpleNamespace(
        _enable=True,
        _rank=2,
        _server_args=SimpleNamespace(
            eplb_min_rebalancing_utilization_threshold=0.5,
            device="npu",
        ),
    )

    assert method(accumulator) == 0.25
    assert calls == [
        (tensor, {"src": 0, "group": context.control_group})
    ]
