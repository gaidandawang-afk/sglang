import ast
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
MC2_MODULE_PATH = REPO_ROOT / "python/sglang/srt/elastic_ep/npu_mc2.py"
DISPATCH_MODULE_PATH = REPO_ROOT / "python/sglang/srt/eplb/expert_location_dispatch.py"
ELASTIC_EP_MODULE_PATH = REPO_ROOT / "python/sglang/srt/elastic_ep/elastic_ep.py"


class FakeTensor:
    def __init__(self, values, *, dtype="int32", device="cpu"):
        self.values = list(values)
        self.dtype = dtype
        self.device = device

    def detach(self):
        return self

    def cpu(self):
        return FakeTensor(self.values, dtype=self.dtype, device="cpu")

    def flatten(self):
        return self

    def tolist(self):
        return list(self.values)

    def to(self, *, dtype=None, device=None):
        return FakeTensor(
            self.values,
            dtype=self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
        )

    def copy_(self, other):
        self.values[:] = other.values
        return self

    def masked_fill(self, mask, value):
        return FakeTensor(
            [
                value if selected else item
                for item, selected in zip(self.values, mask.values)
            ],
            dtype=self.dtype,
            device=self.device,
        )

    def long(self):
        return self.to(dtype="int64")

    def __getitem__(self, index):
        if isinstance(index, slice):
            return FakeTensor(self.values[index], dtype=self.dtype, device=self.device)
        if isinstance(index, FakeTensor):
            return FakeTensor(
                [self.values[item] for item in index.values],
                dtype=self.dtype,
                device=self.device,
            )
        return self.values[index]

    def _binary(self, other, operation, *, dtype=None):
        other_values = other.values if isinstance(other, FakeTensor) else None
        values = [
            operation(item, other_values[i] if other_values is not None else other)
            for i, item in enumerate(self.values)
        ]
        return FakeTensor(values, dtype=dtype or self.dtype, device=self.device)

    def __ge__(self, other):
        return self._binary(other, lambda left, right: left >= right, dtype="bool")

    def __lt__(self, other):
        return self._binary(other, lambda left, right: left < right, dtype="bool")

    def __and__(self, other):
        return self._binary(other, lambda left, right: left and right, dtype="bool")

    def __or__(self, other):
        return self._binary(other, lambda left, right: left or right, dtype="bool")

    def __invert__(self):
        return FakeTensor(
            [not value for value in self.values], dtype="bool", device=self.device
        )

    def __mod__(self, other):
        return self._binary(other, lambda left, right: left % right)

    def __mul__(self, other):
        return self._binary(other, lambda left, right: left * right)

    def __add__(self, other):
        return self._binary(other, lambda left, right: left + right)


def load_mc2_module():
    torch = ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.int32 = "int32"
    torch.tensor = lambda values, dtype: FakeTensor(values, dtype=dtype)
    torch.div = lambda tensor, divisor, rounding_mode: FakeTensor(
        [value // divisor for value in tensor.values],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    spec = importlib.util.spec_from_file_location("npu_mc2_under_test", MC2_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(sys.modules, {"torch": torch}):
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
    return module


def test_builds_bidirectional_original_and_effective_rank_maps():
    module = load_mc2_module()

    values = module.build_mc2_elastic_info_values(
        [True, False, True, True],
        original_ep_size=4,
        num_local_physical_experts=8,
    )

    assert values == [1, 3, 0, 24, 0, -1, 1, 2, 0, 2, 3, -1]


def test_rejects_invalid_mc2_layouts():
    module = load_mc2_module()

    with pytest.raises(ValueError, match="smaller"):
        module.build_mc2_elastic_info_values(
            [True, True], original_ep_size=4, num_local_physical_experts=8
        )
    with pytest.raises(ValueError, match="at least one"):
        module.build_mc2_elastic_info_values(
            [False] * 4, original_ep_size=4, num_local_physical_experts=8
        )
    with pytest.raises(ValueError, match="divisible"):
        module.NpuMC2ElasticInfo.create(
            [True] * 4,
            original_ep_size=4,
            num_physical_experts=31,
            device="npu:0",
        )


def test_update_preserves_elastic_info_storage_address():
    module = load_mc2_module()
    info = module.NpuMC2ElasticInfo.create(
        [True] * 4,
        original_ep_size=4,
        num_physical_experts=32,
        device="npu:0",
    )
    original_storage = info.tensor

    info.update([True, False, True, True])

    assert info.tensor is original_storage
    assert info.tensor.tolist() == [1, 3, 0, 24, 0, -1, 1, 2, 0, 2, 3, -1]


def test_compacts_only_live_original_physical_expert_ids():
    module = load_mc2_module()
    elastic_info = FakeTensor([1, 3, 0, 24, 0, -1, 1, 2, 0, 2, 3, -1], device="npu:0")
    original_ids = FakeTensor(
        [0, 7, 8, 15, 16, 23, 24, 31, -1, 32],
        dtype="int64",
        device="npu:0",
    )

    compact_ids = module.compact_mc2_physical_expert_ids(
        original_ids,
        elastic_info=elastic_info,
        original_ep_size=4,
        num_local_physical_experts=8,
    )

    assert compact_ids.tolist() == [0, 7, -1, -1, 8, 15, 16, 23, -1, -1]
    assert compact_ids.dtype == original_ids.dtype


@contextmanager
def load_dispatch_function(compact):
    tree = ast.parse(DISPATCH_MODULE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "topk_ids_logical_to_physical"
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "Optional": Optional,
        "ExpertLocationDispatchInfo": object,
        "torch": SimpleNamespace(Tensor=object),
        "_topk_ids_logical_to_physical_static": lambda topk_ids, info: "original",
    }
    compact_module = ModuleType("sglang.srt.elastic_ep.npu_mc2")
    compact_module.compact_mc2_physical_expert_ids = compact
    with patch.dict(sys.modules, {compact_module.__name__: compact_module}):
        exec(compile(module, str(DISPATCH_MODULE_PATH), "exec"), namespace)
        yield namespace["topk_ids_logical_to_physical"]


def test_dispatch_compacts_at_the_mc2_boundary_only():
    calls = []

    def compact(ids, **kwargs):
        calls.append((ids, kwargs))
        return "compact"

    with load_dispatch_function(compact) as dispatch:
        no_mc2_info = SimpleNamespace(
            ep_dispatch_algorithm="static", npu_mc2_elastic_info=None
        )
        mc2 = SimpleNamespace(
            tensor="elastic-info",
            original_ep_size=4,
            num_local_physical_experts=8,
        )
        mc2_info = SimpleNamespace(
            ep_dispatch_algorithm="static", npu_mc2_elastic_info=mc2
        )

        assert dispatch("logical", no_mc2_info) == "original"
        assert calls == []
        assert dispatch("logical", mc2_info) == "compact"
    assert calls == [
        (
            "original",
            {
                "elastic_info": "elastic-info",
                "original_ep_size": 4,
                "num_local_physical_experts": 8,
            },
        )
    ]


def test_elastic_ep_selects_an_npu_device():
    tree = ast.parse(ELASTIC_EP_MODULE_PATH.read_text(encoding="utf-8"))
    manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ElasticEPStateManager"
    )
    select_device = next(
        node
        for node in manager.body
        if isinstance(node, ast.FunctionDef) and node.name == "_select_device"
    )
    harness = ast.fix_missing_locations(
        ast.Module(
            body=[
                    ast.ClassDef(
                        name="Harness",
                        bases=[],
                        keywords=[],
                        body=[select_device],
                        decorator_list=[],
                    )
            ],
            type_ignores=[],
        )
    )
    namespace = {
        "is_cuda": lambda: False,
        "is_npu": lambda: True,
        "is_cpu": lambda: False,
        "torch": SimpleNamespace(device=lambda name: name),
    }
    exec(compile(harness, str(ELASTIC_EP_MODULE_PATH), "exec"), namespace)

    assert namespace["Harness"]._select_device() == "npu"
