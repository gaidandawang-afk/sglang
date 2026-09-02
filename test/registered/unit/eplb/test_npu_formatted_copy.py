import ast
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/hardware_backend/npu/utils.py"
)


class FakeDevice:
    type = "npu"


class FakeTensor:
    def __init__(self, values, *, acl_format=29, storage_offset=0, data_ptr=100):
        self.values = list(values)
        self.shape = (2, 2)
        self.dtype = "bfloat16"
        self.device = DEVICE
        self.acl_format = acl_format
        self._storage_offset = storage_offset
        self._data_ptr = data_ptr
        self.physical_storage_size = 8
        self.target = self

    def storage_offset(self):
        return self._storage_offset

    def data_ptr(self):
        return self._data_ptr


DEVICE = FakeDevice()


def load_utils_functions(torch):
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    wanted = {
        "require_torch_npu_nz_p2p_support",
        "copy_npu_formatted_tensor_",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, str(MODULE_PATH), "exec"), namespace)
    return namespace


def test_nz_p2p_requires_post2_but_accepts_post2_dev_build():
    torch = SimpleNamespace(Tensor=FakeTensor)
    require_support = load_utils_functions(torch)[
        "require_torch_npu_nz_p2p_support"
    ]

    with patch.dict(
        sys.modules, {"torch_npu": SimpleNamespace(__version__="2.10.0")}
    ):
        try:
            require_support()
        except RuntimeError as exc:
            assert "torch-npu>=2.10.0.post2" in str(exc)
        else:
            raise AssertionError("TorchNPU 2.10.0 must be rejected for NZ P2P")

    for version in ("2.10.0.post2", "2.10.0.post2.dev20260704", "2.11.0"):
        with patch.dict(
            sys.modules, {"torch_npu": SimpleNamespace(__version__=version)}
        ):
            require_support()


def test_formatted_copy_aliases_full_expert_views_at_offset_zero():
    changed_offsets = []

    def empty_with_format(shape, *, dtype, device, acl_format):
        return FakeTensor(
            [0, 0, 0, 0],
            acl_format=acl_format,
            storage_offset=0,
            data_ptr=1000,
        )

    def change_data_ptr(alias, tensor, storage_offset):
        changed_offsets.append(storage_offset)
        alias._data_ptr = tensor.data_ptr()
        alias.target = tensor

    def copy_memory(destination_alias, source_alias, non_blocking):
        assert not non_blocking
        destination_alias.target.values = list(source_alias.target.values)
        return destination_alias

    torch_npu = SimpleNamespace(
        empty_with_format=empty_with_format,
        get_npu_format=lambda tensor: tensor.acl_format,
        get_storage_size=lambda tensor: tensor.target.physical_storage_size,
        npu_change_data_ptr=change_data_ptr,
    )
    torch = SimpleNamespace(
        Tensor=FakeTensor,
        ops=SimpleNamespace(npu=SimpleNamespace(copy_memory_=copy_memory)),
    )
    copy_formatted = load_utils_functions(torch)["copy_npu_formatted_tensor_"]
    destination = FakeTensor(
        [0, 0, 0, 0], storage_offset=8, data_ptr=108
    )
    source = FakeTensor([1, 2, 3, 4], storage_offset=16, data_ptr=216)

    with patch.dict(sys.modules, {"torch_npu": torch_npu}):
        copy_formatted(destination, source)

    assert changed_offsets == [8, 16]
    assert destination.values == source.values
