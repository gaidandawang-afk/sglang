import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/layers/moe/fused_moe_triton/layer.py"
)


class FakeMatrix:
    def __init__(self, rows, row_indices=None, *, acl_format=29):
        self.rows = rows
        self.row_indices = row_indices or list(range(len(rows)))
        self.device = SimpleNamespace(type="npu")
        self.dtype = "bfloat16"
        self.acl_format = acl_format

    @property
    def shape(self):
        return (len(self.row_indices), len(self.rows[0]))

    def narrow(self, dim, start, length):
        assert dim == 0
        return FakeMatrix(
            self.rows,
            self.row_indices[start : start + length],
            acl_format=self.acl_format,
        )

    def copy_(self, source):
        source_rows = [source.rows[index] for index in source.row_indices]
        assert len(source_rows) == len(self.row_indices)
        for destination_index, source_row in zip(
            self.row_indices, source_rows, strict=True
        ):
            self.rows[destination_index] = list(source_row)
        return self


def load_w13_copy_function():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_copy_loaded_w13_weight_"
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "_is_npu": True,
        "torch": SimpleNamespace(Tensor=FakeMatrix),
    }
    exec(compile(module, str(MODULE_PATH), "exec"), namespace)
    return namespace["_copy_loaded_w13_weight_"]


def test_npu_w13_reload_roundtrips_the_full_expert_through_nd():
    copy_w13 = load_w13_copy_function()
    parent = [
        [[1, 1], [2, 2], [3, 3], [4, 4]],
        [[11, 11], [12, 12], [13, 13], [14, 14]],
    ]
    neighbor_before = copy.deepcopy(parent[1])
    destination = FakeMatrix(parent[0])
    first_half = FakeMatrix([[101, 101], [102, 102]], acl_format=2)
    second_half = FakeMatrix([[201, 201], [202, 202]], acl_format=2)

    def empty_with_format(shape, *, dtype, device, acl_format):
        return FakeMatrix(
            [[0 for _ in range(shape[1])] for _ in range(shape[0])],
            acl_format=acl_format,
        )

    npu_utils = SimpleNamespace(
        NPUACLFormat=SimpleNamespace(ACL_FORMAT_ND=2),
        copy_to_npu_formatted_tensor_=lambda target, source: target.copy_(source),
        is_npu_internal_format_tensor=lambda tensor: tensor.acl_format != 2,
    )
    torch_npu = SimpleNamespace(empty_with_format=empty_with_format)

    with patch.dict(
        sys.modules,
        {
            "torch_npu": torch_npu,
            "sglang.srt.hardware_backend.npu.utils": npu_utils,
        },
    ):
        copy_w13(destination, first_half, shard_dim=0, start=0, shard_size=2)
        assert parent[0] == [[101, 101], [102, 102], [3, 3], [4, 4]]
        assert parent[1] == neighbor_before

        copy_w13(destination, second_half, shard_dim=0, start=2, shard_size=2)

    assert parent[0] == [
        [101, 101],
        [102, 102],
        [201, 201],
        [202, 202],
    ]
    assert parent[1] == neighbor_before
