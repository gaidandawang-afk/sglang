import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock


MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/eplb/expert_location_updater.py"
)


class FakeSlot:
    def __init__(self, owner, index):
        self.owner = owner
        self.index = index

    @property
    def value(self):
        return self.owner.values[self.index]

    def copy_(self, other):
        self.owner.values[self.index] = other.value
        return self


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)

    def __getitem__(self, index):
        return FakeSlot(self, index)


class FakeMap:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        return SimpleNamespace(tolist=lambda: list(self.rows[index]))


class EPLBProcessGroupContext:
    def __init__(
        self,
        *,
        control_group=None,
        device_group=None,
        active_original_ranks=None,
        control_group_uses_cpu=False,
    ):
        self.control_group = control_group
        self.device_group = device_group
        self.active_original_ranks = active_original_ranks
        self.control_group_uses_cpu = control_group_uses_cpu

    def is_active(self, original_rank):
        return (
            self.active_original_ranks is None
            or original_rank in self.active_original_ranks
        )


def load_updater_functions():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    wanted = {
        "_update_expert_weights_raw",
        "update_expert_weights_single_layer",
        "_ChunkUtils",
        "_deduplicate_ordered",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
    ]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    isend = object()
    irecv = object()
    distributed = SimpleNamespace(
        get_world_size=lambda: 3,
        isend=isend,
        irecv=irecv,
        batch_isend_irecv=lambda ops: [SimpleNamespace(wait=lambda: None)],
    )
    torch = SimpleNamespace(Tensor=FakeTensor, distributed=distributed)
    default_context = EPLBProcessGroupContext()
    namespace = {
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "ExpertLocationMetadata": object,
        "ElasticEPStateManager": SimpleNamespace(instance=lambda: None),
        "P2POp": lambda **kwargs: SimpleNamespace(**kwargs),
        "_LOG_INPUT": False,
        "envs": SimpleNamespace(
            SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE=SimpleNamespace(get=lambda: 100)
        ),
        "get_bool_env_var": lambda name: False,
        "get_eplb_process_group_context": lambda: default_context,
        "logger": logging.getLogger(__name__),
        "torch": torch,
    }
    exec(compile(module, str(MODULE_PATH), "exec"), namespace)
    return namespace


def test_gpu_per_node_uses_original_ep_size_after_rank_failure():
    namespace = load_updater_functions()
    update_single_layer = Mock()
    namespace["create_temp_buffers"] = lambda tensors: []
    namespace["update_expert_weights_single_layer"] = update_single_layer
    old_metadata = SimpleNamespace(
        ep_size=4,
        num_local_physical_experts=2,
        physical_to_logical_map_cpu=FakeMap([[0] * 8]),
    )
    new_metadata = SimpleNamespace(
        physical_to_logical_map_cpu=FakeMap([[0] * 8]),
    )

    namespace["_update_expert_weights_raw"](
        routed_experts_weights_of_layer={0: [FakeTensor([0, 0])]},
        old_expert_location_metadata=old_metadata,
        new_expert_location_metadata=new_metadata,
        update_layer_ids=[0],
        nnodes=4,
        rank=0,
    )

    assert update_single_layer.call_args.kwargs["world_size"] == 3
    assert update_single_layer.call_args.kwargs["num_gpu_per_node"] == 1


def test_expert_p2p_reuses_original_device_group_and_rank_namespace():
    namespace = load_updater_functions()
    original_device_group = object()
    context = EPLBProcessGroupContext(
        control_group=object(),
        device_group=original_device_group,
        active_original_ranks=(1, 2, 3),
        control_group_uses_cpu=True,
    )
    fake_ops = []

    def fake_p2p_op(**kwargs):
        op = SimpleNamespace(**kwargs)
        fake_ops.append(op)
        return op

    namespace["get_eplb_process_group_context"] = lambda: context
    namespace["P2POp"] = fake_p2p_op
    logs = namespace["update_expert_weights_single_layer"](
        routed_experts_weights=[FakeTensor([0])],
        temp_buffers=[FakeTensor([0])],
        old_physical_to_logical_map=[7, 0, 7, 1],
        new_physical_to_logical_map=[7, 7, 7, 1],
        num_local_physical_experts=1,
        num_gpu_per_node=1,
        rank=1,
        world_size=4,
        missing_logical_experts_info=[],
        debug=True,
    )

    assert any("chosen_src_rank=2" in line for line in logs)
    assert fake_ops
    assert all(op.peer == 2 for op in fake_ops)
    assert all(op.group is original_device_group for op in fake_ops)


def test_missing_expert_is_reported_when_all_original_sources_failed():
    namespace = load_updater_functions()
    context = EPLBProcessGroupContext(
        control_group=object(),
        device_group=object(),
        active_original_ranks=(1, 2, 3),
        control_group_uses_cpu=True,
    )
    namespace["get_eplb_process_group_context"] = lambda: context
    missing = []

    namespace["update_expert_weights_single_layer"](
        routed_experts_weights=[FakeTensor([0])],
        temp_buffers=[FakeTensor([0])],
        old_physical_to_logical_map=[7, 0, 2, 1],
        new_physical_to_logical_map=[7, 7, 2, 1],
        num_local_physical_experts=1,
        num_gpu_per_node=1,
        rank=1,
        world_size=4,
        missing_logical_experts_info=missing,
    )

    assert missing == [7]
