import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.eplb.expert_location_updater import (
    _copy_expert_tensor_,
    _copy_npu_p2p_recv_,
    _p2p_ops_need_npu_staging,
    _update_expert_weights_raw,
    update_expert_weights_single_layer,
)
from sglang.srt.eplb.process_group_context import EPLBProcessGroupContext
from sglang.srt.hardware_backend.npu.utils import NPUACLFormat


def test_npu_p2p_staging_does_not_depend_on_an_explicit_group():
    npu_op = SimpleNamespace(
        tensor=SimpleNamespace(device=SimpleNamespace(type="npu"))
    )
    cuda_op = SimpleNamespace(
        tensor=SimpleNamespace(device=SimpleNamespace(type="cuda"))
    )

    assert _p2p_ops_need_npu_staging([npu_op])
    assert not _p2p_ops_need_npu_staging([cuda_op])


class _FakeNPUTensor:
    def __init__(self, values, acl_format, storage_offset=0):
        self.values = list(values)
        self.shape = (len(self.values),)
        self.dtype = torch.float16
        self.device = SimpleNamespace(type="npu")
        self.acl_format = acl_format
        self._storage_offset = storage_offset

    def storage_offset(self):
        return self._storage_offset

    def copy_(self, source):
        self.values = list(source.values)
        return self


def test_npu_p2p_uses_nd_staging_regardless_of_storage_offset():
    from sglang.srt.eplb.expert_location_updater import _stage_npu_p2p_ops

    nz_format = int(NPUACLFormat.ACL_FORMAT_FRACTAL_NZ)
    nd_format = int(NPUACLFormat.ACL_FORMAT_ND)
    sources = [
        _FakeNPUTensor([1, 2], nz_format, storage_offset=0),
        _FakeNPUTensor([3, 4], nz_format, storage_offset=8),
    ]
    ops = [
        SimpleNamespace(
            op=torch.distributed.isend,
            tensor=source,
            peer=1,
            group=None,
            tag=0,
        )
        for source in sources
    ]
    fake_torch_npu = SimpleNamespace(
        empty_with_format=lambda shape, **kwargs: _FakeNPUTensor(
            [0] * shape[0], kwargs["acl_format"]
        ),
        get_npu_format=lambda tensor: tensor.acl_format,
    )

    with (
        patch.dict(sys.modules, {"torch_npu": fake_torch_npu}),
        patch(
            "sglang.srt.eplb.expert_location_updater.P2POp",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    ):
        staged_ops, recv_copy_infos = _stage_npu_p2p_ops(ops)

    assert recv_copy_infos == []
    assert all(
        staged.tensor is not source for staged, source in zip(staged_ops, sources)
    )
    assert all(staged.tensor.acl_format == nd_format for staged in staged_ops)
    assert all(staged.tensor.storage_offset() == 0 for staged in staged_ops)
    assert [staged.tensor.values for staged in staged_ops] == [[1, 2], [3, 4]]


def test_internal_format_expert_copy_uses_formatted_copy():
    nz_format = int(NPUACLFormat.ACL_FORMAT_FRACTAL_NZ)
    source = _FakeNPUTensor([1, 2], nz_format, storage_offset=8)
    destination = _FakeNPUTensor([0, 0], nz_format, storage_offset=16)

    with (
        patch(
            "sglang.srt.hardware_backend.npu.utils.is_npu_internal_format_tensor",
            return_value=True,
        ),
        patch(
            "sglang.srt.hardware_backend.npu.utils.copy_npu_formatted_tensor_",
            side_effect=lambda destination, source: destination.copy_(source),
        ) as formatted_copy,
    ):
        _copy_expert_tensor_(destination, source)

    assert destination.values == [1, 2]
    formatted_copy.assert_called_once_with(destination, source)


def test_nd_p2p_recv_converts_to_offset_zero_destination_format():
    nz_format = int(NPUACLFormat.ACL_FORMAT_FRACTAL_NZ)
    nd_format = int(NPUACLFormat.ACL_FORMAT_ND)
    source = _FakeNPUTensor([7, 8], nd_format)
    destination = _FakeNPUTensor([0, 0], nz_format, storage_offset=16)
    fake_torch_npu = SimpleNamespace(
        empty_with_format=lambda shape, **kwargs: _FakeNPUTensor(
            [0] * shape[0], kwargs["acl_format"]
        ),
        get_npu_format=lambda tensor: tensor.acl_format,
    )

    with (
        patch.dict(sys.modules, {"torch_npu": fake_torch_npu}),
        patch(
            "sglang.srt.hardware_backend.npu.utils.is_npu_internal_format_tensor",
            side_effect=lambda tensor: tensor.acl_format != nd_format,
        ),
        patch(
            "sglang.srt.hardware_backend.npu.utils.copy_npu_formatted_tensor_",
            side_effect=lambda destination, source: destination.copy_(source),
        ) as formatted_copy,
    ):
        _copy_npu_p2p_recv_(destination, source)

    assert destination.values == [7, 8]
    assert formatted_copy.call_args.args[1].acl_format == nz_format


def test_gpu_per_node_uses_original_ep_size_after_rank_failure():
    old_metadata = MagicMock()
    old_metadata.ep_size = 4
    old_metadata.num_local_physical_experts = 2
    old_metadata.physical_to_logical_map_cpu = torch.zeros((1, 8), dtype=torch.int64)

    new_metadata = MagicMock()
    new_metadata.physical_to_logical_map_cpu = torch.zeros((1, 8), dtype=torch.int64)

    with (
        patch("torch.distributed.get_world_size", return_value=3),
        patch(
            "sglang.srt.eplb.expert_location_updater.create_temp_buffers",
            return_value=[],
        ),
        patch(
            "sglang.srt.eplb.expert_location_updater.update_expert_weights_single_layer"
        ) as update_single_layer,
    ):
        _update_expert_weights_raw(
            routed_experts_weights_of_layer={0: [torch.empty(2)]},
            old_expert_location_metadata=old_metadata,
            new_expert_location_metadata=new_metadata,
            update_layer_ids=[0],
            nnodes=4,
            rank=0,
        )

    assert update_single_layer.call_args.kwargs["world_size"] == 3
    assert update_single_layer.call_args.kwargs["num_gpu_per_node"] == 1


def test_expert_copy_prefers_a_live_source_rank():
    active_state = SimpleNamespace(
        active_ranks_cpu=torch.tensor([False, True, True, True])
    )
    fake_ops = []

    def fake_p2p_op(op, tensor, peer):
        result = SimpleNamespace(op=op, tensor=tensor, peer=peer)
        fake_ops.append(result)
        return result

    with (
        patch(
            "sglang.srt.eplb.expert_location_updater.ElasticEPStateManager.instance",
            return_value=active_state,
        ),
        patch(
            "sglang.srt.eplb.expert_location_updater.P2POp",
            side_effect=fake_p2p_op,
        ),
        patch(
            "torch.distributed.batch_isend_irecv",
            return_value=[SimpleNamespace(wait=lambda: None)],
        ),
    ):
        logs = update_expert_weights_single_layer(
            routed_experts_weights=[torch.zeros(1)],
            temp_buffers=[torch.zeros(1)],
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
    assert all(op.peer != 0 for op in fake_ops)


def test_expert_p2p_reuses_original_device_group_and_rank_namespace():
    survivor_control_group = object()
    original_device_group = object()
    context = EPLBProcessGroupContext(
        control_group=survivor_control_group,
        device_group=original_device_group,
        active_original_ranks=(1, 2, 3),
        control_group_uses_cpu=True,
    )
    fake_ops = []

    def fake_p2p_op(op, tensor, peer, group=None, tag=0):
        result = SimpleNamespace(
            op=op,
            tensor=tensor,
            peer=peer,
            group=group,
            tag=tag,
        )
        fake_ops.append(result)
        return result

    with (
        patch(
            "sglang.srt.eplb.expert_location_updater.get_eplb_process_group_context",
            return_value=context,
        ),
        patch(
            "sglang.srt.eplb.expert_location_updater.P2POp",
            side_effect=fake_p2p_op,
        ),
        patch(
            "torch.distributed.batch_isend_irecv",
            return_value=[SimpleNamespace(wait=lambda: None)],
        ),
    ):
        logs = update_expert_weights_single_layer(
            routed_experts_weights=[torch.zeros(1)],
            temp_buffers=[torch.zeros(1)],
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
