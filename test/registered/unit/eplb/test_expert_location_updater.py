import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.eplb.expert_location_updater import (
    _copy_npu_p2p_recv_,
    _stage_npu_p2p_ops,
    _update_expert_weights_raw,
    update_expert_weights_single_layer,
)
from sglang.srt.eplb.process_group_context import EPLBProcessGroupContext
from sglang.srt.hardware_backend.npu.utils import NPUACLFormat


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


class _FakeNPUTensor:
    def __init__(self, values, acl_format):
        self.values = list(values)
        self.shape = (len(self.values),)
        self.dtype = torch.float16
        self.device = SimpleNamespace(type="npu")
        self.acl_format = acl_format

    def storage_offset(self):
        return 0

    def copy_(self, source):
        self.values = list(source.values)
        return self


def _fake_torch_npu():
    def empty_with_format(shape, *, dtype, device, acl_format):
        del dtype, device
        return _FakeNPUTensor([0] * shape[0], acl_format)

    return SimpleNamespace(
        empty_with_format=empty_with_format,
        get_npu_format=lambda tensor: tensor.acl_format,
    )


def test_npu_p2p_uses_nd_staging_and_restores_destination_format():
    nz_format = int(NPUACLFormat.ACL_FORMAT_FRACTAL_NZ)
    nd_format = int(NPUACLFormat.ACL_FORMAT_ND)
    send_tensor = _FakeNPUTensor([1, 2], nz_format)
    recv_tensor = _FakeNPUTensor([0, 0], nz_format)
    ops = [
        SimpleNamespace(
            op=torch.distributed.isend,
            tensor=send_tensor,
            peer=1,
            group="hccl",
            tag=3,
        ),
        SimpleNamespace(
            op=torch.distributed.irecv,
            tensor=recv_tensor,
            peer=1,
            group="hccl",
            tag=4,
        ),
    ]

    fake_torch_npu = _fake_torch_npu()
    with (
        patch.dict(sys.modules, {"torch_npu": fake_torch_npu}),
        patch(
            "sglang.srt.eplb.expert_location_updater.P2POp",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    ):
        staged_ops, recv_copy_infos = _stage_npu_p2p_ops(ops)

        assert [op.tensor.acl_format for op in staged_ops] == [nd_format, nd_format]
        assert staged_ops[0].tensor.values == [1, 2]
        assert staged_ops[0].tensor is not send_tensor

        staged_ops[1].tensor.values = [7, 8]
        with patch(
            "sglang.srt.hardware_backend.npu.utils.copy_npu_formatted_tensor_",
            side_effect=lambda destination, source: destination.copy_(source),
        ) as formatted_copy:
            _copy_npu_p2p_recv_(recv_copy_infos[0][1], recv_copy_infos[0][0])

    assert recv_tensor.values == [7, 8]
    assert formatted_copy.call_args.args[1].acl_format == nz_format
