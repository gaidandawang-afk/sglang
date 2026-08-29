from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.eplb.expert_location_updater import (
    _update_expert_weights_raw,
    update_expert_weights_single_layer,
)
from sglang.srt.eplb.process_group_context import EPLBProcessGroupContext


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


def test_expert_p2p_maps_original_rank_to_compact_survivor_rank():
    survivor_group = object()
    context = EPLBProcessGroupContext(
        group=survivor_group,
        active_original_ranks=(1, 2, 3),
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
    assert all(op.peer == 1 for op in fake_ops)
    assert all(op.group is survivor_group for op in fake_ops)
