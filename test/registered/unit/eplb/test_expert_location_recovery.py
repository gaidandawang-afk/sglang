"""Unit tests for the distinct Elastic EP scale and recovery metadata paths."""

from unittest.mock import patch

import torch

from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    broadcast_global_expert_location_metadata_for_recovery,
    broadcast_global_expert_location_metadata_for_scale,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _metadata(fill: int) -> ExpertLocationMetadata:
    physical_to_logical = torch.full((2, 4), fill, dtype=torch.int64)
    logical_to_all = torch.full((2, 3, 4), fill, dtype=torch.int64)
    return ExpertLocationMetadata(
        physical_to_logical_map=physical_to_logical,
        physical_to_logical_map_cpu=physical_to_logical.clone(),
        logical_to_all_physical_map=logical_to_all,
        logical_to_all_physical_map_cpu=logical_to_all.clone(),
        logical_to_all_physical_map_num_valid=torch.full(
            (2, 3), fill, dtype=torch.int64
        ),
        ep_size=2,
        logical_to_rank_dispatch_physical_map=torch.full(
            (2, 3), fill, dtype=torch.int64
        ),
    )


def test_recovery_broadcast_updates_graph_visible_tensors_in_place():
    metadata = _metadata(fill=0)
    source_values = [
        torch.full_like(metadata.physical_to_logical_map, 1),
        torch.full_like(metadata.logical_to_all_physical_map, 2),
        torch.full_like(metadata.logical_to_all_physical_map_num_valid, 3),
        torch.full_like(metadata.logical_to_rank_dispatch_physical_map, 4),
    ]
    tensors = [
        metadata.physical_to_logical_map,
        metadata.logical_to_all_physical_map,
        metadata.logical_to_all_physical_map_num_valid,
        metadata.logical_to_rank_dispatch_physical_map,
    ]
    data_ptrs = [tensor.data_ptr() for tensor in tensors]

    def fake_broadcast(tensor, src, group):
        assert src == 2
        assert group == "recovery-group"
        tensor.copy_(source_values.pop(0))

    with (
        patch(
            "sglang.srt.eplb.expert_location.get_global_expert_location_metadata",
            return_value=metadata,
        ),
        patch("torch.distributed.broadcast", side_effect=fake_broadcast),
    ):
        result = broadcast_global_expert_location_metadata_for_recovery(
            src_rank=2, group="recovery-group"
        )

    assert result is metadata
    assert [tensor.data_ptr() for tensor in tensors] == data_ptrs
    assert torch.equal(
        metadata.physical_to_logical_map_cpu,
        metadata.physical_to_logical_map,
    )
    assert torch.equal(
        metadata.logical_to_all_physical_map_cpu,
        metadata.logical_to_all_physical_map,
    )
    assert not source_values


def test_scale_broadcast_rebuilds_metadata_for_the_expanded_topology():
    old_metadata = _metadata(fill=0)
    new_metadata = _metadata(fill=1)

    with (
        patch(
            "sglang.srt.eplb.expert_location.get_global_expert_location_metadata",
            return_value=old_metadata,
        ),
        patch("sglang.srt.runtime_context.get_server_args", return_value=object()),
        patch("torch.distributed.broadcast"),
        patch.object(
            ExpertLocationMetadata,
            "init_by_mapping",
            return_value=new_metadata,
        ) as init_by_mapping,
        patch(
            "sglang.srt.eplb.expert_location.set_global_expert_location_metadata"
        ) as set_metadata,
    ):
        result = broadcast_global_expert_location_metadata_for_scale(
            model_config=object(), moe_ep_rank=3, src_rank=0
        )

    assert result is new_metadata
    init_by_mapping.assert_called_once()
    set_metadata.assert_called_once_with(new_metadata, allow_overwrite=True)
