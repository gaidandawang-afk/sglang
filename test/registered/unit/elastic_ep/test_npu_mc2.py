import pytest
import torch
from sglang.srt.elastic_ep.npu_mc2 import (
    NpuMC2ElasticInfo,
    build_mc2_elastic_info,
    compact_mc2_physical_expert_ids,
    validate_mc2_dispatch_expert_ids,
    validate_mc2_scale_down_routing,
)


def test_build_elastic_info_for_sparse_survivors():
    info = build_mc2_elastic_info(
        [True, False, True, True],
        original_ep_size=4,
        num_local_physical_experts=3,
    )

    assert info.tolist() == [1, 3, 0, 9, 0, -1, 1, 2, 0, 2, 3, -1]


def test_compact_physical_expert_ids_uses_effective_rank_namespace():
    info = build_mc2_elastic_info(
        [True, False, True, True],
        original_ep_size=4,
        num_local_physical_experts=3,
    )
    physical_ids = torch.tensor([[0, 7, 11, -1]])

    compact = compact_mc2_physical_expert_ids(
        physical_ids,
        elastic_info=info,
        original_ep_size=4,
        num_local_physical_experts=3,
    )

    assert compact.tolist() == [[0, 4, 8, -1]]


def test_update_preserves_graph_captured_tensor_address():
    elastic_info = NpuMC2ElasticInfo.create(
        [True, True, True, True],
        original_ep_size=4,
        num_physical_experts=12,
        device="cpu",
    )
    address = elastic_info.tensor.data_ptr()

    elastic_info.update([True, False, True, True])

    assert elastic_info.tensor.data_ptr() == address
    assert elastic_info.tensor.tolist() == [
        1,
        3,
        0,
        9,
        0,
        -1,
        1,
        2,
        0,
        2,
        3,
        -1,
    ]


def test_scale_down_routing_accepts_live_static_replicas():
    summary = validate_mc2_scale_down_routing(
        [False, True, True, True],
        original_ep_size=4,
        num_local_physical_experts=2,
        ep_dispatch_algorithm="static",
        logical_to_all_physical_map=torch.tensor([[[0, 2], [1, 3], [4, -1], [6, -1]]]),
        logical_to_all_physical_map_num_valid=torch.tensor([[2, 2, 1, 1]]),
        logical_to_rank_dispatch_physical_map=torch.tensor([[2, 3, 4, 6]]),
    )

    assert summary["dead_candidate_count"] == 2
    assert summary["missing_live_expert_count"] == 0
    assert summary["static_dead_reference_count"] == 0


def test_scale_down_routing_rejects_expert_without_live_replica():
    with pytest.raises(RuntimeError, match="no live physical replica"):
        validate_mc2_scale_down_routing(
            [False, True, True, True],
            original_ep_size=4,
            num_local_physical_experts=1,
            ep_dispatch_algorithm=None,
            logical_to_all_physical_map=torch.tensor([[[0], [1], [2], [3]]]),
            logical_to_all_physical_map_num_valid=torch.ones((1, 4), dtype=torch.long),
            logical_to_rank_dispatch_physical_map=None,
        )


def test_dispatch_diagnostic_rejects_selected_dead_expert_but_ignores_padding():
    info = build_mc2_elastic_info(
        [False, True, True, True],
        original_ep_size=4,
        num_local_physical_experts=1,
    )

    with pytest.raises(RuntimeError, match="before kernel launch"):
        validate_mc2_dispatch_expert_ids(
            torch.tensor([[0, 2, -1]]),
            expert_weights=torch.tensor([[1.0, 1.0, 0.0]]),
            elastic_info=info,
            original_ep_size=4,
            num_local_physical_experts=1,
            ids_are_compacted=False,
        )


def test_dispatch_diagnostic_accepts_effective_ids_and_is_one_shot():
    elastic_info = NpuMC2ElasticInfo.create(
        [False, True, True, True],
        original_ep_size=4,
        num_physical_experts=4,
        device="cpu",
    )
    elastic_info.arm_dispatch_validation()

    assert elastic_info.consume_dispatch_validation()
    assert not elastic_info.consume_dispatch_validation()
    validate_mc2_dispatch_expert_ids(
        torch.tensor([[0, 2, -1]]),
        expert_weights=torch.tensor([[1.0, 1.0, 0.0]]),
        elastic_info=elastic_info.tensor,
        original_ep_size=4,
        num_local_physical_experts=1,
        ids_are_compacted=True,
    )
