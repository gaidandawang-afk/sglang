import torch
from sglang.srt.elastic_ep.npu_mc2 import (
    NpuMC2ElasticInfo,
    build_mc2_elastic_info,
    compact_mc2_physical_expert_ids,
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
