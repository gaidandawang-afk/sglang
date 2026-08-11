"""CPU tests for the fixed-address Ascend MC2 elastic metadata view."""

import unittest

import torch

from sglang.srt.elastic_ep.npu_mc2 import (
    NpuMC2ElasticInfo,
    build_mc2_elastic_info_values,
    compact_mc2_physical_expert_ids,
)
from sglang.srt.eplb.eplb_algorithms.elasticity_aware import rebalance_experts
from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNpuMC2ElasticInfo(CustomTestCase):
    def test_healthy_payload_uses_original_identity_namespace(self):
        payload = build_mc2_elastic_info_values(
            [1, 1, 1, 1],
            original_ep_size=4,
            num_local_physical_experts=8,
        )

        self.assertEqual(
            payload.tolist(),
            [0, 4, 0, 32, 0, 1, 2, 3, 0, 1, 2, 3],
        )
        self.assertEqual(payload.dtype, torch.int32)
        self.assertEqual(payload.device.type, "cpu")

    def test_sparse_mask_builds_both_rank_mappings_and_active_expert_count(self):
        payload = build_mc2_elastic_info_values(
            [1, 0, 1, 1],
            original_ep_size=4,
            num_local_physical_experts=8,
        )

        self.assertEqual(
            payload.tolist(),
            [1, 3, 0, 24, 0, -1, 1, 2, 0, 2, 3, -1],
        )

    def test_original_physical_ids_are_compacted_around_rank_holes(self):
        payload = build_mc2_elastic_info_values(
            [1, 0, 1, 1],
            original_ep_size=4,
            num_local_physical_experts=8,
        )
        original_ids = torch.tensor(
            [0, 7, 8, 15, 16, 23, 24, 31, -1], dtype=torch.int64
        )

        compact_ids = compact_mc2_physical_expert_ids(
            original_ids,
            elastic_info=payload,
            original_ep_size=4,
            num_local_physical_experts=8,
        )

        self.assertEqual(
            compact_ids.tolist(),
            [0, 7, -1, -1, 8, 15, 16, 23, -1],
        )

    def test_healthy_physical_id_compaction_is_identity(self):
        payload = build_mc2_elastic_info_values(
            [1, 1, 1, 1],
            original_ep_size=4,
            num_local_physical_experts=8,
        )
        original_ids = torch.tensor([0, 7, 8, 23, 31], dtype=torch.int32)

        compact_ids = compact_mc2_physical_expert_ids(
            original_ids,
            elastic_info=payload,
            original_ep_size=4,
            num_local_physical_experts=8,
        )

        self.assertTrue(torch.equal(compact_ids, original_ids))

    def test_expert_dispatch_compacts_survivor_storage_ids_for_mc2(self):
        payload = build_mc2_elastic_info_values(
            [1, 0, 1, 1],
            original_ep_size=4,
            num_local_physical_experts=8,
        )
        info = ExpertLocationDispatchInfo(
            ep_dispatch_algorithm="static",
            partial_logical_to_rank_dispatch_physical_map=torch.tensor(
                [0, 16, 24], dtype=torch.int64
            ),
            partial_logical_to_all_physical_map=torch.empty(0),
            partial_logical_to_all_physical_map_num_valid=torch.empty(0),
            num_physical_experts=32,
            npu_mc2_elastic_info=payload,
            npu_mc2_original_ep_size=4,
            npu_mc2_num_local_physical_experts=8,
        )

        compact_ids = topk_ids_logical_to_physical(torch.tensor([0, 1, 2]), info)

        self.assertEqual(compact_ids.tolist(), [0, 8, 16])

    def test_update_preserves_graph_captured_storage(self):
        elastic_info = NpuMC2ElasticInfo.create(
            [1, 1, 1, 1],
            original_ep_size=4,
            num_physical_experts=32,
            device="cpu",
        )
        original_tensor = elastic_info.tensor
        original_data_ptr = elastic_info.data_ptr

        elastic_info.update(torch.tensor([1, 0, 1, 1], dtype=torch.int32))

        self.assertIs(elastic_info.tensor, original_tensor)
        self.assertEqual(elastic_info.tensor.data_ptr(), original_data_ptr)
        self.assertEqual(
            elastic_info.tensor.tolist(),
            [1, 3, 0, 24, 0, -1, 1, 2, 0, 2, 3, -1],
        )

    def test_invalid_masks_and_expert_layout_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one EP rank"):
            build_mc2_elastic_info_values(
                [0, 0, 0, 0],
                original_ep_size=4,
                num_local_physical_experts=8,
            )
        with self.assertRaisesRegex(ValueError, "divisible"):
            NpuMC2ElasticInfo.create(
                [1, 1, 1, 1],
                original_ep_size=4,
                num_physical_experts=30,
                device="cpu",
            )

    def test_scale_down_requires_preallocated_slots_for_all_logical_experts(self):
        without_spares = NpuMC2ElasticInfo.create(
            [1, 1, 1, 1],
            original_ep_size=4,
            num_physical_experts=128,
            device="cpu",
        )
        with self.assertRaisesRegex(
            RuntimeError, "--ep-num-redundant-experts at least 44"
        ):
            without_spares.validate_logical_expert_capacity(
                [1, 0, 1, 1], num_logical_experts=128
            )

        with_spares = NpuMC2ElasticInfo.create(
            [1, 1, 1, 1],
            original_ep_size=4,
            num_physical_experts=172,
            device="cpu",
        )
        with_spares.validate_logical_expert_capacity(
            [1, 0, 1, 1], num_logical_experts=128
        )
        with_spares.update([1, 0, 1, 1])
        self.assertEqual(with_spares.tensor[3].item(), 129)

    def test_qwen3_layout_keeps_all_logical_experts_in_survivor_slots(self):
        physical_to_logical, logical_to_physical, logical_count = rebalance_experts(
            weight=torch.arange(1, 129, dtype=torch.float32).repeat(2, 1),
            num_replicas=172,
            num_groups=1,
            num_nodes=1,
            num_gpus=4,
            enable_hierarchical=False,
            active_ranks=torch.tensor([1, 0, 1, 1], dtype=torch.int32),
        )

        self.assertEqual(physical_to_logical.shape, (2, 172))
        self.assertTrue((physical_to_logical[:, 43:86] == 0).all().item())
        self.assertTrue((logical_count >= 1).all().item())
        valid_locations = logical_to_physical[logical_to_physical >= 0]
        self.assertFalse(
            ((valid_locations >= 43) & (valid_locations < 86)).any().item()
        )

    def test_qwen3_redundant_layout_fits_mc2_compact_namespace(self):
        _, logical_to_physical, _ = rebalance_experts(
            weight=torch.arange(1, 129, dtype=torch.float32).repeat(2, 1),
            num_replicas=512,
            num_groups=1,
            num_nodes=1,
            num_gpus=4,
            enable_hierarchical=False,
            active_ranks=torch.tensor([1, 0, 1, 1], dtype=torch.int32),
        )
        payload = build_mc2_elastic_info_values(
            [1, 0, 1, 1],
            original_ep_size=4,
            num_local_physical_experts=128,
        )

        compact_locations = compact_mc2_physical_expert_ids(
            logical_to_physical,
            elastic_info=payload,
            original_ep_size=4,
            num_local_physical_experts=128,
        )

        valid = logical_to_physical >= 0
        self.assertTrue((compact_locations[valid] >= 0).all().item())
        self.assertTrue((compact_locations[valid] < 384).all().item())
        self.assertTrue((compact_locations[~valid] == -1).all().item())


if __name__ == "__main__":
    unittest.main()
