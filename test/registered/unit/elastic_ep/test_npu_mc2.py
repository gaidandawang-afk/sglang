import unittest

import torch

from sglang.srt.elastic_ep.npu_mc2 import (
    NpuMC2ElasticInfo,
    build_mc2_elastic_info,
    compact_mc2_physical_expert_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNpuMC2ElasticInfo(CustomTestCase):
    def test_build_elastic_info_for_sparse_survivors(self):
        info = build_mc2_elastic_info(
            [True, False, True, True],
            original_ep_size=4,
            num_local_physical_experts=3,
        )

        self.assertEqual(
            info.tolist(),
            [1, 3, 0, 9, 0, -1, 1, 2, 0, 2, 3, -1],
        )

    def test_compact_physical_expert_ids_uses_effective_rank_namespace(self):
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

        self.assertEqual(compact.tolist(), [[0, 4, 8, -1]])

    def test_update_preserves_graph_captured_tensor_address(self):
        elastic_info = NpuMC2ElasticInfo.create(
            [True, True, True, True],
            original_ep_size=4,
            num_physical_experts=12,
            device="cpu",
        )
        address = elastic_info.tensor.data_ptr()

        elastic_info.update([True, False, True, True])

        self.assertEqual(elastic_info.tensor.data_ptr(), address)
        self.assertEqual(
            elastic_info.tensor.tolist(),
            [1, 3, 0, 9, 0, -1, 1, 2, 0, 2, 3, -1],
        )


if __name__ == "__main__":
    unittest.main()
