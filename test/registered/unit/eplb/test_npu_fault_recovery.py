"""CPU coverage for Ascend survivor-first expert recovery."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_distribution
from sglang.srt.eplb.expert_location_updater import (
    update_expert_weights_single_layer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeSurvivorGroups:
    def __init__(self, active_original_ranks):
        self.active_original_ranks = tuple(active_original_ranks)
        self.eplb_device_group = "rebuilt-eplb-hccl"

    def group_rank(self, original_rank):
        return self.active_original_ranks.index(original_rank)


class _CompletedWork:
    def wait(self):
        return None


class TestNpuExpertRecovery(CustomTestCase):
    def test_periodic_eplb_stats_use_rebuilt_survivor_group(self):
        accumulator = expert_distribution._StatAccumulator.__new__(
            expert_distribution._StatAccumulator
        )
        accumulator._first_dump = False
        accumulator._rank = 2
        accumulator.get_local_logical_count = lambda: torch.tensor([3, 4])
        accumulator._get_global_average_utilization_rate = lambda: None

        groups = SimpleNamespace(
            active_original_ranks=(0, 2, 3),
            all_reduce_sum_tensor=lambda tensor: tensor + 10,
        )
        with (
            patch.object(
                expert_distribution,
                "_get_rebuilt_npu_ft_process_groups",
                return_value=groups,
            ),
            patch.object(torch.distributed, "all_reduce") as old_all_reduce,
        ):
            output = accumulator.dump(output_mode="object")

        self.assertEqual(output["logical_count"].tolist(), [13, 14])
        old_all_reduce.assert_not_called()

    def test_reuses_existing_local_physical_experts_before_communication(self):
        weights = [torch.tensor([[10.0], [11.0]])]
        temp_buffers = [torch.empty_like(weights[0])]
        missing = []

        with patch(
            "sglang.srt.eplb.expert_location_updater.torch.distributed.batch_isend_irecv"
        ) as batch_p2p:
            update_expert_weights_single_layer(
                routed_experts_weights=weights,
                temp_buffers=temp_buffers,
                old_physical_to_logical_map=[0, 1, 2, 3],
                new_physical_to_logical_map=[1, 0, 2, 3],
                num_local_physical_experts=2,
                num_gpu_per_node=4,
                rank=0,
                world_size=2,
                missing_logical_experts_info=missing,
                survivor_process_groups=_FakeSurvivorGroups([0, 1]),
            )

        self.assertEqual(weights[0].tolist(), [[11.0], [10.0]])
        self.assertEqual(missing, [])
        batch_p2p.assert_not_called()

    def test_copies_from_another_survivor_over_rebuilt_hccl_group(self):
        weights = [torch.tensor([[10.0]])]
        temp_buffers = [torch.empty_like(weights[0])]
        missing = []
        submitted_ops = []

        def execute_p2p(ops):
            submitted_ops.extend(ops)
            for op in ops:
                if op.op == torch.distributed.irecv:
                    op.tensor.fill_(20.0)
            return [_CompletedWork() for _ in ops]

        from torch.distributed.distributed_c10d import _world

        with (
            patch.dict(
                _world.pg_group_ranks,
                {"rebuilt-eplb-hccl": {0: 0, 1: 1, 2: 2}},
            ),
            patch(
                "sglang.srt.eplb.expert_location_updater.torch.distributed.batch_isend_irecv",
                side_effect=execute_p2p,
            ),
        ):
            update_expert_weights_single_layer(
                routed_experts_weights=weights,
                temp_buffers=temp_buffers,
                old_physical_to_logical_map=[0, 1, 2, 3],
                new_physical_to_logical_map=[2, 0, 0, 3],
                num_local_physical_experts=1,
                num_gpu_per_node=4,
                rank=0,
                world_size=4,
                missing_logical_experts_info=missing,
                survivor_process_groups=_FakeSurvivorGroups([0, 2, 3]),
            )

        self.assertEqual(weights[0].item(), 20.0)
        self.assertEqual(missing, [])
        self.assertTrue(submitted_ops)
        self.assertTrue(
            all(op.group == "rebuilt-eplb-hccl" for op in submitted_ops)
        )
        # Original rank 2 is compact rank 1 in survivor membership [0, 2, 3].
        self.assertTrue(all(op.peer == 1 for op in submitted_ops))

    def test_reloads_only_when_no_survivor_contains_the_expert(self):
        weights = [torch.tensor([[10.0]])]
        temp_buffers = [torch.empty_like(weights[0])]
        missing = []

        with patch(
            "sglang.srt.eplb.expert_location_updater.torch.distributed.batch_isend_irecv"
        ) as batch_p2p:
            update_expert_weights_single_layer(
                routed_experts_weights=weights,
                temp_buffers=temp_buffers,
                old_physical_to_logical_map=[0, 1, 2, 3],
                new_physical_to_logical_map=[1, 0, 2, 3],
                num_local_physical_experts=1,
                num_gpu_per_node=4,
                rank=0,
                world_size=4,
                missing_logical_experts_info=missing,
                survivor_process_groups=_FakeSurvivorGroups([0, 2, 3]),
            )

        self.assertEqual(missing, [1])
        self.assertEqual(weights[0].item(), 10.0)
        batch_p2p.assert_not_called()


if __name__ == "__main__":
    unittest.main()
