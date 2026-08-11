"""CPU coverage for Ascend survivor-first expert recovery."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_distribution
from sglang.srt.eplb.expert_location import ExpertLocationMetadata
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


def _make_expert_location_metadata(physical_to_logical, ep_size=4):
    physical_to_logical = torch.tensor(physical_to_logical, dtype=torch.int64)
    num_layers, num_physical_experts = physical_to_logical.shape
    num_logical_experts = int(physical_to_logical.max().item()) + 1
    logical_to_all = torch.full(
        (num_layers, num_logical_experts, num_physical_experts),
        -1,
        dtype=torch.int64,
    )
    counts = torch.zeros(
        (num_layers, num_logical_experts),
        dtype=torch.int64,
    )
    for layer_id in range(num_layers):
        for physical_id in range(num_physical_experts):
            logical_id = int(physical_to_logical[layer_id, physical_id].item())
            offset = int(counts[layer_id, logical_id].item())
            logical_to_all[layer_id, logical_id, offset] = physical_id
            counts[layer_id, logical_id] += 1
    return ExpertLocationMetadata(
        physical_to_logical_map=physical_to_logical.clone(),
        physical_to_logical_map_cpu=physical_to_logical.clone(),
        logical_to_all_physical_map=logical_to_all.clone(),
        logical_to_all_physical_map_cpu=logical_to_all.clone(),
        logical_to_all_physical_map_num_valid=counts,
        ep_size=ep_size,
        logical_to_rank_dispatch_physical_map=None,
    )


class TestNpuExpertRecovery(CustomTestCase):
    def test_fault_layout_preserves_survivor_slots_when_coverage_is_complete(self):
        old_metadata = _make_expert_location_metadata(
            [[0, 1, 2, 3, 0, 1, 2, 3]]
        )
        server_args = SimpleNamespace(ep_dispatch_algorithm="dynamic")

        recovered = ExpertLocationMetadata.init_for_fault_recovery(
            server_args,
            old_metadata,
            active_ranks=[True, True, True, False],
        )

        self.assertTrue(
            torch.equal(
                recovered.physical_to_logical_map_cpu,
                old_metadata.physical_to_logical_map_cpu,
            )
        )
        valid_locations = recovered.logical_to_all_physical_map_cpu[
            recovered.logical_to_all_physical_map_cpu >= 0
        ]
        self.assertFalse((valid_locations >= 6).any().item())
        self.assertTrue(
            (recovered.logical_to_all_physical_map_num_valid >= 1).all().item()
        )

    def test_fault_layout_replaces_only_a_redundant_slot_for_missing_expert(self):
        old_metadata = _make_expert_location_metadata(
            [[0, 0, 1, 1, 2, 2, 3, 3]]
        )
        server_args = SimpleNamespace(ep_dispatch_algorithm="dynamic")

        recovered = ExpertLocationMetadata.init_for_fault_recovery(
            server_args,
            old_metadata,
            active_ranks=[True, True, True, False],
        )

        changed = (
            recovered.physical_to_logical_map_cpu
            != old_metadata.physical_to_logical_map_cpu
        )
        self.assertEqual(changed.sum().item(), 1)
        self.assertEqual(recovered.physical_to_logical_map_cpu[0, 0].item(), 3)
        self.assertEqual(
            recovered.physical_to_logical_map_cpu[0, 6:].tolist(),
            old_metadata.physical_to_logical_map_cpu[0, 6:].tolist(),
        )
        valid_locations = recovered.logical_to_all_physical_map_cpu[
            recovered.logical_to_all_physical_map_cpu >= 0
        ]
        self.assertFalse((valid_locations >= 6).any().item())
        self.assertTrue(
            (recovered.logical_to_all_physical_map_num_valid >= 1).all().item()
        )

    def test_fault_layout_rejects_insufficient_survivor_capacity(self):
        old_metadata = _make_expert_location_metadata([[0, 1, 2, 3]])
        server_args = SimpleNamespace(ep_dispatch_algorithm="dynamic")

        with self.assertRaisesRegex(RuntimeError, "insufficient survivor"):
            ExpertLocationMetadata.init_for_fault_recovery(
                server_args,
                old_metadata,
                active_ranks=[True, True, True, False],
            )

    def test_fault_layout_balances_missing_experts_across_survivors(self):
        old_metadata = _make_expert_location_metadata(
            [[0, 1, 2, 3, 0, 1, 2, 3]]
        )
        server_args = SimpleNamespace(ep_dispatch_algorithm="dynamic")

        recovered = ExpertLocationMetadata.init_for_fault_recovery(
            server_args,
            old_metadata,
            active_ranks=[True, False, True, False],
        )

        changed = (
            recovered.physical_to_logical_map_cpu
            != old_metadata.physical_to_logical_map_cpu
        )
        changed_per_original_rank = changed.view(1, 4, 2).sum(dim=(0, 2))
        self.assertEqual(changed_per_original_rank.tolist(), [1, 0, 1, 0])
        active_locations = recovered.logical_to_all_physical_map_cpu[
            recovered.logical_to_all_physical_map_cpu >= 0
        ]
        self.assertTrue(
            torch.all(
                (active_locations < 2)
                | ((active_locations >= 4) & (active_locations < 6))
            ).item()
        )
        self.assertTrue(
            (recovered.logical_to_all_physical_map_num_valid == 1).all().item()
        )

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
