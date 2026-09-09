"""CPU tests for minimal expert movement during NPU FT recovery."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_metadata(physical_to_logical, *, ep_size=4):
    physical_to_logical = torch.tensor(physical_to_logical, dtype=torch.int64)
    num_layers, num_physical_experts = physical_to_logical.shape
    num_logical_experts = int(physical_to_logical.max().item()) + 1
    logical_to_all = torch.full(
        (num_layers, num_logical_experts, num_physical_experts),
        -1,
        dtype=torch.int64,
    )
    counts = torch.zeros((num_layers, num_logical_experts), dtype=torch.int64)
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


def _recover(old_metadata, active_ranks):
    return ExpertLocationMetadata.init_for_fault_recovery(
        SimpleNamespace(ep_dispatch_algorithm="dynamic"),
        old_metadata,
        active_ranks=active_ranks,
    )


class TestNpuFaultRecoveryLayout(CustomTestCase):
    def test_preserves_survivor_slots_when_coverage_is_complete(self):
        old_metadata = _make_metadata([[0, 1, 2, 3, 0, 1, 2, 3]])

        recovered = _recover(old_metadata, [True, True, True, False])

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

    def test_replaces_only_one_redundant_slot_for_one_missing_expert(self):
        old_metadata = _make_metadata([[0, 0, 1, 1, 2, 2, 3, 3]])

        recovered = _recover(old_metadata, [True, True, True, False])

        changed = (
            recovered.physical_to_logical_map_cpu
            != old_metadata.physical_to_logical_map_cpu
        )
        self.assertEqual(changed.sum().item(), 1)
        self.assertEqual(recovered.physical_to_logical_map_cpu[0, 0].item(), 3)

    def test_spreads_missing_experts_across_survivor_ranks(self):
        old_metadata = _make_metadata(
            [[0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 4, 5, 6, 7]]
        )

        recovered = _recover(old_metadata, [True, True, False, False])

        changed = (
            recovered.physical_to_logical_map_cpu
            != old_metadata.physical_to_logical_map_cpu
        )
        self.assertEqual(changed[0, :4].sum().item(), 2)
        self.assertEqual(changed[0, 4:8].sum().item(), 2)
        self.assertEqual(changed[0, 8:].sum().item(), 0)

    def test_rejects_insufficient_survivor_capacity(self):
        old_metadata = _make_metadata([[0, 1, 2, 3]])

        with self.assertRaisesRegex(RuntimeError, "insufficient survivor"):
            _recover(old_metadata, [True, True, True, False])


if __name__ == "__main__":
    unittest.main()
