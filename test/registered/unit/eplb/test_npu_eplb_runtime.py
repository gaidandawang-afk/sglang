"""CPU tests for NPU-compatible EPLB runtime primitives."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.elastic_ep import elastic_ep
from sglang.srt.eplb import eplb_algorithms, expert_distribution
from sglang.srt.eplb.eplb_algorithms import EplbAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNpuEplbRuntime(CustomTestCase):
    def test_elastic_ep_selects_npu_device(self):
        npu_device = object()

        with (
            patch.object(elastic_ep, "is_cuda", return_value=False),
            patch.object(elastic_ep, "is_npu", return_value=True),
            patch.object(elastic_ep, "is_cpu", return_value=False),
            patch.object(
                elastic_ep.torch, "device", return_value=npu_device
            ) as make_device,
        ):
            selected = elastic_ep.ElasticEPStateManager._select_device()

        self.assertIs(selected, npu_device)
        make_device.assert_called_once_with("npu")

    def test_elasticity_aware_rebalance_uses_cpu_active_mask(self):
        active_ranks_cpu = torch.tensor([1, 0], dtype=torch.int32)
        state = SimpleNamespace(
            active_ranks=object(),
            active_ranks_cpu=active_ranks_cpu,
        )
        expected = object()

        with (
            patch.object(
                elastic_ep.ElasticEPStateManager,
                "instance",
                return_value=state,
            ),
            patch.object(
                eplb_algorithms.elasticity_aware,
                "rebalance_experts",
                return_value=expected,
            ) as rebalance,
        ):
            result = eplb_algorithms.rebalance_experts(
                tokens_per_expert=torch.ones((2, 4)),
                num_physical_experts=8,
                num_local_physical_experts=4,
                num_groups=2,
                num_nodes=1,
                algorithm=EplbAlgorithm.elasticity_aware,
            )

        self.assertIs(result, expected)
        self.assertIs(rebalance.call_args.kwargs["active_ranks"], active_ranks_cpu)

    def test_elasticity_aware_fallback_builds_cpu_mask(self):
        active_ranks_cpu = torch.ones(2, dtype=torch.int32)

        with (
            patch.object(
                elastic_ep.ElasticEPStateManager,
                "instance",
                return_value=None,
            ),
            patch.object(
                elastic_ep.ElasticEPStateManager,
                "healthy_rank_state",
                return_value=active_ranks_cpu,
            ) as healthy_rank_state,
            patch.object(
                eplb_algorithms.elasticity_aware,
                "rebalance_experts",
                return_value=object(),
            ),
        ):
            eplb_algorithms.rebalance_experts(
                tokens_per_expert=torch.ones((2, 4)),
                num_physical_experts=8,
                num_local_physical_experts=4,
                num_groups=2,
                num_nodes=1,
                algorithm=EplbAlgorithm.elasticity_aware,
            )

        healthy_rank_state.assert_called_once_with(device=torch.device("cpu"))

    def test_utilization_metric_uses_configured_device(self):
        metric_tensor = Mock()
        metric_tensor.item.return_value = 0.75
        owner = SimpleNamespace(
            _enable=True,
            _server_args=SimpleNamespace(
                eplb_min_rebalancing_utilization_threshold=0.5,
                device="npu",
            ),
            _rank=0,
            _history=SimpleNamespace(mean=lambda: {4: 0.75}),
            window_sizes=[4],
        )

        with (
            patch.object(
                expert_distribution.torch,
                "tensor",
                return_value=metric_tensor,
            ) as make_tensor,
            patch.object(expert_distribution.torch.distributed, "broadcast"),
        ):
            result = expert_distribution._StatAccumulator._get_global_average_utilization_rate(
                owner
            )

        self.assertEqual(result, 0.75)
        self.assertEqual(make_tensor.call_args.kwargs["device"], "npu")

    def test_non_root_utilization_metric_uses_configured_device(self):
        metric_tensor = Mock()
        metric_tensor.item.return_value = 0.5
        owner = SimpleNamespace(
            _enable=True,
            _server_args=SimpleNamespace(
                eplb_min_rebalancing_utilization_threshold=0.5,
                device="npu",
            ),
            _rank=1,
        )

        with (
            patch.object(
                expert_distribution.torch,
                "empty",
                return_value=metric_tensor,
            ) as make_empty,
            patch.object(expert_distribution.torch.distributed, "broadcast"),
        ):
            result = expert_distribution._StatAccumulator._get_global_average_utilization_rate(
                owner
            )

        self.assertEqual(result, 0.5)
        self.assertEqual(make_empty.call_args.kwargs["device"], "npu")


if __name__ == "__main__":
    unittest.main()
