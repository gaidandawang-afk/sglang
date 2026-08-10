"""CPU-mock tests for TorchNPU device recovery during FT scale-down."""

import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNpuFTDeviceRecovery(CustomTestCase):
    def _make_runner(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner._npu_ft_device_owner_thread_id = threading.get_ident()
        runner.gpu_id = 2
        runner.ps = SimpleNamespace(dp_rank=3)
        return runner

    def test_stop_restart_run_in_order_without_resource_rebuild_flag(self):
        runner = self._make_runner()
        calls = []
        fake_npu = SimpleNamespace(
            stop_device=Mock(
                side_effect=lambda device_id: calls.append(("stop", device_id))
            ),
            restart_device=Mock(
                side_effect=lambda device_id: calls.append(("restart", device_id))
            ),
        )

        with patch.dict(sys.modules, {"torch_npu": SimpleNamespace(npu=fake_npu)}):
            runner._stop_and_restart_npu_device_for_fault_tolerance()

        self.assertEqual(calls, [("stop", 2), ("restart", 2)])
        fake_npu.restart_device.assert_called_once_with(2)

    def test_cross_thread_call_fails_before_touching_torch_npu(self):
        runner = self._make_runner()
        fake_npu = SimpleNamespace(stop_device=Mock(), restart_device=Mock())

        with (
            patch.dict(sys.modules, {"torch_npu": SimpleNamespace(npu=fake_npu)}),
            patch(
                "sglang.srt.model_executor.model_runner.threading.get_ident",
                return_value=runner._npu_ft_device_owner_thread_id + 1,
            ),
            self.assertRaisesRegex(RuntimeError, "device owner thread"),
        ):
            runner._stop_and_restart_npu_device_for_fault_tolerance()

        fake_npu.stop_device.assert_not_called()
        fake_npu.restart_device.assert_not_called()

    def test_scale_down_recovers_device_before_rebuilding_process_groups(self):
        runner = self._make_runner()
        runner.server_args = SimpleNamespace(elastic_ep_backend="mc2")
        runner.device = "cpu"
        calls = []
        runner._stop_and_restart_npu_device_for_fault_tolerance = lambda: calls.append(
            "device-restart"
        )

        elastic_info_tensor = torch.arange(12, dtype=torch.int32)
        elastic_info = SimpleNamespace(
            tensor=elastic_info_tensor,
            validate_logical_expert_capacity=lambda *args, **kwargs: calls.append(
                "validate-capacity"
            ),
        )
        state = SimpleNamespace(
            active_ranks=torch.ones(4, dtype=torch.int32),
            active_ranks_cpu=torch.ones(4, dtype=torch.int32),
            npu_mc2_elastic_info=elastic_info,
            update_npu_mc2_elastic_info=lambda: calls.append("commit-elastic-info"),
            snapshot_active_to_last=lambda: calls.append("snapshot-mask"),
        )

        class _Groups:
            def rebuild(self, **kwargs):
                calls.append("rebuild-groups")

            def all_reduce_sum_tensor(self, tensor):
                calls.append("eplb-all-reduce")
                return tensor

        groups = _Groups()

        class _EPLBManager:
            def rebalance(self, **kwargs):
                calls.append("eplb-rebalance")
                return iter(())

        runner.eplb_manager = _EPLBManager()
        expert_location = SimpleNamespace(num_logical_experts=128)
        recorder = SimpleNamespace(
            dump_local_logical_count=lambda: torch.ones(128, dtype=torch.int64)
        )

        with (
            patch(
                "sglang.srt.model_executor.model_runner.ElasticEPStateManager.instance",
                return_value=state,
            ),
            patch(
                "sglang.srt.model_executor.model_runner.get_global_expert_location_metadata",
                return_value=expert_location,
            ),
            patch(
                "sglang.srt.fault_tolerance.npu_metadata.get_npu_ft_metadata_group",
                return_value=groups,
            ),
            patch(
                "sglang.srt.model_executor.model_runner.get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
        ):
            runner.apply_fault_tolerance_scale_down([True, False, True, True])

        self.assertLess(calls.index("device-restart"), calls.index("rebuild-groups"))
        self.assertLess(calls.index("rebuild-groups"), calls.index("eplb-rebalance"))
        self.assertLess(
            calls.index("eplb-rebalance"), calls.index("commit-elastic-info")
        )


if __name__ == "__main__":
    unittest.main()
