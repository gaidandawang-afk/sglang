import unittest
from datetime import timedelta
from unittest.mock import Mock, patch

import torch

from sglang.srt.fault_tolerance.npu_communication import (
    NpuFTCommunication,
    all_gather_into_tensor_with_timeout,
)


class TestNpuMLPSyncTimeout(unittest.TestCase):
    def test_trace_correlates_mlp_sync_with_first_device_dispatch(self):
        communication = NpuFTCommunication(
            store=Mock(),
            original_rank=2,
            timeout_sec=5,
            mlp_sync_group=Mock(),
            active_original_ranks=(0, 1, 2, 3),
        )

        with self.assertLogs(
            "sglang.srt.fault_tolerance.npu_communication", level="INFO"
        ) as logs:
            communication.record_mlp_sync_complete(
                local_forward_mode="DECODE",
                num_tokens=1,
            )
            communication.record_device_dispatch_enter()
            communication.record_device_dispatch_enter()
            communication.record_device_dispatch_host_return()
            communication.record_device_dispatch_host_return()

        self.assertEqual(len(logs.output), 3)
        self.assertIn("mlp_sync_complete epoch=1", logs.output[0])
        self.assertIn("device_dispatch_enter epoch=1", logs.output[1])
        self.assertIn("forward_mode=DECODE", logs.output[1])
        self.assertIn("device_dispatch_host_return epoch=1", logs.output[2])

    def test_mlp_sync_gather_has_bounded_wait(self):
        work = Mock()
        work.wait.return_value = True
        output = torch.empty(3)
        local = torch.tensor([2])

        with patch(
            "sglang.srt.fault_tolerance.npu_communication.dist.all_gather_into_tensor",
            return_value=work,
        ) as all_gather:
            all_gather_into_tensor_with_timeout(
                output,
                local,
                group="mlp-sync-gloo",
                timeout_sec=5,
            )

        all_gather.assert_called_once_with(
            output,
            local,
            group="mlp-sync-gloo",
            async_op=True,
        )
        work.wait.assert_called_once_with(timeout=timedelta(seconds=5))

    def test_mlp_sync_timeout_enters_ft_control_loop(self):
        work = Mock()
        work.wait.return_value = False

        with (
            patch(
                "sglang.srt.fault_tolerance.npu_communication.dist.all_gather_into_tensor",
                return_value=work,
            ),
            self.assertRaisesRegex(RuntimeError, "entering the FT control loop"),
        ):
            all_gather_into_tensor_with_timeout(
                torch.empty(3),
                torch.tensor([2]),
                group="mlp-sync-gloo",
                timeout_sec=5,
            )

    def test_mlp_sync_wait_failure_is_tagged(self):
        work = Mock()
        work.wait.side_effect = RuntimeError("peer disconnected")

        with (
            patch(
                "sglang.srt.fault_tolerance.npu_communication.dist.all_gather_into_tensor",
                return_value=work,
            ),
            self.assertRaisesRegex(RuntimeError, "entering the FT control loop"),
        ):
            all_gather_into_tensor_with_timeout(
                torch.empty(3),
                torch.tensor([2]),
                group="mlp-sync-gloo",
                timeout_sec=5,
            )


if __name__ == "__main__":
    unittest.main()
