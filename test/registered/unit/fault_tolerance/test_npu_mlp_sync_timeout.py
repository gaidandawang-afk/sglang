import unittest
from datetime import timedelta
from unittest.mock import Mock, patch

import torch

from sglang.srt.fault_tolerance.npu_communication import (
    NpuFTCommunication,
    all_gather_into_tensor_with_timeout,
)


class TestNpuMLPSyncTimeout(unittest.TestCase):
    def test_rebuild_survivor_groups_uses_compact_rank_view(self):
        communication = NpuFTCommunication(
            store=Mock(),
            original_rank=2,
            timeout_sec=5,
            mlp_sync_group=Mock(),
            active_original_ranks=(0, 1, 2, 3),
        )
        mlp_sync_group = object()
        eplb_group = object()

        with (
            patch(
                "sglang.srt.utils.init_custom_process_group",
                side_effect=[mlp_sync_group, eplb_group],
            ) as init_group,
            patch(
                "sglang.srt.distributed.parallel_state.get_torch_distributed_pg_options",
                return_value="hccl-options",
            ),
            patch(
                "sglang.srt.fault_tolerance.npu_communication.PrefixStore",
                side_effect=lambda prefix, store: (prefix, store),
            ),
            patch(
                "sglang.srt.fault_tolerance.npu_communication.dist.barrier"
            ) as barrier,
            patch(
                "sglang.srt.fault_tolerance.npu_communication.dist.all_reduce"
            ) as all_reduce,
            patch(
                "sglang.srt.fault_tolerance.npu_communication.set_eplb_process_group_context"
            ) as set_context,
        ):
            communication.rebuild_survivor_groups(
                [False, True, True, True],
                "cpu",
            )

        self.assertEqual(init_group.call_count, 2)
        for call in init_group.call_args_list:
            self.assertEqual(call.kwargs["world_size"], 3)
            self.assertEqual(call.kwargs["rank"], 1)
        self.assertEqual(init_group.call_args_list[0].kwargs["backend"], "gloo")
        self.assertEqual(init_group.call_args_list[1].kwargs["backend"], "hccl")
        barrier.assert_called_once_with(group=mlp_sync_group)
        all_reduce.assert_not_called()
        context = set_context.call_args.args[0]
        self.assertIs(context.group, eplb_group)
        self.assertEqual(context.active_original_ranks, (1, 2, 3))
        self.assertEqual(communication.active_original_ranks, (1, 2, 3))

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

        self.assertEqual(len(logs.output), 2)
        self.assertIn("device_dispatch_enter epoch=1", logs.output[0])
        self.assertIn("forward_mode=DECODE", logs.output[0])
        self.assertIn("device_dispatch_host_return epoch=1", logs.output[1])

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
