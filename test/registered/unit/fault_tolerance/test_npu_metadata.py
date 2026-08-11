"""CPU tests for rebuilt Ascend FT graph-external process groups."""

import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import torch

from sglang.srt.fault_tolerance.npu_metadata import (
    NpuFTSurvivorProcessGroups,
    prewarm_npu_ft_original_mlp_sync_group,
)
from sglang.srt.managers.prefill_delayer import PrefillDelayer
from sglang.srt.managers.scheduler_components import dp_attn
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNpuFTSurvivorProcessGroups(CustomTestCase):
    def test_original_mlp_sync_gloo_group_is_prewarmed_before_faults(self):
        with patch(
            "sglang.srt.fault_tolerance.npu_metadata.dist.barrier"
        ) as barrier:
            prewarm_npu_ft_original_mlp_sync_group(
                "original-gloo",
                original_rank=3,
            )

        barrier.assert_called_once_with(group="original-gloo")

    def test_rebuild_uses_compact_rank_and_separate_graph_external_groups(self):
        groups = NpuFTSurvivorProcessGroups(
            store=object(),
            original_rank=2,
            original_world_size=4,
            timeout_sec=5,
        )

        with (
            patch(
                "sglang.srt.fault_tolerance.npu_metadata.PrefixStore",
                side_effect=lambda prefix, store: f"store:{prefix}",
            ),
            patch(
                "sglang.srt.utils.init_custom_process_group",
                side_effect=["gloo-pg", "scheduler-hccl-pg", "eplb-hccl-pg"],
            ) as init_group,
            patch(
                "sglang.srt.distributed.parallel_state.get_torch_distributed_pg_options",
                return_value="hccl-options",
            ),
            patch("sglang.srt.fault_tolerance.npu_metadata.dist.barrier") as barrier,
            patch(
                "sglang.srt.fault_tolerance.npu_metadata.dist.all_reduce"
            ) as all_reduce,
        ):
            groups.rebuild(
                active_ranks=[1, 0, 1, 1],
                device=torch.device("cpu"),
            )

        self.assertEqual(groups.active_original_ranks, (0, 2, 3))
        self.assertEqual(groups.compact_rank, 1)
        self.assertEqual(groups.cpu_group, "gloo-pg")
        self.assertEqual(groups.scheduler_device_group, "scheduler-hccl-pg")
        self.assertEqual(groups.eplb_device_group, "eplb-hccl-pg")
        self.assertEqual(init_group.call_count, 3)
        self.assertEqual(init_group.call_args_list[0].kwargs["backend"], "gloo")
        self.assertEqual(init_group.call_args_list[1].kwargs["backend"], "hccl")
        self.assertEqual(init_group.call_args_list[2].kwargs["backend"], "hccl")
        for group_call in init_group.call_args_list:
            self.assertEqual(group_call.kwargs["rank"], 1)
            self.assertEqual(group_call.kwargs["world_size"], 3)
        barrier.assert_called_once_with(group="gloo-pg")
        self.assertEqual(
            [item.kwargs["group"] for item in all_reduce.call_args_list],
            ["scheduler-hccl-pg", "eplb-hccl-pg"],
        )

    def test_repeated_scale_down_retires_old_groups_without_destroying_them(self):
        groups = NpuFTSurvivorProcessGroups(
            store=object(),
            original_rank=0,
            original_world_size=4,
            timeout_sec=5,
        )

        with (
            patch(
                "sglang.srt.fault_tolerance.npu_metadata.PrefixStore",
                side_effect=lambda prefix, store: f"store:{prefix}",
            ),
            patch(
                "sglang.srt.utils.init_custom_process_group",
                side_effect=[
                    "gloo-1",
                    "scheduler-hccl-1",
                    "eplb-hccl-1",
                    "gloo-2",
                    "scheduler-hccl-2",
                    "eplb-hccl-2",
                ],
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_torch_distributed_pg_options"
            ),
            patch("sglang.srt.fault_tolerance.npu_metadata.dist.barrier"),
            patch("sglang.srt.fault_tolerance.npu_metadata.dist.all_reduce"),
            patch(
                "sglang.srt.fault_tolerance.npu_metadata.dist.destroy_process_group"
            ) as destroy,
        ):
            groups.rebuild(
                active_ranks=[1, 0, 1, 1], device=torch.device("cpu")
            )
            groups.rebuild(
                active_ranks=[1, 0, 1, 0], device=torch.device("cpu")
            )

        self.assertEqual(groups.generation, 2)
        self.assertEqual(
            groups._retired_groups,
            ["gloo-1", "scheduler-hccl-1", "eplb-hccl-1"],
        )
        destroy.assert_not_called()

    def test_device_collectives_keep_original_rank_namespace(self):
        groups = NpuFTSurvivorProcessGroups(
            store=object(),
            original_rank=2,
            original_world_size=4,
            timeout_sec=5,
            generation=1,
            active_original_ranks=(0, 2, 3),
            cpu_group="gloo-pg",
            scheduler_device_group="scheduler-hccl-pg",
            eplb_device_group="eplb-hccl-pg",
        )

        def fake_all_gather(outputs, local, group):
            self.assertEqual(group, "scheduler-hccl-pg")
            for output, value in zip(outputs, (10, 20, 30), strict=True):
                output.fill_(value)

        def fake_all_reduce(tensor, **kwargs):
            self.assertEqual(kwargs["group"], "eplb-hccl-pg")
            tensor.add_(7)

        with (
            patch(
                "sglang.srt.fault_tolerance.npu_metadata.dist.all_gather",
                side_effect=fake_all_gather,
            ),
            patch(
                "sglang.srt.fault_tolerance.npu_metadata.dist.all_reduce",
                side_effect=fake_all_reduce,
            ),
        ):
            gathered = groups.all_gather_tensor(torch.tensor([2]))
            reduced = groups.all_reduce_sum_tensor(torch.tensor([3]))

        self.assertEqual(
            {rank: tensor.item() for rank, tensor in gathered.items()},
            {0: 10, 2: 20, 3: 30},
        )
        self.assertEqual(reduced.item(), 10)

    def test_rebuilt_mlp_sync_gather_has_bounded_wait(self):
        groups = NpuFTSurvivorProcessGroups(
            store=object(),
            original_rank=2,
            original_world_size=4,
            timeout_sec=600,
            generation=1,
            active_original_ranks=(0, 2, 3),
            cpu_group="gloo-pg",
            scheduler_device_group="scheduler-hccl-pg",
            eplb_device_group="eplb-hccl-pg",
        )
        work = Mock()
        work.wait.return_value = False

        with (
            patch(
                "sglang.srt.fault_tolerance.npu_metadata.dist.all_gather",
                return_value=work,
            ) as all_gather,
            self.assertRaisesRegex(
                TimeoutError, "returning to the FT control loop"
            ),
        ):
            groups.all_gather_tensor(torch.tensor([2]), timeout_sec=5)

        all_gather.assert_called_once_with(
            [ANY, ANY, ANY],
            ANY,
            group="scheduler-hccl-pg",
            async_op=True,
        )
        work.wait.assert_called_once_with(timeout=timedelta(seconds=5))

    def test_inactive_rank_cannot_rebuild(self):
        groups = NpuFTSurvivorProcessGroups(
            store=object(),
            original_rank=1,
            original_world_size=4,
            timeout_sec=5,
        )
        with self.assertRaisesRegex(RuntimeError, "inactive original rank"):
            groups.rebuild(
                active_ranks=[1, 0, 1, 1], device=torch.device("cpu")
            )

    def test_cpu_gather_uses_rebuilt_gloo_group(self):
        groups = NpuFTSurvivorProcessGroups(
            store=object(),
            original_rank=2,
            original_world_size=4,
            timeout_sec=5,
            generation=1,
            active_original_ranks=(0, 2, 3),
            cpu_group="gloo-pg",
            scheduler_device_group="scheduler-hccl-pg",
            eplb_device_group="eplb-hccl-pg",
        )

        def fake_all_gather(outputs, local, group):
            self.assertEqual(group, "gloo-pg")
            self.assertEqual(local.device.type, "cpu")
            for output, value in zip(outputs, (1, 2, 3), strict=True):
                output.fill_(value)

        with patch(
            "sglang.srt.fault_tolerance.npu_metadata.dist.all_gather",
            side_effect=fake_all_gather,
        ):
            gathered = groups.all_gather_cpu_tensor(torch.tensor([9]))

        self.assertEqual(
            {rank: tensor.item() for rank, tensor in gathered.items()},
            {0: 1, 2: 2, 3: 3},
        )

    def test_control_broadcast_uses_rebuilt_gloo_group(self):
        groups = NpuFTSurvivorProcessGroups(
            store=object(),
            original_rank=2,
            original_world_size=4,
            timeout_sec=5,
            generation=1,
            active_original_ranks=(0, 2, 3),
            cpu_group="gloo-pg",
            scheduler_device_group="scheduler-hccl-pg",
            eplb_device_group="eplb-hccl-pg",
        )

        def fake_broadcast(payload, **kwargs):
            self.assertEqual(payload, [None])
            payload[0] = ["from-root"]

        with patch(
            "sglang.srt.fault_tolerance.npu_metadata.dist.broadcast_object_list",
            side_effect=fake_broadcast,
        ) as broadcast:
            result = groups.broadcast_control(["local-copy"])

        self.assertEqual(result, ["from-root"])
        self.assertEqual(broadcast.call_args.args[0], [["from-root"]])
        self.assertEqual(
            broadcast.call_args.kwargs, {"src": 0, "group": "gloo-pg"}
        )

    def test_prefill_delayer_gathers_only_survivors_over_rebuilt_gloo(self):
        class _Groups:
            is_rebuilt = True
            active_original_ranks = (0, 2, 3)

            def all_gather_cpu_tensor(self, local_tensor):
                self.local_tensor = local_tensor
                return {
                    0: torch.tensor([1, 0, 4, 8, 2]),
                    2: torch.tensor([0, 0, 3, 8, 1]),
                    3: torch.tensor([1, 1, 2, 8, 0]),
                }

        groups = _Groups()
        delayer = PrefillDelayer.__new__(PrefillDelayer)
        delayer._gather_device = "cpu"
        delayer._gather_group = "old-gloo"
        delayer._global_info_buffer = torch.empty((4, 1, 5), dtype=torch.int64)

        resources = SimpleNamespace(
            buffers={"npu_ft_survivor_process_groups": groups}
        )
        with (
            patch(
                "sglang.srt.runtime_context.get_resources", return_value=resources
            ),
            patch.object(
                torch.distributed, "all_gather_into_tensor"
            ) as old_all_gather,
        ):
            gathered = delayer._gather_info(
                local_prefillable=True,
                local_token_watermark_force_allow=False,
                running_batch=4,
                max_prefill_bs=8,
                waiting_queue_len=2,
            )

        self.assertEqual(gathered.shape, (3, 5))
        self.assertEqual(gathered[:, 0].tolist(), [1, 0, 1])
        old_all_gather.assert_not_called()

    def test_prefill_delayer_non_ft_path_keeps_original_group(self):
        delayer = PrefillDelayer.__new__(PrefillDelayer)
        delayer._gather_device = "cpu"
        delayer._gather_group = "original-gloo"
        delayer._global_info_buffer = torch.empty((2, 1, 5), dtype=torch.int64)

        def fake_all_gather(output, local, group):
            self.assertEqual(group, "original-gloo")
            output.view(2, 5)[0].copy_(local)
            output.view(2, 5)[1].copy_(local + 1)

        with (
            patch(
                "sglang.srt.runtime_context.get_resources",
                return_value=SimpleNamespace(buffers={}),
            ),
            patch.object(
                torch.distributed,
                "all_gather_into_tensor",
                side_effect=fake_all_gather,
            ) as old_all_gather,
        ):
            gathered = delayer._gather_info(
                local_prefillable=True,
                local_token_watermark_force_allow=False,
                running_batch=4,
                max_prefill_bs=8,
                waiting_queue_len=2,
            )

        self.assertEqual(gathered.shape, (2, 5))
        old_all_gather.assert_called_once()

    def test_mlp_sync_maps_survivors_back_to_fixed_original_slots(self):
        class _GatherGroup:
            def all_gather_tensor(self, local_tensor, *, timeout_sec=None):
                self.local_tensor = local_tensor
                self.timeout_sec = timeout_sec
                return {
                    0: torch.tensor([2, 2, 1, 0, 1, ForwardMode.DECODE.value, 0]),
                    2: torch.tensor([3, 3, 1, 0, 1, ForwardMode.DECODE.value, 0]),
                    3: torch.tensor([4, 4, 1, 0, 1, ForwardMode.DECODE.value, 0]),
                }

        gather_group = _GatherGroup()
        sync_info = dp_attn.MLPSyncBatchInfo(
            dp_size=4,
            tp_size=1,
            cp_size=1,
            num_tokens=3,
            num_tokens_for_logprob=3,
            can_cuda_graph=True,
            is_extend_in_batch=False,
            local_can_run_tbo=True,
            local_forward_mode=ForwardMode.DECODE.value,
            can_run_breakable_cuda_graph=False,
        )
        active_ranks = torch.tensor([1, 0, 1, 1], dtype=torch.int32)
        tp_group = SimpleNamespace(
            active_ranks_cpu=active_ranks,
            active_ranks=active_ranks,
        )

        with (
            patch.object(
                dp_attn, "_get_npu_ft_active_ranks", return_value=active_ranks
            ),
            patch(
                "sglang.srt.fault_tolerance.npu_metadata.get_npu_ft_metadata_group",
                return_value=gather_group,
            ),
            patch.object(dp_attn, "get_tp_group", return_value=tp_group),
        ):
            sync_info.all_gather(device="cpu", group=None)

        self.assertEqual(sync_info.global_num_tokens, [2, 0, 3, 4])
        self.assertEqual(sync_info.tp0_info[1, 5].item(), ForwardMode.IDLE.value)
        self.assertEqual(gather_group.timeout_sec, 5)

    def test_pre_scale_mlp_sync_has_bounded_wait_for_ft_control_liveness(self):
        work = Mock()
        work.wait.return_value = True
        output = torch.empty(4, dtype=torch.int64)
        local = torch.ones(1, dtype=torch.int64)

        with (
            patch.object(dp_attn, "_is_npu_mc2_ft_enabled", return_value=True),
            patch.object(
                torch.distributed,
                "all_gather_into_tensor",
                return_value=work,
            ) as all_gather,
        ):
            dp_attn._all_gather_original_mlp_sync_group(
                output,
                local,
                group="original-gloo",
            )

        all_gather.assert_called_once_with(
            output,
            local,
            group="original-gloo",
            async_op=True,
        )
        work.wait.assert_called_once_with(timeout=timedelta(seconds=5))

    def test_pre_scale_mlp_sync_timeout_returns_to_ft_loop(self):
        work = Mock()
        work.wait.return_value = False

        with (
            patch.object(dp_attn, "_is_npu_mc2_ft_enabled", return_value=True),
            patch.object(
                torch.distributed,
                "all_gather_into_tensor",
                return_value=work,
            ),
            self.assertRaisesRegex(
                TimeoutError, "returning to the FT control loop"
            ),
        ):
            dp_attn._all_gather_original_mlp_sync_group(
                torch.empty(4, dtype=torch.int64),
                torch.ones(1, dtype=torch.int64),
                group="original-gloo",
            )

    def test_non_ft_mlp_sync_keeps_synchronous_collective(self):
        output = torch.empty(4, dtype=torch.int64)
        local = torch.ones(1, dtype=torch.int64)

        with (
            patch.object(dp_attn, "_is_npu_mc2_ft_enabled", return_value=False),
            patch.object(
                torch.distributed, "all_gather_into_tensor"
            ) as all_gather,
        ):
            dp_attn._all_gather_original_mlp_sync_group(
                output,
                local,
                group="original-gloo",
            )

        all_gather.assert_called_once_with(
            output,
            local,
            group="original-gloo",
        )


if __name__ == "__main__":
    unittest.main()
