import unittest
from types import SimpleNamespace

from sglang.srt.fault_tolerance.controller import (
    FaultToleranceState,
    is_ft_supported_config,
)


def make_state(*, dp_size=2, ranks_per_dp=1, strategy="pause"):
    return FaultToleranceState(
        dp_size=dp_size,
        strategy=strategy,
        global_rank_count=dp_size * ranks_per_dp,
    )


class TestFaultToleranceState(unittest.TestCase):
    def test_npu_mc2_configuration_is_supported(self):
        args = SimpleNamespace(
            pp_size=1,
            elastic_ep_backend="mc2",
            disaggregation_mode="null",
            device="npu",
            tokenizer_worker_num=1,
            use_ray=False,
            enable_dp_attention=True,
            enable_eplb=True,
            tp_size=4,
            dp_size=4,
            ep_size=4,
            moe_dp_size=1,
            attn_cp_size=1,
        )

        self.assertEqual(is_ft_supported_config(args), (True, ""))

    def test_single_dp_is_rejected(self):
        args = SimpleNamespace(
            pp_size=1,
            elastic_ep_backend="mooncake",
            disaggregation_mode="null",
            device="cuda",
            tokenizer_worker_num=1,
            use_ray=False,
            enable_dp_attention=True,
            enable_eplb=True,
            tp_size=1,
            dp_size=1,
            attn_cp_size=1,
        )
        self.assertEqual(is_ft_supported_config(args), (False, "ft_requires_dp_gt1"))

    def test_invalid_dp_attention_topology_is_rejected(self):
        args = SimpleNamespace(
            pp_size=1,
            elastic_ep_backend="mooncake",
            disaggregation_mode="null",
            device="cuda",
            tokenizer_worker_num=1,
            use_ray=False,
            enable_dp_attention=True,
            enable_eplb=True,
            tp_size=3,
            dp_size=2,
            attn_cp_size=1,
        )
        self.assertEqual(
            is_ft_supported_config(args),
            (False, "ft_requires_tp_divisible_by_dp_and_attn_cp"),
        )

    def test_runtime_ep_capacity_different_from_static_tp_size_is_rejected(self):
        args = SimpleNamespace(
            pp_size=1,
            elastic_ep_backend="mooncake",
            disaggregation_mode="null",
            device="cuda",
            tokenizer_worker_num=1,
            use_ray=False,
            enable_dp_attention=True,
            enable_eplb=True,
            tp_size=4,
            dp_size=2,
            attn_cp_size=1,
            max_ep_size=2,
            ep_join_mode=None,
        )
        self.assertEqual(
            is_ft_supported_config(args),
            (False, "ft_unsupported_with_runtime_ep_scale"),
        )

        args.max_ep_size = 8
        self.assertEqual(
            is_ft_supported_config(args),
            (False, "ft_unsupported_with_runtime_ep_scale"),
        )

        args.max_ep_size = 4
        self.assertEqual(is_ft_supported_config(args), (True, ""))

    def test_process_mask_keeps_global_rank_granularity(self):
        state = make_state(ranks_per_dp=2)
        state.observe_process_active_ranks([2], active=False)

        self.assertEqual(
            state.process_alive_global_rank_mask, [True, True, False, True]
        )
        self.assertEqual(state.process_alive_dp_mask(), [True, False])
        self.assertEqual(state.status_response()["ranks"][1]["state"], "dead")

    def test_sparse_expected_mask_expands_to_whole_dp_blocks(self):
        state = make_state(dp_size=4, ranks_per_dp=2)
        self.assertEqual(
            state.expand_dp_mask_to_global_rank_mask([True, True, False, True]),
            [True, True, True, True, False, False, True, True],
        )

    def test_exception_is_unhealthy_without_paused_state(self):
        state = make_state()
        state.observe_rank_fault(1)

        self.assertEqual(
            state.status_response()["ranks"],
            [
                {"rank": 0, "state": "healthy"},
                {"rank": 1, "state": "unhealthy"},
            ],
        )
        self.assertTrue(state.has_incident())

    def test_non_expected_dp_stays_dead_until_manager_reopens_it(self):
        state = make_state()
        state.expected_dp_mask[1] = False
        state.observe_process_active_ranks([1], active=False)
        state.observe_process_active_ranks([1], active=True)
        self.assertEqual(state.status_response()["ranks"][1]["state"], "dead")
        self.assertEqual(state.pending_recovery_global_ranks, {1})

        state.observe_native_active_dp_mask([True, True])
        self.assertEqual(state.pending_recovery_global_ranks, set())
        self.assertEqual(state.status_response()["ranks"][1]["state"], "dead")

        state.observe_process_active_ranks([1], active=False)
        self.assertEqual(state.status_response()["ranks"][1]["state"], "dead")

    def test_native_active_dp_clears_all_pending_global_ranks_in_dp(self):
        state = make_state(dp_size=2, ranks_per_dp=2)
        state.observe_process_active_ranks([2, 3], active=False)

        state.observe_native_active_dp_mask([True, False])
        self.assertEqual(state.pending_recovery_global_ranks, {2, 3})

        state.observe_native_active_dp_mask([True, True])
        self.assertEqual(state.pending_recovery_global_ranks, set())

    def test_excluded_dead_dp_does_not_create_new_incident(self):
        state = make_state()
        state.expected_dp_mask[1] = False
        state.observe_process_active_ranks([1], active=False)

        self.assertFalse(state.has_incident())

    def test_pause_admission_uses_incident_and_operation(self):
        state = make_state()
        self.assertFalse(state.should_reject_admission([True, True]))
        state.observe_rank_fault(0)
        self.assertTrue(state.cluster_paused)
        self.assertTrue(state.should_reject_admission([True, True]))
        state.finish_retry()
        self.assertFalse(state.cluster_paused)
        state.ft_operation_in_progress = True
        self.assertTrue(state.should_reject_admission([True, True]))

    def test_continue_admission_is_not_paused_by_faults(self):
        state = make_state(strategy="continue")
        state.observe_process_active_ranks([1], active=False)
        state.observe_rank_fault(0)
        # continue never pauses admission; faults only drop requests / update status.
        self.assertFalse(state.should_reject_admission([True, True]))
        self.assertEqual(state.unhealthy_dp_ranks, set())
        self.assertEqual(state.status_response()["ranks"][1]["state"], "dead")


if __name__ == "__main__":
    unittest.main()
