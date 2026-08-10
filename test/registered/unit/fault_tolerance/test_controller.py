import unittest
from types import SimpleNamespace

from sglang.srt.fault_tolerance.controller import (
    FaultToleranceState,
    is_ft_supported_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_state(*, dp_size=2, ranks_per_dp=1, strategy="pause"):
    return FaultToleranceState(
        dp_size=dp_size,
        strategy=strategy,
        global_rank_count=dp_size * ranks_per_dp,
    )


class TestFaultToleranceState(CustomTestCase):
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

    def test_process_mask_keeps_global_rank_granularity(self):
        state = make_state(ranks_per_dp=2)
        state.observe_process_active_ranks([2], active=False)

        self.assertEqual(state.process_alive_mask, [True, True, False, True])
        self.assertEqual(state.process_alive_dp_mask(), [True, False])
        self.assertEqual(state.status_response()["ranks"][1]["state"], "dead")

    def test_sparse_expected_mask_expands_to_whole_dp_blocks(self):
        state = make_state(dp_size=4, ranks_per_dp=2)
        self.assertEqual(
            state.expand_dp_mask([True, True, False, True]),
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

    def test_rejoin_needs_native_ready_before_disabled(self):
        state = make_state()
        state.expected_dp_mask[1] = False
        state.observe_process_active_ranks([1], active=False)
        state.observe_process_active_ranks([1], active=True)
        self.assertEqual(state.status_response()["ranks"][1]["state"], "dead")

        state.observe_native_active_ranks([True, True])
        self.assertEqual(state.status_response()["ranks"][1]["state"], "disabled")

        state.observe_process_active_ranks([1], active=False)
        self.assertEqual(state.status_response()["ranks"][1]["state"], "dead")
        self.assertNotIn(1, state.disabled_dp_ranks)

    def test_excluded_dead_dp_does_not_create_new_incident(self):
        state = make_state()
        state.expected_dp_mask[1] = False
        state.observe_process_active_ranks([1], active=False)

        self.assertFalse(state.has_incident())

    def test_pause_admission_uses_incident_and_operation(self):
        state = make_state()
        self.assertFalse(state.should_reject_admission([True, True]))
        state.observe_rank_fault(0)
        self.assertTrue(state.should_reject_admission([True, True]))
        state.finish_retry()
        state.ft_operation_in_progress = True
        self.assertTrue(state.should_reject_admission([True, True]))


class TestNpuFaultToleranceConfig(CustomTestCase):
    def _make_args(self, **overrides):
        values = dict(
            pp_size=1,
            elastic_ep_backend="mc2",
            disaggregation_mode="null",
            device="npu",
            tokenizer_worker_num=1,
            use_ray=False,
            enable_dp_attention=True,
            enable_dp_lm_head=True,
            enable_eplb=True,
            eplb_algorithm="elasticity_aware",
            moe_a2a_backend="deepep",
            deepep_mode="low_latency",
            fault_tolerance_on_error_strategy="pause",
            nnodes=1,
            tp_size=4,
            dp_size=4,
            ep_size=4,
            moe_dp_size=1,
            attn_cp_size=1,
            max_ep_size=None,
            ep_join_mode=None,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_target_ascend_mc2_topology_is_supported(self):
        self.assertEqual(is_ft_supported_config(self._make_args()), (True, ""))

    def test_npu_requires_mc2_active_rank_backend(self):
        self.assertEqual(
            is_ft_supported_config(
                self._make_args(elastic_ep_backend="mooncake")
            ),
            (False, "ft_npu_requires_mc2_active_rank_backend"),
        )

    def test_npu_requires_graph_internal_low_latency_deepep(self):
        self.assertEqual(
            is_ft_supported_config(self._make_args(deepep_mode="normal")),
            (False, "ft_npu_requires_deepep_low_latency"),
        )

    def test_npu_requires_original_tp_dp_ep_namespace_to_match(self):
        self.assertEqual(
            is_ft_supported_config(self._make_args(ep_size=2)),
            (False, "ft_npu_requires_tp_dp_ep_equal"),
        )


if __name__ == "__main__":
    unittest.main()
