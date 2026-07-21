import unittest
from types import SimpleNamespace

from sglang.srt.fault_tolerance.controller import (
    FaultToleranceState,
    is_ft_supported_config,
)


class TestFaultToleranceState(unittest.TestCase):
    def test_single_dp_is_rejected(self):
        args = SimpleNamespace(
            pp_size=1,
            elastic_ep_backend="mooncake",
            disaggregation_mode="null",
            device="cuda",
            tokenizer_worker_num=1,
            use_ray=False,
            enable_dp_attention=False,
            tp_size=1,
            dp_size=1,
            attn_cp_size=1,
        )

        self.assertEqual(
            is_ft_supported_config(args),
            (False, "ft_requires_dp_gt1"),
        )

    def test_invalid_dp_attention_topology_is_rejected(self):
        args = SimpleNamespace(
            pp_size=1,
            elastic_ep_backend="mooncake",
            disaggregation_mode="null",
            device="cuda",
            tokenizer_worker_num=1,
            use_ray=False,
            enable_dp_attention=True,
            tp_size=3,
            dp_size=2,
            attn_cp_size=1,
        )

        self.assertEqual(
            is_ft_supported_config(args),
            (False, "ft_requires_tp_divisible_by_dp_and_attn_cp"),
        )

    def test_effective_mask_is_intersection_of_three_sources(self):
        state = FaultToleranceState(dp_size=3, strategy="pause")
        state.process_active_ranks = [True, False, True]
        state.mooncake_active_ranks = [True, True, False]
        state.disabled_dp_ranks.add(0)

        self.assertEqual(state.runtime_active_mask(), [True, False, False])
        self.assertEqual(state.effective_active_mask(), [False, False, False])

    def test_process_falling_edge_pauses_runtime_ranks_once(self):
        state = FaultToleranceState(dp_size=3, strategy="pause")

        self.assertEqual(
            state.observe_process_active_ranks([1], active=False),
            [0, 2],
        )
        self.assertTrue(state.ft_operation_in_progress)
        self.assertEqual(
            state.observe_process_active_ranks([1], active=False),
            [],
        )
        state.finish_pause({0, 2})

        self.assertEqual(state.paused_dp_ranks, {0, 2})
        self.assertEqual(
            state.status_response()["ranks"],
            [
                {"rank": 0, "state": "paused"},
                {"rank": 1, "state": "dead"},
                {"rank": 2, "state": "paused"},
            ],
        )

    def test_no_effective_route_rejects_admission_for_continue_strategy(self):
        state = FaultToleranceState(dp_size=2, strategy="continue")
        state.process_active_ranks = [False, False]

        self.assertTrue(state.should_reject_admission())

    def test_only_explicit_recovery_clears_disabled(self):
        state = FaultToleranceState(dp_size=2, strategy="continue")
        state.disabled_dp_ranks.add(1)

        state.observe_mooncake_active_ranks([True, False])
        self.assertEqual(state.disabled_dp_ranks, {1})
        state.observe_mooncake_active_ranks([True, True])
        self.assertEqual(state.disabled_dp_ranks, {1})

        state.observe_recovered_dp_ranks([1])
        self.assertEqual(state.disabled_dp_ranks, set())

    def test_process_rejoin_updates_only_reported_ranks(self):
        state = FaultToleranceState(dp_size=3, strategy="continue")
        state.process_active_ranks = [False, False, True]

        state.observe_process_active_ranks([0], active=True)

        self.assertEqual(state.process_active_ranks, [True, False, True])

    def test_scale_down_only_changes_disabled_source(self):
        state = FaultToleranceState(dp_size=2, strategy="pause")
        state.begin_exception_pause()
        state.finish_pause({0, 1})

        resume_targets = state.begin_recover("scale_down", [1])

        self.assertEqual(state.effective_active_mask(), [True, False])
        self.assertEqual(resume_targets, [0, 1])
        self.assertEqual(state.disabled_dp_ranks, {1})
        self.assertEqual(state.process_active_ranks, [True, True])
        self.assertEqual(state.mooncake_active_ranks, [True, True])

        response = state.commit_recover({0, 1})
        self.assertEqual(response["resumed_ranks"], [0, 1])
        self.assertEqual(
            response["ranks"],
            [
                {"rank": 0, "state": "healthy"},
                {"rank": 1, "state": "dead"},
            ],
        )

    def test_scale_down_cannot_remove_last_effective_route(self):
        state = FaultToleranceState(dp_size=2, strategy="pause")
        state.process_active_ranks = [True, False]

        self.assertEqual(
            state.validate_scale_down_ranks([0]),
            "cannot_isolate_all_active_ranks",
        )

    def test_resume_targets_use_paused_runtime_sources(self):
        state = FaultToleranceState(dp_size=3, strategy="pause")
        state.paused_dp_ranks = {0, 1, 2}
        state.process_active_ranks = [True, False, True]
        state.mooncake_active_ranks = [True, True, False]

        self.assertEqual(state.resume_targets(), [0])

    def test_duplicate_exception_pause_is_ignored(self):
        state = FaultToleranceState(dp_size=2, strategy="pause")

        self.assertEqual(state.begin_exception_pause(), [0, 1])
        self.assertEqual(state.begin_exception_pause(), [])
        state.finish_pause({0, 1})
        self.assertEqual(state.begin_exception_pause(), [])


if __name__ == "__main__":
    unittest.main()
