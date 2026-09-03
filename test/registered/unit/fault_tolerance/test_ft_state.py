import unittest

from sglang.srt.fault_tolerance.ft_state import FaultToleranceState


def make_state(*, dp_size=2, ranks_per_dp=1, strategy="pause"):
    return FaultToleranceState(
        dp_size=dp_size,
        strategy=strategy,
        global_rank_count=dp_size * ranks_per_dp,
    )


class TestFaultToleranceState(unittest.TestCase):
    def test_status_uses_vllm_engine_schema_for_all_dps(self):
        state = make_state(dp_size=2)

        self.assertEqual(
            state.status_response(),
            {
                "schema_version": 1,
                "total_engines": 2,
                "engines": [
                    {"id": 0, "status": "healthy"},
                    {"id": 1, "status": "healthy"},
                ],
            },
        )

    def test_process_mask_keeps_global_rank_granularity(self):
        state = make_state(ranks_per_dp=2)
        state.observe_process_active_ranks([2], active=False)

        self.assertEqual(
            state.process_alive_global_rank_mask, [True, True, False, True]
        )
        self.assertEqual(state.process_alive_dp_mask(), [True, False])
        self.assertEqual(state.status_response()["engines"][1]["status"], "dead")

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
            state.status_response()["engines"],
            [
                {"id": 0, "status": "healthy"},
                {"id": 1, "status": "unhealthy"},
            ],
        )
        self.assertTrue(state.has_unresolved_expected_dp_fault())

    def test_non_expected_dp_stays_dead_until_manager_reopens_it(self):
        state = make_state()
        state.expected_dp_mask[1] = False
        state.observe_process_active_ranks([1], active=False)
        state.observe_process_active_ranks([1], active=True)
        self.assertEqual(state.status_response()["engines"][1]["status"], "dead")
        self.assertEqual(state.pending_recovery_global_ranks, {1})

        state.observe_runtime_active_dp_mask([True, True])
        self.assertEqual(state.pending_recovery_global_ranks, set())
        self.assertEqual(state.status_response()["engines"][1]["status"], "dead")

        state.observe_process_active_ranks([1], active=False)
        self.assertEqual(state.status_response()["engines"][1]["status"], "dead")

    def test_runtime_active_dp_clears_all_pending_global_ranks_in_dp(self):
        state = make_state(dp_size=2, ranks_per_dp=2)
        state.observe_process_active_ranks([2, 3], active=False)

        state.observe_runtime_active_dp_mask([True, False])
        self.assertEqual(state.pending_recovery_global_ranks, {2, 3})

        state.observe_runtime_active_dp_mask([True, True])
        self.assertEqual(state.pending_recovery_global_ranks, set())

    def test_excluded_dead_dp_is_not_an_unresolved_expected_dp_fault(self):
        state = make_state()
        state.expected_dp_mask[1] = False
        state.observe_process_active_ranks([1], active=False)

        self.assertFalse(state.has_unresolved_expected_dp_fault())

    def test_pause_admission_uses_cluster_pause_and_operation(self):
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
        self.assertEqual(state.status_response()["engines"][1]["status"], "dead")


if __name__ == "__main__":
    unittest.main()
