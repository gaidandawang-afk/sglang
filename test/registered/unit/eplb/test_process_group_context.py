import unittest

from sglang.srt.eplb.process_group_context import EPLBProcessGroupContext
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestEPLBProcessGroupContext(CustomTestCase):
    def test_default_context_preserves_original_rank_namespace(self):
        context = EPLBProcessGroupContext()

        self.assertTrue(context.is_active(3))
        self.assertTrue(context.is_control_group_root(0))

    def test_survivor_context_tracks_active_original_ranks(self):
        context = EPLBProcessGroupContext(
            control_group=object(),
            device_group=object(),
            active_original_ranks=(1, 2, 3),
            control_group_uses_cpu=True,
        )

        self.assertFalse(context.is_active(0))
        self.assertTrue(context.is_active(1))
        self.assertTrue(context.is_control_group_root(1))
        self.assertFalse(context.is_control_group_root(0))
        self.assertTrue(context.control_group_uses_cpu)


if __name__ == "__main__":
    unittest.main()
