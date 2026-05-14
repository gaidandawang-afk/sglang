import os
import unittest

if os.name == "nt":
    raise unittest.SkipTest("SGLang runtime imports require POSIX modules.")

from sglang.srt.fault_tolerance.state import (
    ComponentState,
    FaultEvent,
    SentinelStatus,
)


class TestFaultToleranceState(unittest.TestCase):
    def test_fault_event_from_exception(self):
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            event = FaultEvent.from_exception(
                exc,
                origin="scheduler",
                scheduler_id=3,
                rank=1,
            )

        self.assertEqual(event.origin, "scheduler")
        self.assertEqual(event.scheduler_id, 3)
        self.assertEqual(event.exception_type, "RuntimeError")
        self.assertIn("boom", event.message)
        self.assertIn("RuntimeError", event.traceback)

    def test_sentinel_status_to_dict_uses_enum_value(self):
        status = SentinelStatus(
            scheduler_id=0,
            pid=123,
            state=ComponentState.HEALTHY,
            last_heartbeat_ts=1.0,
        )

        self.assertEqual(status.to_dict()["state"], "HEALTHY")


if __name__ == "__main__":
    unittest.main()
