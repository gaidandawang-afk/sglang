import asyncio
import os
import types
import unittest

if os.name == "nt":
    raise unittest.SkipTest("SGLang runtime imports require POSIX modules.")

from sglang.srt.fault_tolerance.manager import SentinelManager
from sglang.srt.fault_tolerance.state import FaultEvent, FaultToleranceState


class FakeCondition:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def notify_all(self):
        pass


class FakeTokenizerManager:
    def __init__(self):
        self.is_pause = False
        self.is_pause_cond = FakeCondition()
        self.pause_calls = []
        self.continue_calls = []

    async def pause_generation(self, obj):
        self.pause_calls.append(obj)
        self.is_pause = True

    async def continue_generation(self, obj):
        self.continue_calls.append(obj)
        self.is_pause = False


class TestSentinelManager(unittest.TestCase):
    def make_manager(self):
        server_args = types.SimpleNamespace(
            enable_fault_tolerance=True,
            fault_tolerance_recovery_timeout_sec=1,
            fault_tolerance_comm_abort_timeout_sec=1,
            fault_tolerance_sentinel_cmd_timeout_sec=1,
            fault_tolerance_default_pause_mode="retract",
            fault_tolerance_hard_pause_on_fault=True,
            fault_tolerance_reinit_dist_on_retry=True,
            shutdown_on_fault_tolerance_failure=False,
            tp_size=1,
            pp_size=1,
            dp_size=1,
            ep_size=1,
            attn_cp_size=1,
            moe_dp_size=1,
            nnodes=1,
            node_rank=0,
        )
        tokenizer_manager = FakeTokenizerManager()
        return SentinelManager(server_args, tokenizer_manager=tokenizer_manager)

    def test_report_fault_freezes_admission(self):
        manager = self.make_manager()
        event = FaultEvent.create(
            origin="scheduler",
            scheduler_id=0,
            rank=0,
            fault_type="exception",
            exception_type="RuntimeError",
            message="boom",
            traceback=None,
        )

        manager.report_fault_sync(event)

        self.assertEqual(manager.state, FaultToleranceState.ABORTING_COMM)
        self.assertFalse(manager.accepting_requests)
        self.assertTrue(manager.tokenizer_manager.is_pause)
        self.assertEqual(manager.get_status()["last_fault"]["message"], "boom")

    def test_pause_without_registered_sentinels_reports_failure(self):
        manager = self.make_manager()

        result = asyncio.run(
            manager.apply(
                "pause",
                timeout=1,
                params={"hard": False, "mode": "retract"},
            )
        )

        self.assertFalse(result["success"])
        self.assertEqual(manager.state, FaultToleranceState.WAITING_OPERATOR)
        self.assertTrue(manager.tokenizer_manager.is_pause)
        self.assertEqual(manager.tokenizer_manager.pause_calls[0].mode, "retract")
        self.assertEqual(result["results"][0]["scheduler_id"], 0)
        self.assertFalse(result["results"][0]["success"])
        self.assertIn("not registered", result["results"][0]["message"])

    def test_retry_without_registered_sentinels_reports_failure(self):
        manager = self.make_manager()

        result = asyncio.run(manager.apply("retry", timeout=1, params={}))

        self.assertFalse(result["success"])
        self.assertEqual(manager.state, FaultToleranceState.WAITING_OPERATOR)
        self.assertTrue(manager.tokenizer_manager.is_pause)
        self.assertEqual(result["results"][0]["scheduler_id"], 0)
        self.assertFalse(result["results"][0]["success"])


if __name__ == "__main__":
    unittest.main()
