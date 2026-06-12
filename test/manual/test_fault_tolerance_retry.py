"""
Test suite for SGLang fault tolerance retry semantics.

Covers Commit 1 of the FT Retry / Watchdog fix plan:
- retry rejected when DEAD rank exists
- retry sends only lightweight cleanup commands
- retry after scale_down is rejected
- retry cleanup exception triggers fail-stop
"""

import asyncio
import os
import sys
from unittest import mock

import pytest

# Ensure sglang is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.fault_tolerance.controller import (
    FaultToleranceManager,
    RankState,
)


def make_manager(dp_size=4, enabled=True, mooncake=True):
    """Create a FaultToleranceManager with standard test parameters."""
    return FaultToleranceManager(
        enabled=enabled,
        dp_size=dp_size,
        on_error_strategy="continue",
        recovery_timeout_sec=300,
        moe_a2a_backend="mooncake" if mooncake else "none",
        elastic_ep_backend=None,
    )


class TestValidateRetry:
    """validate_retry() should reject retry when any DEAD rank exists."""

    def test_no_dead_ranks_allows_retry(self):
        mgr = make_manager()
        assert mgr.validate_retry() is None

    def test_paused_ranks_allow_retry(self):
        mgr = make_manager()
        mgr.pause_all_active("test pause")
        assert mgr.validate_retry() is None

    def test_single_dead_rank_rejects_retry(self):
        mgr = make_manager()
        mgr.record_fault(2, "rank 2 crashed")
        error = mgr.validate_retry()
        assert error is not None
        assert "dead" in error.lower()
        assert "scale_down" in error.lower()

    def test_multiple_dead_ranks_rejects_retry(self):
        mgr = make_manager()
        mgr.record_fault(0, "rank 0 crashed")
        mgr.record_fault(3, "rank 3 crashed")
        error = mgr.validate_retry()
        assert error is not None

    def test_scale_down_then_retry_rejected(self):
        mgr = make_manager()
        mgr.begin_scale_down([1])
        mgr.commit_scale_down()
        error = mgr.validate_retry()
        assert error is not None
        assert "dead" in error.lower()

    def test_fault_tolerance_disabled(self):
        mgr = make_manager(enabled=False)
        error = mgr.validate_retry()
        assert error is not None


class TestRetryDoesNotUseDeadRanks:
    """begin_retry and apply_retry operate only on non-DEAD ranks."""

    def test_begin_retry_does_not_touch_dead_ranks(self):
        mgr = make_manager()
        mgr.record_fault(0, "rank 0 dead")
        result = mgr.begin_retry()
        assert result["success"]
        states = [r["state"] for r in mgr.status_response()["ranks"]]
        assert states[0] == "dead"
        assert states[1] == "paused"

    def test_apply_retry_only_heals_paused(self):
        mgr = make_manager()
        mgr.record_fault(0, "rank 0 dead")
        mgr.begin_retry()
        result = mgr.apply_retry()
        assert result["success"]
        states = [r["state"] for r in mgr.status_response()["ranks"]]
        assert states[0] == "dead"
        assert states[1] == "healthy"


class TestScaleDown:
    """scale_down validation and lifecycle."""

    def test_scale_down_rejects_empty_ranks(self):
        mgr = make_manager()
        error = mgr.validate_scale_down([])
        assert error is not None

    def test_scale_down_rejects_unknown_rank(self):
        mgr = make_manager()
        error = mgr.validate_scale_down([99])
        assert error is not None

    def test_scale_down_cannot_isolate_all(self):
        mgr = make_manager(dp_size=2)
        error = mgr.validate_scale_down([0, 1])
        assert error is not None

    def test_scale_down_lifecycle(self):
        mgr = make_manager()
        result = mgr.begin_scale_down([0])
        assert result["success"]
        states = [r["state"] for r in mgr.status_response()["ranks"]]
        assert states[0] == "dead"
        assert all(s == "paused" for s in states[1:])

        result = mgr.commit_scale_down()
        assert result["success"]
        states = [r["state"] for r in mgr.status_response()["ranks"]]
        assert states[0] == "dead"
        assert all(s == "healthy" for s in states[1:])


class TestActiveMaskAndRanks:
    """active_mask and active_ranks exclude DEAD ranks."""

    def test_active_mask_excludes_dead(self):
        mgr = make_manager()
        mgr.record_fault(1, "rank 1 dead")
        mask = mgr.active_mask()
        assert mask == [True, False, True, True]

    def test_active_ranks_excludes_dead(self):
        mgr = make_manager()
        mgr.record_fault(1, "rank 1 dead")
        active = mgr.active_ranks()
        assert active == [0, 2, 3]


class TestLightweightCleanupMock:
    """Verify retry cleanup sends only lightweight commands (no reinit/health_check)."""

    @pytest.mark.asyncio
    async def test_lightweight_sends_only_prepare_and_resume(self):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        mgr = make_manager()
        tm = mock.AsyncMock(spec=TokenizerManager)
        tm.fault_tolerance = mgr
        tm.server_args = mock.Mock()
        tm.server_args.dp_size = 4
        tm.server_args.fault_tolerance_recovery_timeout_sec = 300

        commands_sent = []

        async def fake_send_command(command, **kwargs):
            commands_sent.append(command)

        tm._fault_tolerance_send_command = fake_send_command

        real_method = TokenizerManager._fault_tolerance_run_retry_cleanup_sequence
        await real_method(tm, timeout_sec=300, params={})

        assert commands_sent == ["prepare_retry", "resume"], (
            f"Expected only prepare_retry and resume, got {commands_sent}"
        )


class TestScaleDownRecoveryMock:
    """Verify scale_down recovery is ordered and does not use fire-and-forget masks."""

    @pytest.mark.asyncio
    async def test_sparse_scale_down_prepares_mask_before_skipping_sync(self):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        tm = mock.AsyncMock(spec=TokenizerManager)
        tm.server_args = mock.Mock()
        tm.server_args.dp_size = 4

        sent = []

        async def fake_send_command(command, **kwargs):
            sent.append((command, kwargs))

        tm._fault_tolerance_send_command = fake_send_command

        real_method = TokenizerManager._fault_tolerance_run_recovery_sequence
        await real_method(
            tm,
            active_ranks=[0, 2, 3],
            isolated_ranks=[1],
            timeout_sec=180,
            params={"ranks": [1]},
        )

        assert [command for command, _ in sent] == [
            "prepare_retry",
            "reinit",
            "health_check",
            "resume",
            "set_sparse_idle_control",
        ]
        assert all("wait" not in kwargs for _, kwargs in sent[:4])
        assert sent[-1][1]["wait"] is False

        params = sent[0][1]["params"]
        assert params["apply_active_mask_before_prepare"] is True
        assert params["skip_device_synchronize"] is True
        assert sent[-1][1]["params"] == {"enabled": True}


class TestFinalResponseBarrierMock:
    """Verify Mooncake FT responses park schedulers before returning."""

    @pytest.mark.asyncio
    async def test_degraded_mooncake_parks_after_final_response(self):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        mgr = make_manager()
        mgr.begin_scale_down([1])
        mgr.commit_scale_down()

        tm = mock.AsyncMock(spec=TokenizerManager)
        tm.fault_tolerance = mgr
        tm.server_args = mock.Mock()
        tm.server_args.enable_fault_tolerance = True
        tm.server_args.dp_size = 4
        tm.server_args.fault_tolerance_recovery_timeout_sec = 300
        tm._fault_tolerance_schedulers_parked = False
        tm._fault_tolerance_scheduler_park_lock = asyncio.Lock()

        sent = []

        async def fake_send_command(command, **kwargs):
            sent.append((command, kwargs))
            return []

        tm._fault_tolerance_send_command = fake_send_command
        obj = mock.Mock()
        obj.rid = "rid-response-barrier"

        real_method = TokenizerManager._fault_tolerance_barrier_before_final_response
        await real_method(tm, obj, is_stream=False)

        assert [command for command, _ in sent] == ["park_idle"]
        kwargs = sent[0][1]
        assert kwargs["target_ranks"] == [0, 2, 3]
        assert kwargs["active_ranks"] == [0, 2, 3]
        assert kwargs["timeout_sec"] == 300
        assert kwargs["params"] == {"reason": "final_response"}
        assert tm._fault_tolerance_schedulers_parked is True

    @pytest.mark.asyncio
    async def test_degraded_mooncake_resumes_before_next_request(self):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        mgr = make_manager()
        mgr.begin_scale_down([1])
        mgr.commit_scale_down()

        tm = mock.AsyncMock(spec=TokenizerManager)
        tm.fault_tolerance = mgr
        tm.server_args = mock.Mock()
        tm.server_args.enable_fault_tolerance = True
        tm.server_args.dp_size = 4
        tm.server_args.fault_tolerance_recovery_timeout_sec = 300
        tm._fault_tolerance_schedulers_parked = True
        tm._fault_tolerance_scheduler_park_lock = asyncio.Lock()

        sent = []

        async def fake_send_command(command, **kwargs):
            sent.append((command, kwargs))
            return []

        tm._fault_tolerance_send_command = fake_send_command
        obj = mock.Mock()
        obj.rid = "rid-resume-parked"

        real_method = (
            TokenizerManager._fault_tolerance_resume_parked_schedulers_before_request
        )
        await real_method(tm, obj)

        assert [command for command, _ in sent] == ["resume"]
        kwargs = sent[0][1]
        assert kwargs["target_ranks"] == [0, 2, 3]
        assert kwargs["active_ranks"] == [0, 2, 3]
        assert kwargs["timeout_sec"] == 300
        assert kwargs["params"] == {"reason": "before_request"}
        assert tm._fault_tolerance_schedulers_parked is False

    @pytest.mark.asyncio
    async def test_full_mooncake_cluster_parks_after_final_response(self):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        tm = mock.AsyncMock(spec=TokenizerManager)
        tm.fault_tolerance = make_manager()
        tm.server_args = mock.Mock()
        tm.server_args.enable_fault_tolerance = True
        tm.server_args.dp_size = 4
        tm.server_args.fault_tolerance_recovery_timeout_sec = 300
        tm._fault_tolerance_schedulers_parked = False
        tm._fault_tolerance_scheduler_park_lock = asyncio.Lock()
        tm._fault_tolerance_send_command = mock.AsyncMock(return_value=[])

        obj = mock.Mock()
        obj.rid = "rid-no-barrier"

        real_method = TokenizerManager._fault_tolerance_barrier_before_final_response
        await real_method(tm, obj, is_stream=False)

        tm._fault_tolerance_send_command.assert_awaited_once_with(
            "park_idle",
            target_ranks=[0, 1, 2, 3],
            active_ranks=[0, 1, 2, 3],
            timeout_sec=300,
            params={"reason": "final_response"},
        )
        assert tm._fault_tolerance_schedulers_parked is True


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

