import asyncio
from types import SimpleNamespace

import pytest

from sglang.srt.managers import ascend_ft
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    AscendFTAckOutput,
    AscendFTCommandReq,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Tokenizer:
    def __init__(self):
        self.is_pause = False
        self.is_pause_cond = asyncio.Condition()
        self.events = []
        self.fail_recovery = False

    def auto_create_handle_loop(self):
        pass

    async def _async_dispatch_to_scheduler(self, message):
        assert self.is_pause
        if isinstance(message, ActiveRanksOutput):
            self.events.append(("route", message.status))
        else:
            assert isinstance(message, AscendFTCommandReq)
            self.events.append((message.op, message.target_ranks))
            for rank in message.target_ranks:
                self.manager.handle_ack(
                    AscendFTAckOutput(
                        rid=message.rid,
                        op=message.op,
                        rank=rank,
                        error=(
                            "migration failed"
                            if self.fail_recovery and message.op == "recover"
                            else None
                        ),
                    )
                )

    async def ascend_ft_replay(self, removed_dp_ranks):
        assert self.is_pause
        self.events.append(("replay", removed_dp_ranks))


@pytest.fixture
def tokenizer(monkeypatch):
    monkeypatch.setattr(
        ascend_ft, "get_parallel", lambda: SimpleNamespace(num_dp_ranks=4)
    )
    monkeypatch.setattr(
        ascend_ft, "get_device", lambda: SimpleNamespace(watchdog_timeout=1)
    )
    tokenizer = Tokenizer()
    tokenizer.manager = ascend_ft.AscendFTManager(tokenizer)
    tokenizer.manager.min_survivors = 2
    return tokenizer


def test_pause_then_scale_down(tokenizer):
    async def run():
        manager = tokenizer.manager
        assert manager.apply("pause", [])["phase"] == "pausing"
        with pytest.raises(RuntimeError, match="already running"):
            manager.apply("pause", [])
        await manager.task
        assert manager.status()["phase"] == "paused"
        assert tokenizer.is_pause

        manager.apply("scale_down", [0])
        await manager.task
        assert tokenizer.events == [
            ("pause", [0, 1, 2, 3]),
            ("pause", [1, 2, 3]),
            ("recover", [1, 2, 3]),
            ("route", [False, True, True, True]),
            ("replay", [0]),
            ("resume", [1, 2, 3]),
        ]
        assert manager.status()["active_dp_ranks"] == [1, 2, 3]
        assert manager.status()["phase"] == "running"
        assert not tokenizer.is_pause

    asyncio.run(run())


def test_recovery_error_keeps_requests_paused(tokenizer):
    async def run():
        manager = tokenizer.manager
        with pytest.raises(ValueError, match="enough expert slots"):
            manager.apply("scale_down", [0, 1, 2])
        manager.handle_ack(AscendFTAckOutput(op="fault", rank=1, error="MC2 timeout"))
        assert manager.status()["phase"] == "faulted"

        tokenizer.fail_recovery = True
        manager.apply("scale_down", [1])
        await manager.task
        assert manager.status()["phase"] == "failed"
        assert manager.status()["error"] == "DP rank 0: migration failed"
        assert tokenizer.is_pause
        assert tokenizer.events == [
            ("pause", [0, 2, 3]),
            ("recover", [0, 2, 3]),
        ]
        with pytest.raises(RuntimeError, match="Restart"):
            manager.apply("retry", [])

    asyncio.run(run())
