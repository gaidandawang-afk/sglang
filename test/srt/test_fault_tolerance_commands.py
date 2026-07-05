import asyncio
import importlib.util
import sys
from pathlib import Path


_COMMANDS_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "srt"
    / "fault_tolerance"
    / "commands.py"
)
_SPEC = importlib.util.spec_from_file_location("ft_commands", _COMMANDS_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

PendingFTCommand = _MODULE.PendingFTCommand


def test_pending_ft_command_waits_for_each_global_rank_ack():
    loop = asyncio.new_event_loop()
    try:
        pending = PendingFTCommand(
            target_ranks={2, 3},
            future=loop.create_future(),
        )

        pending.acked.add(2)
        pending.finish_if_ready()
        assert not pending.future.done()

        pending.acked.add(3)
        pending.finish_if_ready()
        assert pending.future.done()
    finally:
        loop.close()


def test_pending_ft_command_accepts_failure_per_global_rank():
    loop = asyncio.new_event_loop()
    try:
        pending = PendingFTCommand(
            target_ranks={0, 1, 3},
            future=loop.create_future(),
        )

        pending.acked.update({0, 1})
        pending.failed[3] = "ft_target_rank_unreachable"
        pending.finish_if_ready()

        assert pending.future.done()
    finally:
        loop.close()


def test_pending_ft_command_waits_until_every_target_has_ack_or_failure():
    loop = asyncio.new_event_loop()
    try:
        pending = PendingFTCommand(
            target_ranks={0, 1, 3},
            future=loop.create_future(),
        )

        pending.acked.add(0)
        pending.failed[3] = "ft_target_rank_unreachable"
        pending.finish_if_ready()
        assert not pending.future.done()

        pending.acked.add(1)
        pending.finish_if_ready()
        assert pending.future.done()
    finally:
        loop.close()
