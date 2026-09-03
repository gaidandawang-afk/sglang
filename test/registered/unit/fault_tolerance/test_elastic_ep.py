import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.elastic_ep import elastic_ep


def test_forced_rank_fault_rebalance_uses_mooncake_survivor_barriers():
    events = []
    state = Mock()
    state.is_active_equal_last.return_value = True
    state.snapshot_active_to_last.side_effect = lambda: events.append("snapshot")
    state.sync_active_to_cpu.side_effect = lambda: events.append("sync-active-cpu")

    group = SimpleNamespace(device_group="device", cpu_group="cpu")
    group.barrier = lambda: events.append("barrier")

    accepted = object()
    pg = SimpleNamespace(
        SyncAfterFailureStatus=SimpleNamespace(Rejected=object()),
        sync_after_failure=lambda backend: (
            events.append(f"sync-failure-{backend}")
            or SimpleNamespace(status=accepted)
        ),
    )

    def rebalance(*, force):
        events.append(f"rebalance-force-{force}")
        yield from ()

    eplb_manager = SimpleNamespace(rebalance=rebalance)

    with (
        patch.object(elastic_ep.ElasticEPStateManager, "instance", return_value=state),
        patch.object(elastic_ep, "get_world_group", return_value=group),
        patch.object(
            elastic_ep.torch.distributed,
            "get_backend",
            return_value="mooncake-cpu",
        ),
        patch.dict(sys.modules, {"mooncake": SimpleNamespace(pg=pg)}),
    ):
        assert elastic_ep.maybe_rebalance_after_rank_fault(
            eplb_manager=eplb_manager,
            force=True,
        )

    assert events == [
        "sync-failure-device",
        "sync-failure-cpu",
        "barrier",
        "snapshot",
        "sync-active-cpu",
        "rebalance-force-True",
        "barrier",
    ]
