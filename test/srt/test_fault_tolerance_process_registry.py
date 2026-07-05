import importlib.util
import sys
from pathlib import Path
from types import ModuleType


_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "srt"
    / "fault_tolerance"
    / "process_registry.py"
)
_TOPOLOGY_PATH = _REGISTRY_PATH.with_name("topology.py")

_MISSING = object()


def _load_process_registry_module():
    topology_spec = importlib.util.spec_from_file_location(
        "ft_topology", _TOPOLOGY_PATH
    )
    topology_module = importlib.util.module_from_spec(topology_spec)
    sys.modules[topology_spec.name] = topology_module
    topology_spec.loader.exec_module(topology_module)

    package_names = [
        "sglang",
        "sglang.srt",
        "sglang.srt.fault_tolerance",
        "sglang.srt.fault_tolerance.topology",
    ]
    previous_modules = {
        name: sys.modules.get(name, _MISSING) for name in package_names
    }
    try:
        sys.modules["sglang"] = ModuleType("sglang")
        sys.modules["sglang.srt"] = ModuleType("sglang.srt")
        sys.modules["sglang.srt.fault_tolerance"] = ModuleType(
            "sglang.srt.fault_tolerance"
        )
        sys.modules["sglang.srt.fault_tolerance.topology"] = topology_module

        spec = importlib.util.spec_from_file_location(
            "ft_process_registry", _REGISTRY_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


_MODULE = _load_process_registry_module()

SchedulerProcessRegistry = _MODULE.SchedulerProcessRegistry


class FakeProcess:
    def __init__(self, pid, alive):
        self.pid = pid
        self._alive = alive

    def is_alive(self):
        return self._alive


def test_tp1_rejoin_pid_overrides_dead_original_process(monkeypatch):
    registry = SchedulerProcessRegistry(dp_size=2, tp_size=1)
    original = FakeProcess(pid=100, alive=False)
    registry.append_process(original)

    old_pid = registry.register_rejoin(rank=0, pid=200)
    monkeypatch.setattr(registry, "_is_pid_alive", lambda pid: pid == 200)

    assert old_pid == 100
    assert registry.dp_rank_for_scheduler_index(0) == 0
    assert registry.is_rank_alive(0, original)
    assert registry.should_ignore_process_exit(0, original)


def test_tp1_rejoined_rank_is_not_treated_as_dead_by_broadcast_guard(monkeypatch):
    registry = SchedulerProcessRegistry(dp_size=2, tp_size=1)
    processes = [
        FakeProcess(pid=100, alive=True),
        FakeProcess(pid=101, alive=False),
    ]
    for process in processes:
        registry.append_process(process)
    registry.register_rejoin(rank=1, pid=201)
    monkeypatch.setattr(registry, "_is_pid_alive", lambda pid: pid == 201)

    assert not registry.has_dead_scheduler_rank(processes)


def test_tpgt1_dp_attention_maps_global_rank_to_dp_rank():
    registry = SchedulerProcessRegistry(
        dp_size=2,
        tp_size=4,
        attn_cp_size=1,
        enable_dp_attention=True,
    )

    assert registry.dp_rank_for_scheduler_index(0) == 0
    assert registry.dp_rank_for_scheduler_index(1) == 0
    assert registry.dp_rank_for_scheduler_index(2) == 1
    assert registry.dp_rank_for_scheduler_index(3) == 1


def test_dead_replacement_pid_marks_rank_inactive(monkeypatch):
    registry = SchedulerProcessRegistry(dp_size=1, tp_size=1)
    original = FakeProcess(pid=100, alive=False)
    registry.append_process(original)
    registry.register_rejoin(rank=0, pid=200)
    monkeypatch.setattr(registry, "_is_pid_alive", lambda pid: False)

    assert not registry.is_rank_alive(0, original)
    assert not registry.should_ignore_process_exit(0, original)


def test_dp_attention_local_control_ignores_dp_route_status_if_leader_alive():
    registry = SchedulerProcessRegistry(
        dp_size=2,
        tp_size=4,
        attn_cp_size=1,
        enable_dp_attention=True,
    )
    processes = [
        FakeProcess(pid=100, alive=True),
        FakeProcess(pid=101, alive=True),
        FakeProcess(pid=102, alive=True),
        FakeProcess(pid=103, alive=False),
    ]

    assert (
        registry.ft_control_rank_for_target(
            3, control_message_step=1, worker_count=2
        )
        == 1
    )
    assert registry.is_ft_control_rank_reachable(
        1,
        control_message_step=1,
        worker_count=2,
        status=[True, False],
        processes=processes,
    )


def test_dp_attention_local_control_rejects_when_group_leader_is_dead():
    registry = SchedulerProcessRegistry(
        dp_size=2,
        tp_size=4,
        attn_cp_size=1,
        enable_dp_attention=True,
    )
    processes = [
        FakeProcess(pid=100, alive=True),
        FakeProcess(pid=101, alive=True),
        FakeProcess(pid=102, alive=False),
        FakeProcess(pid=103, alive=True),
    ]

    assert not registry.is_ft_control_rank_reachable(
        1,
        control_message_step=1,
        worker_count=2,
        status=[True, True],
        processes=processes,
    )


def test_dp_attention_rejoined_group_leader_keeps_ft_control_reachable(
    monkeypatch,
):
    registry = SchedulerProcessRegistry(
        dp_size=2,
        tp_size=4,
        attn_cp_size=1,
        enable_dp_attention=True,
    )
    processes = [
        FakeProcess(pid=100, alive=True),
        FakeProcess(pid=101, alive=True),
        FakeProcess(pid=102, alive=False),
        FakeProcess(pid=103, alive=True),
    ]
    for process in processes:
        registry.append_process(process)
    registry.register_rejoin(rank=2, pid=202)
    monkeypatch.setattr(registry, "_is_pid_alive", lambda pid: pid == 202)

    assert registry.is_ft_control_rank_reachable(
        1,
        control_message_step=1,
        worker_count=2,
        status=[True, False],
        processes=processes,
    )


def test_dp_attention_full_tp_broadcast_detects_dead_member():
    registry = SchedulerProcessRegistry(
        dp_size=2,
        tp_size=4,
        attn_cp_size=1,
        enable_dp_attention=True,
    )
    processes = [
        FakeProcess(pid=100, alive=True),
        FakeProcess(pid=101, alive=True),
        FakeProcess(pid=102, alive=False),
        FakeProcess(pid=103, alive=True),
    ]

    assert registry.ft_control_rank_for_target(
        3, control_message_step=4, worker_count=2
    ) == 0
    assert registry.has_dead_scheduler_rank(processes)
