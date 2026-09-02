import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/eplb/process_group_context.py"
)


def load_context_module():
    dist = ModuleType("torch.distributed")
    dist.ProcessGroup = object
    torch = ModuleType("torch")
    torch.distributed = dist
    spec = importlib.util.spec_from_file_location(
        "eplb_process_group_context_under_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}):
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
    return module


def test_default_context_preserves_original_rank_namespace():
    context = load_context_module().EPLBProcessGroupContext()

    assert context.is_active(3)
    assert context.is_control_group_root(0)


def test_survivor_context_tracks_active_ranks_after_rank_zero_failure():
    context = load_context_module().EPLBProcessGroupContext(
        control_group=object(),
        device_group=object(),
        active_original_ranks=(1, 2, 3),
        control_group_uses_cpu=True,
    )

    assert not context.is_active(0)
    assert context.is_active(1)
    assert context.is_control_group_root(1)
    assert not context.is_control_group_root(0)
    assert context.control_group_uses_cpu
