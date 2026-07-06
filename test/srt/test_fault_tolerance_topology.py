import importlib.util
import sys
from pathlib import Path


_TOPOLOGY_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "srt"
    / "fault_tolerance"
    / "topology.py"
)
_TOKENIZER_MANAGER_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "srt"
    / "managers"
    / "tokenizer_manager.py"
)
_SPEC = importlib.util.spec_from_file_location("ft_topology", _TOPOLOGY_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

resolve_ft_rank_topology = _MODULE.resolve_ft_rank_topology
dp_rank_for_global_rank = _MODULE.dp_rank_for_global_rank
scheduler_ft_rank = _MODULE.scheduler_ft_rank


def test_dp_attention_topology_uses_global_rank_space():
    topology = resolve_ft_rank_topology(
        dp_size=2,
        tp_size=4,
        attn_cp_size=1,
        enable_dp_attention=True,
    )

    assert topology.global_rank_count == 4
    assert topology.ranks_per_dp == 2
    assert [
        dp_rank_for_global_rank(
            rank,
            dp_size=2,
            ranks_per_dp=topology.ranks_per_dp,
        )
        for rank in range(topology.global_rank_count)
    ] == [0, 0, 1, 1]


def test_non_dp_attention_topology_keeps_dp_rank_space():
    topology = resolve_ft_rank_topology(
        dp_size=2,
        tp_size=4,
        attn_cp_size=1,
        enable_dp_attention=False,
    )

    assert topology.global_rank_count == 2
    assert topology.ranks_per_dp == 1


def test_dp_attention_cp_topology_maps_all_cp_members_to_the_same_dp():
    topology = resolve_ft_rank_topology(
        dp_size=2,
        tp_size=8,
        attn_cp_size=2,
        enable_dp_attention=True,
    )

    assert topology.global_rank_count == 8
    assert topology.ranks_per_dp == 4
    assert [
        dp_rank_for_global_rank(
            rank,
            dp_size=2,
            ranks_per_dp=topology.ranks_per_dp,
        )
        for rank in range(topology.global_rank_count)
    ] == [0, 0, 0, 0, 1, 1, 1, 1]


def test_scheduler_ft_rank_uses_global_rank_under_dp_attention():
    assert scheduler_ft_rank(
        enable_dp_attention=True,
        tp_rank=3,
        dp_rank=1,
    ) == 3


def test_scheduler_ft_rank_uses_dp_rank_without_dp_attention():
    assert scheduler_ft_rank(
        enable_dp_attention=False,
        tp_rank=3,
        dp_rank=1,
    ) == 1


def test_scheduler_ft_rank_falls_back_to_tp_rank_without_dp_rank():
    assert scheduler_ft_rank(
        enable_dp_attention=False,
        tp_rank=0,
        dp_rank=None,
    ) == 0


def test_tokenizer_manager_ft_mask_uses_resolved_topology_size():
    source = _TOKENIZER_MANAGER_PATH.read_text(encoding="utf-8")

    assert "ft_global_rank_count" not in source
    assert "ft_rank_topology.global_rank_count" in source
