import importlib.util
import sys
from pathlib import Path


_RANK_SPACE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "srt"
    / "fault_tolerance"
    / "rank_space.py"
)
_SPEC = importlib.util.spec_from_file_location("ft_rank_space", _RANK_SPACE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

FT_RANK_SPACE_DP_ROUTE = _MODULE.FT_RANK_SPACE_DP_ROUTE
FT_RANK_SPACE_EP = _MODULE.FT_RANK_SPACE_EP
is_dp_route_rank_space = _MODULE.is_dp_route_rank_space
is_ep_rank_space = _MODULE.is_ep_rank_space
active_ranks_broadcast_rank_space = _MODULE.active_ranks_broadcast_rank_space


def test_dp_route_rank_space_is_not_ep_backend_mask():
    assert is_dp_route_rank_space(FT_RANK_SPACE_DP_ROUTE)
    assert not is_ep_rank_space(FT_RANK_SPACE_DP_ROUTE)


def test_ep_rank_space_is_only_mooncake_backend_mask_space():
    assert is_ep_rank_space(FT_RANK_SPACE_EP)
    assert not is_dp_route_rank_space(FT_RANK_SPACE_EP)


def test_legacy_active_ranks_broadcast_preserves_tp1_ep_mask():
    assert (
        active_ranks_broadcast_rank_space(mask_size=2, scheduler_rank_count=2)
        == FT_RANK_SPACE_EP
    )


def test_tpgt1_active_ranks_broadcast_stays_dp_route_mask():
    assert (
        active_ranks_broadcast_rank_space(mask_size=2, scheduler_rank_count=4)
        == FT_RANK_SPACE_DP_ROUTE
    )
