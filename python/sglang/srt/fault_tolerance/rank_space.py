FT_RANK_SPACE_DP_ROUTE = "dp_route"
FT_RANK_SPACE_EP = "ep"


def is_ep_rank_space(rank_space: str) -> bool:
    return rank_space == FT_RANK_SPACE_EP


def is_dp_route_rank_space(rank_space: str) -> bool:
    return rank_space == FT_RANK_SPACE_DP_ROUTE


def active_ranks_broadcast_rank_space(
    *, mask_size: int, scheduler_rank_count: int
) -> str:
    """Infer rank-space for legacy ActiveRanksOutput scheduler broadcasts.

    DPC routing state is DP-sized. Mooncake's backend mask is physical EP/global
    rank sized.  In the historical TP=1 path these spaces have the same size,
    so the legacy broadcast can still be treated as EP.  Once TP>1 makes the
    scheduler/global rank count larger than the DP route mask, the same payload
    must not be applied to Mooncake.
    """
    if mask_size == scheduler_rank_count:
        return FT_RANK_SPACE_EP
    return FT_RANK_SPACE_DP_ROUTE
