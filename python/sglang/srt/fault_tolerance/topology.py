from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class FTRankTopology:
    global_rank_count: int
    ranks_per_dp: int


def resolve_ft_rank_topology(
    *,
    dp_size: int,
    tp_size: int,
    attn_cp_size: int = 1,
    enable_dp_attention: bool = False,
) -> FTRankTopology:
    if dp_size <= 0:
        raise ValueError("dp_size must be positive")
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if attn_cp_size <= 0:
        raise ValueError("attn_cp_size must be positive")

    if enable_dp_attention:
        if tp_size % dp_size != 0:
            raise ValueError("tp_size must be divisible by dp_size")
        ranks_per_dp = tp_size // dp_size
        if ranks_per_dp % attn_cp_size != 0:
            raise ValueError("ranks_per_dp must be divisible by attn_cp_size")
        return FTRankTopology(
            global_rank_count=tp_size,
            ranks_per_dp=ranks_per_dp,
        )
    return FTRankTopology(global_rank_count=dp_size, ranks_per_dp=1)


def dp_rank_for_global_rank(
    global_rank: int, *, dp_size: int, ranks_per_dp: int
) -> int | None:
    if global_rank < 0 or ranks_per_dp <= 0:
        return None
    dp_rank = global_rank // ranks_per_dp
    if 0 <= dp_rank < dp_size:
        return dp_rank
    return None


def scheduler_ft_rank(
    *,
    enable_dp_attention: bool,
    tp_rank: int,
    dp_rank: int | None,
) -> int:
    """Return the rank identity used by FT commands, ACKs, faults, and rejoin.

    DP-attention has multiple physical scheduler ranks per public DP route, so
    FT control uses the real TP/global rank.  Non-DP-attention keeps the
    historical DP-rank identity when present, with TP rank as a single-rank
    fallback.
    """
    if enable_dp_attention:
        return tp_rank
    return dp_rank if dp_rank is not None else tp_rank
