from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class FTRankTopology:
    global_rank_count: int
    attention_tp_size: int


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
        return FTRankTopology(
            global_rank_count=tp_size,
            attention_tp_size=max(1, tp_size // dp_size // attn_cp_size),
        )
    return FTRankTopology(global_rank_count=dp_size, attention_tp_size=1)


def dp_rank_for_global_rank(
    global_rank: int, *, dp_size: int, attention_tp_size: int
) -> int | None:
    if global_rank < 0 or attention_tp_size <= 0:
        return None
    dp_rank = global_rank // attention_tp_size
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
