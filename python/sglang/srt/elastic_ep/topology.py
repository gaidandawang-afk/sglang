from __future__ import annotations

from typing import Sequence


def collapse_physical_rank_status(
    status: Sequence[bool], attn_replica_size: int
) -> list[bool]:
    """Collapse physical-rank health with an all-members replica rule."""
    if not status or attn_replica_size <= 0 or len(status) % attn_replica_size != 0:
        raise ValueError(
            f"Physical rank status size {len(status)} must be divisible by "
            f"attention replica size {attn_replica_size}."
        )
    return [
        all(bool(value) for value in status[start : start + attn_replica_size])
        for start in range(0, len(status), attn_replica_size)
    ]
