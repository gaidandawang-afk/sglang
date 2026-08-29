from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.distributed


@dataclass(frozen=True)
class EPLBProcessGroupContext:
    """Communication planes used by EPLB after membership changes."""

    control_group: Optional[torch.distributed.ProcessGroup] = None
    device_group: Optional[torch.distributed.ProcessGroup] = None
    active_original_ranks: Optional[tuple[int, ...]] = None
    control_group_uses_cpu: bool = False

    def is_active(self, original_rank: int) -> bool:
        return (
            self.active_original_ranks is None
            or original_rank in self.active_original_ranks
        )

    def to_control_group_rank(self, original_rank: int) -> int:
        if self.active_original_ranks is None:
            return original_rank
        return self.active_original_ranks.index(original_rank)

    def is_control_group_root(self, original_rank: int) -> bool:
        if self.active_original_ranks is None:
            return original_rank == 0
        return original_rank == self.active_original_ranks[0]


_context = EPLBProcessGroupContext()


def get_eplb_process_group_context() -> EPLBProcessGroupContext:
    return _context


def set_eplb_process_group_context(context: EPLBProcessGroupContext) -> None:
    global _context
    _context = context
