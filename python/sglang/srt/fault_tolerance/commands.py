import asyncio
import dataclasses
from typing import Dict


@dataclasses.dataclass
class PendingFTCommand:
    target_ranks: set[int]
    future: asyncio.Future
    acked: set[int] = dataclasses.field(default_factory=set)
    failed: Dict[int, str] = dataclasses.field(default_factory=dict)

    def finish_if_ready(self):
        if self.future.done():
            return
        if self.acked.union(self.failed) >= self.target_ranks:
            self.future.set_result(None)
