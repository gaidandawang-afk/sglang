from __future__ import annotations

from datetime import timedelta
from urllib.parse import urlsplit

import torch
import torch.distributed as dist

from sglang.srt.distributed.utils import all_gather_single
from sglang.srt.utils.process_group import init_custom_process_group

_STORE_TIMEOUT = timedelta(minutes=3)


class AscendFTPause(RuntimeError):
    """An MLP sync round received a pause vote."""


class AscendFTCollectiveError(RuntimeError):
    """The Ascend FT metadata exchange could not complete."""


def _connect_store(endpoint: str, *, is_master: bool) -> dist.TCPStore:
    url = urlsplit(endpoint if "://" in endpoint else f"tcp://{endpoint}")
    return dist.TCPStore(
        host_name=url.hostname,
        port=url.port,
        is_master=is_master,
        wait_for_workers=False,
        timeout=_STORE_TIMEOUT,
    )


def create_ascend_ft_store(endpoint: str) -> dist.TCPStore:
    return _connect_store(endpoint, is_master=True)


class AscendFTCommunication:
    """Exchange MLP metadata in compact Gloo order and restore physical DP slots."""

    def __init__(self, endpoint: str, *, rank: int, world_size: int, timeout: float):
        self.store = _connect_store(endpoint, is_master=False)
        self.rank = rank
        self.timeout = timedelta(seconds=timeout)
        self.group = None
        self.pause_requested = False
        startup = dist.PrefixStore("startup", self.store)
        startup.set(str(rank), "ready")
        startup.wait([str(peer) for peer in range(world_size)])
        self._install_group(tuple(range(world_size)), "startup")

    def rebuild(
        self, active_ranks: tuple[int, ...], request_id: str
    ) -> dist.ProcessGroup:
        return self._install_group(active_ranks, request_id)

    def _install_group(
        self, active_ranks: tuple[int, ...], request_id: str
    ) -> dist.ProcessGroup:
        # Each operation gets fresh rendezvous keys even if membership is unchanged.
        name = f"ascend_ft_{request_id}"
        group = init_custom_process_group(
            backend="gloo",
            store=dist.PrefixStore(name, self.store),
            world_size=len(active_ranks),
            rank=active_ranks.index(self.rank),
            group_name=name,
            timeout=self.timeout,
        )
        dist.barrier(group=group)
        if self.group is not None:
            dist.destroy_process_group(self.group)
        self.group = group
        self.active_ranks = active_ranks
        self.physical_slots = torch.tensor(
            active_ranks, dtype=torch.int64, device="cpu"
        )
        return group

    def gather(self, local_row: torch.Tensor, physical_rows: torch.Tensor) -> None:
        row = torch.cat((local_row, local_row.new_tensor([self.pause_requested])))
        rows = row.new_empty((len(self.active_ranks), row.numel()))
        try:
            work = all_gather_single(
                rows.flatten(), row, group=self.group, async_op=True
            )
            completed = work.wait(timeout=self.timeout)
        except RuntimeError as exc:
            raise AscendFTCollectiveError("Ascend FT MLP sync failed") from exc
        if completed is False:
            raise AscendFTCollectiveError("Ascend FT MLP sync timed out")
        if rows[:, -1].any():
            raise AscendFTPause()
        physical_rows.index_copy_(0, self.physical_slots, rows[:, :-1])
