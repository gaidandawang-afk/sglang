import multiprocessing
import time

import pytest
import torch
import torch.distributed as dist

from sglang.srt.managers.ascend_ft_communication import (
    AscendFTCollectiveError,
    AscendFTCommunication,
    AscendFTPause,
    create_ascend_ft_store,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _worker(port, rank, retire):
    store = dist.TCPStore("127.0.0.1", port, is_master=False, wait_for_workers=False)
    dist.init_process_group(
        backend="gloo", store=dist.PrefixStore("world", store), rank=rank, world_size=3
    )
    comm = AscendFTCommunication(
        f"127.0.0.1:{port}", rank=rank, world_size=3, timeout=3
    )
    local = torch.tensor([rank])
    rows = torch.full((3, 1), -1)
    comm.gather(local, rows)
    assert rows[:, 0].tolist() == [0, 1, 2]

    comm.pause_requested = rank == 0
    with pytest.raises(AscendFTPause):
        comm.gather(local, rows)
    comm.pause_requested = False
    if rank == 0:
        assert retire.wait(timeout=30)
        return

    # Rank zero is alive but will never enter another collective.
    # Both the outstanding gather and old-group cleanup must be bounded.
    start = time.monotonic()
    with pytest.raises(AscendFTCollectiveError):
        comm.gather(local, rows)
    group = comm.rebuild((1, 2), "after-rank-zero")
    assert time.monotonic() - start < 15

    value = local.clone()
    dist.all_reduce(value, group=group)
    assert value.item() == 3
    rows.fill_(-1)
    comm.gather(local, rows)
    assert rows[:, 0].tolist() == [-1, 1, 2]
    dist.barrier(group=group)
    retire.set()


def test_rank_zero_pause_and_retirement():
    store = create_ascend_ft_store("127.0.0.1:0")
    context = multiprocessing.get_context("spawn")
    retire = context.Event()
    workers = [
        context.Process(target=_worker, args=(store.port, rank, retire))
        for rank in range(3)
    ]
    for worker in workers:
        worker.start()
    try:
        for worker in workers:
            worker.join(timeout=45)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
