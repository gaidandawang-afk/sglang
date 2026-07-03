from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ServerArgs
from sglang.srt.utils import is_cpu, is_cuda

logger = logging.getLogger(__name__)


@dataclass
class ElasticEPState:
    active_ranks: Optional[torch.Tensor]
    last_active_ranks: Optional[torch.Tensor]
    active_ranks_cpu: Optional[torch.Tensor]

    def is_active_equal_last(self) -> bool:
        return torch.equal(self.active_ranks, self.last_active_ranks)

    def sync_active_to_cpu(self):
        if self.active_ranks is not None:
            self.active_ranks_cpu = self.active_ranks.detach().cpu().clone()

    def snapshot_active_to_last(self):
        if self.active_ranks is not None:
            self.last_active_ranks = self.active_ranks.clone()

    def reset(self):
        if self.active_ranks is not None:
            old_active = self.active_ranks.detach().cpu().tolist()
            self.active_ranks = torch.ones(
                self.active_ranks.shape,
                dtype=self.active_ranks.dtype,
                device=self.active_ranks.device,
            )
            self.snapshot_active_to_last()
            self.sync_active_to_cpu()
            if envs.SGLANG_FT_PRECISION_DEBUG.get():
                logger.info(
                    "[FTPrecisionDebug][ElasticEP] reset old_active=%s new_active=%s",
                    old_active,
                    self.active_ranks_cpu.tolist(),
                )

    def maybe_inject_test_rank_fault(self) -> bool:
        ranks_raw = os.environ.get("SGLANG_TEST_ELASTIC_EP_FAULT_RANKS", "")
        if not ranks_raw or self.active_ranks is None:
            return False
        trigger_file = os.environ.get("SGLANG_TEST_ELASTIC_EP_FAULT_FILE", "")
        if trigger_file and not os.path.exists(trigger_file):
            return False
        if getattr(self, "_test_rank_fault_injected", False):
            return False

        ranks = []
        for item in ranks_raw.replace(",", " ").split():
            try:
                rank = int(item)
            except ValueError:
                continue
            if 0 <= rank < self.active_ranks.numel():
                ranks.append(rank)
        if not ranks:
            return False

        self.active_ranks[ranks] = 0
        self.sync_active_to_cpu()
        setattr(self, "_test_rank_fault_injected", True)
        done_file = os.environ.get("SGLANG_TEST_ELASTIC_EP_FAULT_DONE_FILE", "")
        if done_file:
            try:
                with open(done_file, "a", encoding="utf-8") as f:
                    f.write(f"pid={os.getpid()} ranks={ranks}\n")
            except OSError:
                pass
        logger.warning(
            "[ElasticEPTest] Injected inactive ranks %s; active_ranks=%s",
            ranks,
            self.active_ranks.detach().cpu().tolist(),
        )
        return True


class ElasticEPStateManager:
    _instance: Optional[ElasticEPState] = None

    @classmethod
    def instance(cls) -> ElasticEPState:
        return cls._instance

    @classmethod
    def init(cls, server_args: ServerArgs):
        if cls._instance is not None:
            return cls._instance

        if server_args.elastic_ep_backend is not None:
            cls._instance = cls._build_state(ep_size=None, device=None)
            if server_args.elastic_ep_rejoin:
                # Mask out peer ranks to perform cuda graph capture on its own
                cls._instance.active_ranks.zero_()
                cls._instance.active_ranks[torch.distributed.get_rank()] = 1
                cls._instance.snapshot_active_to_last()
                cls._instance.sync_active_to_cpu()

        return cls._instance

    @staticmethod
    def _select_device() -> torch.device:
        if is_cuda():
            return torch.device("cuda")
        elif is_cpu():
            return torch.device("cpu")
        else:
            raise NotImplementedError("Only CUDA and CPU support elastic ep now.")

    @classmethod
    def _build_state(
        cls, *, ep_size: Optional[int] = None, device: Optional[torch.device] = None
    ) -> ElasticEPState:
        active = cls.healthy_rank_state(ep_size=ep_size, device=device)
        return ElasticEPState(
            active_ranks=active,
            last_active_ranks=active.clone(),
            active_ranks_cpu=active.detach().cpu().clone(),
        )

    @classmethod
    def healthy_rank_state(
        cls, *, ep_size: Optional[int] = None, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        size = ep_size if ep_size is not None else torch.distributed.get_world_size()
        dev = device if device is not None else cls._select_device()

        return torch.ones(size, dtype=torch.int32, device=dev)


# ---------------------------------------------------------------------------
# Helpers for elastic EP recovery
# ---------------------------------------------------------------------------


_PEER_STATE_POLL_INTERVAL_SEC = 0.01


def _get_process_group_backend(process_group, device: str):
    return process_group._get_backend(torch.device(device))


def _iter_live_parallel_groups() -> Iterator[parallel_state.GroupCoordinator]:
    groups = []
    for group_ref in parallel_state._groups.values():
        group = group_ref()
        if group is not None:
            groups.append(group)
    for group in sorted(groups, key=lambda x: x.unique_name):
        yield group


def _map_global_to_group_local_ranks(
    group_ranks: List[int], global_ranks: List[int]
) -> List[int]:
    rank_to_local = {rank: idx for idx, rank in enumerate(group_ranks)}
    return [rank_to_local[rank] for rank in global_ranks if rank in rank_to_local]


def _wait_for_peer_state(mooncake_ep, backend, ranks: List[int]) -> None:
    # Relaunched ranks become recoverable asynchronously, so we poll until the
    # target backend reports all requested peers as ready.
    while not all(mooncake_ep.get_peer_state(backend, ranks)):
        time.sleep(_PEER_STATE_POLL_INTERVAL_SEC)


def _maybe_create_message_queue(group) -> None:
    if not group.use_message_queue_broadcaster or group.world_size <= 1:
        return

    from sglang.srt.distributed.device_communicators.shm_broadcast import MessageQueue

    group.mq_broadcaster = MessageQueue.create_from_process_group(
        group.cpu_group, 1 << 22, 6
    )


def _refresh_ep_members(*, allow_missing_buffer_in_fallback: bool = False) -> None:
    from sglang.srt.layers.moe.token_dispatcher.mooncake import EPBuffer

    if EPBuffer._buffer is None:
        if (
            allow_missing_buffer_in_fallback
            and os.environ.get("MOONCAKE_EP_FORCE_FALLBACK") == "1"
        ):
            logger.info(
                "Skip Mooncake EP member refresh before EPBuffer initialization "
                "in Mooncake forced fallback rejoin path."
            )
            return
        raise RuntimeError("Mooncake EPBuffer is not initialized")
    EPBuffer._buffer.update_ep_member()


def apply_active_rank_mask(mask: List[bool]) -> None:
    state = ElasticEPStateManager.instance()
    if state is None or state.active_ranks is None:
        raise RuntimeError("elastic ep state is not initialized")
    original_mask = list(mask)
    old_active = state.active_ranks_cpu.tolist()
    if len(mask) != state.active_ranks.numel() and state.active_ranks.numel() == 1:
        # Native DP launches one-rank scheduler worlds. The FT control plane
        # still carries a DP-rank mask, while each live scheduler only needs its
        # local one-rank Mooncake mirror to stay active; DPC owns DP routing.
        mask = [True]
    if len(mask) != state.active_ranks.numel():
        raise ValueError(
            f"active mask size mismatch: got {len(mask)}, "
            f"expected {state.active_ranks.numel()}"
        )
    tensor = torch.tensor(
        [1 if item else 0 for item in mask],
        dtype=state.active_ranks.dtype,
        device=state.active_ranks.device,
    )
    state.active_ranks.copy_(tensor)
    state.sync_active_to_cpu()
    _refresh_ep_members()
    if envs.SGLANG_FT_PRECISION_DEBUG.get():
        logger.info(
            "[FTPrecisionDebug][ElasticEP] apply_active_rank_mask "
            "requested=%s effective=%s old_active=%s new_active=%s",
            original_mask,
            mask,
            old_active,
            state.active_ranks_cpu.tolist(),
        )


def try_recover_ranks(global_ranks: List[int]) -> bool:
    from mooncake import ep as mooncake_ep

    world_group = torch.distributed.group.WORLD
    if not all(mooncake_ep.get_peer_state(world_group, global_ranks)):
        # The relaunched ranks have not finished initializing yet.
        return False

    # Recover the world backend first, then recover each derived process group
    # using ranks mapped into that group's local rank space.
    mooncake_ep.recover_ranks(world_group, global_ranks)

    for group in _iter_live_parallel_groups():
        group_local_ranks = _map_global_to_group_local_ranks(group.ranks, global_ranks)
        if not group_local_ranks:
            continue

        _wait_for_peer_state(mooncake_ep, group.device_group, group_local_ranks)
        mooncake_ep.recover_ranks(group.device_group, group_local_ranks)

        _wait_for_peer_state(mooncake_ep, group.cpu_group, group_local_ranks)
        mooncake_ep.recover_ranks(group.cpu_group, group_local_ranks)
        _maybe_create_message_queue(group)

    _refresh_ep_members()
    return True


def join_process_groups():
    from mooncake import ep as mooncake_ep

    def join_backend(label: str, backend) -> None:
        logger.info("Recovered rank joining Mooncake backend %s", label)
        mooncake_ep.join_group(backend)

    join_backend(
        "default_world",
        torch.distributed.group.WORLD,
    )

    for group in _iter_live_parallel_groups():
        if group.world_size <= 1:
            continue

        join_backend(
            f"{group.unique_name}:device",
            group.device_group,
        )
        join_backend(
            f"{group.unique_name}:cpu",
            group.cpu_group,
        )
        _maybe_create_message_queue(group)

    _refresh_ep_members(allow_missing_buffer_in_fallback=True)
