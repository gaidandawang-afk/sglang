from __future__ import annotations

import threading
from typing import List, Optional

from sglang.srt.fault_tolerance.topology import (
    dp_rank_for_global_rank,
    resolve_ft_rank_topology,
)


class SchedulerProcessRegistry:
    """Track scheduler process identity across elastic-EP rejoin.

    DPC owns the original ``multiprocessing.Process`` objects, but an elastic-EP
    rejoin rank can be relaunched externally. In that case DPC must stop using
    the stale Process object for liveness and shutdown decisions and use the
    replacement PID reported by the rejoined scheduler.
    """

    def __init__(
        self,
        *,
        dp_size: int,
        tp_size: int,
        attn_cp_size: int = 1,
        enable_dp_attention: bool = False,
    ) -> None:
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.attn_cp_size = attn_cp_size
        self.enable_dp_attention = enable_dp_attention
        self.rank_topology = resolve_ft_rank_topology(
            dp_size=dp_size,
            tp_size=tp_size,
            attn_cp_size=attn_cp_size,
            enable_dp_attention=enable_dp_attention,
        )
        self.pids: List[Optional[int]] = []
        self.reported_exit_ranks: set[int] = set()
        self._lock = threading.Lock()

    def append_process(self, proc) -> None:
        with self._lock:
            self.pids.append(getattr(proc, "pid", None))

    def dp_rank_for_scheduler_index(self, scheduler_index: int) -> Optional[int]:
        if self.enable_dp_attention:
            return dp_rank_for_global_rank(
                scheduler_index,
                dp_size=self.dp_size,
                ranks_per_dp=self.rank_topology.ranks_per_dp,
            )
        if self.tp_size == 1 and 0 <= scheduler_index < self.dp_size:
            return scheduler_index
        dp_rank = scheduler_index // max(1, self.tp_size)
        if 0 <= dp_rank < self.dp_size:
            return dp_rank
        return None

    def register_rejoin(self, rank: int, pid: int) -> Optional[int]:
        with self._lock:
            old_pid = self.pids[rank]
            self.pids[rank] = pid
            self.reported_exit_ranks.discard(rank)
            return old_pid

    def mark_process_exit_reported(self, rank: int) -> bool:
        with self._lock:
            if rank in self.reported_exit_ranks:
                return False
            self.reported_exit_ranks.add(rank)
            return True

    def _is_pid_alive(self, pid: int) -> bool:
        import psutil

        try:
            proc = psutil.Process(pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    def replacement_pid(self, rank: int, proc) -> Optional[int]:
        with self._lock:
            if not (0 <= rank < len(self.pids)):
                return None
            pid = self.pids[rank]
        if pid is not None and pid != getattr(proc, "pid", None):
            return pid
        return None

    def is_rank_alive(self, rank: int, proc) -> bool:
        replacement_pid = self.replacement_pid(rank, proc)
        if replacement_pid is not None:
            return self._is_pid_alive(replacement_pid)
        return proc is not None and proc.is_alive()

    def should_ignore_process_exit(self, rank: int, proc) -> bool:
        replacement_pid = self.replacement_pid(rank, proc)
        return replacement_pid is not None and self._is_pid_alive(replacement_pid)

    def has_dead_scheduler_rank(self, processes) -> bool:
        return any(
            proc is not None and not self.is_rank_alive(rank, proc)
            for rank, proc in enumerate(processes)
        )

    def ft_control_rank_for_target(
        self,
        target_rank: int,
        *,
        control_message_step: int,
        worker_count: int,
    ) -> Optional[int]:
        if self.enable_dp_attention:
            if control_message_step != 1:
                return 0
            return self.dp_rank_for_scheduler_index(target_rank)
        if 0 <= target_rank < worker_count:
            return target_rank
        return None

    def is_ft_control_rank_reachable(
        self,
        control_rank: int,
        *,
        control_message_step: int,
        worker_count: int,
        status: List[bool],
        processes,
    ) -> bool:
        if not (0 <= control_rank < worker_count):
            return False
        if self.enable_dp_attention and control_message_step == 1:
            leader_rank = control_rank * self.rank_topology.ranks_per_dp
            if not (0 <= leader_rank < len(processes)):
                return False
            return self.is_rank_alive(leader_rank, processes[leader_rank])
        return bool(status[control_rank])
