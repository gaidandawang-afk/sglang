from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class RecoveryResult:
    success: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class DistributedRecoveryManager:
    """Best-effort distributed communication cleanup and health checks."""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def disable_communicators(self) -> RecoveryResult:
        try:
            from sglang.srt.distributed.parallel_state import get_all_model_groups

            groups = get_all_model_groups()
            disabled = []
            for group in groups:
                for attr in (
                    "pynccl_comm",
                    "ca_comm",
                    "qr_comm",
                    "pymscclpp_comm",
                    "torch_symm_mem_comm",
                    "hpu_communicator",
                    "xpu_communicator",
                    "npu_communicator",
                ):
                    comm = getattr(group, attr, None)
                    if comm is not None and hasattr(comm, "disabled"):
                        comm.disabled = True
                        disabled.append(f"{group.unique_name}.{attr}")
            return RecoveryResult(
                success=True,
                message="Communicators disabled.",
                details={"disabled": disabled},
            )
        except Exception as exc:
            logger.exception("Failed to disable communicators")
            return RecoveryResult(False, str(exc))

    def abort_communicators(self, timeout_sec: int) -> RecoveryResult:
        try:
            from sglang.srt.distributed.parallel_state import (
                safe_abort_or_destroy_all_groups,
            )

            results = safe_abort_or_destroy_all_groups(timeout_sec=timeout_sec)
            success = all(result.success for result in results)
            return RecoveryResult(
                success=success,
                message=(
                    "Communication groups aborted/destroyed."
                    if success
                    else "Some communication groups failed to abort/destroy."
                ),
                details={"groups": [dataclasses.asdict(result) for result in results]},
            )
        except Exception as exc:
            logger.exception("Failed to abort communicators")
            return RecoveryResult(False, str(exc))

    def cleanup_distributed(self) -> RecoveryResult:
        try:
            from sglang.srt.distributed.parallel_state import cleanup_dist_env_and_memory

            cleanup_dist_env_and_memory()
            return RecoveryResult(True, "Distributed environment cleaned up.")
        except Exception as exc:
            logger.exception("Failed to clean up distributed environment")
            return RecoveryResult(False, str(exc))

    def reinit_distributed_on_main_thread(self, params: Dict[str, Any]) -> RecoveryResult:
        try:
            model_runner = self.scheduler.tp_worker.model_runner
            model_runner.fault_tolerance_reinit_distributed(params)
            model_runner.fault_tolerance_rebind_distributed_groups()
            model_runner.fault_tolerance_invalidate_cuda_graphs()
            if params.get("recapture_cuda_graph", True):
                model_runner.fault_tolerance_recapture_cuda_graphs()
            return RecoveryResult(True, "Distributed environment reinitialized.")
        except Exception as exc:
            logger.exception("Failed to reinitialize distributed environment")
            return RecoveryResult(False, str(exc))

    def health_check(self, timeout_sec: int) -> RecoveryResult:
        del timeout_sec
        start = time.perf_counter()
        try:
            model_runner = self.scheduler.tp_worker.model_runner
            model_runner.fault_tolerance_health_check()
            return RecoveryResult(
                True,
                "Health collective succeeded.",
                {"elapsed_sec": time.perf_counter() - start},
            )
        except Exception as exc:
            logger.exception("Fault tolerance health check failed")
            return RecoveryResult(False, str(exc))
