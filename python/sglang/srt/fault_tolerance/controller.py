from __future__ import annotations

from enum import Enum
from http import HTTPStatus
from typing import Any, Dict, Iterable, List, Optional, Tuple


class RankState(str, Enum):
    HEALTHY = "healthy"
    PAUSED = "paused"
    DISABLED = "disabled"
    DEAD = "dead"


def ft_failure(message: str) -> Dict[str, Any]:
    return {"success": False, "message": message}


def ft_error_status(error: str) -> int:
    if error == "ft_operation_in_progress":
        return HTTPStatus.CONFLICT
    return HTTPStatus.BAD_REQUEST


def _dp_attention_gate(server_args) -> bool:
    if not getattr(server_args, "enable_dp_attention", False):
        return True
    attention_block_count = getattr(server_args, "dp_size", 1) * getattr(
        server_args, "attn_cp_size", 1
    )
    return getattr(server_args, "tp_size", 1) % attention_block_count == 0


_FT_SUPPORT_GATES = [
    (lambda a: getattr(a, "dp_size", 1) > 1, "ft_requires_dp_gt1"),
    (lambda a: getattr(a, "pp_size", 1) == 1, "ft_requires_pp1"),
    (
        lambda a: getattr(a, "elastic_ep_backend", None) == "mooncake",
        "ft_requires_mooncake_active_rank_backend",
    ),
    (
        lambda a: getattr(a, "disaggregation_mode", "null") == "null",
        "ft_unsupported_with_pd",
    ),
    (lambda a: getattr(a, "device", None) != "npu", "ft_unsupported_with_npu"),
    (
        lambda a: getattr(a, "tokenizer_worker_num", 1) <= 1,
        "ft_unsupported_with_multi_tokenizer",
    ),
    (lambda a: not getattr(a, "use_ray", False), "ft_unsupported_with_ray_engine"),
    (_dp_attention_gate, "ft_requires_tp_divisible_by_dp_and_attn_cp"),
]


def is_ft_supported_config(server_args) -> Tuple[bool, str]:
    for supported, error in _FT_SUPPORT_GATES:
        if not supported(server_args):
            return False, error
    return True, ""


class _ApplyOp:
    """一次 /fault_tolerance/apply 指令：封装校验与状态变更语义。"""

    name = ""

    def __init__(self, ranks: Optional[List[int]] = None):
        self.ranks = ranks

    def validate(self, st: "FaultToleranceState") -> Optional[str]:
        raise NotImplementedError

    def apply_to_disabled(self, st: "FaultToleranceState") -> None:
        """apply 成功时对 disabled 集合的副作用。默认不动。"""

    def resume_targets(self, st: "FaultToleranceState") -> List[int]:
        return st.resume_targets()

    def needs_resume(self) -> bool:
        return True


class _RetryOp(_ApplyOp):
    name = "retry"

    def validate(self, st):
        if not st.has_paused_rank():
            return "no_paused_rank"
        if self.ranks is not None:
            return "retry_does_not_accept_ranks"
        return None


class _ScaleDownOp(_ApplyOp):
    name = "scale_down"

    def validate(self, st):
        if not st.has_paused_rank():
            return "no_paused_rank"
        if not self.ranks:
            return "scale_down_requires_non_empty_ranks"
        requested = set(self.ranks)
        if any(rank < 0 or rank >= st.dp_size for rank in requested):
            return "unknown_rank"
        disabled = st.disabled_dp_ranks | requested
        runtime_active = st.runtime_active_mask()
        if not any(
            is_active and rank not in disabled
            for rank, is_active in enumerate(runtime_active)
        ):
            return "cannot_isolate_all_active_ranks"
        return None

    def apply_to_disabled(self, st):
        st.disabled_dp_ranks.update(self.ranks)


class _RecoverOp(_ApplyOp):
    name = "recover"

    def validate(self, st):
        # recover 不查 has_paused_rank：它拉回 disabled rank，不要求有 paused。
        if not self.ranks:
            return "recover_requires_non_empty_ranks"
        requested = set(self.ranks)
        if any(rank < 0 or rank >= st.dp_size for rank in requested):
            return "unknown_rank"
        if not requested.issubset(st.disabled_dp_ranks):
            return "recover_requires_disabled_ranks"
        return None

    def apply_to_disabled(self, st):
        st.disabled_dp_ranks.difference_update(self.ranks)

    def resume_targets(self, st):
        return []

    def needs_resume(self) -> bool:
        return False


_APPLY_OPS = {
    op.name: op for op in (_RetryOp, _ScaleDownOp, _RecoverOp)
}


def build_apply_op(
    instruction: str, ranks: Optional[List[int]] = None
) -> Optional[_ApplyOp]:
    cls = _APPLY_OPS.get(instruction)
    return None if cls is None else cls(ranks)


class FaultToleranceState:
    def __init__(
        self,
        *,
        dp_size: int,
        strategy: str,
    ):
        self.dp_size = dp_size
        self.strategy = strategy
        self.process_active_ranks = [True] * dp_size
        self.mooncake_active_ranks = [True] * dp_size
        self.disabled_dp_ranks: set[int] = set()
        self.paused_dp_ranks: set[int] = set()
        self.ft_operation_in_progress = False
        self._last_published_effective_active_mask = [True] * dp_size

    def _rank_state(self, rank: int, runtime_active: List[bool]) -> RankState:
        if not runtime_active[rank]:
            return RankState.DEAD
        if rank in self.paused_dp_ranks:
            return RankState.PAUSED
        if rank in self.disabled_dp_ranks:
            return RankState.DISABLED
        return RankState.HEALTHY

    def status_response(self) -> Dict[str, Any]:
        runtime_active = self.runtime_active_mask()
        return {
            "ranks": [
                {"rank": rank, "state": self._rank_state(rank, runtime_active).value}
                for rank in range(self.dp_size)
            ]
        }

    def has_paused_rank(self) -> bool:
        return bool(self.paused_dp_ranks)

    def should_reject_admission(self) -> bool:
        if not any(self.effective_active_mask()):
            return True
        return self.strategy == "pause" and (
            self.ft_operation_in_progress or self.has_paused_rank()
        )

    def runtime_active_mask(self) -> List[bool]:
        return [
            self.process_active_ranks[rank] and self.mooncake_active_ranks[rank]
            for rank in range(self.dp_size)
        ]

    def effective_active_mask(self) -> List[bool]:
        return [
            is_active and rank not in self.disabled_dp_ranks
            for rank, is_active in enumerate(self.runtime_active_mask())
        ]

    def pause_targets(self) -> List[int]:
        runtime_active = self.runtime_active_mask()
        return [rank for rank in range(self.dp_size) if runtime_active[rank]]

    def resume_targets(self) -> List[int]:
        runtime_active = self.runtime_active_mask()
        return sorted(rank for rank in self.paused_dp_ranks if runtime_active[rank])

    def is_rank_routable(self, rank: int) -> bool:
        return 0 <= rank < self.dp_size and self.effective_active_mask()[rank]

    def _normalize_mask(self, mask: List[bool]) -> List[bool]:
        return [rank < len(mask) and bool(mask[rank]) for rank in range(self.dp_size)]

    def _falling_edge(self, old_effective: List[bool]) -> bool:
        return any(
            old and not new
            for old, new in zip(old_effective, self.effective_active_mask())
        )

    def _try_begin_pause(self) -> List[int]:
        if self.ft_operation_in_progress or self.has_paused_rank():
            return []
        targets = self.pause_targets()
        if not targets:
            return []
        self.ft_operation_in_progress = True
        return targets

    def _begin_availability_pause(self, old_effective: List[bool]) -> List[int]:
        if self.strategy != "pause" or not self._falling_edge(old_effective):
            return []
        return self._try_begin_pause()

    def _update_availability(self, mutate) -> List[int]:
        old_effective = self.effective_active_mask()
        mutate()
        return self._begin_availability_pause(old_effective)

    def observe_process_active_ranks(
        self, ranks: Iterable[int], *, active: bool
    ) -> List[int]:
        def _mutate():
            for rank in ranks:
                if 0 <= rank < self.dp_size:
                    self.process_active_ranks[rank] = active

        return self._update_availability(_mutate)

    def observe_mooncake_active_ranks(self, new_mask: List[bool]) -> List[int]:
        def _mutate():
            self.mooncake_active_ranks = self._normalize_mask(new_mask)

        return self._update_availability(_mutate)

    def get_unpublished_effective_active_mask(self) -> Optional[List[bool]]:
        effective_active = self.effective_active_mask()
        if effective_active == self._last_published_effective_active_mask:
            return None
        return effective_active

    def mark_effective_active_mask_published(
        self, effective_active: List[bool]
    ) -> None:
        self._last_published_effective_active_mask = list(effective_active)

    def begin_exception_pause(self) -> List[int]:
        # 异常止血：无策略门（continue 在 manager 的 handle_rank_fault 拦截）。
        return self._try_begin_pause()

    def finish_pause(self, paused_ranks: Iterable[int]) -> None:
        self.paused_dp_ranks = {
            rank for rank in paused_ranks if 0 <= rank < self.dp_size
        }
        self.ft_operation_in_progress = False

    def validate_apply(
        self, instruction: str, ranks: Optional[List[int]]
    ) -> Optional[str]:
        if self.ft_operation_in_progress:
            return "ft_operation_in_progress"
        op = build_apply_op(instruction, ranks)
        if op is None:
            return "unknown_instruction"
        return op.validate(self)

    def begin_recover(
        self, instruction: str, ranks: Optional[List[int]] = None
    ) -> List[int]:
        self.ft_operation_in_progress = True
        op = build_apply_op(instruction, ranks)
        op.apply_to_disabled(self)
        return op.resume_targets(self)

    def commit_recover(
        self,
        resumed_ranks: Iterable[int],
        *,
        clear_paused: bool = False,
    ) -> Dict[str, Any]:
        committed_resumed_ranks = sorted(self.paused_dp_ranks & set(resumed_ranks))
        if clear_paused:
            self.paused_dp_ranks.clear()
        self.ft_operation_in_progress = False
        body = self.status_response()
        body.update(
            {
                "success": True,
                "message": "fault_tolerance_apply_committed",
                "resumed_ranks": committed_resumed_ranks,
            }
        )
        return body
