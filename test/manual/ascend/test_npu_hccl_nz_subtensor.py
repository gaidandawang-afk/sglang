"""Isolate HCCL P2P and FRACTAL_NZ sub-tensor writes on two NPUs.

Run from the SGLang repository root:

    torchrun --standalone --nproc-per-node=2 \
      test/manual/ascend/test_npu_hccl_nz_subtensor.py

The last case intentionally submits a non-zero-offset internal-format view to
HCCL. It is expected to fail, so it runs after every non-destructive case.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch_npu

from sglang.srt.hardware_backend.npu.utils import (
    NPUACLFormat,
    copy_to_npu_formatted_tensor_,
    npu_format_cast,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def make_cpu_value(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    # Small integers are exactly representable in bfloat16.
    value = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.int64)
    return ((value.reshape(shape) % 13) + seed).to(torch.bfloat16)


def make_parent(*, rank: int, experts: int, rows: int, cols: int) -> torch.nn.Parameter:
    slots = [
        make_cpu_value((rows, cols), rank * 64 + slot * 16) for slot in range(experts)
    ]
    nd_parent = torch.stack(slots).npu(rank)
    nz_parent = npu_format_cast(nd_parent, NPUACLFormat.ACL_FORMAT_FRACTAL_NZ)
    return torch.nn.Parameter(nz_parent, requires_grad=False)


def make_npu_value(
    shape: tuple[int, ...], seed: int, rank: int, acl_format: NPUACLFormat
) -> torch.Tensor:
    value = make_cpu_value(shape, seed).npu(rank)
    if acl_format == NPUACLFormat.ACL_FORMAT_ND:
        return value
    return npu_format_cast(value, acl_format)


def tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    storage_base_ptr = tensor.untyped_storage().data_ptr()
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "format": int(torch_npu.get_npu_format(tensor)),
        "storage_offset": int(tensor.storage_offset()),
        "data_ptr": int(tensor.data_ptr()),
        "storage_base_ptr": int(storage_base_ptr),
        "data_ptr_delta": int(tensor.data_ptr() - storage_base_ptr),
        "logical_numel": int(tensor.numel()),
        "physical_storage_size": int(torch_npu.get_storage_size(tensor)),
    }


def changed_slots(before: torch.Tensor, after: torch.Tensor) -> list[int]:
    return [
        slot
        for slot in range(before.shape[0])
        if not torch.equal(before[slot], after[slot])
    ]


def run_local_case(
    name: str,
    operation: Callable[[], dict[str, Any]],
    *,
    rank: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name}
    try:
        result.update(operation())
        result.setdefault("local_pass", True)
    except Exception as exc:
        result.update(
            {
                "local_pass": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
        )

    flag = torch.tensor(
        [1 if result["local_pass"] else 0], dtype=torch.int32, device=f"npu:{rank}"
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    result["global_pass"] = bool(flag.item())
    if result["local_pass"] and not result["global_pass"]:
        result["remote_rank_failed"] = True
    return result


def run_p2p_case(
    *,
    rank: int,
    shape: tuple[int, int],
    repeats: int,
    acl_format: NPUACLFormat,
) -> dict[str, Any]:
    expected_cpu = make_cpu_value(shape, 7)
    send_tensor = make_npu_value(shape, 7, rank, acl_format)
    recv_tensor = torch_npu.empty_with_format(
        shape,
        dtype=torch.bfloat16,
        device=f"npu:{rank}",
        acl_format=int(acl_format),
    )
    metadata = tensor_metadata(send_tensor if rank == 0 else recv_tensor)

    for _ in range(repeats):
        recv_tensor.fill_(-1)
        if rank == 0:
            ops = [dist.P2POp(dist.isend, send_tensor, 1)]
        else:
            ops = [dist.P2POp(dist.irecv, recv_tensor, 0)]
        requests = dist.batch_isend_irecv(ops)
        for request in requests:
            request.wait()
        torch_npu.npu.synchronize()
        if rank == 1 and not torch.equal(recv_tensor.cpu(), expected_cpu):
            return {
                "local_pass": False,
                "metadata": metadata,
                "failed_repeat": _,
                "reason": "received logical values differ from rank0 source",
            }

    return {
        "local_pass": True,
        "metadata": metadata,
        "repeats": repeats,
    }


def evaluate_parent_write(
    *,
    rank: int,
    experts: int,
    rows: int,
    cols: int,
    write: Callable[[torch.Tensor, torch.Tensor], None],
    half: bool,
) -> dict[str, Any]:
    parent = make_parent(rank=rank, experts=experts, rows=rows, cols=cols)
    before = parent.detach().cpu().clone()
    destination = parent[1]
    destination_metadata = tensor_metadata(destination)
    if half:
        destination = destination.narrow(0, rows // 2, rows - rows // 2)
        replacement_cpu = make_cpu_value(tuple(destination.shape), 208 + rank)
    else:
        replacement_cpu = make_cpu_value(tuple(destination.shape), 176 + rank)
    replacement_nd = replacement_cpu.npu(rank)

    write(destination, replacement_nd)
    torch_npu.npu.synchronize()
    after = parent.detach().cpu()

    if half:
        expected = before.clone()
        expected[1, rows // 2 :] = replacement_cpu
    else:
        expected = before.clone()
        expected[1] = replacement_cpu

    actual_changed_slots = changed_slots(before, after)
    return {
        "local_pass": torch.equal(after, expected),
        "parent_metadata": tensor_metadata(parent),
        "destination_metadata": destination_metadata,
        "write_view_metadata": tensor_metadata(destination),
        "expected_changed_slots": [1],
        "actual_changed_slots": actual_changed_slots,
        "target_equal": torch.equal(after[1], expected[1]),
        "neighbor_equal": all(
            torch.equal(after[slot], before[slot])
            for slot in range(experts)
            if slot != 1
        ),
    }


def direct_copy(destination: torch.Tensor, source: torch.Tensor) -> None:
    destination.copy_(source)


def standalone_formatted_copy(destination: torch.Tensor, source: torch.Tensor) -> None:
    # This is the rejected approach: construct an independent formatted tensor
    # with the shape of a nested view, then raw-copy it into that view.
    copy_to_npu_formatted_tensor_(destination, source)


def evaluate_full_roundtrip(
    *, rank: int, experts: int, rows: int, cols: int
) -> dict[str, Any]:
    parent = make_parent(rank=rank, experts=experts, rows=rows, cols=cols)
    before = parent.detach().cpu().clone()
    full_expert = parent[1]
    half = full_expert.narrow(0, rows // 2, rows - rows // 2)
    replacement_cpu = make_cpu_value(tuple(half.shape), 240 + rank)

    nd_full = torch_npu.empty_with_format(
        tuple(full_expert.shape),
        dtype=full_expert.dtype,
        device=full_expert.device,
        acl_format=int(NPUACLFormat.ACL_FORMAT_ND),
    )
    nd_full.copy_(full_expert)
    nd_full.narrow(0, rows // 2, rows - rows // 2).copy_(replacement_cpu.npu(rank))
    copy_to_npu_formatted_tensor_(full_expert, nd_full)
    torch_npu.npu.synchronize()

    expected = before.clone()
    expected[1, rows // 2 :] = replacement_cpu
    after = parent.detach().cpu()
    return {
        "local_pass": torch.equal(after, expected),
        "full_expert_metadata": tensor_metadata(full_expert),
        "half_metadata": tensor_metadata(half),
        "nd_full_metadata": tensor_metadata(nd_full),
        "expected_changed_slots": [1],
        "actual_changed_slots": changed_slots(before, after),
        "target_equal": torch.equal(after[1], expected[1]),
        "neighbor_equal": all(
            torch.equal(after[slot], before[slot])
            for slot in range(experts)
            if slot != 1
        ),
    }


def run_expected_offset_error(
    *, rank: int, experts: int, rows: int, cols: int
) -> dict[str, Any]:
    parent = make_parent(rank=rank, experts=experts, rows=rows, cols=cols)
    expert_view = parent[1]
    result = {
        "name": "p2p_parent_nz_view_with_offset",
        "metadata": tensor_metadata(expert_view),
    }
    try:
        if rank == 0:
            ops = [dist.P2POp(dist.isend, expert_view, 1)]
        else:
            ops = [dist.P2POp(dist.irecv, expert_view, 0)]
        requests = dist.batch_isend_irecv(ops)
        for request in requests:
            request.wait()
        result.update(
            {
                "status": "unexpected_success",
                "expected_error_observed": False,
            }
        )
    except Exception as exc:
        message = str(exc)
        result.update(
            {
                "status": "expected_error"
                if "storage_offset" in message
                else "different_error",
                "expected_error_observed": "storage_offset" in message,
                "error_type": type(exc).__name__,
                "error": message,
            }
        )
    return result


def main() -> None:
    args = parse_args()
    if args.experts < 3:
        raise ValueError("--experts must be at least 3")
    if args.rows < 2 or args.rows % 2 != 0:
        raise ValueError("--rows must be a positive even number")
    if args.cols < 1 or args.repeats < 1:
        raise ValueError("--cols and --repeats must be positive")

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError("this test requires exactly two torchrun processes")

    torch_npu.npu.config.allow_internal_format = True
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl")

    shape = (args.rows, args.cols)
    results = []
    results.append(
        run_local_case(
            "p2p_offset_zero_nz",
            lambda: run_p2p_case(
                rank=rank,
                shape=shape,
                repeats=args.repeats,
                acl_format=NPUACLFormat.ACL_FORMAT_FRACTAL_NZ,
            ),
            rank=rank,
        )
    )
    results.append(
        run_local_case(
            "p2p_offset_zero_nd",
            lambda: run_p2p_case(
                rank=rank,
                shape=shape,
                repeats=args.repeats,
                acl_format=NPUACLFormat.ACL_FORMAT_ND,
            ),
            rank=rank,
        )
    )
    results.append(
        run_local_case(
            "copy_nd_to_full_nz_parent_slot",
            lambda: evaluate_parent_write(
                rank=rank,
                experts=args.experts,
                rows=args.rows,
                cols=args.cols,
                write=direct_copy,
                half=False,
            ),
            rank=rank,
        )
    )
    results.append(
        run_local_case(
            "copy_nd_to_half_nz_parent_view",
            lambda: evaluate_parent_write(
                rank=rank,
                experts=args.experts,
                rows=args.rows,
                cols=args.cols,
                write=direct_copy,
                half=True,
            ),
            rank=rank,
        )
    )
    results.append(
        run_local_case(
            "copy_nd_to_half_via_standalone_nz",
            lambda: evaluate_parent_write(
                rank=rank,
                experts=args.experts,
                rows=args.rows,
                cols=args.cols,
                write=standalone_formatted_copy,
                half=True,
            ),
            rank=rank,
        )
    )
    results.append(
        run_local_case(
            "copy_nd_to_half_via_full_nd_roundtrip",
            lambda: evaluate_full_roundtrip(
                rank=rank,
                experts=args.experts,
                rows=args.rows,
                cols=args.cols,
            ),
            rank=rank,
        )
    )

    # Collect all non-destructive results before intentionally triggering the
    # unsupported offset-view P2P path.
    rank_results: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(rank_results, results)
    expected_error_result = run_expected_offset_error(
        rank=rank,
        experts=args.experts,
        rows=args.rows,
        cols=args.cols,
    )

    if rank == 0:
        report = {
            "torch_version": torch.__version__,
            "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
            "device_name": torch_npu.npu.get_device_name(0),
            "arguments": vars(args),
            "rank_results": rank_results,
            "expected_offset_error_results": [expected_error_result],
        }
        print("NPU_HCCL_NZ_SUBTENSOR_REPORT_BEGIN", flush=True)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        print("NPU_HCCL_NZ_SUBTENSOR_REPORT_END", flush=True)


if __name__ == "__main__":
    main()
