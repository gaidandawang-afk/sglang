"""Minimal NPU check for offset-zero aliases used by FT expert recovery."""

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.utils import (
    copy_npu_formatted_tensor_,
    npu_format_cast,
)


def main() -> None:
    device = torch.device("npu:0")
    shape = (3, 32, 32)
    expected = torch.arange(
        shape[0] * shape[1] * shape[2], dtype=torch.float32
    ).reshape(shape)
    parent = npu_format_cast(expected.to(device=device, dtype=torch.bfloat16))
    parent_data_ptr = parent.data_ptr()

    for slot in range(shape[0]):
        replacement_cpu = torch.full(
            shape[1:],
            fill_value=1000 + slot,
            dtype=torch.bfloat16,
        )
        replacement = npu_format_cast(replacement_cpu.to(device))
        destination = parent[slot]
        original_storage_offset = destination.storage_offset()

        copy_npu_formatted_tensor_(destination, replacement)
        torch_npu.npu.synchronize()

        if parent.data_ptr() != parent_data_ptr:
            raise AssertionError("parent data_ptr changed during formatted copy")
        if destination.storage_offset() != original_storage_offset:
            raise AssertionError("destination storage_offset changed during copy")
        actual = destination.to(device="cpu")
        if not torch.equal(actual, replacement_cpu):
            raise AssertionError(
                f"formatted expert copy mismatch at slot={slot}: "
                f"offset={original_storage_offset}"
            )

    print(
        "NPU formatted expert subtensor copy passed: "
        f"parent_data_ptr={parent_data_ptr} slots={shape[0]}"
    )


if __name__ == "__main__":
    main()
