"""Validate exact writes into FRACTAL_NZ expert sub-tensors."""

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.utils import (
    NPUACLFormat,
    copy_npu_formatted_tensor_,
    copy_to_npu_formatted_tensor_,
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
            shape[1:], fill_value=1000 + slot, dtype=torch.bfloat16
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
        if not torch.equal(destination.cpu(), replacement_cpu):
            raise AssertionError(
                f"formatted expert copy mismatch at slot={slot}: "
                f"offset={original_storage_offset}"
            )

        nd_replacement_cpu = torch.full(
            shape[1:], fill_value=2000 + slot, dtype=torch.bfloat16
        )
        nd_replacement = nd_replacement_cpu.to(device)
        copy_to_npu_formatted_tensor_(destination, nd_replacement)
        torch_npu.npu.synchronize()
        if not torch.equal(destination.cpu(), nd_replacement_cpu):
            raise AssertionError(
                f"ND-to-formatted expert copy mismatch at slot={slot}: "
                f"offset={original_storage_offset}"
            )

        # Update two logical halves through a full-size ND tensor. A standalone
        # half-size NZ tensor is not a valid proxy for a half inside parent NZ.
        nd_full = torch_npu.empty_with_format(
            tuple(destination.shape),
            dtype=destination.dtype,
            device=destination.device,
            acl_format=NPUACLFormat.ACL_FORMAT_ND,
        )
        nd_full.copy_(destination)
        half_size = shape[1] // 2
        first_half_cpu = torch.full(
            (half_size, shape[2]), fill_value=3000 + slot, dtype=torch.bfloat16
        )
        nd_full.narrow(0, 0, half_size).copy_(first_half_cpu.to(device))
        copy_to_npu_formatted_tensor_(destination, nd_full)

        nd_full.copy_(destination)
        second_half_cpu = torch.full(
            (half_size, shape[2]), fill_value=4000 + slot, dtype=torch.bfloat16
        )
        nd_full.narrow(0, half_size, half_size).copy_(second_half_cpu.to(device))
        copy_to_npu_formatted_tensor_(destination, nd_full)
        torch_npu.npu.synchronize()
        if not torch.equal(destination[:half_size].cpu(), first_half_cpu):
            raise AssertionError(
                f"ND-to-formatted first-half copy mismatch at slot={slot}"
            )
        if not torch.equal(destination[half_size:].cpu(), second_half_cpu):
            raise AssertionError(
                f"ND-to-formatted second-half copy mismatch at slot={slot}"
            )

    print(
        "NPU formatted expert subtensor copy passed: "
        f"parent_data_ptr={parent_data_ptr} slots={shape[0]}"
    )


if __name__ == "__main__":
    main()
