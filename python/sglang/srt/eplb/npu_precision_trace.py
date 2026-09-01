import logging
import os
from typing import Dict, Iterable, List, Optional

import torch

from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

_TRACE_ENV = "SGLANG_NPU_FT_PRECISION_TRACE"
_TRACE_VALUES_ENV = "SGLANG_NPU_FT_PRECISION_TRACE_VALUES"
_TRACE_LAYERS_ENV = "SGLANG_NPU_FT_PRECISION_TRACE_LAYERS"


def npu_precision_trace_enabled() -> bool:
    return get_bool_env_var(_TRACE_ENV)


def _selected_layer_ids(available_layer_ids: Iterable[int]) -> List[int]:
    available = sorted(available_layer_ids)
    if not available:
        return []

    configured = os.getenv(_TRACE_LAYERS_ENV, "").strip()
    if configured == "all":
        return available
    if configured:
        requested = {int(value.strip()) for value in configured.split(",")}
        return [layer_id for layer_id in available if layer_id in requested]
    return [available[0], available[-1]] if len(available) > 1 else available


def _sample_coordinates(shape: torch.Size) -> List[tuple[int, ...]]:
    if not shape or any(size == 0 for size in shape):
        return []

    candidates = [
        tuple(0 for _ in shape),
        tuple(size - 1 for size in shape),
        tuple(size // 2 for size in shape),
        tuple(size // 3 for size in shape),
        tuple((2 * size) // 3 for size in shape),
    ]
    output = []
    for coordinate in candidates:
        if coordinate not in output:
            output.append(coordinate)
    return output


def _tensor_trace_fields(tensor: torch.Tensor, *, include_values: bool) -> dict:
    fields = {
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "dtype": str(tensor.dtype),
        "storage_offset": tensor.storage_offset(),
        "data_ptr": tensor.data_ptr(),
        "storage_data_ptr": tensor.untyped_storage().data_ptr(),
        "contiguous": tensor.is_contiguous(),
    }
    if tensor.device.type == "npu":
        import torch_npu

        fields["acl_format"] = torch_npu.get_npu_format(tensor)
        fields["npu_storage_size"] = torch_npu.get_storage_size(tensor)

    if include_values:
        coordinates = _sample_coordinates(tensor.shape)
        try:
            samples = torch.stack([tensor[coordinate] for coordinate in coordinates])
            fields["sample_coordinates"] = coordinates
            fields["sample_values"] = samples.detach().cpu().tolist()
        except Exception as exc:
            fields["sample_error"] = repr(exc)
    return fields


def trace_expert_tensor_state(
    *,
    stage: str,
    rank: int,
    routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
    physical_to_logical_map_cpu: Optional[torch.Tensor] = None,
    num_local_physical_experts: Optional[int] = None,
    logical_experts_by_layer: Optional[Dict[int, List[int]]] = None,
) -> None:
    if not npu_precision_trace_enabled():
        return

    include_values = get_bool_env_var(_TRACE_VALUES_ENV)
    available_layer_ids = (
        logical_experts_by_layer.keys()
        if logical_experts_by_layer is not None
        else routed_experts_weights_of_layer.keys()
    )
    for layer_id in _selected_layer_ids(available_layer_ids):
        tensors = routed_experts_weights_of_layer[layer_id]
        if not tensors:
            continue
        local_expert_count = tensors[0].shape[0]
        for local_slot in range(local_expert_count):
            logical_expert_id = None
            if (
                physical_to_logical_map_cpu is not None
                and num_local_physical_experts is not None
            ):
                physical_slot = rank * num_local_physical_experts + local_slot
                logical_expert_id = int(
                    physical_to_logical_map_cpu[layer_id, physical_slot].item()
                )
            if (
                logical_experts_by_layer is not None
                and logical_expert_id not in logical_experts_by_layer[layer_id]
            ):
                continue
            for tensor_index, tensor in enumerate(tensors):
                expert_tensor = tensor[local_slot]
                try:
                    fields = _tensor_trace_fields(
                        expert_tensor, include_values=include_values
                    )
                    logger.info(
                        "NPU FT precision state stage=%s rank=%s layer_id=%s "
                        "local_slot=%s logical_expert_id=%s tensor_index=%s fields=%s",
                        stage,
                        rank,
                        layer_id,
                        local_slot,
                        logical_expert_id,
                        tensor_index,
                        fields,
                    )
                except Exception as exc:
                    logger.exception(
                        "NPU FT precision state stage=%s rank=%s layer_id=%s "
                        "local_slot=%s logical_expert_id=%s tensor_index=%s "
                        "trace_error=%r",
                        stage,
                        rank,
                        layer_id,
                        local_slot,
                        logical_expert_id,
                        tensor_index,
                        exc,
                    )
