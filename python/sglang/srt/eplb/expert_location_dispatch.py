# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.runtime_context import get_server_args


@dataclass
class ExpertLocationDispatchInfo:
    ep_dispatch_algorithm: Literal["static", "dynamic", "fake", "lp"]
    # (num_logical_experts,)
    partial_logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]
    # (num_logical_experts, X)
    partial_logical_to_all_physical_map: torch.Tensor
    # (num_logical_experts,)
    partial_logical_to_all_physical_map_num_valid: torch.Tensor
    num_physical_experts: int
    # NPU MC2 communicates in a compact survivor-rank expert namespace while
    # EPLB keeps storage IDs in the immutable original-rank namespace.  These
    # fields are populated before graph capture and the tensor is updated
    # in-place after scale-down.
    npu_mc2_elastic_info: Optional[torch.Tensor] = None
    npu_mc2_original_ep_size: int = 0
    npu_mc2_num_local_physical_experts: int = 0

    @classmethod
    def init_new(cls, layer_id: int):
        server_args = get_server_args()
        ep_dispatch_algorithm = server_args.ep_dispatch_algorithm
        expert_location_metadata = get_global_expert_location_metadata()
        assert expert_location_metadata is not None

        if ep_dispatch_algorithm is None:
            return None

        npu_mc2_elastic_info = None
        npu_mc2_original_ep_size = 0
        npu_mc2_num_local_physical_experts = 0
        if (
            server_args.device == "npu"
            and server_args.enable_fault_tolerance
            and server_args.elastic_ep_backend == "mc2"
        ):
            from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

            state = ElasticEPStateManager.instance()
            if state is None or state.npu_mc2_elastic_info is None:
                raise RuntimeError(
                    "NPU MC2 fault tolerance requires fixed elastic_info "
                    "before expert dispatch"
                )
            npu_mc2_elastic_info = state.npu_mc2_elastic_info.tensor
            npu_mc2_original_ep_size = state.original_ep_size
            (
                npu_mc2_num_local_physical_experts,
                remainder,
            ) = divmod(
                expert_location_metadata.num_physical_experts,
                npu_mc2_original_ep_size,
            )
            if remainder != 0:
                raise RuntimeError(
                    "NPU MC2 expert storage must be divisible by the original "
                    "EP size"
                )

        return cls(
            ep_dispatch_algorithm=ep_dispatch_algorithm,
            partial_logical_to_rank_dispatch_physical_map=(
                expert_location_metadata.logical_to_rank_dispatch_physical_map[
                    layer_id, :
                ]
                if expert_location_metadata.logical_to_rank_dispatch_physical_map
                is not None
                else None
            ),
            partial_logical_to_all_physical_map=expert_location_metadata.logical_to_all_physical_map[
                layer_id, :
            ],
            partial_logical_to_all_physical_map_num_valid=expert_location_metadata.logical_to_all_physical_map_num_valid[
                layer_id, :
            ],
            num_physical_experts=expert_location_metadata.num_physical_experts,
            npu_mc2_elastic_info=npu_mc2_elastic_info,
            npu_mc2_original_ep_size=npu_mc2_original_ep_size,
            npu_mc2_num_local_physical_experts=npu_mc2_num_local_physical_experts,
        )


def transform_select_experts_inputs(
    router_logits: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    info: Optional[ExpertLocationDispatchInfo],
):
    if (info is not None) and (info.ep_dispatch_algorithm == "fake"):
        router_logits.uniform_(5, 10)
        if correction_bias is not None:
            correction_bias = torch.zeros_like(correction_bias)
    return router_logits, correction_bias


def topk_ids_logical_to_physical(
    topk_ids: torch.Tensor,
    info: Optional[ExpertLocationDispatchInfo],
    log2phy_prob: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if info is None:
        return topk_ids

    if info.ep_dispatch_algorithm == "static":
        physical_topk_ids = _topk_ids_logical_to_physical_static(topk_ids, info)
    elif info.ep_dispatch_algorithm in ["dynamic", "fake"]:
        physical_topk_ids = _topk_ids_logical_to_physical_dynamic(topk_ids, info)
    elif info.ep_dispatch_algorithm == "lp":
        if log2phy_prob is None:
            raise RuntimeError(
                "ep_dispatch_algorithm='lp' but log2phy_prob is None at dispatch "
                f"time (topk_ids.shape={tuple(topk_ids.shape)})."
            )
        physical_topk_ids = _topk_ids_logical_to_physical_probability(
            topk_ids, info, log2phy_prob
        )
    else:
        raise NotImplementedError(f"Unknown algorithm {info.ep_dispatch_algorithm}")

    if info.npu_mc2_elastic_info is not None:
        from sglang.srt.elastic_ep.npu_mc2 import compact_mc2_physical_expert_ids

        physical_topk_ids = compact_mc2_physical_expert_ids(
            physical_topk_ids,
            elastic_info=info.npu_mc2_elastic_info,
            original_ep_size=info.npu_mc2_original_ep_size,
            num_local_physical_experts=info.npu_mc2_num_local_physical_experts,
        )
    return physical_topk_ids


def _topk_ids_logical_to_physical_static(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    physical_topk_ids = info.partial_logical_to_rank_dispatch_physical_map[topk_ids]
    if physical_topk_ids.dtype != topk_ids.dtype:
        physical_topk_ids = physical_topk_ids.to(topk_ids.dtype)
    return physical_topk_ids


def _topk_ids_logical_to_physical_dynamic(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    topk_ids_original_shape = topk_ids.shape
    original_dtype = topk_ids.dtype
    device = topk_ids.device
    topk_ids = topk_ids.flatten()

    chosen_dispatch_index = (
        torch.randint(0, 65536, topk_ids.shape, dtype=torch.int32, device=device)
        % info.partial_logical_to_all_physical_map_num_valid[topk_ids]
    )
    topk_ids = info.partial_logical_to_all_physical_map[topk_ids, chosen_dispatch_index]
    if topk_ids.dtype != original_dtype:
        topk_ids = topk_ids.to(original_dtype)

    topk_ids = topk_ids.view(topk_ids_original_shape)
    return topk_ids


def _topk_ids_logical_to_physical_probability(
    topk_ids: torch.Tensor,
    info: ExpertLocationDispatchInfo,
    log2phy_prob: torch.Tensor,
) -> torch.Tensor:
    """Select physical experts via the JIT-compiled CUDA dispatch kernel.

    Raises if ``topk_ids`` isn't on CUDA — the LP path requires the fused
    kernel and there is no torch reference fallback at runtime.
    """
    if not topk_ids.is_cuda:
        raise RuntimeError(
            "LP dispatch requires CUDA tensors; got topk_ids on " f"{topk_ids.device}."
        )
    from sglang.kernels.ops.lplb import cuda_solver

    return cuda_solver.dispatch_probability(
        topk_ids, log2phy_prob, info.partial_logical_to_all_physical_map
    )
