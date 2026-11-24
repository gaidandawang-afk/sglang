import os

from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend


def validate_ascend_ft(cfg) -> None:
    if not (
        cfg.device == "npu"
        and cfg.dp_size == cfg.pp_size == cfg.attn_cp_size == cfg.attn_dcp_size == 1
        and cfg.tp_size == cfg.attn_dp_size == cfg.ep_size > 1
        and cfg.moe_dense_tp_size == 1
        and cfg.enable_dp_lm_head
    ):
        raise ValueError(
            "Ascend FT requires NPU, --tp == --attn-dp-size == --ep > 1, "
            "--moe-dense-tp-size 1, --enable-dp-lm-head, and DP/PP/attention CP sizes of 1."
        )
    if not (
        cfg.moe_a2a_backend == "ascend_ops"
        and cfg.enable_eplb
        and cfg.disable_shared_experts_fusion
        and cfg.expert_distribution_recorder_mode == "stat"
        and cfg.expert_balancedness_report_mode == "off"
        and cfg.ep_dispatch_algorithm == "static"
        and cfg.elastic_ep_backend is None
        and envs.SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT.get()
    ):
        raise ValueError(
            "Ascend FT requires a fixed EP world, static low-latency ascend_ops/MC2 "
            "with EPLB stat recording, no shared expert fusion or balancedness "
            "reporting, and ND weights (SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1)."
        )
    if not (
        cfg.disable_overlap_schedule
        and not cfg.enable_two_batch_overlap
        and not cfg.enable_single_batch_overlap
        and not cfg.enable_mixed_chunk
        and cfg.cuda_graph_config.decode.backend == Backend.DISABLED
        and cfg.cuda_graph_config.prefill.backend == Backend.DISABLED
        and cfg.speculative_algorithm is None
        and cfg.disaggregation_mode == "null"
        and not cfg.enable_pd_role_switch
        and not cfg.enable_hierarchical_cache
        and not cfg.enable_lmcache
        and not cfg.enable_session_radix_cache
        and not cfg.enable_prefill_delayer
        and not cfg.enable_lora
        and not cfg.lora_paths
    ):
        raise ValueError(
            "Ascend FT supports non-overlap eager generation without speculative "
            "decoding, PD disaggregation, external KV caches, prefill delay, or LoRA."
        )
    pad = envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
    if not (
        0 < (cfg.chunked_prefill_size or 0) <= pad
        and cfg.max_prefill_tokens <= pad
        and 0 < (cfg.max_running_requests or 0) <= pad
    ):
        raise ValueError(
            "Ascend FT requires positive --chunked-prefill-size and "
            "--max-running-requests, and each must fit the MC2 token capacity "
            "along with --max-prefill-tokens."
        )
    if not (
        cfg.tokenizer_worker_num == cfg.detokenizer_worker_num == 1
        and not cfg.skip_tokenizer_init
        and not cfg.use_ray
        and cfg.grpc_port is None
        and not envs.SGLANG_RUST_SERVER.get()
    ):
        raise ValueError("Ascend FT requires one Python tokenizer and detokenizer.")

    if not (
        0
        < cfg.fault_tolerance_gloo_timeout
        < min(cfg.fault_tolerance_timeout, cfg.watchdog_timeout)
        and cfg.fault_tolerance_pause_timeout > 0
    ):
        raise ValueError(
            "The Gloo timeout must be positive and below both the FT and scheduler "
            "watchdog timeouts; the FT pause timeout must be positive."
        )
    if os.environ.setdefault("TASK_QUEUE_ENABLE", "0") != "0":
        raise ValueError("Ascend FT requires TASK_QUEUE_ENABLE=0.")

    abort = cfg.fault_tolerance_communication_abort_timeout
    if abort != 0 and abort < 2:
        raise ValueError(
            "The NPU FT communication abort timeout must be 0 or at least 2 seconds."
        )
    if abort:
        os.environ.setdefault("HCCL_EVENT_TIMEOUT", str(abort))
        os.environ.setdefault("HCCL_EXEC_TIMEOUT", str(abort - 1))
        os.environ.setdefault("ACL_DEVICE_SYNC_TIMEOUT", str(abort))
        os.environ.setdefault("ACL_STREAM_TIMEOUT", str(abort * 1000))
        if int(os.environ["HCCL_EVENT_TIMEOUT"]) <= int(
            os.environ["HCCL_EXEC_TIMEOUT"]
        ):
            raise ValueError(
                "HCCL_EVENT_TIMEOUT must be greater than HCCL_EXEC_TIMEOUT."
            )
