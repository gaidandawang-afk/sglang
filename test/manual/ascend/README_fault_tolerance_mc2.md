# Ascend MC2 fault-tolerance smoke test

This test targets one Atlas A3 node with four Scheduler ranks, Qwen3-30B-A3B,
and the fixed original topology `TP=DP=EP=4`. It validates scale-down after a
real Scheduler process exit. It does not exercise scale-up/recover and must not
depend on NPU graph recapture.

Before starting SGLang, apply
`patches/ascend/sgl-kernel-npu-mc2-elastic-info.patch` to the matching
`sgl-kernel-npu` checkout and rebuild/install its DeepEP package. The patch
passes the optional tensor through both the default C++ strategy and the
`DEEP_USE_MODE=ops` torch-npu strategy down to the ACLNN `elasticInfo` input.

Start a disposable server with the target topology (adapt model and log paths):

```bash
DEEP_USE_MODE=ops python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B \
  --host 0.0.0.0 \
  --port 30000 \
  --device npu \
  --tp-size 4 \
  --dp-size 4 \
  --ep-size 4 \
  --moe-dense-tp-size 1 \
  --moe-dp-size 1 \
  --attn-cp-size 1 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --enable-eplb \
  --eplb-algorithm elasticity_aware \
  --ep-num-redundant-experts 44 \
  --elastic-ep-backend mc2 \
  --enable-fault-tolerance \
  --fault-tolerance-on-error-strategy pause \
  --fault-tolerance-timeout 600 \
  2>&1 | tee /tmp/sglang-npu-ft.log
```

Qwen3-30B-A3B has 128 logical experts. With one of four ranks removed, each
survivor needs 43 fixed expert slots, so launch-time capacity must be
`43 * 4 = 172` physical slots: 128 base experts plus 44 redundant experts.
This is fixed storage allocated before graph capture, not post-failure graph
reconstruction. Startup fails fast if even one-rank scale-down would leave fewer
physical slots than logical experts.

At scale-down, the decode-graph MC2 dispatch/combine window is intentionally not
rebuilt. Its fixed-address `elastic_info` is committed in place only after the
expert layout is ready. All graph-external domains are rebuilt from the
survivors through the controller-hosted rendezvous store:

- Scheduler/control objects and optional PrefillDelayer negotiation use the new
  compact-rank Gloo group after rebuild.
- MLP-sync metadata uses a new compact-rank Scheduler HCCL group.
- EPLB statistics and expert P2P use a separate rebuilt HCCL group; peers are
  translated from immutable original ranks to compact survivor ranks.
- The precompile barrier selects the rebuilt Scheduler HCCL group if invoked
  after scale-down. The required `--deepep-mode low_latency` keeps prefill and
  decode on the MC2 low-latency path, so DeepEP normal prefill communication is
  not enabled in this FT configuration.

Before rebuilding those domains, every survivor calls
`torch_npu.npu.stop_device(local_device_id)` followed immediately by
`restart_device(local_device_id)` on the Scheduler/ModelRunner device-owner
thread. This aborts unfinished device work and resets TorchNPU's HCCL watchdog
state left by the failed rank. The restart call deliberately uses its default
mode: do not pass `rebuild_all_resource(s)=True`, because that mode can rebuild
streams and mark existing tensors unsafe. The default path does not request a
new graph capture or replace the fixed MC2 buffers. The manual test verifies the
stop/restart/rebuild/elastic-info ordering and then exercises the same captured
decode graph with deterministic requests.

Expert restoration has a strict priority order for every destination slot:
reuse an unchanged or duplicate local physical expert, copy it from another
survivor over the rebuilt HCCL group, and use DRAM backup or checkpoint reload
only when no survivor owns that logical expert.

Find a non-controller Scheduler PID and its original DP rank, then run:

```bash
python test/manual/ascend/test_fault_tolerance_mc2_scale_down.py \
  --victim-rank 1 \
  --victim-pid <scheduler-pid-for-rank-1> \
  --server-log /tmp/sglang-npu-ft.log
```

The test requires all four ranks to be healthy first. It sends a deterministic
baseline request, kills exactly the supplied Scheduler PID, waits for the FT
incident, applies sparse scale-down, and sends three more requests. With
`--server-log`, it additionally proves that the MC2 `elastic_info` device
address did not change, the victim is `-1` in the original-to-effective table,
the reverse table remains fixed-width, and every survivor logged the same
rebuilt process-group membership plus the expected compact-rank mapping.

Use `--wait-for-existing-incident` instead of `--victim-pid` when the process
failure is injected externally.
