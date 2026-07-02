# W2 vLLM Source-Diff Checkpoint

Date: 2026-06-16

This is a diagnostic source comparison checkpoint, not release-grade evidence.
It did not run a new paid GPU job and it did not produce
`MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.

## Scope

- vLLM source tree: `/Users/chejinxuan/py_ws/vllm`
- vLLM source head: `0b3ba88f1 Revert "[CPU] Experimentally enable Triton and MRV2 (#43225)"`
- Ferrum checkpoint before this note: `20ee9751 docs(cuda): record active prefill chunk c16 diag`
- Local vLLM Python import check: `torch` is not installed on the Mac host, so
  local Python can only support static/source and shape-level checks, not CUDA
  op execution.

## Direct Source Comparison

Gemma3 model semantics are broadly aligned:

- vLLM `Gemma3MLP` uses one `MergedColumnParallelLinear` for `gate_up_proj`,
  `GeluAndMul(approximate="tanh")`, then `down_proj`.
- vLLM `Gemma3Attention` uses fused QKV projection, per-head `GemmaRMSNorm`
  on Q/K, RoPE, per-layer sliding window, and `query_pre_attn_scalar**-0.5`.
- vLLM `Gemma3DecoderLayer` applies the Gemma3 sandwich norm order:
  attention, post-attention norm, pre-feedforward residual norm, MLP,
  post-feedforward norm.
- Ferrum implements the same Gemma3 surface: fused `qkv_proj`, fused
  `gate_up_proj`, Q/K norm, sliding-window schedule, query scale folded into
  Q norm, GeGLU tanh, and device F32 residual shadow for sandwich norms.

The important performance-path differences are not model semantics:

- vLLM dense GPTQ goes through `apply_gptq_marlin_linear` and vLLM's Marlin op.
  Existing native probes show Ferrum's dense Marlin kernels are effectively tied
  with vLLM Marlin for the W2 shapes under product-relevant weight cycling.
- vLLM V1 has persistent GPU input buffers plus CUDA graph dispatch for uniform
  decode (`CUDAGraphMode.FULL` / `PIECEWISE`) by batch descriptor.
- Ferrum's stable product diagnostics are still eager on the unified Gemma3
  path. `--batched-graph` selects the legacy batched graph path but does not
  help ShareGPT c16/c32. `--unified-graph` is closer to vLLM's model, but it
  has failed under W2 c16 diagnostics with CUDA illegal-address and graph
  instantiation/OOM modes.

## Evidence Reused

- vLLM baseline artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_vllm_sharegpt_baseline_probe_2026-06-15/`
  - vLLM `0.10.1.1`, torch `2.7.1+cu126`, GPTQ Marlin
  - c16 `518.796 tok/s`, c32 `524.128 tok/s`, zero errors on ASCII ShareGPT
- Ferrum latest same-family diagnostic:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_active_decode_prefill_chunk_c16_diag_2026-06-16/`
  - default c16 `320.311 tok/s`, correctness clean, no release PASS
- Dense Marlin native probes:
  `w2_dense_vllm_marlin_native_probe_retry_2026-06-15/` and
  `w2_dense_vllm_marlin_weight_cycle_probe_2026-06-15/`
  - m16 gate_up Ferrum/vLLM both about `134-137 us`
  - m16 down weight-cycle Ferrum/vLLM both about `69 us`
  - conclusion: dense Marlin kernel selection is not the main c16 gap
- Typed profile:
  `w2_marlin_typed_profile_2026-06-16/`
  - m16 decode row: total about `30.2 ms`
  - `tail_mlp` about `14.8 ms`, `marlin_kernel` about `16.6 ms`
  - `attn` about `2.5-2.7 ms`, Q/K/RoPE about `0.69 ms`
  - profiling itself lowers throughput and is diagnostic only
- Graph diagnostics:
  `w2_batched_graph_sharegpt_current_diag_2026-06-16/` and
  `w2_unified_graph_typed_c16_2026-06-16/`
  - legacy batched graph was correctness-clean but did not improve endpoint
    throughput
  - typed unified graph passed run/serve smoke but failed c16 bench with
    `CUDA_ERROR_ILLEGAL_ADDRESS`

## Current Conclusion

There is no new correctness issue in the default product path from this source
comparison. The default path remains functionally clean but not release-grade.

The current bottleneck is no longer likely to be:

- dense GPTQ Marlin kernel choice
- active-decode prefill chunk scheduling
- product `--batched-graph`
- simple Marlin cache-policy or block-policy knobs

The most plausible remaining vLLM/Ferrum gap is the decode integration layer:
vLLM gets more work under persistent-buffer CUDA graph replay, while Ferrum's
usable Gemma3 path is still eager and pays many per-step launches plus
post-model bookkeeping. The closest Ferrum lever is unified graph, but it must
first become correctness-clean and memory-safe before any performance claim.

## Next Minimal Validation

Do not restart or rebuild a GPU environment just to sweep knobs. When a cached
4090 lane is used, run only a targeted unified-graph minimal validation:

1. Native or product micro-run that captures/replays one Gemma3 unified decode
   graph shape with minimal output tokens and records whether graph replay is
   correctness-clean.
2. If it fails, collect the exact graph node/failing kernel evidence and stop.
3. Only after correctness is stable, compare eager vs graph replay for the same
   c16 ShareGPT shape with `n_repeats=1`.

This is the next high-return direction because it directly matches the major
source-level difference found in vLLM.
