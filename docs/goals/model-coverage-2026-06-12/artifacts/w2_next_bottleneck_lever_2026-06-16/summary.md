# W2 Next Bottleneck Lever

This checkpoint adds no new GPU run. It consolidates the current W2 Gemma3
CUDA bottleneck evidence and fixes the next minimal validation target before
another paid benchmark or product patch.

## Current Release-Grade State

- W2 Gemma3 27B CUDA GPTQ remains functional, not release-grade.
- No final validator artifact has printed
  `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.
- Current same-hardware ShareGPT diagnostic endpoint ratio is still around
  60-65% of the vLLM baseline, below the required 80% line.
- No current product correctness blocker is known from the latest run/serve
  smokes, but correctness must still be rerun before any performance claim
  after source changes.

## Evidence Used

- `w2_bottleneck_synthesis_2026-06-15`: c16 gap is split between
  TTFT/prefill/scheduling and sustained decode; decode `tail_mlp` is the
  largest profiled block.
- `w2_typed_decode_profile_2026-06-16`: decode is model-side dominated;
  postprocess/scheduler overhead is not the ordinary bottleneck.
- `w2_batch_prefill_fallback_reason_diag_2026-06-16`: batch prefill is not
  falling back through `LlmExecutor::batch_prefill`; heavy TTFT is inside the
  unified product path.
- `w2_tail_mlp_chain_native_probe_2026-06-16`: native CUDA reproduces product
  tail MLP time: `gate_up + GeGLU + down` is about `216us` per layer at m16,
  or about `13.4ms` over 62 layers.
- `w2_down_input_source_native_probe_2026-06-16`: `down_proj` is fast when
  repeated on constant input, but slows after a `gate_up+GeGLU` producer even
  when reading a separate constant input.
- `w2_down_l2_persist_native_probe_2026-06-16`: single-layer stream access
  policy can restore `down_proj` kernel time.
- `w2_down_l2_persist_cycle_native_probe_2026-06-16`: the simple per-layer
  access-policy win does not survive product-like eight-layer weight rotation.
- `w2_down_prefetch_overlap_native_probe_2026-06-16`: an explicit warm/prefetch
  read can restore the down kernel time, but increases total segment wall time.
- `w2_fa2_source_gemma_full_config_smoke_2026-06-16`: FA2 source product path
  is correct but slower on the current Gemma3 ShareGPT c16 diagnostic.
- `w2_batched_graph_sharegpt_current_diag_2026-06-16`: product
  `--batched-graph` selects the graph path but does not improve endpoint
  throughput.

## Source Audit

The product Gemma3 tail path is a direct per-layer sequence:

1. `gate_up_proj.forward(...)`
2. `fused_gelu_tanh_mul_split(...)`
3. `down_proj.forward(...)`

The unified path is in:

- `crates/ferrum-models/src/models/llama_family_forward_batched.rs`

The non-unified path mirrors the same sequence in:

- `crates/ferrum-models/src/models/llama_family.rs`

CUDA GPTQ dispatch currently goes through the generic `CudaMarlinLinear`
implementation in:

- `crates/ferrum-kernels/src/quant_linear/cuda_marlin.rs`

The Marlin kernel wrapper can identify projection buckets only through the
diagnostic allocation label/profile path in:

- `crates/ferrum-kernels/src/backend/cuda/marlin.rs`

That means a product fix must not rely on hidden env or profile labels as the
semantic API. If a future optimization is specific to Gemma3 `down_proj`, the
projection role/layer context needs to be passed through a typed product path
or otherwise encoded in the loaded linear metadata.

## Falsified Branches

- Missing FA2 product wiring is not the current lever: the corrected FA2 source
  product smoke selected `fa2_source` and was slower than the default path.
- Product batched graph is not the current lever: it selected
  `legacy_batched_decode_graph` and did not improve ShareGPT throughput.
- Batch formation is not the main c16 issue: existing c16 decode stats reach
  m16 and average about 13.6 active items.
- HTTP/postprocess/scheduler overhead is not the ordinary decode bottleneck:
  typed decode profile shows model-side decode dominates.
- Simple Marlin cache-policy/default block tuning is not enough: native gains
  are small or do not survive endpoint validation.
- Existing Triton W4A16 is not a candidate replacement: native evidence shows
  it is materially slower than Marlin for the Gemma3 shapes.
- Simple stream access-policy alone should not be productized: it wins in a
  single-layer loop but fails under multi-layer weight rotation.
- Explicit external down-weight warm/prefetch should not be productized as-is:
  it improves the measured down kernel while increasing segment wall time.

## Next Minimal Validation

The next CUDA-native probe should test whether a producer-integrated down-weight
touch can lower total tail-MLP segment time, not merely the isolated down kernel
time.

Concrete probe shape:

- clone or extend `scripts/microbenches/gemma3_down_prefetch_overlap_perf.cu`;
- keep the product-shaped eight-layer rotation, m16 and m32 rows, and segment
  wall-time measurement;
- add a GeGLU producer variant that optionally touches a configurable slice of
  the next `down_proj` qweight/scales while computing activation;
- compare against:
  - no prefetch,
  - existing second-stream qweight/scales warm,
  - explicit down-warm upper bound;
- accept the branch only if total segment wall time improves, not just
  `down_us`.

Stop condition for that native probe:

- If producer-integrated touch does not reduce m16 segment wall time versus
  no prefetch, abandon the cache-warm branch and move to work-reduction/fusion
  work on the tail MLP.
- If it does reduce segment wall time, only then design a typed product API for
  projection-specific Marlin policy and rerun product `ferrum run`/`serve`
  correctness before endpoint performance diagnostics.

## Current Bottleneck Statement

The best current bottleneck statement is:

1. W2 is no longer blocked on a known correctness failure in the current product
   default path, but performance is still below the release-grade 80% line.
2. The main actionable decode bottleneck is Gemma3 GPTQ dense tail MLP,
   especially the `gate_up -> GeGLU -> down` sequence and its weight/cache
   residency under layer rotation.
3. The cache-residency branch has a real single-layer signal, but the naive
   productization routes have been falsified. The next valid test is a native
   CUDA producer-integrated touch that proves total segment reduction before
   touching product code.

This is planning/diagnostic evidence only. It is not W2 release evidence.
