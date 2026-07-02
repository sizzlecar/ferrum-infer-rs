# W2 Bottleneck Synthesis

This note synthesizes existing 2026-06-15 c16 artifacts. It does not add a new
GPU run and is not release-grade evidence.

## Evidence Used

- Ferrum/vLLM ShareGPT baseline:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_vllm_sharegpt_baseline_probe_2026-06-15/`.
- Ferrum batched decode profile buckets:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_tail_profile_buckets_2026-06-15/`.
- Ferrum Marlin projection profile:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_projection_profile_2026-06-15/`.
- Ferrum/vLLM native Marlin weight-cycle probe:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_dense_vllm_marlin_weight_cycle_probe_2026-06-15/`.
- Product Marlin shape trace:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_shape_trace_probe_2026-06-15/`.

## End-to-End Gap

Same-hardware c16 ShareGPT diagnostic:

- Ferrum: `340.003 tok/s`, `5.328 req/s`.
- vLLM: `518.796 tok/s`, `8.106 req/s`.
- Ratio: `65.5%`; Ferrum is about `34.5%` behind vLLM.

Latency split:

- Ferrum p50 TTFT: `887.683ms`; vLLM p50 TTFT: `411.903ms`;
  gap `+475.780ms`.
- Ferrum p50 TPOT: `32.817ms`; vLLM p50 TPOT: `24.789ms`;
  gap `+8.027ms/token`.
- For roughly 63 inter-token gaps, TPOT accounts for about `506ms`.
- The batch wall-time gap is about `16 / 5.328 - 16 / 8.106 = 1.03s`.
- TTFT plus TPOT explains about `0.98s` of that gap.

Interpretation: the c16 deficit is split between first-token/prefill/scheduling
cost and sustained decode cost. It is not explained by a single dense Marlin
kernel swap.

## Decode Shape And Batch Formation

Existing decode batch stats show c16 already forms large batches:

- c=16 segment: `calls=391`, `total_items=5334`, `avg_m=13.642`, `max_m=16`.
- Batched profile rows include `118` rows at `m=16`.
- The product shape trace confirms the trace is wired to real `ferrum serve`;
  its single-request smoke naturally decodes at `m=1`, but that single-request
  result should not be used to explain c16.

Interpretation: the remaining c16 gap is not primarily "batch never forms".

## Decode Cost Breakdown

From `w2_tail_profile_buckets_2026-06-15`, batch `m=16`:

- mean decode step total: `28.037ms`.
- `tail_mlp`: `13.744ms`, `49.0%`.
- `matmul`: `6.971ms`, `24.9%`.
- `attention`: `2.406ms`, `8.6%`.
- `unwrapped`: `0.649ms`, `2.3%`.

From `w2_marlin_projection_profile_2026-06-15`, batch `m=16` with Marlin
projection profiling enabled:

- mean decode step total: `30.063ms`.
- Marlin kernels: `16.548ms`, `55.0%`.
- `gate_up`: `8.728ms`.
- `down`: `4.352ms`.
- `qkv`: `2.132ms`.
- `o_proj`: `1.336ms`.

The native Ferrum/vLLM Marlin weight-cycle probe shows that vLLM dense Marlin
does not remove the down-projection weight-cycle penalty. Therefore, the next
decode lever should be broader than a direct dense-Marlin kernel replacement:
reduce Gemma3 tail MLP work/copies/synchronization or change how those
projections are scheduled/fused.

## Current Bottleneck Statement

The best current bottleneck statement is:

1. Ferrum c16 batching works well enough to reach `m=16`, but the batch drains
   near the tail and averages `m=13.642`.
2. The end-to-end gap versus vLLM is split roughly half TTFT/prefill/scheduling
   and half sustained TPOT/decode.
3. Within decode, Gemma3 tail MLP and Marlin projection time dominate; unwrapped
   host-side time is small in the profiled path.
4. Dense Marlin single-kernel replacement has been falsified as the main lever.

## Next Levers

- First-token side: inspect whether Ferrum serializes prefill requests in the
  c16 ShareGPT lane and whether chunked/batched prefill can reduce TTFT without
  breaking correctness.
- Decode side: target Gemma3 tail MLP, especially gate_up/down projection
  scheduling, activation, and residual/norm boundaries, using native CUDA
  probes before product reruns.

No `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced. W2 is still not
release-grade.
