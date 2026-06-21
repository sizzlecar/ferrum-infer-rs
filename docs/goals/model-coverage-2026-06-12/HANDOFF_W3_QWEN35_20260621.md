# W3 Qwen3.5 Handoff - 2026-06-22

## Current status

- Branch: `goal/w2-w3-release-grade`.
- Latest pushed commit before this handoff update: `4a0fdbfc640b24534330c2844ba9ee3ee8f09652`.
- Draft PR to `main`: <https://github.com/sizzlecar/ferrum-infer-rs/pull/237>.
- Final W3 is not complete. There is no `MODEL_RELEASE_GRADE_W3 PASS`.
- Correctness/product-path evidence exists for real Qwen3.5 GPTQ through `ferrum run`
  and `ferrum serve`, including streaming, tool-call, structured-output, and L5
  concurrency zero-error checks.
- Performance is still the blocker. The stable c32 range remains about
  `690-695 output tok/s` on the 1x RTX 4090 Vast host, while the accepted vLLM
  c32 mean is `1708.52785 output tok/s`; the W3 80% mean target is
  `1366.82228 output tok/s`.

## Important artifacts

- W3 L2 real product PASS:
  `docs/goals/model-coverage-2026-06-12/artifacts/w3_l2_qwen35_gptq_int4_from_real_product_20260620T025952Z_75ec7e6e`.
- W3 L4/L5 CUDA PASS:
  `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_l4_l5_cuda_20260620T031726Z_ba19f2b9`.
- vLLM baseline:
  `docs/goals/model-coverage-2026-06-12/artifacts/w3_vllm_sharegpt_baseline_20260619/bench_vllm_sharegpt_sweep_100x3.json`.
- Clean decode/Marlin profile:
  `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_decode_marlin_profile_20260621T155747Z_clean`.
- Aux-overlap A/B diagnostic:
  `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_aux_overlap_ab_20260621T153830Z_dirty`.
- MoE block-size/vLLM-policy A/B diagnostic:
  `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_moe_block8_policy_ab_20260621T160946Z_dirty`.
- Earlier MoE-body graph diagnostic:
  `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_moe_body_graph_ab_20260621T144117Z_dirty`.

## Negative result: MoE-body CUDA graph

The Qwen35 MoE-body CUDA graph diagnostic was correct but slower, and the source
change was reverted.

- Correctness: `W3 QWEN35 REAL PRODUCT REPORT PASS`.
- Same-binary c32 A/B:
  - graph off: `690.5137934755809 output tok/s`;
  - graph on: `658.857379674344 output tok/s`;
  - ratio: `0.9541552766934606`.
- Conclusion: do not reintroduce `FERRUM_MOE_GRAPH` without new profiler evidence.

## Negative result: shared expert auxiliary overlap

I tested backend auxiliary stream overlap for Qwen35 shared experts and reverted
the source change.

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_aux_overlap_ab_20260621T153830Z_dirty`.
- Correctness: `W3 QWEN35 REAL PRODUCT REPORT PASS`.
- Same-binary c32 A/B:
  - aux off: `692.0108299917425 output tok/s`;
  - aux on: `695.211236472971 output tok/s`;
  - ratio: `1.0046247924779828`.
- Conclusion: the implementation was correct, but the measured gain was only
  `+0.46%`, not a material lever for the vLLM gap.

## Negative result: vLLM-style MoE block-size policy

I compared vLLM's current Marlin-MoE block-size policy in
`/Users/chejinxuan/py_ws/vllm`:

```text
for block_size_m in [8, 16, 32, 48, 64]:
    if M * topk / E / block_size_m < 0.9:
        break
```

For Qwen35 c32, `M=32`, `topk=8`, `E=256`, so vLLM selects block size `8`.
Ferrum's device-route default was `16`. I prototyped the density policy, ran
correctness, then A/B tested default block8 against forced block16 using the
same binary.

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_moe_block8_policy_ab_20260621T160946Z_dirty`.
- Correctness on default block8:
  `W3 QWEN35 REAL PRODUCT REPORT PASS`.
- n=1 c32 A/B:
  - default block8: `694.0957598710125 output tok/s`;
  - forced block16: `678.0431104857319 output tok/s`;
  - ratio: `1.023674968651744`.
- n=3 c32 A/B with `--require-ci`:
  - default block8: `627.4866287030608 +/- 259.8810517637914 output tok/s`;
  - forced block16: `682.3350864003527 +/- 15.124401845857886 output tok/s`;
  - ratio: `0.9196165362290777`.
- Conclusion: block8 was not reliable in Ferrum's current path. The source
  change was reverted; do not land the vLLM-density policy unless a cleaner
  profile explains the variance and shows a repeatable win.

## Current bottleneck read

- The clean c32 decode/Marlin diagnostic completed `64/64` requests with
  `629.4797629976711 output tok/s` and zero errors. It is diagnostic only.
- In the profile sample, observed decode batches were mostly smaller than the
  requested c32: `batch=8` appeared 16 times, `batch=32` appeared 3 times, and
  `batch=1` appeared once.
- At `batch=32`, mean layer time was about `18.4 ms`, with
  `linear_layer_sum` about `15.0 ms` and `full_layer_sum` about `3.4 ms`.
- This points more at effective decode batching and Qwen35 linear/MoE execution
  than at KV paging, pair-id routing, MoE-body graphing, auxiliary shared expert
  overlap, or block-size selection.

## Next engineering step

Do not continue blind env-flip sweeps. The next useful work is targeted
localization:

1. Instrument the scheduler/serve loop for c32 ShareGPT to record per decode
   step: requested concurrency, active decode sequences, newly admitted
   requests, completed requests, generated token counts, and batch size sent to
   Qwen35.
2. Compare that trace against vLLM's effective batch behavior for the same
   dataset and output length. If Ferrum is not keeping decode batches near 32,
   fix scheduler/admission/finish handling before touching kernels again.
3. If effective batch really is near 32 and linear layers still dominate, compare
   Ferrum's Marlin-MoE call shapes and launch counts against vLLM at the same
   step. Focus on routed-expert linear work, not shared-expert overlap.
4. Only after a profiler-backed lever is identified, make one source change and
   rerun correctness before same-hardware A/B.

## Deadline policy from user

As of `2026-06-22 00:22:24 CST`, the two-hour deadline is still active. If
there is no material performance improvement by the deadline, keep source code
clean, push the handoff/artifact update to the existing PR, and stop further
optimization attempts until the next scoped plan is accepted.
