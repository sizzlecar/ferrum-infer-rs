# W3 Qwen3.5 Handoff - 2026-06-21

## Current status

- Branch: `goal/w2-w3-release-grade`.
- Base commit under test: `690ba923b652109aea09d99ebe5702a4897b6995`.
- Final W3 is not complete. There is no `MODEL_RELEASE_GRADE_W3 PASS`.
- Correctness/product-path evidence exists for real Qwen3.5 GPTQ through `ferrum run`
  and `ferrum serve`, including streaming, tool-call, structured-output, and L5
  concurrency zero-error checks.
- Performance is still the blocker. Latest stable c32 diagnostic is about
  `690-695 output tok/s` on the 1x RTX 4090 Vast host, while the accepted vLLM
  c32 mean is `1708.52785 output tok/s`; the W3 80% mean target is about
  `1366.82228 output tok/s`.

## Important artifacts

- W3 L2 real product PASS:
  `docs/goals/model-coverage-2026-06-12/artifacts/w3_l2_qwen35_gptq_int4_from_real_product_20260620T025952Z_75ec7e6e`.
- W3 L4/L5 CUDA PASS:
  `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_l4_l5_cuda_20260620T031726Z_ba19f2b9`.
- vLLM baseline:
  `docs/goals/model-coverage-2026-06-12/artifacts/w3_vllm_sharegpt_baseline_20260619/bench_vllm_sharegpt_sweep_100x3.json`.
- Latest c32 graph diagnostic:
  `docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_moe_body_graph_ab_20260621T144117Z_dirty`.

## Negative result: MoE-body CUDA graph

I tested a Qwen35 MoE-body CUDA graph diagnostic path and reverted it.

- Correctness: `W3 QWEN35 REAL PRODUCT REPORT PASS`.
- Same-binary c32 A/B:
  - graph off: `690.5137934755809 output tok/s`;
  - graph on: `658.857379674344 output tok/s`;
  - ratio: `0.9541552766934606`.
- Config evidence:
  - graph off had `FERRUM_MOE_GRAPH=0`, source `config_file`,
    `selected_graph_mode=graph_disabled`;
  - graph on had `FERRUM_MOE_GRAPH=1`, source `config_file`,
    `selected_graph_mode=graph_clean_decode`.
- Conclusion: MoE-body graph is correct but slower for this workload. Do not
  reintroduce it without new profiler evidence.

## Current bottleneck read

- The latest detailed profile points at Qwen35 MLP/MoE work, not KV paging or
  pair-id routing.
- Current typed/default path already selects vLLM Marlin MoE device route with
  pair-id layout for the workload preset. Pair-id env flips are redundant.
- `FERRUM_MOE_GRAPH` is not the next lever after the negative A/B above.
- vLLM's relevant architecture difference is shared-expert scheduling: vLLM can
  run shared experts on an auxiliary CUDA stream and then synchronize before
  merge. Ferrum currently runs Qwen35 routed experts and shared experts
  serially through one backend context.

## Next engineering step

Do not hard-code a Qwen35-only CUDA stream hack in model code. The clean path is:

1. Add a backend-level auxiliary execution capability with default no-op behavior
   for CPU/Metal.
2. In CUDA, create an auxiliary `CudaState` with its own stream and a BLAS handle
   correctly bound to that stream, plus event synchronization:
   default stream records input-ready, aux waits, aux runs shared-expert work,
   default waits before merge.
3. Wire Qwen35 shared-expert gate/up/down through that capability only when the
   backend reports it is supported and the batch size is in the small-token
   decode range.
4. Validate with real product correctness first, then same-binary c32 A/B.

## Deadline policy from user

As of `2026-06-21 22:45:47 CST`, the user set a two-hour deadline. If there is
no substantive performance progress by `2026-06-22 00:45:47 CST`, stop
optimization attempts, clean the worktree to goal-related code only, open a PR
to `main`, and use this handoff as the progress document.
