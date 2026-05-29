# M3 80% goal — 2026-05-29 Codex c=4 threshold + pair-id routing

**Pod:** Vast `38237968`, RTX 4090, CUDA 13.0  
**Model:** `Qwen/Qwen3-30B-A3B-GPTQ-Int4`  
**Dataset:** random 256/128, 128 prompts, warmup 10  
**Rule:** correctness gate before performance claims

## Summary

This session fixed the default-path correctness issue from multi-turn chat,
closed the c=4 batching hole, validated two small MoE routing/combine wins,
and falsified one vLLM-source-inspired Marlin scheduling lever. The overall
0.80× vLLM goal is **not complete**.

| Change | Correctness | Result |
|---|---|---:|
| graph-clean default (`MOE_GRAPH=1` → `VLLM_MOE=1`) | Paris 4/4 + multi-turn smoke | fixes wrong-turn contamination |
| server threshold `8 → 4` | Paris 4/4 | c=4 `425.6 ± 36.6 tok/s` |
| pair-id vLLM MoE routing | Paris + c4 smoke | c16 `986.9 ± 10.2`, c32 `1249.5 ± 69.3` |
| pair-id combine fast path | Paris | c16 `993.8 ± 26.6`, c32 `1264.0 ± 29.4` |
| partial vLLM 0.20.2 Marlin scheduling backport | Paris | no c32 gain; reverted |
| block8 vLLM parity override | Paris/no-garbage smoke | c16/c32 regression; keep override-only |

## What changed

- `apply_moe_graph_default()` now defaults `FERRUM_VLLM_MOE=1` when
  `FERRUM_MOE_GRAPH=1` and the binary has vLLM MoE support.
- Qwen3-MoE graph capture is gated to graph-clean MoE only:
  `FERRUM_VLLM_MOE=1` and `FERRUM_MOE_HOST_ROUTE!=1`.
- The server-side MoE batched-decode default threshold is `4`, so c=4 uses
  `decode_batch_internal` instead of the slow per-item loop.
- `FERRUM_VLLM_MOE_PAIR_IDS=1` defaults on with vLLM MoE. This preserves the
  vLLM-native pair-id layout, skips the `x_packed` pre-gather, and keeps a
  `=0` escape hatch.
- The pair-id combine path uses `weighted_sum_batched_f16`, which directly
  reduces `[batch, top_k, hidden]` rows. This is a combine fast path, not a
  residual fusion.
- A partial vLLM 0.20.2 Marlin-MoE scheduling/tile backport was tested:
  `{128,64,128}` was added to auto candidates and `blocks_per_sm` selection
  was made closer to vLLM 0.20.2. It passed Paris but gave no c32 gain, so the
  code change was reverted.
- `FERRUM_MOE_BLOCK_SIZE=8` is now accepted as an explicit override, and the
  process-global vLLM-MoE `c_tmp` scratch is sized for vLLM's block8
  requirement. This is not the default path; it exists to test vLLM block-size
  parity safely.

## Correctness gates

Artifacts:

- `/workspace/m3-graph-loop/chat_default_graphclean_fix.jsonl`
- `/workspace/m3-graph-loop/paris_threshold4_default_final.log`
- `/workspace/m3-graph-loop/pair_ids_paris_min/`

Results:

- Multi-turn chat no longer answers the previous prompt after graph replay.
- Paris bisect after graph-clean + threshold fix: `A_safe`, `B_vllm_moe`,
  `C_vllm_moe_graph`, `D_vllm_moe_host_route` all `PASS`.
- Pair-id path Paris smoke produced `The capital of France is **Paris**.`
- Pair-id combine fast path Paris smoke produced `The capital of France is
  **Paris**.`
- Marlin scheduling negative-control Paris smoke produced `The capital of
  France is **Paris**.`
- Block8 smoke artifacts contain `Paris` and did not show garbage, but the
  first two prompts were truncated in Qwen3 `<think>` output because the gate
  used too few completion tokens. Treat this as a no-garbage smoke, not a
  publication-grade final-answer gate.

## Performance

Current full sweep before pair-id defaulting:

| c | Ferrum tok/s | Notes |
|---:|---:|---|
| 1 | `155.4 ± 1.0` | already near/over 0.80 depending on vLLM denominator |
| 4 | `425.6 ± 36.6` | threshold fix; pre-fix observation was ~`122 tok/s` |
| 16 | `965.8 ± 6.4` | still short under conservative vLLM N=5 denominator |
| 32 | `1205.6 ± 55.1` | still short; MoE remains dominant |

Pair-id focused N=3:

| c | no pair-id | pair-id | Δ |
|---:|---:|---:|---:|
| 16 | `965.8 ± 6.4` | `986.9 ± 10.2` | `+2.2%` |
| 32 | `1205.6 ± 55.1` | `1249.5 ± 69.3` | `+3.6%` |

c=4 pair-id screen was `434.1 tok/s` (N=1), so it did not obviously regress
the already-improved c=4 path.

Pair-id combine fast path N=3:

| c | pair-id baseline | combine fast path | Δ |
|---:|---:|---:|---:|
| 16 | `986.9 ± 10.2` | `993.8 ± 26.6` | `+0.7%` |
| 32 | `1249.5 ± 69.3` | `1264.0 ± 29.4` | `+1.2%` |

The artifact directory name says `residual`, but the code path is not a
residual fusion. The valid artifact is the robust rerun:
`/workspace/m3-graph-loop/bench_pairids_residual_c16_c32_n3_rerun2/`.
Ignore `/workspace/m3-graph-loop/bench_pairids_residual_c16_c32_n3/`; the
server received SIGINT during that run and emitted errors.

Same-pod vLLM 0.20.2 c16/c32 N=3 baseline:

| c | Ferrum current | vLLM 0.20.2 | ratio |
|---:|---:|---:|---:|
| 16 | `993.8 ± 26.6` | `1328.7 ± 44.4` | `0.75×` |
| 32 | `1264.0 ± 29.4` | `1971.8 ± 7.4` | `0.64×` |

Partial Marlin scheduling backport N=3:

| c | current fast path | scheduling backport | decision |
|---:|---:|---:|---|
| 16 | `993.8 ± 26.6` | `1000.9 ± 34.7` | within noise |
| 32 | `1264.0 ± 29.4` | `1250.4 ± 65.5` | no gain; reverted |

Block8 full-model validation N=3:

| c | current block16 fast path | block8 override | decision |
|---:|---:|---:|---|
| 16 | `993.8 ± 26.6` | `975.1 ± 3.6` | regression |
| 32 | `1264.0 ± 29.4` | `1209.5 ± 37.2` | regression |

The block8 hypothesis came from vLLM 0.20.2 source: for Qwen3 small-M MoE,
`M * topk / E / block_size` selects block size 8. Ferrum's current vendored
kernel still regresses under block8 even after sizing `c_tmp` for the vLLM
block8 requirement. Do not make block8 default or repeat block8-only testing;
revisit only as part of a full vLLM 0.20.2 Marlin template/source parity port.

Current-default c32 profile after pair-id combine:

Artifact: `/workspace/m3-graph-loop/profile_current_pairid_combine_c32/`.

Representative steady m≈30/31 lines, with graph disabled because the stage
timers synchronize:

```text
[batched-decode-prof] m=30/31 layers=48 total=16–17 ms
  dense≈2 ms
  attn_peritem≈2 ms
  moe≈10–11 ms (~64–66%)

[bucket-prof] layers=48 bk_total≈9–10 ms
  gemm1≈6.2–6.6 ms
  silu≈0.27–0.29 ms
  gemm3≈3.0–3.2 ms
  combine≈0.25 ms
```

This reconfirms the priority: gate_up/down Marlin GEMM body or a full vLLM
0.20.2 Marlin source-parity port. Combine is already too small to be a primary
lever.

Continuation notes after the pair-id combine profile:

- Added debug-only real-routing dump hooks for both packed and pair-id
  `moe_align_block_size` paths. Use `FERRUM_MOE_DUMP=1` plus
  `FERRUM_MOE_DUMP_BATCH_X_TOPK=256` to skip prefill and capture one c=32
  decode routing shape: `block_size`, `total_post_pad`, `active_blocks`, and
  `unique_experts`.
- Added `FERRUM_UNIFIED_POST_PROF=1` to the unified engine path. It prints
  `[unified-prof]` every 32 calls with total/model/decode-post timings split
  into sample, scheduler, stream, stop, and complete. This is needed because
  the current production path uses `process_batch_unified`, not the older
  `run_batch_decode` path that emits `[rbd-prof]`.
- vLLM 0.20.2 source comparison found that Ferrum's vendored
  `vllm_marlin_moe/ops.cu` differs beyond Torch-wrapper removal: upstream has
  dynamic `stages`/`is_a_8bit` shared-memory sizing, different block metadata
  sizing, an extra `{thread_k=128, thread_n=64, threads=128}` candidate, and
  a fuller `blocks_per_sm` selection for large-batch tiles. The already-tested
  partial scheduling backport did not help c=32, so do not cherry-pick these
  blindly; use the real c=32 route dump plus `[unified-prof]` before a full
  parity port.
- OpenAI-chat `vllm bench serve` is a streaming benchmark, so
  `send_stream_update` is on the hot path. A Qwen tokenizer CPU microbench of
  the current full-history decode pattern for `32 × 128` streaming updates was
  only about `0.064 s` total, or roughly `0.5 ms` per c=32 decode iteration.
  This is not large enough to explain the observed `~6 ms` gap between the
  model profile and benchmark TPOT, so streaming text decode is not the next
  primary lever unless `[unified-prof]` contradicts this.
- Vast instance `38237968` stopped while the route-dump CUDA build was running.
  Restart returned `resources_unavailable`, and renting a replacement 48GB
  RTX 4090 failed with `insufficient_credit`. No new GPU correctness or
  performance claims were made after this point.
- A follow-up short-context vLLM paged-attention v1 path was prepared locally:
  it calls vLLM `paged_attention_v1_kernel` when `max_seq_len <= 512`, with
  `FERRUM_VLLM_PAGED_ATTN_V1_SHORT=0` as the old-v2 kill switch. This is
  intended to remove the v2 reduce launch for random 256/128 decode, but it is
  not a validated win yet. Local gates passed (`cargo fmt --all -- --check`,
  `cargo check -p ferrum-cli`); GPU Paris/c32 A/B is pending because the Vast
  account currently reports negative balance and replacement rentals fail with
  `insufficient_credit`. Use `scripts/m3_attn_v1_ab.sh` when a pod is restored.
- Added `scripts/m3_route_unified_profile.sh` so the next pod run has one
  scoped command for the real c=32 MoE route shape plus unified engine timing.
  It sets `FERRUM_MOE_DUMP_BATCH_X_TOPK=$((CONCURRENCY * TOP_K))`,
  `FERRUM_UNIFIED_POST_PROF=1`, `FERRUM_BATCH_DECODE_PROF=1`,
  `FERRUM_NEXT_BATCH_PROF=1`, `FERRUM_MOE_PROFILE=1`, and
  `FERRUM_MOE_GRAPH=0` because route dumping synchronizes/copies routing
  buffers and should not run inside CUDA graph capture. The script fails fast
  if either `[MOE_DUMP:*]` or `[unified-prof]` is missing. It also writes
  `profile_summary.json` with medians for unified, iteration, and bucket
  timings.

Continuation after pod restore:

- The GPU pod was restored and tested from a dirty remote worktree intended to
  match the pushed `codex/m3-20260529-checkpoint` contents. The remote metadata
  still reports `git_head=a9dc705` because the checkout was dirty `main`, so
  treat these artifacts as restored-pod dirty-binary evidence rather than a
  clean-checkout baseline.
- Valid restored-pod route/unified profile:
  `/workspace/m3-moe-parity-lite-profile-20260529_033812/`.
  Correctness gate passed (`Paris`). The captured route shape was
  `batch_x_topk=256`, `block_size=16`, `total_post_pad=832`,
  `active_blocks=52`, `unique_experts=48`.
- The same profile shows postprocess is not the blocker: `unified-prof`
  median `total≈15.1 ms`, `model≈14.6 ms`, `decode_post≈0.34 ms`, with
  `stream≈0.23 ms` and scheduler effectively zero. The early c32 prefill waves
  were large (`~190–288 ms` mixed-prefill iterations), so TTFT/prefill remains
  a real part of the gap.
- Prompt-token-estimate prefill scheduling passed local unit test
  `cargo test -q -p ferrum-scheduler prompt_token_metadata_expands_prefill_admission`
  and GPU Paris. In the current branch it is gated behind
  `FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE=1`; the artifact below was produced by
  the dirty restored-pod binary where this candidate was still default-on.
  Stable c32 N=3 artifact:
  `/workspace/m3-prefill-est-c32-stable-20260529_034726/`.
  Result: `1288.2 ± 13.0 tok/s`, `TPOT p50=20.1 ms`, `ITL p50=15.1 ms`,
  `TTFT p50=576.0 ms`. This is a small, sub-10% delta against the prior
  `1264.0 ± 29.4` c32 row, so keep it opt-in until a same-binary direct A/B
  confirms the effect.
- Discard these restored-pod artifacts as contaminated by delayed old scripts,
  process-group kills, duplicate servers, or concurrent builds:
  `/workspace/m3-prefill-est-c32-20260529_033714/`,
  `/workspace/m3-prefill-est-c32-clean-*`,
  `/workspace/m3-prefill-est-c32-foreground*`,
  `/workspace/m3-prefill-est-c32-rerun-*`, and
  `/workspace/m3-prefill-est-c32-setsid-*`.
- `FERRUM_MAX_BATCHED_TOKENS=4096` was tested as a targeted prefill-wave
  hypothesis, not an env sweep. Paris passed, but N=1 artifact
  `/workspace/m3-mbt4096-c32-n1-20260529_035206/` measured only
  `1304.9 tok/s`; `TPOT p50` improved to `19.0 ms`, but `TTFT p50` regressed
  to `723.7 ms`. Do not make it default or repeat max-batched-token-only
  tests without new evidence.

Primary artifacts:

- `/workspace/m3-graph-loop/bench_threshold4_current_full_c1_c4_c16_c32_n3/`
- `/workspace/m3-graph-loop/bench_pair_ids_c16_c32_n3/`
- `/workspace/m3-graph-loop/bench_pair_ids_c4_n1/`
- `/workspace/m3-graph-loop/bench_pairids_residual_c16_c32_n3_rerun2/`
- `/workspace/m3-graph-loop/vllm0202_baseline_c16_c32_n3_retry/`
- `/workspace/m3-graph-loop/bench_marlin_sched_c16_c32_n3/`
- `/workspace/m3-graph-loop/block8_validation_rerun/`
- `/workspace/m3-graph-loop/block8_paris128/`
- `/workspace/m3-graph-loop/profile_current_pairid_combine_c32/`
- `/workspace/m3-moe-parity-lite-profile-20260529_033812/`
- `/workspace/m3-prefill-est-c32-stable-20260529_034726/`
- `/workspace/m3-mbt4096-c32-n1-20260529_035206/`

## Current target status

Use the ratios below as directional only until vLLM is rerun with
`n_repeats >= 5`.

- c=1 and c=4 are now the healthy cells.
- c=16 is close but not over 0.80 against same-pod vLLM 0.20.2 N=3:
  `993.8 / 1328.7 ≈ 0.75`.
- c=32 remains the real blocker: default fast path is `1264.0 / 1971.8 ≈
  0.64`; the opt-in prompt-token-estimate candidate reached
  `1288.2 / 1971.8 ≈ 0.65`, so c32 still needs roughly `+22–25%` throughput
  to clear 0.80× on this pod.

## Next lever

Do not repeat env sweeps, the partial Marlin scheduling backport, block8-only
testing, or `FERRUM_MAX_BATCHED_TOKENS=4096` as a standalone lever. The next
high-return loop should target model time directly:

1. restore a GPU pod with enough credit and 48GB-class memory if possible;
2. first run the scoped attention A/B gate if the local v1 patch is still
   present: `REPEATS=3 bash scripts/m3_attn_v1_ab.sh`; keep it only if Paris
   passes and same-pod v1 beats forced-v2;
3. add or run a scoped unified-layer profile that separates prefill and decode
   dense/attention/MoE time; the current restored profile already rules out
   scheduler/postprocess as the main gap;
4. if MoE remains dominant, prototype either a full vLLM 0.20.2 Marlin-MoE
   source-parity port behind the existing C ABI or one small-m gate_up/down
   fused kernel;
6. run Paris, then c16/c32 N=3 A/B on the same pod.
