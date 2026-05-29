# M3 80% goal — 2026-05-29 Codex c=4 threshold + pair-id routing

**Pod:** Vast `38237968`, RTX 4090, CUDA 13.0  
**Model:** `Qwen/Qwen3-30B-A3B-GPTQ-Int4`  
**Dataset:** random 256/128, 128 prompts, warmup 10  
**Rule:** correctness gate before performance claims

## Summary

This session fixed the default-path correctness issue from multi-turn chat,
closed the c=4 batching hole, validated two small MoE routing/combine wins,
validated a short-context paged-attention v1 default, and falsified multiple
low-yield or unsafe Marlin/env levers. The overall 0.80× vLLM goal is **not
complete**.

| Change | Correctness | Result |
|---|---|---:|
| graph-clean default (`MOE_GRAPH=1` → `VLLM_MOE=1`) | Paris 4/4 + multi-turn smoke | fixes wrong-turn contamination |
| server threshold `8 → 4` | Paris 4/4 | c=4 `425.6 ± 36.6 tok/s` |
| pair-id vLLM MoE routing | Paris + c4 smoke | c16 `986.9 ± 10.2`, c32 `1249.5 ± 69.3` |
| pair-id combine fast path | Paris | c16 `993.8 ± 26.6`, c32 `1264.0 ± 29.4` |
| short paged-attn v1 default | Paris on both A/B paths | c32 `1301.1 ± 25.0` vs forced-v2 `1275.2 ± 56.7` |
| current binary vs pre-binary, forced v2 | Paris all rows | c32 `1299.0 ± 48.4` vs `1244.4 ± 53.7` |
| partial vLLM 0.20.2 Marlin scheduling backport | Paris | no c32 gain; reverted |
| block8 vLLM parity override | Paris/no-garbage smoke | c16/c32 regression; keep override-only |
| broad `{128,64,128}` Marlin tile instantiation | graph-on load failed | 48GB OOM near layer 40; reverted |
| vLLM raw Marlin block8/block16 op probe | synthetic op only | no kernel-time delta; block-size-only ruled out |
| Ferrum/vLLM raw Marlin active65 op probe | synthetic op only | gate/up and down match within noise; down-source-parity ruled out |
| graph-on production c32 profile | Paris | graph/post/scheduler not a 25% standalone gap |
| prompt-token-estimate scheduler A/B | Paris both rows | +2.8% throughput but TTFT worse; keep opt-in |
| split mixed prefill/decode | Paris both rows | c32 N=3 `+1.46%`; reverted as noise |
| unified mixed layer-loop graph | Paris | c32 N=1 `-3.82%`; reverted |
| unified GPU greedy argmax | Paris both rows | c32 N=1 `+0.97%`; keep as cleanup, not a main lever |
| prefill-first admission | Paris both rows | c32 N=1 `+3.77%`, but TTFT worse; prompt-est combo `-0.58%`; keep opt-in |
| unified greedy prefill token handling | Paris | correctness cleanup; current-default c32 N=1 `1299.7 tok/s` |

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
- `FERRUM_VLLM_PAGED_ATTN_V1_SHORT` now defaults on; `=0` forces the old v2
  path. This removes the v2 reduce launch for short-context decode where
  `max_seq_len <= 512`.
- The MoE Marlin config logger now supports
  `FERRUM_VLLM_MOE_LOG_CONFIG_LIMIT` and
  `FERRUM_VLLM_MOE_LOG_CONFIG_MIN_PAIRS`, so profiling can skip early small-M
  calls and capture the real c32 decode config.
- `FERRUM_MOE_LARGE_M_BLOCK_SIZE` plus `FERRUM_MOE_LARGE_M_MIN_PAIRS` is an
  opt-in large-M device-route experiment hook. The default c32 decode block
  size remains unchanged.

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
- Short paged-attention v1 A/B produced `Paris` for both v1-short and forced-v2
  paths.
- Current-binary vs pre-binary A/B rows for c=1/4/16/32 all passed the Paris
  gate before throughput was counted.

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

Restored-pod final current-binary vs pre-binary A/B, with attention forced to
old v2 (`FERRUM_VLLM_PAGED_ATTN_V1_SHORT=0`) to avoid mixing in the new
attention default:

| c | current binary | pre binary | decision |
|---:|---:|---:|---|
| 1 | `161.9 ± 1.2` | `160.6 ± 0.8` | flat / slight positive |
| 4 | `424.2 ± 14.1` | `428.0 ± 35.1` | flat |
| 16 | `1001.5 ± 21.8` | `999.9 ± 32.0` | flat |
| 32 | `1299.0 ± 48.4` | `1244.4 ± 53.7` | `+4.4%`, small but real |

Artifacts:

- c32: `/workspace/m3-moe-parity-lite-binary-ab-trimmed-20260529_043153/`
- c16: `/workspace/m3-moe-parity-lite-binary-ab-c16-20260529_043419/`
- c4: `/workspace/m3-moe-parity-lite-binary-ab-c4-20260529_043830/`
- c1: `/workspace/m3-moe-parity-lite-binary-ab-c1-20260529_044647/`

The short-attention v1 A/B is a separate same-binary effect:
`/workspace/m3-attn-v1-ab-codex-20260529_040601/` measured c32
`1301.1 ± 25.0` for v1-short versus `1275.2 ± 56.7` for forced-v2, a
`+2.0%` directional attention-path win. There is still no clean final
combined-default c32 N=3 row, so use the `1299.0 ± 48.4` forced-v2 row as the
conservative current-binary baseline and the v1 result only as A/B evidence
for the attention switch.

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
  performance claims were made during this outage window; later claims below
  come from the restored pod and explicit artifact directories.
- A follow-up short-context vLLM paged-attention v1 path was prepared locally:
  it calls vLLM `paged_attention_v1_kernel` when `max_seq_len <= 512`, with
  `FERRUM_VLLM_PAGED_ATTN_V1_SHORT=0` as the old-v2 kill switch. This is
  intended to remove the v2 reduce launch for random 256/128 decode. Local
  gates passed (`cargo fmt --all -- --check`, `cargo check -p ferrum-cli`);
  GPU validation was initially blocked by credit, then completed after the pod
  was restored.
- Added `scripts/m3_route_unified_profile.sh` so the next pod run has one
  scoped command for the real c=32 MoE route shape plus unified engine timing.
  It sets `FERRUM_MOE_DUMP_BATCH_X_TOPK=$((CONCURRENCY * TOP_K))`,
  `FERRUM_VLLM_MOE_LOG_CONFIG=1`,
  `FERRUM_VLLM_MOE_LOG_CONFIG_MIN_PAIRS=$((CONCURRENCY * TOP_K))`,
  `FERRUM_VLLM_MOE_LOG_CONFIG_MAX_PAIRS=$((CONCURRENCY * TOP_K))`,
  `FERRUM_UNIFIED_POST_PROF=1`, `FERRUM_DECODE_OP_PROFILE=1`,
  `FERRUM_BATCH_DECODE_PROF=1`, `FERRUM_NEXT_BATCH_PROF=1`, `FERRUM_MOE_PROFILE=1`, and
  `FERRUM_MOE_GRAPH=0` because route dumping synchronizes/copies routing
  buffers and should not run inside CUDA graph capture. The script fails fast
  if `[MOE_DUMP:*]`, `[vllm-moe-config]`, `[unified-prof]`, or a decode-stage
  profile is missing. It also writes `profile_summary.json` with medians for
  unified, iteration, bucket, and Marlin config fields.

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
- Short-context vLLM paged-attention v1 was validated as a same-binary A/B on
  the restored dirty remote binary after fixing the default switch to match the
  intended semantics (`default on`, `FERRUM_VLLM_PAGED_ATTN_V1_SHORT=0` forces
  old v2). Artifact:
  `/workspace/m3-attn-v1-ab-codex-20260529_040601/`. Both paths passed Paris.
  `v1_short` c32 N=3 measured `1301.1 ± 25.0 tok/s`, `TPOT p50=19.7 ms`;
  `v2_forced` measured `1275.2 ± 56.7 tok/s`, `TPOT p50=20.6 ms`. This is a
  small `+2.0%` directional win and supports keeping the short-v1 default, but
  the absolute `1301 tok/s` row is not a clean current-default baseline because
  the remote dirty binary still had prompt-token-estimate scheduling default-on.
  Treat this artifact as attention-path A/B evidence only.
- Added a targeted large-M MoE block-size experiment hook, not a new default:
  `FERRUM_MOE_LARGE_M_BLOCK_SIZE=32/48/64` applies only on the device-route
  path when `batch * top_k >= FERRUM_MOE_LARGE_M_MIN_PAIRS` (default threshold
  `1024`). This is intended to test whether prefill/unified large-M work can use
  the wider Marlin tile without repeating the known-bad c32 sparse-decode
  `FERRUM_MOE_BLOCK_SIZE=64` behavior. Local tests cover thresholding and the
  global `FERRUM_MOE_BLOCK_SIZE` override precedence; there is no GPU
  performance claim yet.
- Extended `FERRUM_VLLM_MOE_LOG_CONFIG=1` with
  `FERRUM_VLLM_MOE_LOG_CONFIG_LIMIT` and
  `FERRUM_VLLM_MOE_LOG_CONFIG_MIN_PAIRS` so the next clean GPU build can skip
  early small-M Marlin calls and capture the actual c32 decode config. The
  restored dirty-pod probe `/workspace/m3-moe-config-probe-20260529_044441/`
  passed Paris and captured route shape `batch_x_topk=256`,
  `total_post_pad=800`, `active_blocks=50`, `unique_experts=44`, but the old
  fixed log cap only recorded smaller `batch_x_topk=104` calls selecting
  `thread_k=64`, `thread_n=128`, `threads=128`, `blocks_per_sm=3`. Treat that
  run as debug evidence only, not throughput data.
- A broad Marlin-MoE `{thread_k=128, thread_n=64, threads=128}` tile
  instantiation was tested and reverted. The larger CUDA module/binary loaded
  graph-on into OOM near layer 40 on the 48GB 4090, before a valid Paris or
  throughput result. This is a hard safety signal: do not add broad Marlin
  template coverage without measuring binary/module/graph memory pressure.
- Final restored-pod current-binary vs pre-binary A/B forced attention old-v2
  (`FERRUM_VLLM_PAGED_ATTN_V1_SHORT=0`) to isolate the non-attention code
  state. All rows passed Paris. c32 artifact
  `/workspace/m3-moe-parity-lite-binary-ab-trimmed-20260529_043153/` measured
  current `1299.0 ± 48.4 tok/s` vs pre `1244.4 ± 53.7` (`+4.4%`). c16
  `/workspace/m3-moe-parity-lite-binary-ab-c16-20260529_043419/` measured
  `1001.5 ± 21.8` vs `999.9 ± 32.0`; c4
  `/workspace/m3-moe-parity-lite-binary-ab-c4-20260529_043830/` measured
  `424.2 ± 14.1` vs `428.0 ± 35.1`; c1
  `/workspace/m3-moe-parity-lite-binary-ab-c1-20260529_044647/` measured
  `161.9 ± 1.2` vs `160.6 ± 0.8`. Treat lower-concurrency rows as flat and
  c32 as the only meaningful small win.
- Clean-checkout c=32 profile after decode-shape filtering:
  `/workspace/m3-route-unified-layer-relaxed-clean-20260529_060400/`
  (`c55e9c4`, current branch clean binary). Paris passed. This is a scoped
  profiling artifact with graph disabled and sync timers on, not a throughput
  baseline. Route shape: `batch_x_topk=256`, `block_size=16`,
  `total_post_pad=1040`, `active_blocks=65`, `unique_experts=61`.
  Marlin config now correctly records only decode rows:
  gate/up `prob_m=32 prob_n=1536 prob_k=2048 top_k=8` and down
  `prob_m=256 prob_n=2048 prob_k=768 top_k=1`; both select `thread_k=64`,
  `thread_n=128`, `threads=128`, `blocks_per_sm=3`.
- Filtering the same clean profile to full `m=32` decode gives
  `unified-prof` median `total≈15.9 ms`, `model≈15.5 ms`,
  `decode_post≈0.35 ms`. `batched-decode-prof` median is `total≈16 ms` with
  MoE `≈10 ms` (`~62.5%`), attention `≈2 ms`, dense `≈2 ms`, and other
  `≈1 ms`. Paired `bucket-prof` median is `bk_total≈9 ms`, dominated by
  `gemm1≈5.86 ms` and `gemm3≈2.89 ms` (`~97%` of the bucket), while
  `combine≈0.25 ms`. This clean profile closes the attribution loop:
  scheduler/postprocess/streaming/combine/route are not the primary c=32 gap;
  the remaining high-return lever is the vLLM-Marlin MoE GEMM body/source
  parity or a new small-M fused MoE kernel.
- Targeted Marlin thread-config A/B tested whether the real decode shape should
  override auto's `thread_k=64`, `thread_n=128`, `blocks_per_sm=3` selection
  with `FERRUM_VLLM_MOE_THREAD_K=128`,
  `FERRUM_VLLM_MOE_THREAD_N=128`, and
  `FERRUM_VLLM_MOE_BLOCKS_PER_SM=1`. Artifact:
  `/workspace/m3-marlin-thread-ab-n3-20260529_060835/`. Both rows passed
  Paris. N=1 initially looked positive (`+5.7%`), but the required N=3 A/B
  reversed it: auto measured `1257.2 ± 50.2 tok/s`; forced `128x128,bps=1`
  measured `1235.4 ± 35.1 tok/s` (`-1.7%`). Do not turn this into a default or
  shape guard; the next MoE lever still needs source-parity/kernel work, not
  this scheduling override.
- vLLM 0.20.2 source inspection confirmed that the single-GPU DP=1 path uses the
  standard `MarlinExperts`, not the `BatchedMarlinExperts` DeepEP/NIXL
  activation format. A raw-op probe then tested vLLM's block-size selector
  directly with `/workspace/vllm-venv/bin/python` and
  `moe_wna16_marlin_gemm` at M=32/topk=8. Artifact:
  `/workspace/m3-admin/vllm-raw-marlin-block8-block16-20260529_0624.json`.
  This was a synthetic random route (`unique_experts=114`, `blocks=114`), not a
  full-model throughput row, but it is a valid block8-vs-block16 comparison on
  the same route.
- Raw-op result: gate/up block8 `198.949 µs` vs block16 `200.144 µs`; down
  block8 `104.090 µs` vs block16 `104.322 µs`. Block8 halves the `npp` scratch
  requirement but did not reduce active block count or kernel time for this
  sparse c32-style route. This reinforces the earlier Ferrum full-model block8
  regression: do not spend another loop on block-size-only work.
- Process hygiene checkpoint: stale `m3build128x64d` / `ssh_tmux` 128x64 build
  jobs were killed after they repeatedly dirtied
  `crates/ferrum-kernels/kernels/vllm_marlin_moe/{ops.cu,kernel_instantiations.cu}`.
  The remote diff was saved to
  `/workspace/m3-admin/unknown-128x64-dirty-20260529_0620.diff` and stashed as
  `codex-save-unknown-128x64-dirty-20260529_0620`. A final audit showed
  `/workspace/ferrum-codex-clean` clean on `codex/m3-20260529-checkpoint`, no
  matching cargo/nvcc/ferrum/vLLM/tmux processes, and RTX 4090 GPU usage
  `0%`, `1 MiB / 49140 MiB`. The matching local 128x64 edits were preserved in
  stash `codex-save-local-unknown-128x64-dirty-20260529_0620` so they cannot
  contaminate this checkpoint.
- Graph pre-sync removal was tested as a deliberately small host-barrier
  lever and reverted. The candidate skipped the unconditional
  `B::sync(&mut ctx)` before graph replay/eager layer-loop while still syncing
  before graph capture. Paris passed in
  `/workspace/m3-presync-skip-paris-20260529_074417`, but same-binary c32 N=3
  `/workspace/m3-presync-ab-20260529_074643/` measured skip-sync
  `1249.0 ± 27.3 tok/s` vs forced old pre-sync `1252.6 ± 81.4 tok/s`
  (`-0.28%`). This is noise and directionally negative; do not keep or repeat
  graph-pre-sync removal as a primary lever.
- vLLM 0.20.2 DP + two-tile split-K Marlin-MoE scheduler parity was tested as
  a scoped kernel-body candidate and reverted. The patch only changed
  `vllm_marlin_moe/marlin_template.h`, preserving Ferrum's EP handling and
  local scale/zero-point pointer layout. Static slice accounting after the
  patch showed limited headroom: for the clean c32 `active_blocks=65` route,
  gate/up kept the same total slice count while down only dropped from 1324 to
  1312 slices. The build completed in `23m51s`, Paris passed all cells in
  `/workspace/m3-dp2sk-paris-20260529_083316`, but the single c32 smoke
  `/workspace/m3-dp2sk-c32-smoke-20260529_083702/` measured only
  `1262.0 tok/s`, `TPOT p50=21.84 ms`, `ITL p50=15.25 ms`,
  `TTFT p50=379 ms`. This is below the current c32 range and not worth N=3
  A/B. Do not repeat scheduler-only parity as a primary lever; use full
  vLLM-Marlin source parity or a small-M fused MoE kernel instead.
- A high-return raw-op localization compared vLLM 0.20.2 Marlin-MoE against
  Ferrum's direct C ABI on the same active65 route class. The route was
  `batch_x_topk=256`, `block_size=16`, `total_post_pad=1040`,
  `active_blocks=65`, `unique_experts=61`, matching
  `/workspace/m3-route-unified-layer-relaxed-clean-20260529_060400/`. The
  corrected vLLM probe used `mul_topk_weights=false`, matching Ferrum's down
  GEMM plus separate pair-id combine path.
- Raw-op result: vLLM artifact
  `/workspace/m3-admin/vllm-raw-marlin-active65-mulfalse-20260529_084848.json`
  measured gate/up `107.136 µs` and down `32.108 µs`. Ferrum artifact
  `/workspace/m3-admin/ferrum-raw-marlin-active65-20260529_085145.json`
  measured gate/up `106.936 µs` and down `32.210 µs`. This invalidates the
  earlier down-source-parity hypothesis: the full-model graph-off bucket profile
  reported down around `60 µs/layer` because the profiler includes launch/sync
  and timer overhead around the kernel body. Do not target Marlin source/body
  parity again unless a fresh raw-op or graph-on profile shows kernel divergence.
- Graph-on production c32 profile then checked whether graph replay, model
  post-process, or scheduler overhead explains the end-to-end gap. Artifact:
  `/workspace/m3-graph-prod-profile-20260529_085701/`. Paris passed; c32 N=1
  measured `1217.6 tok/s`, `TPOT p50=22.7 ms`, `ITL p50=15.0 ms`,
  `TTFT p50=383 ms`. Steady full-batch snippets showed `model≈12.5–14.6 ms`,
  `decode_post≈0.15–0.8 ms`, scheduler `≈20–44 µs`, and graph replay
  `upload≈0.10–0.13 ms`, `launch≈0.20–0.26 ms`. This rules out graph
  upload/launch, decode post-process, and scheduler call overhead as standalone
  25% levers.
- Prompt-token-estimate scheduler A/B tested the concrete prefill-admission
  mechanism exposed by that profile: without the env, new 256-token random
  prompts are budgeted as `prefill_chunk_size=512`, so only four fit in a
  2048-token first prefill batch. Artifact:
  `/workspace/m3-sched-prompt-est-ab-20260529_090018/`. Both rows passed Paris.
  c32 N=3 default measured `1278.6 tok/s`, `TPOT p50=22.0 ms`,
  `ITL p50=15.0 ms`, `TTFT p50=374 ms`; `FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE=1`
  measured `1314.0 tok/s` (`+2.8%`), `TPOT p50=19.4 ms`, `ITL p50=15.1 ms`,
  `TTFT p50=603 ms`. This is below the default-worthy gain threshold and
  worsens TTFT, so keep the feature opt-in and do not pursue scheduler env
  sweeps without a new occupancy profile.
- Unified chunked prefill was tested as a targeted vLLM-scheduler parity
  candidate and reverted. The patch added opt-in
  `FERRUM_UNIFIED_CHUNKED_PREFILL=64` so unified prefill requests were scheduled
  and executed in 64-token chunks with non-final chunks skipping sampling.
  This was meant to reduce mixed prefill/decode ITL tails, not to change Marlin
  GEMM. The same-binary c32 N=1 artifact
  `/workspace/m3-unified-chunked-prefill-ab-20260529_094117/` passed Paris for
  both rows. `chunk64` measured `1219.9 tok/s`, `TPOT p50=18.27 ms`,
  `ITL p50=15.05 ms`, `ITL p95=45.2 ms`, `TTFT p50=991 ms`; default measured
  `1232.6 tok/s`, `TPOT p50=22.62 ms`, `ITL p50=15.24 ms`,
  `ITL p95=98.7 ms`, `TTFT p50=391 ms`. The candidate improved tail metrics
  but lost `1.03%` throughput and nearly tripled TTFT, so it failed the c32
  smoke gate and was not promoted to N=3. Do not repeat chunk64 or chunk-size
  sweeps unless a new admission/fill design can cap the TTFT cost.
- A graph-on timeline and graph-shape prewarm check narrowed the current c32
  tail issue. Artifact `/workspace/m3-graphon-timeline-c32-20260529_095135/`
  passed Paris and measured Ferrum c32 N=1 `1212.4 tok/s`, `TPOT p50=23.58 ms`,
  `ITL p50=15.19 ms`, `ITL p95=95.15 ms`, `TTFT p50=351 ms`. Same-pod vLLM
  N=1 in the same artifact measured `1983.5 tok/s`, `ITL p95=14.01 ms`. Graph
  replay upload/launch was only hundreds of microseconds, while large Ferrum
  `iter-prof` spikes were mixed/full prefill batches. External prewarm of
  c30/c31/c32 graph shapes in `/workspace/m3-graph-shape-prewarm-c32-20260529_095611/`
  improved the single N=1 throughput to `1267.8 tok/s` but left `ITL p95=96.84 ms`,
  so lazy shape capture is not the primary c32 tail explanation.
- Mixed-prefill layer profiling showed a concrete attention gap. With
  vLLM-layout paged attention enabled, `/workspace/m3-mixed-prefill-layer-prof-20260529_095945/`
  measured mixed rows around `m≈832-876` with median layer-stage totals
  `attn≈61.5 ms`, `moe≈36.5 ms`, `qkv≈6.7 ms`, `o_proj≈5.8 ms` under sync
  profiling. The legacy-layout comparison
  `/workspace/m3-mixed-prefill-legacy-varlen-prof-20260529_100212/` measured
  `attn≈46.5 ms` and `moe≈36.3 ms` on the same style of run. This points at
  the vLLM-layout varlen prefill attention bridge as a real tail contributor,
  but it does not by itself prove that legacy layout should be default because
  VPA remains useful for decode.
- vLLM-layout varlen split-K was implemented as an opt-in kernel experiment and
  then reverted after failing the c32 smoke gate. The candidate added
  `FERRUM_VLLM_VARLEN_SPLIT_K=1`, a vLLM-layout split-K phase1 kernel, and
  reused the existing varlen split-K reduce. CUDA release build took `22m17s`
  because `ferrum-kernels/build.rs` recompiles unrelated Marlin/MoE-Marlin
  template libraries when one attention `.cu` changes. Same-binary c32 N=1
  artifact `/workspace/m3-vllm-varlen-splitk-ab-20260529_183142/` passed Paris
  for both rows. `splitk` measured `1164.6 tok/s`, `TPOT p50=24.19 ms`,
  `ITL p50=15.01 ms`, `ITL p95=104.57 ms`, `TTFT p50=419 ms`; default measured
  `1239.8 tok/s`, `TPOT p50=22.61 ms`, `ITL p50=15.07 ms`,
  `ITL p95=96.68 ms`, `TTFT p50=383 ms`. Split-K was `-6.06%`, so do not
  reintroduce that two-phase path without a new kernel design. The next
  attention attempt should be a proper tiled vLLM/FlashAttention-style varlen
  prefill kernel or a microbench-proven equivalent, not another split-K wrapper.
- Added `scripts/m3_vllm_varlen_splitk_ab.sh` to make this negative control
  reproducible. It runs split-K first so a candidate Paris failure stops before
  spending time on the default row, then runs the same-binary default baseline
  and prints throughput/TPOT/ITL/TTFT summaries. It is a scoped experiment
  script, not a default benchmark sweep.
- Build-cycle efficiency is now a concrete blocker for CUDA iteration speed.
  The observed attention-only edit rebuilt unrelated `vllm_marlin/*` and
  `vllm_marlin_moe/*` static libraries before linking. A near-term engineering
  fix is to add build-script stamp/artifact caching or split those static
  libraries so attention-only PTX changes do not pay the Marlin template cost.
- Build-cache checkpoint completed after the split-K negative control.
  `crates/ferrum-kernels/build.rs` now writes dependency/flag signatures for
  the Marlin, vLLM-Marlin, vLLM-MoE-Marlin, and vLLM paged-attention static
  libraries before invoking nvcc. The first pod build paid the one-time stamp
  generation cost (`22m21s`). After that, touching only
  `kernels/paged_varlen_attention_vllm.cu` rebuilt the release `ferrum-cli`
  binary in `3m13s`, and the final content-hash verbose verification log
  `/workspace/m3-build-cache-contenthash-final-vv-20260529.log` completed in
  `194s` with cache hits for all four static libraries. Process audit during
  the rebuild showed no Marlin/MoE-Marlin `nvcc`/`cicc`/`ptxas` work. This is
  not a throughput claim; it removes the worst unrelated CUDA rebuild cost
  before the next tiled varlen prefill-attention kernel experiment.
- The first post-cache attention experiment was a negative control. A scoped
  `FERRUM_VLLM_VARLEN_TILED_V=1` candidate changed only the weighted-V phase of
  `paged_varlen_attn_vllm_f16`: after the existing QK/softmax work it loaded
  vLLM-layout V as 16-slot tiles and reduced per head dimension from shared
  memory. The candidate built in `196s` with the new build-cache path and
  passed Paris, but same-binary c32 N=1 artifact
  `/workspace/m3-vllm-varlen-tiled-v-ab-20260529_115504/` regressed. Tiled-V
  measured `1165.0 tok/s`, `TPOT p50=23.88 ms`, `ITL p95=133.58 ms`,
  `TTFT p50=458 ms`; default measured `1219.6 tok/s`, `TPOT p50=22.83 ms`,
  `ITL p95=120.40 ms`, `TTFT p50=392 ms`. The code was reverted. Do not repeat
  V-stage-only tiling; it adds shared-memory/barrier overhead without fixing the
  real QK/V reuse problem. A future attention lever needs a full Q/K/V tiled
  FlashAttention-style varlen design or a standalone microbench showing the
  intended kernel wins before full-model testing.
- A clean HEAD graph-on production profile after the tiled-V revert confirmed
  the same end-to-end shape. Artifact
  `/workspace/m3-graphon-prod-profile-8ef71ce-20260529_120645/` passed Paris
  and measured c32 N=1 `1255.7 tok/s`, `TPOT p50=22.0 ms`,
  `ITL p95=127.0 ms`, `TTFT p50=393 ms`. Graph replay median was
  `total≈13.38 ms`, with upload+launch only `≈0.32 ms`; sampled full-batch
  model calls stayed around `12–16 ms`. This reconfirms that graph
  upload/launch, decode postprocess, and scheduler call overhead are not
  standalone 25% gaps.
- A split mixed-prefill/decode candidate was tested and reverted. The opt-in
  `FERRUM_SPLIT_MIXED_PREFILL_DECODE=1` routed mixed batches through the
  existing pure-prefill plus pure-decode graph paths instead of one Qwen
  unified mixed forward. Paris passed. N=1 artifact
  `/workspace/m3-split-mixed-ab-20260529_121843/` showed split `1250.8` vs
  default `1221.5 tok/s` (`+2.4%`), but N=3 artifact
  `/workspace/m3-split-mixed-ab-n3-20260529_122047/` measured default
  `1256.3 ± 60.4` and split `1274.7 ± 33.9 tok/s` (`+1.46%`, overlapping
  noise). Do not repeat split-only mixed-batch work without a new profile.
- A unified mixed/prefill CUDA graph candidate was tested and reverted. The
  opt-in `FERRUM_UNIFIED_MIXED_GRAPH=1` captured only the Qwen unified
  layer loop; index uploads, embedding, final norm/lm-head, and readback
  stayed eager. Paris passed, but c32 N=1 artifact
  `/workspace/m3-unified-mixed-graph-ab-20260529_123549/` regressed: default
  measured `1250.4 tok/s`, `TPOT p50=22.20 ms`, `ITL p95=98.35 ms`,
  `TTFT p50=390 ms`; candidate measured `1202.7 tok/s`, `TPOT p50=22.93 ms`,
  `ITL p95=106.34 ms`, `TTFT p50=412 ms` (`-3.82%`). Do not repeat mixed
  layer-loop graph capture unless a shape-stability profile first proves
  repeated mixed shapes.
- A simple Q-tiled vLLM-layout varlen attention candidate is correctness-safe
  but not a meaningful full-model lever. Standalone CUDA microbench
  `scripts/microbenches/varlen_vllm_tiled_q_perf.cu` showed `tiled_q4`
  materially faster than the current one-query-per-block bridge on synthetic
  prefill/mixed shapes: `prefill_4x256` `719.8 -> 533.7 us` (`+34.9%`),
  `mixed_3x256_4x1` `518.7 -> 421.1 us` (`+23.2%`), and `prefill_4x512`
  `2679.6 -> 1969.3 us` (`+36.1%`), with zero max_abs_err. The opt-in
  production path `FERRUM_VLLM_VARLEN_TILED_Q4=1` and reproducible
  `scripts/m3_vllm_varlen_tiled_q4_ab.sh` were committed through Git
  (`6a40de0`, `73b1f58`) and validated on the pod from a clean checkout.
  Release build took `3m14s` with the build-cache path. Both rows passed Paris
  in `/workspace/m3-vllm-varlen-tiled-q4-ab-20260529_git_n1/`; c32 N=1 measured
  tiled-Q4 `1251.1 tok/s`, `TPOT p50=22.72 ms`, `ITL p95=96.91 ms`,
  `TTFT p50=360 ms` versus default `1246.0 tok/s`, `TPOT p50=22.33 ms`,
  `ITL p95=128.86 ms`, `TTFT p50=397 ms` (`+0.41%`). Do not default it or
  spend N=3 time on this simple Q-only tiling without a new profile showing
  prefill attention dominates the current end-to-end run.
- Added `scripts/microbenches/vllm_flash_attn_varlen_probe.py` and ran it
  from the clean Git checkout on the restored pod using
  `/workspace/vllm-venv/bin/python`. Artifact:
  `/workspace/m3-vllm-fa2-varlen-probe-20260529_git.log`. vLLM 0.20.2
  `flash_attn_varlen_func` on the same Qwen3 paged-varlen shapes measured
  `prefill_4x256` `40.35 us`, `mixed_3x256_4x1` `47.95 us`, and
  `prefill_4x512` `109.47 us`. Compared with Ferrum's simple
  vLLM-layout varlen bridge and the q4 microbench (`~422-1970 us` for the
  same synthetic shapes), this confirms the large attention gap is kernel
  architecture, not q-only/v-only tiling. The next attention lever should be a
  real FlashAttention-style paged-varlen port or an FA-compatible KV path; it
  is not a direct function swap because Ferrum's vLLM paged-decode K/V layout
  is not the FA paged layout used by vLLM prefill.
- The unified path now uses GPU greedy argmax when `FERRUM_GREEDY_ARGMAX=1`.
  Previously the unified decode path still read full logits back to host even
  in greedy mode; `FERRUM_UNIFIED_GREEDY_ARGMAX=0` keeps an escape hatch for the
  old behavior. Clean Git-flow validation from commit `b81866c` passed Paris for
  both rows in `/workspace/m3-unified-greedy-ab-20260529_git_n1/`. c32 N=1
  measured unified greedy `1247.6 tok/s`, `TPOT p50=22.54 ms`,
  `ITL p95=83.67 ms`, `TTFT p50=295 ms` versus old full-logits readback
  `1235.6 tok/s`, `TPOT p50=22.45 ms`, `ITL p95=128.86 ms`, `TTFT p50=393 ms`
  (`+0.97%`). Keep the cleanup, but do not spend N=3 GPU time on it unless a
  later profile shows readback dominating again.
- Follow-up correctness fix: the engine now treats unified greedy prefill
  results the same way as decode results when the model returns a single f32
  token-id sentinel, and prefix-cache stores are skipped when prefix cache is
  disabled or the result is only that sentinel. Local gates passed
  `cargo fmt --all -- --check`, `cargo check -q -p ferrum-engine`, and
  `cargo test -q -p ferrum-engine --test continuous_batch_test`. Remote build
  from Git commit `2b1e2e8` took `2m57s`. Smoke artifact
  `/workspace/m3-greedy-prefill-fix-smoke-20260529_git_n1/` passed Paris and
  measured current-default c32 N=1 `1299.7 tok/s`, `TPOT p50=21.78 ms`,
  `ITL p95=85.19 ms`, `TTFT p50=325 ms`. Treat this as correctness cleanup and
  current-default smoke only, not an N=3 performance claim.
- A scheduler prefill-first admission experiment was added as an opt-in only
  lever. `FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE=32` skips early decode
  scheduling while there are waiting/prefill requests and the active decode
  cohort is below 32, trying to reduce the c32 first-wave mixed
  prefill+decode spike. It passed local scheduler tests and remote release
  build took `2m56s` without nvcc. Clean Git-flow c32 N=1
  `/workspace/m3-prefill-first-ab-20260529_git_n1/` passed Paris for both rows:
  prefill-first measured `1287.1 tok/s`, `TPOT p50=20.47 ms`,
  `ITL p95=22.93 ms`, `TTFT p50=466 ms`; default measured `1240.4 tok/s`,
  `TPOT p50=22.64 ms`, `ITL p95=82.92 ms`, `TTFT p50=320 ms` (`+3.77%`).
  This hit the tail spike but missed the `+5%` continuation bar and worsened
  TTFT. With prompt-token estimate enabled on both rows,
  `/workspace/m3-prefill-first-prompt-est-ab-20260529_git_n1/` regressed
  slightly: prefill-first `1267.0 tok/s` versus default `1274.4 tok/s`
  (`-0.58%`). Do not default it or spend N=3 time without a stronger signal.
- Clean Git-flow graph-on runtime profile
  `/workspace/m3-graph-runtime-profile-20260529_git_n1/` passed Paris and
  measured c32 N=1 `1309.3 tok/s`, `TPOT p50=21.17 ms`, `ITL p50=15.12 ms`,
  `ITL p95=117.9 ms`, `TTFT p50=360 ms`. Full-ish `items>=31` snippets had
  iteration median `14.9 ms`, model median `14.0 ms`, decode-post median
  `0.61 ms`; graph replay median was `13.16 ms` with upload+launch only
  `~0.35 ms`. This rules out graph upload/launch, scheduler calls, and
  postprocess as the standalone c32 20%+ throughput gap. The visible gap is
  fill/mixed-prefill tail behavior plus remaining model work.
- Active-decode-only prefill chunking was added as an opt-in experiment:
  `FERRUM_ACTIVE_DECODE_PREFILL_CHUNK=64` keeps initial empty-queue prefills
  full-size but chunks prefills admitted while decode is already active. Local
  gates passed: `cargo fmt --all -- --check`, `git diff --check`,
  `cargo check -q -p ferrum-engine`, `cargo test -q -p ferrum-scheduler`,
  `cargo test -q -p ferrum-engine --test continuous_batch_test`, and
  `cargo test -q -p ferrum-engine --test chunked_prefill_test`. Remote Rust
  release build took `2m56s` with no nvcc. Same-binary c32 N=1
  `/workspace/m3-active-prefill-chunk-ab-20260529_git_n1/` passed Paris for
  both rows: active chunk measured `1305.0 tok/s`, `TPOT p50=15.87 ms`,
  `ITL p50=14.60 ms`, `ITL p95=31.03 ms`, `TTFT p50=1065 ms`; default measured
  `1304.3 tok/s`, `TPOT p50=21.19 ms`, `ITL p50=15.04 ms`,
  `ITL p95=116.11 ms`, `TTFT p50=362 ms`. This is tail-latency evidence, not
  a throughput lever: throughput was flat and TTFT worsened badly. Keep it
  opt-in only; do not default or N=3 it for the M3 throughput goal unless a new
  admission policy bounds TTFT and improves tok/s.

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
- `/workspace/m3-moe-config-probe-20260529_044441/`
- `/workspace/m3-attn-v1-ab-codex-20260529_040601/`
- `/workspace/m3-moe-parity-lite-binary-ab-trimmed-20260529_043153/`
- `/workspace/m3-moe-parity-lite-binary-ab-c16-20260529_043419/`
- `/workspace/m3-moe-parity-lite-binary-ab-c4-20260529_043830/`
- `/workspace/m3-moe-parity-lite-binary-ab-c1-20260529_044647/`
- `/workspace/m3-route-unified-layer-relaxed-clean-20260529_060400/`
- `/workspace/m3-marlin-thread-ab-n3-20260529_060835/`
- `/workspace/m3-admin/vllm-raw-marlin-block8-block16-20260529_0624.json`
- `/workspace/m3-presync-skip-paris-20260529_074417`
- `/workspace/m3-presync-ab-20260529_074643/`
- `/workspace/m3-dp2sk-paris-20260529_083316`
- `/workspace/m3-dp2sk-c32-smoke-20260529_083702/`
- `/workspace/m3-admin/vllm-raw-marlin-active65-20260529_084340.json`
- `/workspace/m3-admin/vllm-raw-marlin-active65-mulfalse-20260529_084848.json`
- `/workspace/m3-admin/ferrum-raw-marlin-active65-20260529_085145.json`
- `/workspace/m3-graph-prod-profile-20260529_085701/`
- `/workspace/m3-sched-prompt-est-ab-20260529_090018/`
- `/workspace/m3-unified-chunked-prefill-ab-20260529_094117/`
- `/workspace/m3-graphon-timeline-c32-20260529_095135/`
- `/workspace/m3-graph-shape-prewarm-c32-20260529_095611/`
- `/workspace/m3-mixed-prefill-layer-prof-20260529_095945/`
- `/workspace/m3-mixed-prefill-legacy-varlen-prof-20260529_100212/`
- `/workspace/m3-vllm-varlen-splitk-ab-20260529_183142/`
- `/workspace/m3-build-cache-touch-attn-20260529.log`
- `/workspace/m3-build-cache-final-vv-20260529.log`
- `/workspace/m3-build-cache-contenthash-final-vv-20260529.log`
- `/workspace/m3-unified-trace-c32-20260529_114549/`
- `/workspace/m3-varlen-tiled-v-build-20260529.log`
- `/workspace/m3-vllm-varlen-tiled-v-ab-20260529_115504/`
- `/workspace/m3-graphon-prod-profile-8ef71ce-20260529_120645/`
- `/workspace/m3-split-mixed-ab-20260529_121843/`
- `/workspace/m3-split-mixed-ab-n3-20260529_122047/`
- `/workspace/m3-unified-mixed-graph-ab-20260529_123549/`
- `/workspace/m3-vllm-varlen-tiled-q4-ab-20260529_git_n1/`
- `/workspace/m3-vllm-fa2-varlen-probe-20260529_git.log`
- `/workspace/m3-unified-greedy-ab-20260529_git_n1/`
- `/workspace/m3-prefill-first-ab-20260529_git_n1/`
- `/workspace/m3-prefill-first-prompt-est-ab-20260529_git_n1/`
- `/workspace/m3-greedy-prefill-fix-smoke-20260529_git_n1/`
- `/workspace/m3-graph-runtime-profile-20260529_git_n1/`
- `/workspace/m3-active-prefill-chunk-ab-20260529_git_n1/`

## Current target status

Use the ratios below as directional only until vLLM is rerun with
`n_repeats >= 5`.

- c=1 and c=4 are now the healthy cells.
- c=16 is close but not over 0.80 against same-pod vLLM 0.20.2 N=3:
  the latest current-binary A/B row is `1001.5 / 1328.7 ≈ 0.75`.
- c=32 remains the real blocker: the conservative current-binary forced-v2 row
  is `1299.0 / 1971.8 ≈ 0.66`. Short-v1 attention is separately validated as
  a `+2.0%` same-binary effect, but there is no clean final combined-default
  N=3 row yet. Prompt-token-estimate reached `1314.0 tok/s` in same-binary A/B
  but is not default-worthy because TTFT regressed. The split mixed-batch
  candidate only reached `1274.7 ± 33.9` and was reverted; simple Q-tiled
  varlen attention was only `+0.41%` at c32 N=1, and unified GPU argmax was
  only `+0.97%` at c32 N=1. Prefill-first admission was only `+3.77%` at c32
  N=1 and regressed when combined with prompt-token estimate. Active-decode
  prefill chunking flattened throughput while worsening TTFT. On the
  conservative row c32 still needs roughly `+21%` throughput to clear 0.80× on
  this pod.

## Next lever

Do not repeat env sweeps, the partial Marlin scheduling backport, DP + two-tile
split-K scheduler-only parity, block8-only testing, block-size-only vLLM raw-op
probes, forcing Marlin `128x128,bps=1`, graph-pre-sync removal, Marlin
source/body parity without new raw-op evidence,
`FERRUM_UNIFIED_CHUNKED_PREFILL=64`, chunk-size-only prefill sweeps, or
`FERRUM_MAX_BATCHED_TOKENS=4096` as a standalone lever. Also do not repeat the
two-phase `FERRUM_VLLM_VARLEN_SPLIT_K=1` wrapper; it passed Paris but regressed
c32 throughput by `6.06%`. Do not repeat simple vLLM-layout weighted-V tiling;
`FERRUM_VLLM_VARLEN_TILED_V=1` passed Paris but regressed c32 N=1 by about
`4.5%`. Do not spend another loop on simple Q-only varlen tiling:
`FERRUM_VLLM_VARLEN_TILED_Q4=1` passed Paris and had a positive standalone
microbench, but full-model c32 N=1 was only `+0.41%`. Do not repeat split-only
mixed prefill/decode routing; it was only
`+1.46%` at c32 N=3 and within noise. Do not repeat unified mixed layer-loop
graph capture; it regressed c32 N=1 by `3.82%`. Unified GPU argmax is already
committed as a cleanup but only measured `+0.97%` at c32 N=1, so it is not a
primary lever. Do not default or N=3 the prefill-first admission experiment:
it reduced `ITL p95` but only measured `+3.77%` at c32 N=1, worsened TTFT, and
regressed with prompt-token estimate. Do not default or N=3
`FERRUM_ACTIVE_DECODE_PREFILL_CHUNK=64` for throughput: it fixed much of the
ITL tail but was throughput-flat and made TTFT about `3x` worse. The next
high-return loop should target full-model time directly:

1. keep using the restored GPU pod if available; otherwise restore a 48GB-class
   pod before making new performance claims;
2. profile graph-on production c32 with `[graph-prof]`, `[iter-prof]`, and
   `[rbd-prof]` before changing code; the graph-off route profile already
   overstates per-kernel bucket time;
3. do not default prompt-token-estimate from the current evidence; it is only
   `+2.8%` and worsens TTFT;
4. before another scheduler change, the next profile should explain end-to-end
   occupancy/fill gaps and mixed-prefill attention cost (batch-size over time,
   warmup/non-warmup separation, streaming backpressure, and whether a proper
   tiled varlen prefill attention kernel can remove the `~61 ms` vLLM-layout
   attention bucket);
5. for kernel work, skip source parity unless a new raw-op mismatch appears;
   the remaining high-upside kernel path is a genuinely fused small-M MoE design;
6. reduce CUDA iteration waste before another `.cu` experiment by adding
   build-script cache/stamp logic or splitting Marlin static libraries, then run
   Paris and c16/c32 N=3 A/B on the same pod for any positive candidate.
