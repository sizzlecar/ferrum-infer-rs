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

## Current target status

Use the ratios below as directional only until vLLM is rerun with
`n_repeats >= 5`.

- c=1 and c=4 are now the healthy cells.
- c=16 is close but not over 0.80 against same-pod vLLM 0.20.2 N=3:
  the latest current-binary A/B row is `1001.5 / 1328.7 ≈ 0.75`.
- c=32 remains the real blocker: the conservative current-binary forced-v2 row
  is `1299.0 / 1971.8 ≈ 0.66`. Short-v1 attention is separately validated as
  a `+2.0%` same-binary effect, but there is no clean final combined-default
  N=3 row yet. On the conservative row c32 still needs roughly `+21%`
  throughput to clear 0.80× on this pod.

## Next lever

Do not repeat env sweeps, the partial Marlin scheduling backport, DP + two-tile
split-K scheduler-only parity, block8-only testing, block-size-only vLLM raw-op
probes, forcing Marlin `128x128,bps=1`, graph-pre-sync removal, or
`FERRUM_MAX_BATCHED_TOKENS=4096` as a standalone lever. The next high-return
loop should target model time directly:

1. keep using the restored GPU pod if available; otherwise restore a 48GB-class
   pod before making new performance claims;
2. add or run a scoped unified-layer profile that separates prefill and decode
   dense/attention/MoE time; the current restored profile already rules out
   scheduler/postprocess as the main gap;
3. with a clean build, capture c32 Marlin config using
   `FERRUM_VLLM_MOE_LOG_CONFIG=1 FERRUM_VLLM_MOE_LOG_CONFIG_MIN_PAIRS=256`, and
   run a scoped A/B for the opt-in large-M block hook only if prefill/unified
   MoE remains material;
4. if MoE remains dominant, prototype either a full vLLM 0.20.2 Marlin-MoE
   source-parity port behind the existing C ABI or one small-m gate_up/down
   fused kernel, and require a microbench/profile reason before another
   release build;
5. run Paris, then c16/c32 N=3 A/B on the same pod.
