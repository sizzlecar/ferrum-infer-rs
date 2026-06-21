# W3 Qwen3.5 Handoff - 2026-06-22

## Status

- Branch: `goal/w2-w3-release-grade`.
- PR: <https://github.com/sizzlecar/ferrum-infer-rs/pull/237>.
- W3 is not complete. There is no `MODEL_RELEASE_GRADE_W3 PASS`.
- Correctness evidence exists for real Qwen3.5 GPTQ product paths, including
  `ferrum run`, `ferrum serve`, streaming usage, tool calls, structured output,
  and L5 zero-error concurrency.
- Performance remains below the W3 80% vLLM target. The accepted vLLM c32 mean
  baseline is `1708.52785 output tok/s`, so the 80% target is
  `1366.82228 output tok/s`. Recent Ferrum c32 diagnostics have been around
  `629-695 output tok/s`.

## Latest Source Progress

This update moves Qwen3.5 GDN prefill toward the vLLM architecture instead of
adding model-name defaults or hidden environment switches.

- Added backend capability:
  `supports_qwen35_packed_gdn_prefill_prepare`.
- Added packed varlen prefill prepare API:
  `linear_attention_prepare_varlen_packed_qkvz_ba_f32`.
- Added CPU reference implementation.
- Added CUDA launcher and `.cu` kernel for packed `[q,k,v,z]` plus `[b,a]`
  varlen GDN prefill prepare.
- Routed Qwen3.5 product prefill through fused `qkvz_proj` and `ba_proj` when
  the backend supports packed prefill prepare.
- Routed product varlen prefill through compact core outputs so
  `query/key/value/g/beta/delta_core` debug/reference intermediates do not stay
  live after the GDN core boundary.
- Avoided duplicate product batch prefill state scatter when indexed linear
  state pools are present: final GDN/conv state is written to the slot pool,
  and sequence-local buffers are synchronized from the slot only when needed.
- Avoided the matching fresh-prefill initial state gather: the product batch
  prefill entry only accepts fresh final chunks, so linear-attention layers now
  pass zero-initialized initial conv/GDN state slabs directly to the varlen core
  instead of copying per-sequence zero state buffers every layer. The non-fresh
  gather path remains for future chunked/non-fresh prefill.
- Kept separate `qkv/z/b/a` projection fallback for unsupported backends.
- Added prefill profile fields for `qkvz_proj` and `ba_proj`.
- Added CPU packed-vs-separate contract test and a CUDA packed-vs-CPU parity
  test for the next GPU lane.
- Added a CPU fresh-prefill initial-state unit test that verifies
  `fresh_initial_linear_state=true` produces zero slabs even if sequence-local
  recurrent state is dirty, while `fresh_initial_linear_state=false` still
  gathers the existing state.
- Fixed continuous scheduler mixed-prefill budgeting so the aggregate
  active-decode prefill cap uses the effective per-request chunk
  `min(active_decode_prefill_chunk, prefill_step_chunk)`. This targets the W3
  trace pattern where an explicit large active chunk such as `8192` could still
  admit many small prefills into a decode iteration even though individual
  prefills were capped by `prefill_step_chunk`.
- Added a scheduler regression for the W3-like shape `decode=7`,
  `waiting_prefill=25`, `max_batch=32`, `max_tokens=8192`,
  `active_decode_prefill_chunk=8192`, `prefill_step_chunk=64`; the scheduler
  now emits `7 decode + 4 prefill chunks`, not all 25 free prefill slots.
- Added a Qwen3.5 model-level `forward_stateful_unified_items` path that
  classifies unified work into fresh prefill, continuation/chunked prefill, and
  decode. This removes the old assumption that Qwen35 unified work is only
  fresh final prefill.
- Fresh batch prefill now accepts non-final chunks and returns `None` for
  non-final rows. If only some rows are final, final norm/lm_head/readback is
  run only for those final rows.
- Continuation chunks now use the existing stateful Qwen35 path and only return
  logits for final chunks; decode rows still use batched decode and preserve
  `LogitsReturnPolicy` behavior.
- Added duplicate `cache_id` rejection and checked decode position conversion
  before batching.
- Added a focused CPU regression that runs real Qwen35 forward with a tiny
  valid config and verifies a mixed continuation chunk plus decode frame
  advances both sequence states while returning logits only for the decode row.
- Split Qwen3.5 stateful chunk execution into logits-returning and no-logits
  paths. Non-final continuation/chunked prefill rows now advance state and sync
  the linear state slot without running final norm, final-token gather,
  lm_head, or logits readback.
- Added a regression that replaces final norm/lm_head with deliberately broken
  test fixtures after the seed prefill, then proves a non-final continuation
  chunk still advances state without touching the logits tail.
- Removed the Qwen3.5 stateful continuation special case that split
  already-started multi-token chunks into one model pass per token. Final and
  non-final continuation chunks now advance through the stateful layer path as
  one chunk.
- The tiny Qwen35 forward test loader now uses small deterministic non-zero
  weights instead of all-zero weights, and a model-level parity regression
  compares stepwise continuation `[4] + [5]` with one final continuation chunk
  `[4, 5]`, including both returned logits and next-token decode logits.
- Added a paged-KV continuation batch path for Qwen3.5 unified prefill rows:
  when `use_paged_kv` is active, continuation/chunked prefill rows are grouped
  and sent through the existing varlen batch prefill layer path instead of the
  row-by-row stateful path.
- Generalized `forward_stateful_prefill_batch_taken` with an explicit
  `fresh_initial_linear_state` flag. Fresh prefill keeps zero initial recurrent
  state; continuation batch prefill syncs indexed linear-state slots back to
  sequence-local state before gathering.
- Fixed batch prefill KV allocation to target
  `state.tokens.len() + q_lens[row]`, which is required for continuation
  chunks and is equivalent to the old value for fresh prefill.
- Full-logits mixed continuation+decode frames can now share the paged
  continuation varlen prefill batch when it is semantically safe. Decode rows
  are merged only when paged KV is active, a continuation/chunked row is
  present, and every decode row requires full logits. Greedy-argmax/no-policy
  rows keep the decode batch path so the model-side argmax sentinel contract is
  preserved.
- Paged continuation varlen prefill batch now accepts
  `Qwen35DecodeLogitsReturn` and reuses the same model-side argmax helper as
  decode batch. Complete explicit `LogitsReturnPolicy` batches, including
  `GreedyArgmax`, may merge into the continuation batch and return argmax
  sentinels; legacy no-policy `FERRUM_GREEDY_ARGMAX=1` still stays on the
  decode batch path to preserve old no-policy behavior.
- Added regressions that lock this boundary and directly verify the shared
  argmax helper returns token sentinels for raw greedy policy rows.
- `LlmExecutor::batch_decode` now forwards `LogitsReturnPolicy` through
  `unified_forward_with_logits_policy` on the fast unified path. The fallback
  already used `decode_batch_with_logits_policy`; this closes the no-policy
  unified gap for decode-only batches using model-side greedy argmax, token
  masks, or sparse repetition penalties.

The key vLLM reference is:

```text
/Users/chejinxuan/py_ws/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
```

The relevant vLLM behavior is that Qwen3.5 prefill uses `in_proj_qkvz` and
`in_proj_ba`, then splits `[q,k,v,z]` and `[b,a]`. Ferrum prefill previously
still launched four projections.

## Local Validation

Passed locally:

```bash
cargo fmt --all
cargo test -p ferrum-kernels --test linear_attention_cpu \
  linear_attention_prepare_varlen_packed_cpu_matches_separate_prepare -- --nocapture
cargo check -p ferrum-kernels -p ferrum-models
cargo test -p ferrum-models \
  linear_attention_prefill_varlen_compact_core_matches_full_core_outputs -- --nocapture
cargo test -p ferrum-models \
  linear_attention_prefill_varlen_backend_matches_per_sequence_stateful_reference -- --nocapture
cargo fmt --all -- --check
cargo test -p ferrum-models \
  qwen35_fresh_prefill_initial_state_slabs_are_zero_not_gathered -- --nocapture
cargo check --workspace --all-targets
cargo test -p ferrum-scheduler active_decode_prefill -- --nocapture
cargo test -p ferrum-scheduler \
  newly_admitted_prefill_uses_remaining_budget_with_decode -- --nocapture
cargo test -p ferrum-scheduler -- --nocapture
cargo check -p ferrum-types -p ferrum-scheduler -p ferrum-engine -p ferrum-cli
cargo test -p ferrum-models \
  qwen35_unified_forward_mixes_decode_and_continuation_chunk -- --nocapture
cargo test -p ferrum-models \
  qwen35_unified_forward_requires_paged_kv_for_fresh_batch_prefill -- --nocapture
cargo check -p ferrum-models
cargo test -p ferrum-models \
  qwen35_unified_forward_non_final_continuation_skips_logits_tail -- --nocapture
cargo test -p ferrum-models \
  qwen35_unified_forward_multitoken_continuation_matches_stepwise -- --nocapture
cargo test -p ferrum-models \
  qwen35_unified_forward_mixes_decode_and_continuation_chunk -- --nocapture
cargo test -p ferrum-models \
  qwen35_decode_merge_policy_preserves_legacy_no_policy_contract -- --nocapture
cargo test -p ferrum-models \
  qwen35_try_argmax_logits_rows_returns_policy_sentinel -- --nocapture
cargo test -p ferrum-models \
  qwen35_unified_forward_mixes_decode_and_continuation_chunk -- --nocapture
cargo test -p ferrum-models \
  qwen35_unified_forward_multitoken_continuation_matches_stepwise -- --nocapture
cargo test -p ferrum-models \
  qwen35_unified_forward_non_final_continuation_skips_logits_tail -- --nocapture
cargo test -p ferrum-models \
  batch_decode_forwards_logits_policy_to_unified_model -- --nocapture
cargo test -p ferrum-models \
  unified_decode_forwards_logits_policy_to_unified_model -- --nocapture
cargo check -p ferrum-models
cargo fmt --all -- --check
git diff --check
```

Not yet validated locally:

- CUDA feature build, because this Mac has no `nvcc`.
- CUDA packed varlen parity test:

```bash
cargo test -p ferrum-kernels --features cuda --test linear_attention_cuda_eq \
  linear_attention_prepare_varlen_packed_cuda_matches_cpu_reference -- --nocapture
```

## GPU Next Lane

Paid GPU policy for the next run:

- Lane: W3 Qwen3.5 packed-prefill CUDA smoke and c32 diagnostic on exactly
  1x RTX 4090.
- Expected runtime/cost: about 30-60 minutes; on the previous Vast price
  `$0.662962962962963/hour`, expected cost is about `$0.33-$0.66`.
- Stop condition: stop after CUDA build/test failure, product smoke failure,
  one c32 diagnostic artifact, or a clear external blocker.
- Correctness gate:

```bash
cargo build --release -p ferrum-cli --bin ferrum \
  --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
cargo test -p ferrum-kernels --features cuda --test linear_attention_cuda_eq \
  linear_attention_prepare_varlen_packed_cuda_matches_cpu_reference -- --nocapture
```

Then run real Qwen3.5 product smoke for both `ferrum run` and `ferrum serve`,
including streaming with `stream_options.include_usage=true`.

- Performance command shape:

```bash
ferrum bench-serve \
  --dataset sharegpt \
  --num-prompts 64 \
  --warmup-requests 8 \
  --random-output-len 128 \
  --concurrency 32 \
  --n-repeats 1 \
  --seed 9271 \
  --fail-on-error
```

If c32 materially improves and correctness remains clean, run the release-shape
L5 cells with `c=1/4/16/32`, `--require-ci`, and `--n-repeats 3`.

## Current GPU Blocker

- Existing cached Vast instance `41422823` was last observed stopped/exited.
- A replacement 1x RTX 4090 attempt was externally blocked by Vast
  `insufficient_credit`.
- On 2026-06-21 UTC, direct SSH to `ssh7.vast.ai:22822` returned
  `Connection refused`.
- On 2026-06-22 Asia/Shanghai, direct SSH to `ssh7.vast.ai:22822` still
  returned `Connection refused`.
- The Vast API still listed instance `41422823` as 1x RTX 4090 with
  `cur_state=stopped`, `actual_status=exited`.
- A start request returned
  `Required resources are currently unavailable, state change queued`; a
  5-minute poll kept reporting `cur_state=stopped`,
  `actual_status=exited`, so no CUDA artifact was produced from that attempt.
- Do not keep cycling rental attempts until credit is restored.

## What To Watch

- Profile should show prefill projection time moving from separate
  `qkv/z/b/a` entries to `qkvz/ba`.
- Product prefill should use compact core output; debug/reference tests still
  keep full intermediates for parity checks.
- On CUDA indexed-state runs, `qwen35_linear_prefill_state_scatter` should be
  near zero and `qwen35_linear_prefill_pool_scatter` should contain the state
  write cost.
- Fresh product batch prefill should also show little/no
  `qwen35_linear_prefill_state_gather` work, because initial recurrent state is
  now a zero slab instead of per-sequence copies.
- Scheduler trace should no longer show large active-decode mixed-prefill
  cohorts caused by explicit `active_decode_prefill_chunk=8192` bypassing the
  smaller `prefill_step_chunk`; the W3-like source regression caps that case to
  four 64-token prefill chunks when seven decodes are active.
- Qwen35 unified trace should no longer reject a mixed frame solely because it
  contains continuation/chunked prefill. Fresh prefill still requires paged KV;
  CUDA paged-KV continuation chunks should use the varlen batch prefill path.
- Non-final continuation chunks should show no final norm/lm_head/readback
  tail; if scheduler traces still show high chunk cost, profile the layer body
  rather than the logits tail first.
- Multi-token continuation chunks should show one stateful chunk pass rather
  than per-token model passes. The local parity test allows small float-path
  differences (`1e-3`) against stepwise continuation and also compares the next
  decode logits.
- With paged KV active on CUDA, multiple continuation/chunked prefill rows
  should enter the varlen batch prefill path rather than row-by-row stateful
  calls. CPU cannot execute this paged varlen path; local evidence only proves
  non-paged fallback and type-level compilation.
- Full-logits mixed continuation+decode frames should also enter that same
  paged continuation batch. Explicit `GreedyArgmax` policy rows can now enter
  the same batch and should return model-side argmax sentinels; no-policy
  `FERRUM_GREEDY_ARGMAX=1` remains on decode batch by design.
- Qwen3.5 prefill profiles now include `argmax` time; for policy-driven merged
  continuation batches, confirm the profile shows argmax rather than full
  vocab readback when masks/penalties allow it.
- Decode-only `batch_decode` should now show the same policy-aware model-side
  argmax behavior as `unified_decode`; if masked/repetition-penalty decode
  correctness differs between the two product paths, inspect executor policy
  construction before changing model kernels.
- CUDA build must confirm the new `.cu` symbols are present.
- If packed prefill improves projection cost but c32 remains far below target,
  continue with profiler-backed bottleneck localization; do not revert blindly
  and do not start env-flag sweeps.
- This update is source progress only. Do not make a performance claim without
  same-hardware artifacts and command logs.
