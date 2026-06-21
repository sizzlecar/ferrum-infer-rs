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

### 2026-06-22 Manifest Self-Test Contract Sync

- `scripts/release/model_release_grade_manifest.py --self-test` was updated
  after the stricter L2/L5 final-validator contracts.
- Its synthetic W3 L1 fixture now includes
  `full_attention_official_shape=true`.
- Its synthetic W3 L2 fixture now includes real `ferrum run` and
  `ferrum serve` command lines.
- Its synthetic W3 L5 fixture now includes a compliant
  `bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`
  command covering `c=1/4/16/32`.
- Validation passed locally:
  `python3 scripts/release/model_release_grade_manifest.py --self-test`,
  `python3 scripts/release/model_release_grade_goal_gate.py --self-test`,
  `python3 scripts/release/w3_l2_quantized_gate.py --self-test`, Python
  compile checks, and `git diff --check`.
- The `MODEL_RELEASE_GRADE_W3 PASS` printed by this self-test is synthetic
  temporary-directory evidence only. W3 real-model release-grade remains
  incomplete.

### 2026-06-22 L2 Product-Command Gate Hardening

- `scripts/release/w3_l2_quantized_gate.py` now requires real
  `command_line` evidence for both `ferrum run` and `ferrum serve`.
- Declaration-only evidence such as `product_entrypoints` or
  `{"entrypoint": "ferrum run"}` no longer counts toward W3 L2 product
  coverage.
- L2 command evidence is normalized, must invoke the `ferrum` binary, must
  contain exactly one of `run` / `serve`, and must not embed hidden
  `FERRUM_*=` overrides.
- `scripts/release/model_release_grade_goal_gate.py` now re-checks those L2
  commands in the final W3 validator. It still tolerates older artifacts with
  extra declaration-only entries as long as real command lines cover both
  required product entrypoints.
- Validation passed locally:
  `python3 -m py_compile scripts/release/w3_l2_quantized_gate.py scripts/release/model_release_grade_goal_gate.py`,
  `python3 scripts/release/w3_l2_quantized_gate.py --self-test`,
  `python3 scripts/release/model_release_grade_goal_gate.py --self-test`,
  `python3 scripts/release/w3_qwen35_real_product_report.py --self-test`, an
  existing-artifact L2 final-validator probe, a temporary re-package of the
  historical real known-answer report, and `git diff --check`.
- Direct SSH probe to `ssh7.vast.ai:22822` still returned
  `Connection refused`, so no CUDA run was started from this checkpoint.

### 2026-06-22 L5 Release-Command Gate Hardening

- `scripts/release/w3_l5_concurrency_gate.py` now requires saved
  `bench-serve` command evidence before packaging a W3 L5 artifact.
- The command evidence is parsed with `shlex`, normalized into
  `command_line`, and must prove closed-loop coverage of `c=1/4/16/32`.
- The L5 gate rejects hidden `FERRUM_*` env overrides, `--request-rate`, missing
  `--fail-on-error`, missing `--require-ci`, wrong seed, and wrong repeat
  count.
- `scripts/release/model_release_grade_goal_gate.py` now re-checks those L5
  commands in the final W3 validator, so a hand-built zero-error L5 report
  cannot satisfy release-grade evidence without the required benchmark command.
- Validation passed locally:
  `python3 -m py_compile scripts/release/w3_l5_concurrency_gate.py scripts/release/model_release_grade_goal_gate.py`,
  `python3 scripts/release/w3_l5_concurrency_gate.py --self-test`,
  `python3 scripts/release/model_release_grade_goal_gate.py --self-test`, an
  existing-artifact L5 command validation probe, and `git diff --check`.
- This is tooling/evidence progress only. It does not add new CUDA correctness
  or performance evidence, and W3 still has no `MODEL_RELEASE_GRADE_W3 PASS`.

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
- Fresh Qwen3.5 batch prefill now has a policy-aware entry point and
  `forward_stateful_unified_items` derives logits return policy for final fresh
  prefill rows. Product unified first-token generation can therefore return
  model-side greedy argmax sentinels when policy-compatible, instead of forcing
  full vocab logits readback for fresh final prefill.
- Added an engine-level product-path regression that goes through
  `ContinuousBatchEngine::infer`, captures the generated `UnifiedBatch`, and
  verifies final prefill work carries `LogitsReturnPolicy::GreedyArgmax` with
  the expected token mask instead of silently falling back to full logits.
- Runtime chunked prefill no longer forces `process_batch` onto the legacy
  split path. The unified producer now treats scheduler `tokens_to_process`,
  active-decode prefill chunk, and typed `chunked_prefill_size` as coexisting
  upper bounds and uses the smallest cap, so explicit chunked-prefill can still
  use the unified/chunked prefill architecture instead of bypassing the Qwen3.5
  paged continuation path.
- Qwen3.5 paged prefill now has one typed batch entry for fresh-only,
  continuation-only, and mixed prefill rows. This removes duplicated validation
  and gives `forward_stateful_unified_items` one place to run mixed varlen
  prefill work.
- Linear-attention batch prefill initial state is now selected per row instead
  of per batch: fresh first chunks keep zero initial conv/GDN state, while
  continuation and decode-candidate rows gather existing recurrent state in the
  same batch.
- Paged unified Qwen3.5 work can now place fresh first chunks, continuation
  chunks, and eligible decode candidates into one mixed prefill batch. This is
  the source-side architecture change needed after runtime chunked prefill was
  kept on the unified path; it avoids splitting fresh first chunk work from
  decode solely because the prefill row starts at position 0.
- The continuous-engine product path now has a regression proving active decode
  work and a fresh first prefill chunk are emitted in the same `UnifiedBatch`.
  The test constructs a real `BatchPlan`, preloads one decode-ready sequence,
  and captures one unified model call with a non-final fresh chunk at
  `pos_offset=0` plus a decode row at `pos_offset>0`.
- `LlmExecutor::unified_decode` now has an executor-level regression proving
  the same mixed `UnifiedBatch` shape is forwarded as one
  `unified_forward_with_logits_policy` call to the model layer. The regression
  locks row order, `pos_offset`, non-final `None` output for the fresh chunk,
  and no fallback to split prefill/decode.
- Qwen3.5 dense full-attention backend now has a source parity regression for
  the official scaled shape family: `hidden_size != q_total`,
  `q_proj_total = 2 * q_total`, `attn_output_gate=true`,
  `num_heads > num_kv_heads`, `rope_dim < head_dim`, interleaved partial RoPE,
  and non-zero `position_offset`. The test compares the backend layer path
  against the CPU reference across q projection/gate split, Q/K RMSNorm,
  partial RoPE, head-major attention, context gating, `o_proj`, and dense MLP
  output.
- Qwen3.5/Qwen3.6 weight inventory now accepts complete GPTQ tensor sets for
  required linear roles. A required linear can resolve from `.weight` or from
  `.qweight` + `.scales` + `.qzeros`; the resolved `.qweight` still maps back
  to the module name before `WeightLoader::load_linear`, matching the existing
  `NativeSafetensorsLoader` GPTQ path. Incomplete GPTQ aliases still fail the
  manifest instead of being treated as usable weights.
- `mlp.shared_expert_gate` now also uses the linear-loader boundary instead of
  a raw tensor GEMM. This matches vLLM's `ReplicatedLinear(hidden_size, 1)`
  modeling and lets dense or GPTQ shared-expert gate weights use the same
  `WeightLoader::load_linear` path as router and shared expert projections.
- Added a real HF metadata probe:
  `scripts/release/w3_qwen35_weight_index_probe.py`. It reads only
  `config.json`, `model.safetensors.index.json`, and optional
  `quantize_config.json`, then validates Ferrum's Qwen3.5 manifest against the
  actual `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` checkpoint at
  `3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b`.
- The real probe produced:
  `W3 QWEN35 WEIGHT INDEX PROBE PASS:
  docs/goals/model-coverage-2026-06-12/artifacts/w3_qwen35_weight_index_probe_20260622`.
  Key results: selected prefix `model.language_model`, zero missing required
  tensors, `124611` tensor names across `14` shards, GPTQ config
  `bits=4/group_size=128/desc_act=false/sym=true`, and complete sparse-MoE
  per-expert GPTQ coverage for `40` layers x `256` experts
  (`92160` checked `.qweight/.scales/.qzeros` tensors, `g_idx` present for all
  layers). Required non-expert tensors resolve as dense/meta tensors:
  `552` dense `.weight`, `60` non-linear tensors, and top-level
  `lm_head.weight`.
- Existing Vast instance `41422823` was checked for a W3 Qwen35 mixed-prefill
  CUDA smoke/c32 diagnostic. SSH to `ssh7.vast.ai:22822` returned connection
  refused, and the sanitized API summary says `cur_state=stopped`,
  `actual_status=exited`, `intended_status=stopped`. CUDA validation did not
  run; the sanitized status artifact is
  `artifacts/w3_qwen35_mixed_prefill_cuda_95adb578_20260622/local_vast/status_summary.json`.
- A later current-SHA start attempt for the same cached instance also did not
  produce a runnable GPU lane. The lane was W3 Qwen35 GPTQ-Int4 current-SHA
  CUDA correctness smoke at `7ba1f415c54f7eab050563b801a37fb38f0f28af`;
  intended correctness path was `w3_qwen35_real_product_report.py` followed by
  `w3_l2_quantized_gate.py`, with performance deferred. Vast `PUT
  state=running` returned an empty object and the follow-up API query still
  reported `cur_state=stopped`, `actual_status=exited`,
  `intended_status=stopped`; no remote SSH/build/gate command ran. Sanitized
  artifact:
  `artifacts/w3_qwen35_cuda_current_sha_7ba1f415_start_20260621T214634Z/summary.json`.
- A follow-up read-only Vast inventory/credit check saved
  `artifacts/w3_qwen35_vast_credit_inventory_20260622T0554CST/summary.json`.
  It reports `credit=0`, negative balance state, and only the stopped/exited
  cached 1x4090 instance. Do not create replacement instances until credit is
  restored.
- While the GPU lane was unavailable, the local evidence pipeline was checked:
  `w3_qwen35_real_product_report.py --self-test`,
  `w3_l2_quantized_gate.py --self-test`, `w3_l4_agent_gate.py --self-test`,
  `w3_l5_concurrency_gate.py --self-test`, and
  `model_release_grade_goal_gate.py --self-test` all passed.
- `w3_qwen35_real_product_report.py` now captures git status before creating
  the artifact directory and writes that pre-run snapshot into the S2 product
  artifact and report summary. This prevents a clean GPU run from being marked
  dirty solely because the runner wrote its own evidence files under the repo.

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
cargo test -p ferrum-models \
  unified_decode_forwards_prefill_logits_policy_to_unified_model -- --nocapture
cargo test -p ferrum-engine \
  process_batch_unified_forwards_prefill_logits_policy -- --nocapture
cargo test -p ferrum-engine \
  process_batch_unified_honors_runtime_chunked_prefill -- --nocapture
cargo test -p ferrum-engine \
  process_batch_unified_co_batches_active_decode_with_fresh_prefill_chunk -- --nocapture
cargo test -p ferrum-models \
  unified_decode_forwards_mixed_fresh_prefill_and_decode_to_unified_model -- --nocapture
cargo test -p ferrum-models \
  unified_decode_forwards_logits_policy_to_unified_model -- --nocapture
cargo test -p ferrum-models \
  unified_decode_forwards_prefill_logits_policy_to_unified_model -- --nocapture
cargo test -p ferrum-models \
  batch_decode_forwards_logits_policy_to_unified_model -- --nocapture
cargo test -p ferrum-models qwen35_unified_forward -- --nocapture
cargo test -p ferrum-models \
  qwen35_fresh_prefill_initial_state_slabs_are_zero_not_gathered -- --nocapture
cargo test -p ferrum-models \
  qwen35_decode_merge_policy_preserves_legacy_no_policy_contract -- --nocapture
cargo check -p ferrum-models
cargo test -p ferrum-models \
  dense_full_attention_backend_matches_reference_for_qwen35_gated_official_like_shape -- --nocapture
cargo test -p ferrum-models dense_full_attention -- --nocapture
cargo test -p ferrum-models \
  full_attention_backend_core_matches_reference -- --nocapture
cargo check -p ferrum-models
cargo test -p ferrum-models qwen35_weights -- --nocapture
cargo test -p ferrum-models \
  sparse_moe_shared_expert_composes_router_fused_experts_and_shared_gate -- --nocapture
cargo check -p ferrum-models
cargo check -p ferrum-engine
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
- Fresh final prefill in unified batches should also show policy-aware argmax
  when the request is greedy-compatible; if first-token profiles still show
  full vocab readback for ordinary greedy requests, first rule out non-greedy
  sampling, structured-output requests, prefix-cache full-logits mode, or token
  mask inconsistency before changing Qwen3.5 kernels. The engine-level source
  regression now proves ordinary greedy product prefill attaches the policy.
- Explicit typed chunked-prefill should now appear as non-final/final unified
  prefill items, not as a legacy split-path prefill loop. If a CUDA run still
  shows split-path behavior, inspect whether speculative decoding is enabled or
  whether the executor reports `supports_native_unified_decode=false`.
- Full-attention CUDA parity should include the official gated shape, not only
  old dense assumptions: q projection rows are `[query, gate]` per head,
  `q_proj_total` is twice `q_total`, RoPE is partial/interleaved, and `o_proj`
  consumes `q_total`, not `hidden_size`.
- Real GPTQ checkpoint inventory should now pass for required linears that are
  present only as `.qweight/.scales/.qzeros`, including
  `mlp.shared_expert_gate`. If real load still fails at this boundary, inspect
  the checkpoint tensor names first before adding aliases.
- CUDA build must confirm the new `.cu` symbols are present.
- If packed prefill improves projection cost but c32 remains far below target,
  continue with profiler-backed bottleneck localization; do not revert blindly
  and do not start env-flag sweeps.
- This update is source progress only. Do not make a performance claim without
  same-hardware artifacts and command logs.
