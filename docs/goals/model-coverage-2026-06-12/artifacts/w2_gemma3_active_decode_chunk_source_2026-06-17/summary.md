# W2 Gemma3 Active-Decode Prefill Chunk Source Checkpoint

## Scope

- Source-only checkpoint. No GPU instance was started.
- No release performance claim and no `MODEL_RELEASE_GRADE_W2 PASS` was produced.
- Goal: turn the current bottleneck evidence into a typed product default so the
  next CUDA diagnostic can test a focused scheduler hypothesis.

## Evidence Reviewed

- Previous dense vLLM Marlin probes already ruled out a direct dense Marlin
  kernel swap as the primary W2 gap:
  `w2_dense_vllm_marlin_native_probe_retry_2026-06-15` and
  `w2_dense_vllm_marlin_weight_cycle_probe_2026-06-15`.
- Latest c16 profile after the logits-readback fix:
  `w2_unified_argmax_c16_cuda_diag_2026-06-17`.
- The target sampled frame was:
  `call#21 m_total=897 num_seqs=16 prefill=12 decode=4 total=383684us`.
- The same profile shows pure decode batches at `prefill=0 decode=4` taking
  about `28-35ms`, while mixed or prefill-heavy frames dominate wall time.

## vLLM Comparison

- Local vLLM source schedules running requests first, then uses a token budget
  to schedule waiting/prefill work.
- vLLM exposes chunked prefill through scheduler config fields such as
  `max_num_batched_tokens`, `enable_chunked_prefill`,
  `max_num_partial_prefills`, and `long_prefill_token_threshold`.
- Ferrum already has the analogous typed control plane:
  `FERRUM_ACTIVE_DECODE_PREFILL_CHUNK`, surfaced through
  `scheduler_active_decode_prefill_chunk` and materialized into
  `EngineConfig.scheduler.active_decode_prefill_chunk`.

## Source Change

- Added a Gemma3 CUDA GPTQ/int4 default in `ferrum-types` auto-config:
  `FERRUM_ACTIVE_DECODE_PREFILL_CHUNK=16`.
- The default is capability-gated:
  - backend is CUDA;
  - model architecture is Gemma3;
  - quantization mentions GPTQ or int4.
- Explicit user/config/CLI scheduler choices still win:
  - `FERRUM_ACTIVE_DECODE_PREFILL_CHUNK`;
  - `FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE`;
  - `FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE`.
- The decision trace now selects
  `scheduler_admission_policy=active_decode_prefill_chunk:16` for Gemma3 CUDA
  GPTQ when no explicit scheduler policy is provided, and the effective runtime
  snapshot materializes `FERRUM_ACTIVE_DECODE_PREFILL_CHUNK=16`.

## Local Validation

- `cargo fmt --all -- --check`
- `cargo test -p ferrum-types auto_config -- --nocapture`
- `cargo test -p ferrum-scheduler active_decode_prefill_chunk_only_caps_when_decode_is_active -- --nocapture`
- `cargo test -p ferrum-engine continuous_engine_runtime_config_parses_env_snapshot -- --nocapture`
- `cargo check -q -p ferrum-types -p ferrum-scheduler -p ferrum-engine -p ferrum-cli`
- `git diff --check`

All commands passed.

## Next CUDA Check

Run a paid 1x RTX 4090 W2 diagnostic only after stating the GPU contract. The
minimum useful check is:

- correctness first: CUDA release build, `ferrum run` smoke, `ferrum serve`
  streaming smoke with usage and exactly one `[DONE]`;
- performance diagnostic: c16 `bench-serve --fail-on-error` with
  `FERRUM_BATCH_DECODE_PROF=1`, `FERRUM_NEXT_BATCH_PROF=1`,
  `FERRUM_UNIFIED_POST_PROF=1`, `FERRUM_DECODE_OP_PROFILE=1`, and
  `FERRUM_MARLIN_PROFILE=1` in typed artifact config;
- expected profile change: mixed frames should show prefill chunks of about 16
  tokens per active prefill request instead of full-prompt `12 prefill + 4
  decode` frames near `m_total=897`.
