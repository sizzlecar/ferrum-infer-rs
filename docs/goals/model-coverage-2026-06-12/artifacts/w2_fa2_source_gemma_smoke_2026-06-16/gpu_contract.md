# W2 Gemma3 FA2 Source Product Smoke GPU Contract

- Lane: W2 Gemma3 typed FA2-source product smoke.
- Hardware: cached Vast instance `40826362`, 1x RTX 4090.
- Expected runtime/cost: 10-20 minutes, about USD 0.07-0.15 at
  USD 0.42488888888888887/h.
- Stop condition: startup, SSH, CUDA, config assertion, serve readiness,
  chat smoke, or minimal c16 bench first failure, or artifact collection
  complete, then stop the instance.
- Correctness gate: `ferrum serve` from artifact-local `ferrum.toml` with
  `runtime.use_vllm_paged_attn=true`, `runtime.fa2_source=true`, and
  `runtime.fa2_direct_ffi=false`; decision trace must select
  `attention_prefill_mixed_backend=fa2_source`; chat smoke must return `5`
  with usage; minimal bench must return rc 0 with zero request errors.
- Performance command: diagnostic-only natural ASCII ShareGPT c16,
  `bench-serve --fail-on-error --seed 9271 --n-repeats 1 --num-prompts 16`.
- Scope: diagnostic only. This run intentionally omits `--require-ci` and does
  not produce `MODEL_RELEASE_GRADE_W2 PASS`.
