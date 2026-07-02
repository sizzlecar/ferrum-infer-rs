# W2 dense unified logits-policy source checkpoint

Status: source checkpoint only; no `MODEL_RELEASE_GRADE_W2 PASS` was produced.

## Scope

- No GPU instance was started.
- Target: Gemma3 dense unified forward readback path.
- Motivation: `w2_unified_op_profile_c16_rerun_2026-06-16` measured
  `readback=22039us` in a mixed c16 frame because dense unified forward always
  downloaded `sampled * vocab` logits.

## Change

- Added `DecoderOnlyLLM::unified_forward_with_logits_policy(...)` with a
  backwards-compatible default.
- `LlmExecutor::unified_decode` now forwards each
  `UnifiedBatchItem.logits_policy` into real unified model execution and treats
  policy-required full logits as a full-logits condition.
- Dense `LlamaFamilyModel` unified forward now uses existing GPU
  `argmax_rows_f16` / `argmax_rows_f16_masked` when every sampled row is
  greedy-compatible and masks are uniform.
- Full logits are still returned when any sampled row requires CPU-side
  sampling, structured output, or incompatible/mixed masks.
- In the default no-prefix-cache product path, final prefill chunks now carry
  the same model-side greedy policy as decode rows; prefix-cache-enabled runs
  still force full logits so cached logits remain available.

## Local Validation

```text
cargo fmt --all -- --check
git diff --check
cargo test -p ferrum-models unified_decode_ -- --nocapture
cargo test -p ferrum-engine model_decode_logits_policy -- --nocapture
cargo check -q -p ferrum-models -p ferrum-engine
```

All commands passed locally.

## Required Next GPU Validation

Run one cached 1x RTX 4090 diagnostic before treating this as a performance
win:

1. Build CUDA release binary with
   `cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`.
2. Run `ferrum run` 2+3 and `ferrum serve` chat smoke for
   `gemma3:27b-gptq`.
3. Run c16 diagnostic with `FERRUM_DECODE_OP_PROFILE=1` or config equivalent
   and verify `unified-op-profile readback` drops while completed requests stay
   full and errors remain zero.
4. Only after correctness is clean, compare c16 throughput against the existing
   same-dataset vLLM baseline.
