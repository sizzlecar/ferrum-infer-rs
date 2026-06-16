# W2 Active-Decode Chunk C16 CUDA Diagnostic GPU Contract

Date: 2026-06-17

## Lane

W2 Gemma3-27B GPTQ active-decode prefill chunk c16 CUDA diagnostic on exactly
one RTX 4090.

## Expected Runtime And Cost

- Prefer reusing an existing stopped/cached Vast instance when it can start
  cleanly and still exposes one RTX 4090.
- Expected runtime with cache: 20-45 minutes.
- Fresh-instance upper bound: 60-75 minutes.
- Cost estimate: about USD 0.4-0.8/hour depending on selected Vast offer.
- Hard stop: 75 minutes unless the command is already copying artifacts or
  shutting the instance down.

## Stop Condition

- Stop immediately if instance start, SSH, CUDA visibility, source sync, or
  CUDA release build fails.
- Stop immediately if product-path correctness fails for either `ferrum run`
  or `ferrum serve`.
- After c16 diagnostic profile artifacts are copied back, stop or destroy the
  instance and verify it is no longer running.

## Correctness Gate

- CUDA release build with features:
  `cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`.
- `ferrum run` smoke on the W2 Gemma3 GPTQ model.
- `ferrum serve` streaming smoke requiring usage, output tokens, and exactly
  one `data: [DONE]`.

## Performance Command

Run only a c16 diagnostic `bench-serve --fail-on-error` profile, not a release
full sweep. Required profile knobs:

- `FERRUM_BATCH_DECODE_PROF=1`
- `FERRUM_NEXT_BATCH_PROF=1`
- `FERRUM_UNIFIED_POST_PROF=1`
- `FERRUM_DECODE_OP_PROFILE=1`
- `FERRUM_MARLIN_PROFILE=1`

Acceptance for the lever:

- effective config shows typed default
  `FERRUM_ACTIVE_DECODE_PREFILL_CHUNK=16`;
- profile frames replace the previous full-prompt mixed frame shape
  `prefill=12 decode=4 m_total=897` with small active-decode prefill chunks or
  otherwise show that decode-active prefill is no longer monopolizing c16
  batches.

No release performance claim is made from this diagnostic, and no W2 completion
claim is valid unless the final model-release validator later prints its exact
PASS line.
