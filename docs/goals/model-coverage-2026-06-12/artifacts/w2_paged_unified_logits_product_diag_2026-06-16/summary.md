# W2 paged-unified product logits diagnostic summary

- Scope: diagnostic product-path smoke only. This did not run a performance
  benchmark or a release-grade validator.
- Source: local/remote base `b768073a80c8a7519c1107083f5a10b478d0fe1a`.
- Diagnostic patch: remote-only temporary change to allow paged KV for
  windowed Gemma3 when the CUDA backend supports varlen QKV. The checked-in
  default path remains protected.
- Instance: Vast `40826362`, 1x RTX 4090.
- Build: CUDA release build rc `0`; binary SHA256
  `1d046a81f5194f80a946b2c0e2f37f1de97fdde69668ad359135f032e32af5d9`.
- Command shape: `ferrum serve --model gemma3:27b-gptq --backend cuda`
  with `FERRUM_DECODE_OP_PROFILE=1`, then one non-stream chat request with
  `max_tokens=1`.
- Response: empty content, `finish_reason=stop`, `completion_tokens=1`.
- Unified path evidence:
  - `[unified-decode] call#0 ... attempted_unified=true fallback=false
    fallback_reason=none`.
  - `[unified-logits] call#0 ... finite=262208 nan=0 pos_inf=0 neg_inf=0
    top=[106:11.039062,108:9.445312,107:8.882812,245526:8.460938,236743:8.304688]`.
- Token decode: generation config lists eos token ids `[1, 106]`; token id
  `106` is `<end_of_turn>`.
- Interpretation: the paged-unified path is producing finite logits, but the
  first sampled row ranks the stop token highest. This rules out NaN/Inf or
  uninitialized sampled logits for this repro and points above the standalone
  split-QKV + paged-varlen attention pair. The most likely source-level gap is
  that `unified_forward_internal` omitted Gemma3 `embed_scale` after
  `embedding_lookup`, unlike the legacy decode/prefill paths.
- Cleanup: service was stopped; GPU memory returned to 1 MiB before Vast
  shutdown. Vast final poll recorded `cur_state=stopped actual_status=exited`.
