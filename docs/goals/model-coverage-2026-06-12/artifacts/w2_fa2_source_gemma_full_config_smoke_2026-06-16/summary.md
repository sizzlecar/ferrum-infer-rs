# W2 Gemma3 FA2 Source Full Config Product Smoke

## Scope

This artifact validates whether current W2 Gemma3 CUDA GPTQ product `serve` can
select the source-linked FA2 prefill/mixed attention path through typed runtime
configuration, then runs one minimal ShareGPT c16 diagnostic cell. It is not
release-grade evidence because it uses `n_repeats=1`, omits `--require-ci`, and
does not run the final W2 validator.

## Evidence

- Remote clean worktree HEAD: `017300426514d62e8e50ac1546ff77d4d54fd6ce`.
- Binary SHA256:
  `d38caf704f252045c29bdfe02795606937f400ab00edef05647da74179b215d5`.
- Artifact-local complete `ferrum.toml` copied from the repo root, with
  `runtime.use_vllm_paged_attn=true`, `runtime.fa2_source=true`, and
  `runtime.fa2_direct_ffi=false` appended.
- Effective config selected `attention_prefill_mixed_backend=fa2_source` from
  config-file `FERRUM_FA2_SOURCE`.
- Decode backend selected `vllm_paged_attn_v1_short`.
- Autosize logged `KV pool copies=2 (FA-compatible attention path)`.
- Chat smoke returned content `5` with usage present.
- Bench rc `0`; server error scan had `0` lines.
- Vast cleanup confirmed `cur_state=stopped` and `actual_status=exited`.

## Results

ShareGPT c16 diagnostic:

- `16 completed / 0 errored / 0 bad_output / 0 zero_output`;
- `output_token_count_source=usage`;
- output throughput `313.472 tok/s`;
- TTFT p50 `489.564ms`, TPOT p50 `43.178ms`, ITL p50 `38.637ms`.

Compared with current graph-disabled Ferrum same-dataset c16
`339.9306 tok/s`, this is `-26.4586 tok/s`, or `-7.78%`.
Compared with the clean vLLM ShareGPT c16 baseline `518.796 tok/s`, the ratio is
`0.6042`.

## Interpretation

FA2 is not the missing W2 Gemma3 ShareGPT performance lever. Once the product
path actually selects `fa2_source`, correctness remains clean in this minimal
smoke, but endpoint throughput regresses versus the current default path. This
matches the profiler direction: W2's current gap is dominated by model-step and
Gemma GPTQ MLP/tail behavior, not by prefill/mixed attention alone.
