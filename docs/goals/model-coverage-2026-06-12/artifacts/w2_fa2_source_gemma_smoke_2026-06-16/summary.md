# W2 Gemma3 FA2 Source Minimal Config Smoke

## Scope

This artifact attempted to force `runtime.fa2_source=true` with an artifact-local
minimal `ferrum.toml` containing only a `[runtime]` section. It is not FA2
performance evidence.

## Result

The server became ready, but the decision assertion failed:

- expected `attention_prefill_mixed_backend=fa2_source`;
- observed `attention_prefill_mixed_backend=legacy_paged_varlen`;
- no chat smoke or bench was run after the failed assertion.

The local `ferrum.toml` file exists and contains the intended runtime keys, but
the effective runtime config did not include `FERRUM_FA2_SOURCE` or
`FERRUM_USE_VLLM_PAGED_ATTN`. The likely cause is that a partial config file does
not deserialize as the full `CliConfig`, causing the CLI to fall back to default
configuration without `--verbose` output.

## Interpretation

This artifact only explains why the first FA2 attempt did not exercise FA2. The
follow-up artifact `w2_fa2_source_gemma_full_config_smoke_2026-06-16` uses a
complete config file and is the valid FA2 product-path smoke.
