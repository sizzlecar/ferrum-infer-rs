# Codebase Shape Status - 2026-05-30

Milestone H is not complete. This checkpoint reduces the backend trait
surface file, Qwen3-MoE model surface, Qwen3-MoE unified-forward surface, and
continuous engine wrapper below their H line-count thresholds without changing
backend or model behavior.

## Added

- `crates/ferrum-kernels/src/backend/types.rs` now owns backend data types
  that were previously embedded in `backend/traits.rs`:
  - source dtype and quantization descriptors;
  - attention config;
  - KV cache containers;
  - MoE routing buffers.
- `crates/ferrum-kernels/src/backend/capabilities.rs` now owns optional
  backend capability traits layered on top of `Backend`:
  - graph capture/replay;
  - collective ops;
  - Marlin/GGUF quant loading;
  - fused MoE routing/post-op kernels.
- `backend/traits.rs` now keeps the core `Backend` trait plus paged-KV and
  KV-dtype capability axis. Existing public imports remain available through
  `ferrum_kernels::backend::*`.
- `crates/ferrum-models/src/models/qwen3_moe_forward_unified_layer.rs` now
  owns the per-transformer-layer unified forward implementation. The main
  `qwen3_moe_forward_unified.rs` file keeps mixed-batch setup, final sampling,
  readback, cache length updates, and unified profile helpers.
- `crates/ferrum-engine/src/continuous_engine/inner.rs` now owns the
  `EngineInner` helpers and iteration loop. The request-processing
  implementation is split under `crates/ferrum-engine/src/continuous_engine/inner/`:
  - `batch.rs`: batch classification, unified path, legacy split path, and
    preemption;
  - `prefill.rs`: single and batched prefill;
  - `decode.rs`: batched decode, single-step decode, and speculative decode;
  - `completion.rs`: streaming deltas and request completion.
  The main `continuous_engine.rs` file keeps runtime config parsing, sequence
  state, the engine wrapper, trait impls, and focused tests.
- `crates/ferrum-models/src/models/qwen3_moe_runtime.rs` now owns
  `Qwen3MoeRuntimeEnv`, its one-time env parsing helpers, and parser unit
  tests. The main `qwen3_moe.rs` file now imports the typed runtime snapshot
  instead of owning env parsing logic inline.
- `crates/ferrum-models/src/models/qwen3_moe_profile.rs` now owns the
  Qwen3-MoE profiling counters shared by model forward and MoE helper paths.
- `crates/ferrum-models/src/models/qwen3_moe/` now owns the main Qwen3-MoE
  implementation by operational boundary:
  - `scratch.rs`: layer state and scratch allocation;
  - `load.rs`: constructors, safetensors/GGUF loading helpers, graph-clean
    test accessors;
  - `kv.rs`: scratch and KV-cache allocation;
  - `forward_layer.rs`: legacy per-layer attention + MoE forward;
  - `prefix_cache.rs`: block-level prefix-cache helpers;
  - `prefill_decode.rs`: single-sequence prefill/decode paths;
  - `decode_batch.rs`: batched decode path;
  - `api.rs`: `DecoderOnlyLLM` adapter.
- Long repeated Qwen3-MoE hot-path call groups now use typed parameter
  objects:
  - `UnifiedLayerParams` replaces the 22-argument
    `Qwen3MoeModel::unified_forward_layer` signature;
  - `MoeForwardParams` replaces the legacy fallback MoE forward argument
    list;
  - `MoeForwardBucketedParams` replaces the bucketed MoE forward argument
    list shared by legacy, batched decode, and unified paths.
- `crates/ferrum-types/tests/codebase_shape_test.rs` locks the explicit
  Milestone H line-count limits for the four goal-owned surface files and is
  wired into CI.
- The same Rust integration test now also locks the existing model/backend
  hot-path long-signature baseline. It scans
  `crates/ferrum-models/src/{models,moe}` and
  `crates/ferrum-kernels/src/backend`, ignores low-level `extern` ABI
  declarations, and fails if a new ordinary Rust function/method/trait method
  exceeds 15 typed parameters. The current legacy baseline is intentionally
  explicit so new long signatures cannot enter silently.

## Current Line Counts

```text
 814 crates/ferrum-engine/src/continuous_engine.rs
 161 crates/ferrum-engine/src/continuous_engine/inner.rs
 681 crates/ferrum-engine/src/continuous_engine/inner/batch.rs
 185 crates/ferrum-engine/src/continuous_engine/inner/completion.rs
 519 crates/ferrum-engine/src/continuous_engine/inner/decode.rs
 445 crates/ferrum-engine/src/continuous_engine/inner/prefill.rs
 136 crates/ferrum-models/src/models/qwen3_moe.rs
 186 crates/ferrum-models/src/models/qwen3_moe/api.rs
1339 crates/ferrum-models/src/models/qwen3_moe/decode_batch.rs
 622 crates/ferrum-models/src/models/qwen3_moe/forward_layer.rs
 189 crates/ferrum-models/src/models/qwen3_moe/kv.rs
 403 crates/ferrum-models/src/models/qwen3_moe/load.rs
 481 crates/ferrum-models/src/models/qwen3_moe/prefill_decode.rs
 160 crates/ferrum-models/src/models/qwen3_moe/prefix_cache.rs
 455 crates/ferrum-models/src/models/qwen3_moe/scratch.rs
  60 crates/ferrum-models/src/models/qwen3_moe_profile.rs
 253 crates/ferrum-models/src/models/qwen3_moe_runtime.rs
1320 crates/ferrum-kernels/src/backend/traits.rs
 441 crates/ferrum-models/src/models/qwen3_moe_forward_unified.rs
 451 crates/ferrum-models/src/models/qwen3_moe_forward_unified_layer.rs
 724 crates/ferrum-kernels/src/backend/capabilities.rs
 230 crates/ferrum-kernels/src/backend/types.rs
```

`backend/traits.rs` now satisfies the Milestone H `<=1500` line-count target.
`qwen3_moe.rs` now satisfies the Milestone H `<=1500` line-count target, and
the largest Qwen3-MoE child file is `decode_batch.rs` at 1339 lines.
`qwen3_moe_forward_unified.rs` now satisfies the Milestone H `<=700`
line-count target. `continuous_engine.rs` now satisfies the explicit
Milestone H `<=1500` line-count target, and `EngineInner` is split by
request-processing stage. The most obvious Qwen3-MoE repeated hot-path
parameter groups are now typed. The remaining long backend/attention/KV launch
signatures are now visible in the Rust code-shape test baseline; reducing that
baseline is future cleanup, while adding to it requires an explicit test diff.

## Validation

Local:

```bash
cargo fmt --all -- --check
cargo check -q -p ferrum-kernels
cargo check -q -p ferrum-models
cargo check -q -p ferrum-cli
cargo test -q -p ferrum-models qwen3_moe_runtime_env -- --nocapture
cargo test -q -p ferrum-models --lib
cargo test -q -p ferrum-models --test moe_bucketed_parity_test
cargo test -q -p ferrum-engine --test continuous_batch_test
cargo test -q -p ferrum-engine continuous_engine --lib
cargo test -q -p ferrum-scheduler
cargo test -q -p ferrum-types --test codebase_shape_test -- --nocapture
git diff --check
python3 scripts/check_ferrum_env_registry.py --fail-on-registry-gap
```

All checks listed above passed after the splits. No GPU performance run was
executed for this checkpoint because the changes are Rust module ownership
splits with no kernel or runtime behavior change.

## Remaining H Gaps

- Continue shrinking the explicit long-signature baseline by converting
  repeated backend/attention/KV launch parameter groups into typed structs
  where it reduces call-site complexity.
