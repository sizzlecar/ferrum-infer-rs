# Auto Config Status - 2026-05-30

Milestone E is not complete. This checkpoint moves the selector decision trace
from a runner-only artifact toward a Rust startup control-plane surface.

## Added

- `ferrum-types::FerrumConfigBuilder` resolves a typed
  `ResolvedFerrumConfig` from:
  - `RuntimeConfigSnapshot`
  - `ModelCapabilities`
  - `HardwareCapabilities`
  - `WorkloadProfile`
- Capability structs now represent the Milestone E inputs:
  `ModelCapabilities`, `MoeCapabilities`, `HardwareCapabilities`,
  `CompiledKernelFeatures`, and `WorkloadProfile`.
- The decision trace covers the required M3 selector surface:
  - attention prefill/mixed backend
  - attention decode backend
  - MoE implementation
  - MoE graph policy
  - KV block count
  - max sequences
  - max batched tokens
  - prefix cache policy
  - scheduler admission policy
  - sampling/readback path
- The M3 workload preset name is defined in Rust as
  `m3_qwen3_30b_a3b_int4`.
- Invalid override tests now cover more than 10 combinations, including:
  graph with graph-unsafe/disabled MoE, FA2 without source-built support,
  direct FFI without shim support, FA-layout without vLLM paged attention,
  pair-id routing without vLLM MoE, BF16/FP8 unsupported dtype selections,
  zero KV/max-seq values, `max_batched_tokens < max_sequences`,
  invalid `FERRUM_MAX_MODEL_LEN`, model length above metadata, and KV token
  capacity smaller than the requested max model length.
- `/health` now includes both the existing runtime config snapshot and an
  `auto_config` object. The CLI `serve` path resolves startup auto-config once
  from the selected device, detected model metadata, compiled feature flags,
  and current runtime env snapshot, then attaches it to the server state.
- `ferrum serve` now accepts `--effective-config-json <path>` and
  `--decision-trace-jsonl <path>`. When set, startup writes the product-server
  effective config artifact and JSONL decision trace directly.
- `scripts/m3_ab_runner.py` passes those paths to `ferrum serve` for every
  case and records `auto_config_decision_count` from the server-written JSONL
  trace instead of relying only on the runner-side selector scaffold.
- Runtime config entry sources now propagate into selector decisions. Decision
  traces preserve `env`, `cli`, `config_file`, and `script_case` source
  attribution instead of marking every present key as an env override.
- `RuntimeConfigSnapshot` now has stable explicit-entry/upsert helpers for
  non-env sources. `ferrum serve` uses them to carry CLI runtime/profile
  inputs such as `--kv-dtype` and `--profile-*` into the startup
  `effective_config.json` snapshot with `source=cli`.
- `ferrum.toml` now has a `[runtime]` config surface for `kv_dtype`,
  `kv_max_blocks`, `paged_max_seqs`, `max_batched_tokens`, `prefix_cache`, and
  `moe_graph`. `ferrum serve` bridges these config-file values into the
  existing startup env snapshot only when the corresponding env var is absent,
  records them with `source=config_file`, and uses an explicit
  `CLI > env > config_file > default` precedence rule for both runtime
  behavior and startup artifacts.
- The runner artifact validator now schema-checks server-written
  `effective_config.json` entries, including sorted `FERRUM_*` keys,
  non-empty `affects`, allowed source/effect values, and decision parity with
  `decision_trace.jsonl`.
- Decision-trace validation now rejects unknown source values, non-`FERRUM_*`
  source keys, empty candidate/effect lists, and malformed rejected-candidate
  reasons.
- `ferrum-types` now has a locked artifact-shape unit test that parses
  `effective_config_document()` and `decision_trace_jsonl()` as real artifacts
  and checks sorted entries, source/effect vocabulary, non-empty selector
  fields, `FERRUM_*` source keys, and exact parity between embedded decisions
  and JSONL trace lines.

## Validation

Local:

```bash
cargo test -q -p ferrum-types auto_config -- --nocapture
cargo test -q -p ferrum-types runtime_config -- --nocapture
cargo check -q -p ferrum-types
cargo check -q -p ferrum-server
cargo check -q -p ferrum-cli
cargo test -q -p ferrum-cli config -- --nocapture
cargo test -q -p ferrum-cli commands::serve -- --nocapture
cargo test -q -p ferrum-server route_health_includes_runtime_config_snapshot -- --nocapture
python3 -m py_compile scripts/m3_ab_runner.py scripts/m3_validate_runner_artifact.py
python3 scripts/m3_ab_runner.py --self-test
python3 scripts/m3_validate_runner_artifact.py --self-test
MODEL_DIR=/tmp BIN=/bin/echo OUT_ROOT=/tmp/m3-runner-validate-fa-layout \
  VALIDATE_ONLY=1 bash scripts/m3_fa_layout_varlen_ab.sh
MODEL_DIR=/tmp BIN=/bin/echo OUT_ROOT=/tmp/m3-runner-validate-route \
  VALIDATE_ONLY=1 bash scripts/m3_route_unified_profile.sh
cargo fmt --all -- --check
git diff --check
python3 scripts/check_ferrum_env_registry.py --fail-on-registry-gap
```

Evidence:

- `ferrum-types` auto-config tests passed: `10 passed`, including
  non-env source attribution for config-file, CLI, and script-case entries plus
  locked artifact schema-shape coverage and requested max-model-len budget
  validation. The M3 preset test also asserts prefix cache remains disabled by
  the Rust selector unless explicitly overridden.
- `ferrum-types` runtime-config tests passed: `6 passed`, including stable
  upsert/override behavior for CLI-sourced entries.
- `ferrum-cli` serve tests passed: `6 passed`, covering CLI-sourced runtime
  entries for `--kv-dtype` and profile sink arguments plus explicit runtime
  source precedence for config-file, env, and CLI inputs. The serve tests also
  verify that config-file values bridged into the env snapshot keep
  `source=config_file` in startup artifacts.
- `ferrum-cli` config tests passed: `6 passed`, including config-file
  runtime source attribution and missing `[runtime]` backwards
  compatibility.
- `ferrum-server` health test passed and verifies `auto_config` is present
  with either a decision list or an explicit resolver error.
- `cargo check` passed for `ferrum-types`, `ferrum-server`, and `ferrum-cli`.
- Runner and artifact-validator self-tests passed after switching the runtime
  case path to server-written config artifacts.
- Env registry stayed at full coverage with no new unregistered names:
  `147/147`, hot direct env reads `4`.

## Remaining E Gaps

- The builder is not yet the sole owner of runtime defaults. Existing
  startup env setters and model/backend runtime configs still consume parts
  of the old env surface.
- CLI/config-file/script-case source attribution is represented in the
  builder and decision trace. `ferrum serve` now carries selected CLI
  runtime/profile inputs plus a small set of config-file runtime fields into
  the startup snapshot. Broader config-file runtime coverage still needs
  expansion beyond this initial startup surface.
- Hardware capability data is currently compiled-feature/device based; it does
  not yet include a real startup memory profile or CUDA capability probe.
- The M3 preset selector is wired for exact Qwen3-MoE 30B-A3B metadata, but
  the old script preset and Rust startup preset still need to converge.
- No new GPU performance claim is made by this checkpoint. Default-path
  equivalence for the Rust auto-config path still needs the Milestone I
  correctness and non-regression packet.
