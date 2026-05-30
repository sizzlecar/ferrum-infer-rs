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
  trace. The old runner-side selector scaffold has been removed from the
  runtime harness.
- Runtime config entry sources now propagate into selector decisions. Decision
  traces preserve `env`, `cli`, `config_file`, `script_case`, and
  `memory_profile` source attribution instead of marking every present key as
  an env override.
- `RuntimeConfigSnapshot` now has stable explicit-entry/upsert helpers for
  non-env sources. `ferrum serve` uses them to carry CLI runtime/profile
  inputs such as `--kv-dtype` and `--profile-*` into the startup
  `effective_config.json` snapshot with `source=cli`.
- The CLI config schema now has a `[runtime]` config surface for
  M3/auto-config selectors: `kv_dtype`, KV/max-batch sizing, prefix cache, MoE
  graph, vLLM paged attention, vLLM-MoE, pair-id routing, greedy argmax,
  FA-layout, source FA2, diagnostic FA2 direct FFI, max model length, and MoE
  batch threshold. `ferrum serve` bridges these config-file values into the
  existing startup env snapshot only when the corresponding env var is absent,
  records them with `source=config_file`, and uses an explicit `CLI > env >
  config_file > default` precedence rule for both runtime behavior and startup
  artifacts. The checked-in product-facing `ferrum.toml` intentionally lists
  only the small stable/common subset as commented examples, and the env
  registry checker now enforces that it does not advertise advanced selector
  keys or raw `FERRUM_*` env names.
- `ferrum serve --runtime-preset m3_qwen3_30b_a3b_int4` and
  `[runtime].preset = "m3_qwen3_30b_a3b_int4"` now select the product-server
  M3 runtime preset. The preset materializes the common M3 startup defaults
  before env snapshot capture and records them with CLI/config-file source
  attribution; explicit runtime config fields, env vars, and CLI flags still
  override those defaults.
- `ferrum serve` also infers the same M3 runtime preset from Qwen3-30B-A3B
  model metadata when no preset is explicitly configured. Those inferred
  defaults are materialized through the typed preset entries with
  `source=default`, so the M3 serve path no longer gets those startup
  defaults from the legacy `apply_moe_graph_default()` env setter.
- Non-preset Qwen3-MoE `ferrum serve` now uses typed default entries for the
  historical graph-clean MoE defaults instead of calling
  `apply_moe_graph_default()`. This keeps the compatibility values visible to
  the builder, effective-config artifact, and decision trace before they are
  materialized for older backend readers.
- The graph-clean MoE default path is now a shared typed CLI resolver.
  `ferrum run` consumes that resolver from a startup snapshot instead of
  calling the old process-wide setter, so explicit overrides are preserved
  before compatibility env values are materialized for model/backend readers.
- `source_resolver` chat-profile defaults now resolve into typed entries
  before compatibility env materialization, covering dense/MoE KV capacity,
  Metal paged KV, single-user paged sizing, and MoE batching defaults.
- GPU autosizing now uses the startup runtime snapshot for override detection
  and emits `source=memory_profile` typed entries before compatibility
  materialization, instead of directly reading/writing the sizing env keys.
- `ferrum serve` now records GPU-memory autosizer-created runtime keys
  (`FERRUM_MAX_BATCHED_TOKENS`, `FERRUM_KV_MAX_BLOCKS`,
  `FERRUM_PAGED_MAX_SEQS`, `FERRUM_KV_CAPACITY`) with
  `source=memory_profile` when they are introduced during startup, so startup
  artifacts distinguish memory-profile sizing from user env overrides.
- The LLM `serve` path now passes the resolved target model path and
  speculative decoding draft settings through typed `EngineConfig`
  backend options instead of process-wide `FERRUM_MODEL_PATH` /
  `FERRUM_SPEC_DRAFT` / `FERRUM_SPEC_N` writes. Those env names remain
  available only as compatibility aliases for external engine-builder callers.
- The CUDA `serve` startup path now performs a runtime hardware probe with
  `nvidia-smi`/`nvcc` when available. It fills `HardwareCapabilities` with
  CUDA runtime/toolkit version, compute capability, total VRAM, and SM count
  (`multiprocessor_count` when reported; conservative RTX 4090 fallback
  otherwise) instead of leaving those fields as empty compile-feature stubs.
- The M3 selector now uses `HardwareCapabilities.backend` before selecting
  CUDA-only defaults. If compiled CUDA fast-path features are present but the
  resolved backend is not CUDA, the preset falls back to compatible legacy
  prefill/decode/MoE/readback selections instead of picking an unusable
  compiled path. Forced CUDA-only overrides for vLLM paged attention,
  vLLM-MoE, GPU greedy argmax, and FA2 now fail fast on non-CUDA backends;
  source/direct FA2 also rejects known CUDA compute capabilities below `8.0`.
- `scripts/m3_ab_runner.py` now passes the M3 preset through
  `--runtime-preset`, so the migrated wrappers no longer inject the common M3
  `FERRUM_*` bundle as process env. Completed runner manifests now copy the
  runtime snapshot from server-written `effective_config.json`, and native
  profile events default to the server's resolved env hash/runtime flags.
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
cargo test -q -p ferrum-cli runtime_env -- --nocapture
cargo test -q -p ferrum-cli source_resolver -- --nocapture
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

- `ferrum-types` auto-config tests passed: `13 passed`, including
  non-env source attribution for config-file, CLI, and script-case entries plus
  locked artifact schema-shape coverage and requested max-model-len budget
  validation. The M3 preset test also asserts prefix cache remains disabled by
  the Rust selector unless explicitly overridden. New hardware-selector tests
  cover non-CUDA backend fallback, forced CUDA-only override rejection, and
  CUDA compute capability parsing for FA2 eligibility.
- `ferrum-types` runtime-config tests passed: `6 passed`, including stable
  upsert/override behavior for CLI-sourced entries.
- `ferrum-cli` serve tests passed: `19 passed`, covering CLI-sourced runtime
  entries for `--kv-dtype`, `--runtime-preset`, and profile sink arguments
  plus explicit runtime source precedence for config-file, env, and CLI
  inputs. The serve tests also verify that config-file values bridged into the
  env snapshot keep `source=config_file` in startup artifacts, and that
  model-inferred M3 preset defaults keep `source=default` instead of being
  reported as env overrides after materialization. New serve tests verify
  autosizer-added keys are source-attributed as `memory_profile`, CUDA
  hardware probe output is parsed into `HardwareCapabilities`, and non-preset
  Qwen3-MoE serve defaults are represented as typed `source=default` entries
  while preserving config-file overrides.
- `ferrum-cli` config tests passed: `8 passed`, including config-file runtime
  source attribution, `[runtime].preset` parsing, and missing `[runtime]`
  backwards compatibility.
- `ferrum-cli` runtime-env tests passed: `3 passed`, covering missing-default
  insertion, `FERRUM_MOE_GRAPH=0` override preservation, and graph-enabled
  snapshot completion.
- `ferrum-cli` source-resolver tests passed: `3 passed`, covering chat-profile
  typed defaults for dense safetensors, MoE safetensors, and explicit
  snapshot overrides.
- `ferrum-server` health test passed and verifies `auto_config` is present
  with either a decision list or an explicit resolver error.
- `cargo check` passed for `ferrum-types`, `ferrum-server`, and `ferrum-cli`.
- Runner and artifact-validator self-tests passed after switching the runtime
  case path to server-written config artifacts and removing the runner-side
  selector implementation. Migrated wrapper syntax checks and
  `VALIDATE_ONLY=1` config checks passed for the current migrated set.
- Env registry stayed at full coverage with no new unregistered names:
  `147/147`, hot direct env reads `4`.

## Remaining E Gaps

- The builder is not yet the sole owner of runtime defaults. The serve path no
  longer gets M3 or non-preset Qwen3-MoE startup defaults from the old MoE
  graph default setter, and `ferrum run` now uses the same typed resolver for
  that graph-clean default. Source resolver chat-profile defaults are also
  typed before compatibility materialization, and autosizing now produces
  typed `memory_profile` entries. Model/backend runtime configs still consume
  parts of the old env surface.
- CLI/config-file/script-case/memory-profile source attribution is represented
  in the builder and decision trace. `ferrum serve` now carries the named M3
  runtime preset, selected CLI runtime/profile inputs, config-file runtime
  fields, and autosizer-created runtime keys into the startup snapshot. Broader
  non-M3 and diagnostic config-file coverage still needs expansion.
- CUDA hardware capability data is no longer only compiled-feature/device
  based on the `serve` startup path. Selectors now use backend and known
  compute capability for CUDA-only attention/MoE/sampling compatibility.
  Remaining work is to validate the runtime probe on the GPU pod and make
  selectors consume probed VRAM/SM data for memory and throughput policy, not
  only backend compatibility. Startup autosizer outputs are now
  source-attributed as `memory_profile`, but this is not yet a full
  memory-capability selector.
- The M3 preset selector is now exposed through the Rust startup path and used
  by the migrated runner. Server-written effective-config data now owns
  completed-run runtime snapshots, and the runner no longer has a parallel
  selector implementation.
- Scheduler admission-policy overrides selected by the builder are now
  consumed through typed `SchedulerConfig` fields in `ferrum run`,
  `ferrum bench`, and `ferrum serve` instead of relying on the continuous
  scheduler to read process env at construction time.
- KV block budget and max-batched-token selections are also applied from the
  startup/runtime snapshot into typed `EngineConfig`, so the default config
  constructors no longer hide process-env reads for those autosized values.
- No new GPU performance claim is made by this checkpoint. Default-path
  equivalence for the Rust auto-config path still needs the Milestone I
  correctness and non-regression packet.
