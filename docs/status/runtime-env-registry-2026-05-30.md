# Runtime Env Registry Status - 2026-05-30

Milestone D is not complete. This checkpoint adds a repeatable audit tool and brings the static registry to 100% coverage for the current scanned `FERRUM_*` env candidates.

## Added

- `scripts/check_ferrum_env_registry.py` scans the product/build/bench source surface for Ferrum env tokens and direct env reads.
- `docs/runtime-env-registry.tsv` is the stable registry path. It currently covers every scanned `FERRUM_*` env candidate.
- `docs/runtime-env-registry-ignore.txt` allowlists scanned `FERRUM_*` symbols that are not environment variables.
- `docs/runtime-env-registry-missing-baseline.txt` is now empty and kept only for transitional local workflows.
- `.github/workflows/ci.yml` now runs the full static registry gate:
  `python3 scripts/check_ferrum_env_registry.py --fail-on-registry-gap`.
  This gate now also fails on any unclassified hot-path direct env read.
- The registry checker now validates `type`, `scope`, `stability`, and
  `read_phase` against controlled vocabularies, and rejects
  `experimental`/`diagnostic`/`deprecated` entries that do not state a sunset
  condition. Its `--self-test` covers valid rows plus invalid type, invalid
  scope, and missing-sunset negative cases, and CI runs that self-test before
  the repository scan.
- The same checker now audits the checked-in product-facing `ferrum.toml`
  surface. It fails if `[runtime]` advertises advanced/internal selector keys
  outside the small user-facing sample set, or if the file mentions raw
  `FERRUM_*` env names.
- The JSON report includes `name_paths` for every scanned token, so registry owner rows can be audited against source locations.
- The JSON report also records hot-path direct env reads by individual call site
  with source line, parsed env name when available, and an explicit
  classification. The current residual hot direct reads are all classified as
  diagnostic/startup/test-only exceptions rather than hidden by file-level
  counts.
- The same static gate now audits process-wide env writes
  (`std::env::set_var`, `env::set_var`, `std::env::remove_var`, and
  `env::remove_var`). `--fail-on-registry-gap` fails on any unclassified
  product/runtime env write, while test fixtures and the two remaining
  compatibility bridges are explicitly classified.
- `ferrum-types::RuntimeConfigSnapshot` captures stable sorted `FERRUM_*` overrides with source/effect metadata; `/health` and bench `Env` now include it.
- `RuntimeConfigSnapshot` now supports explicit non-env entries and stable
  upserts. `ferrum serve` uses this for selected CLI runtime/profile inputs,
  so `--kv-dtype` and `--profile-*` appear in startup config artifacts with
  `source=cli`.
- `RuntimeConfigSource` now includes `memory_profile`. `ferrum serve`
  source-attributes GPU autosizer-created `FERRUM_MAX_BATCHED_TOKENS`,
  `FERRUM_KV_MAX_BLOCKS`, `FERRUM_PAGED_MAX_SEQS`, and `FERRUM_KV_CAPACITY`
  entries as `source=memory_profile` instead of reporting them as user env
  overrides.
- The CLI config schema supports a `[runtime]` surface for startup/runtime
  performance selectors: `kv_dtype`, KV/max-batch sizing, prefix cache, MoE
  graph, vLLM paged attention, vLLM-MoE, pair-id routing, greedy argmax,
  FA-layout, source FA2, diagnostic FA2 direct FFI, max model length, and MoE
  batch threshold. `ferrum serve` records these as `source=config_file`,
  bridges them into the existing startup env snapshot only when the
  corresponding env var is absent, and participates in the explicit startup
  precedence order `CLI > env > config_file > default`. The checked-in
  product-facing `ferrum.toml` intentionally lists only the small stable/common
  subset as commented examples.
- `ferrum-types::runtime_config` parser tests cover boolean, integer, path,
  and tri-state default/forced-off/forced-on env values.
- `ferrum-scheduler` now parses the prompt-token-estimate, prefill-first, active-decode-prefill-chunk, and scheduler-none-prof env switches once during `ContinuousBatchScheduler` construction; batch planning reads typed fields instead of calling env APIs.
- `ferrum-engine::ContinuousBatchEngine` now parses its chunked-prefill, prefix-cache, active-prefill-chunk, and profiling env switches once during engine construction from a startup env snapshot; iteration, prefill, unified, and background-loop paths read typed fields.
- `Qwen3MoeModel` now captures the Qwen/M3 runtime env surface once at model construction with parser tests. The core Qwen3-MoE Rust forward files now have no direct env calls in hot paths; `std::env::vars()` is used only for the startup snapshot.
- `moe::dispatch` now captures MoE profiling, vLLM-MoE routing, pair-id, workspace-zeroing, load-trace, and block-size overrides in a typed runtime config. The hot bucketed dispatch path reads cached fields; block-size policy tests use explicit config inputs instead of mutating process env.
- `moe::forward` now captures fused gate/up/SILU, host top-k, and direct-dispatch toggles in a typed runtime config. The stacked decode and batched prefill paths no longer call env APIs directly.
- `LlamaFamilyModel` batched/unified forward now captures graph, profile, trace, and greedy-argmax toggles in a typed runtime config. The batched Llama hot path no longer has direct env reads beyond startup snapshot iteration.
- `LlamaFamilyModel` single-item/prefill paths now capture KV capacity, paged-KV, paged max sequence, CUDA graph, decode profile, prefill profile, and decode-layer profile toggles in a typed runtime config.
- CUDA paged attention now captures KV capacity, paged-flash, split count, split-K, and FA2-source toggles in a typed runtime config. The CUDA paged attention hot path no longer reads env directly beyond startup snapshot iteration.
- The legacy CUDA decode runner now captures diagnostic flags, paged-KV enablement, and KV block count in a typed runtime config at runner construction.
- CUDA quant/Marlin/graph wrappers now capture Triton/vLLM-Marlin/vLLM-MoE selection, Marlin tuning switches, MoE fused/multistream dispatch, graph replay diagnostics, and vLLM paged-attn short-context selection in typed runtime configs.
- The engine builder and component registry now snapshot model-path, speculative decoding, dtype, and tensor-parallel env knobs once. Tokenizer/executor resolution and model factory setup no longer call env APIs directly.
- `ferrum run`, `ferrum bench`, and the LLM branch of `ferrum serve` no longer
  write `FERRUM_MODEL_PATH` into process env before engine construction.
  They pass `model_path` through `EngineConfig.backend_options`; `serve`
  also passes speculative decoding `spec_draft`/`spec_n` through typed backend
  options instead of writing `FERRUM_SPEC_DRAFT`/`FERRUM_SPEC_N`. The env names
  remain registered as startup compatibility aliases for external callers.
- TTS, HuggingFace download/source resolution, and LLM batch profile flags now use typed one-time env snapshots with parser tests.
- Remaining CUDA module-level backend selectors (`FERRUM_MOE_STREAMS`, `FERRUM_CUDA_MAX_KV`, `FERRUM_CUDA_DEVICE`) and fused-attention CPU/Metal selectors now use cached runtime config.
- The vLLM-MoE Marlin C++ bridge now reads its diagnostic logging and thread/block override env knobs through a single process-static runtime config helper instead of per-call `getenv` checks.
- `ferrum serve` now accepts `--runtime-preset m3_qwen3_30b_a3b_int4`, and
  `[runtime].preset` can select the same preset from config files. The preset
  materializes the common M3 runtime defaults before startup snapshot capture,
  while explicit runtime config fields, env vars, and CLI flags can still
  override those defaults. When no preset is explicitly configured, `serve`
  now infers the M3 preset from Qwen3-30B-A3B model metadata and materializes
  those defaults through the same typed preset entries before the legacy MoE
  graph compatibility setter would otherwise be needed.
- Non-preset Qwen3-MoE `ferrum serve` now preserves the historical graph-clean
  MoE defaults through typed runtime entries (`FERRUM_MOE_GRAPH`, and when
  built, `FERRUM_VLLM_MOE` / `FERRUM_VLLM_MOE_PAIR_IDS`) before the startup
  snapshot is resolved. Those defaults now appear as `source=default` in
  effective config / decision artifacts instead of being introduced by a
  hidden process-wide env setter.
- The MoE graph default resolver is now a shared typed CLI helper. `ferrum run`
  no longer calls the legacy `apply_moe_graph_default()` setter; it resolves
  the same defaults from the current startup snapshot, preserves explicit
  env/config overrides such as `FERRUM_MOE_GRAPH=0`, and only then materializes
  missing compatibility env values for older backend readers.
- `source_resolver` chat-profile defaults for `ferrum run` now resolve to
  typed runtime entries before compatibility materialization. Dense, MoE, and
  explicit-override tests cover `FERRUM_KV_CAPACITY`,
  `FERRUM_METAL_PAGED_KV`, single-user paged KV sizing, and MoE batching
  defaults without local direct `set_var` calls.
- GPU autosizing now detects user/config overrides from a startup
  `RuntimeConfigSnapshot` and emits `source=memory_profile` typed entries for
  `FERRUM_MAX_BATCHED_TOKENS`, `FERRUM_KV_MAX_BLOCKS`,
  `FERRUM_PAGED_MAX_SEQS`, and `FERRUM_KV_CAPACITY` before compatibility
  materialization. The autosizer no longer owns direct FERRUM env reads/writes.
- `scripts/m3_ab_runner.py` now passes the named M3 preset to the product
  server through `--runtime-preset` instead of injecting the common M3
  `FERRUM_*` env bundle. It keeps only path-like preset env such as `HF_HOME`
  and uses server-written `effective_config.json` as the completed manifest
  runtime snapshot.
- Native structured profile emission now uses typed runner/server
  `ProfileSinkConfig` plumbing for Rust emitters and the vLLM-MoE C++ config
  emitter through a typed C ABI sink. Registered `FERRUM_PROFILE_*` names remain
  as a backwards-compatible fallback, but `ferrum serve --profile-*` no longer
  exports them for the normal native profile path.
- Runner summaries now include per-row `env_hash`,
  `runtime_config_entry_count`, and a machine-readable
  `runtime_config_diff_vs_baseline` object for A/B cases. The diff records
  added/removed/changed runtime config entries with source and effect
  classification.
- Runner-side runtime config snapshots now contain only `FERRUM_*` entries.
  Completed artifacts copy those entries from server-written
  `effective_config.json`; the artifact validator rejects a manifest snapshot
  whose env hash or entries differ from the server artifact.
- Runner cleanup artifacts now include global process hygiene status for
  residual `ferrum`, `bench-serve`, `cargo`, `nvcc`, and `vllm` processes so
  publishable artifacts can prove the post-run process state was clean.
- The backend timer helper now takes an already-resolved boolean gate instead
  of reading `FERRUM_DECODE_OP_PROFILE` inside token/layer probes. Qwen3-MoE
  and MoE forward paths pass cached typed runtime config fields into the
  timer.
- CUDA MoE route dump, CUDA TP/rank collectives, FA2 direct-FFI shim path,
  Metal attention dispatch policy, Metal mmap/capture/dtype policy, Metal
  quant profiling, and the Qwen3-TTS Candle fallback now resolve their env
  knobs once through cached startup/runtime helpers instead of per-call direct
  env reads.
- `ferrum-types::FerrumConfigBuilder` adds a typed selector decision trace
  over the registered env surface. It now includes explicit
  `FERRUM_MAX_MODEL_LEN` validation so requested serving length overrides are
  checked against model metadata and KV token capacity. `/health` exposes this
  as `auto_config` next to the runtime config snapshot.

## Current Audit

Run:

```bash
python3 scripts/check_ferrum_env_registry.py --json
python3 scripts/check_ferrum_env_registry.py --self-test
python3 scripts/check_ferrum_env_registry.py --fail-on-registry-gap
```

Current local scan:

| metric | value |
|---|---:|
| files scanned | 582 |
| unique `FERRUM_*` tokens | 153 |
| unique standalone candidates | 152 |
| direct env read calls | 94 |
| process env write calls | 42 |
| classified process env write calls | 42 |
| unclassified process env write calls | 0 |
| hot-path unique `FERRUM_*` tokens | 121 |
| hot-path direct env read calls | 4 |
| classified hot-path direct env read calls | 4 |
| unclassified hot-path direct env read calls | 0 |
| ignored non-env symbols | 5 |
| registry entries | 147 |
| registry coverage | 147 / 147 env candidates |
| unregistered baseline backlog | 0 |
| new unregistered names versus baseline | 0 |
| product `ferrum.toml` runtime sample keys | `kv_dtype`, `kv_max_blocks`, `max_batched_tokens`, `moe_graph`, `paged_max_seqs`, `prefix_cache` |
| product `ferrum.toml` raw `FERRUM_*` mentions | 0 |
| product `ferrum.toml` surface errors | 0 |

The hot-path name count is above the original `116`-name snapshot because the structured profile metadata bridge adds diagnostic `FERRUM_PROFILE_*` names in Rust and the vLLM-MoE C++ bridge. The direct-read scanner now requires an actual function call and excludes `std::env::vars()` snapshot iteration, so the current counts are `94` direct reads whole-tree and `4` in hot paths. The hot-path direct-read count is well below the Milestone D quantitative target of `<=26`. The whole-tree token counts are now `153` token names, `152` standalone env candidates, and `147` registered env candidates after explicit non-env ignores because recent local work added FA2/API/profile development scripts, runtime gates, and explicit requested max-model-len validation after the original `143`-name snapshot.

The classified residual hot-path direct-read call sites are:

- `crates/ferrum-kernels/kernels/marlin_cuda_kernel.cu`: two diagnostic
  `FERRUM_MARLIN_TILE` override reads in the legacy Marlin C++ path.
- `crates/ferrum-kernels/kernels/vllm_marlin_moe/ops.cu`: one process-static
  helper uses `std::getenv` internally for the vLLM-MoE diagnostic/tuning
  runtime config.
- `crates/ferrum-kernels/src/backend/metal/q4_k_moe_id_gemv_batched.rs`: one
  ignored manual Metal capture test reads `MTL_CAPTURE_ENABLED`.

The scanner reports `FERRUM_2M` as an embedded token, not a standalone env candidate; it comes from a microbench label string rather than a real runtime knob.

The classified process-env write call sites are:

- `34` test-only fixture setup/cleanup calls in Rust test modules or
  integration tests.
- `7` profile sink compatibility bridge calls in
  `crates/ferrum-bench-core/src/profile.rs`. Normal `ferrum serve`
  structured-profile wiring uses typed `--profile-*` arguments; this bridge
  remains for backwards-compatible `FERRUM_PROFILE_*` propagation.
- `1` CLI runtime compatibility bridge call in
  `crates/ferrum-cli/src/runtime_env.rs`, which materializes typed
  `RuntimeConfigEntry` defaults only for backend paths that have not yet been
  converted to typed config.

## Config Diff Artifact

Remote smoke with the migrated FA-layout wrapper:

```bash
cd /workspace/ferrum-codex-clean
OUT_ROOT=/workspace/m3-runner-config-diff-smoke2-20260530_003637 \
  BUILD=0 CONCURRENCY=1 NUM_PROMPTS=1 WARMUP_REQUESTS=0 \
  REPEATS=1 PORT_BASE=18780 \
  bash scripts/m3_fa_layout_varlen_ab.sh
python3 scripts/m3_validate_runner_artifact.py \
  /workspace/m3-runner-config-diff-smoke2-20260530_003637
```

Evidence:

- `summary.json.runtime_config_diff_vs_baseline.fa_layout.changed` identified
  the only intended case diff: `FERRUM_FA_LAYOUT_VARLEN` from `0` to `1`,
  source `script_case`, effect `performance`.
- Summary rows included stable env hashes for both rows and
  `runtime_config_entry_count=13`.
- The runner and artifact validator self-tests cover the diff schema.

## Remaining D Gaps

- Continue typed config coverage for the remaining low-count hot-path surfaces
  and non-hot CLI/build surfaces.
- Continue shrinking the remaining non-hot compatibility materialization bridge
  and model/backend runtime configs that still consume the old env surface.
- Continue extending first-class config-file runtime knobs for non-M3
  surfaces and any remaining diagnostics that need source-attributed
  artifacts.
- Keep advanced/experimental config-file fields available for artifacts while
  keeping checked-in `ferrum.toml` as a small user-facing sample surface.
- Replace the remaining classified C++ hot-path direct env reads with typed
  launch parameters when those legacy/diagnostic paths become product-critical.
- Remove the backwards-compatible `FERRUM_PROFILE_*` metadata fallback after any
  external profile users have moved to `ferrum serve --profile-*` or another
  typed configuration path.
- Extend named presets beyond the initial M3 server/runner surface and wire
  them into future artifact/config summaries.
