# Repository Guidelines

## Read First: Non-Negotiable Rules

- Do not claim a task is complete, release-ready, or performance-ready unless the required gate produced an artifact directory and the final validator printed the exact PASS line.
- Do not make performance claims without same-hardware evidence, benchmark command lines, git SHA, dirty status, binary SHA256 when available, runtime/config evidence, and saved artifacts.
- Validate both product entrypoints when a change can affect user-visible behavior: `ferrum run` and `ferrum serve`.
- Do not validate product behavior with hidden environment-variable combinations that users will not know to set. Required product behavior must be wired through typed defaults, CLI/config options, or documented presets.
- Correctness gates must pass before performance measurements are treated as evidence.
- Official accelerator release evidence must cover both correctness and performance on each shipped accelerator backend, and must include at least one Llama 8B-class dense model in addition to the Qwen3-30B-A3B MoE/GPTQ model.
- Paid GPU work requires a stated lane, expected runtime/cost, stop condition, correctness gate, and performance command before starting.
- Prefer small, surgical changes. Do not combine release gate changes, kernel/model changes, and large repository cleanup in the same patch unless the goal explicitly asks for that.

## Project Structure

- This repository is a Rust workspace. Root configuration lives in `Cargo.toml`; workspace crates live under `crates/`.
- Core contracts are in `crates/ferrum-types` and `crates/ferrum-interfaces`.
- Runtime/product implementations live in crates such as `ferrum-engine`, `ferrum-models`, `ferrum-kernels`, `ferrum-quantization`, `ferrum-server`, and `ferrum-cli`.
- Bench schema, aggregation, and report contracts live in `crates/ferrum-bench-core`.
- Integration tests are primarily in `crates/*/tests`.
- Release and regression gate scripts live in `scripts/release/`; CUDA/M3 runner infrastructure remains in `scripts/m3_ab_runner.py` and related M3 scripts.
- Local runtime defaults are in `ferrum.toml`.

## Build, Test, and Development Commands

- `cargo check --workspace --all-targets` — fast compile validation across all crates/targets.
- `cargo test --workspace --all-targets` — required source gate before PR-ready or release-ready claims.
- `cargo build --workspace` — full workspace build.
- `cargo fmt --all -- --check` — formatting check.
- `cargo clippy --workspace --all-targets -- -A warnings` — advisory lint pass matching CI behavior.
- `cargo run -p ferrum-cli -- list` — run the CLI crate locally; swap `list` for `pull`, `run`, `serve`, `bench-serve`, etc.
- macOS Metal test build: `cargo build --release -p ferrum-cli --features metal --tests`.
- CUDA release build: `cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`.

## Active G0 Gate Scripts

These scripts are the current source of truth for G0 release validation.

- `scripts/release/g0_source_gate.sh`
  - Lanes: `unit`, `metal`, `cuda-smoke`, `cuda-full`, `cuda-llama-dense`, `all-source`.
  - Required PASS lines:
    - `G0 SOURCE unit PASS: <out_root>`
    - `G0 SOURCE metal PASS: <out_root>`
    - `G0 SOURCE g0_cuda4090_smoke PASS: <out_root>`
    - `G0 SOURCE g0_cuda4090_full PASS: <out_root>`
    - `G0 SOURCE g0_cuda4090_llama_dense PASS: <out_root>`
    - `G0 SOURCE all-source PASS: <out_root>`
- `scripts/release/validate_metal_readme_regression.py`
  - Validates artifacts from `scripts/metal_readme_regression.py`.
  - Required PASS line: `METAL README GATE PASS: <out_dir>`.
- `scripts/release/release_binary_gate.py`
  - Modes: `metal-tarball`, `cuda-tarball`, `homebrew-metal`, `homebrew-cuda-fetch`.
  - Required PASS lines:
    - `METAL TARBALL GATE PASS: <out_dir>`
    - `CUDA TARBALL GATE PASS: <out_dir>`
    - `HOMEBREW METAL GATE PASS: <out_dir>`
    - `HOMEBREW CUDA FETCH GATE PASS: <out_dir>`
- `scripts/release/g0_release_summary.py`
  - Aggregates required G0 gate artifacts.
  - Required final release PASS line: `G0 RELEASE PASS: <root>`.
- `scripts/release/inventory_tree.py`
  - Required before moving, archiving, or deleting files under `crates/`, `docs/`, or `scripts/`.
  - Required PASS line: `INVENTORY PASS: <out_file>`.
- `scripts/release/selftest_g0_validators.py`
  - Unit/self-test for release validators.
  - It is run by `g0_source_gate.sh unit`.

## G0 Source Gate Policy

`cargo test --workspace --all-targets` is always the first source gate.

Use `scripts/release/g0_source_gate.sh` instead of private one-off gate wrappers:

```bash
scripts/release/g0_source_gate.sh unit <out_root>
scripts/release/g0_source_gate.sh metal <out_root>
scripts/release/g0_source_gate.sh cuda-smoke <out_root>
scripts/release/g0_source_gate.sh cuda-full <out_root>
scripts/release/g0_source_gate.sh cuda-llama-dense <out_root>
scripts/release/g0_source_gate.sh all-source <out_root>
```

Lane rules:

- `unit`
  - Runs workspace tests, release script Python compilation, shell syntax checks, and validator self-tests.
  - Use this for every PR-like source change.
- `metal`
  - Builds release Metal tests.
  - Runs `scripts/metal_readme_regression.py`.
  - Runs `scripts/release/validate_metal_readme_regression.py`.
  - Use this for source releases on macOS and for changes touching CLI/server/model/tokenizer/runtime defaults that can affect Metal.
- `cuda-smoke`
  - Builds the CUDA release binary with `cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`.
  - Expands `scripts/release/configs/g0_cuda4090_smoke.json`.
  - Runs `scripts/m3_ab_runner.py` and `scripts/m3_validate_runner_artifact.py`.
  - Use this for CUDA release candidates and backend-adjacent smoke validation.
- `cuda-full`
  - Uses `scripts/release/configs/g0_cuda4090_full.json`.
  - Covers c=1/4/16/32 with repeats and default-path performance regression checks.
  - Use this before official CUDA release-ready claims or when default-path performance/runtime behavior changes.
- `cuda-llama-dense`
  - Uses `scripts/release/configs/g0_cuda4090_llama_dense.json`.
  - Covers a Llama 8B-class dense CUDA model with `ferrum run`, `ferrum serve`, streaming usage, and `bench-serve --fail-on-error --require-ci`.
  - Use this as required supplemental official CUDA release evidence; it does not replace Qwen3 MoE/GPTQ CUDA full.
- `all-source`
  - Runs `unit`.
  - Runs `metal` only on macOS.
  - Does not automatically run CUDA.

Regression selection rules:

- If a change touches shared model code, scheduler/admission, runtime defaults, tokenizer/template, CLI `run`, CLI `serve`, OpenAI server API, sampler, or response formatting, regress both Metal and CUDA before release-ready claims.
- Backend-local CUDA changes require CUDA smoke/full as appropriate plus at least a Metal smoke when shared code paths may be affected.
- Backend-local Metal changes require Metal full plus at least CUDA smoke when shared code paths may be affected.
- A passing server-only gate is not evidence that `ferrum run` works. A passing `run` gate is not evidence that OpenAI-compatible serving works.

## CUDA G0 Policy

- CUDA G0 gates target exactly one RTX 4090.
- Official CUDA release validation must include at least two same-hardware architecture lanes:
  - `Qwen/Qwen3-30B-A3B-GPTQ-Int4` for MoE/GPTQ behavior.
  - A Llama 8B-class dense model for dense transformer behavior.
- If the named G0 CUDA configs do not cover the full official release model matrix, add a supplemental release artifact with its own PASS line before making any CUDA release-ready claim.
- The smoke config is `scripts/release/configs/g0_cuda4090_smoke.json`.
  - It covers c=1 and c=32.
  - It is diagnostic/smoke evidence, not a full release performance claim.
- The full config is `scripts/release/configs/g0_cuda4090_full.json`.
  - It covers c=1/4/16/32.
  - It is the required default-path CUDA performance gate before official CUDA release-ready claims.
- Do not broaden paid CUDA G0 beyond one RTX 4090 or beyond the official Qwen3 MoE/GPTQ plus Llama dense matrix without explicit user approval.
- Do not rerun expensive full sweeps repeatedly after a failure. Collect the failing artifact, stop extra processes, inspect the failure, and propose a targeted fix/gate.

## Metal G0 Policy

- Metal source release validation must use:
  1. `scripts/metal_readme_regression.py`
  2. `scripts/release/validate_metal_readme_regression.py`
- The Metal validator is the hard gate. The runner alone is not enough release evidence.
- Metal source release validation is both a correctness gate and a README performance gate. Do not call Metal release-ready unless the validator accepts the performance rows for the advertised models.
- Official Metal release validation must include at least one Llama 8B-class dense model and the Qwen3-30B-A3B MoE model when both are shipped or advertised.
- Metal evidence must include `ferrum run` multi-turn correctness, `serve` Paris/multi-turn/stream correctness, throughput rows, completed/failed request counts, and log scans.
- Metal gate failures must point to the exact model/cell/artifact path before any fix is claimed complete.

## Release Asset Gate Policy

Source gates do not validate official release assets. Before creating an official release tag, manually stage production-path Metal and CUDA binaries with `workflow_dispatch` and `publish_release=false`, then run the appropriate tarball gates against those staged assets. After publishing or staging release assets, run the appropriate release binary gates.

Metal tarball:

```bash
python3 scripts/release/release_binary_gate.py metal-tarball \
  --version <VERSION> \
  --out docs/release/g0/<VERSION>/metal-tarball
```

For a staged pre-tag tarball, add `--asset-path <PATH_TO_TARBALL>` and provide the adjacent `.sha256` file or `--sha256 <SHA256>`.

CUDA tarball:

```bash
python3 scripts/release/release_binary_gate.py cuda-tarball \
  --version <VERSION> \
  --out docs/release/g0/<VERSION>/cuda-tarball
```

For a staged pre-tag tarball, add `--asset-path <PATH_TO_TARBALL>` and provide the adjacent `.sha256` file or `--sha256 <SHA256>`.

Homebrew Metal:

```bash
python3 scripts/release/release_binary_gate.py homebrew-metal \
  --version <VERSION> \
  --out docs/release/g0/<VERSION>/homebrew-metal
```

Homebrew CUDA fetch:

```bash
python3 scripts/release/release_binary_gate.py homebrew-cuda-fetch \
  --version <VERSION> \
  --out docs/release/g0/<VERSION>/homebrew-cuda-fetch
```

Release asset gates must check version, asset integrity, CLI behavior, serve behavior, logs, and platform-specific dependencies. CUDA tarball gates must reject missing libraries and accidental Python/Torch/vLLM runtime linkage.

Final G0 release summary:

```bash
python3 scripts/release/g0_release_summary.py docs/release/g0/<VERSION>
```

No official release-ready claim is valid unless the final line is:

```text
G0 RELEASE PASS: docs/release/g0/<VERSION>
```

## bench-serve Policy

`ferrum bench-serve` is the canonical HTTP performance client for `/v1/chat/completions`. Do not add a second release throughput path for the same endpoint.

Release performance claims must use:

```bash
ferrum bench-serve ... \
  --fail-on-error \
  --require-ci \
  --seed 9271 \
  --n-repeats 3
```

Smoke runs may omit `--require-ci`, but must still use `--fail-on-error`.

Benchmark correctness rules:

- Streaming requests must send `stream_options.include_usage=true`.
- A streaming request is successful only when it receives exactly one `data: [DONE]`, emits at least one output token, and has no stream or malformed-SSE JSON error.
- Release performance artifacts must require `output_token_count_source == "usage"` unless the goal explicitly marks the run as diagnostic.
- Reports must use actual tokenizer-counted input lengths for `random`, `sharegpt`, and `shared-prefix` prompts.
- Do not reuse `random_input_len` as the true prompt length for non-random datasets.
- If any measured request errors and `--fail-on-error` is set, the process must return non-zero.
- If `--require-ci` is set and `n_repeats < 3`, the process must return non-zero.
- Random prompt generation for release evidence must be reproducible with `--seed`.

## Directory Cleanup Policy

Before moving, deleting, or archiving files under `crates/`, `docs/`, or `scripts/`, run:

```bash
python3 scripts/release/inventory_tree.py \
  --out docs/release/cleanup/<YYYYMMDD>-inventory.md
```

Rules:

- Commit the inventory with the cleanup PR.
- Do not delete benchmark or release evidence unless the cleanup manifest lists the file, category, reference count, and reason.
- Keep workspace crate membership in root `Cargo.toml` synchronized with actual `crates/*/Cargo.toml` directories.
- Prefer classification and archive-candidate documentation before physical moves.
- Avoid large script-layout churn in the same PR as gate logic changes.
- If moving scripts, keep thin compatibility wrappers when existing docs or automation reference the old path.

## Product Goal Policy

Product-specific goals belong in separate goal documents, not in `AGENTS.md`.

Rules:

- `AGENTS.md` may name a product goal family, but it must not contain feature-level acceptance matrices, specific flag mappings, exact endpoint behavior, or per-goal implementation details.
- Put concrete product requirements in `docs/goals/`, `docs/release/`, or the goal document supplied by the user.
- Before implementing or claiming completion for a product goal, read the relevant goal document and follow its exact acceptance criteria.
- For release-critical, multi-step product, or performance-claiming goals, use the relevant goal document. If none exists, create a focused goal document or ask before inferring large product scope from `AGENTS.md`.
- For small, localized fixes, tests, or documentation updates, proceed directly without creating a goal document unless the user asks for one.
- Product goal completion claims must include the PASS line defined by that goal document and the artifact directory produced by its gate.
- Product feature gates do not imply tagging, publishing, Cargo release, Homebrew release, or GitHub release unless the user explicitly asks for those release steps.

## GPU Cost Policy

- Before using a paid GPU, state:
  - lane,
  - expected runtime/cost,
  - stop condition,
  - correctness gate,
  - performance command.
- Do not leave paid GPU instances idle. Stop or destroy them after validation evidence is collected unless the user explicitly asks to keep them running.
- If CUDA validation fails, collect the exact failing artifact/log, stop unnecessary processes, and avoid repeated full sweeps until the failure mode is understood.
- Prefer CUDA smoke before CUDA full unless the task specifically requires full release evidence.

## Release Regression Lessons

- Release readiness must cover both product entrypoints: `ferrum run` and `ferrum serve`.
- `ferrum run` multi-turn correctness must be tested with product defaults and deterministic diagnostic settings when appropriate.
- Do not treat `--temperature 0` as a fix for garbage output; it is only a diagnostic.
- For MoE models, verify that runtime presets select the intended scheduler/KV path. Unexpected fallback to a priority scheduler or default KV manager is a release blocker.
- Every release candidate must include `run` multi-turn correctness and basic `run` performance evidence, not only HTTP benchmark artifacts.
- If code changes after CUDA validation, rerun a CUDA quick regression before calling the release ready.
- Do not publish or tag after a failed local product-path smoke. Fix the failed path, rerun Metal locally, rerun CUDA remotely when needed, then record evidence.
- Do not hard-code model-family prompt hacks such as forced empty Qwen3 `<think>` blocks. Prefer model-provided chat templates from GGUF/HF metadata, and keep template rendering shared between `run` and `serve`.
- Do not infer release readiness for one model architecture from another. Dense Llama-style models and Qwen3 MoE/GPTQ models exercise different scheduler, KV, attention, quantization, tokenizer, and chat-template paths.
- Do not hide invalid model output by filtering decoded text. If `<unk>`, `[PAD...]`, tokenizer-reserved IDs, invalid UTF-8, or mojibake appear, trace token IDs and fix sampling/logit masking or KV/logits state before accepting the regression.
- Release blocker scans should include panic, KV cache overflow, `<unk>`, `[PAD]`, invalid UTF-8/mojibake, missing or duplicate `[DONE]`, stream bulk-flush behavior, strict-schema failures, required-tool failures, and silent fallback from a requested feature to base behavior.

## Performance and M3 Work Protocol

- Current CUDA M3 goal details belong in `docs/bench/m3-80pct-goal-2026-05-25/GOAL.md` and current status docs, not in `AGENTS.md`.
- Use vLLM source and release behavior as the comparison baseline before inventing new kernels.
- Do not run unscoped env-flip sweeps for M3 c=32. Use fresh profiler evidence to choose a lever.
- Work one high-return lever at a time.
- During long CUDA builds/tests, use the time for non-overlapping source tracing, vLLM comparison, kernel review, or microbench design.
- Performance claims need same-pod A/B and `N >= 3` for deltas under 10%. Single-pod or single-run numbers may be used as smoke or diagnostic evidence only.
- Correctness gates precede performance claims. At minimum run Paris and multi-turn gates for MoE, attention routing, scheduler, or runtime default changes.

## Coding Style and Naming Conventions

- Follow Rust 2021 idioms and keep code `rustfmt`-clean.
- Formatting is defined in `rustfmt.toml`: 4-space indentation, max width 100, reordered imports/modules.
- Use `snake_case` for functions/modules/files, `CamelCase` for types/traits, and `SCREAMING_SNAKE_CASE` for constants.
- Keep crate boundaries clear: shared types/traits belong in `ferrum-types` or `ferrum-interfaces`, not duplicated in implementation crates.
- Do not introduce broad abstractions unless the current goal needs them.

## Testing Guidelines

- Prefer crate-local integration tests in `crates/<crate>/tests`.
- Use descriptive `snake_case` test names focused on behavior.
- Add tests for new public APIs, serialization changes, scheduler/cache logic, CLI flags, OpenAI wire compatibility, and runtime-default behavior.
- For `bench-serve` changes, include unit tests for stream error handling, EOF before `[DONE]`, malformed SSE, usage token counting, `--fail-on-error`, `--require-ci`, and seeded prompt generation.
- Run `cargo test --workspace --all-targets` before opening a PR-ready claim.

## Commit and Pull Request Guidelines

- Follow conventional prefixes plus scope when useful, for example:
  - `feat(cli): ...`
  - `fix(server): ...`
  - `perf(cuda): ...`
  - `test(release): ...`
  - `docs(agents): ...`
- Keep commits focused and imperative.
- Avoid mixing unrelated crates in one commit when possible.
- PRs should include:
  - purpose,
  - affected crates/scripts/docs,
  - key design notes,
  - validation commands run,
  - artifact directories and PASS lines.
- Include sample CLI/API output when behavior changes.

## Long-Lived Goal and Evidence Policy

- `AGENTS.md` is for durable engineering rules only.
- Do not add one-off user requests, date-specific acceptance matrices, or long session logs here.
- Put temporary goal scope, per-goal acceptance criteria, and date-specific test matrices in `docs/` or `docs/release/`, then reference that document from commits or summaries.
- When a lesson should be written into `AGENTS.md`, generalize it into a reusable rule about correctness, release gates, paid GPU use, evidence, or performance claims.
- Detailed release evidence requirements are defined above in the non-negotiable rules and release gate policies. Accelerator evidence must be explicit: CPU gates are useful for contracts, but they do not prove Metal or CUDA behavior.
