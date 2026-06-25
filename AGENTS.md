# Repository Guidelines

## Read First: Non-Negotiable Rules

- Do not claim a task is complete, release-ready, or performance-ready unless the required gate produced an artifact directory and the final validator printed the exact PASS line.
- Do not make performance claims without same-hardware evidence, benchmark command lines, git SHA, dirty status, binary SHA256 when available, runtime/config evidence, and saved artifacts.
- Validate both product entrypoints when a change can affect user-visible behavior: `ferrum run` and `ferrum serve`.
- Do not validate product behavior with hidden environment-variable combinations that users will not know to set. Required product behavior must be wired through typed defaults, CLI/config options, or documented presets.
- Correctness gates must pass before performance measurements are treated as evidence.
- Official accelerator release evidence must cover both correctness and performance on each shipped accelerator backend, and must include at least one Llama 8B-class dense model in addition to the Qwen3-30B-A3B MoE/GPTQ model.
- Paid GPU work requires a stated lane, expected runtime/cost, stop condition, correctness gate, and performance command before starting.
- Before creating, starting, or resuming paid GPU capacity, reconcile the current instance inventory and state which existing instance will be reused or why a new one is necessary.
- Long-running performance or GPU work must convert time into saved evidence: a PASS line, KEEP/REJECT diagnostic artifact, or explicit blocker. Do not count elapsed time, setup, compilation, or partial stability as final-goal progress.
- On long-lived goal branches, synchronize with `git pull --rebase --autostash` before staging or committing, and push validated commits promptly unless the user explicitly asks to keep work local.
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

- `scripts/release/run_gate.py`
  - Unified gate entrypoint. Use `python3 scripts/release/run_gate.py --list-lanes` to list lanes.
  - Lanes: `unit`, `metal`, `cuda-smoke`, `cuda-full`, `cuda-llama-dense`, `cuda-llama33-70b-4bit-2x4090-smoke`, `cuda-llama33-70b-4bit-2x4090`, `metal-tarball`, `cuda-tarball`, `homebrew-metal`, `homebrew-cuda-fetch`, `release-summary`, `release-complete`.
  - Each run writes `<out_dir>/gate.manifest.json` with lane, status, command line, git SHA, dirty status, artifact dir, timestamps, duration, binary SHA256 when available, model id/path when applicable, sanitized env, and PASS line.
  - Self-test: `python3 scripts/release/run_gate.py --self-test`; this is also run by `scripts/release/selftest_g0_validators.py`.
  - Required PASS line: `FERRUM GATE <lane> PASS: <out_dir>`.
  - `scripts/release.sh` is only a compatibility wrapper and intentionally fails; do not use it as a release source of truth.
- `scripts/release/g0_source_gate.sh`
  - Lanes: `unit`, `metal`, `cuda-smoke`, `cuda-full`, `cuda-llama-dense`, `cuda-llama33-70b-4bit-2x4090-smoke`, `cuda-llama33-70b-4bit-2x4090`, `all-source`.
  - Required PASS lines:
    - `G0 SOURCE unit PASS: <out_root>`
    - `G0 SOURCE metal PASS: <out_root>`
    - `G0 SOURCE g0_cuda4090_smoke PASS: <out_root>`
    - `G0 SOURCE g0_cuda4090_full PASS: <out_root>`
    - `G0 SOURCE g0_cuda4090_llama_dense PASS: <out_root>`
    - `G0 SOURCE g0_cuda2x4090_llama33_70b_4bit_smoke PASS: <out_root>`
    - `G0 SOURCE g0_cuda2x4090_llama33_70b_4bit PASS: <out_root>`
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
- `scripts/release/validate_release_completion_manifest.py`
  - Validates the local release completion manifest shape for the `run_gate.py release-complete` lane.
  - Required PASS line: `FERRUM RELEASE COMPLETION PASS: <out_dir>`.
- `scripts/release/backend_boundary_audit.py`
  - Cheap audit for backend-specific `cuda` / `metal` decisions outside allowed backend, registry, resolver, release, and allowlisted paths.
  - Checked-in allowlist: `scripts/release/backend_boundary_allowlist.json`.
  - Required PASS line: `BACKEND BOUNDARY AUDIT PASS: <out_dir>`.
- `scripts/release/backend_runtime_preset_snapshot.py`
  - Generates and validates no-weight runtime preset snapshots via `crates/ferrum-types/examples/backend_runtime_preset_snapshot.rs`.
  - Checked-in snapshots live in `scripts/release/snapshots/backend_runtime_preset/`.
  - Required PASS line: `BACKEND PRESET SNAPSHOT PASS: <out_dir>`.
- `scripts/release/backend_runtime_preset_goal_gate.py`
  - Final validator for backend runtime preset fast-iteration goals.
  - Aggregates unit, Metal, CUDA, backend boundary, runtime preset snapshot, and product scenario artifacts.
  - Required PASS line: `BACKEND RUNTIME PRESET GOAL PASS: <out_dir>`.
- `scripts/release/llama33_70b_4bit_2x4090_goal_gate.py`
  - Final validator for the Llama 3.3 70B 4bit 2x4090 goal.
  - Aggregates Metal, existing 1x4090 CUDA full/dense, and the 2x4090 70B artifact.
  - Required PASS line: `LLAMA33_70B_4BIT_2X4090 GOAL PASS: <out_dir>`.
- `scripts/release/run_scenarios.py`
  - Manifest-driven product regression runner for shared `ferrum run` and `ferrum serve` scenarios.
  - Manifests: `scripts/release/scenarios/product_regression.json` and `scripts/release/scenarios/product_regression_smoke.json`.
  - Self-test: `python3 scripts/release/run_scenarios.py --self-test`; this is also run by `scripts/release/selftest_g0_validators.py`.
  - Required PASS line: `BACKEND REGRESSION SMOKE PASS: <out_dir>`.
- `scripts/release/inventory_tree.py`
  - Required before moving, archiving, or deleting files under `crates/`, `docs/`, or `scripts/`.
  - Required PASS line: `INVENTORY PASS: <out_file>`.
- `scripts/release/selftest_g0_validators.py`
  - Unit/self-test for release validators.
  - It is run by `g0_source_gate.sh unit`.

## G0 Source Gate Policy

`cargo test --workspace --all-targets` is always the first source gate.

Use `scripts/release/run_gate.py` for unified source, binary, summary, and completion lanes when practical:

```bash
python3 scripts/release/run_gate.py unit --out <out_root>
python3 scripts/release/run_gate.py metal --out <out_root>
python3 scripts/release/run_gate.py cuda-smoke --out <out_root>
python3 scripts/release/run_gate.py cuda-full --out <out_root>
python3 scripts/release/run_gate.py cuda-llama-dense --out <out_root>
```

`scripts/release/g0_source_gate.sh` remains the delegated source gate implementation:

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
- Failure triage comes before machine churn. If a failure has a small, bounded follow-up that can reuse the same warmed instance, model cache, and build cache, record the keep-alive window and stop condition, then run only that follow-up. Otherwise stop the instance after artifacts are copied.
- Account for restart, bootstrap, build, and model-cache cost when choosing between a bounded keep-alive window and stopping the instance. Do not destroy a reusable cached environment just to recover from a source-level or script-level mistake.
- Prefer CUDA smoke before CUDA full unless the task specifically requires full release evidence.

## Vast GPU Runner Policy

Use Vast only after the paid GPU cost policy has been stated for the lane.

Credential and secret rules:

- Load `VAST_API_KEY` from `.env.local` when present, but never print the value, shell-trace it, commit it, copy it into artifacts, or write it into remote logs.
- Prefer non-interactive HTTP API calls over the Vast CLI because CLI behavior can drift across versions.
- Use the user's existing SSH public key, typically `~/.ssh/id_ed25519.pub`, for instance bootstrap. Never copy or print private keys.

Offer selection:

- Before creating a new Vast instance, query existing instances and record whether each relevant instance is running, stopped, scheduling, loading, or exited. Prefer a matching retained instance with useful caches over a fresh rental.
- Search rentable offers through the Vast HTTP API and filter locally for the exact hardware required by the lane, for example a 2x RTX 4090 host when a two-GPU CUDA lane requires it.
- Prefer low hourly cost only after hardware count, GPU memory, disk capacity, CUDA capability, network bandwidth, and reliability are sufficient for the gate.
- Do not broaden from the lane's required hardware, model matrix, or GPU count without explicit user approval.
- If the Vast API reports `insufficient_credit`, stop rental attempts immediately, report the external blocker, and do not keep cycling offers. Clean up any newly-created stopped diagnostic instance that has no unique evidence, while preserving older stopped instances that may contain model caches or artifacts.

Instance creation:

- Create instances with an NVIDIA CUDA devel image suitable for building Ferrum CUDA binaries, such as `nvidia/cuda:12.4.0-devel-ubuntu22.04`, unless the lane requires a newer CUDA base.
- Configure SSH access at creation time and install only required bootstrap packages such as `openssh-server`, `git`, `curl`, `ca-certificates`, `build-essential`, `pkg-config`, `libssl-dev`, Python, `jq`, and `rsync`.
- Request enough disk for source, build outputs, downloaded models, and artifacts. If the model size is uncertain, stop before creating the instance and estimate disk from the goal document or model metadata.
- Ensure the startup command keeps the container alive after SSH starts, for example by starting `sshd` and ending with `sleep infinity`. Do not rely on a foreground bootstrap command that can exit and stop the instance during a download or build.
- Save the selected offer id, created instance id, Vast response metadata, lane, expected runtime/cost, and stop condition in the local artifact notes without secrets.

Instance lifecycle:

- Query an instance with `GET https://console.vast.ai/api/v0/instances/<INSTANCE_ID>/` and header `Authorization: Bearer $VAST_API_KEY`.
- Start or stop an existing instance with `PUT https://console.vast.ai/api/v0/instances/<INSTANCE_ID>/` and JSON body `{"state":"running"}` or `{"state":"stopped"}`. Do not use guessed subpaths such as `/start/`.
- For a single-lane diagnostic, keep at most one non-stopped paid instance unless the lane explicitly requires multiple machines or the user approves parallel capacity. Treat `scheduling`, `loading`, `running`, and unknown transitional states as potentially billable until the API proves otherwise.
- If multiple unexpected paid or scheduling instances exist, pause new rentals and new GPU work. Pick the canonical instance, preserve any unique artifacts or caches that matter, then stop or destroy the extras according to their evidence/cache value.
- After requesting start, wait for both `cur_state=running` and `actual_status=running`, then verify SSH and CUDA with `nvidia-smi` and `nvcc --version` before running paid work.
- After requesting stop, keep polling until `actual_status=exited`; `cur_state=stopped` alone can appear while the container is still shutting down.
- Prefer reusing a retained stopped or already-running instance with valid caches over creating a fresh machine, when it matches the lane hardware and reliability requirements. Do not reinstall or recreate the environment just to recover from a source-level mistake.

Remote validation workflow:

- Before running a gate, verify the remote host with `nvidia-smi` and record GPU names, count, driver/CUDA visibility, and GPU memory in the artifact directory.
- Synchronize the exact local worktree to the instance, including `.git`, while excluding local secrets and build caches such as `.env.local` and `target/`.
- On the remote host, verify `git rev-parse HEAD` and `git status --short` before collecting evidence. Dirty remote state is not acceptable for performance claims unless the dirty files are explicitly listed in the artifacts.
- Install Rust and other build dependencies on the instance only as needed for the lane. Keep environment variables and runtime options visible in the saved command log, except for secrets.
- Run long downloads, builds, and gates inside `tmux`, `nohup`, or another detached session with logs under the artifact root. SSH disconnects must not kill model downloads, CUDA builds, or benchmark runs.
- For large Hugging Face model lanes, start a model snapshot prefetch into the documented remote `HF_HOME` in parallel with long CUDA builds when the cache is empty. Keep the official gate command unchanged, save the prefetch log as an auxiliary non-gate artifact, and do not copy or pass HF secrets unless the user explicitly approves that lane.
- When using Python `huggingface_hub.snapshot_download` for that prefetch, align the cache layout with Ferrum's `HF_HOME` lookup: set `HF_HOME=/workspace/hf-cache` and use `cache_dir="$HF_HOME/hub"` or omit `cache_dir`; do not use `cache_dir="$HF_HOME"` because Ferrum expects snapshots under `$HF_HOME/hub/models--...`.
- Prefer the current Hugging Face high-performance transfer path when available, for example `HF_XET_HIGH_PERFORMANCE=1` with recent `huggingface_hub`. If no `HF_TOKEN` is available, record that the prefetch is anonymous and expect slower or rate-limited public downloads rather than treating the model format as failed.
- Run the lane's correctness gate before any performance command. Product-path CUDA evidence must include the required `ferrum run` and `ferrum serve` coverage from the relevant gate or goal document.
- Run benchmark commands from the goal document or release policy, not ad hoc hidden environment-variable combinations.

Shutdown and artifact handling:

- Always copy back the gate artifact directory, command logs, Vast instance metadata, and any failure logs before destroying the instance.
- Destroy or stop the Vast instance immediately after PASS, failure triage, or the stated stop condition unless the user explicitly asks to keep it running.
- After cleanup, verify through the Vast API that the target instance is no longer running and that no unexpected paid/scheduling sibling instances remain. Record that cleanup check in the local notes.

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
- When historical vLLM artifacts or published/source behavior are sufficient for comparison, use them instead of spending GPU time on a live vLLM rerun. Only run live vLLM when the goal explicitly requires fresh same-pod vLLM evidence.
- Do not run unscoped env-flip sweeps for M3 c=32. Use fresh profiler evidence to choose a lever.
- Work one high-return lever at a time.
- Before a new paid performance diagnostic, compare the latest relevant Ferrum artifact against the best historical Ferrum artifact and the vLLM source/release baseline. Record the specific runtime/config, scheduler trace, token length, profile, or log difference that the next lever is expected to change.
- Do not use a GPU run to discover what should have been learned from existing artifacts. If the same failure class has an artifact, inspect that artifact first and write the falsifiable hypothesis before starting another instance.
- Each performance diagnostic must have an expected signal and reject threshold, for example a target trace-shape change, throughput floor, error-count limit, or profile counter movement. If the threshold is missed, stop, classify the candidate as rejected or diagnostic-only, and do not stack another source change on top without a new artifact-based explanation.
- Classify CUDA memory failures before fixing them: distinguish driver/CUDA OOM, allocator exhaustion, model-owned KV admission failure, scheduler capacity deferral, decode cancellation, and livelock. The fix direction must follow that classification.
- For KV/cache capacity pressure, prefer dynamic admission, waiting, release, or backpressure logic that is derived from runtime state. Do not replace a scheduling bug with a hard-coded model, GPU, VRAM, or concurrency enumeration unless the goal document explicitly defines that product behavior.
- Treat effective-concurrency or slot caps as diagnostic levers unless the goal document explicitly accepts them as product behavior. Do not substitute a lower effective concurrency for a requested c=32 release/performance target without making the effective value visible in artifacts and status notes.
- During long CUDA builds/tests, use the time for non-overlapping source tracing, vLLM comparison, kernel review, or microbench design.
- Performance claims need same-pod A/B and `N >= 3` for deltas under 10%. Single-pod or single-run numbers may be used as smoke or diagnostic evidence only.
- Correctness gates precede performance claims. At minimum run Paris and multi-turn gates for MoE, attention routing, scheduler, or runtime default changes.

## Long-Running Work Review Protocol

- When a goal spans long-running builds, paid GPU diagnostics, or repeated candidate failures, keep a local time ledger if the user requested one. The ledger must stay uncommitted unless explicitly requested otherwise.
- Record time by action category such as source reading, code edit, local validation, remote setup, compile wait, GPU run, artifact handling, artifact analysis, status update, and git sync.
- Use the ledger during long tasks to check whether time is converting into goal evidence. If repeated GPU runs or validation cycles do not move the target metric, pause implementation and do a written retrospective before continuing.
- A retrospective must distinguish stability progress, correctness progress, performance progress, and final-goal progress. Do not present stability or diagnostic progress as target-performance progress.
- During long-running goals, periodically summarize elapsed time by category and name the next artifact or PASS line that would count as real progress. If the next action cannot plausibly move that evidence, stop and choose a narrower diagnostic.
- Status updates for long-running work must name the current artifact or PASS/KEEP/REJECT/blocker, the remaining gap to the target, the paid-machine state, and the next stop condition. Avoid vague progress claims.
- If a candidate improves stability or error counts but remains far from the target throughput, label it as partial or diagnostic progress and identify the next bottleneck evidence before making another source change.
- After a rejected candidate, either revert it or document why it remains useful before starting the next paid diagnostic. Avoid accumulating unproven scheduler/runtime changes that make later artifacts hard to interpret.

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
- For long-running goal branches, synchronize with `git pull --rebase --autostash` before staging or committing, and push completed commits promptly after validation. Keep local ledgers, scratch scripts, and non-evidence state directories untracked unless the user explicitly asks to commit them.
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
