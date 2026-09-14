# Ferrum Repository Guidelines

## Scope and workflow

- Ferrum is a Rust workspace. `Cargo.toml` defines membership; crates live in
  `crates/`. Keep membership synchronized when adding or removing crates.
- Inspect current code, worktree status, and upstream state before acting.
  Preserve unrelated changes and keep each fix focused on its original issue.
  Report additional fixes separately; they do not complete the original task.
- Keep reusable tests and benchmark logic in Rust. Do not add Python or shell
  test wrappers or one-off validation scripts.

## Product behavior

- Expose behavior through typed defaults, CLI/config options, or documented
  presets. Hidden environment combinations are not a product interface.
- Prefer protocol rules and declared model capabilities over client-name or
  model-name special cases. Preserve the model template's intended behavior.
- For shared inference changes, cover both `ferrum run` and `ferrum serve`.
  For entrypoint-specific changes, test the affected flow and explain why the
  other entrypoint is unaffected.

## Tests and validation

- Test code, protocol behavior, and real safety/capacity boundaries. Do not gate
  on machine IDs, commits, dirty status, artifact paths, exact PASS text, fixed
  benchmark matrices, arbitrary style counts, repetition totals, or PASS ratios.
- Prefer hardware-independent correctness tests. Backend tests should exercise
  the affected backend; select hardware/model samples from the change's reach.
- Start with the smallest affected test. At a stable code milestone, run the
  workspace checks below and relevant backend checks before calling a PR validated.
  Documentation-only edits need content/link and diff checks, not a full build.
- Distinguish compilation, protocol validity, semantic correctness, and measured
  performance. Report failures and skipped checks explicitly; a green CI job with
  tolerated failures is not proof that every check passed.

Workspace checks (Clippy currently permits warnings and is non-blocking in CI):

- `cargo fmt --all -- --check`
- `cargo check --workspace --all-targets`
- `cargo test --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -A warnings`

Backend compile checks (not runtime regression tests):

- Metal, on macOS: `cargo check --workspace --all-targets --features metal`
- CUDA CLI: `cargo check -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2`
  Requires a configured CUDA host and the pinned native operator-set lock via
  `FERRUM_NATIVE_OPERATOR_SET_LOCK`, as in `.github/workflows/ci.yml`.

## Performance evidence

- Support performance claims with same-hardware measurements. Record model and
  precision, server/client versions, dataset, input/output lengths, commands,
  configuration, and repetitions. State intentional differences in comparisons.
- Check output validity and report errors alongside timing. Distinguish usage
  tokens from SSE text events; report latency/throughput tradeoffs and uncertainty.
  Do not generalize one model or backend result to all supported configurations.

## Repository size and Rust hygiene

- Version source and small deterministic fixtures. Keep model weights, benchmark
  results, logs, profiles, archives, dumps, and copied binaries outside the repo.
  Cover build/local output paths in `.gitignore`; reuse Cargo caches instead of
  duplicating builds.
- Before adding a file over 1 MiB, minimize the reproducer. A required full asset
  needs a documented reason and explicit review approval. Git LFS is an exception,
  not storage for disposable output.
- Keep modules cohesive and split by responsibility when needed. Module/signature
  size is review guidance; avoid mechanical splits that make code harder to follow.
- Prefer typed Rust fixtures/builders. Keep JSON for parsing or wire compatibility,
  omit irrelevant metadata, and store fixtures beside their consuming tests/modules.
- Use Cargo.lock and registry/cache mechanisms. Do not commit generated dependency
  trees or vendor crates without an explicit offline or supply-chain requirement.

## Code map

All paths below are under `crates/`:

- Contracts: `ferrum-types`, `ferrum-interfaces`.
- Execution: `ferrum-engine`, `ferrum-scheduler`, `ferrum-kv`, `ferrum-models`,
  `ferrum-kernels`, `ferrum-quantization`, `ferrum-tokenizer`, `ferrum-sampler`.
- Product entrypoints: `ferrum-cli`, `ferrum-server`.
- Native operators: `ferrum-native-ops`, `ferrum-native-ops-builder`.
- Validation: `ferrum-bench-core`, `ferrum-testkit`, and `*/tests`.
