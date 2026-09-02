# Repository Guidelines

## Scope

- Ferrum is a Rust workspace. Workspace members are declared in the root
  `Cargo.toml`; implementation crates live under `crates/`.
- Keep changes small and focused. Preserve unrelated user changes in a dirty
  worktree.
- Tests should fail for product code or protocol behavior, not repository
  process policy. Do not gate tests on a machine identifier, git commit, dirty
  status, artifact directory, exact PASS text, or a fixed benchmark matrix.
- Do not add Python or shell test wrappers. Put reusable test logic in the
  relevant crate's Rust unit or integration tests.

## Product behavior

- When a change affects user-visible inference behavior, cover both product
  entrypoints: `ferrum run` and `ferrum serve`.
- Product behavior must be reachable through typed defaults, CLI/config
  options, or documented presets. Hidden environment combinations are not a
  product interface.
- Keep correctness tests hardware-independent when practical. Hardware-specific
  tests should exercise backend code, not identify a particular host.
- Do not claim accelerator performance without actual same-hardware benchmark
  evidence and the command/config used to produce it.

## Repository size and Rust hygiene

- Git stores source and small deterministic fixtures, not build products,
  benchmark output, logs, profiles, model weights, archives, core dumps, or
  copied third-party binaries. Keep those outside the repository and cover
  local output paths in `.gitignore`.
- Before adding any file larger than 1 MiB, reduce it to the smallest fixture
  that reproduces the code behavior. If the full asset is genuinely required,
  document why it must be versioned and obtain explicit review approval.
- Keep Rust modules cohesive. Split oversized hand-written modules by
  responsibility instead of accumulating unrelated implementations in one
  file; do not split mechanically when it would make the code harder to follow.
- Treat module size and signature length as review guidance. Do not add tests
  that fail on arbitrary line counts, parameter counts, file counts, fixed
  repetition totals, or required PASS ratios.
- Prefer typed Rust fixtures and builders over duplicated JSON blobs. Keep
  checked-in JSON only when parsing or wire compatibility is the behavior under
  test, and remove metadata that does not affect that behavior.
- Do not commit generated dependency trees or vendored crates by default. Use
  Cargo's lockfile and registry/cache mechanisms; vendor only for an explicit
  offline or supply-chain requirement.
- Treat Git LFS as an exception, not a way to preserve disposable artifacts.
  Model and benchmark assets belong in external artifact storage unless they
  are intentionally shipped as repository content.
- Derive stress cases from real code boundaries and invariants. A test may
  enforce a safety or capacity limit implemented by the product, but it must
  not fail merely because a process checklist, machine identity, commit, or
  expected run count changed.

## Test commands

- `cargo fmt --all -- --check`
- `cargo check --workspace --all-targets`
- `cargo test --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -A warnings`
- macOS Metal: `cargo check --workspace --all-targets --features metal`
- CUDA: `cargo check -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2`

Run the smallest affected test first after a failure. Run the workspace gate at
a stable milestone before presenting a PR as validated.

## Repository structure

- Core contracts: `crates/ferrum-types`, `crates/ferrum-interfaces`.
- Runtime and product: `ferrum-engine`, `ferrum-models`, `ferrum-kernels`,
  `ferrum-quantization`, `ferrum-server`, `ferrum-cli`.
- Benchmark schema and aggregation: `crates/ferrum-bench-core`.
- Shared Rust test utilities: `crates/ferrum-testkit`.
- Integration tests: `crates/*/tests`.

Keep workspace membership synchronized with actual crate directories. Store
test fixtures beside the Rust tests or source modules that consume them.
