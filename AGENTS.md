# Repository Guidelines

## Project Structure & Module Organization
- This repository is a Rust workspace. Root configuration lives in `Cargo.toml`, with crates under `crates/`.
- Core contracts are in `crates/ferrum-types` and `crates/ferrum-interfaces`; implementations live in crates like `ferrum-engine`, `ferrum-runtime`, `ferrum-models`, `ferrum-cli`, and `ferrum-server`.
- Integration tests are primarily in `crates/*/tests` (for example, `crates/ferrum-types/tests`).
- CI configuration is in `.github/workflows/ci.yml`; local runtime defaults are in `ferrum.toml`.

## Build, Test, and Development Commands
- `cargo check --workspace --all-targets` — fast compile validation across all crates/targets.
- `cargo build --workspace` — full workspace build.
- `cargo test --workspace` — run unit and integration tests.
- `cargo fmt --all -- --check` — verify formatting (same check used in CI).
- `cargo clippy --workspace --all-targets -- -A warnings` — advisory lint pass matching CI behavior.
- `cargo run -p ferrum-cli -- list` — run the CLI crate locally (swap `list` for `pull`, `run`, `serve`, etc.).

## Coding Style & Naming Conventions
- Follow Rust 2021 idioms and keep code `rustfmt`-clean.
- Formatting is defined in `rustfmt.toml`: 4-space indentation, max width 100, reordered imports/modules.
- Use `snake_case` for functions/modules/files, `CamelCase` for types/traits, and `SCREAMING_SNAKE_CASE` for constants.
- Keep crate boundaries clear: shared types/traits belong in `ferrum-types` or `ferrum-interfaces`, not duplicated in implementation crates.

## Testing Guidelines
- Prefer crate-local integration tests in `crates/<crate>/tests`.
- Use descriptive `snake_case` test names focused on behavior (example: `engine_status_serde_roundtrip`).
- Add tests for new public APIs, serialization changes, and scheduler/cache logic.
- Run `cargo test --workspace` before opening a PR.

## Commit & Pull Request Guidelines
- Follow the existing commit style: conventional prefixes plus scope when useful, e.g. `feat(cli): ...`, `refactor(engine): ...`, `feat(cli, models): ...`.
- Keep commits focused and imperative; avoid mixing unrelated crates in one commit when possible.
- PRs should include: purpose, affected crates, key design notes, and validation steps/commands run.
- Link related issues and include sample CLI/API output when behavior changes.
