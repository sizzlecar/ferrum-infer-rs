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

## Active M3 Performance Goal
- Current stretch target: `Qwen/Qwen3-30B-A3B-GPTQ-Int4` reaches at least `0.80 × vLLM` throughput for `c=1/4/16/32`; source of truth is `docs/bench/m3-80pct-goal-2026-05-25/GOAL.md`.
- Latest useful c=32 progress includes the 2026-05-28 VPA bridge: `FERRUM_USE_VLLM_PAGED_ATTN=1` now passes the Paris smoke and full-sweep same-pod N=3 improved throughput by about `+3.1%/+2.8%/+3.4%/+6.5%` for `c=1/4/16/32`; c=32 reached `1127.9 ± 101.4 tok/s` versus no-VPA `1058.9 ± 9.2 tok/s`.
- VPA is a correctness/unlock plus modest performance lever, not enough for the 0.80× target. A 2026-05-28 c=32 profile after VPA still shows MoE as the main decode bottleneck: active m=32 `batched-decode-prof` has roughly `~69%` MoE, `~11%` attention, `~14%` dense; within MoE, vLLM-Marlin gate_up/down GEMMs dominate.
- The 2026-05-28 session is a negative control: env-flip sweeps for `FERRUM_VLLM_MOE`, `FERRUM_MOE_GRAPH`, and `FERRUM_PAGED_FLASH_SPLITS` produced no new performance gain; do not repeat that strategy.
- A 2026-05-28 vLLM-source-inspired `moe_block_size=8` microbench is also a negative control: Qwen3-shaped c=32 all-active Marlin MoE regressed from about `224.7 µs` at block 16 to `247.4 µs` at block 8, so do not chase block-size 8 without a new full-model profile.
- The existing Triton fused-MoE PTX is not a viable drop-in for this M3 shape: `triton_fused_moe_bench_qwen3_shape` measured about `453 µs/layer` for gate_up only, versus the profiled vLLM-Marlin gate_up path at about `150 µs/layer`; do not port that path without a new kernel design.
- Skipping redundant vLLM-Marlin MoE workspace zeroing is correct but small: Paris passed and same-binary c=32 N=3 measured `1153.1 ± 152.7 tok/s` vs forced old zeroing `1130.4 ± 168.7 tok/s`; treat as a cleanup/modest win, not a primary lever.
- 2026-05-29 Codex handoff progress:
  - Fixed the default graph-clean path: `FERRUM_MOE_GRAPH=1` now defaults `FERRUM_VLLM_MOE=1`, and graph capture is gated off when the MoE path is not graph-clean. This fixes the multi-turn contamination seen in `ferrum run` without disabling the graph fast path.
  - Lowered the server MoE batched-decode default threshold to `FERRUM_MOE_BATCH_THRESHOLD=4`. c=4 no longer stays on the slow per-item path: current full sweep measured `425.6 ± 36.6 tok/s` at c=4 versus the pre-fix ~`122 tok/s` observation.
  - `FERRUM_VLLM_MOE_PAIR_IDS=1` is now a graph-clean default when vLLM MoE is enabled. Paris passed; same-pod N=3 measured c=16 `986.9 ± 10.2 tok/s` and c=32 `1249.5 ± 69.3 tok/s`, small but positive over the non-pair-id path.
  - Added the pair-id combine fast path (`weighted_sum_batched_f16`), not a residual fusion. Paris passed; valid same-pod N=3 artifact `/workspace/m3-graph-loop/bench_pairids_residual_c16_c32_n3_rerun2/` measured c=16 `993.8 ± 26.6` and c=32 `1264.0 ± 29.4` tok/s. Ignore `/workspace/m3-graph-loop/bench_pairids_residual_c16_c32_n3/`: the server was killed by process-group SIGINT during the run.
  - Reran vLLM 0.20.2 on the same pod for c=16/c=32 (`random 256/128`, N=3): `/workspace/m3-graph-loop/vllm0202_baseline_c16_c32_n3_retry/` measured c=16 `1328.7 ± 44.4`, c=32 `1971.8 ± 7.4` tok/s. Current Ferrum ratios are therefore about c=16 `0.75×`, c=32 `0.64×`; c=32 still needs roughly `+25%` throughput to clear 0.80×.
  - A partial vLLM 0.20.2 Marlin-MoE scheduling backport was tested and reverted: Paris passed, but c=16 `1000.9 ± 34.7` and c=32 `1250.4 ± 65.5` tok/s showed no c=32 gain. Do not repeat this partial tile/thread-config tweak without a new profiler reason.
  - A full-model `FERRUM_MOE_BLOCK_SIZE=8` validation was tested because vLLM 0.20.2 selects block size 8 for small-M MoE. Ferrum now has an override and a larger vLLM-MoE `c_tmp` scratch so block8 can be tested safely, but the result is negative: `/workspace/m3-graph-loop/block8_validation_rerun/` measured c=16 `975.1 ± 3.6`, c=32 `1209.5 ± 37.2`, below the block16 fast-path baseline. Do not make block8 default or repeat block8-only tests; revisit only as part of a full vLLM 0.20.2 Marlin template/source parity port.
  - Fresh current-default c32 profile after pair-id combine: `/workspace/m3-graph-loop/profile_current_pairid_combine_c32/`. With graph disabled for sync timers, steady m≈30/31 decode stays around `16–17 ms`; MoE is still `~64–66%`, and bucket timing is dominated by `gemm1≈6.2–6.6 ms` plus `gemm3≈3.0–3.2 ms`; combine is only `~0.25 ms`. Do not spend another primary lever on combine.
  - Added debug-only bottleneck tools for the next GPU run: `FERRUM_MOE_DUMP=1 FERRUM_MOE_DUMP_BATCH_X_TOPK=256` captures real c=32 decode `active_blocks/unique_experts`, and `FERRUM_UNIFIED_POST_PROF=1` separates unified model time from decode post-process/sample/scheduler/stream/stop/complete. Use these before any new Marlin or engine optimization claim.
  - Use `scripts/m3_route_unified_profile.sh` on the restored pod to capture `[MOE_DUMP:*]`, `[unified-prof]`, `[iter-prof]`, and `[bucket-prof]` in one scoped c=32 run; it sets `FERRUM_MOE_GRAPH=0` because route dumping syncs/copies GPU buffers and fails fast if route shape or unified timing is missing.
  - A short-context vLLM paged-attention v1 path has been implemented locally behind `FERRUM_VLLM_PAGED_ATTN_V1_SHORT` (default on, `=0` forces old v2). It should remove the v2 reduce launch when `max_seq_len <= 512`, but it has only passed local `cargo fmt` / `cargo check -p ferrum-cli`; no GPU correctness or performance claim is valid until Paris and same-pod c32 A/B run. Use `scripts/m3_attn_v1_ab.sh` on the restored pod.
  - Vast instance `38237968` stopped during the route-dump build; restart returned `resources_unavailable`, and renting a replacement 48GB RTX 4090 failed with `insufficient_credit`. No GPU-backed performance claims after this interruption are valid until a pod is restored.
  - Restored GPU continuation after the credit issue used a dirty-main-equivalent remote binary rather than a clean checkout. Route/unified profile `/workspace/m3-moe-parity-lite-profile-20260529_033812/` passed Paris, captured real c=32 route shape `batch_x_topk=256`, `total_post_pad=832`, `active_blocks=52`, `unique_experts=48`, and showed graph-off steady decode median `total≈15.1 ms`, `model≈14.6 ms`, `decode_post≈0.34 ms`; postprocess/scheduler is not the primary gap.
  - The prompt-token-estimate scheduler candidate (`FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE=1` in current code; it was default-on in the dirty binary used for this artifact) passed local scheduler unit test and GPU Paris. Stable c32 N=3 artifact `/workspace/m3-prefill-est-c32-stable-20260529_034726/` measured `1288.2 ± 13.0 tok/s`, `TPOT p50=20.1 ms`, `ITL p50=15.1 ms`, `TTFT p50=576 ms`. This is only a small improvement over `1264.0 ± 29.4`, inside the <10% A/B caution zone, and still about `0.65×` of same-pod vLLM c32 `1971.8 ± 7.4`; keep it opt-in until a direct same-binary A/B confirms.
  - Discard contaminated artifacts from the same continuation where delayed old scripts or tmux builds interrupted runs: `/workspace/m3-prefill-est-c32-20260529_033714/`, `/workspace/m3-prefill-est-c32-clean-*`, `/workspace/m3-prefill-est-c32-foreground*`, `/workspace/m3-prefill-est-c32-rerun-*`, and `/workspace/m3-prefill-est-c32-setsid-*`.
  - `FERRUM_MAX_BATCHED_TOKENS=4096` is a negative control for c32. Paris passed and N=1 artifact `/workspace/m3-mbt4096-c32-n1-20260529_035206/` measured `1304.9 tok/s` with `TPOT p50=19.0 ms`, but TTFT worsened to `723.7 ms`; do not make 4096 default or spend a primary loop on max-batched-token env sweeps without a new profile.
- Current Ferrum full-sweep shape after the c=4 fix, before pair-id defaulting, was c=1 `155.4 ± 1.0`, c=4 `425.6 ± 36.6`, c=16 `965.8 ± 6.4`, c=32 `1205.6 ± 55.1` tok/s. Pair-id plus combine fast path replaces the c=16/c=32 rows with c=16 `993.8 ± 26.6`, c=32 `1264.0 ± 29.4`. The prompt-token-estimate candidate produced c32 `1288.2 ± 13.0` in a restored-pod N=3 run but is opt-in pending a direct A/B. The 0.80× goal is not complete.
- Treat c=16/c=32 vLLM ratios as same-pod directional because they are N=3; rerun an apples-to-apples vLLM 0.20.2 sweep with `random 256/128`, `n_repeats >= 5`, and committed artifacts before final goal accounting.

## M3 Work Protocol
- Before using a paid GPU pod, state the single lever, expected gain, budget cap, stop condition, correctness gate, and benchmark command; get user approval if cost or runtime is material.
- Ferrum is the challenger. After locating the bottleneck, bias toward the highest expected throughput gain, not the lowest-risk cleanup. The required loop is: bottleneck localization → minimal GPU/CUDA validation → optimization patch → correctness gate → performance test → repeat.
- Work one high-return lever at a time, but do not wait idly: during long CUDA builds/tests, run non-overlapping source tracing, vLLM comparison, kernel review, or microbench design in parallel.
- Valid next levers are: full vLLM 0.20.2 Marlin-MoE source parity or a small-m fused MoE kernel, reduce vLLM-Marlin gate_up/down overhead with profiler evidence, rewrite/port `moe_align` only if fresh profiling shows route/align material, add block-table shared-memory cache in paged decode, or extend graph coverage beyond the MoE layer loop.
- Use vLLM source as the comparison baseline before inventing new kernels. For M3 benchmarking, inspect the `v0.20.2` tag first; HEAD-only findings are useful for ideas but not authoritative for the current ratio target.
- Do not run unscoped env-flip sweeps for M3 c=32. Current defaults already wire the cheap wins: `FERRUM_MOE_GRAPH=1` via `apply_moe_graph_default`, `FERRUM_DECODE_OP_PROFILE` is opt-in via `FERRUM_PROFILE_STAGES`, and c=32 paged decode chooses split-K=4.
- Correctness gates precede performance claims. At minimum run the Paris smoke/bisect for MoE or attention routing changes; never benchmark a path known to emit garbage except while fixing that bug.
- Performance claims need same-pod A/B and `N >= 3` for deltas under 10%; single Vast pod numbers can vary by about 10–17% from hardware, clocks, and neighbor load.

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
