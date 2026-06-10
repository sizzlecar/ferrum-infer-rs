# Test Architecture Status

Last updated: 2026-06-10.

This goal is not complete. Completion still requires the final validator to print:

```text
TEST_ARCH GOAL PASS: <out_dir>
```

## Current State

- Branch: `goal/test-architecture-20260610`, based on main `097b3ec7`.
- Stage 0 done; stage 1 substantially done (see below).
- Decisions locked in GOAL.md: L1-cuda batched execution protocol; models.json
  manifest generates the README support table (Plan A).
- Gate tooling landed: `scripts/release/test_arch_goal_gate.py`
  (`--self-test` green; `--baseline` written at clean sha `2b0135db`).
- Measured audit-scope baseline (see `baseline.json`): env_var 7
  (models 4 + engine 3), cfg_branch 22 (all engine), supports_branch 18,
  once_lock 7; conformance coverage 1/20 ops; kill-list 11 entries
  (8 cpu-reachable, 3 cuda).
- All kill-list fix commits verified as ancestors of main except the
  branch-side `241dbc0f` (superseded by `049b3a42` for hb-10).

## Stage Progress

| Stage | State |
| --- | --- |
| 0 baseline + tooling | done |
| 1 tiny-model full-stack suite | done (hb-07 deferred) |
| 2 op conformance matrix + fallback law | partial — manifest 6/20 (added residual_add, fused_add_rms_norm), Llama fallback law done; 14 op parity + MoE fallback + launch-plan pending (need Metal/CUDA) |
| 3 main-path decoupling | partial — hot-path cfg coupling removed + device-init allowlist (Gate A2: cfg outside allowlist = 0); env/supports/OnceLock migration pending |
| 4 regression tiers + README matrix | partial — manifest + README guard + explosion-radius classifier done (local); lane wiring + matrix RUN pending (need GPU) |
| 5 kill-rate + stability + final PASS | partial — new-test anti-flake 10/10+5/5 green; CUDA kills + full lanes pending (need pod) |

## Gate A (platform isolation) — current measured state

| Sub-gate | Target | Now (outside allowlist) |
| --- | --- | --- |
| A1 env_var | 0 | 7 (migration target) |
| A2 cfg_branch | 0 | **0** — 20 legitimate device-init branches allowlisted |
| A3 supports_*() | 0 | 18 (structural migration target) |
| A3b OnceLock | 0 | 7 (config-freeze migration target) |
| A4 op conformance | 20/20 | 6/20 |

The one genuine hot-path platform coupling — the engine's
`cfg(target_os=macos)` unified-vs-legacy routing — is removed: the
device→capability decision now lives in `ModelExecutor::
supports_native_unified_decode()` (executor is backend-aware), so the
engine carries no platform cfg. Behavior preserved (CPU verified by
tiny_stack; Metal/CUDA preserved by construction). The remaining cfg
branches are device construction/detection at the composition root and
are legitimately allowlisted.

## Remaining decoupling plan (env / supports / OnceLock)

These are NOT allowlistable — the GOAL targets them for migration:

- **env_var (7) + OnceLock (7)**: all the `*_runtime_config()` accessors
  that do `OnceLock.get_or_init(|| Config::from_env_vars(std::env::vars()))`
  (llama_family, llama_family_forward_batched, qwen3_moe_runtime, engine
  registry/builder/continuous_engine). Fix: read env once at the CLI/serve
  composition root, build the typed config, thread it through construction.
  Bounded but broad (touches every accessor call site). Low behavioral risk
  (env isn't mutated at runtime) but should land with the L0 suite green.
- **supports_*() (18)**: structural — these choose between genuinely
  different forward paths (varlen-batched vs per-item, paged setup). Cannot
  collapse to a single trait method without restructuring model forward
  logic. Needs Metal + CUDA parity validation before moving, per the GOAL's
  behavior-invariance rule. Highest-risk, last.

## Metal IS the local GPU — corrected boundary

Earlier this status mis-filed all GPU work as pod-only. This is an Apple
Silicon host, so every Metal validation runs locally. Done on this Mac:

- **Metal op-parity**: `cargo test -p ferrum-testkit --features metal --test
  op_diff` → 11/11 cells green (6 ops, CPU-vs-Metal NMSE < 1e-6).
  Evidence: `evidence/metal-op-parity-20260610.md`.
- **Engine under metal**: `cargo test -p ferrum-engine --features metal
  --test tiny_stack` → 10/10 green. Confirms the stage-3 decoupling compiles
  and runs correctly in the Metal configuration.
- **L1-metal lane (Gate C2)**: `scripts/release/lane_l1_metal.sh` bundles
  op-parity + tiny_stack(metal) + server wire, **81 s wall** (budget 900 s),
  prints `TEST_ARCH L1_METAL PASS`. Artifact: `evidence/l1-metal-run/`.

Only **CUDA-specific** work genuinely needs a pod (per AGENTS.md GPU budget
approval):
- CUDA op-parity column for the conformance matrix.
- hb-09 / hb-10 / hb-11 CUDA kill verification (L1-cuda batch).
- L1-cuda lane RUN + the Qwen72B/Llama70B matrix cells (Gate C3/C4).
- The CUDA half of the final aggregated PASS.

The stage-3 hot-path decoupling already landed is now Metal-validated
(op-parity + engine-under-metal both green); the remaining supports_*
migration still wants the CUDA parity column before it moves.

## Stage 1 Detail

- `ferrum-models` `test-support` feature: deterministic tiny CPU LlamaFamily
  model (`SyntheticLlamaLoader`) wrapped in the real `LlmExecutor`, plus an
  in-memory `TinyTokenizer` (real EOS, composite-token + EOS overrides).
- 10 full-stack scenarios, all green, sub-second on CPU:
  - 9 in `ferrum-engine/tests/tiny_stack.rs` (real engine + forward + KV +
    GreedySampler, no mock).
  - 1 in `ferrum-server/tests/tiny_stack_wire.rs` (real AxumServer over a stub
    engine, in-process).
- Verified CPU kills (apply repro patch → mapped test red, revert → green):
  hb-01, hb-02, hb-03, hb-04, hb-05, hb-06. hb-08 exempt (not reproducible at
  tiny CPU scale — `LlmExecutor` never drives the batched paged-decode scratch
  path). hb-07 deferred to stage 2 (needs a tiny Qwen3-MoE + the capability
  fallback law test).
- All 9 revert-fix repro patches apply cleanly and are product-code-only
  (hb-03 regenerated to drop a co-mutated sampler test hunk).

### Deviations from the GOAL.md stage-1 plan (intentional)

- `serve.rs build_app` extraction not done: the in-process composition point
  already exists as `AxumServer::from_llm`; scenario 9 uses it directly. No
  server refactor was warranted.
- `ParityLoader` promoted by adding `SyntheticLlamaLoader` in `test_support`;
  the private `ParityLoader` in `llama_family_pipeline.rs` is left in place
  (still used by pipeline parity tests). Full dedup is a later cleanup.

## Environment Notes

- Disk is near-full. The worktree shares the main repo's `target/` via
  `CARGO_TARGET_DIR=/Users/chejinxuan/rust_ws/ferrum-infer-rs/target` to avoid
  duplicating the ~7 GB candle/dep build. Building from this worktree rebuilds
  the ferrum crates (branch source differs) but reuses heavy externals.

## Long-term Heartbeat Metrics (tracked per GOAL.md)

- Bugs found manually/by users per month: (start tracking at stage 1 landing)
- Kill-list length: 11; verified CPU kills wired: 6; exempt: 1 (hb-08);
  deferred: hb-07 (stage 2), hb-09/10/11 (CUDA, stage 4-5).
