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
| 1 tiny-model full-stack suite | substantially done (hb-07 deferred to stage 2) |
| 2 op conformance matrix + fallback law | pending |
| 3 main-path decoupling | pending |
| 4 regression tiers + README matrix | pending |
| 5 kill-rate + stability + final PASS | pending |

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
