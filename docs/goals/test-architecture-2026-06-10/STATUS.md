# Test Architecture Status

Last updated: 2026-06-10.

This goal is not complete. Completion still requires the final validator to print:

```text
TEST_ARCH GOAL PASS: <out_dir>
```

## Current State

- Branch: `goal/test-architecture-20260610`, based on main `097b3ec7`.
- Stage 0 (baseline + acceptance tooling) in progress.
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
| 0 baseline + tooling | in progress |
| 1 tiny-model full-stack suite | pending |
| 2 op conformance matrix + fallback law | pending |
| 3 main-path decoupling | pending |
| 4 regression tiers + README matrix | pending |
| 5 kill-rate + stability + final PASS | pending |

## Long-term Heartbeat Metrics (tracked per GOAL.md)

- Bugs found manually/by users per month: (start tracking at stage 1 landing)
- Kill-list length: 11 (see `historical_bugs.json` once landed)
