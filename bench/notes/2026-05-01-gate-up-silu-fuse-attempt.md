=== Fused gate+up+silu kernel attempt ===
Date: 2026-05-01 (commit pending)
Branch: perf/moe-gate-up-silu-fuse
Model: Qwen3-30B-A3B-Q4_K_M.gguf (17 GB)

Baseline (FERRUM_DISABLE_GATE_UP_SILU_FUSE=1): 37.2 / 38.2 / 38.3 t/s
Fused:                                          38.0 / 37.7 / 37.5 / 37.7 / 38.1 t/s

Result: WITHIN NOISE — no measurable improvement.

Memory state at time of test (pre-revert):
  swap used: 3.82 GB
  pages free: 4.0 GB
  big RSS (non-ferrum): claude 905 MB, iTerm 649 MB, mds 344 MB,
                       spotlight 355 MB, Chrome 156 MB
  → bench environment under paging pressure, results unreliable

Lesson: every perf test must capture vm_stat + swapusage alongside
the throughput numbers, AND benches should refuse to run when swap
is active above some threshold.

Decision: revert this branch (no PR), capture environment monitoring
hygiene as a separate small infra change, then re-attempt the fusion
with a clean machine state.
