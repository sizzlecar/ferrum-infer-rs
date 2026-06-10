# CUDA pod execution contract (test-architecture goal)

Authorized by user 2026-06-10 ("需要cuda 就去开机器").

- **Lever**: run the CUDA-gated validator checks — CUDA op-parity column,
  hb-09/hb-10/hb-11 kill verification, the L1-cuda lane, and the model
  matrix CUDA cells.
- **Expected gain**: flip the CUDA-gated subset of the 38 failing checks
  toward `TEST_ARCH GOAL PASS`. (Non-CUDA gates — env/supports refactor,
  7 complex op references — are still local work, done on the pod's disk
  or locally.)
- **Files/paths**: scripts/release/lane_l1_cuda.sh, crates/ferrum-testkit
  op_diff CUDA cells, docs/.../patches/hb-10.patch, scripts/release/
  readme_model_matrix.py.
- **Correctness gate**: op-parity NMSE < fp16 tol on CUDA; hb-10 patch
  flips a CUDA test red; hb-09/hb-11 verify-live probes must NOT crash
  (if they do, fix first).
- **Benchmark gate**: N/A — this goal is correctness/coverage, not perf.
- **Budget cap**: one RTX 4090-class pod, target <= 4 GPU-hours for the
  L1-cuda lane + op columns; matrix runs are a separate budgeted step.
- **Stop condition**: L1-cuda lane PASS captured, or 4h elapsed, or an
  unrecoverable build failure reproduced 3×. Stop and report either way.
- **Discipline**: code reaches the pod via git pull (no scp); CUDA build
  runs on the pod (Mac cannot compile --features cuda); bench/run wrapped
  in `timeout` to avoid SIGINT-shutdown hangs burning rented minutes.
