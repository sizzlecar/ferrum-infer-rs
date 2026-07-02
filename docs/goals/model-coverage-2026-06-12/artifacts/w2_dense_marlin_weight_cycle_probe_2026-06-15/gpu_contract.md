# W2 Gemma3 CUDA dense Marlin weight-cycle native probe

- Lane: W2 Gemma3 CUDA dense Marlin weight-cycle native probe.
- Instance: Vast 40826362, cache-retained 1x RTX 4090.
- Expected runtime/cost: 8-20 minutes, hard cap 30 minutes, about USD 0.402/hr, expected cost below USD 0.25.
- Stop condition: start, SSH, CUDA/nvcc, source sync, compile, or probe first failure; successful probe with `VERDICT: dense Marlin native CUDA probe complete`; or 30 minute cap.
- Correctness gate: native CUDA probe compiles and exits 0 with the VERDICT line. This is not product correctness evidence.
- Performance command: `bash scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh`.
- Evidence scope: diagnostic only; no `ferrum run`, no `ferrum serve`, no release-grade performance claim.
