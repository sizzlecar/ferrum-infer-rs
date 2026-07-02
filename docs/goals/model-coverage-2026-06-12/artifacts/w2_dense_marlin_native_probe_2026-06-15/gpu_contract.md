# W2 dense Marlin native CUDA Gemma3 probe

- lane: W2 dense Marlin native CUDA Gemma3 probe
- instance: Vast 40826362, 1x RTX 4090, cache-retained
- expected runtime/cost: 5-15 minutes, hard cap 25 minutes, about USD 0.425/hr while running
- stop condition: start/SSH/CUDA/source sync failure; native CUDA compile failure; native probe completes and artifacts are copied; or 25 minute hard cap
- correctness gate: none; native CUDA kernel-ceiling diagnostic only, not a product correctness/release gate
- performance command: `bash scripts/microbenches/build_and_run_dense_marlin_gemma3_perf.sh`
