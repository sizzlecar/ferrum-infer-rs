# W2 Gemma3 shadow graph native CUDA probe

- lane: W2 Gemma3 shadow graph native CUDA probe
- instance: Vast 40826362, 1x RTX 4090, cache-retained
- expected runtime/cost: 5-10 min, hard cap 20 min, about USD 0.425/hr
- stop condition: Vast start/SSH/CUDA/nvcc compile/probe first failure, or probe artifact copied locally
- correctness gate: nvcc compile rc 0, probe rc 0, finite checksum, stdout contains `VERDICT: Gemma3 shadow graph native CUDA probe complete`
- performance command: `bash scripts/microbenches/build_and_run_gemma3_shadow_graph_bench.sh`
- evidence status: diagnostic only, not release performance evidence
