# W2 paged-varlen split-QKV native correctness probe
- lane: W2 paged-varlen split-QKV native correctness probe
- instance: Vast 40826362, 1x RTX 4090, cache-retained native CUDA machine
- expected runtime/cost: 5-10 minutes, about USD 0.425/hr while running
- stop condition: start/SSH/CUDA/source sync/compile/probe first failure, or probe PASS, then copy artifacts and stop instance
- correctness command: bash scripts/microbenches/build_and_run_paged_varlen_split_qkv_correctness.sh
- expected PASS line: VERDICT: paged varlen split-qkv correctness PASS
- performance command: none; diagnostic correctness only
