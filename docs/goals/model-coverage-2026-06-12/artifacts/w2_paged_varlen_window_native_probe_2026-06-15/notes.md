# W2 paged-varlen sliding-window native CUDA probe

- lane: W2 Gemma3 CUDA paged-varlen sliding-window native correctness probe
- expected runtime/cost: 5-15 minutes, hard cap 25 minutes, approximately USD 0.425/hr on existing 1x RTX 4090 Vast instance 40826362
- stop condition: instance start/SSH/CUDA/source sync/compile/probe first failure, or probe PASS after artifacts copied back
- correctness command: `bash scripts/microbenches/build_and_run_paged_varlen_window_correctness.sh`
- performance command: none; this checkpoint is correctness-only and produces no performance claim
- instance policy: reuse cached instance once for this checkpoint, then stop after evidence is copied
