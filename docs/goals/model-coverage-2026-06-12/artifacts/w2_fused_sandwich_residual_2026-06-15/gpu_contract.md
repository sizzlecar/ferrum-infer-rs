lane: W2 Gemma3 CUDA fused sandwich residual-add minimal validation
instance: Vast 40826362, 1x RTX 4090, cache-retained native CUDA machine
expected_runtime_cost: 15-35min, hard cap 45min, reused instance at about USD 0.425/hr when running
stop_condition: start/SSH/CUDA/source sync/build first failure, ferrum run correctness failure, serve/bench diagnostic complete and artifacts copied, or 45min hard cap
correctness_gate: CUDA release build plus product ferrum run smoke before any performance diagnostic; serve readiness and bench-serve --fail-on-error for product serve path
performance_command: diagnostic-only natural ASCII ShareGPT c16/c32 small sample with bench-serve --fail-on-error, seed 9271, n_repeats=1, FERRUM_DECODE_OP_PROFILE=1
release_grade_status: not release-grade evidence; no MODEL_RELEASE_GRADE_W2 PASS is expected from this checkpoint
