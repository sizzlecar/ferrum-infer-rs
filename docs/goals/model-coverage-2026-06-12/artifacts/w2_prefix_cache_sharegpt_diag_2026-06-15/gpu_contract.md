lane: W2 Gemma3 CUDA typed prefix-cache ShareGPT diagnostic
instance: Vast 40826362, 1x RTX 4090, cache-retained native CUDA machine
expected_runtime_cost: 10-25min, hard cap 35min, reused instance at about USD 0.425/hr when running
stop_condition: start/SSH/CUDA/server readiness first failure, chat smoke failure, c16/c32 ShareGPT diagnostic complete and artifacts copied, or 35min hard cap
correctness_gate: ferrum serve --enable-prefix-cache readiness plus non-stream chat smoke before bench-serve
performance_command: diagnostic-only natural ASCII ShareGPT c16/c32 with bench-serve --fail-on-error, seed 9271, n_repeats=1, num_prompts=16
release_grade_status: not release-grade evidence; tests whether typed product prefix cache closes the repeated-prompt gap versus vLLM
