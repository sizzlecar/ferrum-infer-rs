lane: W2 Gemma3 CUDA decode-op-profile diagnostic
expected_runtime_cost: 10-25min, hard cap 40min, reused Vast 40826362 1x RTX 4090 at about USD 0.402/hr
stop_condition: startup/SSH/CUDA/server readiness first failure, profile c16/c32 small sample complete and copied, or 40min cap
correctness_gate: server readiness plus bench-serve --fail-on-error zero-error diagnostic
performance_command: bench-serve sharegpt natural ASCII, c16/c32, num_prompts=16, n_repeats=1, seed 9271, diagnostic only
