lane: W2 Gemma3 CUDA projection-level dense Marlin profile diagnostic
expected_runtime_cost: 15-35min, hard cap 45min, reused Vast 40826362 1x RTX 4090 at about USD 0.425/hr
stop_condition: start/SSH/CUDA/source sync/build/server readiness first failure, projection-level dense Marlin profile c16/c32 small sample complete and copied, or 45min cap
correctness_gate: release build plus server readiness plus bench-serve --fail-on-error zero-error diagnostic
performance_command: bench-serve sharegpt natural ASCII, FERRUM_DECODE_OP_PROFILE=1 and FERRUM_MARLIN_PROFILE=1, c16/c32, num_prompts=16, n_repeats=1, random-output-len=64, seed 9271, diagnostic only
