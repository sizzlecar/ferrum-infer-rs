lane: W2 Gemma3 CUDA dense vLLM Marlin first-fail diagnostic
expected_runtime_cost: 15-35min, hard cap 45min, reused Vast 40826362 1x RTX 4090 at about USD 0.425/hr
stop_condition: start/SSH/CUDA/source sync/build/server readiness, vLLM dense Marlin load, ferrum run smoke, or c16/c32 small sample first failure; diagnostic complete and copied; or 45min cap
correctness_gate: release build plus FERRUM_VLLM_MARLIN=1 ferrum run smoke plus server readiness plus bench-serve --fail-on-error zero-error diagnostic
performance_command: bench-serve sharegpt natural ASCII, FERRUM_VLLM_MARLIN=1, c16/c32, num_prompts=16, n_repeats=1, random-output-len=64, seed 9271, diagnostic only
