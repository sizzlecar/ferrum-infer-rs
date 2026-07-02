lane: W2 Gemma3 CUDA vLLM ShareGPT baseline-cleanliness probe
instance: Vast 40826362, 1x RTX 4090, cache-retained native CUDA machine
expected_runtime_cost: 20-45min, hard cap 60min, reused instance at about USD 0.425/hr when running
stop_condition: start/SSH/CUDA/vLLM server first failure, baseline smoke failure, c16/c32 ShareGPT diagnostic complete and artifacts copied, or 60min hard cap
correctness_gate: vLLM OpenAI /v1/models plus non-stream chat smoke before bench-serve
performance_command: diagnostic-only natural ASCII ShareGPT c16/c32 with bench-serve --fail-on-error, seed 9271, n_repeats=1, num_prompts=16
release_grade_status: not release-grade evidence; this only tests whether vLLM is baseline-clean on natural prompts after random-prompt invalid-UTF8 failures
