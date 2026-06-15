lane: W2 Gemma3 CUDA prefill/TTFT profile diagnostic
instance: Vast 40826362, 1x RTX 4090, cache-retained native CUDA machine
expected_runtime_cost: 8-20min, hard cap 30min, reused instance at about USD 0.425/hr when running
stop_condition: start/SSH/CUDA/server readiness first failure, chat smoke failure, c16 ShareGPT diagnostic complete and artifacts copied, or 30min hard cap
correctness_gate: ferrum serve readiness plus non-stream chat smoke before bench-serve; bench-serve must use --fail-on-error
performance_command: diagnostic-only natural ASCII ShareGPT c16 with bench-serve --fail-on-error, seed 9271, n_repeats=1, num_prompts=16
profile_scope: server runs with FERRUM_PREFILL_OP_PROFILE=1 to split first-token prefill work; not release-grade evidence and not a product behavior validation path
baseline_engine_version_build: vLLM 0.10.1.1 CUDA12 baseline from w2_vllm_sharegpt_baseline_probe_2026-06-15; same RTX 4090, same HF/safetensors GPTQ model, same ShareGPT dataset
release_grade_status: not release-grade evidence; diagnostic only
