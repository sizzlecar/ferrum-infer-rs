lane: W2 Gemma3 CUDA typed vLLM paged-attn ShareGPT diagnostic
instance: Vast 40826362, 1x RTX 4090, cache-retained native CUDA machine
expected_runtime_cost: 10-25min, hard cap 35min, reused instance at about USD 0.425/hr when running
stop_condition: start/SSH/CUDA/server readiness first failure, typed attention-selection assertion failure, chat smoke failure, c16/c32 ShareGPT diagnostic complete and artifacts copied, or 35min hard cap
correctness_gate: ferrum serve from an artifact-local ferrum.toml with runtime.use_vllm_paged_attn=true, readiness, decision-trace assertion, and non-stream chat smoke before bench-serve
performance_command: diagnostic-only natural ASCII ShareGPT c16/c32 with bench-serve --fail-on-error, seed 9271, n_repeats=1, num_prompts=16
baseline_engine_version_build: vLLM 0.10.1.1 CUDA12 baseline from w2_vllm_sharegpt_baseline_probe_2026-06-15; same RTX 4090, same HF/safetensors GPTQ model, same ShareGPT dataset
release_grade_status: not release-grade evidence; tests whether a typed product config for vLLM paged attention closes the clean vLLM ShareGPT gap
