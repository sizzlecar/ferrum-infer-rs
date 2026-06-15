# W2 Ferrum natural-prompt diagnostic GPU contract

- lane: W2 Gemma3 CUDA natural-prompt Ferrum diagnostic
- expected runtime/cost: 20-45min, hard cap 60min, 1x RTX 4090 instance 40826362 at about USD 0.425/hr
- stop condition: startup/SSH/CUDA/build/product-smoke first failure, c16/c32 diagnostic complete and artifact copied, or 60min cap
- correctness gate: target/release/ferrum run plus scripts/model_coverage_smoke.sh gemma3:27b-gptq
- performance command: diagnostic-only ferrum bench-serve --dataset sharegpt --sharegpt-path ascii_sharegpt.jsonl --fail-on-error --seed 9271, c16/c32 small sample first
- baseline: vLLM 0.10.1.1 natural ASCII ShareGPT artifact w2_natural_prompt_baseline_probe_2026-06-15
