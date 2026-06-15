lane: W2 Gemma3 CUDA natural-prompt baseline safety probe
expected_runtime_cost: 20-50min, hard cap 60min, 1x RTX 4090 instance 40826362 at about USD 0.425/hr
stop_condition: startup/SSH/CUDA/vLLM smoke failure, sharegpt c16 nonzero, c16 pass then c32/cap16 nonzero, probe complete and artifact copied, or 60min cap
correctness_gate: torch CUDA smoke plus vLLM OpenAI smoke
performance_command: diagnostic-only ferrum bench-serve --dataset sharegpt --fail-on-error --require-ci --seed 9271 --n-repeats 3 --num-prompts 100 for c16 then c32
