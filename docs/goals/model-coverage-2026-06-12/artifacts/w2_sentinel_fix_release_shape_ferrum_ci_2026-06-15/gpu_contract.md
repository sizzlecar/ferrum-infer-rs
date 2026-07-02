lane: W2 Gemma3 CUDA sentinel-fix release-shape Ferrum CI matrix
expected_runtime_cost: 1.5-3h, hard cap 3h, 1x RTX 4090 instance 40826362 at about USD 0.425/hr
stop_condition: startup/SSH/CUDA/build/correctness first failure, any bench cell nonzero or blocker warning, full matrix artifact copied, or 3h cap
correctness_gate: CUDA argmax_rows test, ferrum run, scripts/model_coverage_smoke.sh gemma3:27b-gptq
performance_command: ferrum bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3 --num-prompts 100 for c=1/4/16/32
