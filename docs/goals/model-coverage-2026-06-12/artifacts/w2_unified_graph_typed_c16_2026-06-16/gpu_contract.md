lane: W2 Gemma3-27B CUDA typed unified graph c16 diagnostic
expected_runtime_cost: 20-45min, hard cap 60min, reused Vast 40826362 1x RTX 4090 at about USD 0.425/hr
stop_condition: start/SSH/CUDA/source sync/build, ferrum run --unified-graph, ferrum serve --unified-graph, or c16 bench first failure; otherwise copy artifacts and stop the instance
correctness_gate: release build plus ferrum run --unified-graph known-answer smoke plus ferrum serve --unified-graph chat smoke with usage
performance_command: ferrum bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3, c16 only, diagnostic not release-grade
baseline: same-host vLLM c16 orientation baseline 518.7959572662905 tok/s from w2_vllm0101_cuda12_baseline_probe_2026-06-15
release_grade: false
