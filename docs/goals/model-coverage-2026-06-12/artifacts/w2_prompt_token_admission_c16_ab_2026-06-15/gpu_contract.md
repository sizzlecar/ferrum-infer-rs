# W2 prompt-token admission c16 A/B diagnostic

- lane: W2 Gemma3 CUDA prompt-token admission c16 A/B diagnostic
- instance: Vast 40826362, 1x RTX 4090, about USD 0.425/hr
- expected runtime/cost: 10-20min, hard cap 30min
- stop condition: startup, SSH, CUDA visibility, clean worktree checkout, build, product smoke, or c16 benchmark first failure; otherwise stop after artifact copy
- correctness gate: `ferrum run` known-answer smoke plus `ferrum serve` chat smoke with usage and zero benchmark errors
- performance command: diagnostic-only `ferrum bench-serve --dataset sharegpt --sharegpt-path ascii_sharegpt.jsonl --concurrency-sweep 16 --num-prompts 16 --n-repeats 1 --fail-on-error --seed 9271`
- baseline: same-host Ferrum/vLLM natural ASCII ShareGPT artifact `w2_vllm_sharegpt_baseline_probe_2026-06-15`
- release status: not release-grade evidence; no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` is expected from this checkpoint
