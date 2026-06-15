# W2 greedy-argmax default validation GPU contract

- lane: W2 Gemma3 CUDA greedy-argmax default validation
- expected runtime/cost: 20-45min, hard cap 60min, 1x RTX 4090 instance 40826362 at about USD 0.425/hr
- stop condition: startup/SSH/CUDA/build/product-smoke first failure, decision trace missing gpu_greedy_argmax, c16/c32 diagnostic complete and artifact copied, or 60min cap
- correctness gate: target/release/ferrum run plus scripts/model_coverage_smoke.sh gemma3:27b-gptq
- performance command: diagnostic-only ferrum bench-serve --dataset sharegpt --sharegpt-path ascii_sharegpt.jsonl --fail-on-error --seed 9271 --num-prompts 32 --n-repeats 1
- local target git head: 9a3382357a1f8578421eeab8a93878d6f70a9cd5
- remote source note: auto_config.rs synced from local target head; remote git status records the dirty source file because this is a targeted diagnostic, not final release evidence
