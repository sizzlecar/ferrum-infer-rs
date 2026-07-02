lane: W2 current Ferrum ShareGPT same-dataset diagnostic after Marlin evict-first default
instance: Vast 40826362, 1x RTX 4090
expected_runtime_cost: 20-40 minutes, about USD 0.14-0.28 at USD 0.42488888888888887/h
stop_condition: startup/SSH/CUDA/sync/build/serve/bench first failure, or diagnostic artifact collected, then stop instance
correctness_gate: prior product run/serve correctness artifact plus server ready, chat smoke, bench rc 0, completed requests, zero request errors, and clean server log scan
performance_command: ferrum bench-serve --dataset sharegpt --sharegpt-path ascii_sharegpt.jsonl --random-output-len 64 --concurrency-sweep 16,32 --num-prompts 16 --n-repeats 1 --fail-on-error --seed 9271
release_grade_status: diagnostic only; n_repeats=1, no --require-ci, no MODEL_RELEASE_GRADE_W2 PASS
