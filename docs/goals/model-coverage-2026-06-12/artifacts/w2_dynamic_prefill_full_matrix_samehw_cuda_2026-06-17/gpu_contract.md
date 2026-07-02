# W2 dynamic-prefill full-matrix CUDA contract

- Lane: W2 Gemma 3 27B GPTQ same-hardware vLLM/Ferrum full matrix diagnostic for dynamic active-decode prefill release-grade gap closure.
- Hardware: one RTX 4090 Vast instance, reusing cached model and native CUDA build environment when available.
- Expected runtime/cost: 60-120 minutes at approximately 0.471 USD/hour, estimated 0.50-0.95 USD.
- Stop condition: stop after correctness failure, vLLM baseline failure, Ferrum performance failure with artifacts copied back, successful full matrix artifact, or a 2 hour cap.
- Correctness gate: Ferrum `run` smoke plus Ferrum `serve` streaming smoke with usage; vLLM streaming smoke for baseline sanity.
- Performance command: `ferrum bench-serve --dataset sharegpt --sharegpt-path /workspace/ascii_sharegpt_w2_100.jsonl --random-output-len 128 --concurrency-sweep 1,4,16,32 --num-prompts 100 --n-repeats 3 --fail-on-error --require-ci --seed 9271`.
- Baseline engine: vLLM 0.10.1.1 on the same instance, same model snapshot, same prompt dataset, same c=1/4/16/32 sweep.
- Note: this run is still diagnostic unless the final W2 validator later prints `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.
