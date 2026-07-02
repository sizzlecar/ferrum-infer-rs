# W2 dynamic KV full-matrix CUDA run contract

Lane: W2 release-grade full matrix diagnostic after dynamic on-demand KV.

Hardware: reuse stopped Vast instance `41241013`, 1x RTX 4090, same cached host.

Expected runtime/cost: 1.5-3 hours at `0.47111111111111115 USD/h`, approximately
`$0.70-$1.45`.

Stop condition: copy back the full remote artifact after PASS/FAIL, then stop the
instance and poll until `actual_status=exited`.

Correctness gate:

```text
cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
ferrum run gemma3:27b-gptq --backend cuda
ferrum serve gemma3:27b-gptq --backend cuda --max-num-seqs 32 --kv-capacity 512
```

The serve smoke must produce non-empty streamed output, exactly one `[DONE]`,
usage tokens, and no OOM/panic blocker scan hits.

Performance command:

```text
ferrum bench-serve --dataset sharegpt --sharegpt-path /workspace/ascii_sharegpt_w2_100.jsonl --random-output-len 128 --concurrency-sweep 1,4,16,32 --num-prompts 100 --n-repeats 3 --fail-on-error --require-ci --seed 9271
```

Run the same `bench-serve` client against the same-pod vLLM baseline and Ferrum.

Baseline:

- engine: vLLM
- expected version: `0.10.1.1`
- model: `circulus/gemma-3-27b-it-gptq`
- dataset: `/workspace/ascii_sharegpt_w2_100.jsonl`
- same hardware/model/quantization/prompt set as Ferrum

Release-grade note: this run can become source evidence for the final W2
manifest, but it is not itself the final `MODEL_RELEASE_GRADE_W2 PASS` unless
the release-grade validator is run successfully afterward.
