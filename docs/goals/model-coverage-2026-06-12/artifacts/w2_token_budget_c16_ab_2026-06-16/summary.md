# W2 token-budget c16 A/B diagnostic

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_token_budget_c16_ab_2026-06-16/`.
- Remote artifact copy: `remote/`.
- Product path: `ferrum serve` plus `ferrum bench-serve`.
- Model: `gemma3:27b-gptq`.
- Hardware: Vast `41212840`, 1x NVIDIA GeForce RTX 4090, driver `580.119.02`, CUDA `12.4`.
- Remote source: clean worktree at `5f01af002d44ec58e2242f63ff085e54ba9a9e8c`.
- Binary SHA256: `649a73fc2ec46ab4272a14390422a1bfc565243a4b638c6f775e6cc5b15d8962`.
- Cleanup: instance `41212840` deleted after artifact copy; Vast API returned HTTP 200 success.

## Command shape

Both cells used typed product CLI flags:

```text
ferrum serve --model gemma3:27b-gptq --backend cuda --kv-capacity 512 \
  --max-num-seqs 16 --max-num-batched-tokens <1024|512>
```

Each cell then ran streaming smoke for `2+3`, followed by diagnostic
`bench-serve` c16 random 64/16, `num_prompts=16`, `warmup_requests=4`,
`n_repeats=1`, `--fail-on-error`, seed `9271`.

## Results

| max_num_batched_tokens | completed | errored | req/s | output tok/s | TTFT p50 ms | TTFT p95 ms | TPOT p50 ms | ITL p95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 16 | 0 | 12.722 | 203.552 | 610.561 | 662.281 | 42.716 | 60.847 |
| 512 | 16 | 0 | 12.226 | 195.621 | 536.344 | 720.189 | 49.293 | 159.156 |

Both cells used `output_token_count_source=usage`.

## Interpretation

Correctness stayed green, but lowering `max_num_batched_tokens` from `1024`
to `512` did not improve throughput. It reduced TTFT p50, but worsened request
throughput, output throughput, TTFT p95, TPOT p50, and ITL p95.

The profile still shows Gemma3 GPTQ work dominated by mixed-prefill and MLP
projection cost rather than attention:

- `1024`: `items=16 prefill=11 decode=5 total_q=823`, model batch `334383us`.
- `512`: `items=16 prefill=3 decode=13 total_q=235`, model batch `118779us`,
  but downstream tail latency and throughput were worse.

Conclusion: a simple typed token-budget reduction is not a high-return W2
performance lever. Next work should focus on Gemma3 GPTQ dense MLP Marlin
projection behavior, weight residency/permute overhead, or a more targeted
admission policy than globally reducing `max_num_batched_tokens`.

This is diagnostic evidence only. It did not produce
`MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.
