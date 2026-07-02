# W2 default-path paged-unified CUDA smoke

## Verdict

Default-path minimal correctness passed for both product entrypoints:
`ferrum run` and `ferrum serve`.

This is not release-grade evidence. It is a focused checkpoint after enabling
paged KV for windowed Gemma3 on varlen-QKV backends.

## Source And Build

- Source checkpoint: `d6d872c1e12fc364886117b0431aec752b2d78ac`
- Remote source state: clean `git status --short`
- Remote source delivery: git bundle clone, no remote diagnostic patch
- Vast instance: `40826362`, 1x RTX 4090
- CUDA: driver `565.77`, runtime-reported CUDA `12.7`, `nvcc 12.4.131`
- Build rc: `0`
- Binary SHA256:
  `11b26df2b8dccf3138b2fe294e80ef618cc6255e56af626213e6aaabe8b2e48f`
- Build command:
  `cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`

## Correctness

`ferrum run`:

- Command shape: one-shot prompt, `--backend cuda`, `--max-tokens 8`,
  `--temperature 0`, `--kv-capacity 2560`, `--max-num-seqs 2`,
  `--output-format jsonl`
- rc: `0`
- Output:
  `{"event":"assistant","turn":0,"content":"5","finish_reason":"stop","n_tokens":3,"chunk_count":1,"ms":393.64389700000004}`

`ferrum serve`:

- Command shape: `--model gemma3:27b-gptq`, `--backend cuda`,
  `--max-num-seqs 16`, `--max-num-batched-tokens 2048`,
  `--kv-capacity 512`
- readiness: `/v1/models` ready on poll `8`
- chat rc: `0`
- Response:
  `{"choices":[{"message":{"content":"5"},"finish_reason":"length"}],"usage":{"prompt_tokens":23,"completion_tokens":1,"total_tokens":24}}`
- Health after bench: `331` successful requests, `0` failed requests
- Server log scan: `0` matches for panic/error/NaN/`<unk>`/`[PAD]`/invalid UTF
  patterns used in this artifact

Effective server config:

- `selected_kv_layout`: `paged`
- `selected_attention_impl`: `legacy_paged_decode`
- `selected_graph_mode`: `graph_disabled`
- `selected_max_sequences`: `16`
- `selected_kv_capacity`: `512`
- `selected_max_batched_tokens`: `2048`

## Performance Diagnostic

Ferrum c16 random-prompt diagnostic:

- Command: `bench-serve --random-input-len 256 --random-output-len 128
  --concurrency-sweep 16 --num-prompts 100 --n-repeats 3 --fail-on-error
  --require-ci --seed 9271`
- rc: `0`
- completed per run: `[100, 100, 100]`
- errored per run: `[0, 0, 0]`
- output token count source: `usage`
- output throughput: `295.8064415567493 ± 5.210666937312439 tok/s`
- goodput: `2.3204614024423846 ± 0.031239672060048216 req/s`
- TTFT p50: `798.7481571666667 ms`
- TPOT p50: `45.52809012467191 ms`

Comparison caveats:

- The directly matching random-prompt vLLM artifact in
  `w2_vllm0101_cuda12_baseline_probe_2026-06-15` reports about
  `381.5 tok/s`, but that vLLM run had `1` bad output/errored request, so it is
  diagnostic only. Against that imperfect baseline, this Ferrum run is about
  `77.6%`.
- The cleaner same-instance vLLM ShareGPT baseline reports `518.796 tok/s`, but
  this artifact did not rerun Ferrum on the same ShareGPT dataset after the
  default-path fix. Do not use that as a strict current same-dataset ratio.
- Therefore this checkpoint does not prove the `>=80%` mainstream-engine target.

## Shutdown

- Server process was stopped.
- `nvidia_smi_after_server_stop.txt` showed no running GPU processes.
- Vast shutdown poll verified `cur_state=stopped actual_status=exited`.

## Next

- Formalize the next same-dataset comparison before making a new performance
  claim.
- Since correctness now passes on the default path, the remaining c16 gap is a
  performance bottleneck, not the first-token correctness bug fixed by
  `embed_scale`.
