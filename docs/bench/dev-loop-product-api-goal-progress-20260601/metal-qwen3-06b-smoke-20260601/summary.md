# Metal Qwen3-0.6B Smoke - 2026-06-01

Scope: local Apple Metal correctness and small-model decode performance smoke
for `qwen3:0.6b` before the formal `0.7.3` release.

Binary:

- `target/release/ferrum`
- `ferrum 0.7.3`
- Built locally with `cargo build --release -p ferrum-cli --features metal --bin ferrum`

Model:

- Alias: `qwen3:0.6b`
- Canonical model: `Qwen/Qwen3-0.6B`
- Backend: `metal`

Correctness smoke:

```bash
target/release/ferrum run qwen3:0.6b \
  --backend metal \
  --temperature 0 \
  --max-tokens 16 \
  --prompt 'What is the capital of France? Answer with just the city name.'
```

Observed stdout:

```text
Paris
```

Result: pass.

Performance smoke:

```bash
target/release/ferrum run qwen3:0.6b \
  --backend metal \
  --temperature 0 \
  --max-tokens 64 \
  --bench-mode \
  --prompt 'Write a concise technical explanation of why caching improves inference throughput. Continue until the answer is complete.'
```

Runs:

- `run_1`: `[64 tokens, 42.1 tok/s, 1.5s]`
- `run_2`: `[64 tokens, 43.9 tok/s, 1.5s]`
- `run_3`: `[64 tokens, 43.0 tok/s, 1.5s]`

Summary:

- Decode throughput median: `43.0 tok/s`
- Token count: `64` tokens per run
- Correctness: pass

OpenAI-compatible server smoke:

```bash
FERRUM_KV_CAPACITY=1024 target/release/ferrum serve qwen3:0.6b \
  --host 127.0.0.1 \
  --port 18080
```

- `/health`: pass
- Backend selection: local release binary with Metal feature enabled

Multi-turn chat smoke:

- Request shape: one request with prior user/assistant turns in
  `/v1/chat/completions`
- User asks the model to remember code word `basalt`
- Follow-up asks for the remembered word
- Observed content: `basalt`
- Result: pass

Concurrent chat completions smoke:

```bash
scripts/bench_chat_completions.sh \
  --url http://127.0.0.1:18080/v1/chat/completions \
  --model qwen3:0.6b \
  --requests 8 \
  --concurrency 4 \
  --max-tokens 32 \
  --temperature 0 \
  --prompt 'What is the capital of France? Answer with one short sentence.'
```

Observed:

- Success: `8/8`
- Failure: `0`
- Wall time: `4.376132s`
- Requests/s: `1.828`
- Completion tokens: `239`
- Prompt tokens: `200`
- Aggregate throughput: `54.614 tok/s`
- Average request throughput: `15.255 tok/s`
- Average latency: `1.961s`
- P50 latency: `1.947512s`
- P95 latency: `2.206260s`

Caveat:

- Local swap was active during/after the run:
  `vm.swapusage: total = 2048.00M used = 1644.06M free = 403.94M`.
  Treat this as a Metal smoke/regression check, not a clean headline
  performance benchmark.
