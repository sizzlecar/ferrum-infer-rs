# M1 Max Q4_K_M Benchmarks

Head-to-head decoding throughput on Apple Silicon, GGUF Q4_K_M format,
identical files across engines so the comparison is engine quality, not
quantisation.

## Hardware

| Field | Value |
|---|---|
| Machine | Apple M1 Max |
| Memory | 32 GB unified |
| OS | macOS (host) |

## Models (download list, ~30 GB total)

| Model | HuggingFace repo | File | Size |
|---|---|---|---|
| Qwen3-8B | `Qwen/Qwen3-8B-GGUF` | `Qwen3-8B-Q4_K_M.gguf` | ~4.5 GB |
| Llama-3.1-8B-Instruct | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` (or similar) | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | ~4.6 GB |
| Qwen3-30B-A3B | `Qwen/Qwen3-30B-A3B-GGUF` | `Qwen3-30B-A3B-Q4_K_M.gguf` | ~17 GB |

For each model, also download the corresponding `tokenizer.json` from
the **non-GGUF** HuggingFace repo (e.g. `Qwen/Qwen3-8B`,
`meta-llama/Meta-Llama-3.1-8B-Instruct`, `Qwen/Qwen3-30B-A3B`).

## Engines

| Engine | Build / install | Source format |
|---|---|---|
| ferrum-infer-rs (this repo) | `cargo build -p ferrum-cli --bin ferrum --features metal --release` | `.gguf` via candle-transformers' Q4_K_M Metal kernels |
| mistral.rs | `cargo install mistralrs` (or build from source) | same `.gguf` |
| llama.cpp | `brew install llama.cpp` | same `.gguf` |

## Workloads

Phase 1 (this commit): single-stream decode, deterministic.

| Workload | Prompt tokens | Output tokens | Concurrency | What it tests |
|---|---|---|---|---|
| Single decode | ~16 (default prompt) | 256 | 1 | Decode throughput |
| TTFT (long context) | 2048 | 1 | 1 | Prefill speed |
| Concurrent | 512 | 256 | 16 | Continuous batching (deferred) |

The concurrent workload requires ferrum's HTTP server path with batched
scheduling; will land in a follow-up.

## Running

```bash
# 1. Build ferrum (once):
cargo build -p ferrum-cli --bin ferrum --features metal --release

# 2. Lay out files:
#    ~/Downloads/ferrum-bench/
#      models/
#        Qwen3-8B-Q4_K_M.gguf
#        Llama-3.1-8B-Instruct-Q4_K_M.gguf
#        Qwen3-30B-A3B-Q4_K_M.gguf
#      tokenizers/
#        Qwen3-8B.tokenizer.json
#        Llama-3.1-8B-Instruct.tokenizer.json
#        Qwen3-30B-A3B.tokenizer.json

# 3. Run the bench:
./scripts/bench_m1_q4_k_m.sh \
  ~/Downloads/ferrum-bench/models \
  ~/Downloads/ferrum-bench/tokenizers

# 4. Inspect bench_results/<timestamp>/summary.md
```

Or run a single model manually:

```bash
# One-shot generation + timing (the bench path)
./target/release/ferrum run ~/Downloads/Qwen3-8B-Q4_K_M.gguf \
  --tokenizer ~/Downloads/Qwen3-8B.tokenizer.json \
  --prompt "Explain the theory of relativity in two sentences." \
  --max-tokens 256 \
  --temperature 0

# Interactive chat (multi-turn, with chat template + sampling)
./target/release/ferrum run ~/Downloads/Qwen3-8B-Q4_K_M.gguf \
  --tokenizer ~/Downloads/Qwen3-8B.tokenizer.json \
  --temperature 0.7 --top-k 50 --top-p 0.95 --repeat-penalty 1.1
```

In the REPL:
- `/exit` or Ctrl-D — quit
- `/clear` — reset KV cache (re-opens the model)
- `/system <text>` — change system prompt + reset
- `/help` — list commands

## Results

Filled in by each `bench_results/<timestamp>/summary.md`. Commit those
artifacts back to this file's history when stabilising a release.

### 2026-04-29 — placeholder

| Model | Engine | Decode tok/s | Prefill (s) | Decode (s) |
|---|---|---|---|---|
| Qwen3-8B | ferrum | _pending first run_ | — | — |
| Qwen3-8B | mistral.rs | _pending_ | — | — |
| Qwen3-8B | llama.cpp | _pending_ | — | — |
| Llama-3.1-8B-Instruct | ferrum | _pending_ | — | — |
| Qwen3-30B-A3B | ferrum | _pending_ | — | — |

## Notes on the ferrum path

Phase 3A (this commit) wires candle-transformers' quantized loaders
(`quantized_qwen3`, `quantized_qwen3_moe`, `quantized_llama`) behind a
new `ferrum run-gguf` subcommand. This deliberately *bypasses* ferrum's
`Backend<B>` abstraction for the Q4_K_M path — candle's Metal Q4_K_M
dequant kernels are mature and what mistral.rs / llama.cpp use, so
parity at the kernel layer is the floor.

The competitive moat (engine scheduler, custom kernels, MoE perf)
slots in as targeted optimisations once we have baseline numbers and
know where ferrum loses to llama.cpp / wins vs mistral.rs.
