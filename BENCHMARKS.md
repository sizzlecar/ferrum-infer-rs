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

## Methodology

Following `llama-bench` (the canonical GGUF benchmark tool, included
with llama.cpp):

  - **Sequential**: one engine + one workload + one model = one
    process. Engines never run concurrently; would contend for the
    Metal device queue and unified memory bandwidth, results would be
    meaningless.
  - **Workloads** (fixed shape per cell):
    - `pp512` — prompt processing: model digests a 512-token prompt;
      tokens/sec measures **prefill throughput**
    - `tg128` — token generation: model decodes 128 fresh tokens;
      tokens/sec measures **decode throughput**
  - **Repetition**: 1 warm-up + ≥5 timed runs per cell. Reported as
    `mean ± stddev`. Warm-up discards cold-cache effects.
  - **Determinism**: temperature = 0 (greedy) so different runs
    produce identical token sequences and timing differences are
    purely engine-side.
  - **Hardware state**: M1 Max, 32 GB unified memory, AC-powered.
    Other heavy apps closed during runs (low-memory pressure regime).
  - **Reproducibility**: `llama-bench -m <model.gguf> -p 512 -n 128`
    (default 5 reps, mean ± stddev printed).

References:
  - llama-bench manual: `man llama-bench` after `brew install llama.cpp`
  - vLLM: `benchmark_throughput.py` follows the same shape (separate
    pp / tg, multiple reps, deterministic decode).

## Results

### llama.cpp baseline — 2026-04-29 (M1 Max, build 8960)

#### Hardware

| Field | Value |
|---|---|
| Machine | Apple M1 Max |
| CPU | 8 performance + 2 efficiency cores (10 total) |
| GPU | 24-core (this binning of M1 Max; 32-core also exists) |
| Memory | 32 GB unified |
| Power | AC, 100% battery, lid open |
| Thermal | Cold start at first run; subsequent runs back-to-back |

#### Software

| Field | Value |
|---|---|
| OS | macOS 15.1.1 (build 24B91) |
| llama.cpp | brew bottle `8960`, build commit `19821178b` |
| ggml | `0.10.0` (libggml-blas, libggml-metal, libggml-cpu-apple_m1) |
| Metal device | MTL0, MTLGPUFamilyApple7 / Common3 / Metal3 |
| Working set | `recommendedMaxWorkingSetSize = 22906.50 MB` |
| Build flags | brew bottle defaults (`GGML_METAL=ON`, BLAS via Accelerate); not built from source |

#### Command

```bash
llama-bench \
  -m <model.gguf> \
  -p 512 -n 128 \
  -t 8 -ngl 99 -r 5 \
  -o md
```

  - `-p 512` = prompt-processing test (prefill 512 tokens)
  - `-n 128` = token-generation test (decode 128 tokens)
  - `-t 8`   = 8 CPU threads (perf cores only — leave efficiency cores for the OS)
  - `-ngl 99` = offload all layers to Metal
  - `-r 5`   = 5 timed reps (one warm-up included by default)
  - `-o md`  = markdown output (also saved as raw stdout)

#### KV cache + context

  - KV dtype: `fp16` (llama-bench default)
  - Context size: `4096` (llama-bench default)

#### Models

| Model | HuggingFace source | Size | SHA-256 (truncated) |
|---|---|---:|---|
| Qwen3-8B Q4_K_M | `Qwen/Qwen3-8B-GGUF`/`Qwen3-8B-Q4_K_M.gguf` | 4.68 GiB | `d98cdcbd…45785` |
| Llama-3.1-8B-Instruct Q4_K_M | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`/`Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 4.58 GiB | `7b064f58…3557c` |
| Qwen3-30B-A3B Q4_K_M | `Qwen/Qwen3-30B-A3B-GGUF`/`Qwen3-30B-A3B-Q4_K_M.gguf` | 17.28 GiB | `0d003f66…aa8f5` |

Full SHA-256 list and raw `llama-bench` stdout/stderr are saved in
`bench_results/` (`metadata.txt`, `model_sha256.txt`, `llamacpp_*.md`,
`llamacpp_*.stderr.log`) so the numbers below can be reproduced and
audited.

#### Results

| Model | Size | Backend | pp512 (prefill tok/s) | tg128 (decode tok/s) |
|---|---:|---|---:|---:|
| Qwen3-8B Q4_K_M | 4.68 GiB | BLAS+MTL | **346.52 ± 25.26** | **27.94 ± 0.90** |
| Llama-3.1-8B-Instruct Q4_K_M | 4.58 GiB | BLAS+MTL | **335.13 ± 24.86** | **29.20 ± 0.44** |
| Qwen3-30B-A3B Q4_K_M | 17.28 GiB | BLAS+MTL | **596.58 ± 3.89** | **44.52 ± 6.80** |

Observation: **Qwen3-30B-A3B MoE decode (44.5 t/s) is ~1.6× faster
than dense Qwen3-8B (27.9 t/s)** despite having ~3.7× more total
parameters — only 3B active params per token. ferrum needs to match
this MoE-on-Mac advantage to be competitive on the flagship target.

The pp512 stddev for the 8B models (~7%) is higher than the 30B model's
(~0.7%); this matches the fact that the 8B models share the M1 Max with
other macOS workloads (Spotlight, mds, etc.), while the 30B model
saturates the GPU and washes those out. Re-running with the macOS in
"low-overhead" mode (Spotlight indexing paused, no other apps) is on
the to-do list.

> Earlier revisions of this document published higher numbers
> (`373.51 / 31.17 / 375.24 / 29.06 / 598.23 / 52.68`); those came from
> a single un-saved run and didn't survive re-measurement. The numbers
> above were captured with the saved `bench_results/llamacpp_*.md`
> outputs as audit trail.

These are the numbers ferrum is catching up to. ferrum's own runs go
in the next subsection once Phase 1D is hooked into LlamaFamilyModel
end-to-end.

### ferrum — pending

| Model | Engine | pp512 | tg128 | vs llama.cpp |
|---|---|---:|---:|---:|
| Qwen3-8B Q4_K_M | ferrum (Phase 1D) | _pending_ | _pending_ | _pending_ |
| Llama-3.1-8B Q4_K_M | ferrum (Phase 1D) | _pending_ | _pending_ | _pending_ |
| Qwen3-30B-A3B Q4_K_M | ferrum (Phase 2 MoE + 1D) | _pending_ | _pending_ | _pending_ |

## Notes on the ferrum path

`ferrum run <path.gguf>` routes through ferrum's own
`LlamaFamilyModel<B>` runtime. candle is touched only inside
`GgufLoader<B>` for parsing the GGUF binary and Phase 1D's
`QuantLinear<B>` for the raw Q4 → Metal MTLBuffer upload. Every line of
math from `model.prefill()` onward — RMSNorm, RoPE, attention, fp16
GEMM, Q4_K_M dequant — runs through ferrum's own kernels in
`ferrum-kernels` / `ferrum-attention`.

The Phase 1D-1 dequant kernel
(`ferrum-kernels/src/q4_k_dequant.metal`) is the foundation: 1
thread per super-block expands 256 weights into 256 fp16 outputs,
called by `MetalBackend::gemm_quant` per matmul to produce a
transient fp16 buffer that the existing fp16 GEMM kernel consumes.

The competitive moat (engine scheduler, custom kernels, MoE perf)
slots in as targeted optimisations once ferrum's first end-to-end
numbers are in and we know where ferrum loses ground vs llama.cpp.
