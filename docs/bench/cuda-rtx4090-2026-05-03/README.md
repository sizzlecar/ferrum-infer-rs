# Group A vs vLLM CUDA Benchmark — RTX 4090

**Date:** 2026-05-03
**Engines:** ferrum-infer-rs · vLLM
**Models:** Llama-3.1-8B FP16 · Llama-3.1-8B GPTQ-Int4 · Qwen3-30B-A3B GPTQ-Int4
**Concurrency levels:** c = 1, 4, 16, 32
**Cells:** 3 models × 2 engines × 4 concurrencies × 3 reps = **72 runs**

This report mirrors the audit-quality template at
[`docs/bench/macos-2026-05-02/`](../macos-2026-05-02/). Every cell has its
own JSON result, server log, and bench-harness log. The full environment
fingerprint lives at [`_env.txt`](./_env.txt). Re-run with
[`bench/v0.2-cuda/setup.sh`](../../../bench/v0.2-cuda/setup.sh) →
[`smoke_engines.sh`](../../../bench/v0.2-cuda/smoke_engines.sh) →
[`run_sweep.sh`](../../../bench/v0.2-cuda/run_sweep.sh).

> **STATUS: in flight as of 2026-05-03** — numbers in the tables below
> are placeholders (`(TBD)`) until the sweep completes. This file gets
> filled by re-running the same JSON-extract pass we used for the
> macOS bench.

---

## Headline (output throughput tok/s, c = 16)

| Model | ferrum | vLLM 0.20 | ferrum vs vLLM |
|---|---:|---:|---:|
| Llama-3.1-8B FP16 | (TBD) | (TBD) | (TBD) |
| Llama-3.1-8B GPTQ-Int4 | (TBD) | (TBD) | (TBD) |
| Qwen3-30B-A3B GPTQ-Int4 | (TBD) | (TBD) | (TBD) |

## Hardware

| Field | Value |
|---|---|
| Vendor | vast.ai (on-demand, contract `36047161`) |
| Region | Spain |
| GPU | NVIDIA RTX 4090 24 GB (sm_89, Ada) |
| Driver | (filled from `_env.txt`) |
| CUDA toolkit (host) | (from `_env.txt`) |
| CPU | 192-core (vast.ai shared host) |
| RAM | 503 GB shared (≥ 60 GB available to our cgroup) |
| Disk | 100 GB / overlay |
| Network | ~7.5 Gbps down |
| Cost | ~$0.32 / hr |

## Software

| Component | Version | Source |
|---|---|---|
| ferrum-infer-rs | branch `bench/v0.2-cuda` (build flag `--features cuda`) | this repo |
| vLLM | 0.20.0 | `pip install vllm==0.20.0` (PyPI) |
| benchmark harness | `vllm bench serve` | shipped with vLLM 0.20.0 |
| ShareGPT subset | `anon8231489123/ShareGPT_Vicuna_unfiltered`, 128 deterministic prompts (seed = repo HEAD short hash) | `bench/v0.2-cuda/prompts.json` |

## Models

| # | HF source | Format | Size on disk |
|---|---|---|---|
| M1 | `unsloth/Meta-Llama-3.1-8B-Instruct` | safetensors FP16 | ~16 GB |
| M2 | `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4` | safetensors GPTQ Int4 | ~5.7 GB |
| M3 | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` (Qwen-official) | safetensors GPTQ Int4 | ~17 GB |

## Bench harness

`vllm bench serve` (0.20+ CLI; replaces deprecated standalone `benchmarks/benchmark_serving.py`). Same OpenAI HTTP shape as the macOS bench so the engines see identical protocol load.

| Knob | Value | Why |
|---|---|---|
| Endpoint | `POST /v1/chat/completions` (SSE streaming) | OpenAI-compatible across both engines |
| Sampling | `temperature=0`, `top_p=1` | deterministic — RNG noise out, engine variance in |
| `max_tokens` | 512 (W1, c=1) / 256 (W2-W4) | covers a real chat-reply length |
| Per-cell `num_prompts` | `min(128, 4 × c)` | enough samples to amortise head/tail |
| Prefix cache | DISABLED on every engine | otherwise repeats hit cache and throughput inflates |

## Per-engine launch

### ferrum

```bash
FERRUM_KV_PAGED=1 FERRUM_PAGED_MAX_SEQS=64 FERRUM_KV_CAPACITY=1024 \
CUDA_VISIBLE_DEVICES=0 \
ferrum serve --model /workspace/models/<MX> --port 8800
```

### vLLM

```bash
CUDA_VISIBLE_DEVICES=0 \
python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/<MX> --port 8800 --max-num-seqs 64 \
  --no-enable-prefix-caching --disable-log-requests \
  --quantization gptq_marlin   # only for INT4 models
```

---

## Results

### Headline c = 16 throughput (tok/s)

| Model | ferrum | vLLM 0.20 | ferrum vs vLLM |
|---|---:|---:|---:|
| Llama-3.1-8B FP16 | (TBD) | (TBD) | (TBD) |
| Llama-3.1-8B GPTQ-Int4 | (TBD) | (TBD) | (TBD) |
| Qwen3-30B-A3B GPTQ-Int4 | (TBD) | (TBD) | (TBD) |

### Full grid: output throughput (tok/s)

| Model | Engine | c=1 | c=4 | c=16 | c=32 |
|---|---|---:|---:|---:|---:|
| Llama-3.1-8B FP16 | ferrum | (TBD) | (TBD) | (TBD) | (TBD) |
| Llama-3.1-8B FP16 | vLLM | (TBD) | (TBD) | (TBD) | (TBD) |
| Llama-3.1-8B GPTQ | ferrum | (TBD) | (TBD) | (TBD) | (TBD) |
| Llama-3.1-8B GPTQ | vLLM | (TBD) | (TBD) | (TBD) | (TBD) |
| Qwen3-30B-A3B GPTQ | ferrum | (TBD) | (TBD) | (TBD) | (TBD) |
| Qwen3-30B-A3B GPTQ | vLLM | (TBD) | (TBD) | (TBD) | (TBD) |

### TPOT median (ms) at c = 16

| Model | ferrum | vLLM |
|---|---:|---:|
| Llama-3.1-8B FP16 | (TBD) | (TBD) |
| Llama-3.1-8B GPTQ | (TBD) | (TBD) |
| Qwen3-30B-A3B GPTQ | (TBD) | (TBD) |

### Memory footprint at c = 32 (GB peak `nvidia-smi memory.used`)

| Model | ferrum | vLLM |
|---|---:|---:|
| Llama-3.1-8B FP16 | (TBD) | (TBD) |
| Llama-3.1-8B GPTQ | (TBD) | (TBD) |
| Qwen3-30B-A3B GPTQ | (TBD) | (TBD) |

---

## Methodology / caveats

(Filled in once we have data + post-mortem of any failed cells.)

---

## v0.2 scope notes

- **mistralrs dropped.** PoisonError on Qwen3-MoE GGUF in the macOS Group A bench, brittle install path. Not central to the ferrum-vs-vLLM story.
- **Qwen3-Coder-30B-A3B (M4) dropped.** Only available as community GPTQ pack (no official Qwen pack). One variable too many.
- **Llama-3.1-8B FP16 source:** `unsloth/Meta-Llama-3.1-8B-Instruct` mirror, not `meta-llama/*` (which requires per-account approval).

## How to reproduce

```bash
# On a vast.ai RTX 4090 pod with cuda_max_good >= 12.6 (driver ≥ 555):
git clone --depth 50 --branch bench/v0.2-cuda \
  https://github.com/sizzlecar/ferrum-infer-rs.git
export HF_TOKEN=hf_...
bash ferrum-infer-rs/bench/v0.2-cuda/setup.sh
bash ferrum-infer-rs/bench/v0.2-cuda/smoke_engines.sh    # ~2 min, must pass
bash ferrum-infer-rs/bench/v0.2-cuda/run_sweep.sh        # ~3-4 hr
# Locally:
POD=root@ssh8.vast.ai:17160 bash ferrum-infer-rs/bench/v0.2-cuda/pull_results.sh
```

## Related

- [`docs/bench/v0.2-cuda/README.md`](../v0.2-cuda/README.md) — the plan that drove this run
- [`docs/bench/macos-2026-05-02/`](../macos-2026-05-02/) — sister bench on Apple Silicon (same template)
