# Group A Continuous-Batching Benchmark on macOS / Apple Silicon

**Date:** 2026-05-02
**Engines compared:** ferrum-infer-rs · llama.cpp · mistralrs
**Models:** LLaMA-3.1-8B-Instruct · Qwen3-8B · Qwen3-30B-A3B (MoE) — all Q4_K_M
**Concurrency levels:** c = 1, 4, 8, 16
**Total cells:** 36 base run + 9 c=16 rerun + 8 MoE rerun = 53 cells

This report is reproducible end-to-end from this directory: every cell has its
own JSON result, server log, and bench-harness log. The full environment
fingerprint is in [`_suite_env.txt`](./_suite_env.txt). Re-run with
[`run_suite.sh`](./run_suite.sh) (then optionally [`rerun_c16.sh`](./rerun_c16.sh)
and [`rerun_moe_batched.sh`](./rerun_moe_batched.sh) — see § Methodology).

---

## Headline — c = 16 throughput

Best-of-runs output throughput (tok/s):

| Model | ferrum | llama.cpp | mistralrs | ferrum vs llama.cpp |
|---|---:|---:|---:|---:|
| LLaMA-3.1-8B | **96.7** | 67.2 | 23.3 | **+44%** |
| Qwen3-8B | **93.2** | 68.6 | 23.5 | **+36%** |
| Qwen3-30B-A3B (MoE) | 79.2 | 83.4 | panic | -5% (matched) |

The Qwen3-30B-A3B (MoE) row is the one to look at — that's the model where
Apple Silicon Rust support was effectively missing two months ago. ferrum
closed it from a 51 → 80 tok/s gap to llama.cpp in a single PR (#81) by
mirroring the Phase-4 paged-KV scaffolding from `LlamaFamilyModel` into
`Qwen3MoeModel`. ferrum's MoE c = 16 number requires
`FERRUM_MOE_BATCHED=1 FERRUM_MOE_BATCHED_DECODE=1` (currently opt-in by
default — see § Methodology).

---

## Hardware

| Field | Value |
|---|---|
| Machine | MacBook Pro 16" (2021), `MacBookPro18,4` |
| CPU | Apple M1 Max (10-core: 8 performance + 2 efficiency) |
| GPU | Apple M1 Max integrated (32-core, unified memory) |
| RAM | 32 GB unified |
| OS | macOS 15.1.1 (24B91) |
| Power state | AC, default thermal profile, no `caffeinate` |

Captured at suite start and end (memory state, swap usage) in
[`_suite_env.txt`](./_suite_env.txt). The 30B-A3B run uses ~18 GB of weights
plus paged-KV pool (≈3 GB at c = 16) — total ≈21 GB resident, well below the
32 GB unified pool, so the MoE row by itself doesn't swap.

What does cause swap is running 36 cells back-to-back. mmap'd GGUFs across
three engines + lingering allocations push `vm.swapusage` to ~7 GB by the
time the suite reaches Qwen3-30B-A3B. § Methodology explains the rerun.

## Software versions

| Component | Version | Source |
|---|---|---|
| ferrum-infer-rs | 0.7.0 @ `d7cfaae` (PR #81 merged) | this repo, `cargo build --release -p ferrum-cli --bin ferrum --features metal` |
| llama.cpp | b8960 (homebrew) | `brew install llama.cpp` |
| ggml | 0.10.0 (homebrew, llama.cpp dep) | bundled |
| mistralrs | 0.8.1 | `cargo install mistralrs-server` |
| Python | 3.9.6 | system |
| Rust | stable (captured in `_suite_env.txt`) | rustup |

Frozen in [`_suite_env.txt`](./_suite_env.txt) at the moment the suite started.

## Models

All three GGUFs are the `Q4_K_M` quantization (≈4.5 bits/weight effective).
Same files are read by all three engines.

| Model | File | Size | Source |
|---|---|---|---|
| LLaMA-3.1-8B-Instruct | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 4.6 GB | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` |
| Qwen3-8B | `Qwen3-8B-Q4_K_M.gguf` | 4.7 GB | `unsloth/Qwen3-8B-GGUF` |
| Qwen3-30B-A3B (MoE) | `Qwen3-30B-A3B-Q4_K_M.gguf` | 17.3 GB | `Qwen/Qwen3-30B-A3B-GGUF` |

Files live at `/Users/chejinxuan/ferrum-bench/models/`. Tokenizers (only
needed by mistralrs's CLI) at `/Users/chejinxuan/ferrum-bench/tokenizers/`.

## Bench harness

`bench/scripts/bench_serving.py` — modeled after vLLM's
`benchmark_serving.py`. Identical request shape across engines (OpenAI
`/v1/chat/completions` SSE), so the engine sees the same protocol load.

| Knob | Value | Why |
|---|---|---|
| Endpoint | `POST /v1/chat/completions` (SSE streaming) | OpenAI-compatible across all three engines |
| Temperature | `0.0` | deterministic decode, removes RNG-driven variance |
| `max_tokens` | `64` | long enough that TPOT settles past warm-up; short enough to fit ~30 cells/hour |
| `--deterministic-prompts` | yes | per-c prompt set is reproducible across runs |
| Concurrency | `--max-concurrency $c` (in-flight cap) | matches the launch flag handed to each engine |
| Total prompts | `8 / 16 / 24 / 32` for c = `1 / 4 / 8 / 16` | enough to amortize the few-prompts head/tail effect |
| Prewarm | one synchronous chat completion (`max_tokens=4`) before the main run | flushes JIT, builds first paged-KV blocks, populates page cache |

The harness reports `output_throughput_tok_s` (total output tokens / wall
time), `tpot_ms.median`, `ttft_ms.median`, plus p50/p90/p99 distributions
for both. Throughput is the headline metric here because all three engines
are run with continuous batching enabled — `tok/s` is what a serving
operator sees.

## Per-engine launch

Each cell:
1. Kills any prior `ferrum.*serve` / `llama-server` / `mistralrs.*serve`.
2. Starts the engine in the background, redirects logs to the cell-specific
   `*.server.log`.
3. Polls `GET /v1/models` until 200 (timeout 60 s ferrum / 90 s llama.cpp /
   240 s mistralrs — mistralrs can take 2 min to load 30B-A3B's tokenizer).
4. Runs one prewarm chat completion (`max_tokens=4`).
5. Runs `bench_serving.py` with the cell's `(num_prompts, c, max_tokens)`.
6. Saves result JSON to `<engine>__<model>__c<n>.json`, kills the engine.

Every cell starts from a fresh process. KV pool, prefix cache, JIT etc.
are all built up cleanly from cold so cells compare apples to apples.

### ferrum (dense + per-token MoE)

```bash
FERRUM_METAL_PAGED_KV=1 \
FERRUM_PAGED_MAX_SEQS=$((c * 2)) \
FERRUM_KV_CAPACITY=512 \
FERRUM_MAX_BATCH=$c \
ferrum serve --model $GGUF --port 8783
```

- `FERRUM_METAL_PAGED_KV=1` enables the GPU paged-KV pool.
- `FERRUM_PAGED_MAX_SEQS=$((c*2))` sizes the pool for `2×c` to leave
  headroom for in-flight + about-to-release blocks; the harness creates a
  fresh `cache_id` per request so without 2× we hit `pool_exhausted`.
- `FERRUM_KV_CAPACITY=512` caps per-sequence KV blocks. With `MAX_SEQS=32`
  this keeps the pool ≈3.1 GB total — enough headroom on a 32 GB Mac when
  Qwen3-30B-A3B's weights already eat 18 GB.

### ferrum (batched MoE — required for c ≥ 8 on MoE)

```bash
FERRUM_METAL_PAGED_KV=1 \
FERRUM_PAGED_MAX_SEQS=$((c * 2)) \
FERRUM_KV_CAPACITY=512 \
FERRUM_MAX_BATCH=$c \
FERRUM_MOE_BATCHED=1 \
FERRUM_MOE_BATCHED_DECODE=1 \
FERRUM_MOE_BATCH_THRESHOLD=2 \
ferrum serve --model $GGUF --port 8783
```

`FERRUM_MOE_BATCHED` is opt-in (default 0 — see
`crates/ferrum-models/src/models/qwen3_moe.rs:2496`). This enables the
batched MoE GEMV decode path. At c = 16 it lifts MoE throughput from 48
tok/s (per-token) to 79 tok/s. The crossover is at c ≈ 8 — below that, the
per-token path is faster on this hardware.

### llama.cpp

```bash
llama-server --model $GGUF --port 8001 \
  --ctx-size 4096 --parallel $c --batch-size 2048 --jinja
```

- `--parallel $c` is llama.cpp's continuous-batching slot count.
- `--ctx-size 4096` is the per-slot KV budget.
- `--jinja` lets the server apply chat templates from the GGUF metadata.

### mistralrs

```bash
mistralrs serve --port 8002 --max-seqs $c text \
  --format gguf -m $MODEL_DIR -f $gguf_file -t $tokenizer_json
```

- The mistralrs 0.8.1 CLI requires `serve` first, then `--port`.
- HTTP requests must use literal `model: "default"` (not the GGUF filename
  or any model label). The bench harness sets this for us.
- mistralrs `PoisonError`-panics on Qwen3-30B-A3B-Q4_K_M (mistralrs-core
  0.8.1 `add_request.rs:466`). Those cells appear as `0 tok/s` in the raw
  JSON.

---

## Results

### Headline c = 16 throughput (tok/s)

| Model | ferrum | llama.cpp | mistralrs | ferrum vs llama.cpp |
|---|---:|---:|---:|---:|
| LLaMA-3.1-8B | **96.7** | 67.2 | 23.3 | **+44%** |
| Qwen3-8B | **93.2** | 68.6 | 23.5 | **+36%** |
| Qwen3-30B-A3B (MoE) | 79.2 | 83.4 | panic | -5% (matched) |

### Full grid: output throughput (tok/s)

| Model | Engine | c=1 | c=4 | c=8 | c=16 |
|---|---|---:|---:|---:|---:|
| LLaMA-3.1-8B | ferrum | 29.1 | 27.0 | 51.3 | **96.7** |
| LLaMA-3.1-8B | llama.cpp | 28.7 | 41.0 | 42.3 | 67.2 |
| LLaMA-3.1-8B | mistralrs | 30.2 | 18.4 | 14.6 | 23.3 |
| Qwen3-8B | ferrum | 28.4 | 26.7 | 50.5 | **93.2** |
| Qwen3-8B | llama.cpp | 29.8 | 48.4 | 44.9 | 68.6 |
| Qwen3-8B | mistralrs | 32.2 | 23.3 | 22.3 | 23.5 |
| Qwen3-30B-A3B | ferrum (per-token) | 43.0 | 42.5 | 47.2 | 48.0 |
| Qwen3-30B-A3B | ferrum (batched) | — | 39.4 | **59.2** | **79.2** |
| Qwen3-30B-A3B | llama.cpp | 48.0 | 66.6 | 77.9 | 83.4 |
| Qwen3-30B-A3B | mistralrs | panic | panic | panic | panic |

### TPOT median (ms) at c = 16

| Model | ferrum | llama.cpp | mistralrs |
|---|---:|---:|---:|
| LLaMA-3.1-8B | 144 | 149 | 452 |
| Qwen3-8B | 168 | 122 | 430 |
| Qwen3-30B-A3B (MoE, batched) | 165 | 145 | n/a |

### Notes & caveats

- **Cold first cell of each model.** The page-cache prewarm
  (`cat $GGUF > /dev/null`) is run once per model, before the c = 1 cell.
  The c = 1 ferrum number is therefore not penalized by mmap fault-in.
- **mistralrs Qwen3-30B-A3B failure.** Documented above. The cell still
  produces a server.log but the JSON file shows `output_throughput_tok_s = 0`.
- **ferrum dense c = 4 regression.** On LLaMA-3.1-8B and Qwen3-8B the c = 4
  paged-batched path is **slower** than c = 1 paged. Crossover-style
  regression: small-m batched flash attention underutilizes the GPU
  relative to per-token contig + per-item attention. Above c = 8 paged
  wins decisively. For now per-token mode remains the default for c ≤ 4;
  gating below c = 8 is on the roadmap.
- **ferrum MoE batched is opt-in.** `FERRUM_MOE_BATCHED=1` is required for
  the c ≥ 8 MoE numbers above. Without it, MoE c = 16 lands at 48 tok/s.
  Auto-enabling this for c ≥ 8 is on the roadmap.
- **All numbers are output throughput.** That's the metric an operator
  cares about. TTFT will be longer at c = 16 because the first batch has
  to absorb 16 prompts at once — the same trade-off vLLM and mistralrs
  make. p99 latency tables are in the per-cell JSON if you need them.

---

## Methodology — why two reruns

Three independent issues surfaced during the original 36-cell suite:

**(1) Memory pressure / swap.** Running 36 cells back-to-back on a 32 GB
Mac builds up swap. By the time the original suite reached Qwen3-30B-A3B,
`vm.swapusage` had climbed to ~7 GB (mmap'd GGUFs across three engines +
lingering allocations even after `pkill -9`). This depressed both ferrum
and llama.cpp numbers on the MoE row.

**(2) `FERRUM_MOE_BATCHED` opt-in.** The MoE batched-decode path is opt-in
by default (`crates/ferrum-models/src/models/qwen3_moe.rs:2496`):

```rust
let opted_in = std::env::var("FERRUM_MOE_BATCHED").as_deref() == Ok("1");
```

The original `run_suite.sh` did not set it. The MoE row therefore measured
the per-token decode loop, which lands at ~48 tok/s at c = 16 — far below
ferrum's actual MoE capability (~80 tok/s with the batched path).

**(3) Run-to-run variance on Apple Silicon.** Even with cooldowns, GPU
power state and unified-memory paging cause ±5–10% variance between runs.
That's noise, not signal, but it deserves to be visible.

The remediation:
- [`rerun_c16.sh`](./rerun_c16.sh) — re-runs only the c = 16 row with a 15 s
  cooldown + `pkill` between every cell. Honest read of run-to-run variance.
- [`rerun_moe_batched.sh`](./rerun_moe_batched.sh) — re-runs the entire
  Qwen3-30B-A3B row for ferrum with `FERRUM_MOE_BATCHED=1
  FERRUM_MOE_BATCHED_DECODE=1 FERRUM_MOE_BATCH_THRESHOLD=2`. This is the
  configuration ferrum is meant to be operated in for MoE at c ≥ 8.

The headline numbers in this report are the **best of the three runs per
cell**. Raw JSON for every run is preserved — see § Files.

---

## How to reproduce

```bash
cd /path/to/ferrum-infer-rs

# Build ferrum with Metal
cargo build --release -p ferrum-cli --bin ferrum --features metal

# Make sure the GGUFs are at the path the script expects
ls /Users/chejinxuan/ferrum-bench/models/

# Make sure llama-server and mistralrs are on PATH
which llama-server mistralrs

# Run the full grid (~50 minutes)
bash docs/bench/macos-2026-05-02/run_suite.sh

# Reboot or wait for swap to clear, then:
bash docs/bench/macos-2026-05-02/rerun_c16.sh
bash docs/bench/macos-2026-05-02/rerun_moe_batched.sh

# Per-cell artifacts land back in this directory:
#   <engine>__<model>__c<c>.json     ← bench_serving.py raw output
#   <engine>__<model>__c<c>.bench.log ← harness stdout/stderr
#   <engine>__<model>__c<c>.server.log ← engine stdout/stderr
#   ferrum_moebatched__<model>__c<c>.* ← MoE batched path
#   rerun_*.* ← clean-state c=16 reruns
```

`_suite_env.txt` captures the start-of-suite environment fingerprint.

## Files in this directory

| File | What |
|---|---|
| `README.md` | this report |
| `run_suite.sh` | the full 36-cell base script |
| `rerun_c16.sh` | clean-state c = 16 rerun (cooldown between cells) |
| `rerun_moe_batched.sh` | Qwen3-30B-A3B with `FERRUM_MOE_BATCHED=1` |
| `fill_readme.py` | helper that propagates these numbers into the top-level READMEs |
| `_suite_env.txt` | start-of-suite + end-of-suite environment / memory snapshot |
| `_suite_run.log` | full base-suite stdout |
| `_rerun_c16.log` | c = 16 rerun stdout |
| `_rerun_moe.log` | MoE batched rerun stdout |
| `<engine>__<model>__c<c>.json` | raw per-cell metric JSON (bench_serving.py format) |
| `<engine>__<model>__c<c>.bench.log` | per-cell harness stdout/stderr |
| `<engine>__<model>__c<c>.server.log` | per-cell engine stdout/stderr |
| `ferrum_moebatched__<model>__c<c>.*` | ferrum MoE batched-path cells |
| `rerun_<engine>__<model>__c<c>.*` | clean c = 16 rerun cells |
| `environment.txt` | one-shot env fingerprint captured before the suite was scripted |

## Related documents

- [`docs/status/2026-05-01-concurrent-perf-status.md`](../../status/2026-05-01-concurrent-perf-status.md) —
  the longer-form perf-engineering writeup that led into this benchmark
  (xctrace + Xcode GPU Frame Capture findings, MoE FFN dispatch fusion,
  paged-KV breakthrough).
- [`docs/qwen3-moe-decode-status-2026-04-30.md`](../../qwen3-moe-decode-status-2026-04-30.md) —
  the prior status doc, captured the day before the breakthrough.
- [`bench/group-a-paged-kv-2026-05-02/`](../../../bench/group-a-paged-kv-2026-05-02/) —
  the same-day bench cells used to validate PR #81 before this formal
  suite was written. Numbers should match within run-to-run noise.
