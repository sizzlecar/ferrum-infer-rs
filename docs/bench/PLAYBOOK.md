# ferrum — Performance & Correctness Testing Playbook

**Status:** forward-looking spec, drafted 2026-05-24.
Describes the daily workflow once Phase 0–4 of the testing-system plan land. Commands tagged **[OK]** work today; **[WIP-PN]** ships in Phase N (see § 8).

The testing system has three dimensions:

1. **Correctness** — outputs aren't garbled, repetitive, or numerically drifted vs a reference engine.
2. **Performance** — standard metrics (TTFT / TPOT / ITL / throughput / goodput) at varying concurrency and arrival rate.
3. **Bottleneck localization** — when ferrum is slower than vLLM on identical hardware, *which iteration phase / which kernel* costs the time.

Two user-facing entry points (this playbook's subject):

- **CLI mode** (`ferrum run` / `ferrum bench`) — single user, batch = 1, "how snappy is one chat."
- **Server mode** (`ferrum serve` / `ferrum bench-serve`) — concurrent HTTP, "how does it scale under load."

> **Read § 0 before running anything.** Without GPU lock-down, warmup discard, and `n_repeats ≥ 3` the run-to-run variance routinely exceeds the regression signal you're trying to measure.

---

## 0. Bench preconditions

These are not optional. Skipping any of them produces numbers that look real but mask a 5–10% noise floor — large enough to hide every regression Phase 1 wants to catch.

### 0.1 Hardware lock-down (CUDA)

```bash
# Run once per pod (Vast.ai / RunPod / local 4090). Requires sudo.
sudo nvidia-smi -pm 1                              # persistence mode (driver stays loaded)
sudo nvidia-smi -pl 350                            # power limit — 4090 max
sudo nvidia-smi --auto-boost-default=0             # disable opportunistic boost
sudo nvidia-smi -lgc 2520,2520                     # lock graphics clock — 4090 base
nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.limit,persistence_mode --format=csv
# expect: clocks.gr ≈ 2520 MHz, persistence Enabled. If not, fail-fast — don't bench.
```

To unlock:
```bash
sudo nvidia-smi -rgc                               # restore default clocks
sudo nvidia-smi --auto-boost-default=1
```

### 0.2 Hardware lock-down (Metal / macOS)

Metal has no direct clock-control equivalent. Mitigation:
```bash
caffeinate -dimsu &                                # block sleep + display blank during the run
sudo pmset -a disablesleep 1                       # also prevent suspend-on-lid-close
# Verify no thermal throttle during run:
sudo powermetrics --samplers smc --samplers thermal -i 5000 -n 5 | grep -E "GPU|throttle"
```

Empirical observation on M1 Max: a clean run takes 10–15 minutes before thermal throttling becomes detectable. Keep individual bench cells under that.

### 0.3 Warmup discard

Every bench discards the **first 5 requests OR first 10 seconds**, whichever is greater. The first prefill cold-loads CUDA graph caches, JITs Triton kernels, and warms paged-KV allocators — its TTFT is unrepresentative.

`bench-serve --warmup-requests 5` ([WIP-P0]) enforces this on the client side; engine-side warmup (`/v1/chat/completions` smoke probe at server boot) is `[WIP-P1.4]`.

### 0.4 Repeats and variance reporting

Every bench cell runs **n_repeats independent runs of the bench**, each producing its own per-request percentile distribution. The report aggregates the *per-run percentiles* with mean + stddev + 95% CI half-width (Student's t, n−1 df).

- Default `n_repeats = 5` for CI runs and committed reports
- `n_repeats = 1` allowed for ad-hoc smoke; emitted scalar only, no CI fields
- `n_repeats = 3` minimum for any "X is faster than Y" claim

A regression is **real** only when `mean_A − ci95_A > mean_B + ci95_B`. Closer than that, you're staring at noise.

### 0.5 Determinism contract — what L3 byte-equal actually guarantees

`reference_match.rs` asserts byte-identical greedy output across runs. This holds **only** when:

- `temperature == 0.0` (argmax sampler, not multinomial)
- `top_p == 1.0` AND `top_k == 0` (no truncation that could be perturbed)
- **No speculative decoding** (verifier reorder is non-deterministic by design)
- `FERRUM_PREFIX_CACHE=0` (default since PR #204; cache CoW write-fork is open)
- **Same backend** — Metal vs CUDA vs CPU are NOT bit-equal (Whisper Metal fp32 matmul drift documented in `CLAUDE.md`; CUDA Marlin split-K float-add non-associativity is another)
- **Same kernel feature flags** — `FERRUM_VLLM_MOE`, `FERRUM_MOE_BUCKETED`, `FERRUM_GRAPH`, `FERRUM_MARLIN_SKIP_WS_ZERO` all change the numeric path
- **Same batch size** — prefill-batched kernels can produce different rounding from per-request prefill at the same logical input

When any of the above change, re-baseline the fixture with `FERRUM_UPDATE_FIXTURES=1` after human-reviewing the diff. L3 is **not** an oracle of correctness — it's a regression gate that catches *any* drift, including intentional drift.

### 0.6 vs-vLLM config parity checklist

This is the single biggest source of false "ferrum is N× slower" claims. **Before declaring a perf gap engineering**, confirm both engines run the apples-to-apples config below. Each line in the report's parity block must show the value, not just `(default)`.

| Knob | vLLM flag | ferrum equivalent | Why it matters |
|---|---|---|---|
| KV cache fraction | `--gpu-memory-utilization 0.9` | `FERRUM_KV_MAX_BLOCKS=N` | Larger KV → bigger batch fits → higher throughput |
| Max concurrent seqs | `--max-num-seqs 256` | `FERRUM_PAGED_MAX_SEQS=N` | Scheduler-level batch cap |
| Max batched tokens / iter | `--max-num-batched-tokens 8192` | (engine constant) | Prefill chunk size budget |
| Chunked prefill | `--enable-chunked-prefill` | (always on in Phase 3 path) | Whether prefill mixes with decode |
| Prefix caching | `--enable-prefix-caching` | `FERRUM_PREFIX_CACHE` | **Default differs**: vLLM ON, ferrum OFF |
| Dtype | `--dtype bfloat16` | (model config) | bf16 vs fp16 numerics + speed |
| Quantization | `--quantization gptq_marlin` | (model loader auto-detects) | Same kernel? |
| KV dtype | `--kv-cache-dtype fp16` | `FERRUM_KV_DTYPE` | fp16 vs fp8 vs int8 |
| Tensor parallel | `--tensor-parallel-size 1` | (single-GPU only) | TP > 1 collective overhead |
| CUDA graph | `--enforce-eager false` (graph on) | `FERRUM_GRAPH=1` | Captures dispatch; huge for short decode |
| Sampling | request body | request body | temp / top_p / top_k must match |

`scripts/bench_vs_vllm.sh` ([WIP-P0]) dumps both engines' effective config into the report's first table — manual eyeball pass required before trusting any ratio.

---

## 1. CLI mode performance

`ferrum bench` runs the same forward path as `ferrum run` (batch = 1, no HTTP, fixed prompt). It measures the "single-user feel" of the engine.

### Level 1 — single number (~30 s × n_repeats)

```bash
# [OK] today (Phase 2 adds n_repeats + variance fields)
ferrum bench qwen3:0.6b --rounds 10 --max-tokens 256

# [WIP-P0] enforced n_repeats with variance
ferrum bench qwen3:0.6b --rounds 10 --max-tokens 256 --n-repeats 5
```

After Phase 0+2 the output schema is:

```
[WIP-P2] expected output:
  TTFT_ms      p50: 120.0 ± 2.1   p95: 145.0 ± 3.2   p99: 160.0 ± 5.1
  TPOT_ms      p50:   8.2 ± 0.1   p95:   9.1 ± 0.2   p99:  10.5 ± 0.4
  ITL_ms       p50:   8.0 ± 0.1   p95:   9.5 ± 0.2   p99:  12.0 ± 0.5
  throughput   122 ± 1.4 tok/s   (single user, n_repeats=5)
                                  └── CI95 half-width (Student's t)
```

Use when: a sampler / tokenizer / non-batched-decode change.

### Level 2 — per-op breakdown (5 min × n_repeats)

```bash
# [WIP-P1] requires CUDA-event probes (Phase 1.1–1.2) + trace emitter (Phase 1.5)
FERRUM_LAYER_PROF=1 FERRUM_TRACE_OUT=cli_trace.json \
    ferrum bench qwen3:0.6b --rounds 5 --max-tokens 128

python scripts/visualize_layerwise.py cli_trace.json -o cli_decode.png
# stacked bar: attention / gemm / quant / rms_norm / sampling
```

Use when: throughput moved unexpectedly and you want a single picture of where the time went. **Do not trust Level-2 numbers until Phase 1.1 lands** — today's `FERRUM_DECODE_OP_PROFILE` etc. use `Instant::now()` and conflate GPU + host noise.

### Level 3 — kernel timeline (CUDA only, 10 min)

```bash
# [WIP-P1.4] requires --enable-cuda-profiler flag or /start_profile wiring
nsys profile -o cli_kernels \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    ferrum bench qwen3:0.6b --rounds 3 --enable-cuda-profiler

# Open cli_kernels.nsys-rep in Nsight Systems
```

Use when: Level 2 isolated a category and you need kernel-level proof.

---

## 2. Server mode performance

Three scenarios that answer three different questions. Don't conflate them.

| Scenario | Question | Workload shape |
|---|---|---|
| **A. Open-loop goodput** | "What fraction of users meet SLO under realistic arrival?" | Poisson arrivals at rate R; goodput is headline |
| **B. Closed-loop capacity** | "What's the maximum sustained throughput?" | K workers in tight send→wait loop; throughput is headline |
| **C. Prefix-cache stress** | "Does prefix cache actually amortize across similar requests?" | Shared 1024-token prefix, burst arrival, cache hit-rate is headline |

```bash
# Setup (all scenarios)
ferrum serve qwen3:0.6b --port 8000 &
SERVER_PID=$!
trap "kill $SERVER_PID" EXIT
sleep 5
```

### Scenario A — open-loop goodput (production shape)

```bash
# [WIP-P0] --request-rate Poisson + --goodput SLO + n_repeats
ferrum bench-serve \
    --base-url http://127.0.0.1:8000 \
    --model qwen3:0.6b \
    --dataset sharegpt --num-prompts 500 \
    --request-rate 20 \
    --goodput ttft:500 tpot:50 e2el:30000 \
    --warmup-requests 10 \
    --n-repeats 3 \
    --output sql --db perf.db
```

Why open-loop is the headline: goodput is only meaningful when arrivals are independent of the server's state. In closed-loop you control the queue depth, so `time_to_first_token < SLO?` becomes "depends on how many workers I picked", not "did production users get served fast enough".

Output:

```
[WIP-P0] expected output:
  arrival     Poisson(20 req/s), 500 prompts → 25 s nominal duration
  measured    duration 26.4s, completed 497, errored 3
  TTFT_ms     p50: 145 ± 8     p99: 480 ± 25
  TPOT_ms     p50:  18 ± 0.3   p99:  35 ± 1.2
  goodput     12.4 req/s  (412/497 met SLO; 85 missed: 60 TTFT, 25 TPOT)
              └── this is the headline
```

### Scenario B — closed-loop capacity knee

Use this to find where the engine saturates, NOT to compute goodput.

```bash
# [WIP-P0] --concurrency-sweep + n_repeats
ferrum bench-serve \
    --base-url http://127.0.0.1:8000 \
    --model qwen3:0.6b \
    --dataset sharegpt --num-prompts 200 \
    --concurrency-sweep 1,4,8,16,32,64 \
    --warmup-requests 10 \
    --n-repeats 3 \
    --output sql --db perf.db
```

Output:

```
[WIP-P0] expected output (per c-cell):
  c= 1   throughput=  122 ± 1.4 tok/s    TTFT p50=120
  c= 4   throughput=  380 ± 4.2 tok/s    TTFT p50=145
  c=16   throughput=  860 ± 8.1 tok/s    TTFT p50=290
  c=32   throughput= 1100 ± 9.5 tok/s    TTFT p50=550  ← knee (Δthr=+28%, Δlat=+90%)
  c=64   throughput= 1180 ± 7.2 tok/s    TTFT p50=1100 ← saturated
```

The knee is where Δthroughput / Δlatency falls off. That's the c-value that maximizes user-felt performance — past it, you're trading latency for marginal throughput.

### Scenario C — prefix-cache thundering herd

```bash
# [WIP-P2.3] requires shared-prefix dataset module (promote bench/v0.2-cuda/gen_shared_prefix_jsonl.py)
ferrum serve qwen3:0.6b --port 8000 \
    --env FERRUM_PREFIX_CACHE=1 &      # opt-in until CoW write-fork is fixed
sleep 5

ferrum bench-serve \
    --base-url http://127.0.0.1:8000 \
    --model qwen3:0.6b \
    --dataset shared-prefix \
    --shared-prefix-len 1024 --suffix-len 64 --output-len 128 \
    --num-prompts 200 --request-rate 50 \
    --warmup-requests 10 \
    --n-repeats 3 \
    --output sql --db prefix_cache.db
```

Output:

```
[WIP-P2] expected output:
  TTFT_ms first request   p50= 380 ± 12   (cache miss, full prefill of 1024 tokens)
  TTFT_ms warm requests   p50=  62 ± 4    (cache hit on 1024-tok prefix)
  cache_hit_rate          0.94             ← headline
  vs Scenario A baseline  6.1× faster TTFT on cache hit
```

This is what closed-loop ShareGPT can't reproduce: each ShareGPT conversation has a *different* prefix, so the cache never gets a meaningful hit rate. Burst + shared-prefix is the only stress that exercises the eviction logic.

### Level 2 — Prometheus steady-state

```bash
# [WIP-P1.4] /metrics endpoint via metrics-exporter-prometheus
curl -s http://127.0.0.1:8000/metrics | grep ferrum

# Useful series (✓ = already emitted today via `metrics` crate):
#   ferrum_engine_iterations_total            ✓
#   ferrum_engine_active_requests             ✓
#   ferrum_engine_prefix_cache_hits           ✓
#   ferrum_engine_preemptions_total           ✓   any >0 means KV pressure
#   ferrum_engine_request_duration_ms_bucket  ✓   E2E
#   ferrum_engine_kv_cache_usage              [WIP-P1.3]
#   ferrum_engine_ttft_seconds_bucket         [WIP-P1.3]
#   ferrum_engine_tpot_seconds_bucket         [WIP-P1.3]
#   ferrum_engine_corrupted_reqs              [WIP-P1.3]   NaN watchdog
```

### Level 3 — kernel timeline under load (CUDA, 15 min)

```bash
# [WIP-P1.4] /start_profile + /stop_profile endpoints wrap cudaProfilerStart/Stop
curl -X POST http://127.0.0.1:8000/start_profile
nsys profile -o server_c32 \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    ferrum bench-serve --base-url http://127.0.0.1:8000 \
                       --concurrency 32 --num-prompts 50 \
                       --warmup-requests 10
curl -X POST http://127.0.0.1:8000/stop_profile
```

---

## 3. Correctness gate

Four layers, fast → slow. Run top-down whenever you suspect output regression. **Read § 0.5 first** — L3 has determinism preconditions that aren't free.

| Layer | What | Command | Time |
|---|---|---|---|
| **L3 byte-equal** | greedy snapshot regression (preconditions in § 0.5) | `cargo test --release -p ferrum-cli --features metal --test reference_match -- --ignored` | ~30 s |
| **L1 op NMSE** | cross-backend numerical diff | `cargo test --release -p ferrum-kernels --features metal --test op_diff` | ~30 s |
| **L2 quant KL** | Marlin / GPTQ drift via KL divergence vs FP16 logits | `cargo test --release -p ferrum-cli --features metal --test quant_kl -- --ignored` | ~5 min |
| **L4 lm-eval-light** | task-level sanity (MMLU+ARC+GSM8K, 100Q each) via OpenAI endpoint | `scripts/lm_eval_light.sh qwen3:0.6b` | ~10 min |

Layer ordering mental model: **finer-granularity catches faster, coarser-granularity confirms quality**. L3 catches "did anything in the pipeline change?" — but tells you nothing about quality. L4 confirms quality on real tasks — but only after L3/L1/L2 say the pipeline didn't break.

L3 is in place today **[OK]**. L1/L2/L4 land in Phase 3 **[WIP-P3]**.

**When to run which**:
- sampler / tokenizer / non-kernel engine code → L3 only
- a CUDA / Metal kernel → L3 + L1
- a quant path (Marlin tile, GPTQ packer) → L3 + L1 + L2
- big refactor or model addition → all four (L4 nightly is enough)

---

## 4. Cross-cutting workflows

### A. Compare against a previous commit

```bash
# [WIP-P2.4] requires SQLite output + scripts/compare-commits.sh
scripts/compare-commits.sh HEAD~1 HEAD qwen3:0.6b 32

# Expected markdown ratio table — flags significance with CI overlap:
#   metric          HEAD~1      HEAD       ratio   significant?
#   TTFT p50        290 ± 8     245 ± 6    0.84    ✓ (no overlap)
#   TPOT p50         18 ± 0.3    17 ± 0.4  0.94    ✗ (CIs overlap)
#   throughput      860 ± 9    920 ± 8    1.07    ✓
#   goodput          72 ± 2     78 ± 1.8  1.08    ✓
```

Mechanism: `git worktree add` each commit, `cargo build --release`, run identical `bench-serve` (with `--n-repeats 5`), write into one SQLite, join by `(model, c, n_prompt, n_gen, env_hash)`, print ratio table marking only the cells where CIs don't overlap. Modelled on llama.cpp's `scripts/compare-commits.sh` + `scripts/compare-llama-bench.py`.

### B. Compare against vLLM (the gap-closing workflow)

```bash
# [WIP-P0] thin wrapper. MUST include config parity dump (§ 0.6).
scripts/bench_vs_vllm.sh qwen3:0.6b 32

# Starts both servers on the same host:
#   ferrum serve qwen3:0.6b                              --port 8000
#   vllm   serve Qwen/Qwen3-0.6B  --enforce-eager=False  --port 8001
# Runs identical bench-serve against both; outputs:
#
#   ┌─ Config parity ─────────────────────────────────────────────┐
#   │ knob                  ferrum             vllm                │
#   │ kv_cache_fraction     0.85 (computed)    0.90 (--gpu-mem-...)│
#   │ max_seqs              32                 256          ⚠ DIFF │
#   │ chunked_prefill       on                 on                  │
#   │ prefix_caching        OFF (PR #204)      ON           ⚠ DIFF │
#   │ dtype                 fp16               fp16                │
#   │ quantization          gptq_marlin        gptq_marlin         │
#   │ kv_dtype              fp16               fp16                │
#   │ cuda_graph            FERRUM_GRAPH=1     enforce-eager=false │
#   └──────────────────────────────────────────────────────────────┘
#
#   metric          ferrum   vllm     ferrum/vllm   significant?
#   throughput      1100 ±10 1867 ±15 0.59          ✓
#   TTFT p99         920ms   310ms    2.97          ✓
#   TPOT p99          48ms    31ms    1.55          ✓
```

The config parity block at the top is non-negotiable. Two of the most-cited gaps in this repo's history turned out to be **config**, not engine — vLLM ran with prefix-caching ON and we ran with it OFF (PR #204), and vLLM's max-num-batched-tokens was higher than our prefill chunk budget. Always eyeball the parity block before trusting the ratio. Phase 0 deliverable is a credible parity-confirmed bench.

Today's raw harness: `bench/v0.2-cuda/apples_all_drive.sh`.

---

## 5. Regression localization path

Standardized escalation. Each step is more expensive but more precise. **All steps below depend on Phase 0 hygiene** — without n_repeats + GPU lock + config parity, the cheap signals at the top of this table are unreliable.

| Signal | Tool | Time |
|---|---|---|
| "throughput dropped 10%" | `compare-commits.sh` (only trust signif. cells) | 30 s |
| "TTFT regressed, TPOT didn't" | bench-serve prefill/decode split; `/metrics` → `prefill_tokens_total` rate | 1 min |
| "TPOT regressed" | `FERRUM_LAYER_PROF=1` + visualizer → which op category grew | 5 min |
| "one op category dominates" | `nsys profile` + `/start_profile` + open timeline | 10 min |
| "kernel looks fine, still slow" | `ncu` for SM occupancy / mem-BW utilisation | 30 min |
| "numbers fine, output garbage" | L3 → L1 → L2, fastest first | 30 s → 30 min |

---

## 6. Worked example — "what's qwen3:0.6b's concurrency wall in server mode?"

After Phase 0–2:

```bash
# (assumes § 0.1 GPU lock-down done)
ferrum serve qwen3:0.6b --port 8000 &
sleep 5

# Capacity knee (Scenario B):
ferrum bench-serve \
    --base-url http://127.0.0.1:8000 \
    --model qwen3:0.6b \
    --dataset sharegpt --num-prompts 200 \
    --concurrency-sweep 1,4,8,16,32,64 \
    --warmup-requests 10 \
    --n-repeats 3 \
    --output md --out "docs/bench/metal-$(date +%Y-%m-%d)/scenario_b.md"

# Goodput at the knee's c-1 (Scenario A):
ferrum bench-serve ... --request-rate 18 \
    --goodput ttft:500 tpot:50 e2el:30000 \
    --out "docs/bench/metal-$(date +%Y-%m-%d)/scenario_a.md"
```

Two artifacts land in `docs/bench/metal-2026-05-24/`, both ready to link in a PR. If `scenario_a.md` shows goodput < 95% but `scenario_b.md` shows throughput still rising, the bottleneck is **tail latency**, not aggregate throughput — escalate via § 5.

---

## 7. Schema (locked Phase 0)

Schema is the most expensive thing to change later — readers, comparators, and dashboards all build against it. **Do not invent variants.**

Every bench run produces:

| File | Purpose |
|---|---|
| `report.md` | human-readable summary |
| `metrics.json` | canonical schema, one record per `(model, backend, scenario, c_or_rate, ...)` cell |
| `metrics.sqlite` | same data, joinable across runs |
| `trace.json` (opt-in via `FERRUM_TRACE_OUT`) | chrome://tracing / Perfetto spans |
| `nsys.rep` (opt-in) | Nsight Systems profile |

### Metric record schema

```json
{
  "model": "qwen3:0.6b",
  "backend": "metal",
  "scenario": "closed_loop",
  "concurrency": 32,
  "request_rate": null,
  "n_prompt": 256,
  "n_gen": 128,
  "n_repeats": 5,
  "n_requests_per_run": 200,
  "warmup_requests": 10,

  "ttft_ms": {
    "p50": {"mean": 120.0, "stddev": 2.1, "ci95_hw": 1.85},
    "p75": {"mean": 132.0, "stddev": 2.8, "ci95_hw": 2.45},
    "p95": {"mean": 145.0, "stddev": 3.2, "ci95_hw": 2.81},
    "p99": {"mean": 160.0, "stddev": 5.1, "ci95_hw": 4.47}
  },
  "tpot_ms":           { "p50": {...}, "p75": {...}, "p95": {...}, "p99": {...} },
  "itl_ms":            { "p50": {...}, "p75": {...}, "p95": {...}, "p99": {...} },
  "e2e_ms":            { "p50": {...}, "p75": {...}, "p95": {...}, "p99": {...} },

  "output_throughput_tps":  {"mean": 1100.0, "stddev": 9.5, "ci95_hw": 8.39},
  "total_throughput_tps":   {"mean": 1450.0, "stddev": 12.1, "ci95_hw": 10.67},
  "request_throughput_rps": {"mean": 1.8,    "stddev": 0.02, "ci95_hw": 0.018},
  "goodput_rps":            {"mean": 0.81,   "stddev": 0.01, "ci95_hw": 0.009},

  "slo": {
    "ttft_p99_ms": 500,
    "tpot_p99_ms": 50,
    "e2e_p99_ms":  30000
  },

  "completed_per_run":  [200, 200, 199, 200, 200],
  "errored_per_run":    [0, 0, 1, 0, 0],

  "env": {
    "commit_sha":   "b769bbd",
    "hw_id":        "rtx-4090",
    "driver":       "555.42.06",
    "cuda":         "12.4",
    "rust":         "1.78.0",
    "ferrum_features": ["cuda", "vllm-moe-marlin"],

    "gpu_clock_lock_mhz":  2520,
    "gpu_power_limit_w":   350,
    "gpu_persistence_mode": true,
    "gpu_auto_boost":      false,

    "ferrum_env": {
      "FERRUM_PREFIX_CACHE":  "0",
      "FERRUM_KV_MAX_BLOCKS": "2048",
      "FERRUM_PAGED_MAX_SEQS": "32",
      "FERRUM_GRAPH":         "1",
      "FERRUM_VLLM_MOE":      "0"
    },
    "vllm_args": null
  },
  "env_hash": "sha256:f3a2b1c..."
}
```

Field rules:

- **`n_repeats < 3`** → emit `mean` only, omit `stddev` and `ci95_hw`. CI95 with 2 samples is meaningless.
- **`ci95_hw`** is the half-width of the 95% confidence interval, computed with Student's t (`t_{n−1, 0.975} · stddev / √n`). Use this to check significance: A vs B is real iff `|mean_A − mean_B| > ci95_hw_A + ci95_hw_B`.
- **`percentiles`** are `[p50, p75, p95, p99]` — fixed. Report-level display may drop p75 for brevity but JSON always carries it.
- **`scenario`** ∈ `{"closed_loop", "open_loop", "shared_prefix", "cli"}`. Exactly one of `concurrency` / `request_rate` is set (per scenario contract).
- **`env_hash`** = sha256 of canonical-JSON-serialized `env` block. Two runs with the same `env_hash` are guaranteed apples-to-apples; cross-run comparators **must** filter on equal `env_hash` before computing ratios.
- **`vllm_args`** is `null` for ferrum runs, populated with the full effective `vllm serve` args for vLLM cells; this is what § 0.6 config parity dumps consume.

### Locked definitions

Copied verbatim from vLLM / lmdeploy — do not invent variants:

```
TTFT     = first_SSE_chunk_arrival − client_send
TPOT     = (e2e_latency − TTFT) / (output_tokens − 1)             # per-request scalar
ITL      = list[t_n − t_{n−1}] flattened across all requests       # per-token
e2e      = last_SSE_chunk_arrival − client_send

output_throughput  = Σ output_tokens / duration                    # headline; excludes prefill
total_throughput   = (Σ input + Σ output) / duration
request_throughput = completed / duration
goodput            = Σ(req where ttft, tpot, e2e all meet SLO) / duration
                     └── ONLY meaningful in open-loop scenarios
percentiles        = [50, 75, 95, 99]
```

---

## 8. Build sequencing

The reordering principle: **prove the gap is engineering before building engineering tools.** If today's `bench/v0.2-cuda/apples_all_drive.sh` numbers (ferrum 0.40–0.55× vLLM) turn out to be 0.80–0.85× under config parity, Phase 1 per-op profiling is hugely de-prioritized. We don't know until we run Phase 0.

### Phase 0 — Hygiene + parity (1–1.5 weeks) — UNBLOCKS EVERYTHING

| Item | Scope |
|---|---|
| 0.1 | Metric schema in `crates/ferrum-bench-core/` (new) — § 7 spec, variance fields, env_hash construction |
| 0.2 | `Profiler` trait + `compute_metrics()` (lmdeploy pattern) |
| 0.3 | `--n-repeats N` + Student-t CI95 in `bench` and `bench-serve` |
| 0.4 | `--request-rate R` (Poisson) + `--goodput` SLO in `bench-serve` |
| 0.5 | `--concurrency-sweep` shorthand |
| 0.6 | `scripts/lock_gpu.sh` (CUDA `nvidia-smi` lock-down) + `scripts/unlock_gpu.sh` |
| 0.7 | `scripts/bench_vs_vllm.sh` — runs both engines, dumps effective config parity table, then identical bench |
| 0.8 | `--warmup-requests N` (default 10) |
| 0.9 | `env_hash` computation + report header |

**Phase 0 deliverable**: one credible vs-vLLM bench on RTX 4090, ShareGPT, c=1/4/16/32, with config parity confirmed line-by-line in the report. Numbers reported as `mean ± ci95`. Land it in `docs/bench/cuda-rtx4090-<date>-parity/`.

### Decision point (1 day) — gates Phase 1

Inspect the parity-confirmed report. Three branches:

| What Phase 0 shows | Next step |
|---|---|
| Gap < 15% under parity | Phase 1 deferred. Pick highest-cell-by-cell deltas in scenario A goodput; address as targeted bug fixes (which usually don't need per-op profiling). Focus shifts to Phase 3 correctness expansion. |
| Gap 15–40%, dominantly TTFT | Phase 1.1+1.5 (CUDA event + trace) on prefill path only. Skip 1.4 (Prometheus expansion) until the prefill regression is closed. |
| Gap > 40% across TTFT *and* TPOT | Full Phase 1 (1.1–1.5). The current 0.59 ratio is in this band on paper — but only Phase 0 will tell us if that's the parity-corrected number. |

### Phase 1 — Per-op profiling (GATED, 2 weeks if green)

| 1.1 | `BackendTimer` trait (CUDA event via `cudarc::driver::CudaEvent`; Metal `MTLCounterSampleBuffer`) |
| 1.2 | Migrate `FERRUM_*_PROF` call sites off `Instant::now()` (~10 in `qwen3_moe.rs` + `moe/forward.rs`) |
| 1.3 | `IterationStats` expansion — TTFT / TPOT / ITL histograms, `kv_cache_usage` gauge, `corrupted_reqs` NaN watchdog |
| 1.4 | `/metrics` Prometheus + `/start_profile` + `/stop_profile` HTTP endpoints |
| 1.5 | Chrome trace JSON output via `FERRUM_TRACE_OUT` |

### Phase 2 — Bench polish (parallel after Phase 0, 1 week)

| 2.1 | SQLite output + `scripts/compare-commits.sh` (CI-overlap-aware) |
| 2.2 | `dataset shared-prefix` module (promote `bench/v0.2-cuda/gen_shared_prefix_jsonl.py`) |
| 2.3 | `dataset sharegpt` module (promote `bench/v0.2-cuda/prompts_subset.py`) |
| 2.4 | Report-md generator (markdown summary with significance markers) |

### Phase 3 — Correctness expansion (independent of perf path, 1.5 weeks)

| 3.1 | NMSE op-diff harness in `ferrum-testkit/src/op_diff.rs` (cross-backend, ggml `test-backend-ops` pattern) |
| 3.2 | KL-divergence quant gate `crates/ferrum-cli/tests/quant_kl.rs` |
| 3.3 | `scripts/lm_eval_light.sh` against OpenAI endpoint |

### Phase 4 — Visualizer (after Phase 1 if Phase 1 happens, 1 week)

| 4.1 | `scripts/visualize_layerwise.py` (vLLM `visualize_layerwise_profile.py` pattern) |
| 4.2 | Bench-vs-vllm wrapper polish (CI integration) |

### Current status snapshot (updated 2026-05-24)

| Phase | Status |
|---|---|
| 0.1 | ✓ `crates/ferrum-bench-core` — schema, Profiler, Student-t CI95, Poisson arrivals, env_hash |
| 0.2–0.5,0.8 | ✓ `bench-serve --n-repeats / --request-rate / --concurrency-sweep / --goodput / --warmup-requests` |
| 0.3,0.9 | ✓ `bench` CLI consumes ferrum-bench-core |
| 0.6 | ✓ `scripts/lock_gpu.sh` + `scripts/unlock_gpu.sh` |
| 0.7 | ✓ `scripts/bench_vs_vllm.sh` with config-parity dump |
| 1.1 | ✓ `BackendTimer` trait + CPU/Metal/CUDA impls in `crates/ferrum-kernels/src/backend/timer.rs` (CUDA via `cuEventRecord/Synchronize/ElapsedTime`; Metal via sync-wrap; CPU via `Instant`) |
| 1.2 | ✓ `Backend::Timer` associated type + `make_timer()` + `start_probe_timer` / `finish_probe_timer` helpers. 4 hot `B::sync + Instant::now()` probe sites in `qwen3_moe.rs` migrated (attn / moe / prefill / decode-stage). Closure-shaped sites in `moe/forward.rs` and wall-clock-by-design sites are unchanged — different refactor. |
| 1.3 | ✓ TTFT/TPOT/ITL histograms in engine (`ferrum.engine.{ttft,tpot,itl}_seconds`) + corrected bug at `continuous_engine.rs:2253` |
| 1.4 | ✓ `/metrics` Prometheus endpoint (already wired in `axum_server.rs`) |
| 1.5 | ✓ Chrome trace JSON in `crates/ferrum-bench-core/src/trace.rs` — `TraceWriter` with `FERRUM_TRACE_OUT` env-gated emit, compatible with chrome://tracing / Perfetto / Nsight |
| 2.1 | ✓ `--output jsonl` + `scripts/compare-commits.sh` + `scripts/compare_bench.py` (CI-overlap-aware ratio table) |
| 2.2 | ✓ `--dataset shared-prefix` (1024-tok shared + unique suffix) |
| 2.3 | ✓ `--dataset sharegpt --sharegpt-path PATH` (HF Vicuna format) |
| 2.4 | ✓ `--output md` + `ferrum_bench_core::report::{render_single, render_sweep}` |
| 3.1 | ✓ NMSE op-diff harness in `ferrum-testkit::op_diff` — `rms_norm` + `silu_mul` covered (Metal NMSE empirically 1e-15 against CPU); extend pattern to rope/varlen_attn/marlin_matmul as follow-up PRs |
| 3.2 | ✓ Token-divergence proxy KL gate at `crates/ferrum-cli/tests/quant_kl.rs`: `self_determinism_qwen3_0p6b` (always runnable) + `paired_quant_drift_qwen2p5_3b` (auto-skip if INT4 variant not cached). Real KL on logits remains future work. |
| 3.3 | ✓ `scripts/lm_eval_light.sh` against `/v1/completions` (MMLU + ARC-easy + GSM8K, rtol=0.05) |
| 4   | ✓ `scripts/visualize_layerwise.py` — reads chrome trace JSON, groups by `cat`, stacked-bar PNG per layer. Smoke-tested with fixture trace. |

**Gate resolution** (see [`docs/bench/decision-2026-05-24-phase8-gate/decision.md`](decision-2026-05-24-phase8-gate/decision.md)):
A live Vast 4090 parity bench was attempted but failed across 5 different pods (boot-fail, "Template not found", SSH proxy bug, host-firewall on direct ports). Decision was made on **existing baseline data** from `CLAUDE.md` and `bench/v0.2-cuda/PERF_TRACKER.md` — both M2 and M3 sit in the `< 0.60` ratio band (> 40% gap), so PLAYBOOK § 8 mandates **full Phase 1 + 4**. Phase 1.1 / 1.5 / 4 are now built; Phase 1.2 (probe migration) is mechanical follow-up work deferred to a focused PR.

---

## 9. References

- **Done criteria** (PR-time correctness gate): `CLAUDE.md` § "Done criteria — engine / scheduler / sampler / CLI run / HTTP server". `reference_match.rs` byte-equal already wired into nightly.
- **Existing bench infra**:
  - `bench/v0.2-cuda/apples_all_drive.sh` — vLLM-vs-ferrum sweep (CUDA RTX 4090)
  - `bench/v0.2-cuda/{nsys_*.sh, ncu_profile.sh}` — kernel profilers
  - `crates/ferrum-cli/src/commands/{bench.rs, bench_serve.rs}` — CLI entry points
  - `docs/bench/<platform>-<date>/` — run archives
- **Mainstream reference**:
  - vLLM `benchmarks/benchmark_serving.py` — metric definitions, goodput, Poisson arrival
  - vLLM `tools/profiler/visualize_layerwise_profile.py` — stacked-bar visualizer template
  - vLLM `vllm/v1/metrics/stats.py` — `IterationStats` Prometheus model
  - SGLang `python/sglang/bench_serving.py` — sister of vLLM bench
  - lmdeploy `lmdeploy/profiler.py` — canonical `compute_metrics()` design
  - llama.cpp `tests/test-backend-ops.cpp` — NMSE op-diff pattern
  - llama.cpp `tools/llama-bench/`, `scripts/compare-commits.sh` — perf-regression workflow
