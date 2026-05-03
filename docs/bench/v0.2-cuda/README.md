# ferrum v0.2 — CUDA / RTX 4090 Benchmark Plan

**Status:** drafted 2026-05-03, not yet executed
**Target test runner:** RunPod RTX 4090 (24 GB), $0.40/hr, ≤ $10 total
**Headline goal:** Publish a v0.2 bench report comparing **ferrum vs vLLM vs mistralrs** across **4 model × precision configs × 4 concurrency levels = 48 cells × 3 repeats = 144 runs**, with the same audit-quality rigor as the macOS Group A bench at [`docs/bench/macos-2026-05-02/`](../macos-2026-05-02/).

This doc is the **plan**. The actual run, raw artifacts, and final report will be at `docs/bench/cuda-rtx4090-2026-05-XX/` (date filled in when we kick off).

---

## 1. What we're proving

| Claim | Bench cell that proves it |
|---|---|
| ferrum's CUDA decode runner holds up at single-request | c=1 across all 4 models |
| ferrum's continuous batching scales to c=16/32 | c=16, c=32 across Llama-8B INT4 + 30B-A3B INT4 |
| ferrum is competitive with vLLM in the price-conscious-deployment lane | Side-by-side same hardware / same dataset |
| ferrum doesn't fall apart on MoE under concurrent load | Qwen3-30B-A3B + Qwen3-Coder-30B-A3B at c=16+ |
| GPTQ INT4 + Marlin gets us to vLLM-level memory efficiency | INT4 cells fit comfortably; FP16-8B is the contrast |

Non-goal: be the fastest engine on H100 8×. We're pitching **single-GPU 4090 / single binary / Python-free** — the bench has to mirror that deployment.

---

## 2. Budget math

| Item | Estimate |
|---|---|
| Wall-clock at $0.40/hr | 25 hours (≤ $10) |
| Cold-start engine (boot + load) | 30 s (8B INT4) → 90 s (30B INT4) |
| Per bench cell wall time | c=1: ~60 s · c=4: ~150 s · c=16: ~120 s · c=32: ~120 s |
| Total bench wall | ~7-8 hr (see § 6 for derivation) |
| Setup + model download | 30-45 min one-time |
| Buffer (debugging, OOM retries, partial reruns) | ~6 hr |

**Cost target: $5-7 used, $3-5 buffer.** Headroom matters because the first 1-2 hours **will** discover something (vLLM CUDA mismatch, Qwen3-Coder-30B GPTQ doesn't exist, mistralrs panics on MoE again, ...).

---

## 3. Test matrix

### 3.1 Models × precision

| # | Model | Precision | HF source | On-disk |
|---|---|---|---|---|
| M1 | Llama-3.1-8B-Instruct | FP16 | `unsloth/Meta-Llama-3.1-8B-Instruct` (open mirror — meta-llama/* requires per-account approval) | 16 GB |
| M2 | Llama-3.1-8B-Instruct | GPTQ-INT4 | `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4` | 5.7 GB |
| M3 | Qwen3-30B-A3B | GPTQ-INT4 | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` (official Qwen org) | 17 GB |

**Out of v0.2 scope**: ~~M4 Qwen3-Coder-30B-A3B~~ — only available as community pack (no official Qwen GPTQ); one variable too many.

### 3.2 Engines

| # | Engine | Pin | Notes |
|---|---|---|---|
| E1 | ferrum | this repo @ branch `bench/v0.2-cuda` (`cargo build --release --features cuda`) | Marlin INT4 path; paged-KV auto |
| E2 | vLLM | `vllm==0.20.0` (latest PyPI; **requires CUDA ≥12.6 / driver ≥555** — pick pod accordingly) | `--quantization gptq_marlin` for INT4 |

**Out of v0.2 scope**: ~~mistralrs~~ — PoisonError on MoE (per macOS bench), brittle install path, not the primary point of comparison.

### 3.3 Workloads

| # | Workload | c | Prompt tokens | Output tokens | Total tokens/req |
|---|---|---:|---:|---:|---:|
| W1 | single decode | 1 | 128 | 512 | 640 |
| W2 | mid concurrency | 4 | 512 | 256 | 768 |
| W3 | high concurrency | 16 | 512 | 256 | 768 |
| W4 | extreme concurrency | 32 | 512 | 256 | 768 |

Per-c prompt counts (so the run isn't dominated by head/tail): **`num_prompts = 4 × c`** → 4, 16, 64, 128.

### 3.4 Held constant across all cells

- **Compute precision**: BF16 (or whatever the engine defaults to for INT4 matmul; vLLM does FP16/BF16 + Marlin INT4 dequant on-the-fly, ferrum ditto)
- **KV cache dtype**: FP16 (vLLM `--kv-cache-dtype auto` defaults to model dtype; ferrum's KV is FP16 by default)
- **Sampler**: greedy / temperature 0 / top-p 1 (deterministic — every run produces the same logits, so noise is engine, not RNG)
- **Tokenizer**: HF official tokenizer.json shipped with the model
- **Bench harness**: vLLM's `benchmarks/benchmark_serving.py` — same script, same dataset, same protocol (OpenAI SSE)
- **Dataset**: ShareGPT v3 unfiltered, deterministic 128-prompt subset (seeded), padded/truncated to W's prompt-length budget
- **Prefix cache**: explicitly disabled on every engine (otherwise the deterministic prompt set hits the cache after the first repeat and the second/third repeats land at unrealistic throughput)

### 3.5 Cell counting

```
3 models × 2 engines × 4 workloads = 24 cells
× 3 repetitions (for noise floor / median)
= 72 runs total
```

(Was 144 in the original plan; halved by dropping mistralrs + M4.)

**Reporting**: median of 3 (drops the worst run). p50/p95 of TTFT and TPOT come from the per-request distribution **within** each median run — not across runs.

---

## 4. Held metric definitions

To prevent ambiguity at report time:

| Metric | Definition | Where it comes from |
|---|---|---|
| Aggregate output throughput (tok/s) | total output tokens / wall time of the bench | `benchmark_serving.py: output_throughput_tok_s` |
| TTFT (Time to First Token) | wall time from POST sent to first SSE chunk | `benchmark_serving.py: ttft_ms` distribution |
| TPOT (Time per Output Token) | (E2E time − TTFT) / (output tokens − 1), per request | `benchmark_serving.py: tpot_ms` |
| ITL (Inter-Token Latency) | time between consecutive output tokens | `benchmark_serving.py: itl_ms` |
| Memory occupied | peak `nvidia-smi memory.used` during the run | sample every 5s with `nvidia-smi --query-gpu=memory.used --format=csv,noheader -lms 5000` |

p50 = median, p95 = 95th percentile. Distribution is across all requests in a single bench run.

---

## 5. Pre-flight (LOCAL, FREE — must pass before renting)

Every minute we spend on the rented box at $0.40/hr is a minute not spent locally for free. The pre-flight is the unfair advantage.

### 5.1 Engine availability

- [ ] `cargo build --release -p ferrum-cli --features cuda` builds on the M1 Max (it cross-compiles the Rust glue; CUDA kernels need an NVCC step that fails locally — that's expected, defer to RunPod). Confirm the **non-CUDA-kernel** parts of the cuda feature compile.
- [ ] `cargo install vllm` — N/A; vLLM is Python. Verify `pip install vllm==<pin>` resolves on Linux x86_64 + CUDA 12.x.
- [ ] `cargo install mistralrs-server --version <pin>` — at least lock the version we'll bench so the report is reproducible.

### 5.2 Model availability

- [ ] M1: `huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --revision <commit> --token $HF_TOKEN` (gated — confirm token works)
- [ ] M2: similarly, find a working GPTQ-INT4 pack. Candidates: `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4`. Verify it loads in vLLM and ferrum locally (CPU loader can at least parse).
- [ ] **M3 / M4**: Search HF for `Qwen3-30B-A3B-*-GPTQ-Int4` and `Qwen3-Coder-30B-A3B-*-GPTQ-Int4`. If only AWQ exists, decide: switch to AWQ across the board, or drop the cell.
- [ ] All 4 model IDs frozen in `bench/v0.2-cuda/models.txt` BEFORE renting.

### 5.3 Bench harness sanity (M1 Max)

- [ ] Pull vLLM's `benchmarks/benchmark_serving.py` (their version, not ours from the macOS bench — vLLM's is what they cite, so cite back). Pin it to a vLLM tag.
- [ ] Run it against ferrum locally on `qwen3:0.6b` at c=1 — proves the harness wiring works. Won't tell us anything about CUDA performance but rules out tokenizer / OpenAI-shape mismatches.
- [ ] Source ShareGPT v3 subset; freeze the 128-prompt seed-deterministic slice as `bench/v0.2-cuda/prompts.json`.

### 5.4 Scripts authored locally

`bench/v0.2-cuda/` will hold:
- `setup.sh` — one-shot, run on rented box first (apt deps, Rust, build ferrum, pip install vllm, cargo install mistralrs, download models to network volume).
- `run_ferrum.sh` `run_vllm.sh` `run_mistralrs.sh` — single-engine launchers parameterized by `<MODEL_ID> <port>`.
- `run_cell.sh <engine> <model> <c>` — runs one cell: starts engine if not running, prewarms, calls `benchmark_serving.py`, saves JSON, kills engine.
- `run_sweep.sh` — outer loop over the matrix, calls `run_cell.sh` 144 times. Skips cells whose output JSON already exists (resume safe).
- `pull_results.sh` — runs locally; rsync results back from the pod.

All scripts authored and committed BEFORE the pod boots. Mistakes in shell logic at $0.40/hr are real money.

### 5.5 RunPod template selection

- [ ] Pick a template with PyTorch + CUDA pre-baked (saves ~20 min CUDA install). Suggest `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
- [ ] Network volume: 100 GB at `/workspace`. Models persist across pod restarts so we don't re-download on resume.
- [ ] On-demand instance (cheaper than reserved). RTX 4090 24GB single-GPU. ≥ 64 GB system RAM (some engines are RAM-hungry on prefill).
- [ ] **Pre-rent**: one cheap 5-min boot to verify driver version supports our vLLM pin (CUDA 12.4 + driver ≥ 550 for vLLM 0.7+).

### 5.6 SSH + data exfil

- [ ] SSH key registered on vast / RunPod profile.
- [ ] `pull_results.sh` is `rsync -avz pod:/workspace/ferrum-infer-rs/bench/v0.2-cuda/results/ ./bench/v0.2-cuda/results/`. Test rsync mid-bench (after 10 cells) to confirm exfil works — don't wait until the end.

---

## 6. Sweep execution order (fail-fast)

The naive nested loop `for model: for engine: for c: for repeat:` is 144 cold starts. **We don't do that.**

Better: keep each (engine, model) server up while sweeping the c axis. Cold-start is ~30-90 s, so collapsing 16 c-values per server start saves ~10 min/server × 12 servers = 2 hr.

```
for model in [M2 (Llama-INT4), M1 (Llama-FP16), M3 (Qwen3-30B-INT4), M4 (Qwen3-Coder-30B-INT4)]:
    for engine in [ferrum, vllm, mistralrs]:
        start engine ONCE with --max-seqs=32 (covers all c)
        prewarm with 1 small request
        for workload in [W1 c=1, W2 c=4, W3 c=16, W4 c=32]:
            for repeat in [1, 2, 3]:
                run benchmark_serving.py — saves JSON
        kill engine
```

**Order rationale**:

1. **M2 (Llama-INT4) first** — smallest, lowest blast radius. Uncovers most engine bugs cheaply. If vLLM's `gptq_marlin` flag is wrong, we find out in cell 1 not cell 80.
2. **M1 (Llama-FP16) second** — only at this point are we paying real GPU memory cost. FP16-8B + KV at c=32 is the tightest fit on 24 GB; if anything OOMs it's here.
3. **M3 (Qwen3-30B-INT4) third** — first MoE, biggest weights. mistralrs may PoisonError; document and skip.
4. **M4 (Qwen3-Coder-30B-INT4) last** — same shape as M3, so any infrastructure issue is already debugged.

**Within each (engine, model)**: c=1 first, c=32 last. c=1 surfaces correctness issues (gibberish output → tokenizer mismatch), c=32 surfaces resource issues. Walking up the c ladder makes failures attributable.

**Cell wall-time budget**:

| Phase | Cells | Per-cell | Subtotal |
|---|---|---|---|
| Smoke (engine boot + 1 c=1 cell each) | 12 | 90 s | 18 min |
| Full matrix bench runs | 144 | ~150 s | 360 min = 6.0 hr |
| Inter-engine model load | 12 server starts | 60 s avg | 12 min |
| Total inside the pod | | | **~6.5 hr** |
| Setup + model download | | one-shot | 30-45 min |
| Buffer (OOMs / retries / debug) | | | 4-6 hr |
| **Grand total** | | | **~12 hr — safely under 25 hr / $10** |

Per-cell wall: `prewarm (5s) + bench_serving (variable: 60 s @ c=1 with 512 outputs and ~30 tok/s, 120 s @ c=32 with 256 outputs / 32 prompts and ~150 tok/s aggregate) + cooldown (5s)`. ~150 s average is the working number.

---

## 7. Failure handling

### 7.1 Per-cell timeout

Every `bench_serving.py` call wrapped in `timeout 900 ...` (15 min). If hit, kill the engine, log the cell as `FAILED`, continue. Don't let one stuck request torch the whole budget.

### 7.2 Per-(engine, model) timeout

If a server can't get past the smoke c=1 in 5 min, mark the whole (engine, model) pair as broken and skip its 12 cells. Document in the report.

### 7.3 Resume protocol

`run_sweep.sh` checks for `results/<engine>__<model>__c<n>__r<rep>.json` before each cell. If exists with `output_throughput_tok_s > 0`, skip. So a pod restart loses ≤ 1 cell.

### 7.4 Disk space

`/workspace` should have ~100 GB. Models eat ~55 GB. Per-run JSON + log = a few hundred KB. Should never fill, but log `df -h /workspace` every 10 cells defensively.

### 7.5 vLLM-specific risks

- **Marlin requires Ampere+** — RTX 4090 is Ada (sm_89), should be fine. Verify in pre-flight.
- **Long prefill OOMs**: at c=32 + prompt 512, prefill batch is up to 16 384 tokens. vLLM may need `--max-num-batched-tokens` tuned. Plan: start with default, lower if OOM.
- **Rope scaling on Llama-3.1**: vLLM auto-detects from config; verify it doesn't warn.

### 7.6 mistralrs-specific risks

- Known: PoisonError on Qwen3-30B-A3B Q4_K_M GGUF (per memory). Try the GPTQ safetensors path which goes through a different loader.
- If still panics: report as `panic` in the cell and move on. We did the same in macOS bench.

### 7.7 ferrum-specific risks

- Triton w4a16 + ContinuousBatch scheduler hits NaN on prefill (per CLAUDE.md). For this bench, use **Marlin** path explicitly (`FERRUM_TRITON_INT4=0`, the default) and document.
- 30B-A3B + paged-KV at c=32 might exceed 24 GB. Plan: keep `KV_CAPACITY=512` default, watch GPU memory, drop max_seqs if OOM.

---

## 8. Data collection

### 8.1 Per-cell artifacts (saved to `/workspace/.../results/`)

```
<engine>__<model>__c<n>__r<rep>.json    # benchmark_serving.py output
<engine>__<model>__c<n>__r<rep>.bench.log  # harness stdout
<engine>__<model>__c<n>__r<rep>.server.log # engine stdout (last 500 lines)
<engine>__<model>__c<n>__r<rep>.gpu.csv  # nvidia-smi mem trace
```

### 8.2 Suite-level fingerprint (saved once)

`_suite_env.txt`:
- `nvidia-smi -q` (driver, GPU, memory, ECC state)
- `nvcc --version`
- `cat /proc/cpuinfo | head` (for completeness)
- `pip freeze` (vLLM + deps)
- `cargo --version`, `git rev-parse HEAD` for ferrum + mistralrs
- HF model commit hashes (resolved from snapshot dir)

### 8.3 Final report structure

`docs/bench/cuda-rtx4090-2026-05-XX/README.md`, mirroring the macOS report:
1. Headline tables (4 models × 3 engines, output throughput at each c)
2. TTFT p50/p95 at c=16 (latency-sensitive)
3. Memory at c=32 (capacity story)
4. Hardware + software fingerprint
5. Methodology (ShareGPT subset, harness pin, prompt-length normalization)
6. How to reproduce (the same scripts, with model IDs filled in)
7. Caveats (mistralrs MoE failures, vLLM Marlin sm_89 confirmation, ferrum CUDA Graph capture state at run time)

---

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Qwen3-Coder-30B-A3B GPTQ-Int4 doesn't exist on HF | Medium | Pre-flight § 5.2; if missing, drop M4 or AWQ-quant on-pod (60 min one-time) |
| RTX 4090 OOM on FP16-8B + c=32 | Low (math says 19 GB worst case, fits) | Pre-compute KV math; if hits, drop c=32 for M1 only |
| vLLM Marlin path not usable on Ada (sm_89) | Low | Verify in pre-flight; fallback `--quantization gptq` (Marlin's predecessor) |
| mistralrs panics on M3+M4 (MoE) | High (per macOS history) | Document, skip those cells; M3+M4 mistralrs columns will show `panic` |
| Pod gets pre-empted mid-bench (RunPod can do this on spot pricing) | Low if on-demand | Always on-demand pricing; resume protocol § 7.3 covers it |
| ShareGPT licensing | Medium | Use the publicly-available unfiltered snapshot; cite source. If concerned, swap for vLLM's synthetic prompts |
| Wall-clock blows past 25 hr | Low (estimate is 12 hr) | If we burn 15 hr and only 50% done, drop the 3 repetitions to 1 — single-run instead of median |
| GPU is cold the first 2-3 min and skews c=1 numbers | Low | First cell of each (engine, model) is a smoke run that's discarded |

---

## 10. Out of scope (explicit, per user)

- H100 / A100 / multi-GPU (4090 single-card is the price-conscious story)
- Tensor / pipeline parallelism (single 4090 can't fit a model that needs it for this matrix)
- KV cache quantization (FP16 KV across all engines, fair comparison only)
- Perplexity / accuracy validation (correctness is verified by spot-checking a few outputs in pre-flight; quality benchmarks are a separate concern)
- Model architectures outside Llama / Qwen3 dense / Qwen3-MoE
- Qwen3.6 (vLLM doesn't support it on RTX 4090 per user)
- Custom datasets — ShareGPT only

---

## 11. Open questions (resolve in pre-flight)

1. **Which exact GPTQ-Int4 of Qwen3-30B-A3B / Qwen3-Coder-30B-A3B do we use?** Answer required before renting.
2. **Pin vLLM to which version?** Latest stable that still supports Llama-3.1 GPTQ-Marlin on Ada (sm_89). Likely 0.7.x.
3. **Pin mistralrs to which version?** Latest stable.
4. **ShareGPT exact subset?** Seed = ferrum's repo rev hash (deterministic), 128 prompts, prompt-length filter ≥ 128 + ≤ 512 tokens.
5. **Do we test with sampling temperature 0 only, or also 0.7?** Plan: 0.0 (deterministic, removes RNG noise). Anyone interested in sampling-cost can re-run with the same scripts later.
6. **Do we report cost-per-million-tokens?** Yes — derived metric: `$/hr ÷ aggregate tok/s × 3600 × 1e6`. Adds a "ferrum is cheaper per token" angle.

---

## 12. Execution checklist (sign off before renting)

- [ ] All four model HF repos exist and have GPTQ-Int4 if needed. **OWNER: pre-flight day**
- [ ] All scripts in `bench/v0.2-cuda/` committed and reviewed
- [ ] `prompts.json` (128 deterministic prompts) committed
- [ ] vLLM + mistralrs versions pinned, RunPod image template chosen
- [ ] `setup.sh` smoke-tested: at minimum, `bash -n setup.sh` lints; ideally one cheap 30-min pod boot to validate the install path
- [ ] `pull_results.sh` rsync target ready locally
- [ ] HF_TOKEN exported (Llama-3.1 is gated)
- [ ] CARGO_REGISTRY_TOKEN NOT exported on the pod (we don't publish from there)
- [ ] $10 of credit on the GPU vendor account
- [ ] Local M1 Max ferrum smoke run on the harness passes

---

## 13. Reference

- macOS Group A bench: [`docs/bench/macos-2026-05-02/`](../macos-2026-05-02/) — same rigor template we're matching
- vLLM benchmark harness: https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py
- ShareGPT v3 unfiltered: typical mirror at `lmsys/lmsys-chat-1m` or `anon8231489123/ShareGPT_Vicuna_unfiltered`
- ferrum CUDA decode runner: `crates/ferrum-kernels/src/cuda/decode_runner.rs`
- ferrum Marlin INT4: `crates/ferrum-kernels/src/cuda/marlin.rs`
