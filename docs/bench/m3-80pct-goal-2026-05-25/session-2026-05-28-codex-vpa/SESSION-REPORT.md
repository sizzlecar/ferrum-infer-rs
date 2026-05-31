# M3 80% goal — session 2026-05-28 Codex VPA + MoE loop

**Date:** 2026-05-28  
**Pod:** Vast instance `38237968`, RTX 4090 sm_89, driver 580.95.05, CUDA 13.0  
**Base commit:** `a9dc705`  
**Branch state:** local working tree; no commit created  

---

## TL;DR

This session produced one correctness fix with a modest performance gain:

| Change | Gate | c=32 result |
|---|---|---:|
| VPA bridge for `FERRUM_USE_VLLM_PAGED_ATTN=1` | Paris ✓ | `1127.9 ± 101.4 tok/s` |
| Same binary, VPA off | — | `1058.9 ± 9.2 tok/s` |
| Delta | N=3 same-pod | `+6.5%` |

Full VPA sweep on the same pod:

| c | no-VPA tok/s | VPA tok/s | ratio |
|---:|---:|---:|---:|
| 1 | `145.6 ± 1.3` | `150.0 ± 1.1` | `1.031` |
| 4 | `147.1 ± 1.2` | `151.3 ± 2.5` | `1.028` |
| 16 | `855.8 ± 18.2` | `884.9 ± 42.3` | `1.034` |
| 32 | `1058.9 ± 9.2` | `1127.9 ± 101.4` | `1.065` |

VPA is useful, but it does not close the M3 0.80× vLLM gap.

---

## Loop 1 — fix VPA correctness

### Bottleneck / hypothesis

The previous session found `FERRUM_USE_VLLM_PAGED_ATTN=1` emitted garbage on the
Paris smoke. vLLM source showed decode should use `paged_attention_v2`, while
prefill/chunk-prefill needs a varlen bridge over vLLM KV layout.

### Patch

- Added `paged_varlen_attention_vllm.cu` as a correctness bridge for q_len > 1.
- Wired `Backend::paged_varlen_attention_vllm_layout`.
- Changed Qwen3-MoE single-seq paged attention routing:
  - decode token: `paged_decode_attention_v2`
  - q_len > 1: new VPA varlen bridge
  - no VPA: legacy paged path

### Correctness

Command class:

```bash
HF_HOME=/workspace/hf-cache \
FERRUM_BACKEND=cuda \
FERRUM_MOE_DEVICE_ROUTE=1 \
FERRUM_MOE_STREAMS=4 \
FERRUM_GREEDY_ARGMAX=1 \
FERRUM_KV_MAX_BLOCKS=2048 \
FERRUM_PAGED_MAX_SEQS=32 \
FERRUM_VLLM_MOE=1 \
FERRUM_MOE_GRAPH=0 \
FERRUM_USE_VLLM_PAGED_ATTN=1 \
./target/release/ferrum run Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
  --backend cuda \
  --prompt "What is the capital of France?" \
  --output-format jsonl \
  --max-tokens 64 \
  --temperature 0.0
```

Output contained: `The capital of France is **Paris**.`

### Performance artifacts

| Artifact | Meaning |
|---|---|
| `/workspace/m3-graph-loop/sweep_baseline_new_n3.json` | no-VPA full sweep |
| `/workspace/m3-graph-loop/sweep_vpa_n3.json` | VPA full sweep |
| `/workspace/m3-graph-loop/c32_baseline_new_n3.json` | no-VPA c=32 focused N=3 |
| `/workspace/m3-graph-loop/c32_vpa_n3.json` | VPA c=32 focused N=3 |

---

## Loop 2 — post-VPA bottleneck localization

Profile server used `FERRUM_DECODE_OP_PROFILE=1` and `FERRUM_RBD_PROF=1`.
`FERRUM_MOE_GRAPH=0` was required because stage timers sync inside graph capture
and can trigger `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.

Active c=32 m=32 profile:

```text
[batched-decode-prof] m=32 layers=48 total=19 ms
  dense=2 ms (~14%)
  attn_peritem=2 ms (~11%)
  moe=13 ms (~70%)
  other=1 ms (~6%)

[bucket-prof] layers=48 bk_total=11 ms
  gather=~0.39 ms
  gemm1=~7.3 ms
  silu=~0.29 ms
  gemm3=~3.6 ms
  combine=~0.29 ms
```

Conclusion: attention improved after VPA; MoE remains the main bottleneck, and
within MoE the vLLM-Marlin gate_up/down GEMMs dominate.

Artifact: `/workspace/m3-graph-loop/profile_vpa_c32_g0/`.

---

## Loop 3 — small-m fused MoE validation

Ran the existing Triton fused-MoE microbench:

```bash
CUDA_COMPUTE_CAP=89 /root/.cargo/bin/cargo test -p ferrum-kernels \
  --features cuda,triton-kernels \
  --release \
  --test triton_fused_moe_bench \
  -- --ignored --nocapture
```

Result:

```text
Triton fused MoE: 453.0 µs/iter
```

This covers gate_up only. The profiled vLLM-Marlin gate_up path is about
`7.3 ms / 48 = 150 µs/layer`, so the existing Triton PTX is not a viable
drop-in replacement for this M3 shape.

Artifact: `/workspace/m3-graph-loop/triton_fused_moe_bench.log`.

---

## Loop 4 — skip redundant vLLM workspace zeroing

### Hypothesis

Ferrum bulk-zeroed the vLLM Marlin workspace before both gate_up and down in
each MoE layer. vLLM initializes the workspace with `torch.zeros` at allocation
time and the marlin_moe reduce path resets lock slots internally. Therefore the
per-layer bulk-zero is redundant on the vLLM MoE path.

### Patch

`moe_forward_bucketed` now skips `zero_workspace()` when `FERRUM_VLLM_MOE=1`.
`FERRUM_VLLM_MOE_ZERO_WS=1` forces the old behavior for A/B or rollback.

### Correctness

Paris smoke passed with default skip-zero and VPA:

```text
The capital of France is **Paris**.
```

### Performance

Same binary c=32 N=3, random 256/128:

| Mode | tok/s | TPOT p50 |
|---|---:|---:|
| default skip zero | `1153.1 ± 152.7` | `20.57 ms` |
| forced old zero | `1130.4 ± 168.7` | `20.67 ms` |

Conclusion: correct and slightly positive, but the delta is below 10% with
overlapping intervals. Treat this as a small cleanup, not the primary M3 lever.

Artifacts:

| Artifact | Meaning |
|---|---|
| `/workspace/m3-graph-loop/paris_skip_ws_zero.log` | correctness smoke |
| `/workspace/m3-graph-loop/skip_ws_zero_ab/skip/c32_n3.json` | default skip-zero |
| `/workspace/m3-graph-loop/skip_ws_zero_ab/zero/c32_n3.json` | forced old zeroing |

---

## Next target

The next high-return work still has to reduce MoE GEMM cost or remove MoE work
from the critical path. The existing Triton PTX is too slow, and non-GEMM MoE
overheads are small. Do not spend another session on broad env flips; use vLLM
source first and target a concrete MoE kernel/Marlin path change.
