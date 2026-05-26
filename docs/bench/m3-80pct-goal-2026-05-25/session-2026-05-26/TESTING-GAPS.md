# Testing gaps exposed in 2026-05-26 session

Honest inventory of what the current test infrastructure does NOT
catch. Maintained so subsequent sessions don't re-discover the same
bugs by accident.

## What the existing tests DO cover

Per `CLAUDE.md` § "Done criteria":

| suite | gating | what it tests |
|---|---|---|
| `cargo test -p ferrum-cli --features metal --test chat_smoke` | metal | single-turn REPL output is non-empty + non-NaN |
| `cargo test -p ferrum-cli --features metal --test chat_pty` | metal | **multi-turn** PTY chat — N turns, no crash |
| `cargo test -p ferrum-cli --features metal --test chat_stress` | metal | concurrent / pressure on chat state |
| `cargo test -p ferrum-cli --features metal --test server_*` | metal | HTTP server compliance + multi-turn via /v1/chat/completions |
| `cargo test -p ferrum-cli --features metal --test reference_match` | metal | byte-equal output vs fixture |

CI runs these on macos-latest nightly. PR-time CI is `cargo check`
only (no GPU runner).

## What we added this session (for the M3 sweep)

| added | covers | doesn't cover |
|---|---|---|
| `sweep_bottleneck.sh` per-cell `vllm_baseline.json` | apples vLLM ratio | correctness |
| `scripts/safe_sweep.sh` `"Paris"` gate per cell | **single-prompt** sanity at cN | multi-turn |
| `scripts/compare_nsys_kernels.py` | kernel attribution | runtime semantics |
| `scripts/microbenches/*.cu` | GPU primitive perf | model semantics |
| `pod_correctness_check.sh` (3-prompt smoke) | single-shot QA | multi-turn KV state |

→ **all single-prompt**. None exercise the path that hits the
multi-turn `paged_varlen_attn` crash documented below.

## Known CUDA-specific gaps (no test coverage today)

### 1. Multi-turn KV-state on CUDA path

**Repro** (manually observed in `ferrum run qwen3-30b-a3b-int4`
build 241dbc0 on Vast 4090, CUDA 13.0.48):
- Turn 1: "你好" → "你好！很高兴见到你..." ✓
- Turn 2: "你谁?" → "你好！我是通义千问..." ✓
- Turn 3: "你会什么" → **panic**:
```
thread 'tokio-runtime-worker' panicked at
crates/ferrum-models/src/models/qwen3_moe_forward_unified.rs:368:14:
Qwen3Moe unified: paged_varlen_attention: Model { message:
"paged_varlen_attn: DriverError(CUDA_ERROR_INVALID_VALUE,
 \"invalid argument\")" }
```

**Why no test catches this**:
- `chat_pty` / `chat_stress` are Metal-only — different attn path
  (`paged_decode_attention` Metal kernel, not `paged_varlen_attn`)
- Single-prompt correctness gates don't probe cumulative KV state
- CUDA CI is `cargo check` only (no real inference)

**Where to add coverage**: `cargo test -p ferrum-cli --features cuda
--test chat_pty` (gated behind a `CUDA_TESTS=1` env so it runs only
when a GPU is wired). Run nightly via a self-hosted GPU runner OR
manually as part of every CUDA-touching PR.

### 2. Long-context / KV growth boundary

`paged_varlen_attn` uses `safe_kv_max = max(FERRUM_KV_CAPACITY=512,
max_kv_len)` for shared mem allocation. When `max_kv_len > 24576` (=
96KB / 4 bytes shared budget on sm_89), kernel launch fails with
INVALID_VALUE. Not tested anywhere.

### 3. Mixed batch (prefill + decode) across non-pure-decode paths

Per `qwen3_moe.rs:3738` `unified_forward` returns Unsupported for
pure-decode, but mixed batches at turn N>1 in the chat path DO go
through unified. Coverage of mixed-batch paths on CUDA is zero in
the test suite.

### 4. CUDA 13 vs CUDA 12 ABI / template mangling

The vllm-moe-marlin garbage-output regression (b0da10d → 241dbc0
session-2026-05-26 fix) was a CUDA-13-specific template-instantiation
bug. CI runs metal only — would never have caught this. The
`reference_match` byte-equal test runs only on Metal too.

## Testing gaps that ARE NOT my excuse

This session's test infrastructure (sweep correctness gate, microbenches,
nsys comparison) **did its job**: caught that the entire prior session's
ratio numbers were measuring garbage output. Without those gates I'd
still be quoting "0.81 at c=1" instead of the real 0.585.

The structural gap is in the existing test suite, not in what we added.

## Concrete proposals for next session

| proposal | cost | value |
|---|---|---|
| Mirror `chat_pty` to `--features cuda`; gate by `CUDA_TESTS=1` env | 1 day | catches multi-turn KV bugs on CUDA |
| Add a CI nightly job that rents a GPU and runs `cuda chat_pty` | medium (need org GPU runner OR rent each night) | continuous coverage |
| For each CUDA-touching PR, mandate `bash scripts/safe_sweep.sh + scripts/cuda_chat_pty_smoke.sh` on a pod before merge | 1 hour/PR | local human-driven coverage |
| Per-kernel unit tests (paged_varlen_attn with varied (q_len, kv_len, num_blocks)) | 2-3 days | catches shape-boundary bugs |
