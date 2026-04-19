#!/usr/bin/env bash
# Phase E one-shot verification — self-bootstrapping GPU runner.
#
# Designed for a freshly rented Linux CUDA box. Idempotent — rerun is safe.
#
# Usage:
#   git clone https://github.com/sizzlecar/ferrum-infer-rs.git -b feat/cuda-tts
#   cd ferrum-infer-rs
#   bash scripts/phase-e-verify.sh                # defaults to qwen3:0.6b
#   bash scripts/phase-e-verify.sh qwen3:4b       # larger model
#
# Prereqs on host:
#   - NVIDIA GPU (SM >= 7.0; SM >= 8.0 for Marlin GPTQ section)
#   - NVIDIA driver (`nvidia-smi` works)
#   - CUDA toolkit 12.x installed (nvcc + cuda.h). If missing, see:
#       https://developer.nvidia.com/cuda-downloads
#     Or run the rented-box's built-in installer (most cloud GPUs pre-install).
#   - ~10 GB free disk (build artifacts + weights)
#
# First run ≈ 30-45 min (mostly cargo build + model download).
# Subsequent runs ≈ 5-10 min (cargo incremental).
#
# Output:
#   /tmp/phase-e-report.md   — human-readable report (scp back to local)
#   /tmp/phase-e-verify.log  — full command log

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

MODEL="${1:-qwen3:0.6b}"
HF_MODEL_ID="Qwen/Qwen3-0.6B"  # adjust if MODEL != qwen3:0.6b
REPORT="/tmp/phase-e-report.md"
LOG="/tmp/phase-e-verify.log"

: > "${REPORT}"
: > "${LOG}"

# ── Logging helpers ─────────────────────────────────────────────────────
section()   { printf '\n===== %s =====\n' "$*"       | tee -a "${REPORT}" "${LOG}"; }
record()    { printf '%s\n'  "$*"                    | tee -a "${REPORT}" "${LOG}"; }
run_logged(){ printf '\n$ %s\n' "$*"                 >> "${LOG}"; eval "$*" 2>&1 | tee -a "${LOG}"; }
fail()      { record "❌ $*"; echo "Full log: ${LOG}"; exit 1; }

record "# Phase E Verification Report"
record ""
record "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
record "Host:      $(uname -a)"
record "Model:     ${MODEL}  (${HF_MODEL_ID})"
record "Commit:    $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
record ""

# ────────────────────────────────────────────────────────────────────────
# 0. GPU driver + CUDA toolkit
# ────────────────────────────────────────────────────────────────────────
section "0. GPU driver + CUDA toolkit"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found — no NVIDIA driver on this host"
fi
record '```'
run_logged "nvidia-smi" | head -20
record '```'
record ""

# Auto-detect CUDA_HOME if not set.
if [ -z "${CUDA_HOME:-}" ]; then
    for candidate in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12.2 /opt/cuda; do
        if [ -f "$candidate/include/cuda.h" ]; then
            export CUDA_HOME="$candidate"
            break
        fi
    done
fi
if [ -z "${CUDA_HOME:-}" ] || [ ! -f "$CUDA_HOME/include/cuda.h" ]; then
    fail "CUDA_HOME not set and cuda.h not found in standard locations. Install cuda-toolkit-12 or set CUDA_HOME manually."
fi
record "CUDA_HOME = ${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
run_logged "nvcc --version | tail -3"
record ""

# Compute capability — Marlin (GPTQ) needs SM >= 8.0.
#
# Clamp to the highest arch nvcc actually supports (newer GPUs like Blackwell
# SM 12.0 need nvcc 12.8+; on older nvcc the PTX JIT handles forward-compat
# at runtime). Don't just propagate raw `nvidia-smi` output — nvcc 12.6 will
# reject `compute_cap=120` with "nvcc cannot target gpu arch 120".
CC_RAW="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')"
NVCC_MAX_CAP="$(nvcc --list-gpu-arch 2>/dev/null | grep -oE 'compute_[0-9]+' | grep -oE '[0-9]+$' | sort -n | tail -1)"
: "${NVCC_MAX_CAP:=90}"
if [ -n "$CC_RAW" ] && [ "$CC_RAW" -le "$NVCC_MAX_CAP" ]; then
    export CUDA_COMPUTE_CAP="$CC_RAW"
else
    export CUDA_COMPUTE_CAP="$NVCC_MAX_CAP"
fi
record "GPU compute_cap = ${CC_RAW:-unknown}"
record "nvcc max cap    = ${NVCC_MAX_CAP}"
record "CUDA_COMPUTE_CAP = ${CUDA_COMPUTE_CAP} (PTX JIT bridges any gap on newer GPUs)"

# ────────────────────────────────────────────────────────────────────────
# 1. System deps
# ────────────────────────────────────────────────────────────────────────
section "1. System deps (build-essential, libssl, ffmpeg, python3)"
if command -v apt-get >/dev/null 2>&1; then
    record 'Installing / verifying via apt-get...'
    # libnccl-dev: cudarc's `nccl` feature links -lnccl at build time.
    # Without this apt step, the release link step fails even though CUDA
    # compiles cleanly.
    run_logged "sudo apt-get update -qq && sudo apt-get install -y --no-install-recommends \
        build-essential pkg-config libssl-dev git curl \
        python3 python3-pip ffmpeg libsndfile1 \
        libnccl2 libnccl-dev 2>&1 | tail -3" || \
        record 'apt-get hit an issue — retry manually if next steps fail'
else
    record 'Non-Debian host — ensure build-essential, libssl-dev, pkg-config, python3 are installed manually.'
fi

# ────────────────────────────────────────────────────────────────────────
# 2. Rust toolchain
# ────────────────────────────────────────────────────────────────────────
section "2. Rust toolchain"
if ! command -v cargo >/dev/null 2>&1; then
    record 'Installing rustup (stable)...'
    run_logged "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal" \
        || fail 'rustup install failed'
    # shellcheck source=/dev/null
    . "$HOME/.cargo/env"
fi
export PATH="$HOME/.cargo/bin:${PATH}"
run_logged "rustc --version"
run_logged "cargo --version"
record ""

# ────────────────────────────────────────────────────────────────────────
# 3. Python deps (for HF download)
# ────────────────────────────────────────────────────────────────────────
section "3. Python deps"
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    run_logged "pip3 install --quiet --no-cache-dir huggingface-hub numpy safetensors" \
        || record 'pip3 install flaky — continuing; download step will retry'
fi
record ""

# ────────────────────────────────────────────────────────────────────────
# 4. Download model weights
# ────────────────────────────────────────────────────────────────────────
section "4. Weights: ${HF_MODEL_ID}"
HF_CACHE_BASE="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_CACHE="${HF_CACHE_BASE}/hub/models--${HF_MODEL_ID//\//--}"
if [ -d "$HF_CACHE/snapshots" ] && find "$HF_CACHE/snapshots" -name "*.safetensors" -print -quit | grep -q .; then
    record "✅ already cached: ${HF_CACHE}"
else
    record "Downloading from HuggingFace..."
    run_logged "python3 -c \"
from huggingface_hub import snapshot_download
import sys
path = snapshot_download('${HF_MODEL_ID}', allow_patterns=['*.safetensors', '*.json', 'tokenizer*'])
print('downloaded to:', path, file=sys.stderr)
\"" || fail 'HF download failed — check network / HF_HUB_OFFLINE / token'
fi
record ""

# ────────────────────────────────────────────────────────────────────────
# 5. cargo check --features cuda
# ────────────────────────────────────────────────────────────────────────
section "5. cargo check --features cuda"
# Target ferrum-cli specifically, not the whole workspace — `--workspace
# --features cuda` activates every workspace member's cuda feature, which
# includes ferrum-attention's dead-stub cuda feature (pinned on cudarc 0.12,
# max CUDA 12.6). Scoping to ferrum-cli pulls only the live CUDA deps.
record '```'
if run_logged "cargo check --features cuda -p ferrum-cli 2>&1 | tail -10"; then
    record "✅ type check passed"
else
    record "❌ type check failed"
    fail 'type check'
fi
record '```'
record ""

# ────────────────────────────────────────────────────────────────────────
# 6. cargo build --release --features cuda
# ────────────────────────────────────────────────────────────────────────
section "6. cargo build --release --features cuda"
BUILD_START=$(date +%s)
record '```'
if run_logged "cargo build --release --features cuda -p ferrum-cli --bin ferrum 2>&1 | tail -5"; then
    record "Build time: $(( $(date +%s) - BUILD_START )) s"
    record "✅ release build OK"
else
    record "❌ release build failed"
    fail 'build'
fi
record '```'
record ""

FERRUM_BIN="${ROOT}/target/release/ferrum"
if [ ! -x "${FERRUM_BIN}" ]; then
    fail "${FERRUM_BIN} missing after build"
fi

# ────────────────────────────────────────────────────────────────────────
# 7. Correctness: Cpu ↔ Cuda parity test
# ────────────────────────────────────────────────────────────────────────
section "7. Parity — CpuBackend vs CudaBackend on Qwen3-0.6B"
record ''
record 'Prefill (8 tokens) + 5 decode steps. Asserts argmax equality and'
record 'cosine similarity ≥ 0.999 every step. First failing step is printed.'
record ''
record '```'
PARITY_OK=0
if run_logged "cargo test -p ferrum-models --features cuda --release --test qwen3_cuda_parity_test -- --ignored --nocapture 2>&1 | tail -25"; then
    record "✅ parity passed"
    PARITY_OK=1
else
    record "❌ parity FAILED — see ${LOG}"
fi
record '```'
record ''

# ────────────────────────────────────────────────────────────────────────
# 8. Perf: decode bench (short / medium / long)
# ────────────────────────────────────────────────────────────────────────
section "8. Decode bench (short / medium / long)"
for max_tok in 64 256 1024; do
    record ""
    record "### ${max_tok} tokens"
    record '```'
    run_logged "RUST_LOG=warn ${FERRUM_BIN} bench ${MODEL} --backend cuda --max-tokens ${max_tok} 2>&1 | tail -14" \
        || record '(bench failed — see log)'
    record '```'
done

# ────────────────────────────────────────────────────────────────────────
# 9. Perf: long prefill (~2k prompt)
# ────────────────────────────────────────────────────────────────────────
section "9. Long prefill (~2k prompt, 256 decode)"
record '```'
run_logged "RUST_LOG=warn ${FERRUM_BIN} bench ${MODEL} --backend cuda --long-context 2>&1 | tail -14" \
    || record '(bench failed — see log)'
record '```'
record ""

# ────────────────────────────────────────────────────────────────────────
# 10. Perf: concurrent batch decode
# ────────────────────────────────────────────────────────────────────────
section "10. Concurrent batch decode (4 streams × 64 tokens)"
record '```'
run_logged "RUST_LOG=warn FERRUM_MAX_BATCH=4 ${FERRUM_BIN} bench ${MODEL} --backend cuda --concurrency 4 --max-tokens 64 2>&1 | tail -14" \
    || record '(bench failed — see log)'
record '```'
record ""

# ────────────────────────────────────────────────────────────────────────
# 11. GPTQ smoke — expect clean "unsupported" error (Phase E-GPTQ followup)
# ────────────────────────────────────────────────────────────────────────
GPTQ_DIR="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct-GPTQ-Int4"
if [ -d "${GPTQ_DIR}" ]; then
    section "11. GPTQ smoke — Marlin path (expected clean error)"
    record ''
    record 'Phase E-GPTQ is not yet wired. CudaBackend::gemm_quant returns'
    record '"unsupported" for all three QuantKind variants. This step verifies'
    record 'the error surfaces cleanly (not a segfault / silent wrong output).'
    record ''
    record '```'
    run_logged "RUST_LOG=warn ${FERRUM_BIN} bench qwen2.5-3b-instruct-gptq-int4 --backend cuda --max-tokens 32 2>&1 | tail -14 || true"
    record '```'
else
    section "11. GPTQ smoke — SKIPPED (no GPTQ weights cached)"
    record 'To enable: run `ferrum pull qwen2.5-3b-instruct-gptq-int4` first.'
fi

# ────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────
section "Summary"
record ""
if [ "${PARITY_OK}" = "1" ]; then
    record "- Correctness: ✅ Cpu↔Cuda parity matches on Qwen3-0.6B"
else
    record "- Correctness: ❌ parity FAILED — block Phase E landing until fixed"
    record "  Look at first divergent layer in the parity test output above."
    record "  Likely suspect kernels: qk_norm_rope, kv_cache_append_head_major,"
    record "  flash_attn_full_f16 (prefill), decode_attention_f16 (decode step)."
fi
record ""
record "- Report:   ${REPORT}"
record "- Full log: ${LOG}"
record ""
record "## What was tested"
record ""
record "| Area              | How                                                         |"
record "|-------------------|-------------------------------------------------------------|"
record "| CUDA type check   | cargo check --workspace --features cuda                     |"
record "| PTX compilation   | cargo build --release --features cuda (bindgen_cuda path)   |"
record "| Correctness       | Cpu↔Cuda logit parity, prefill + 5 decode steps             |"
record "| Decode perf       | bench at 64 / 256 / 1024 output tokens                      |"
record "| Prefill perf      | bench --long-context (~2k prompt)                           |"
record "| Batch decode      | bench --concurrency 4                                       |"
record "| GPTQ error path   | Marlin bench (expected clean unsupported)                   |"
record ""
record "## Kernels exercised (indirectly, via Qwen3 LLM forward)"
record ""
record "- embedding_lookup_f16, rms_norm_f16, fused_add_rms_norm_inplace_f16"
record "- cuBLAS hgemm (Linear.forward), split_qkv_f16, qk_norm_rope_transpose_f16"
record "- kv_cache_append_head_major_f16, flash_attn_full_f16 (prefill)"
record "- decode_attention_f16 (decode), transpose_head_to_token_f16"
record "- fused_silu_mul_interleaved_f16, residual_add_inplace_f16"
record ""
record "## Kernels NOT exercised (future phases)"
record ""
record "- layer_norm_f16, gelu_f16, add_bias_f16 (wired but unused — Bert / Clip path)"
record "- marlin_cuda (GPTQ, Phase E-GPTQ)"
record "- NCCL all_reduce / all_gather / broadcast (Phase E-TP)"

echo
echo "✅ Report: ${REPORT}"
echo "   Log:    ${LOG}"
echo
echo "Copy the report back for review:"
echo "   scp <user>@<host>:${REPORT} ./"
