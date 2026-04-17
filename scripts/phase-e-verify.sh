#!/usr/bin/env bash
# Phase E one-shot verification — run on a rented CUDA machine.
#
# Goal: minimise GPU-clock time by batching every check into one script.
# Expected wall time end-to-end: 15-25 min on an A100 / RTX 4090, assuming
# Qwen3-0.6B weights are cached. First run adds ~3 min for weight download.
#
# Usage (on GPU host):
#   ./scripts/phase-e-verify.sh [model-id]
#
# Output: /tmp/phase-e-report.md  (collects every result in one file; scp it
# back to local for review).

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

MODEL="${1:-qwen3:0.6b}"
REPORT="/tmp/phase-e-report.md"
LOG="/tmp/phase-e-verify.log"

# ── Utility ─────────────────────────────────────────────────────────────
section()   { printf '\n===== %s =====\n' "$*"       | tee -a "${REPORT}" "${LOG}"; }
record()    { printf '%s\n'  "$*"                    | tee -a "${REPORT}" "${LOG}"; }
run_logged(){ printf '\n$ %s\n' "$*"                 >> "${LOG}"; eval "$*" 2>&1 | tee -a "${LOG}"; }

: > "${REPORT}"
: > "${LOG}"

record "# Phase E Verification Report"
record ""
record "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
record "Host:      $(uname -a)"
record "Model:     ${MODEL}"
record ""

# ── 1. Environment sanity ────────────────────────────────────────────────
section "0. Environment"
record '```'
run_logged "nvidia-smi || echo 'nvidia-smi unavailable'" | tail -20
record '```'
record ""
record '```'
run_logged "rustc --version"
run_logged "cargo --version"
record '```'
record ""

# ── 2. Compile check first (cheap; fails fast before build) ──────────────
section "1. cargo check --features cuda"
if run_logged "cargo check --workspace --features cuda 2>&1 | tail -15"; then
    record ""
    record "✅ cargo check --features cuda passed"
else
    record ""
    record "❌ cargo check failed — abort"
    exit 1
fi
record ""

# ── 3. Build release ─────────────────────────────────────────────────────
section "2. cargo build --release --features cuda"
BUILD_START=$(date +%s)
if run_logged "cargo build --release --features cuda -p ferrum-cli --bin ferrum 2>&1 | tail -5"; then
    BUILD_END=$(date +%s)
    record ""
    record "Build time: $((BUILD_END - BUILD_START)) s"
    record "✅ release build OK"
else
    record ""
    record "❌ release build failed — abort"
    exit 1
fi
record ""

# ── 4. Parity test: CudaBackend vs CpuBackend on LlamaFamilyModel ───────
section "3. Parity (CudaBackend ↔ CpuBackend, same weights)"
record ""
record 'Runs `qwen3_cuda_parity_test::qwen3model_cpu_vs_cuda` on Qwen3-0.6B.'
record 'Fails loudly on the first argmax/cosine mismatch (per-step diff printed).'
record ""
record '```'
if run_logged "cargo test -p ferrum-models --features cuda --release --test qwen3_cuda_parity_test -- --ignored --nocapture 2>&1 | tail -20"; then
    record ""
    record "✅ parity passed"
else
    record ""
    record "❌ parity FAILED — see ${LOG}"
    # Don't abort; still collect bench below so one GPU session captures max data.
fi
record '```'
record ""

# ── 5. Bench decode ──────────────────────────────────────────────────────
section "4. Bench decode"
for max_tok in 64 128; do
    record ""
    record "### ${max_tok} tokens"
    record '```'
    run_logged "RUST_LOG=warn target/release/ferrum bench ${MODEL} --backend cuda --max-tokens ${max_tok} 2>&1 | tail -14" || true
    record '```'
done

# ── 6. Bench concurrent ──────────────────────────────────────────────────
section "5. Bench concurrent (4)"
record '```'
run_logged "RUST_LOG=warn target/release/ferrum bench ${MODEL} --backend cuda --concurrency 4 --max-tokens 64 2>&1 | tail -14" || true
record '```'
record ""

# ── 7. GPTQ smoke (if GPTQ weights available) ────────────────────────────
GPTQ_DIR="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct-GPTQ-Int4"
if [ -d "${GPTQ_DIR}" ]; then
    section "6. GPTQ smoke (Qwen2.5-3B-Instruct-GPTQ-Int4)"
    record ""
    record 'NOTE: This will fail until Phase E GPTQ kernel work is done.'
    record 'The failure mode should be a clean "gemm_quant not implemented" error,'
    record 'not a crash or wrong output.'
    record ""
    record '```'
    run_logged "RUST_LOG=warn target/release/ferrum bench qwen2.5-3b-instruct-gptq-int4 --backend cuda --max-tokens 32 2>&1 | tail -14 || true"
    record '```'
fi

# ── 8. Summary ───────────────────────────────────────────────────────────
section "Summary"
record ""
record '- **cargo check --features cuda**:  see section 1'
record '- **Release build**:                 see section 2'
record '- **Parity (Cpu vs Cuda)**:          see section 3'
record '- **Decode bench**:                  see section 4'
record '- **Concurrent bench**:              see section 5'
record '- **GPTQ (if present)**:             see section 6'
record ''
record "Full log: ${LOG}"
record ""
record "Next steps depend on these results. If parity fails at a specific"
record "layer, look at CudaBackend method for that op and compare to the"
record "CpuBackend reference in \`ferrum-kernels/src/backend/cpu.rs\`."

echo
echo "Report: ${REPORT}"
echo "Log:    ${LOG}"
