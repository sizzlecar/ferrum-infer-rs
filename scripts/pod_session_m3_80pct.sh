#!/usr/bin/env bash
#
# pod_session_m3_80pct.sh — single-pod orchestrator that retires the
# data-quality caveats in
#   docs/bench/m3-80pct-goal-2026-05-25/GOAL.md
#
# Does:
#   - apt + rust + repo + venv setup (sequential, ~3 min)
#   - PARALLEL: ferrum build (cuda+vllm-moe-marlin+vllm-paged-attn-v2),
#               pip install vllm==0.20.2, HF download Qwen3-30B-A3B-Int4
#   - lock_gpu
#   - ferrum ON-path sweep (FERRUM_VLLM_MOE=1, c=1/4/16/32, n_repeats=5,
#                            num_prompts=128, random 256/128)
#   - vLLM 0.20.2 apples-to-apples sweep (random 256/128, c=1/4/16/32, n=5)
#   - nsys profile on ON-path at c=32 → captures missing bottleneck data
#   - unlock_gpu
#
# Every long step wrapped in `timeout` per feedback_bench_hang_guard.
# Background launches via nohup so SSH disconnects don't kill them.
#
# Usage:
#   bash scripts/pod_session_m3_80pct.sh [BRANCH]
#
# All output → /workspace/m3-80pct-session/
# Final results → docs/bench/m3-80pct-goal-2026-05-25/session-2026-05-26/

set -euo pipefail

BRANCH="${1:-claude-md-behavioral-guidelines}"
REPO_URL="${REPO_URL:-https://github.com/sizzlecar/ferrum-infer-rs.git}"
SESSION_DIR="/workspace/m3-80pct-session"
mkdir -p "$SESSION_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SESSION_DIR/orchestrator.log" >&2; }

# ──────────────────────────────────────────────────────────────────────
# Phase 0 — apt + rust + repo (sequential, must finish before parallel)
# ──────────────────────────────────────────────────────────────────────
log "▶ Phase 0: apt tooling"
export DEBIAN_FRONTEND=noninteractive
timeout 600 apt-get update -qq
timeout 1200 apt-get install -y -qq \
    curl git build-essential pkg-config libssl-dev cmake \
    python3-venv python3-pip jq bc rsync \
    >/dev/null

log "▶ Phase 0: Rust"
if [ ! -d "$HOME/.cargo" ]; then
    timeout 300 curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal >/dev/null
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
log "  $(rustc --version)"

log "▶ Phase 0: git clone"
mkdir -p /workspace
cd /workspace
if [ ! -d ferrum-infer-rs ]; then
    timeout 300 git clone "$REPO_URL"
fi
cd ferrum-infer-rs
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"
log "  ferrum at $(git rev-parse --short HEAD)"

log "▶ Phase 0: nvidia-smi sanity"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || {
    log "ERROR: nvidia-smi failed — pod has no GPU"; exit 1; }

# ──────────────────────────────────────────────────────────────────────
# Phase 1 — PARALLEL setup: ferrum build || pip vllm || HF download
# Each task → standalone temp script (avoids declare -f / bash-c quoting
# hell). Each writes its own marker into orchestrator.log on completion.
# ──────────────────────────────────────────────────────────────────────
log "▶ Phase 1: writing parallel task scripts"

cat > /tmp/task_build.sh <<EOSH
#!/usr/bin/env bash
set -e
cd /workspace/ferrum-infer-rs
source "\$HOME/.cargo/env"
export CUDA_COMPUTE_CAP=89
timeout 2700 cargo build --release -p ferrum-cli \\
    --features 'cuda,vllm-moe-marlin,vllm-paged-attn-v2' \\
    2>&1 | grep -E "Compiling|Finished|error|warning:" | tail -200
if [ -x ./target/release/ferrum ]; then
    echo "BUILD_OK" >> "$SESSION_DIR/orchestrator.log"
else
    echo "BUILD_FAIL" >> "$SESSION_DIR/orchestrator.log"
fi
EOSH

cat > /tmp/task_vllm.sh <<EOSH
#!/usr/bin/env bash
set -e
if [ ! -d "/workspace/vllm-venv" ]; then
    python3 -m venv /workspace/vllm-venv
fi
source /workspace/vllm-venv/bin/activate
timeout 60 pip install -U -q pip wheel setuptools
# vllm 0.20.2 pulls torch transitively. cu128 wheel works on driver 580.
timeout 1800 pip install -q \\
    --extra-index-url https://download.pytorch.org/whl/cu128 \\
    'vllm==0.20.2' 'huggingface_hub[cli]' || true
if /workspace/vllm-venv/bin/vllm --version >/dev/null 2>&1; then
    echo "VLLM_OK" >> "$SESSION_DIR/orchestrator.log"
else
    echo "VLLM_FAIL" >> "$SESSION_DIR/orchestrator.log"
fi
EOSH

cat > /tmp/task_hf.sh <<EOSH
#!/usr/bin/env bash
set -e
export HF_HOME="\${HF_HOME:-/workspace/hf-cache}"
mkdir -p "\$HF_HOME"
# huggingface_hub >=1.x deprecates huggingface-cli in favor of \`hf\`.
# Install + use the new CLI.
if ! command -v hf >/dev/null 2>&1; then
    pip3 install -q --user 'huggingface_hub' >/dev/null 2>&1 || \\
        pip3 install -q --user --break-system-packages 'huggingface_hub' || true
    export PATH="\$HOME/.local/bin:\$PATH"
fi
timeout 1800 hf download Qwen/Qwen3-30B-A3B-GPTQ-Int4 || true
if [ -d "\$HF_HOME/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4" ]; then
    echo "MODEL_OK" >> "$SESSION_DIR/orchestrator.log"
else
    echo "MODEL_FAIL" >> "$SESSION_DIR/orchestrator.log"
fi
EOSH

chmod +x /tmp/task_build.sh /tmp/task_vllm.sh /tmp/task_hf.sh

log "▶ Phase 1: launching 3 parallel tasks (nohup)"
nohup bash /tmp/task_build.sh > "$SESSION_DIR/build.log" 2>&1 &
BUILD_PID=$!
log "  task A (ferrum build) PID=$BUILD_PID"

nohup bash /tmp/task_vllm.sh > "$SESSION_DIR/vllm_install.log" 2>&1 &
VLLM_PID=$!
log "  task B (vllm install) PID=$VLLM_PID"

nohup bash /tmp/task_hf.sh > "$SESSION_DIR/hf_download.log" 2>&1 &
HF_PID=$!
log "  task C (HF download) PID=$HF_PID"

# Wait for all three. Show progress every 60s.
log "▶ Phase 1: waiting for parallel tasks (build/vllm/hf)"
while kill -0 "$BUILD_PID" 2>/dev/null || kill -0 "$VLLM_PID" 2>/dev/null || kill -0 "$HF_PID" 2>/dev/null; do
    B=$(kill -0 "$BUILD_PID" 2>/dev/null && echo "running" || echo "done")
    V=$(kill -0 "$VLLM_PID" 2>/dev/null && echo "running" || echo "done")
    H=$(kill -0 "$HF_PID" 2>/dev/null && echo "running" || echo "done")
    log "  build=$B vllm=$V hf=$H"
    sleep 60
done

# Status check
for marker in BUILD_OK VLLM_OK MODEL_OK; do
    if grep -q "$marker" "$SESSION_DIR/orchestrator.log"; then
        log "  ✓ $marker"
    else
        log "  ✗ $marker missing — check $SESSION_DIR/*.log"
    fi
done

# Hard fail if build didn't produce binary (sweep can't run without it)
if [ ! -x /workspace/ferrum-infer-rs/target/release/ferrum ]; then
    log "FATAL: ferrum binary missing. Aborting sweep."
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────
# Phase 2 — Lock GPU
# ──────────────────────────────────────────────────────────────────────
log "▶ Phase 2: lock GPU clocks"
cd /workspace/ferrum-infer-rs
bash scripts/lock_gpu.sh 2>&1 | tail -5 || log "  (lock failed — proceeding unlocked)"

# Cleanup trap
cleanup() {
    log "▶ cleanup"
    pkill -f "target/release/ferrum serve" 2>/dev/null || true
    pkill -f "target/release/ferrum run" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true
    bash /workspace/ferrum-infer-rs/scripts/unlock_gpu.sh 2>&1 | tail -3 || true
}
trap cleanup EXIT INT TERM

# ──────────────────────────────────────────────────────────────────────
# Phase 2.5 — Paris bisect (gate). If the moe_align_block_size.cu fix
# is wrong or the device-route still emits garbage, abort BEFORE the
# 90-min sweep so we don't collect more "garbage emission rate" numbers
# like session-2026-05-25 did.
# ──────────────────────────────────────────────────────────────────────
log "▶ Phase 2.5: Paris bisect gate"
if bash scripts/paris_bisect.sh ./target/release/ferrum "$SESSION_DIR/paris" \
        > "$SESSION_DIR/paris.log" 2>&1; then
    log "  ✓ all 4 cells produced 'Paris' — sweep is meaningful"
else
    log "  ✗ Paris bisect FAILED — see $SESSION_DIR/paris.log + $SESSION_DIR/paris/"
    log "    aborting sweep; the fix needs more work."
    cat "$SESSION_DIR/paris.log" | tail -40 || true
    echo "PARIS_FAIL" >> "$SESSION_DIR/orchestrator.log"
    exit 3
fi

# ──────────────────────────────────────────────────────────────────────
# Phase 3 — Run ON-path sweep with FERRUM_VLLM_MOE=1, n=5 prompts=128
# Uses scripts/sweep_bottleneck.sh but with publication-grade overrides
# ──────────────────────────────────────────────────────────────────────
log "▶ Phase 3: ON-path sweep (FERRUM_VLLM_MOE=1, n=5, prompts=128)"

export FERRUM_VLLM_MOE=1
# Activate vllm venv if available; otherwise skip vllm half of sweep.
if [ -d /workspace/vllm-venv ] && [ -x /workspace/vllm-venv/bin/vllm ]; then
    source /workspace/vllm-venv/bin/activate
    log "  vllm available: $(vllm --version 2>&1 | head -1)"
    export SKIP_VLLM=0
else
    log "  vllm NOT available — SKIP_VLLM=1 (ferrum-only sweep)"
    export SKIP_VLLM=1
fi
export N_REPEATS=5
export NUM_PROMPTS=128
export WARMUP=10

# sweep_bottleneck.sh runs ferrum + vllm per cell; output → docs/bench/sweep-<date>-<model>/c{N}/
# Force a stable directory name we can grab afterwards.
SWEEP_OUT="docs/bench/m3-80pct-session-2026-05-26"
mkdir -p "$SWEEP_OUT"
# Patch sweep_bottleneck.sh's OUT_ROOT for this session via env (or
# symlink after). Cleanest: run the sweep, then move its output.
timeout 5400 bash scripts/sweep_bottleneck.sh qwen3-moe-30b-int4 1,4,16,32 \
    > "$SESSION_DIR/sweep.log" 2>&1 || {
        log "  sweep returned non-zero — collecting partial results anyway"
    }

# Grab the most recent sweep output directory
ACTUAL_OUT=$(ls -td docs/bench/sweep-*-qwen3-moe-30b-int4 2>/dev/null | head -1)
if [ -n "$ACTUAL_OUT" ] && [ -d "$ACTUAL_OUT" ]; then
    log "  sweep output: $ACTUAL_OUT"
    cp -a "$ACTUAL_OUT"/. "$SWEEP_OUT"/
    log "  also copied to $SWEEP_OUT/"
fi

# ──────────────────────────────────────────────────────────────────────
# Phase 4 — Aggregate
# ──────────────────────────────────────────────────────────────────────
log "▶ Phase 4: aggregate"
if [ -f scripts/aggregate_sweep.py ]; then
    timeout 60 python3 scripts/aggregate_sweep.py "$SWEEP_OUT" \
        > "$SWEEP_OUT/aggregate.md" 2>&1 || true
fi

log "▶ all phases complete"
log "  artifacts under $SWEEP_OUT/"
log "  c=32 nsys at $SWEEP_OUT/c32/ferrum_nsys.nsys-rep"
ls -la "$SWEEP_OUT" 2>/dev/null | tee -a "$SESSION_DIR/orchestrator.log"

# Final marker for the local poll loop to grep on.
echo "SESSION_COMPLETE" >> "$SESSION_DIR/orchestrator.log"
log "✓ session done — fetch results via scp"
