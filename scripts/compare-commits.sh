#!/usr/bin/env bash
#
# Bench-compare two commits — PLAYBOOK § 4.A.
#
# For each commit: git worktree → cargo build --release → run identical
# bench-serve against a local model → write JSONL → join → emit ratio
# table marking only the cells where 95% CIs don't overlap.
#
# Usage:
#   scripts/compare-commits.sh <commit_a> <commit_b> <model> <concurrency>
#
# Example:
#   scripts/compare-commits.sh HEAD~1 HEAD qwen3:0.6b 32
#
# Env:
#   FERRUM_BIN_FEATURES   features for cargo build (default: metal on
#                         macOS, empty otherwise)
#   N_REPEATS             default 5
#   NUM_PROMPTS           default 100
#   WARMUP                default 10

set -euo pipefail

if [ $# -lt 4 ]; then
    sed -n '2,15p' "$0" | sed 's/^# \?//'
    exit 1
fi

COMMIT_A="$1"
COMMIT_B="$2"
MODEL="$3"
CONCURRENCY="$4"

N_REPEATS="${N_REPEATS:-5}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
WARMUP="${WARMUP:-10}"

if [ "$(uname -s)" = "Darwin" ]; then
    FERRUM_BIN_FEATURES="${FERRUM_BIN_FEATURES:-metal}"
else
    FERRUM_BIN_FEATURES="${FERRUM_BIN_FEATURES:-}"
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
RESULTS_DIR="$REPO_ROOT/.compare-commits-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

cleanup() {
    local ec=$?
    # Best-effort kill any lingering ferrum serve.
    pkill -f "target/release/ferrum serve" 2>/dev/null || true
    return $ec
}
trap cleanup EXIT INT TERM

SHA_A=$(cd "$REPO_ROOT" && git rev-parse --short "$COMMIT_A")
SHA_B=$(cd "$REPO_ROOT" && git rev-parse --short "$COMMIT_B")

build_and_bench() {
    local sha="$1"
    local out_file="$2"

    local wt_dir="$RESULTS_DIR/wt-$sha"
    if [ -d "$wt_dir" ]; then
        echo "▶ worktree for $sha already exists — reusing"
    else
        echo "▶ creating worktree for $sha"
        git -C "$REPO_ROOT" worktree add --detach "$wt_dir" "$sha"
    fi

    echo "▶ building $sha (features='$FERRUM_BIN_FEATURES')"
    local build_args=()
    [ -n "$FERRUM_BIN_FEATURES" ] && build_args=(--features "$FERRUM_BIN_FEATURES")
    (
        cd "$wt_dir"
        cargo build --release -p ferrum-cli "${build_args[@]}" 2>&1 \
            | grep -E "Compiling|Finished|error" || true
    )

    local bin="$wt_dir/target/release/ferrum"
    if [ ! -x "$bin" ]; then
        echo "ERROR: binary missing at $bin" >&2
        return 1
    fi

    # Resolve a tokenizer dir for the model (best-effort via HF cache).
    local hf="${HF_HOME:-$HOME/.cache/huggingface}"
    local tok_dir
    tok_dir=$(find "$hf/hub" -type d -name "models--*$(echo "$MODEL" | sed 's,[/:],-,g')*" 2>/dev/null | head -1)
    [ -z "$tok_dir" ] && tok_dir=$(find "$hf/hub" -type d -name "models--Qwen--Qwen3-0.6B*" 2>/dev/null | head -1)
    local tok_snap
    tok_snap=$(find "$tok_dir/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
    if [ -z "$tok_snap" ] || [ ! -f "$tok_snap/tokenizer.json" ]; then
        echo "ERROR: tokenizer.json not located under $tok_dir" >&2
        return 1
    fi

    local port
    port=$((RANDOM % 1000 + 18000))
    echo "▶ starting $sha serve on port $port"
    "$bin" serve "$MODEL" --port "$port" \
        > "$RESULTS_DIR/serve-$sha.log" 2>&1 &
    local serve_pid=$!

    # Wait for /health
    local i=0
    while ! curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
        i=$((i + 1))
        if [ "$i" -gt 120 ]; then
            echo "ERROR: $sha serve did not become healthy" >&2
            kill "$serve_pid" 2>/dev/null || true
            return 1
        fi
        sleep 1
    done

    echo "▶ bench-serve against $sha"
    "$bin" bench-serve \
        --base-url "http://127.0.0.1:$port" \
        --model "$MODEL" --tokenizer "$tok_snap" \
        --dataset random --random-input-len 256 --random-output-len 128 \
        --num-prompts "$NUM_PROMPTS" --warmup-requests "$WARMUP" \
        --n-repeats "$N_REPEATS" \
        --concurrency "$CONCURRENCY" \
        --output jsonl --out "$out_file" \
        --commit-sha "$sha"

    kill "$serve_pid" 2>/dev/null || true
    wait "$serve_pid" 2>/dev/null || true
    sleep 2
}

build_and_bench "$SHA_A" "$RESULTS_DIR/a.jsonl"
build_and_bench "$SHA_B" "$RESULTS_DIR/b.jsonl"

echo
echo "▶ joining and rendering"
python3 "$REPO_ROOT/scripts/compare_bench.py" \
    --label-a "$SHA_A" --label-b "$SHA_B" \
    "$RESULTS_DIR/a.jsonl" "$RESULTS_DIR/b.jsonl" \
    | tee "$RESULTS_DIR/compare.md"

cat <<EOF

✓ compare complete

  worktree A:  $RESULTS_DIR/wt-$SHA_A
  worktree B:  $RESULTS_DIR/wt-$SHA_B
  bench A:     $RESULTS_DIR/a.jsonl
  bench B:     $RESULTS_DIR/b.jsonl
  rendered:    $RESULTS_DIR/compare.md

Remove worktrees with:
  git worktree remove $RESULTS_DIR/wt-$SHA_A
  git worktree remove $RESULTS_DIR/wt-$SHA_B
EOF
