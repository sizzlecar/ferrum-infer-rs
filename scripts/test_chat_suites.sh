#!/usr/bin/env bash
#
# Run every chat / HTTP smoke suite locally — the same 6 test files that
# nightly CI (chat-smoke.yml) runs against real models.
#
# Total runtime: ~115 s on M1 Metal with --release (a fresh build adds
# ~3 min compile).
#
# Usage:
#   scripts/test_chat_suites.sh             # all 7 suites
#   scripts/test_chat_suites.sh cli         # just the 3 CLI suites
#   scripts/test_chat_suites.sh http        # just the 3 HTTP suites
#   scripts/test_chat_suites.sh correctness # reference_match (byte-equal baseline)
#   scripts/test_chat_suites.sh chat_smoke  # just one named suite
#
# Env:
#   FEATURES=metal       — feature flags passed to cargo (default: metal on macOS, empty otherwise)
#   FERRUM_PREFIX_CACHE  — opt in to prefix cache (otherwise default OFF, see PR #204)

set -u
set -o pipefail

cd "$(dirname "$0")/.."

if [ "$(uname -s)" = "Darwin" ]; then
    FEATURES="${FEATURES:-metal}"
else
    FEATURES="${FEATURES:-}"
fi

# Map filter arg → suites to run.
case "${1:-all}" in
    all)
        suites=(chat_smoke chat_pty chat_stress server_smoke server_openai_compat server_stress reference_match)
        ;;
    cli)
        suites=(chat_smoke chat_pty chat_stress)
        ;;
    http)
        suites=(server_smoke server_openai_compat server_stress)
        ;;
    correctness)
        suites=(reference_match)
        ;;
    *)
        suites=("$1")
        ;;
esac

# Pre-flight: required HF model.
hf_root="${HF_HOME:-$HOME/.cache/huggingface}"
if [ ! -d "$hf_root/hub/models--Qwen--Qwen3-0.6B" ]; then
    echo "ERROR: Qwen/Qwen3-0.6B not found in $hf_root/hub/"
    echo "       Run: ferrum pull qwen3:0.6b"
    exit 1
fi

# TinyLlama + Qwen2.5-0.5B are only needed by chat_smoke's no_template_leak
# multi-model cases. Warn if missing but don't block — most tests pass.
optional=(
    "TinyLlama--TinyLlama-1.1B-Chat-v1.0:tinyllama"
    "Qwen--Qwen2.5-0.5B-Instruct:qwen2.5:0.5b"
)
for entry in "${optional[@]}"; do
    # Repo and alias are separated by the FIRST ':' — but the alias itself
    # contains ':' (e.g. `qwen2.5:0.5b`), so use longest-suffix-strip on the
    # repo side and shortest-prefix-strip on the alias side.
    repo="${entry%%:*}"
    alias_name="${entry#*:}"
    if [ ! -d "$hf_root/hub/models--$repo" ]; then
        echo "WARN: missing optional model $repo — chat_smoke multi-model cases will fail."
        echo "      Run: ferrum pull $alias_name"
    fi
done

# Build once so per-suite cargo test calls don't repeat compile work.
echo
echo "==> Building release tests (FEATURES=$FEATURES)"
if [ -n "$FEATURES" ]; then
    build_args=(--features "$FEATURES")
else
    build_args=()
fi
cargo build --release -p ferrum-cli "${build_args[@]}" --tests 2>&1 \
    | grep -E "Compiling|Finished|error" || true

# Run each suite serially. Aggregate pass/fail/duration.
declare -a results=()
overall_start=$(date +%s)
exit_code=0

for suite in "${suites[@]}"; do
    echo
    echo "==> $suite"
    start=$(date +%s)
    if cargo test --release -p ferrum-cli "${build_args[@]}" \
            --test "$suite" -- --ignored --test-threads=1; then
        status="PASS"
    else
        status="FAIL"
        exit_code=1
    fi
    end=$(date +%s)
    duration=$((end - start))
    results+=("$status $suite ${duration}s")
done

# Summary table.
overall=$(($(date +%s) - overall_start))
echo
echo "==================================================="
echo "Summary"
echo "==================================================="
for line in "${results[@]}"; do
    echo "  $line"
done
echo "---------------------------------------------------"
echo "  total: ${overall}s"
echo "==================================================="

exit "$exit_code"
