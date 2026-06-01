#!/usr/bin/env bash
#
# Real-model OpenAI API compatibility smoke wrapper for F/G evidence capture.
# Runs ignored async-openai + optional Python OpenAI SDK tests against a cached
# real model and writes structured command logs under OUT_ROOT.

set -euo pipefail

OUT_ROOT="${OUT_ROOT:-/workspace/m3-real-model-api-smoke-$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-qwen3:0.6b}"
CARGO_FEATURES="${CARGO_FEATURES:-metal}"
FERRUM_BIN="${FERRUM_BIN:-ferrum}"
PYTHON_TEST="${PYTHON_TEST:-1}"
ASYNC_TESTS="${ASYNC_TESTS:-1}"
PULL_MODEL="${PULL_MODEL:-1}"
RUN_PYTHON_CHECK="${RUN_PYTHON_CHECK:-1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Async-openai test filters that exercise the critical non-streaming/streaming
# compatibility path on real model.
ASYNC_OPENAI_TESTS=(
    test_openai_client_chat_basic
    test_openai_client_chat_usage_fields
    test_openai_client_chat_streaming
    test_openai_client_tools_stream_options_include_usage
    test_openai_client_response_format_json_object
    test_openai_client_strict_json_schema_20_runs
    test_openai_client_multi_turn
)

mkdir -p "$OUT_ROOT"

MANIFEST="$OUT_ROOT/commands.md"
SUMMARY_JSON="$OUT_ROOT/run_summary.json"
LOG_PREFIX="$OUT_ROOT/cargo-test"
FAILURES=0
COMMAND_COUNT=0

now_ms() {
    python3 - <<'PY'
import time

print(time.time_ns() // 1_000_000)
PY
}

{
    echo "# Real-model API smoke execution"
    echo "- date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- model: ${MODEL}"
    echo "- cargo_features: ${CARGO_FEATURES}"
    echo "- ferrum_bin: ${FERRUM_BIN}"
    echo "- repository_head: $(git -C "$REPO_ROOT" rev-parse --short HEAD)"
    echo "- rustc: $(rustc --version)"
    echo
} > "$MANIFEST"

run_cmd() {
    local name="$1"
    shift
    local log="$LOG_PREFIX-${name}.log"
    local start_ts_ms
    local end_ts_ms
    local elapsed_ms
    COMMAND_COUNT=$((COMMAND_COUNT + 1))
    printf '\n### %s\n' "$name" >> "$MANIFEST"
    echo "- started_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
    echo "- cmd: ${*}" >> "$MANIFEST"
    start_ts_ms="$(now_ms)"

    # shellcheck disable=SC2317
    {
        set +e
        ("$@" 2>&1 | tee "$log")
        local rc=$?
        set -e
        end_ts_ms="$(now_ms)"
        elapsed_ms=$(( end_ts_ms - start_ts_ms ))
        if (( rc != 0 )); then
            echo "- elapsed_ms: ${elapsed_ms}" >> "$MANIFEST"
            echo "- rc: ${rc}" >> "$MANIFEST"
            echo "- finished_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
            FAILURES=1
            return 0
        fi
        echo "- elapsed_ms: ${elapsed_ms}" >> "$MANIFEST"
        echo "- rc: 0" >> "$MANIFEST"
        echo "- finished_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
    }
}

if [[ "$PULL_MODEL" == "1" ]]; then
    run_cmd pull_model "$FERRUM_BIN" pull "$MODEL"
fi

if [[ "$RUN_PYTHON_CHECK" == "1" ]]; then
    run_cmd python_pkg python3 -m pip show openai >/dev/null
fi

if [[ "$ASYNC_TESTS" == "1" ]]; then
    for test_name in "${ASYNC_OPENAI_TESTS[@]}"; do
        run_cmd "${test_name}" \
            cargo test -q -p ferrum-cli --features "$CARGO_FEATURES" \
            --test server_openai_compat -- --ignored "$test_name" --test-threads=1
    done
fi

if [[ "$PYTHON_TEST" == "1" ]]; then
    run_cmd python_openai_test \
        cargo test -q -p ferrum-cli --features "$CARGO_FEATURES" \
        --test server_openai_compat -- --ignored test_python_openai_sdk_chat_and_stream_smoke --test-threads=1
fi

if (( COMMAND_COUNT == 0 )); then
    FAILURES=1
fi

python3 - "$SUMMARY_JSON" "$MANIFEST" <<'PY'
import json
import re
from pathlib import Path

summary_path = Path(__import__('sys').argv[1])
manifest_path = Path(__import__('sys').argv[2])
text = manifest_path.read_text().splitlines()

status = {
    "script": "m3_real_model_api_smoke.sh",
    "commands": [],
}
current = None
for line in text:
    if line.startswith("### "):
        if current:
            status["commands"].append(current)
        current = {"name": line[4:].strip(), "rc": None}
    elif line.startswith("- started_at_utc:") and current is not None:
        current["started_at_utc"] = line.split(":", 1)[1].strip()
    elif line.startswith("- finished_at_utc:") and current is not None:
        current["finished_at_utc"] = line.split(":", 1)[1].strip()
    elif line.startswith("- cmd:") and current is not None:
        current["cmd"] = line.split(":", 1)[1].strip()
    elif line.startswith("- rc:") and current is not None:
        current["rc"] = int(re.findall(r"-?\d+", line)[0])
    elif line.startswith("- elapsed_ms:") and current is not None:
        current["elapsed_ms"] = int(re.findall(r"-?\d+", line)[0])

if current:
    status["commands"].append(current)

status["all_passed"] = bool(status["commands"]) and all(
    cmd.get("rc") == 0 for cmd in status["commands"]
)
status_path = summary_path
summary_path.write_text(json.dumps(status, indent=2) + "\n")
PY

echo "out_root: $OUT_ROOT"
echo "summary: $SUMMARY_JSON"

if [[ "${CI:-}" == "1" && -f "$SUMMARY_JSON" ]]; then
    python3 - "$SUMMARY_JSON" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text())
if not summary.get("all_passed"):
    raise SystemExit("m3_real_model_api_smoke.sh did not pass all commands")
PY
fi

exit "$FAILURES"
