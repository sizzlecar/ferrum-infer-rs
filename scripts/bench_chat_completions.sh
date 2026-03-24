#!/usr/bin/env bash

set -euo pipefail

URL="http://127.0.0.1:8000/v1/chat/completions"
MODEL=""
PROMPT="Explain continuous batching in one paragraph."
REQUESTS=20
CONCURRENCY=4
MAX_TOKENS=128
TEMPERATURE="0.7"
CONNECT_TIMEOUT=10
REQUEST_TIMEOUT=600

usage() {
    cat <<'EOF'
Usage:
  scripts/bench_chat_completions.sh --model MODEL [options]

Options:
  --url URL                 Chat completions endpoint
  --model MODEL             Model name sent in the request body
  --prompt TEXT             Single prompt used for every request
  --requests N              Total requests to send
  --concurrency N           Max in-flight requests
  --max-tokens N            max_tokens in request body
  --temperature FLOAT       temperature in request body
  --connect-timeout SEC     curl connect timeout
  --request-timeout SEC     curl total timeout
  -h, --help                Show this help

Example:
  scripts/bench_chat_completions.sh \
    --model Qwen/Qwen3-4B \
    --requests 100 \
    --concurrency 8 \
    --max-tokens 128 \
    --prompt "Write a short summary of Rust async."
EOF
}

json_escape() {
    awk '
        BEGIN { ORS = ""; first = 1 }
        {
            gsub(/\\/,"\\\\");
            gsub(/"/,"\\\"");
            gsub(/\t/,"\\t");
            gsub(/\r/,"\\r");
            if (!first) {
                printf "\\n";
            }
            first = 0;
            printf "%s", $0;
        }
    ' <<<"$1"
}

require_number() {
    local value="$1"
    local name="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "invalid $name: $value" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --url)
            URL="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --requests)
            REQUESTS="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --connect-timeout)
            CONNECT_TIMEOUT="$2"
            shift 2
            ;;
        --request-timeout)
            REQUEST_TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "--model is required" >&2
    usage >&2
    exit 1
fi

require_number "$REQUESTS" "requests"
require_number "$CONCURRENCY" "concurrency"
require_number "$MAX_TOKENS" "max-tokens"
require_number "$CONNECT_TIMEOUT" "connect-timeout"
require_number "$REQUEST_TIMEOUT" "request-timeout"

if (( REQUESTS == 0 || CONCURRENCY == 0 )); then
    echo "requests and concurrency must be > 0" >&2
    exit 1
fi

if (( CONCURRENCY > REQUESTS )); then
    CONCURRENCY="$REQUESTS"
fi

MODEL_JSON=$(json_escape "$MODEL")
PROMPT_JSON=$(json_escape "$PROMPT")

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ferrum-http-bench.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

run_one() {
    local id="$1"
    local out_file="$TMP_DIR/$id.result"
    local payload
    payload=$(cat <<EOF
{"model":"$MODEL_JSON","messages":[{"role":"user","content":"$PROMPT_JSON"}],"max_tokens":$MAX_TOKENS,"temperature":$TEMPERATURE,"stream":false}
EOF
)

    local curl_output
    local curl_status=0
    curl_output="$(
        curl -sS \
            --noproxy '*' \
            --connect-timeout "$CONNECT_TIMEOUT" \
            --max-time "$REQUEST_TIMEOUT" \
            -H "Content-Type: application/json" \
            -d "$payload" \
            -w $'\n__HTTP_CODE__=%{http_code}\n__TIME_TOTAL__=%{time_total}\n' \
            "$URL"
    )" || curl_status=$?

    local http_code="000"
    local time_total="0"
    http_code="$(printf '%s\n' "$curl_output" | sed -n 's/^__HTTP_CODE__=//p' | tail -n1)"
    time_total="$(printf '%s\n' "$curl_output" | sed -n 's/^__TIME_TOTAL__=//p' | tail -n1)"

    local body
    body="$(printf '%s\n' "$curl_output" | sed '/^__HTTP_CODE__=/d; /^__TIME_TOTAL__=/d')"

    local prompt_tokens=""
    local completion_tokens=""
    prompt_tokens="$(printf '%s' "$body" | sed -n 's/.*"prompt_tokens"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -n1)"
    completion_tokens="$(printf '%s' "$body" | sed -n 's/.*"completion_tokens"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -n1)"

    {
        printf 'id=%s\n' "$id"
        printf 'curl_status=%s\n' "$curl_status"
        printf 'http_code=%s\n' "$http_code"
        printf 'time_total=%s\n' "$time_total"
        printf 'prompt_tokens=%s\n' "$prompt_tokens"
        printf 'completion_tokens=%s\n' "$completion_tokens"
        printf 'body=%s\n' "$(printf '%s' "$body" | tr '\n' ' ' | cut -c1-400)"
    } >"$out_file"
}

get_result_field() {
    local file="$1"
    local key="$2"
    sed -n "s/^${key}=//p" "$file" | tail -n1
}

now_ns() {
    if command -v python3 >/dev/null 2>&1; then
        python3 -c 'import time; print(time.time_ns())'
        return
    fi

    if command -v perl >/dev/null 2>&1; then
        perl -MTime::HiRes=time -e 'printf "%.0f\n", time() * 1e9'
        return
    fi

    date +%s | awk '{ printf "%.0f\n", $1 * 1000000000 }'
}

wait_for_one_pid() {
    local i
    local pid

    while true; do
        for i in "${!PIDS[@]}"; do
            pid="${PIDS[$i]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid"
                unset 'PIDS[$i]'
                PIDS=("${PIDS[@]}")
                return
            fi
        done
        sleep 0.05
    done
}

echo "Endpoint:     $URL"
echo "Model:        $MODEL"
echo "Requests:     $REQUESTS"
echo "Concurrency:  $CONCURRENCY"
echo "Max tokens:   $MAX_TOKENS"
echo "Temperature:  $TEMPERATURE"
echo

bench_start_ns="$(now_ns)"

PIDS=()
for id in $(seq 1 "$REQUESTS"); do
    run_one "$id" &
    PIDS+=("$!")

    if (( ${#PIDS[@]} >= CONCURRENCY )); then
        wait_for_one_pid
    fi
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

bench_end_ns="$(now_ns)"
wall_seconds="$(
    awk -v start="$bench_start_ns" -v end="$bench_end_ns" 'BEGIN { printf "%.6f", (end - start) / 1000000000 }'
)"

summary="$TMP_DIR/summary.tsv"
for result_file in "$TMP_DIR"/*.result; do
    id="$(get_result_field "$result_file" "id")"
    curl_status="$(get_result_field "$result_file" "curl_status")"
    http_code="$(get_result_field "$result_file" "http_code")"
    time_total="$(get_result_field "$result_file" "time_total")"
    prompt_tokens="$(get_result_field "$result_file" "prompt_tokens")"
    completion_tokens="$(get_result_field "$result_file" "completion_tokens")"

    success=0
    req_tps=0
    if [[ "$curl_status" == "0" && "$http_code" =~ ^2 && -n "${completion_tokens:-}" && -n "${time_total:-}" ]]; then
        success=1
        req_tps="$(
            awk -v tok="$completion_tokens" -v secs="$time_total" 'BEGIN {
                if (secs > 0) {
                    printf "%.6f", tok / secs
                } else {
                    printf "0"
                }
            }'
        )"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$id" \
        "$success" \
        "$http_code" \
        "${time_total:-0}" \
        "${prompt_tokens:-0}" \
        "${completion_tokens:-0}" \
        "$req_tps" >>"$summary"
done

success_count="$(awk -F '\t' '$2 == 1 { count += 1 } END { print count + 0 }' "$summary")"
failure_count=$((REQUESTS - success_count))
total_completion_tokens="$(awk -F '\t' '$2 == 1 { sum += $6 } END { print sum + 0 }' "$summary")"
total_prompt_tokens="$(awk -F '\t' '$2 == 1 { sum += $5 } END { print sum + 0 }' "$summary")"
avg_latency="$(awk -F '\t' '$2 == 1 { sum += $4; count += 1 } END { if (count > 0) printf "%.3f", sum / count; else print "0" }' "$summary")"
aggregate_tps="$(awk -v tok="$total_completion_tokens" -v secs="$wall_seconds" 'BEGIN {
    if (secs > 0) {
        printf "%.3f", tok / secs
    } else {
        print "0"
    }
}')"
avg_req_tps="$(awk -F '\t' '$2 == 1 { sum += $7; count += 1 } END { if (count > 0) printf "%.3f", sum / count; else print "0" }' "$summary")"
requests_per_second="$(awk -v ok="$success_count" -v secs="$wall_seconds" 'BEGIN {
    if (secs > 0) {
        printf "%.3f", ok / secs
    } else {
        print "0"
    }
}')"

success_times="$TMP_DIR/success_times.txt"
awk -F '\t' '$2 == 1 { print $4 }' "$summary" | sort -n >"$success_times"
success_lines="$(wc -l <"$success_times" | tr -d ' ')"

p50_latency="0"
p95_latency="0"
if (( success_lines > 0 )); then
    p50_index=$(( (success_lines * 50 + 99) / 100 ))
    p95_index=$(( (success_lines * 95 + 99) / 100 ))
    p50_latency="$(sed -n "${p50_index}p" "$success_times")"
    p95_latency="$(sed -n "${p95_index}p" "$success_times")"
fi

echo "Summary"
echo "  Success:            $success_count/$REQUESTS"
echo "  Failure:            $failure_count"
echo "  Wall time:          ${wall_seconds}s"
echo "  Requests/s:         ${requests_per_second}"
echo "  Completion tokens:  $total_completion_tokens"
echo "  Prompt tokens:      $total_prompt_tokens"
echo "  Aggregate tok/s:    ${aggregate_tps}"
echo "  Avg req tok/s:      ${avg_req_tps}"
echo "  Avg latency:        ${avg_latency}s"
echo "  P50 latency:        ${p50_latency}s"
echo "  P95 latency:        ${p95_latency}s"

if (( failure_count > 0 )); then
    echo
    echo "Failed requests"
    awk -F '\t' '$2 != 1 { print $1 }' "$summary" | while read -r failed_id; do
        echo "  #$failed_id"
        sed -n 's/^curl_status=//p; s/^http_code=//p; s/^body=//p' "$TMP_DIR/$failed_id.result" \
            | sed '1s/^/    curl_status: /; 2s/^/    http_code: /; 3s/^/    body: /'
    done
fi
