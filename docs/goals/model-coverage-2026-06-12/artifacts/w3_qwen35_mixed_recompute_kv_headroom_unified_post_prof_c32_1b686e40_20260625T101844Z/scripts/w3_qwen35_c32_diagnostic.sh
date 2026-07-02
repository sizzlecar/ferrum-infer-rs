#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/release/w3_qwen35_c32_diagnostic.sh [options]

Runs a focused remote CUDA diagnostic for the W3 Qwen3.5 c32 lane.
This is diagnostic evidence only; it never prints MODEL_RELEASE_GRADE_W3 PASS.

Options:
  --sha <sha>                         Git SHA to test.
  --tag <name>                        Artifact name tag.
  --throughput-floor <tok/s>          Required output throughput floor.
  --max-kv-admission-failed <n>       Max allowed "Unified KV admission failed" count.
  --max-capacity-deferred <n>         Max allowed final capacity_deferred_total.
  --min-mixed-iterations <n>          Min scheduler iterations with prefill+decode work.
  --max-p95-itl-ms <ms>               Max allowed p95 inter-token latency; empty disables.
  --port <port>                       Local serve port.
  --repo <path>                       Remote repo path.
  --artifact-root <path>              Remote artifact root.
  --branch <branch>                   Branch to fetch before detached checkout.
  --model <model-id>                  HF model id.
  --dataset <path>                    ShareGPT JSONL dataset path.
  --self-test                         Run local verdict self-test and exit.
  -h, --help                          Show this help.
EOF
}

write_diagnostic_summary() {
  python3 - "$1" "$2" "$3" "$4" "$5" "$6" "$7" <<'PY_SUMMARY'
import gzip
import json
import math
import pathlib
import re
import sys

art = pathlib.Path(sys.argv[1])
sha = sys.argv[2]
throughput_floor = float(sys.argv[3])
max_kv_admission_failed = int(sys.argv[4])
max_capacity_deferred = int(sys.argv[5])
min_mixed_iterations = int(sys.argv[6])
max_p95_itl_ms = None if sys.argv[7] == "" else float(sys.argv[7])
UNIFIED_PROF_RE = re.compile(
    r"\[unified-prof\]\s+iter#(?P<iter>\d+)\s+items=(?P<items>\d+)\s+"
    r"prefill=(?P<prefill>\d+)\s+decode=(?P<decode>\d+)\s+"
    r"total=(?P<total>\d+)us\s+model=(?P<model>\d+)us\s+"
    r"decode_post=(?P<decode_post>\d+)us\s+\|\s+sample=(?P<sample>\d+)\s+"
    r"sched=(?P<sched>\d+)\s+stream=(?P<stream>\d+)\s+stop=(?P<stop>\d+)\s+"
    r"complete=(?P<complete>\d+)"
)
summary = {
    "artifact": str(art),
    "sha": sha,
    "release_evidence": False,
    "no_live_vllm": True,
    "thresholds": {
        "output_throughput_floor_tps": throughput_floor,
        "max_kv_admission_failed": max_kv_admission_failed,
        "max_capacity_deferred_total": max_capacity_deferred,
        "min_mixed_iterations": min_mixed_iterations,
        "max_p95_itl_ms": max_p95_itl_ms,
        "required_completed": 32,
        "required_errors": 0,
    },
    "bench_exit": (art / "perf/bench.exit").read_text().strip(),
}

bench_path = art / "perf/bench_ferrum_c32_32x1.json"
bench = {}
if bench_path.exists():
    try:
        bench = json.load(open(bench_path))
    except Exception as exc:
        summary["bench_parse_error"] = str(exc)

def mean_metric(name):
    value = bench.get(name)
    if isinstance(value, dict):
        mean = value.get("mean")
        if isinstance(mean, (int, float)):
            return float(mean)
    return None

def mean_percentile_metric(name, percentile):
    value = bench.get(name)
    if not isinstance(value, dict):
        return None
    point = value.get(percentile)
    if isinstance(point, dict):
        mean = point.get("mean")
        if isinstance(mean, (int, float)):
            return float(mean)
    if isinstance(point, (int, float)):
        return float(point)
    return None

summary["bench"] = {
    "concurrency": bench.get("concurrency"),
    "n_requests_per_run": bench.get("n_requests_per_run"),
    "n_repeats": bench.get("n_repeats"),
    "output_token_count_source": bench.get("output_token_count_source"),
    "completed_per_run": bench.get("completed_per_run"),
    "errored_per_run": bench.get("errored_per_run"),
    "http_500_per_run": bench.get("http_500_per_run"),
    "panic_per_run": bench.get("panic_per_run"),
    "output_throughput_tps": mean_metric("output_throughput_tps"),
    "total_throughput_tps": mean_metric("total_throughput_tps"),
    "request_throughput_rps": mean_metric("request_throughput_rps"),
    "p95_ttft_ms": mean_percentile_metric("ttft_ms", "p95"),
    "p95_tpot_ms": mean_percentile_metric("tpot_ms", "p95"),
    "p95_itl_ms": mean_percentile_metric("itl_ms", "p95"),
}

trace = art / "server/scheduler_trace.jsonl"
if trace.exists():
    lines = trace.read_text(errors="replace").splitlines()
elif (art / "server/scheduler_trace.jsonl.gz").exists():
    lines = gzip.open(art / "server/scheduler_trace.jsonl.gz", "rt", errors="replace").read().splitlines()
else:
    lines = []

trace_stats = {
    "some_ok_iterations": 0,
    "prefill_only_iterations": 0,
    "decode_only_iterations": 0,
    "mixed_iterations": 0,
    "scheduled_decode_tokens": 0,
    "scheduled_prefill_tokens": 0,
    "max_active_len": 0,
    "max_capacity_deferred_total": 0,
    "avg_process_us": None,
    "avg_decode_items": None,
}
process_us_total = 0
decode_items_total = 0
last = None
for line in lines:
    if line.startswith("{"):
        try:
            row = json.loads(line)
        except Exception:
            continue
        last = row
        if row.get("event") != "scheduler_iteration" or row.get("result") != "some_ok":
            continue
        plan = row.get("plan") if isinstance(row.get("plan"), dict) else {}
        before = row.get("scheduler_before") if isinstance(row.get("scheduler_before"), dict) else {}
        timing = row.get("timing_us") if isinstance(row.get("timing_us"), dict) else {}
        prefill_items = int(plan.get("prefill_items") or 0)
        decode_items = int(plan.get("decode_items") or 0)
        prefill_tokens = int(plan.get("prefill_tokens") or 0)
        decode_tokens = int(plan.get("decode_tokens") or 0)
        trace_stats["some_ok_iterations"] += 1
        if prefill_items > 0 and decode_items > 0:
            trace_stats["mixed_iterations"] += 1
        elif prefill_items > 0:
            trace_stats["prefill_only_iterations"] += 1
        elif decode_items > 0:
            trace_stats["decode_only_iterations"] += 1
        trace_stats["scheduled_decode_tokens"] += decode_tokens
        trace_stats["scheduled_prefill_tokens"] += prefill_tokens
        trace_stats["max_active_len"] = max(trace_stats["max_active_len"], int(before.get("active_len") or 0))
        trace_stats["max_capacity_deferred_total"] = max(
            trace_stats["max_capacity_deferred_total"],
            int(before.get("capacity_deferred_total") or 0),
        )
        process_us_total += int(timing.get("process") or 0)
        decode_items_total += decode_items
if trace_stats["some_ok_iterations"]:
    trace_stats["avg_process_us"] = process_us_total / trace_stats["some_ok_iterations"]
    trace_stats["avg_decode_items"] = decode_items_total / trace_stats["some_ok_iterations"]
summary["trace_stats"] = trace_stats
if last:
    summary["last_iteration"] = last.get("iteration")
    for section in ("scheduler_before", "scheduler_after_schedule", "scheduler_after_process"):
        value = last.get(section)
        if isinstance(value, dict):
            summary["last_" + section] = {
                key: value.get(key)
                for key in (
                    "waiting_queue_len",
                    "prefill_queue_len",
                    "decode_queue_len",
                    "active_len",
                    "completed_total",
                    "failed_total",
                    "cancelled_total",
                    "admitted_total",
                    "capacity_deferred_total",
                    "capacity_blocked_waiting_len",
                )
            }

serve = art / "server/serve.log"
text = serve.read_text(errors="replace") if serve.exists() else ""
profile_rows = [
    {key: int(value) for key, value in match.groupdict().items()}
    for match in UNIFIED_PROF_RE.finditer(text)
]

def profile_stats(rows):
    if not rows:
        return {"count": 0}

    def values(name):
        return [row[name] for row in rows]

    def avg(name):
        data = values(name)
        return sum(data) / len(data)

    def p95(name):
        data = sorted(values(name))
        index = min(len(data) - 1, max(0, math.ceil(len(data) * 0.95) - 1))
        return data[index]

    return {
        "count": len(rows),
        "avg_total_us": avg("total"),
        "p95_total_us": p95("total"),
        "max_total_us": max(values("total")),
        "avg_model_us": avg("model"),
        "avg_decode_post_us": avg("decode_post"),
        "avg_sample_us": avg("sample"),
        "avg_sched_us": avg("sched"),
        "avg_stream_us": avg("stream"),
        "avg_stop_us": avg("stop"),
        "avg_complete_us": avg("complete"),
    }

summary["unified_prof"] = {
    "all": profile_stats(profile_rows),
    "mixed": profile_stats([row for row in profile_rows if row["prefill"] > 0 and row["decode"] > 0]),
    "prefill_only": profile_stats([row for row in profile_rows if row["prefill"] > 0 and row["decode"] == 0]),
    "decode_only": profile_stats([row for row in profile_rows if row["decode"] > 0 and row["prefill"] == 0]),
}
summary["log_counts"] = {
    "unified_kv_admission_failed": text.count("Unified KV admission failed"),
    "unified_prof_events": len(profile_rows),
    "cancelled_during_decode": text.count("cancelled during decode"),
    "preempting_request": text.count("Preempting request"),
    "oom_mentions": len(re.findall(r"out of memory|OutOfMemory|OOM", text)),
    "panic_mentions": text.lower().count("panic"),
    "block_pool_exhausted": text.count("Block pool exhausted"),
    "unified_prefill_alloc_deferred": text.count("Unified prefill alloc deferred"),
}

capacity_deferred = None
for key in ("last_scheduler_after_process", "last_scheduler_after_schedule", "last_scheduler_before"):
    value = summary.get(key)
    if isinstance(value, dict) and value.get("capacity_deferred_total") is not None:
        capacity_deferred = value.get("capacity_deferred_total")
        break

completed = summary["bench"].get("completed_per_run") or []
errored = summary["bench"].get("errored_per_run") or []
http_500 = summary["bench"].get("http_500_per_run") or []
panic = summary["bench"].get("panic_per_run") or []
throughput = summary["bench"].get("output_throughput_tps")
p95_itl_ms = summary["bench"].get("p95_itl_ms")
kv_failed = summary["log_counts"]["unified_kv_admission_failed"]
mixed_iterations = summary["trace_stats"]["mixed_iterations"]

failures = []
if summary["bench_exit"] != "0":
    failures.append(f"bench_exit={summary['bench_exit']}")
if completed != [32]:
    failures.append(f"completed_per_run={completed}")
if errored != [0]:
    failures.append(f"errored_per_run={errored}")
if http_500 and http_500 != [0]:
    failures.append(f"http_500_per_run={http_500}")
if panic and panic != [0]:
    failures.append(f"panic_per_run={panic}")
if throughput is None or throughput <= throughput_floor:
    failures.append(f"output_throughput_tps={throughput} <= floor={throughput_floor}")
if mixed_iterations < min_mixed_iterations:
    failures.append(f"mixed_iterations={mixed_iterations} < {min_mixed_iterations}")
if max_p95_itl_ms is not None and (p95_itl_ms is None or p95_itl_ms > max_p95_itl_ms):
    failures.append(f"p95_itl_ms={p95_itl_ms} > {max_p95_itl_ms}")
if kv_failed > max_kv_admission_failed:
    failures.append(f"unified_kv_admission_failed={kv_failed} > {max_kv_admission_failed}")
if capacity_deferred is None:
    failures.append("capacity_deferred_total=missing")
elif int(capacity_deferred) > max_capacity_deferred:
    failures.append(f"capacity_deferred_total={capacity_deferred} > {max_capacity_deferred}")

summary["verdict"] = "keep" if not failures else "reject"
summary["reject_reasons"] = failures
(art / "perf/bench_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
(art / "perf/diagnostic_verdict.json").write_text(
    json.dumps({"verdict": summary["verdict"], "reject_reasons": failures}, indent=2, sort_keys=True) + "\n"
)
PY_SUMMARY
}

run_self_test() {
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  make_case() {
    local name="$1"
    local throughput="$2"
    local kv_failed="$3"
    local capacity_deferred="$4"
    local bench_exit="$5"
    local completed="$6"
    local errored="$7"
    local mixed_iterations="$8"
    local p95_itl_ms="$9"
    local art="$tmp/$name"
    mkdir -p "$art/perf" "$art/server"
    printf '%s\n' "$bench_exit" > "$art/perf/bench.exit"
    python3 - "$art/perf/bench_ferrum_c32_32x1.json" "$throughput" "$completed" "$errored" "$p95_itl_ms" <<'PY_CASE_BENCH'
import json
import sys

path = sys.argv[1]
throughput = float(sys.argv[2])
completed = int(sys.argv[3])
errored = int(sys.argv[4])
p95_itl_ms = float(sys.argv[5])
json.dump(
    {
        "concurrency": 32,
        "n_requests_per_run": 32,
        "n_repeats": 1,
        "output_token_count_source": "usage",
        "completed_per_run": [completed],
        "errored_per_run": [errored],
        "http_500_per_run": [0],
        "panic_per_run": [0],
        "output_throughput_tps": {"mean": throughput},
        "total_throughput_tps": {"mean": throughput * 2},
        "request_throughput_rps": {"mean": 3.5},
        "itl_ms": {"p95": {"mean": p95_itl_ms}},
    },
    open(path, "w"),
)
PY_CASE_BENCH
    python3 - "$art/server/scheduler_trace.jsonl" "$capacity_deferred" "$mixed_iterations" <<'PY_CASE_TRACE'
import json
import sys

path = sys.argv[1]
capacity = int(sys.argv[2])
mixed = int(sys.argv[3])
rows = []
for iteration in range(max(mixed, 1)):
    plan = (
        {"prefill_items": 1, "decode_items": 1, "prefill_tokens": 6, "decode_tokens": 1}
        if iteration < mixed
        else {"prefill_items": 0, "decode_items": 1, "prefill_tokens": 0, "decode_tokens": 1}
    )
    rows.append(
        {
            "event": "scheduler_iteration",
            "iteration": iteration,
            "result": "some_ok",
            "plan": plan,
            "scheduler_before": {
                "active_len": 32,
                "capacity_deferred_total": capacity,
            },
            "scheduler_after_process": {
                "waiting_queue_len": 0,
                "prefill_queue_len": 0,
                "decode_queue_len": 0,
                "active_len": 0,
                "completed_total": 32,
                "failed_total": 0,
                "cancelled_total": 0,
                "admitted_total": 64,
                "capacity_deferred_total": capacity,
                "capacity_blocked_waiting_len": 0,
            },
            "timing_us": {"process": 10000},
        }
    )
open(path, "w").write("".join(json.dumps(row) + "\n" for row in rows))
PY_CASE_TRACE
    python3 - "$art/server/serve.log" "$kv_failed" <<'PY_CASE_LOG'
import sys

path, count = sys.argv[1], int(sys.argv[2])
profile = "\n".join(
    [
        "[unified-prof] iter#1 items=32 prefill=0 decode=32 total=28000us model=27600us decode_post=300us | sample=20 sched=0 stream=280 stop=0 complete=0 (us)",
        "[unified-prof] iter#2 items=33 prefill=1 decode=32 total=42000us model=41000us decode_post=800us | sample=30 sched=0 stream=760 stop=0 complete=10 (us)",
        "[unified-prof] iter#3 items=1 prefill=1 decode=0 total=50000us model=49900us decode_post=0us | sample=0 sched=0 stream=0 stop=0 complete=0 (us)",
    ]
)
open(path, "w").write(("Unified KV admission failed\n" * count) + profile + "\n")
PY_CASE_LOG
    write_diagnostic_summary "$art" "selftest" "600" "13" "32" "64" "25"
  }

  make_case keep 601 13 32 0 32 0 64 25
  make_case reject_throughput 600 13 32 0 32 0 64 25
  make_case reject_kv 601 14 32 0 32 0 64 25
  make_case reject_capacity 601 13 33 0 32 0 64 25
  make_case reject_errors 601 13 32 0 31 1 64 25
  make_case reject_mixed 601 13 32 0 32 0 63 25
  make_case reject_itl 601 13 32 0 32 0 64 25.1

  python3 - "$tmp" <<'PY_ASSERT_SELFTEST'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = {
    "keep": "keep",
    "reject_throughput": "reject",
    "reject_kv": "reject",
    "reject_capacity": "reject",
    "reject_errors": "reject",
    "reject_mixed": "reject",
    "reject_itl": "reject",
}
for name, verdict in expected.items():
    data = json.load(open(root / name / "perf" / "diagnostic_verdict.json"))
    actual = data["verdict"]
    assert actual == verdict, (name, actual, verdict, data)
keep_summary = json.load(open(root / "keep" / "perf" / "bench_summary.json"))
assert keep_summary["log_counts"]["unified_prof_events"] == 3, keep_summary["log_counts"]
assert keep_summary["unified_prof"]["mixed"]["count"] == 1, keep_summary["unified_prof"]
assert keep_summary["unified_prof"]["decode_only"]["count"] == 1, keep_summary["unified_prof"]
assert keep_summary["unified_prof"]["prefill_only"]["count"] == 1, keep_summary["unified_prof"]
print("W3 QWEN35 C32 DIAGNOSTIC SELFTEST PASS")
PY_ASSERT_SELFTEST
}

SHA="${FERRUM_W3_SHA:-9fda11014c17483b0ceca479e3704b3767db0cbe}"
TAG="${FERRUM_W3_TAG:-mixed_prefill_immediate_kv}"
THROUGHPUT_FLOOR="${FERRUM_W3_THROUGHPUT_FLOOR:-458.0662677685816}"
MAX_KV_ADMISSION_FAILED="${FERRUM_W3_MAX_KV_ADMISSION_FAILED:-13}"
MAX_CAPACITY_DEFERRED="${FERRUM_W3_MAX_CAPACITY_DEFERRED:-32}"
MIN_MIXED_ITERATIONS="${FERRUM_W3_MIN_MIXED_ITERATIONS:-0}"
MAX_P95_ITL_MS="${FERRUM_W3_MAX_P95_ITL_MS:-}"
PORT="${FERRUM_W3_PORT:-55994}"
BRANCH="${FERRUM_W3_BRANCH:-goal/w2-w3-release-grade}"
REPO_URL="${FERRUM_W3_REPO_URL:-https://github.com/sizzlecar/ferrum-infer-rs.git}"
REPO="${FERRUM_W3_REPO:-/workspace/ferrum-infer-rs}"
MODEL="${FERRUM_W3_MODEL:-Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}"
ARTIFACT_ROOT="${FERRUM_W3_ARTIFACT_ROOT:-/workspace/artifacts}"
DATASET="${FERRUM_W3_DATASET:-}"
SELF_TEST=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sha)
      SHA="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --throughput-floor)
      THROUGHPUT_FLOOR="$2"
      shift 2
      ;;
    --max-kv-admission-failed)
      MAX_KV_ADMISSION_FAILED="$2"
      shift 2
      ;;
    --max-capacity-deferred)
      MAX_CAPACITY_DEFERRED="$2"
      shift 2
      ;;
    --min-mixed-iterations)
      MIN_MIXED_ITERATIONS="$2"
      shift 2
      ;;
    --max-p95-itl-ms)
      MAX_P95_ITL_MS="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --repo)
      REPO="$2"
      shift 2
      ;;
    --artifact-root)
      ARTIFACT_ROOT="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --self-test)
      SELF_TEST=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$SELF_TEST" == true ]]; then
  run_self_test
  exit 0
fi

if [[ -z "$DATASET" ]]; then
  DATASET="$REPO/docs/goals/model-coverage-2026-06-12/artifacts/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl"
fi

export DEBIAN_FRONTEND=noninteractive
export PATH=/root/.cargo/bin:/root/.rustup/toolchains/1.96.0-x86_64-unknown-linux-gnu/bin:$PATH
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export RUST_BACKTRACE=1
export FERRUM_SCHED_TRACE=1
export FERRUM_UNIFIED_POST_PROF=1

SHORT_SHA="${SHA:0:8}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ART="$ARTIFACT_ROOT/w3_qwen35_${TAG}_c32_${SHORT_SHA}_${STAMP}"
mkdir -p "$ART"/{env,hardware,logs,run,server,perf,scripts,prefetch}
cp "$0" "$ART/scripts/$(basename "$0")"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$ART/logs/lane.log"
}

cleanup() {
  set +e
  if [[ -f "$ART/server/server.pid" ]]; then
    kill "$(cat "$ART/server/server.pid")" 2>/dev/null || true
  fi
  pkill -P $$ 2>/dev/null || true
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits > "$ART/logs/nvidia_after.csv" 2>/dev/null || true
}
trap cleanup EXIT

compress_large_logs() {
  find "$ART" -type f \( -name '*.log' -o -name '*.jsonl' \) \
    ! -name '*.gz' -size +1M -print0 | xargs -0 -r gzip -f
}

log "lane_start art=$ART sha=$SHA release_evidence=false no_live_vllm=true"
{
  date -u
  hostname
  uname -a
  nvidia-smi || true
  nvcc --version || true
  df -h /workspace || true
} > "$ART/logs/preflight.log" 2>&1
nvidia-smi > "$ART/hardware/nvidia_smi_before.txt" 2>&1 || true

log bootstrap_start
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > "$ART/logs/rustup-init.sh"
  sh "$ART/logs/rustup-init.sh" -y --default-toolchain 1.96.0 \
    > "$ART/logs/rustup-install.log" 2>&1
else
  rustup toolchain install 1.96.0 > "$ART/logs/rustup-toolchain.log" 2>&1 || true
  rustup default 1.96.0 > "$ART/logs/rustup-default.log" 2>&1 || true
fi
export PATH=/root/.cargo/bin:/root/.rustup/toolchains/1.96.0-x86_64-unknown-linux-gnu/bin:$PATH
python3 -m pip install -q --upgrade huggingface_hub hf_xet > "$ART/logs/pip-hf.log" 2>&1 || true
rustc --version > "$ART/env/rustc.txt" 2>&1 || true
cargo --version > "$ART/env/cargo.txt" 2>&1 || true
log bootstrap_done

log git_sync_start
if [[ ! -d "$REPO/.git" ]]; then
  git clone "$REPO_URL" "$REPO" > "$ART/logs/git-clone.log" 2>&1
fi
cd "$REPO"
git remote set-url origin "$REPO_URL" || true
git fetch origin "$BRANCH" --prune > "$ART/logs/git-fetch.log" 2>&1
git checkout "$BRANCH" > "$ART/logs/git-checkout-branch.log" 2>&1 \
  || git checkout -B "$BRANCH" "origin/$BRANCH" > "$ART/logs/git-checkout-branch.log" 2>&1
git pull --rebase --autostash origin "$BRANCH" > "$ART/logs/git-pull.log" 2>&1
git checkout --detach "$SHA" > "$ART/logs/git-checkout-detach.log" 2>&1
git rev-parse HEAD > "$ART/env/git_sha.txt"
if [[ "$(cat "$ART/env/git_sha.txt")" != "$SHA" ]]; then
  log git_sha_mismatch
  exit 22
fi
git status --short > "$ART/env/git_status_short.txt"
log git_sync_done

if [[ ! -f "$DATASET" ]]; then
  log dataset_missing path="$DATASET"
  exit 23
fi

log hf_prefetch_start
HF_HOME="$HF_HOME" HF_XET_HIGH_PERFORMANCE=1 python3 - <<'PY_HF' > "$ART/prefetch/hf-prefetch.log" 2>&1 &
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", cache_dir="/workspace/hf-cache/hub")
PY_HF
PREFETCH_PID=$!

log cargo_check_start
set +e
cargo check -p ferrum-kernels -p ferrum-models -p ferrum-cli \
  > "$ART/logs/cargo-check.stdout.log" 2> "$ART/logs/cargo-check.stderr.log"
cargo_check_rc=$?
set -e
echo "$cargo_check_rc" > "$ART/env/cargo-check.exit"
if [[ "$(cat "$ART/env/cargo-check.exit")" != 0 ]]; then
  log cargo_check_failed
  wait "$PREFETCH_PID" || true
  exit 20
fi
log cargo_check_done

log release_build_start
set +e
cargo build --release -p ferrum-cli --bin ferrum \
  --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
  > "$ART/logs/build.stdout.log" 2> "$ART/logs/build.stderr.log"
build_rc=$?
set -e
echo "$build_rc" > "$ART/env/build.exit"
if [[ "$(cat "$ART/env/build.exit")" != 0 ]]; then
  log release_build_failed
  wait "$PREFETCH_PID" || true
  exit 21
fi
sha256sum target/release/ferrum > "$ART/env/ferrum.sha256"
log release_build_done

log hf_prefetch_wait_start
set +e
wait "$PREFETCH_PID"
prefetch_rc=$?
set -e
echo "$prefetch_rc" > "$ART/prefetch/hf-prefetch.exit"
if [[ "$prefetch_rc" != 0 ]]; then
  log hf_prefetch_failed
  exit 24
fi
TOKENIZER="$(find "$HF_HOME/hub" -path '*models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4*/snapshots/*/tokenizer.json' -print | head -n 1 | xargs -r dirname)"
if [[ -z "$TOKENIZER" || ! -f "$TOKENIZER/tokenizer.json" ]]; then
  log tokenizer_missing
  exit 25
fi
printf '%s\n' "$TOKENIZER" > "$ART/env/tokenizer_path.txt"
log hf_prefetch_done tokenizer="$TOKENIZER"

log ferrum_run_smoke_start
RUN_CMD=(target/release/ferrum run "$MODEL"
  --backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90
  --max-num-seqs 32 --kv-capacity 512 --max-num-batched-tokens 192
  --temperature 0 --max-tokens 8 --output-format jsonl
  --prompt 'What is 2+3? Answer with only the number.'
  --effective-config-json "$ART/run/effective_config.json"
  --decision-trace-jsonl "$ART/run/decision_trace.jsonl")
printf '%q ' "${RUN_CMD[@]}" > "$ART/run/run.command.txt"
printf '\n' >> "$ART/run/run.command.txt"
set +e
"${RUN_CMD[@]}" > "$ART/run/run.stdout.jsonl" 2> "$ART/run/run.stderr.log"
run_rc=$?
set -e
echo "$run_rc" > "$ART/run/run.exit"
if [[ "$(cat "$ART/run/run.exit")" != 0 ]]; then
  log ferrum_run_failed
  exit 30
fi
python3 - "$ART/run/effective_config.json" <<'PY_RUN_ASSERT' > "$ART/run/effective_config.assert.txt"
import json, sys
data = json.load(open(sys.argv[1]))
assert data.get("selected_max_sequences") == 32, data.get("selected_max_sequences")
assert data.get("selected_recurrent_state_max_slots") == 32, data.get("selected_recurrent_state_max_slots")
assert data.get("selected_admission_limit") == 32, data.get("selected_admission_limit")
print("ok")
PY_RUN_ASSERT
log ferrum_run_smoke_done

log serve_start
SERVE_CMD=(target/release/ferrum serve "$MODEL"
  --host 127.0.0.1 --port "$PORT"
  --backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90
  --max-num-seqs 32 --kv-capacity 512 --max-num-batched-tokens 192
  --scheduler-prefill-first-until-active 32 --scheduler-prefill-step-chunk 6
  --effective-config-json "$ART/server/effective_config.json"
  --profile-jsonl "$ART/server/profile.jsonl"
  --decision-trace-jsonl "$ART/server/decision_trace.jsonl"
  --scheduler-trace-jsonl "$ART/server/scheduler_trace.jsonl")
printf '%q ' "${SERVE_CMD[@]}" > "$ART/server/serve.command.txt"
printf '\n' >> "$ART/server/serve.command.txt"
"${SERVE_CMD[@]}" > "$ART/server/serve.stdout.log" 2> "$ART/server/serve.log" &
echo $! > "$ART/server/server.pid"
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" \
    > "$ART/server/models.json" 2> "$ART/server/models.curl.err"; then
    break
  fi
  if ! kill -0 "$(cat "$ART/server/server.pid")" 2>/dev/null; then
    log serve_exited_before_ready
    exit 31
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" \
  > "$ART/server/models.json" 2> "$ART/server/models.curl.err"
python3 - "$ART/server/effective_config.json" <<'PY_SERVE_ASSERT' > "$ART/server/effective_config.assert.txt"
import json, sys
data = json.load(open(sys.argv[1]))
assert data.get("selected_max_sequences") == 32, data.get("selected_max_sequences")
assert data.get("selected_recurrent_state_max_slots") == 32, data.get("selected_recurrent_state_max_slots")
assert data.get("selected_admission_limit") == 32, data.get("selected_admission_limit")
print("ok")
PY_SERVE_ASSERT
log serve_ready

python3 - <<'PY_PAYLOAD' > "$ART/server/chat_smoke_payload.json"
import json
print(json.dumps({
    "model": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
    "messages": [{"role": "user", "content": "What is 2+3? Answer with only the number."}],
    "temperature": 0,
    "max_tokens": 8,
    "stream": False,
}))
PY_PAYLOAD
curl -fsS -H 'Content-Type: application/json' \
  --data-binary "@$ART/server/chat_smoke_payload.json" \
  "http://127.0.0.1:${PORT}/v1/chat/completions" \
  > "$ART/server/chat_smoke_response.json" 2> "$ART/server/chat_smoke.curl.err"
python3 - "$ART/server/chat_smoke_response.json" <<'PY_CHAT_ASSERT'
import json, sys
data = json.load(open(sys.argv[1]))
content = data["choices"][0]["message"]["content"].strip()
assert content == "5", content
PY_CHAT_ASSERT
log serve_chat_smoke_done

log bench_c32_start
BENCH_CMD=(target/release/ferrum bench-serve
  --base-url "http://127.0.0.1:${PORT}"
  --model "$MODEL"
  --tokenizer "$TOKENIZER"
  --dataset sharegpt
  --sharegpt-path "$DATASET"
  --random-output-len 128
  --ignore-eos
  --concurrency 32
  --num-prompts 32
  --warmup-requests 4
  --n-repeats 1
  --fail-on-error
  --seed 9271
  --timeout 600
  --out "$ART/perf/bench_ferrum_c32_32x1.json")
printf '%q ' "${BENCH_CMD[@]}" > "$ART/perf/bench-ferrum-c32.command.txt"
printf '\n' >> "$ART/perf/bench-ferrum-c32.command.txt"
set +e
"${BENCH_CMD[@]}" > "$ART/perf/bench.stdout.log" 2> "$ART/perf/bench.stderr.log"
bench_rc=$?
set -e
echo "$bench_rc" > "$ART/perf/bench.exit"

write_diagnostic_summary \
  "$ART" "$SHA" "$THROUGHPUT_FLOOR" "$MAX_KV_ADMISSION_FAILED" "$MAX_CAPACITY_DEFERRED" \
  "$MIN_MIXED_ITERATIONS" "$MAX_P95_ITL_MS"

verdict="$(python3 - <<'PY_VERDICT' "$ART/perf/diagnostic_verdict.json"
import json, sys
print(json.load(open(sys.argv[1]))["verdict"])
PY_VERDICT
)"

compress_large_logs
if [[ "$bench_rc" != 0 ]]; then
  log "FERRUM W3 QWEN35 C32 DIAG REJECT: $ART"
  exit 50
fi
if [[ "$verdict" != "keep" ]]; then
  log "FERRUM W3 QWEN35 C32 DIAG REJECT: $ART"
  exit 60
fi

log "FERRUM W3 QWEN35 C32 DIAG KEEP: $ART"
