#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

LANE="${1:-}"
if [[ -z "$LANE" ]]; then
  echo "usage: scripts/release/g0_source_gate.sh {unit|metal|cuda-smoke|cuda-full|cuda-llama-dense|cuda-llama33-70b-4bit-2x4090-smoke|cuda-llama33-70b-4bit-2x4090|all-source} [OUT_ROOT]" >&2
  exit 2
fi
OUT_ROOT="${2:-docs/release/g0/source-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_ROOT"

pass() { echo "G0 SOURCE ${1} PASS: $OUT_ROOT"; }

run_unit() {
  local unit_bounded_root="$OUT_ROOT/unit-bounded"
  local unit_receipt="$unit_bounded_root/receipt.json"
  local unit_stdout="$unit_bounded_root/stdout.log"
  local unit_stderr="$unit_bounded_root/stderr.log"
  mkdir -p "$unit_bounded_root"
  python3 scripts/release/bounded_command.py \
    --receipt "$unit_receipt" \
    --stdout-log "$unit_stdout" \
    --stderr-log "$unit_stderr" \
    --cwd "$PWD" \
    --wall-timeout-seconds 1800 \
    --max-processes 16 \
    --max-group-threads 96 \
    --max-per-process-threads 48 \
    --sample-interval-seconds 0.05 \
    --max-sampling-errors 3 \
    --term-grace-seconds 1 \
    -- env PYTHONDONTWRITEBYTECODE=1 CARGO_BUILD_JOBS=2 RUST_TEST_THREADS=1 \
      cargo test --workspace --all-targets
  python3 - "$OUT_ROOT/release-scripts-pycompile-cache" \
    scripts/metal_readme_regression.py \
    scripts/release/inventory_tree.py \
    scripts/release/validate_metal_readme_regression.py \
    scripts/release/release_binary_gate.py \
    scripts/release/g0_release_summary.py \
    scripts/release/g0_cuda_llama_dense_gate.py \
    scripts/release/g0_cuda_llama33_70b_4bit_2x4090_gate.py \
    scripts/release/backend_runtime_preset_goal_gate.py \
    scripts/release/llama33_70b_4bit_2x4090_goal_gate.py \
    scripts/release/layer_split_perf_goal_gate.py \
    scripts/release/run_layer_split_perf_goal.py \
    scripts/release/backend_boundary_audit.py \
    scripts/release/backend_runtime_preset_snapshot.py \
    scripts/release/openai_concurrency_quality_regression.py \
    scripts/release/openai_tool_call_regression.py \
    scripts/release/runtime_vnext_baseline_gate.py \
    scripts/release/runtime_vnext_inventory.py \
    scripts/release/runtime_vnext_model_resolver.py \
    scripts/release/runtime_vnext_hardware_probe.py \
    scripts/release/runtime_vnext_build_timing.py \
    scripts/release/runtime_vnext_baseline_scenarios.py \
    scripts/release/runtime_vnext_expectation_amendment.py \
    scripts/release/runtime_vnext_blocked_lane.py \
    scripts/release/runtime_vnext_resource_sampler.py \
    scripts/release/runtime_vnext_performance_collector.py \
    scripts/release/runtime_vnext_g00a_checkpoint.py \
    scripts/release/runtime_vnext_g00_orchestrator.py \
    scripts/release/runtime_vnext_historical_corpus.py \
    scripts/release/runtime_vnext_historical_replay.py \
    scripts/release/runtime_vnext_historical_capture.py \
    scripts/release/runtime_vnext_g01a_checkpoint.py \
    scripts/release/bounded_command.py \
    scripts/release/run_gate.py \
    scripts/release/run_scenarios.py \
    scripts/release/selftest_g0_validators.py \
    scripts/release/selftest_g1_g3_g4_release_regression.py \
    scripts/release/validate_release_completion_manifest.py <<'PY' 2>&1 | tee "$OUT_ROOT/release-scripts-pycompile.log"
import pathlib
import py_compile
import sys

cache_dir = pathlib.Path(sys.argv[1])
cache_dir.mkdir(parents=True, exist_ok=True)
for raw in sys.argv[2:]:
    cfile = cache_dir / (raw.replace("/", "__") + ".pyc")
    py_compile.compile(raw, cfile=str(cfile), doraise=True)
    print(f"compiled {raw}")
PY
  bash -n scripts/release/g0_source_gate.sh | tee "$OUT_ROOT/g0-source-bashn.log"
  python3 scripts/release/selftest_g0_validators.py | tee "$OUT_ROOT/g0-validator-selftest.log"
  python3 scripts/release/selftest_g1_g3_g4_release_regression.py | tee "$OUT_ROOT/g1-g3-g4-validator-selftest.log"
  python3 - "$OUT_ROOT" "$unit_receipt" "$unit_stdout" "$unit_stderr" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

out_raw = sys.argv[1]
out_root = pathlib.Path(out_raw).resolve()
receipt_path = pathlib.Path(sys.argv[2]).resolve()
stdout_path = pathlib.Path(sys.argv[3]).resolve()
stderr_path = pathlib.Path(sys.argv[4]).resolve()
expected_command = [
    "env",
    "PYTHONDONTWRITEBYTECODE=1",
    "CARGO_BUILD_JOBS=2",
    "RUST_TEST_THREADS=1",
    "cargo",
    "test",
    "--workspace",
    "--all-targets",
]
expected_limits = {
    "wall_timeout_seconds": 1800.0,
    "max_processes": 16,
    "max_group_threads": 96,
    "max_per_process_threads": 48,
    "sample_interval_seconds": 0.05,
    "max_sampling_errors": 3,
    "term_grace_seconds": 1.0,
}
expected_fields = {
    "schema",
    "command",
    "cwd",
    "pid",
    "pgid",
    "limits",
    "peaks",
    "started_at",
    "ended_at",
    "duration_seconds",
    "reason",
    "rc",
    "status",
    "successful_samples",
    "sampling_error_count",
    "sampling_errors",
    "violation",
    "termination",
    "cleanup",
    "stdout",
    "stderr",
}


def require(condition, message):
    if not condition:
        raise SystemExit(f"G0 source unit bounded receipt ERROR: {message}")


def identity(path):
    payload = path.read_bytes()
    return {
        "path": path.relative_to(out_root).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
require(isinstance(receipt, dict) and set(receipt) == expected_fields, "receipt field set mismatch")
require(receipt.get("schema") == "ferrum.bounded-command-receipt.v1", "receipt schema mismatch")
require(receipt.get("command") == expected_command, "receipt command mismatch")
require(pathlib.Path(receipt.get("cwd", "")).resolve() == pathlib.Path.cwd().resolve(), "receipt cwd mismatch")
pid = receipt.get("pid")
require(isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 and receipt.get("pgid") == pid, "receipt pid/pgid mismatch")
limits = receipt.get("limits")
require(isinstance(limits, dict) and set(limits) == set(expected_limits), "receipt limit field set mismatch")
require(all(isinstance(limits.get(key), (int, float)) and not isinstance(limits[key], bool) and limits[key] == value for key, value in expected_limits.items()), "receipt limits mismatch")
rc = receipt.get("rc")
require(receipt.get("status") == "pass" and receipt.get("reason") == "command_completed" and isinstance(rc, int) and not isinstance(rc, bool) and rc == 0, "receipt status mismatch")
require(isinstance(receipt.get("successful_samples"), int) and not isinstance(receipt["successful_samples"], bool) and receipt["successful_samples"] >= 1, "receipt has no successful sample")
sampling_error_count = receipt.get("sampling_error_count")
require(isinstance(sampling_error_count, int) and not isinstance(sampling_error_count, bool) and sampling_error_count == 0 and receipt.get("sampling_errors") == [], "receipt contains sampling errors")
require(receipt.get("violation") is None, "receipt contains a resource violation")
require(receipt.get("termination") == {"signals": [], "errors": []}, "receipt termination is not clean")
cleanup = receipt.get("cleanup")
require(isinstance(cleanup, dict) and set(cleanup) == {"process_group_gone"} and cleanup.get("process_group_gone") is True, "receipt process group cleanup failed")
require(isinstance(receipt.get("started_at"), str) and receipt["started_at"] and isinstance(receipt.get("ended_at"), str) and receipt["ended_at"], "receipt timestamps missing")
require(isinstance(receipt.get("duration_seconds"), (int, float)) and not isinstance(receipt["duration_seconds"], bool) and receipt["duration_seconds"] >= 0, "receipt duration invalid")
peaks = receipt.get("peaks")
require(isinstance(peaks, dict) and set(peaks) == {"processes", "group_threads", "per_process_threads", "per_process_threads_pid"}, "receipt peaks field set mismatch")
for key in ("processes", "group_threads", "per_process_threads"):
    require(isinstance(peaks.get(key), int) and not isinstance(peaks[key], bool) and peaks[key] >= 1, f"receipt {key} peak invalid")
require(isinstance(peaks.get("per_process_threads_pid"), int) and not isinstance(peaks["per_process_threads_pid"], bool) and peaks["per_process_threads_pid"] > 0, "receipt peak pid invalid")
require(peaks["processes"] <= expected_limits["max_processes"], "receipt process peak exceeded")
require(peaks["group_threads"] <= expected_limits["max_group_threads"], "receipt group thread peak exceeded")
require(peaks["per_process_threads"] <= expected_limits["max_per_process_threads"], "receipt per-process thread peak exceeded")
require(peaks["group_threads"] >= peaks["processes"] and peaks["group_threads"] >= peaks["per_process_threads"], "receipt peak relationship invalid")
for stream, path in (("stdout", stdout_path), ("stderr", stderr_path)):
    row = receipt.get(stream)
    require(isinstance(row, dict) and set(row) == {"path", "sha256", "size_bytes"}, f"receipt {stream} identity invalid")
    payload = path.read_bytes()
    require(pathlib.Path(row["path"]).resolve() == path, f"receipt {stream} path mismatch")
    require(row["sha256"] == hashlib.sha256(payload).hexdigest() and isinstance(row["size_bytes"], int) and not isinstance(row["size_bytes"], bool) and row["size_bytes"] == len(payload), f"receipt {stream} content mismatch")

stdout_text = stdout_path.read_text(encoding="utf-8")
stderr_text = stderr_path.read_text(encoding="utf-8")
bench_cases = (
    "single_request/tokens/1",
    "single_request/tokens/5",
    "single_request/tokens/10",
    "single_request/tokens/20",
    "single_request/tokens/50",
    "concurrent_throughput/concurrency/1",
    "concurrent_throughput/concurrency/2",
    "concurrent_throughput/concurrency/4",
    "concurrent_throughput/concurrency/8",
    "concurrent_throughput/concurrency/16",
    "scheduling_overhead/single_request_overhead",
    "scheduling_overhead/sequential_10_requests",
)
require("Running benches/engine_bench.rs" in stderr_text, "engine_bench execution witness missing")
for bench_case in bench_cases:
    require(f"Testing {bench_case}\nSuccess" in stdout_text, f"engine_bench case witness missing: {bench_case}")

manifest = {
    "schema_version": 1,
    "artifact_type": "g0_source_unit_bounded_gate",
    "status": "pass",
    "lane": "unit",
    "pass_line": f"G0 SOURCE unit PASS: {out_raw}",
    "command": expected_command,
    "env_overrides": {
        "PYTHONDONTWRITEBYTECODE": "1",
        "CARGO_BUILD_JOBS": "2",
        "RUST_TEST_THREADS": "1",
    },
    "receipt_schema": "ferrum.bounded-command-receipt.v1",
    "limits": expected_limits,
    "peaks": peaks,
    "cleanup": {"process_group_gone": True},
    "bounded_receipt": identity(receipt_path),
    "stdout_log": identity(stdout_path),
    "stderr_log": identity(stderr_path),
}
destination = out_root / "unit.gate.json"
temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, destination)
print(f"G0 SOURCE UNIT BOUNDED RECEIPT PASS: {receipt_path}")
PY
  pass unit
}

run_metal() {
  cargo build --release -p ferrum-cli --features metal --tests | tee "$OUT_ROOT/metal-build.log"
  local metal_out="$OUT_ROOT/metal-readme"
  python3 scripts/metal_readme_regression.py --out "$metal_out" --ferrum-bin ./target/release/ferrum | tee "$OUT_ROOT/metal-runner.log"
  python3 scripts/release/validate_metal_readme_regression.py "$metal_out" | tee "$OUT_ROOT/metal-validator.log"
  echo '{"status":"pass","lane":"metal","artifact":"metal-readme"}' > "$OUT_ROOT/metal.gate.json"
  pass metal
}

cuda_build() {
  cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2 | tee "$OUT_ROOT/cuda-build.log"
}

run_cuda_template() {
  local template="$1"
  local label="$2"
  local config_list="$OUT_ROOT/${label}-configs.txt"
  local change_type
  change_type="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("validation",{}).get("change_type","opt_in_experiment"))' "$template")"
  python3 - "$template" "$OUT_ROOT" "$config_list" <<'PY'
import json, pathlib, sys
src = pathlib.Path(sys.argv[1])
out_root = pathlib.Path(sys.argv[2])
config_list = pathlib.Path(sys.argv[3])
base = json.load(open(src))
cells = base.pop("concurrency_cells")
paths = []
for i, c in enumerate(cells):
    cfg = dict(base)
    cfg["concurrency"] = c
    cfg["out_root"] = str(out_root / f"c{c}")
    cfg["port_base"] = int(base.get("port_base", 19000)) + i * 10
    cfg["validation"] = dict(base.get("validation", {}))
    cfg["validation"]["required_concurrency_cells"] = [c]
    for j, case in enumerate(cfg.get("cases", [])):
        case["port"] = cfg["port_base"] + j
    path = out_root / f"{src.stem}-c{c}.json"
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    paths.append(path)
    print(path)
config_list.write_text("".join(f"{path}\n" for path in paths))
PY
  while IFS= read -r cfg; do
    python3 scripts/m3_ab_runner.py --config "$cfg"
    art=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["out_root"])' "$cfg")
    python3 scripts/m3_validate_runner_artifact.py "$art"
  done < "$config_list"
  python3 scripts/m3_collect_allcell_runner_artifacts.py "$OUT_ROOT" \
    --baseline-case baseline \
    --candidate candidate \
    --change-type "$change_type"
  python3 scripts/m3_validate_runner_artifact.py "$OUT_ROOT"
  echo "{\"status\":\"pass\",\"lane\":\"$label\"}" > "$OUT_ROOT/$label.gate.json"
  pass "$label"
}

run_cuda_llama_dense() {
  python3 scripts/release/g0_cuda_llama_dense_gate.py \
    --config scripts/release/configs/g0_cuda4090_llama_dense.json \
    --out "$OUT_ROOT" \
    --ferrum-bin ./target/release/ferrum | tee "$OUT_ROOT/cuda-llama-dense.log"
  echo '{"status":"pass","lane":"g0_cuda4090_llama_dense"}' > "$OUT_ROOT/g0_cuda4090_llama_dense.gate.json"
  pass g0_cuda4090_llama_dense
}

run_cuda_llama33_70b_4bit_2x4090() {
  python3 scripts/release/g0_cuda_llama33_70b_4bit_2x4090_gate.py \
    --config scripts/release/configs/g0_cuda2x4090_llama33_70b_4bit.json \
    --out "$OUT_ROOT" \
    --ferrum-bin ./target/release/ferrum | tee "$OUT_ROOT/cuda-llama33-70b-4bit-2x4090.log"
  pass g0_cuda2x4090_llama33_70b_4bit
}

run_cuda_llama33_70b_4bit_2x4090_smoke() {
  python3 scripts/release/g0_cuda_llama33_70b_4bit_2x4090_gate.py \
    --config scripts/release/configs/g0_cuda2x4090_llama33_70b_4bit_smoke.json \
    --out "$OUT_ROOT" \
    --ferrum-bin ./target/release/ferrum \
    --lane-name g0_cuda2x4090_llama33_70b_4bit_smoke \
    | tee "$OUT_ROOT/cuda-llama33-70b-4bit-2x4090-smoke.log"
  pass g0_cuda2x4090_llama33_70b_4bit_smoke
}

case "$LANE" in
  unit) run_unit ;;
  metal) run_metal ;;
  cuda-smoke) cuda_build; run_cuda_template scripts/release/configs/g0_cuda4090_smoke.json g0_cuda4090_smoke ;;
  cuda-full) cuda_build; run_cuda_template scripts/release/configs/g0_cuda4090_full.json g0_cuda4090_full ;;
  cuda-llama-dense) cuda_build; run_cuda_llama_dense ;;
  cuda-llama33-70b-4bit-2x4090-smoke) cuda_build; run_cuda_llama33_70b_4bit_2x4090_smoke ;;
  cuda-llama33-70b-4bit-2x4090) cuda_build; run_cuda_llama33_70b_4bit_2x4090 ;;
  all-source)
    run_unit
    if [[ "$(uname -s)" == "Darwin" ]]; then
      run_metal
    else
      echo "G0 SOURCE all-source: skipping metal on non-macOS"
    fi
    echo '{"status":"pass","lane":"all-source"}' > "$OUT_ROOT/all-source.gate.json"
    pass all-source
    ;;
  *) echo "unknown lane: $LANE" >&2; exit 2 ;;
esac
