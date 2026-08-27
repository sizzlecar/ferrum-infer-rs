#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

LANE="${1:-}"
if [[ -z "$LANE" ]]; then
  echo "usage: scripts/release/g0_source_gate.sh {unit|metal|cuda-smoke|cuda-full|cuda-llama-dense|cuda-llama33-70b-4bit-2x4090-smoke|cuda-llama33-70b-4bit-2x4090|all-source} [OUT_ROOT] [NATIVE_OPERATOR_SET_LOCK]" >&2
  exit 2
fi
OUT_ROOT="${2:-docs/release/g0/source-$(date +%Y%m%d-%H%M%S)}"
NATIVE_OPERATOR_SET_LOCK="${3:-}"
mkdir -p "$OUT_ROOT"

pass() { echo "G0 SOURCE ${1} PASS: $OUT_ROOT"; }

run_unit() {
  local unit_bounded_root="$OUT_ROOT/unit-bounded"
  local unit_receipt="$unit_bounded_root/receipt.json"
  local unit_stdout="$unit_bounded_root/stdout.log"
  local unit_stderr="$unit_bounded_root/stderr.log"
  local unit_source="$unit_bounded_root/source.before.json"
  mkdir -p "$unit_bounded_root"
  python3 - "$unit_source" <<'PY'
import json
import os
import pathlib
import re
import subprocess
import sys

destination = pathlib.Path(sys.argv[1]).resolve()


def git_output(*args):
    proc = subprocess.run(
        ["git", *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"G0 source unit checkout receipt ERROR: git {' '.join(args)} "
            f"failed rc={proc.returncode}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


git_sha = git_output("rev-parse", "HEAD")
git_tree_sha = git_output("rev-parse", "HEAD^{tree}")
status_short = [
    line
    for line in git_output(
        "status",
        "--short",
        "--untracked-files=all",
    ).splitlines()
    if line.strip()
]
if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
    raise SystemExit("G0 source unit checkout receipt ERROR: invalid HEAD SHA")
if re.fullmatch(r"[0-9a-f]{40}", git_tree_sha) is None:
    raise SystemExit("G0 source unit checkout receipt ERROR: invalid tree SHA")
receipt = {
    "schema_version": 1,
    "artifact_type": "g0_source_checkout_receipt",
    "git_sha": git_sha,
    "git_tree_sha": git_tree_sha,
    "dirty_status": {
        "is_dirty": bool(status_short),
        "status_short": status_short,
    },
}
temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, destination)
PY
  python3 scripts/release/bounded_command.py \
    --receipt "$unit_receipt" \
    --stdout-log "$unit_stdout" \
    --stderr-log "$unit_stderr" \
    --cwd "$PWD" \
    --wall-timeout-seconds 1800 \
    --max-processes 128 \
    --max-group-threads 1024 \
    --max-per-process-threads 256 \
    --sample-interval-seconds 0.05 \
    --max-sampling-errors 3 \
    --term-grace-seconds 1 \
    -- env PYTHONDONTWRITEBYTECODE=1 CARGO_BUILD_JOBS=8 RUST_TEST_THREADS=1 \
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
    scripts/release/native_operator_source_bundle.py \
    scripts/release/runtime_vnext_model_resolver.py \
    scripts/release/runtime_vnext_hardware_probe.py \
    scripts/release/runtime_vnext_build_timing.py \
    scripts/release/runtime_vnext_baseline_scenarios.py \
    scripts/release/runtime_vnext_blocked_lane.py \
    scripts/release/runtime_vnext_resource_sampler.py \
    scripts/release/runtime_vnext_performance_collector.py \
    scripts/release/runtime_vnext_g00a_checkpoint.py \
    scripts/release/runtime_vnext_historical_corpus.py \
    scripts/release/runtime_vnext_historical_replay.py \
    scripts/release/runtime_vnext_g01a_checkpoint.py \
    scripts/release/runtime_vnext_plan_reference.py \
    scripts/release/runtime_vnext_cuda_correctness_build.py \
    scripts/release/runtime_vnext_g07a_checkpoint.py \
    scripts/release/runtime_vnext_numerical_tolerances.py \
    scripts/release/runtime_vnext_checkpoint_artifact.py \
    scripts/release/qwen35_gguf_linear_attention_reference.py \
    scripts/release/qwen35_gguf_full_attention_reference.py \
    scripts/release/qwen35_gguf_model_reference.py \
    scripts/release/runtime_vnext_qwen35_layer_reference_gate.py \
    scripts/release/runtime_vnext_qwen35_full_attention_gate.py \
    scripts/release/runtime_vnext_qwen35_model_reference_gate.py \
    scripts/release/bounded_command.py \
    scripts/release/run_gate.py \
    scripts/release/run_scenarios.py \
    scripts/release/runtime_vnext_crates_io_release.py \
    scripts/release/runtime_vnext_g0_llama_sampled_execution.py \
    scripts/release/runtime_vnext_g08b_cuda_matrix_prepare.py \
    scripts/release/runtime_vnext_goal_gate.py \
    scripts/release/runtime_vnext_homebrew_release.py \
    scripts/release/runtime_vnext_prepromotion_bundle.py \
    scripts/release/runtime_vnext_release_workflow_policy.py \
    scripts/release/runtime_vnext_r2_ferrum_collector.py \
    scripts/release/runtime_vnext_sampled_final.py \
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
  python3 - "$OUT_ROOT" "$unit_receipt" "$unit_stdout" "$unit_stderr" "$unit_source" <<'PY'
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

out_raw = sys.argv[1]
out_root = pathlib.Path(out_raw).resolve()
receipt_path = pathlib.Path(sys.argv[2]).resolve()
stdout_path = pathlib.Path(sys.argv[3]).resolve()
stderr_path = pathlib.Path(sys.argv[4]).resolve()
source_path = pathlib.Path(sys.argv[5]).resolve()
expected_command = [
    "env",
    "PYTHONDONTWRITEBYTECODE=1",
    "CARGO_BUILD_JOBS=8",
    "RUST_TEST_THREADS=1",
    "cargo",
    "test",
    "--workspace",
    "--all-targets",
]
expected_limits = {
    "wall_timeout_seconds": 1800.0,
    "max_processes": 128,
    "max_group_threads": 1024,
    "max_per_process_threads": 256,
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


def git_output(*args):
    proc = subprocess.run(
        ["git", *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    require(
        proc.returncode == 0,
        f"git {' '.join(args)} failed rc={proc.returncode}: {proc.stderr.strip()}",
    )
    return proc.stdout.strip()


source_receipt = json.loads(source_path.read_text(encoding="utf-8"))
require(
    isinstance(source_receipt, dict)
    and set(source_receipt)
    == {
        "schema_version",
        "artifact_type",
        "git_sha",
        "git_tree_sha",
        "dirty_status",
    }
    and source_receipt.get("schema_version") == 1
    and source_receipt.get("artifact_type") == "g0_source_checkout_receipt",
    "source checkout receipt schema mismatch",
)
source_git_sha = source_receipt.get("git_sha")
source_tree_sha = source_receipt.get("git_tree_sha")
source_dirty = source_receipt.get("dirty_status")
require(
    isinstance(source_git_sha, str)
    and re.fullmatch(r"[0-9a-f]{40}", source_git_sha) is not None
    and isinstance(source_tree_sha, str)
    and re.fullmatch(r"[0-9a-f]{40}", source_tree_sha) is not None,
    "source checkout receipt SHA mismatch",
)
require(
    isinstance(source_dirty, dict)
    and set(source_dirty) == {"is_dirty", "status_short"}
    and isinstance(source_dirty.get("is_dirty"), bool)
    and isinstance(source_dirty.get("status_short"), list)
    and all(
        isinstance(line, str) and line
        for line in source_dirty["status_short"]
    )
    and source_dirty["is_dirty"] == bool(source_dirty["status_short"]),
    "source checkout dirty status mismatch",
)
current_status = [
    line
    for line in git_output(
        "status",
        "--short",
        "--untracked-files=all",
    ).splitlines()
    if line.strip()
]
require(
    git_output("rev-parse", "HEAD") == source_git_sha
    and git_output("rev-parse", "HEAD^{tree}") == source_tree_sha
    and current_status == source_dirty["status_short"],
    "source checkout changed while the unit gate was running",
)


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
        "CARGO_BUILD_JOBS": "8",
        "RUST_TEST_THREADS": "1",
    },
    "receipt_schema": "ferrum.bounded-command-receipt.v1",
    "limits": expected_limits,
    "peaks": peaks,
    "cleanup": {"process_group_gone": True},
    "source": {
        "git_sha": source_git_sha,
        "git_tree_sha": source_tree_sha,
        "dirty_status": source_dirty,
    },
    "source_receipt": identity(source_path),
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
  if [[ -z "$NATIVE_OPERATOR_SET_LOCK" || ! -f "$NATIVE_OPERATOR_SET_LOCK" || -L "$NATIVE_OPERATOR_SET_LOCK" ]]; then
    echo "CUDA source lanes require a regular native operator set lock as argument 3" >&2
    exit 2
  fi
  local lock_dir
  lock_dir="$(cd "$(dirname "$NATIVE_OPERATOR_SET_LOCK")" && pwd -P)"
  NATIVE_OPERATOR_SET_LOCK="$lock_dir/$(basename "$NATIVE_OPERATOR_SET_LOCK")"
  FERRUM_NATIVE_OPERATOR_SET_LOCK="$NATIVE_OPERATOR_SET_LOCK" \
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
