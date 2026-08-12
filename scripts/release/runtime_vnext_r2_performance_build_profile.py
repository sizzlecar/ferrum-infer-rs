#!/usr/bin/env python3
"""Validate the thin Runtime vNext R2 performance/build/profile checkpoint.

R2 is deliberately an evidence aggregator.  It never launches a model, a
benchmark, a profiler, or a build.  It consumes the formal R1 gate, six
model/backend performance manifests, the frozen Ferrum floor catalog, a profile
bundle, and one independently verifiable G07A build-evidence root.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
PERFORMANCE_AMENDMENT = (
    REPO_ROOT
    / "docs/goals/runtime-vnext-0.8.0-2026-07-10/PERFORMANCE_ACCEPTANCE_AMENDMENT_2026-08-06.md"
)
SCHEMA_VERSION = 1
PASS_PREFIX = "FERRUM RUNTIME VNEXT R2 PERFORMANCE BUILD PROFILE PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT R2 PERFORMANCE BUILD PROFILE SELFTEST PASS"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

MODELS = (
    "m1-qwen35-4b",
    "m2-qwen35-35b-a3b",
    "m3-qwen3-30b-a3b",
)
BACKENDS = ("cuda", "metal")
LANE_KEYS = {
    f"{model_key.split('-', 1)[0]}_{backend}": (model_key, backend)
    for model_key in MODELS
    for backend in BACKENDS
}
MAIN_CONCURRENCY = {"cuda": (1, 4, 16, 32), "metal": (1, 4, 16)}
SENTINEL_CONCURRENCY = {"cuda": (1, 32), "metal": (1, 16)}
SENTINEL_DATASET = {"cuda": "sharegpt", "metal": "real-chat"}
ACTIVE_FLOORS = {
    ("m1-qwen35-4b", "cuda"): 32,
    ("m2-qwen35-35b-a3b", "cuda"): 16,
    ("m3-qwen3-30b-a3b", "cuda"): 32,
    ("m1-qwen35-4b", "metal"): 16,
    ("m2-qwen35-35b-a3b", "metal"): 4,
    ("m3-qwen3-30b-a3b", "metal"): 16,
}
ABSOLUTE_RUN_FLOORS = {
    ("m1-qwen35-4b", "cuda"): 50.0,
    ("m2-qwen35-35b-a3b", "cuda"): 50.0,
    ("m3-qwen3-30b-a3b", "cuda"): 100.0,
    ("m1-qwen35-4b", "metal"): 20.0,
    ("m2-qwen35-35b-a3b", "metal"): 5.0,
    ("m3-qwen3-30b-a3b", "metal"): 5.0,
}
HIGHEST_CONCURRENCY_SCALE = {"cuda": 1.25, "metal": 1.10}
PHYSICAL_HEADROOM_FLOOR_BYTES = {
    "cuda": 512 * 1024**2,
    "metal": 2 * 1024**3,
}
FLOOR_METRICS = (
    "throughput",
    "ttft_p95",
    "tpot_p95",
    "peak_accelerator_or_unified_memory",
)
FLOOR_UNITS = {
    "throughput": "output_tokens_per_second",
    "ttft_p95": "milliseconds",
    "tpot_p95": "milliseconds",
    "peak_accelerator_or_unified_memory": "bytes",
}
BUILD_TARGETS = {
    "noop": 30.0,
    "rust-model-leaf": 90.0,
    "rust-runtime-leaf": 90.0,
    "core-ptx": 120.0,
    "native-tu": 300.0,
    "clean-release": 900.0,
}
PROFILE_IDENTITY_FIELDS = {
    "plan",
    "node",
    "operation",
    "resource",
    "provider",
    "kernel",
}
POST_CELL_OCCUPANCY_COUNTER_FIELDS = (
    "claim_count",
    "segment_count",
    "physical_bytes",
)
POST_CELL_OCCUPANCY_SCOPE_FIELDS = (
    "plan",
    "request",
    "sequence",
    "step",
    "invocation",
    "initial_sequence_bundle",
)
R2_CONTROL_PLANE_FILES = frozenset(
    {
        "docs/goals/runtime-vnext-0.8.0-2026-07-10/METAL_HOST_GLOBAL_SWAP_NOISE_AMENDMENT_2026-08-13.md",
        "docs/goals/runtime-vnext-0.8.0-2026-07-10/PERFORMANCE_ACCEPTANCE_AMENDMENT_2026-08-06.md",
        "scripts/release/bounded_command.py",
        "scripts/release/runtime_vnext_r2_ferrum_collector.py",
        "scripts/release/runtime_vnext_r2_ferrum_terminal_recovery.py",
        "scripts/release/runtime_vnext_r2_performance_build_profile.py",
        "scripts/release/runtime_vnext_r2_profile_collector.py",
        "scripts/release/runtime_vnext_g07a_build_iteration.py",
        "scripts/release/run_gate.py",
        "scripts/release/configs/runtime_vnext_r2_ferrum_floors.json",
    }
)
DOES_NOT_PROVE = [
    "R3 release freeze, staged assets, publication, or installed regression",
    "vLLM or llama.cpp competitiveness",
    "v0.8.0 release readiness",
]


class R2Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R2Error(message)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise R2Error(f"invalid {label} JSON {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="ascii") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def collector_canonical_json_sha256(value: Any) -> str:
    """Match the checked-in Ferrum performance collector hash encoding."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_sha(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} is not a SHA256",
    )
    return value


def finite_positive(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0,
        f"{label} must be positive and finite",
    )
    return float(value)


def finite_nonnegative(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0,
        f"{label} must be nonnegative and finite",
    )
    return float(value)


def parse_timestamp(value: Any, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise R2Error(f"{label} is not ISO-8601: {value}") from error
    require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return value


def timestamp_nanos(value: Any, label: str) -> int:
    raw = parse_timestamp(value, label)
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = parsed - epoch
    return (
        delta.days * 86_400 * 1_000_000_000
        + delta.seconds * 1_000_000_000
        + delta.microseconds * 1_000
    )


def timestamp_datetime(value: Any, label: str) -> datetime:
    raw = parse_timestamp(value, label)
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def normalize_source(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} source identity is missing")
    source = {
        "git_sha": value.get("git_sha"),
        "git_tree_sha": value.get("git_tree_sha"),
        "dirty": value.get("dirty"),
    }
    require(
        isinstance(source["git_sha"], str)
        and GIT_SHA_RE.fullmatch(source["git_sha"]) is not None
        and isinstance(source["git_tree_sha"], str)
        and GIT_SHA_RE.fullmatch(source["git_tree_sha"]) is not None
        and source["dirty"] is False,
        f"{label} source identity is invalid",
    )
    return source


def git_text(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    require(
        process.returncode == 0,
        f"git {' '.join(args)} failed: {process.stderr.strip()}",
    )
    return process.stdout.strip()


def require_recorded_source_in_history(path: str, digest: Any, label: str) -> None:
    recorded = require_sha(digest, label)
    commits = git_text("log", "--format=%H", "--", path).splitlines()
    require(commits, f"{label} path has no Git history")
    matches = False
    for commit in commits:
        process = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        if process.returncode == 0 and hashlib.sha256(process.stdout).hexdigest() == recorded:
            matches = True
            break
    require(matches, f"{label} is not a reviewed Git-history source")


def current_source() -> dict[str, Any]:
    status = [line for line in git_text("status", "--short").splitlines() if line]
    require(not status, f"R2 source must be clean: {status[:8]}")
    return {
        "git_sha": git_text("rev-parse", "HEAD"),
        "git_tree_sha": git_text("rev-parse", "HEAD^{tree}"),
        "dirty": False,
    }


def source_closure(recorded: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    if recorded == current:
        return {
            "from_git_sha": recorded["git_sha"],
            "to_git_sha": current["git_sha"],
            "policy": "exact-source",
            "changed_file_count": 0,
            "changed_files": [],
        }
    require(
        git_text("rev-parse", f"{recorded['git_sha']}^{{tree}}")
        == recorded["git_tree_sha"],
        "recorded evidence tree differs from git",
    )
    require(
        git_text("merge-base", "--is-ancestor", recorded["git_sha"], current["git_sha"])
        == "",
        "recorded evidence source is not an ancestor of R2",
    )
    changed = [
        row
        for row in git_text(
            "diff", "--name-only", f"{recorded['git_sha']}..{current['git_sha']}", "--"
        ).splitlines()
        if row
    ]
    rejected = [path for path in changed if path not in R2_CONTROL_PLANE_FILES]
    require(
        not rejected,
        f"evidence is stale after non-R2-control-plane changes: {rejected[:8]}",
    )
    return {
        "from_git_sha": recorded["git_sha"],
        "to_git_sha": current["git_sha"],
        "policy": "r2-control-plane-only",
        "changed_file_count": len(changed),
        "changed_files": changed,
    }


def file_ref(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    require(
        resolved.is_file() and not resolved.is_symlink(),
        f"artifact is missing: {resolved}",
    )
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def relative_ref(root: Path, path: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved = path.resolve()
    require(
        resolved.is_file() and not resolved.is_symlink(),
        f"artifact is missing: {resolved}",
    )
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise R2Error(f"artifact escaped root: {resolved}") from error
    return {
        "path": relative.as_posix(),
        "sha256": sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def validate_ref(
    value: Any,
    label: str,
    *,
    root: Path | None = None,
) -> Path:
    require(
        isinstance(value, dict)
        and set(value) == {"path", "sha256", "size_bytes"},
        f"{label} reference fields differ",
    )
    raw = Path(str(value["path"])).expanduser()
    if root is not None:
        require(
            not raw.is_absolute() and ".." not in raw.parts,
            f"{label} path must be relative and contained",
        )
        path = (root.resolve() / raw).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise R2Error(f"{label} escaped artifact root") from error
    else:
        path = raw.resolve()
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    require(
        isinstance(value["size_bytes"], int)
        and value["size_bytes"] >= 0
        and path.stat().st_size == value["size_bytes"],
        f"{label} size mismatch",
    )
    require_sha(value["sha256"], f"{label}.sha256")
    require(sha256(path) == value["sha256"], f"{label} SHA256 mismatch")
    return path


def input_manifest(path: Path, default_name: str, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / default_name
    require(
        resolved.is_file() and not resolved.is_symlink(),
        f"{label} manifest is missing: {resolved}",
    )
    return resolved


def expected_cells(backend: str) -> set[tuple[str, int]]:
    return {
        *(("random", concurrency) for concurrency in MAIN_CONCURRENCY[backend]),
        *(
            (SENTINEL_DATASET[backend], concurrency)
            for concurrency in SENTINEL_CONCURRENCY[backend]
        ),
    }


def requires_active_floor(dataset: str, concurrency: int, backend: str) -> bool:
    """The active-floor proof belongs to the high-concurrency random cell."""

    return dataset == "random" and concurrency == max(MAIN_CONCURRENCY[backend])


def requests_per_repeat(dataset: str) -> int:
    return 100 if dataset == "random" else 30


def derive_metal_swap_scope_evidence(
    observations_path: Path,
    observations_sha256: str,
    summary: dict[str, Any],
    *,
    label: str,
    measurement_started_at: str | None = None,
    measurement_finished_at: str | None = None,
) -> dict[str, Any]:
    """Derive transparent host-global swap movement from immutable raw rows."""

    rows = read_jsonl_objects(observations_path, f"{label} raw resource observations")
    require(
        rows[0].get("record_type") == "header"
        and rows[-1].get("record_type") == "footer",
        f"{label} raw resource observations lack header/footer",
    )
    header = rows[0]
    samples = [row for row in rows[1:-1] if row.get("record_type") == "sample"]
    require(len(samples) >= 2, f"{label} raw swap evidence has fewer than two samples")
    parsed = [
        timestamp_datetime(row.get("sampled_at"), f"{label} sample timestamp")
        for row in samples
    ]
    require(
        all(right > left for left, right in zip(parsed, parsed[1:])),
        f"{label} raw swap timestamps are not strictly increasing",
    )
    if measurement_started_at is not None or measurement_finished_at is not None:
        require(
            measurement_started_at is not None
            and measurement_finished_at is not None,
            f"{label} swap measurement window is incomplete",
        )
        started = timestamp_datetime(
            measurement_started_at, f"{label} measurement start"
        )
        finished = timestamp_datetime(
            measurement_finished_at, f"{label} measurement finish"
        )
        before = [index for index, value in enumerate(parsed) if value <= started]
        after = [index for index, value in enumerate(parsed) if value >= finished]
        require(before and after, f"{label} raw swap evidence does not bracket measurement")
        start_index = before[-1]
        finish_index = after[0]
        samples = samples[start_index : finish_index + 1]
        parsed = parsed[start_index : finish_index + 1]

    swap_values = [
        int(finite_nonnegative(row.get("swap_used_bytes"), f"{label} raw swap"))
        for row in samples
    ]
    require(
        swap_values[0] == summary.get("swap_start_bytes")
        and swap_values[-1] == summary.get("swap_end_bytes"),
        f"{label} raw swap endpoints differ from the collector summary",
    )
    deltas = [right - left for left, right in zip(swap_values, swap_values[1:])]
    changed = [delta for delta in deltas if delta != 0]
    positive_steps = [
        {
            "before_sample_index": index,
            "after_sample_index": index + 1,
            "before_sample_at": samples[index]["sampled_at"],
            "after_sample_at": samples[index + 1]["sampled_at"],
            "before_bytes": swap_values[index],
            "after_bytes": swap_values[index + 1],
            "delta_bytes": delta,
        }
        for index, delta in enumerate(deltas)
        if delta > 0
    ]
    decreases = [delta for delta in deltas if delta < 0]
    headroom_values = [
        int(
            finite_nonnegative(
                row.get("physical_headroom_bytes"), f"{label} raw physical headroom"
            )
        )
        for row in samples
    ]
    require(
        min(headroom_values) == summary.get("physical_headroom_bytes"),
        f"{label} raw physical headroom differs from the collector summary",
    )
    thermal_states = [row.get("thermal_state") for row in samples]
    require(
        all(value in {"nominal", "throttled"} for value in thermal_states),
        f"{label} raw thermal state is invalid",
    )
    raw_thermal_throttling_count = sum(
        value != "nominal" for value in thermal_states
    )
    require(
        raw_thermal_throttling_count == summary.get("thermal_throttling_count"),
        f"{label} raw thermal state differs from the collector summary",
    )
    power_modes = [row.get("power_mode") for row in samples]
    require(
        all(value in {"normal", "low-power"} for value in power_modes),
        f"{label} raw power mode is invalid",
    )
    low_power_sample_count = sum(value != "normal" for value in power_modes)
    raw_total_positive_growth_bytes = sum(
        row["delta_bytes"] for row in positive_steps
    )
    raw_max_positive_step_bytes = max(
        (row["delta_bytes"] for row in positive_steps), default=0
    )
    classification = (
        "exact_zero" if not positive_steps else "host_global_swap_observed"
    )
    return {
        "classification": classification,
        "classification_reason": (
            "no-positive-vm.swapusage-step-observed"
            if classification == "exact_zero"
            else "host-global-vm.swapusage-positive-step-observed-not-process-attributable"
        ),
        "raw_collector_sha256": require_sha(
            header.get("collector_sha256"), f"{label} raw collector SHA256"
        ),
        "resource_observations_sha256": require_sha(
            observations_sha256, f"{label} resource observations SHA256"
        ),
        "raw_sample_count": len(samples),
        "raw_step_count": len(changed),
        "raw_positive_step_count": len(positive_steps),
        "raw_negative_step_count": len(decreases),
        "raw_max_absolute_step_bytes": max(
            (abs(delta) for delta in changed), default=0
        ),
        "raw_max_positive_step_bytes": raw_max_positive_step_bytes,
        "raw_total_positive_growth_bytes": raw_total_positive_growth_bytes,
        "raw_net_growth_bytes": swap_values[-1] - swap_values[0],
        "swap_start_bytes": swap_values[0],
        "swap_end_bytes": swap_values[-1],
        "positive_steps": positive_steps,
        "physical_headroom_bytes": int(summary["physical_headroom_bytes"]),
        "thermal_throttling_count": summary.get("thermal_throttling_count"),
        "low_power_sample_count": low_power_sample_count,
        "oom_count": summary.get("oom_count"),
        "admission_error_count": summary.get("admission_error_count"),
        "active_probe_retry_error_count": summary.get(
            "active_probe_retry_error_count"
        ),
        "first_sample_at": samples[0]["sampled_at"],
        "last_sample_at": samples[-1]["sampled_at"],
    }


def aggregate_metal_swap_evidence(
    scopes: list[dict[str, Any]], label: str
) -> dict[str, Any]:
    """Aggregate non-attributable host-global swap telemetry without gating it."""

    require(len(scopes) == 8, f"{label} must cover five cells and three run samples")
    require(
        all(
            row.get("classification")
            == (
                "exact_zero"
                if row.get("raw_positive_step_count") == 0
                else "host_global_swap_observed"
            )
            for row in scopes
        ),
        f"{label} scope swap classification is not raw-derived",
    )
    positive_step_count = sum(row["raw_positive_step_count"] for row in scopes)
    total_positive_growth = sum(
        row["raw_total_positive_growth_bytes"] for row in scopes
    )
    max_positive_step = max(row["raw_max_positive_step_bytes"] for row in scopes)
    classification = (
        "exact_zero"
        if positive_step_count == 0
        else "host_global_swap_observed"
    )
    return {
        "classification": classification,
        "classification_reason": (
            "no-positive-vm.swapusage-step-observed-across-lane"
            if classification == "exact_zero"
            else "host-global-vm.swapusage-positive-step-observed-not-process-attributable"
        ),
        "contract_scope": "five-formal-http-cells-plus-three-run-samples",
        "policy": {
            "signal": "macos-vm.swapusage",
            "attribution": "host-global-non-attributable",
            "resource_gate": False,
            "classification_basis": "positive-step-count-only",
        },
        "scope_count": len(scopes),
        "raw_step_count": sum(row["raw_step_count"] for row in scopes),
        "raw_positive_step_count": positive_step_count,
        "raw_negative_step_count": sum(
            row["raw_negative_step_count"] for row in scopes
        ),
        "raw_max_absolute_step_bytes": max(
            row["raw_max_absolute_step_bytes"] for row in scopes
        ),
        "raw_max_positive_step_bytes": max_positive_step,
        "raw_total_positive_growth_bytes": total_positive_growth,
        "raw_net_growth_bytes": sum(row["raw_net_growth_bytes"] for row in scopes),
        "minimum_physical_headroom_bytes": min(
            row["physical_headroom_bytes"] for row in scopes
        ),
        "raw_collector_sha256": sorted(
            {row["raw_collector_sha256"] for row in scopes}
        ),
        "scopes": copy.deepcopy(scopes),
    }


def validate_metal_resource_contract(
    summary: dict[str, Any],
    label: str,
    *,
    typed_memory_budget_bytes: Any,
    resource_evidence: dict[str, Any],
) -> None:
    typed_budget = finite_positive(
        typed_memory_budget_bytes, f"{label} typed Metal memory budget"
    )
    summary_budget = finite_positive(
        summary.get("memory_budget_bytes"), f"{label} Metal summary memory budget"
    )
    require(
        summary_budget == typed_budget,
        f"{label} Metal summary memory budget differs from typed config",
    )
    require(
        finite_positive(summary.get("peak_memory_bytes"), f"{label} Metal peak RSS")
        <= typed_budget,
        f"{label} Metal peak process-group RSS exceeds the typed memory budget",
    )
    physical_headroom = finite_nonnegative(
        summary.get("physical_headroom_bytes"), f"{label} Metal physical headroom"
    )
    physical_headroom_floor = PHYSICAL_HEADROOM_FLOOR_BYTES["metal"]
    require(
        physical_headroom >= physical_headroom_floor,
        f"{label} Metal physical headroom is below the "
        f"{physical_headroom_floor}-byte floor",
    )
    require(
        resource_evidence.get("classification")
        in {"exact_zero", "host_global_swap_observed"},
        f"{label} Metal host-global swap telemetry classification is invalid",
    )
    require(
        resource_evidence.get("physical_headroom_bytes")
        == summary.get("physical_headroom_bytes")
        and resource_evidence.get("thermal_throttling_count")
        == summary.get("thermal_throttling_count")
        and resource_evidence.get("oom_count") == summary.get("oom_count")
        and resource_evidence.get("admission_error_count")
        == summary.get("admission_error_count")
        and resource_evidence.get("active_probe_retry_error_count")
        == summary.get("active_probe_retry_error_count"),
        f"{label} Metal raw resource evidence differs from the collector summary",
    )
    require(
        summary.get("thermal_throttling_count") == 0
        and summary.get("thermal_start") == "nominal"
        and summary.get("thermal_end") == "nominal",
        f"{label} Metal thermal state is not nominal",
    )
    require(
        summary.get("power_mode_start") == "normal"
        and summary.get("power_mode_end") == "normal"
        and resource_evidence.get("low_power_sample_count") == 0,
        f"{label} Metal power mode is not normal",
    )
    require(
        summary.get("oom_count") == 0
        and summary.get("admission_error_count") == 0
        and summary.get("active_probe_retry_error_count") == 0,
        f"{label} Metal OOM/admission/probe error contract failed",
    )


def nearest_rank(values: list[float], percentile: float) -> float:
    require(values, "percentile input is empty")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def flag_value(argv: list[str], flag: str, label: str) -> str:
    require(argv.count(flag) == 1, f"{label} must contain {flag} exactly once")
    index = argv.index(flag)
    require(index + 1 < len(argv), f"{label} lacks a value for {flag}")
    value = argv[index + 1]
    require(value and not value.startswith("--"), f"{label} has invalid {flag}")
    return value


def validate_benchmark_argv(
    argv: Any,
    *,
    backend: str,
    dataset: str,
    concurrency: int,
    request_count: int,
) -> list[str]:
    require(
        isinstance(argv, list)
        and len(argv) >= 10
        and all(isinstance(item, str) and item for item in argv),
        "benchmark argv is invalid",
    )
    require(argv.count("bench-serve") == 1, "benchmark argv must run bench-serve")
    require(
        argv.count("--fail-on-error") == 1
        and argv.count("--require-ci") == 1,
        "benchmark argv lacks the canonical failure/CI flags",
    )
    require(flag_value(argv, "--seed", "benchmark argv") == "9271", "seed differs")
    require(
        flag_value(argv, "--n-repeats", "benchmark argv") == "3",
        "repeat count differs",
    )
    require(
        flag_value(argv, "--concurrency", "benchmark argv") == str(concurrency),
        "concurrency differs",
    )
    require(
        flag_value(argv, "--num-prompts", "benchmark argv") == str(request_count),
        "request count differs",
    )
    expected_dataset = "random" if dataset == "random" else "sharegpt"
    require(
        flag_value(argv, "--dataset", "benchmark argv") == expected_dataset,
        "dataset differs",
    )
    if dataset == "random":
        require(
            flag_value(argv, "--random-input-len", "benchmark argv")
            == ("256" if backend == "cuda" else "64"),
            "random input length differs",
        )
        require(
            flag_value(argv, "--random-output-len", "benchmark argv") == "128",
            "random output length differs",
        )
    return argv


def default_r1_verifier(path: Path) -> dict[str, Any]:
    try:
        import runtime_vnext_r1_product_correctness as r1

        return r1.verify_manifest(path, verify_checkout=False)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise R2Error(f"R1 child provenance failed: {error}") from error


def trace_r1_full_anchor(
    child_path: Path,
    child_manifest: dict[str, Any],
    r1_module: Any,
    *,
    reference_validator: Callable[[Any, str], Path] = validate_ref,
    manifest_reader: Callable[[Path, str], dict[str, Any]] = read_json,
) -> tuple[Path, dict[str, Any]]:
    """Follow authenticated cumulative R1 links to their final full aggregate."""
    current_path = child_path.expanduser().resolve()
    current_manifest = child_manifest
    visited: set[Path] = set()
    depth = 0
    while True:
        require(current_path not in visited, "cumulative R1 dependency cycle detected")
        visited.add(current_path)
        dependencies = current_manifest.get("dependencies")
        require(isinstance(dependencies, dict), "R1 child dependencies are missing")
        dependency_keys = set(dependencies)
        require(
            dependency_keys == r1_module.FULL_DEPENDENCY_KEYS
            or dependency_keys == r1_module.CUMULATIVE_DEPENDENCY_KEYS,
            "R1 child dependency shape differs",
        )
        if dependency_keys == r1_module.FULL_DEPENDENCY_KEYS:
            try:
                r1_module.require_full_prior_manifest_shape(current_manifest)
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                raise R2Error(f"final R1 full anchor shape failed: {error}") from error
            return current_path, current_manifest

        depth += 1
        require(depth <= 128, "cumulative R1 dependency depth exceeds 128")
        # Authenticate every edge, including qualifications that are consumed by
        # the canonical recursive verifier, before allowing it to recurse.  This
        # makes cycles and malformed chains fail closed instead of risking an
        # unbounded verifier recursion.
        reference_validator(
            dependencies.get("impact_qualification"),
            f"cumulative R1 depth {depth} impact qualification",
        )
        prior_path = reference_validator(
            dependencies.get("prior_r1"),
            f"cumulative R1 depth {depth} prior manifest",
        ).expanduser().resolve()
        require(prior_path not in visited, "cumulative R1 dependency cycle detected")
        current_path = prior_path
        current_manifest = manifest_reader(
            current_path, f"cumulative R1 depth {depth} prior manifest"
        )


def validate_cumulative_profile_off_evidence(
    child_path: Path,
    child_manifest: dict[str, Any],
    summary: dict[str, Any],
    r1_module: Any,
) -> dict[str, Any] | None:
    dependencies = child_manifest.get("dependencies")
    require(isinstance(dependencies, dict), "R1 child dependencies are missing")
    dependency_keys = set(dependencies)
    require(
        dependency_keys == r1_module.FULL_DEPENDENCY_KEYS
        or dependency_keys == r1_module.CUMULATIVE_DEPENDENCY_KEYS,
        "R1 child dependency shape differs",
    )
    acceptance = summary["acceptance"]
    if dependency_keys == r1_module.FULL_DEPENDENCY_KEYS:
        require(
            "evidence_backend_binary_sha256" not in acceptance,
            "full R1 cannot expose cumulative performance evidence",
        )
        return None

    anchor_path, anchor_manifest = trace_r1_full_anchor(
        child_path, child_manifest, r1_module
    )

    # The cumulative exception is available only after the canonical R1 verifier
    # has recursively revalidated every qualification and prior R1 in the safe,
    # structurally authenticated chain above.
    try:
        deep_summary = r1_module.verify_manifest(
            child_path,
            verify_checkout=False,
            expected_source=summary["source"],
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise R2Error(f"cumulative R1 deep provenance failed: {error}") from error
    require(deep_summary == summary, "cumulative R1 deep summary differs")

    anchor_source = normalize_source(anchor_manifest.get("source"), "final full R1")
    try:
        anchor_summary = r1_module.verify_manifest(
            anchor_path,
            verify_checkout=False,
            expected_source=anchor_source,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise R2Error(f"final full R1 canonical verification failed: {error}") from error
    require(
        isinstance(anchor_summary, dict)
        and anchor_summary.get("kind") == "vnext-r1"
        and anchor_summary.get("source") == anchor_source,
        "final full R1 canonical summary differs",
    )
    anchor_acceptance = anchor_summary.get("acceptance")
    require(isinstance(anchor_acceptance, dict), "final full R1 acceptance is missing")
    anchor_binaries = anchor_acceptance.get("backend_binary_sha256")
    evidence_binaries = acceptance.get("evidence_backend_binary_sha256")
    anchor_hardware = anchor_acceptance.get("backend_hardware_id")
    require(
        isinstance(anchor_binaries, dict)
        and set(anchor_binaries) == set(BACKENDS)
        and all(SHA256_RE.fullmatch(str(value)) for value in anchor_binaries.values())
        and evidence_binaries == anchor_binaries,
        "cumulative R1 evidence binary authority differs from final full R1",
    )
    require(
        isinstance(anchor_hardware, dict)
        and set(anchor_hardware) == set(BACKENDS)
        and acceptance.get("backend_hardware_id") == anchor_hardware,
        "cumulative R1 hardware authority differs from final full R1",
    )
    qualification_path = validate_ref(
        dependencies.get("impact_qualification"),
        "cumulative R1 impact qualification",
    )
    anchor_ref = file_ref(anchor_path)
    qualification_ref = file_ref(qualification_path)
    require(
        qualification_ref == dependencies["impact_qualification"],
        "cumulative R1 dependency reference drifted",
    )
    return {
        "mode": "cumulative-r1-prior-full-profile-off",
        "source": anchor_source,
        "r1_child_manifest": anchor_ref,
        "impact_qualification": qualification_ref,
        "backend_binary_sha256": copy.deepcopy(evidence_binaries),
        "backend_hardware_id": copy.deepcopy(anchor_hardware),
    }


def validate_r1_outer(
    path: Path,
    source: dict[str, Any],
    *,
    verifier: Callable[[Path], dict[str, Any]] = default_r1_verifier,
) -> dict[str, Any]:
    try:
        import runtime_vnext_r1_product_correctness as r1
    except ModuleNotFoundError as error:
        raise R2Error(f"cannot load R1 validator: {error}") from error
    outer_path = input_manifest(path, "gate.manifest.json", "R1 outer")
    outer = read_json(outer_path, "R1 outer")
    require(set(outer) == r1.OUTER_GATE_FIELDS, "R1 outer field set mismatch")
    actual_root = outer_path.parent
    recorded_root = Path(str(outer.get("artifact_dir", "")))
    require(recorded_root.is_absolute(), "R1 recorded artifact root is invalid")
    require(
        outer.get("schema_version") == 1
        and outer.get("lane") == "vnext-r1"
        and outer.get("status") == "pass"
        and outer.get("child_returncode") == 0
        and outer.get("error") is None
        and outer.get("dirty_status") == {"is_dirty": False, "status_short": []},
        "R1 outer identity/status differs",
    )
    delegated = outer.get("delegated_command_line")
    require(
        isinstance(delegated, list)
        and len(delegated) >= 4
        and Path(str(delegated[1])).as_posix().endswith(
            "scripts/release/runtime_vnext_r1_product_correctness.py"
        ),
        "R1 delegated command identity differs",
    )
    require(
        Path(r1.command_flag(delegated, "--out", "R1 delegated command"))
        == recorded_root,
        "R1 delegated output differs",
    )
    expected_child = f"FERRUM RUNTIME VNEXT R1 PRODUCT CORRECTNESS PASS: {recorded_root}"
    require(
        outer.get("child_pass_line") == expected_child
        and outer.get("pass_line") == f"FERRUM GATE vnext-r1 PASS: {recorded_root}",
        "R1 PASS lines differ",
    )
    try:
        r1.validate_outer_receipts(outer, actual_root, delegated, expected_child)
    except (OSError, RuntimeError, ValueError) as error:
        raise R2Error(f"R1 outer receipts failed: {error}") from error
    child_path = actual_root / "manifest.json"
    summary = verifier(child_path)
    require(
        isinstance(summary, dict)
        and summary.get("kind") == "vnext-r1"
        and isinstance(summary.get("acceptance"), dict),
        "R1 verifier returned an invalid summary",
    )
    child_manifest = read_json(child_path, "R1 child")
    recorded_source = normalize_source(summary.get("source"), "R1")
    require(
        outer.get("git_sha") == recorded_source["git_sha"],
        "R1 outer/child source differs",
    )
    closure = source_closure(recorded_source, source)
    artifacts = outer.get("child_artifacts")
    require(isinstance(artifacts, dict), "R1 outer child provenance is missing")
    require(
        artifacts.get("kind") == "vnext-r1"
        and artifacts.get("source") == recorded_source
        and artifacts.get("acceptance") == summary["acceptance"]
        and isinstance(artifacts.get("child_manifest"), dict)
        and artifacts["child_manifest"].get("sha256") == sha256(child_path),
        "R1 outer child provenance differs",
    )
    acceptance = summary["acceptance"]
    binaries = acceptance.get("backend_binary_sha256")
    hardware = acceptance.get("backend_hardware_id")
    require(
        isinstance(binaries, dict)
        and set(binaries) == set(BACKENDS)
        and all(SHA256_RE.fullmatch(str(value)) for value in binaries.values()),
        "R1 backend binary authority is invalid",
    )
    require(
        isinstance(hardware, dict)
        and set(hardware) == set(BACKENDS)
        and all(isinstance(value, str) and value for value in hardware.values()),
        "R1 backend hardware authority is invalid",
    )
    cumulative_profile_off_evidence = validate_cumulative_profile_off_evidence(
        child_path,
        child_manifest,
        summary,
        r1,
    )
    result = {
        "outer_manifest": file_ref(outer_path),
        "child_manifest": file_ref(child_path),
        "source": recorded_source,
        "source_closure": closure,
        "backend_binary_sha256": copy.deepcopy(binaries),
        "backend_hardware_id": copy.deepcopy(hardware),
        "acceptance": copy.deepcopy(acceptance),
    }
    if cumulative_profile_off_evidence is not None:
        result["cumulative_profile_off_evidence"] = (
            cumulative_profile_off_evidence
        )
    return result


def catalog_key(
    model_key: str, backend: str, dataset: str, concurrency: int, metric: str
) -> tuple[str, str, str, int, str]:
    return model_key, backend, dataset, concurrency, metric


def validate_floor_catalog(
    path: Path, *, require_checked_in: bool
) -> tuple[dict[tuple[str, str, str, int, str], dict[str, Any]], dict[str, Any]]:
    catalog_path = input_manifest(path, "floor-catalog.json", "floor catalog")
    if require_checked_in:
        try:
            relative = catalog_path.relative_to(REPO_ROOT)
        except ValueError as error:
            raise R2Error("floor catalog must be checked into the source tree") from error
        process = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative.as_posix()],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        require(process.returncode == 0, "floor catalog is not tracked by git")
    raw = read_json(catalog_path, "floor catalog")
    required = {
        "schema_version",
        "artifact_type",
        "status",
        "frozen_at",
        "goal_contract_sha256",
        "collectors",
        "rows",
        "canonical_sha256_scope",
        "canonical_sha256",
    }
    require(set(raw) == required, "floor catalog field set mismatch")
    require(
        raw.get("schema_version") == SCHEMA_VERSION
        and raw.get("artifact_type") == "runtime_vnext_r2_floor_catalog"
        and raw.get("status") == "frozen"
        and raw.get("canonical_sha256_scope")
        == "document_without_canonical_sha256",
        "floor catalog identity/status differs",
    )
    parse_timestamp(raw.get("frozen_at"), "floor catalog frozen_at")
    require(
        raw.get("goal_contract_sha256") == sha256(PERFORMANCE_AMENDMENT),
        "floor catalog does not bind the active performance amendment",
    )
    collectors = raw.get("collectors")
    require(
        isinstance(collectors, list) and len(collectors) == len(LANE_KEYS),
        "floor catalog must bind exactly six collector manifests",
    )
    collector_index: dict[tuple[str, str], str] = {}
    for index, collector in enumerate(collectors):
        require(
            isinstance(collector, dict)
            and set(collector)
            == {"lane_key", "model_key", "backend", "manifest_sha256"},
            f"floor collector {index} fields differ",
        )
        lane_key = collector.get("lane_key")
        require(
            lane_key in LANE_KEYS
            and (collector.get("model_key"), collector.get("backend"))
            == LANE_KEYS[lane_key]
            and LANE_KEYS[lane_key] not in collector_index,
            f"floor collector {index} identity differs",
        )
        collector_index[LANE_KEYS[lane_key]] = require_sha(
            collector.get("manifest_sha256"), f"floor collector {index} manifest"
        )
    canonical = copy.deepcopy(raw)
    declared = require_sha(canonical.pop("canonical_sha256"), "floor canonical SHA256")
    require(
        canonical_json_sha256(canonical) == declared,
        "floor catalog canonical SHA256 mismatch",
    )
    rows = raw.get("rows")
    require(isinstance(rows, list), "floor catalog rows are missing")
    expected = {
        catalog_key(model_key, backend, dataset, concurrency, metric)
        for model_key in MODELS
        for backend in BACKENDS
        for dataset, concurrency in expected_cells(backend)
        for metric in FLOOR_METRICS
    }
    require(len(rows) == len(expected) == 132, "floor catalog must contain 132 rows")
    indexed: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    row_fields = {
        "key",
        "baseline_kind",
        "value",
        "unit",
        "source_git_sha",
        "dirty",
        "binary_sha256",
        "model_sha256",
        "hardware_id",
        "hardware_sha256",
        "typed_config_sha256",
        "dataset_sha256",
        "command_sha256",
        "collector_manifest_sha256",
        "artifact_sha256",
        "raw_repeats",
        "raw_value_scope",
        "resource_observations_sha256",
        "frozen_at",
    }
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and set(row) == row_fields, f"floor row {index} fields differ")
        key = row.get("key")
        require(
            isinstance(key, dict)
            and set(key) == {"model_key", "backend", "dataset", "concurrency", "metric"},
            f"floor row {index} key differs",
        )
        normalized = catalog_key(
            str(key["model_key"]),
            str(key["backend"]),
            str(key["dataset"]),
            int(key["concurrency"]),
            str(key["metric"]),
        )
        require(normalized in expected, f"floor row {index} key is unexpected: {normalized}")
        require(normalized not in indexed, f"duplicate floor row: {normalized}")
        metric = normalized[-1]
        value = finite_positive(row.get("value"), f"floor row {index} value")
        require(row.get("unit") == FLOOR_UNITS[metric], f"floor row {index} unit differs")
        require(row.get("baseline_kind") in {"historical", "calibration"}, f"floor row {index} baseline kind differs")
        require(
            isinstance(row.get("source_git_sha"), str)
            and GIT_SHA_RE.fullmatch(row["source_git_sha"]) is not None
            and row.get("dirty") is False,
            f"floor row {index} source is invalid",
        )
        for name in (
            "binary_sha256",
            "model_sha256",
            "hardware_sha256",
            "typed_config_sha256",
            "dataset_sha256",
            "command_sha256",
            "collector_manifest_sha256",
            "artifact_sha256",
            "resource_observations_sha256",
        ):
            require_sha(row.get(name), f"floor row {index} {name}")
        require(
            row["collector_manifest_sha256"]
            == collector_index[(normalized[0], normalized[1])],
            f"floor row {index} collector binding differs",
        )
        require(
            isinstance(row.get("hardware_id"), str) and row["hardware_id"],
            f"floor row {index} hardware is missing",
        )
        parse_timestamp(row.get("frozen_at"), f"floor row {index} frozen_at")
        require(
            row.get("frozen_at") == raw.get("frozen_at"),
            f"floor row {index} freeze timestamp differs",
        )
        repeats = row.get("raw_repeats")
        expected_repeat_count = 1 if metric == "peak_accelerator_or_unified_memory" else 3
        expected_scope = (
            "cell-aggregate-peak-over-three-benchmark-repeats"
            if metric == "peak_accelerator_or_unified_memory"
            else "three-benchmark-repeat-values"
        )
        require(
            isinstance(repeats, list) and len(repeats) == expected_repeat_count,
            f"floor row {index} raw value denominator differs: expected {expected_repeat_count}",
        )
        require(
            row.get("raw_value_scope") == expected_scope,
            f"floor row {index} raw value scope differs",
        )
        repeat_values = [
            finite_positive(item, f"floor row {index} raw repeat") for item in repeats
        ]
        recomputed = (
            max(repeat_values)
            if metric == "peak_accelerator_or_unified_memory"
            else statistics.median(repeat_values)
        )
        require(
            math.isclose(value, recomputed, rel_tol=1e-9, abs_tol=1e-9),
            f"floor row {index} value does not derive from raw repeats",
        )
        indexed[normalized] = copy.deepcopy(row)
    require(set(indexed) == expected, "floor catalog key set differs")
    return indexed, {"catalog": file_ref(catalog_path), "row_count": len(indexed), "canonical_sha256": declared}


def validate_request_rows(
    rows: Any,
    *,
    backend: str,
    dataset: str,
    request_count: int,
) -> dict[int, list[dict[str, Any]]]:
    require(isinstance(rows, list), "request rows are missing")
    require(len(rows) == request_count * 3, "measured request denominator differs")
    expected_fields = {
        "repeat",
        "index",
        "completed",
        "usage_token_source",
        "input_tokens",
        "output_tokens",
        "error",
        "bad_output",
        "malformed_sse",
        "done_count",
        "ttft_ms",
        "tpot_ms",
        "itl_eligible",
        "itl_ms",
        "fields_complete",
    }
    grouped: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}
    identities: set[tuple[int, int]] = set()
    for row in rows:
        require(isinstance(row, dict) and set(row) == expected_fields, "measured request fields differ")
        repeat = row.get("repeat")
        index = row.get("index")
        require(
            repeat in grouped
            and isinstance(index, int)
            and not isinstance(index, bool)
            and 1 <= index <= request_count,
            "measured request identity differs",
        )
        identity = (repeat, index)
        require(identity not in identities, f"duplicate measured request: {identity}")
        identities.add(identity)
        require(
            row.get("completed") is True
            and row.get("usage_token_source") == "usage"
            and row.get("error") is None
            and row.get("bad_output") is False
            and row.get("malformed_sse") is False
            and row.get("done_count") == 1
            and row.get("fields_complete") is True,
            f"measured request failed correctness/completeness: {identity}",
        )
        input_tokens = row.get("input_tokens")
        output_tokens = row.get("output_tokens")
        require(
            isinstance(input_tokens, int)
            and input_tokens > 0
            and isinstance(output_tokens, int)
            and output_tokens > 0,
            f"measured request token counts are invalid: {identity}",
        )
        if dataset == "random":
            require(
                input_tokens == (256 if backend == "cuda" else 64)
                and output_tokens == 128,
                f"random request lengths differ: {identity}",
            )
        else:
            require(output_tokens <= 128, f"sentinel output exceeds policy: {identity}")
        finite_nonnegative(row.get("ttft_ms"), f"request {identity} TTFT")
        finite_nonnegative(row.get("tpot_ms"), f"request {identity} TPOT")
        require(isinstance(row.get("itl_eligible"), bool), f"request {identity} ITL eligibility is missing")
        if row["itl_eligible"]:
            finite_nonnegative(row.get("itl_ms"), f"request {identity} ITL")
        else:
            require(row.get("itl_ms") is None, f"ineligible request {identity} has ITL")
        grouped[repeat].append(row)
    require(
        all(len(grouped[index]) == request_count for index in grouped),
        "per-repeat measured request denominator differs",
    )
    return grouped


def validate_warmups(rows: Any) -> None:
    require(isinstance(rows, list) and rows, "warmup rows are missing")
    expected_fields = {"repeat", "index", "completed", "error"}
    grouped = {1: 0, 2: 0, 3: 0}
    identities: set[tuple[int, int]] = set()
    for row in rows:
        require(isinstance(row, dict) and set(row) == expected_fields, "warmup row fields differ")
        repeat = row.get("repeat")
        index = row.get("index")
        require(
            repeat in grouped
            and isinstance(index, int)
            and not isinstance(index, bool)
            and index > 0,
            "warmup identity differs",
        )
        identity = (repeat, index)
        require(identity not in identities, f"duplicate warmup request: {identity}")
        identities.add(identity)
        require(row.get("completed") is True and row.get("error") is None, f"warmup request failed: {identity}")
        grouped[repeat] += 1
    require(all(count > 0 for count in grouped.values()), "every repeat requires warmup completion")


def validate_repeat_rows(
    rows: Any,
    *,
    grouped_requests: dict[int, list[dict[str, Any]]],
    backend: str,
) -> list[dict[str, Any]]:
    require(isinstance(rows, list) and len(rows) == 3, "repeat summary count differs")
    common_fields = {
        "repeat",
        "wall_time_seconds",
        "output_tokens",
        "output_throughput_tps",
        "ttft_p95_ms",
        "tpot_p95_ms",
        "steady_decode_tps",
        "ci_low_tps",
        "ci_high_tps",
        "observed_max_active",
        "eligible_interval_seconds",
        "active_floor_duty_cycle",
        "active_timeline_complete",
        "peak_memory_bytes",
        "memory_budget_bytes",
    }
    backend_fields = (
        {"physical_vram_headroom_bytes"}
        if backend == "cuda"
        else {"swap_growth_bytes", "thermal_throttling_count"}
    )
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        require(
            isinstance(row, dict) and set(row) == common_fields | backend_fields,
            "repeat summary fields differ",
        )
        repeat = row.get("repeat")
        require(repeat in grouped_requests and repeat not in indexed, "repeat identity differs")
        requests = grouped_requests[repeat]
        output_tokens = sum(int(item["output_tokens"]) for item in requests)
        require(row.get("output_tokens") == output_tokens, f"repeat {repeat} output-token total differs")
        wall = finite_positive(row.get("wall_time_seconds"), f"repeat {repeat} wall time")
        throughput = finite_positive(row.get("output_throughput_tps"), f"repeat {repeat} throughput")
        require(
            math.isclose(throughput, output_tokens / wall, rel_tol=1e-6, abs_tol=1e-6),
            f"repeat {repeat} throughput is not derived from tokens/time",
        )
        ttft = nearest_rank([float(item["ttft_ms"]) for item in requests], 0.95)
        tpot = nearest_rank([float(item["tpot_ms"]) for item in requests], 0.95)
        require(
            math.isclose(float(row.get("ttft_p95_ms", -1)), ttft, abs_tol=1e-9)
            and math.isclose(float(row.get("tpot_p95_ms", -1)), tpot, abs_tol=1e-9),
            f"repeat {repeat} latency p95 differs",
        )
        steady = finite_positive(row.get("steady_decode_tps"), f"repeat {repeat} steady decode")
        low = finite_positive(row.get("ci_low_tps"), f"repeat {repeat} CI low")
        high = finite_positive(row.get("ci_high_tps"), f"repeat {repeat} CI high")
        require(low <= throughput <= high, f"repeat {repeat} CI does not contain throughput")
        require(
            isinstance(row.get("observed_max_active"), int)
            and not isinstance(row["observed_max_active"], bool)
            and row["observed_max_active"] > 0
            and row.get("active_timeline_complete") is True,
            f"repeat {repeat} active timeline differs",
        )
        finite_positive(row.get("eligible_interval_seconds"), f"repeat {repeat} eligible interval")
        duty = finite_nonnegative(row.get("active_floor_duty_cycle"), f"repeat {repeat} active duty")
        require(duty <= 1.0, f"repeat {repeat} active duty exceeds one")
        peak = finite_positive(row.get("peak_memory_bytes"), f"repeat {repeat} peak memory")
        budget = finite_positive(row.get("memory_budget_bytes"), f"repeat {repeat} memory budget")
        require(peak <= budget, f"repeat {repeat} exceeds typed memory budget")
        if backend == "cuda":
            require(
                finite_nonnegative(
                    row.get("physical_vram_headroom_bytes"),
                    f"repeat {repeat} CUDA headroom",
                )
                >= 512 * 1024 * 1024,
                f"repeat {repeat} CUDA headroom is below 512 MiB",
            )
        else:
            require(
                row.get("swap_growth_bytes") == 0
                and row.get("thermal_throttling_count") == 0,
                f"repeat {repeat} Metal swap/thermal contract failed",
            )
        indexed[repeat] = {**copy.deepcopy(row), "steady_decode_tps": steady}
    require(set(indexed) == {1, 2, 3}, "repeat summary set differs")
    return [indexed[index] for index in (1, 2, 3)]


def validate_performance_report(
    path: Path,
    *,
    model_key: str,
    backend: str,
    dataset: str,
    concurrency: int,
    source: dict[str, Any],
    binary_sha256: str,
    hardware_id: str,
    model_sha256: str,
    typed_config_sha256: str,
) -> dict[str, Any]:
    report = read_json(path, f"{model_key}/{backend}/{dataset}/c{concurrency} report")
    required = {
        "schema_version",
        "artifact_type",
        "model_key",
        "backend",
        "dataset",
        "dataset_sha256",
        "concurrency",
        "source",
        "binary_sha256",
        "hardware_id",
        "model_sha256",
        "typed_config_sha256",
        "profile_mode",
        "benchmark_argv",
        "requested_input_tokens",
        "requested_output_tokens",
        "requests_per_repeat",
        "n_repeats",
        "warmups",
        "requests",
        "repeats",
    }
    require(set(report) == required, "performance report field set differs")
    require(
        report.get("schema_version") == SCHEMA_VERSION
        and report.get("artifact_type") == "runtime_vnext_r2_performance_cell_report"
        and report.get("model_key") == model_key
        and report.get("backend") == backend
        and report.get("dataset") == dataset
        and report.get("concurrency") == concurrency
        and normalize_source(report.get("source"), "performance report") == source
        and report.get("binary_sha256") == binary_sha256
        and report.get("hardware_id") == hardware_id
        and report.get("model_sha256") == model_sha256
        and report.get("typed_config_sha256") == typed_config_sha256
        and report.get("profile_mode") == "off"
        and report.get("requested_output_tokens") == 128
        and report.get("n_repeats") == 3,
        "performance report identity/config differs",
    )
    require_sha(report.get("dataset_sha256"), "performance dataset SHA256")
    if dataset == "random":
        require(
            report.get("requested_input_tokens") == (256 if backend == "cuda" else 64),
            "random input-token policy differs",
        )
    else:
        require(report.get("requested_input_tokens") is None, "sentinel input length must use tokenizer counts")
    count = requests_per_repeat(dataset)
    require(report.get("requests_per_repeat") == count, "requests-per-repeat differs")
    validate_benchmark_argv(
        report.get("benchmark_argv"),
        backend=backend,
        dataset=dataset,
        concurrency=concurrency,
        request_count=count,
    )
    validate_warmups(report.get("warmups"))
    grouped = validate_request_rows(
        report.get("requests"), backend=backend, dataset=dataset, request_count=count
    )
    repeats = validate_repeat_rows(report.get("repeats"), grouped_requests=grouped, backend=backend)
    throughputs = [float(row["output_throughput_tps"]) for row in repeats]
    mean = statistics.mean(throughputs)
    cv = statistics.pstdev(throughputs) / mean
    require(cv <= 0.08 + 1e-12, f"{model_key}/{backend}/{dataset}/c{concurrency} CV exceeds 8%")
    return {
        "dataset": dataset,
        "dataset_sha256": report["dataset_sha256"],
        "concurrency": concurrency,
        "request_count": count * 3,
        "repeat_count": 3,
        "throughput": statistics.median(throughputs),
        "ttft_p95": statistics.median(float(row["ttft_p95_ms"]) for row in repeats),
        "tpot_p95": statistics.median(float(row["tpot_p95_ms"]) for row in repeats),
        "peak_accelerator_or_unified_memory": max(float(row["peak_memory_bytes"]) for row in repeats),
        "steady_decode": statistics.median(float(row["steady_decode_tps"]) for row in repeats),
        "cv": cv,
        "observed_max_active": min(int(row["observed_max_active"]) for row in repeats),
        "active_floor_duty_cycle": min(float(row["active_floor_duty_cycle"]) for row in repeats),
        "raw_repeats": {
            "throughput": [float(row["output_throughput_tps"]) for row in repeats],
            "ttft_p95": [float(row["ttft_p95_ms"]) for row in repeats],
            "tpot_p95": [float(row["tpot_p95_ms"]) for row in repeats],
            "peak_accelerator_or_unified_memory": [
                max(float(row["peak_memory_bytes"]) for row in repeats)
            ],
        },
        "command_sha256": canonical_json_sha256(report["benchmark_argv"]),
        "resource_observations_sha256": sha256(path),
        "metric_artifact_sha256": {
            metric: sha256(path) for metric in FLOOR_METRICS
        },
        "raw_value_scope": {
            metric: (
                "cell-aggregate-peak-over-three-benchmark-repeats"
                if metric == "peak_accelerator_or_unified_memory"
                else "three-benchmark-repeat-values"
            )
            for metric in FLOOR_METRICS
        },
        "report": file_ref(path),
    }


def validate_run_samples(
    rows: Any,
    *,
    root: Path,
    model_key: str,
    backend: str,
    source: dict[str, Any],
    binary_sha256: str,
    hardware_id: str,
    model_sha256: str,
    typed_config_sha256: str,
) -> dict[str, Any]:
    require(isinstance(rows, list) and len(rows) == 3, f"{model_key}/{backend} requires three run samples")
    expected_row_fields = {"process_index", "command", "result"}
    expected_result_fields = {
        "schema_version",
        "artifact_type",
        "source",
        "model_key",
        "backend",
        "binary_sha256",
        "hardware_id",
        "model_sha256",
        "typed_config_sha256",
        "process_id",
        "status",
        "output_tokens",
        "steady_decode_tps",
        "engine_infer_e2e_tps",
        "error_count",
        "independent_process",
    }
    indexes: set[int] = set()
    pids: set[int] = set()
    speeds: list[float] = []
    refs: list[dict[str, Any]] = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == expected_row_fields, "run sample fields differ")
        index = row.get("process_index")
        require(index in {1, 2, 3} and index not in indexes, "run process index differs")
        indexes.add(index)
        command_path = validate_ref(row.get("command"), "run command", root=root)
        command = read_json(command_path, "run command")
        argv = command.get("argv")
        require(
            isinstance(argv, list)
            and len(argv) >= 2
            and argv.count("run") == 1,
            "run command does not execute ferrum run",
        )
        result_path = validate_ref(row.get("result"), "run result", root=root)
        result = read_json(result_path, "run result")
        require(set(result) == expected_result_fields, "run result fields differ")
        require(
            result.get("schema_version") == SCHEMA_VERSION
            and result.get("artifact_type") == "runtime_vnext_r2_run_sample"
            and normalize_source(result.get("source"), "run result") == source
            and result.get("model_key") == model_key
            and result.get("backend") == backend
            and result.get("binary_sha256") == binary_sha256
            and result.get("hardware_id") == hardware_id
            and result.get("model_sha256") == model_sha256
            and result.get("typed_config_sha256") == typed_config_sha256
            and result.get("status") == "pass"
            and result.get("error_count") == 0
            and result.get("independent_process") is True
            and isinstance(result.get("output_tokens"), int)
            and result["output_tokens"] > 0,
            "run sample identity/status differs",
        )
        pid = result.get("process_id")
        require(isinstance(pid, int) and pid > 0 and pid not in pids, "run process identity is not independent")
        pids.add(pid)
        speeds.append(finite_positive(result.get("steady_decode_tps"), "run steady decode"))
        finite_positive(result.get("engine_infer_e2e_tps"), "run E2E throughput")
        refs.extend([file_ref(command_path), file_ref(result_path)])
    require(indexes == {1, 2, 3}, "run process denominator differs")
    return {"sample_count": 3, "steady_decode_median": statistics.median(speeds), "artifacts": refs}


def load_calibration_lane(
    path: Path,
    *,
    expected_model: str,
    expected_backend: str,
    expected_source: dict[str, Any] | None,
) -> dict[str, Any]:
    """Independently validate collector data without consulting any floor."""
    manifest_path = input_manifest(
        path, "manifest.json", f"{expected_model}/{expected_backend} performance"
    )
    root = manifest_path.parent
    manifest = read_json(manifest_path, "performance lane manifest")
    required = {
        "schema_version",
        "artifact_type",
        "status",
        "created_at",
        "source",
        "model_key",
        "backend",
        "binary_sha256",
        "hardware_id",
        "model_sha256",
        "typed_config",
        "profile_mode",
        "hidden_env_names",
        "production_legacy_selection_count",
        "cells",
        "run_samples",
    }
    require(set(manifest) == required, "performance lane manifest fields differ")
    source = normalize_source(manifest.get("source"), "performance lane")
    require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type")
        == "runtime_vnext_r2_performance_lane_manifest"
        and manifest.get("status") == "ready"
        and (expected_source is None or source == expected_source)
        and manifest.get("model_key") == expected_model
        and manifest.get("backend") == expected_backend
        and isinstance(manifest.get("hardware_id"), str)
        and bool(manifest["hardware_id"])
        and manifest.get("profile_mode") == "off"
        and manifest.get("hidden_env_names") == []
        and manifest.get("production_legacy_selection_count") == 0,
        "performance lane identity/source/config differs",
    )
    created_at = parse_timestamp(
        manifest.get("created_at"), "performance lane created_at"
    )
    binary = require_sha(manifest.get("binary_sha256"), "performance binary")
    model_sha = require_sha(manifest.get("model_sha256"), "performance model")
    typed_config_path = validate_ref(
        manifest.get("typed_config"), "typed config", root=root
    )
    typed_config = read_json(typed_config_path, "typed config")
    require(
        set(typed_config)
        == {
            "schema_version",
            "model_key",
            "backend",
            "typed_active_cap",
            "memory_budget_bytes",
            "profile_detail",
            "hidden_env_names",
        }
        and typed_config.get("schema_version") == SCHEMA_VERSION
        and typed_config.get("model_key") == expected_model
        and typed_config.get("backend") == expected_backend
        and isinstance(typed_config.get("typed_active_cap"), int)
        and not isinstance(typed_config["typed_active_cap"], bool)
        and typed_config["typed_active_cap"]
        >= ACTIVE_FLOORS[(expected_model, expected_backend)]
        and finite_positive(
            typed_config.get("memory_budget_bytes"), "typed memory budget"
        )
        > 0
        and typed_config.get("profile_detail") == "off"
        and typed_config.get("hidden_env_names") == [],
        "typed config contract differs",
    )
    config_sha = sha256(typed_config_path)
    raw_cells = manifest.get("cells")
    expected = expected_cells(expected_backend)
    require(
        isinstance(raw_cells, list) and len(raw_cells) == len(expected),
        "performance cell denominator differs",
    )
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    for row in raw_cells:
        require(
            isinstance(row, dict)
            and set(row) == {"dataset", "concurrency", "report"},
            "performance cell reference fields differ",
        )
        concurrency = row.get("concurrency")
        require(
            isinstance(concurrency, int) and not isinstance(concurrency, bool),
            "performance cell concurrency differs",
        )
        key = (str(row.get("dataset")), concurrency)
        require(
            key in expected and key not in summaries,
            f"unexpected or duplicate performance cell: {key}",
        )
        report_path = validate_ref(
            row.get("report"), f"performance report {key}", root=root
        )
        summaries[key] = validate_performance_report(
            report_path,
            model_key=expected_model,
            backend=expected_backend,
            dataset=key[0],
            concurrency=key[1],
            source=source,
            binary_sha256=binary,
            hardware_id=manifest["hardware_id"],
            model_sha256=model_sha,
            typed_config_sha256=config_sha,
        )
    require(set(summaries) == expected, "performance cell tuple set differs")
    run = validate_run_samples(
        manifest.get("run_samples"),
        root=root,
        model_key=expected_model,
        backend=expected_backend,
        source=source,
        binary_sha256=binary,
        hardware_id=manifest["hardware_id"],
        model_sha256=model_sha,
        typed_config_sha256=config_sha,
    )
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": sha256(manifest_path),
        "source": source,
        "created_at": created_at,
        "binary_sha256": binary,
        "hardware_id": manifest["hardware_id"],
        "hardware_sha256": canonical_json_sha256(
            {"hardware_id": manifest["hardware_id"]}
        ),
        "model_sha256": model_sha,
        "typed_config_sha256": config_sha,
        "summaries": summaries,
        "run": run,
    }


def validate_collector_ref(root: Path, value: Any, label: str) -> Path:
    require(
        isinstance(value, dict)
        and set(value) >= {"kind", "path", "sha256", "size_bytes"},
        f"{label} collector reference fields differ",
    )
    raw = Path(str(value.get("path", "")))
    require(
        str(raw) and not raw.is_absolute() and ".." not in raw.parts,
        f"{label} collector path is invalid",
    )
    path = (root / raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise R2Error(f"{label} escaped the collector root") from error
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    require(
        isinstance(value.get("size_bytes"), int)
        and not isinstance(value["size_bytes"], bool)
        and value["size_bytes"] > 0
        and path.stat().st_size == value["size_bytes"]
        and require_sha(value.get("sha256"), f"{label}.sha256") == sha256(path),
        f"{label} size/SHA256 differs",
    )
    return path


def post_cell_nonnegative_int(value: Any, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a nonnegative integer",
    )
    return value


def validate_post_cell_occupancy_counter(
    value: Any, label: str
) -> dict[str, int]:
    require(isinstance(value, dict), f"{label} occupancy counter is missing")
    return {
        field: post_cell_nonnegative_int(value.get(field), f"{label}.{field}")
        for field in POST_CELL_OCCUPANCY_COUNTER_FIELDS
    }


def validate_post_cell_residency_occupancy(
    value: Any, label: str
) -> dict[str, dict[str, int]]:
    require(isinstance(value, dict), f"{label} residency occupancy is missing")
    normalized = {
        field: validate_post_cell_occupancy_counter(
            value.get(field), f"{label}.{field}"
        )
        for field in ("total", *POST_CELL_OCCUPANCY_SCOPE_FIELDS)
    }
    for metric in POST_CELL_OCCUPANCY_COUNTER_FIELDS:
        require(
            normalized["total"][metric]
            == sum(
                normalized[scope][metric]
                for scope in POST_CELL_OCCUPANCY_SCOPE_FIELDS
            ),
            f"{label} scope occupancy does not sum to total {metric}",
        )
    return normalized


def validate_post_cell_typed_ledger(
    body: Any,
    *,
    typed_active_cap: int,
    memory_budget_bytes: int,
    label: str,
) -> dict[str, Any]:
    """Validate one quiescent product health body and its typed pool ledger."""

    require(isinstance(body, dict), f"{label} health body is invalid")
    post_cell_nonnegative_int(typed_active_cap, f"{label} typed active cap")
    post_cell_nonnegative_int(memory_budget_bytes, f"{label} memory budget")
    require(typed_active_cap > 0, f"{label} typed active cap must be positive")
    require(memory_budget_bytes > 0, f"{label} memory budget must be positive")

    engine = body.get("engine")
    admission = body.get("admission")
    require(
        isinstance(engine, dict) and isinstance(admission, dict),
        f"{label} engine/admission health is missing",
    )
    for field in ("active_requests", "queued_requests"):
        require(
            post_cell_nonnegative_int(engine.get(field), f"{label} engine.{field}")
            == 0,
            f"{label} engine is not quiescent: {field}",
        )
    require(
        admission.get("runtime_snapshot_available") is True
        and admission.get("runtime_contract_error") is None
        and admission.get("resource_authority") == "plan_runtime",
        f"{label} admission runtime authority is invalid",
    )
    for field in (
        "queue_depth",
        "active_sequences",
        "active_prefill",
        "active_decode",
        "rejected_requests_total",
        "failed_requests_total",
    ):
        require(
            post_cell_nonnegative_int(
                admission.get(field), f"{label} admission.{field}"
            )
            == 0,
            f"{label} admission is not quiescent: {field}",
        )
    current_batch_size = admission.get("current_batch_size")
    require(
        current_batch_size is None
        or (
            isinstance(current_batch_size, int)
            and not isinstance(current_batch_size, bool)
            and current_batch_size == 0
        ),
        f"{label} admission current batch is active",
    )
    for field in (
        "effective_max_concurrent",
        "maximum_active_sequences",
        "preflight_effective_max_concurrent",
    ):
        require(
            post_cell_nonnegative_int(
                admission.get(field), f"{label} admission.{field}"
            )
            == typed_active_cap,
            f"{label} admission active ceiling differs: {field}",
        )

    cache = body.get("cache")
    executor = cache.get("prefix_cache") if isinstance(cache, dict) else None
    require(
        isinstance(executor, dict)
        and executor.get("schema") == "ferrum.runtime-vnext.executor-trace.v1",
        f"{label} vNext executor health is missing",
    )
    for field in (
        "pending_sequences",
        "active_sequences",
        "staged_prefill_requests",
        "staged_prefill_sequences",
        "pending_prefill_maintenance",
        "executing_prefills",
    ):
        require(
            post_cell_nonnegative_int(
                executor.get(field), f"{label} executor.{field}"
            )
            == 0,
            f"{label} executor is not quiescent: {field}",
        )
    deferred_cleanup = executor.get("deferred_cleanup")
    require(
        isinstance(deferred_cleanup, dict),
        f"{label} deferred cleanup status is missing",
    )
    for field in (
        "queued",
        "in_progress",
        "retryable",
        "quarantined",
        "panicked",
        "panicked_total",
    ):
        require(
            post_cell_nonnegative_int(
                deferred_cleanup.get(field), f"{label} deferred_cleanup.{field}"
            )
            == 0,
            f"{label} deferred cleanup is not quiescent: {field}",
        )

    static_bytes = post_cell_nonnegative_int(
        executor.get("static_bytes"), f"{label} static bytes"
    )
    require(static_bytes > 0, f"{label} static bytes must be positive")
    runtime_memory = executor.get("runtime_memory_policy")
    pools_status = executor.get("dynamic_pools")
    require(
        isinstance(runtime_memory, dict) and isinstance(pools_status, dict),
        f"{label} runtime memory/pool status is missing",
    )
    runtime_capacity = post_cell_nonnegative_int(
        runtime_memory.get("capacity_bytes"), f"{label} runtime capacity"
    )
    runtime_reserve = post_cell_nonnegative_int(
        runtime_memory.get("reserve_bytes"), f"{label} runtime reserve"
    )
    require(
        post_cell_nonnegative_int(
            runtime_memory.get("maximum_active_sequences"),
            f"{label} runtime maximum active sequences",
        )
        == typed_active_cap,
        f"{label} runtime active ceiling differs",
    )
    require(
        post_cell_nonnegative_int(
            pools_status.get("maximum_active_sequences"),
            f"{label} pool maximum active sequences",
        )
        == typed_active_cap,
        f"{label} pool active ceiling differs",
    )

    device_capacity = post_cell_nonnegative_int(
        pools_status.get("device_capacity_bytes"), f"{label} device capacity"
    )
    effective_ceiling = post_cell_nonnegative_int(
        pools_status.get("effective_device_usable_ceiling_bytes"),
        f"{label} effective device ceiling",
    )
    budget_ceiling = post_cell_nonnegative_int(
        pools_status.get("budget_device_wide_usable_ceiling_bytes"),
        f"{label} budget device ceiling",
    )
    process_claimed = post_cell_nonnegative_int(
        pools_status.get("process_claimed_bytes"), f"{label} process claim"
    )
    budget_claimed = post_cell_nonnegative_int(
        pools_status.get("budget_claimed_bytes"), f"{label} budget claim"
    )
    require(
        runtime_capacity == device_capacity
        and runtime_reserve <= runtime_capacity
        and runtime_capacity - runtime_reserve == budget_ceiling
        and effective_ceiling == budget_ceiling == memory_budget_bytes
        and budget_claimed == process_claimed <= effective_ceiling,
        f"{label} effective ceiling/process/budget claims differ",
    )

    pools = pools_status.get("pools")
    require(isinstance(pools, list) and pools, f"{label} dynamic pool list is empty")
    seen_pool_ids: set[str] = set()
    resident_total = 0
    used_total = 0
    live_physical_total = 0
    for index, pool in enumerate(pools, start=1):
        pool_label = f"{label} pool {index}"
        require(isinstance(pool, dict), f"{pool_label} status is invalid")
        pool_id = pool.get("pool_id")
        require(
            isinstance(pool_id, str)
            and bool(pool_id)
            and pool_id not in seen_pool_ids,
            f"{pool_label} identity is invalid",
        )
        seen_pool_ids.add(pool_id)
        require(
            post_cell_nonnegative_int(pool.get("domain_id"), f"{pool_label} domain")
            > 0,
            f"{pool_label} domain is invalid",
        )
        resident = post_cell_nonnegative_int(
            pool.get("resident_bytes"), f"{pool_label} resident bytes"
        )
        free = post_cell_nonnegative_int(
            pool.get("free_bytes"), f"{pool_label} free bytes"
        )
        resident_chunks = post_cell_nonnegative_int(
            pool.get("resident_chunks"), f"{pool_label} resident chunks"
        )
        live_segments = post_cell_nonnegative_int(
            pool.get("live_segments"), f"{pool_label} live segments"
        )
        largest_contiguous = post_cell_nonnegative_int(
            pool.get("largest_contiguous_bytes"),
            f"{pool_label} largest contiguous bytes",
        )
        require(
            free <= resident
            and largest_contiguous <= free
            and (resident == 0) == (resident_chunks == 0),
            f"{pool_label} resident/free/chunk accounting differs",
        )
        for field in (
            "pending_growth_bytes",
            "quarantined_bytes",
            "quarantined_chunks",
            "descriptor_mismatch_chunks",
            "publication_rejected_chunks",
        ):
            require(
                post_cell_nonnegative_int(pool.get(field), f"{pool_label}.{field}")
                == 0,
                f"{pool_label} contains pending/quarantined/rejected backing: {field}",
            )
        require(pool.get("poisoned") is False, f"{pool_label} is poisoned")

        live = pool.get("live_occupancy")
        require(isinstance(live, dict), f"{pool_label} live occupancy is missing")
        live_total = validate_post_cell_occupancy_counter(
            live.get("total"), f"{pool_label}.live.total"
        )
        transient = validate_post_cell_residency_occupancy(
            live.get("transient"), f"{pool_label}.live.transient"
        )
        lane_stable = validate_post_cell_residency_occupancy(
            live.get("lane_stable"), f"{pool_label}.live.lane_stable"
        )
        for metric in POST_CELL_OCCUPANCY_COUNTER_FIELDS:
            require(
                live_total[metric]
                == transient["total"][metric] + lane_stable["total"][metric],
                f"{pool_label} residency occupancy does not sum to live {metric}",
            )
            require(
                transient["total"][metric] == 0,
                f"{pool_label} transient {metric} remains after the cell",
            )
        used = resident - free
        require(
            live_total["physical_bytes"] == used
            and live_total["segment_count"] == live_segments,
            f"{pool_label} live ledger differs from allocator occupancy",
        )
        resident_total += resident
        used_total += used
        live_physical_total += live_total["physical_bytes"]

    require(
        budget_claimed == static_bytes + resident_total
        and used_total == live_physical_total,
        f"{label} aggregate pool/budget accounting differs",
    )
    return {
        "pool_count": len(pools),
        "resident_bytes": resident_total,
        "used_bytes": used_total,
        "budget_claimed_bytes": budget_claimed,
        "process_claimed_bytes": process_claimed,
        "effective_device_usable_ceiling_bytes": effective_ceiling,
        "typed_active_cap": typed_active_cap,
    }


def validate_legacy_post_cell_path(
    root: Path, relative: Any, digest: Any, label: str
) -> Path:
    raw = Path(str(relative or ""))
    require(
        str(raw) and not raw.is_absolute() and ".." not in raw.parts,
        f"{label} path is invalid",
    )
    path = (root / raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise R2Error(f"{label} escaped the collector root") from error
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    require(
        require_sha(digest, f"{label}.sha256") == sha256(path),
        f"{label} SHA256 differs",
    )
    return path


def validate_post_cell_probe_receipt(
    body_path: Path, receipt_path: Path, label: str
) -> dict[str, Any]:
    receipt = read_json(receipt_path, f"{label} receipt")
    require(
        receipt.get("returncode") == 0
        and receipt.get("http_status") == 200
        and receipt.get("body_sha256") == sha256(body_path)
        and receipt.get("body_size_bytes") == body_path.stat().st_size,
        f"{label} receipt/body binding differs",
    )
    return read_json(body_path, f"{label} health body")


def load_post_cell_health_bodies(
    root: Path,
    server: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    legacy_single_epoch: bool,
) -> list[dict[str, Any]]:
    """Load current checkpoint refs or the frozen legacy quiescence refs."""

    checkpoint_refs = server.get("completed_cell_checkpoints")
    if checkpoint_refs is not None:
        require(
            isinstance(checkpoint_refs, list)
            and len(checkpoint_refs) == len(records),
            "Ferrum completed-cell checkpoint denominator differs",
        )
        bodies: list[dict[str, Any]] = []
        for record, checkpoint_ref in zip(records, checkpoint_refs, strict=True):
            label = f"Ferrum post-cell {record.get('cell_id')}"
            checkpoint_path = validate_collector_ref(
                root, checkpoint_ref, f"{label} checkpoint"
            )
            checkpoint = read_json(checkpoint_path, f"{label} checkpoint")
            require(
                checkpoint.get("contract")
                == "ferrum.runtime-vnext.r2.completed-cell-checkpoint.v1"
                and checkpoint.get("sequence") == record.get("sequence")
                and checkpoint.get("cell_id") == record.get("cell_id"),
                f"{label} checkpoint/record binding differs",
            )
            post = checkpoint.get("post_cell_idle")
            require(
                isinstance(post, dict) and post.get("status") == "quiescent",
                f"{label} checkpoint is not quiescent",
            )
            body_path = validate_collector_ref(
                root, post.get("body"), f"{label} body"
            )
            receipt_path = validate_collector_ref(
                root, post.get("receipt"), f"{label} receipt"
            )
            bodies.append(
                validate_post_cell_probe_receipt(body_path, receipt_path, label)
            )
        return bodies

    require(legacy_single_epoch, "Ferrum post-cell checkpoints are missing")
    session = server.get("session")
    quiescence = session.get("cell_quiescence") if isinstance(session, dict) else None
    require(
        isinstance(quiescence, list) and len(quiescence) == len(records),
        "legacy Ferrum cell-quiescence denominator differs",
    )
    bodies = []
    for record, row in zip(records, quiescence, strict=True):
        label = f"legacy Ferrum post-cell {record.get('cell_id')}"
        require(
            isinstance(row, dict)
            and row.get("status") == "quiescent"
            and row.get("active_requests") == 0
            and row.get("queued_requests") == 0
            and row.get("admission_queue_depth") == 0,
            f"{label} quiescence receipt differs",
        )
        probe = row.get("probe")
        require(isinstance(probe, dict), f"{label} probe is missing")
        expected_prefix = (
            f"post-{record.get('sequence'):02d}-{record.get('dataset')}"
            f"-c{record.get('concurrency')}-"
        )
        body_relative = probe.get("body")
        receipt_relative = probe.get("receipt")
        require(
            isinstance(body_relative, str)
            and isinstance(receipt_relative, str)
            and Path(body_relative).name.startswith(expected_prefix)
            and Path(body_relative).name.endswith(".body.json")
            and Path(receipt_relative).name.startswith(expected_prefix)
            and Path(receipt_relative).name.endswith(".receipt.json"),
            f"{label} probe/record path binding differs",
        )
        body_path = validate_legacy_post_cell_path(
            root, body_relative, probe.get("body_sha256"), f"{label} body"
        )
        receipt_path = validate_legacy_post_cell_path(
            root,
            receipt_relative,
            probe.get("receipt_sha256"),
            f"{label} receipt",
        )
        bodies.append(validate_post_cell_probe_receipt(body_path, receipt_path, label))
    return bodies


def ferrum_collector_root(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    plan = manifest.get("plan")
    require(isinstance(plan, dict), "Ferrum collector plan reference is missing")
    relative = Path(str(plan.get("path", "")))
    require(
        str(relative)
        and not relative.is_absolute()
        and ".." not in relative.parts
        and relative.name == "plan.json",
        "Ferrum collector plan path is invalid",
    )
    root = manifest_path.parent
    for _ in relative.parent.parts:
        root = root.parent
    require(
        (root / relative).resolve().parent == manifest_path.parent,
        "Ferrum collector artifact-root inference failed",
    )
    validate_collector_ref(root, plan, "Ferrum collector plan")
    return root.resolve()


TERMINAL_CUDA_EXIT_RECOVERY_CONTRACT = (
    "ferrum.runtime-vnext.r2.cuda-terminal-exit-recovery.v1"
)
TERMINAL_CUDA_EXIT_REJECTION = (
    "R2CollectorError: CUDA bridge server PID left its process group"
)


def read_jsonl_objects(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise R2Error(f"invalid {label} JSONL {path}: {error}") from error
    require(
        rows and all(isinstance(row, dict) for row in rows),
        f"{label} must contain JSON objects",
    )
    return rows


def parsed_utc_timestamp(value: Any, label: str) -> datetime:
    require(isinstance(value, str) and value, f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise R2Error(f"{label} is not ISO-8601: {value}") from error
    require(parsed.tzinfo is not None, f"{label} lacks a timezone")
    return parsed.astimezone(timezone.utc)


def validate_terminal_cuda_exit_recovery(
    root: Path,
    bundle: dict[str, Any],
    *,
    fingerprint: str,
    sample_ordinal: int,
    recovery: dict[str, Any],
    expected_collector_sha256: str,
) -> dict[str, Any]:
    """Accept one immutable legacy audit race without weakening live-PID checks."""

    try:
        import runtime_vnext_r2_ferrum_collector as ferrum_collector
        import runtime_vnext_resource_sampler as resource_sampler
    except ModuleNotFoundError as error:
        raise R2Error(f"Ferrum recovery validator import failed: {error}") from error

    require(
        bundle.get("schema_version") == 1
        and bundle.get("config_fingerprint") == fingerprint,
        "Ferrum recovered run bundle identity differs",
    )
    sample = bundle.get("sample")
    require(
        isinstance(sample, dict)
        and sample.get("sample_ordinal") == sample_ordinal
        and sample.get("independent_process") is True
        and sample.get("returncode") == 0,
        "Ferrum recovered run sample did not exit successfully",
    )
    stdout_path = validate_collector_ref(
        root, sample.get("stdout"), "recovered run stdout"
    )
    arrival_path = validate_collector_ref(
        root, sample.get("arrival_timeline"), "recovered run arrival timeline"
    )
    stderr_path = validate_collector_ref(
        root, sample.get("stderr"), "recovered run stderr"
    )
    validate_collector_ref(
        root, sample.get("product_effective_config"), "recovered run effective config"
    )
    process_receipt_ref = sample.get("process_receipt")
    require(
        isinstance(process_receipt_ref, dict)
        and isinstance(process_receipt_ref.get("path"), str)
        and isinstance(process_receipt_ref.get("sha256"), str),
        "recovered run process receipt reference is invalid",
    )
    process_receipt_path = (root / process_receipt_ref["path"]).resolve()
    try:
        process_receipt_path.relative_to(root.resolve())
    except ValueError as error:
        raise R2Error("recovered run process receipt escaped artifact root") from error
    require(
        process_receipt_path.is_file()
        and not process_receipt_path.is_symlink()
        and sha256(process_receipt_path) == process_receipt_ref["sha256"],
        "recovered run process receipt differs",
    )
    process_receipt = read_json(process_receipt_path, "recovered run process receipt")
    require(
        process_receipt.get("pid") == sample.get("pid")
        and process_receipt.get("pgid") == sample.get("pgid")
        and process_receipt.get("process_start_marker")
        == sample.get("process_start_marker")
        and process_receipt.get("argv") == sample.get("argv")
        and process_receipt.get("environment") == sample.get("environment"),
        "recovered run process receipt identity differs",
    )

    resources = sample.get("resources")
    require(isinstance(resources, dict), "recovered run resources are missing")
    observations_path = validate_collector_ref(
        root, resources.get("observations"), "recovered run resource observations"
    )
    observation_rows = read_jsonl_objects(
        observations_path, "recovered run resource observations"
    )
    header, footer = observation_rows[0], observation_rows[-1]
    samples = observation_rows[1:-1]
    require(
        samples
        and footer.get("record_type") == "footer"
        and footer.get("exit_reason") == "process-exit",
        "recovered run lacks a process-exit resource footer",
    )
    resource_summary = resources.get("summary")
    require(isinstance(resource_summary, dict), "recovered run resource summary is missing")
    recomputed_resource = resource_sampler.derive_summary(
        observations_path,
        session_id=sample["sample_id"],
        cell_id="run:c1",
        backend="cuda",
        hardware_id=sample["hardware"]["id"],
        pid=sample["pid"],
        pgid=sample["pgid"],
        process_start_marker=sample["process_start_marker"],
        base_url=f"process://{sample['sample_id']}",
        session_started_at=sample["started_at"],
        session_finished_at=sample["finished_at"],
        measurement_started_at=samples[0]["sampled_at"],
        measurement_finished_at=samples[-1]["sampled_at"],
        memory_budget_bytes=resource_summary["memory_budget_bytes"],
        requested_concurrency=1,
        typed_active_cap=1,
        runtime_log_path=header["runtime_log_path"],
        runtime_log_evidence_path=stderr_path,
    )
    require(
        recomputed_resource == resource_summary
        and resource_summary.get("oom_count") == 0
        and resource_summary.get("admission_error_count") == 0,
        "recovered run resources are not raw-derived or contain runtime errors",
    )

    bridge = resources.get("cuda_pid_namespace_bridge")
    require(isinstance(bridge, dict), "recovered run CUDA bridge is missing")
    original_audit_path = validate_collector_ref(
        root, bridge.get("audit"), "recovered run original CUDA audit"
    )
    original_rows = read_jsonl_objects(
        original_audit_path, "recovered run original CUDA audit"
    )
    terminal = original_rows[-1]
    prefix_ref = recovery.get("accepted_prefix_audit")
    prefix_path = validate_collector_ref(
        root, prefix_ref, "recovered run accepted CUDA audit prefix"
    )
    prefix_rows = read_jsonl_objects(
        prefix_path, "recovered run accepted CUDA audit prefix"
    )
    require(
        prefix_rows == original_rows[:-1],
        "recovered CUDA audit prefix is not the immutable original prefix",
    )
    prefix_resources = copy.deepcopy(resources)
    prefix_resources["cuda_pid_namespace_bridge"]["audit"] = copy.deepcopy(prefix_ref)
    ferrum_collector.validate_cuda_bridge_evidence(
        root,
        prefix_resources,
        backend="cuda",
        label=f"recovered run sample {sample_ordinal} prefix",
        expected_collector_sha256=expected_collector_sha256,
    )

    compute_options = set(ferrum_collector.CUDA_COMPUTE_QUERY)
    prior_compute = [
        row
        for row in prefix_rows
        if compute_options <= set(row.get("nvidia_smi_argv", []))
        and row.get("status") == "pass"
        and row.get("strategy")
        in {
            "native-process-group-pid",
            "single-new-host-pid-mapped-to-container-server",
        }
    ]
    require(len(prior_compute) >= 3, "recovered CUDA audit lacks prior mapped samples")
    terminal_raw = ferrum_collector.parse_cuda_compute_rows(
        str(terminal.get("raw_stdout", "")), "terminal CUDA compute query"
    )
    require(
        terminal is original_rows[-1]
        and compute_options <= set(terminal.get("nvidia_smi_argv", []))
        and terminal.get("contract") == bridge.get("contract")
        and terminal.get("collector_sha256") == bridge.get("bridge_source_sha256")
        and terminal.get("server_pid") == sample.get("pid")
        and terminal.get("server_pgid") == sample.get("pgid")
        and terminal.get("status") == "reject"
        and terminal.get("error") == TERMINAL_CUDA_EXIT_REJECTION
        and terminal.get("real_returncode") == 0
        and terminal.get("raw_stderr") == ""
        and len(terminal_raw) == 1,
        "recovered CUDA terminal rejection shape differs",
    )
    prior_raw_for_pid = [
        app["used_gpu_memory_mib"]
        for row in prior_compute
        for app in row.get("raw_compute_apps", [])
        if isinstance(app, dict) and app.get("pid") == terminal_raw[0]["pid"]
    ]
    require(
        prior_raw_for_pid
        and terminal_raw[0]["used_gpu_memory_mib"] > 0
        and terminal_raw[0]["used_gpu_memory_mib"] < prior_raw_for_pid[-1],
        "recovered CUDA terminal row is not a draining previously mapped PID",
    )
    last_sample_at = parsed_utc_timestamp(samples[-1]["sampled_at"], "last resource sample")
    terminal_at = parsed_utc_timestamp(terminal.get("observed_at"), "terminal CUDA audit")
    footer_at = parsed_utc_timestamp(footer.get("finished_at"), "resource footer")
    sample_finished_at = parsed_utc_timestamp(sample.get("finished_at"), "run finish")
    require(
        last_sample_at <= terminal_at <= footer_at <= sample_finished_at
        and (footer_at - terminal_at).total_seconds() <= 1.0,
        "recovered CUDA rejection is not the terminal process-exit edge",
    )

    require(
        recovery.get("schema_version") == 1
        and recovery.get("contract") == TERMINAL_CUDA_EXIT_RECOVERY_CONTRACT
        and recovery.get("artifact_type")
        == "runtime_vnext_r2_cuda_terminal_exit_recovery"
        and recovery.get("status") == "pass"
        and recovery.get("sample_ordinal") == sample_ordinal
        and isinstance(recovery.get("run_sample"), dict)
        and recovery.get("raw_evidence_mutated") is False
        and recovery.get("original_audit") == bridge.get("audit")
        and recovery.get("terminal_rejection_sha256")
        == canonical_json_sha256(terminal),
        "Ferrum terminal-exit recovery receipt differs",
    )
    recovered_bundle_path = validate_collector_ref(
        root, recovery.get("run_sample"), "recovered run sample bundle"
    )
    require(
        read_json(recovered_bundle_path, "recovered run sample bundle") == bundle,
        "Ferrum terminal-exit recovery references a different run bundle",
    )
    recovery_source = recovery.get("recovery_source")
    classifier_source = recovery.get("classifier_source")
    recovery_script = REPO_ROOT / "scripts/release/runtime_vnext_r2_ferrum_terminal_recovery.py"
    require(
        isinstance(recovery_source, dict)
        and recovery_source.get("path")
        == "scripts/release/runtime_vnext_r2_ferrum_terminal_recovery.py"
        and recovery_source.get("sha256") == sha256(recovery_script)
        and isinstance(classifier_source, dict)
        and classifier_source.get("path")
        == SCRIPT_PATH.relative_to(REPO_ROOT).as_posix(),
        "Ferrum terminal-exit recovery source identity differs",
    )
    require_recorded_source_in_history(
        classifier_source["path"],
        classifier_source.get("sha256"),
        "Ferrum terminal-exit recovery classifier source",
    )
    postflight_path = validate_collector_ref(
        root, recovery.get("cuda_idle_postflight"), "recovered CUDA idle postflight"
    )
    postflight = read_json(postflight_path, "recovered CUDA idle postflight")
    require(
        postflight.get("returncode") == 0
        and postflight.get("compute_apps") == []
        and postflight.get("gpu_uuids")
        == read_json(
            validate_collector_ref(root, bridge.get("preflight"), "recovered CUDA preflight"),
            "recovered CUDA preflight",
        ).get("gpu_uuids"),
        "recovered CUDA idle postflight did not prove context cleanup",
    )

    events = read_jsonl_objects(stdout_path, "recovered run stdout")
    arrivals = read_jsonl_objects(arrival_path, "recovered run arrival timeline")
    metrics = ferrum_collector.validate_run_events(
        events, arrivals, f"run sample {sample_ordinal}"
    )
    require(metrics == sample.get("metrics"), "recovered run metrics differ")
    return {
        "sample_ordinal": sample_ordinal,
        "terminal_used_gpu_memory_mib": terminal_raw[0]["used_gpu_memory_mib"],
        "prior_used_gpu_memory_mib": prior_raw_for_pid[-1],
        "terminal_rejection_sha256": canonical_json_sha256(terminal),
        "resource_exit_reason": resource_summary["exit_reason"],
    }


def default_collector_verifier(
    root: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
    server: dict[str, Any],
    runs: list[dict[str, Any]],
) -> None:
    try:
        import runtime_vnext_r2_ferrum_collector as ferrum_collector

        fingerprint = str(manifest.get("config_fingerprint", ""))
        ferrum_collector.validate_final_manifest(root, manifest, fingerprint)
        inputs = {
            "binary_path": validate_collector_ref(
                root, manifest.get("inputs", {}).get("binary"), "collector input binary"
            ),
            "binary": copy.deepcopy(manifest.get("inputs", {}).get("binary")),
            "tokenizer_path": validate_collector_ref(
                root, manifest.get("inputs", {}).get("tokenizer"), "collector input tokenizer"
            ),
            "realistic_dataset_path": validate_collector_ref(
                root,
                manifest.get("inputs", {}).get("realistic_dataset"),
                "collector input realistic dataset",
            ),
            "run_parity_dataset_path": validate_collector_ref(
                root,
                manifest.get("inputs", {}).get("run_parity_dataset"),
                "collector input run parity dataset",
            ),
        }
        ferrum_collector.validate_server_bundle(
            root, server, fingerprint, config, inputs
        )
        recovery_ref = manifest.get("terminal_exit_recovery")
        recovery = None
        recovery_ordinal = None
        if recovery_ref is not None:
            recovery_path = validate_collector_ref(
                root, recovery_ref, "Ferrum terminal-exit recovery"
            )
            recovery = read_json(recovery_path, "Ferrum terminal-exit recovery")
            recovery_ordinal = recovery.get("sample_ordinal")
            require(
                isinstance(recovery_ordinal, int)
                and not isinstance(recovery_ordinal, bool)
                and 1 <= recovery_ordinal <= len(runs),
                "Ferrum terminal-exit recovery sample ordinal is invalid",
            )
        legacy_epoch = ferrum_collector.frozen_collection_epoch(
            ferrum_collector.read_json(
                ferrum_collector.lane_dir(root, config) / "plan.json"
            )
        )
        for ordinal, bundle in enumerate(runs, start=1):
            if ordinal == recovery_ordinal:
                require(recovery is not None, "Ferrum terminal-exit recovery is missing")
                validate_terminal_cuda_exit_recovery(
                    root,
                    bundle,
                    fingerprint=fingerprint,
                    sample_ordinal=ordinal,
                    recovery=recovery,
                    expected_collector_sha256=legacy_epoch["collector_sha256"],
                )
            else:
                ferrum_collector.validate_run_bundle(
                    root,
                    bundle,
                    fingerprint,
                    ordinal,
                    legacy_collection_epoch=legacy_epoch,
                )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise R2Error(f"Ferrum collector provenance failed: {error}") from error


def validate_request_sidecar(
    root: Path,
    value: Any,
    *,
    report_ref: dict[str, Any],
    cell: dict[str, Any],
    report: dict[str, Any],
) -> int:
    path = validate_collector_ref(root, value, "raw request sidecar")
    sidecar = read_json(path, "raw request sidecar")
    require(
        sidecar.get("schema_version") == SCHEMA_VERSION
        and sidecar.get("artifact_type")
        == "runtime_vnext_r2_bench_request_evidence_sidecar"
        and sidecar.get("source_report") == report_ref
        and sidecar.get("cell") == cell
        and sidecar.get("output_token_count_source") == "usage"
        and sidecar.get("per_request_token_and_itl_evidence_complete") is True,
        "raw request sidecar identity/completeness differs",
    )
    rows = sidecar.get("repeats")
    require(isinstance(rows, list) and len(rows) == 3, "request sidecar repeat denominator differs")
    measured = 0
    for repeat, row in enumerate(rows, start=1):
        require(
            isinstance(row, dict)
            and row.get("repeat") == repeat
            and row.get("aggregate_metrics") == report["repeat_metrics"][repeat - 1],
            f"request sidecar repeat {repeat} aggregate differs",
        )
        requests = row.get("requests")
        require(
            isinstance(requests, list) and len(requests) == cell["num_prompts"],
            f"request sidecar repeat {repeat} denominator differs",
        )
        for ordinal, request in enumerate(requests, start=1):
            require(
                isinstance(request, dict)
                and request.get("request_ordinal") == ordinal
                and request.get("output_token_count_source") == "usage"
                and isinstance(request.get("actual_input_tokens"), int)
                and not isinstance(request["actual_input_tokens"], bool)
                and request["actual_input_tokens"] > 0
                and isinstance(request.get("usage_output_tokens"), int)
                and not isinstance(request["usage_output_tokens"], bool)
                and request["usage_output_tokens"] > 0
                and isinstance(request.get("itl_evidence"), dict),
                f"request sidecar repeat {repeat} request {ordinal} is incomplete",
            )
            if cell["dataset"] == "random":
                require(
                    request["actual_input_tokens"] == cell["input_tokens"]
                    and request["usage_output_tokens"] == cell["output_tokens"],
                    f"random request lengths differ at repeat {repeat}/{ordinal}",
                )
            else:
                require(
                    request["usage_output_tokens"] <= cell["output_tokens"],
                    f"realistic request output exceeds the locked maximum at repeat {repeat}/{ordinal}",
                )
        require(
            sum(request["actual_input_tokens"] for request in requests)
            == report["repeat_metrics"][repeat - 1]["actual_input_tokens"]
            and sum(request["usage_output_tokens"] for request in requests)
            == report["repeat_metrics"][repeat - 1]["output_tokens"],
            f"request sidecar repeat {repeat} token aggregates differ",
        )
        measured += len(requests)
    return measured


def load_product_request_lifecycle(path: Path) -> dict[str, dict[str, int]]:
    """Load the immutable product request boundaries from one scheduler trace."""

    lifecycle: dict[str, dict[str, int]] = {}
    accepted_phase = "vnext.request_accepted"
    completed_phase = "vnext.request_completed"
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if accepted_phase not in line and completed_phase not in line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise R2Error(
                    f"scheduler lifecycle JSON is invalid at line {line_number}"
                ) from error
            phase = event.get("phase")
            if phase not in {accepted_phase, completed_phase}:
                continue
            request_id = event.get("request_id")
            if not isinstance(request_id, str) or not request_id.startswith(
                "request.product."
            ):
                continue
            attributes = event.get("attributes")
            require(
                event.get("entrypoint") == "serve"
                and event.get("status") == "ok"
                and isinstance(attributes, dict)
                and attributes.get("execution_request_origin") == "product",
                f"scheduler lifecycle identity differs at line {line_number}",
            )
            timestamp = event.get("ts_unix_nanos")
            require(
                isinstance(timestamp, int)
                and not isinstance(timestamp, bool)
                and timestamp > 0,
                f"scheduler lifecycle timestamp is invalid at line {line_number}",
            )
            kind = "accepted_ns" if phase == accepted_phase else "completed_ns"
            slots = lifecycle.setdefault(request_id, {})
            require(
                kind not in slots,
                f"scheduler lifecycle has duplicate {kind}: {request_id}",
            )
            slots[kind] = timestamp
    require(lifecycle, "scheduler trace contains no product request lifecycle")
    for request_id, slots in lifecycle.items():
        require(
            set(slots) == {"accepted_ns", "completed_ns"},
            f"scheduler lifecycle pair is incomplete: {request_id}",
        )
        require(
            slots["completed_ns"] >= slots["accepted_ns"],
            f"scheduler lifecycle completion precedes acceptance: {request_id}",
        )
    return lifecycle


def closed_loop_request_groups(
    lifecycle: dict[str, dict[str, int]], record: dict[str, Any]
) -> list[dict[str, Any]]:
    """Split a bench cell into the strict warmup/measured barriers it executed."""

    started_ns = timestamp_nanos(record.get("started_at"), "cell started_at")
    finished_ns = timestamp_nanos(record.get("finished_at"), "cell finished_at")
    counts = [record.get("warmup_requests"), record.get("num_prompts")] * int(
        record.get("n_repeats", 0)
    )
    require(
        finished_ns > started_ns
        and counts
        and all(isinstance(count, int) and count > 0 for count in counts),
        "cell lifecycle window/counts are invalid",
    )
    selected = sorted(
        (
            (request_id, slots)
            for request_id, slots in lifecycle.items()
            if started_ns <= slots["accepted_ns"] <= finished_ns
        ),
        key=lambda item: (item[1]["accepted_ns"], item[0]),
    )
    require(
        len(selected) == sum(counts)
        and all(slots["completed_ns"] <= finished_ns for _, slots in selected),
        "cell lifecycle request denominator/window differs",
    )
    groups: list[dict[str, Any]] = []
    offset = 0
    for count in counts:
        chunk = selected[offset : offset + count]
        offset += count
        request_ids = [request_id for request_id, _ in chunk]
        events: dict[int, int] = {}
        for _, slots in chunk:
            events[slots["accepted_ns"]] = events.get(slots["accepted_ns"], 0) + 1
            events[slots["completed_ns"]] = events.get(slots["completed_ns"], 0) - 1
        outstanding = 0
        previous_ns: int | None = None
        intervals: list[dict[str, int]] = []
        for event_ns in sorted(events):
            if previous_ns is not None and event_ns > previous_ns and outstanding > 0:
                intervals.append(
                    {
                        "start_unix_nanos": previous_ns,
                        "end_unix_nanos": event_ns,
                        "outstanding_request_count": outstanding,
                    }
                )
            outstanding += events[event_ns]
            require(outstanding >= 0, "cell lifecycle completion precedes acceptance")
            previous_ns = event_ns
        require(outstanding == 0 and intervals, "cell lifecycle group is incomplete")
        group = {
            "started_unix_nanos": min(slots["accepted_ns"] for _, slots in chunk),
            "finished_unix_nanos": max(slots["completed_ns"] for _, slots in chunk),
            "request_ids": request_ids,
            "outstanding_intervals": intervals,
        }
        if groups:
            require(
                groups[-1]["finished_unix_nanos"] <= group["started_unix_nanos"],
                "cell warmup/measured phase barriers overlap",
            )
        groups.append(group)
    return groups


def replay_active_floor_duty(
    *,
    groups: list[dict[str, Any]],
    active_rows: list[dict[str, Any]],
    active_floor: int,
) -> dict[str, Any]:
    """Intersect measured outstanding>=floor time with conservative probes."""

    measured_groups = groups[1::2]
    require(measured_groups, "measured request groups are missing")
    replay_rows: list[dict[str, Any]] = []
    total_eligible_ns = 0
    total_at_floor_ns = 0
    for repeat, group in enumerate(measured_groups, start=1):
        eligible_intervals = [
            copy.deepcopy(interval)
            for interval in group["outstanding_intervals"]
            if interval["outstanding_request_count"] >= active_floor
        ]
        require(
            eligible_intervals,
            f"repeat {repeat} never has {active_floor} outstanding requests",
        )
        eligible_ns = 0
        at_floor_ns = 0
        observation_indexes: set[int] = set()
        for interval in eligible_intervals:
            start_ns = interval["start_unix_nanos"]
            end_ns = interval["end_unix_nanos"]
            expected_ns = end_ns - start_ns
            covered_ns = 0
            for row_index, row in enumerate(active_rows):
                overlap_start = max(start_ns, row["started_unix_nanos"])
                overlap_end = min(end_ns, row["finished_unix_nanos"])
                if overlap_end <= overlap_start:
                    continue
                overlap_ns = overlap_end - overlap_start
                require(
                    row["eligible"],
                    f"repeat {repeat} eligible lifecycle interval lacks a valid active probe",
                )
                covered_ns += overlap_ns
                observation_indexes.add(row_index)
                if row["active_requests_conservative"] >= active_floor:
                    at_floor_ns += overlap_ns
            require(
                covered_ns == expected_ns,
                f"repeat {repeat} active probes do not fully cover eligible lifecycle time",
            )
            eligible_ns += expected_ns
        require(eligible_ns > 0, f"repeat {repeat} eligible duration is zero")
        duty = at_floor_ns / eligible_ns
        replay_rows.append(
            {
                "repeat": repeat,
                "warmup_request_count": len(groups[(repeat - 1) * 2]["request_ids"]),
                "measured_request_count": len(group["request_ids"]),
                "warmup_request_id_set_sha256": canonical_json_sha256(
                    sorted(groups[(repeat - 1) * 2]["request_ids"])
                ),
                "measured_request_id_set_sha256": canonical_json_sha256(
                    sorted(group["request_ids"])
                ),
                "eligible_intervals": eligible_intervals,
                "eligible_duration_ns": eligible_ns,
                "active_at_or_above_floor_duration_ns": at_floor_ns,
                "active_duty_cycle": duty,
                "active_interval_count": len(observation_indexes),
            }
        )
        total_eligible_ns += eligible_ns
        total_at_floor_ns += at_floor_ns
    return {
        "schema_version": 1,
        "algorithm": "measured-product-outstanding-intersect-conservative-active-v1",
        "request_group_counts": [len(group["request_ids"]) for group in groups],
        "repeats": replay_rows,
        "eligible_duration_ns": total_eligible_ns,
        "active_at_or_above_floor_duration_ns": total_at_floor_ns,
        "active_duty_cycle": total_at_floor_ns / total_eligible_ns,
    }


def derive_active_interval_rows(
    samples: list[dict[str, Any]], record: dict[str, Any]
) -> list[dict[str, Any]]:
    """Reproduce the collector sidecar directly from immutable probe samples."""

    require(len(samples) >= 2, "resource observations contain fewer than two samples")
    measurement_start = timestamp_datetime(record.get("started_at"), "cell started_at")
    measurement_finish = timestamp_datetime(record.get("finished_at"), "cell finished_at")
    require(measurement_finish > measurement_start, "cell measurement window is invalid")
    rows: list[dict[str, Any]] = []
    for sample_index, (left, right) in enumerate(
        zip(samples, samples[1:]), start=1
    ):
        require(
            isinstance(left, dict) and isinstance(right, dict),
            f"resource sample pair {sample_index} is invalid",
        )
        left_at = timestamp_datetime(
            left.get("sampled_at"), f"resource sample {sample_index} sampled_at"
        )
        right_at = timestamp_datetime(
            right.get("sampled_at"),
            f"resource sample {sample_index + 1} sampled_at",
        )
        clipped_start = max(left_at, measurement_start)
        clipped_finish = min(right_at, measurement_finish)
        duration_ms = (clipped_finish - clipped_start).total_seconds() * 1000.0
        if duration_ms <= 0:
            continue
        left_errors = left.get("active_probe_errors", [])
        right_errors = right.get("active_probe_errors", [])
        require(
            isinstance(left_errors, list) and isinstance(right_errors, list),
            f"resource sample pair {sample_index} probe errors are invalid",
        )
        errors = [*left_errors, *right_errors]
        eligible = (
            left.get("process_alive") is True
            and right.get("process_alive") is True
            and not errors
            and isinstance(left.get("active_requests"), int)
            and isinstance(right.get("active_requests"), int)
        )
        rows.append(
            {
                "sequence": len(rows) + 1,
                "started_at": clipped_start.isoformat().replace("+00:00", "Z"),
                "finished_at": clipped_finish.isoformat().replace("+00:00", "Z"),
                "duration_ms": duration_ms,
                "eligible": eligible,
                "active_requests_conservative": (
                    min(left["active_requests"], right["active_requests"])
                    if eligible
                    else None
                ),
                "left_sample_sequence": left.get("sequence"),
                "right_sample_sequence": right.get("sequence"),
                "probe_errors": errors,
            }
        )
    require(rows, "resource observations produce no active intervals")
    return rows


def load_raw_active_interval_rows(
    observations_path: Path, record: dict[str, Any]
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    try:
        with observations_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                require(
                    isinstance(row, dict),
                    f"resource observation row {line_number} is invalid",
                )
                if row.get("record_type") == "sample":
                    samples.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise R2Error(
            f"cannot derive active intervals from resource observations: {error}"
        ) from error
    return derive_active_interval_rows(samples, record)


def require_raw_derived_active_rows(
    rows: Any, expected_rows: list[dict[str, Any]]
) -> None:
    require(rows == expected_rows, "active interval sidecar is not raw-derived")


def require_repeat_active_floor_duty(evidence: dict[str, Any]) -> None:
    require(
        all(
            row["active_duty_cycle"] >= 0.80 - 1e-12
            for row in evidence["repeats"]
        ),
        "highest-concurrency active-floor duty is below 80%",
    )


def validate_active_intervals(
    root: Path,
    value: Any,
    *,
    observation_ref: dict[str, Any],
    observations_path: Path,
    record: dict[str, Any],
    active_floor: int,
    require_floor: bool,
    product_lifecycle: dict[str, dict[str, int]] | None = None,
    scheduler_trace_ref: dict[str, Any] | None = None,
    report_ref: dict[str, Any] | None = None,
    typed_active_cap: int | None = None,
) -> tuple[float, dict[str, Any] | None]:
    path = validate_collector_ref(root, value, "active interval sidecar")
    document = read_json(path, "active interval sidecar")
    require(
        document.get("schema_version") == SCHEMA_VERSION
        and document.get("artifact_type")
        == "runtime_vnext_r2_active_interval_sidecar"
        and document.get("source_observations") == observation_ref
        and document.get("cell_id") == record.get("cell_id")
        and document.get("measurement_started_at") == record.get("started_at")
        and document.get("measurement_finished_at") == record.get("finished_at"),
        "active interval sidecar identity differs",
    )
    rows = document.get("intervals")
    require(isinstance(rows, list) and rows, "active interval timeline is empty")
    require_raw_derived_active_rows(
        rows, load_raw_active_interval_rows(observations_path, record)
    )
    eligible = 0.0
    at_floor = 0.0
    total = 0.0
    active_rows: list[dict[str, Any]] = []
    previous_finish_ns: int | None = None
    for sequence, row in enumerate(rows, start=1):
        require(
            isinstance(row, dict)
            and row.get("sequence") == sequence
            and isinstance(row.get("eligible"), bool),
            "active interval identity differs",
        )
        duration = finite_positive(row.get("duration_ms"), "active interval duration")
        started_ns = timestamp_nanos(
            row.get("started_at"), f"active interval {sequence} started_at"
        )
        finished_ns = timestamp_nanos(
            row.get("finished_at"), f"active interval {sequence} finished_at"
        )
        require(
            finished_ns > started_ns
            and (previous_finish_ns is None or started_ns >= previous_finish_ns),
            "active interval timestamps overlap or are not increasing",
        )
        previous_finish_ns = finished_ns
        total += duration
        if row["eligible"]:
            active = row.get("active_requests_conservative")
            require(
                isinstance(active, int) and not isinstance(active, bool) and active >= 0,
                "eligible active interval lacks a conservative active count",
            )
            eligible += duration
            if active >= active_floor:
                at_floor += duration
        else:
            require(
                row.get("active_requests_conservative") is None,
                "ineligible active interval has an active count",
            )
        active_rows.append(
            {
                "started_unix_nanos": started_ns,
                "finished_unix_nanos": finished_ns,
                "eligible": row["eligible"],
                "active_requests_conservative": row.get(
                    "active_requests_conservative"
                ),
            }
        )
    require(
        math.isclose(
            finite_positive(document.get("eligible_duration_ms"), "eligible interval duration"),
            eligible,
            abs_tol=1e-6,
        )
        and math.isclose(
            finite_positive(document.get("total_interval_duration_ms"), "total interval duration"),
            total,
            abs_tol=1e-6,
        ),
        "active interval duration summary differs",
    )
    fraction = at_floor / eligible
    if not require_floor:
        return fraction, None
    require(
        product_lifecycle is not None
        and isinstance(scheduler_trace_ref, dict)
        and isinstance(report_ref, dict)
        and isinstance(typed_active_cap, int)
        and not isinstance(typed_active_cap, bool)
        and typed_active_cap >= active_floor,
        "highest-concurrency lifecycle replay inputs are missing",
    )
    groups = closed_loop_request_groups(product_lifecycle, record)
    evidence = replay_active_floor_duty(
        groups=groups,
        active_rows=active_rows,
        active_floor=active_floor,
    )
    require_repeat_active_floor_duty(evidence)
    evidence.update(
        {
            "typed_active_cap": typed_active_cap,
            "active_floor": active_floor,
            "requested_concurrency": record.get("concurrency"),
            "source_scheduler_trace": copy.deepcopy(scheduler_trace_ref),
            "source_resource_observations": copy.deepcopy(observation_ref),
            "source_active_intervals": copy.deepcopy(value),
            "source_bench_report": copy.deepcopy(report_ref),
            "validator": {
                "path": SCRIPT_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256(SCRIPT_PATH),
                "size_bytes": SCRIPT_PATH.stat().st_size,
            },
            "lifecycle_integrity": {
                "event_count": sum(
                    len(group["request_ids"]) * 2 for group in groups
                ),
                "request_count": sum(len(group["request_ids"]) for group in groups),
                "duplicate_count": 0,
                "missing_count": 0,
                "out_of_order_count": 0,
                "foreign_count": 0,
                "phase_barriers": "pass",
            },
            "observed_max_active": max(
                row["active_requests_conservative"]
                for row in active_rows
                if row["eligible"]
            ),
            "legacy_full_command_window_duty_cycle": fraction,
        }
    )
    return float(evidence["active_duty_cycle"]), evidence


def load_ferrum_collector_lane(
    path: Path,
    *,
    expected_model: str,
    expected_backend: str,
    expected_source: dict[str, Any] | None,
    collector_verifier: Callable[
        [Path, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]],
        None,
    ] = default_collector_verifier,
) -> dict[str, Any]:
    manifest_path = input_manifest(path, "manifest.json", "Ferrum collector lane")
    manifest = read_json(manifest_path, "Ferrum collector manifest")
    require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("contract") == "ferrum.runtime-vnext.r2.ferrum-collector.v1"
        and manifest.get("artifact_type") == "runtime_vnext_r2_ferrum_lane_manifest"
        and manifest.get("status") == "pass"
        and manifest.get("formal_r2_aggregate_status") == "not-evaluated"
        and manifest.get("model_key") == expected_model
        and manifest.get("backend") == expected_backend
        and manifest.get("profile_detail") == "off"
        and manifest.get("dirty_status")
        == {"is_dirty": False, "status_short": []},
        "Ferrum collector manifest identity/status differs",
    )
    source = normalize_source(
        {
            "git_sha": manifest.get("source_git_sha"),
            "git_tree_sha": manifest.get("source_tree_sha"),
            "dirty": False,
        },
        "Ferrum collector",
    )
    require(
        expected_source is None or source == expected_source,
        "Ferrum collector candidate source differs",
    )
    root = ferrum_collector_root(manifest_path, manifest)
    plan_path = validate_collector_ref(root, manifest.get("plan"), "collector plan")
    plan = read_json(plan_path, "collector plan")
    config_path = validate_collector_ref(root, plan.get("config"), "normalized collector config")
    config = read_json(config_path, "normalized collector config")
    require(
        plan.get("model_key") == expected_model
        and plan.get("backend") == expected_backend
        and plan.get("profile_detail") == "off"
        and plan.get("external_engine") is None
        and plan.get("legacy_binary") is None
        and plan.get("abba_order") is None
        and config.get("model_key") == expected_model
        and config.get("backend") == expected_backend
        and config.get("candidate", {}).get("source_git_sha") == source["git_sha"]
        and config.get("candidate", {}).get("source_tree_sha")
        == source["git_tree_sha"],
        "Ferrum collector plan/config is not the profile-off product-only lane",
    )
    try:
        import runtime_vnext_baseline_scenarios as baseline_scenarios

        baseline_scenarios.validate_sanitized_environment(
            config.get("candidate", {}).get("env"), "collector candidate environment"
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise R2Error(f"collector candidate uses hidden environment: {error}") from error
    binary = require_sha(
        manifest.get("candidate_binary_sha256"), "Ferrum collector binary"
    )
    hardware = manifest.get("hardware")
    require(
        isinstance(hardware, dict)
        and isinstance(hardware.get("id"), str)
        and bool(hardware["id"])
        and isinstance(hardware.get("fingerprint"), str)
        and bool(hardware["fingerprint"]),
        "Ferrum collector hardware identity is missing",
    )
    inputs = manifest.get("inputs")
    require(isinstance(inputs, dict), "Ferrum collector input provenance is missing")
    for name in (
        "binary",
        "correctness_manifest",
        "tokenizer",
        "realistic_dataset",
    ):
        validate_collector_ref(root, inputs.get(name), f"collector input {name}")
    require(
        inputs["binary"]["sha256"] == binary,
        "Ferrum collector staged binary differs from the manifest",
    )
    server_path = validate_collector_ref(
        root, manifest.get("server_session"), "server session"
    )
    server = read_json(server_path, "server session")
    run_paths = [
        validate_collector_ref(root, ref, f"run sample {ordinal}")
        for ordinal, ref in enumerate(manifest.get("run_samples", []), start=1)
    ]
    require(len(run_paths) == 3, "Ferrum collector run sample denominator differs")
    runs = [read_json(run_path, "run sample bundle") for run_path in run_paths]
    collector_verifier(root, manifest, config, server, runs)
    session = server.get("session")
    session_epochs = server.get("session_epochs")
    records = server.get("formal_reports")
    legacy_single_epoch = session_epochs is None
    require(
        isinstance(session, dict)
        and (
            (isinstance(session_epochs, list) and bool(session_epochs))
            or (legacy_single_epoch and session.get("server_process_ordinal") == 1)
        )
        and session.get("shutdown_clean") is True
        and isinstance(records, list),
        "Ferrum collector server session is incomplete",
    )
    all_server_records = [*records, server.get("run_serve_parity_report")]
    if legacy_single_epoch:
        session_epochs = [
            {
                "kind": "legacy-single-session",
                "session_id": session.get("session_id"),
                "completed_cell_sequences": [
                    row.get("sequence")
                    for row in all_server_records
                    if isinstance(row, dict)
                ],
            }
        ]
    epoch_authority: dict[str, dict[str, Any]] = {}
    for ordinal, epoch_row in enumerate(session_epochs, start=1):
        require(isinstance(epoch_row, dict), f"Ferrum server epoch {ordinal} is invalid")
        epoch_session_id = epoch_row.get("session_id")
        require(
            isinstance(epoch_session_id, str)
            and bool(epoch_session_id)
            and epoch_session_id not in epoch_authority,
            f"Ferrum server epoch {ordinal} identity is invalid",
        )
        if epoch_row.get("kind") == "recovered-completed-cell":
            checkpoint_refs = epoch_row.get("checkpoints")
            completed_sequences = epoch_row.get("completed_cell_sequences")
            require(
                isinstance(checkpoint_refs, list)
                and isinstance(completed_sequences, list)
                and len(checkpoint_refs) == len(completed_sequences)
                and bool(checkpoint_refs),
                f"Ferrum recovered epoch {ordinal} checkpoint set is invalid",
            )
            checkpoint_documents = [
                read_json(
                    validate_collector_ref(
                        root, ref, f"Ferrum recovered epoch {ordinal} checkpoint {index}"
                    ),
                    f"Ferrum recovered epoch {ordinal} checkpoint {index}",
                )
                for index, ref in enumerate(checkpoint_refs, start=1)
            ]
            checkpoint = checkpoint_documents[0]
            require(
                [row.get("sequence") for row in checkpoint_documents] == completed_sequences
                and all(
                    row.get("contract") == "ferrum.runtime-vnext.r2.completed-cell-checkpoint.v1"
                    and row.get("config_fingerprint") == manifest.get("config_fingerprint")
                    and row.get("collection_epoch") == epoch_row.get("collection_epoch")
                    and row.get("epoch", {}).get("session_id") == epoch_session_id
                    and row.get("epoch") == checkpoint.get("epoch")
                    and row.get("attempt_cleanup") == checkpoint.get("attempt_cleanup")
                    for row in checkpoint_documents
                ),
                f"Ferrum recovered epoch {ordinal} checkpoint binding differs",
            )
            authority = copy.deepcopy(checkpoint["epoch"])
            authority["runtime_log"] = checkpoint["epoch"]["runtime_log_evidence"]
            authority["scheduler_trace"] = checkpoint["epoch"]["scheduler_trace"]
            authority["product_effective_config"] = checkpoint["epoch"]["product_effective_config"]
            failure_path = validate_collector_ref(
                root,
                checkpoint.get("attempt_cleanup"),
                f"Ferrum recovered epoch {ordinal} cleanup",
            )
            failure = read_json(failure_path, f"Ferrum recovered epoch {ordinal} cleanup")
            require(
                failure.get("cleanup_process_group_gone") is True
                and failure.get("cleanup_error") is None,
                f"Ferrum recovered epoch {ordinal} cleanup differs",
            )
            authority["finished_at"] = failure["failed_at"]
            authority["completed_cell_sequences"] = completed_sequences
        elif epoch_row.get("kind") == "legacy-single-session":
            require(
                epoch_session_id == session.get("session_id"),
                f"Ferrum legacy epoch {ordinal} identity differs",
            )
            authority = {
                **session,
                "completed_cell_sequences": epoch_row["completed_cell_sequences"],
            }
        else:
            require(
                epoch_row.get("kind") == "current-server-session"
                and epoch_session_id == session.get("session_id"),
                f"Ferrum server epoch {ordinal} kind differs",
            )
            authority = session
            completed_sequences = epoch_row.get("completed_cell_sequences")
            require(
                isinstance(completed_sequences, list)
                and completed_sequences
                and all(isinstance(value, int) for value in completed_sequences),
                f"Ferrum current epoch {ordinal} completed-cell set is invalid",
            )
            authority = {**authority, "completed_cell_sequences": completed_sequences}
        for ref_name in ("runtime_log", "scheduler_trace", "product_effective_config"):
            validate_collector_ref(
                root, authority.get(ref_name), f"Ferrum server epoch {ordinal} {ref_name}"
            )
        epoch_authority[epoch_session_id] = authority
    observed_epoch_sequences: dict[str, list[int]] = {
        epoch_id: [] for epoch_id in epoch_authority
    }
    for record in all_server_records:
        require(isinstance(record, dict), "Ferrum server record is invalid")
        epoch_id = record.get("session_id")
        require(epoch_id in observed_epoch_sequences, "Ferrum server record uses an unknown epoch")
        observed_epoch_sequences[epoch_id].append(record.get("sequence"))
    for epoch_id, authority in epoch_authority.items():
        require(
            observed_epoch_sequences[epoch_id] == authority["completed_cell_sequences"],
            f"Ferrum server epoch {epoch_id} completed-cell binding differs",
        )
    require(
        session.get("server_process_ordinal") == len(epoch_authority),
        "Ferrum collector server epoch/process denominator differs",
    )
    post_cell_bodies = load_post_cell_health_bodies(
        root,
        server,
        all_server_records,
        legacy_single_epoch=legacy_single_epoch,
    )
    for record, body in zip(all_server_records, post_cell_bodies, strict=True):
        validate_post_cell_typed_ledger(
            body,
            typed_active_cap=config.get("typed_active_cap"),
            memory_budget_bytes=config.get("memory_budget_bytes"),
            label=f"Ferrum post-cell {record.get('cell_id')}",
        )
    try:
        baseline_scenarios.validate_sanitized_environment(
            session.get("environment"), "Ferrum serve environment"
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise R2Error(f"Ferrum serve contains hidden environment: {error}") from error
    require(
        session.get("environment_sha256")
        == collector_canonical_json_sha256(session["environment"]),
        "Ferrum serve environment hash differs",
    )
    server_argv = session.get("server_argv")
    require(
        isinstance(server_argv, list)
        and server_argv.count("serve") == 1
        and flag_value(server_argv, "--profile-detail", "Ferrum serve argv")
        == "off"
        and session.get("server_argv_sha256")
        == collector_canonical_json_sha256(server_argv),
        "Ferrum serve product argv/hash differs",
    )
    run_pids: set[int] = set()
    metal_swap_scopes: list[dict[str, Any]] = []
    for ordinal, bundle in enumerate(runs, start=1):
        sample = bundle.get("sample")
        require(
            isinstance(sample, dict)
            and sample.get("sample_ordinal") == ordinal
            and sample.get("candidate_binary_sha256") == binary
            and sample.get("source_git_sha") == source["git_sha"]
            and sample.get("profile_detail") == "off"
            and sample.get("returncode") == 0,
            f"Ferrum run sample {ordinal} identity differs",
        )
        pid = sample.get("pid")
        require(
            isinstance(pid, int)
            and not isinstance(pid, bool)
            and pid > 0
            and pid not in run_pids,
            f"Ferrum run sample {ordinal} process is not independent",
        )
        run_pids.add(pid)
        try:
            baseline_scenarios.validate_sanitized_environment(
                sample.get("environment"), f"Ferrum run sample {ordinal} environment"
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise R2Error(
                f"Ferrum run sample {ordinal} contains hidden environment: {error}"
            ) from error
        require(
            sample.get("environment_sha256")
            == collector_canonical_json_sha256(sample["environment"])
            and sample.get("argv_sha256")
            == collector_canonical_json_sha256(sample.get("argv")),
            f"Ferrum run sample {ordinal} command/environment hash differs",
        )
        require(
            isinstance(sample.get("argv"), list)
            and sample["argv"].count("run") == 1
            and flag_value(
                sample["argv"],
                "--profile-detail",
                f"Ferrum run sample {ordinal} argv",
            )
            == "off",
            f"Ferrum run sample {ordinal} is not the profile-off product path",
        )
        resources = sample.get("resources")
        require(
            isinstance(resources, dict)
            and isinstance(resources.get("summary"), dict)
            and resources["summary"].get("exit_reason") == "process-exit",
            f"Ferrum run sample {ordinal} lacks a clean process-exit resource footer",
        )
        if expected_backend == "metal":
            observations_ref = resources.get("observations")
            observations_path = validate_collector_ref(
                root,
                observations_ref,
                f"Ferrum run sample {ordinal} resource observations",
            )
            resource_summary = resources["summary"]
            observation_rows = read_jsonl_objects(
                observations_path,
                f"Ferrum run sample {ordinal} resource observations",
            )
            observation_samples = observation_rows[1:-1]
            try:
                import runtime_vnext_resource_sampler as resource_sampler

                recomputed_resource = resource_sampler.derive_summary(
                    observations_path,
                    session_id=sample["sample_id"],
                    cell_id="run:c1",
                    backend="metal",
                    hardware_id=hardware["id"],
                    pid=sample["pid"],
                    pgid=sample["pgid"],
                    process_start_marker=sample["process_start_marker"],
                    base_url=f"process://{sample['sample_id']}",
                    session_started_at=sample["started_at"],
                    session_finished_at=sample["finished_at"],
                    measurement_started_at=observation_samples[0]["sampled_at"],
                    measurement_finished_at=observation_samples[-1]["sampled_at"],
                    memory_budget_bytes=config["memory_budget_bytes"],
                    requested_concurrency=1,
                    typed_active_cap=1,
                    runtime_log_path=observation_rows[0]["runtime_log_path"],
                )
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                raise R2Error(
                    f"Ferrum run sample {ordinal} resource evidence failed: {error}"
                ) from error
            require(
                recomputed_resource == resource_summary,
                f"Ferrum run sample {ordinal} resource summary is not raw-derived",
            )
            swap_evidence = derive_metal_swap_scope_evidence(
                observations_path,
                observations_ref["sha256"],
                resource_summary,
                label=f"Ferrum run sample {ordinal}",
            )
            validate_metal_resource_contract(
                resource_summary,
                f"Ferrum run sample {ordinal}",
                typed_memory_budget_bytes=config["memory_budget_bytes"],
                resource_evidence=swap_evidence,
            )
            metal_swap_scopes.append(
                {"scope": f"run:c1:sample-{ordinal}", **swap_evidence}
            )
    expected = expected_cells(expected_backend)
    product_lifecycle_by_epoch: dict[str, dict[str, dict[str, int]]] = {}
    require(
        len(records) == len(expected)
        and manifest.get("formal_http_cell_count") == len(expected),
        "Ferrum collector formal cell denominator differs",
    )
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    measured_requests = 0
    for record, cell in zip(records, plan.get("server_cell_order", []), strict=True):
        key = (str(cell.get("dataset")), int(cell.get("concurrency", -1)))
        record_session_id = record.get("session_id")
        require(
            isinstance(record_session_id, str) and record_session_id in epoch_authority,
            f"Ferrum collector cell epoch is unknown: {key}",
        )
        record_session = epoch_authority[record_session_id]
        require(
            key in expected
            and key not in summaries
            and record.get("dataset") == key[0]
            and record.get("concurrency") == key[1]
            and record.get("num_prompts") == requests_per_repeat(key[0])
            and record.get("n_repeats") == 3
            and record.get("warmup_requests") == 10
            and record.get("returncode") == 0
            and record.get("candidate_binary_sha256") == binary,
            f"Ferrum collector cell identity/status differs: {key}",
        )
        environment = record.get("environment")
        try:
            baseline_scenarios.validate_sanitized_environment(
                environment, f"Ferrum collector cell environment {key}"
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise R2Error(
                f"Ferrum collector cell contains hidden environment {key}: {error}"
            ) from error
        require(
            record.get("environment_sha256")
            == collector_canonical_json_sha256(environment),
            f"Ferrum collector cell environment hash differs: {key}",
        )
        validate_benchmark_argv(
            record.get("bench_argv"),
            backend=expected_backend,
            dataset=key[0],
            concurrency=key[1],
            request_count=requests_per_repeat(key[0]),
        )
        require(
            record.get("bench_argv_sha256")
            == collector_canonical_json_sha256(record["bench_argv"]),
            f"Ferrum collector cell command hash differs: {key}",
        )
        report_ref = record.get("raw_report")
        report_path = validate_collector_ref(root, report_ref, f"raw report {key}")
        report = read_json(report_path, f"raw report {key}")
        repeats = report.get("repeat_metrics")
        require(isinstance(repeats, list) and len(repeats) == 3, f"raw repeat denominator differs: {key}")
        measured_requests += validate_request_sidecar(
            root,
            record.get("raw_request_evidence"),
            report_ref=report_ref,
            cell=cell,
            report=report,
        )
        resources = record.get("resources")
        require(isinstance(resources, dict), f"resource evidence is missing: {key}")
        observations_ref = resources.get("observations")
        observations_path = validate_collector_ref(
            root, observations_ref, f"resource observations {key}"
        )
        resource_summary = resources.get("summary")
        require(isinstance(resource_summary, dict), f"resource summary is missing: {key}")
        runtime_log_path = validate_collector_ref(
            root, record_session.get("runtime_log"), f"server epoch runtime log {key}"
        )
        try:
            import runtime_vnext_resource_sampler as resource_sampler

            recomputed_resource = resource_sampler.derive_summary(
                observations_path,
                session_id=record_session["session_id"],
                cell_id=record["cell_id"],
                backend=expected_backend,
                hardware_id=hardware["id"],
                pid=record_session["pid"],
                pgid=record_session["pgid"],
                process_start_marker=record_session["process_start_marker"],
                base_url=record_session["base_url"],
                session_started_at=record_session["started_at"],
                session_finished_at=record_session["finished_at"],
                measurement_started_at=record["started_at"],
                measurement_finished_at=record["finished_at"],
                memory_budget_bytes=config["memory_budget_bytes"],
                requested_concurrency=record["concurrency"],
                typed_active_cap=config["typed_active_cap"],
                runtime_log_path=record_session["runtime_log_origin_path"],
                runtime_log_evidence_path=runtime_log_path,
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise R2Error(f"resource evidence failed for {key}: {error}") from error
        require(recomputed_resource == resource_summary, f"resource summary is not raw-derived: {key}")
        highest = requires_active_floor(key[0], key[1], expected_backend)
        scheduler_trace_ref = record_session.get("scheduler_trace")
        if highest and record_session_id not in product_lifecycle_by_epoch:
            scheduler_trace_path = validate_collector_ref(
                root, scheduler_trace_ref, f"server epoch scheduler trace {key}"
            )
            product_lifecycle_by_epoch[record_session_id] = load_product_request_lifecycle(
                scheduler_trace_path
            )
        duty, duty_evidence = validate_active_intervals(
            root,
            resources.get("active_intervals"),
            observation_ref=observations_ref,
            observations_path=observations_path,
            record=record,
            active_floor=ACTIVE_FLOORS[(expected_model, expected_backend)],
            require_floor=highest,
            product_lifecycle=(
                product_lifecycle_by_epoch[record_session_id] if highest else None
            ),
            scheduler_trace_ref=scheduler_trace_ref if highest else None,
            report_ref=report_ref if highest else None,
            typed_active_cap=config.get("typed_active_cap") if highest else None,
        )
        swap_evidence: dict[str, Any] | None = None
        if expected_backend == "cuda":
            require(
                finite_positive(resource_summary.get("peak_memory_bytes"), "peak memory")
                <= finite_positive(resource_summary.get("memory_budget_bytes"), "memory budget")
                and resource_summary.get("oom_count") == 0
                and resource_summary.get("admission_error_count") == 0,
                f"resource budget/error contract failed: {key}",
            )
            require(
                finite_nonnegative(
                    resource_summary.get("physical_headroom_bytes"), "CUDA headroom"
                )
                >= PHYSICAL_HEADROOM_FLOOR_BYTES["cuda"],
                f"CUDA headroom is below 512 MiB: {key}",
            )
        else:
            swap_evidence = derive_metal_swap_scope_evidence(
                observations_path,
                observations_ref["sha256"],
                resource_summary,
                label=f"Ferrum collector cell {key}",
                measurement_started_at=record["started_at"],
                measurement_finished_at=record["finished_at"],
            )
            validate_metal_resource_contract(
                resource_summary,
                f"Ferrum collector cell {key}",
                typed_memory_budget_bytes=config["memory_budget_bytes"],
                resource_evidence=swap_evidence,
            )
            metal_swap_scopes.append(
                {"scope": f"{key[0]}:c{key[1]}", **swap_evidence}
            )
        throughputs = [finite_positive(row.get("output_throughput_tps"), "repeat throughput") for row in repeats]
        cv = statistics.pstdev(throughputs) / statistics.mean(throughputs)
        require(cv <= 0.08 + 1e-12, f"Ferrum collector cell CV exceeds 8%: {key}")
        raw_metrics = {
            "throughput": throughputs,
            "ttft_p95": [finite_positive(row.get("ttft_ms", {}).get("p95"), "repeat TTFT p95") for row in repeats],
            "tpot_p95": [finite_positive(row.get("tpot_ms", {}).get("p95"), "repeat TPOT p95") for row in repeats],
            "peak_accelerator_or_unified_memory": [
                float(resource_summary["peak_memory_bytes"])
            ],
        }
        summaries[key] = {
            "dataset_sha256": (
                canonical_json_sha256(
                    {
                        "dataset": "random",
                        "input_tokens": cell["input_tokens"],
                        "output_tokens": cell["output_tokens"],
                        "seed": 9271,
                    }
                )
                if key[0] == "random"
                else inputs["realistic_dataset"]["sha256"]
            ),
            "throughput": statistics.median(raw_metrics["throughput"]),
            "ttft_p95": statistics.median(raw_metrics["ttft_p95"]),
            "tpot_p95": statistics.median(raw_metrics["tpot_p95"]),
            "peak_accelerator_or_unified_memory": raw_metrics[
                "peak_accelerator_or_unified_memory"
            ][0],
            "physical_headroom_bytes": resource_summary[
                "physical_headroom_bytes"
            ],
            "metal_swap_evidence": copy.deepcopy(swap_evidence),
            "steady_decode": statistics.median(
                1000.0 / finite_positive(row.get("tpot_ms", {}).get("p50"), "repeat TPOT p50")
                for row in repeats
            ),
            "cv": cv,
            "observed_max_active": int(resource_summary["observed_max_active"]),
            "active_floor_duty_cycle": duty,
            "active_floor_duty_evidence": duty_evidence,
            "raw_repeats": raw_metrics,
            "command_sha256": record["bench_argv_sha256"],
            "resource_observations_sha256": observations_ref["sha256"],
            "metric_artifact_sha256": {
                metric: (
                    observations_ref["sha256"]
                    if metric == "peak_accelerator_or_unified_memory"
                    else report_ref["sha256"]
                )
                for metric in FLOOR_METRICS
            },
            "raw_value_scope": {
                metric: (
                    "cell-aggregate-peak-over-three-benchmark-repeats"
                    if metric == "peak_accelerator_or_unified_memory"
                    else "three-benchmark-repeat-values"
                )
                for metric in FLOOR_METRICS
            },
            "report": copy.deepcopy(report_ref),
        }
    require(set(summaries) == expected, "Ferrum collector cell tuple set differs")
    metal_swap_evidence = (
        aggregate_metal_swap_evidence(
            metal_swap_scopes,
            f"Ferrum collector {expected_model}/{expected_backend}",
        )
        if expected_backend == "metal"
        else None
    )
    require(measured_requests == sum(requests_per_repeat(dataset) * 3 for dataset, _ in expected), "Ferrum collector measured request denominator differs")
    parity = server.get("run_serve_parity_report")
    require(isinstance(parity, dict), "run/serve parity evidence is missing")
    if expected_backend == "metal":
        parity_session_id = parity.get("session_id")
        require(
            isinstance(parity_session_id, str)
            and parity_session_id in epoch_authority,
            "run/serve parity resource epoch is unknown",
        )
        parity_session = epoch_authority[parity_session_id]
        parity_resources = parity.get("resources")
        require(
            isinstance(parity_resources, dict),
            "run/serve parity resource evidence is missing",
        )
        parity_observations_ref = parity_resources.get("observations")
        parity_observations_path = validate_collector_ref(
            root,
            parity_observations_ref,
            "run/serve parity resource observations",
        )
        parity_resource_summary = parity_resources.get("summary")
        require(
            isinstance(parity_resource_summary, dict),
            "run/serve parity resource summary is missing",
        )
        parity_runtime_log_path = validate_collector_ref(
            root,
            parity_session.get("runtime_log"),
            "run/serve parity epoch runtime log",
        )
        try:
            import runtime_vnext_resource_sampler as resource_sampler

            recomputed_parity_resource = resource_sampler.derive_summary(
                parity_observations_path,
                session_id=parity_session["session_id"],
                cell_id=parity["cell_id"],
                backend="metal",
                hardware_id=hardware["id"],
                pid=parity_session["pid"],
                pgid=parity_session["pgid"],
                process_start_marker=parity_session["process_start_marker"],
                base_url=parity_session["base_url"],
                session_started_at=parity_session["started_at"],
                session_finished_at=parity_session["finished_at"],
                measurement_started_at=parity["started_at"],
                measurement_finished_at=parity["finished_at"],
                memory_budget_bytes=config["memory_budget_bytes"],
                requested_concurrency=parity["concurrency"],
                typed_active_cap=config["typed_active_cap"],
                runtime_log_path=parity_session["runtime_log_origin_path"],
                runtime_log_evidence_path=parity_runtime_log_path,
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise R2Error(
                f"run/serve parity resource evidence failed: {error}"
            ) from error
        require(
            recomputed_parity_resource == parity_resource_summary,
            "run/serve parity resource summary is not raw-derived",
        )
        parity_resource_evidence = derive_metal_swap_scope_evidence(
            parity_observations_path,
            parity_observations_ref["sha256"],
            parity_resource_summary,
            label="run/serve parity resource",
            measurement_started_at=parity["started_at"],
            measurement_finished_at=parity["finished_at"],
        )
        validate_metal_resource_contract(
            parity_resource_summary,
            "run/serve parity resource",
            typed_memory_budget_bytes=config["memory_budget_bytes"],
            resource_evidence=parity_resource_evidence,
        )
    try:
        import runtime_vnext_r2_ferrum_collector as ferrum_collector

        recomputed_run = ferrum_collector.run_summary(runs, parity)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise R2Error(f"run summary failed: {error}") from error
    require(recomputed_run == manifest.get("run_performance"), "run performance is not raw-derived")
    require(
        recomputed_run.get("sample_count") == 3
        and recomputed_run.get("independent_process_count") == 3
        and finite_positive(recomputed_run.get("run_to_serve_c1_steady_decode_ratio"), "run/serve ratio")
        >= 0.90 - 1e-12,
        "three-process ferrum run/serve parity contract failed",
    )
    model_files = manifest.get("model_files")
    require(isinstance(model_files, dict) and model_files, "locked model file identity is missing")
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": sha256(manifest_path),
        "artifact_root": root,
        "source": source,
        "profile_detail": "off",
        "created_at": session["formal_measurement_finished_at"],
        "binary_sha256": binary,
        "hardware_id": hardware["id"],
        "hardware_sha256": canonical_json_sha256(hardware),
        "model_sha256": canonical_json_sha256(model_files),
        "typed_config_sha256": sha256(config_path),
        "correctness_manifest_sha256": inputs["correctness_manifest"]["sha256"],
        "summaries": summaries,
        "minimum_cell_physical_headroom_bytes": min(
            summary["physical_headroom_bytes"] for summary in summaries.values()
        ),
        "physical_headroom_floor_bytes": PHYSICAL_HEADROOM_FLOOR_BYTES[
            expected_backend
        ],
        "metal_swap_evidence": metal_swap_evidence,
        "measured_request_count": measured_requests,
        "run": {
            "sample_count": 3,
            "steady_decode_median": recomputed_run["steady_decode_tps_median"],
            "serve_c1_median": recomputed_run[
                "serve_c1_same_prompt_steady_decode_tps_median"
            ],
        },
    }


def performance_lane_r1_source_closure(
    lane: dict[str, Any],
    source: dict[str, Any],
    r1: dict[str, Any],
    expected_backend: str,
) -> dict[str, Any]:
    require(
        lane.get("profile_detail") == "off",
        "cumulative R1 evidence is restricted to profile-off performance lanes",
    )
    if (
        lane.get("binary_sha256")
        == r1["backend_binary_sha256"][expected_backend]
        and lane.get("correctness_manifest_sha256")
        == r1["child_manifest"]["sha256"]
    ):
        return source_closure(lane["source"], source)

    evidence = r1.get("cumulative_profile_off_evidence")
    require(
        isinstance(evidence, dict)
        and set(evidence)
        == {
            "mode",
            "source",
            "r1_child_manifest",
            "impact_qualification",
            "backend_binary_sha256",
            "backend_hardware_id",
        }
        and evidence.get("mode") == "cumulative-r1-prior-full-profile-off",
        "Ferrum collector binary/correctness authority differs from R1",
    )
    prior_source = normalize_source(
        evidence.get("source"), "cumulative prior full R1"
    )
    evidence_binaries = evidence.get("backend_binary_sha256")
    prior_child = evidence.get("r1_child_manifest")
    require(
        lane.get("source") == prior_source
        and isinstance(evidence_binaries, dict)
        and set(evidence_binaries) == set(BACKENDS)
        and lane.get("binary_sha256") == evidence_binaries.get(expected_backend)
        and isinstance(prior_child, dict)
        and lane.get("correctness_manifest_sha256") == prior_child.get("sha256"),
        "Ferrum collector is not bound to cumulative R1 prior profile-off evidence",
    )
    return {
        "from_git_sha": prior_source["git_sha"],
        "to_git_sha": source["git_sha"],
        "policy": "cumulative-r1-prior-full-profile-off",
        "prior_r1_child_manifest": copy.deepcopy(prior_child),
        "impact_qualification": copy.deepcopy(evidence["impact_qualification"]),
        "current_r1_child_manifest": copy.deepcopy(r1["child_manifest"]),
        "current_r1_source_closure": copy.deepcopy(r1["source_closure"]),
    }


def validate_ferrum_performance_lane(
    path: Path,
    *,
    expected_model: str,
    expected_backend: str,
    source: dict[str, Any],
    r1: dict[str, Any],
    floors: dict[tuple[str, str, str, int, str], dict[str, Any]],
) -> dict[str, Any]:
    lane = load_ferrum_collector_lane(
        path,
        expected_model=expected_model,
        expected_backend=expected_backend,
        expected_source=None,
    )
    closure = performance_lane_r1_source_closure(
        lane,
        source,
        r1,
        expected_backend,
    )
    throughput_ratios: list[float] = []
    calibration = False
    for (dataset, concurrency), summary in lane["summaries"].items():
        for metric in FLOOR_METRICS:
            floor = floors[
                catalog_key(
                    expected_model,
                    expected_backend,
                    dataset,
                    concurrency,
                    metric,
                )
            ]
            require(
                floor["model_sha256"] == lane["model_sha256"]
                and floor["hardware_id"] == lane["hardware_id"]
                and floor["hardware_sha256"] == lane["hardware_sha256"]
                and floor["typed_config_sha256"] == lane["typed_config_sha256"]
                and floor["dataset_sha256"] == summary["dataset_sha256"],
                f"floor identity differs for {expected_model}/{expected_backend}/{dataset}/c{concurrency}/{metric}",
            )
            calibration = calibration or floor["baseline_kind"] == "calibration"
            if floor["baseline_kind"] == "calibration":
                require(
                    floor["source_git_sha"] == lane["source"]["git_sha"]
                    and floor["binary_sha256"] == lane["binary_sha256"]
                    and floor["collector_manifest_sha256"]
                    == lane["manifest_sha256"]
                    and floor["artifact_sha256"]
                    == summary["metric_artifact_sha256"][metric]
                    and floor["command_sha256"] == summary["command_sha256"]
                    and floor["raw_repeats"] == summary["raw_repeats"][metric]
                    and floor["raw_value_scope"]
                    == summary["raw_value_scope"][metric]
                    and floor["resource_observations_sha256"]
                    == summary["resource_observations_sha256"],
                    "calibration floor is not bound to the supplied Ferrum collector "
                    f"for {expected_model}/{expected_backend}/{dataset}/c{concurrency}/{metric}",
                )
            candidate = float(summary[metric])
            baseline = float(floor["value"])
            if metric == "throughput":
                ratio = candidate / baseline
                require(
                    ratio >= 0.95 - 1e-12,
                    f"throughput ratio below 0.95 for {dataset}/c{concurrency}",
                )
                throughput_ratios.append(ratio)
            elif metric in {"ttft_p95", "tpot_p95"}:
                require(
                    candidate / baseline <= 1.10 + 1e-12,
                    f"latency ratio above 1.10 for {dataset}/c{concurrency}/{metric}",
                )
            else:
                require(
                    candidate / baseline <= 1.05 + 1e-12,
                    f"memory ratio above 1.05 for {dataset}/c{concurrency}",
                )
    geometric_mean = math.exp(
        sum(math.log(value) for value in throughput_ratios)
        / len(throughput_ratios)
    )
    require(
        geometric_mean >= 1.0 - 1e-12,
        "model/backend throughput geometric mean is below 1.00",
    )
    highest = max(MAIN_CONCURRENCY[expected_backend])
    high = lane["summaries"][("random", highest)]
    c1 = lane["summaries"][("random", 1)]
    require(
        high["observed_max_active"]
        >= ACTIVE_FLOORS[(expected_model, expected_backend)]
        and high["active_floor_duty_cycle"] >= 0.80 - 1e-12
        and high["throughput"] / c1["throughput"]
        >= HIGHEST_CONCURRENCY_SCALE[expected_backend] - 1e-12,
        "highest-concurrency active/scaling contract failed",
    )
    require(
        lane["run"]["steady_decode_median"] / lane["run"]["serve_c1_median"]
        >= 0.90 - 1e-12,
        "ferrum run steady decode is below 0.90x same-prompt serve c1",
    )
    if calibration:
        require(
            lane["run"]["steady_decode_median"]
            >= ABSOLUTE_RUN_FLOORS[(expected_model, expected_backend)] - 1e-12,
            "calibration lane misses the absolute run-c1 floor",
        )
    return {
        "manifest": file_ref(lane["manifest_path"]),
        "source": lane["source"],
        "source_closure": closure,
        "model_key": expected_model,
        "backend": expected_backend,
        "binary_sha256": lane["binary_sha256"],
        "hardware_id": lane["hardware_id"],
        "r1_correctness_hardware_id": r1["backend_hardware_id"][expected_backend],
        "model_sha256": lane["model_sha256"],
        "typed_config_sha256": lane["typed_config_sha256"],
        "cell_count": len(lane["summaries"]),
        "repeat_count": len(lane["summaries"]) * 3,
        "measured_request_count": lane["measured_request_count"],
        "run_sample_count": lane["run"]["sample_count"],
        "minimum_cell_physical_headroom_bytes": lane[
            "minimum_cell_physical_headroom_bytes"
        ],
        "physical_headroom_floor_bytes": lane[
            "physical_headroom_floor_bytes"
        ],
        "metal_swap_evidence": copy.deepcopy(lane["metal_swap_evidence"]),
        "throughput_geometric_mean_ratio": geometric_mean,
        "max_cv": max(
            float(row["cv"]) for row in lane["summaries"].values()
        ),
        "highest_concurrency_active_floor_duty_cycle": high[
            "active_floor_duty_cycle"
        ],
        "highest_concurrency_active_floor_duty_evidence": copy.deepcopy(
            high["active_floor_duty_evidence"]
        ),
        "calibration": calibration,
    }


def floor_catalog_from_collectors(
    paths: dict[str, Path],
    *,
    lane_loader: Callable[..., dict[str, Any]] = load_ferrum_collector_lane,
) -> dict[str, Any]:
    """Create a deterministic calibration catalog from six immutable collectors."""
    require(set(paths) == set(LANE_KEYS), "floor template requires six performance lanes")
    lanes: dict[str, dict[str, Any]] = {}
    common_source: dict[str, Any] | None = None
    for lane_key, (model_key, backend) in LANE_KEYS.items():
        lane = lane_loader(
            paths[lane_key],
            expected_model=model_key,
            expected_backend=backend,
            expected_source=common_source,
        )
        if common_source is None:
            common_source = lane["source"]
        lanes[lane_key] = lane
    require(common_source is not None, "floor template source is missing")
    frozen_dt = max(
        datetime.fromisoformat(lane["created_at"].replace("Z", "+00:00"))
        for lane in lanes.values()
    )
    frozen_at = frozen_dt.astimezone(timezone.utc).isoformat()
    collectors = [
        {
            "lane_key": lane_key,
            "model_key": LANE_KEYS[lane_key][0],
            "backend": LANE_KEYS[lane_key][1],
            "manifest_sha256": lanes[lane_key]["manifest_sha256"],
        }
        for lane_key in sorted(LANE_KEYS)
    ]
    rows: list[dict[str, Any]] = []
    for model_key in MODELS:
        for backend in BACKENDS:
            lane_key = f"{model_key.split('-', 1)[0]}_{backend}"
            lane = lanes[lane_key]
            ordered_cells = sorted(
                expected_cells(backend),
                key=lambda item: (item[0] != "random", item[0], item[1]),
            )
            for dataset, concurrency in ordered_cells:
                summary = lane["summaries"][(dataset, concurrency)]
                for metric in FLOOR_METRICS:
                    repeats = copy.deepcopy(summary["raw_repeats"][metric])
                    value = (
                        max(repeats)
                        if metric == "peak_accelerator_or_unified_memory"
                        else statistics.median(repeats)
                    )
                    rows.append(
                        {
                            "key": {
                                "model_key": model_key,
                                "backend": backend,
                                "dataset": dataset,
                                "concurrency": concurrency,
                                "metric": metric,
                            },
                            "baseline_kind": "calibration",
                            "value": value,
                            "unit": FLOOR_UNITS[metric],
                            "source_git_sha": lane["source"]["git_sha"],
                            "dirty": False,
                            "binary_sha256": lane["binary_sha256"],
                            "model_sha256": lane["model_sha256"],
                            "hardware_id": lane["hardware_id"],
                            "hardware_sha256": lane["hardware_sha256"],
                            "typed_config_sha256": lane["typed_config_sha256"],
                            "dataset_sha256": summary["dataset_sha256"],
                            "command_sha256": summary["command_sha256"],
                            "collector_manifest_sha256": lane[
                                "manifest_sha256"
                            ],
                            "artifact_sha256": summary[
                                "metric_artifact_sha256"
                            ][metric],
                            "raw_repeats": repeats,
                            "raw_value_scope": summary["raw_value_scope"][metric],
                            "resource_observations_sha256": summary[
                                "resource_observations_sha256"
                            ],
                            "frozen_at": frozen_at,
                        }
                    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_r2_floor_catalog",
        "status": "frozen",
        "frozen_at": frozen_at,
        "goal_contract_sha256": sha256(PERFORMANCE_AMENDMENT),
        "collectors": collectors,
        "rows": rows,
        "canonical_sha256_scope": "document_without_canonical_sha256",
    }
    document["canonical_sha256"] = canonical_json_sha256(document)
    return document


def write_floor_catalog_template(
    paths: dict[str, Path],
    output: Path,
    *,
    lane_loader: Callable[..., dict[str, Any]] = load_ferrum_collector_lane,
) -> str:
    target = output.expanduser().resolve()
    require(
        REPO_ROOT not in target.parents and target != REPO_ROOT,
        "floor template output must stay outside the source tree",
    )
    require(not target.exists(), f"floor template output already exists: {target}")
    document = floor_catalog_from_collectors(paths, lane_loader=lane_loader)
    write_json(target, document, exclusive=True)
    try:
        validate_floor_catalog(target, require_checked_in=False)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return f"FERRUM RUNTIME VNEXT R2 FLOOR CATALOG TEMPLATE WRITTEN: {target}"


def validate_performance_lane(
    path: Path,
    *,
    expected_model: str,
    expected_backend: str,
    source: dict[str, Any],
    r1: dict[str, Any],
    floors: dict[tuple[str, str, str, int, str], dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = input_manifest(path, "manifest.json", f"{expected_model}/{expected_backend} performance")
    root = manifest_path.parent
    manifest = read_json(manifest_path, "performance lane manifest")
    required = {
        "schema_version",
        "artifact_type",
        "status",
        "created_at",
        "source",
        "model_key",
        "backend",
        "binary_sha256",
        "hardware_id",
        "model_sha256",
        "typed_config",
        "profile_mode",
        "hidden_env_names",
        "production_legacy_selection_count",
        "cells",
        "run_samples",
    }
    require(set(manifest) == required, "performance lane manifest fields differ")
    recorded_source = normalize_source(manifest.get("source"), "performance lane")
    require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type") == "runtime_vnext_r2_performance_lane_manifest"
        and manifest.get("status") == "ready"
        and manifest.get("model_key") == expected_model
        and manifest.get("backend") == expected_backend
        and manifest.get("binary_sha256")
        == r1["backend_binary_sha256"][expected_backend]
        and manifest.get("hardware_id") == r1["backend_hardware_id"][expected_backend]
        and manifest.get("profile_mode") == "off"
        and manifest.get("hidden_env_names") == []
        and manifest.get("production_legacy_selection_count") == 0,
        "performance lane identity/source/config differs",
    )
    parse_timestamp(manifest.get("created_at"), "performance lane created_at")
    closure = source_closure(recorded_source, source)
    binary = require_sha(manifest.get("binary_sha256"), "performance binary")
    model_sha = require_sha(manifest.get("model_sha256"), "performance model")
    typed_config_path = validate_ref(manifest.get("typed_config"), "typed config", root=root)
    typed_config = read_json(typed_config_path, "typed config")
    require(
        set(typed_config)
        == {
            "schema_version",
            "model_key",
            "backend",
            "typed_active_cap",
            "memory_budget_bytes",
            "profile_detail",
            "hidden_env_names",
        }
        and typed_config.get("schema_version") == SCHEMA_VERSION
        and typed_config.get("model_key") == expected_model
        and typed_config.get("backend") == expected_backend
        and isinstance(typed_config.get("typed_active_cap"), int)
        and typed_config["typed_active_cap"] >= ACTIVE_FLOORS[(expected_model, expected_backend)]
        and finite_positive(typed_config.get("memory_budget_bytes"), "typed memory budget") > 0
        and typed_config.get("profile_detail") == "off"
        and typed_config.get("hidden_env_names") == [],
        "typed config contract differs",
    )
    config_sha = sha256(typed_config_path)
    raw_cells = manifest.get("cells")
    require(isinstance(raw_cells, list), "performance cells are missing")
    expected = expected_cells(expected_backend)
    require(len(raw_cells) == len(expected), "performance cell denominator differs")
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    for row in raw_cells:
        require(
            isinstance(row, dict) and set(row) == {"dataset", "concurrency", "report"},
            "performance cell reference fields differ",
        )
        key = (str(row.get("dataset")), int(row.get("concurrency", -1)))
        require(key in expected and key not in summaries, f"unexpected or duplicate performance cell: {key}")
        report_path = validate_ref(row.get("report"), f"performance report {key}", root=root)
        summaries[key] = validate_performance_report(
            report_path,
            model_key=expected_model,
            backend=expected_backend,
            dataset=key[0],
            concurrency=key[1],
            source=recorded_source,
            binary_sha256=binary,
            hardware_id=manifest["hardware_id"],
            model_sha256=model_sha,
            typed_config_sha256=config_sha,
        )
    require(set(summaries) == expected, "performance cell tuple set differs")
    throughput_ratios: list[float] = []
    calibration = False
    for (dataset, concurrency), summary in summaries.items():
        for metric in FLOOR_METRICS:
            floor = floors[catalog_key(expected_model, expected_backend, dataset, concurrency, metric)]
            require(
                floor["model_sha256"] == model_sha
                and floor["hardware_id"] == manifest["hardware_id"]
                and floor["hardware_sha256"]
                == canonical_json_sha256(
                    {"hardware_id": manifest["hardware_id"]}
                )
                and floor["typed_config_sha256"] == config_sha
                and floor["dataset_sha256"] == summary["dataset_sha256"],
                f"floor identity differs for {expected_model}/{expected_backend}/{dataset}/c{concurrency}/{metric}",
            )
            if floor["baseline_kind"] == "calibration":
                require(
                    floor["source_git_sha"] == recorded_source["git_sha"]
                    and floor["binary_sha256"] == binary
                    and floor["collector_manifest_sha256"]
                    == sha256(manifest_path)
                    and floor["artifact_sha256"]
                    == summary["metric_artifact_sha256"][metric]
                    and floor["command_sha256"] == summary["command_sha256"]
                    and floor["raw_repeats"] == summary["raw_repeats"][metric]
                    and floor["raw_value_scope"]
                    == summary["raw_value_scope"][metric]
                    and floor["resource_observations_sha256"]
                    == summary["resource_observations_sha256"],
                    "calibration floor is not bound to the supplied collector artifact "
                    f"for {expected_model}/{expected_backend}/{dataset}/c{concurrency}/{metric}",
                )
            calibration = calibration or floor["baseline_kind"] == "calibration"
            candidate = float(summary[metric])
            baseline = float(floor["value"])
            if metric == "throughput":
                ratio = candidate / baseline
                require(ratio >= 0.95 - 1e-12, f"throughput ratio below 0.95 for {dataset}/c{concurrency}")
                throughput_ratios.append(ratio)
            elif metric in {"ttft_p95", "tpot_p95"}:
                require(candidate / baseline <= 1.10 + 1e-12, f"latency ratio above 1.10 for {dataset}/c{concurrency}/{metric}")
            else:
                require(candidate / baseline <= 1.05 + 1e-12, f"memory ratio above 1.05 for {dataset}/c{concurrency}")
    geometric_mean = math.exp(sum(math.log(value) for value in throughput_ratios) / len(throughput_ratios))
    require(geometric_mean >= 1.0 - 1e-12, "model/backend throughput geometric mean is below 1.00")
    highest = max(MAIN_CONCURRENCY[expected_backend])
    high = summaries[("random", highest)]
    c1 = summaries[("random", 1)]
    floor = ACTIVE_FLOORS[(expected_model, expected_backend)]
    require(
        high["observed_max_active"] >= floor
        and high["active_floor_duty_cycle"] >= 0.80 - 1e-12,
        "highest-concurrency active floor/duty contract failed",
    )
    require(
        high["throughput"] / c1["throughput"]
        >= HIGHEST_CONCURRENCY_SCALE[expected_backend] - 1e-12,
        "highest-concurrency scaling contract failed",
    )
    run = validate_run_samples(
        manifest.get("run_samples"),
        root=root,
        model_key=expected_model,
        backend=expected_backend,
        source=recorded_source,
        binary_sha256=binary,
        hardware_id=manifest["hardware_id"],
        model_sha256=model_sha,
        typed_config_sha256=config_sha,
    )
    require(
        run["steady_decode_median"] / c1["steady_decode"] >= 0.90 - 1e-12,
        "ferrum run steady decode is below 0.90x serve c1",
    )
    if calibration:
        require(
            run["steady_decode_median"]
            >= ABSOLUTE_RUN_FLOORS[(expected_model, expected_backend)] - 1e-12,
            "calibration lane misses the absolute run-c1 floor",
        )
    return {
        "manifest": file_ref(manifest_path),
        "source": recorded_source,
        "source_closure": closure,
        "model_key": expected_model,
        "backend": expected_backend,
        "binary_sha256": binary,
        "hardware_id": manifest["hardware_id"],
        "model_sha256": model_sha,
        "typed_config_sha256": config_sha,
        "cell_count": len(summaries),
        "repeat_count": len(summaries) * 3,
        "measured_request_count": sum(row["request_count"] for row in summaries.values()),
        "run_sample_count": run["sample_count"],
        "throughput_geometric_mean_ratio": geometric_mean,
        "max_cv": max(float(row["cv"]) for row in summaries.values()),
        "calibration": calibration,
    }


def validate_profile_overhead_artifact(
    path: Path,
    *,
    mode: str,
    source: dict[str, Any],
    model_key: str,
    backend: str,
    binary_sha256: str,
    hardware_id: str,
    workload_sha256: str,
) -> dict[str, Any]:
    artifact = read_json(path, f"{backend} profile-{mode} overhead artifact")
    require(
        set(artifact)
        == {
            "schema_version",
            "artifact_type",
            "status",
            "source",
            "model_key",
            "backend",
            "binary_sha256",
            "hardware_id",
            "workload_sha256",
            "profile_detail",
            "hidden_env_names",
            "samples",
        }
        and artifact.get("schema_version") == SCHEMA_VERSION
        and artifact.get("artifact_type")
        == "runtime_vnext_r2_profile_overhead_samples"
        and artifact.get("status") == "ready"
        and normalize_source(artifact.get("source"), "profile overhead artifact")
        == source
        and artifact.get("model_key") == model_key
        and artifact.get("backend") == backend
        and artifact.get("binary_sha256") == binary_sha256
        and artifact.get("hardware_id") == hardware_id
        and artifact.get("workload_sha256") == workload_sha256
        and artifact.get("profile_detail") == mode
        and artifact.get("hidden_env_names") == [],
        f"{backend} profile-{mode} overhead artifact identity differs",
    )
    samples = artifact.get("samples")
    require(
        isinstance(samples, list) and len(samples) == 3,
        f"{backend} profile-{mode} requires exactly three samples",
    )
    pids: set[int] = set()
    throughputs: list[float] = []
    entrypoint: str | None = None
    for index, sample in enumerate(samples, start=1):
        require(
            isinstance(sample, dict)
            and set(sample)
            == {
                "process_index",
                "process_id",
                "command_argv",
                "status",
                "output_throughput_tps",
                "error_count",
            }
            and sample.get("process_index") == index
            and sample.get("status") == "pass"
            and sample.get("error_count") == 0,
            f"{backend} profile-{mode} sample {index} identity/status differs",
        )
        pid = sample.get("process_id")
        require(
            isinstance(pid, int)
            and not isinstance(pid, bool)
            and pid > 0
            and pid not in pids,
            f"{backend} profile-{mode} sample process is not independent",
        )
        pids.add(pid)
        argv = sample.get("command_argv")
        require(
            isinstance(argv, list)
            and argv
            and all(isinstance(value, str) and value for value in argv),
            f"{backend} profile-{mode} sample command is invalid",
        )
        matches = [candidate for candidate in ("run", "serve") if candidate in argv]
        require(
            len(matches) == 1,
            f"{backend} profile-{mode} sample is not a real ferrum run/serve path",
        )
        require(
            entrypoint is None or entrypoint == matches[0],
            f"{backend} profile-{mode} sample entrypoint drifted",
        )
        entrypoint = matches[0]
        throughputs.append(
            finite_positive(
                sample.get("output_throughput_tps"),
                f"{backend} profile-{mode} sample throughput",
            )
        )
    return {"throughputs": throughputs, "entrypoint": entrypoint}


def validate_profile_jsonl(path: Path, *, mode: str, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise R2Error(f"cannot read {label}: {error}") from error
    require(lines and all(line.strip() for line in lines), f"{label} JSONL is empty or sparse")
    rows: list[dict[str, Any]] = []
    required = PROFILE_IDENTITY_FIELDS | {
        "profile_detail",
        "entrypoint",
        "stage_covered",
        "top_op_or_kernel",
        "device_attributed",
        "direct_fallback",
        "catalog_epoch_miss",
        "product_throughput_claim",
        "execution_fingerprint",
    }
    for index, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise R2Error(f"invalid {label} JSONL row {index}: {error}") from error
        require(
            isinstance(row, dict) and required.issubset(row),
            f"{label} JSONL row {index} lacks the profile identity/coverage fields",
        )
        require(
            row.get("profile_detail") == mode
            and row.get("entrypoint") in {"run", "serve"}
            and all(
                isinstance(row.get(field), str) and bool(row[field])
                for field in PROFILE_IDENTITY_FIELDS | {"execution_fingerprint"}
            )
            and all(
                isinstance(row.get(field), bool)
                for field in {
                    "stage_covered",
                    "top_op_or_kernel",
                    "device_attributed",
                    "direct_fallback",
                    "catalog_epoch_miss",
                    "product_throughput_claim",
                }
            ),
            f"{label} JSONL row {index} has an invalid identity/coverage value",
        )
        rows.append(row)
    return rows


def validate_profile(
    path: Path,
    *,
    source: dict[str, Any],
    r1: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = input_manifest(path, "manifest.json", "profile")
    root = manifest_path.parent
    manifest = read_json(manifest_path, "profile manifest")
    recorded_source = normalize_source(manifest.get("source"), "profile")
    require(
        set(manifest)
        == {
            "schema_version",
            "artifact_type",
            "status",
            "source",
            "overhead",
            "attribution",
        }
        and manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type") == "runtime_vnext_r2_profile_manifest"
        and manifest.get("status") == "ready",
        "profile manifest identity/source differs",
    )
    closure = source_closure(recorded_source, source)
    overhead = manifest.get("overhead")
    require(isinstance(overhead, list) and len(overhead) == 2, "profile requires one overhead row per backend")
    overhead_backends: set[str] = set()
    overhead_values: list[float] = []
    hardening_misses = 0
    for row in overhead:
        require(isinstance(row, dict) and set(row) == {"backend", "model_key", "summary"}, "profile overhead reference fields differ")
        backend = row.get("backend")
        model_key = row.get("model_key")
        require(
            backend in BACKENDS
            and backend not in overhead_backends
            and model_key == MODELS[0],
            "profile overhead must use the M1 product path once per backend",
        )
        overhead_backends.add(backend)
        summary_path = validate_ref(row.get("summary"), f"{backend} profile overhead", root=root)
        summary = read_json(summary_path, "profile overhead summary")
        require(
            set(summary)
            == {
                "schema_version",
                "artifact_type",
                "source",
                "backend",
                "model_key",
                "binary_sha256",
                "hardware_id",
                "workload_sha256",
                "off_throughput_repeats",
                "basic_throughput_repeats",
                "reported_overhead_ratio",
                "hidden_env_names",
                "off_artifact",
                "basic_artifact",
            }
            and summary.get("schema_version") == SCHEMA_VERSION
            and summary.get("artifact_type") == "runtime_vnext_r2_profile_overhead"
            and normalize_source(summary.get("source"), "profile overhead")
            == recorded_source
            and summary.get("backend") == backend
            and summary.get("model_key") == model_key
            and summary.get("binary_sha256") == r1["backend_binary_sha256"][backend]
            and summary.get("hardware_id") == r1["backend_hardware_id"][backend]
            and summary.get("hidden_env_names") == [],
            "profile overhead identity differs",
        )
        workload_sha = require_sha(
            summary.get("workload_sha256"), "profile overhead workload"
        )
        off = summary.get("off_throughput_repeats")
        basic = summary.get("basic_throughput_repeats")
        require(isinstance(off, list) and len(off) == 3 and isinstance(basic, list) and len(basic) == 3, "profile overhead requires three paired repeats")
        off_values = [finite_positive(value, "profile-off throughput") for value in off]
        basic_values = [finite_positive(value, "basic-profile throughput") for value in basic]
        off_path = validate_ref(
            summary.get("off_artifact"),
            f"{backend} profile-off artifact",
            root=root,
        )
        basic_path = validate_ref(
            summary.get("basic_artifact"),
            f"{backend} basic artifact",
            root=root,
        )
        off_evidence = validate_profile_overhead_artifact(
            off_path,
            mode="off",
            source=recorded_source,
            model_key=model_key,
            backend=backend,
            binary_sha256=r1["backend_binary_sha256"][backend],
            hardware_id=r1["backend_hardware_id"][backend],
            workload_sha256=workload_sha,
        )
        basic_evidence = validate_profile_overhead_artifact(
            basic_path,
            mode="basic",
            source=recorded_source,
            model_key=model_key,
            backend=backend,
            binary_sha256=r1["backend_binary_sha256"][backend],
            hardware_id=r1["backend_hardware_id"][backend],
            workload_sha256=workload_sha,
        )
        require(
            off_values == off_evidence["throughputs"]
            and basic_values == basic_evidence["throughputs"]
            and off_evidence["entrypoint"] == basic_evidence["entrypoint"],
            f"{backend} profile off/basic samples do not bind the same product workload",
        )
        ratio = max(0.0, (statistics.median(off_values) - statistics.median(basic_values)) / statistics.median(off_values))
        require(
            math.isclose(float(summary.get("reported_overhead_ratio", -1)), ratio, abs_tol=1e-9),
            "profile overhead summary differs",
        )
        require(ratio <= 0.07 + 1e-12, f"{backend} basic profile overhead exceeds 7%")
        hardening_misses += int(ratio > 0.02 + 1e-12)
        overhead_values.append(ratio)
    require(overhead_backends == set(BACKENDS), "profile overhead backend set differs")
    attribution = manifest.get("attribution")
    require(isinstance(attribution, list) and len(attribution) == 6, "profile requires six attribution lanes")
    seen: set[tuple[str, str]] = set()
    for row in attribution:
        require(isinstance(row, dict) and set(row) == {"model_key", "backend", "summary"}, "profile attribution reference fields differ")
        key = (str(row.get("model_key")), str(row.get("backend")))
        require(key[0] in MODELS and key[1] in BACKENDS and key not in seen, f"profile attribution identity differs: {key}")
        seen.add(key)
        summary_path = validate_ref(row.get("summary"), f"{key} profile attribution", root=root)
        summary = read_json(summary_path, "profile attribution summary")
        require(
            set(summary)
            == {
                "schema_version",
                "artifact_type",
                "source",
                "model_key",
                "backend",
                "binary_sha256",
                "hardware_id",
                "modes",
                "identity_completeness",
                "stage_coverage",
                "top_op_kernel_coverage",
                "run_profile_count",
                "serve_profile_count",
                "off_basic_direct_fallback_count",
                "off_basic_catalog_epoch_miss_count",
                "replay_full_product_throughput_claim_count",
                "full_basic_fingerprint_match_fraction",
                "hidden_env_names",
                "artifacts",
            }
            and summary.get("schema_version") == SCHEMA_VERSION
            and summary.get("artifact_type") == "runtime_vnext_r2_profile_attribution"
            and normalize_source(summary.get("source"), "profile attribution")
            == recorded_source
            and summary.get("model_key") == key[0]
            and summary.get("backend") == key[1]
            and summary.get("binary_sha256") == r1["backend_binary_sha256"][key[1]]
            and summary.get("hardware_id") == r1["backend_hardware_id"][key[1]]
            and summary.get("modes") == ["basic", "replay", "full"]
            and summary.get("hidden_env_names") == [],
            f"profile attribution identity differs: {key}",
        )
        identity = summary.get("identity_completeness")
        require(
            isinstance(identity, dict)
            and set(identity) == PROFILE_IDENTITY_FIELDS
            and all(float(value) == 1.0 for value in identity.values()),
            f"profile attribution identity chain is incomplete: {key}",
        )
        artifacts = summary.get("artifacts")
        require(
            isinstance(artifacts, dict)
            and set(artifacts) == {"basic", "replay", "full"},
            f"profile artifacts differ: {key}",
        )
        mode_rows: dict[str, list[dict[str, Any]]] = {}
        for mode, ref in artifacts.items():
            artifact_path = validate_ref(
                ref, f"{key} {mode} profile artifact", root=root
            )
            require(
                artifact_path.suffix == ".jsonl",
                f"{key} {mode} profile artifact must be JSONL",
            )
            mode_rows[mode] = validate_profile_jsonl(
                artifact_path, mode=mode, label=f"{key} {mode} profile"
            )
        combined = [row for rows in mode_rows.values() for row in rows]
        stage_coverage = sum(bool(row["stage_covered"]) for row in combined) / len(
            combined
        )
        top_rows = [row for row in combined if row["top_op_or_kernel"]]
        require(top_rows, f"profile attribution has no top op/kernel denominator: {key}")
        top_coverage = sum(bool(row["device_attributed"]) for row in top_rows) / len(
            top_rows
        )
        direct_fallbacks = sum(bool(row["direct_fallback"]) for row in combined)
        epoch_misses = sum(bool(row["catalog_epoch_miss"]) for row in combined)
        replay_full_claims = sum(
            bool(row["product_throughput_claim"])
            for mode in ("replay", "full")
            for row in mode_rows[mode]
        )
        basic_fingerprints = {
            str(row["execution_fingerprint"]) for row in mode_rows["basic"]
        }
        full_match_fraction = sum(
            str(row["execution_fingerprint"]) in basic_fingerprints
            for row in mode_rows["full"]
        ) / len(mode_rows["full"])
        run_count = sum(row["entrypoint"] == "run" for row in combined)
        serve_count = sum(row["entrypoint"] == "serve" for row in combined)
        require(
            math.isclose(
                float(summary.get("stage_coverage", -1)),
                stage_coverage,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(summary.get("top_op_kernel_coverage", -1)),
                top_coverage,
                abs_tol=1e-12,
            )
            and summary.get("run_profile_count") == run_count
            and summary.get("serve_profile_count") == serve_count
            and summary.get("off_basic_direct_fallback_count")
            == direct_fallbacks
            and summary.get("off_basic_catalog_epoch_miss_count") == epoch_misses
            and summary.get("replay_full_product_throughput_claim_count")
            == replay_full_claims
            and math.isclose(
                float(summary.get("full_basic_fingerprint_match_fraction", -1)),
                full_match_fraction,
                abs_tol=1e-12,
            ),
            f"profile attribution summary is not derived from JSONL: {key}",
        )
        require(
            float(summary.get("stage_coverage", -1)) >= 0.90
            and float(summary.get("stage_coverage", -1)) <= 1.0
            and float(summary.get("top_op_kernel_coverage", -1)) >= 0.80
            and float(summary.get("top_op_kernel_coverage", -1)) <= 1.0
            and isinstance(summary.get("run_profile_count"), int)
            and summary["run_profile_count"] > 0
            and isinstance(summary.get("serve_profile_count"), int)
            and summary["serve_profile_count"] > 0
            and summary.get("off_basic_direct_fallback_count") == 0
            and summary.get("off_basic_catalog_epoch_miss_count") == 0
            and summary.get("replay_full_product_throughput_claim_count") == 0
            and float(summary.get("full_basic_fingerprint_match_fraction", -1)) == 1.0,
            f"profile attribution acceptance failed: {key}",
        )
    require(seen == {(model, backend) for model in MODELS for backend in BACKENDS}, "profile attribution lane set differs")
    return {
        "manifest": file_ref(manifest_path),
        "source": recorded_source,
        "source_closure": closure,
        "overhead_backends": "2/2",
        "profile_backends": "2/2",
        "profile_mode_artifacts": "10/10",
        "attribution_lanes": "6/6",
        "max_basic_overhead_ratio": max(overhead_values),
        "hardening_over_2pct_count": hardening_misses,
    }


def contained_child_path(root: Path, raw: Any, label: str) -> Path:
    relative = Path(str(raw))
    require(
        str(relative) and not relative.is_absolute() and ".." not in relative.parts,
        f"{label} path is invalid",
    )
    path = (root.resolve() / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise R2Error(f"{label} escaped the profile artifact root") from error
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    return path


def validate_profile_aggregate(
    path: Path,
    *,
    source: dict[str, Any],
    r1: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = input_manifest(path, "manifest.json", "profile aggregate")
    root = manifest_path.parent.resolve()
    aggregate = read_json(manifest_path, "profile aggregate")
    required = {
        "schema_version",
        "schema",
        "artifact_type",
        "status",
        "artifact_dir",
        "model_key",
        "backends",
        "source",
        "backend_source_bindings",
        "workload_contract_sha256",
        "children",
        "summary",
        "evidence_files",
        "evidence_closure_sha256",
        "created_at",
        "pass_line",
    }
    require(
        set(aggregate) == required
        and aggregate.get("schema_version") == SCHEMA_VERSION
        and aggregate.get("schema")
        == "ferrum.runtime-vnext-r2-profile-aggregate.v1"
        and aggregate.get("artifact_type")
        == "runtime_vnext_r2_profile_aggregate"
        and aggregate.get("status") == "pass"
        and Path(str(aggregate.get("artifact_dir", ""))).resolve() == root
        and aggregate.get("model_key") == "m1"
        and aggregate.get("backends") == ["cuda", "metal"]
        and aggregate.get("pass_line")
        == f"FERRUM RUNTIME VNEXT R2 PROFILE AGGREGATE PASS: {root}",
        "profile aggregate identity/status/PASS differs",
    )
    parse_timestamp(aggregate.get("created_at"), "profile aggregate created_at")
    recorded = aggregate.get("source")
    require(isinstance(recorded, dict), "profile aggregate source is missing")
    recorded_source = normalize_source(
        {
            "git_sha": recorded.get("git_sha"),
            "git_tree_sha": recorded.get("tree_sha"),
            "dirty": recorded.get("dirty_status", {}).get("is_dirty"),
        },
        "profile aggregate",
    )
    require(
        recorded.get("dirty_status")
        == {"is_dirty": False, "status_short": []},
        "profile aggregate source is dirty",
    )
    closure = source_closure(recorded_source, source)
    try:
        import runtime_vnext_r2_profile_collector as profile_collector

        profile_collector.verify_evidence(root, aggregate.get("evidence_files"))
        require(
            aggregate.get("evidence_closure_sha256")
            == profile_collector.canonical_sha256(aggregate["evidence_files"]),
            "profile aggregate evidence closure differs",
        )
        current_profile_source = profile_collector.source_identity()
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise R2Error(f"profile aggregate provenance failed: {error}") from error
    require(
        recorded.get("collector_path")
        == "scripts/release/runtime_vnext_r2_profile_collector.py"
        and recorded.get("collector_sha256")
        == current_profile_source.get("collector_sha256")
        and recorded.get("product_source_closure")
        == current_profile_source.get("product_source_closure"),
        "profile aggregate collector/product source closure differs",
    )
    bindings = aggregate.get("backend_source_bindings")
    children = aggregate.get("children")
    require(
        isinstance(bindings, dict)
        and set(bindings) == set(BACKENDS)
        and isinstance(children, dict)
        and set(children) == set(BACKENDS),
        "profile aggregate backend child set differs",
    )
    workload_sha = require_sha(
        aggregate.get("workload_contract_sha256"),
        "profile aggregate workload contract",
    )
    child_manifests: dict[str, dict[str, Any]] = {}
    overheads: dict[str, float] = {}
    for backend in BACKENDS:
        metadata = children[backend]
        require(
            isinstance(metadata, dict)
            and set(metadata)
            == {
                "source_path",
                "artifact_dir",
                "manifest",
                "manifest_sha256",
                "binary_sha256",
                "pass_line",
            },
            f"profile {backend} child metadata differs",
        )
        staged_path = contained_child_path(
            root, metadata.get("manifest"), f"profile {backend} staged manifest"
        )
        original_path = Path(str(metadata.get("source_path", ""))).expanduser().resolve()
        original_root = Path(str(metadata.get("artifact_dir", ""))).expanduser().resolve()
        require(
            original_path == original_root / "manifest.json"
            and original_path.is_file()
            and not original_path.is_symlink()
            and sha256(staged_path)
            == sha256(original_path)
            == require_sha(
                metadata.get("manifest_sha256"),
                f"profile {backend} child manifest",
            ),
            f"profile {backend} staged/original manifest binding differs",
        )
        try:
            child = profile_collector.validate_backend_manifest(
                original_path, backend
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise R2Error(f"profile {backend} child failed: {error}") from error
        require(
            child == read_json(staged_path, f"profile {backend} staged manifest")
            and child.get("workload_contract_sha256") == workload_sha
            and metadata.get("pass_line") == child.get("pass_line")
            and metadata.get("binary_sha256")
            == child.get("inputs", {}).get("binary", {}).get("closure_sha256")
            == r1["backend_binary_sha256"][backend]
            and child.get("hidden_ferrum_env_count") == 0,
            f"profile {backend} child binary/workload/R1 binding differs",
        )
        child_source = child.get("source")
        binding = bindings[backend]
        require(
            isinstance(child_source, dict)
            and isinstance(binding, dict)
            and binding.get("git_sha") == child_source.get("git_sha")
            and binding.get("tree_sha") == child_source.get("tree_sha")
            and binding.get("product_source_closure")
            == child_source.get("product_source_closure")
            == recorded.get("product_source_closure"),
            f"profile {backend} product source binding differs",
        )
        runs = child.get("runs")
        require(isinstance(runs, list) and len(runs) == 8, f"profile {backend} run denominator differs")
        counts = {
            mode: sum(
                isinstance(row, dict) and row.get("mode") == mode for row in runs
            )
            for mode in ("off", "basic", "replay", "full")
        }
        require(
            counts == {"off": 3, "basic": 3, "replay": 1, "full": 1},
            f"profile {backend} mode denominator differs",
        )
        for row in runs:
            require(isinstance(row, dict), f"profile {backend} run row is invalid")
            stdout_path = contained_child_path(
                original_root,
                row.get("stdout"),
                f"profile {backend}/{row.get('name')} stdout",
            )
            try:
                recomputed_run = profile_collector.validate_run_stdout(stdout_path)
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                raise R2Error(
                    f"profile {backend}/{row.get('name')} stdout failed: {error}"
                ) from error
            require(
                recomputed_run == row.get("run_summary"),
                f"profile {backend}/{row.get('name')} run summary is not raw-derived",
            )
            receipt_path = contained_child_path(
                original_root,
                row.get("receipt"),
                f"profile {backend}/{row.get('name')} receipt",
            )
            timeout = 1800 if row.get("mode") == "full" else (900 if row.get("mode") == "replay" else 600)
            try:
                profile_collector.validate_receipt(
                    receipt_path,
                    expected_command=row.get("sanitized_command"),
                    expected_timeout=timeout,
                )
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                raise R2Error(
                    f"profile {backend}/{row.get('name')} receipt failed: {error}"
                ) from error
        try:
            recomputed_overhead = profile_collector.validate_overhead(runs)
            basic_events = [
                profile_collector.read_profile_events(
                    contained_child_path(
                        original_root,
                        row["profile"],
                        f"profile {backend}/{row['name']} JSONL",
                    )
                )
                for row in runs
                if row["mode"] == "basic"
            ]
            replay_row = next(row for row in runs if row["mode"] == "replay")
            full_row = next(row for row in runs if row["mode"] == "full")
            replay_events = profile_collector.read_profile_events(
                contained_child_path(
                    original_root,
                    replay_row["profile"],
                    f"profile {backend}/replay JSONL",
                )
            )
            full_events = profile_collector.read_profile_events(
                contained_child_path(
                    original_root,
                    full_row["profile"],
                    f"profile {backend}/full JSONL",
                )
            )
            recomputed_contract = (
                profile_collector.validate_identity_and_device_contract(
                    backend=backend,
                    basic_events=basic_events,
                    replay_events=replay_events,
                    full_events=full_events,
                )
            )
        except (KeyError, OSError, RuntimeError, StopIteration, TypeError, ValueError) as error:
            raise R2Error(f"profile {backend} raw contract failed: {error}") from error
        require(
            recomputed_overhead == child.get("overhead")
            and recomputed_contract == child.get("profile_contract")
            and replay_row["run_summary"]["content_sha256"]
            == full_row["run_summary"]["content_sha256"],
            f"profile {backend} overhead/identity summary is not raw-derived",
        )
        overheads[backend] = float(
            recomputed_overhead["median_duration_overhead"]
        )
        child_manifests[backend] = child
    expected_summary = {
        "backend_pass_count": 2,
        "profile_off_basic_independent_processes": 12,
        "diagnostic_processes": 4,
        "cuda_stage_coverage": child_manifests["cuda"]["profile_contract"][
            "stage_timing"
        ]["coverage"],
        "cuda_device_attribution_coverage": child_manifests["cuda"][
            "profile_contract"
        ]["device_timing"]["attribution_coverage"],
        "cuda_dispatch_timing_coverage": child_manifests["cuda"][
            "profile_contract"
        ]["device_timing"]["dispatch_timing_coverage"],
        "metal_device_timing_status": child_manifests["metal"][
            "profile_contract"
        ]["device_timing"]["status"],
        "metal_fabricated_device_time_count": 0,
        "cuda_basic_overhead": overheads["cuda"],
        "metal_basic_overhead": overheads["metal"],
    }
    require(
        aggregate.get("summary") == expected_summary,
        "profile aggregate summary is not child-derived",
    )
    return {
        "manifest": file_ref(manifest_path),
        "source": recorded_source,
        "source_closure": closure,
        "overhead_backends": "2/2",
        "profile_backends": "2/2",
        "profile_mode_artifacts": "10/10",
        "max_basic_overhead_ratio": max(overheads.values()),
        "hardening_over_2pct_count": sum(
            value > 0.02 + 1e-12 for value in overheads.values()
        ),
    }


def default_build_verifier(root: Path) -> dict[str, Any]:
    try:
        import validate_runtime_vnext_g07a_build_iteration as g07a

        return g07a.verify_manifest(
            root,
            REPO_ROOT,
            require_canonical=False,
            verify_checkout=False,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise R2Error(f"G07A build evidence failed: {error}") from error


def validate_build(
    path: Path,
    *,
    source: dict[str, Any],
    verifier: Callable[[Path], dict[str, Any]] = default_build_verifier,
) -> dict[str, Any]:
    manifest_path = input_manifest(path, "evidence.manifest.json", "build evidence")
    root = manifest_path.parent
    manifest = read_json(manifest_path, "G07A evidence manifest")
    recorded_source = normalize_source(manifest.get("source"), "build evidence")
    require(
        manifest.get("artifact_type") == "runtime_vnext_g07a_build_iteration_evidence"
        and manifest.get("status") == "ready"
        and manifest.get("mode") == "diagnostic"
        and manifest.get("repeats") == 1,
        "build evidence must be the source-matched G07A diagnostic repeats=1 lane",
    )
    closure = source_closure(recorded_source, source)
    summary = verifier(root)
    require(
        isinstance(summary, dict) and summary.get("mode") == "diagnostic",
        "G07A verifier did not confirm diagnostic mode",
    )
    targets = summary.get("scenario_targets") if isinstance(summary, dict) else None
    require(isinstance(targets, dict) and set(targets) == set(BUILD_TARGETS), "build scenario target set differs")
    for name, target in BUILD_TARGETS.items():
        row = targets[name]
        require(
            isinstance(row, dict)
            and float(row.get("target_seconds", -1)) == target
            and row.get("target_met") is True
            and finite_positive(row.get("p95_seconds"), f"build {name} duration") <= target,
            f"build scenario failed: {name}",
        )
    scenarios = manifest.get("scenarios")
    require(isinstance(scenarios, list) and len(scenarios) == 6, "build raw scenario denominator differs")
    seen: set[str] = set()
    cold = False
    hot = False
    for row in scenarios:
        require(isinstance(row, dict), "build raw scenario row is invalid")
        name = row.get("name")
        require(name in BUILD_TARGETS and name not in seen, f"build raw scenario identity differs: {name}")
        seen.add(name)
        samples = row.get("samples")
        require(
            row.get("sample_count") == 1
            and isinstance(samples, list)
            and len(samples) == 1,
            f"build diagnostic scenario must contain exactly one sample: {name}",
        )
        for sample in samples:
            require(isinstance(sample, dict), f"build sample is invalid: {name}")
            cache = sample.get("cache")
            require(isinstance(cache, dict) and isinstance(cache.get("scope"), str), f"build sample cache state is missing: {name}")
            scope = cache["scope"]
            cold = cold or "fresh" in scope or name == "clean-release"
            hot = hot or "shared" in scope or "clone" in scope or isinstance(sample.get("prewarm"), dict)
            build_step = sample.get("build")
            require(
                isinstance(build_step, dict)
                and isinstance(build_step.get("command"), list)
                and bool(build_step["command"])
                and all(
                    isinstance(value, str) and value
                    for value in build_step["command"]
                ),
                f"build sample command is missing: {name}",
            )
            output = sample.get("output")
            require(
                isinstance(output, dict)
                and SHA256_RE.fullmatch(str(output.get("sha256"))) is not None,
                f"build sample binary/archive SHA is missing: {name}",
            )
    require(seen == set(BUILD_TARGETS) and cold and hot, "build evidence lacks the six-scenario cold/hot contract")
    return {
        "manifest": file_ref(manifest_path),
        "source": recorded_source,
        "source_closure": closure,
        "mode": manifest["mode"],
        "scenario_count": 6,
        "targets": copy.deepcopy(targets),
        "cold_sample_present": cold,
        "hot_sample_present": hot,
    }


def acceptance(
    lanes: dict[str, dict[str, Any]],
    profile: dict[str, Any],
    build: dict[str, Any],
    floor: dict[str, Any],
) -> dict[str, Any]:
    require(set(lanes) == set(LANE_KEYS), "R2 performance lane set differs")
    cell_count = sum(int(row["cell_count"]) for row in lanes.values())
    repeat_count = sum(int(row["repeat_count"]) for row in lanes.values())
    request_count = sum(int(row["measured_request_count"]) for row in lanes.values())
    run_count = sum(int(row["run_sample_count"]) for row in lanes.values())
    require(cell_count == 33, f"R2 required cell denominator differs: {cell_count}")
    require(repeat_count == 99, f"R2 repeat denominator differs: {repeat_count}")
    require(request_count == 7380, f"R2 measured request denominator differs: {request_count}")
    require(run_count == 18, f"R2 run sample denominator differs: {run_count}")
    require(
        profile.get("overhead_backends") == "2/2"
        and profile.get("profile_backends") == "2/2"
        and profile.get("profile_mode_artifacts") == "10/10"
        and float(profile.get("max_basic_overhead_ratio", 1.0)) <= 0.07 + 1e-12,
        "R2 profile denominator/overhead differs",
    )
    require(build.get("scenario_count") == 6, "R2 build denominator differs")
    require(floor.get("row_count") == 132, "R2 floor denominator differs")
    return {
        "performance_lanes": "6/6",
        "required_cell_count": 33,
        "repeat_report_count": 99,
        "measured_request_count": 7380,
        "run_sample_count": 18,
        "floor_row_count": 132,
        "profile_overhead_backends": "2/2",
        "profile_backends": "2/2",
        "profile_mode_artifacts": "10/10",
        "build_scenarios": "6/6",
        "max_cell_cv": max(float(row["max_cv"]) for row in lanes.values()),
        "max_basic_profile_overhead_ratio": profile["max_basic_overhead_ratio"],
        "profile_hardening_over_2pct_count": profile["hardening_over_2pct_count"],
        "waiver_count": 0,
        "error_count": 0,
        "external_comparator_count": 0,
    }


def validate_inputs(
    paths: dict[str, Path],
    source: dict[str, Any],
    *,
    require_checked_in_catalog: bool,
    r1_verifier: Callable[[Path], dict[str, Any]] = default_r1_verifier,
    build_verifier: Callable[[Path], dict[str, Any]] = default_build_verifier,
    performance_validator: Callable[..., dict[str, Any]] = validate_ferrum_performance_lane,
    profile_validator: Callable[..., dict[str, Any]] = validate_profile_aggregate,
) -> dict[str, Any]:
    expected_paths = {"r1", *LANE_KEYS, "profile", "build", "floor_catalog"}
    require(set(paths) == expected_paths, "R2 input path set differs")
    r1 = validate_r1_outer(paths["r1"], source, verifier=r1_verifier)
    floors, floor_summary = validate_floor_catalog(
        paths["floor_catalog"], require_checked_in=require_checked_in_catalog
    )
    lanes = {
        key: performance_validator(
            paths[key],
            expected_model=model_key,
            expected_backend=backend,
            source=source,
            r1=r1,
            floors=floors,
        )
        for key, (model_key, backend) in LANE_KEYS.items()
    }
    validate_performance_hardware_cohort(lanes)
    profile = profile_validator(paths["profile"], source=source, r1=r1)
    build = validate_build(paths["build"], source=source, verifier=build_verifier)
    candidate_sources = {
        canonical_json_sha256(row["source"]) for row in lanes.values()
    }
    require(len(candidate_sources) == 1, "six performance lanes must share one candidate source")
    accepted = acceptance(lanes, profile, build, floor_summary)
    return {
        "r1": r1,
        "floor_catalog": floor_summary,
        "performance": lanes,
        "profile": profile,
        "build": build,
        "acceptance": accepted,
    }


def validate_performance_hardware_cohort(lanes: dict[str, dict[str, Any]]) -> None:
    for backend in BACKENDS:
        hardware_ids = {
            lanes[key]["hardware_id"]
            for key, (_, lane_backend) in LANE_KEYS.items()
            if lane_backend == backend
        }
        require(
            len(hardware_ids) == 1,
            f"three {backend} performance lanes must share one R2 hardware identity",
        )


def build_with_source(
    paths: dict[str, Path],
    out: Path,
    source: dict[str, Any],
    *,
    require_checked_in_catalog: bool,
    r1_verifier: Callable[[Path], dict[str, Any]],
    build_verifier: Callable[[Path], dict[str, Any]],
    performance_validator: Callable[..., dict[str, Any]] = validate_ferrum_performance_lane,
    profile_validator: Callable[..., dict[str, Any]] = validate_profile_aggregate,
) -> str:
    output = out.expanduser().resolve()
    require(
        REPO_ROOT not in output.parents and output != REPO_ROOT,
        "R2 output must stay outside the source tree",
    )
    require(
        not output.exists() or not any(output.iterdir()),
        f"R2 output must be absent or empty: {output}",
    )
    dependencies = validate_inputs(
        paths,
        source,
        require_checked_in_catalog=require_checked_in_catalog,
        r1_verifier=r1_verifier,
        build_verifier=build_verifier,
        performance_validator=performance_validator,
        profile_validator=profile_validator,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        pass_line = f"{PASS_PREFIX}: {output}"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_r2_performance_build_profile_manifest",
            "checkpoint_id": "R2",
            "lane": "runtime-vnext-r2",
            "status": "pass",
            "canonical": True,
            "artifact_dir": str(output),
            "source": source,
            "source_closure": dependencies["r1"]["source_closure"],
            "inputs": {key: file_ref(input_manifest(path, "evidence.manifest.json" if key == "build" else ("gate.manifest.json" if key == "r1" else ("floor-catalog.json" if key == "floor_catalog" else "manifest.json")), key)) for key, path in paths.items()},
            "dependencies": dependencies,
            "acceptance": dependencies["acceptance"],
            "unlocks": ["R3"],
            "does_not_prove": DOES_NOT_PROVE,
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "pass_line": pass_line,
        }
        write_json(staging / "manifest.json", manifest, exclusive=True)
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
        verify_manifest(
            output / "manifest.json",
            verify_checkout=False,
            expected_source=source,
            require_checked_in_catalog=require_checked_in_catalog,
            r1_verifier=r1_verifier,
            build_verifier=build_verifier,
            performance_validator=performance_validator,
            profile_validator=profile_validator,
        )
        return pass_line
    except BaseException:
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        if output.exists() and output.is_dir() and not any(output.iterdir()):
            output.rmdir()
        raise


def build(paths: dict[str, Path], out: Path) -> str:
    source = current_source()
    result = build_with_source(
        paths,
        out,
        source,
        require_checked_in_catalog=True,
        r1_verifier=default_r1_verifier,
        build_verifier=default_build_verifier,
        performance_validator=validate_ferrum_performance_lane,
        profile_validator=validate_profile_aggregate,
    )
    require(current_source() == source, "R2 source changed during validation")
    return result


def verify_manifest(
    path: Path,
    *,
    verify_checkout: bool = True,
    expected_source: dict[str, Any] | None = None,
    require_checked_in_catalog: bool = True,
    r1_verifier: Callable[[Path], dict[str, Any]] = default_r1_verifier,
    build_verifier: Callable[[Path], dict[str, Any]] = default_build_verifier,
    performance_validator: Callable[..., dict[str, Any]] = validate_ferrum_performance_lane,
    profile_validator: Callable[..., dict[str, Any]] = validate_profile_aggregate,
) -> dict[str, Any]:
    manifest_path = input_manifest(path, "manifest.json", "R2")
    root = manifest_path.parent
    manifest = read_json(manifest_path, "R2 manifest")
    required = {
        "schema_version",
        "artifact_type",
        "checkpoint_id",
        "lane",
        "status",
        "canonical",
        "artifact_dir",
        "source",
        "source_closure",
        "inputs",
        "dependencies",
        "acceptance",
        "unlocks",
        "does_not_prove",
        "created_at",
        "pass_line",
    }
    require(set(manifest) == required, "R2 manifest field set mismatch")
    require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type")
        == "runtime_vnext_r2_performance_build_profile_manifest"
        and manifest.get("checkpoint_id") == "R2"
        and manifest.get("lane") == "runtime-vnext-r2"
        and manifest.get("status") == "pass"
        and manifest.get("canonical") is True
        and Path(str(manifest.get("artifact_dir", ""))).resolve() == root
        and manifest.get("unlocks") == ["R3"]
        and manifest.get("does_not_prove") == DOES_NOT_PROVE
        and manifest.get("pass_line") == f"{PASS_PREFIX}: {root}",
        "R2 manifest identity/status/PASS differs",
    )
    parse_timestamp(manifest.get("created_at"), "R2 created_at")
    source = normalize_source(manifest.get("source"), "R2")
    expected = current_source() if verify_checkout else expected_source
    if expected is not None:
        require(source == expected, "R2 manifest source is stale")
    refs = manifest.get("inputs")
    require(isinstance(refs, dict) and set(refs) == {"r1", *LANE_KEYS, "profile", "build", "floor_catalog"}, "R2 input reference set differs")
    paths = {key: validate_ref(value, f"R2 input {key}") for key, value in refs.items()}
    dependencies = validate_inputs(
        paths,
        source,
        require_checked_in_catalog=require_checked_in_catalog,
        r1_verifier=r1_verifier,
        build_verifier=build_verifier,
        performance_validator=performance_validator,
        profile_validator=profile_validator,
    )
    require(
        manifest.get("source_closure") == dependencies["r1"]["source_closure"]
        and manifest.get("dependencies") == dependencies
        and manifest.get("acceptance") == dependencies["acceptance"],
        "R2 manifest dependency/acceptance summary drifted",
    )
    return {
        "kind": "vnext-r2",
        "child_manifest": file_ref(manifest_path),
        "source": source,
        "acceptance": copy.deepcopy(dependencies["acceptance"]),
    }


def expect_reject(action: Callable[[], Any], label: str) -> None:
    try:
        action()
    except R2Error:
        return
    raise AssertionError(f"{label}: accepted invalid evidence")


def self_test() -> None:
    require(
        "scripts/release/bounded_command.py" in R2_CONTROL_PLANE_FILES,
        "bounded command must remain an R2 evidence-control-plane file",
    )

    def occupancy_counter(
        claim_count: int, segment_count: int, physical_bytes: int
    ) -> dict[str, int]:
        return {
            "claim_count": claim_count,
            "segment_count": segment_count,
            "physical_bytes": physical_bytes,
        }

    def residency_occupancy(
        claim_count: int = 0,
        segment_count: int = 0,
        physical_bytes: int = 0,
    ) -> dict[str, dict[str, int]]:
        zero = occupancy_counter(0, 0, 0)
        return {
            "total": occupancy_counter(
                claim_count, segment_count, physical_bytes
            ),
            **{
                scope: (
                    occupancy_counter(claim_count, segment_count, physical_bytes)
                    if scope == "step"
                    else copy.deepcopy(zero)
                )
                for scope in POST_CELL_OCCUPANCY_SCOPE_FIELDS
            },
        }

    post_cell_fixture = {
        "engine": {"active_requests": 0, "queued_requests": 0},
        "admission": {
            "runtime_snapshot_available": True,
            "runtime_contract_error": None,
            "resource_authority": "plan_runtime",
            "queue_depth": 0,
            "active_sequences": 0,
            "active_prefill": 0,
            "active_decode": 0,
            "current_batch_size": None,
            "rejected_requests_total": 0,
            "failed_requests_total": 0,
            "effective_max_concurrent": 4,
            "maximum_active_sequences": 4,
            "preflight_effective_max_concurrent": 4,
        },
        "cache": {
            "prefix_cache": {
                "schema": "ferrum.runtime-vnext.executor-trace.v1",
                "pending_sequences": 0,
                "active_sequences": 0,
                "staged_prefill_requests": 0,
                "staged_prefill_sequences": 0,
                "pending_prefill_maintenance": 0,
                "executing_prefills": 0,
                "deferred_cleanup": {
                    "queued": 0,
                    "in_progress": 0,
                    "retryable": 0,
                    "quarantined": 0,
                    "panicked": 0,
                    "panicked_total": 0,
                },
                "static_bytes": 100,
                "runtime_memory_policy": {
                    "capacity_bytes": 1000,
                    "reserve_bytes": 400,
                    "maximum_active_sequences": 4,
                },
                "dynamic_pools": {
                    "maximum_active_sequences": 4,
                    "device_capacity_bytes": 1000,
                    "effective_device_usable_ceiling_bytes": 600,
                    "process_claimed_bytes": 164,
                    "budget_device_wide_usable_ceiling_bytes": 600,
                    "budget_claimed_bytes": 164,
                    "pools": [
                        {
                            "pool_id": "pool-a",
                            "domain_id": 1,
                            "resident_bytes": 64,
                            "pending_growth_bytes": 0,
                            "free_bytes": 16,
                            "largest_contiguous_bytes": 16,
                            "resident_chunks": 1,
                            "live_segments": 2,
                            "live_occupancy": {
                                "total": occupancy_counter(1, 2, 48),
                                "transient": residency_occupancy(),
                                "lane_stable": residency_occupancy(1, 2, 48),
                            },
                            "quarantined_chunks": 0,
                            "quarantined_bytes": 0,
                            "descriptor_mismatch_chunks": 0,
                            "publication_rejected_chunks": 0,
                            "poisoned": False,
                        }
                    ],
                },
            }
        },
    }
    post_cell_summary = validate_post_cell_typed_ledger(
        post_cell_fixture,
        typed_active_cap=4,
        memory_budget_bytes=600,
        label="post-cell fixture",
    )
    require(
        post_cell_summary["pool_count"] == 1
        and post_cell_summary["resident_bytes"] == 64
        and post_cell_summary["used_bytes"] == 48,
        "post-cell typed ledger summary differs",
    )
    active_post_cell = copy.deepcopy(post_cell_fixture)
    active_post_cell["engine"]["active_requests"] = 1
    expect_reject(
        lambda: validate_post_cell_typed_ledger(
            active_post_cell,
            typed_active_cap=4,
            memory_budget_bytes=600,
            label="active post-cell fixture",
        ),
        "active post-cell health",
    )
    transient_post_cell = copy.deepcopy(post_cell_fixture)
    transient_live = transient_post_cell["cache"]["prefix_cache"]["dynamic_pools"][
        "pools"
    ][0]["live_occupancy"]
    transient_live["transient"] = residency_occupancy(1, 2, 48)
    transient_live["lane_stable"] = residency_occupancy()
    expect_reject(
        lambda: validate_post_cell_typed_ledger(
            transient_post_cell,
            typed_active_cap=4,
            memory_budget_bytes=600,
            label="transient post-cell fixture",
        ),
        "transient post-cell occupancy",
    )
    mismatched_budget = copy.deepcopy(post_cell_fixture)
    mismatched_budget["cache"]["prefix_cache"]["dynamic_pools"][
        "budget_claimed_bytes"
    ] += 1
    expect_reject(
        lambda: validate_post_cell_typed_ledger(
            mismatched_budget,
            typed_active_cap=4,
            memory_budget_bytes=600,
            label="mismatched post-cell budget fixture",
        ),
        "mismatched post-cell budget",
    )

    with tempfile.TemporaryDirectory(prefix="ferrum-post-cell-selftest-") as raw_temp:
        fixture_root = Path(raw_temp)
        fixture_dir = fixture_root / "lane" / "attempts"
        body_path = fixture_dir / "post-01-random-c1-001.body.json"
        receipt_path = fixture_dir / "post-01-random-c1-001.receipt.json"
        checkpoint_path = fixture_root / "lane" / "cell-01.json"
        write_json(body_path, post_cell_fixture)
        write_json(
            receipt_path,
            {
                "returncode": 0,
                "http_status": 200,
                "body_sha256": sha256(body_path),
                "body_size_bytes": body_path.stat().st_size,
            },
        )

        def collector_ref(path: Path, kind: str) -> dict[str, Any]:
            return {
                "kind": kind,
                "path": path.relative_to(fixture_root).as_posix(),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }

        record_fixture = {
            "sequence": 1,
            "cell_id": "random:c1",
            "dataset": "random",
            "concurrency": 1,
        }
        write_json(
            checkpoint_path,
            {
                "contract": "ferrum.runtime-vnext.r2.completed-cell-checkpoint.v1",
                "sequence": 1,
                "cell_id": "random:c1",
                "completion_provenance": "legacy-active-envelope",
                "post_cell_idle": {
                    "status": "quiescent",
                    "body": collector_ref(body_path, "post-cell-idle-body"),
                    "receipt": collector_ref(
                        receipt_path, "post-cell-idle-receipt"
                    ),
                },
            },
        )
        current_bodies = load_post_cell_health_bodies(
            fixture_root,
            {"completed_cell_checkpoints": [collector_ref(checkpoint_path, "checkpoint")]},
            [record_fixture],
            legacy_single_epoch=False,
        )
        legacy_bodies = load_post_cell_health_bodies(
            fixture_root,
            {
                "session": {
                    "cell_quiescence": [
                        {
                            "status": "quiescent",
                            "active_requests": 0,
                            "queued_requests": 0,
                            "admission_queue_depth": 0,
                            "probe": {
                                "body": body_path.relative_to(fixture_root).as_posix(),
                                "body_sha256": sha256(body_path),
                                "receipt": receipt_path.relative_to(
                                    fixture_root
                                ).as_posix(),
                                "receipt_sha256": sha256(receipt_path),
                                "body_origin_path": "/stale/remote/body.json",
                            },
                        }
                    ]
                }
            },
            [record_fixture],
            legacy_single_epoch=True,
        )
        require(
            current_bodies == legacy_bodies == [post_cell_fixture],
            "current/legacy post-cell health adapters differ",
        )

    lifecycle_fixture: dict[str, dict[str, int]] = {}
    active_fixture: list[dict[str, Any]] = []
    cursor_ns = 10_000
    group_counts = [2, 4, 2, 4, 2, 4]
    for group_index, request_count in enumerate(group_counts):
        duration_ns = 10_000 if group_index % 2 == 0 else 100_000
        group_start = cursor_ns
        group_finish = group_start + duration_ns
        for request_index in range(request_count):
            accepted_ns = group_start
            completed_ns = group_finish
            if group_index == 0:
                midpoint_ns = group_start + duration_ns // 2
                accepted_ns = group_start if request_index == 0 else midpoint_ns
                completed_ns = midpoint_ns if request_index == 0 else group_finish
            lifecycle_fixture[f"fixture-{group_index}-{request_index}"] = {
                "accepted_ns": accepted_ns,
                "completed_ns": completed_ns,
            }
        if group_index == 3:
            active_fixture.extend(
                [
                    {
                        "started_unix_nanos": group_start,
                        "finished_unix_nanos": group_start + 70_000,
                        "eligible": True,
                        "active_requests_conservative": 4,
                    },
                    {
                        "started_unix_nanos": group_start + 70_000,
                        "finished_unix_nanos": group_finish,
                        "eligible": True,
                        "active_requests_conservative": 3,
                    },
                ]
            )
        else:
            active_fixture.append(
                {
                    "started_unix_nanos": group_start,
                    "finished_unix_nanos": group_finish,
                    "eligible": True,
                    # Warmup can reach the floor but must remain outside the denominator.
                    "active_requests_conservative": 4,
                }
            )
        cursor_ns = group_finish + 10_000
    replay_record = {
        "started_at": "1970-01-01T00:00:00Z",
        "finished_at": "1970-01-01T00:00:00.001000Z",
        "warmup_requests": 2,
        "num_prompts": 4,
        "n_repeats": 3,
    }
    replay_groups = closed_loop_request_groups(lifecycle_fixture, replay_record)
    require(
        [row["outstanding_request_count"] for row in replay_groups[0]["outstanding_intervals"]]
        == [1, 1]
        and replay_groups[0]["outstanding_intervals"][0]["end_unix_nanos"]
        == replay_groups[0]["outstanding_intervals"][1]["start_unix_nanos"],
        "same-timestamp completion/acceptance net delta differs",
    )
    replay = replay_active_floor_duty(
        groups=replay_groups,
        active_rows=active_fixture,
        active_floor=4,
    )
    require(
        replay["request_group_counts"] == group_counts
        and replay["eligible_duration_ns"] == 300_000
        and replay["active_at_or_above_floor_duration_ns"] == 270_000
        and math.isclose(replay["active_duty_cycle"], 0.9, abs_tol=1e-12)
        and math.isclose(
            replay["repeats"][1]["active_duty_cycle"], 0.7, abs_tol=1e-12
        ),
        "measured outstanding active-duty replay differs",
    )
    expect_reject(
        lambda: require_repeat_active_floor_duty(replay),
        "per-repeat active-duty floor",
    )
    sample_fixture = [
        {
            "record_type": "sample",
            "sequence": 1,
            "sampled_at": "1970-01-01T00:00:00.000010Z",
            "process_alive": True,
            "active_requests": 4,
            "active_probe_errors": [],
        },
        {
            "record_type": "sample",
            "sequence": 2,
            "sampled_at": "1970-01-01T00:00:00.000020Z",
            "process_alive": True,
            "active_requests": 3,
            "active_probe_errors": [],
        },
    ]
    interval_fixture = derive_active_interval_rows(sample_fixture, replay_record)
    for field, value in (
        ("active_requests_conservative", 4),
        ("eligible", False),
        ("started_at", "1970-01-01T00:00:00.000011Z"),
    ):
        tampered = copy.deepcopy(interval_fixture)
        tampered[0][field] = value
        expect_reject(
            lambda tampered=tampered: require_raw_derived_active_rows(
                tampered, interval_fixture
            ),
            f"tampered active interval {field}",
        )
    truncated_lifecycle = copy.deepcopy(lifecycle_fixture)
    truncated_lifecycle.pop("fixture-5-3")
    expect_reject(
        lambda: closed_loop_request_groups(truncated_lifecycle, replay_record),
        "truncated closed-loop lifecycle",
    )
    require(
        requires_active_floor("random", 32, "cuda")
        and requires_active_floor("random", 16, "metal")
        and not requires_active_floor("sharegpt", 32, "cuda")
        and not requires_active_floor("real-chat", 16, "metal"),
        "active-floor cell selection differs",
    )
    typed_metal_budget = 20 * 1024**3
    stable_summary = {
        "peak_memory_bytes": 1024**3,
        "memory_budget_bytes": typed_metal_budget,
        "physical_headroom_bytes": 4 * 1024**3,
        "swap_start_bytes": 10,
        "swap_end_bytes": 10,
        "thermal_start": "nominal",
        "thermal_end": "nominal",
        "power_mode_start": "normal",
        "power_mode_end": "normal",
        "thermal_throttling_count": 0,
        "oom_count": 0,
        "admission_error_count": 0,
        "active_probe_retry_error_count": 0,
    }
    with tempfile.TemporaryDirectory(prefix="ferrum-metal-swap-selftest-") as swap_temp:
        swap_root = Path(swap_temp)

        def swap_fixture(
            name: str,
            values: list[int],
            *,
            summary_overrides: dict[str, Any] | None = None,
            power_modes: list[str] | None = None,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            path = swap_root / f"{name}.jsonl"
            summary = {
                **stable_summary,
                "swap_start_bytes": values[0],
                "swap_end_bytes": values[-1],
                **(summary_overrides or {}),
            }
            modes = power_modes or ["normal"] * len(values)
            require(len(modes) == len(values), f"{name} power-mode fixture differs")
            rows: list[dict[str, Any]] = [
                {
                    "record_type": "header",
                    "collector_sha256": "a" * 64,
                }
            ]
            rows.extend(
                {
                    "record_type": "sample",
                    "sampled_at": (
                        datetime(2026, 8, 13, tzinfo=timezone.utc)
                        + timedelta(seconds=index)
                    ).isoformat(),
                    "swap_used_bytes": value,
                    "physical_headroom_bytes": summary[
                        "physical_headroom_bytes"
                    ],
                    "thermal_state": "nominal",
                    "power_mode": modes[index],
                }
                for index, value in enumerate(values)
            )
            rows.append({"record_type": "footer"})
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            evidence = derive_metal_swap_scope_evidence(
                path,
                sha256(path),
                summary,
                label=f"{name} fixture",
            )
            return summary, {"scope": name, **evidence}

        exact_summary, exact_swap = swap_fixture("exact-zero", [100] * 42)
        require(
            exact_swap["classification"] == "exact_zero"
            and exact_swap["raw_step_count"] == 0
            and exact_swap["raw_net_growth_bytes"] == 0,
            "exact-zero Metal swap evidence differs",
        )
        validate_metal_resource_contract(
            exact_summary,
            "exact-zero fixture",
            typed_memory_budget_bytes=typed_metal_budget,
            resource_evidence=exact_swap,
        )

        actual_step_sizes = [199_230, 587_202, 398_459, 450_888]
        actual_values = [1_754_854_850] * 3
        for step in actual_step_sizes:
            actual_values.extend([actual_values[-1] + step] * 3)
        actual_summary, actual_swap = swap_fixture(
            "actual-four-step-observation", actual_values
        )
        require(
            actual_swap["classification"] == "host_global_swap_observed"
            and actual_swap["raw_positive_step_count"] == 4
            and actual_swap["raw_max_positive_step_bytes"] == 587_202
            and actual_swap["raw_total_positive_growth_bytes"] == 1_635_779
            and len(actual_swap["positive_steps"]) == 4,
            "actual four-step Metal host-global observation differs",
        )
        validate_metal_resource_contract(
            actual_summary,
            "actual-four-step fixture",
            typed_memory_budget_bytes=typed_metal_budget,
            resource_evidence=actual_swap,
        )
        observed_lane = aggregate_metal_swap_evidence(
            [
                actual_swap,
                *(
                    {**exact_swap, "scope": f"actual-exact-{index}"}
                    for index in range(7)
                ),
            ],
            "actual-observation lane fixture",
        )
        require(
            observed_lane["classification"] == "host_global_swap_observed"
            and observed_lane["scope_count"] == 8
            and observed_lane["raw_positive_step_count"] == 4
            and observed_lane["raw_total_positive_growth_bytes"] == 1_635_779,
            "lane-global Metal host swap observation classification differs",
        )

        low_headroom = int(1.607 * 1024**3)
        m2_summary, m2_swap = swap_fixture(
            "m2-646mib-observation",
            [100] * 10 + [100 + 646 * 1024**2] * 32,
            summary_overrides={"physical_headroom_bytes": low_headroom},
        )
        require(
            m2_swap["classification"] == "host_global_swap_observed"
            and m2_swap["raw_total_positive_growth_bytes"] == 646 * 1024**2,
            "646 MiB Metal host-global observation was not preserved",
        )
        aggregate_metal_swap_evidence(
            [
                m2_swap,
                *(
                    {**exact_swap, "scope": f"m2-exact-{index}"}
                    for index in range(7)
                ),
            ],
            "m2-observation lane fixture",
        )
        expect_reject(
            lambda: validate_metal_resource_contract(
                m2_summary,
                "m2-low-headroom fixture",
                typed_memory_budget_bytes=typed_metal_budget,
                resource_evidence=m2_swap,
            ),
            "646 MiB observation with 1.607 GiB physical headroom",
        )

        _, low_power_swap = swap_fixture(
            "low-power-observation",
            [100] * 5,
            power_modes=["normal", "normal", "low-power", "normal", "normal"],
        )
        require(
            low_power_swap["low_power_sample_count"] == 1,
            "low-power raw evidence count differs",
        )
        hard_gate_rejects = (
            (
                {**exact_summary, "memory_budget_bytes": typed_metal_budget + 1},
                exact_swap,
                "typed budget mismatch",
            ),
            (
                {**exact_summary, "peak_memory_bytes": typed_metal_budget + 1},
                exact_swap,
                "peak RSS above typed budget",
            ),
            (
                {**exact_summary, "physical_headroom_bytes": 2 * 1024**3 - 1},
                {
                    **exact_swap,
                    "physical_headroom_bytes": 2 * 1024**3 - 1,
                },
                "physical headroom below 2 GiB",
            ),
            (
                {
                    **exact_summary,
                    "thermal_start": "throttled",
                    "thermal_end": "throttled",
                    "thermal_throttling_count": 1,
                },
                {**exact_swap, "thermal_throttling_count": 1},
                "thermal throttling",
            ),
            (exact_summary, low_power_swap, "low-power mode"),
            (
                {**exact_summary, "oom_count": 1},
                {**exact_swap, "oom_count": 1},
                "OOM event",
            ),
            (
                {**exact_summary, "admission_error_count": 1},
                {**exact_swap, "admission_error_count": 1},
                "admission error",
            ),
            (
                {**exact_summary, "active_probe_retry_error_count": 1},
                {**exact_swap, "active_probe_retry_error_count": 1},
                "resource probe error",
            ),
        )
        for rejected_summary, rejected_evidence, rejected_label in hard_gate_rejects:
            expect_reject(
                lambda rejected_summary=rejected_summary, rejected_evidence=rejected_evidence: validate_metal_resource_contract(
                    rejected_summary,
                    f"{rejected_label} fixture",
                    typed_memory_budget_bytes=typed_metal_budget,
                    resource_evidence=rejected_evidence,
                ),
                f"Metal hard resource gate: {rejected_label}",
            )
    source = {
        "git_sha": "a" * 40,
        "git_tree_sha": "b" * 40,
        "dirty": False,
    }
    binaries = {"cuda": "1" * 64, "metal": "2" * 64}
    hardware = {"cuda": "selftest-rtx4090", "metal": "selftest-mac"}
    timestamp = "2026-08-09T00:00:00+00:00"
    current_child_ref = {
        "path": "/fixture/current-r1.json",
        "sha256": "3" * 64,
        "size_bytes": 1,
    }
    current_r1_fixture = {
        "source": source,
        "source_closure": source_closure(source, source),
        "child_manifest": current_child_ref,
        "backend_binary_sha256": binaries,
        "backend_hardware_id": hardware,
    }
    exact_lane_fixture = {
        "profile_detail": "off",
        "source": source,
        "binary_sha256": binaries["cuda"],
        "correctness_manifest_sha256": current_child_ref["sha256"],
    }
    require(
        performance_lane_r1_source_closure(
            exact_lane_fixture, source, current_r1_fixture, "cuda"
        )["policy"]
        == "exact-source",
        "current R1 performance authority closure differs",
    )
    prior_source = {
        "git_sha": "c" * 40,
        "git_tree_sha": "d" * 40,
        "dirty": False,
    }
    prior_child_ref = {
        "path": "/fixture/prior-full-r1.json",
        "sha256": "4" * 64,
        "size_bytes": 1,
    }
    qualification_ref = {
        "path": "/fixture/impact-qualification.json",
        "sha256": "5" * 64,
        "size_bytes": 1,
    }
    evidence_binaries = {"cuda": "6" * 64, "metal": "7" * 64}
    cumulative_r1_fixture = copy.deepcopy(current_r1_fixture)
    cumulative_r1_fixture["cumulative_profile_off_evidence"] = {
        "mode": "cumulative-r1-prior-full-profile-off",
        "source": prior_source,
        "r1_child_manifest": prior_child_ref,
        "impact_qualification": qualification_ref,
        "backend_binary_sha256": evidence_binaries,
        "backend_hardware_id": hardware,
    }
    cumulative_lane_fixture = {
        "profile_detail": "off",
        "source": prior_source,
        "binary_sha256": evidence_binaries["cuda"],
        "correctness_manifest_sha256": prior_child_ref["sha256"],
    }
    cumulative_closure = performance_lane_r1_source_closure(
        cumulative_lane_fixture,
        source,
        cumulative_r1_fixture,
        "cuda",
    )
    require(
        cumulative_closure["policy"]
        == "cumulative-r1-prior-full-profile-off"
        and cumulative_closure["prior_r1_child_manifest"] == prior_child_ref
        and cumulative_closure["impact_qualification"] == qualification_ref,
        "cumulative R1 profile-off performance closure differs",
    )
    for marker, mutation in (
        ("profile mode", {"profile_detail": "basic"}),
        ("evidence binary", {"binary_sha256": "8" * 64}),
        (
            "prior correctness child",
            {"correctness_manifest_sha256": current_child_ref["sha256"]},
        ),
        (
            "prior source",
            {
                "source": {
                    "git_sha": "e" * 40,
                    "git_tree_sha": "f" * 40,
                    "dirty": False,
                }
            },
        ),
    ):
        invalid_lane = copy.deepcopy(cumulative_lane_fixture)
        invalid_lane.update(mutation)
        expect_reject(
            lambda invalid_lane=invalid_lane: performance_lane_r1_source_closure(
                invalid_lane,
                source,
                cumulative_r1_fixture,
                "cuda",
            ),
            f"cumulative profile-off {marker}",
        )
    expect_reject(
        lambda: performance_lane_r1_source_closure(
            cumulative_lane_fixture,
            source,
            current_r1_fixture,
            "cuda",
        ),
        "profile-off evidence without cumulative R1",
    )

    with tempfile.TemporaryDirectory(
        prefix="ferrum-r2-cumulative-chain-selftest-"
    ) as chain_temp:
        chain_root = Path(chain_temp).resolve()
        anchor_source = copy.deepcopy(prior_source)
        middle_source = {
            "git_sha": "8" * 40,
            "git_tree_sha": "9" * 40,
            "dirty": False,
        }
        anchor_acceptance = {
            "backend_binary_sha256": copy.deepcopy(evidence_binaries),
            "backend_hardware_id": copy.deepcopy(hardware),
        }
        middle_acceptance = {
            **copy.deepcopy(anchor_acceptance),
            "backend_binary_sha256": {"cuda": "a" * 64, "metal": "b" * 64},
            "evidence_backend_binary_sha256": copy.deepcopy(evidence_binaries),
        }
        head_acceptance = {
            **copy.deepcopy(middle_acceptance),
            "backend_binary_sha256": copy.deepcopy(binaries),
        }
        anchor_path = chain_root / "full.json"
        qualification_one = chain_root / "qualification-1.json"
        qualification_two = chain_root / "qualification-2.json"
        write_json(qualification_one, {}, exclusive=True)
        write_json(qualification_two, {}, exclusive=True)
        write_json(
            anchor_path,
            {
                "source": anchor_source,
                "dependencies": {
                    key: {} for key in ("r0", "matrices", "llama_dense", "acceptance")
                },
                "acceptance": anchor_acceptance,
            },
            exclusive=True,
        )
        middle_path = chain_root / "middle.json"
        write_json(
            middle_path,
            {
                "source": middle_source,
                "dependencies": {
                    "prior_r1": file_ref(anchor_path),
                    "impact_qualification": file_ref(qualification_one),
                    "acceptance": middle_acceptance,
                },
                "acceptance": middle_acceptance,
            },
            exclusive=True,
        )
        head_path = chain_root / "head.json"
        head_manifest = {
            "source": source,
            "dependencies": {
                "prior_r1": file_ref(middle_path),
                "impact_qualification": file_ref(qualification_two),
                "acceptance": head_acceptance,
            },
            "acceptance": head_acceptance,
        }
        write_json(head_path, head_manifest, exclusive=True)

        class FixtureR1Module:
            FULL_DEPENDENCY_KEYS = {
                "r0",
                "matrices",
                "llama_dense",
                "acceptance",
            }
            CUMULATIVE_DEPENDENCY_KEYS = {
                "prior_r1",
                "impact_qualification",
                "acceptance",
            }

            def __init__(self) -> None:
                self.calls: list[Path] = []

            @staticmethod
            def require_full_prior_manifest_shape(manifest: dict[str, Any]) -> None:
                require(
                    set(manifest.get("dependencies", {}))
                    == FixtureR1Module.FULL_DEPENDENCY_KEYS,
                    "fixture full R1 shape differs",
                )

            def verify_manifest(
                self,
                path: Path,
                *,
                verify_checkout: bool,
                expected_source: dict[str, Any],
            ) -> dict[str, Any]:
                del verify_checkout
                resolved = path.expanduser().resolve()
                self.calls.append(resolved)
                manifest = read_json(resolved, "fixture R1")
                require(
                    manifest.get("source") == expected_source,
                    "fixture R1 source differs",
                )
                dependencies = manifest.get("dependencies")
                if set(dependencies) == self.CUMULATIVE_DEPENDENCY_KEYS:
                    prior = validate_ref(dependencies["prior_r1"], "fixture prior R1")
                    prior_manifest = read_json(prior, "fixture prior R1")
                    self.verify_manifest(
                        prior,
                        verify_checkout=False,
                        expected_source=prior_manifest["source"],
                    )
                return {
                    "kind": "vnext-r1",
                    "source": copy.deepcopy(manifest["source"]),
                    "acceptance": copy.deepcopy(manifest["acceptance"]),
                }

        fixture_r1 = FixtureR1Module()
        head_summary = fixture_r1.verify_manifest(
            head_path, verify_checkout=False, expected_source=source
        )
        fixture_r1.calls.clear()
        chained_evidence = validate_cumulative_profile_off_evidence(
            head_path,
            head_manifest,
            head_summary,
            fixture_r1,
        )
        require(
            chained_evidence is not None
            and chained_evidence["source"] == anchor_source
            and chained_evidence["r1_child_manifest"] == file_ref(anchor_path)
            and chained_evidence["backend_binary_sha256"] == evidence_binaries
            and fixture_r1.calls.count(anchor_path) == 2,
            "cumulative-on-cumulative R1 did not bind the final full anchor",
        )

        forged_acceptance = copy.deepcopy(head_acceptance)
        forged_acceptance["evidence_backend_binary_sha256"]["cuda"] = "f" * 64
        forged_manifest = copy.deepcopy(head_manifest)
        forged_manifest["acceptance"] = forged_acceptance
        forged_manifest["dependencies"]["acceptance"] = forged_acceptance
        write_json(head_path, forged_manifest)
        forged_summary = fixture_r1.verify_manifest(
            head_path, verify_checkout=False, expected_source=source
        )
        expect_reject(
            lambda: validate_cumulative_profile_off_evidence(
                head_path,
                forged_manifest,
                forged_summary,
                fixture_r1,
            ),
            "cumulative chain forged final evidence authority",
        )

    cycle_a = Path("/fixture/cycle-a.json")
    cycle_b = Path("/fixture/cycle-b.json")
    cycle_manifests = {
        cycle_a: {
            "dependencies": {
                "prior_r1": {"path": str(cycle_b)},
                "impact_qualification": {"path": "/fixture/q-a.json"},
                "acceptance": {},
            }
        },
        cycle_b: {
            "dependencies": {
                "prior_r1": {"path": str(cycle_a)},
                "impact_qualification": {"path": "/fixture/q-b.json"},
                "acceptance": {},
            }
        },
    }

    def fixture_reference(value: Any, _: str) -> Path:
        require(isinstance(value, dict) and isinstance(value.get("path"), str), "fixture reference differs")
        return Path(value["path"])

    expect_reject(
        lambda: trace_r1_full_anchor(
            cycle_a,
            cycle_manifests[cycle_a],
            FixtureR1Module(),
            reference_validator=fixture_reference,
            manifest_reader=lambda path, _: cycle_manifests[path],
        ),
        "cumulative R1 dependency cycle",
    )
    expect_reject(
        lambda: trace_r1_full_anchor(
            Path("/fixture/bad-shape.json"),
            {"dependencies": {"prior_r1": {}}},
            FixtureR1Module(),
            reference_validator=fixture_reference,
            manifest_reader=lambda path, _: cycle_manifests[path],
        ),
        "cumulative R1 unsupported dependency shape",
    )

    with tempfile.TemporaryDirectory(prefix="ferrum-r2-selftest-") as raw_temp:
        temp = Path(raw_temp).resolve()

        def jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="ascii") as handle:
                for row in rows:
                    handle.write(
                        json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
                    )

        r1_root = temp / "r1"
        r1_root.mkdir()
        child_path = r1_root / "manifest.json"
        write_json(
            child_path,
            {
                "dependencies": {
                    key: {}
                    for key in ("r0", "matrices", "llama_dense", "acceptance")
                }
            },
            exclusive=True,
        )
        r1_acceptance = {
            "backend_binary_sha256": binaries,
            "backend_hardware_id": hardware,
        }
        r1_summary = {
            "kind": "vnext-r1",
            "child_manifest": file_ref(child_path),
            "source": source,
            "acceptance": r1_acceptance,
        }

        def r1_verifier(_: Path) -> dict[str, Any]:
            return copy.deepcopy(r1_summary)

        delegated = [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts/release/runtime_vnext_r1_product_correctness.py"
            ),
            "--out",
            str(r1_root),
        ]
        child_pass = (
            f"FERRUM RUNTIME VNEXT R1 PRODUCT CORRECTNESS PASS: {r1_root}"
        )
        receipt_files = {
            "run_gate.child.command.json": {"cmd": delegated},
            "run_gate.child.stdout": child_pass + "\n",
            "run_gate.child.stderr": "",
        }
        for name, value in receipt_files.items():
            path = r1_root / name
            if isinstance(value, dict):
                write_json(path, value, exclusive=True)
            else:
                path.write_text(value, encoding="utf-8")
        execution_refs = [
            relative_ref(r1_root, r1_root / name) for name in receipt_files
        ]
        outer = {
            "artifact_dir": str(r1_root),
            "binary": None,
            "child_artifacts": r1_summary,
            "child_execution_artifacts": execution_refs,
            "child_pass_line": child_pass,
            "child_returncode": 0,
            "command_line": ["run_gate.py", "vnext-r1"],
            "delegated_command_line": delegated,
            "dirty_status": {"is_dirty": False, "status_short": []},
            "duration_sec": 1.0,
            "error": None,
            "finished_at": timestamp,
            "git_sha": source["git_sha"],
            "lane": "vnext-r1",
            "model": None,
            "pass_line": f"FERRUM GATE vnext-r1 PASS: {r1_root}",
            "sanitized_env": {},
            "schema_version": 1,
            "started_at": timestamp,
            "status": "pass",
        }
        write_json(r1_root / "gate.manifest.json", outer, exclusive=True)

        performance_paths: dict[str, Path] = {}
        report_context: dict[str, Any] | None = None
        for lane_number, (lane_key, identity) in enumerate(
            LANE_KEYS.items(), start=1
        ):
            model_key, backend = identity
            root = temp / lane_key
            root.mkdir()
            typed_config = {
                "schema_version": SCHEMA_VERSION,
                "model_key": model_key,
                "backend": backend,
                "typed_active_cap": ACTIVE_FLOORS[(model_key, backend)],
                "memory_budget_bytes": 1024 * 1024 * 1024,
                "profile_detail": "off",
                "hidden_env_names": [],
            }
            typed_path = root / "typed-config.json"
            write_json(typed_path, typed_config, exclusive=True)
            config_sha = sha256(typed_path)
            model_sha = canonical_json_sha256({"model": model_key})
            cells: list[dict[str, Any]] = []
            ordered_cells = sorted(
                expected_cells(backend),
                key=lambda item: (item[0] != "random", item[0], item[1]),
            )
            for cell_number, (dataset, concurrency) in enumerate(
                ordered_cells, start=1
            ):
                count = requests_per_repeat(dataset)
                throughput = 100.0
                if dataset == "random":
                    if backend == "cuda":
                        throughput = {1: 100.0, 4: 110.0, 16: 120.0, 32: 130.0}[
                            concurrency
                        ]
                    else:
                        throughput = {1: 100.0, 4: 110.0, 16: 115.0}[
                            concurrency
                        ]
                requests: list[dict[str, Any]] = []
                for repeat in (1, 2, 3):
                    for index in range(1, count + 1):
                        requests.append(
                            {
                                "repeat": repeat,
                                "index": index,
                                "completed": True,
                                "usage_token_source": "usage",
                                "input_tokens": (
                                    (256 if backend == "cuda" else 64)
                                    if dataset == "random"
                                    else 48
                                ),
                                "output_tokens": 128,
                                "error": None,
                                "bad_output": False,
                                "malformed_sse": False,
                                "done_count": 1,
                                "ttft_ms": 10.0,
                                "tpot_ms": 5.0,
                                "itl_eligible": True,
                                "itl_ms": 1.0,
                                "fields_complete": True,
                            }
                        )
                repeats: list[dict[str, Any]] = []
                output_tokens = count * 128
                for repeat in (1, 2, 3):
                    row = {
                        "repeat": repeat,
                        "wall_time_seconds": output_tokens / throughput,
                        "output_tokens": output_tokens,
                        "output_throughput_tps": throughput,
                        "ttft_p95_ms": 10.0,
                        "tpot_p95_ms": 5.0,
                        "steady_decode_tps": 100.0,
                        "ci_low_tps": throughput * 0.99,
                        "ci_high_tps": throughput * 1.01,
                        "observed_max_active": (
                            ACTIVE_FLOORS[(model_key, backend)]
                            if dataset == "random"
                            and concurrency == max(MAIN_CONCURRENCY[backend])
                            else 1
                        ),
                        "eligible_interval_seconds": 1.0,
                        "active_floor_duty_cycle": 0.9,
                        "active_timeline_complete": True,
                        "peak_memory_bytes": 100 * 1024 * 1024,
                        "memory_budget_bytes": 1024 * 1024 * 1024,
                    }
                    if backend == "cuda":
                        row["physical_vram_headroom_bytes"] = 1024 * 1024 * 1024
                    else:
                        row["swap_growth_bytes"] = 0
                        row["thermal_throttling_count"] = 0
                    repeats.append(row)
                argv = [
                    "ferrum",
                    "bench-serve",
                    "--fail-on-error",
                    "--require-ci",
                    "--seed",
                    "9271",
                    "--n-repeats",
                    "3",
                    "--concurrency",
                    str(concurrency),
                    "--num-prompts",
                    str(count),
                    "--dataset",
                    "random" if dataset == "random" else "sharegpt",
                ]
                if dataset == "random":
                    argv.extend(
                        [
                            "--random-input-len",
                            "256" if backend == "cuda" else "64",
                            "--random-output-len",
                            "128",
                        ]
                    )
                report = {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": "runtime_vnext_r2_performance_cell_report",
                    "model_key": model_key,
                    "backend": backend,
                    "dataset": dataset,
                    "dataset_sha256": canonical_json_sha256(
                        {"dataset": dataset, "model": model_key}
                    ),
                    "concurrency": concurrency,
                    "source": source,
                    "binary_sha256": binaries[backend],
                    "hardware_id": hardware[backend],
                    "model_sha256": model_sha,
                    "typed_config_sha256": config_sha,
                    "profile_mode": "off",
                    "benchmark_argv": argv,
                    "requested_input_tokens": (
                        (256 if backend == "cuda" else 64)
                        if dataset == "random"
                        else None
                    ),
                    "requested_output_tokens": 128,
                    "requests_per_repeat": count,
                    "n_repeats": 3,
                    "warmups": [
                        {
                            "repeat": repeat,
                            "index": 1,
                            "completed": True,
                            "error": None,
                        }
                        for repeat in (1, 2, 3)
                    ],
                    "requests": requests,
                    "repeats": repeats,
                }
                report_path = root / f"cell-{cell_number}.json"
                write_json(report_path, report, exclusive=True)
                cells.append(
                    {
                        "dataset": dataset,
                        "concurrency": concurrency,
                        "report": relative_ref(root, report_path),
                    }
                )
                if report_context is None:
                    report_context = {
                        "path": report_path,
                        "report": copy.deepcopy(report),
                        "model_key": model_key,
                        "backend": backend,
                        "dataset": dataset,
                        "concurrency": concurrency,
                        "model_sha": model_sha,
                        "config_sha": config_sha,
                    }
            run_samples: list[dict[str, Any]] = []
            for index in (1, 2, 3):
                command_path = root / f"run-{index}-command.json"
                result_path = root / f"run-{index}-result.json"
                write_json(
                    command_path,
                    {"argv": ["ferrum", "run", "--model", model_key]},
                    exclusive=True,
                )
                write_json(
                    result_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_type": "runtime_vnext_r2_run_sample",
                        "source": source,
                        "model_key": model_key,
                        "backend": backend,
                        "binary_sha256": binaries[backend],
                        "hardware_id": hardware[backend],
                        "model_sha256": model_sha,
                        "typed_config_sha256": config_sha,
                        "process_id": lane_number * 100 + index,
                        "status": "pass",
                        "output_tokens": 64,
                        "steady_decode_tps": 100.0,
                        "engine_infer_e2e_tps": 95.0,
                        "error_count": 0,
                        "independent_process": True,
                    },
                    exclusive=True,
                )
                run_samples.append(
                    {
                        "process_index": index,
                        "command": relative_ref(root, command_path),
                        "result": relative_ref(root, result_path),
                    }
                )
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "runtime_vnext_r2_performance_lane_manifest",
                "status": "ready",
                "created_at": timestamp,
                "source": source,
                "model_key": model_key,
                "backend": backend,
                "binary_sha256": binaries[backend],
                "hardware_id": hardware[backend],
                "model_sha256": model_sha,
                "typed_config": relative_ref(root, typed_path),
                "profile_mode": "off",
                "hidden_env_names": [],
                "production_legacy_selection_count": 0,
                "cells": cells,
                "run_samples": run_samples,
            }
            write_json(root / "manifest.json", manifest, exclusive=True)
            performance_paths[lane_key] = root

        catalog_path = temp / "generated-floor-catalog.json"
        first_catalog = floor_catalog_from_collectors(
            performance_paths, lane_loader=load_calibration_lane
        )
        require(
            first_catalog
            == floor_catalog_from_collectors(
                performance_paths, lane_loader=load_calibration_lane
            ),
            "floor catalog generation is not deterministic",
        )
        write_floor_catalog_template(
            performance_paths,
            catalog_path,
            lane_loader=load_calibration_lane,
        )

        profile_root = temp / "profile"
        profile_root.mkdir()
        overhead_refs: list[dict[str, Any]] = []
        for backend_number, backend in enumerate(BACKENDS, start=1):
            workload_sha = canonical_json_sha256(
                {"model": MODELS[0], "backend": backend, "workload": "m1"}
            )
            mode_values = {"off": [100.0, 100.0, 100.0], "basic": [96.0] * 3}
            artifacts: dict[str, Path] = {}
            for mode, values in mode_values.items():
                artifact_path = profile_root / f"overhead-{backend}-{mode}.json"
                write_json(
                    artifact_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_type": "runtime_vnext_r2_profile_overhead_samples",
                        "status": "ready",
                        "source": source,
                        "model_key": MODELS[0],
                        "backend": backend,
                        "binary_sha256": binaries[backend],
                        "hardware_id": hardware[backend],
                        "workload_sha256": workload_sha,
                        "profile_detail": mode,
                        "hidden_env_names": [],
                        "samples": [
                            {
                                "process_index": index,
                                "process_id": backend_number * 1000
                                + (0 if mode == "off" else 10)
                                + index,
                                "command_argv": [
                                    "ferrum",
                                    "serve",
                                    "--profile-detail",
                                    mode,
                                ],
                                "status": "pass",
                                "output_throughput_tps": value,
                                "error_count": 0,
                            }
                            for index, value in enumerate(values, start=1)
                        ],
                    },
                    exclusive=True,
                )
                artifacts[mode] = artifact_path
            summary_path = profile_root / f"overhead-{backend}-summary.json"
            write_json(
                summary_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": "runtime_vnext_r2_profile_overhead",
                    "source": source,
                    "backend": backend,
                    "model_key": MODELS[0],
                    "binary_sha256": binaries[backend],
                    "hardware_id": hardware[backend],
                    "workload_sha256": workload_sha,
                    "off_throughput_repeats": mode_values["off"],
                    "basic_throughput_repeats": mode_values["basic"],
                    "reported_overhead_ratio": 0.04,
                    "hidden_env_names": [],
                    "off_artifact": relative_ref(profile_root, artifacts["off"]),
                    "basic_artifact": relative_ref(
                        profile_root, artifacts["basic"]
                    ),
                },
                exclusive=True,
            )
            overhead_refs.append(
                {
                    "backend": backend,
                    "model_key": MODELS[0],
                    "summary": relative_ref(profile_root, summary_path),
                }
            )
        attribution_refs: list[dict[str, Any]] = []
        for lane_key, (model_key, backend) in LANE_KEYS.items():
            artifacts: dict[str, dict[str, Any]] = {}
            for mode in ("basic", "replay", "full"):
                rows = []
                for entrypoint in ("run", "serve"):
                    rows.append(
                        {
                            "profile_detail": mode,
                            "entrypoint": entrypoint,
                            "plan": f"{lane_key}-plan",
                            "node": f"{entrypoint}-node",
                            "operation": "attention",
                            "resource": backend,
                            "provider": "ferrum",
                            "kernel": "selftest-kernel",
                            "stage_covered": True,
                            "top_op_or_kernel": True,
                            "device_attributed": True,
                            "direct_fallback": False,
                            "catalog_epoch_miss": False,
                            "product_throughput_claim": False,
                            "execution_fingerprint": f"{lane_key}-{entrypoint}",
                        }
                    )
                artifact_path = profile_root / f"{lane_key}-{mode}.jsonl"
                jsonl(artifact_path, rows)
                artifacts[mode] = relative_ref(profile_root, artifact_path)
            summary_path = profile_root / f"{lane_key}-profile-summary.json"
            write_json(
                summary_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": "runtime_vnext_r2_profile_attribution",
                    "source": source,
                    "model_key": model_key,
                    "backend": backend,
                    "binary_sha256": binaries[backend],
                    "hardware_id": hardware[backend],
                    "modes": ["basic", "replay", "full"],
                    "identity_completeness": {
                        field: 1.0 for field in PROFILE_IDENTITY_FIELDS
                    },
                    "stage_coverage": 1.0,
                    "top_op_kernel_coverage": 1.0,
                    "run_profile_count": 3,
                    "serve_profile_count": 3,
                    "off_basic_direct_fallback_count": 0,
                    "off_basic_catalog_epoch_miss_count": 0,
                    "replay_full_product_throughput_claim_count": 0,
                    "full_basic_fingerprint_match_fraction": 1.0,
                    "hidden_env_names": [],
                    "artifacts": artifacts,
                },
                exclusive=True,
            )
            attribution_refs.append(
                {
                    "model_key": model_key,
                    "backend": backend,
                    "summary": relative_ref(profile_root, summary_path),
                }
            )
        profile_manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_r2_profile_manifest",
            "status": "ready",
            "source": source,
            "overhead": overhead_refs,
            "attribution": attribution_refs,
        }
        write_json(
            profile_root / "manifest.json", profile_manifest, exclusive=True
        )

        build_root = temp / "build"
        build_root.mkdir()
        scenarios = []
        for name in BUILD_TARGETS:
            scenarios.append(
                {
                    "name": name,
                    "sample_count": 1,
                    "samples": [
                        {
                            "cache": {
                                "scope": (
                                    "fresh-per-sample"
                                    if name == "clean-release"
                                    else "declared-shared-incremental"
                                )
                            },
                            "prewarm": None,
                            "build": {"command": ["cargo", "build", name]},
                            "output": {
                                "sha256": canonical_json_sha256({"build": name})
                            },
                        }
                    ],
                }
            )
        build_manifest = {
            "schema_version": 4,
            "artifact_type": "runtime_vnext_g07a_build_iteration_evidence",
            "status": "ready",
            "mode": "diagnostic",
            "repeats": 1,
            "source": source,
            "scenarios": scenarios,
        }
        write_json(
            build_root / "evidence.manifest.json",
            build_manifest,
            exclusive=True,
        )

        def build_verifier(_: Path) -> dict[str, Any]:
            return {
                "mode": "diagnostic",
                "scenario_targets": {
                    name: {
                        "p95_seconds": target / 2,
                        "target_seconds": target,
                        "target_met": True,
                    }
                    for name, target in BUILD_TARGETS.items()
                },
            }

        paths = {
            "r1": r1_root,
            **performance_paths,
            "profile": profile_root,
            "build": build_root,
            "floor_catalog": catalog_path,
        }
        dependencies = validate_inputs(
            paths,
            source,
            require_checked_in_catalog=False,
            r1_verifier=r1_verifier,
            build_verifier=build_verifier,
            performance_validator=validate_performance_lane,
            profile_validator=validate_profile,
        )
        require(
            dependencies["acceptance"]["required_cell_count"] == 33
            and dependencies["acceptance"]["repeat_report_count"] == 99
            and dependencies["acceptance"]["measured_request_count"] == 7380
            and dependencies["acceptance"]["run_sample_count"] == 18,
            "positive fixture denominator differs",
        )
        mixed_hardware_lanes = {
            key: {"hardware_id": hardware[backend]}
            for key, (_, backend) in LANE_KEYS.items()
        }
        mixed_hardware_lanes["m3_cuda"]["hardware_id"] = "other-rtx4090"
        expect_reject(
            lambda: validate_performance_hardware_cohort(mixed_hardware_lanes),
            "mixed R2 hardware cohort",
        )
        output = temp / "r2-output"
        pass_line = build_with_source(
            paths,
            output,
            source,
            require_checked_in_catalog=False,
            r1_verifier=r1_verifier,
            build_verifier=build_verifier,
            performance_validator=validate_performance_lane,
            profile_validator=validate_profile,
        )
        require(pass_line == f"{PASS_PREFIX}: {output}", "R2 PASS line differs")
        verify_manifest(
            output,
            verify_checkout=False,
            expected_source=source,
            require_checked_in_catalog=False,
            r1_verifier=r1_verifier,
            build_verifier=build_verifier,
            performance_validator=validate_performance_lane,
            profile_validator=validate_profile,
        )

        bad_catalog = copy.deepcopy(first_catalog)
        bad_catalog["rows"][0]["value"] *= 2
        bad_catalog.pop("canonical_sha256")
        bad_catalog["canonical_sha256"] = canonical_json_sha256(bad_catalog)
        bad_catalog_path = temp / "bad-floor-catalog.json"
        write_json(bad_catalog_path, bad_catalog, exclusive=True)
        expect_reject(
            lambda: validate_floor_catalog(
                bad_catalog_path, require_checked_in=False
            ),
            "non-derived floor",
        )

        bad_build = copy.deepcopy(build_manifest)
        bad_build["scenarios"][0]["samples"][0]["build"]["command"] = []
        bad_build_root = temp / "bad-build"
        bad_build_root.mkdir()
        write_json(
            bad_build_root / "evidence.manifest.json", bad_build, exclusive=True
        )
        expect_reject(
            lambda: validate_build(
                bad_build_root, source=source, verifier=build_verifier
            ),
            "missing build command",
        )

        assert report_context is not None
        bad_report = copy.deepcopy(report_context["report"])
        bad_report["requests"].pop()
        bad_report_path = temp / "bad-report.json"
        write_json(bad_report_path, bad_report, exclusive=True)
        expect_reject(
            lambda: validate_performance_report(
                bad_report_path,
                model_key=report_context["model_key"],
                backend=report_context["backend"],
                dataset=report_context["dataset"],
                concurrency=report_context["concurrency"],
                source=source,
                binary_sha256=binaries[report_context["backend"]],
                hardware_id=hardware[report_context["backend"]],
                model_sha256=report_context["model_sha"],
                typed_config_sha256=report_context["config_sha"],
            ),
            "missing measured request",
        )

        bad_profile = copy.deepcopy(profile_manifest)
        bad_profile["overhead"].pop()
        bad_profile_root = temp / "bad-profile"
        bad_profile_root.mkdir()
        write_json(
            bad_profile_root / "manifest.json", bad_profile, exclusive=True
        )
        expect_reject(
            lambda: validate_profile(
                bad_profile_root,
                source=source,
                r1={
                    "backend_binary_sha256": binaries,
                    "backend_hardware_id": hardware,
                },
            ),
            "missing profile backend",
        )
    print(SELFTEST_PASS_LINE)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the Runtime vNext R2 performance/build/profile checkpoint"
    )
    parser.add_argument("--r1", type=Path)
    for lane_key in LANE_KEYS:
        parser.add_argument("--" + lane_key.replace("_", "-"), dest=lane_key, type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--floor-catalog", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--write-floor-catalog-template",
        type=Path,
        metavar="PATH",
        help="write a frozen calibration catalog from the six collector manifests",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            self_test()
            return 0
        performance_paths = {
            lane_key: getattr(args, lane_key) for lane_key in LANE_KEYS
        }
        missing_performance = [
            "--" + key.replace("_", "-")
            for key, value in performance_paths.items()
            if value is None
        ]
        if missing_performance:
            parser.error(
                "the following six-lane inputs are required: "
                + ", ".join(missing_performance)
            )
        if args.write_floor_catalog_template is not None:
            print(
                write_floor_catalog_template(
                    performance_paths, args.write_floor_catalog_template
                )
            )
            return 0
        formal = {
            "r1": args.r1,
            **performance_paths,
            "profile": args.profile,
            "build": args.build,
            "floor_catalog": args.floor_catalog,
        }
        missing = [key for key, value in formal.items() if value is None]
        if args.out is None:
            missing.append("out")
        if missing:
            parser.error(
                "formal R2 validation requires: "
                + ", ".join("--" + key.replace("_", "-") for key in missing)
            )
        print(build({key: value for key, value in formal.items()}, args.out))
        return 0
    except (OSError, R2Error, TypeError, ValueError) as error:
        print(f"{PASS_PREFIX.replace(' PASS', ' FAIL')}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
