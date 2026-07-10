#!/usr/bin/env python3
"""Validate the frozen Runtime vNext G00 legacy baseline.

The gate is deliberately evidence-only.  It does not build binaries, run
models, download weights, or manufacture missing rows.  Collectors place
artifacts under one root; this validator rejects incomplete, stale, dirty, or
internally inconsistent evidence and writes a normalized manifest.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import runtime_vnext_baseline_scenarios as scenario_runner
    import runtime_vnext_build_timing as build_timing
    import runtime_vnext_hardware_probe as hardware_probe
    import runtime_vnext_resource_sampler as resource_sampler
except ModuleNotFoundError:
    from scripts.release import runtime_vnext_baseline_scenarios as scenario_runner
    from scripts.release import runtime_vnext_build_timing as build_timing
    from scripts.release import runtime_vnext_hardware_probe as hardware_probe
    from scripts.release import runtime_vnext_resource_sampler as resource_sampler


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_CATALOG_PATH = REPO_ROOT / "scripts/release/configs/runtime_vnext_models.json"
PRESETS_CATALOG_PATH = REPO_ROOT / "scripts/release/configs/runtime_vnext_generation_presets.json"
BUG_CATALOG_PATH = REPO_ROOT / "scripts/release/configs/runtime_vnext_historical_bugs.json"
CORRECTNESS_EXPECTATIONS_PATH = (
    REPO_ROOT / "scripts/release/configs/runtime_vnext_legacy_correctness_expectations.json"
)
INVENTORY_ANALYZER_PATH = REPO_ROOT / "scripts/release/runtime_vnext_inventory.py"
MODEL_RESOLVER_PATH = REPO_ROOT / "scripts/release/runtime_vnext_model_resolver.py"
HARDWARE_PROBE_PATH = REPO_ROOT / "scripts/release/runtime_vnext_hardware_probe.py"
BUILD_TIMING_COLLECTOR_PATH = REPO_ROOT / "scripts/release/runtime_vnext_build_timing.py"
INVENTORY_REVIEW_PATH = REPO_ROOT / "scripts/release/configs/runtime_vnext_inventory_review.json"
SCENARIO_RUNNER_PATH = REPO_ROOT / "scripts/release/runtime_vnext_baseline_scenarios.py"
RESOURCE_SAMPLER_PATH = REPO_ROOT / "scripts/release/runtime_vnext_resource_sampler.py"
CONTRACT_PATHS = (
    REPO_ROOT / "docs/goals/runtime-vnext-0.8.0-2026-07-10/G00_BASELINE.md",
    REPO_ROOT / "docs/goals/runtime-vnext-0.8.0-2026-07-10/MODEL_MATRIX.md",
    MODELS_CATALOG_PATH,
    PRESETS_CATALOG_PATH,
    BUG_CATALOG_PATH,
    CORRECTNESS_EXPECTATIONS_PATH,
    INVENTORY_ANALYZER_PATH,
    MODEL_RESOLVER_PATH,
    HARDWARE_PROBE_PATH,
    BUILD_TIMING_COLLECTOR_PATH,
    SCENARIO_RUNNER_PATH,
    RESOURCE_SAMPLER_PATH,
    Path(__file__).resolve(),
)
FROZEN_LEGACY_SHA = "cff4c47765ef3259b8a04890187d99c60da86394"
PASS_PREFIX = "FERRUM RUNTIME VNEXT G00 BASELINE PASS"
CORRECTNESS_PASS_PREFIX = "FERRUM RUNTIME VNEXT G00 LEGACY CORRECTNESS PASS"
SELFTEST_FAST_PASS_LINE = "FERRUM RUNTIME VNEXT G00 BASELINE FAST SELFTEST PASS"
SELFTEST_FULL_PASS_LINE = "FERRUM RUNTIME VNEXT G00 BASELINE FULL SELFTEST PASS"
PATH_SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT G00 EXTERNAL PATH SELFTEST PASS"
SELFTEST_SUMMARY_PREFIX = "FERRUM RUNTIME VNEXT G00 BASELINE SELFTEST SUMMARY:"
SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SLOT_ORDER = ["A", "B", "B", "A", "B", "A", "A", "B"]
COLLECTOR_SCHEMA_VERSION = 1
HTTP_SCALAR_METRICS = (
    "output_throughput_tps",
    "total_throughput_tps",
    "request_throughput_rps",
    "goodput_rps",
)
HTTP_LATENCY_METRICS = ("ttft_ms", "tpot_ms", "itl_ms", "e2e_ms")
HTTP_ALWAYS_AVAILABLE_LATENCY_METRICS = ("ttft_ms", "tpot_ms", "e2e_ms")
HTTP_PERCENTILES = ("p50", "p75", "p95", "p99")
ITL_EVIDENCE_FIELDS = {
    "source",
    "output_events",
    "usage_output_tokens",
    "observed_intervals",
    "transport_coalesced_output_chunks",
    "eligibility",
}
ITL_ELIGIBILITY_FIELDS = {
    "eligible",
    "missing_evidence",
    "request_failed",
    "missing_usage",
    "too_short",
    "event_usage_mismatch",
    "interval_count_mismatch",
    "transport_coalesced",
}
BENCH_QUALITY_FIELDS = (
    "bad_output_per_run",
    "malformed_stream_per_run",
    "missing_done_per_run",
    "duplicate_done_per_run",
    "zero_output_tokens_per_run",
    "stream_bulk_flush_per_run",
    "http_500_per_run",
    "panic_per_run",
)
SELFTEST_MUTATION_NAMES = (
    "dirty",
    "stale",
    "preset-policy-drift",
    "expectations-lock-sha",
    "preset-semantic-chain-forgery",
    "chat-template-chain-forgery",
    "generation-config-chain-forgery",
    "llama-official-chain-forgery",
    "model-resolution-revision",
    "model-resolution-resolver-sha",
    "model-resolution-shard-index",
    "model-resolution-expected-sha-non-lfs",
    "hardware-derived-fingerprint",
    "hardware-probe-command",
    "hardware-probe-raw-derivation",
    "historical-identical-mutation",
    "historical-missing-signature",
    "historical-success-returncode",
    "model-sha",
    "cuda-primary-blocked",
    "scenario-no-ferrum-argv",
    "scenario-missing-tools",
    "scenario-missing-schema",
    "scenario-missing-utf8",
    "scenario-missing-thinking",
    "scenario-missing-cancel",
    "scenario-artifact-sha",
    "scenario-fake-pass",
    "cross-hardware",
    "repeat-count",
    "expected-request-accounting",
    "expected-requests-absolute",
    "expected-requests-missing",
    "errors",
    "usage",
    "ab-identity-swap",
    "duplicate-server-session",
    "server-session-same-lane-overlap",
    "cross-lane-session-id-conflict",
    "server-cell-window-overlap",
    "report-outside-cell-window",
    "server-process-start-marker",
    "ready-probe-returncode",
    "loaded-model-probe",
    "server-effective-config-model",
    "server-product-config-cap",
    "server-effective-config-argv",
    "benchmark-client-tree-binding",
    "benchmark-client-rust-allowlist",
    "bench-canonical-argv",
    "dataset-sha",
    "tokenizer-sha",
    "config-sha",
    "hardware-fingerprint",
    "active-cap",
    "observed-active",
    "zero-observed-active",
    "resource-observation-pid",
    "resource-observation-process-start",
    "resource-summary-forgery",
    "resource-http-process-probe",
    "raw-report-sha",
    "raw-report-metric",
    "raw-report-usage",
    "raw-report-quality",
    "itl-evidence-missing",
    "itl-evidence-cardinality",
    "itl-source-forged",
    "itl-usage-event-claimed-eligible",
    "itl-interval-claimed-eligible",
    "itl-coalesced-claimed-eligible",
    "itl-repeat-counts-forged",
    "itl-repeat-intervals-forged",
    "itl-ineligible-partial-percentiles",
    "swap-growth",
    "duplicate-repeat-ordinal",
    "warmup-error",
    "bench-thinking-payload",
    "bench-env-hash",
    "run-real-command",
    "run-session-global-overlap",
    "run-command-window-binding",
    "inventory-source-coverage",
    "inventory-review-binding",
    "artifact-index-empty-file",
    "artifact-index-symlink",
    "build-real-command",
    "build-raw-summary",
    "build-finished-failure",
    "build-content-evidence",
    "build-native-log-derivation",
    "build-restore-fresh",
    "build-restore-binary",
    "build-restore-mtime",
    "malformed-artifact-type",
)
SELFTEST_MODEL_LOCK_MUTATIONS = frozenset(
    {
        "stale",
        "preset-policy-drift",
        "expectations-lock-sha",
        "preset-semantic-chain-forgery",
        "chat-template-chain-forgery",
        "generation-config-chain-forgery",
        "llama-official-chain-forgery",
        "model-resolution-revision",
        "model-resolution-resolver-sha",
        "model-resolution-shard-index",
        "model-resolution-expected-sha-non-lfs",
        "hardware-derived-fingerprint",
        "hardware-probe-command",
        "hardware-probe-raw-derivation",
    }
)
SELFTEST_HISTORICAL_MUTATIONS = frozenset(
    {
        "historical-identical-mutation",
        "historical-missing-signature",
        "historical-success-returncode",
    }
)
SELFTEST_SCENARIO_MUTATIONS = frozenset(
    {
        "scenario-no-ferrum-argv",
        "scenario-missing-tools",
        "scenario-missing-schema",
        "scenario-missing-utf8",
        "scenario-missing-thinking",
        "scenario-missing-cancel",
        "scenario-artifact-sha",
        "scenario-fake-pass",
    }
)
SELFTEST_PERFORMANCE_MUTATIONS = frozenset(
    {
        "cross-hardware",
        "repeat-count",
        "expected-request-accounting",
        "expected-requests-absolute",
        "expected-requests-missing",
        "errors",
        "usage",
        "ab-identity-swap",
        "duplicate-server-session",
        "server-session-same-lane-overlap",
        "server-cell-window-overlap",
        "report-outside-cell-window",
        "server-process-start-marker",
        "ready-probe-returncode",
        "loaded-model-probe",
        "server-effective-config-model",
        "server-product-config-cap",
        "server-effective-config-argv",
        "benchmark-client-tree-binding",
        "benchmark-client-rust-allowlist",
        "bench-canonical-argv",
        "dataset-sha",
        "tokenizer-sha",
        "config-sha",
        "hardware-fingerprint",
        "active-cap",
        "observed-active",
        "zero-observed-active",
        "resource-observation-pid",
        "resource-observation-process-start",
        "resource-summary-forgery",
        "resource-http-process-probe",
        "raw-report-sha",
        "raw-report-metric",
        "raw-report-usage",
        "raw-report-quality",
        "itl-evidence-missing",
        "itl-evidence-cardinality",
        "itl-source-forged",
        "itl-usage-event-claimed-eligible",
        "itl-interval-claimed-eligible",
        "itl-coalesced-claimed-eligible",
        "itl-repeat-counts-forged",
        "itl-repeat-intervals-forged",
        "itl-ineligible-partial-percentiles",
        "swap-growth",
        "duplicate-repeat-ordinal",
        "warmup-error",
        "bench-thinking-payload",
        "bench-env-hash",
        "run-real-command",
        "run-session-global-overlap",
        "run-command-window-binding",
        "malformed-artifact-type",
    }
)
SELFTEST_ROOT_INTEGRATION_MUTATIONS = frozenset(
    {
        "cross-lane-session-id-conflict",
        "inventory-source-coverage",
        "artifact-index-empty-file",
        "artifact-index-symlink",
    }
)
SELFTEST_INVENTORY_MUTATIONS = frozenset({"inventory-review-binding"})
SELFTEST_BUILD_MUTATIONS = frozenset(
    {
        "build-real-command",
        "build-raw-summary",
        "build-finished-failure",
        "build-content-evidence",
        "build-native-log-derivation",
        "build-restore-fresh",
        "build-restore-binary",
        "build-restore-mtime",
    }
)
BUILD_SCENARIOS = {
    "noop",
    "rust-model-leaf",
    "rust-runtime-leaf",
    "core-ptx",
    "native-tu",
    "clean-release",
}
CUDA_BUILD_ARGV = [
    "cargo",
    "build",
    "--release",
    "-p",
    "ferrum-cli",
    "--bin",
    "ferrum",
    "--features",
    "cuda,vllm-moe-marlin,vllm-paged-attn-v2",
    "--message-format=json-render-diagnostics",
    "--timings",
    "-vv",
]
BUILD_SCENARIO_INPUTS = {
    "noop": ("none", None, None),
    "rust-model-leaf": ("content-mutation", "crates/ferrum-models/src/lib.rs", "ferrum-models"),
    "rust-runtime-leaf": ("content-mutation", "crates/ferrum-engine/src/lib.rs", "ferrum-engine"),
    "core-ptx": ("content-mutation", "crates/ferrum-kernels/triton_ptx/add_bias_f16.ptx", "ferrum-kernels"),
    "native-tu": ("content-mutation", "crates/ferrum-kernels/vllm_marlin/gptq_marlin_repack.cu", "ferrum-kernels"),
    "clean-release": ("cargo-clean", None, None),
}
PRIMARY_MODELS = {
    "m1-qwen35-4b": "Qwen/Qwen3.5-4B",
    "m2-qwen35-35b-a3b": "Qwen/Qwen3.5-35B-A3B",
    "m3-qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
}
SUPPLEMENTAL_MODELS = {
    "qwen3-coder-30b-a3b": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "deepseek-r1-qwen3-8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "llama31-8b-compat": "meta-llama/Llama-3.1-8B-Instruct",
}
CATALOG_MODEL_KEYS = {
    "M1": "m1-qwen35-4b",
    "M2": "m2-qwen35-35b-a3b",
    "M3": "m3-qwen3-30b-a3b",
    "Qwen3-Coder-30B-A3B-Instruct": "qwen3-coder-30b-a3b",
    "DeepSeek-R1-0528-Qwen3-8B": "deepseek-r1-qwen3-8b",
    "Llama-3.1-8B-Instruct": "llama31-8b-compat",
}
LLAMA31_OFFICIAL_MATCHES = {
    "config.json": {
        "git_oid": "0bb6fd75b3ad2fe988565929f329945262c2814e",
        "content_sha256": "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
        "size_bytes": 855,
    },
    "generation_config.json": {
        "git_oid": "cc7276afd599de091142c6ed3005faf8a74aa257",
        "content_sha256": "189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd",
        "size_bytes": 184,
    },
    "tokenizer.json": {
        "git_oid": "5cc5f00a5b203e90a27a3bd60d1ec393b07971e8",
        "content_sha256": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
        "size_bytes": 9085657,
    },
}
REQUIRED_PRESETS = {
    "P_DETERMINISTIC",
    "P_NO_THINKING",
    "P_THINKING",
    "P_OFFICIAL_DEFAULT",
}
PRESET_FIELDS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
    "seed",
    "max_tokens",
    "stop",
    "eos_token_ids",
    "enable_thinking",
    "template_kwargs",
    "source",
}
FORBIDDEN_OUTPUT_KEYS = {"waiver", "waivers", "skipped", "placeholder"}
BENCHMARK_CLIENT_RUST_ALLOWLIST = (
    "crates/ferrum-bench-core/src/lib.rs",
    "crates/ferrum-bench-core/src/report.rs",
    "crates/ferrum-cli/src/commands/bench.rs",
    "crates/ferrum-cli/src/commands/bench_serve.rs",
)


class BaselineError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BaselineError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BaselineError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BaselineError(f"{path}: expected JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineError(message)


def require_object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a JSON array")
    return value


def require_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and value.strip(), f"{label} must be a non-empty string")
    return value.strip()


def require_sha256(value: Any, label: str) -> str:
    digest = require_string(value, label).lower()
    require(SHA256_RE.fullmatch(digest) is not None, f"{label} must be a lowercase SHA256")
    return digest


def require_git_sha(value: Any, label: str, *, frozen: bool = True) -> str:
    sha = require_string(value, label).lower()
    require(GIT_SHA_RE.fullmatch(sha) is not None, f"{label} must be a 40-character git SHA")
    if frozen:
        require(sha == FROZEN_LEGACY_SHA, f"{label} is stale: {sha} != {FROZEN_LEGACY_SHA}")
    return sha


def require_schema(data: dict[str, Any], label: str) -> None:
    require(data.get("schema_version") == SCHEMA_VERSION, f"{label}.schema_version must be {SCHEMA_VERSION}")


def require_clean(data: dict[str, Any], label: str) -> None:
    dirty = data.get("dirty_status")
    if isinstance(dirty, bool):
        require(not dirty, f"{label} dirty baseline is forbidden")
        return
    dirty_obj = require_object(dirty, f"{label}.dirty_status")
    require(dirty_obj.get("is_dirty") is False, f"{label} dirty baseline is forbidden")
    status = dirty_obj.get("status_short", [])
    require(isinstance(status, list) and not status, f"{label}.dirty_status.status_short must be empty")


def require_no_forbidden_markers(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_OUTPUT_KEYS:
                if child not in (False, 0, None, [], {}):
                    raise BaselineError(f"{label}.{key} contains forbidden waiver/skip/placeholder evidence")
            require_no_forbidden_markers(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            require_no_forbidden_markers(child, f"{label}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        require("selftest pass" not in lowered and "self-test pass" not in lowered, f"{label} uses selftest evidence")


def reject_synthetic_value(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            reject_synthetic_value(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_synthetic_value(child, f"{label}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        forbidden = ("selftest", "self-test", "synthetic", "example.invalid")
        require(not any(marker in lowered for marker in forbidden), f"{label} contains synthetic/self-test evidence")


def reject_synthetic_artifacts(root: Path) -> None:
    for path in sorted(root.rglob("*.json")):
        if path.name in {
            "manifest.json",
            "gate.manifest.json",
            "coupling-inventory.json",
            "historical-bug-corpus.json",
        }:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BaselineError(f"invalid JSON artifact while scanning synthetic markers: {path}: {exc}") from exc
        reject_synthetic_value(value, str(path.relative_to(root)))


def artifact_path(root: Path, raw: Any, label: str) -> Path:
    text = require_string(raw, label)
    path = Path(text)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise BaselineError(f"{label} must stay inside artifact root: {text}") from exc
    return resolved


def require_external_artifact_path(raw: str | Path, label: str) -> Path:
    resolved = Path(raw).expanduser().resolve(strict=False)
    repo_root = REPO_ROOT.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        return resolved
    raise BaselineError(f"{label} must resolve outside REPO_ROOT ({repo_root})")


def validate_child_output_paths(raw_argv: Any, label: str) -> tuple[Path, Path]:
    _, options, _ = parse_argv(raw_argv, label)
    artifact_root = require_external_artifact_path(
        require_string(options.get("--artifact-root"), f"{label}.--artifact-root"),
        f"{label}.--artifact-root",
    )
    out = require_external_artifact_path(
        require_string(options.get("--out"), f"{label}.--out"),
        f"{label}.--out",
    )
    try:
        out.relative_to(artifact_root)
    except ValueError as exc:
        raise BaselineError(f"{label}.--out must resolve inside --artifact-root") from exc
    return artifact_root, out


def require_log(root: Path, raw: Any, label: str) -> Path:
    path = artifact_path(root, raw, label)
    require(path.is_file(), f"{label} missing log: {path}")
    require(path.stat().st_size > 0, f"{label} log is empty: {path}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_timestamp(value: Any, label: str) -> datetime:
    text = require_string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BaselineError(f"{label} must be an ISO-8601 timestamp") from exc
    require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return parsed


def validate_execution_window(raw: dict[str, Any], label: str) -> float:
    started = parse_timestamp(raw.get("started_at"), f"{label}.started_at")
    finished = parse_timestamp(raw.get("finished_at"), f"{label}.finished_at")
    elapsed = (finished - started).total_seconds()
    require(elapsed > 0, f"{label} must finish after it starts")
    duration = require_number(raw.get("duration_sec"), f"{label}.duration_sec", positive=True)
    require(math.isclose(duration, elapsed, rel_tol=0.02, abs_tol=0.05), f"{label}.duration_sec does not match timestamps")
    return duration


def require_number(value: Any, label: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value),
        f"{label} must be finite numeric",
    )
    number = float(value)
    if positive:
        require(number > 0, f"{label} must be positive")
    if nonnegative:
        require(number >= 0, f"{label} must be non-negative")
    return number


def require_nonnegative_int(value: Any, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return int(value)


def require_positive_int(value: Any, label: str) -> int:
    result = require_nonnegative_int(value, label)
    require(result > 0, f"{label} must be positive")
    return result


def require_artifact_sha(root: Path, raw_path: Any, raw_sha: Any, label: str) -> Path:
    path = artifact_path(root, raw_path, f"{label}.path")
    require(path.is_file(), f"{label}.path missing artifact: {path}")
    expected = require_sha256(raw_sha, f"{label}.sha256")
    require(file_sha256(path) == expected, f"{label} SHA256 mismatch")
    return path


def parse_argv(argv: Any, label: str) -> tuple[list[str], dict[str, str], set[str]]:
    parts = require_list(argv, label)
    require(parts and all(isinstance(part, str) and part for part in parts), f"{label} must be argv")
    options: dict[str, str] = {}
    switches: set[str] = set()
    index = 0
    while index < len(parts):
        part = parts[index]
        if not part.startswith("--"):
            index += 1
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            require(key not in options and key not in switches, f"{label} duplicate option {key}")
            require(value != "", f"{label} empty option value for {key}")
            options[key] = value
            index += 1
            continue
        if index + 1 < len(parts) and not parts[index + 1].startswith("--"):
            require(part not in options and part not in switches, f"{label} duplicate option {part}")
            options[part] = parts[index + 1]
            index += 2
            continue
        require(part not in options and part not in switches, f"{label} duplicate option {part}")
        switches.add(part)
        index += 1
    return parts, options, switches


def require_option(options: dict[str, str], key: str, expected: Any, label: str) -> None:
    require(options.get(key) == str(expected), f"{label} {key} must be {expected}")


def remove_option_with_value(argv: list[str], key: str) -> None:
    index = argv.index(key)
    del argv[index : index + 2]


def close_enough(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8)


def scalar_stats(values: list[float]) -> dict[str, float]:
    require(len(values) >= 3, "scalar stats require at least three samples")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    stddev = math.sqrt(variance)
    # All HTTP outer reports contain exactly three inner repeats (df=2).
    ci95_hw = 4.303 * stddev / math.sqrt(len(values))
    return {"mean": mean, "stddev": stddev, "ci95_hw": ci95_hw}


def percentile_linear(values: list[float], quantile: float) -> float:
    require(values and 0.0 <= quantile <= 1.0, "percentile requires values and q in [0,1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def recompute_bench_env_hash(raw: dict[str, Any], label: str) -> str:
    field_order = (
        "commit_sha",
        "hw_id",
        "driver",
        "cuda",
        "rust",
        "ferrum_features",
        "gpu_clock_lock_mhz",
        "gpu_power_limit_w",
        "gpu_persistence_mode",
        "gpu_auto_boost",
        "ferrum_env",
        "runtime_config",
        "vllm_args",
    )
    required = {"commit_sha", "hw_id", "rust", "ferrum_features", "ferrum_env", "runtime_config"}
    require(required <= set(raw) <= set(field_order), f"{label} field set does not match ferrum_bench_core::Env")
    features = require_list(raw.get("ferrum_features"), f"{label}.ferrum_features")
    require(features == sorted(set(features)), f"{label}.ferrum_features must be sorted and unique")
    ferrum_env = require_object(raw.get("ferrum_env"), f"{label}.ferrum_env")
    ordered_env: dict[str, Any] = {}
    for field in field_order:
        if field not in raw:
            continue
        value = raw[field]
        if field == "ferrum_env":
            value = {key: ferrum_env[key] for key in sorted(ferrum_env)}
        elif field == "runtime_config":
            runtime = require_object(value, f"{label}.runtime_config")
            require(set(runtime) == {"entries"}, f"{label}.runtime_config must contain only entries")
            entries = require_list(runtime.get("entries"), f"{label}.runtime_config.entries")
            normalized_entries = []
            for index, entry_raw in enumerate(entries):
                entry = require_object(entry_raw, f"{label}.runtime_config.entries[{index}]")
                entry_order = ("key", "effective_value", "source", "affects")
                require(set(entry) == set(entry_order), f"{label}.runtime_config.entries[{index}] field set mismatch")
                normalized_entries.append({key: entry[key] for key in entry_order})
            value = {"entries": normalized_entries}
        ordered_env[field] = value
    payload = json.dumps(ordered_env, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def validate_scalar_stats(raw: Any, values: list[float], label: str) -> None:
    stats = require_object(raw, label)
    expected = scalar_stats(values)
    for field, expected_value in expected.items():
        raw_value = stats.get(field, 0.0 if field != "mean" else None)
        actual = require_number(raw_value, f"{label}.{field}", nonnegative=True)
        require(close_enough(actual, expected_value), f"{label}.{field} cannot be recomputed from raw repeats")


def git_value(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(proc.returncode == 0, f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def git_bytes(args: list[str]) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(
        proc.returncode == 0,
        f"git {' '.join(args)} failed: {proc.stderr.decode('utf-8', errors='replace').strip()}",
    )
    return proc.stdout


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_file_sha256(revision: str, path: str) -> str:
    return bytes_sha256(git_bytes(["show", f"{revision}:{path}"]))


def frozen_tree_sha() -> str:
    value = git_value(["rev-parse", f"{FROZEN_LEGACY_SHA}^{{tree}}"])
    require(GIT_SHA_RE.fullmatch(value) is not None, "frozen source tree SHA is invalid")
    return value


def require_source_identity(data: dict[str, Any], label: str) -> None:
    require_git_sha(data.get("source_git_sha"), f"{label}.source_git_sha")
    require(data.get("source_tree_sha") == frozen_tree_sha(), f"{label}.source_tree_sha mismatch")
    require_clean(data, label)


def contract_files() -> list[dict[str, Any]]:
    paths = list(CONTRACT_PATHS)
    if INVENTORY_REVIEW_PATH.is_file():
        paths.append(INVENTORY_REVIEW_PATH)
    rows: list[dict[str, Any]] = []
    for path in paths:
        require(path.is_file(), f"missing Runtime vNext contract file: {path}")
        rows.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def combined_contract_sha(rows: list[dict[str, Any]]) -> str:
    payload = "".join(f"{row['path']}\0{row['sha256']}\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_catalog_sha_map(
    raw: Any,
    expected_paths: set[str],
    label: str,
) -> dict[str, str]:
    mapping = require_object(raw, label)
    require(set(mapping) == expected_paths, f"{label} must cover exactly {sorted(expected_paths)}")
    return {
        require_string(path, f"{label}.path"): require_sha256(digest, f"{label}.{path}")
        for path, digest in mapping.items()
    }


def validate_catalog_reference(row: dict[str, Any], catalog_id: str) -> None:
    label = f"models catalog[{catalog_id}].reference"
    reference = require_object(row.get("reference"), label)
    semantic_repo = require_string(reference.get("semantic_repo"), f"{label}.semantic_repo")
    semantic_revision = require_object(reference.get("semantic_revision"), f"{label}.semantic_revision")
    semantic_status = semantic_revision.get("status")
    require(semantic_status in {"pinned", "same_as_weight_revision"}, f"{label}.semantic_revision status must be pinned or same_as_weight_revision")
    if semantic_status == "pinned":
        require_git_sha(semantic_revision.get("value"), f"{label}.semantic_revision.value", frozen=False)
    else:
        require(semantic_repo == row.get("repo"), f"{label}.same_as_weight_revision must use the weight repo")
        require(row["revision"].get("status") == "pinned", f"{label}.same_as_weight_revision requires a pinned weight revision")
    required_semantic = [
        require_string(path, f"{label}.required_semantic_files")
        for path in require_list(reference.get("required_semantic_files"), f"{label}.required_semantic_files")
    ]
    require(len(required_semantic) == len(set(required_semantic)), f"{label}.required_semantic_files contains duplicates")
    require({"README.md", "config.json", "tokenizer.json"} <= set(required_semantic), f"{label} must bind README.md, config.json, and tokenizer.json")
    semantic_hashes = validate_catalog_sha_map(
        reference.get("semantic_file_sha256"),
        set(required_semantic),
        f"{label}.semantic_file_sha256",
    )

    generation = require_object(reference.get("generation_config_source"), f"{label}.generation_config_source")
    require(generation.get("source") == "semantic_source", f"{label}.generation_config_source.source must be semantic_source")
    generation_path = require_string(generation.get("path"), f"{label}.generation_config_source.path")
    require(generation_path == "generation_config.json", f"{label}.generation_config_source.path must be generation_config.json")
    generation_policy = generation.get("policy")
    require(generation_policy in {"required", "absent"}, f"{label}.generation_config_source.policy is invalid")
    if generation_policy == "required":
        require(generation_path in semantic_hashes, f"{label} required generation config is not semantically locked")
        require(
            require_sha256(generation.get("content_sha256"), f"{label}.generation_config_source.content_sha256")
            == semantic_hashes[generation_path],
            f"{label}.generation_config_source SHA256 differs from semantic lock",
        )
    else:
        require(generation_path not in semantic_hashes, f"{label} absent generation config cannot be semantically locked")
        require("content_sha256" not in generation, f"{label} absent generation config cannot declare content_sha256")

    tokenizer_repo = reference.get("tokenizer_repo")
    tokenizer_hashes: dict[str, str] | None = None
    tokenizer_required: list[str] = []
    if tokenizer_repo is not None:
        require_string(tokenizer_repo, f"{label}.tokenizer_repo")
        tokenizer_revision = require_object(reference.get("tokenizer_revision"), f"{label}.tokenizer_revision")
        require(tokenizer_revision.get("status") == "pinned", f"{label}.tokenizer_revision must be pinned")
        require_git_sha(tokenizer_revision.get("value"), f"{label}.tokenizer_revision.value", frozen=False)
        tokenizer_required = [
            require_string(path, f"{label}.required_tokenizer_files")
            for path in require_list(reference.get("required_tokenizer_files"), f"{label}.required_tokenizer_files")
        ]
        tokenizer_hashes = validate_catalog_sha_map(
            reference.get("tokenizer_file_sha256"),
            set(tokenizer_required),
            f"{label}.tokenizer_file_sha256",
        )

    chat = require_object(reference.get("chat_template_source"), f"{label}.chat_template_source")
    chat_source = chat.get("source")
    require(chat_source in {"semantic_source", "tokenizer_source"}, f"{label}.chat_template_source.source is invalid")
    require(chat.get("json_pointer") == "/chat_template", f"{label}.chat_template_source.json_pointer must be /chat_template")
    chat_path = require_string(chat.get("path"), f"{label}.chat_template_source.path")
    selected_hashes = semantic_hashes if chat_source == "semantic_source" else tokenizer_hashes
    selected_paths = required_semantic if chat_source == "semantic_source" else tokenizer_required
    require(selected_hashes is not None, f"{label}.chat_template_source selects a missing tokenizer source")
    require(chat_path in selected_paths, f"{label}.chat_template_source container is not a required source file")
    require(
        require_sha256(chat.get("container_sha256"), f"{label}.chat_template_source.container_sha256")
        == selected_hashes[chat_path],
        f"{label}.chat_template_source container SHA256 differs from its source lock",
    )
    require_sha256(chat.get("content_sha256"), f"{label}.chat_template_source.content_sha256")

    upstream = reference.get("official_upstream")
    is_llama = row.get("model_id") == "Llama-3.1-8B-Instruct"
    require((upstream is not None) == is_llama, f"{label}.official_upstream must exist exactly for Llama")
    if upstream is not None:
        require(
            semantic_repo == "NousResearch/Meta-Llama-3.1-8B-Instruct"
            and semantic_revision
            == {"status": "pinned", "value": "d10aef7999a2b5ba950ab3974312feeedbfe0b77"},
            f"{label} Llama public semantic mirror source mismatch",
        )
        require(
            reference.get("tokenizer_repo") == "unsloth/Meta-Llama-3.1-8B-Instruct"
            and reference.get("tokenizer_revision")
            == {"status": "pinned", "value": "a2856192dd7c25b842431f39c179a6c2c2f627d1"},
            f"{label} Llama tokenizer source mismatch",
        )
        require(
            chat.get("source") == "tokenizer_source"
            and chat.get("container_sha256")
            == "4d18eae3017839b17cb3598f589b62ec212652d183582ee0fc33a4f1e8b3da67"
            and chat.get("content_sha256")
            == "e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65",
            f"{label} Llama chat template binding mismatch",
        )
        official = require_object(upstream, f"{label}.official_upstream")
        require(official.get("repo") == "meta-llama/Llama-3.1-8B-Instruct", f"{label}.official_upstream repo mismatch")
        revision = require_object(official.get("revision"), f"{label}.official_upstream.revision")
        require(
            revision == {"status": "pinned", "value": "0e9e39f249a16976918f6564b8830bc894c89659"},
            f"{label}.official_upstream revision mismatch",
        )
        require(official.get("required_gated") is True, f"{label}.official_upstream must require gated provenance")
        match_paths = [
            require_string(path, f"{label}.official_upstream.blob_oid_match_files")
            for path in require_list(official.get("blob_oid_match_files"), f"{label}.official_upstream.blob_oid_match_files")
        ]
        require(match_paths == ["config.json", "generation_config.json", "tokenizer.json"], f"{label}.official_upstream match file matrix mismatch")
        oid_map = require_object(official.get("expected_git_oids"), f"{label}.official_upstream.expected_git_oids")
        require(set(oid_map) == set(match_paths), f"{label}.official_upstream Git OID map mismatch")
        for path, oid in oid_map.items():
            require(isinstance(oid, str) and GIT_SHA_RE.fullmatch(oid) is not None, f"{label}.official_upstream Git OID invalid for {path}")
        content_map = validate_catalog_sha_map(
            official.get("expected_content_sha256"),
            set(match_paths),
            f"{label}.official_upstream.expected_content_sha256",
        )
        require(
            all(content_map[path] == semantic_hashes[path] for path in match_paths),
            f"{label}.official_upstream content hashes differ from the public mirror",
        )
        sizes = require_object(official.get("expected_size_bytes"), f"{label}.official_upstream.expected_size_bytes")
        require(set(sizes) == set(match_paths), f"{label}.official_upstream size map mismatch")
        for path, size in sizes.items():
            require_positive_int(size, f"{label}.official_upstream.expected_size_bytes.{path}")
        require(
            oid_map == {path: row["git_oid"] for path, row in LLAMA31_OFFICIAL_MATCHES.items()}
            and content_map
            == {path: row["content_sha256"] for path, row in LLAMA31_OFFICIAL_MATCHES.items()}
            and sizes == {path: row["size_bytes"] for path, row in LLAMA31_OFFICIAL_MATCHES.items()},
            f"{label}.official_upstream frozen official evidence mismatch",
        )
        require("gated" in require_string(official.get("access_note"), f"{label}.official_upstream.access_note").lower(), f"{label}.official_upstream access note must explain gated access")


def validate_models_catalog() -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    data = read_json(MODELS_CATALOG_PATH)
    require_schema(data, "runtime_vnext_models catalog")
    require(data.get("generated_from_git_sha") == FROZEN_LEGACY_SHA, "models catalog baseline SHA mismatch")
    rows = require_list(data.get("models"), "runtime_vnext_models.models")
    require(len(rows) == 12, "models catalog must contain 12 backend lanes")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    ids: set[str] = set()
    for index, raw in enumerate(rows):
        row = require_object(raw, f"runtime_vnext_models.models[{index}]")
        catalog_id = require_string(row.get("id"), f"models catalog[{index}].id")
        require(catalog_id not in ids, f"duplicate models catalog id: {catalog_id}")
        ids.add(catalog_id)
        model_id = require_string(row.get("model_id"), f"models catalog[{catalog_id}].model_id")
        model_key = CATALOG_MODEL_KEYS.get(model_id)
        require(model_key is not None, f"models catalog[{catalog_id}] unknown model_id {model_id}")
        backend = require_string(row.get("backend"), f"models catalog[{catalog_id}].backend")
        require(backend in {"cuda", "metal"}, f"models catalog[{catalog_id}] invalid backend")
        require((model_key, backend) not in indexed, f"duplicate catalog lane: {model_key}/{backend}")
        revision = require_object(row.get("revision"), f"models catalog[{catalog_id}].revision")
        status = revision.get("status")
        require(status in {"pinned", "resolution_required"}, f"models catalog[{catalog_id}] invalid revision status")
        if status == "pinned":
            require_git_sha(revision.get("value"), f"models catalog[{catalog_id}].revision.value", frozen=False)
        else:
            require(revision.get("value") is None, f"models catalog[{catalog_id}] unresolved revision must be null")
        require_list(row.get("files"), f"models catalog[{catalog_id}].files")
        validate_catalog_reference(row, catalog_id)
        indexed[(model_key, backend)] = row
    expected = {(key, backend) for key in set(PRIMARY_MODELS) | set(SUPPLEMENTAL_MODELS) for backend in ("cuda", "metal")}
    require(set(indexed) == expected, "models catalog lane matrix mismatch")
    return data, indexed


def validate_presets_catalog(model_catalog: dict[str, Any]) -> dict[str, Any]:
    data = read_json(PRESETS_CATALOG_PATH)
    require_schema(data, "runtime_vnext_generation_presets catalog")
    require(
        data.get("model_catalog_id") == model_catalog.get("catalog_id"),
        "generation preset catalog model_catalog_id mismatch",
    )
    require(data.get("seed") == 9271, "generation preset catalog seed must be 9271")
    models = require_object(data.get("models"), "generation preset catalog models")
    require(set(models) == set(PRIMARY_MODELS), "generation preset catalog must contain exactly M1-M3")
    catalog_lanes: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in require_list(model_catalog.get("models"), "runtime_vnext_models.models"):
        lane = require_object(raw, "runtime_vnext_models.models lane")
        key = CATALOG_MODEL_KEYS.get(str(lane.get("model_id")))
        if key in PRIMARY_MODELS:
            catalog_lanes[(str(key), str(lane.get("backend")))] = lane
    for model_key, raw in models.items():
        model = require_object(raw, f"generation preset catalog {model_key}")
        require_git_sha(
            model.get("metadata_revision"),
            f"generation preset catalog {model_key}.metadata_revision",
            frozen=False,
        )
        require_string(model.get("metadata_repo"), f"generation preset catalog {model_key}.metadata_repo")
        evidence = require_object(model.get("evidence"), f"generation preset catalog {model_key}.evidence")
        require("README.md" in evidence, f"generation preset catalog {model_key} must bind README.md")
        for path, digest in evidence.items():
            require_string(path, f"generation preset catalog {model_key}.evidence path")
            require_sha256(digest, f"generation preset catalog {model_key}.evidence.{path}")
        presets = require_object(model.get("presets"), f"generation preset catalog {model_key}.presets")
        require(set(presets) == REQUIRED_PRESETS, f"generation preset catalog {model_key} preset matrix mismatch")
        for backend in ("cuda", "metal"):
            catalog_lane = catalog_lanes[(model_key, backend)]
            reference = require_object(
                catalog_lane.get("reference"),
                f"generation preset catalog {model_key}/{backend}.reference",
            )
            require(
                reference.get("semantic_repo") == model.get("metadata_repo"),
                f"generation preset catalog {model_key}/{backend} metadata repo differs from semantic source",
            )
            semantic_revision = require_object(
                reference.get("semantic_revision"),
                f"generation preset catalog {model_key}/{backend}.semantic_revision",
            )
            resolved_revision = (
                catalog_lane["revision"].get("value")
                if semantic_revision.get("status") == "same_as_weight_revision"
                else semantic_revision.get("value")
            )
            require(
                resolved_revision == model.get("metadata_revision"),
                f"generation preset catalog {model_key}/{backend} metadata revision differs from semantic source",
            )
            semantic_hashes = require_object(
                reference.get("semantic_file_sha256"),
                f"generation preset catalog {model_key}/{backend}.semantic_file_sha256",
            )
            require(
                all(semantic_hashes.get(path) == digest for path, digest in evidence.items()),
                f"generation preset catalog {model_key}/{backend} evidence is not bound to semantic file SHA256",
            )
    return data


def required_catalog_files(catalog_lane: dict[str, Any], locked: list[dict[str, Any]], label: str) -> None:
    paths = [str(item["path"]) for item in locked]
    for index, raw in enumerate(require_list(catalog_lane.get("files"), f"{label}.catalog.files")):
        spec = require_object(raw, f"{label}.catalog.files[{index}]")
        required = spec.get("required") is True
        required_if_sharded = spec.get("required_if_sharded") is True
        if not required and not required_if_sharded:
            continue
        if required_if_sharded:
            is_sharded = any(re.search(r"-\d{5}-of-\d{5}(?:\.|$)", path) for path in paths)
            if not is_sharded:
                continue
        if "path" in spec:
            expected = require_string(spec.get("path"), f"{label}.catalog.files[{index}].path")
            require(expected in paths, f"{label} missing required catalog file {expected}")
            expected_size = spec.get("expected_size_bytes")
            if expected_size is not None:
                actual = next(item["size_bytes"] for item in locked if item["path"] == expected)
                require(actual == expected_size, f"{label} file size mismatch for {expected}")
            expected_sha256 = spec.get("expected_sha256")
            if expected_sha256 is not None:
                require(
                    spec.get("required") is True,
                    f"{label}.catalog.files[{index}].expected_sha256 requires required=true",
                )
                expected_digest = require_sha256(
                    expected_sha256,
                    f"{label}.catalog.files[{index}].expected_sha256",
                )
                actual_digest = next(item["sha256"] for item in locked if item["path"] == expected)
                require(actual_digest == expected_digest, f"{label} file SHA256 mismatch for {expected}")
        elif "glob" in spec:
            require(
                "expected_sha256" not in spec,
                f"{label}.catalog.files[{index}].expected_sha256 requires an exact path selector",
            )
            pattern = require_string(spec.get("glob"), f"{label}.catalog.files[{index}].glob")
            require(any(fnmatch.fnmatch(path, pattern) for path in paths), f"{label} missing required catalog glob {pattern}")
        else:
            raise BaselineError(f"{label}.catalog.files[{index}] needs path or glob")


def require_catalog_expected_weight_identities(
    catalog_lane: dict[str, Any],
    weight_source: dict[str, Any],
    label: str,
) -> None:
    files = {
        str(item["path"]): item
        for item in validate_file_locks(
            weight_source.get("files"),
            f"{label}.weight_source.files",
        )
    }
    for index, raw in enumerate(
        require_list(catalog_lane.get("files"), f"{label}.catalog.files")
    ):
        spec = require_object(raw, f"{label}.catalog.files[{index}]")
        if "expected_sha256" not in spec:
            continue
        require(
            "path" in spec and "glob" not in spec,
            f"{label}.catalog.files[{index}].expected_sha256 requires an exact path selector",
        )
        require(
            spec.get("required") is True,
            f"{label}.catalog.files[{index}].expected_sha256 requires required=true",
        )
        path = require_string(spec.get("path"), f"{label}.catalog.files[{index}].path")
        expected_sha256 = require_sha256(
            spec.get("expected_sha256"),
            f"{label}.catalog.files[{index}].expected_sha256",
        )
        file_row = require_object(files.get(path), f"{label}.weight_source expected file {path}")
        require(
            file_row.get("sha256") == expected_sha256,
            f"{label}.weight_source expected SHA256 mismatch for {path}",
        )
        require(
            file_row.get("sha256_source") == "hugging_face_lfs_oid",
            f"{label}.weight_source expected SHA256 for {path} requires Hugging Face LFS identity",
        )
        require(
            file_row.get("lfs_oid") == expected_sha256,
            f"{label}.weight_source expected SHA256 for {path} differs from its Hugging Face LFS OID",
        )


def normalized_file_locks(raw: Any, label: str) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "path": str(item["path"]),
                "sha256": str(item["sha256"]),
                "size_bytes": int(item["size_bytes"]),
            }
            for item in validate_file_locks(raw, label)
        ),
        key=lambda item: item["path"],
    )


def validate_resolution_source_hashes(
    source: dict[str, Any],
    expected_raw: Any,
    label: str,
    *,
    allow_synthetic: bool,
) -> dict[str, dict[str, Any]]:
    rows = validate_file_locks(source.get("files"), f"{label}.files")
    indexed = {str(row["path"]): row for row in rows}
    expected = require_object(expected_raw, f"{label}.expected_sha256")
    require(set(indexed) == set(expected), f"{label}.files do not exactly match the catalog source lock")
    for path, digest_raw in expected.items():
        digest = require_sha256(digest_raw, f"{label}.expected_sha256.{path}")
        if allow_synthetic and path == "tokenizer.json":
            continue
        require(indexed[path].get("sha256") == digest, f"{label}.files content SHA256 mismatch for {path}")
    return indexed


def validate_resolution_lane_bindings(
    lane: dict[str, Any],
    catalog_lane: dict[str, Any],
    request_lookup: dict[tuple[str, str], dict[str, Any]],
    label: str,
    *,
    allow_synthetic: bool,
) -> None:
    reference = require_object(catalog_lane.get("reference"), f"{label}.catalog.reference")
    weight = require_object(lane.get("weight_source"), f"{label}.weight_source")
    require_catalog_expected_weight_identities(catalog_lane, weight, label)
    semantic = require_object(lane.get("semantic_source"), f"{label}.semantic_source")
    require(semantic.get("repo") == reference.get("semantic_repo"), f"{label}.semantic_source.repo mismatch")
    revision_rule = require_object(reference.get("semantic_revision"), f"{label}.catalog.semantic_revision")
    expected_semantic_revision = (
        weight.get("revision")
        if revision_rule.get("status") == "same_as_weight_revision"
        else revision_rule.get("value")
    )
    require(
        semantic.get("revision") == expected_semantic_revision,
        f"{label}.semantic_source.revision differs from the pinned catalog source",
    )
    semantic_files = validate_resolution_source_hashes(
        semantic,
        reference.get("semantic_file_sha256"),
        f"{label}.semantic_source",
        allow_synthetic=allow_synthetic,
    )

    tokenizer: dict[str, Any] | None = None
    tokenizer_files: dict[str, dict[str, Any]] | None = None
    if reference.get("tokenizer_repo") is not None:
        tokenizer = require_object(lane.get("tokenizer_source"), f"{label}.tokenizer_source")
        require(tokenizer.get("repo") == reference.get("tokenizer_repo"), f"{label}.tokenizer_source.repo mismatch")
        tokenizer_revision = require_object(reference.get("tokenizer_revision"), f"{label}.catalog.tokenizer_revision")
        require(tokenizer.get("revision") == tokenizer_revision.get("value"), f"{label}.tokenizer_source.revision mismatch")
        tokenizer_files = validate_resolution_source_hashes(
            tokenizer,
            reference.get("tokenizer_file_sha256"),
            f"{label}.tokenizer_source",
            allow_synthetic=allow_synthetic,
        )
    else:
        require("tokenizer_source" not in lane, f"{label} has an undeclared tokenizer_source")

    generation_rule = require_object(reference.get("generation_config_source"), f"{label}.catalog.generation_config_source")
    generation = require_object(lane.get("generation_config"), f"{label}.generation_config")
    require(generation.get("source") == "semantic_source", f"{label}.generation_config.source mismatch")
    require(generation.get("repo") == semantic.get("repo"), f"{label}.generation_config.repo mismatch")
    require(generation.get("revision") == semantic.get("revision"), f"{label}.generation_config.revision mismatch")
    require(generation.get("path") == generation_rule.get("path"), f"{label}.generation_config.path mismatch")
    require(generation.get("policy") == generation_rule.get("policy"), f"{label}.generation_config.policy mismatch")
    if generation_rule.get("policy") == "required":
        require(generation.get("present") is True, f"{label}.generation_config must be present")
        generation_file = require_object(generation.get("file"), f"{label}.generation_config.file")
        expected_file = semantic_files[str(generation_rule.get("path"))]
        require(
            normalized_file_locks([generation_file], f"{label}.generation_config.file")
            == normalized_file_locks([expected_file], f"{label}.semantic generation file"),
            f"{label}.generation_config.file differs from semantic source",
        )
        require(
            generation_file.get("sha256") == generation_rule.get("content_sha256"),
            f"{label}.generation_config content SHA256 mismatch",
        )
    else:
        require(generation.get("present") is False, f"{label}.generation_config must prove absence")
        require("file" not in generation, f"{label}.absent generation_config cannot contain a file lock")

    chat_rule = require_object(reference.get("chat_template_source"), f"{label}.catalog.chat_template_source")
    chat = require_object(lane.get("chat_template"), f"{label}.chat_template")
    require(chat.get("source") == chat_rule.get("source"), f"{label}.chat_template.source mismatch")
    selected_source = semantic if chat.get("source") == "semantic_source" else tokenizer
    selected_files = semantic_files if chat.get("source") == "semantic_source" else tokenizer_files
    require(selected_source is not None and selected_files is not None, f"{label}.chat_template selects a missing source")
    require(chat.get("repo") == selected_source.get("repo"), f"{label}.chat_template.repo mismatch")
    require(chat.get("revision") == selected_source.get("revision"), f"{label}.chat_template.revision mismatch")
    require(chat.get("path") == chat_rule.get("path"), f"{label}.chat_template.path mismatch")
    require(chat.get("json_pointer") == chat_rule.get("json_pointer"), f"{label}.chat_template.json_pointer mismatch")
    container = selected_files[str(chat.get("path"))]
    if not (allow_synthetic and chat.get("path") == "tokenizer.json"):
        require(chat.get("container_sha256") == container.get("sha256"), f"{label}.chat_template.container_sha256 differs from source file")
    require(chat.get("container_sha256") == chat_rule.get("container_sha256"), f"{label}.chat_template.container_sha256 differs from catalog")
    require(chat.get("content_sha256") == chat_rule.get("content_sha256"), f"{label}.chat_template.content_sha256 differs from catalog")
    require_positive_int(chat.get("content_bytes"), f"{label}.chat_template.content_bytes")

    official_rule_raw = reference.get("official_upstream")
    if official_rule_raw is None:
        require("official_upstream" not in lane, f"{label} has an undeclared official_upstream")
        return
    official_rule = require_object(official_rule_raw, f"{label}.catalog.official_upstream")
    official = require_object(lane.get("official_upstream"), f"{label}.official_upstream")
    require(official.get("repo") == official_rule.get("repo"), f"{label}.official_upstream.repo mismatch")
    require(
        official.get("revision") == require_object(official_rule.get("revision"), f"{label}.catalog.official revision").get("value"),
        f"{label}.official_upstream.revision mismatch",
    )
    require(official.get("mirror_repo") == semantic.get("repo"), f"{label}.official_upstream.mirror_repo mismatch")
    require(official.get("mirror_revision") == semantic.get("revision"), f"{label}.official_upstream.mirror_revision mismatch")
    require(official.get("gated") not in {None, False}, f"{label}.official_upstream must be gated")
    require(
        official.get("verification_method") == "mirror_content_sha256_and_official_git_blob_oid",
        f"{label}.official_upstream verification method mismatch",
    )
    require(official.get("access_note") == official_rule.get("access_note"), f"{label}.official_upstream.access_note mismatch")
    model_url = require_string(official.get("model_request_url"), f"{label}.official_upstream.model_request_url")
    require(
        model_url
        == f"https://huggingface.co/api/models/{official['repo']}/revision/{official['revision']}",
        f"{label}.official_upstream model URL is not bound to its repo/revision",
    )
    require(("model-info", model_url) in request_lookup, f"{label}.official_upstream model request is absent from provenance")
    tree_urls = require_list(official.get("tree_request_urls"), f"{label}.official_upstream.tree_request_urls")
    require(tree_urls, f"{label}.official_upstream tree request matrix is empty")
    for tree_url_raw in tree_urls:
        tree_url = require_string(tree_url_raw, f"{label}.official_upstream.tree_request_url")
        require(("repo-tree", tree_url) in request_lookup, f"{label}.official_upstream tree request is absent from provenance")
        require(
            f"https://huggingface.co/api/models/{official['repo']}/tree/{official['revision']}?" in tree_url,
            f"{label}.official_upstream tree URL is not bound to its repo/revision",
        )
    matches = [
        require_object(item, f"{label}.official_upstream.mirror_blob_oid_matches")
        for item in require_list(official.get("mirror_blob_oid_matches"), f"{label}.official_upstream.mirror_blob_oid_matches")
    ]
    expected_paths = list(require_list(official_rule.get("blob_oid_match_files"), f"{label}.catalog.official match files"))
    require([item.get("path") for item in matches] == expected_paths, f"{label}.official_upstream match file matrix mismatch")
    expected_oids = require_object(official_rule.get("expected_git_oids"), f"{label}.catalog.official expected Git OIDs")
    expected_hashes = require_object(official_rule.get("expected_content_sha256"), f"{label}.catalog.official expected hashes")
    expected_sizes = require_object(official_rule.get("expected_size_bytes"), f"{label}.catalog.official expected sizes")
    for item in matches:
        path = str(item["path"])
        require(item.get("git_oid") == expected_oids[path], f"{label}.official_upstream Git OID mismatch for {path}")
        require(item.get("content_sha256") == expected_hashes[path], f"{label}.official_upstream content SHA256 mismatch for {path}")
        require(item.get("size_bytes") == expected_sizes[path], f"{label}.official_upstream size mismatch for {path}")
        require(semantic_files[path].get("sha256") == item.get("content_sha256"), f"{label}.official_upstream mirror content mismatch for {path}")


def validate_model_resolution(
    root: Path,
    lock: dict[str, Any],
    catalog: dict[str, Any],
    catalog_lanes: dict[tuple[str, str], dict[str, Any]],
    *,
    allow_synthetic: bool = False,
) -> dict[str, dict[str, Any]]:
    reference = require_object(lock.get("model_resolution"), "models.lock.model_resolution")
    require(reference.get("path") == "model-resolution.json", "models.lock.model_resolution.path must be model-resolution.json")
    path = require_artifact_sha(
        root,
        reference.get("path"),
        reference.get("sha256"),
        "models.lock.model_resolution",
    )
    resolution = read_json(path)
    require_schema(resolution, "model-resolution")
    require(resolution.get("artifact_type") == "runtime_vnext_model_resolution", "model-resolution artifact_type mismatch")
    require(resolution.get("catalog_id") == catalog.get("catalog_id"), "model-resolution catalog_id mismatch")
    require(resolution.get("catalog_sha256") == file_sha256(MODELS_CATALOG_PATH), "model-resolution catalog_sha256 mismatch")
    resolver = require_object(resolution.get("resolver"), "model-resolution.resolver")
    require(resolver.get("path") == MODEL_RESOLVER_PATH.relative_to(REPO_ROOT).as_posix(), "model-resolution resolver path mismatch")
    require(resolver.get("sha256") == file_sha256(MODEL_RESOLVER_PATH), "model-resolution resolver SHA256 mismatch")
    source = require_object(resolution.get("source"), "model-resolution.source")
    require_git_sha(source.get("git_sha"), "model-resolution.source.git_sha", frozen=False)
    require(source.get("dirty") is False, "model-resolution source must be clean")
    require(source.get("status_short") == [], "model-resolution source status must be empty")
    policy = require_object(resolution.get("policy"), "model-resolution.policy")
    require(policy.get("revision") == "full_hugging_face_commit", "model-resolution revision policy mismatch")
    require(policy.get("large_weight_downloaded") is False, "model-resolution must not claim downloaded weights")
    require(policy.get("lfs_sha256_source") == "Hugging Face tree lfs.oid", "model-resolution LFS policy mismatch")
    expected_transport = "internal_selftest_fixture" if allow_synthetic else "network_huggingface_https"
    require(policy.get("transport") == expected_transport, f"model-resolution transport must be {expected_transport}")
    require(policy.get("non_lfs_max_download_bytes") == 32 * 1024 * 1024, "model-resolution metadata download limit mismatch")
    requests = require_list(resolution.get("requests"), "model-resolution.requests")
    require(requests, "model-resolution must contain live request provenance")
    request_kinds: set[str] = set()
    request_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for request_index, request_raw in enumerate(requests):
        request = require_object(request_raw, f"model-resolution.requests[{request_index}]")
        require(request.get("method") == "GET", f"model-resolution.requests[{request_index}] method must be GET")
        kind = require_string(request.get("kind"), f"model-resolution.requests[{request_index}].kind")
        require(kind in {"model-info", "repo-tree", "metadata-file"}, f"model-resolution.requests[{request_index}] kind is invalid")
        url = require_string(request.get("url"), f"model-resolution.requests[{request_index}].url")
        require(url.startswith("https://huggingface.co/"), f"model-resolution.requests[{request_index}] is not Hugging Face HTTPS")
        status = request.get("status")
        require(isinstance(status, int) and not isinstance(status, bool) and 200 <= status < 300, f"model-resolution.requests[{request_index}] status is not successful")
        require_positive_int(request.get("response_bytes"), f"model-resolution.requests[{request_index}].response_bytes")
        require_sha256(request.get("response_sha256"), f"model-resolution.requests[{request_index}].response_sha256")
        request_kinds.add(kind)
        key = (kind, url)
        require(key not in request_lookup, f"model-resolution contains duplicate request provenance for {kind} {url}")
        request_lookup[key] = request
    require(request_kinds == {"model-info", "repo-tree", "metadata-file"}, "model-resolution request kind matrix incomplete")
    lanes = require_list(resolution.get("lanes"), "model-resolution.lanes")
    require(len(lanes) == len(catalog_lanes), "model-resolution lane count mismatch")
    indexed: dict[str, dict[str, Any]] = {}
    expected_ids = {str(row["id"]) for row in catalog_lanes.values()}
    for index, raw in enumerate(lanes):
        lane = require_object(raw, f"model-resolution.lanes[{index}]")
        lane_id = require_string(lane.get("catalog_lane_id"), f"model-resolution.lanes[{index}].catalog_lane_id")
        require(lane_id not in indexed, f"duplicate model-resolution lane {lane_id}")
        require(lane_id in expected_ids, f"unknown model-resolution lane {lane_id}")
        indexed[lane_id] = lane
    require(set(indexed) == expected_ids, "model-resolution lane matrix mismatch")
    for lane_id, lane in indexed.items():
        for source_name in ("weight_source", "semantic_source", "tokenizer_source"):
            if source_name not in lane:
                continue
            source = require_object(lane[source_name], f"model-resolution.{lane_id}.{source_name}")
            repo = require_string(source.get("repo"), f"model-resolution.{lane_id}.{source_name}.repo")
            revision = require_git_sha(
                source.get("revision"),
                f"model-resolution.{lane_id}.{source_name}.revision",
                frozen=False,
            )
            requested_revision = require_object(
                source.get("requested_revision"),
                f"model-resolution.{lane_id}.{source_name}.requested_revision",
            )
            requested_status = requested_revision.get("status")
            require(
                requested_status in {"pinned", "resolution_required", "same_as_weight_revision"},
                f"model-resolution.{lane_id}.{source_name}.requested_revision.status is invalid",
            )
            if requested_status == "resolution_required":
                require(
                    requested_revision.get("value") is None,
                    f"model-resolution.{lane_id}.{source_name} unresolved request value must be null",
                )
            else:
                require(
                    requested_revision.get("value") == revision,
                    f"model-resolution.{lane_id}.{source_name} requested revision differs from resolved revision",
                )
            model_url = require_string(source.get("model_request_url"), f"model-resolution.{lane_id}.{source_name}.model_request_url")
            require(("model-info", model_url) in request_lookup, f"model-resolution.{lane_id}.{source_name} model request is absent from provenance")
            require(repo in model_url, f"model-resolution.{lane_id}.{source_name} model URL does not name its repo")
            if requested_status != "resolution_required":
                require(
                    model_url == f"https://huggingface.co/api/models/{repo}/revision/{revision}",
                    f"model-resolution.{lane_id}.{source_name} model URL is not bound to its pinned revision",
                )
            tree_urls = require_list(source.get("tree_request_urls"), f"model-resolution.{lane_id}.{source_name}.tree_request_urls")
            require(tree_urls, f"model-resolution.{lane_id}.{source_name} has no tree request")
            for tree_url_raw in tree_urls:
                tree_url = require_string(tree_url_raw, f"model-resolution.{lane_id}.{source_name}.tree_request_url")
                require(("repo-tree", tree_url) in request_lookup, f"model-resolution.{lane_id}.{source_name} tree request is absent from provenance")
                require(
                    f"https://huggingface.co/api/models/{repo}/tree/{revision}?" in tree_url,
                    f"model-resolution.{lane_id}.{source_name} tree URL is not bound to its repo/revision",
                )
            referenced_files = list(
                validate_file_locks(
                    source.get("files"),
                    f"model-resolution.{lane_id}.{source_name}.files",
                )
            )
            license_info = require_object(source.get("license"), f"model-resolution.{lane_id}.{source_name}.license")
            license_files = require_list(
                license_info.get("files"),
                f"model-resolution.{lane_id}.{source_name}.license.files",
            )
            if license_files:
                referenced_files += validate_file_locks(
                    license_files,
                    f"model-resolution.{lane_id}.{source_name}.license.files",
                )
            for file_index, file_raw in enumerate(referenced_files):
                file_row = require_object(file_raw, f"model-resolution.{lane_id}.{source_name}.files[{file_index}]")
                if file_row.get("sha256_source") != "downloaded_content":
                    if file_row.get("sha256_source") == "hugging_face_lfs_oid":
                        require(
                            file_row.get("lfs_oid") == file_row.get("sha256"),
                            f"model-resolution.{lane_id}.{source_name}.files[{file_index}] LFS OID differs from SHA256",
                        )
                    continue
                content_url = require_string(file_row.get("content_request_url"), f"model-resolution.{lane_id}.{source_name}.files[{file_index}].content_request_url")
                require(("metadata-file", content_url) in request_lookup, f"model-resolution.{lane_id}.{source_name} metadata request is absent from provenance")
                require(
                    content_url
                    == f"https://huggingface.co/{repo}/resolve/{revision}/{file_row['path']}",
                    f"model-resolution.{lane_id}.{source_name} metadata URL is not bound to its repo/revision/path",
                )
                content_request = request_lookup[("metadata-file", content_url)]
                require(
                    file_row.get("sha256") == content_request.get("response_sha256"),
                    f"model-resolution.{lane_id}.{source_name} downloaded file SHA256 differs from request provenance",
                )
                require(
                    file_row.get("size_bytes") == content_request.get("response_bytes"),
                    f"model-resolution.{lane_id}.{source_name} downloaded file size differs from request provenance",
                )
    catalog_by_id = {
        str(row["id"]): row
        for row in catalog_lanes.values()
    }
    for lane_id, lane in indexed.items():
        validate_resolution_lane_bindings(
            lane,
            catalog_by_id[lane_id],
            request_lookup,
            f"model-resolution.{lane_id}",
            allow_synthetic=allow_synthetic,
        )
    return indexed


def validate_semantic_source(lane: dict[str, Any], catalog_lane: dict[str, Any], label: str) -> None:
    reference = require_object(catalog_lane.get("reference"), f"{label}.catalog.reference")
    semantic = require_object(lane.get("semantic_source"), f"{label}.semantic_source")
    expected_repo = require_string(reference.get("semantic_repo"), f"{label}.catalog.reference.semantic_repo")
    require(semantic.get("repo") == expected_repo, f"{label}.semantic_source.repo mismatch")
    semantic_revision = require_git_sha(semantic.get("revision"), f"{label}.semantic_source.revision", frozen=False)
    revision_rule = require_object(reference.get("semantic_revision"), f"{label}.catalog.reference.semantic_revision")
    if revision_rule.get("status") == "same_as_weight_revision":
        require(semantic_revision == lane.get("revision"), f"{label}.semantic_source.revision must match weight revision")
    semantic_files = validate_file_locks(semantic.get("files"), f"{label}.semantic_source.files")
    semantic_paths = {str(item["path"]) for item in semantic_files}
    required_semantic = reference.get(
        "required_semantic_files",
        ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json"],
    )
    for path in require_list(required_semantic, f"{label}.required_semantic_files"):
        require(path in semantic_paths, f"{label}.semantic_source missing {path}")
    tokenizer_repo = reference.get("tokenizer_repo")
    if tokenizer_repo is not None:
        tokenizer = require_object(lane.get("tokenizer_source"), f"{label}.tokenizer_source")
        require(tokenizer.get("repo") == tokenizer_repo, f"{label}.tokenizer_source.repo mismatch")
        require_git_sha(tokenizer.get("revision"), f"{label}.tokenizer_source.revision", frozen=False)
        validate_file_locks(tokenizer.get("files"), f"{label}.tokenizer_source.files")


def validate_hardware(root: Path, data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    hardware = require_list(data.get("hardware"), "models.lock.hardware")
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(hardware):
        item = require_object(raw, f"models.lock.hardware[{index}]")
        hardware_id = require_string(item.get("id"), f"models.lock.hardware[{index}].id")
        require(hardware_id not in indexed, f"duplicate hardware id: {hardware_id}")
        backend = require_string(item.get("backend"), f"hardware[{hardware_id}].backend")
        require(backend in {"cuda", "metal"}, f"hardware[{hardware_id}].backend must be cuda or metal")
        fingerprint = require_sha256(item.get("fingerprint"), f"hardware[{hardware_id}].fingerprint")
        policy_id = require_string(item.get("policy_id"), f"hardware[{hardware_id}].policy_id")
        host = require_string(item.get("host"), f"hardware[{hardware_id}].host")
        device_name = require_string(item.get("device_name"), f"hardware[{hardware_id}].device_name")
        device_count = require_positive_int(item.get("device_count"), f"hardware[{hardware_id}].device_count")
        memory = item.get("memory_bytes")
        require(isinstance(memory, int) and not isinstance(memory, bool) and memory > 0, f"hardware[{hardware_id}].memory_bytes must be positive")
        if backend == "cuda":
            require(policy_id == "cuda-g0-1x-rtx4090", f"hardware[{hardware_id}] CUDA policy mismatch")
            require(device_count == 1, f"hardware[{hardware_id}] must have exactly one CUDA device")
            require("4090" in device_name, f"hardware[{hardware_id}] must be an RTX 4090")
            require(memory >= 23 * 1024**3, f"hardware[{hardware_id}] RTX 4090 memory is implausibly small")
        else:
            require(policy_id == "metal-reference-m1-max-32gb", f"hardware[{hardware_id}] Metal policy mismatch")
            require(device_count == 1, f"hardware[{hardware_id}] must have exactly one Metal device")
            require("M1 Max" in device_name, f"hardware[{hardware_id}] must be the fixed M1 Max host")
            require(memory == 32 * 1024**3, f"hardware[{hardware_id}] Metal unified memory must be exactly 32 GiB")
            require(item.get("gpu_core_count") == 24, f"hardware[{hardware_id}] Metal GPU core count must be 24")
        runtime = require_object(item.get("runtime"), f"hardware[{hardware_id}].runtime")
        require(runtime, f"hardware[{hardware_id}].runtime must not be empty")
        require(all(isinstance(key, str) and key and isinstance(value, str) and value for key, value in runtime.items()), f"hardware[{hardware_id}].runtime must contain string facts")
        system = require_object(item.get("system"), f"hardware[{hardware_id}].system")
        for field in ("os", "arch", "cpu"):
            require_string(system.get(field), f"hardware[{hardware_id}].system.{field}")
        require_positive_int(system.get("logical_cpu_count"), f"hardware[{hardware_id}].system.logical_cpu_count")
        require_positive_int(system.get("host_memory_bytes"), f"hardware[{hardware_id}].system.host_memory_bytes")
        material: dict[str, Any] = {
            "schema_version": 1,
            "backend": backend,
            "policy_id": policy_id,
            "host": host,
            "device_name": device_name,
            "device_count": device_count,
            "memory_bytes": memory,
            "runtime": runtime,
            "system": system,
        }
        if backend == "metal":
            material["gpu_core_count"] = item["gpu_core_count"]
        require(item.get("fingerprint_material") == material, f"hardware[{hardware_id}].fingerprint_material mismatch")
        require(fingerprint == canonical_json_sha256(material), f"hardware[{hardware_id}].fingerprint is not derived from normalized hardware")

        probe_ref = require_object(item.get("probe"), f"hardware[{hardware_id}].probe")
        probe_path = require_artifact_sha(
            root,
            probe_ref.get("path"),
            probe_ref.get("sha256"),
            f"hardware[{hardware_id}].probe",
        )
        probe = read_json(probe_path)
        require_schema(probe, f"hardware[{hardware_id}].probe")
        require_source_identity(probe, f"hardware[{hardware_id}].probe")
        require(probe.get("hardware_id") == hardware_id, f"hardware[{hardware_id}].probe hardware_id mismatch")
        collector = require_object(probe.get("collector"), f"hardware[{hardware_id}].probe.collector")
        require(collector.get("path") == HARDWARE_PROBE_PATH.relative_to(REPO_ROOT).as_posix(), f"hardware[{hardware_id}].probe collector path mismatch")
        require(collector.get("sha256") == file_sha256(HARDWARE_PROBE_PATH), f"hardware[{hardware_id}].probe collector SHA256 mismatch")
        require(probe.get("normalized") == material, f"hardware[{hardware_id}].probe normalized facts mismatch")
        require(probe.get("fingerprint") == fingerprint, f"hardware[{hardware_id}].probe fingerprint mismatch")
        commands = require_list(probe.get("commands"), f"hardware[{hardware_id}].probe.commands")
        command_kinds: set[str] = set()
        command_stdout: dict[str, str] = {}
        command_stderr: dict[str, str] = {}
        expected_commands = hardware_probe.PROBE_ARGV[backend]
        for command_index, command_raw in enumerate(commands):
            command = require_object(command_raw, f"hardware[{hardware_id}].probe.commands[{command_index}]")
            kind = require_string(command.get("kind"), f"hardware[{hardware_id}].probe.commands[{command_index}].kind")
            require(kind in expected_commands and kind not in command_kinds, f"hardware[{hardware_id}].probe invalid or duplicate command kind {kind}")
            command_kinds.add(kind)
            argv = require_list(command.get("argv"), f"hardware[{hardware_id}].probe.commands[{kind}].argv")
            require(argv and all(isinstance(part, str) and part for part in argv), f"hardware[{hardware_id}].probe.commands[{kind}].argv must be argv")
            require(argv == expected_commands[kind], f"hardware[{hardware_id}].probe.commands[{kind}] argv mismatch")
            require(command.get("returncode") == 0, f"hardware[{hardware_id}].probe.commands[{kind}] failed")
            validate_execution_window(command, f"hardware[{hardware_id}].probe.commands[{kind}]")
            stdout_path = require_artifact_sha(probe_path.parent, command.get("stdout"), command.get("stdout_sha256"), f"hardware[{hardware_id}].probe.commands[{kind}].stdout")
            stderr_path = require_artifact_sha(probe_path.parent, command.get("stderr"), command.get("stderr_sha256"), f"hardware[{hardware_id}].probe.commands[{kind}].stderr")
            command_stdout[kind] = stdout_path.read_text(encoding="utf-8", errors="replace")
            command_stderr[kind] = stderr_path.read_text(encoding="utf-8", errors="replace")
            combined = command_stdout[kind] + command_stderr[kind]
            require(not re.search(r"Serial Number|Hardware UUID|Authorization:\s*Bearer", combined, re.IGNORECASE), f"hardware[{hardware_id}].probe.commands[{kind}] leaks private identifiers")
        require(command_kinds == set(expected_commands), f"hardware[{hardware_id}].probe command matrix incomplete")
        try:
            recomputed = hardware_probe.normalized_from_outputs(
                backend,
                policy_id,
                command_stdout,
                command_stderr,
            )
        except hardware_probe.ProbeError as exc:
            raise BaselineError(f"hardware[{hardware_id}].probe raw output is invalid: {exc}") from exc
        require(recomputed == material, f"hardware[{hardware_id}].probe normalized facts are not derived from raw outputs")
        indexed[hardware_id] = item
    require({item["backend"] for item in indexed.values()} == {"cuda", "metal"}, "hardware must include CUDA and Metal")
    return indexed


def validate_file_locks(files: Any, label: str) -> list[dict[str, Any]]:
    rows = require_list(files, label)
    require(rows, f"{label} must not be empty")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        item = require_object(raw, f"{label}[{index}]")
        path = require_string(item.get("path"), f"{label}[{index}].path")
        require(path not in seen, f"{label} duplicate path: {path}")
        seen.add(path)
        require_sha256(item.get("sha256"), f"{label}[{index}].sha256")
        size = item.get("size_bytes")
        require(isinstance(size, int) and not isinstance(size, bool) and size > 0, f"{label}[{index}].size_bytes must be positive")
        normalized.append(item)
    return normalized


def validate_models_lock(
    root: Path,
    *,
    allow_synthetic: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    catalog, catalog_lanes = validate_models_catalog()
    preset_catalog = validate_presets_catalog(catalog)
    data = read_json(root / "models.lock.json")
    require_schema(data, "models.lock")
    require_source_identity(data, "models.lock")
    require(
        data.get("catalog_sha256") == file_sha256(MODELS_CATALOG_PATH),
        "models.lock.catalog_sha256 mismatch",
    )
    require(data.get("catalog_id") == catalog.get("catalog_id"), "models.lock.catalog_id mismatch")
    require(
        data.get("preset_catalog_id") == preset_catalog.get("catalog_id"),
        "models.lock.preset_catalog_id mismatch",
    )
    require(
        data.get("preset_catalog_sha256") == file_sha256(PRESETS_CATALOG_PATH),
        "models.lock.preset_catalog_sha256 mismatch",
    )
    expectations_binding = require_object(
        data.get("expectations_catalog"), "models.lock.expectations_catalog"
    )
    expectations_path = require_string(
        expectations_binding.get("path"), "models.lock.expectations_catalog.path"
    )
    expectations_sha = require_sha256(
        expectations_binding.get("sha256"), "models.lock.expectations_catalog.sha256"
    )
    if allow_synthetic:
        expectations_artifact = artifact_path(
            root, expectations_path, "models.lock.expectations_catalog.path"
        )
        require(
            expectations_artifact.is_file()
            and file_sha256(expectations_artifact) == expectations_sha,
            "models.lock.expectations_catalog synthetic artifact mismatch",
        )
    else:
        require(
            expectations_path == CORRECTNESS_EXPECTATIONS_PATH.relative_to(REPO_ROOT).as_posix(),
            "models.lock.expectations_catalog.path mismatch",
        )
        require(
            expectations_sha == file_sha256(CORRECTNESS_EXPECTATIONS_PATH),
            "models.lock.expectations_catalog.sha256 mismatch",
        )
    require_no_forbidden_markers(data, "models.lock")
    resolution_lanes = validate_model_resolution(
        root,
        data,
        catalog,
        catalog_lanes,
        allow_synthetic=allow_synthetic,
    )
    hardware = validate_hardware(root, data)
    rows = require_list(data.get("models"), "models.lock.models")
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        model = require_object(raw, f"models.lock.models[{index}]")
        key = require_string(model.get("key"), f"models.lock.models[{index}].key")
        require(key not in indexed, f"duplicate model key: {key}")
        model_id = require_string(model.get("official_model_id"), f"models[{key}].official_model_id")
        role = require_string(model.get("role"), f"models[{key}].role")
        require(role in {"primary", "supplemental"}, f"models[{key}].role must be primary or supplemental")
        expected = (PRIMARY_MODELS if role == "primary" else SUPPLEMENTAL_MODELS).get(key)
        require(expected is not None, f"models[{key}] is not in the frozen model matrix")
        require(model_id == expected, f"models[{key}].official_model_id {model_id!r} != {expected!r}")
        lanes = require_object(model.get("lanes"), f"models[{key}].lanes")
        require(set(lanes) == {"cuda", "metal"}, f"models[{key}].lanes must contain exactly cuda and metal")
        for backend in ("cuda", "metal"):
            lane = require_object(lanes[backend], f"models[{key}].lanes.{backend}")
            catalog_lane = catalog_lanes[(key, backend)]
            label = f"models[{key}].lanes.{backend}"
            require(lane.get("catalog_lane_id") == catalog_lane.get("id"), f"{label}.catalog_lane_id mismatch")
            resolution_lane = resolution_lanes[str(catalog_lane.get("id"))]
            require(resolution_lane.get("backend") == backend, f"{label} model-resolution backend mismatch")
            require(resolution_lane.get("model_id") == catalog_lane.get("model_id"), f"{label} model-resolution model_id mismatch")
            require(resolution_lane.get("format") == catalog_lane.get("format"), f"{label} model-resolution format mismatch")
            weight_source = require_object(resolution_lane.get("weight_source"), f"{label}.model-resolution.weight_source")
            require(lane.get("repo") == catalog_lane.get("repo"), f"{label}.repo mismatch")
            require(lane.get("repo") == weight_source.get("repo"), f"{label}.repo differs from model-resolution")
            revision = require_git_sha(lane.get("revision"), f"{label}.revision", frozen=False)
            require(revision == weight_source.get("revision"), f"{label}.revision differs from model-resolution")
            catalog_revision = require_object(catalog_lane.get("revision"), f"{label}.catalog.revision")
            if catalog_revision.get("status") == "pinned":
                require(revision == catalog_revision.get("value"), f"{label}.revision differs from pinned catalog revision")
            require(lane.get("format") == catalog_lane.get("format"), f"{label}.format mismatch")
            locked_files = validate_file_locks(lane.get("files"), f"{label}.files")
            require(
                normalized_file_locks(lane.get("files"), f"{label}.files")
                == normalized_file_locks(weight_source.get("files"), f"{label}.model-resolution.weight_source.files"),
                f"{label}.files differ from model-resolution",
            )
            required_catalog_files(catalog_lane, locked_files, label)
            require(lane.get("hardware_policy") == catalog_lane.get("hardware_policy"), f"{label}.hardware_policy mismatch")
            license_info = require_object(lane.get("license"), f"{label}.license")
            require_string(license_info.get("spdx"), f"{label}.license.spdx")
            require_string(license_info.get("source"), f"{label}.license.source")
            validate_semantic_source(lane, catalog_lane, label)
            resolved_semantic = require_object(resolution_lane.get("semantic_source"), f"{label}.model-resolution.semantic_source")
            require(lane["semantic_source"].get("repo") == resolved_semantic.get("repo"), f"{label}.semantic_source.repo differs from model-resolution")
            require(lane["semantic_source"].get("revision") == resolved_semantic.get("revision"), f"{label}.semantic_source.revision differs from model-resolution")
            require(
                normalized_file_locks(lane["semantic_source"].get("files"), f"{label}.semantic_source.files")
                == normalized_file_locks(resolved_semantic.get("files"), f"{label}.model-resolution.semantic_source.files"),
                f"{label}.semantic_source.files differ from model-resolution",
            )
            resolved_tokenizer = resolution_lane.get("tokenizer_source")
            if resolved_tokenizer is None:
                require("tokenizer_source" not in lane, f"{label}.tokenizer_source absent from model-resolution")
            else:
                resolved_tokenizer = require_object(resolved_tokenizer, f"{label}.model-resolution.tokenizer_source")
                locked_tokenizer = require_object(lane.get("tokenizer_source"), f"{label}.tokenizer_source")
                require(locked_tokenizer.get("repo") == resolved_tokenizer.get("repo"), f"{label}.tokenizer_source.repo differs from model-resolution")
                require(locked_tokenizer.get("revision") == resolved_tokenizer.get("revision"), f"{label}.tokenizer_source.revision differs from model-resolution")
                require(
                    normalized_file_locks(locked_tokenizer.get("files"), f"{label}.tokenizer_source.files")
                    == normalized_file_locks(resolved_tokenizer.get("files"), f"{label}.model-resolution.tokenizer_source.files"),
                    f"{label}.tokenizer_source.files differ from model-resolution",
                )
            for binding_name in ("generation_config", "chat_template"):
                require(
                    lane.get(binding_name) == resolution_lane.get(binding_name),
                    f"{label}.{binding_name} differs from model-resolution",
                )
            if "official_upstream" in resolution_lane:
                require(
                    lane.get("official_upstream") == resolution_lane.get("official_upstream"),
                    f"{label}.official_upstream differs from model-resolution",
                )
            else:
                require("official_upstream" not in lane, f"{label}.official_upstream absent from model-resolution")
            hardware_id = require_string(lane.get("hardware_id"), f"models[{key}].lanes.{backend}.hardware_id")
            require(hardware_id in hardware, f"models[{key}].lanes.{backend} references unknown hardware {hardware_id}")
            require(hardware[hardware_id]["backend"] == backend, f"models[{key}].lanes.{backend} hardware backend mismatch")
            require(hardware[hardware_id]["policy_id"] == lane["hardware_policy"], f"models[{key}].lanes.{backend} hardware policy mismatch")
        if role == "primary":
            presets = require_object(model.get("generation_presets"), f"models[{key}].generation_presets")
            require(set(presets) == REQUIRED_PRESETS, f"models[{key}] must lock exactly {sorted(REQUIRED_PRESETS)}")
            preset_policy = require_object(
                preset_catalog["models"].get(key),
                f"generation preset catalog {key}",
            )
            require(
                presets == preset_policy.get("presets"),
                f"models[{key}].generation_presets differ from checked-in preset policy",
            )
            for backend in ("cuda", "metal"):
                require(
                    lanes[backend]["semantic_source"]["repo"] == preset_policy.get("metadata_repo")
                    and lanes[backend]["semantic_source"]["revision"]
                    == preset_policy.get("metadata_revision"),
                    f"models[{key}] preset metadata source does not match {backend} semantic lock",
                )
            preset_evidence = require_list(
                model.get("generation_preset_evidence"),
                f"models[{key}].generation_preset_evidence",
            )
            evidence_rows = [
                require_object(item, f"models[{key}].generation_preset_evidence")
                for item in preset_evidence
            ]
            evidence_map: dict[str, str] = {}
            for row in evidence_rows:
                path = require_string(row.get("path"), f"models[{key}].generation_preset_evidence.path")
                require(path not in evidence_map, f"models[{key}].generation_preset_evidence duplicate path {path}")
                evidence_map[path] = require_sha256(
                    row.get("sha256"),
                    f"models[{key}].generation_preset_evidence.sha256",
                )
                require_positive_int(
                    row.get("size_bytes"),
                    f"models[{key}].generation_preset_evidence[{path}].size_bytes",
                )
            require(
                evidence_map == preset_policy.get("evidence"),
                f"models[{key}].generation_preset_evidence mismatch",
            )
            evidence_by_path = {str(row["path"]): row for row in evidence_rows}
            for backend in ("cuda", "metal"):
                semantic_by_path = {
                    str(row["path"]): row
                    for row in lanes[backend]["semantic_source"]["files"]
                }
                for path, digest in evidence_map.items():
                    require(path in semantic_by_path, f"models[{key}] preset evidence {path} is absent from {backend} semantic source")
                    require(
                        semantic_by_path[path].get("sha256") == digest,
                        f"models[{key}] preset evidence {path} SHA256 differs from {backend} semantic source",
                    )
                    require(
                        semantic_by_path[path].get("size_bytes") == evidence_by_path[path].get("size_bytes"),
                        f"models[{key}] preset evidence {path} size differs from {backend} semantic source",
                    )
            for preset_name, preset_raw in presets.items():
                preset = require_object(preset_raw, f"models[{key}].generation_presets.{preset_name}")
                missing = sorted(PRESET_FIELDS - set(preset))
                require(not missing, f"models[{key}].generation_presets.{preset_name} missing fields: {missing}")
                require(preset.get("seed") == 9271, f"models[{key}].generation_presets.{preset_name}.seed must be 9271")
                for field in (
                    "temperature",
                    "top_p",
                    "min_p",
                    "presence_penalty",
                    "repetition_penalty",
                ):
                    value = preset.get(field)
                    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f"models[{key}].generation_presets.{preset_name}.{field} must be finite numeric")
                require(0 <= preset["top_p"] <= 1, f"models[{key}].generation_presets.{preset_name}.top_p must be in [0,1]")
                require(0 <= preset["min_p"] <= 1, f"models[{key}].generation_presets.{preset_name}.min_p must be in [0,1]")
                require(0 <= preset["presence_penalty"] <= 2, f"models[{key}].generation_presets.{preset_name}.presence_penalty must be in [0,2]")
                require(preset["temperature"] >= 0, f"models[{key}].generation_presets.{preset_name}.temperature must be non-negative")
                require(preset["repetition_penalty"] > 0, f"models[{key}].generation_presets.{preset_name}.repetition_penalty must be positive")
                top_k = preset.get("top_k")
                require(isinstance(top_k, int) and not isinstance(top_k, bool) and top_k >= 0, f"models[{key}].generation_presets.{preset_name}.top_k must be a non-negative integer")
                max_tokens = preset.get("max_tokens")
                require(isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > 0, f"models[{key}].generation_presets.{preset_name}.max_tokens must be positive")
                require(isinstance(preset.get("stop"), list), f"models[{key}].generation_presets.{preset_name}.stop must be a list")
                eos_ids = preset.get("eos_token_ids")
                require(isinstance(eos_ids, list) and eos_ids and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in eos_ids), f"models[{key}].generation_presets.{preset_name}.eos_token_ids invalid")
                require(isinstance(preset.get("template_kwargs"), dict), f"models[{key}].generation_presets.{preset_name}.template_kwargs must be an object")
                require_string(preset.get("source"), f"models[{key}].generation_presets.{preset_name}.source")
                thinking = preset.get("enable_thinking")
                if preset_name in {"P_DETERMINISTIC", "P_NO_THINKING"}:
                    require(thinking is False, f"models[{key}].generation_presets.{preset_name}.enable_thinking must be false")
                elif preset_name == "P_THINKING":
                    require(thinking is True, f"models[{key}].generation_presets.P_THINKING.enable_thinking must be true")
                else:
                    require(thinking == "model-default", f"models[{key}].generation_presets.P_OFFICIAL_DEFAULT.enable_thinking must be model-default")
                if preset_name == "P_DETERMINISTIC":
                    require(preset["temperature"] == 0, f"models[{key}].generation_presets.P_DETERMINISTIC.temperature must be 0")
        indexed[key] = model
    require(set(indexed) == set(PRIMARY_MODELS) | set(SUPPLEMENTAL_MODELS), "models.lock must contain the exact primary and supplemental matrix")
    return data, hardware, indexed


def validate_legacy_binaries(root: Path, hardware: dict[str, dict[str, Any]]) -> dict[str, str]:
    data = read_json(root / "legacy-binaries.json")
    require_schema(data, "legacy-binaries")
    require_source_identity(data, "legacy-binaries")
    require_no_forbidden_markers(data, "legacy-binaries")
    rows = require_list(data.get("binaries"), "legacy-binaries.binaries")
    indexed: dict[str, str] = {}
    for index, raw in enumerate(rows):
        item = require_object(raw, f"legacy-binaries.binaries[{index}]")
        backend = require_string(item.get("backend"), f"legacy-binaries.binaries[{index}].backend")
        require(backend in {"cuda", "metal"} and backend not in indexed, f"duplicate or invalid legacy binary backend: {backend}")
        hardware_id = require_string(item.get("hardware_id"), f"legacy-binaries.{backend}.hardware_id")
        require(hardware_id in hardware and hardware[hardware_id]["backend"] == backend, f"legacy-binaries.{backend} hardware mismatch")
        command = require_list(item.get("build_command"), f"legacy-binaries.{backend}.build_command")
        require(command and all(isinstance(part, str) and part for part in command), f"legacy-binaries.{backend}.build_command must be argv")
        features = require_list(item.get("cargo_features"), f"legacy-binaries.{backend}.cargo_features")
        require(features and all(isinstance(feature, str) and feature for feature in features), f"legacy-binaries.{backend}.cargo_features must not be empty")
        require_log(root, item.get("build_log"), f"legacy-binaries.{backend}.build_log")
        require_log(root, item.get("sha256_log"), f"legacy-binaries.{backend}.sha256_log")
        require_string(item.get("binary_path"), f"legacy-binaries.{backend}.binary_path")
        digest = require_sha256(item.get("binary_sha256"), f"legacy-binaries.{backend}.binary_sha256")
        binary = artifact_path(root, item.get("artifact_binary"), f"legacy-binaries.{backend}.artifact_binary")
        require(binary.is_file(), f"legacy-binaries.{backend}.artifact_binary missing")
        require(file_sha256(binary) == digest, f"legacy-binaries.{backend}.artifact_binary SHA256 mismatch")
        indexed[backend] = digest
    require(set(indexed) == {"cuda", "metal"}, "legacy-binaries must contain exactly CUDA and Metal")
    return indexed


def locked_file_map(model: dict[str, Any], backend: str) -> dict[str, str]:
    return {str(row["path"]): str(row["sha256"]) for row in model["lanes"][backend]["files"]}


def tokenizer_sha_set(model: dict[str, Any], backend: str) -> set[str]:
    lane = model["lanes"][backend]
    sources = [lane.get("tokenizer_source") or lane.get("semantic_source")]
    return {
        str(row["sha256"])
        for source in sources
        if isinstance(source, dict)
        for row in source.get("files", [])
        if isinstance(row, dict) and str(row.get("path", "")).endswith("tokenizer.json")
    }


def locked_tokenizer_source(model: dict[str, Any], backend: str) -> dict[str, Any]:
    lane = model["lanes"][backend]
    return require_object(lane.get("tokenizer_source") or lane.get("semantic_source"), f"models[{model['key']}].lanes.{backend}.tokenizer_source")


def validate_lane_identity(
    lane: dict[str, Any],
    *,
    label: str,
    model: dict[str, Any],
    backend: str,
    hardware: dict[str, dict[str, Any]],
    binaries: dict[str, str],
) -> None:
    require_schema(lane, label)
    require_source_identity(lane, label)
    require(lane.get("model_key") == model["key"], f"{label}.model_key mismatch")
    require(lane.get("backend") == backend, f"{label}.backend mismatch")
    model_lane = model["lanes"][backend]
    require(lane.get("model_revision") == model_lane["revision"], f"{label}.model_revision mismatch")
    require(lane.get("model_files") == locked_file_map(model, backend), f"{label}.model_files do not match models.lock")
    hardware_id = lane.get("hardware_id")
    require(hardware_id == model_lane["hardware_id"], f"{label}.hardware_id cross-hardware mismatch")
    require(hardware_id in hardware, f"{label}.hardware_id unknown")
    require(lane.get("binary_sha256") == binaries[backend], f"{label}.binary_sha256 mismatch")
    require_no_forbidden_markers(lane, label)


def validate_blocked_lane(root: Path, lane: dict[str, Any], label: str) -> None:
    require(lane.get("current_support") is False, f"{label}.current_support must be false")
    require(lane.get("comparable") is False, f"{label}.comparable must be false")
    require(lane.get("waiver") is False, f"{label}.waiver must be false")
    require_string(lane.get("failure_class"), f"{label}.failure_class")
    require_string(lane.get("reason"), f"{label}.reason")
    require_string(lane.get("first_failure"), f"{label}.first_failure")
    require_string(lane.get("downstream_goal"), f"{label}.downstream_goal")
    require_string(lane.get("implementation_path"), f"{label}.implementation_path")
    require_string(lane.get("acceptance_path"), f"{label}.acceptance_path")
    downstream_pass = require_string(lane.get("downstream_acceptance_pass_line"), f"{label}.downstream_acceptance_pass_line")
    require(" PASS:" in downstream_pass, f"{label}.downstream_acceptance_pass_line must name an exact PASS line")
    attempted = require_list(lane.get("attempted_command"), f"{label}.attempted_command")
    require(attempted and all(isinstance(part, str) and part for part in attempted), f"{label}.attempted_command must be argv")
    require(Path(attempted[0]).name == "ferrum" and any(part in {"run", "serve"} for part in attempted[1:]), f"{label}.attempted_command must execute a product ferrum run/serve path")
    returncode = lane.get("attempted_returncode")
    require(isinstance(returncode, int) and not isinstance(returncode, bool) and returncode != 0, f"{label}.attempted_returncode must be non-zero")
    require_log(root, lane.get("failure_log"), f"{label}.failure_log")
    require(not lane.get("pass_line"), f"{label}.pass_line forbidden for blocked lane")
    forbidden = {"ratio", "throughput_tok_s", "cells", "run_legacy"} & set(lane)
    require(not forbidden, f"{label} blocked lane contains fabricated performance fields: {sorted(forbidden)}")


def validate_pass_lane(
    root: Path,
    lane: dict[str, Any],
    label: str,
    *,
    allow_synthetic: bool,
    expectations_binding: dict[str, Any],
) -> dict[str, Any] | None:
    pass_line = require_string(lane.get("pass_line"), f"{label}.pass_line")
    expected_pass_line = f"{CORRECTNESS_PASS_PREFIX}: {lane['model_key']}/{lane['backend']}"
    require(pass_line == expected_pass_line, f"{label}.pass_line must equal {expected_pass_line}")
    require("SELFTEST" not in pass_line.upper(), f"{label}.pass_line must not be selftest evidence")
    require("commands" not in lane and "checks" not in lane, f"{label} legacy aggregate commands/checks are forbidden; use scenario_report")
    report_ref = require_object(lane.get("scenario_report"), f"{label}.scenario_report")
    report_path = artifact_path(root, report_ref.get("path"), f"{label}.scenario_report.path")
    require(report_path.is_file() and report_path.stat().st_size > 0, f"{label}.scenario_report is missing or empty")
    report_digest = require_sha256(report_ref.get("sha256"), f"{label}.scenario_report.sha256")
    require(file_sha256(report_path) == report_digest, f"{label}.scenario_report SHA256 mismatch")
    report = read_json(report_path)
    try:
        scenario_runner.validate_report_document(
            report,
            root,
            report_path=report_path,
            allow_internal_fixture=allow_synthetic,
        )
    except scenario_runner.ScenarioError as exc:
        raise BaselineError(f"{label}.scenario_report rejected: {exc}") from exc
    expectations_ref = require_object(
        report.get("expectations_catalog"), f"{label}.scenario_report.expectations_catalog"
    )
    require(
        expectations_ref.get("sha256") == expectations_binding["sha256"]
        and report.get("expectations_catalog_sha256") == expectations_binding["sha256"],
        f"{label}.scenario_report expectations catalog differs from models.lock",
    )
    bindings = {
        "source_git_sha": lane["source_git_sha"],
        "source_tree_sha": lane["source_tree_sha"],
        "model_key": lane["model_key"],
        "backend": lane["backend"],
        "model_revision": lane["model_revision"],
        "model_files": lane["model_files"],
        "hardware_id": lane["hardware_id"],
        "binary_sha256": lane["binary_sha256"],
        "models_lock_sha256": file_sha256(root / "models.lock.json"),
    }
    for key, expected in bindings.items():
        require(report.get(key) == expected, f"{label}.scenario_report.{key} binding mismatch")
    require(report["runner"]["path"] == SCENARIO_RUNNER_PATH.relative_to(REPO_ROOT).as_posix(), f"{label}.scenario_report runner path mismatch")
    invocation_ref = report.get("executor_invocation")
    if invocation_ref is None:
        require(allow_synthetic, f"{label}.scenario_report.executor_invocation is required")
        return None
    invocation_ref = require_object(invocation_ref, f"{label}.scenario_report.executor_invocation")
    invocation_path = require_artifact_sha(
        root,
        invocation_ref.get("path"),
        invocation_ref.get("sha256"),
        f"{label}.scenario_report.executor_invocation",
    )
    invocation = read_json(invocation_path)
    validate_child_output_paths(invocation.get("argv"), f"{label}.scenario_report.executor_invocation.argv")
    require(invocation.get("runner_path") == report["runner"]["path"], f"{label} executor invocation runner path mismatch")
    require(invocation.get("runner_sha256") == report["runner"]["sha256"], f"{label} executor invocation runner SHA mismatch")
    require(invocation.get("mode") == "canonical", f"{label} executor invocation must be canonical")
    return {
        "path": invocation_path.relative_to(root.resolve()).as_posix(),
        "sha256": file_sha256(invocation_path),
        "runner_path": invocation["runner_path"],
        "runner_sha256": invocation["runner_sha256"],
        "mode": invocation["mode"],
    }


def validate_correctness(
    root: Path,
    models: dict[str, dict[str, Any]],
    hardware: dict[str, dict[str, Any]],
    binaries: dict[str, str],
    *,
    allow_synthetic: bool,
    expectations_binding: dict[str, Any],
) -> tuple[dict[tuple[str, str], str], dict[str, dict[str, Any]]]:
    statuses: dict[tuple[str, str], str] = {}
    executor_invocations: dict[str, dict[str, Any]] = {}
    for model_key in PRIMARY_MODELS:
        model = models[model_key]
        for backend in ("cuda", "metal"):
            path = root / "correctness" / model_key / backend / "lane.json"
            lane = read_json(path)
            label = f"correctness.{model_key}.{backend}"
            validate_lane_identity(lane, label=label, model=model, backend=backend, hardware=hardware, binaries=binaries)
            status = lane.get("status")
            require(status in {"pass", "blocked"}, f"{label}.status must be pass or blocked")
            if backend == "cuda":
                require(status == "pass", f"{label} CUDA primary lane must PASS")
            if model_key == "m3-qwen3-30b-a3b":
                require(status == "pass", f"{label} M3 legacy lane must PASS")
            if model_key in {"m1-qwen35-4b", "m2-qwen35-35b-a3b"} and backend == "metal":
                require(status == "blocked", f"{label} must truthfully capture current unsupported Metal lane")
            if status == "pass":
                invocation = validate_pass_lane(
                    root,
                    lane,
                    label,
                    allow_synthetic=allow_synthetic,
                    expectations_binding=expectations_binding,
                )
                if invocation is not None:
                    executor_invocations[f"{model_key}/{backend}"] = invocation
            else:
                validate_blocked_lane(root, lane, label)
            statuses[(model_key, backend)] = str(status)
    require(len(statuses) == 6, "correctness matrix must contain six primary lanes")
    return statuses, executor_invocations


def expected_cells(backend: str) -> set[tuple[str, int]]:
    datasets = {"random", "sharegpt"} if backend == "cuda" else {"random", "real-chat"}
    concurrencies = {1, 4, 16, 32} if backend == "cuda" else {1, 4, 16}
    return {(dataset, concurrency) for dataset in datasets for concurrency in concurrencies}


def validate_benchmark_client(
    root: Path,
    raw: Any,
    label: str,
    *,
    allow_synthetic: bool,
) -> dict[str, Any]:
    client = require_object(raw, label)
    require(Path(require_string(client.get("binary_path"), f"{label}.binary_path")).name == "ferrum", f"{label}.binary_path must identify ferrum")
    digest = require_sha256(client.get("binary_sha256"), f"{label}.binary_sha256")
    binary = require_artifact_sha(root, client.get("artifact_binary"), digest, f"{label}.artifact_binary")
    require(binary.stat().st_size > 0, f"{label}.artifact_binary is empty")
    source_git_sha = require_git_sha(client.get("source_git_sha"), f"{label}.source_git_sha", frozen=False)
    source_tree_sha = require_git_sha(client.get("source_tree_sha"), f"{label}.source_tree_sha", frozen=False)
    require_clean(client, label)
    collector_rows = require_list(client.get("collector_source_files"), f"{label}.collector_source_files")
    collector_files: dict[str, str] = {}
    for index, raw_row in enumerate(collector_rows):
        row = require_object(raw_row, f"{label}.collector_source_files[{index}]")
        path = require_string(row.get("path"), f"{label}.collector_source_files[{index}].path")
        require(path not in collector_files, f"{label}.collector_source_files duplicates {path}")
        collector_files[path] = require_sha256(
            row.get("sha256"), f"{label}.collector_source_files[{index}].sha256"
        )
    require(
        set(collector_files) == set(BENCHMARK_CLIENT_RUST_ALLOWLIST),
        f"{label}.collector_source_files must lock the canonical evidence-only Rust files",
    )
    declared_diff = require_list(client.get("production_rust_diff"), f"{label}.production_rust_diff")
    require(
        declared_diff == list(BENCHMARK_CLIENT_RUST_ALLOWLIST),
        f"{label}.production_rust_diff must equal the canonical evidence-only allowlist",
    )
    collector_identity = {
        "source_git_sha": source_git_sha,
        "source_tree_sha": source_tree_sha,
        "collector_source_files": [
            {"path": path, "sha256": collector_files[path]} for path in sorted(collector_files)
        ],
        "production_rust_diff": sorted(declared_diff),
    }
    require(
        client.get("collector_identity_sha256") == canonical_json_sha256(collector_identity),
        f"{label}.collector_identity_sha256 is not derived from the collector identity",
    )
    if not allow_synthetic:
        require(
            git_value(["cat-file", "-t", source_git_sha]) == "commit",
            f"{label}.source_git_sha is not a local commit object",
        )
        git_value(["merge-base", "--is-ancestor", FROZEN_LEGACY_SHA, source_git_sha])
        git_value(["merge-base", "--is-ancestor", source_git_sha, "HEAD"])
        recomputed_tree = git_value(["rev-parse", f"{source_git_sha}^{{tree}}"])
        require(source_tree_sha == recomputed_tree, f"{label}.source_tree_sha is not derived from source_git_sha")
        changed_paths = git_value(
            ["diff", "--name-only", FROZEN_LEGACY_SHA, source_git_sha, "--", "*.rs"]
        ).splitlines()
        production_rust = sorted(
            path
            for path in changed_paths
            if "/tests/" not in path
            and "/examples/" not in path
            and "/benches/" not in path
            and not Path(path).name.endswith("_test.rs")
        )
        require(
            production_rust == sorted(BENCHMARK_CLIENT_RUST_ALLOWLIST),
            f"{label} source commit changes forbidden production Rust paths: {production_rust}",
        )
        require(
            all(git_file_sha256(source_git_sha, path) == digest for path, digest in collector_files.items()),
            f"{label}.collector_source_files are not derived from the source commit",
        )
    features = require_list(client.get("cargo_features"), f"{label}.cargo_features")
    require(features and all(isinstance(item, str) and item for item in features), f"{label}.cargo_features must not be empty")
    require_log(root, client.get("build_log"), f"{label}.build_log")
    return client


def validate_server_identity(
    root: Path,
    raw: Any,
    label: str,
    *,
    implementation: str,
    backend: str,
    expected_binary_sha256: str | None,
    model: dict[str, Any],
    hardware_memory_bytes: int,
) -> tuple[dict[str, Any], int]:
    identity = require_object(raw, label)
    require(identity.get("implementation") == implementation, f"{label}.implementation mismatch")
    role = "external" if implementation == "A" else "legacy"
    require(identity.get("role") == role, f"{label}.role must be {role}")
    engine = require_string(identity.get("engine"), f"{label}.engine")
    expected_engine = "vllm" if backend == "cuda" else "llama.cpp"
    if implementation == "A":
        require(engine == expected_engine, f"{label}.engine must be {expected_engine}")
        require_string(identity.get("engine_version"), f"{label}.engine_version")
        require_git_sha(identity.get("engine_revision"), f"{label}.engine_revision", frozen=False)
    else:
        require(engine == "ferrum", f"{label}.engine must be ferrum")
        require_git_sha(identity.get("source_git_sha"), f"{label}.source_git_sha")
    digest = require_sha256(identity.get("binary_sha256"), f"{label}.binary_sha256")
    if expected_binary_sha256 is not None:
        require(digest == expected_binary_sha256, f"{label}.binary_sha256 mismatch")
    config_path = require_artifact_sha(
        root,
        identity.get("effective_config"),
        identity.get("effective_config_sha256"),
        f"{label}.effective_config",
    )
    config = read_json(config_path)
    require(config.get("schema_version") == COLLECTOR_SCHEMA_VERSION, f"{label}.effective_config schema mismatch")
    model_lane = model["lanes"][backend]
    required_config = {
        "config_source": "normalized-server-argv",
        "model_key": model["key"],
        "backend": backend,
        "model_repo": model_lane["repo"],
        "model_revision": model_lane["revision"],
        "model_format": model_lane["format"],
        "model_files": locked_file_map(model, backend),
        "request_model": model["official_model_id"],
        "enable_thinking": False,
    }
    for field, expected in required_config.items():
        require(config.get(field) == expected, f"{label}.effective_config.{field} mismatch")
    model_origin_path = require_string(config.get("model_origin_path"), f"{label}.effective_config.model_origin_path")
    cap = require_positive_int(config.get("typed_active_cap"), f"{label}.effective_config.typed_active_cap")
    memory_budget = require_positive_int(
        config.get("memory_budget_bytes"), f"{label}.effective_config.memory_budget_bytes"
    )
    require(memory_budget <= hardware_memory_bytes, f"{label}.effective_config memory budget exceeds hardware")
    require(identity.get("typed_active_cap") == cap, f"{label}.typed_active_cap mismatch")
    for field, expected in (
        ("model_key", model["key"]),
        ("model_repo", model_lane["repo"]),
        ("model_revision", model_lane["revision"]),
        ("model_format", model_lane["format"]),
        ("model_files", locked_file_map(model, backend)),
        ("request_model", model["official_model_id"]),
        ("model_origin_path", model_origin_path),
        ("memory_budget_bytes", memory_budget),
    ):
        require(identity.get(field) == expected, f"{label}.{field} mismatch")
    identity["_config"] = config
    return identity, cap


def validate_endpoint_probe(
    root: Path,
    raw: Any,
    label: str,
    *,
    expected_url: str,
) -> tuple[dict[str, Any], dict[str, Any], datetime, datetime]:
    evidence = require_object(raw, label)
    receipt_origin = require_string(evidence.get("receipt_origin_path"), f"{label}.receipt_origin_path")
    body_origin = require_string(evidence.get("body_origin_path"), f"{label}.body_origin_path")
    receipt_path = require_artifact_sha(
        root, evidence.get("receipt"), evidence.get("receipt_sha256"), f"{label}.receipt"
    )
    body_path = require_artifact_sha(
        root, evidence.get("body"), evidence.get("body_sha256"), f"{label}.body"
    )
    receipt = read_json(receipt_path)
    require(receipt.get("schema_version") == resource_sampler.SCHEMA_VERSION, f"{label}.receipt schema mismatch")
    require(receipt.get("collector_path") == resource_sampler.COLLECTOR_RELATIVE_PATH, f"{label}.receipt collector path mismatch")
    require(receipt.get("collector_sha256") == file_sha256(RESOURCE_SAMPLER_PATH), f"{label}.receipt collector SHA256 mismatch")
    require(receipt.get("url") == expected_url, f"{label}.receipt URL mismatch")
    require(receipt.get("http_status") == 200 and receipt.get("returncode") == 0, f"{label}.receipt did not capture HTTP 200 success")
    require(receipt.get("body_origin_path") == body_origin, f"{label}.receipt body origin mismatch")
    require(receipt.get("body_sha256") == file_sha256(body_path), f"{label}.receipt body SHA256 mismatch")
    require(receipt.get("body_size_bytes") == body_path.stat().st_size, f"{label}.receipt body size mismatch")
    argv, options, _ = parse_argv(receipt.get("argv"), f"{label}.receipt.argv")
    require(
        len(argv) >= 2
        and Path(argv[0]).name.startswith("python")
        and Path(argv[1]).name == RESOURCE_SAMPLER_PATH.name,
        f"{label}.receipt.argv must execute the checked-in endpoint collector",
    )
    require_option(options, "--probe-url", expected_url, f"{label}.receipt.argv")
    require_option(options, "--probe-body-out", body_origin, f"{label}.receipt.argv")
    require_option(options, "--probe-receipt-out", receipt_origin, f"{label}.receipt.argv")
    require_option(options, "--probe-timeout-sec", 10, f"{label}.receipt.argv")
    validate_execution_window(receipt, f"{label}.receipt")
    started = parse_timestamp(receipt.get("started_at"), f"{label}.receipt.started_at")
    finished = parse_timestamp(receipt.get("finished_at"), f"{label}.receipt.finished_at")
    body = read_json(body_path)
    return receipt, body, started, finished


def validate_server_sessions(
    root: Path,
    raw: Any,
    label: str,
    *,
    identities: dict[str, tuple[dict[str, Any], int]],
    hardware_id: str,
    hardware_fingerprint: str,
    require_abba: bool,
    backend: str,
    model: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows = require_list(raw, label)
    required_cells = sorted(expected_cells(backend))
    expected_slots = set(range(1, 9)) if require_abba else {
        slot for slot, owner in enumerate(SLOT_ORDER, start=1) if owner == "A"
    }
    expected_count = len(expected_slots)
    require(len(rows) == expected_count, f"{label} must contain exactly {expected_count} independent server sessions")
    indexed: dict[str, dict[str, Any]] = {}
    seen_slots: set[int] = set()
    by_sequence: dict[int, dict[str, Any]] = {}
    for index, raw_session in enumerate(rows):
        session = require_object(raw_session, f"{label}[{index}]")
        session_id = require_string(session.get("session_id"), f"{label}[{index}].session_id")
        require(session_id not in indexed, f"{label} duplicate session_id {session_id}")
        implementation = require_string(session.get("implementation"), f"{label}[{index}].implementation")
        require(implementation in identities, f"{label}[{index}] unknown implementation {implementation}")
        slot = require_positive_int(session.get("slot"), f"{label}[{index}].slot")
        require(slot in expected_slots and slot not in seen_slots, f"{label} duplicate or invalid slot {slot}")
        seen_slots.add(slot)
        if require_abba:
            require(SLOT_ORDER[slot - 1] == implementation, f"{label}[{index}] violates ABBA-BAAB slot ownership")
        else:
            require(implementation == "A", f"{label}[{index}] standalone sessions must be external A")
        sequence = require_positive_int(session.get("sequence"), f"{label}[{index}].sequence")
        require(sequence == slot and sequence not in by_sequence, f"{label}[{index}].sequence does not bind canonical slot order")
        by_sequence[sequence] = session
        identity, cap = identities[implementation]
        require(session.get("hardware_id") == hardware_id, f"{label}[{index}].hardware_id mismatch")
        require(session.get("hardware_fingerprint") == hardware_fingerprint, f"{label}[{index}].hardware_fingerprint mismatch")
        require(session.get("effective_config") == identity.get("effective_config"), f"{label}[{index}].effective_config mismatch")
        require(session.get("effective_config_sha256") == identity.get("effective_config_sha256"), f"{label}[{index}].effective_config_sha256 mismatch")
        require(session.get("typed_active_cap") == cap, f"{label}[{index}].typed_active_cap mismatch")
        require(session.get("executed_binary_sha256") == identity["binary_sha256"], f"{label}[{index}].executed_binary_sha256 mismatch")
        pid = require_positive_int(session.get("pid"), f"{label}[{index}].pid")
        pgid = require_positive_int(session.get("pgid"), f"{label}[{index}].pgid")
        process_marker = require_string(session.get("process_start_marker"), f"{label}[{index}].process_start_marker")
        try:
            derived_process_marker = resource_sampler.process_marker_from_source(
                pid, session.get("process_start_source")
            )
        except resource_sampler.ResourceEvidenceError as exc:
            raise BaselineError(f"{label}[{index}].process_start_source rejected: {exc}") from exc
        require(process_marker == derived_process_marker, f"{label}[{index}].process_start_marker is not derived from raw OS identity evidence")
        argv = require_list(session.get("server_argv"), f"{label}[{index}].server_argv")
        require(argv and all(isinstance(part, str) and part for part in argv), f"{label}[{index}].server_argv must be argv")
        _, options, _ = parse_argv(argv, f"{label}[{index}].server_argv")
        joined = " ".join(argv).lower()
        if implementation == "B":
            require(Path(argv[0]).name == "ferrum" and "serve" in argv[1:], f"{label}[{index}] must execute frozen ferrum serve")
            serve_index = argv.index("serve")
            require(
                serve_index + 1 < len(argv) and argv[serve_index + 1] == identity["model_origin_path"],
                f"{label}[{index}].server_argv model origin mismatch",
            )
            require_option(options, "--backend", backend, f"{label}[{index}].server_argv")
            product_origin = require_string(
                session.get("product_effective_config_origin_path"),
                f"{label}[{index}].product_effective_config_origin_path",
            )
            require_option(options, "--effective-config-json", product_origin, f"{label}[{index}].server_argv")
            product_config_path = require_artifact_sha(
                root,
                session.get("product_effective_config"),
                session.get("product_effective_config_sha256"),
                f"{label}[{index}].product_effective_config",
            )
            product_config = read_json(product_config_path)
            require(product_config.get("schema_version") == 1, f"{label}[{index}].product_effective_config schema mismatch")
            for field, kind in (
                ("entries", list),
                ("model_capabilities", dict),
                ("hardware_capabilities", dict),
                ("workload_profile", dict),
                ("decisions", list),
                ("admission", dict),
            ):
                require(isinstance(product_config.get(field), kind), f"{label}[{index}].product_effective_config.{field} has wrong type")
            require(
                product_config["hardware_capabilities"].get("backend") == backend,
                f"{label}[{index}].product_effective_config backend mismatch",
            )
            require(
                product_config["admission"].get("effective_max_concurrent") == cap,
                f"{label}[{index}].product_effective_config active cap mismatch",
            )
        else:
            marker = "vllm" if identity["engine"] == "vllm" else "llama"
            require(marker in joined, f"{label}[{index}].server_argv does not identify {identity['engine']}")
            require_option(options, "--model", identity["model_origin_path"], f"{label}[{index}].server_argv")
            alias_option = "--served-model-name" if identity["engine"] == "vllm" else "--alias"
            require_option(options, alias_option, identity["request_model"], f"{label}[{index}].server_argv")
        require_string(session.get("base_url"), f"{label}[{index}].base_url")
        validate_execution_window(session, f"{label}[{index}]")
        started_at = parse_timestamp(session.get("started_at"), f"{label}[{index}].started_at")
        ready_at = parse_timestamp(session.get("ready_at"), f"{label}[{index}].ready_at")
        measurement_started_at = parse_timestamp(
            session.get("measurement_started_at"), f"{label}[{index}].measurement_started_at"
        )
        measurement_finished_at = parse_timestamp(
            session.get("measurement_finished_at"), f"{label}[{index}].measurement_finished_at"
        )
        shutdown_started_at = parse_timestamp(
            session.get("shutdown_started_at"), f"{label}[{index}].shutdown_started_at"
        )
        finished_at = parse_timestamp(session.get("finished_at"), f"{label}[{index}].finished_at")
        require(
            started_at < ready_at < measurement_started_at < measurement_finished_at < shutdown_started_at < finished_at,
            f"{label}[{index}] must follow start -> ready -> measurement -> shutdown -> finish",
        )
        cell_windows_raw = require_list(session.get("cell_windows"), f"{label}[{index}].cell_windows")
        require(
            len(cell_windows_raw) == len(required_cells),
            f"{label}[{index}].cell_windows must cover the complete fixed cell matrix",
        )
        cell_windows: dict[str, dict[str, Any]] = {}
        previous_finished: datetime | None = None
        for cell_index, ((expected_dataset, expected_concurrency), raw_window) in enumerate(
            zip(required_cells, cell_windows_raw), start=1
        ):
            window = require_object(raw_window, f"{label}[{index}].cell_windows[{cell_index - 1}]")
            expected_cell_id = f"{expected_dataset}:c{expected_concurrency}"
            require(window.get("sequence") == cell_index, f"{label}[{index}].cell_windows sequence mismatch")
            require(window.get("cell_id") == expected_cell_id, f"{label}[{index}].cell_windows cell_id/order mismatch")
            require(window.get("dataset") == expected_dataset, f"{label}[{index}].cell_windows dataset mismatch")
            require(window.get("concurrency") == expected_concurrency, f"{label}[{index}].cell_windows concurrency mismatch")
            window_started = parse_timestamp(window.get("started_at"), f"{label}[{index}].cell_windows[{cell_index - 1}].started_at")
            window_finished = parse_timestamp(window.get("finished_at"), f"{label}[{index}].cell_windows[{cell_index - 1}].finished_at")
            require(window_started < window_finished, f"{label}[{index}].cell_windows[{cell_index - 1}] has an invalid window")
            if previous_finished is not None:
                require(previous_finished <= window_started, f"{label}[{index}].cell_windows overlap or are out of order")
            previous_finished = window_finished
            window["_started_at"] = window_started
            window["_finished_at"] = window_finished
            cell_windows[expected_cell_id] = window
        first_window = cell_windows[f"{required_cells[0][0]}:c{required_cells[0][1]}"]
        last_window = cell_windows[f"{required_cells[-1][0]}:c{required_cells[-1][1]}"]
        require(
            first_window["_started_at"] == measurement_started_at
            and last_window["_finished_at"] == measurement_finished_at,
            f"{label}[{index}] overall measurement bounds must equal first/last cell windows",
        )
        require(
            all(
                ready_at < window["_started_at"] < window["_finished_at"] < shutdown_started_at
                for window in cell_windows.values()
            ),
            f"{label}[{index}].cell_windows must be between ready and shutdown",
        )
        probe_url = f"{session['base_url'].rstrip('/')}/v1/models"
        _, ready_body, ready_probe_started, ready_probe_finished = validate_endpoint_probe(
            root,
            session.get("ready_probe"),
            f"{label}[{index}].ready_probe",
            expected_url=probe_url,
        )
        require(
            started_at < ready_probe_started < ready_probe_finished == ready_at,
            f"{label}[{index}].ready_at is not bound to its collected HTTP probe",
        )
        require(bool(require_list(ready_body.get("data"), f"{label}[{index}].ready_probe.body.data")), f"{label}[{index}].ready_probe observed no loaded models")
        _, model_body, model_probe_started, model_probe_finished = validate_endpoint_probe(
            root,
            session.get("model_probe"),
            f"{label}[{index}].model_probe",
            expected_url=probe_url,
        )
        require(
            ready_at <= model_probe_started < model_probe_finished < measurement_started_at,
            f"{label}[{index}].model_probe must run after ready and before measurement",
        )
        model_rows = require_list(model_body.get("data"), f"{label}[{index}].model_probe.body.data")
        observed_model_ids = [
            require_string(require_object(row, f"{label}[{index}].model_probe.model").get("id"), f"{label}[{index}].model_probe.model.id")
            for row in model_rows
        ]
        require(observed_model_ids == [identity["request_model"]], f"{label}[{index}].model_probe loaded model identity mismatch")
        require(session.get("model_key") == model["key"], f"{label}[{index}].model_key mismatch")
        require(session.get("model_revision") == model["lanes"][backend]["revision"], f"{label}[{index}].model_revision mismatch")
        require(session.get("model_files") == locked_file_map(model, backend), f"{label}[{index}].model_files mismatch")
        require(isinstance(session.get("returncode"), int) and not isinstance(session.get("returncode"), bool), f"{label}[{index}].returncode must be an integer")
        require(session.get("returncode") == 0, f"{label}[{index}].returncode must be zero")
        require(session.get("shutdown_clean") is True, f"{label}[{index}].shutdown_clean must be true")
        runtime_log = require_artifact_sha(root, session.get("runtime_log"), session.get("runtime_log_sha256"), f"{label}[{index}].runtime_log")
        require(runtime_log.stat().st_size >= 32, f"{label}[{index}].runtime_log is too small")
        require_string(session.get("runtime_log_origin_path"), f"{label}[{index}].runtime_log_origin_path")
        session["_pid"] = pid
        session["_pgid"] = pgid
        session["_identity"] = identity
        session["_timeline"] = {
            "started_at": started_at,
            "ready_at": ready_at,
            "measurement_started_at": measurement_started_at,
            "measurement_finished_at": measurement_finished_at,
            "shutdown_started_at": shutdown_started_at,
            "finished_at": finished_at,
        }
        session["_cell_windows"] = cell_windows
        indexed[session_id] = session
    require(seen_slots == expected_slots, f"{label} must cover every required outer slot exactly once")
    require({session["implementation"] for session in indexed.values()} == set(identities), f"{label} implementation coverage mismatch")
    ordered = [by_sequence[key] for key in sorted(by_sequence)]
    for previous, current in zip(ordered, ordered[1:]):
        require(
            previous["_timeline"]["finished_at"] <= current["_timeline"]["started_at"],
            f"{label} server sessions overlap globally between sequence {previous['sequence']} and {current['sequence']}",
        )
    return indexed


def validate_resource_metrics(
    root: Path,
    raw: Any,
    label: str,
    *,
    backend: str,
    hardware_memory_bytes: int,
    requested_concurrency: int,
    typed_active_cap: int,
    memory_budget_bytes: int,
    session: dict[str, Any],
    cell_id: str,
    measurement_started_at: str,
    measurement_finished_at: str,
    runtime_log_origin_path: str,
    allow_process_probe: bool = False,
) -> dict[str, Any]:
    evidence = require_object(raw, label)
    require(
        evidence.get("collector_sha256") == file_sha256(RESOURCE_SAMPLER_PATH),
        f"{label}.collector_sha256 mismatch",
    )
    observation_origin = require_string(
        evidence.get("observation_origin_path"), f"{label}.observation_origin_path"
    )
    observation_path = require_artifact_sha(
        root,
        evidence.get("observations"),
        evidence.get("observations_sha256"),
        f"{label}.observations",
    )
    argv, options, _ = parse_argv(evidence.get("sampler_argv"), f"{label}.sampler_argv")
    require(
        len(argv) >= 2
        and Path(argv[0]).name in {"python", "python3"}
        and Path(argv[1]).name == RESOURCE_SAMPLER_PATH.name,
        f"{label}.sampler_argv must execute the checked-in resource sampler",
    )
    required_options = {
        "--out": observation_origin,
        "--pid": session["_pid"],
        "--pgid": session["_pgid"],
        "--session-id": session["session_id"],
        "--cell-id": cell_id,
        "--backend": backend,
        "--hardware-id": session["hardware_id"],
        "--base-url": session["base_url"],
        "--runtime-log": runtime_log_origin_path,
    }
    for option, expected in required_options.items():
        require_option(options, option, expected, f"{label}.sampler_argv")
    require_option(options, "--interval-ms", 250, f"{label}.sampler_argv")
    require_option(options, "--max-duration-sec", 7200, f"{label}.sampler_argv")
    require("--stop-file" in options, f"{label}.sampler_argv must use a bounded stop file")
    probe_format = options.get("--active-probe-format")
    require(probe_format in {"json", "prometheus", "process"}, f"{label}.sampler_argv active probe format missing")
    if allow_process_probe:
        require(probe_format == "process", f"{label}.sampler_argv one-shot run must use process-alive active evidence")
        require_option(options, "--active-selector", "process-alive", f"{label}.sampler_argv")
        require_option(options, "--active-semantics", "process-alive", f"{label}.sampler_argv")
    else:
        require(probe_format != "process", f"{label}.sampler_argv HTTP measurements cannot use process-alive active evidence")
        require("--active-path" in options and "--active-selector" in options, f"{label}.sampler_argv active HTTP probe is incomplete")
        require_option(
            options,
            "--active-semantics",
            "scheduler-active-high-water",
            f"{label}.sampler_argv",
        )
    try:
        recomputed = resource_sampler.derive_summary(
            observation_path,
            session_id=session["session_id"],
            cell_id=cell_id,
            backend=backend,
            hardware_id=session["hardware_id"],
            pid=session["_pid"],
            pgid=session["_pgid"],
            process_start_marker=session["process_start_marker"],
            base_url=session["base_url"],
            session_started_at=session["started_at"],
            session_finished_at=session["finished_at"],
            measurement_started_at=measurement_started_at,
            measurement_finished_at=measurement_finished_at,
            memory_budget_bytes=memory_budget_bytes,
            requested_concurrency=requested_concurrency,
            typed_active_cap=typed_active_cap,
            runtime_log_path=runtime_log_origin_path,
        )
    except resource_sampler.ResourceEvidenceError as exc:
        raise BaselineError(f"{label} raw resource evidence rejected: {exc}") from exc
    summary = require_object(evidence.get("summary"), f"{label}.summary")
    require(summary == recomputed, f"{label}.summary is not derived from raw resource observations")
    peak = require_positive_int(summary.get("peak_memory_bytes"), f"{label}.summary.peak_memory_bytes")
    budget = require_positive_int(summary.get("memory_budget_bytes"), f"{label}.summary.memory_budget_bytes")
    require(peak <= budget <= hardware_memory_bytes, f"{label} memory peak/budget exceeds frozen hardware")
    headroom = require_nonnegative_int(summary.get("physical_headroom_bytes"), f"{label}.summary.physical_headroom_bytes")
    minimum_headroom = 2 * 1024**3 if backend == "metal" else 512 * 1024**2
    require(headroom >= minimum_headroom, f"{label}.summary.physical_headroom_bytes is below the {minimum_headroom}-byte floor")
    swap_start = require_nonnegative_int(summary.get("swap_start_bytes"), f"{label}.summary.swap_start_bytes")
    swap_end = require_nonnegative_int(summary.get("swap_end_bytes"), f"{label}.summary.swap_end_bytes")
    require(swap_end == swap_start, f"{label} measured swap growth must be zero")
    require(summary.get("oom_count") == 0, f"{label}.summary.oom_count must be zero")
    require(summary.get("admission_error_count") == 0, f"{label}.summary.admission_error_count must be zero")
    observed = require_positive_int(summary.get("observed_max_active"), f"{label}.summary.observed_max_active")
    require(
        observed == min(requested_concurrency, typed_active_cap),
        f"{label}.observed_max_active must reach the requested/typed scheduler high-water",
    )
    if backend == "metal":
        for field in ("thermal_start", "thermal_end", "power_mode_start", "power_mode_end"):
            require_string(summary.get(field), f"{label}.summary.{field}")
        require(summary.get("thermal_throttling_count") == 0, f"{label}.summary.thermal_throttling_count must be zero")
    return recomputed


def validate_workload_identity(
    root: Path,
    raw: Any,
    label: str,
    *,
    model: dict[str, Any],
    backend: str,
    dataset: str,
    concurrency: int,
    typed_active_cap: int,
) -> dict[str, Any]:
    workload = require_object(raw, label)
    require(workload.get("dataset_id") == dataset, f"{label}.dataset_id mismatch")
    dataset_path = require_artifact_sha(root, workload.get("dataset_artifact"), workload.get("dataset_sha256"), f"{label}.dataset_artifact")
    require(dataset_path.stat().st_size > 0, f"{label}.dataset_artifact is empty")
    tokenizer_digest = require_sha256(workload.get("tokenizer_sha256"), f"{label}.tokenizer_sha256")
    require(tokenizer_digest in tokenizer_sha_set(model, backend), f"{label}.tokenizer_sha256 is not locked by models.lock")
    tokenizer_path = require_artifact_sha(root, workload.get("tokenizer_artifact"), tokenizer_digest, f"{label}.tokenizer_artifact")
    require(tokenizer_path.name == "tokenizer.json", f"{label}.tokenizer_artifact must be tokenizer.json")
    tokenizer_source = locked_tokenizer_source(model, backend)
    require(workload.get("tokenizer_id") == tokenizer_source["repo"], f"{label}.tokenizer_id mismatch")
    require(workload.get("tokenizer_revision") == tokenizer_source["revision"], f"{label}.tokenizer_revision mismatch")
    require_string(workload.get("tokenizer_origin_path"), f"{label}.tokenizer_origin_path")
    require_string(workload.get("dataset_origin_path"), f"{label}.dataset_origin_path")
    config_path = require_artifact_sha(root, workload.get("effective_config"), workload.get("effective_config_sha256"), f"{label}.effective_config")
    config = read_json(config_path)
    require(config.get("schema_version") == COLLECTOR_SCHEMA_VERSION, f"{label}.effective_config schema mismatch")
    expected_ignore_eos = dataset == "random"
    required_config = {
        "dataset_id": dataset,
        "dataset_sha256": workload["dataset_sha256"],
        "tokenizer_sha256": tokenizer_digest,
        "model_revision": model["lanes"][backend]["revision"],
        "seed": 9271,
        "max_output_tokens": 128,
        "ignore_eos": expected_ignore_eos,
        "enable_thinking": False,
        "requested_concurrency": concurrency,
        "typed_active_cap": typed_active_cap,
    }
    for field, expected in required_config.items():
        require(config.get(field) == expected, f"{label}.effective_config.{field} mismatch")
    require_string(config.get("request_model"), f"{label}.effective_config.request_model")
    workload["_config"] = config
    return workload


def validate_bench_argv(
    raw: Any,
    label: str,
    *,
    client: dict[str, Any],
    workload: dict[str, Any],
    session: dict[str, Any],
    backend: str,
    dataset: str,
    concurrency: int,
    raw_report_origin_path: str,
) -> dict[str, float]:
    argv, options, switches = parse_argv(raw, label)
    require(argv[0] == client["binary_path"] and "bench-serve" in argv[1:], f"{label} must execute the identified ferrum benchmark client")
    for switch in ("--fail-on-error", "--require-ci"):
        require(switch in switches, f"{label} missing canonical {switch}")
    require("--concurrency-sweep" not in options, f"{label} must capture one explicit concurrency cell")
    require_option(options, "--base-url", session["base_url"], label)
    require_option(options, "--model", workload["_config"]["request_model"], label)
    require_option(options, "--tokenizer", workload["tokenizer_origin_path"], label)
    require_option(options, "--concurrency", concurrency, label)
    require_option(options, "--random-output-len", 128, label)
    require_option(options, "--num-prompts", 100, label)
    require_option(options, "--warmup-requests", 10, label)
    require_option(options, "--n-repeats", 3, label)
    require_option(options, "--seed", 9271, label)
    require_option(options, "--output", "json", label)
    require_option(options, "--out", raw_report_origin_path, label)
    require_option(options, "--hw-id", session["hardware_id"], label)
    require_option(options, "--commit-sha", client["source_git_sha"], label)
    require("--goodput" in options, f"{label} must explicitly lock the goodput SLO")
    parsed_slo: dict[str, float] = {}
    for token in options["--goodput"].replace(",", " ").split():
        require(":" in token, f"{label} --goodput contains malformed token {token}")
        key, value = token.split(":", 1)
        normalized = "e2e" if key == "e2el" else key
        require(normalized in {"ttft", "tpot", "e2e"} and normalized not in parsed_slo, f"{label} --goodput contains invalid/duplicate metric {key}")
        try:
            parsed_slo[normalized] = float(value)
        except ValueError as exc:
            raise BaselineError(f"{label} --goodput {key} must be numeric") from exc
        require(math.isfinite(parsed_slo[normalized]) and parsed_slo[normalized] > 0, f"{label} --goodput {key} must be positive finite")
    require(set(parsed_slo) == {"ttft", "tpot", "e2e"}, f"{label} --goodput must lock ttft/tpot/e2e")
    dataset_arg = "random" if dataset == "random" else "sharegpt"
    require_option(options, "--dataset", dataset_arg, label)
    require_option(options, "--random-input-len", 256 if backend == "cuda" else 64, label)
    if dataset == "random":
        require("--ignore-eos" in switches, f"{label} random workload requires --ignore-eos")
        require("--sharegpt-path" not in options, f"{label} random workload cannot use --sharegpt-path")
    else:
        require("--ignore-eos" not in switches, f"{label} realistic workload cannot use --ignore-eos")
        require_option(options, "--sharegpt-path", workload["dataset_origin_path"], label)
    require_option(options, "--enable-thinking", "false", label)
    return {
        "ttft_p99_ms": parsed_slo["ttft"],
        "tpot_p99_ms": parsed_slo["tpot"],
        "e2e_p99_ms": parsed_slo["e2e"],
    }


def validate_zero_list(raw: Any, label: str, *, length: int = 3) -> None:
    values = require_list(raw, label)
    require(len(values) == length and all(value == 0 and not isinstance(value, bool) for value in values), f"{label} must contain {length} zeros")


def validate_repeat_row(
    raw: Any,
    label: str,
    *,
    expected_repeat: int,
) -> dict[str, Any]:
    row = require_object(raw, label)
    repeat = require_positive_int(row.get("repeat"), f"{label}.repeat")
    require(repeat == expected_repeat, f"{label}.repeat must be {expected_repeat}")
    duration = require_number(row.get("duration_s"), f"{label}.duration_s", positive=True)
    for metric in HTTP_SCALAR_METRICS:
        require_number(row.get(metric), f"{label}.{metric}", positive=metric != "goodput_rps", nonnegative=metric == "goodput_rps")
    for metric in HTTP_ALWAYS_AVAILABLE_LATENCY_METRICS:
        percentiles = require_object(row.get(metric), f"{label}.{metric}")
        require(set(percentiles) == set(HTTP_PERCENTILES), f"{label}.{metric} percentile set mismatch")
        previous = 0.0
        for percentile in HTTP_PERCENTILES:
            value = require_number(percentiles.get(percentile), f"{label}.{metric}.{percentile}", positive=True)
            require(value >= previous, f"{label}.{metric} percentiles must be monotonic")
            previous = value
    itl_percentiles = require_object(row.get("itl_ms"), f"{label}.itl_ms")
    require(set(itl_percentiles) == set(HTTP_PERCENTILES), f"{label}.itl_ms percentile set mismatch")
    previous_itl = 0.0
    for percentile in HTTP_PERCENTILES:
        value = require_number(itl_percentiles.get(percentile), f"{label}.itl_ms.{percentile}", nonnegative=True)
        require(value >= previous_itl, f"{label}.itl_ms percentiles must be monotonic")
        previous_itl = value
    expected_requests = require_positive_int(row.get("expected_requests"), f"{label}.expected_requests")
    completed_requests = require_nonnegative_int(row.get("completed_requests"), f"{label}.completed_requests")
    errored_requests = require_nonnegative_int(row.get("errored_requests"), f"{label}.errored_requests")
    require(expected_requests == 100, f"{label}.expected_requests must be 100")
    require(
        expected_requests == completed_requests + errored_requests,
        f"{label}.expected_requests must equal completed_requests + errored_requests",
    )
    require(completed_requests == 100, f"{label}.completed_requests must be 100")
    require(errored_requests == 0, f"{label}.errored_requests must be zero")
    itl_eligible = require_nonnegative_int(row.get("itl_eligible_requests"), f"{label}.itl_eligible_requests")
    itl_ineligible = require_nonnegative_int(row.get("itl_ineligible_requests"), f"{label}.itl_ineligible_requests")
    require(itl_eligible + itl_ineligible == expected_requests, f"{label} ITL eligible/ineligible request count mismatch")
    itl_expected = require_nonnegative_int(row.get("itl_expected_intervals"), f"{label}.itl_expected_intervals")
    itl_observed = require_nonnegative_int(row.get("itl_observed_intervals"), f"{label}.itl_observed_intervals")
    eligibility_counts = require_object(row.get("itl_eligibility_counts"), f"{label}.itl_eligibility_counts")
    require(set(eligibility_counts) == ITL_ELIGIBILITY_FIELDS, f"{label}.itl_eligibility_counts field set mismatch")
    normalized_eligibility_counts = {
        field: require_nonnegative_int(eligibility_counts.get(field), f"{label}.itl_eligibility_counts.{field}")
        for field in ITL_ELIGIBILITY_FIELDS
    }
    require(sum(normalized_eligibility_counts.values()) == expected_requests, f"{label}.itl_eligibility_counts total mismatch")
    require(normalized_eligibility_counts["eligible"] == itl_eligible, f"{label}.itl_eligibility_counts eligible mismatch")
    require(sum(value for field, value in normalized_eligibility_counts.items() if field != "eligible") == itl_ineligible, f"{label}.itl_eligibility_counts ineligible mismatch")
    if itl_ineligible == 0:
        require(itl_eligible == expected_requests, f"{label} complete ITL evidence must cover every request")
        require(itl_expected == itl_observed and itl_expected > 0, f"{label} eligible ITL interval totals mismatch")
        require(all(float(itl_percentiles[field]) > 0.0 for field in HTTP_PERCENTILES), f"{label} eligible ITL percentiles must be positive")
    else:
        require(all(float(itl_percentiles[field]) == 0.0 for field in HTTP_PERCENTILES), f"{label} ineligible ITL repeat must not expose partial percentiles")
    require(row.get("warmup_expected") == 10 and row.get("warmup_completed") == 10, f"{label} warmup must be 10/10")
    require(row.get("warmup_errored") == 0, f"{label}.warmup_errored must be zero")
    require(row.get("output_token_count_source") == "usage", f"{label}.output_token_count_source must be usage")
    output_tokens = require_positive_int(row.get("output_tokens"), f"{label}.output_tokens")
    actual_input_tokens = require_positive_int(row.get("actual_input_tokens"), f"{label}.actual_input_tokens")
    require(close_enough(float(row["output_throughput_tps"]), output_tokens / duration), f"{label}.output_throughput_tps is not derived from tokens/duration")
    require(close_enough(float(row["total_throughput_tps"]), (actual_input_tokens + output_tokens) / duration), f"{label}.total_throughput_tps is not derived from tokens/duration")
    require(close_enough(float(row["request_throughput_rps"]), 100.0 / duration), f"{label}.request_throughput_rps is not derived from completions/duration")
    require(float(row["goodput_rps"]) <= float(row["request_throughput_rps"]), f"{label}.goodput_rps exceeds request throughput")
    for percentile in HTTP_PERCENTILES:
        require(float(row["e2e_ms"][percentile]) >= float(row["ttft_ms"][percentile]), f"{label}.e2e_ms.{percentile} is below TTFT")
    quality = require_object(row.get("quality_issues"), f"{label}.quality_issues")
    expected_quality = {field.removesuffix("_per_run") for field in BENCH_QUALITY_FIELDS}
    require(set(quality) == expected_quality and all(quality[field] == 0 for field in expected_quality), f"{label}.quality_issues must contain only zero canonical counters")
    warmup_quality = require_object(row.get("warmup_quality_issues"), f"{label}.warmup_quality_issues")
    require(set(warmup_quality) == expected_quality and all(warmup_quality[field] == 0 for field in expected_quality), f"{label}.warmup_quality_issues must contain only zero canonical counters")
    return row


def validate_request_itl_evidence(
    raw: Any,
    *,
    output_tokens: int,
    label: str,
) -> str:
    evidence = require_object(raw, label)
    require(set(evidence) == ITL_EVIDENCE_FIELDS, f"{label} field set mismatch")
    require(evidence.get("source") == "sse_delta_events", f"{label}.source must be sse_delta_events for G00 HTTP evidence")
    output_events = require_nonnegative_int(evidence.get("output_events"), f"{label}.output_events")
    usage_tokens = evidence.get("usage_output_tokens")
    if usage_tokens is not None:
        usage_tokens = require_nonnegative_int(usage_tokens, f"{label}.usage_output_tokens")
    observed_intervals = require_nonnegative_int(evidence.get("observed_intervals"), f"{label}.observed_intervals")
    coalesced_chunks = require_nonnegative_int(
        evidence.get("transport_coalesced_output_chunks"),
        f"{label}.transport_coalesced_output_chunks",
    )
    require(observed_intervals == max(output_events - 1, 0), f"{label}.observed_intervals is not derived from output events")
    require(coalesced_chunks <= output_events // 2, f"{label}.transport coalescing count is impossible")

    if usage_tokens is None:
        expected_eligibility = "missing_usage"
    elif usage_tokens < 2:
        expected_eligibility = "too_short"
    elif usage_tokens != output_events:
        expected_eligibility = "event_usage_mismatch"
    elif observed_intervals != usage_tokens - 1:
        expected_eligibility = "interval_count_mismatch"
    elif coalesced_chunks > 0:
        expected_eligibility = "transport_coalesced"
    else:
        expected_eligibility = "eligible"
    require(evidence.get("eligibility") == expected_eligibility, f"{label}.eligibility is not derived from raw ITL evidence")
    require(usage_tokens == output_tokens, f"{label}.usage_output_tokens differs from output_tokens_per_request")
    return expected_eligibility


def validate_raw_bench_report(
    report: dict[str, Any],
    repeats: list[dict[str, Any]],
    label: str,
    *,
    client: dict[str, Any],
    workload: dict[str, Any],
    session: dict[str, Any],
    backend: str,
    dataset: str,
    concurrency: int,
) -> float:
    require(report.get("model") == workload["_config"]["request_model"], f"{label}.model mismatch")
    require(report.get("backend") == backend, f"{label}.backend mismatch")
    require(report.get("scenario") == "closed_loop", f"{label}.scenario must be closed_loop")
    require(report.get("concurrency") == concurrency, f"{label}.concurrency mismatch")
    require(report.get("request_rate") in (None, 0), f"{label}.request_rate must be absent for closed loop")
    require_positive_int(report.get("n_prompt"), f"{label}.n_prompt")
    if dataset == "random":
        require(report["n_prompt"] == (256 if backend == "cuda" else 64), f"{label}.n_prompt mismatch")
    require(report.get("n_gen") == 128, f"{label}.n_gen must be 128")
    require(report.get("n_repeats") == 3, f"{label}.n_repeats must be 3")
    require(report.get("n_requests_per_run") == 100, f"{label}.n_requests_per_run must be 100")
    require(report.get("warmup_requests") == 10, f"{label}.warmup_requests must be 10")
    require(report.get("output_token_count_source") == "usage", f"{label}.output_token_count_source must be usage")
    completed = require_list(report.get("completed_per_run"), f"{label}.completed_per_run")
    require(completed == [100, 100, 100], f"{label}.completed_per_run must be 100/100/100")
    require(completed == [row["completed_requests"] for row in repeats], f"{label}.completed_per_run differs from repeat metrics")
    validate_zero_list(report.get("errored_per_run"), f"{label}.errored_per_run")
    require(report["errored_per_run"] == [row["errored_requests"] for row in repeats], f"{label}.errored_per_run differs from repeat metrics")
    for field in BENCH_QUALITY_FIELDS:
        validate_zero_list(report.get(field), f"{label}.{field}")
    quality_rows = require_list(report.get("quality_issues_per_run"), f"{label}.quality_issues_per_run")
    require(len(quality_rows) == 3, f"{label}.quality_issues_per_run must have three rows")
    for index, raw_quality in enumerate(quality_rows):
        quality = require_object(raw_quality, f"{label}.quality_issues_per_run[{index}]")
        expected_quality = {field.removesuffix("_per_run") for field in BENCH_QUALITY_FIELDS}
        require(set(quality) == expected_quality, f"{label}.quality_issues_per_run[{index}] field set mismatch")
        require(all(value == 0 and not isinstance(value, bool) for value in quality.values()), f"{label}.quality_issues_per_run[{index}] must be all zero")
        require(quality == repeats[index]["quality_issues"], f"{label}.quality_issues_per_run[{index}] differs from repeat metrics")
    input_rows = require_list(report.get("actual_input_tokens_per_request"), f"{label}.actual_input_tokens_per_request")
    output_rows = require_list(report.get("output_tokens_per_request"), f"{label}.output_tokens_per_request")
    itl_rows = require_list(report.get("itl_evidence_per_request"), f"{label}.itl_evidence_per_request")
    require(len(input_rows) == len(output_rows) == len(itl_rows) == 3, f"{label} token/ITL vectors must contain three repeats")
    all_repeats_itl_eligible = True
    for index, (inputs, outputs, request_itl_rows) in enumerate(zip(input_rows, output_rows, itl_rows)):
        require(isinstance(inputs, list) and len(inputs) == 100 and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in inputs), f"{label}.actual_input_tokens_per_request[{index}] invalid")
        require(isinstance(outputs, list) and len(outputs) == 100 and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in outputs), f"{label}.output_tokens_per_request[{index}] invalid")
        require(isinstance(request_itl_rows, list) and len(request_itl_rows) == 100, f"{label}.itl_evidence_per_request[{index}] invalid")
        require(sum(inputs) == repeats[index]["actual_input_tokens"], f"{label} repeat {index + 1} actual input token total mismatch")
        require(sum(outputs) == repeats[index]["output_tokens"], f"{label} repeat {index + 1} output token total mismatch")
        eligibility_counts = {field: 0 for field in ITL_ELIGIBILITY_FIELDS}
        observed_intervals = 0
        for request_index, (output_tokens, raw_itl) in enumerate(zip(outputs, request_itl_rows)):
            eligibility = validate_request_itl_evidence(
                raw_itl,
                output_tokens=output_tokens,
                label=f"{label}.itl_evidence_per_request[{index}][{request_index}]",
            )
            eligibility_counts[eligibility] += 1
            observed_intervals += int(raw_itl["observed_intervals"])
        expected_intervals = sum(max(output_tokens - 1, 0) for output_tokens in outputs)
        repeat = repeats[index]
        eligible_requests = eligibility_counts["eligible"]
        ineligible_requests = 100 - eligible_requests
        require(repeat["itl_eligible_requests"] == eligible_requests, f"{label} repeat {index + 1} ITL eligible count is not derived")
        require(repeat["itl_ineligible_requests"] == ineligible_requests, f"{label} repeat {index + 1} ITL ineligible count is not derived")
        require(repeat["itl_expected_intervals"] == expected_intervals, f"{label} repeat {index + 1} ITL expected intervals are not derived")
        require(repeat["itl_observed_intervals"] == observed_intervals, f"{label} repeat {index + 1} ITL observed intervals are not derived")
        require(repeat["itl_eligibility_counts"] == eligibility_counts, f"{label} repeat {index + 1} ITL eligibility counters are not derived")
        all_repeats_itl_eligible = all_repeats_itl_eligible and ineligible_requests == 0
    flattened_inputs = [value for row in input_rows for value in row]
    input_stats = require_object(report.get("actual_input_tokens"), f"{label}.actual_input_tokens")
    require(input_stats.get("requested") == report["n_prompt"], f"{label}.actual_input_tokens.requested mismatch")
    require(input_stats.get("min") == min(flattened_inputs), f"{label}.actual_input_tokens.min mismatch")
    require(input_stats.get("max") == max(flattened_inputs), f"{label}.actual_input_tokens.max mismatch")
    require(close_enough(require_number(input_stats.get("mean"), f"{label}.actual_input_tokens.mean", positive=True), sum(flattened_inputs) / len(flattened_inputs)), f"{label}.actual_input_tokens.mean mismatch")
    for metric in HTTP_SCALAR_METRICS:
        validate_scalar_stats(report.get(metric), [float(row[metric]) for row in repeats], f"{label}.{metric}")
    for metric in HTTP_ALWAYS_AVAILABLE_LATENCY_METRICS:
        metric_set = require_object(report.get(metric), f"{label}.{metric}")
        require(set(metric_set) == set(HTTP_PERCENTILES), f"{label}.{metric} percentile set mismatch")
        for percentile in HTTP_PERCENTILES:
            validate_scalar_stats(metric_set.get(percentile), [float(row[metric][percentile]) for row in repeats], f"{label}.{metric}.{percentile}")
    itl_metric_set = require_object(report.get("itl_ms"), f"{label}.itl_ms")
    require(set(itl_metric_set) == set(HTTP_PERCENTILES), f"{label}.itl_ms percentile set mismatch")
    for percentile in HTTP_PERCENTILES:
        samples = (
            [float(row["itl_ms"][percentile]) for row in repeats]
            if all_repeats_itl_eligible
            else [0.0] * len(repeats)
        )
        validate_scalar_stats(itl_metric_set.get(percentile), samples, f"{label}.itl_ms.{percentile}")
    slo = require_object(report.get("slo"), f"{label}.slo")
    for field in ("ttft_p99_ms", "tpot_p99_ms", "e2e_p99_ms"):
        require_number(slo.get(field), f"{label}.slo.{field}", positive=True)
    env = require_object(report.get("env"), f"{label}.env")
    require(env.get("commit_sha") == client["source_git_sha"], f"{label}.env.commit_sha must identify benchmark client")
    require(env.get("hw_id") == session["hardware_id"], f"{label}.env.hw_id mismatch")
    require(isinstance(env.get("ferrum_features"), list), f"{label}.env.ferrum_features must be a list")
    require(env.get("ferrum_env") == {}, f"{label}.env.ferrum_env must not contain hidden benchmark-client switches")
    env_hash = require_string(report.get("env_hash"), f"{label}.env_hash")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", env_hash) is not None, f"{label}.env_hash must be sha256:<digest>")
    expected_env_hash = recompute_bench_env_hash(env, f"{label}.env")
    require(env_hash == expected_env_hash, f"{label}.env_hash is not derived from env")
    return sum(flattened_inputs) / len(flattened_inputs)


def validate_http_implementation(
    root: Path,
    raw: Any,
    label: str,
    *,
    implementation: str,
    sessions: dict[str, dict[str, Any]],
    client: dict[str, Any],
    workload: dict[str, Any],
    backend: str,
    dataset: str,
    concurrency: int,
    typed_active_cap: int,
    hardware_memory_bytes: int,
    global_sample_ids: set[str],
) -> dict[str, float]:
    impl = require_object(raw, label)
    require("samples" not in impl, f"{label}.samples hand-filled summaries are forbidden; use raw repeat sidecars")
    reports = require_list(impl.get("reports"), f"{label}.reports")
    owned_sessions = {
        session_id: session
        for session_id, session in sessions.items()
        if session["implementation"] == implementation
    }
    require(len(reports) == len(owned_sessions) == 4, f"{label} must contain four independent outer-session reports")
    seen_sessions: set[str] = set()
    measured = completed = warmup = warmup_completed = 0
    actual_input_means: list[float] = []
    for index, raw_record in enumerate(reports):
        record = require_object(raw_record, f"{label}.reports[{index}]")
        session_id = require_string(record.get("session_id"), f"{label}.reports[{index}].session_id")
        require(session_id in owned_sessions and session_id not in seen_sessions, f"{label} duplicate or wrong session_id {session_id}")
        seen_sessions.add(session_id)
        session = owned_sessions[session_id]
        require(record.get("slot") == session["slot"], f"{label}.reports[{index}].slot mismatch")
        cell_id = f"{dataset}:c{concurrency}"
        require(record.get("cell_id") == cell_id, f"{label}.reports[{index}].cell_id mismatch")
        require(record.get("dataset") == dataset, f"{label}.reports[{index}].dataset mismatch")
        require(record.get("concurrency") == concurrency, f"{label}.reports[{index}].concurrency mismatch")
        require(record.get("benchmark_client_binary_sha256") == client["binary_sha256"], f"{label}.reports[{index}] benchmark client binary mismatch")
        raw_report_origin = require_string(record.get("raw_report_origin_path"), f"{label}.reports[{index}].raw_report_origin_path")
        expected_slo = validate_bench_argv(
            record.get("bench_argv"),
            f"{label}.reports[{index}].bench_argv",
            client=client,
            workload=workload,
            session=session,
            backend=backend,
            dataset=dataset,
            concurrency=concurrency,
            raw_report_origin_path=raw_report_origin,
        )
        require_object(record.get("env"), f"{label}.reports[{index}].env")
        hidden = sorted(key for key in record["env"] if str(key).startswith("FERRUM_"))
        require(not hidden, f"{label}.reports[{index}].env uses hidden FERRUM_* switches: {hidden}")
        require(record.get("returncode") == 0, f"{label}.reports[{index}].returncode must be zero")
        validate_execution_window(record, f"{label}.reports[{index}]")
        command_started = parse_timestamp(record.get("started_at"), f"{label}.reports[{index}].started_at")
        command_finished = parse_timestamp(record.get("finished_at"), f"{label}.reports[{index}].finished_at")
        session_started = parse_timestamp(session.get("started_at"), f"{label}.reports[{index}].session.started_at")
        session_finished = parse_timestamp(session.get("finished_at"), f"{label}.reports[{index}].session.finished_at")
        require(session_started <= command_started < command_finished <= session_finished, f"{label}.reports[{index}] is outside its server session")
        cell_window = session["_cell_windows"][cell_id]
        require(
            command_started == cell_window["_started_at"]
            and command_finished == cell_window["_finished_at"],
            f"{label}.reports[{index}] command/report window must equal its session measurement window",
        )
        require_artifact_sha(root, record.get("stdout"), record.get("stdout_sha256"), f"{label}.reports[{index}].stdout")
        require_artifact_sha(root, record.get("stderr"), record.get("stderr_sha256"), f"{label}.reports[{index}].stderr")
        report_path = require_artifact_sha(root, record.get("raw_report"), record.get("raw_report_sha256"), f"{label}.reports[{index}].raw_report")
        report = read_json(report_path)
        require(report.get("slo") == expected_slo, f"{label}.reports[{index}].raw_report SLO differs from --goodput")
        require("raw_repeats" not in record and "raw_repeats_sha256" not in record, f"{label}.reports[{index}] second-source repeat sidecars are forbidden")
        repeats_raw = require_list(report.get("repeat_metrics"), f"{label}.reports[{index}].raw_report.repeat_metrics")
        require(len(repeats_raw) == 3, f"{label}.reports[{index}] canonical BenchReport must contain three repeat metrics")
        repeats = [
            validate_repeat_row(
                row,
                f"{label}.reports[{index}].raw_report.repeat_metrics[{repeat_index}]",
                expected_repeat=repeat_index + 1,
            )
            for repeat_index, row in enumerate(repeats_raw)
        ]
        for repeat in repeats:
            sample_id = f"{session_id}:{dataset}:c{concurrency}:r{repeat['repeat']}"
            require(sample_id not in global_sample_ids, f"duplicate HTTP sample identity {sample_id}")
            global_sample_ids.add(sample_id)
        validate_resource_metrics(
            root,
            record.get("resources"),
            f"{label}.reports[{index}].resources",
            backend=backend,
            hardware_memory_bytes=hardware_memory_bytes,
            requested_concurrency=concurrency,
            typed_active_cap=typed_active_cap,
            memory_budget_bytes=int(session["_identity"]["memory_budget_bytes"]),
            session=session,
            cell_id=cell_id,
            measurement_started_at=str(record["started_at"]),
            measurement_finished_at=str(record["finished_at"]),
            runtime_log_origin_path=str(session["runtime_log_origin_path"]),
        )
        actual_input_means.append(
            validate_raw_bench_report(
                report,
                repeats,
                f"{label}.reports[{index}].raw_report",
                client=client,
                workload=workload,
                session=session,
                backend=backend,
                dataset=dataset,
                concurrency=concurrency,
            )
        )
        measured += sum(row["completed_requests"] + row["errored_requests"] for row in repeats)
        completed += sum(row["completed_requests"] for row in repeats)
        warmup += sum(row["warmup_expected"] for row in repeats)
        warmup_completed += sum(row["warmup_completed"] for row in repeats)
    require(seen_sessions == set(owned_sessions), f"{label} did not use every owned outer session exactly once")
    require(impl.get("measured_requests") == measured == 1200, f"{label}.measured_requests must be 1200")
    require(impl.get("completed_requests") == completed == 1200, f"{label}.completed_requests must be 1200")
    require(impl.get("warmup_requests") == warmup == 120, f"{label}.warmup_requests must be 120")
    require(impl.get("warmup_completed") == warmup_completed == 120, f"{label}.warmup_completed must be 120")
    require(impl.get("error_count") == 0, f"{label}.error_count must be zero")
    require(impl.get("bad_output_count") == 0, f"{label}.bad_output_count must be zero")
    require(impl.get("output_token_count_source") == "usage", f"{label}.output_token_count_source must be usage")
    return {"actual_input_token_mean": sum(actual_input_means) / len(actual_input_means)}


def validate_run_workload(root: Path, raw: Any, label: str, *, model: dict[str, Any], backend: str) -> tuple[dict[str, Any], str]:
    workload = require_object(raw, label)
    prompt_path = require_artifact_sha(root, workload.get("prompt_artifact"), workload.get("prompt_sha256"), f"{label}.prompt_artifact")
    prompt_data = read_json(prompt_path)
    prompt = require_string(prompt_data.get("prompt"), f"{label}.prompt_artifact.prompt")
    tokenizer_digest = require_sha256(workload.get("tokenizer_sha256"), f"{label}.tokenizer_sha256")
    require(tokenizer_digest in tokenizer_sha_set(model, backend), f"{label}.tokenizer_sha256 is not locked by models.lock")
    tokenizer = require_artifact_sha(root, workload.get("tokenizer_artifact"), tokenizer_digest, f"{label}.tokenizer_artifact")
    require(tokenizer.name == "tokenizer.json", f"{label}.tokenizer_artifact must be tokenizer.json")
    tokenizer_source = locked_tokenizer_source(model, backend)
    require(workload.get("tokenizer_id") == tokenizer_source["repo"], f"{label}.tokenizer_id mismatch")
    require(workload.get("tokenizer_revision") == tokenizer_source["revision"], f"{label}.tokenizer_revision mismatch")
    require_string(workload.get("tokenizer_origin_path"), f"{label}.tokenizer_origin_path")
    config_path = require_artifact_sha(root, workload.get("effective_config"), workload.get("effective_config_sha256"), f"{label}.effective_config")
    config = read_json(config_path)
    required_config = {
        "schema_version": COLLECTOR_SCHEMA_VERSION,
        "model_revision": model["lanes"][backend]["revision"],
        "prompt_sha256": workload["prompt_sha256"],
        "tokenizer_sha256": tokenizer_digest,
        "seed": 9271,
        "max_output_tokens": 128,
        "enable_thinking": False,
        "temperature": 0.0,
        "eos_policy": "model-metadata",
        "backend": backend,
    }
    for field, expected in required_config.items():
        require(config.get(field) == expected, f"{label}.effective_config.{field} mismatch")
    for field in ("top_k", "top_p", "repeat_penalty"):
        require_number(config.get(field), f"{label}.effective_config.{field}", nonnegative=True)
    require_string(config.get("request_model"), f"{label}.effective_config.request_model")
    workload["_config"] = config
    return workload, prompt


def validate_run_stdout(path: Path, label: str) -> tuple[int, float, float]:
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BaselineError(f"{label}:{line_no} invalid run JSONL: {exc}") from exc
        require(isinstance(event, dict), f"{label}:{line_no} run JSONL event must be an object")
        events.append(event)
    assistant = [event for event in events if event.get("event") == "assistant"]
    require(len(assistant) == 1, f"{label} must contain exactly one assistant event")
    row = assistant[0]
    output_tokens = require_positive_int(row.get("n_tokens"), f"{label} assistant n_tokens")
    require(output_tokens <= 128, f"{label} assistant n_tokens exceeds max_tokens")
    content = require_string(row.get("content"), f"{label} assistant content")
    forbidden = ("<unk>", "[PAD", "<pad>", "<|reserved_special_token", "\ufffd", "Ã©", "Â©", "â€")
    require(not any(marker in content for marker in forbidden), f"{label} assistant content contains reserved-token/UTF-8 corruption")
    require(row.get("finish_reason") in {"stop", "eos", "length"}, f"{label} finish_reason must be stop, eos, or length")
    require(row.get("chunk_count") == 1, f"{label} one-shot JSONL must report chunk_count=1")
    elapsed_ms = require_number(row.get("ms"), f"{label} assistant ms", positive=True)
    output_tps = output_tokens * 1000.0 / elapsed_ms
    return output_tokens, elapsed_ms, output_tps


def validate_product_effective_config(root: Path, sample: dict[str, Any], label: str) -> tuple[str, str]:
    config_path = require_artifact_sha(
        root,
        sample.get("product_effective_config"),
        sample.get("product_effective_config_sha256"),
        f"{label}.product_effective_config",
    )
    document = read_json(config_path)
    require(document.get("schema_version") == 1, f"{label}.product_effective_config schema mismatch")
    for field, kind in (
        ("entries", list),
        ("model_capabilities", dict),
        ("hardware_capabilities", dict),
        ("workload_profile", dict),
        ("decisions", list),
    ):
        require(isinstance(document.get(field), kind), f"{label}.product_effective_config.{field} has wrong type")
    origin = require_string(
        sample.get("product_effective_config_origin_path"),
        f"{label}.product_effective_config_origin_path",
    )
    return config_path.relative_to(root.resolve()).as_posix(), origin


def validate_run_legacy(
    root: Path,
    raw: Any,
    label: str,
    *,
    model: dict[str, Any],
    backend: str,
    binary_sha256: str,
    hardware: dict[str, Any],
    hardware_id: str,
    hardware_fingerprint: str,
) -> None:
    run = require_object(raw, label)
    require(run.get("comparison_id") == "g00-run-legacy", f"{label}.comparison_id must be g00-run-legacy")
    workload, prompt = validate_run_workload(root, run.get("workload"), f"{label}.workload", model=model, backend=backend)
    require(
        require_positive_int(
            workload["_config"].get("memory_budget_bytes"),
            f"{label}.workload.effective_config.memory_budget_bytes",
        )
        <= int(hardware["memory_bytes"]),
        f"{label}.workload.effective_config memory budget exceeds hardware",
    )
    sessions_raw = require_list(run.get("sessions"), f"{label}.sessions")
    expected_slots = {slot for slot, owner in enumerate(SLOT_ORDER, start=1) if owner == "B"}
    require(len(sessions_raw) == 4, f"{label}.sessions must contain four outer sessions")
    sessions: dict[str, dict[str, Any]] = {}
    seen_slots: set[int] = set()
    by_sequence: dict[int, dict[str, Any]] = {}
    for index, raw_session in enumerate(sessions_raw):
        session = require_object(raw_session, f"{label}.sessions[{index}]")
        session_id = require_string(session.get("session_id"), f"{label}.sessions[{index}].session_id")
        require(session_id not in sessions, f"{label} duplicate run session_id {session_id}")
        slot = require_positive_int(session.get("slot"), f"{label}.sessions[{index}].slot")
        require(slot in expected_slots and slot not in seen_slots, f"{label} duplicate or invalid B slot {slot}")
        seen_slots.add(slot)
        sequence = require_positive_int(session.get("sequence"), f"{label}.sessions[{index}].sequence")
        require(sequence == slot and sequence not in by_sequence, f"{label}.sessions[{index}].sequence must bind the B slot")
        by_sequence[sequence] = session
        require(session.get("hardware_id") == hardware_id, f"{label}.sessions[{index}].hardware_id mismatch")
        require(session.get("hardware_fingerprint") == hardware_fingerprint, f"{label}.sessions[{index}].hardware_fingerprint mismatch")
        validate_execution_window(session, f"{label}.sessions[{index}]")
        session_started = parse_timestamp(session.get("started_at"), f"{label}.sessions[{index}].started_at")
        session_finished = parse_timestamp(session.get("finished_at"), f"{label}.sessions[{index}].finished_at")
        windows_raw = require_list(session.get("sample_windows"), f"{label}.sessions[{index}].sample_windows")
        require(len(windows_raw) == 3, f"{label}.sessions[{index}].sample_windows must contain three repeats")
        windows: dict[int, dict[str, Any]] = {}
        previous_finished: datetime | None = None
        for repeat_index, raw_window in enumerate(windows_raw, start=1):
            window = require_object(raw_window, f"{label}.sessions[{index}].sample_windows[{repeat_index - 1}]")
            require(window.get("repeat") == repeat_index, f"{label}.sessions[{index}].sample_windows repeat order mismatch")
            require(
                window.get("sample_id") == f"run-{model['key']}-{backend}-s{slot}-r{repeat_index}",
                f"{label}.sessions[{index}].sample_windows sample_id mismatch",
            )
            window_started = parse_timestamp(window.get("started_at"), f"{label}.sessions[{index}].sample_windows[{repeat_index - 1}].started_at")
            window_finished = parse_timestamp(window.get("finished_at"), f"{label}.sessions[{index}].sample_windows[{repeat_index - 1}].finished_at")
            require(session_started < window_started < window_finished < session_finished, f"{label}.sessions[{index}].sample_windows is outside session")
            if previous_finished is not None:
                require(previous_finished <= window_started, f"{label}.sessions[{index}].sample_windows overlap")
            previous_finished = window_finished
            window["_started_at"] = window_started
            window["_finished_at"] = window_finished
            windows[repeat_index] = window
        require(
            parse_timestamp(session.get("measurement_started_at"), f"{label}.sessions[{index}].measurement_started_at")
            == windows[1]["_started_at"]
            and parse_timestamp(session.get("measurement_finished_at"), f"{label}.sessions[{index}].measurement_finished_at")
            == windows[3]["_finished_at"],
            f"{label}.sessions[{index}] measurement bounds must cover its sample windows",
        )
        session["_started_at"] = session_started
        session["_finished_at"] = session_finished
        session["_sample_windows"] = windows
        sessions[session_id] = session
    require(seen_slots == expected_slots, f"{label}.sessions must cover every B slot")
    ordered_sessions = [by_sequence[key] for key in sorted(by_sequence)]
    for previous, current in zip(ordered_sessions, ordered_sessions[1:]):
        require(
            previous["_finished_at"] <= current["_started_at"],
            f"{label}.sessions overlap globally between slots {previous['slot']} and {current['slot']}",
        )
    samples = require_list(run.get("samples"), f"{label}.samples")
    require(len(samples) == 12, f"{label}.samples must contain 12 real ferrum run commands")
    seen_pairs: set[tuple[int, int]] = set()
    sample_ids: set[str] = set()
    product_config_paths: set[str] = set()
    product_config_shas: set[str] = set()
    e2e_ms_values: list[float] = []
    e2e_tps_values: list[float] = []
    for index, raw_sample in enumerate(samples):
        sample = require_object(raw_sample, f"{label}.samples[{index}]")
        sample_id = require_string(sample.get("sample_id"), f"{label}.samples[{index}].sample_id")
        require(sample_id not in sample_ids, f"{label} duplicate sample_id {sample_id}")
        sample_ids.add(sample_id)
        session_id = require_string(sample.get("session_id"), f"{label}.samples[{index}].session_id")
        require(session_id in sessions, f"{label}.samples[{index}] unknown session_id")
        slot = sessions[session_id]["slot"]
        require(sample.get("slot") == slot, f"{label}.samples[{index}].slot mismatch")
        repeat = require_positive_int(sample.get("repeat"), f"{label}.samples[{index}].repeat")
        pair = (slot, repeat)
        require(1 <= repeat <= 3 and pair not in seen_pairs, f"{label} duplicate or invalid slot/repeat {pair}")
        seen_pairs.add(pair)
        sample_window = sessions[session_id]["_sample_windows"][repeat]
        require(sample.get("binary_sha256") == binary_sha256, f"{label}.samples[{index}].binary_sha256 mismatch")
        require(sample.get("hardware_id") == hardware_id, f"{label}.samples[{index}].hardware_id mismatch")
        require(sample.get("hardware_fingerprint") == hardware_fingerprint, f"{label}.samples[{index}].hardware_fingerprint mismatch")
        require(sample.get("effective_config") == workload["effective_config"], f"{label}.samples[{index}].effective_config mismatch")
        require(sample.get("effective_config_sha256") == workload["effective_config_sha256"], f"{label}.samples[{index}].effective_config_sha256 mismatch")
        require(sample.get("prompt_sha256") == workload["prompt_sha256"], f"{label}.samples[{index}].prompt_sha256 mismatch")
        require(sample.get("tokenizer_sha256") == workload["tokenizer_sha256"], f"{label}.samples[{index}].tokenizer_sha256 mismatch")
        argv, options, switches = parse_argv(sample.get("argv"), f"{label}.samples[{index}].argv")
        require(Path(argv[0]).name == "ferrum" and "run" in argv[1:], f"{label}.samples[{index}].argv must execute ferrum run")
        run_index = argv.index("run")
        require(run_index + 1 < len(argv) and argv[run_index + 1] == workload["_config"]["request_model"], f"{label}.samples[{index}].argv model mismatch")
        require_option(options, "--prompt", prompt, f"{label}.samples[{index}].argv")
        require_option(options, "--tokenizer", workload["tokenizer_origin_path"], f"{label}.samples[{index}].argv")
        require_option(options, "--max-tokens", 128, f"{label}.samples[{index}].argv")
        require_option(options, "--seed", 9271, f"{label}.samples[{index}].argv")
        require_option(options, "--temperature", workload["_config"]["temperature"], f"{label}.samples[{index}].argv")
        require_option(options, "--top-k", workload["_config"]["top_k"], f"{label}.samples[{index}].argv")
        require_option(options, "--top-p", workload["_config"]["top_p"], f"{label}.samples[{index}].argv")
        require_option(options, "--repeat-penalty", workload["_config"]["repeat_penalty"], f"{label}.samples[{index}].argv")
        require_option(options, "--backend", backend, f"{label}.samples[{index}].argv")
        require_option(options, "--output-format", "jsonl", f"{label}.samples[{index}].argv")
        product_config_path, product_config_origin = validate_product_effective_config(
            root, sample, f"{label}.samples[{index}]"
        )
        require_option(options, "--effective-config-json", product_config_origin, f"{label}.samples[{index}].argv")
        require(product_config_path not in product_config_paths, f"{label}.samples[{index}] must use a unique effective-config artifact")
        product_config_paths.add(product_config_path)
        product_config_shas.add(str(sample["product_effective_config_sha256"]))
        require("--disable-thinking" in switches and "--enable-thinking" not in switches, f"{label}.samples[{index}].argv must explicitly disable thinking")
        env = require_object(sample.get("env"), f"{label}.samples[{index}].env")
        hidden = sorted(key for key in env if str(key).startswith("FERRUM_"))
        require(not hidden, f"{label}.samples[{index}].env uses hidden FERRUM_* switches: {hidden}")
        require(sample.get("returncode") == 0, f"{label}.samples[{index}].returncode must be zero")
        validate_execution_window(sample, f"{label}.samples[{index}]")
        sample_started = parse_timestamp(sample.get("started_at"), f"{label}.samples[{index}].started_at")
        sample_finished = parse_timestamp(sample.get("finished_at"), f"{label}.samples[{index}].finished_at")
        require(
            sample_started == sample_window["_started_at"]
            and sample_finished == sample_window["_finished_at"],
            f"{label}.samples[{index}] command window does not match its outer slot sample window",
        )
        measurement_started = parse_timestamp(
            sample.get("measurement_started_at"), f"{label}.samples[{index}].measurement_started_at"
        )
        measurement_finished = parse_timestamp(
            sample.get("measurement_finished_at"), f"{label}.samples[{index}].measurement_finished_at"
        )
        require(
            sample_started < measurement_started < measurement_finished < sample_finished,
            f"{label}.samples[{index}] resource measurement window must be inside command execution",
        )
        pid = require_positive_int(sample.get("pid"), f"{label}.samples[{index}].pid")
        pgid = require_positive_int(sample.get("pgid"), f"{label}.samples[{index}].pgid")
        process_marker = require_string(sample.get("process_start_marker"), f"{label}.samples[{index}].process_start_marker")
        try:
            derived_process_marker = resource_sampler.process_marker_from_source(
                pid, sample.get("process_start_source")
            )
        except resource_sampler.ResourceEvidenceError as exc:
            raise BaselineError(f"{label}.samples[{index}].process_start_source rejected: {exc}") from exc
        require(process_marker == derived_process_marker, f"{label}.samples[{index}].process_start_marker is not derived from raw OS identity evidence")
        stdout = require_artifact_sha(root, sample.get("stdout"), sample.get("stdout_sha256"), f"{label}.samples[{index}].stdout")
        stderr = require_artifact_sha(root, sample.get("stderr"), sample.get("stderr_sha256"), f"{label}.samples[{index}].stderr")
        stderr_text = stderr.read_text(encoding="utf-8", errors="replace").lower()
        require(not any(marker in stderr_text for marker in ("panicked", "segmentation fault", "traceback", "out of memory")), f"{label}.samples[{index}].stderr contains a runtime blocker")
        require("token_timing" not in sample and "token_timing_sha256" not in sample, f"{label}.samples[{index}] impossible token timing sidecar is forbidden")
        output_tokens, inference_ms, output_tps = validate_run_stdout(stdout, f"{label}.samples[{index}].stdout")
        require(sample.get("output_tokens") == output_tokens, f"{label}.samples[{index}].output_tokens mismatch")
        require(close_enough(require_number(sample.get("legacy_inference_e2e_ms"), f"{label}.samples[{index}].legacy_inference_e2e_ms", positive=True), inference_ms), f"{label}.samples[{index}].legacy_inference_e2e_ms mismatch")
        require(close_enough(require_number(sample.get("legacy_inference_e2e_output_tps"), f"{label}.samples[{index}].legacy_inference_e2e_output_tps", positive=True), output_tps), f"{label}.samples[{index}].legacy_inference_e2e_output_tps mismatch")
        require(sample.get("cold_process_first_request") is True, f"{label}.samples[{index}] must identify the cold-process first-request boundary")
        e2e_ms_values.append(inference_ms)
        e2e_tps_values.append(output_tps)
        resource_session = {
            "session_id": sample_id,
            "hardware_id": hardware_id,
            "base_url": f"process://{sample_id}",
            "started_at": sample["started_at"],
            "finished_at": sample["finished_at"],
            "_pid": pid,
            "_pgid": pgid,
            "process_start_marker": sample["process_start_marker"],
        }
        runtime_log_origin = require_string(
            sample.get("stderr_origin_path"), f"{label}.samples[{index}].stderr_origin_path"
        )
        validate_resource_metrics(
            root,
            sample.get("resources"),
            f"{label}.samples[{index}].resources",
            backend=backend,
            hardware_memory_bytes=int(hardware["memory_bytes"]),
            requested_concurrency=1,
            typed_active_cap=1,
            memory_budget_bytes=require_positive_int(
                workload["_config"].get("memory_budget_bytes"),
                f"{label}.workload.effective_config.memory_budget_bytes",
            ),
            session=resource_session,
            cell_id="run:c1",
            measurement_started_at=str(sample["measurement_started_at"]),
            measurement_finished_at=str(sample["measurement_finished_at"]),
            runtime_log_origin_path=runtime_log_origin,
            allow_process_probe=True,
        )
    expected_pairs = {(slot, repeat) for slot in expected_slots for repeat in range(1, 4)}
    require(seen_pairs == expected_pairs, f"{label}.samples do not cover all B slot/repeat pairs")
    require(run.get("measured_samples") == 12 and run.get("completed_samples") == 12, f"{label} measured/completed samples must be 12/12")
    require(run.get("error_count") == 0, f"{label}.error_count must be zero")
    require(run.get("output_token_count_source") == "generated_tokens", f"{label}.output_token_count_source must be generated_tokens")
    require(run.get("metric_boundary") == "engine.infer_e2e", f"{label}.metric_boundary must be engine.infer_e2e")
    require(len(product_config_paths) == 12 and len(product_config_shas) == 1, f"{label} product effective configs must be unique artifacts with identical content")
    summary = require_object(run.get("summary"), f"{label}.summary")
    for metric, values in (
        ("legacy_inference_e2e_ms", e2e_ms_values),
        ("legacy_inference_e2e_output_tps", e2e_tps_values),
    ):
        stats = require_object(summary.get(metric), f"{label}.summary.{metric}")
        require(close_enough(require_number(stats.get("median"), f"{label}.summary.{metric}.median", positive=True), percentile_linear(values, 0.5)), f"{label}.summary.{metric}.median mismatch")
        require(close_enough(require_number(stats.get("p95"), f"{label}.summary.{metric}.p95", positive=True), percentile_linear(values, 0.95)), f"{label}.summary.{metric}.p95 mismatch")


def validate_external_summary(
    root: Path,
    *,
    model: dict[str, Any],
    backend: str,
    hardware: dict[str, Any],
    allow_synthetic: bool,
) -> dict[str, Any]:
    path = root / "external-baselines" / model["key"] / backend / "summary.json"
    data = read_json(path)
    label = f"external-baselines.{model['key']}.{backend}"
    require_schema(data, label)
    require(data.get("status") == "pass", f"{label}.status must be pass")
    require(data.get("model_key") == model["key"], f"{label}.model_key mismatch")
    require(data.get("backend") == backend, f"{label}.backend mismatch")
    hardware_id = model["lanes"][backend]["hardware_id"]
    require(data.get("hardware_id") == hardware_id, f"{label}.hardware_id cross-hardware mismatch")
    require(data.get("hardware_fingerprint") == hardware["fingerprint"], f"{label}.hardware_fingerprint mismatch")
    require(data.get("model_revision") == model["lanes"][backend]["revision"], f"{label}.model_revision mismatch")
    require(data.get("model_files") == locked_file_map(model, backend), f"{label}.model_files mismatch")
    require_no_forbidden_markers(data, label)
    client = validate_benchmark_client(
        root,
        data.get("benchmark_client"),
        f"{label}.benchmark_client",
        allow_synthetic=allow_synthetic,
    )
    identity, cap = validate_server_identity(
        root,
        data.get("server_identity"),
        f"{label}.server_identity",
        implementation="A",
        backend=backend,
        expected_binary_sha256=None,
        model=model,
        hardware_memory_bytes=int(hardware["memory_bytes"]),
    )
    sessions = validate_server_sessions(
        root,
        data.get("sessions"),
        f"{label}.sessions",
        identities={"A": (identity, cap)},
        hardware_id=hardware_id,
        hardware_fingerprint=hardware["fingerprint"],
        require_abba=False,
        backend=backend,
        model=model,
    )
    require_log(root, data.get("command_log"), f"{label}.command_log")
    require_log(root, data.get("runtime_log"), f"{label}.runtime_log")
    cells = require_list(data.get("cells"), f"{label}.cells")
    seen: set[tuple[str, int]] = set()
    global_sample_ids: set[str] = set()
    for index, raw in enumerate(cells):
        cell = require_object(raw, f"{label}.cells[{index}]")
        key = (str(cell.get("dataset")), cell.get("concurrency"))
        require(key not in seen, f"{label} duplicate cell {key}")
        seen.add(key)
        dataset, concurrency = key
        require((dataset, concurrency) in expected_cells(backend), f"{label}.cells[{index}] unexpected cell {key}")
        workload = validate_workload_identity(
            root,
            cell.get("workload"),
            f"{label}.cells[{index}].workload",
            model=model,
            backend=backend,
            dataset=dataset,
            concurrency=int(concurrency),
            typed_active_cap=cap,
        )
        validate_http_implementation(
            root,
            cell.get("implementation"),
            f"{label}.cells[{index}].implementation",
            implementation="A",
            sessions=sessions,
            client=client,
            workload=workload,
            backend=backend,
            dataset=dataset,
            concurrency=int(concurrency),
            typed_active_cap=cap,
            hardware_memory_bytes=int(hardware["memory_bytes"]),
            global_sample_ids=global_sample_ids,
        )
    require(seen == expected_cells(backend), f"{label} required cell matrix mismatch")
    data["_validated_client"] = client
    data["_validated_identity"] = identity
    data["_validated_cap"] = cap
    data["_validated_sessions"] = sessions
    return data


def validate_performance_lane(
    root: Path,
    *,
    model: dict[str, Any],
    backend: str,
    correctness_status: str,
    hardware: dict[str, dict[str, Any]],
    binaries: dict[str, str],
    allow_synthetic: bool,
) -> list[dict[str, Any]]:
    path = root / "performance" / model["key"] / backend / "summary.json"
    data = read_json(path)
    label = f"performance.{model['key']}.{backend}"
    validate_lane_identity(data, label=label, model=model, backend=backend, hardware=hardware, binaries=binaries)
    expected_status = "pass" if correctness_status == "pass" else "blocked"
    require(data.get("status") == expected_status, f"{label}.status must be {expected_status}")
    hardware_item = hardware[model["lanes"][backend]["hardware_id"]]
    external = validate_external_summary(
        root,
        model=model,
        backend=backend,
        hardware=hardware_item,
        allow_synthetic=allow_synthetic,
    )
    if expected_status == "blocked":
        require(data.get("comparable") is False, f"{label}.comparable must be false")
        require_string(data.get("reason"), f"{label}.reason")
        require_string(data.get("downstream_goal"), f"{label}.downstream_goal")
        forbidden = {"ratio", "throughput_tok_s", "cells", "run_legacy", "comparison_id", "slot_order"} & set(data)
        require(not forbidden, f"{label} blocked performance contains fabricated fields: {sorted(forbidden)}")
        return list(external["_validated_sessions"].values())
    require(data.get("comparable") is True, f"{label}.comparable must be true")
    require(data.get("hardware_fingerprint") == hardware_item["fingerprint"], f"{label}.hardware_fingerprint mismatch")
    require(data.get("slot_order") == SLOT_ORDER, f"{label}.slot_order must be ABBA-BAAB")
    comparison_id = require_string(data.get("comparison_id"), f"{label}.comparison_id")
    require(comparison_id == "g00-legacy-external", f"{label}.comparison_id must be g00-legacy-external")
    require_log(root, data.get("command_log"), f"{label}.command_log")
    require_log(root, data.get("runtime_log"), f"{label}.runtime_log")
    client = validate_benchmark_client(
        root,
        data.get("benchmark_client"),
        f"{label}.benchmark_client",
        allow_synthetic=allow_synthetic,
    )
    require(client == external["_validated_client"], f"{label}.benchmark_client differs from external baseline client")
    implementations = require_object(data.get("implementations"), f"{label}.implementations")
    require(set(implementations) == {"A", "B"}, f"{label}.implementations must contain A and B")
    identity_a, cap_a = validate_server_identity(
        root,
        implementations["A"],
        f"{label}.implementations.A",
        implementation="A",
        backend=backend,
        expected_binary_sha256=str(external["_validated_identity"]["binary_sha256"]),
        model=model,
        hardware_memory_bytes=int(hardware_item["memory_bytes"]),
    )
    require(identity_a == external["_validated_identity"], f"{label}.implementations.A differs from external baseline identity")
    identity_b, cap_b = validate_server_identity(
        root,
        implementations["B"],
        f"{label}.implementations.B",
        implementation="B",
        backend=backend,
        expected_binary_sha256=binaries[backend],
        model=model,
        hardware_memory_bytes=int(hardware_item["memory_bytes"]),
    )
    require(cap_a == cap_b, f"{label} A/B typed active caps must match")
    if model["key"] == "m2-qwen35-35b-a3b" and backend == "cuda":
        require(cap_b >= 16, f"{label} M2 CUDA typed active cap must be at least 16")
    sessions = validate_server_sessions(
        root,
        data.get("sessions"),
        f"{label}.sessions",
        identities={"A": (identity_a, cap_a), "B": (identity_b, cap_b)},
        hardware_id=str(data["hardware_id"]),
        hardware_fingerprint=str(hardware_item["fingerprint"]),
        require_abba=True,
        backend=backend,
        model=model,
    )
    cells = require_list(data.get("cells"), f"{label}.cells")
    seen: set[tuple[str, int]] = set()
    global_sample_ids: set[str] = set()
    for index, raw in enumerate(cells):
        cell = require_object(raw, f"{label}.cells[{index}]")
        key = (str(cell.get("dataset")), cell.get("concurrency"))
        require(key not in seen, f"{label} duplicate cell {key}")
        seen.add(key)
        dataset, concurrency = key
        require((dataset, concurrency) in expected_cells(backend), f"{label}.cells[{index}] unexpected cell {key}")
        tokenizer_diff = cell.get("tokenizer_input_len_diff_pct")
        require(
            isinstance(tokenizer_diff, (int, float))
            and not isinstance(tokenizer_diff, bool)
            and math.isfinite(tokenizer_diff)
            and 0 <= tokenizer_diff <= 1.0,
            f"{label}.cells[{index}] tokenizer length difference must be numeric and <=1%",
        )
        implementations = require_object(cell.get("implementations"), f"{label}.cells[{index}].implementations")
        require(set(implementations) == {"A", "B"}, f"{label}.cells[{index}] must contain A and B")
        external_cell = next(
            candidate
            for candidate in external["cells"]
            if candidate.get("dataset") == dataset and candidate.get("concurrency") == concurrency
        )
        require(
            implementations["A"] == external_cell["implementation"],
            f"{label}.cells[{index}] external A report evidence differs from standalone baseline",
        )
        workload = validate_workload_identity(
            root,
            cell.get("workload"),
            f"{label}.cells[{index}].workload",
            model=model,
            backend=backend,
            dataset=dataset,
            concurrency=int(concurrency),
            typed_active_cap=cap_a,
        )
        implementation_summaries: dict[str, dict[str, float]] = {}
        for implementation in ("A", "B"):
            implementation_summaries[implementation] = validate_http_implementation(
                root,
                implementations[implementation],
                f"{label}.cells[{index}].implementations.{implementation}",
                implementation=implementation,
                sessions=sessions,
                client=client,
                workload=workload,
                backend=backend,
                dataset=dataset,
                concurrency=int(concurrency),
                typed_active_cap=cap_a,
                hardware_memory_bytes=int(hardware_item["memory_bytes"]),
                global_sample_ids=global_sample_ids,
            )
        a_input = implementation_summaries["A"]["actual_input_token_mean"]
        b_input = implementation_summaries["B"]["actual_input_token_mean"]
        recomputed_diff = abs(b_input - a_input) / a_input * 100.0
        require(close_enough(float(tokenizer_diff), recomputed_diff), f"{label}.cells[{index}] tokenizer length difference is not derived from raw BenchReports")
    require(seen == expected_cells(backend), f"{label} required cell matrix mismatch")
    validate_run_legacy(
        root,
        data.get("run_legacy"),
        f"{label}.run_legacy",
        model=model,
        backend=backend,
        binary_sha256=binaries[backend],
        hardware=hardware_item,
        hardware_id=str(data["hardware_id"]),
        hardware_fingerprint=str(hardware_item["fingerprint"]),
    )
    external_sessions = external["_validated_sessions"]
    for session_id, external_session in external_sessions.items():
        require(session_id in sessions, f"{label} comparable sessions omit external session {session_id}")
        comparable_session = sessions[session_id]
        for field in (
            "implementation",
            "slot",
            "sequence",
            "pid",
            "pgid",
            "process_start_marker",
            "started_at",
            "ready_at",
            "measurement_started_at",
            "measurement_finished_at",
            "shutdown_started_at",
            "finished_at",
            "runtime_log_sha256",
        ):
            require(
                comparable_session.get(field) == external_session.get(field),
                f"{label} external/comparable session {session_id} differs at {field}",
            )
    return list(sessions.values())


def validate_performance(
    root: Path,
    models: dict[str, dict[str, Any]],
    statuses: dict[tuple[str, str], str],
    hardware: dict[str, dict[str, Any]],
    binaries: dict[str, str],
    *,
    allow_synthetic: bool,
) -> None:
    global_sessions: dict[str, dict[str, Any]] = {}
    for model_key in PRIMARY_MODELS:
        for backend in ("cuda", "metal"):
            lane_sessions = validate_performance_lane(
                root,
                model=models[model_key],
                backend=backend,
                correctness_status=statuses[(model_key, backend)],
                hardware=hardware,
                binaries=binaries,
                allow_synthetic=allow_synthetic,
            )
            for session in lane_sessions:
                session_id = str(session["session_id"])
                if session_id in global_sessions:
                    previous = global_sessions[session_id]
                    require(
                        previous["hardware_id"] == session["hardware_id"]
                        and previous["started_at"] == session["started_at"]
                        and previous["finished_at"] == session["finished_at"]
                        and previous["process_start_marker"] == session["process_start_marker"],
                        f"global server session {session_id} has conflicting duplicate evidence",
                    )
                else:
                    global_sessions[session_id] = session
    by_hardware: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in global_sessions.values():
        by_hardware[str(session["hardware_id"])].append(session)
    for hardware_id, sessions in by_hardware.items():
        ordered = sorted(sessions, key=lambda row: parse_timestamp(row["started_at"], "global.session.started_at"))
        for previous, current in zip(ordered, ordered[1:]):
            require(
                parse_timestamp(previous["finished_at"], "global.session.finished_at")
                <= parse_timestamp(current["started_at"], "global.session.started_at"),
                f"global server sessions overlap on {hardware_id}: {previous['session_id']} -> {current['session_id']}",
            )


def build_artifact_ref(root: Path, raw: Any, label: str, *, nonempty: bool = True) -> Path:
    reference = require_object(raw, label)
    path = require_artifact_sha(root, reference.get("path"), reference.get("sha256"), label)
    if nonempty:
        require(path.stat().st_size > 0, f"{label} artifact is empty")
    return path


def validate_cargo_messages(path: Path, raw_summary: Any, label: str) -> dict[str, Any]:
    try:
        expected = build_timing.parse_cargo_messages(path)
    except build_timing.BuildTimingError as exc:
        raise BaselineError(f"{label} is invalid: {exc}") from exc
    require(require_object(raw_summary, f"{label}.summary") == expected, f"{label} summary cannot be recomputed from Cargo messages")
    return expected


def validate_build_record(
    root: Path,
    raw: Any,
    label: str,
    *,
    require_binary_sha: str | None,
    require_output_binary: bool = False,
) -> tuple[float, dict[str, Any], str | None, str]:
    record = require_object(raw, label)
    require(record.get("argv") == CUDA_BUILD_ARGV, f"{label}.argv is not the canonical CUDA release build")
    require(record.get("returncode") == 0, f"{label}.returncode must be zero")
    duration = validate_execution_window(record, label)
    cargo_path = build_artifact_ref(root, record.get("cargo_messages"), f"{label}.cargo_messages")
    summary = validate_cargo_messages(cargo_path, record.get("cargo_summary"), f"{label}.cargo_messages")
    log_path = build_artifact_ref(root, record.get("log"), f"{label}.log")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    require("Finished" in log_text, f"{label}.log lacks Cargo completion evidence")
    timings_path = build_artifact_ref(root, record.get("cargo_timings"), f"{label}.cargo_timings")
    timings_text = timings_path.read_text(encoding="utf-8", errors="replace")
    require("<html" in timings_text.lower() and "cargo" in timings_text.lower(), f"{label}.cargo_timings is not Cargo timing HTML")
    binary_path_value: str | None = None
    if require_output_binary or require_binary_sha is not None:
        binary_ref = require_object(record.get("output_binary"), f"{label}.output_binary")
        binary = build_artifact_ref(root, binary_ref, f"{label}.output_binary")
        if require_binary_sha is not None:
            require(binary_ref.get("sha256") == require_binary_sha, f"{label}.output_binary differs from frozen CUDA binary")
        binary_path_value = binary.relative_to(root.resolve()).as_posix()
    require(record.get("post_git_status", []) == [], f"{label} left a dirty source worktree")
    return duration, summary, binary_path_value, log_text


def validate_build_timings(
    root: Path,
    hardware: dict[str, dict[str, Any]],
    binaries: dict[str, str],
) -> None:
    data = read_json(root / "build-timings" / "summary.json")
    require_schema(data, "build-timings")
    require_source_identity(data, "build-timings")
    collector = require_object(data.get("collector"), "build-timings.collector")
    require(collector.get("path") == BUILD_TIMING_COLLECTOR_PATH.relative_to(REPO_ROOT).as_posix(), "build-timings collector path mismatch")
    require(collector.get("sha256") == file_sha256(BUILD_TIMING_COLLECTOR_PATH), "build-timings collector SHA256 mismatch")
    hardware_id = require_string(data.get("hardware_id"), "build-timings.hardware_id")
    require(hardware_id in hardware and hardware[hardware_id]["backend"] == "cuda", "build-timings must use frozen CUDA hardware")
    require(data.get("hardware_fingerprint") == hardware[hardware_id]["fingerprint"], "build-timings hardware fingerprint mismatch")
    _, prewarm_summary, _, _ = validate_build_record(
        root,
        data.get("prewarm"),
        "build-timings.prewarm",
        require_binary_sha=None,
        require_output_binary=False,
    )
    require(prewarm_summary["compiler_artifact_count"] > 0, "build-timings prewarm did not execute Cargo")
    rows = require_list(data.get("scenarios"), "build-timings.scenarios")
    seen: set[str] = set()
    sample_ids: set[str] = set()
    binary_paths: set[str] = set()
    for index, raw in enumerate(rows):
        row = require_object(raw, f"build-timings.scenarios[{index}]")
        name = require_string(row.get("name"), f"build-timings.scenarios[{index}].name")
        require(name in BUILD_SCENARIOS and name not in seen, f"invalid or duplicate build scenario: {name}")
        seen.add(name)
        require(row.get("command") == CUDA_BUILD_ARGV, f"build-timings.{name}.command is not canonical")
        samples = require_list(row.get("samples"), f"build-timings.{name}.samples")
        require(len(samples) == 5, f"build-timings.{name} must contain exactly five samples")
        expected_setup, expected_input, expected_package = BUILD_SCENARIO_INPUTS[name]
        durations = []
        content_before_shas: set[str] = set()
        for sample_index, sample_raw in enumerate(samples):
            label = f"build-timings.{name}.samples[{sample_index}]"
            sample = require_object(sample_raw, label)
            sample_id = require_string(sample.get("sample_id"), f"{label}.sample_id")
            require(sample_id not in sample_ids, f"duplicate build sample id {sample_id}")
            sample_ids.add(sample_id)
            duration, cargo_summary, binary_path, log_text = validate_build_record(
                root,
                sample,
                label,
                require_binary_sha=binaries["cuda"] if name in {"noop", "clean-release"} else None,
                require_output_binary=True,
            )
            durations.append(duration)
            require(binary_path is not None and binary_path not in binary_paths, f"{label}.output_binary must be a unique copied artifact")
            binary_paths.add(binary_path)
            setup = require_object(sample.get("setup"), f"{label}.setup")
            require(setup.get("kind") == expected_setup, f"{label}.setup.kind mismatch")
            if expected_setup == "none":
                require(set(setup) == {"kind"}, f"{label}.setup contains fabricated no-op evidence")
                require(cargo_summary["nonfresh_artifact_count"] == 0, f"{label} no-op build compiled non-fresh artifacts")
            elif expected_setup == "content-mutation":
                require(setup.get("input_path") == expected_input, f"{label}.setup.input_path mismatch")
                before_sha = require_sha256(setup.get("before_sha256"), f"{label}.setup.before_sha256")
                during_sha = require_sha256(setup.get("during_sha256"), f"{label}.setup.during_sha256")
                content_before_shas.add(before_sha)
                require(during_sha != before_sha and setup.get("after_sha256") == before_sha, f"{label}.setup does not prove a reversible content edit")
                require(setup.get("mutation_kind") == "append-comment", f"{label}.setup mutation kind mismatch")
                mutation_sha = require_sha256(setup.get("mutation_sha256"), f"{label}.setup.mutation_sha256")
                mutation_bytes = require_positive_int(setup.get("mutation_bytes"), f"{label}.setup.mutation_bytes")
                before_input = build_artifact_ref(root, setup.get("before_input"), f"{label}.setup.before_input")
                during_input = build_artifact_ref(root, setup.get("during_input"), f"{label}.setup.during_input")
                mutation_input = build_artifact_ref(root, setup.get("mutation_artifact"), f"{label}.setup.mutation_artifact")
                before_content = before_input.read_bytes()
                during_content = during_input.read_bytes()
                mutation_content = mutation_input.read_bytes()
                require(file_sha256(before_input) == before_sha, f"{label}.setup.before_input SHA mismatch")
                require(file_sha256(during_input) == during_sha, f"{label}.setup.during_input SHA mismatch")
                require(file_sha256(mutation_input) == mutation_sha, f"{label}.setup.mutation_artifact SHA mismatch")
                require(len(mutation_content) == mutation_bytes, f"{label}.setup mutation byte count mismatch")
                require(during_content == before_content + mutation_content, f"{label}.setup content mutation cannot be reproduced")
                before_mtime = require_positive_int(setup.get("before_mtime_ns"), f"{label}.setup.before_mtime_ns")
                during_mtime = require_positive_int(setup.get("during_mtime_ns"), f"{label}.setup.during_mtime_ns")
                after_mtime = require_positive_int(setup.get("after_mtime_ns"), f"{label}.setup.after_mtime_ns")
                require(during_mtime > before_mtime and after_mtime == before_mtime, f"{label}.setup does not prove touch and restore")
                require(cargo_summary["nonfresh_artifact_count"] > 0, f"{label} touch did not invalidate any Cargo artifact")
                require(any(expected_package in package for package in cargo_summary["nonfresh_packages"]), f"{label} did not invalidate {expected_package}")
                if name in {"core-ptx", "native-tu"}:
                    expected_log_input = str(expected_input).removeprefix("crates/ferrum-kernels/")
                    require(expected_log_input in log_text, f"{label}.log does not bind the edited input")
            else:
                require(setup.get("argv") == ["cargo", "clean"], f"{label}.setup.argv must be cargo clean")
                require(setup.get("returncode") == 0, f"{label}.setup cargo clean failed")
                require(setup.get("target_absent_after_clean") is True, f"{label}.setup did not remove the release binary")
                build_artifact_ref(root, setup.get("log"), f"{label}.setup.log")
                require(cargo_summary["nonfresh_artifact_count"] > 0, f"{label} clean release contains no non-fresh artifacts")
            native_build = require_object(sample.get("native_build"), f"{label}.native_build")
            expected_native = build_timing.native_build_summary(log_text, expected_input)
            require(native_build == expected_native, f"{label}.native_build cannot be recomputed from raw log")
            if name in {"noop", "rust-model-leaf", "rust-runtime-leaf", "core-ptx"}:
                require(native_build["compiled_tu_count"] == 0, f"{label} unexpectedly recompiles a CUDA translation unit")
            else:
                require(native_build["compiled_tu_count"] > 0, f"{label} CUDA build did not compile a translation unit")
            if name == "native-tu":
                require(any(path.endswith("vllm_marlin/gptq_marlin_repack.cu") for path in native_build["compiled_tu_paths"]), f"{label} native edit did not compile the selected TU")
        if expected_setup == "content-mutation":
            require(len(content_before_shas) == 1, f"build-timings.{name} samples do not start from one canonical input")
            require("restore_verification" in row, f"build-timings.{name} lacks canonical restore verification")
            _, restore_summary, _, _ = validate_build_record(
                root,
                row.get("restore_verification"),
                f"build-timings.{name}.restore_verification",
                require_binary_sha=binaries["cuda"],
                require_output_binary=True,
            )
            require(restore_summary["nonfresh_artifact_count"] > 0, f"build-timings.{name}.restore_verification was Fresh")
            require(any(expected_package in package for package in restore_summary["nonfresh_packages"]), f"build-timings.{name}.restore_verification did not rebuild {expected_package}")
            restored = require_object(row["restore_verification"].get("restored_input"), f"build-timings.{name}.restore_verification.restored_input")
            require(restored.get("input_path") == expected_input, f"build-timings.{name}.restore_verification input mismatch")
            restored_sha = require_sha256(restored.get("sha256"), f"build-timings.{name}.restore_verification restored SHA256")
            require(restored_sha in content_before_shas, f"build-timings.{name}.restore_verification restored the wrong content")
            before_verify = require_positive_int(restored.get("before_verification_mtime_ns"), f"build-timings.{name}.restore_verification before mtime")
            verify_mtime = require_positive_int(restored.get("verification_mtime_ns"), f"build-timings.{name}.restore_verification verification mtime")
            after_verify = require_positive_int(restored.get("after_verification_mtime_ns"), f"build-timings.{name}.restore_verification after mtime")
            require(verify_mtime > before_verify and after_verify == before_verify, f"build-timings.{name}.restore_verification did not force and then restore mtime")
        else:
            require("restore_verification" not in row, f"build-timings.{name} has unexpected restore verification")
        ordered = sorted(durations)
        expected_p50 = ordered[2]
        expected_p95 = ordered[4]
        actual_p50 = require_number(row.get("p50_sec"), f"build-timings.{name}.p50_sec", positive=True)
        actual_p95 = require_number(row.get("p95_sec"), f"build-timings.{name}.p95_sec", positive=True)
        require(math.isclose(actual_p50, expected_p50, rel_tol=1e-9), f"build-timings.{name}.p50_sec mismatch")
        require(math.isclose(actual_p95, expected_p95, rel_tol=1e-9), f"build-timings.{name}.p95_sec mismatch")
    require(seen == BUILD_SCENARIOS, "build-timings scenario matrix incomplete")


def validate_inventory(root: Path) -> None:
    data = read_json(root / "coupling-inventory.json")
    require_schema(data, "coupling-inventory")
    analyzer = require_object(data.get("analyzer"), "coupling-inventory.analyzer")
    require(analyzer.get("path") == "scripts/release/runtime_vnext_inventory.py", "coupling-inventory analyzer path mismatch")
    require(analyzer.get("identity_key") == "sha256", "coupling-inventory identity key must be sha256")
    require(
        set(require_list(analyzer.get("loc_languages"), "coupling-inventory.analyzer.loc_languages"))
        == {
            "rust", "c", "cpp", "cuda", "cuda-header", "c-header", "cpp-header",
            "objective-c", "objective-cpp", "metal", "python", "shell", "makefile", "dockerfile",
        },
        "coupling-inventory LOC language policy mismatch",
    )
    require(
        set(require_list(
            analyzer.get("source_discovery_excluded_dirs"),
            "coupling-inventory.analyzer.source_discovery_excluded_dirs",
        )) == {".git", ".venv", "__pycache__", "node_modules", "target"},
        "coupling-inventory source discovery exclusion policy mismatch",
    )
    require(
        analyzer.get("large_native_thresholds")
        == {
            "production_loc_gte": 10_000,
            "source_bytes_gte": 5 * 1024 * 1024,
            "translation_units_gte": 10,
            "qualifier": "any",
        },
        "coupling-inventory large native thresholds mismatch",
    )
    git = require_object(data.get("git"), "coupling-inventory.git")
    require(git.get("sha") == FROZEN_LEGACY_SHA, "coupling-inventory git SHA is stale")
    require(git.get("tree_sha") == frozen_tree_sha(), "coupling-inventory tree SHA mismatch")
    require(git.get("dirty") is False, "coupling-inventory must come from a clean frozen worktree")
    require(git.get("status_short") == [], "coupling-inventory git status must be empty")
    scope = require_object(data.get("scope"), "coupling-inventory.scope")
    require(scope.get("scan_roots") == ["crates", "scripts"], "coupling-inventory must scan crates and scripts")
    require(scope.get("scripts_policy") == "all scripts (superset of product/release scripts)", "coupling-inventory scripts policy mismatch")
    require(scope.get("coverage_ratio") == 1.0, "coupling-inventory coverage must be 100%")
    require(scope.get("discovered_file_count") == scope.get("inventoried_file_count"), "coupling-inventory discovered/inventoried count mismatch")
    counts = require_object(scope.get("file_count_by_root"), "coupling-inventory.scope.file_count_by_root")
    require(set(counts) == {"crates", "scripts"} and all(isinstance(value, int) and value > 0 for value in counts.values()), "coupling-inventory root coverage is incomplete")
    files = require_list(data.get("files"), "coupling-inventory.files")
    require(files, "coupling-inventory.files must not be empty")
    required_file_keys = {"path", "sha256", "content_id", "size_bytes", "language", "classification", "logical_loc", "logical_loc_by_classification", "coupling_finding_count", "coupling_counts"}
    allowed_classifications = {"production", "test", "generated", "vendor", "example", "fixture"}
    seen: set[str] = set()
    for index, raw in enumerate(files):
        row = require_object(raw, f"coupling-inventory.files[{index}]")
        missing = required_file_keys - set(row)
        require(not missing, f"coupling-inventory.files[{index}] missing {sorted(missing)}")
        path = require_string(row.get("path"), f"coupling-inventory.files[{index}].path")
        require(path not in seen, f"coupling-inventory duplicate path: {path}")
        seen.add(path)
        digest = require_sha256(row.get("sha256"), f"coupling-inventory.files[{index}].sha256")
        require(row.get("content_id") == f"sha256:{digest}", f"coupling-inventory.files[{index}].content_id mismatch")
        require(row.get("classification") in allowed_classifications, f"coupling-inventory.files[{index}].classification invalid")
        require(isinstance(row.get("logical_loc"), int) and row["logical_loc"] >= 0, f"coupling-inventory.files[{index}].logical_loc invalid")
    require(len(seen) == scope.get("inventoried_file_count"), "coupling-inventory file count mismatch")
    summary = require_object(data.get("summary"), "coupling-inventory.summary")
    require(summary.get("file_count") == len(files), "coupling-inventory.summary.file_count mismatch")
    category_counts = require_object(summary.get("coupling_count_by_category"), "coupling-inventory.summary.coupling_count_by_category")
    required_categories = {
        "qwen35_symbol",
        "architecture_named_api",
        "backend_trait_method",
        "backend_cfg",
        "ferrum_env_read",
        "legacy_factory_candidate",
        "model_runner_candidate",
        "model_scaffolding_candidate",
        "product_decision_candidate",
    }
    require(required_categories.issubset(category_counts), "coupling-inventory coupling categories incomplete")
    coupling = require_object(data.get("coupling"), "coupling-inventory.coupling")
    findings = require_list(coupling.get("findings"), "coupling-inventory.coupling.findings")
    require(len(findings) == summary.get("coupling_finding_count"), "coupling-inventory finding count mismatch")
    duplicate_decisions = require_list(coupling.get("potential_run_serve_duplicate_decisions"), "coupling-inventory.coupling.potential_run_serve_duplicate_decisions")
    require(duplicate_decisions, "coupling-inventory must capture run/serve duplicate decision candidates")
    native = require_list(data.get("large_native_source_trees"), "coupling-inventory.large_native_source_trees")
    require(summary.get("native_source_tree_count") == len(native), "coupling-inventory native tree count mismatch")
    require(
        summary.get("large_third_party_native_source_count")
        == sum(1 for row in native if isinstance(row, dict) and row.get("is_large") is True),
        "coupling-inventory large native count mismatch",
    )
    review = read_json(INVENTORY_REVIEW_PATH)
    require_schema(review, "runtime_vnext_inventory_review")
    require(review.get("reviewed_at_git_sha") == FROZEN_LEGACY_SHA, "inventory review SHA mismatch")
    require(review.get("candidate_identity") == "path+symbol", "inventory review identity must be path+symbol")
    require(review.get("unresolved_count") == 0, "inventory review has unresolved model scaffolding candidates")
    decisions = require_list(review.get("decisions"), "inventory review decisions")
    expected_decisions = {"scaffolding-owned", "excluded-math", "excluded-parser", "excluded-weights", "excluded-other"}
    require(set(decisions) == expected_decisions, "inventory review decision vocabulary mismatch")
    candidates: dict[tuple[str, str], set[int]] = defaultdict(set)
    for finding in findings:
        if isinstance(finding, dict) and finding.get("category") == "model_scaffolding_candidate":
            path = require_string(finding.get("path"), "model scaffolding candidate.path")
            symbol = require_string(finding.get("symbol"), "model scaffolding candidate.symbol")
            line = finding.get("line")
            require(isinstance(line, int) and line > 0, "model scaffolding candidate.line must be positive")
            candidates[(path, symbol)].add(line)
    review_rows = require_list(review.get("reviews"), "inventory review rows")
    reviewed: dict[tuple[str, str], set[int]] = {}
    classification_counts: dict[str, int] = {decision: 0 for decision in expected_decisions}
    for index, raw in enumerate(review_rows):
        row = require_object(raw, f"inventory review rows[{index}]")
        key = (
            require_string(row.get("path"), f"inventory review rows[{index}].path"),
            require_string(row.get("symbol"), f"inventory review rows[{index}].symbol"),
        )
        require(key not in reviewed, f"duplicate inventory review candidate: {key}")
        line_hints = require_list(row.get("line_hints"), f"inventory review rows[{index}].line_hints")
        require(line_hints and all(isinstance(line, int) and line > 0 for line in line_hints), f"inventory review rows[{index}].line_hints invalid")
        reviewed[key] = set(line_hints)
        classification = require_string(row.get("classification"), f"inventory review rows[{index}].classification")
        require(classification in expected_decisions, f"inventory review rows[{index}].classification invalid")
        classification_counts[classification] += 1
        require_string(row.get("reason"), f"inventory review rows[{index}].reason")
        require_string(row.get("owner"), f"inventory review rows[{index}].owner")
        require(row.get("reviewed_at_git_sha") == FROZEN_LEGACY_SHA, f"inventory review rows[{index}] SHA mismatch")
    require(set(reviewed) == set(candidates), "inventory review does not exactly cover model scaffolding candidates")
    require(all(reviewed[key] == candidates[key] for key in candidates), "inventory review line hints are stale")
    require(review.get("candidate_count") == len(candidates), "inventory review candidate_count mismatch")
    require(review.get("reviewed_count") == len(reviewed), "inventory review reviewed_count mismatch")
    require(review.get("classification_counts") == classification_counts, "inventory review classification_counts mismatch")
    reviewed_native = require_list(review.get("large_native_content_roots"), "inventory review large native roots")
    native_keys = {
        (row.get("tree_key"), row.get("content_root_sha256"))
        for row in native
        if isinstance(row, dict) and row.get("is_large") is True
    }
    review_native_keys: set[tuple[Any, Any]] = set()
    for index, raw in enumerate(reviewed_native):
        row = require_object(raw, f"inventory review large native roots[{index}]")
        key = (row.get("tree_key"), row.get("content_root_sha256"))
        require(key not in review_native_keys, f"duplicate inventory native review: {key}")
        review_native_keys.add(key)
        require(row.get("counted_build_input_count") == 1, f"inventory native review {key} must count one content root")
        require_string(row.get("aggregation_decision"), f"inventory native review {key}.aggregation_decision")
        require_string(row.get("reason"), f"inventory native review {key}.reason")
        require_string(row.get("owner"), f"inventory native review {key}.owner")
        require(row.get("reviewed_at_git_sha") == FROZEN_LEGACY_SHA, f"inventory native review {key} SHA mismatch")
    require(review_native_keys == native_keys, "inventory native content-root review is stale")
    require(review.get("large_native_content_root_count") == len(native_keys), "inventory review native count mismatch")
    require(review.get("large_native_content_root_reviewed_count") == len(review_native_keys), "inventory review native reviewed count mismatch")
    require(review.get("large_native_content_root_unresolved_count") == 0, "inventory review has unresolved native roots")


def validate_historical_bugs(root: Path) -> None:
    catalog = read_json(BUG_CATALOG_PATH)
    require_schema(catalog, "runtime_vnext_historical_bugs catalog")
    require(catalog.get("baseline_git_sha") == FROZEN_LEGACY_SHA, "historical bug catalog baseline SHA mismatch")
    catalog_families = require_list(catalog.get("families"), "historical bug catalog families")
    expected_cases: dict[str, dict[str, Any]] = {}
    expected_family_cases: dict[str, set[str]] = {}
    verified_commits: set[str] = set()
    for family_raw in catalog_families:
        family = require_object(family_raw, "historical bug catalog family")
        family_id = require_string(family.get("id"), "historical bug catalog family.id")
        case_ids: set[str] = set()
        for case_raw in require_list(family.get("cases"), f"historical bug catalog {family_id}.cases"):
            case = require_object(case_raw, f"historical bug catalog {family_id}.case")
            case_id = require_string(case.get("id"), f"historical bug catalog {family_id}.case.id")
            require(case_id not in expected_cases, f"duplicate historical bug catalog case {case_id}")
            expected_cases[case_id] = case
            case_ids.add(case_id)
            for commit_index, commit_raw in enumerate(require_list(case.get("commits", []), f"historical bug catalog {case_id}.commits")):
                commit = require_object(commit_raw, f"historical bug catalog {case_id}.commits[{commit_index}]")
                sha = require_git_sha(commit.get("sha"), f"historical bug catalog {case_id}.commits[{commit_index}].sha", frozen=False)
                if sha not in verified_commits:
                    require(git_value(["cat-file", "-t", sha]) == "commit", f"historical bug catalog commit is not present: {sha}")
                    verified_commits.add(sha)
            for ref in [*case.get("historical_artifacts", []), *case.get("reproducer_paths", [])]:
                ref_path = Path(require_string(ref, f"historical bug catalog {case_id} evidence path"))
                require(not ref_path.is_absolute() and ".." not in ref_path.parts, f"historical bug catalog {case_id} evidence path escapes repository")
                require((REPO_ROOT / ref_path).is_file(), f"historical bug catalog {case_id} evidence path is missing: {ref}")
        expected_family_cases[family_id] = case_ids

    data = read_json(root / "historical-bug-corpus.json")
    require_schema(data, "historical-bug-corpus")
    require_source_identity(data, "historical-bug-corpus")
    require(data.get("catalog_id") == catalog.get("catalog_id"), "historical-bug-corpus.catalog_id mismatch")
    require(data.get("catalog_sha256") == file_sha256(BUG_CATALOG_PATH), "historical-bug-corpus.catalog_sha256 mismatch")
    require(data.get("family_count") == 15, "historical-bug-corpus.family_count must be 15")
    families = require_list(data.get("families"), "historical-bug-corpus.families")
    require(len(families) == 15, "historical-bug-corpus must contain 15 families")
    ids: set[str] = set()
    cases: set[str] = set()
    for index, raw in enumerate(families):
        family = require_object(raw, f"historical-bug-corpus.families[{index}]")
        family_id = require_string(family.get("id"), f"historical-bug-corpus.families[{index}].id")
        require(re.fullmatch(r"H(0[1-9]|1[0-5])", family_id) is not None, f"invalid historical family id: {family_id}")
        require(family_id not in ids, f"duplicate historical family: {family_id}")
        ids.add(family_id)
        case_rows = require_list(family.get("cases"), f"historical-bug-corpus.{family_id}.cases")
        family_case_ids: set[str] = set()
        for case_index, case_raw in enumerate(case_rows):
            case = require_object(case_raw, f"historical-bug-corpus.{family_id}.cases[{case_index}]")
            case_id = require_string(case.get("id"), f"historical-bug-corpus.{family_id}.cases[{case_index}].id")
            require(case_id not in cases, f"duplicate historical case: {case_id}")
            cases.add(case_id)
            family_case_ids.add(case_id)
            catalog_case = expected_cases.get(case_id)
            require(catalog_case is not None, f"historical-bug-corpus unknown case {case_id}")
            require(case.get("failure_class") == catalog_case.get("failure_class"), f"historical-bug-corpus.{case_id}.failure_class mismatch")
            require(case.get("status") == "frozen", f"historical-bug-corpus.{case_id}.status must be frozen")
            require(case.get("entrypoints") == catalog_case.get("entrypoints"), f"historical-bug-corpus.{case_id}.entrypoints mismatch")
            require(case.get("backends") == catalog_case.get("backends"), f"historical-bug-corpus.{case_id}.backends mismatch")
            evidence = require_list(case.get("source_evidence"), f"historical-bug-corpus.{case_id}.source_evidence")
            require(evidence, f"historical-bug-corpus.{case_id} needs source evidence")
            evidence_pairs: set[tuple[str, str]] = set()
            for evidence_index, evidence_raw in enumerate(evidence):
                item = require_object(evidence_raw, f"historical-bug-corpus.{case_id}.source_evidence[{evidence_index}]")
                kind = require_string(item.get("kind"), f"historical-bug-corpus.{case_id}.source_evidence[{evidence_index}].kind")
                require(kind in {"commit", "artifact", "source", "retrospective"}, f"historical-bug-corpus.{case_id} invalid evidence kind")
                ref = require_string(item.get("ref"), f"historical-bug-corpus.{case_id}.source_evidence[{evidence_index}].ref")
                if kind == "commit":
                    require_git_sha(ref, f"historical-bug-corpus.{case_id}.source_evidence[{evidence_index}].ref", frozen=False)
                    require(ref in verified_commits, f"historical-bug-corpus.{case_id} references an unverified commit")
                else:
                    ref_path = Path(ref)
                    require(not ref_path.is_absolute() and ".." not in ref_path.parts, f"historical-bug-corpus.{case_id} evidence path escapes repository")
                    source_path = REPO_ROOT / ref_path
                    require(source_path.is_file(), f"historical-bug-corpus.{case_id} evidence path missing: {ref}")
                    require(item.get("sha256") == file_sha256(source_path), f"historical-bug-corpus.{case_id} evidence SHA256 mismatch: {ref}")
                    require(item.get("size_bytes") == source_path.stat().st_size, f"historical-bug-corpus.{case_id} evidence size mismatch: {ref}")
                evidence_pairs.add((kind, ref))
            for commit in catalog_case.get("commits", []):
                require(("commit", commit["sha"]) in evidence_pairs, f"historical-bug-corpus.{case_id} missing catalog commit {commit['sha']}")
            for artifact in catalog_case.get("historical_artifacts", []):
                require(("artifact", artifact) in evidence_pairs, f"historical-bug-corpus.{case_id} missing catalog artifact {artifact}")
            reproducer = require_object(case.get("reproducer"), f"historical-bug-corpus.{case_id}.reproducer")
            input_path = artifact_path(root, reproducer.get("input_path"), f"historical-bug-corpus.{case_id}.reproducer.input_path")
            mutation_path = artifact_path(root, reproducer.get("mutation_path"), f"historical-bug-corpus.{case_id}.reproducer.mutation_path")
            failure_log = require_log(root, reproducer.get("failure_log"), f"historical-bug-corpus.{case_id}.reproducer.failure_log")
            require(input_path.is_file() and input_path.stat().st_size > 0, f"historical-bug-corpus.{case_id} frozen input missing")
            require(mutation_path.is_file() and mutation_path.stat().st_size > 0, f"historical-bug-corpus.{case_id} mutation missing")
            input_sha = require_sha256(reproducer.get("input_sha256"), f"historical-bug-corpus.{case_id}.reproducer.input_sha256")
            mutation_sha = require_sha256(reproducer.get("mutation_sha256"), f"historical-bug-corpus.{case_id}.reproducer.mutation_sha256")
            require(file_sha256(input_path) == input_sha, f"historical-bug-corpus.{case_id} input SHA mismatch")
            require(file_sha256(mutation_path) == mutation_sha, f"historical-bug-corpus.{case_id} mutation SHA mismatch")
            require(input_sha != mutation_sha, f"historical-bug-corpus.{case_id} mutation must differ from frozen input")
            require_string(reproducer.get("expected_invariant"), f"historical-bug-corpus.{case_id}.reproducer.expected_invariant")
            failure_signature = require_string(reproducer.get("failure_signature"), f"historical-bug-corpus.{case_id}.reproducer.failure_signature")
            require(failure_log.stat().st_size > 0, f"historical-bug-corpus.{case_id} failure log is empty")
            require(
                failure_signature in failure_log.read_text(encoding="utf-8", errors="replace"),
                f"historical-bug-corpus.{case_id} failure signature is absent from failure log",
            )
            command = require_list(reproducer.get("command"), f"historical-bug-corpus.{case_id}.reproducer.command")
            require(command and all(isinstance(part, str) and part for part in command), f"historical-bug-corpus.{case_id} reproducer command must be argv")
            require(
                Path(command[0]).name in {"cargo", "python", "python3", "ferrum"},
                f"historical-bug-corpus.{case_id} reproducer command is not an approved real runner",
            )
            require(case_id in command, f"historical-bug-corpus.{case_id} reproducer command does not bind case id")
            require(str(reproducer.get("input_path")) in command, f"historical-bug-corpus.{case_id} reproducer command does not bind frozen input")
            require(str(reproducer.get("mutation_path")) in command, f"historical-bug-corpus.{case_id} reproducer command does not bind mutation")
            returncode = reproducer.get("returncode")
            require(
                isinstance(returncode, int) and not isinstance(returncode, bool) and returncode != 0,
                f"historical-bug-corpus.{case_id} reproducer must capture a non-zero failing result",
            )
            validate_execution_window(reproducer, f"historical-bug-corpus.{case_id}.reproducer")
            require(
                reproducer.get("mutation_kind") in {"revert_patch", "fault_injection", "frozen_bad_input"},
                f"historical-bug-corpus.{case_id} mutation_kind is invalid",
            )
        require(family_case_ids == expected_family_cases.get(family_id), f"historical-bug-corpus.{family_id} case coverage mismatch")
    require(ids == {f"H{index:02d}" for index in range(1, 16)}, "historical bug families must be H01-H15")
    require(cases == set(expected_cases), "historical-bug-corpus concrete case coverage mismatch")
    require(data.get("concrete_case_count") == len(cases), "historical-bug-corpus concrete_case_count mismatch")
    require(data.get("orphan_case_count") == 0, "historical-bug-corpus orphan_case_count must be zero")
    require(data.get("duplicate_case_count") == 0, "historical-bug-corpus duplicate_case_count must be zero")


def build_artifact_index(root: Path) -> list[dict[str, Any]]:
    excluded = {
        "manifest.json",
        "gate.manifest.json",
        "run_gate.child.stdout",
        "run_gate.child.stderr",
        "run_gate.child.command.json",
    }
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"artifact tree contains forbidden symlink: {path}")
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in excluded:
            continue
        require(path.stat().st_size > 0, f"artifact file is empty: {rel}")
        rows.append(
            {
                "path": rel,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
                "role": rel.split("/", 1)[0] if "/" in rel else "root-manifest",
            }
        )
    require(rows, "artifact index is empty")
    return rows


def _validate_root_impl(root: Path, *, allow_synthetic: bool = False) -> dict[str, Any]:
    require(root.is_dir(), f"artifact root does not exist: {root}")
    if not allow_synthetic:
        validator_dirty = git_value(["status", "--short"]).splitlines()
        require(not validator_dirty, f"validator worktree must be clean: {validator_dirty}")
        reject_synthetic_artifacts(root)
    models_lock, hardware, models = validate_models_lock(root, allow_synthetic=allow_synthetic)
    binaries = validate_legacy_binaries(root, hardware)
    statuses, executor_invocations = validate_correctness(
        root,
        models,
        hardware,
        binaries,
        allow_synthetic=allow_synthetic,
        expectations_binding=models_lock["expectations_catalog"],
    )
    validate_performance(
        root,
        models,
        statuses,
        hardware,
        binaries,
        allow_synthetic=allow_synthetic,
    )
    validate_build_timings(root, hardware, binaries)
    validate_inventory(root)
    validate_historical_bugs(root)
    contracts = contract_files()
    artifacts = build_artifact_index(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "source_git_sha": FROZEN_LEGACY_SHA,
        "validated_at": now_iso(),
        "validator_git_sha": git_value(["rev-parse", "HEAD"]),
        "validator_dirty_status": git_value(["status", "--short"]).splitlines(),
        "artifact_dir": str(root),
        "contract_files": contracts,
        "contract_sha256": combined_contract_sha(contracts),
        "artifact_index": artifacts,
        "artifact_count": len(artifacts),
        "models_lock_sha256": file_sha256(root / "models.lock.json"),
        "expectations_catalog": models_lock["expectations_catalog"],
        "correctness_executor_invocations": executor_invocations,
        "legacy_binaries_sha256": file_sha256(root / "legacy-binaries.json"),
        "coupling_inventory_sha256": file_sha256(root / "coupling-inventory.json"),
        "historical_bug_corpus_sha256": file_sha256(root / "historical-bug-corpus.json"),
        "primary_models": sorted(PRIMARY_MODELS),
        "supplemental_models": sorted(SUPPLEMENTAL_MODELS),
        "correctness_lanes": {f"{model}/{backend}": status for (model, backend), status in sorted(statuses.items())},
        "waiver_count": 0,
        "pass_line": f"{PASS_PREFIX}: {root}",
    }


def validate_root(root: Path, *, allow_synthetic: bool = False) -> dict[str, Any]:
    try:
        return _validate_root_impl(root, allow_synthetic=allow_synthetic)
    except BaselineError:
        raise
    except (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError) as exc:
        raise BaselineError(f"malformed baseline artifact ({type(exc).__name__}): {exc}") from exc


def synthetic_sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def synthetic_log(root: Path, rel: str, text: str = "synthetic evidence\n") -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return rel


def synthetic_endpoint_probe(
    root: Path,
    *,
    stem: str,
    url: str,
    started_at: datetime,
    finished_at: datetime,
    body: dict[str, Any],
) -> dict[str, Any]:
    body_rel = f"{stem}.body.json"
    receipt_rel = f"{stem}.receipt.json"
    body_origin = f"/remote/{body_rel}"
    receipt_origin = f"/remote/{receipt_rel}"
    write_json(root / body_rel, body)
    argv = [
        "python3",
        f"/repo/{resource_sampler.COLLECTOR_RELATIVE_PATH}",
        "--probe-url",
        url,
        "--probe-body-out",
        body_origin,
        "--probe-receipt-out",
        receipt_origin,
        "--probe-timeout-sec",
        "10",
    ]
    write_json(
        root / receipt_rel,
        {
            "schema_version": resource_sampler.SCHEMA_VERSION,
            "collector_path": resource_sampler.COLLECTOR_RELATIVE_PATH,
            "collector_sha256": file_sha256(RESOURCE_SAMPLER_PATH),
            "argv": argv,
            "started_at": iso_at(started_at),
            "finished_at": iso_at(finished_at),
            "duration_sec": (finished_at - started_at).total_seconds(),
            "returncode": 0,
            "url": url,
            "http_status": 200,
            "body_origin_path": body_origin,
            "body_sha256": file_sha256(root / body_rel),
            "body_size_bytes": (root / body_rel).stat().st_size,
        },
    )
    return {
        "receipt_origin_path": receipt_origin,
        "body_origin_path": body_origin,
        "receipt": receipt_rel,
        "receipt_sha256": file_sha256(root / receipt_rel),
        "body": body_rel,
        "body_sha256": file_sha256(root / body_rel),
    }


def iso_at(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def synthetic_process_identity(pid: int, started_at: datetime) -> tuple[str, dict[str, str]]:
    source = {"kind": "ps-lstart", "raw_output": started_at.strftime("%a %b %d %H:%M:%S %Y")}
    return resource_sampler.process_marker_from_source(pid, source), source


def synthetic_resource_evidence(
    root: Path,
    *,
    stem: str,
    session: dict[str, Any],
    cell_id: str,
    backend: str,
    measurement_started_at: str,
    measurement_finished_at: str,
    runtime_log_origin_path: str,
    memory_bytes: int,
    concurrency: int,
    cap: int,
    process_probe: bool = False,
) -> dict[str, Any]:
    observation_rel = f"{stem}.resource-observations.jsonl"
    observation_origin = f"/remote/{observation_rel}"
    stop_origin = f"/remote/{stem}.resource-stop"
    started = parse_timestamp(measurement_started_at, "synthetic.resource.measurement_started_at")
    finished = parse_timestamp(measurement_finished_at, "synthetic.resource.measurement_finished_at")
    sample_times = [
        started - timedelta(seconds=1) + timedelta(seconds=offset)
        for offset in range(int((finished - started).total_seconds()) + 3)
    ]
    probe_format = "process" if process_probe else "json"
    probe_path = "" if process_probe else "/health"
    probe_url = "" if process_probe else f"{session['base_url'].rstrip('/')}{probe_path}"
    header = {
        "record_type": "header",
        "schema_version": resource_sampler.SCHEMA_VERSION,
        "collector_path": resource_sampler.COLLECTOR_RELATIVE_PATH,
        "collector_sha256": file_sha256(RESOURCE_SAMPLER_PATH),
        "session_id": session["session_id"],
        "cell_id": cell_id,
        "backend": backend,
        "hardware_id": session["hardware_id"],
        "pid": session["pid"],
        "pgid": session["pgid"],
        "process_start_marker": session["process_start_marker"],
        "process_start_source": session["process_start_source"],
        "base_url": session["base_url"],
        "started_at": iso_at(started - timedelta(seconds=2)),
        "interval_ms": 250,
        "runtime_log_path": runtime_log_origin_path,
        "active_probe": {
            "format": probe_format,
            "path": probe_path,
            "url": probe_url,
            "selector": "process-alive" if process_probe else "engine.active_requests",
            "semantics": "process-alive" if process_probe else "scheduler-active-high-water",
        },
    }
    active = min(concurrency, cap)
    samples: list[dict[str, Any]] = []
    for sequence, sampled_at in enumerate(sample_times):
        row: dict[str, Any] = {
            "record_type": "sample",
            "sequence": sequence,
            "sampled_at": iso_at(sampled_at),
            "pid": session["pid"],
            "pgid": session["pgid"],
            "process_start_marker": session["process_start_marker"],
            "process_alive": True,
            "process_rss_bytes": 1024 * 1024,
            "memory_used_bytes": 1024 * 1024,
            "physical_headroom_bytes": memory_bytes // 2,
            "swap_used_bytes": 0,
            "active_requests": active,
            "oom_count": 0,
            "admission_error_count": 0,
        }
        if backend == "metal":
            row.update({"thermal_state": "nominal", "power_mode": "normal"})
        else:
            row["device_memory_bytes"] = 1024 * 1024
        samples.append(row)
    footer = {
        "record_type": "footer",
        "finished_at": iso_at(finished + timedelta(seconds=2)),
        "sample_count": len(samples),
        "exit_reason": "stop-file",
    }
    observation_path = root / observation_rel
    observation_path.parent.mkdir(parents=True, exist_ok=True)
    observation_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in [header, *samples, footer]),
        encoding="utf-8",
    )
    argv = [
        "python3",
        f"/repo/{resource_sampler.COLLECTOR_RELATIVE_PATH}",
        "--out",
        observation_origin,
        "--pid",
        str(session["pid"]),
        "--pgid",
        str(session["pgid"]),
        "--session-id",
        session["session_id"],
        "--cell-id",
        cell_id,
        "--backend",
        backend,
        "--hardware-id",
        session["hardware_id"],
        "--base-url",
        session["base_url"],
        "--active-probe-format",
        probe_format,
        "--active-selector",
        "process-alive" if process_probe else "engine.active_requests",
        "--active-semantics",
        "process-alive" if process_probe else "scheduler-active-high-water",
        "--runtime-log",
        runtime_log_origin_path,
        "--stop-file",
        stop_origin,
        "--interval-ms",
        "250",
        "--max-duration-sec",
        "7200",
    ]
    if not process_probe:
        argv.extend(["--active-path", "/health"])
    summary = resource_sampler.derive_summary(
        observation_path,
        session_id=session["session_id"],
        cell_id=cell_id,
        backend=backend,
        hardware_id=session["hardware_id"],
        pid=session["pid"],
        pgid=session["pgid"],
        process_start_marker=session["process_start_marker"],
        base_url=session["base_url"],
        session_started_at=session["started_at"],
        session_finished_at=session["finished_at"],
        measurement_started_at=measurement_started_at,
        measurement_finished_at=measurement_finished_at,
        memory_budget_bytes=memory_bytes // 2,
        requested_concurrency=concurrency,
        typed_active_cap=cap,
        runtime_log_path=runtime_log_origin_path,
    )
    return {
        "collector_sha256": file_sha256(RESOURCE_SAMPLER_PATH),
        "sampler_argv": argv,
        "observation_origin_path": observation_origin,
        "observations": observation_rel,
        "observations_sha256": file_sha256(observation_path),
        "summary": summary,
    }


def synthetic_itl_evidence(output_tokens: int) -> dict[str, Any]:
    require(output_tokens >= 2, "synthetic ITL evidence requires at least two output tokens")
    return {
        "source": "sse_delta_events",
        "output_events": output_tokens,
        "usage_output_tokens": output_tokens,
        "observed_intervals": output_tokens - 1,
        "transport_coalesced_output_chunks": 0,
        "eligibility": "eligible",
    }


def synthetic_repeat_row(
    *,
    implementation: str,
    session_id: str,
    slot: int,
    repeat: int,
    dataset: str,
    concurrency: int,
    backend: str,
    memory_bytes: int,
    cap: int,
    input_len: int,
    output_len: int,
) -> dict[str, Any]:
    base = 100.0 + slot + repeat + (0.5 if implementation == "B" else 0.0)
    output_tokens = output_len * 100
    actual_input_tokens = input_len * 100
    duration_s = output_tokens / base
    latencies = {
        metric: {
            "p50": 10.0 + offset + repeat,
            "p75": 12.0 + offset + repeat,
            "p95": 15.0 + offset + repeat,
            "p99": 18.0 + offset + repeat,
        }
        for offset, metric in enumerate(HTTP_LATENCY_METRICS)
    }
    quality = {field.removesuffix("_per_run"): 0 for field in BENCH_QUALITY_FIELDS}
    return {
        "repeat": repeat,
        "duration_s": duration_s,
        "output_throughput_tps": base,
        "total_throughput_tps": (actual_input_tokens + output_tokens) / duration_s,
        "request_throughput_rps": 100.0 / duration_s,
        "goodput_rps": 100.0 / duration_s,
        **latencies,
        "expected_requests": 100,
        "completed_requests": 100,
        "errored_requests": 0,
        "warmup_expected": 10,
        "warmup_completed": 10,
        "warmup_errored": 0,
        "output_token_count_source": "usage",
        "output_tokens": output_tokens,
        "actual_input_tokens": actual_input_tokens,
        "itl_eligible_requests": 100,
        "itl_ineligible_requests": 0,
        "itl_expected_intervals": 100 * (output_len - 1),
        "itl_observed_intervals": 100 * (output_len - 1),
        "itl_eligibility_counts": {
            field: 100 if field == "eligible" else 0 for field in ITL_ELIGIBILITY_FIELDS
        },
        "quality_issues": quality,
        "warmup_quality_issues": dict(quality),
    }


def synthetic_bench_report(
    repeats: list[dict[str, Any]],
    *,
    request_model: str,
    backend: str,
    concurrency: int,
    n_prompt: int,
    input_len: int,
    output_len: int,
    client_sha: str,
    hardware_id: str,
) -> dict[str, Any]:
    metric_sets = {
        metric: {
            percentile: scalar_stats([float(row[metric][percentile]) for row in repeats])
            for percentile in HTTP_PERCENTILES
        }
        for metric in HTTP_LATENCY_METRICS
    }
    scalar_sets = {
        metric: scalar_stats([float(row[metric]) for row in repeats])
        for metric in HTTP_SCALAR_METRICS
    }
    quality = {field.removesuffix("_per_run"): 0 for field in BENCH_QUALITY_FIELDS}
    env = {
        "commit_sha": client_sha,
        "hw_id": hardware_id,
        "rust": "rustc-selftest",
        "ferrum_features": [backend],
        "ferrum_env": {},
        "runtime_config": {"entries": []},
    }
    env_hash = recompute_bench_env_hash(env, "selftest.bench.env").removeprefix("sha256:")
    return {
        "model": request_model,
        "backend": backend,
        "scenario": "closed_loop",
        "concurrency": concurrency,
        "n_prompt": n_prompt,
        "n_gen": 128,
        "actual_input_tokens": {
            "requested": n_prompt,
            "min": input_len,
            "max": input_len,
            "mean": float(input_len),
        },
        "actual_input_tokens_per_request": [[input_len] * 100 for _ in range(3)],
        "output_tokens_per_request": [[output_len] * 100 for _ in range(3)],
        "itl_evidence_per_request": [
            [synthetic_itl_evidence(output_len) for _ in range(100)] for _ in range(3)
        ],
        "output_token_count_source": "usage",
        "n_repeats": 3,
        "n_requests_per_run": 100,
        "warmup_requests": 10,
        "repeat_metrics": repeats,
        **metric_sets,
        **scalar_sets,
        "slo": {"ttft_p99_ms": 500.0, "tpot_p99_ms": 50.0, "e2e_p99_ms": 30000.0},
        "completed_per_run": [100, 100, 100],
        "errored_per_run": [0, 0, 0],
        **{field: [0, 0, 0] for field in BENCH_QUALITY_FIELDS},
        "quality_issues_per_run": [quality, quality, quality],
        "env": env,
        "env_hash": f"sha256:{env_hash}",
    }


def synthetic_http_implementation(
    root: Path,
    *,
    implementation: str,
    sessions: list[dict[str, Any]],
    workload: dict[str, Any],
    backend: str,
    dataset: str,
    concurrency: int,
    memory_bytes: int,
    cap: int,
    client: dict[str, Any],
    artifact_key: str,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    workload_config = read_json(root / str(workload["effective_config"]))
    input_len = 256 if backend == "cuda" else 64
    if dataset != "random":
        input_len += 7
    output_len = 128 if dataset == "random" else 120
    for session in [row for row in sessions if row["implementation"] == implementation]:
        slot = session["slot"]
        session_id = session["session_id"]
        cell_id = f"{dataset}:c{concurrency}"
        cell_window = next(window for window in session["cell_windows"] if window["cell_id"] == cell_id)
        repeats = [
            synthetic_repeat_row(
                implementation=implementation,
                session_id=session_id,
                slot=slot,
                repeat=repeat,
                dataset=dataset,
                concurrency=concurrency,
                backend=backend,
                memory_bytes=memory_bytes,
                cap=cap,
                input_len=input_len,
                output_len=output_len,
            )
            for repeat in range(1, 4)
        ]
        stem = f"http/{artifact_key}/{backend}/{dataset}/c{concurrency}/{implementation}/slot-{slot}"
        report_rel = f"{stem}/bench-report.json"
        report = synthetic_bench_report(
            repeats,
            request_model=workload_config["request_model"],
            backend=backend,
            concurrency=concurrency,
            n_prompt=256 if backend == "cuda" else 64,
            input_len=input_len,
            output_len=output_len,
            client_sha=client["source_git_sha"],
            hardware_id=session["hardware_id"],
        )
        write_json(root / report_rel, report)
        stdout_rel = synthetic_log(root, f"{stem}/bench.stdout", "bench-serve completed 300 measured requests with usage and CI evidence\n")
        stderr_rel = synthetic_log(root, f"{stem}/bench.stderr", "bench-serve completed without errors or malformed streams\n")
        dataset_arg = "random" if dataset == "random" else "sharegpt"
        argv = [
            client["binary_path"],
            "bench-serve",
            "--base-url",
            session["base_url"],
            "--model",
            workload_config["request_model"],
            "--tokenizer",
            workload["tokenizer_origin_path"],
            "--concurrency",
            str(concurrency),
            "--dataset",
            dataset_arg,
            "--random-input-len",
            str(256 if backend == "cuda" else 64),
            "--random-output-len",
            "128",
            "--num-prompts",
            "100",
            "--warmup-requests",
            "10",
            "--n-repeats",
            "3",
            "--seed",
            "9271",
            "--goodput",
            "ttft:500,tpot:50,e2e:30000",
            "--output",
            "json",
            "--out",
            f"/remote/{report_rel}",
            "--hw-id",
            session["hardware_id"],
            "--commit-sha",
            client["source_git_sha"],
            "--fail-on-error",
            "--require-ci",
            "--enable-thinking",
            "false",
        ]
        if dataset == "random":
            argv.append("--ignore-eos")
        else:
            argv.extend(["--sharegpt-path", workload["dataset_origin_path"]])
        resources = synthetic_resource_evidence(
            root,
            stem=f"{stem}/resource",
            session=session,
            cell_id=cell_id,
            backend=backend,
            measurement_started_at=cell_window["started_at"],
            measurement_finished_at=cell_window["finished_at"],
            runtime_log_origin_path=session["runtime_log_origin_path"],
            memory_bytes=memory_bytes,
            concurrency=concurrency,
            cap=cap,
        )
        reports.append(
            {
                "session_id": session_id,
                "slot": slot,
                "cell_id": cell_id,
                "dataset": dataset,
                "concurrency": concurrency,
                "benchmark_client_binary_sha256": client["binary_sha256"],
                "bench_argv": argv,
                "env": {"RUST_LOG": "info"},
                "raw_report_origin_path": f"/remote/{report_rel}",
                "raw_report": report_rel,
                "raw_report_sha256": file_sha256(root / report_rel),
                "started_at": cell_window["started_at"],
                "finished_at": cell_window["finished_at"],
                "duration_sec": (
                    parse_timestamp(cell_window["finished_at"], "synthetic.http.finished")
                    - parse_timestamp(cell_window["started_at"], "synthetic.http.started")
                ).total_seconds(),
                "returncode": 0,
                "stdout": stdout_rel,
                "stdout_sha256": file_sha256(root / stdout_rel),
                "stderr": stderr_rel,
                "stderr_sha256": file_sha256(root / stderr_rel),
                "resources": resources,
            }
        )
    return {
        "measured_requests": 1200,
        "completed_requests": 1200,
        "warmup_requests": 120,
        "warmup_completed": 120,
        "error_count": 0,
        "bad_output_count": 0,
        "output_token_count_source": "usage",
        "reports": reports,
    }


def synthetic_catalog_files(catalog_lane: dict[str, Any], seed: str) -> list[dict[str, Any]]:
    paths: list[tuple[str, int, str | None]] = []
    for raw in catalog_lane["files"]:
        if raw.get("required") is not True and raw.get("required_if_sharded") is not True:
            continue
        if "path" in raw:
            path = str(raw["path"])
        else:
            pattern = str(raw["glob"])
            if pattern == "*.safetensors":
                path = "model-00001-of-00001.safetensors"
            elif "quantize" in pattern:
                path = "quantize_config.json"
            else:
                path = pattern.replace("*", "synthetic")
        expected_sha256 = raw.get("expected_sha256")
        if expected_sha256 is not None:
            expected_sha256 = require_sha256(
                expected_sha256,
                f"selftest catalog file {path}.expected_sha256",
            )
        paths.append((path, int(raw.get("expected_size_bytes", 1024)), expected_sha256))
    return [
        {
            "path": path,
            "sha256": expected_sha256 or synthetic_sha(f"{seed}-{path}"),
            "size_bytes": size,
        }
        for path, (size, expected_sha256) in {
            path: (size, expected_sha256)
            for path, size, expected_sha256 in paths
        }.items()
    ]


def synthetic_semantic_files(
    seed: str,
    expected_sha256: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    expected_sha256 = expected_sha256 or {
        path: synthetic_sha(f"{seed}-{path}")
        for path in ("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json")
    }
    return [
        {
            "path": path,
            "sha256": digest,
            "size_bytes": 128,
        }
        for path, digest in expected_sha256.items()
    ]


def make_synthetic_hardware(
    root: Path,
    *,
    hardware_id: str,
    backend: str,
    policy_id: str,
) -> dict[str, Any]:
    if backend == "cuda":
        device_name = "NVIDIA GeForce RTX 4090"
        memory_bytes = 24 * 1024**3
        runtime = {
            "driver_version": "self-test-driver",
            "cuda_version": "12.4",
            "nvcc_version": "Cuda compilation tools, release 12.4, V12.4.99",
        }
        system = {
            "os": "Linux 6.8.0",
            "arch": "x86_64",
            "cpu": "self-test-cpu",
            "logical_cpu_count": 32,
            "host_memory_bytes": 64 * 1024**3,
        }
        argv = copy.deepcopy(hardware_probe.PROBE_ARGV[backend])
        outputs = {
            "host": "Linux cuda-self-test-host 6.8.0 x86_64\n",
            "gpu": f"{device_name}, 24576, self-test-driver\n",
            "toolchain": runtime["nvcc_version"] + "\n",
            "memory": "Mem: 68719476736 0 0 0 0 0\n",
            "cpu": json.dumps({"lscpu": [{"field": "CPU(s):", "data": "32"}, {"field": "Model name:", "data": "self-test-cpu"}]}) + "\n",
        }
    else:
        device_name = "Apple M1 Max"
        memory_bytes = 32 * 1024**3
        runtime = {"macos_version": "15.1.1", "metal_toolchain": "metal version self-test"}
        system = {
            "os": "Darwin 24.1.0",
            "arch": "arm64",
            "cpu": "Apple M1 Max",
            "logical_cpu_count": 10,
            "host_memory_bytes": memory_bytes,
        }
        argv = copy.deepcopy(hardware_probe.PROBE_ARGV[backend])
        outputs = {
            "host": "Darwin metal-self-test-host 24.1.0 arm64\n",
            "gpu": "Chipset Model: Apple M1 Max\nTotal Number of Cores: 24\n",
            "toolchain": runtime["metal_toolchain"] + "\n",
            "memory": f"{memory_bytes}\n",
            "cpu": "10\n",
            "os": "15.1.1\n",
        }
    material: dict[str, Any] = {
        "schema_version": 1,
        "backend": backend,
        "policy_id": policy_id,
        "host": f"{backend}-self-test-host",
        "device_name": device_name,
        "device_count": 1,
        "memory_bytes": memory_bytes,
        "runtime": runtime,
        "system": system,
    }
    if backend == "metal":
        material["gpu_core_count"] = 24
    probe_dir = root / "hardware" / hardware_id
    commands = []
    for kind in argv:
        stdout_rel = f"hardware/{hardware_id}/raw/{kind}.stdout.txt"
        stderr_rel = f"hardware/{hardware_id}/raw/{kind}.stderr.txt"
        synthetic_log(root, stdout_rel, outputs[kind])
        synthetic_log(root, stderr_rel, "self-test stderr capture\n")
        commands.append(
            {
                "kind": kind,
                "argv": argv[kind],
                "returncode": 0,
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:00:01Z",
                "duration_sec": 1.0,
                "stdout": f"raw/{kind}.stdout.txt",
                "stdout_sha256": file_sha256(root / stdout_rel),
                "stderr": f"raw/{kind}.stderr.txt",
                "stderr_sha256": file_sha256(root / stderr_rel),
            }
        )
    probe_path = probe_dir / "probe.json"
    write_json(
        probe_path,
        {
            "schema_version": 1,
            "source_git_sha": FROZEN_LEGACY_SHA,
            "source_tree_sha": frozen_tree_sha(),
            "dirty_status": {"is_dirty": False, "status_short": []},
            "collector": {
                "path": HARDWARE_PROBE_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": file_sha256(HARDWARE_PROBE_PATH),
            },
            "hardware_id": hardware_id,
            "normalized": material,
            "fingerprint": canonical_json_sha256(material),
            "commands": commands,
        },
    )
    item = {
        "id": hardware_id,
        **{key: value for key, value in material.items() if key != "schema_version"},
        "fingerprint_material": material,
        "fingerprint": canonical_json_sha256(material),
        "probe": {
            "path": probe_path.relative_to(root).as_posix(),
            "sha256": file_sha256(probe_path),
        },
    }
    return item


def synthetic_benchmark_client(root: Path, backend: str) -> dict[str, Any]:
    binary = root / "benchmark-client" / backend / "ferrum"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(f"checked-in {backend} benchmark client\n", encoding="utf-8")
    source_git_sha = synthetic_sha(f"client-git-{backend}")[:40]
    source_tree_sha = synthetic_sha(f"client-tree-{backend}")[:40]
    collector_rows = [
        {"path": path, "sha256": file_sha256(REPO_ROOT / path)}
        for path in BENCHMARK_CLIENT_RUST_ALLOWLIST
    ]
    identity = {
        "source_git_sha": source_git_sha,
        "source_tree_sha": source_tree_sha,
        "collector_source_files": sorted(collector_rows, key=lambda row: row["path"]),
        "production_rust_diff": sorted(BENCHMARK_CLIENT_RUST_ALLOWLIST),
    }
    return {
        "binary_path": f"/selftest/benchmark-client/{backend}/ferrum",
        "artifact_binary": binary.relative_to(root).as_posix(),
        "binary_sha256": file_sha256(binary),
        "source_git_sha": source_git_sha,
        "source_tree_sha": source_tree_sha,
        "dirty_status": {"is_dirty": False, "status_short": []},
        "collector_source_files": collector_rows,
        "production_rust_diff": list(BENCHMARK_CLIENT_RUST_ALLOWLIST),
        "collector_identity_sha256": canonical_json_sha256(identity),
        "cargo_features": [backend],
        "build_log": synthetic_log(root, f"benchmark-client/{backend}/build.log", "benchmark client build completed\n"),
    }


def synthetic_server_identity(
    root: Path,
    *,
    model: dict[str, Any],
    backend: str,
    implementation: str,
    cap: int,
    binary_sha256: str,
    memory_bytes: int,
) -> dict[str, Any]:
    model_key = str(model["key"])
    model_lane = model["lanes"][backend]
    model_origin_path = f"/models/{model_key}/{backend}"
    config_rel = f"server-config/{model_key}/{backend}/{implementation}.json"
    write_json(
        root / config_rel,
        {
            "schema_version": 1,
            "config_source": "normalized-server-argv",
            "enable_thinking": False,
            "typed_active_cap": cap,
            "model_key": model_key,
            "backend": backend,
            "model_repo": model_lane["repo"],
            "model_revision": model_lane["revision"],
            "model_format": model_lane["format"],
            "model_files": locked_file_map(model, backend),
            "request_model": model["official_model_id"],
            "model_origin_path": model_origin_path,
            "memory_budget_bytes": memory_bytes // 2,
        },
    )
    identity: dict[str, Any] = {
        "implementation": implementation,
        "role": "external" if implementation == "A" else "legacy",
        "engine": ("vllm" if backend == "cuda" else "llama.cpp") if implementation == "A" else "ferrum",
        "binary_sha256": binary_sha256,
        "effective_config": config_rel,
        "effective_config_sha256": file_sha256(root / config_rel),
        "typed_active_cap": cap,
        "model_key": model_key,
        "model_repo": model_lane["repo"],
        "model_revision": model_lane["revision"],
        "model_format": model_lane["format"],
        "model_files": locked_file_map(model, backend),
        "request_model": model["official_model_id"],
        "model_origin_path": model_origin_path,
        "memory_budget_bytes": memory_bytes // 2,
    }
    if implementation == "A":
        identity.update(
            {
                "engine_version": "selftest-1.0",
                "engine_revision": synthetic_sha(f"external-revision-{backend}")[:40],
            }
        )
    else:
        identity["source_git_sha"] = FROZEN_LEGACY_SHA
    return identity


def synthetic_server_sessions(
    root: Path,
    *,
    model: dict[str, Any],
    backend: str,
    implementation: str,
    identity: dict[str, Any],
    hardware_id: str,
    hardware_fingerprint: str,
) -> list[dict[str, Any]]:
    model_key = str(model["key"])
    slots = [slot for slot, owner in enumerate(SLOT_ORDER, start=1) if owner == implementation]
    required_cells = sorted(expected_cells(backend))
    model_offset = list(PRIMARY_MODELS).index(model_key) * 10_000
    rows: list[dict[str, Any]] = []
    for slot in slots:
        session_id = f"{model_key}-{backend}-{implementation}-session-{slot}"
        session_start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
            seconds=model_offset + (slot - 1) * 1000
        )
        ready_at = session_start + timedelta(seconds=10)
        cell_windows: list[dict[str, Any]] = []
        for sequence, (dataset, concurrency) in enumerate(required_cells, start=1):
            cell_start = session_start + timedelta(seconds=20 + (sequence - 1) * 70)
            cell_finish = cell_start + timedelta(seconds=60)
            cell_windows.append(
                {
                    "sequence": sequence,
                    "cell_id": f"{dataset}:c{concurrency}",
                    "dataset": dataset,
                    "concurrency": concurrency,
                    "started_at": iso_at(cell_start),
                    "finished_at": iso_at(cell_finish),
                }
            )
        measurement_start = parse_timestamp(cell_windows[0]["started_at"], "synthetic.cell.start")
        measurement_finish = parse_timestamp(cell_windows[-1]["finished_at"], "synthetic.cell.finish")
        shutdown_at = measurement_finish + timedelta(seconds=10)
        session_finish = shutdown_at + timedelta(seconds=10)
        pid = 10_000 + slot + (100 if backend == "cuda" else 200) + (0 if implementation == "A" else 10)
        pgid = pid
        process_marker, process_source = synthetic_process_identity(pid, session_start)
        runtime_rel = synthetic_log(
            root,
            f"sessions/{model_key}/{backend}/{implementation}/slot-{slot}.log",
            f"server session {session_id} loaded model and shut down cleanly after measured cells\n",
        )
        runtime_origin = f"/remote/{runtime_rel}"
        base_url = f"http://127.0.0.1:{9100 + slot}"
        models_body = {"object": "list", "data": [{"id": identity["request_model"]}]}
        ready_probe = synthetic_endpoint_probe(
            root,
            stem=f"sessions/{model_key}/{backend}/{implementation}/slot-{slot}.ready",
            url=f"{base_url}/v1/models",
            started_at=ready_at - timedelta(seconds=1),
            finished_at=ready_at,
            body=models_body,
        )
        model_probe = synthetic_endpoint_probe(
            root,
            stem=f"sessions/{model_key}/{backend}/{implementation}/slot-{slot}.models",
            url=f"{base_url}/v1/models",
            started_at=ready_at + timedelta(seconds=1),
            finished_at=ready_at + timedelta(seconds=2),
            body=models_body,
        )
        product_fields: dict[str, Any] = {}
        if implementation == "B":
            product_rel = f"sessions/{model_key}/{backend}/{implementation}/slot-{slot}.effective-config.json"
            product_origin = f"/remote/{product_rel}"
            write_json(
                root / product_rel,
                {
                    "schema_version": 1,
                    "entries": [],
                    "model_capabilities": {"architecture": model_key},
                    "hardware_capabilities": {"backend": backend},
                    "workload_profile": {"entrypoint": "serve"},
                    "decisions": [],
                    "admission": {"effective_max_concurrent": identity["typed_active_cap"]},
                },
            )
            argv = [
                "/selftest/legacy/ferrum",
                "serve",
                identity["model_origin_path"],
                "--backend",
                backend,
                "--effective-config-json",
                product_origin,
            ]
            product_fields = {
                "product_effective_config": product_rel,
                "product_effective_config_sha256": file_sha256(root / product_rel),
                "product_effective_config_origin_path": product_origin,
            }
        elif backend == "cuda":
            argv = [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                identity["model_origin_path"],
                "--served-model-name",
                identity["request_model"],
            ]
        else:
            argv = [
                "llama-server",
                "--model",
                identity["model_origin_path"],
                "--alias",
                identity["request_model"],
            ]
        rows.append(
            {
                "session_id": session_id,
                "implementation": implementation,
                "slot": slot,
                "sequence": slot,
                "hardware_id": hardware_id,
                "hardware_fingerprint": hardware_fingerprint,
                "effective_config": identity["effective_config"],
                "effective_config_sha256": identity["effective_config_sha256"],
                "typed_active_cap": identity["typed_active_cap"],
                "executed_binary_sha256": identity["binary_sha256"],
                "pid": pid,
                "pgid": pgid,
                "process_start_marker": process_marker,
                "process_start_source": process_source,
                "server_argv": argv,
                "base_url": base_url,
                "started_at": iso_at(session_start),
                "ready_at": iso_at(ready_at),
                "measurement_started_at": iso_at(measurement_start),
                "measurement_finished_at": iso_at(measurement_finish),
                "shutdown_started_at": iso_at(shutdown_at),
                "finished_at": iso_at(session_finish),
                "duration_sec": (session_finish - session_start).total_seconds(),
                "cell_windows": cell_windows,
                "ready_probe": ready_probe,
                "model_probe": model_probe,
                "model_key": model_key,
                "model_revision": model["lanes"][backend]["revision"],
                "model_files": locked_file_map(model, backend),
                "returncode": 0,
                "shutdown_clean": True,
                "runtime_log": runtime_rel,
                "runtime_log_sha256": file_sha256(root / runtime_rel),
                "runtime_log_origin_path": runtime_origin,
                **product_fields,
            }
        )
    return rows


def synthetic_workload(
    root: Path,
    *,
    model: dict[str, Any],
    model_key: str,
    backend: str,
    dataset: str,
    concurrency: int,
    cap: int,
    tokenizer_rel: str,
) -> dict[str, Any]:
    dataset_rel = f"workloads/{model_key}/{backend}/{dataset}/dataset.json"
    write_json(root / dataset_rel, {"schema_version": 1, "dataset_id": dataset, "frozen_prompts": 100})
    dataset_sha = file_sha256(root / dataset_rel)
    tokenizer_path = root / tokenizer_rel
    config_rel = f"workloads/{model_key}/{backend}/{dataset}/c{concurrency}.config.json"
    config = {
        "schema_version": 1,
        "dataset_id": dataset,
        "dataset_sha256": dataset_sha,
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "model_revision": model["lanes"][backend]["revision"],
        "seed": 9271,
        "max_output_tokens": 128,
        "ignore_eos": dataset == "random",
        "enable_thinking": False,
        "requested_concurrency": concurrency,
        "typed_active_cap": cap,
        "request_model": model["official_model_id"],
    }
    write_json(root / config_rel, config)
    tokenizer_source = model["lanes"][backend].get("tokenizer_source") or model["lanes"][backend]["semantic_source"]
    return {
        "dataset_id": dataset,
        "dataset_artifact": dataset_rel,
        "dataset_sha256": dataset_sha,
        "dataset_origin_path": f"/remote/{dataset_rel}",
        "tokenizer_id": tokenizer_source["repo"],
        "tokenizer_revision": tokenizer_source["revision"],
        "tokenizer_artifact": tokenizer_rel,
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "tokenizer_origin_path": f"/remote/tokenizers/{model_key}/{backend}",
        "effective_config": config_rel,
        "effective_config_sha256": file_sha256(root / config_rel),
    }


def synthetic_run_legacy(
    root: Path,
    *,
    model: dict[str, Any],
    model_key: str,
    backend: str,
    binary_sha256: str,
    hardware_id: str,
    hardware_fingerprint: str,
    memory_bytes: int,
    tokenizer_rel: str,
) -> dict[str, Any]:
    prompt_rel = f"run/{model_key}/{backend}/prompt.json"
    write_json(root / prompt_rel, {"prompt": "Return the word Paris and then stop."})
    prompt_sha = file_sha256(root / prompt_rel)
    tokenizer_sha = file_sha256(root / tokenizer_rel)
    config_rel = f"run/{model_key}/{backend}/effective-config.json"
    config = {
        "schema_version": 1,
        "model_revision": model["lanes"][backend]["revision"],
        "prompt_sha256": prompt_sha,
        "tokenizer_sha256": tokenizer_sha,
        "seed": 9271,
        "max_output_tokens": 128,
        "enable_thinking": False,
        "temperature": 0.0,
        "top_k": 20,
        "top_p": 0.8,
        "repeat_penalty": 1.0,
        "eos_policy": "model-metadata",
        "backend": backend,
        "request_model": model["official_model_id"],
        "memory_budget_bytes": memory_bytes // 2,
    }
    write_json(root / config_rel, config)
    tokenizer_source = model["lanes"][backend].get("tokenizer_source") or model["lanes"][backend]["semantic_source"]
    workload = {
        "prompt_artifact": prompt_rel,
        "prompt_sha256": prompt_sha,
        "tokenizer_artifact": tokenizer_rel,
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_id": tokenizer_source["repo"],
        "tokenizer_revision": tokenizer_source["revision"],
        "tokenizer_origin_path": f"/remote/tokenizers/{model_key}/{backend}",
        "effective_config": config_rel,
        "effective_config_sha256": file_sha256(root / config_rel),
    }
    sessions: list[dict[str, Any]] = []
    for slot, owner in enumerate(SLOT_ORDER, start=1):
        if owner != "B":
            continue
        session_start = datetime(2026, 2, 1, tzinfo=timezone.utc) + timedelta(seconds=(slot - 1) * 200)
        windows = []
        for repeat in range(1, 4):
            window_start = session_start + timedelta(seconds=10 + (repeat - 1) * 30)
            window_finish = window_start + timedelta(seconds=20)
            windows.append(
                {
                    "repeat": repeat,
                    "sample_id": f"run-{model_key}-{backend}-s{slot}-r{repeat}",
                    "started_at": iso_at(window_start),
                    "finished_at": iso_at(window_finish),
                }
            )
        session_finish = parse_timestamp(windows[-1]["finished_at"], "synthetic.run.window.finish") + timedelta(seconds=10)
        sessions.append(
            {
                "session_id": f"{model_key}-{backend}-run-session-{slot}",
                "slot": slot,
                "sequence": slot,
                "hardware_id": hardware_id,
                "hardware_fingerprint": hardware_fingerprint,
                "started_at": iso_at(session_start),
                "measurement_started_at": windows[0]["started_at"],
                "measurement_finished_at": windows[-1]["finished_at"],
                "finished_at": iso_at(session_finish),
                "duration_sec": (session_finish - session_start).total_seconds(),
                "sample_windows": windows,
            }
        )
    samples: list[dict[str, Any]] = []
    for session in sessions:
        for repeat in range(1, 4):
            slot = session["slot"]
            sample_id = f"run-{model_key}-{backend}-s{slot}-r{repeat}"
            window = session["sample_windows"][repeat - 1]
            command_started = parse_timestamp(window["started_at"], "synthetic.run.command.started")
            command_finished = parse_timestamp(window["finished_at"], "synthetic.run.command.finished")
            measurement_started = command_started + timedelta(seconds=2)
            measurement_finished = command_finished - timedelta(seconds=2)
            pid = 20_000 + slot * 10 + repeat
            process_marker, process_source = synthetic_process_identity(pid, command_started)
            stem = f"run/{model_key}/{backend}/slot-{slot}/repeat-{repeat}"
            inference_ms = 1300.0 + slot * 10.0 + repeat
            output_tps = 128_000.0 / inference_ms
            product_config_rel = f"{stem}.effective-config.json"
            product_config_origin = f"/remote/{product_config_rel}"
            write_json(
                root / product_config_rel,
                {
                    "schema_version": 1,
                    "entries": [],
                    "model_capabilities": {"model_key": model_key},
                    "hardware_capabilities": {"backend": backend},
                    "workload_profile": {"entrypoint": "run"},
                    "decisions": [],
                },
            )
            stdout_rel = synthetic_log(
                root,
                f"{stem}.stdout.jsonl",
                json.dumps(
                    {
                        "event": "assistant",
                        "turn": 0,
                        "content": "Paris",
                        "finish_reason": "length",
                        "n_tokens": 128,
                        "chunk_count": 1,
                        "ms": inference_ms,
                    }
                )
                + "\n",
            )
            stderr_rel = synthetic_log(root, f"{stem}.stderr", "run completed with product E2E evidence\n")
            stderr_origin = f"/remote/{stderr_rel}"
            resource_session = {
                "session_id": sample_id,
                "hardware_id": hardware_id,
                "base_url": f"process://{sample_id}",
                "started_at": window["started_at"],
                "finished_at": window["finished_at"],
                "pid": pid,
                "pgid": pid,
                "process_start_marker": process_marker,
                "process_start_source": process_source,
            }
            resources = synthetic_resource_evidence(
                root,
                stem=f"{stem}.resource",
                session=resource_session,
                cell_id="run:c1",
                backend=backend,
                measurement_started_at=iso_at(measurement_started),
                measurement_finished_at=iso_at(measurement_finished),
                runtime_log_origin_path=stderr_origin,
                memory_bytes=memory_bytes,
                concurrency=1,
                cap=1,
                process_probe=True,
            )
            samples.append(
                {
                    "sample_id": sample_id,
                    "session_id": session["session_id"],
                    "slot": slot,
                    "repeat": repeat,
                    "binary_sha256": binary_sha256,
                    "hardware_id": hardware_id,
                    "hardware_fingerprint": hardware_fingerprint,
                    "effective_config": config_rel,
                    "effective_config_sha256": file_sha256(root / config_rel),
                    "prompt_sha256": prompt_sha,
                    "tokenizer_sha256": tokenizer_sha,
                    "argv": [
                        "/selftest/legacy/ferrum",
                        "run",
                        model["official_model_id"],
                        "--prompt",
                        "Return the word Paris and then stop.",
                        "--tokenizer",
                        workload["tokenizer_origin_path"],
                        "--max-tokens",
                        "128",
                        "--disable-thinking",
                        "--seed",
                        "9271",
                        "--temperature",
                        "0.0",
                        "--top-k",
                        "20",
                        "--top-p",
                        "0.8",
                        "--repeat-penalty",
                        "1.0",
                        "--backend",
                        backend,
                        "--output-format",
                        "jsonl",
                        "--effective-config-json",
                        product_config_origin,
                    ],
                    "env": {"RUST_LOG": "info"},
                    "started_at": window["started_at"],
                    "finished_at": window["finished_at"],
                    "duration_sec": (command_finished - command_started).total_seconds(),
                    "measurement_started_at": iso_at(measurement_started),
                    "measurement_finished_at": iso_at(measurement_finished),
                    "pid": pid,
                    "pgid": pid,
                    "process_start_marker": process_marker,
                    "process_start_source": process_source,
                    "returncode": 0,
                    "stdout": stdout_rel,
                    "stdout_sha256": file_sha256(root / stdout_rel),
                    "stderr": stderr_rel,
                    "stderr_sha256": file_sha256(root / stderr_rel),
                    "stderr_origin_path": stderr_origin,
                    "output_tokens": 128,
                    "legacy_inference_e2e_ms": inference_ms,
                    "legacy_inference_e2e_output_tps": output_tps,
                    "cold_process_first_request": True,
                    "product_effective_config": product_config_rel,
                    "product_effective_config_sha256": file_sha256(root / product_config_rel),
                    "product_effective_config_origin_path": product_config_origin,
                    "resources": resources,
                }
            )
    e2e_ms_values = [float(sample["legacy_inference_e2e_ms"]) for sample in samples]
    e2e_tps_values = [float(sample["legacy_inference_e2e_output_tps"]) for sample in samples]
    return {
        "comparison_id": "g00-run-legacy",
        "workload": workload,
        "sessions": sessions,
        "measured_samples": 12,
        "completed_samples": 12,
        "error_count": 0,
        "output_token_count_source": "generated_tokens",
        "metric_boundary": "engine.infer_e2e",
        "summary": {
            "legacy_inference_e2e_ms": {
                "median": percentile_linear(e2e_ms_values, 0.5),
                "p95": percentile_linear(e2e_ms_values, 0.95),
            },
            "legacy_inference_e2e_output_tps": {
                "median": percentile_linear(e2e_tps_values, 0.5),
                "p95": percentile_linear(e2e_tps_values, 0.95),
            },
        },
        "samples": samples,
    }


def make_synthetic_root(root: Path) -> None:
    catalog, catalog_lanes = validate_models_catalog()
    preset_catalog = validate_presets_catalog(catalog)
    tree_sha = frozen_tree_sha()
    cuda_id = "cuda-4090-selftest"
    metal_id = "metal-32g-selftest"
    hardware = [
        make_synthetic_hardware(root, hardware_id=cuda_id, backend="cuda", policy_id="cuda-g0-1x-rtx4090"),
        make_synthetic_hardware(root, hardware_id=metal_id, backend="metal", policy_id="metal-reference-m1-max-32gb"),
    ]
    models = []
    synthetic_tokenizer_paths: dict[tuple[str, str], str] = {}
    for role, mapping in (("primary", PRIMARY_MODELS), ("supplemental", SUPPLEMENTAL_MODELS)):
        for key, model_id in mapping.items():
            lanes = {}
            for backend, hardware_id in (("cuda", cuda_id), ("metal", metal_id)):
                catalog_lane = catalog_lanes[(key, backend)]
                revision_spec = catalog_lane["revision"]
                revision = revision_spec.get("value") or synthetic_sha(f"revision-{key}-{backend}")[:40]
                reference = catalog_lane["reference"]
                semantic_revision = (
                    revision
                    if reference["semantic_revision"].get("status") == "same_as_weight_revision"
                    else reference["semantic_revision"].get("value")
                )
                require_git_sha(semantic_revision, f"selftest {key}/{backend} semantic revision", frozen=False)
                tokenizer_rel = f"tokenizers/{key}/{backend}/tokenizer.json"
                tokenizer_path = root / tokenizer_rel
                tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
                tokenizer_path.write_text(json.dumps({"model": key}) + "\n", encoding="utf-8")
                synthetic_tokenizer_paths[(key, backend)] = tokenizer_rel
                semantic_files = synthetic_semantic_files(
                    f"semantic-{key}-{backend}",
                    require_object(
                        reference.get("semantic_file_sha256"),
                        f"selftest {key}/{backend}.semantic_file_sha256",
                    ),
                )
                for semantic_file in semantic_files:
                    if (
                        semantic_file["path"] == "tokenizer.json"
                        and reference.get("tokenizer_repo") is None
                    ):
                        semantic_file["sha256"] = file_sha256(tokenizer_path)
                        semantic_file["size_bytes"] = tokenizer_path.stat().st_size
                weight_files = synthetic_catalog_files(
                    catalog_lane,
                    f"weights-{key}-{backend}",
                )
                if (
                    catalog_lane["repo"] == reference["semantic_repo"]
                    and revision == semantic_revision
                ):
                    semantic_by_path = {
                        str(item["path"]): item
                        for item in semantic_files
                    }
                    for weight_file in weight_files:
                        semantic_file = semantic_by_path.get(str(weight_file["path"]))
                        if semantic_file is not None:
                            weight_file["sha256"] = semantic_file["sha256"]
                            weight_file["size_bytes"] = semantic_file["size_bytes"]
                lanes[backend] = {
                    "catalog_lane_id": catalog_lane["id"],
                    "repo": catalog_lane["repo"],
                    "revision": revision,
                    "format": catalog_lane["format"],
                    "hardware_policy": catalog_lane["hardware_policy"],
                    "hardware_id": hardware_id,
                    "files": weight_files,
                    "semantic_source": {
                        "repo": reference["semantic_repo"],
                        "revision": semantic_revision,
                        "files": semantic_files,
                    },
                    "license": {
                        "spdx": "Apache-2.0",
                        "source": f"https://example.invalid/{key}/{backend}/license",
                    },
                }
                generation_rule = require_object(
                    reference.get("generation_config_source"),
                    f"selftest {key}/{backend}.generation_config_source",
                )
                generation_binding: dict[str, Any] = {
                    "source": "semantic_source",
                    "repo": reference["semantic_repo"],
                    "revision": semantic_revision,
                    "path": generation_rule["path"],
                    "policy": generation_rule["policy"],
                    "present": generation_rule["policy"] == "required",
                }
                if generation_rule["policy"] == "required":
                    generation_binding["file"] = copy.deepcopy(
                        next(
                            item
                            for item in semantic_files
                            if item["path"] == generation_rule["path"]
                        )
                    )
                lanes[backend]["generation_config"] = generation_binding
                if reference.get("tokenizer_repo") is not None:
                    tokenizer_source_files = synthetic_semantic_files(
                        f"tokenizer-{key}-{backend}",
                        require_object(
                            reference.get("tokenizer_file_sha256"),
                            f"selftest {key}/{backend}.tokenizer_file_sha256",
                        ),
                    )
                    for tokenizer_file in tokenizer_source_files:
                        if tokenizer_file["path"] == "tokenizer.json":
                            tokenizer_file["sha256"] = file_sha256(tokenizer_path)
                            tokenizer_file["size_bytes"] = tokenizer_path.stat().st_size
                    lanes[backend]["tokenizer_source"] = {
                        "repo": reference["tokenizer_repo"],
                        "revision": reference["tokenizer_revision"]["value"],
                        "files": tokenizer_source_files,
                    }
                chat_rule = require_object(
                    reference.get("chat_template_source"),
                    f"selftest {key}/{backend}.chat_template_source",
                )
                chat_source = lanes[backend][
                    "semantic_source"
                    if chat_rule["source"] == "semantic_source"
                    else "tokenizer_source"
                ]
                lanes[backend]["chat_template"] = {
                    "source": chat_rule["source"],
                    "repo": chat_source["repo"],
                    "revision": chat_source["revision"],
                    "path": chat_rule["path"],
                    "json_pointer": chat_rule["json_pointer"],
                    "container_sha256": chat_rule["container_sha256"],
                    "content_sha256": chat_rule["content_sha256"],
                    "content_bytes": 16,
                }
            row: dict[str, Any] = {
                "key": key,
                "official_model_id": model_id,
                "role": role,
                "lanes": lanes,
            }
            if role == "primary":
                preset_policy = preset_catalog["models"][key]
                row["generation_presets"] = copy.deepcopy(preset_policy["presets"])
                row["generation_preset_evidence"] = [
                    {
                        "path": path,
                        "sha256": digest,
                        "size_bytes": next(
                            item["size_bytes"]
                            for item in lanes["cuda"]["semantic_source"]["files"]
                            if item["path"] == path
                        ),
                    }
                    for path, digest in preset_policy["evidence"].items()
                ]
            models.append(row)
    resolution_lanes = []
    for model in models:
        for backend in ("cuda", "metal"):
            lane = model["lanes"][backend]
            catalog_lane = catalog_lanes[(model["key"], backend)]
            resolved: dict[str, Any] = {
                "catalog_lane_id": catalog_lane["id"],
                "model_id": catalog_lane["model_id"],
                "backend": backend,
                "format": catalog_lane["format"],
                "weight_source": {
                    "repo": lane["repo"],
                    "revision": lane["revision"],
                    "requested_revision": {
                        "status": catalog_lane["revision"]["status"],
                        "value": (
                            lane["revision"]
                            if catalog_lane["revision"]["status"] == "pinned"
                            else None
                        ),
                    },
                    "files": copy.deepcopy(lane["files"]),
                    "gated": False,
                    "license": {"hugging_face_id": lane["license"]["spdx"], "files": []},
                },
                "semantic_source": {
                    "repo": lane["semantic_source"]["repo"],
                    "revision": lane["semantic_source"]["revision"],
                    "requested_revision": {
                        "status": catalog_lane["reference"]["semantic_revision"]["status"],
                        "value": lane["semantic_source"]["revision"],
                    },
                    "files": copy.deepcopy(lane["semantic_source"]["files"]),
                    "gated": False,
                    "license": {"hugging_face_id": lane["license"]["spdx"], "files": []},
                },
                "generation_config": copy.deepcopy(lane["generation_config"]),
                "chat_template": copy.deepcopy(lane["chat_template"]),
            }
            if "tokenizer_source" in lane:
                resolved["tokenizer_source"] = {
                    "repo": lane["tokenizer_source"]["repo"],
                    "revision": lane["tokenizer_source"]["revision"],
                    "requested_revision": {
                        "status": "pinned",
                        "value": lane["tokenizer_source"]["revision"],
                    },
                    "files": copy.deepcopy(lane["tokenizer_source"]["files"]),
                    "gated": False,
                    "license": {"hugging_face_id": lane["license"]["spdx"], "files": []},
                }
            official_rule_raw = catalog_lane["reference"].get("official_upstream")
            if official_rule_raw is not None:
                official_rule = require_object(
                    official_rule_raw,
                    f"selftest {catalog_lane['id']}.official_upstream",
                )
                official_revision = official_rule["revision"]["value"]
                official_repo = official_rule["repo"]
                official = {
                    "repo": official_repo,
                    "revision": official_revision,
                    "mirror_repo": lane["semantic_source"]["repo"],
                    "mirror_revision": lane["semantic_source"]["revision"],
                    "model_request_url": f"https://huggingface.co/api/models/{official_repo}/revision/{official_revision}",
                    "tree_request_urls": [
                        f"https://huggingface.co/api/models/{official_repo}/tree/{official_revision}?recursive=true&expand=true"
                    ],
                    "gated": "manual",
                    "verification_method": "mirror_content_sha256_and_official_git_blob_oid",
                    "mirror_blob_oid_matches": [
                        {
                            "path": path,
                            "git_oid": official_rule["expected_git_oids"][path],
                            "size_bytes": official_rule["expected_size_bytes"][path],
                            "content_sha256": official_rule["expected_content_sha256"][path],
                        }
                        for path in official_rule["blob_oid_match_files"]
                    ],
                    "access_note": official_rule["access_note"],
                }
                resolved["official_upstream"] = official
                lane["official_upstream"] = copy.deepcopy(official)
            resolution_lanes.append(resolved)
    synthetic_requests: list[dict[str, Any]] = []
    for resolved_lane in resolution_lanes:
        for source_name in ("weight_source", "semantic_source", "tokenizer_source"):
            if source_name not in resolved_lane:
                continue
            source = resolved_lane[source_name]
            repo = source["repo"]
            revision = source["revision"]
            model_url = f"https://huggingface.co/api/models/{repo}/revision/{revision}"
            tree_url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&expand=true"
            source["model_request_url"] = model_url
            source["tree_request_urls"] = [tree_url]
            for kind, url in (("model-info", model_url), ("repo-tree", tree_url)):
                synthetic_requests.append(
                    {
                        "method": "GET",
                        "kind": kind,
                        "url": url,
                        "status": 200,
                        "response_bytes": 128,
                        "response_sha256": synthetic_sha(f"response-{url}"),
                    }
                )
            for file_row in source["files"]:
                if str(file_row["path"]).endswith((".safetensors", ".gguf")):
                    file_row["sha256_source"] = "hugging_face_lfs_oid"
                    file_row["lfs_oid"] = file_row["sha256"]
                    continue
                content_url = f"https://huggingface.co/{repo}/resolve/{revision}/{file_row['path']}"
                file_row["sha256_source"] = "downloaded_content"
                file_row["content_request_url"] = content_url
                synthetic_requests.append(
                    {
                        "method": "GET",
                        "kind": "metadata-file",
                        "url": content_url,
                        "status": 200,
                        "response_bytes": file_row["size_bytes"],
                        "response_sha256": file_row["sha256"],
                    }
                )
        if "official_upstream" in resolved_lane:
            official = resolved_lane["official_upstream"]
            for kind, url in (
                ("model-info", official["model_request_url"]),
                ("repo-tree", official["tree_request_urls"][0]),
            ):
                synthetic_requests.append(
                    {
                        "method": "GET",
                        "kind": kind,
                        "url": url,
                        "status": 200,
                        "response_bytes": 128,
                        "response_sha256": synthetic_sha(f"response-{url}"),
                    }
                )
    synthetic_requests = list(
        {
            (str(row["kind"]), str(row["url"])): row
            for row in synthetic_requests
        }.values()
    )
    resolution_path = root / "model-resolution.json"
    write_json(
        resolution_path,
        {
            "schema_version": 1,
            "artifact_type": "runtime_vnext_model_resolution",
            "generated_at": "2026-01-01T00:00:00Z",
            "source": {"git_sha": FROZEN_LEGACY_SHA, "dirty": False, "status_short": []},
            "catalog_id": catalog["catalog_id"],
            "catalog_path": MODELS_CATALOG_PATH.relative_to(REPO_ROOT).as_posix(),
            "catalog_sha256": file_sha256(MODELS_CATALOG_PATH),
            "resolver": {
                "path": MODEL_RESOLVER_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": file_sha256(MODEL_RESOLVER_PATH),
            },
            "policy": {
                "revision": "full_hugging_face_commit",
                "large_weight_downloaded": False,
                "lfs_sha256_source": "Hugging Face tree lfs.oid",
                "non_lfs_max_download_bytes": 32 * 1024 * 1024,
                "transport": "internal_selftest_fixture",
            },
            "lanes": resolution_lanes,
            "requests": synthetic_requests,
        },
    )
    expectations_rel = "legacy-correctness-expectations.json"
    write_json(root / expectations_rel, scenario_runner.internal_expectations_catalog())
    write_json(
        root / "models.lock.json",
        {
            "schema_version": 1,
            "source_git_sha": FROZEN_LEGACY_SHA,
            "source_tree_sha": tree_sha,
            "dirty_status": {"is_dirty": False, "status_short": []},
            "catalog_id": catalog["catalog_id"],
            "catalog_sha256": file_sha256(MODELS_CATALOG_PATH),
            "preset_catalog_id": preset_catalog["catalog_id"],
            "preset_catalog_sha256": file_sha256(PRESETS_CATALOG_PATH),
            "expectations_catalog": {
                "path": expectations_rel,
                "sha256": file_sha256(root / expectations_rel),
            },
            "model_resolution": {
                "path": resolution_path.relative_to(root).as_posix(),
                "sha256": file_sha256(resolution_path),
            },
            "hardware": hardware,
            "models": models,
        },
    )
    binaries = []
    for backend, hardware_id in (("cuda", cuda_id), ("metal", metal_id)):
        binary = root / "binaries" / backend / "ferrum"
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text(f"{backend} binary\n", encoding="utf-8")
        digest = file_sha256(binary)
        binaries.append(
            {
                "backend": backend,
                "hardware_id": hardware_id,
                "build_command": ["cargo", "build", "--release"],
                "cargo_features": [backend],
                "build_log": synthetic_log(root, f"binaries/{backend}/build.log"),
                "sha256_log": synthetic_log(root, f"binaries/{backend}/sha256.log", f"{digest}\n"),
                "binary_path": f"/selftest/{backend}/ferrum",
                "artifact_binary": f"binaries/{backend}/ferrum",
                "binary_sha256": digest,
            }
        )
    write_json(
        root / "legacy-binaries.json",
        {
            "schema_version": 1,
            "source_git_sha": FROZEN_LEGACY_SHA,
            "source_tree_sha": tree_sha,
            "dirty_status": {"is_dirty": False, "status_short": []},
            "binaries": binaries,
        },
    )
    indexed_models = {row["key"]: row for row in models}
    binary_map = {row["backend"]: row["binary_sha256"] for row in binaries}
    for model_key in PRIMARY_MODELS:
        model = indexed_models[model_key]
        for backend in ("cuda", "metal"):
            base = {
                "schema_version": 1,
                "source_git_sha": FROZEN_LEGACY_SHA,
                "source_tree_sha": tree_sha,
                "dirty_status": {"is_dirty": False, "status_short": []},
                "model_key": model_key,
                "backend": backend,
                "model_revision": model["lanes"][backend]["revision"],
                "model_files": locked_file_map(model, backend),
                "hardware_id": model["lanes"][backend]["hardware_id"],
                "binary_sha256": binary_map[backend],
            }
            blocked = backend == "metal" and model_key in {"m1-qwen35-4b", "m2-qwen35-35b-a3b"}
            lane = copy.deepcopy(base)
            if blocked:
                lane.update(
                    {
                        "status": "blocked",
                        "current_support": False,
                        "comparable": False,
                        "waiver": False,
                        "failure_class": "unsupported_architecture",
                        "reason": "legacy does not support this architecture on Metal",
                        "first_failure": "model loader rejected the architecture before execution",
                        "downstream_goal": "G08A",
                        "implementation_path": "add vNext operation providers",
                        "acceptance_path": "run the full product matrix",
                        "downstream_acceptance_pass_line": "FERRUM RUNTIME VNEXT G08A MODEL PASS: <out_dir>",
                        "attempted_command": ["ferrum", "run", model_key],
                        "attempted_returncode": 1,
                        "failure_log": synthetic_log(root, f"correctness/{model_key}/{backend}/failure.log"),
                    }
                )
            else:
                report_rel = f"correctness/{model_key}/{backend}/scenario-report.json"
                report_path = root / report_rel
                scenario_manifest = scenario_runner.make_internal_fixture_manifest(
                    root,
                    model_key=model_key,
                    backend=backend,
                    model_revision=model["lanes"][backend]["revision"],
                    model_files=locked_file_map(model, backend),
                    hardware_id=model["lanes"][backend]["hardware_id"],
                    binary_artifact=f"binaries/{backend}/ferrum",
                    models_lock_artifact="models.lock.json",
                )
                scenario_report = scenario_runner.collect_manifest(
                    scenario_manifest,
                    root,
                    report_path,
                    allow_internal_fixture=True,
                )
                write_json(report_path, scenario_report)
                lane.update(
                    {
                        "status": "pass",
                        "pass_line": f"{CORRECTNESS_PASS_PREFIX}: {model_key}/{backend}",
                        "scenario_report": {
                            "path": report_rel,
                            "sha256": file_sha256(report_path),
                        },
                    }
                )
            write_json(root / "correctness" / model_key / backend / "lane.json", lane)
            hardware_item = next(item for item in hardware if item["id"] == model["lanes"][backend]["hardware_id"])
            cap = 32 if backend == "cuda" else 16
            client = synthetic_benchmark_client(root, backend)
            identity_a = synthetic_server_identity(
                root,
                model=model,
                backend=backend,
                implementation="A",
                cap=cap,
                binary_sha256=synthetic_sha(f"external-{model_key}-{backend}"),
                memory_bytes=hardware_item["memory_bytes"],
            )
            identity_b = synthetic_server_identity(
                root,
                model=model,
                backend=backend,
                implementation="B",
                cap=cap,
                binary_sha256=binary_map[backend],
                memory_bytes=hardware_item["memory_bytes"],
            )
            sessions_a = synthetic_server_sessions(
                root,
                model=model,
                backend=backend,
                implementation="A",
                identity=identity_a,
                hardware_id=hardware_item["id"],
                hardware_fingerprint=hardware_item["fingerprint"],
            )
            sessions_b = synthetic_server_sessions(
                root,
                model=model,
                backend=backend,
                implementation="B",
                identity=identity_b,
                hardware_id=hardware_item["id"],
                hardware_fingerprint=hardware_item["fingerprint"],
            )
            all_sessions = sorted(sessions_a + sessions_b, key=lambda row: int(row["slot"]))
            external_cells = []
            perf_cells = []
            for dataset, concurrency in sorted(expected_cells(backend)):
                workload = synthetic_workload(
                    root,
                    model=model,
                    model_key=model_key,
                    backend=backend,
                    dataset=dataset,
                    concurrency=concurrency,
                    cap=cap,
                    tokenizer_rel=synthetic_tokenizer_paths[(model_key, backend)],
                )
                implementation_a = synthetic_http_implementation(
                    root,
                    implementation="A",
                    sessions=sessions_a,
                    workload=workload,
                    backend=backend,
                    dataset=dataset,
                    concurrency=concurrency,
                    memory_bytes=hardware_item["memory_bytes"],
                    cap=cap,
                    client=client,
                    artifact_key=model_key,
                )
                implementation_b = synthetic_http_implementation(
                    root,
                    implementation="B",
                    sessions=sessions_b,
                    workload=workload,
                    backend=backend,
                    dataset=dataset,
                    concurrency=concurrency,
                    memory_bytes=hardware_item["memory_bytes"],
                    cap=cap,
                    client=client,
                    artifact_key=model_key,
                )
                external_cells.append(
                    {
                        "dataset": dataset,
                        "concurrency": concurrency,
                        "workload": copy.deepcopy(workload),
                        "implementation": copy.deepcopy(implementation_a),
                    }
                )
                perf_cells.append(
                    {
                        "dataset": dataset,
                        "concurrency": concurrency,
                        "tokenizer_input_len_diff_pct": 0.0,
                        "workload": copy.deepcopy(workload),
                        "implementations": {
                            "A": copy.deepcopy(implementation_a),
                            "B": implementation_b,
                        },
                    }
                )
            external = {
                "schema_version": 1,
                "status": "pass",
                "model_key": model_key,
                "backend": backend,
                "hardware_id": model["lanes"][backend]["hardware_id"],
                "hardware_fingerprint": hardware_item["fingerprint"],
                "model_revision": model["lanes"][backend]["revision"],
                "model_files": locked_file_map(model, backend),
                "benchmark_client": client,
                "server_identity": identity_a,
                "sessions": sessions_a,
                "command_log": synthetic_log(root, f"external-baselines/{model_key}/{backend}/command.log"),
                "runtime_log": synthetic_log(root, f"external-baselines/{model_key}/{backend}/runtime.log"),
                "cells": external_cells,
            }
            write_json(root / "external-baselines" / model_key / backend / "summary.json", external)
            perf = copy.deepcopy(base)
            if blocked:
                perf.update(
                    {
                        "status": "blocked",
                        "comparable": False,
                        "reason": "legacy Ferrum lane is unavailable",
                        "downstream_goal": "G08A",
                    }
                )
            else:
                perf.update(
                    {
                        "status": "pass",
                        "comparable": True,
                        "hardware_fingerprint": hardware_item["fingerprint"],
                        "comparison_id": "g00-legacy-external",
                        "slot_order": SLOT_ORDER,
                        "benchmark_client": client,
                        "implementations": {"A": identity_a, "B": identity_b},
                        "sessions": all_sessions,
                        "command_log": synthetic_log(root, f"performance/{model_key}/{backend}/command.log"),
                        "runtime_log": synthetic_log(root, f"performance/{model_key}/{backend}/runtime.log"),
                        "cells": perf_cells,
                        "run_legacy": synthetic_run_legacy(
                            root,
                            model=model,
                            model_key=model_key,
                            backend=backend,
                            binary_sha256=binary_map[backend],
                            hardware_id=hardware_item["id"],
                            hardware_fingerprint=hardware_item["fingerprint"],
                            memory_bytes=hardware_item["memory_bytes"],
                            tokenizer_rel=synthetic_tokenizer_paths[(model_key, backend)],
                        ),
                    }
                )
            write_json(root / "performance" / model_key / backend / "summary.json", perf)
    def selftest_ref(rel: str) -> dict[str, Any]:
        return {"path": rel, "sha256": file_sha256(root / rel)}

    def selftest_cargo_record(
        name: str,
        index: int,
        package: str | None,
        fresh: bool,
        *,
        base: str | None = None,
    ) -> dict[str, Any]:
        base = base or f"build-timings/{name}/sample-{index + 1}"
        messages_rel = f"{base}/cargo-messages.jsonl"
        log_rel = f"{base}/cargo.log"
        timings_rel = f"{base}/cargo-timing.html"
        package_id = f"path+file:///self-test#{package or 'ferrum-cli'}@0.8.0"
        cargo_messages = [
            {
                "reason": "compiler-artifact",
                "package_id": package_id,
                "fresh": fresh,
                "filenames": ["/self-test/target/release/ferrum"],
                "target": {"kind": ["bin"]},
            },
            {"reason": "build-finished", "success": True},
        ]
        synthetic_log(root, messages_rel, "".join(json.dumps(row) + "\n" for row in cargo_messages))
        native_input = BUILD_SCENARIO_INPUTS.get(name, (None, None, None))[1]
        log_lines: list[str] = []
        if native_input:
            log_input = str(native_input).removeprefix("crates/ferrum-kernels/")
            log_lines.append(f"[self-test] cargo:rerun-if-changed={log_input}")
        if name in {"native-tu", "clean-release"}:
            log_lines.extend(
                [
                    "[vllm-marlin] compiling vllm_marlin/gptq_marlin_repack.cu -> /self-test/repack.o",
                    "[vllm-marlin] static lib built: /self-test/libvllm-marlin.a",
                ]
            )
        log_lines.append(f"Finished release [optimized] target(s) in {index + 1}.0s")
        synthetic_log(root, log_rel, "\n".join(log_lines) + "\n")
        synthetic_log(root, timings_rel, "<html><title>Cargo Build Timings</title></html>\n")
        summary = build_timing.parse_cargo_messages(root / messages_rel)
        return {
            "argv": copy.deepcopy(CUDA_BUILD_ARGV),
            "returncode": 0,
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": f"2026-01-01T00:00:0{index + 1}Z",
            "duration_sec": float(index + 1),
            "cargo_messages": selftest_ref(messages_rel),
            "log": selftest_ref(log_rel),
            "cargo_summary": summary,
            "cargo_timings": selftest_ref(timings_rel),
            "post_git_status": [],
        }

    prewarm = selftest_cargo_record("prewarm", 0, None, True)
    scenario_rows = []
    for name in sorted(BUILD_SCENARIOS):
        setup_kind, input_rel, package = BUILD_SCENARIO_INPUTS[name]
        samples = []
        for index in range(5):
            record = selftest_cargo_record(name, index, package, name == "noop")
            if setup_kind == "none":
                setup: dict[str, Any] = {"kind": "none"}
            elif setup_kind == "content-mutation":
                sample_base = Path(record["cargo_messages"]["path"]).parent
                before_rel = (sample_base / "input.before").as_posix()
                mutation_rel = (sample_base / "mutation.append").as_posix()
                during_rel = (sample_base / "input.during").as_posix()
                before_content = f"canonical self-test input for {name}\n".encode("ascii")
                mutation_content = f"\n// runtime-vnext-build-timing:{name}-{index + 1}\n".encode("ascii")
                (root / before_rel).write_bytes(before_content)
                (root / mutation_rel).write_bytes(mutation_content)
                (root / during_rel).write_bytes(before_content + mutation_content)
                before_sha = file_sha256(root / before_rel)
                setup = {
                    "kind": "content-mutation",
                    "input_path": input_rel,
                    "before_sha256": before_sha,
                    "during_sha256": file_sha256(root / during_rel),
                    "after_sha256": before_sha,
                    "mutation_kind": "append-comment",
                    "mutation_sha256": file_sha256(root / mutation_rel),
                    "mutation_bytes": len(mutation_content),
                    "before_input": selftest_ref(before_rel),
                    "mutation_artifact": selftest_ref(mutation_rel),
                    "during_input": selftest_ref(during_rel),
                    "before_mtime_ns": 100 + index,
                    "during_mtime_ns": 200 + index,
                    "after_mtime_ns": 100 + index,
                }
            else:
                clean_rel = f"build-timings/{name}/sample-{index + 1}/cargo-clean.log"
                synthetic_log(root, clean_rel, "Removed self-test release artifacts\n")
                setup = {
                    "kind": "cargo-clean",
                    "argv": ["cargo", "clean"],
                    "returncode": 0,
                    "log": selftest_ref(clean_rel),
                    "target_absent_after_clean": True,
                }
            binary_rel = f"build-timings/{name}/sample-{index + 1}/ferrum"
            binary_path = root / binary_rel
            binary_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / "binaries/cuda/ferrum", binary_path)
            record.update(
                {
                    "sample_id": f"{name}-{index + 1}",
                    "setup": setup,
                    "output_binary": selftest_ref(binary_rel),
                    "native_build": build_timing.native_build_summary(
                        (root / record["log"]["path"]).read_text(encoding="utf-8"), input_rel
                    ),
                }
            )
            samples.append(record)
        scenario_row: dict[str, Any] = {
            "name": name,
            "command": copy.deepcopy(CUDA_BUILD_ARGV),
            "samples": samples,
            "p50_sec": 3.0,
            "p95_sec": 5.0,
        }
        if setup_kind == "content-mutation":
            restore_base = f"build-timings/{name}/restore-verification"
            restore = selftest_cargo_record(name, 5, package, False, base=restore_base)
            restore_binary_rel = f"{restore_base}/ferrum"
            restore_binary_path = root / restore_binary_rel
            restore_binary_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / "binaries/cuda/ferrum", restore_binary_path)
            restore.update(
                {
                    "output_binary": selftest_ref(restore_binary_rel),
                    "restored_input": {
                        "input_path": input_rel,
                        "sha256": samples[0]["setup"]["before_sha256"],
                        "before_verification_mtime_ns": 100,
                        "verification_mtime_ns": 300,
                        "after_verification_mtime_ns": 100,
                    },
                }
            )
            scenario_row["restore_verification"] = restore
        scenario_rows.append(scenario_row)
    cuda_hardware = next(item for item in hardware if item["id"] == cuda_id)
    write_json(
        root / "build-timings" / "summary.json",
        {
            "schema_version": 1,
            "source_git_sha": FROZEN_LEGACY_SHA,
            "source_tree_sha": tree_sha,
            "dirty_status": {"is_dirty": False, "status_short": []},
            "collector": {
                "path": BUILD_TIMING_COLLECTOR_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": file_sha256(BUILD_TIMING_COLLECTOR_PATH),
            },
            "hardware_id": cuda_id,
            "hardware_fingerprint": cuda_hardware["fingerprint"],
            "prewarm": prewarm,
            "scenarios": scenario_rows,
        },
    )
    inventory_review = read_json(INVENTORY_REVIEW_PATH)
    scaffolding_findings = [
        {
            "category": "model_scaffolding_candidate",
            "path": review["path"],
            "line": line,
            "line_classification": "production",
            "text": review["symbol"],
            "symbol": review["symbol"],
        }
        for review in inventory_review["reviews"]
        for line in review["line_hints"]
    ]
    other_categories = [
        "qwen35_symbol",
        "architecture_named_api",
        "backend_trait_method",
        "backend_cfg",
        "ferrum_env_read",
        "legacy_factory_candidate",
        "model_runner_candidate",
        "product_decision_candidate",
    ]
    other_findings = [
        {
            "category": category,
            "path": "crates/selftest.rs",
            "line": index + 1,
            "line_classification": "production",
            "text": category,
            "symbol": f"symbol_{index}",
        }
        for index, category in enumerate(other_categories)
    ]
    inventory_findings = scaffolding_findings + other_findings
    inventory_category_counts: dict[str, int] = defaultdict(int)
    for finding in inventory_findings:
        inventory_category_counts[finding["category"]] += 1
    inventory_native = [
        {
            "tree_key": row["tree_key"],
            "content_root_sha256": row["content_root_sha256"],
            "is_large": True,
            "qualifying_reasons": ["self-test threshold"],
        }
        for row in inventory_review["large_native_content_roots"]
    ]
    write_json(
        root / "coupling-inventory.json",
        {
            "schema_version": 1,
            "analyzer": {
                "path": "scripts/release/runtime_vnext_inventory.py",
                "classification_values": ["production", "test", "generated", "vendor", "example", "fixture"],
                "loc_languages": [
                    "rust", "c", "cpp", "cuda", "cuda-header", "c-header", "cpp-header",
                    "objective-c", "objective-cpp", "metal", "python", "shell", "makefile", "dockerfile",
                ],
                "source_discovery_excluded_dirs": [".git", ".venv", "__pycache__", "node_modules", "target"],
                "identity_key": "sha256",
                "large_native_thresholds": {
                    "production_loc_gte": 10_000,
                    "source_bytes_gte": 5 * 1024 * 1024,
                    "translation_units_gte": 10,
                    "qualifier": "any",
                },
            },
            "root": "/selftest/frozen-worktree",
            "git": {
                "sha": FROZEN_LEGACY_SHA,
                "tree_sha": tree_sha,
                "dirty": False,
                "status_short": [],
            },
            "scope": {
                "scan_roots": ["crates", "scripts"],
                "scripts_policy": "all scripts (superset of product/release scripts)",
                "discovery_method": "git tracked plus untracked non-ignored files",
                "discovered_file_count": 2,
                "inventoried_file_count": 2,
                "coverage_ratio": 1.0,
                "file_count_by_root": {"crates": 1, "scripts": 1},
                "excluded_paths": [],
            },
            "files": [
                {
                    "path": "crates/selftest.rs",
                    "sha256": synthetic_sha("inventory-file"),
                    "content_id": f"sha256:{synthetic_sha('inventory-file')}",
                    "size_bytes": 16,
                    "language": "rust",
                    "classification": "production",
                    "logical_loc": 1,
                    "logical_loc_by_classification": {"production": 1},
                    "coupling_finding_count": len(inventory_findings),
                    "coupling_counts": dict(inventory_category_counts),
                },
                {
                    "path": "scripts/selftest.py",
                    "sha256": synthetic_sha("inventory-script"),
                    "content_id": f"sha256:{synthetic_sha('inventory-script')}",
                    "size_bytes": 16,
                    "language": "python",
                    "classification": "production",
                    "logical_loc": 1,
                    "logical_loc_by_classification": {"production": 1},
                    "coupling_finding_count": 0,
                    "coupling_counts": {},
                },
            ],
            "summary": {
                "file_count": 2,
                "file_count_by_classification": {"production": 2, "test": 0, "generated": 0, "vendor": 0, "example": 0, "fixture": 0},
                "logical_loc": 2,
                "logical_loc_by_classification": {"production": 2, "test": 0, "generated": 0, "vendor": 0, "example": 0, "fixture": 0},
                "coupling_finding_count": len(inventory_findings),
                "coupling_count_by_category": dict(inventory_category_counts),
                "native_source_tree_count": len(inventory_native),
                "large_third_party_native_source_count": len(inventory_native),
            },
            "content_identities": [],
            "move_tracking": {"baseline": None, "baseline_file_count": 0, "movement_count": 0, "movements": []},
            "coupling": {
                "findings": inventory_findings,
                "potential_run_serve_duplicate_decisions": [
                    {"symbol": "resolve_model", "occurrences": [{"path": "crates/selftest.rs", "line": 1}, {"path": "scripts/selftest.py", "line": 1}]}
                ],
            },
            "large_native_source_trees": inventory_native,
        },
    )
    bug_catalog = read_json(BUG_CATALOG_PATH)
    families = []
    concrete_case_count = 0
    for family_raw in bug_catalog["families"]:
        family_cases = []
        for catalog_case in family_raw["cases"]:
            case_id = catalog_case["id"]
            concrete_case_count += 1
            input_rel = f"historical-bugs/{case_id}/input.json"
            mutation_rel = f"historical-bugs/{case_id}/mutation.patch"
            failure_rel = f"historical-bugs/{case_id}/failure.log"
            input_path = root / input_rel
            mutation_path = root / mutation_rel
            failure_signature = f"synthetic-{case_id}-failure"
            synthetic_log(root, input_rel, json.dumps({"case_id": case_id}) + "\n")
            synthetic_log(root, mutation_rel, f"synthetic mutation for {case_id}\n")
            synthetic_log(root, failure_rel, f"{failure_signature}\n")
            evidence = [
                {"kind": "commit", "ref": commit["sha"]}
                for commit in catalog_case.get("commits", [])
            ] + [
                {
                    "kind": "artifact",
                    "ref": artifact,
                    "sha256": file_sha256(REPO_ROOT / artifact),
                    "size_bytes": (REPO_ROOT / artifact).stat().st_size,
                }
                for artifact in catalog_case.get("historical_artifacts", [])
            ]
            if not evidence:
                source_ref = catalog_case["reproducer_paths"][0]
                evidence.append(
                    {
                        "kind": "source",
                        "ref": source_ref,
                        "sha256": file_sha256(REPO_ROOT / source_ref),
                        "size_bytes": (REPO_ROOT / source_ref).stat().st_size,
                    }
                )
            family_cases.append(
                {
                    "id": case_id,
                    "failure_class": catalog_case["failure_class"],
                    "status": "frozen",
                    "entrypoints": catalog_case["entrypoints"],
                    "backends": catalog_case["backends"],
                    "source_evidence": evidence,
                    "reproducer": {
                        "input_path": input_rel,
                        "input_sha256": file_sha256(input_path),
                        "mutation_path": mutation_rel,
                        "mutation_sha256": file_sha256(mutation_path),
                        "mutation_kind": "fault_injection",
                        "failure_log": failure_rel,
                        "failure_signature": failure_signature,
                        "command": ["python3", "historical_reproducer.py", case_id, input_rel, mutation_rel],
                        "returncode": 1,
                        "started_at": "2026-01-01T00:00:00Z",
                        "finished_at": "2026-01-01T00:00:01Z",
                        "duration_sec": 1.0,
                        "expected_invariant": "synthetic invariant",
                    },
                }
            )
        families.append({"id": family_raw["id"], "cases": family_cases})
    write_json(
        root / "historical-bug-corpus.json",
        {
            "schema_version": 1,
            "source_git_sha": FROZEN_LEGACY_SHA,
            "source_tree_sha": tree_sha,
            "dirty_status": {"is_dirty": False, "status_short": []},
            "catalog_id": bug_catalog["catalog_id"],
            "catalog_sha256": file_sha256(BUG_CATALOG_PATH),
            "family_count": 15,
            "concrete_case_count": concrete_case_count,
            "orphan_case_count": 0,
            "duplicate_case_count": 0,
            "families": families,
        },
    )


_ACTIVE_SELFTEST: dict[str, Any] | None = None


def clone_tree(source: Path, destination: Path) -> None:
    shutil.rmtree(destination, ignore_errors=True)
    copied = subprocess.run(
        ["cp", "-cR", str(source), str(destination)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if copied.returncode != 0:
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(source, destination)


class SelftestTreeRestorer:
    def __init__(self, root: Path, pristine: Path) -> None:
        self.root = root
        self.pristine = pristine
        clone_tree(root, pristine)
        self.entries = self._tree_state(pristine)

    @staticmethod
    def _tree_state(root: Path) -> dict[str, tuple[str, int, str | None]]:
        states: dict[str, tuple[str, int, str | None]] = {}
        pending = [root]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    rel = path.relative_to(root).as_posix()
                    metadata = entry.stat(follow_symlinks=False)
                    mode = stat.S_IMODE(metadata.st_mode)
                    if stat.S_ISDIR(metadata.st_mode):
                        states[rel] = ("directory", mode, None)
                        pending.append(path)
                    elif stat.S_ISREG(metadata.st_mode):
                        states[rel] = ("file", mode, file_sha256(path))
                    elif stat.S_ISLNK(metadata.st_mode):
                        states[rel] = ("symlink", mode, os.readlink(path))
                    else:
                        states[rel] = ("other", mode, None)
        return states

    @staticmethod
    def _remove_path(path: Path) -> None:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISDIR(metadata.st_mode):
            shutil.rmtree(path)
        else:
            path.unlink()

    def _restore_entry(self, rel: str, expected: tuple[str, int, str | None]) -> None:
        source = self.pristine / rel
        destination = self.root / rel
        kind, mode, target = expected
        destination.parent.mkdir(parents=True, exist_ok=True)
        if kind == "directory":
            destination.mkdir(exist_ok=True)
            destination.chmod(mode)
        elif kind == "file":
            shutil.copy2(source, destination, follow_symlinks=False)
        elif kind == "symlink":
            require(target is not None, f"self-test pristine symlink lacks target: {rel}")
            os.symlink(target, destination)
        else:
            raise AssertionError(f"unsupported pristine self-test entry type: {rel}: {kind}")

    def restore(self) -> None:
        current = self._tree_state(self.root)
        changed = {
            rel
            for rel in current.keys() | self.entries.keys()
            if current.get(rel) != self.entries.get(rel)
        }
        directory_mode_changes = {
            rel
            for rel in changed
            if current.get(rel, (None,))[0] == "directory"
            and self.entries.get(rel, (None,))[0] == "directory"
        }
        for rel in directory_mode_changes:
            (self.root / rel).chmod(self.entries[rel][1])
        changed -= directory_mode_changes
        for rel in sorted(changed, key=lambda value: len(Path(value).parts), reverse=True):
            self._remove_path(self.root / rel)
        for rel in sorted(changed, key=lambda value: len(Path(value).parts)):
            expected = self.entries.get(rel)
            if expected is not None:
                self._restore_entry(rel, expected)


def selftest_mutation_scope(name: str) -> str:
    if name == "dirty":
        return "legacy-binaries"
    if name in SELFTEST_MODEL_LOCK_MUTATIONS:
        return "models-lock"
    if name in SELFTEST_HISTORICAL_MUTATIONS:
        return "historical-bugs"
    if name == "model-sha":
        return "correctness-identity"
    if name == "cuda-primary-blocked":
        return "correctness-matrix"
    if name in SELFTEST_SCENARIO_MUTATIONS:
        return "correctness-scenario"
    if name in SELFTEST_ROOT_INTEGRATION_MUTATIONS:
        return "root-integration"
    if name in SELFTEST_PERFORMANCE_MUTATIONS:
        return "performance-lane"
    if name in SELFTEST_INVENTORY_MUTATIONS:
        return "inventory"
    if name in SELFTEST_BUILD_MUTATIONS:
        return "build-timings"
    raise AssertionError(f"self-test mutation has no scoped validator: {name}")


def validate_selftest_mutation(root: Path, name: str, context: dict[str, Any]) -> None:
    try:
        scope = selftest_mutation_scope(name)
        if scope == "legacy-binaries":
            validate_legacy_binaries(root, context["hardware"])
        elif scope == "models-lock":
            validate_models_lock(root, allow_synthetic=True)
        elif scope == "historical-bugs":
            validate_historical_bugs(root)
        elif scope == "correctness-identity":
            model = context["models"]["m3-qwen3-30b-a3b"]
            lane = read_json(root / "correctness/m3-qwen3-30b-a3b/cuda/lane.json")
            validate_lane_identity(
                lane,
                label="correctness.m3-qwen3-30b-a3b.cuda",
                model=model,
                backend="cuda",
                hardware=context["hardware"],
                binaries=context["binaries"],
            )
        elif scope == "correctness-matrix":
            validate_correctness(
                root,
                context["models"],
                context["hardware"],
                context["binaries"],
                allow_synthetic=True,
                expectations_binding=context["models_lock"]["expectations_catalog"],
            )
        elif scope == "correctness-scenario":
            model = context["models"]["m3-qwen3-30b-a3b"]
            lane = read_json(root / "correctness/m3-qwen3-30b-a3b/cuda/lane.json")
            label = "correctness.m3-qwen3-30b-a3b.cuda"
            validate_lane_identity(
                lane,
                label=label,
                model=model,
                backend="cuda",
                hardware=context["hardware"],
                binaries=context["binaries"],
            )
            validate_pass_lane(
                root,
                lane,
                label,
                allow_synthetic=True,
                expectations_binding=context["models_lock"]["expectations_catalog"],
            )
        elif scope == "performance-lane":
            validate_performance_lane(
                root,
                model=context["models"]["m3-qwen3-30b-a3b"],
                backend="cuda",
                correctness_status="pass",
                hardware=context["hardware"],
                binaries=context["binaries"],
                allow_synthetic=True,
            )
        elif scope == "root-integration":
            _validate_root_impl(root, allow_synthetic=True)
        elif scope == "inventory":
            validate_inventory(root)
        elif scope == "build-timings":
            validate_build_timings(root, context["hardware"], context["binaries"])
        else:
            raise AssertionError(f"unsupported self-test validator scope: {scope}")
    except BaselineError:
        raise
    except (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError) as exc:
        raise BaselineError(f"malformed baseline artifact ({type(exc).__name__}): {exc}") from exc


def expect_reject(root: Path, name: str, mutate: Callable[[Path], None], marker: str) -> None:
    runtime = _ACTIVE_SELFTEST
    fast = runtime is not None and runtime["mode"] == "fast"
    case = root if fast else root.parent / f"reject-{name}"
    if not fast:
        clone_tree(root, case)
    try:
        mutate(case)
        try:
            if fast:
                validate_selftest_mutation(case, name, runtime["context"])
            else:
                validate_root(case, allow_synthetic=True)
        except BaselineError as exc:
            require(marker.lower() in str(exc).lower(), f"{name} rejected for unexpected reason: {exc}")
            if runtime is not None:
                require(name not in runtime["mutations"], f"duplicate self-test mutation assertion: {name}")
                runtime["mutations"].append(name)
                scope = selftest_mutation_scope(name) if fast else "full-root"
                runtime["validator_counts"][scope] += 1
            return
        raise AssertionError(f"{name} unexpectedly passed")
    finally:
        if fast:
            runtime["restorer"].restore()
        else:
            shutil.rmtree(case, ignore_errors=True)


def mutate_json(path: Path, update: Callable[[dict[str, Any]], None]) -> None:
    data = read_json(path)
    update(data)
    write_json(path, data)


def mutate_perf_sidecar(root: Path, update: Callable[[dict[str, Any]], None]) -> None:
    summary_path = root / "performance/m3-qwen3-30b-a3b/cuda/summary.json"
    summary = read_json(summary_path)
    record = summary["cells"][0]["implementations"]["B"]["reports"][0]
    report_path = root / record["raw_report"]
    report = read_json(report_path)
    proxy = {"repeats": report["repeat_metrics"]}
    update(proxy)
    report["repeat_metrics"] = proxy["repeats"]
    write_json(report_path, report)
    record["raw_report_sha256"] = file_sha256(report_path)
    write_json(summary_path, summary)


def mutate_perf_report(root: Path, update: Callable[[dict[str, Any]], None]) -> None:
    summary_path = root / "performance/m3-qwen3-30b-a3b/cuda/summary.json"
    summary = read_json(summary_path)
    record = summary["cells"][0]["implementations"]["B"]["reports"][0]
    report_path = root / record["raw_report"]
    mutate_json(report_path, update)
    record["raw_report_sha256"] = file_sha256(report_path)
    write_json(summary_path, summary)


def make_itl_partial_subset_look_valid(report: dict[str, Any]) -> None:
    evidence = report["itl_evidence_per_request"][0][0]
    evidence["transport_coalesced_output_chunks"] = 1
    evidence["eligibility"] = "transport_coalesced"
    repeat = report["repeat_metrics"][0]
    repeat["itl_eligible_requests"] = 99
    repeat["itl_ineligible_requests"] = 1
    repeat["itl_eligibility_counts"]["eligible"] = 99
    repeat["itl_eligibility_counts"]["transport_coalesced"] = 1


def first_perf_resource(root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    summary_path = root / "performance/m3-qwen3-30b-a3b/cuda/summary.json"
    summary = read_json(summary_path)
    resource = summary["cells"][0]["implementations"]["B"]["reports"][0]["resources"]
    return summary_path, summary, resource


def mutate_perf_resource_summary(root: Path, update: Callable[[dict[str, Any]], None]) -> None:
    summary_path, summary, resource = first_perf_resource(root)
    update(resource["summary"])
    write_json(summary_path, summary)


def mutate_perf_resource_observations(
    root: Path,
    update: Callable[[list[dict[str, Any]]], None],
    *,
    summary_update: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    summary_path, summary, resource = first_perf_resource(root)
    observation_path = root / resource["observations"]
    rows = [json.loads(line) for line in observation_path.read_text(encoding="utf-8").splitlines()]
    update(rows)
    observation_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    resource["observations_sha256"] = file_sha256(observation_path)
    if summary_update is not None:
        summary_update(resource["summary"])
    write_json(summary_path, summary)


def mutate_perf_server_probe(
    root: Path,
    field: str,
    update: Callable[[dict[str, Any]], None],
) -> None:
    summary_path = root / "performance/m3-qwen3-30b-a3b/cuda/summary.json"
    summary = read_json(summary_path)
    session = next(row for row in summary["sessions"] if row["implementation"] == "B")
    evidence = session[field]
    body_path = root / evidence["body"]
    mutate_json(body_path, update)
    evidence["body_sha256"] = file_sha256(body_path)
    receipt_path = root / evidence["receipt"]
    receipt = read_json(receipt_path)
    receipt["body_sha256"] = file_sha256(body_path)
    receipt["body_size_bytes"] = body_path.stat().st_size
    write_json(receipt_path, receipt)
    evidence["receipt_sha256"] = file_sha256(receipt_path)
    write_json(summary_path, summary)


def mutate_benchmark_client(root: Path, update: Callable[[dict[str, Any]], None]) -> None:
    path = root / "external-baselines/m3-qwen3-30b-a3b/cuda/summary.json"
    mutate_json(path, lambda data: update(data["benchmark_client"]))


def mutate_perf_server_config(root: Path, update: Callable[[dict[str, Any]], None]) -> None:
    summary_path = root / "performance/m3-qwen3-30b-a3b/cuda/summary.json"
    summary = read_json(summary_path)
    identity = summary["implementations"]["B"]
    config_path = root / identity["effective_config"]
    mutate_json(config_path, update)
    digest = file_sha256(config_path)
    identity["effective_config_sha256"] = digest
    for session in summary["sessions"]:
        if session["implementation"] == "B":
            session["effective_config_sha256"] = digest
    write_json(summary_path, summary)


def mutate_perf_product_config(root: Path, update: Callable[[dict[str, Any]], None]) -> None:
    summary_path = root / "performance/m3-qwen3-30b-a3b/cuda/summary.json"
    summary = read_json(summary_path)
    session = next(row for row in summary["sessions"] if row["implementation"] == "B")
    config_path = root / session["product_effective_config"]
    mutate_json(config_path, update)
    session["product_effective_config_sha256"] = file_sha256(config_path)
    write_json(summary_path, summary)


def overlap_perf_sessions(root: Path) -> None:
    path = root / "performance/m3-qwen3-30b-a3b/cuda/summary.json"
    data = read_json(path)
    sessions = sorted(data["sessions"], key=lambda row: row["slot"])
    previous, current = sessions[0], sessions[1]
    current["started_at"] = previous["started_at"]
    current["duration_sec"] = (
        parse_timestamp(current["finished_at"], "selftest.overlap.finished")
        - parse_timestamp(current["started_at"], "selftest.overlap.started")
    ).total_seconds()
    write_json(path, data)


def overlap_run_sessions(root: Path) -> None:
    path = root / "performance/m3-qwen3-30b-a3b/cuda/summary.json"
    data = read_json(path)
    sessions = data["run_legacy"]["sessions"]
    current = sessions[1]
    current["started_at"] = sessions[0]["started_at"]
    current["duration_sec"] = (
        parse_timestamp(current["finished_at"], "selftest.run-overlap.finished")
        - parse_timestamp(current["started_at"], "selftest.run-overlap.started")
    ).total_seconds()
    write_json(path, data)


def mutate_perf_probe_receipt(
    root: Path,
    field: str,
    update: Callable[[dict[str, Any]], None],
) -> None:
    summary_path = root / "performance/m3-qwen3-30b-a3b/cuda/summary.json"
    summary = read_json(summary_path)
    session = next(row for row in summary["sessions"] if row["implementation"] == "B")
    evidence = session[field]
    receipt_path = root / evidence["receipt"]
    mutate_json(receipt_path, update)
    evidence["receipt_sha256"] = file_sha256(receipt_path)
    write_json(summary_path, summary)


def mutate_scenario_report(case: Path, update: Callable[[dict[str, Any]], None]) -> None:
    lane_path = case / "correctness/m3-qwen3-30b-a3b/cuda/lane.json"
    lane = read_json(lane_path)
    report_path = artifact_path(case, lane["scenario_report"]["path"], "selftest.scenario_report.path")
    report = read_json(report_path)
    update(report)
    write_json(report_path, report)
    lane["scenario_report"]["sha256"] = file_sha256(report_path)
    write_json(lane_path, lane)


def mutate_first_historical_reproducer(
    root: Path,
    update: Callable[[dict[str, Any]], None],
) -> None:
    path = root / "historical-bug-corpus.json"
    data = read_json(path)
    update(data["families"][0]["cases"][0]["reproducer"])
    write_json(path, data)


def make_first_historical_mutation_identical(root: Path) -> None:
    path = root / "historical-bug-corpus.json"
    data = read_json(path)
    reproducer = data["families"][0]["cases"][0]["reproducer"]
    input_path = artifact_path(root, reproducer["input_path"], "selftest.historical.input")
    mutation_path = artifact_path(root, reproducer["mutation_path"], "selftest.historical.mutation")
    mutation_path.write_bytes(input_path.read_bytes())
    reproducer["mutation_sha256"] = file_sha256(mutation_path)
    write_json(path, data)


def mutate_model_resolution(root: Path, update: Callable[[dict[str, Any]], None]) -> None:
    resolution_path = root / "model-resolution.json"
    resolution = read_json(resolution_path)
    update(resolution)
    write_json(resolution_path, resolution)
    lock_path = root / "models.lock.json"
    lock = read_json(lock_path)
    lock["model_resolution"]["sha256"] = file_sha256(resolution_path)
    write_json(lock_path, lock)


def mutate_model_binding_chain(
    root: Path,
    *,
    update_lock: Callable[[dict[str, Any]], None],
    update_resolution: Callable[[dict[str, Any]], None],
) -> None:
    resolution_path = root / "model-resolution.json"
    resolution = read_json(resolution_path)
    update_resolution(resolution)
    write_json(resolution_path, resolution)
    lock_path = root / "models.lock.json"
    lock = read_json(lock_path)
    update_lock(lock)
    lock["model_resolution"]["sha256"] = file_sha256(resolution_path)
    write_json(lock_path, lock)


def mutate_preset_semantic_chain(root: Path) -> None:
    forged = "0" * 64

    def update_resolution(data: dict[str, Any]) -> None:
        affected_urls: set[str] = set()
        for lane in data["lanes"]:
            if lane["catalog_lane_id"] not in {"M1-CUDA", "M1-METAL"}:
                continue
            readme = next(row for row in lane["semantic_source"]["files"] if row["path"] == "README.md")
            readme["sha256"] = forged
            affected_urls.add(readme["content_request_url"])
        for request in data["requests"]:
            if request["kind"] == "metadata-file" and request["url"] in affected_urls:
                request["response_sha256"] = forged

    def update_lock(data: dict[str, Any]) -> None:
        model = next(row for row in data["models"] if row["key"] == "m1-qwen35-4b")
        next(row for row in model["generation_preset_evidence"] if row["path"] == "README.md")[
            "sha256"
        ] = forged
        for lane in model["lanes"].values():
            next(row for row in lane["semantic_source"]["files"] if row["path"] == "README.md")[
                "sha256"
            ] = forged

    mutate_model_binding_chain(
        root,
        update_lock=update_lock,
        update_resolution=update_resolution,
    )


def mutate_chat_template_chain(root: Path) -> None:
    forged = "0" * 64

    def update_resolution(data: dict[str, Any]) -> None:
        lane = next(row for row in data["lanes"] if row["catalog_lane_id"] == "M1-CUDA")
        lane["chat_template"]["content_sha256"] = forged

    def update_lock(data: dict[str, Any]) -> None:
        model = next(row for row in data["models"] if row["key"] == "m1-qwen35-4b")
        model["lanes"]["cuda"]["chat_template"]["content_sha256"] = forged

    mutate_model_binding_chain(
        root,
        update_lock=update_lock,
        update_resolution=update_resolution,
    )


def mutate_generation_config_chain(root: Path) -> None:
    def update_resolution(data: dict[str, Any]) -> None:
        lane = next(row for row in data["lanes"] if row["catalog_lane_id"] == "M1-CUDA")
        lane["generation_config"]["policy"] = "required"
        lane["generation_config"]["present"] = True

    def update_lock(data: dict[str, Any]) -> None:
        model = next(row for row in data["models"] if row["key"] == "m1-qwen35-4b")
        model["lanes"]["cuda"]["generation_config"]["policy"] = "required"
        model["lanes"]["cuda"]["generation_config"]["present"] = True

    mutate_model_binding_chain(
        root,
        update_lock=update_lock,
        update_resolution=update_resolution,
    )


def mutate_llama_official_chain(root: Path) -> None:
    forged = "0" * 64

    def update_resolution(data: dict[str, Any]) -> None:
        lane = next(row for row in data["lanes"] if row["catalog_lane_id"] == "S3-LLAMA31-8B-CUDA")
        lane["official_upstream"]["mirror_blob_oid_matches"][0]["content_sha256"] = forged

    def update_lock(data: dict[str, Any]) -> None:
        model = next(row for row in data["models"] if row["key"] == "llama31-8b-compat")
        model["lanes"]["cuda"]["official_upstream"]["mirror_blob_oid_matches"][0][
            "content_sha256"
        ] = forged

    mutate_model_binding_chain(
        root,
        update_lock=update_lock,
        update_resolution=update_resolution,
    )


def mutate_hardware_probe(root: Path, update: Callable[[dict[str, Any]], None]) -> None:
    lock_path = root / "models.lock.json"
    lock = read_json(lock_path)
    reference = lock["hardware"][0]["probe"]
    probe_path = artifact_path(root, reference["path"], "selftest.hardware.probe")
    probe = read_json(probe_path)
    update(probe)
    write_json(probe_path, probe)
    reference["sha256"] = file_sha256(probe_path)
    write_json(lock_path, lock)


def mutate_build_cargo_messages(
    root: Path,
    scenario_name: str,
    *,
    restore_verification: bool,
    update: Callable[[list[dict[str, Any]]], None],
    recompute_summary: bool,
) -> None:
    summary_path = root / "build-timings/summary.json"
    summary = read_json(summary_path)
    scenario = next(row for row in summary["scenarios"] if row["name"] == scenario_name)
    record = scenario["restore_verification"] if restore_verification else scenario["samples"][0]
    messages_ref = record["cargo_messages"]
    messages_path = artifact_path(root, messages_ref["path"], "selftest.build.cargo_messages")
    messages = [json.loads(line) for line in messages_path.read_text(encoding="utf-8").splitlines() if line]
    update(messages)
    messages_path.write_text("".join(json.dumps(row) + "\n" for row in messages), encoding="utf-8")
    messages_ref["sha256"] = file_sha256(messages_path)
    if recompute_summary:
        record["cargo_summary"] = build_timing.parse_cargo_messages(messages_path)
    write_json(summary_path, summary)


def tamper_build_content_evidence(root: Path) -> None:
    summary_path = root / "build-timings/summary.json"
    summary = read_json(summary_path)
    scenario = next(row for row in summary["scenarios"] if row["name"] == "core-ptx")
    setup = scenario["samples"][0]["setup"]
    during_path = artifact_path(root, setup["during_input"]["path"], "selftest.build.during_input")
    during_path.write_bytes(during_path.read_bytes() + b"forged")
    digest = file_sha256(during_path)
    setup["during_input"]["sha256"] = digest
    setup["during_sha256"] = digest
    write_json(summary_path, summary)


def tamper_restore_binary(root: Path) -> None:
    summary_path = root / "build-timings/summary.json"
    summary = read_json(summary_path)
    scenario = next(row for row in summary["scenarios"] if row["name"] == "rust-model-leaf")
    binary_ref = scenario["restore_verification"]["output_binary"]
    binary_path = artifact_path(root, binary_ref["path"], "selftest.build.restore_binary")
    binary_path.write_bytes(binary_path.read_bytes() + b"forged")
    binary_ref["sha256"] = file_sha256(binary_path)
    write_json(summary_path, summary)


def tamper_hardware_raw_output(root: Path) -> None:
    lock_path = root / "models.lock.json"
    lock = read_json(lock_path)
    reference = lock["hardware"][0]["probe"]
    probe_path = artifact_path(root, reference["path"], "selftest.hardware.probe")
    probe = read_json(probe_path)
    gpu_command = next(row for row in probe["commands"] if row["kind"] == "gpu")
    stdout_path = artifact_path(probe_path.parent, gpu_command["stdout"], "selftest.hardware.gpu.stdout")
    stdout_path.write_text(
        stdout_path.read_text(encoding="utf-8").replace("24576", "24000"),
        encoding="utf-8",
    )
    gpu_command["stdout_sha256"] = file_sha256(stdout_path)
    write_json(probe_path, probe)
    reference["sha256"] = file_sha256(probe_path)
    write_json(lock_path, lock)


def remove_first_shard_index(root: Path) -> None:
    lock_path = root / "models.lock.json"
    lock = read_json(lock_path)
    model = next(row for row in lock["models"] if row["key"] == "m1-qwen35-4b")
    lane = model["lanes"]["cuda"]
    lane["files"] = [row for row in lane["files"] if row["path"] != "model.safetensors.index.json"]
    write_json(lock_path, lock)

    def update(resolution: dict[str, Any]) -> None:
        resolved = next(row for row in resolution["lanes"] if row["catalog_lane_id"] == lane["catalog_lane_id"])
        resolved["weight_source"]["files"] = [
            row for row in resolved["weight_source"]["files"] if row["path"] != "model.safetensors.index.json"
        ]

    mutate_model_resolution(root, update)


def mutate_expected_sha_non_lfs_chain(root: Path) -> None:
    def update(resolution: dict[str, Any]) -> None:
        lane = next(
            row
            for row in resolution["lanes"]
            if row["catalog_lane_id"] == "M2-METAL"
        )
        weight = lane["weight_source"]
        file_row = next(
            row
            for row in weight["files"]
            if row["path"] == "Qwen3.5-35B-A3B-Q4_K_S.gguf"
        )
        content_url = (
            f"https://huggingface.co/{weight['repo']}/resolve/"
            f"{weight['revision']}/{file_row['path']}"
        )
        file_row.pop("lfs_oid", None)
        file_row["sha256_source"] = "downloaded_content"
        file_row["content_request_url"] = content_url
        resolution["requests"].append(
            {
                "method": "GET",
                "kind": "metadata-file",
                "url": content_url,
                "status": 200,
                "response_bytes": file_row["size_bytes"],
                "response_sha256": file_row["sha256"],
            }
        )

    mutate_model_resolution(root, update)


def replace_exact_string(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {key: replace_exact_string(child, old, new) for key, child in value.items()}
    if isinstance(value, list):
        return [replace_exact_string(child, old, new) for child in value]
    return new if value == old else value


def mutate_cross_lane_session_id_conflict(root: Path) -> None:
    source_summary = read_json(
        root / "performance/m1-qwen35-4b/cuda/summary.json"
    )
    conflicting_id = next(
        row["session_id"]
        for row in source_summary["sessions"]
        if row["implementation"] == "A" and row["slot"] == 1
    )
    summary_paths = [
        root / "external-baselines/m2-qwen35-35b-a3b/cuda/summary.json",
        root / "performance/m2-qwen35-35b-a3b/cuda/summary.json",
    ]
    summaries = [read_json(path) for path in summary_paths]
    performance_summary = summaries[1]
    old_id = next(
        row["session_id"]
        for row in performance_summary["sessions"]
        if row["implementation"] == "A" and row["slot"] == 1
    )
    summaries = [replace_exact_string(summary, old_id, conflicting_id) for summary in summaries]

    def resource_rows(value: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        if isinstance(value, dict):
            if "observations" in value and "observations_sha256" in value:
                found.append(value)
            for child in value.values():
                found.extend(resource_rows(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(resource_rows(child))
        return found

    rewritten_observations: set[str] = set()
    for summary in summaries:
        for resources in resource_rows(summary):
            observations_rel = str(resources["observations"])
            observations_path = root / observations_rel
            if observations_rel not in rewritten_observations:
                rows = [
                    replace_exact_string(json.loads(line), old_id, conflicting_id)
                    for line in observations_path.read_text(encoding="utf-8").splitlines()
                    if line
                ]
                observations_path.write_text(
                    "".join(
                        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                        for row in rows
                    ),
                    encoding="utf-8",
                )
                rewritten_observations.add(observations_rel)
            resources["observations_sha256"] = file_sha256(observations_path)
    for summary_path, summary in zip(summary_paths, summaries):
        write_json(summary_path, summary)


def mutate_inventory_review_binding(root: Path) -> None:
    path = root / "coupling-inventory.json"
    data = read_json(path)
    finding = next(
        row
        for row in data["coupling"]["findings"]
        if row["category"] == "model_scaffolding_candidate"
    )
    finding["line"] += 1
    write_json(path, data)


def add_artifact_index_symlink(root: Path) -> None:
    os.symlink("models.lock.json", root / "artifact-index-selftest-link")


def external_path_self_test() -> int:
    external_root = (Path(tempfile.gettempdir()) / "ferrum-vnext-external-path-selftest").resolve()
    external_out = external_root / "correctness" / "report.json"
    argv = [
        str(SCENARIO_RUNNER_PATH),
        "--manifest",
        str(external_root / "manifest.json"),
        "--artifact-root",
        str(external_root),
        "--out",
        str(external_out),
    ]
    resolved_root, resolved_out = validate_child_output_paths(argv, "external-path-selftest.positive")
    require(resolved_root == external_root and resolved_out == external_out, "external path positive case mismatch")
    negative_cases = (
        (REPO_ROOT / "docs/release/runtime-vnext", REPO_ROOT / "docs/release/runtime-vnext/report.json", "--artifact-root"),
        (external_root, REPO_ROOT / "docs/release/runtime-vnext/report.json", "--out"),
    )
    for artifact_root, out, marker in negative_cases:
        candidate = list(argv)
        candidate[candidate.index("--artifact-root") + 1] = str(artifact_root)
        candidate[candidate.index("--out") + 1] = str(out)
        try:
            validate_child_output_paths(candidate, "external-path-selftest.negative")
        except BaselineError as exc:
            require(marker in str(exc) and "outside REPO_ROOT" in str(exc), f"external path negative case rejected unexpectedly: {exc}")
        else:
            raise AssertionError(f"repo-internal child path unexpectedly passed: {marker}")
    try:
        require_external_artifact_path(REPO_ROOT / "g00-baseline", "external-path-selftest.baseline-out")
    except BaselineError as exc:
        require("outside REPO_ROOT" in str(exc), f"baseline --out negative case rejected unexpectedly: {exc}")
    else:
        raise AssertionError("repo-internal baseline --out unexpectedly passed")
    print(PATH_SELFTEST_PASS_LINE)
    return 0


def _run_self_test(mode: str) -> int:
    global _ACTIVE_SELFTEST

    require(mode in {"fast", "full"}, f"unsupported self-test mode: {mode}")
    require(external_path_self_test() == 0, "external path self-test failed")
    with tempfile.TemporaryDirectory(prefix="ferrum-vnext-g00-selftest-") as tmp:
        root = Path(tmp) / "valid"
        root.mkdir()
        make_synthetic_root(root)
        manifest = validate_root(root, allow_synthetic=True)
        require(manifest["status"] == "pass", "valid synthetic baseline did not pass")
        try:
            reject_synthetic_artifacts(root)
            raise AssertionError("canonical validation unexpectedly accepted synthetic evidence")
        except BaselineError as exc:
            require("synthetic/self-test" in str(exc), f"synthetic fixture rejected for unexpected reason: {exc}")

        context: dict[str, Any] = {}
        restorer: SelftestTreeRestorer | None = None
        if mode == "fast":
            models_lock, hardware, models = validate_models_lock(root, allow_synthetic=True)
            binaries = validate_legacy_binaries(root, hardware)
            context = {
                "models_lock": models_lock,
                "hardware": hardware,
                "models": models,
                "binaries": binaries,
            }
            restorer = SelftestTreeRestorer(root, Path(tmp) / "pristine")
        runtime = {
            "mode": mode,
            "root": root,
            "context": context,
            "restorer": restorer,
            "mutations": [],
            "validator_counts": defaultdict(int),
        }
        _ACTIVE_SELFTEST = runtime

        expect_reject(
            root,
            "dirty",
            lambda case: mutate_json(case / "legacy-binaries.json", lambda data: data["dirty_status"].update({"is_dirty": True})),
            "dirty baseline",
        )
        expect_reject(
            root,
            "stale",
            lambda case: mutate_json(case / "models.lock.json", lambda data: data.update({"source_git_sha": "0" * 40})),
            "stale",
        )
        expect_reject(
            root,
            "preset-policy-drift",
            lambda case: mutate_json(
                case / "models.lock.json",
                lambda data: data["models"][0]["generation_presets"]["P_THINKING"].update(
                    {"presence_penalty": 0.0}
                ),
            ),
            "differ from checked-in preset policy",
        )
        expect_reject(
            root,
            "expectations-lock-sha",
            lambda case: mutate_json(
                case / "models.lock.json",
                lambda data: data["expectations_catalog"].update({"sha256": "0" * 64}),
            ),
            "expectations_catalog synthetic artifact mismatch",
        )
        expect_reject(
            root,
            "preset-semantic-chain-forgery",
            mutate_preset_semantic_chain,
            "content sha256 mismatch for readme.md",
        )
        expect_reject(
            root,
            "chat-template-chain-forgery",
            mutate_chat_template_chain,
            "chat_template.content_sha256 differs from catalog",
        )
        expect_reject(
            root,
            "generation-config-chain-forgery",
            mutate_generation_config_chain,
            "generation_config.policy mismatch",
        )
        expect_reject(
            root,
            "llama-official-chain-forgery",
            mutate_llama_official_chain,
            "official_upstream content sha256 mismatch for config.json",
        )
        expect_reject(
            root,
            "model-resolution-revision",
            lambda case: mutate_model_resolution(
                case,
                lambda data: data["lanes"][0]["weight_source"].update({"revision": "0" * 40}),
            ),
            "revision differs",
        )
        expect_reject(
            root,
            "model-resolution-resolver-sha",
            lambda case: mutate_model_resolution(
                case,
                lambda data: data["resolver"].update({"sha256": "0" * 64}),
            ),
            "resolver sha256 mismatch",
        )
        expect_reject(
            root,
            "model-resolution-shard-index",
            remove_first_shard_index,
            "missing required catalog file model.safetensors.index.json",
        )
        expect_reject(
            root,
            "model-resolution-expected-sha-non-lfs",
            mutate_expected_sha_non_lfs_chain,
            "requires Hugging Face LFS identity",
        )
        expect_reject(
            root,
            "hardware-derived-fingerprint",
            lambda case: mutate_json(
                case / "models.lock.json",
                lambda data: data["hardware"][0].update({"device_name": "NVIDIA GeForce RTX 4090 forged"}),
            ),
            "fingerprint_material mismatch",
        )
        expect_reject(
            root,
            "hardware-probe-command",
            lambda case: mutate_hardware_probe(
                case,
                lambda data: data["commands"][1].update({"argv": ["true"]}),
            ),
            "argv mismatch",
        )
        expect_reject(
            root,
            "hardware-probe-raw-derivation",
            tamper_hardware_raw_output,
            "normalized facts are not derived from raw outputs",
        )
        expect_reject(
            root,
            "historical-identical-mutation",
            make_first_historical_mutation_identical,
            "mutation must differ",
        )
        expect_reject(
            root,
            "historical-missing-signature",
            lambda case: mutate_first_historical_reproducer(
                case,
                lambda reproducer: reproducer.update({"failure_signature": "absent-signature"}),
            ),
            "failure signature is absent",
        )
        expect_reject(
            root,
            "historical-success-returncode",
            lambda case: mutate_first_historical_reproducer(
                case,
                lambda reproducer: reproducer.update({"returncode": 0}),
            ),
            "non-zero failing result",
        )
        expect_reject(
            root,
            "model-sha",
            lambda case: mutate_json(case / "correctness/m3-qwen3-30b-a3b/cuda/lane.json", lambda data: data["model_files"].update({next(iter(data["model_files"])): "0" * 64})),
            "model_files",
        )
        expect_reject(
            root,
            "cuda-primary-blocked",
            lambda case: mutate_json(
                case / "correctness/m1-qwen35-4b/cuda/lane.json",
                lambda data: data.update({"status": "blocked"}),
            ),
            "CUDA primary lane must PASS",
        )
        expect_reject(
            root,
            "scenario-no-ferrum-argv",
            lambda case: mutate_scenario_report(case, lambda data: data["commands"][0].update({"argv": ["python", "run", "model"]})),
            "execute ferrum",
        )
        expect_reject(
            root,
            "scenario-missing-tools",
            lambda case: mutate_scenario_report(case, lambda data: data["scenarios"].pop(9)),
            "exactly C01-C21",
        )
        expect_reject(
            root,
            "scenario-missing-schema",
            lambda case: mutate_scenario_report(case, lambda data: data["scenarios"][13]["assertions"].pop("valid_json_count")),
            "valid schema count",
        )
        expect_reject(
            root,
            "scenario-missing-utf8",
            lambda case: mutate_scenario_report(case, lambda data: data["scenarios"][16]["variants"].pop("emoji")),
            "variants.emoji",
        )
        expect_reject(
            root,
            "scenario-missing-thinking",
            lambda case: mutate_scenario_report(case, lambda data: data["scenarios"][18]["variants"].pop("soft-think")),
            "soft-think",
        )
        expect_reject(
            root,
            "scenario-missing-cancel",
            lambda case: mutate_scenario_report(case, lambda data: data["scenarios"][8]["variants"].pop("cancel")),
            "variants.cancel",
        )
        expect_reject(
            root,
            "scenario-artifact-sha",
            lambda case: mutate_scenario_report(case, lambda data: data["scenarios"][0]["artifacts"][0].update({"sha256": "0" * 64})),
            "artifact SHA256 mismatch",
        )
        expect_reject(
            root,
            "scenario-fake-pass",
            lambda case: mutate_scenario_report(case, lambda data: data["scenarios"][0].update({"status": "PASS"})),
            "status invalid",
        )
        expect_reject(
            root,
            "cross-hardware",
            lambda case: mutate_json(case / "performance/m3-qwen3-30b-a3b/cuda/summary.json", lambda data: data.update({"hardware_id": "metal-64g-selftest"})),
            "cross-hardware",
        )
        expect_reject(
            root,
            "repeat-count",
            lambda case: mutate_perf_sidecar(case, lambda data: data["repeats"].pop()),
            "three repeat metrics",
        )
        expect_reject(
            root,
            "expected-request-accounting",
            lambda case: mutate_perf_sidecar(
                case,
                lambda data: data["repeats"][0].update({"completed_requests": 99}),
            ),
            "expected_requests must equal completed_requests + errored_requests",
        )
        expect_reject(
            root,
            "expected-requests-absolute",
            lambda case: mutate_perf_sidecar(
                case,
                lambda data: data["repeats"][0].update({"expected_requests": 99}),
            ),
            "expected_requests must be 100",
        )
        expect_reject(
            root,
            "expected-requests-missing",
            lambda case: mutate_perf_sidecar(
                case,
                lambda data: data["repeats"][0].pop("expected_requests"),
            ),
            "expected_requests must be a non-negative integer",
        )
        expect_reject(
            root,
            "errors",
            lambda case: mutate_json(case / "performance/m3-qwen3-30b-a3b/cuda/summary.json", lambda data: data["cells"][0]["implementations"]["B"].update({"error_count": 1})),
            "error_count",
        )
        expect_reject(
            root,
            "usage",
            lambda case: mutate_json(case / "performance/m3-qwen3-30b-a3b/cuda/summary.json", lambda data: data["cells"][0]["implementations"]["B"].update({"output_token_count_source": "estimated"})),
            "must be usage",
        )
        expect_reject(
            root,
            "ab-identity-swap",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: data["implementations"].update(
                    {"A": data["implementations"]["B"], "B": data["implementations"]["A"]}
                ),
            ),
            "implementation mismatch",
        )
        expect_reject(
            root,
            "duplicate-server-session",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: data["sessions"][1].update({"session_id": data["sessions"][0]["session_id"]}),
            ),
            "duplicate session_id",
        )
        expect_reject(
            root,
            "server-session-same-lane-overlap",
            overlap_perf_sessions,
            "server sessions overlap globally",
        )
        expect_reject(
            root,
            "cross-lane-session-id-conflict",
            mutate_cross_lane_session_id_conflict,
            "global server session",
        )
        expect_reject(
            root,
            "server-cell-window-overlap",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: data["sessions"][1]["cell_windows"][1].update(
                    {"started_at": data["sessions"][1]["cell_windows"][0]["started_at"]}
                ),
            ),
            "cell_windows overlap",
        )
        expect_reject(
            root,
            "report-outside-cell-window",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: data["cells"][0]["implementations"]["B"]["reports"][0].update(
                    {
                        "started_at": iso_at(
                            parse_timestamp(
                                data["cells"][0]["implementations"]["B"]["reports"][0]["started_at"],
                                "selftest.report.started",
                            )
                            + timedelta(seconds=1)
                        ),
                        "duration_sec": 59.0,
                    }
                ),
            ),
            "command/report window must equal",
        )
        expect_reject(
            root,
            "server-process-start-marker",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: next(
                    row for row in data["sessions"] if row["implementation"] == "B"
                ).update({"process_start_marker": "darwin:999:forged"}),
            ),
            "not derived from raw OS identity evidence",
        )
        expect_reject(
            root,
            "ready-probe-returncode",
            lambda case: mutate_perf_probe_receipt(
                case, "ready_probe", lambda receipt: receipt.update({"returncode": 1})
            ),
            "did not capture HTTP 200 success",
        )
        expect_reject(
            root,
            "loaded-model-probe",
            lambda case: mutate_perf_server_probe(
                case,
                "model_probe",
                lambda body: body["data"][0].update({"id": "forged-model"}),
            ),
            "loaded model identity mismatch",
        )
        expect_reject(
            root,
            "server-effective-config-model",
            lambda case: mutate_perf_server_config(
                case, lambda config: config.update({"model_revision": "0" * 40})
            ),
            "effective_config.model_revision mismatch",
        )
        expect_reject(
            root,
            "server-product-config-cap",
            lambda case: mutate_perf_product_config(
                case, lambda config: config["admission"].update({"effective_max_concurrent": 1})
            ),
            "product_effective_config active cap mismatch",
        )
        expect_reject(
            root,
            "server-effective-config-argv",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: remove_option_with_value(
                    next(row for row in data["sessions"] if row["implementation"] == "B")["server_argv"],
                    "--effective-config-json",
                ),
            ),
            "--effective-config-json",
        )
        expect_reject(
            root,
            "benchmark-client-tree-binding",
            lambda case: mutate_benchmark_client(
                case, lambda client: client.update({"source_tree_sha": "0" * 40})
            ),
            "collector_identity_sha256 is not derived",
        )
        expect_reject(
            root,
            "benchmark-client-rust-allowlist",
            lambda case: mutate_benchmark_client(
                case,
                lambda client: client["production_rust_diff"].append(
                    "crates/ferrum-engine/src/continuous_engine.rs"
                ),
            ),
            "production_rust_diff must equal",
        )
        expect_reject(
            root,
            "bench-canonical-argv",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: data["cells"][0]["implementations"]["B"]["reports"][0]["bench_argv"].remove("--require-ci"),
            ),
            "missing canonical --require-ci",
        )
        expect_reject(
            root,
            "dataset-sha",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: data["cells"][0]["workload"].update({"dataset_sha256": "0" * 64}),
            ),
            "dataset_artifact sha256 mismatch",
        )
        expect_reject(
            root,
            "tokenizer-sha",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: data["cells"][0]["workload"].update({"tokenizer_sha256": "0" * 64}),
            ),
            "not locked by models.lock",
        )
        expect_reject(
            root,
            "config-sha",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: data["cells"][0]["workload"].update({"effective_config_sha256": "0" * 64}),
            ),
            "effective_config sha256 mismatch",
        )
        expect_reject(
            root,
            "hardware-fingerprint",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: data.update({"hardware_fingerprint": "0" * 64}),
            ),
            "hardware_fingerprint mismatch",
        )
        expect_reject(
            root,
            "active-cap",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: data["implementations"]["B"].update({"typed_active_cap": 31}),
            ),
            "typed_active_cap mismatch",
        )
        expect_reject(
            root,
            "observed-active",
            lambda case: mutate_perf_resource_observations(
                case,
                lambda rows: rows[2].update({"active_requests": 33}),
            ),
            "raw active requests exceed",
        )
        expect_reject(
            root,
            "zero-observed-active",
            lambda case: mutate_perf_resource_observations(
                case,
                lambda rows: [row.update({"active_requests": 0}) for row in rows[1:-1]],
            ),
            "never observed active work",
        )
        expect_reject(
            root,
            "resource-observation-pid",
            lambda case: mutate_perf_resource_observations(
                case,
                lambda rows: rows[2].update({"pid": rows[2]["pid"] + 1}),
            ),
            "pid/pgid mismatch",
        )
        expect_reject(
            root,
            "resource-observation-process-start",
            lambda case: mutate_perf_resource_observations(
                case,
                lambda rows: (
                    rows[0].update({"process_start_marker": "darwin:999:forged"}),
                    [row.update({"process_start_marker": "darwin:999:forged"}) for row in rows[1:-1]],
                ),
            ),
            "resource header process_start_marker mismatch",
        )
        expect_reject(
            root,
            "resource-summary-forgery",
            lambda case: mutate_perf_resource_summary(
                case, lambda summary: summary.update({"peak_memory_bytes": 2 * 1024 * 1024})
            ),
            "summary is not derived from raw resource observations",
        )
        expect_reject(
            root,
            "resource-http-process-probe",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: (
                    remove_option_with_value(
                        data["cells"][0]["implementations"]["B"]["reports"][0]["resources"]["sampler_argv"],
                        "--active-probe-format",
                    ),
                    data["cells"][0]["implementations"]["B"]["reports"][0]["resources"]["sampler_argv"].extend(
                        ["--active-probe-format", "process"]
                    ),
                ),
            ),
            "HTTP measurements cannot use process-alive",
        )
        expect_reject(
            root,
            "raw-report-sha",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: data["cells"][0]["implementations"]["B"]["reports"][0].update({"raw_report_sha256": "0" * 64}),
            ),
            "raw_report sha256 mismatch",
        )
        expect_reject(
            root,
            "raw-report-metric",
            lambda case: mutate_perf_report(case, lambda data: data.pop("goodput_rps")),
            "goodput_rps must be a json object",
        )
        expect_reject(
            root,
            "raw-report-usage",
            lambda case: mutate_perf_report(case, lambda data: data.update({"output_token_count_source": "stream_chunks"})),
            "output_token_count_source must be usage",
        )
        expect_reject(
            root,
            "raw-report-quality",
            lambda case: mutate_perf_report(case, lambda data: data["bad_output_per_run"].__setitem__(0, 1)),
            "bad_output_per_run must contain 3 zeros",
        )
        expect_reject(
            root,
            "itl-evidence-missing",
            lambda case: mutate_perf_report(case, lambda data: data.pop("itl_evidence_per_request")),
            "itl_evidence_per_request must be a JSON array",
        )
        expect_reject(
            root,
            "itl-evidence-cardinality",
            lambda case: mutate_perf_report(case, lambda data: data["itl_evidence_per_request"][0].pop()),
            "itl_evidence_per_request[0] invalid",
        )
        expect_reject(
            root,
            "itl-source-forged",
            lambda case: mutate_perf_report(
                case,
                lambda data: data["itl_evidence_per_request"][0][0].update(
                    {"source": "engine_token_events"}
                ),
            ),
            "source must be sse_delta_events",
        )
        expect_reject(
            root,
            "itl-usage-event-claimed-eligible",
            lambda case: mutate_perf_report(
                case,
                lambda data: data["itl_evidence_per_request"][0][0].update(
                    {"usage_output_tokens": data["itl_evidence_per_request"][0][0]["output_events"] - 1}
                ),
            ),
            "eligibility is not derived from raw ITL evidence",
        )
        expect_reject(
            root,
            "itl-interval-claimed-eligible",
            lambda case: mutate_perf_report(
                case,
                lambda data: data["itl_evidence_per_request"][0][0].update(
                    {"observed_intervals": data["itl_evidence_per_request"][0][0]["observed_intervals"] - 1}
                ),
            ),
            "observed_intervals is not derived from output events",
        )
        expect_reject(
            root,
            "itl-coalesced-claimed-eligible",
            lambda case: mutate_perf_report(
                case,
                lambda data: data["itl_evidence_per_request"][0][0].update(
                    {"transport_coalesced_output_chunks": 1}
                ),
            ),
            "eligibility is not derived from raw ITL evidence",
        )
        expect_reject(
            root,
            "itl-repeat-counts-forged",
            lambda case: mutate_perf_report(
                case,
                lambda data: data["repeat_metrics"][0].update({"itl_eligible_requests": 99}),
            ),
            "ITL eligible/ineligible request count mismatch",
        )
        expect_reject(
            root,
            "itl-repeat-intervals-forged",
            lambda case: mutate_perf_report(
                case,
                lambda data: data["repeat_metrics"][0].update(
                    {"itl_expected_intervals": data["repeat_metrics"][0]["itl_expected_intervals"] - 1}
                ),
            ),
            "eligible ITL interval totals mismatch",
        )
        expect_reject(
            root,
            "itl-ineligible-partial-percentiles",
            lambda case: mutate_perf_report(case, make_itl_partial_subset_look_valid),
            "ineligible ITL repeat must not expose partial percentiles",
        )
        expect_reject(
            root,
            "swap-growth",
            lambda case: mutate_perf_resource_observations(
                case,
                lambda rows: rows[-3].update({"swap_used_bytes": 4096}),
                summary_update=lambda summary: summary.update({"swap_end_bytes": 4096}),
            ),
            "swap growth must be zero",
        )
        expect_reject(
            root,
            "duplicate-repeat-ordinal",
            lambda case: mutate_perf_sidecar(
                case,
                lambda data: data["repeats"][1].update({"repeat": 1}),
            ),
            "repeat must be 2",
        )
        expect_reject(
            root,
            "warmup-error",
            lambda case: mutate_perf_sidecar(
                case,
                lambda data: data["repeats"][0].update({"warmup_completed": 9, "warmup_errored": 1}),
            ),
            "warmup must be 10/10",
        )
        expect_reject(
            root,
            "bench-thinking-payload",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: remove_option_with_value(
                    data["cells"][0]["implementations"]["B"]["reports"][0]["bench_argv"],
                    "--enable-thinking",
                ),
            ),
            "--enable-thinking must be false",
        )
        expect_reject(
            root,
            "bench-env-hash",
            lambda case: mutate_perf_report(case, lambda data: data.update({"env_hash": "sha256:" + "0" * 64})),
            "env_hash is not derived",
        )
        expect_reject(
            root,
            "run-real-command",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: data["run_legacy"]["samples"][0]["argv"].remove("--disable-thinking"),
            ),
            "must explicitly disable thinking",
        )
        expect_reject(
            root,
            "run-session-global-overlap",
            overlap_run_sessions,
            "sessions overlap globally",
        )
        expect_reject(
            root,
            "run-command-window-binding",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: data["run_legacy"]["samples"][0].update(
                    {
                        "started_at": iso_at(
                            parse_timestamp(
                                data["run_legacy"]["samples"][0]["started_at"],
                                "selftest.run.started",
                            )
                            + timedelta(seconds=1)
                        ),
                        "duration_sec": 19.0,
                    }
                ),
            ),
            "command window does not match",
        )
        expect_reject(
            root,
            "inventory-source-coverage",
            lambda case: mutate_json(
                case / "coupling-inventory.json",
                lambda data: data["files"].pop(),
            ),
            "coupling-inventory file count mismatch",
        )
        expect_reject(
            root,
            "inventory-review-binding",
            mutate_inventory_review_binding,
            "inventory review line hints are stale",
        )
        expect_reject(
            root,
            "artifact-index-empty-file",
            lambda case: (case / "artifact-index-selftest-empty").write_bytes(b""),
            "artifact file is empty",
        )
        expect_reject(
            root,
            "artifact-index-symlink",
            add_artifact_index_symlink,
            "artifact tree contains forbidden symlink",
        )
        expect_reject(
            root,
            "build-real-command",
            lambda case: mutate_json(
                case / "build-timings/summary.json",
                lambda data: data["scenarios"][0].update({"command": ["true"]}),
            ),
            "command is not canonical",
        )
        expect_reject(
            root,
            "build-raw-summary",
            lambda case: mutate_json(
                case / "build-timings/summary.json",
                lambda data: data["scenarios"][0]["samples"][0]["cargo_summary"].update(
                    {"compiler_artifact_count": 99}
                ),
            ),
            "summary cannot be recomputed",
        )
        expect_reject(
            root,
            "build-finished-failure",
            lambda case: mutate_build_cargo_messages(
                case,
                "noop",
                restore_verification=False,
                update=lambda rows: next(row for row in rows if row.get("reason") == "build-finished").update(
                    {"success": False}
                ),
                recompute_summary=False,
            ),
            "unsuccessful build",
        )
        expect_reject(
            root,
            "build-content-evidence",
            tamper_build_content_evidence,
            "content mutation cannot be reproduced",
        )
        expect_reject(
            root,
            "build-native-log-derivation",
            lambda case: mutate_json(
                case / "build-timings/summary.json",
                lambda data: data["scenarios"][next(
                    index for index, row in enumerate(data["scenarios"]) if row["name"] == "native-tu"
                )]["samples"][0]["native_build"].update({"compiled_tu_count": 99}),
            ),
            "native_build cannot be recomputed",
        )
        expect_reject(
            root,
            "build-restore-fresh",
            lambda case: mutate_build_cargo_messages(
                case,
                "rust-model-leaf",
                restore_verification=True,
                update=lambda rows: next(row for row in rows if row.get("reason") == "compiler-artifact").update(
                    {"fresh": True}
                ),
                recompute_summary=True,
            ),
            "restore_verification was Fresh",
        )
        expect_reject(
            root,
            "build-restore-binary",
            tamper_restore_binary,
            "differs from frozen CUDA binary",
        )
        expect_reject(
            root,
            "build-restore-mtime",
            lambda case: mutate_json(
                case / "build-timings/summary.json",
                lambda data: next(
                    row for row in data["scenarios"] if row["name"] == "rust-model-leaf"
                )["restore_verification"]["restored_input"].update({"verification_mtime_ns": 100}),
            ),
            "did not force and then restore mtime",
        )
        expect_reject(
            root,
            "malformed-artifact-type",
            lambda case: mutate_json(
                case / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
                lambda data: data["cells"][0].update({"concurrency": {"forged": 1}}),
            ),
            "malformed baseline artifact (TypeError)",
        )
        require(
            tuple(runtime["mutations"]) == SELFTEST_MUTATION_NAMES,
            "self-test mutation names/order differ from the locked red-team matrix",
        )
        if mode == "fast":
            restored_manifest = validate_root(root, allow_synthetic=True)
            require(restored_manifest["status"] == "pass", "fast self-test did not restore the valid fixture")
        _ACTIVE_SELFTEST = None
        summary = {
            "schema_version": 1,
            "mode": mode,
            "mutation_assertion_count": len(runtime["mutations"]),
            "expected_mutation_assertion_count": len(SELFTEST_MUTATION_NAMES),
            "mutation_names": runtime["mutations"],
            "validator_counts": dict(sorted(runtime["validator_counts"].items())),
            "valid_fixture_assertion_count": 3 if mode == "fast" else 2,
        }
    print(f"{SELFTEST_SUMMARY_PREFIX} {json.dumps(summary, sort_keys=True, separators=(',', ':'))}")
    print(SELFTEST_FAST_PASS_LINE if mode == "fast" else SELFTEST_FULL_PASS_LINE)
    return 0


def self_test() -> int:
    return _run_self_test("fast")


def full_self_test() -> int:
    return _run_self_test("full")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    selftests = parser.add_mutually_exclusive_group()
    selftests.add_argument("--self-test", action="store_true")
    selftests.add_argument("--self-test-full", action="store_true")
    selftests.add_argument("--external-path-self-test", action="store_true")
    parser.add_argument("--require-full-self-test", action="store_true")
    args = parser.parse_args()
    if args.require_full_self_test and (
        args.self_test or args.self_test_full or args.external_path_self_test
    ):
        parser.error("--require-full-self-test is only valid with a real --out validation")
    if args.self_test:
        return self_test()
    if args.self_test_full:
        return full_self_test()
    if args.external_path_self_test:
        return external_path_self_test()
    if args.out is None:
        parser.error("--out is required")
    try:
        root = require_external_artifact_path(args.out, "baseline --out")
    except BaselineError as exc:
        print(f"FERRUM RUNTIME VNEXT G00 BASELINE FAIL: {args.out}: {exc}", file=sys.stderr)
        return 1
    if not args.require_full_self_test:
        parser.error("--require-full-self-test is required for real --out validation")
    full_self_test()
    try:
        manifest = validate_root(root)
        write_json(root / "manifest.json", manifest)
    except BaselineError as exc:
        root.mkdir(parents=True, exist_ok=True)
        write_json(
            root / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "fail",
                "source_git_sha": FROZEN_LEGACY_SHA,
                "validated_at": now_iso(),
                "artifact_dir": str(root),
                "error": str(exc),
                "pass_line": None,
            },
        )
        print(f"FERRUM RUNTIME VNEXT G00 BASELINE FAIL: {root}: {exc}", file=sys.stderr)
        return 1
    print(manifest["pass_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
