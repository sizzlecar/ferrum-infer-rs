#!/usr/bin/env python3
"""Validate artifacts produced by scripts/m3_ab_runner.py."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any


ROOT_REQUIRED = {
    "runner",
    "schema_version",
    "name",
    "created_at",
    "artifact_verdict",
    "not_publishable",
    "not_publishable_reason",
    "validation_checklist",
    "preflight",
    "runtime_preset",
    "cases",
    "summary_json",
}

CASE_REQUIRED = {
    "schema_version",
    "name",
    "started_at",
    "artifact_verdict",
    "not_publishable",
    "not_publishable_reason",
    "port",
    "git_head",
    "git_status_short",
    "binary_sha256",
    "features",
    "runtime_preset",
    "env_hash",
    "preset_env",
    "base_env",
    "effective_env",
    "case_env",
    "runtime_config_snapshot",
    "model_dir",
    "server_log",
    "bench_json",
    "bench_log",
    "effective_config_json",
    "decision_trace_jsonl",
    "auto_config_decision_count",
    "profile_jsonl",
    "correctness_gates",
    "validation_checklist",
    "cleanup_status",
    "status",
}

ARTIFACT_VERDICTS = {"pass", "fail", "diagnostic-only"}
VALIDATION_CHANGE_TYPES = {
    "default_path",
    "opt_in_experiment",
    "diagnostic",
    "api_only",
    "build_loop",
}
DEFAULT_PATH_REQUIRED_CONCURRENCY_CELLS = {1, 4, 16, 32}

VALIDATION_TOUCHED_AREAS = {
    "attention_decode_path",
    "attention_prefill_mixed_path",
    "benchmark_harness",
    "cuda_build_logic",
    "cuda_kernel",
    "fa2_runtime_path",
    "model_forward",
    "moe_route_dump",
    "openai_server_api",
    "profile_output",
    "runtime_defaults",
    "sampling",
    "scheduler_admission_policy",
    "structured_output",
}

TOUCHED_AREA_REQUIRED_GATES = {
    "attention_decode_path": {"paris", "bench_completion"},
    "attention_prefill_mixed_path": {"paris", "bench_completion"},
    "benchmark_harness": {"bench_completion"},
    "cuda_kernel": {"paris", "bench_completion"},
    "fa2_runtime_path": {"paris", "bench_completion"},
    "model_forward": {"paris", "bench_completion"},
    "moe_route_dump": {"paris", "bench_completion"},
    "openai_server_api": {"bench_completion"},
    "profile_output": {"bench_completion"},
    "runtime_defaults": {"paris", "bench_completion"},
    "sampling": {"paris", "bench_completion"},
    "scheduler_admission_policy": {"paris", "bench_completion"},
    "structured_output": {"bench_completion"},
}

SUMMARY_METRICS = {
    "throughput_mean",
    "throughput_stddev",
    "throughput_ci95_hw",
    "ttft_p50",
    "tpot_p50",
    "itl_p95",
    "completed",
    "errored",
}

PROFILE_REQUIRED = {
    "event",
    "commit_sha",
    "env_hash",
    "model",
    "concurrency",
    "shape",
    "stage_us",
    "graph_enabled",
    "runtime_flags",
}

DECISION_TRACE_REQUIRED = {
    "schema_version",
    "selection",
    "selected",
    "source",
    "source_key",
    "candidates",
    "rejected",
    "affects",
}

DECISION_SOURCES = {
    "default",
    "cli",
    "config_file",
    "env",
    "script_case",
    "model_metadata",
    "hardware_capability",
    "memory_profile",
    "workload_preset",
    "compiled_feature",
    # Transitional runner-side scaffold sources. Runtime artifacts should come
    # from the product server and use the Rust source names above.
    "case",
    "preset",
}

EFFECTIVE_CONFIG_ENTRY_REQUIRED = {
    "key",
    "effective_value",
    "source",
    "affects",
}

EFFECTIVE_CONFIG_REQUIRED = {
    "schema_version",
    "preset",
    "env_hash",
    "entries",
    "model_capabilities",
    "hardware_capabilities",
    "workload_profile",
    "decisions",
}

EFFECTIVE_CONFIG_SOURCES = {
    "default",
    "config_file",
    "cli",
    "env",
    "script_case",
    "memory_profile",
}

EFFECTIVE_CONFIG_EFFECTS = {
    "correctness",
    "performance",
    "memory",
    "diagnostics",
}

COMPILED_FEATURE_KEYS = {
    "cuda",
    "vllm_paged_attn",
    "vllm_moe_marlin",
    "cuda_graph",
    "greedy_argmax",
    "fa2_source",
    "fa2_direct_ffi",
}

VALIDATION_CHECKLIST_REQUIRED = {
    "schema_version",
    "change_type",
    "touched_areas",
    "required_correctness_gates",
    "observed_correctness_gates",
    "bench_completion",
    "local_gates",
    "skipped_gates",
    "performance_regression_required",
    "baseline_case",
}

BENCHMARK_IMPACT_REQUIRED = {
    "m3_benchmark_exercised",
    "reason",
    "evidence",
}

REQUIRED_DECISIONS = {
    "attention_prefill_mixed_backend",
    "attention_decode_backend",
    "moe_implementation",
    "moe_graph_policy",
    "kv_block_count",
    "max_sequences",
    "max_batched_tokens",
    "prefix_cache_policy",
    "scheduler_admission_policy",
    "sampling_readback_path",
}


def required_gates_for_touched_areas(touched_areas: list[str]) -> set[str]:
    required: set[str] = set()
    for area in touched_areas:
        required.update(TOUCHED_AREA_REQUIRED_GATES.get(area, set()))
    return required


class ValidationError(Exception):
    pass


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def require_keys(where: str, value: dict[str, Any], required: set[str]) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValidationError(f"{where}: missing keys: {', '.join(missing)}")


def resolve(path: str, root: Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return root / p


def validate_runtime_snapshot(case_name: str, snapshot: dict[str, Any]) -> None:
    require_keys(
        f"{case_name}.runtime_config_snapshot",
        snapshot,
        {"schema_version", "preset", "env_hash", "entries"},
    )
    entries = snapshot["entries"]
    if not isinstance(entries, list):
        raise ValidationError(f"{case_name}.runtime_config_snapshot.entries is not a list")
    keys = [entry.get("key") for entry in entries]
    if keys != sorted(keys):
        raise ValidationError(f"{case_name}.runtime_config_snapshot.entries are not sorted")
    for i, entry in enumerate(entries):
        require_keys(
            f"{case_name}.runtime_config_snapshot.entries[{i}]",
            entry,
            {"key", "effective_value", "source", "effect"},
        )
        if not isinstance(entry["key"], str) or not entry["key"].startswith("FERRUM_"):
            raise ValidationError(
                f"{case_name}.runtime_config_snapshot.entries[{i}].key must be FERRUM_*"
            )
        if not isinstance(entry["effective_value"], str):
            raise ValidationError(
                f"{case_name}.runtime_config_snapshot.entries[{i}].effective_value must be string"
            )
        if entry["source"] not in EFFECTIVE_CONFIG_SOURCES:
            raise ValidationError(
                f"{case_name}.runtime_config_snapshot.entries[{i}].source invalid: {entry['source']!r}"
            )
        if entry["effect"] not in EFFECTIVE_CONFIG_EFFECTS:
            raise ValidationError(
                f"{case_name}.runtime_config_snapshot.entries[{i}].effect invalid: {entry['effect']!r}"
            )
        affects = entry.get("affects")
        if affects is not None:
            if not isinstance(affects, list) or not affects:
                raise ValidationError(
                    f"{case_name}.runtime_config_snapshot.entries[{i}].affects must be non-empty list"
                )
            invalid_affects = [effect for effect in affects if effect not in EFFECTIVE_CONFIG_EFFECTS]
            if invalid_affects:
                raise ValidationError(
                    f"{case_name}.runtime_config_snapshot.entries[{i}].affects invalid: {invalid_affects}"
                )


def validate_runtime_diff_entry(where: str, entry: dict[str, Any]) -> None:
    require_keys(where, entry, {"key", "effective_value", "source", "effect"})
    if not isinstance(entry["key"], str) or not entry["key"].startswith("FERRUM_"):
        raise ValidationError(f"{where}.key must be FERRUM_*")
    if not isinstance(entry["effective_value"], str):
        raise ValidationError(f"{where}.effective_value must be string")
    if entry["source"] not in EFFECTIVE_CONFIG_SOURCES:
        raise ValidationError(f"{where}.source invalid: {entry['source']!r}")
    if entry["effect"] not in EFFECTIVE_CONFIG_EFFECTS:
        raise ValidationError(f"{where}.effect invalid: {entry['effect']!r}")
    affects = entry.get("affects")
    if affects is not None:
        if not isinstance(affects, list) or not affects:
            raise ValidationError(f"{where}.affects must be non-empty list")
        invalid_affects = [effect for effect in affects if effect not in EFFECTIVE_CONFIG_EFFECTS]
        if invalid_affects:
            raise ValidationError(f"{where}.affects invalid: {invalid_affects}")


def validate_runtime_config_diff(summary: dict[str, Any]) -> None:
    diffs = summary.get("runtime_config_diff_vs_baseline")
    if diffs is None:
        return
    if not isinstance(diffs, dict):
        raise ValidationError("summary.runtime_config_diff_vs_baseline must be an object")
    for case_name, diff in diffs.items():
        if not isinstance(diff, dict):
            raise ValidationError(f"runtime_config_diff {case_name}: diff must be an object")
        require_keys(
            f"runtime_config_diff {case_name}",
            diff,
            {"baseline_env_hash", "candidate_env_hash", "added", "removed", "changed"},
        )
        for list_key in ("added", "removed", "changed"):
            if not isinstance(diff[list_key], list):
                raise ValidationError(
                    f"runtime_config_diff {case_name}.{list_key} must be a list"
                )
        for item in diff["added"] + diff["removed"]:
            if not isinstance(item, dict):
                raise ValidationError(
                    f"runtime_config_diff {case_name}: added/removed entries must be objects"
                )
            validate_runtime_diff_entry(f"runtime_config_diff {case_name} entry", item)
        for item in diff["changed"]:
            if not isinstance(item, dict):
                raise ValidationError(
                    f"runtime_config_diff {case_name}: changed entries must be objects"
                )
            require_keys(
                f"runtime_config_diff {case_name} changed entry",
                item,
                {"key", "baseline", "candidate"},
            )
            for side in ("baseline", "candidate"):
                if not isinstance(item[side], dict):
                    raise ValidationError(
                        f"runtime_config_diff {case_name}.{side} must be object"
                    )
                validate_runtime_diff_entry(
                    f"runtime_config_diff {case_name}.{side}",
                    item[side],
                )


def validate_performance_regression_gates(summary: dict[str, Any]) -> None:
    gates = summary.get("performance_regression_gates")
    if gates is None:
        return
    if not isinstance(gates, dict):
        raise ValidationError("summary.performance_regression_gates must be an object")
    require_keys(
        "summary.performance_regression_gates",
        gates,
        {"schema_version", "enabled", "baseline_case", "thresholds", "cases"},
    )
    if gates["schema_version"] != 1:
        raise ValidationError("performance_regression_gates.schema_version must be 1")
    if not isinstance(gates["enabled"], bool):
        raise ValidationError("performance_regression_gates.enabled must be boolean")
    if gates["baseline_case"] is not None and not isinstance(gates["baseline_case"], str):
        raise ValidationError("performance_regression_gates.baseline_case must be null or string")
    if not isinstance(gates["thresholds"], dict):
        raise ValidationError("performance_regression_gates.thresholds must be an object")
    if not isinstance(gates["cases"], dict):
        raise ValidationError("performance_regression_gates.cases must be an object")
    for key in ("required_concurrency_cells", "observed_concurrency_cells"):
        cells = gates.get(key, [])
        if not isinstance(cells, list) or not all(
            isinstance(cell, int) and not isinstance(cell, bool) and cell > 0
            for cell in cells
        ):
            raise ValidationError(f"performance_regression_gates.{key} must be positive integers")
    if "concurrency_cells_ok" in gates and not isinstance(gates["concurrency_cells_ok"], bool):
        raise ValidationError("performance_regression_gates.concurrency_cells_ok must be boolean")
    for case_name, case_gates in gates["cases"].items():
        if not isinstance(case_gates, dict):
            raise ValidationError(f"performance_regression_gates {case_name}: must be object")
        require_keys(
            f"performance_regression_gates {case_name}",
            case_gates,
            {"baseline_case", "ok", "metrics"},
        )
        if not isinstance(case_gates["ok"], bool):
            raise ValidationError(f"performance_regression_gates {case_name}.ok must be boolean")
        if not isinstance(case_gates["metrics"], list) or not case_gates["metrics"]:
            raise ValidationError(
                f"performance_regression_gates {case_name}.metrics must be non-empty list"
            )
        for metric in case_gates["metrics"]:
            if not isinstance(metric, dict):
                raise ValidationError(
                    f"performance_regression_gates {case_name}.metrics entries must be objects"
                )
            require_keys(
                f"performance_regression_gates {case_name}.metric",
                metric,
                {"metric", "baseline", "candidate", "delta_pct", "threshold", "ok", "reason"},
            )
            if not isinstance(metric["metric"], str) or not metric["metric"].strip():
                raise ValidationError(
                    f"performance_regression_gates {case_name}.metric must be non-empty string"
                )
            if not isinstance(metric["ok"], bool):
                raise ValidationError(
                    f"performance_regression_gates {case_name}.{metric['metric']}.ok must be boolean"
                )
            if not isinstance(metric["reason"], str) or not metric["reason"].strip():
                raise ValidationError(
                    f"performance_regression_gates {case_name}.{metric['metric']}.reason must be non-empty string"
                )
            threshold = metric["threshold"]
            if not isinstance(threshold, dict):
                raise ValidationError(
                    f"performance_regression_gates {case_name}.{metric['metric']}.threshold must be object"
                )
            require_keys(
                f"performance_regression_gates {case_name}.{metric['metric']}.threshold",
                threshold,
                {"type", "value"},
            )
            if threshold["type"] not in {"min_delta_pct", "max_regression_pct"}:
                raise ValidationError(
                    f"performance_regression_gates {case_name}.{metric['metric']}.threshold.type invalid"
                )
            if not isinstance(threshold["value"], (int, float)) or isinstance(
                threshold["value"], bool
            ):
                raise ValidationError(
                    f"performance_regression_gates {case_name}.{metric['metric']}.threshold.value must be number"
                )


def validate_publishability(where: str, value: dict[str, Any]) -> None:
    verdict = value.get("artifact_verdict")
    if verdict not in ARTIFACT_VERDICTS:
        raise ValidationError(
            f"{where}: artifact_verdict must be one of {sorted(ARTIFACT_VERDICTS)}"
        )
    if not isinstance(value.get("not_publishable"), bool):
        raise ValidationError(f"{where}: not_publishable must be boolean")
    reason = value.get("not_publishable_reason")
    if reason is not None and not isinstance(reason, str):
        raise ValidationError(f"{where}: not_publishable_reason must be null or string")
    if verdict == "diagnostic-only" and not value["not_publishable"]:
        raise ValidationError(f"{where}: diagnostic-only artifacts must be not_publishable")
    if value["not_publishable"] and (reason is None or not reason.strip()):
        raise ValidationError(f"{where}: not_publishable artifacts require a reason")


def validate_validation_gate_list(where: str, gates: Any, *, require_ok: bool) -> None:
    if not isinstance(gates, list):
        raise ValidationError(f"{where} must be a list")
    for i, gate in enumerate(gates):
        if not isinstance(gate, dict):
            raise ValidationError(f"{where}[{i}] must be an object")
        require_keys(f"{where}[{i}]", gate, {"name"})
        if not isinstance(gate["name"], str) or not gate["name"].strip():
            raise ValidationError(f"{where}[{i}].name must be non-empty string")
        required = bool(gate.get("required", True))
        if "ok" in gate and gate["ok"] is not None and not isinstance(gate["ok"], bool):
            raise ValidationError(f"{where}[{i}].ok must be boolean or null")
        if require_ok and required and gate.get("ok") is not True:
            raise ValidationError(f"{where}[{i}] required gate is not ok: {gate['name']}")


def validate_benchmark_impact(where: str, impact: Any) -> dict[str, Any] | None:
    if impact is None:
        return None
    if not isinstance(impact, dict):
        raise ValidationError(f"{where}.validation_checklist.benchmark_impact must be object")
    require_keys(
        f"{where}.validation_checklist.benchmark_impact",
        impact,
        BENCHMARK_IMPACT_REQUIRED,
    )
    if not isinstance(impact["m3_benchmark_exercised"], bool):
        raise ValidationError(
            f"{where}.validation_checklist.benchmark_impact.m3_benchmark_exercised must be boolean"
        )
    for key in ("reason", "evidence"):
        if not isinstance(impact[key], str) or not impact[key].strip():
            raise ValidationError(
                f"{where}.validation_checklist.benchmark_impact.{key} must be non-empty string"
            )
    return impact


def performance_gate_ok(summary: dict[str, Any], case_name: str, baseline_case: str | None) -> bool:
    gates = summary.get("performance_regression_gates")
    if not isinstance(gates, dict) or gates.get("enabled") is not True:
        return False
    if baseline_case and case_name == baseline_case:
        return True
    case_gates = gates.get("cases", {}).get(case_name)
    return isinstance(case_gates, dict) and case_gates.get("ok") is True


def missing_default_path_concurrency_cells(summary: dict[str, Any] | None) -> list[int]:
    if summary is None:
        return sorted(DEFAULT_PATH_REQUIRED_CONCURRENCY_CELLS)
    gates = summary.get("performance_regression_gates")
    if not isinstance(gates, dict):
        return sorted(DEFAULT_PATH_REQUIRED_CONCURRENCY_CELLS)
    required = gates.get("required_concurrency_cells")
    if isinstance(required, list) and required:
        required_cells = {int(cell) for cell in required}
    else:
        required_cells = set(DEFAULT_PATH_REQUIRED_CONCURRENCY_CELLS)
    observed = gates.get("observed_concurrency_cells")
    observed_cells = {int(cell) for cell in observed} if isinstance(observed, list) else set()
    return sorted(required_cells - observed_cells)


def validate_validation_checklist(
    where: str,
    checklist: dict[str, Any],
    *,
    publishable: bool,
    correctness_gates: list[dict[str, Any]] | None = None,
    metrics: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    case_name: str | None = None,
    enforce_correctness: bool = True,
) -> None:
    if not isinstance(checklist, dict):
        raise ValidationError(f"{where}.validation_checklist must be an object")
    require_keys(f"{where}.validation_checklist", checklist, VALIDATION_CHECKLIST_REQUIRED)
    if checklist["schema_version"] != 1:
        raise ValidationError(f"{where}.validation_checklist.schema_version must be 1")
    if checklist["change_type"] not in VALIDATION_CHANGE_TYPES:
        raise ValidationError(f"{where}.validation_checklist.change_type invalid")
    if not isinstance(checklist["touched_areas"], list) or not checklist["touched_areas"]:
        raise ValidationError(f"{where}.validation_checklist.touched_areas must be non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in checklist["touched_areas"]):
        raise ValidationError(f"{where}.validation_checklist.touched_areas entries invalid")
    invalid_areas = sorted(
        area for area in checklist["touched_areas"] if area not in VALIDATION_TOUCHED_AREAS
    )
    if invalid_areas:
        raise ValidationError(
            f"{where}.validation_checklist.touched_areas invalid: {invalid_areas}"
        )
    required_gates = checklist["required_correctness_gates"]
    if not isinstance(required_gates, list):
        raise ValidationError(
            f"{where}.validation_checklist.required_correctness_gates must be list"
        )
    if not all(isinstance(item, str) and item.strip() for item in required_gates):
        raise ValidationError(
            f"{where}.validation_checklist.required_correctness_gates entries invalid"
        )
    area_required_gates = required_gates_for_touched_areas(checklist["touched_areas"])
    missing_area_gates = sorted(area_required_gates - set(required_gates))
    if missing_area_gates:
        raise ValidationError(
            f"{where}.validation_checklist missing required gates for touched areas: "
            f"{missing_area_gates}"
        )
    validate_validation_gate_list(
        f"{where}.validation_checklist.local_gates",
        checklist["local_gates"],
        require_ok=publishable,
    )
    validate_validation_gate_list(
        f"{where}.validation_checklist.skipped_gates",
        checklist["skipped_gates"],
        require_ok=False,
    )
    if publishable:
        required_skips = [
            gate.get("name")
            for gate in checklist["skipped_gates"]
            if bool(gate.get("required", True))
        ]
        if required_skips:
            raise ValidationError(
                f"{where}.validation_checklist skipped required gates: {required_skips}"
            )

    observed = checklist["observed_correctness_gates"]
    validate_validation_gate_list(
        f"{where}.validation_checklist.observed_correctness_gates",
        observed,
        require_ok=False,
    )
    observed_by_name = {gate["name"]: gate for gate in observed}
    if enforce_correctness:
        actual_by_name = {
            str(gate.get("name")): gate for gate in (correctness_gates or []) if gate.get("name")
        }
        for gate_name, gate in actual_by_name.items():
            if gate_name not in observed_by_name:
                raise ValidationError(
                    f"{where}.validation_checklist missing observed gate from correctness_gates: {gate_name}"
                )
            if bool(gate.get("ok")) != bool(observed_by_name[gate_name].get("ok")):
                raise ValidationError(
                    f"{where}.validation_checklist observed gate mismatch: {gate_name}"
                )

    bench = checklist["bench_completion"]
    if not isinstance(bench, dict):
        raise ValidationError(f"{where}.validation_checklist.bench_completion must be object")
    require_keys(
        f"{where}.validation_checklist.bench_completion",
        bench,
        {"required", "ok", "completed", "errored"},
    )
    if not isinstance(bench["required"], bool):
        raise ValidationError(f"{where}.validation_checklist.bench_completion.required invalid")
    if bench["ok"] is not None and not isinstance(bench["ok"], bool):
        raise ValidationError(f"{where}.validation_checklist.bench_completion.ok invalid")
    metrics = metrics or {}
    if enforce_correctness and metrics:
        if bench.get("completed") != metrics.get("completed"):
            raise ValidationError(f"{where}.validation_checklist completed metric mismatch")
        if bench.get("errored") != metrics.get("errored"):
            raise ValidationError(f"{where}.validation_checklist errored metric mismatch")

    if publishable and enforce_correctness:
        for gate_name in required_gates:
            if gate_name == "bench_completion":
                if bench.get("ok") is not True:
                    raise ValidationError(f"{where}: required bench_completion gate is not ok")
                continue
            gate = observed_by_name.get(gate_name)
            if gate is None or gate.get("ok") is not True:
                raise ValidationError(f"{where}: required correctness gate is not ok: {gate_name}")

    perf_required = checklist["performance_regression_required"]
    if not isinstance(perf_required, bool):
        raise ValidationError(
            f"{where}.validation_checklist.performance_regression_required must be boolean"
        )
    benchmark_impact = validate_benchmark_impact(
        where, checklist.get("benchmark_impact")
    )
    if publishable and checklist["change_type"] == "api_only" and not perf_required:
        if benchmark_impact is None:
            raise ValidationError(
                f"{where}.validation_checklist api_only without performance gate requires benchmark_impact proof"
            )
        if benchmark_impact["m3_benchmark_exercised"]:
            raise ValidationError(
                f"{where}.validation_checklist api_only exercises M3 benchmark path; "
                "set performance_regression_required=true"
            )
    if publishable and checklist["change_type"] == "default_path" and not perf_required:
        raise ValidationError(
            f"{where}.validation_checklist default_path requires performance regression gate"
        )
    if publishable and checklist["change_type"] == "default_path" and perf_required:
        missing_cells = missing_default_path_concurrency_cells(summary)
        if missing_cells:
            raise ValidationError(
                f"{where}.validation_checklist default_path requires full concurrency sweep; "
                f"missing c={missing_cells}"
            )
        gates = summary.get("performance_regression_gates") if summary else None
        if not isinstance(gates, dict) or gates.get("concurrency_cells_ok") is not True:
            raise ValidationError(
                f"{where}.validation_checklist default_path concurrency sweep gate failed"
            )
    baseline_case = checklist["baseline_case"]
    if baseline_case is not None and not isinstance(baseline_case, str):
        raise ValidationError(f"{where}.validation_checklist.baseline_case must be null or string")
    if publishable and perf_required:
        if not baseline_case:
            raise ValidationError(
                f"{where}.validation_checklist performance gate requires baseline_case"
            )
        if summary is not None and case_name is not None:
            if not performance_gate_ok(summary, case_name, baseline_case):
                raise ValidationError(
                    f"{where}.validation_checklist performance gate failed or missing"
                )
        elif summary is not None:
            gates = summary.get("performance_regression_gates")
            if not isinstance(gates, dict) or gates.get("enabled") is not True:
                raise ValidationError(
                    f"{where}.validation_checklist performance gates are not enabled"
                )


def validate_profile_event(where: str, event: dict[str, Any]) -> None:
    require_keys(where, event, PROFILE_REQUIRED)
    if not isinstance(event.get("event"), str) or not event["event"].strip():
        raise ValidationError(f"{where}: event must be a non-empty string")
    commit_sha = event.get("commit_sha")
    if commit_sha is not None and (
        not isinstance(commit_sha, str) or not commit_sha.strip()
    ):
        raise ValidationError(f"{where}: commit_sha must be null or a non-empty string")
    if not isinstance(event.get("env_hash"), str) or not event["env_hash"].startswith(
        "sha256:"
    ):
        raise ValidationError(f"{where}: env_hash must start with sha256:")
    if not isinstance(event.get("model"), str) or not event["model"].strip():
        raise ValidationError(f"{where}: model must be a non-empty string")
    concurrency = event.get("concurrency")
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency <= 0:
        raise ValidationError(f"{where}: concurrency must be a positive integer")
    if not isinstance(event.get("shape"), dict):
        raise ValidationError(f"{where}: shape must be an object")
    if not isinstance(event.get("stage_us"), dict):
        raise ValidationError(f"{where}: stage_us must be an object")
    if not isinstance(event.get("runtime_flags"), dict):
        raise ValidationError(f"{where}: runtime_flags must be an object")
    if not isinstance(event.get("graph_enabled"), bool):
        raise ValidationError(f"{where}: graph_enabled must be boolean")


def load_profile_events(path: Path, *, require_events: bool) -> list[dict[str, Any]]:
    if not path.exists():
        if require_events:
            raise ValidationError(f"profile_jsonl missing: {path}")
        return []

    events: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(event, dict):
                raise ValidationError(f"{path}:{line_no}: profile event must be an object")
            validate_profile_event(f"{path}:{line_no}", event)
            events.append(event)
    if require_events and not events:
        raise ValidationError(f"profile_jsonl has no events: {path}")
    return events


def validate_profile_jsonl(path: Path, *, require_events: bool) -> int:
    return len(load_profile_events(path, require_events=require_events))


def validate_profile_manifest(
    case_name: str,
    profile: Any,
    events: list[dict[str, Any]],
) -> None:
    if profile is None:
        return
    if not isinstance(profile, dict):
        raise ValidationError(f"{case_name}: profile manifest must be an object")
    if profile.get("enabled") is False:
        return
    if profile.get("ok") is not True:
        raise ValidationError(f"{case_name}: profile manifest reports failure: {profile!r}")
    mode = profile.get("mode")
    if mode is not None and mode not in {"structured_jsonl", "log_snippet_derived"}:
        raise ValidationError(f"{case_name}: profile.mode invalid: {mode!r}")
    if "event_count" in profile:
        event_count = profile["event_count"]
        if not isinstance(event_count, int) or isinstance(event_count, bool):
            raise ValidationError(f"{case_name}: profile.event_count must be an integer")
        if event_count != len(events):
            raise ValidationError(
                f"{case_name}: profile.event_count {event_count} "
                f"does not match profile_jsonl event count {len(events)}"
            )

    event_names = {event["event"] for event in events}
    required = profile.get("required_events", [])
    if required is None:
        required = []
    if not isinstance(required, list) or not all(
        isinstance(name, str) and name.strip() for name in required
    ):
        raise ValidationError(f"{case_name}: profile.required_events must be strings")
    missing = sorted(name for name in required if name not in event_names)
    if missing:
        raise ValidationError(f"{case_name}: profile missing required events: {missing}")

    required_any = profile.get("required_any_events", [])
    if required_any is None:
        required_any = []
    if not isinstance(required_any, list):
        raise ValidationError(f"{case_name}: profile.required_any_events must be a list")
    for group in required_any:
        if not isinstance(group, list) or not all(
            isinstance(name, str) and name.strip() for name in group
        ):
            raise ValidationError(
                f"{case_name}: profile.required_any_events groups must be string lists"
            )
        if group and not any(name in event_names for name in group):
            raise ValidationError(
                f"{case_name}: profile missing one of required event group: {group}"
            )
    if (required or required_any) and mode != "structured_jsonl":
        raise ValidationError(
            f"{case_name}: profile required_events require mode=structured_jsonl"
        )

    for key in ("missing_events", "missing_any_events", "errors"):
        value = profile.get(key, [])
        if value:
            raise ValidationError(f"{case_name}: profile.{key} is not empty: {value!r}")


def validate_decision_event(where: str, event: dict[str, Any]) -> None:
    require_keys(where, event, DECISION_TRACE_REQUIRED)
    if event.get("schema_version") != 1:
        raise ValidationError(f"{where}: schema_version must be 1")
    for key in ("selection", "selected", "source"):
        if not isinstance(event.get(key), str) or not event[key].strip():
            raise ValidationError(f"{where}: {key} must be a non-empty string")
    if event["source"] not in DECISION_SOURCES:
        raise ValidationError(f"{where}: source invalid: {event['source']!r}")
    source_key = event.get("source_key")
    if source_key is not None:
        if not isinstance(source_key, str) or not source_key.startswith("FERRUM_"):
            raise ValidationError(f"{where}: source_key must be null or FERRUM_* string")
    for key in ("candidates", "rejected", "affects"):
        if not isinstance(event.get(key), list):
            raise ValidationError(f"{where}: {key} must be a list")
    if not event["candidates"]:
        raise ValidationError(f"{where}: candidates must not be empty")
    if not all(isinstance(item, str) and item.strip() for item in event["candidates"]):
        raise ValidationError(f"{where}: candidates entries must be non-empty strings")
    if not event["affects"]:
        raise ValidationError(f"{where}: affects must not be empty")
    invalid_affects = [
        effect for effect in event["affects"] if effect not in EFFECTIVE_CONFIG_EFFECTS
    ]
    if invalid_affects:
        raise ValidationError(f"{where}: affects invalid: {invalid_affects}")
    for item in event["rejected"]:
        if not isinstance(item, dict):
            raise ValidationError(f"{where}: rejected entries must be objects")
        require_keys(f"{where}.rejected", item, {"value", "reason"})
        if not isinstance(item["value"], str) or not item["value"].strip():
            raise ValidationError(f"{where}: rejected.value must be non-empty string")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise ValidationError(f"{where}: rejected.reason must be non-empty string")


def validate_decision_trace_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValidationError(f"decision_trace_jsonl missing: {path}")
    events: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(event, dict):
                raise ValidationError(f"{path}:{line_no}: decision event must be an object")
            validate_decision_event(f"{path}:{line_no}", event)
            events.append(event)
    if not events:
        raise ValidationError(f"decision_trace_jsonl has no events: {path}")
    selections = {event["selection"] for event in events}
    missing = sorted(REQUIRED_DECISIONS - selections)
    if missing:
        raise ValidationError(f"decision_trace_jsonl missing selections: {', '.join(missing)}")
    return events


def canonicalize(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def expect_optional_positive_int(where: str, value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"{where} must be null or a positive integer")


def expect_string_list(where: str, value: Any, *, allow_empty: bool = False) -> None:
    if not isinstance(value, list):
        raise ValidationError(f"{where} must be a list")
    if not allow_empty and not value:
        raise ValidationError(f"{where} must not be empty")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValidationError(f"{where} entries must be non-empty strings")


def validate_model_capabilities(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValidationError("effective_config_json.model_capabilities must be an object")
    require_keys(
        "effective_config_json.model_capabilities",
        value,
        {
            "architecture",
            "quantization",
            "moe",
            "max_context_len",
            "num_hidden_layers",
            "head_dim",
            "kv_heads",
            "estimated_weight_bytes",
            "supported_dtypes",
            "graph_safe_moe",
        },
    )
    if not isinstance(value["architecture"], str) or not value["architecture"].strip():
        raise ValidationError("effective_config_json.model_capabilities.architecture invalid")
    if value["quantization"] is not None and not isinstance(value["quantization"], str):
        raise ValidationError("effective_config_json.model_capabilities.quantization invalid")
    for key in (
        "max_context_len",
        "num_hidden_layers",
        "head_dim",
        "kv_heads",
        "estimated_weight_bytes",
    ):
        expect_optional_positive_int(f"effective_config_json.model_capabilities.{key}", value[key])
    expect_string_list(
        "effective_config_json.model_capabilities.supported_dtypes",
        value["supported_dtypes"],
    )
    if not isinstance(value["graph_safe_moe"], bool):
        raise ValidationError("effective_config_json.model_capabilities.graph_safe_moe invalid")
    moe = value["moe"]
    if moe is None:
        return
    if not isinstance(moe, dict):
        raise ValidationError("effective_config_json.model_capabilities.moe must be object")
    require_keys(
        "effective_config_json.model_capabilities.moe",
        moe,
        {"num_experts", "experts_per_token", "moe_intermediate_size"},
    )
    for key in ("num_experts", "experts_per_token"):
        if not isinstance(moe[key], int) or isinstance(moe[key], bool) or moe[key] <= 0:
            raise ValidationError(
                f"effective_config_json.model_capabilities.moe.{key} must be positive integer"
            )
    expect_optional_positive_int(
        "effective_config_json.model_capabilities.moe.moe_intermediate_size",
        moe["moe_intermediate_size"],
    )


def validate_hardware_capabilities(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValidationError("effective_config_json.hardware_capabilities must be an object")
    require_keys(
        "effective_config_json.hardware_capabilities",
        value,
        {
            "backend",
            "cuda_runtime",
            "compute_capability",
            "vram_bytes",
            "sm_count",
            "supported_dtypes",
            "supported_kv_dtypes",
            "graph_support",
            "compiled_features",
        },
    )
    if not isinstance(value["backend"], str) or not value["backend"].strip():
        raise ValidationError("effective_config_json.hardware_capabilities.backend invalid")
    for key in ("cuda_runtime", "compute_capability"):
        if value[key] is not None and not isinstance(value[key], str):
            raise ValidationError(f"effective_config_json.hardware_capabilities.{key} invalid")
    expect_optional_positive_int(
        "effective_config_json.hardware_capabilities.vram_bytes", value["vram_bytes"]
    )
    expect_optional_positive_int(
        "effective_config_json.hardware_capabilities.sm_count", value["sm_count"]
    )
    expect_string_list(
        "effective_config_json.hardware_capabilities.supported_dtypes",
        value["supported_dtypes"],
    )
    expect_string_list(
        "effective_config_json.hardware_capabilities.supported_kv_dtypes",
        value["supported_kv_dtypes"],
    )
    if not isinstance(value["graph_support"], bool):
        raise ValidationError("effective_config_json.hardware_capabilities.graph_support invalid")
    features = value["compiled_features"]
    if not isinstance(features, dict):
        raise ValidationError(
            "effective_config_json.hardware_capabilities.compiled_features must be object"
        )
    require_keys(
        "effective_config_json.hardware_capabilities.compiled_features",
        features,
        COMPILED_FEATURE_KEYS,
    )
    for key in COMPILED_FEATURE_KEYS:
        if not isinstance(features[key], bool):
            raise ValidationError(
                f"effective_config_json.hardware_capabilities.compiled_features.{key} invalid"
            )


def validate_workload_profile(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValidationError("effective_config_json.workload_profile must be an object")
    require_keys(
        "effective_config_json.workload_profile",
        value,
        {
            "preset",
            "serving_mode",
            "target_concurrency",
            "prompt_length_class",
            "output_length_class",
            "priority",
        },
    )
    if value["preset"] is not None and not isinstance(value["preset"], str):
        raise ValidationError("effective_config_json.workload_profile.preset invalid")
    for key in ("serving_mode", "prompt_length_class", "output_length_class"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValidationError(f"effective_config_json.workload_profile.{key} invalid")
    if (
        not isinstance(value["target_concurrency"], int)
        or isinstance(value["target_concurrency"], bool)
        or value["target_concurrency"] <= 0
    ):
        raise ValidationError(
            "effective_config_json.workload_profile.target_concurrency must be positive integer"
        )
    if value["priority"] not in {"latency", "throughput", "balanced"}:
        raise ValidationError("effective_config_json.workload_profile.priority invalid")


def validate_effective_config(
    path: Path, decision_events: list[dict[str, Any]]
) -> dict[str, Any]:
    if not path.exists():
        raise ValidationError(f"effective_config_json missing: {path}")
    data = load_json(path)
    require_keys("effective_config_json", data, EFFECTIVE_CONFIG_REQUIRED)
    if data["schema_version"] != 1:
        raise ValidationError("effective_config_json.schema_version must be 1")
    if data["preset"] is not None and not isinstance(data["preset"], str):
        raise ValidationError("effective_config_json.preset must be null or string")
    if not isinstance(data["env_hash"], str) or not data["env_hash"].startswith("sha256:"):
        raise ValidationError("effective_config_json.env_hash must start with sha256:")
    if not isinstance(data["entries"], list):
        raise ValidationError("effective_config_json.entries must be a list")
    entry_keys = [entry.get("key") if isinstance(entry, dict) else None for entry in data["entries"]]
    if entry_keys != sorted(entry_keys):
        raise ValidationError("effective_config_json.entries are not sorted")
    for i, entry in enumerate(data["entries"]):
        if not isinstance(entry, dict):
            raise ValidationError(f"effective_config_json.entries[{i}] must be an object")
        require_keys(
            f"effective_config_json.entries[{i}]",
            entry,
            EFFECTIVE_CONFIG_ENTRY_REQUIRED,
        )
        if not isinstance(entry["key"], str) or not entry["key"].startswith("FERRUM_"):
            raise ValidationError(
                f"effective_config_json.entries[{i}].key must be a FERRUM_* string"
            )
        if not isinstance(entry["effective_value"], str):
            raise ValidationError(
                f"effective_config_json.entries[{i}].effective_value must be string"
            )
        if entry["source"] not in EFFECTIVE_CONFIG_SOURCES:
            raise ValidationError(
                f"effective_config_json.entries[{i}].source invalid: {entry['source']!r}"
            )
        if not isinstance(entry["affects"], list) or not entry["affects"]:
            raise ValidationError(
                f"effective_config_json.entries[{i}].affects must be a non-empty list"
            )
        invalid_effects = [
            effect for effect in entry["affects"] if effect not in EFFECTIVE_CONFIG_EFFECTS
        ]
        if invalid_effects:
            raise ValidationError(
                f"effective_config_json.entries[{i}].affects invalid: {invalid_effects}"
            )
    validate_model_capabilities(data["model_capabilities"])
    validate_hardware_capabilities(data["hardware_capabilities"])
    validate_workload_profile(data["workload_profile"])
    decisions = data["decisions"]
    if not isinstance(decisions, list):
        raise ValidationError("effective_config_json.decisions must be a list")
    if canonicalize(decisions) != canonicalize(decision_events):
        raise ValidationError("effective_config_json decisions do not match decision_trace_jsonl")
    return data


def validate_case(
    root: Path,
    case_ref: dict[str, Any],
    *,
    require_bench: bool,
    require_profile_events: bool,
    summary: dict[str, Any],
) -> dict[str, Any]:
    if "manifest" not in case_ref:
        raise ValidationError("root manifest case is missing manifest path")
    manifest_path = resolve(case_ref["manifest"], root)
    if not manifest_path.exists():
        raise ValidationError(f"case manifest not found: {manifest_path}")
    case = load_json(manifest_path)
    require_keys(f"{case_ref.get('name', manifest_path.name)} manifest", case, CASE_REQUIRED)
    validate_publishability(f"{case_ref.get('name', manifest_path.name)} manifest", case)

    if case["status"] != "pass":
        raise ValidationError(f"{case['name']}: status is {case['status']!r}")
    if not str(case["env_hash"]).startswith("sha256:"):
        raise ValidationError(f"{case['name']}: env_hash is not sha256")
    validate_runtime_snapshot(case["name"], case["runtime_config_snapshot"])
    decision_events = validate_decision_trace_jsonl(
        resolve(case["decision_trace_jsonl"], root)
    )
    effective_config = validate_effective_config(
        resolve(case["effective_config_json"], root), decision_events
    )
    snapshot = case["runtime_config_snapshot"]
    if snapshot.get("env_hash") != effective_config.get("env_hash"):
        raise ValidationError(
            f"{case['name']}: runtime_config_snapshot.env_hash does not match effective_config_json"
        )
    snapshot_entries = [
        {
            "key": entry.get("key"),
            "effective_value": entry.get("effective_value"),
            "source": entry.get("source"),
            "affects": entry.get("affects", [entry.get("effect")]),
        }
        for entry in snapshot.get("entries", [])
    ]
    if canonicalize(snapshot_entries) != canonicalize(effective_config.get("entries", [])):
        raise ValidationError(
            f"{case['name']}: runtime_config_snapshot.entries do not match effective_config_json"
        )
    decision_count = case["auto_config_decision_count"]
    if (
        not isinstance(decision_count, int)
        or isinstance(decision_count, bool)
        or decision_count != len(decision_events)
    ):
        raise ValidationError(
            f"{case['name']}: auto_config_decision_count {decision_count!r} "
            f"does not match decision trace count {len(decision_events)}"
        )

    gates = case.get("correctness_gates") or []
    if not gates:
        raise ValidationError(f"{case['name']}: no correctness gates recorded")
    failed = [gate for gate in gates if not gate.get("ok")]
    if failed:
        raise ValidationError(f"{case['name']}: failed gates: {failed}")

    cleanup = case.get("cleanup_status") or {}
    if cleanup.get("sent_kill"):
        raise ValidationError(f"{case['name']}: cleanup needed SIGKILL")
    if cleanup.get("returncode") not in (0, None):
        raise ValidationError(f"{case['name']}: server returncode {cleanup.get('returncode')}")
    if cleanup.get("process_leak_ok") is False:
        raise ValidationError(f"{case['name']}: runner-owned process leak after cleanup")
    if cleanup.get("process_leaks"):
        raise ValidationError(
            f"{case['name']}: runner-owned process leaks: {cleanup.get('process_leaks')}"
        )
    if cleanup.get("global_process_hygiene_ok") is False:
        raise ValidationError(f"{case['name']}: global process hygiene failed after cleanup")
    if cleanup.get("global_process_findings"):
        raise ValidationError(
            f"{case['name']}: global process findings: {cleanup.get('global_process_findings')}"
        )

    bench_path = resolve(case["bench_json"], root)
    if require_bench and not bench_path.exists():
        raise ValidationError(f"{case['name']}: bench_json missing: {bench_path}")
    metrics = case.get("metrics") or {}
    if require_bench:
        if metrics.get("completed") is None:
            raise ValidationError(f"{case['name']}: completed metric missing")
        if metrics.get("errored") is None:
            raise ValidationError(f"{case['name']}: errored metric missing")
        if metrics.get("errored") != 0:
            raise ValidationError(f"{case['name']}: errored metric is {metrics.get('errored')}")
    validate_validation_checklist(
        f"{case['name']} manifest",
        case["validation_checklist"],
        publishable=not bool(case["not_publishable"]),
        correctness_gates=gates,
        metrics=metrics,
        summary=summary,
        case_name=case["name"],
    )

    profile_event_values = load_profile_events(
        resolve(case["profile_jsonl"], root),
        require_events=require_profile_events,
    )
    validate_profile_manifest(case["name"], case.get("profile"), profile_event_values)
    return {
        "name": case["name"],
        "bench": bench_path.exists(),
        "profile_events": len(profile_event_values),
    }


def validate_artifact(
    root: Path,
    *,
    require_bench: bool,
    require_profile_events: bool,
) -> dict[str, Any]:
    root_manifest_path = root / "manifest.json"
    if not root_manifest_path.exists():
        raise ValidationError(f"root manifest missing: {root_manifest_path}")
    manifest = load_json(root_manifest_path)
    require_keys("root manifest", manifest, ROOT_REQUIRED)
    validate_publishability("root manifest", manifest)

    summary_path = resolve(manifest["summary_json"], root)
    if not summary_path.exists():
        raise ValidationError(f"summary missing: {summary_path}")
    summary = load_json(summary_path)
    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValidationError("summary.rows must be a non-empty list")
    for row in rows:
        require_keys(
            f"summary row {row.get('name')}",
            row,
            {"name", "status", "artifact_verdict", "not_publishable", *SUMMARY_METRICS},
        )
    validate_runtime_config_diff(summary)
    validate_performance_regression_gates(summary)
    validate_validation_checklist(
        "root manifest",
        manifest["validation_checklist"],
        publishable=not bool(manifest["not_publishable"]),
        summary=summary,
        case_name=None,
        enforce_correctness=False,
    )

    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValidationError("root manifest cases must be a non-empty list")
    case_results = [
        validate_case(
            root,
            case,
            require_bench=require_bench,
            require_profile_events=require_profile_events,
            summary=summary,
        )
        for case in cases
    ]
    return {
        "root": str(root),
        "cases": case_results,
        "summary_rows": len(rows),
        "ok": True,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def self_test() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as td:
        root = Path(td).relative_to(Path.cwd())
        write_json(root / "summary.json", {"ok": True})
        assert resolve(str(root / "summary.json"), root) == root / "summary.json"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        effective_entries = [
            {
                "key": "FERRUM_KV_DTYPE",
                "effective_value": "2",
                "source": "cli",
                "affects": ["correctness", "memory"],
            },
            {
                "key": "FERRUM_MAX_BATCHED_TOKENS",
                "effective_value": "2048",
                "source": "memory_profile",
                "affects": ["performance", "memory"],
            },
            {
                "key": "FERRUM_MOE_GRAPH",
                "effective_value": "1",
                "source": "env",
                "affects": ["performance"],
            },
        ]
        snapshot = {
            "schema_version": 1,
            "preset": "preset",
            "env_hash": "sha256:env",
            "entries": [
                {
                    **entry,
                    "effect": entry["affects"][0],
                }
                for entry in effective_entries
            ],
        }
        validate_runtime_config_diff(
            {
                "runtime_config_diff_vs_baseline": {
                    "case": {
                        "baseline_env_hash": "sha256:base",
                        "candidate_env_hash": "sha256:env",
                        "added": [snapshot["entries"][0]],
                        "removed": [],
                        "changed": [
                            {
                                "key": "FERRUM_FA_LAYOUT_VARLEN",
                                "baseline": {
                                    "key": "FERRUM_FA_LAYOUT_VARLEN",
                                    "effective_value": "0",
                                    "source": "script_case",
                                    "effect": "performance",
                                    "affects": ["performance"],
                                },
                                "candidate": {
                                    "key": "FERRUM_FA_LAYOUT_VARLEN",
                                    "effective_value": "1",
                                    "source": "script_case",
                                    "effect": "performance",
                                    "affects": ["performance"],
                                },
                            }
                        ],
                    }
                }
            }
        )
        case_dir = root / "case_c1_n1"
        decisions = [
            {
                "schema_version": 1,
                "selection": selection,
                "selected": "selected",
                "source": "default",
                "source_key": None,
                "candidates": ["selected"],
                "rejected": [],
                "affects": ["performance"],
            }
            for selection in sorted(REQUIRED_DECISIONS)
        ]
        auto_config_inputs = {
            "model_capabilities": {
                "architecture": "qwen3_moe",
                "quantization": "gptq_int4",
                "moe": {
                    "num_experts": 128,
                    "experts_per_token": 8,
                    "moe_intermediate_size": 768,
                },
                "max_context_len": 40960,
                "num_hidden_layers": 48,
                "head_dim": 128,
                "kv_heads": 4,
                "estimated_weight_bytes": 19327352832,
                "supported_dtypes": ["fp16"],
                "graph_safe_moe": True,
            },
            "hardware_capabilities": {
                "backend": "cuda",
                "cuda_runtime": "12.8",
                "compute_capability": "8.9",
                "vram_bytes": 25753026560,
                "sm_count": 128,
                "supported_dtypes": ["fp16", "fp32"],
                "supported_kv_dtypes": ["fp16", "int8"],
                "graph_support": True,
                "compiled_features": {
                    "cuda": True,
                    "vllm_paged_attn": True,
                    "vllm_moe_marlin": True,
                    "cuda_graph": True,
                    "greedy_argmax": True,
                    "fa2_source": False,
                    "fa2_direct_ffi": False,
                },
            },
            "workload_profile": {
                "preset": "m3_qwen3_30b_a3b_int4",
                "serving_mode": "bench_serve",
                "target_concurrency": 32,
                "prompt_length_class": "random_256",
                "output_length_class": "random_128",
                "priority": "throughput",
            },
        }
        write_json(
            case_dir / "effective_config.json",
            {
                "schema_version": 1,
                "preset": snapshot["preset"],
                "env_hash": snapshot["env_hash"],
                "entries": effective_entries,
                **auto_config_inputs,
                "decisions": decisions,
            },
        )
        (case_dir / "decision_trace.jsonl").write_text(
            "".join(json.dumps(decision, sort_keys=True) + "\n" for decision in decisions)
        )
        write_json(case_dir / "bench.json", {"ok": True})
        validation_checklist = {
            "schema_version": 1,
            "change_type": "opt_in_experiment",
            "touched_areas": ["benchmark_harness"],
            "required_correctness_gates": ["paris", "bench_completion"],
            "observed_correctness_gates": [{"name": "paris", "ok": True}],
            "bench_completion": {
                "required": True,
                "ok": True,
                "completed": 1,
                "errored": 0,
            },
            "local_gates": [],
            "skipped_gates": [],
            "performance_regression_required": False,
            "baseline_case": None,
        }
        write_json(
            case_dir / "manifest.json",
            {
                "schema_version": 1,
                "name": "case",
                "started_at": "now",
                "artifact_verdict": "pass",
                "not_publishable": False,
                "not_publishable_reason": None,
                "port": 1,
                "git_head": "abc",
                "git_status_short": [],
                "binary_sha256": "sha256:bin",
                "features": "cuda",
                "runtime_preset": "preset",
                "env_hash": "sha256:env",
                "preset_env": {},
                "base_env": {},
                "effective_env": {},
                "case_env": {},
                "runtime_config_snapshot": snapshot,
                "model_dir": "/model",
                "server_log": str(case_dir / "server.log"),
                "bench_json": str(case_dir / "bench.json"),
                "bench_log": str(case_dir / "bench.log"),
                "effective_config_json": str(case_dir / "effective_config.json"),
                "decision_trace_jsonl": str(case_dir / "decision_trace.jsonl"),
                "auto_config_decision_count": len(decisions),
                "profile_jsonl": str(case_dir / "profile.jsonl"),
                "correctness_gates": [{"name": "paris", "ok": True}],
                "validation_checklist": validation_checklist,
                "cleanup_status": {
                    "sent_kill": False,
                    "returncode": 0,
                    "process_leak_ok": True,
                    "process_leaks": [],
                    "global_process_hygiene_ok": True,
                    "global_process_findings": [],
                },
                "metrics": {
                    "completed": 1,
                    "errored": 0,
                    "throughput_mean": 1.0,
                    "throughput_stddev": None,
                    "throughput_ci95_hw": None,
                    "ttft_p50": 1.0,
                    "tpot_p50": 1.0,
                    "itl_p95": 1.0,
                },
                "status": "pass",
            },
        )
        write_json(
            root / "summary.json",
            {
                "rows": [
                    {
                        "name": "case",
                        "status": "pass",
                        "artifact_verdict": "pass",
                        "not_publishable": False,
                        **(load_json(case_dir / "manifest.json")["metrics"]),
                    }
                ],
                "performance_regression_gates": {
                    "schema_version": 1,
                    "enabled": True,
                    "baseline_case": "baseline",
                    "required_concurrency_cells": [1, 4, 16, 32],
                    "observed_concurrency_cells": [32],
                    "concurrency_cells_ok": False,
                    "thresholds": {
                        "enabled": True,
                        "throughput_min_delta_pct": -3.0,
                        "ttft_max_regression_pct": 10.0,
                        "tpot_max_regression_pct": 5.0,
                        "itl_p95_max_regression_pct": 10.0,
                    },
                    "cases": {
                        "case": {
                            "baseline_case": "baseline",
                            "ok": True,
                            "metrics": [
                                {
                                    "metric": "throughput_mean",
                                    "baseline": 1.0,
                                    "candidate": 1.0,
                                    "delta_pct": 0.0,
                                    "threshold": {
                                        "type": "min_delta_pct",
                                        "value": -3.0,
                                    },
                                    "ok": True,
                                    "reason": "ok",
                                }
                            ],
                        }
                    },
                },
            },
        )
        write_json(
            root / "manifest.json",
            {
                "runner": "scripts/m3_ab_runner.py",
                "schema_version": 1,
                "name": "test",
                "created_at": "now",
                "artifact_verdict": "pass",
                "not_publishable": False,
                "not_publishable_reason": None,
                "validation_checklist": validation_checklist,
                "preflight": {},
                "runtime_preset": "preset",
                "cases": [{"name": "case", "manifest": str(case_dir / "manifest.json")}],
                "summary_json": str(root / "summary.json"),
            },
        )
        result = validate_artifact(root, require_bench=True, require_profile_events=False)
        assert result["ok"]

        case_manifest = load_json(case_dir / "manifest.json")
        case_manifest["auto_config_decision_count"] = len(decisions) - 1
        write_json(case_dir / "manifest.json", case_manifest)
        try:
            validate_artifact(root, require_bench=True, require_profile_events=False)
        except ValidationError as exc:
            assert "auto_config_decision_count" in str(exc)
        else:
            raise AssertionError("decision count mismatch unexpectedly passed")
        case_manifest["auto_config_decision_count"] = len(decisions)
        write_json(case_dir / "manifest.json", case_manifest)

        bad_decisions = [dict(decision) for decision in decisions]
        bad_decisions[0]["source"] = "mystery"
        (case_dir / "decision_trace.jsonl").write_text(
            "".join(json.dumps(decision, sort_keys=True) + "\n" for decision in bad_decisions)
        )
        try:
            validate_artifact(root, require_bench=True, require_profile_events=False)
        except ValidationError as exc:
            assert "source invalid" in str(exc)
        else:
            raise AssertionError("invalid decision source unexpectedly passed")
        (case_dir / "decision_trace.jsonl").write_text(
            "".join(json.dumps(decision, sort_keys=True) + "\n" for decision in decisions)
        )

        profile_event = {
            "event": "unified_prof",
            "commit_sha": "abc",
            "env_hash": "sha256:env",
            "model": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
            "concurrency": 32,
            "shape": {"batch": 32},
            "stage_us": {"model": 12500.0},
            "graph_enabled": True,
            "runtime_flags": snapshot,
        }
        (case_dir / "profile.jsonl").write_text(
            json.dumps(profile_event, sort_keys=True) + "\n"
        )
        case_manifest = load_json(case_dir / "manifest.json")
        case_manifest["profile"] = {
            "enabled": True,
            "mode": "structured_jsonl",
            "profile_jsonl": str(case_dir / "profile.jsonl"),
            "event_count": 1,
            "events": ["unified_prof"],
            "required_events": ["unified_prof"],
            "required_any_events": [["bucket_prof", "unified_prof"]],
            "missing_events": [],
            "missing_any_events": [],
            "errors": [],
            "ok": True,
        }
        write_json(case_dir / "manifest.json", case_manifest)
        result = validate_artifact(root, require_bench=True, require_profile_events=True)
        assert result["cases"][0]["profile_events"] == 1

        case_manifest["profile"]["required_events"] = ["bucket_prof"]
        write_json(case_dir / "manifest.json", case_manifest)
        try:
            validate_artifact(root, require_bench=True, require_profile_events=True)
        except ValidationError as exc:
            assert "missing required events" in str(exc)
        else:
            raise AssertionError("missing required profile event unexpectedly passed")
        case_manifest["profile"]["required_events"] = ["unified_prof"]
        write_json(case_dir / "manifest.json", case_manifest)

        case_manifest["profile"]["mode"] = "log_snippet_derived"
        write_json(case_dir / "manifest.json", case_manifest)
        try:
            validate_artifact(root, require_bench=True, require_profile_events=True)
        except ValidationError as exc:
            assert "mode=structured_jsonl" in str(exc)
        else:
            raise AssertionError("log-derived profile satisfied structured events")
        case_manifest["profile"]["mode"] = "structured_jsonl"
        write_json(case_dir / "manifest.json", case_manifest)

        effective_config = load_json(case_dir / "effective_config.json")
        effective_config["entries"][0]["affects"] = []
        write_json(case_dir / "effective_config.json", effective_config)
        try:
            validate_artifact(root, require_bench=True, require_profile_events=False)
        except ValidationError as exc:
            assert "affects" in str(exc)
        else:
            raise AssertionError("invalid effective_config entry unexpectedly passed")
        effective_config["entries"] = effective_entries
        write_json(case_dir / "effective_config.json", effective_config)

        effective_config["hardware_capabilities"]["compiled_features"]["cuda"] = "yes"
        write_json(case_dir / "effective_config.json", effective_config)
        try:
            validate_artifact(root, require_bench=True, require_profile_events=False)
        except ValidationError as exc:
            assert "compiled_features.cuda" in str(exc)
        else:
            raise AssertionError("invalid effective_config hardware input unexpectedly passed")
        effective_config["hardware_capabilities"]["compiled_features"]["cuda"] = True
        write_json(case_dir / "effective_config.json", effective_config)

        case_manifest = load_json(case_dir / "manifest.json")
        case_manifest["cleanup_status"]["process_leak_ok"] = False
        case_manifest["cleanup_status"]["process_leaks"] = [
            {"pid": 123, "cmd": "target/release/ferrum serve --port 1"}
        ]
        write_json(case_dir / "manifest.json", case_manifest)
        try:
            validate_artifact(root, require_bench=True, require_profile_events=False)
        except ValidationError as exc:
            assert "process leak" in str(exc)
        else:
            raise AssertionError("process leak cleanup unexpectedly passed")
        case_manifest["cleanup_status"]["process_leak_ok"] = True
        case_manifest["cleanup_status"]["process_leaks"] = []
        write_json(case_dir / "manifest.json", case_manifest)

        case_manifest["cleanup_status"]["global_process_hygiene_ok"] = False
        case_manifest["cleanup_status"]["global_process_findings"] = [
            {"pid": 456, "cmd": "cargo build --release", "reason": "cargo-global"}
        ]
        write_json(case_dir / "manifest.json", case_manifest)
        try:
            validate_artifact(root, require_bench=True, require_profile_events=False)
        except ValidationError as exc:
            assert "global process hygiene" in str(exc)
        else:
            raise AssertionError("global process hygiene unexpectedly passed")
        case_manifest["cleanup_status"]["global_process_hygiene_ok"] = True
        case_manifest["cleanup_status"]["global_process_findings"] = []
        write_json(case_dir / "manifest.json", case_manifest)

        bad_checklist = load_json(case_dir / "manifest.json")
        bad_checklist["validation_checklist"]["observed_correctness_gates"][0]["ok"] = False
        write_json(case_dir / "manifest.json", bad_checklist)
        try:
            validate_artifact(root, require_bench=True, require_profile_events=False)
        except ValidationError as exc:
            assert "observed gate mismatch" in str(exc) or "required correctness gate" in str(exc)
        else:
            raise AssertionError("failed validation checklist unexpectedly passed")
        write_json(case_dir / "manifest.json", case_manifest)

        bad_checklist = load_json(case_dir / "manifest.json")
        bad_checklist["validation_checklist"]["touched_areas"] = ["model_forward"]
        bad_checklist["validation_checklist"]["required_correctness_gates"] = [
            "bench_completion"
        ]
        write_json(case_dir / "manifest.json", bad_checklist)
        try:
            validate_artifact(root, require_bench=True, require_profile_events=False)
        except ValidationError as exc:
            assert "missing required gates for touched areas" in str(exc)
        else:
            raise AssertionError("touched-area required gate omission unexpectedly passed")
        write_json(case_dir / "manifest.json", case_manifest)

        bad_checklist = load_json(case_dir / "manifest.json")
        bad_checklist["validation_checklist"]["touched_areas"] = ["unknown_area"]
        write_json(case_dir / "manifest.json", bad_checklist)
        try:
            validate_artifact(root, require_bench=True, require_profile_events=False)
        except ValidationError as exc:
            assert "touched_areas invalid" in str(exc)
        else:
            raise AssertionError("unknown touched area unexpectedly passed")
        write_json(case_dir / "manifest.json", case_manifest)

        bad_checklist = load_json(case_dir / "manifest.json")
        bad_checklist["validation_checklist"]["change_type"] = "default_path"
        bad_checklist["validation_checklist"]["performance_regression_required"] = False
        write_json(case_dir / "manifest.json", bad_checklist)
        try:
            validate_artifact(root, require_bench=True, require_profile_events=False)
        except ValidationError as exc:
            assert "default_path requires performance regression gate" in str(exc)
        else:
            raise AssertionError("publishable default-path artifact without perf gate passed")
        write_json(case_dir / "manifest.json", case_manifest)

        bad_checklist = load_json(case_dir / "manifest.json")
        bad_checklist["validation_checklist"]["change_type"] = "default_path"
        bad_checklist["validation_checklist"]["performance_regression_required"] = True
        bad_checklist["validation_checklist"]["baseline_case"] = "baseline"
        write_json(case_dir / "manifest.json", bad_checklist)
        try:
            validate_artifact(root, require_bench=True, require_profile_events=False)
        except ValidationError as exc:
            assert "full concurrency sweep" in str(exc)
        else:
            raise AssertionError("publishable default-path artifact without full sweep passed")
        write_json(case_dir / "manifest.json", case_manifest)

        api_checklist = load_json(case_dir / "manifest.json")
        api_checklist["validation_checklist"]["change_type"] = "api_only"
        api_checklist["validation_checklist"]["touched_areas"] = ["openai_server_api"]
        api_checklist["validation_checklist"]["performance_regression_required"] = False
        write_json(case_dir / "manifest.json", api_checklist)
        try:
            validate_artifact(root, require_bench=True, require_profile_events=False)
        except ValidationError as exc:
            assert "benchmark_impact proof" in str(exc)
        else:
            raise AssertionError("api-only artifact without benchmark impact proof passed")

        api_checklist["validation_checklist"]["benchmark_impact"] = {
            "m3_benchmark_exercised": True,
            "reason": "fixture says API path is exercised",
            "evidence": "self-test",
        }
        write_json(case_dir / "manifest.json", api_checklist)
        try:
            validate_artifact(root, require_bench=True, require_profile_events=False)
        except ValidationError as exc:
            assert "set performance_regression_required=true" in str(exc)
        else:
            raise AssertionError("api-only exercised benchmark without perf gate passed")

        api_checklist["validation_checklist"]["benchmark_impact"] = {
            "m3_benchmark_exercised": False,
            "reason": "OpenAI HTTP route is not used by M3 bench-serve hot path",
            "evidence": "touched_areas=openai_server_api; bench uses generated endpoint flow only",
        }
        write_json(case_dir / "manifest.json", api_checklist)
        result = validate_artifact(root, require_bench=True, require_profile_events=False)
        assert result["ok"]
        write_json(case_dir / "manifest.json", case_manifest)

        for missing_key in sorted(PROFILE_REQUIRED):
            bad_profile = dict(profile_event)
            del bad_profile[missing_key]
            (case_dir / "bad-profile.jsonl").write_text(
                json.dumps(bad_profile, sort_keys=True) + "\n"
            )
            try:
                validate_profile_jsonl(case_dir / "bad-profile.jsonl", require_events=True)
            except ValidationError as exc:
                assert f"missing keys: {missing_key}" in str(exc)
            else:
                raise AssertionError(
                    f"missing profile field {missing_key} unexpectedly passed"
                )

        bad_profile = dict(profile_event)
        bad_profile["runtime_flags"] = []
        (case_dir / "bad-profile.jsonl").write_text(
            json.dumps(bad_profile, sort_keys=True) + "\n"
        )
        try:
            validate_profile_jsonl(case_dir / "bad-profile.jsonl", require_events=True)
        except ValidationError as exc:
            assert "runtime_flags" in str(exc)
        else:
            raise AssertionError("invalid profile event unexpectedly passed")
    print("m3_validate_runner_artifact self-test ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path, nargs="?", help="runner artifact root")
    parser.add_argument("--require-bench", action="store_true", default=True)
    parser.add_argument("--no-require-bench", dest="require_bench", action="store_false")
    parser.add_argument("--require-profile-events", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.artifact is None:
        parser.error("artifact is required unless --self-test is set")
    try:
        result = validate_artifact(
            args.artifact,
            require_bench=args.require_bench,
            require_profile_events=args.require_profile_events,
        )
    except ValidationError as exc:
        print(f"artifact validation failed: {exc}", flush=True)
        raise SystemExit(1)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
