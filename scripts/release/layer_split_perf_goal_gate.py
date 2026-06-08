#!/usr/bin/env python3
"""Final validator for the Llama layer-split performance goal."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS_PREFIX = "LAYER_SPLIT_PERF GOAL PASS"
SELFTEST_PASS = "LAYER_SPLIT_PERF GOAL SELFTEST PASS"
REQUIRED_CONCURRENCY_CELLS = {1, 4, 8, 16}
THROUGHPUT_TARGET_CELLS = {4, 8, 16}
FIXED_PUBLIC_TARGET_TPS = 27.6
STRETCH_TARGET_TPS = 33.0
PUBLIC_REFERENCE_TPS = 34.5
REQUIRED_EFFECTIVE_CONFIG_FIELDS = {
    "selected_distributed_strategy",
    "selected_layer_split_plan",
    "selected_pipeline_mode",
    "selected_microbatch_size",
    "selected_stage_bridge",
    "selected_max_sequences",
    "selected_max_batched_tokens",
    "selected_admission_limit",
    "selected_kv_capacity",
    "selected_max_model_len",
}
PIPELINE_CACHE_METRICS_FILES = [
    "serve.health.after.json",
    "server.health.after.json",
    "health.after.json",
    "cache_metrics.json",
    "serve.cache_metrics.json",
    "health.json",
    "serve.health.json",
    "server.health.json",
]
REQUIRED_CORRECTNESS_CHECKS = {
    "ferrum_run_single",
    "ferrum_run_multiturn",
    "ferrum_serve_single",
    "ferrum_serve_multiturn",
    "streaming_done",
    "streaming_usage",
    "tool_calling",
    "structured_output",
    "log_scan",
}
BAD_COUNT_FIELDS = {
    "errored_per_run",
    "bad_output_per_run",
    "malformed_stream_per_run",
    "missing_done_per_run",
    "duplicate_done_per_run",
    "zero_output_tokens_per_run",
    "stream_bulk_flush_per_run",
    "http_500_per_run",
    "panic_per_run",
}
OPTIONAL_BAD_COUNT_FIELDS = {"failed_per_run"}
BAD_LOG_PATTERNS = [
    "panic",
    "cuda error",
    "out of memory",
    " oom",
    "<unk>",
    "[pad]",
    "mojibake",
    "duplicate [done]",
    "missing [done]",
    "malformed sse",
    "silent fallback",
]


class ValidationError(Exception):
    pass


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValidationError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def require_dir(path: Path, label: str) -> Path:
    if not path.exists():
        raise ValidationError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise ValidationError(f"{label} must be a directory: {path}")
    return path


def status_pass_obj(data: Any, label: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValidationError(f"{label} must contain a JSON object")
    status = data.get("status")
    if status is not None and status != "pass":
        raise ValidationError(f"{label} status is not pass: {status!r}")
    if data.get("ok") is False or data.get("passed") is False:
        raise ValidationError(f"{label} is not passing")
    return data


def require_true(value: Any, label: str) -> None:
    if value is not True:
        raise ValidationError(f"{label} must be true")


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return status_pass_obj(load_json(path), str(path))


def first_json(path: Path, rels: list[str], label: str) -> dict[str, Any]:
    for rel in rels:
        candidate = path / rel
        if candidate.is_file():
            return status_pass_obj(load_json(candidate), str(candidate))
    raise ValidationError(f"{label} missing one of: {', '.join(rels)}")


def is_clean_dirty_status(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("is_dirty") is False
    if isinstance(value, str):
        return value.strip() in {"", "clean"}
    return False


def validate_metadata(path: Path, label: str, rels: list[str] | None = None) -> dict[str, Any]:
    metadata = first_json(
        path,
        rels or ["metadata.json", "gate.manifest.json"],
        f"{label} metadata",
    )
    git_sha = metadata.get("git_sha")
    if not isinstance(git_sha, str) or len(git_sha.strip()) < 7:
        raise ValidationError(f"{label}: missing git_sha")
    if not is_clean_dirty_status(metadata.get("dirty_status")):
        raise ValidationError(f"{label}: dirty or missing git metadata")
    digest = metadata.get("binary_sha256")
    if digest is None and isinstance(metadata.get("binary"), dict):
        digest = metadata["binary"].get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValidationError(f"{label}: missing 64-char binary SHA256")
    if metadata.get("diagnostic_only") is True:
        raise ValidationError(f"{label}: artifact is diagnostic-only")
    model = model_identity(metadata)
    if not model:
        raise ValidationError(f"{label}: metadata missing model_id or model_path")
    for key in ["cuda_version", "driver_version"]:
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip() or value == "unknown":
            raise ValidationError(f"{label}: metadata missing {key}")
    for key in ["gpu_names", "gpu_uuids"]:
        value = metadata.get(key)
        if not isinstance(value, list) or len(value) != 2 or not all(value):
            raise ValidationError(f"{label}: metadata {key} must contain two GPUs")
    for key in ["requested_gpu_devices", "selected_gpu_devices"]:
        if metadata.get(key) != [0, 1]:
            raise ValidationError(f"{label}: metadata {key} must be [0, 1]")
    return metadata


def binary_digest(metadata: dict[str, Any]) -> str:
    digest = metadata.get("binary_sha256")
    if digest is None and isinstance(metadata.get("binary"), dict):
        digest = metadata["binary"].get("sha256")
    return str(digest or "")


def validate_same_metadata_value(
    baseline_metadata: dict[str, Any],
    candidate_metadata: dict[str, Any],
    key: str,
) -> None:
    if baseline_metadata.get(key) != candidate_metadata.get(key):
        raise ValidationError(f"baseline and candidate metadata {key} differ")


def validate_baseline_candidate_metadata_match(
    baseline_metadata: dict[str, Any],
    candidate_metadata: dict[str, Any],
) -> None:
    validate_same_metadata_value(baseline_metadata, candidate_metadata, "git_sha")
    if binary_digest(baseline_metadata) != binary_digest(candidate_metadata):
        raise ValidationError("baseline and candidate binary SHA256 differ")
    for key in [
        "cuda_version",
        "driver_version",
        "gpu_names",
        "gpu_uuids",
        "requested_gpu_devices",
        "selected_gpu_devices",
    ]:
        validate_same_metadata_value(baseline_metadata, candidate_metadata, key)


def require_int_range(value: Any, label: str, *, minimum: int, maximum: int | None = None) -> int:
    if not isinstance(value, int):
        raise ValidationError(f"{label} must be an integer")
    if value < minimum:
        raise ValidationError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValidationError(f"{label} must be <= {maximum}")
    return value


def validate_gpu_rows(
    rows: Any,
    label: str,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValidationError(f"{label}: must contain exactly two GPU rows")
    gpu_names = metadata.get("gpu_names")
    gpu_uuids = metadata.get("gpu_uuids")
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValidationError(f"{label}: GPU row {idx} must be an object")
        if row.get("name") != gpu_names[idx]:
            raise ValidationError(f"{label}: GPU row {idx} name does not match metadata")
        if row.get("uuid") != gpu_uuids[idx]:
            raise ValidationError(f"{label}: GPU row {idx} uuid does not match metadata")
        normalized_row = {
            "index": row.get("index"),
            "name": row.get("name"),
            "uuid": row.get("uuid"),
            "memory_total_mib": require_int_range(
                row.get("memory_total_mib"), f"{label}: GPU {idx} memory_total_mib", minimum=1
            ),
            "memory_used_mib": require_int_range(
                row.get("memory_used_mib"), f"{label}: GPU {idx} memory_used_mib", minimum=0
            ),
            "utilization_gpu_percent": require_int_range(
                row.get("utilization_gpu_percent"),
                f"{label}: GPU {idx} utilization_gpu_percent",
                minimum=0,
                maximum=100,
            ),
            "utilization_memory_percent": require_int_range(
                row.get("utilization_memory_percent"),
                f"{label}: GPU {idx} utilization_memory_percent",
                minimum=0,
                maximum=100,
            ),
            "pcie_link_gen_current": require_int_range(
                row.get("pcie_link_gen_current"),
                f"{label}: GPU {idx} pcie_link_gen_current",
                minimum=1,
            ),
            "pcie_link_width_current": require_int_range(
                row.get("pcie_link_width_current"),
                f"{label}: GPU {idx} pcie_link_width_current",
                minimum=1,
            ),
        }
        normalized.append(normalized_row)
    return normalized


def validate_hardware_evidence(
    path: Path,
    label: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    hardware = first_json(path, ["hardware.json"], f"{label} hardware")
    if hardware.get("cuda_device_count") != 2:
        raise ValidationError(f"{label}: hardware cuda_device_count must be 2")
    rows = validate_gpu_rows(hardware.get("gpus"), f"{label} hardware.json", metadata)
    snapshots: dict[str, Any] = {}
    for snapshot_label in ["before", "during", "after"]:
        snapshot = first_json(
            path,
            [f"nvidia-smi.{snapshot_label}.json"],
            f"{label} nvidia-smi {snapshot_label} snapshot",
        )
        snapshot_rows = validate_gpu_rows(
            snapshot.get("gpus"),
            f"{label} nvidia-smi.{snapshot_label}.json",
            metadata,
        )
        snapshots[snapshot_label] = {
            "gpu_utilization_percent": [
                row["utilization_gpu_percent"] for row in snapshot_rows
            ],
            "memory_used_mib": [row["memory_used_mib"] for row in snapshot_rows],
            "pcie_link_width_current": [
                row["pcie_link_width_current"] for row in snapshot_rows
            ],
        }
    samples_path = path / "nvidia-smi.bench.samples.jsonl"
    if not samples_path.is_file():
        raise ValidationError(f"{label}: missing nvidia-smi.bench.samples.jsonl")
    sample_count = 0
    max_gpu_utilization = [0, 0]
    for line_no, line in enumerate(samples_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"{label}: invalid nvidia-smi bench sample JSON at line {line_no}"
            ) from exc
        if not isinstance(sample, dict):
            raise ValidationError(f"{label}: bench sample line {line_no} must be an object")
        if sample.get("status") != "pass":
            continue
        sample_rows = validate_gpu_rows(
            sample.get("gpus"),
            f"{label} nvidia-smi.bench.samples.jsonl line {line_no}",
            metadata,
        )
        sample_count += 1
        for idx, row in enumerate(sample_rows):
            max_gpu_utilization[idx] = max(
                max_gpu_utilization[idx], row["utilization_gpu_percent"]
            )
    if sample_count <= 0:
        raise ValidationError(f"{label}: missing passing bench-period GPU samples")
    if any(value <= 0 for value in max_gpu_utilization):
        raise ValidationError(
            f"{label}: bench-period GPU samples must show non-zero utilization on both GPUs"
        )
    return {
        "gpu_names": metadata.get("gpu_names"),
        "gpu_uuids": metadata.get("gpu_uuids"),
        "pcie_link_width_current": [row["pcie_link_width_current"] for row in rows],
        "pcie_link_gen_current": [row["pcie_link_gen_current"] for row in rows],
        "memory_total_mib": [row["memory_total_mib"] for row in rows],
        "snapshots": snapshots,
        "bench_sample_count": sample_count,
        "bench_max_gpu_utilization_percent": max_gpu_utilization,
    }


def validate_optional_vllm_metadata(
    optional_vllm_artifact: Path,
    candidate_metadata: dict[str, Any],
) -> dict[str, Any]:
    metadata = validate_metadata(
        optional_vllm_artifact,
        "vllm",
        rels=["vllm-baseline.metadata.json", "metadata.json", "gate.manifest.json"],
    )
    if metadata.get("engine") != "vllm":
        raise ValidationError("vllm metadata engine must be vllm")
    if model_identity(metadata) != model_identity(candidate_metadata):
        raise ValidationError("vllm and candidate model identity differ")
    for key in [
        "cuda_version",
        "driver_version",
        "gpu_names",
        "gpu_uuids",
        "requested_gpu_devices",
        "selected_gpu_devices",
    ]:
        if metadata.get(key) != candidate_metadata.get(key):
            raise ValidationError(f"vllm and candidate metadata {key} differ")
    return metadata


def validate_hardware_summaries_match(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    for key in [
        "pcie_link_width_current",
        "pcie_link_gen_current",
        "memory_total_mib",
    ]:
        if baseline.get(key) != candidate.get(key):
            raise ValidationError(f"baseline and candidate hardware {key} differ")


def model_identity(metadata: dict[str, Any]) -> str:
    value = metadata.get("model_id") or metadata.get("model_path") or metadata.get("model")
    if isinstance(value, dict):
        value = value.get("id") or value.get("path")
    return str(value) if value else ""


def is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def validate_file_manifest_item(item: Any, label: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValidationError(f"{label}: file manifest item must be an object")
    path = item.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValidationError(f"{label}: file manifest item missing path")
    try:
        size = int(item.get("size_bytes"))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label}: file manifest item has invalid size_bytes") from exc
    if size <= 0:
        raise ValidationError(f"{label}: file manifest item size_bytes must be > 0")
    if not is_sha256_digest(item.get("sha256")):
        raise ValidationError(f"{label}: file manifest item missing SHA256")
    return item


def validate_model_manifest(
    path: Path,
    label: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = path / "model_manifest.json"
    if not manifest_path.is_file():
        raise ValidationError(f"{label}: missing model_manifest.json")
    manifest = status_pass_obj(load_json(manifest_path), f"{label} model manifest")
    if manifest.get("diagnostic_only") is True:
        raise ValidationError(f"{label}: model manifest is diagnostic-only")
    if manifest.get("model") != model_identity(metadata):
        raise ValidationError(f"{label}: model manifest identity does not match metadata")
    if not isinstance(manifest.get("model_path"), str) or not manifest["model_path"].strip():
        raise ValidationError(f"{label}: model manifest missing model_path")
    for key in [
        "config_sha256",
        "tokenizer_sha256",
        "tokenizer_metadata_sha256",
        "weight_manifest_sha256",
    ]:
        if not is_sha256_digest(manifest.get(key)):
            raise ValidationError(f"{label}: model manifest missing {key}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValidationError(f"{label}: model manifest missing file list")
    for item in files:
        validate_file_manifest_item(item, f"{label} model manifest")
    tokenizer_files = manifest.get("tokenizer_files")
    if not isinstance(tokenizer_files, list) or not tokenizer_files:
        raise ValidationError(f"{label}: model manifest missing tokenizer metadata files")
    for item in tokenizer_files:
        validate_file_manifest_item(item, f"{label} tokenizer manifest")
    if int(manifest.get("weight_file_count") or 0) <= 0:
        raise ValidationError(f"{label}: model manifest missing weight files")
    return {
        "model": manifest.get("model"),
        "model_path": manifest.get("model_path"),
        "resolved_from": manifest.get("resolved_from"),
        "config_sha256": manifest.get("config_sha256"),
        "tokenizer_sha256": manifest.get("tokenizer_sha256"),
        "tokenizer_metadata_sha256": manifest.get("tokenizer_metadata_sha256"),
        "weight_manifest_sha256": manifest.get("weight_manifest_sha256"),
        "file_count": len(files),
        "tokenizer_file_count": len(tokenizer_files),
        "weight_file_count": int(manifest.get("weight_file_count") or 0),
    }


def validate_model_manifests_match(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    for key in [
        "model",
        "config_sha256",
        "tokenizer_sha256",
        "tokenizer_metadata_sha256",
        "weight_manifest_sha256",
        "weight_file_count",
    ]:
        if baseline.get(key) != candidate.get(key):
            raise ValidationError(f"baseline and candidate model manifest {key} differ")


def validate_effective_config(path: Path, label: str) -> dict[str, Any]:
    config = first_json(
        path,
        ["effective_config.json", "serve.effective_config.json"],
        f"{label} effective config",
    )
    missing = sorted(REQUIRED_EFFECTIVE_CONFIG_FIELDS - set(config))
    if missing:
        raise ValidationError(f"{label}: effective config missing fields: {', '.join(missing)}")
    if config["selected_distributed_strategy"] != "layer_split":
        raise ValidationError(f"{label}: selected_distributed_strategy must be layer_split")
    if not isinstance(config["selected_layer_split_plan"], str) or not config[
        "selected_layer_split_plan"
    ].strip():
        raise ValidationError(f"{label}: selected_layer_split_plan must be non-empty")
    if config["selected_pipeline_mode"] not in {"sequential", "batch", "overlapped"}:
        raise ValidationError(f"{label}: invalid selected_pipeline_mode")
    if config["selected_stage_bridge"] not in {"host", "cuda_peer", "cuda_device_staged"}:
        raise ValidationError(f"{label}: invalid selected_stage_bridge")
    for key in [
        "selected_microbatch_size",
        "selected_max_sequences",
        "selected_max_batched_tokens",
        "selected_admission_limit",
        "selected_kv_capacity",
        "selected_max_model_len",
    ]:
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise ValidationError(f"{label}: {key} must be a positive integer")
    if config["selected_pipeline_mode"] == "overlapped":
        expected_microbatch = max(1, (config["selected_max_sequences"] + 1) // 2)
        if config["selected_microbatch_size"] != expected_microbatch:
            raise ValidationError(
                f"{label}: overlapped selected_microbatch_size must be {expected_microbatch}"
            )
    elif config["selected_pipeline_mode"] == "batch":
        if config["selected_microbatch_size"] != config["selected_max_sequences"]:
            raise ValidationError(
                f"{label}: batch selected_microbatch_size must equal selected_max_sequences"
            )
    elif config["selected_pipeline_mode"] == "sequential" and config["selected_microbatch_size"] != 1:
        raise ValidationError(f"{label}: sequential selected_microbatch_size must be 1")
    if config.get("diagnostic_only") is True:
        raise ValidationError(f"{label}: effective config is diagnostic-only")
    return config


def pipeline_metric_candidates(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    candidates: list[dict[str, Any]] = [data]
    for key in ["cache_metrics", "engine_cache", "model_cache", "prefix_cache"]:
        value = data.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    cache = data.get("cache")
    if isinstance(cache, dict):
        candidates.append(cache)
        prefix = cache.get("prefix_cache")
        if isinstance(prefix, dict):
            candidates.append(prefix)
    health = data.get("health")
    if isinstance(health, dict):
        candidates.extend(pipeline_metric_candidates(health))
    return candidates


def extract_pipeline_cache_metrics(data: Any, label: str) -> dict[str, Any]:
    for candidate in pipeline_metric_candidates(data):
        if (
            candidate.get("position") == "llama-layer-split-pipeline"
            or "selected_stage_bridge" in candidate
            or "selected_pipeline_mode" in candidate
        ):
            return candidate
    raise ValidationError(f"{label}: cache metrics missing llama layer-split pipeline snapshot")


def summarize_pipeline_cache_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "position": metrics.get("position"),
        "stage_count": metrics.get("stage_count"),
        "stage_device_ordinals": metrics.get("stage_device_ordinals"),
        "transport": metrics.get("transport"),
        "selected_pipeline_mode": metrics.get("selected_pipeline_mode"),
        "selected_stage_bridge": metrics.get("selected_stage_bridge"),
        "pipeline_hidden": metrics.get("pipeline_hidden"),
        "pipeline_decode": metrics.get("pipeline_decode"),
    }


def json_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int):
        raise ValidationError(f"{label} must be an integer")
    if value < minimum:
        raise ValidationError(f"{label} must be >= {minimum}")
    return value


def validate_admission_health(
    path: Path,
    label: str,
    effective_config: dict[str, Any],
) -> dict[str, Any]:
    health = None
    for rel in ["serve.health.after.json", "server.health.after.json", "health.after.json"]:
        candidate = path / rel
        if candidate.is_file():
            health = load_json(candidate)
            break
    if not isinstance(health, dict):
        raise ValidationError(f"{label}: missing post-bench health JSON")
    if health.get("status") not in {"healthy", "pass", None}:
        raise ValidationError(f"{label}: post-bench health status is not healthy")
    admission = health.get("admission")
    if not isinstance(admission, dict):
        raise ValidationError(f"{label}: post-bench health missing admission object")
    if admission.get("schema_version") != 1:
        raise ValidationError(f"{label}: admission schema_version must be 1")
    effective_max = json_int(
        admission.get("effective_max_concurrent"),
        f"{label}: admission.effective_max_concurrent",
        minimum=1,
    )
    if effective_max != effective_config["selected_admission_limit"]:
        raise ValidationError(
            f"{label}: admission effective_max_concurrent does not match effective config"
        )
    summary = {
        "effective_max_concurrent": effective_max,
        "queue_depth": json_int(admission.get("queue_depth"), f"{label}: admission.queue_depth"),
        "active_prefill": json_int(
            admission.get("active_prefill"), f"{label}: admission.active_prefill"
        ),
        "active_decode": json_int(
            admission.get("active_decode"), f"{label}: admission.active_decode"
        ),
        "current_batch_size": json_int(
            admission.get("current_batch_size"), f"{label}: admission.current_batch_size"
        ),
        "rejected_requests_total": json_int(
            admission.get("rejected_requests_total"),
            f"{label}: admission.rejected_requests_total",
        ),
        "failed_requests_total": json_int(
            admission.get("failed_requests_total"),
            f"{label}: admission.failed_requests_total",
        ),
        "completed_requests_total": json_int(
            admission.get("completed_requests_total"),
            f"{label}: admission.completed_requests_total",
        ),
        "scheduler_policy": admission.get("scheduler_policy"),
    }
    if summary["rejected_requests_total"] != 0:
        raise ValidationError(f"{label}: admission rejected_requests_total must be 0")
    if summary["failed_requests_total"] != 0:
        raise ValidationError(f"{label}: admission failed_requests_total must be 0")
    if summary["completed_requests_total"] <= 0:
        raise ValidationError(f"{label}: admission completed_requests_total must be > 0")

    scheduler = health.get("scheduler")
    if not isinstance(scheduler, dict):
        raise ValidationError(f"{label}: post-bench health missing scheduler object")
    scheduler_summary = {
        "total_requests": json_int(
            scheduler.get("total_requests"), f"{label}: scheduler.total_requests"
        ),
        "successful_requests": json_int(
            scheduler.get("successful_requests"), f"{label}: scheduler.successful_requests"
        ),
        "failed_requests": json_int(
            scheduler.get("failed_requests"), f"{label}: scheduler.failed_requests"
        ),
    }
    if scheduler_summary["failed_requests"] != 0:
        raise ValidationError(f"{label}: scheduler failed_requests must be 0")
    if scheduler_summary["successful_requests"] <= 0:
        raise ValidationError(f"{label}: scheduler successful_requests must be > 0")
    summary["scheduler"] = scheduler_summary
    return summary


def validate_pipeline_cache_metrics(
    path: Path, label: str, effective_config: dict[str, Any]
) -> dict[str, Any]:
    metrics: dict[str, Any] | None = None
    source: Path | None = None
    for rel in PIPELINE_CACHE_METRICS_FILES:
        candidate = path / rel
        if candidate.is_file():
            source = candidate
            metrics = extract_pipeline_cache_metrics(load_json(candidate), label)
            break
    if metrics is None:
        raise ValidationError(
            f"{label}: missing pipeline cache metrics; expected one of "
            + ", ".join(PIPELINE_CACHE_METRICS_FILES)
        )
    if metrics.get("position") != "llama-layer-split-pipeline":
        raise ValidationError(f"{label}: cache metrics position must be llama-layer-split-pipeline")
    if metrics.get("selected_pipeline_mode") != effective_config["selected_pipeline_mode"]:
        raise ValidationError(
            f"{label}: cache metrics selected_pipeline_mode does not match effective config"
        )
    if metrics.get("selected_stage_bridge") != effective_config["selected_stage_bridge"]:
        raise ValidationError(
            f"{label}: cache metrics selected_stage_bridge does not match effective config"
        )
    stage_count = metrics.get("stage_count")
    if not isinstance(stage_count, int) or stage_count < 2:
        raise ValidationError(f"{label}: cache metrics stage_count must be an integer >= 2")
    stage_device_ordinals = metrics.get("stage_device_ordinals")
    if not isinstance(stage_device_ordinals, list) or len(stage_device_ordinals) != stage_count:
        raise ValidationError(
            f"{label}: cache metrics stage_device_ordinals must match stage_count"
        )
    if not isinstance(metrics.get("transport"), str) or not metrics["transport"].strip():
        raise ValidationError(f"{label}: cache metrics transport must be non-empty")
    hidden = metrics.get("pipeline_hidden")
    if not isinstance(hidden, dict):
        raise ValidationError(f"{label}: cache metrics missing pipeline_hidden object")
    if hidden.get("dtype") != "f32":
        raise ValidationError(f"{label}: cache metrics pipeline_hidden.dtype must be f32")
    if hidden.get("layout") != "row_major":
        raise ValidationError(f"{label}: cache metrics pipeline_hidden.layout must be row_major")
    bridge = metrics["selected_stage_bridge"]
    hidden_device = hidden.get("device")
    if bridge == "host" and hidden_device != "host":
        raise ValidationError(f"{label}: host bridge must report host pipeline hidden device")
    if bridge in {"cuda_peer", "cuda_device_staged"} and hidden_device == "host":
        raise ValidationError(
            f"{label}: device bridge cannot report host pipeline hidden device"
        )
    decode = metrics.get("pipeline_decode")
    if not isinstance(decode, dict):
        raise ValidationError(f"{label}: cache metrics missing pipeline_decode object")
    if not isinstance(decode.get("calls"), int) or decode["calls"] <= 0:
        raise ValidationError(f"{label}: pipeline_decode.calls must be > 0")
    if not isinstance(decode.get("rows"), int) or decode["rows"] < decode["calls"]:
        raise ValidationError(f"{label}: pipeline_decode.rows must be >= calls")
    if not isinstance(decode.get("max_batch"), int) or decode["max_batch"] <= 0:
        raise ValidationError(f"{label}: pipeline_decode.max_batch must be > 0")
    if not isinstance(decode.get("microbatch_count_max"), int) or decode["microbatch_count_max"] <= 0:
        raise ValidationError(f"{label}: pipeline_decode.microbatch_count_max must be > 0")
    if not isinstance(decode.get("microbatch_size_max"), int) or decode["microbatch_size_max"] <= 0:
        raise ValidationError(f"{label}: pipeline_decode.microbatch_size_max must be > 0")
    if decode["microbatch_size_max"] > effective_config["selected_microbatch_size"]:
        raise ValidationError(
            f"{label}: pipeline_decode.microbatch_size_max exceeds selected_microbatch_size"
        )
    if decode["max_batch"] > effective_config["selected_max_sequences"]:
        raise ValidationError(f"{label}: pipeline_decode.max_batch exceeds selected_max_sequences")
    if (
        not isinstance(decode.get("in_flight_stage_count_max"), int)
        or decode["in_flight_stage_count_max"] <= 0
    ):
        raise ValidationError(f"{label}: pipeline_decode.in_flight_stage_count_max must be > 0")
    if not isinstance(decode.get("queue_depth_max"), int) or decode["queue_depth_max"] < 0:
        raise ValidationError(f"{label}: pipeline_decode.queue_depth_max must be >= 0")
    if metrics["selected_pipeline_mode"] == "overlapped":
        if not isinstance(decode.get("overlapped_calls"), int) or decode["overlapped_calls"] <= 0:
            raise ValidationError(f"{label}: overlapped pipeline must report overlapped_calls > 0")
        if decode["microbatch_count_max"] < 2:
            raise ValidationError(f"{label}: overlapped pipeline must report microbatch_count_max >= 2")
        if decode["in_flight_stage_count_max"] < 2:
            raise ValidationError(
                f"{label}: overlapped pipeline must report in_flight_stage_count_max >= 2"
            )
    for key in [
        "host_bridge_bytes_total",
        "logits_us_total",
        "total_us_total",
    ]:
        if not isinstance(decode.get(key), int) or decode[key] <= 0:
            raise ValidationError(f"{label}: pipeline_decode.{key} must be > 0")
    for key in ["stage_us_total", "stage_us_last", "stage_us_avg"]:
        values = decode.get(key)
        if not isinstance(values, list) or len(values) != stage_count:
            raise ValidationError(
                f"{label}: pipeline_decode.{key} must be a stage_count-length list"
            )
    if not all(isinstance(value, int) and value > 0 for value in decode["stage_us_total"]):
        raise ValidationError(f"{label}: pipeline_decode.stage_us_total values must be > 0")
    metrics = dict(metrics)
    metrics["_source"] = str(source)
    return metrics


def command_list(path: Path, label: str) -> list[str]:
    data = load_json(path)
    raw = None
    if isinstance(data, dict):
        raw = data.get("cmd") or data.get("bench_cmd")
    else:
        raw = data
    if not isinstance(raw, list) or not raw:
        raise ValidationError(f"{label}: command must be a non-empty list")
    return [str(part) for part in raw]


def flag_value(cmd: list[str], flag: str) -> str | None:
    prefix = flag + "="
    for idx, part in enumerate(cmd):
        if part == flag and idx + 1 < len(cmd):
            return cmd[idx + 1]
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def normalized_concurrency_sweep(value: str | None, label: str) -> str:
    if value is None:
        raise ValidationError(f"{label}: command missing --concurrency-sweep")
    try:
        cells = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise ValidationError(f"{label}: invalid --concurrency-sweep") from exc
    missing = sorted(REQUIRED_CONCURRENCY_CELLS - set(cells))
    if missing:
        raise ValidationError(f"{label}: command missing concurrency cells {missing}")
    return ",".join(str(cell) for cell in cells)


def require_product_command(path: Path, label: str, subcommand: str, pipeline_mode: str) -> dict[str, str]:
    file = path / f"{subcommand}.command.json"
    cmd = command_list(file, f"{label} {subcommand}")
    if subcommand not in cmd:
        raise ValidationError(f"{label}: {subcommand} command is not ferrum {subcommand}")
    if flag_value(cmd, "--backend") != "cuda":
        raise ValidationError(f"{label}: {subcommand} command must use --backend cuda")
    if flag_value(cmd, "--gpu-devices") != "0,1":
        raise ValidationError(f"{label}: {subcommand} command must use --gpu-devices 0,1")
    if flag_value(cmd, "--layer-split-pipeline-mode") != pipeline_mode:
        raise ValidationError(
            f"{label}: {subcommand} command must use --layer-split-pipeline-mode {pipeline_mode}"
        )
    if flag_value(cmd, "--effective-config-json") is None:
        raise ValidationError(f"{label}: {subcommand} command missing --effective-config-json")
    data = load_json(file)
    if isinstance(data, dict):
        expected = data.get("expected_layer_split_pipeline_mode")
        if expected is not None and expected != pipeline_mode:
            raise ValidationError(
                f"{label}: {subcommand} expected_layer_split_pipeline_mode mismatch"
            )
    return {
        "backend": flag_value(cmd, "--backend") or "",
        "gpu_devices": flag_value(cmd, "--gpu-devices") or "",
        "pipeline_mode": pipeline_mode,
    }


def require_bench_command(path: Path, label: str, expected_model: str) -> dict[str, str]:
    return require_bench_command_file(
        path / "bench-serve.command.json",
        label,
        expected_model=expected_model,
        expected_output_name="bench-serve.json",
    )


def require_bench_command_file(
    path: Path,
    label: str,
    *,
    expected_model: str,
    expected_output_name: str,
) -> dict[str, str]:
    cmd = command_list(path, label)
    if "bench-serve" not in cmd:
        raise ValidationError(f"{label}: command is not ferrum bench-serve")
    base_url = flag_value(cmd, "--base-url")
    if base_url is None or not (
        base_url.startswith("http://127.0.0.1:") or base_url.startswith("http://localhost:")
    ):
        raise ValidationError(f"{label}: bench command must use localhost --base-url")
    if flag_value(cmd, "--model") != expected_model:
        raise ValidationError(f"{label}: bench command --model must be {expected_model}")
    tokenizer = flag_value(cmd, "--tokenizer")
    if tokenizer is None or not tokenizer.strip():
        raise ValidationError(f"{label}: bench command missing --tokenizer")
    out_file = flag_value(cmd, "--out")
    if out_file is None or Path(out_file).name != expected_output_name:
        raise ValidationError(
            f"{label}: bench command --out must write {expected_output_name}"
        )
    for flag in ["--fail-on-error", "--require-ci"]:
        if flag not in cmd:
            raise ValidationError(f"{label}: bench command missing {flag}")
    if flag_value(cmd, "--seed") != "9271":
        raise ValidationError(f"{label}: bench command must use --seed 9271")
    try:
        repeats = int(flag_value(cmd, "--n-repeats") or "0")
    except ValueError as exc:
        raise ValidationError(f"{label}: invalid --n-repeats") from exc
    if repeats < 3:
        raise ValidationError(f"{label}: bench command must use --n-repeats >= 3")
    sweep = normalized_concurrency_sweep(flag_value(cmd, "--concurrency-sweep"), label)
    required_values = {
        "--dataset": "random",
        "--random-input-len": "256",
        "--random-output-len": "128",
        "--num-prompts": "96",
        "--warmup-requests": "10",
        "--output": "json",
    }
    for flag, expected in required_values.items():
        if flag_value(cmd, flag) != expected:
            raise ValidationError(f"{label}: bench command must use {flag} {expected}")
    return {
        "model": flag_value(cmd, "--model") or "",
        "tokenizer": tokenizer,
        "dataset": flag_value(cmd, "--dataset") or "",
        "random_input_len": flag_value(cmd, "--random-input-len") or "",
        "random_output_len": flag_value(cmd, "--random-output-len") or "",
        "num_prompts": flag_value(cmd, "--num-prompts") or "",
        "warmup_requests": flag_value(cmd, "--warmup-requests") or "",
        "n_repeats": str(repeats),
        "concurrency_sweep": sweep,
        "seed": flag_value(cmd, "--seed") or "",
        "output": flag_value(cmd, "--output") or "",
    }


def validate_vllm_bench_artifact(path: Path, expected_model: str) -> dict[str, Any]:
    path = require_dir(path, "vllm artifact")
    command_signature = require_bench_command_file(
        path / "vllm-baseline.command.json",
        "vllm",
        expected_model=expected_model,
        expected_output_name="vllm-baseline.json",
    )
    reports = reports_from_json(load_json(path / "vllm-baseline.json"), "vllm")
    rows: dict[int, dict[str, Any]] = {}
    throughput: dict[int, float] = {}
    expected_repeats = int(command_signature["n_repeats"])
    expected_num_prompts = int(command_signature["num_prompts"])
    for report in reports:
        concurrency, tps = validate_report(
            report,
            "vllm",
            expected_repeats=expected_repeats,
            expected_num_prompts=expected_num_prompts,
        )
        rows[concurrency] = report
        throughput[concurrency] = tps
    missing = sorted(REQUIRED_CONCURRENCY_CELLS - set(rows))
    if missing:
        raise ValidationError(f"vllm: bench reports missing cells {missing}")
    return {
        "command_signature": command_signature,
        "throughput_by_concurrency": throughput,
        "max_target_tps": max(throughput[c] for c in THROUGHPUT_TARGET_CELLS),
    }


def reports_from_json(data: Any, label: str) -> list[dict[str, Any]]:
    reports = data if isinstance(data, list) else [data]
    if not reports or not all(isinstance(report, dict) for report in reports):
        raise ValidationError(f"{label}: bench-serve.json must contain report object(s)")
    return reports


def scalar_mean(report: dict[str, Any], key: str) -> float | None:
    value: Any = report.get(key)
    if isinstance(value, dict):
        value = value.get("mean", value.get("value"))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def nested_positive(report: dict[str, Any], key: str, percentile: str) -> bool:
    value: Any = report.get(key)
    if isinstance(value, dict):
        value = value.get(percentile)
    if isinstance(value, dict):
        value = value.get("mean", value.get("value"))
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def require_number(value: Any, label: str, *, positive: bool) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be numeric") from exc
    if positive and number <= 0:
        raise ValidationError(f"{label} must be > 0")
    if not positive and number < 0:
        raise ValidationError(f"{label} must be >= 0")
    return number


def require_scalar_stats(value: Any, label: str) -> float:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must contain mean/stddev/ci95_hw")
    mean = require_number(value.get("mean"), f"{label}.mean", positive=True)
    require_number(value.get("stddev"), f"{label}.stddev", positive=False)
    require_number(value.get("ci95_hw"), f"{label}.ci95_hw", positive=False)
    return mean


def require_percentile_stats(
    report: dict[str, Any],
    metric: str,
    percentile: str,
    label: str,
) -> None:
    metric_obj = report.get(metric)
    if not isinstance(metric_obj, dict):
        raise ValidationError(f"{label}: missing {metric}")
    require_scalar_stats(metric_obj.get(percentile), f"{label}: {metric}.{percentile}")


def list_sum(value: Any) -> int:
    if value is None:
        return 0
    if not isinstance(value, list):
        raise ValidationError(f"expected list, got {type(value).__name__}")
    return sum(int(item) for item in value)


def require_int_list(
    report: dict[str, Any],
    field: str,
    label: str,
    expected_len: int,
) -> list[int]:
    value = report.get(field)
    if not isinstance(value, list):
        raise ValidationError(f"{label}: missing {field}")
    if len(value) != expected_len:
        raise ValidationError(
            f"{label}: {field} length {len(value)} != n_repeats {expected_len}"
        )
    ints: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValidationError(f"{label}: {field} contains non-integer {item!r}")
        if item < 0:
            raise ValidationError(f"{label}: {field} contains negative value {item}")
        ints.append(item)
    return ints


def validate_actual_input_tokens(
    report: dict[str, Any],
    label: str,
    expected_repeats: int,
    expected_num_prompts: int,
) -> None:
    stats = report.get("actual_input_tokens")
    if not isinstance(stats, dict):
        raise ValidationError(f"{label}: missing actual_input_tokens stats")
    for key in ["min", "max", "mean"]:
        require_number(stats.get(key), f"{label}: actual_input_tokens.{key}", positive=True)
    per_request = report.get("actual_input_tokens_per_request")
    if not isinstance(per_request, list):
        raise ValidationError(f"{label}: missing actual_input_tokens_per_request")
    if len(per_request) != expected_repeats:
        raise ValidationError(
            f"{label}: actual_input_tokens_per_request length "
            f"{len(per_request)} != n_repeats {expected_repeats}"
        )
    for repeat_idx, values in enumerate(per_request):
        if not isinstance(values, list):
            raise ValidationError(
                f"{label}: actual_input_tokens_per_request[{repeat_idx}] must be a list"
            )
        if len(values) != expected_num_prompts:
            raise ValidationError(
                f"{label}: actual_input_tokens_per_request[{repeat_idx}] length "
                f"{len(values)} != num_prompts {expected_num_prompts}"
            )
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(
                    f"{label}: actual_input_tokens_per_request contains invalid value {value!r}"
                )


def validate_report(
    report: dict[str, Any],
    label: str,
    *,
    expected_repeats: int,
    expected_num_prompts: int,
) -> tuple[int, float]:
    try:
        concurrency = int(report.get("concurrency") or report.get("c") or 0)
    except ValueError as exc:
        raise ValidationError(f"{label}: invalid concurrency") from exc
    if concurrency <= 0:
        raise ValidationError(f"{label}: missing positive concurrency")
    if report.get("diagnostic_only") is True:
        raise ValidationError(f"{label} c{concurrency}: report is diagnostic-only")
    n_repeats = int(report.get("n_repeats") or 0)
    if n_repeats < 3:
        raise ValidationError(f"{label} c{concurrency}: n_repeats must be >= 3")
    if n_repeats != expected_repeats:
        raise ValidationError(
            f"{label} c{concurrency}: n_repeats {n_repeats} != command n_repeats "
            f"{expected_repeats}"
        )
    if int(report.get("n_requests_per_run") or 0) != expected_num_prompts:
        raise ValidationError(
            f"{label} c{concurrency}: n_requests_per_run must be {expected_num_prompts}"
        )
    if report.get("output_token_count_source") != "usage":
        raise ValidationError(
            f"{label} c{concurrency}: output_token_count_source must be usage"
        )
    completed_values = require_int_list(
        report, "completed_per_run", f"{label} c{concurrency}", n_repeats
    )
    for value in completed_values:
        if value != expected_num_prompts:
            raise ValidationError(
                f"{label} c{concurrency}: completed_per_run must be "
                f"{expected_num_prompts} for every repeat"
            )
    for field in BAD_COUNT_FIELDS:
        values = require_int_list(report, field, f"{label} c{concurrency}", n_repeats)
        if sum(values) != 0:
            raise ValidationError(f"{label} c{concurrency}: {field} must sum to 0")
    for field in OPTIONAL_BAD_COUNT_FIELDS:
        if field not in report:
            continue
        values = require_int_list(report, field, f"{label} c{concurrency}", n_repeats)
        if sum(values) != 0:
            raise ValidationError(f"{label} c{concurrency}: {field} must sum to 0")
    throughput = require_scalar_stats(
        report.get("output_throughput_tps"),
        f"{label} c{concurrency}: output_throughput_tps",
    )
    for metric, percentile in [
        ("ttft_ms", "p50"),
        ("tpot_ms", "p50"),
        ("e2e_ms", "p95"),
    ]:
        require_percentile_stats(report, metric, percentile, f"{label} c{concurrency}")
    validate_actual_input_tokens(
        report,
        f"{label} c{concurrency}",
        n_repeats,
        expected_num_prompts,
    )
    return concurrency, throughput


def validate_bench_artifact(path: Path, label: str, expected_model: str) -> dict[str, Any]:
    command_signature = require_bench_command(path, label, expected_model)
    reports = reports_from_json(load_json(path / "bench-serve.json"), label)
    rows: dict[int, dict[str, Any]] = {}
    throughput: dict[int, float] = {}
    expected_repeats = int(command_signature["n_repeats"])
    expected_num_prompts = int(command_signature["num_prompts"])
    for report in reports:
        concurrency, tps = validate_report(
            report,
            label,
            expected_repeats=expected_repeats,
            expected_num_prompts=expected_num_prompts,
        )
        rows[concurrency] = report
        throughput[concurrency] = tps
    missing = sorted(REQUIRED_CONCURRENCY_CELLS - set(rows))
    if missing:
        raise ValidationError(f"{label}: bench reports missing cells {missing}")
    return {
        "command_signature": command_signature,
        "throughput_by_concurrency": throughput,
        "max_target_tps": max(throughput[c] for c in THROUGHPUT_TARGET_CELLS),
    }


def validate_correctness_artifact(path: Path, label: str = "correctness artifact") -> dict[str, Any]:
    path = require_dir(path, label)
    data = first_json(path, ["correctness.json", "gate.json", "gate.manifest.json"], label)
    if data.get("diagnostic_only") is True:
        raise ValidationError(f"{label}: artifact is diagnostic-only")
    checks = data.get("checks")
    if not isinstance(checks, dict):
        raise ValidationError(f"{label}: must include checks object")
    missing = sorted(REQUIRED_CORRECTNESS_CHECKS - set(checks))
    if missing:
        raise ValidationError(f"{label}: missing checks: " + ", ".join(missing))
    for name in sorted(REQUIRED_CORRECTNESS_CHECKS):
        status_pass_obj(checks[name], f"{label} check {name}")
    run_single = checks["ferrum_run_single"]
    assistant_turns = run_single.get("assistant_turns")
    if not isinstance(assistant_turns, int) or assistant_turns < 1:
        raise ValidationError(f"{label}: ferrum_run_single.assistant_turns must be >= 1")
    require_true(
        checks["ferrum_run_multiturn"].get("has_precise_recall"),
        f"{label}: ferrum_run_multiturn.has_precise_recall",
    )
    require_true(
        checks["ferrum_serve_single"].get("contains_expected_answer"),
        f"{label}: ferrum_serve_single.contains_expected_answer",
    )
    require_true(
        checks["ferrum_serve_multiturn"].get("has_precise_recall"),
        f"{label}: ferrum_serve_multiturn.has_precise_recall",
    )
    streaming_done = checks["streaming_done"]
    if int(streaming_done.get("done_count", 0)) != 1:
        raise ValidationError(f"{label}: streaming_done.done_count must be exactly 1")
    streaming_usage = checks["streaming_usage"]
    if streaming_usage.get("include_usage") is not True:
        raise ValidationError(f"{label}: streaming_usage.include_usage must be true")
    if streaming_usage.get("usage_received") is not True:
        raise ValidationError(f"{label}: streaming usage was not received")
    if int(streaming_usage.get("usage_chunk_count", 0)) != 1:
        raise ValidationError(f"{label}: streaming_usage.usage_chunk_count must be exactly 1")
    tool_calling = checks["tool_calling"]
    details = tool_calling.get("details")
    if isinstance(details, dict):
        status_pass_obj(details, f"{label}: tool_calling.details")
        detail_checks = details.get("checks")
        if isinstance(detail_checks, dict):
            if not detail_checks:
                raise ValidationError(f"{label}: tool_calling.details.checks must be non-empty")
            for detail_name, detail in sorted(detail_checks.items()):
                status_pass_obj(detail, f"{label}: tool_calling.details.{detail_name}")
    else:
        require_true(tool_calling.get("tool_call_ok"), f"{label}: tool_calling.tool_call_ok")
    structured = checks["structured_output"]
    require_true(structured.get("json_ok"), f"{label}: structured_output.json_ok")
    require_true(structured.get("schema_ok"), f"{label}: structured_output.schema_ok")
    log_scan = checks["log_scan"]
    if int(log_scan.get("bad_pattern_count", 0)) != 0:
        raise ValidationError(f"{label}: log_scan.bad_pattern_count must be 0")
    return {"checks": sorted(checks)}


def scan_artifact_logs(path: Path, label: str) -> None:
    for rel in [
        "run.stdout",
        "run.stderr",
        "serve.log",
        "bench-serve.stdout",
        "bench-serve.stderr",
    ]:
        file = path / rel
        if not file.is_file():
            continue
        text = file.read_text(errors="replace").lower()
        for pattern in BAD_LOG_PATTERNS:
            if pattern in text:
                raise ValidationError(f"{label}: bad log pattern {pattern!r} in {file}")


def vllm_tps_from_artifact(path: Path) -> float:
    path = require_dir(path, "vllm artifact")
    candidates = ["vllm-baseline.json", "bench-serve.json", "vllm.json"]
    data = None
    for rel in candidates:
        file = path / rel
        if file.is_file():
            data = load_json(file)
            break
    if data is None:
        data = load_json(path)
    reports = reports_from_json(data, "vllm")
    values: list[float] = []
    for report in reports:
        c = int(report.get("concurrency") or report.get("c") or 0)
        if c not in THROUGHPUT_TARGET_CELLS:
            continue
        tps = scalar_mean(report, "output_throughput_tps")
        if tps is None:
            tps = scalar_mean(report, "output_throughput")
        if tps is not None:
            values.append(tps)
    if not values:
        raise ValidationError("vllm artifact missing c4/c8/c16 throughput")
    return max(values)


def validate_perf_goal(
    *,
    out_dir: Path,
    baseline_artifact: Path,
    candidate_artifact: Path,
    correctness_artifact: Path,
    optional_vllm_artifact: Path | None,
) -> dict[str, Any]:
    baseline_artifact = require_dir(baseline_artifact, "baseline artifact")
    candidate_artifact = require_dir(candidate_artifact, "candidate artifact")

    baseline_metadata = validate_metadata(baseline_artifact, "baseline")
    candidate_metadata = validate_metadata(candidate_artifact, "candidate")
    baseline_hardware = validate_hardware_evidence(
        baseline_artifact, "baseline", baseline_metadata
    )
    candidate_hardware = validate_hardware_evidence(
        candidate_artifact, "candidate", candidate_metadata
    )
    validate_hardware_summaries_match(baseline_hardware, candidate_hardware)
    baseline_model_manifest = validate_model_manifest(
        baseline_artifact, "baseline", baseline_metadata
    )
    candidate_model_manifest = validate_model_manifest(
        candidate_artifact, "candidate", candidate_metadata
    )
    validate_model_manifests_match(baseline_model_manifest, candidate_model_manifest)
    baseline_config = validate_effective_config(baseline_artifact, "baseline")
    candidate_config = validate_effective_config(candidate_artifact, "candidate")
    if model_identity(baseline_metadata) != model_identity(candidate_metadata):
        raise ValidationError("baseline and candidate model identity differ")
    validate_baseline_candidate_metadata_match(baseline_metadata, candidate_metadata)
    if baseline_config["selected_layer_split_plan"] != candidate_config["selected_layer_split_plan"]:
        raise ValidationError("baseline and candidate layer split plans differ")
    if baseline_config["selected_pipeline_mode"] != "batch":
        raise ValidationError("baseline selected_pipeline_mode must be batch")
    if candidate_config["selected_pipeline_mode"] != "overlapped":
        raise ValidationError("candidate selected_pipeline_mode must be overlapped")
    baseline_admission = validate_admission_health(
        baseline_artifact, "baseline", baseline_config
    )
    candidate_admission = validate_admission_health(
        candidate_artifact, "candidate", candidate_config
    )
    baseline_run_command = require_product_command(
        baseline_artifact, "baseline", "run", baseline_config["selected_pipeline_mode"]
    )
    baseline_serve_command = require_product_command(
        baseline_artifact, "baseline", "serve", baseline_config["selected_pipeline_mode"]
    )
    candidate_run_command = require_product_command(
        candidate_artifact, "candidate", "run", candidate_config["selected_pipeline_mode"]
    )
    candidate_serve_command = require_product_command(
        candidate_artifact, "candidate", "serve", candidate_config["selected_pipeline_mode"]
    )
    baseline_cache_metrics = validate_pipeline_cache_metrics(
        baseline_artifact, "baseline", baseline_config
    )
    candidate_cache_metrics = validate_pipeline_cache_metrics(
        candidate_artifact, "candidate", candidate_config
    )

    baseline_bench = validate_bench_artifact(
        baseline_artifact, "baseline", model_identity(baseline_metadata)
    )
    candidate_bench = validate_bench_artifact(
        candidate_artifact, "candidate", model_identity(candidate_metadata)
    )
    if baseline_bench["command_signature"] != candidate_bench["command_signature"]:
        raise ValidationError("baseline and candidate bench command parameters differ")
    baseline_correctness = validate_correctness_artifact(
        baseline_artifact, "baseline correctness artifact"
    )
    candidate_correctness = validate_correctness_artifact(
        candidate_artifact, "candidate correctness artifact"
    )
    correctness = validate_correctness_artifact(correctness_artifact, "correctness artifact")
    scan_artifact_logs(baseline_artifact, "baseline")
    scan_artifact_logs(candidate_artifact, "candidate")

    vllm_tps = None
    vllm_metadata = None
    vllm_bench = None
    target_tps = FIXED_PUBLIC_TARGET_TPS
    target_mode = "fixed_public_lower_bound"
    if optional_vllm_artifact is not None:
        vllm_metadata = validate_optional_vllm_metadata(
            optional_vllm_artifact, candidate_metadata
        )
        vllm_bench = validate_vllm_bench_artifact(
            optional_vllm_artifact, model_identity(candidate_metadata)
        )
        if vllm_bench["command_signature"] != candidate_bench["command_signature"]:
            raise ValidationError("vllm and candidate bench command parameters differ")
        vllm_tps = float(vllm_bench["max_target_tps"])
        if vllm_tps > PUBLIC_REFERENCE_TPS:
            target_tps = 0.80 * vllm_tps
            target_mode = "same_pod_vllm_80pct"

    baseline_max = float(baseline_bench["max_target_tps"])
    candidate_max = float(candidate_bench["max_target_tps"])
    if candidate_max <= baseline_max:
        raise ValidationError(
            f"candidate max c4/c8/c16 throughput {candidate_max:.3f} <= baseline {baseline_max:.3f}"
        )
    if candidate_max < target_tps:
        raise ValidationError(
            f"candidate max c4/c8/c16 throughput {candidate_max:.3f} < target {target_tps:.3f}"
        )
    fixed_public_target_passed = candidate_max >= FIXED_PUBLIC_TARGET_TPS
    same_pod_vllm_target_tps = 0.80 * vllm_tps if vllm_tps is not None else None
    same_pod_vllm_target_passed = (
        candidate_max >= same_pod_vllm_target_tps
        if same_pod_vllm_target_tps is not None
        else None
    )
    if same_pod_vllm_target_passed is True and fixed_public_target_passed:
        target_pass_summary = "fixed_public_lower_bound_and_same_pod_vllm_80pct"
    elif same_pod_vllm_target_passed is True:
        target_pass_summary = "same_pod_vllm_80pct_only"
    elif fixed_public_target_passed:
        target_pass_summary = "fixed_public_lower_bound_only"
    else:
        target_pass_summary = "none"

    pass_line = f"{PASS_PREFIX}: {out_dir}"
    result = {
        "schema_version": 1,
        "status": "pass",
        "created_at": iso_now(),
        "baseline_artifact": str(baseline_artifact),
        "candidate_artifact": str(candidate_artifact),
        "correctness_artifact": str(correctness_artifact),
        "optional_vllm_artifact": str(optional_vllm_artifact) if optional_vllm_artifact else None,
        "model": model_identity(candidate_metadata),
        "git_sha": candidate_metadata["git_sha"],
        "binary_sha256": binary_digest(candidate_metadata),
        "cuda_version": candidate_metadata.get("cuda_version"),
        "driver_version": candidate_metadata.get("driver_version"),
        "gpu_names": candidate_metadata.get("gpu_names"),
        "gpu_uuids": candidate_metadata.get("gpu_uuids"),
        "baseline_hardware_evidence": baseline_hardware,
        "hardware_evidence": candidate_hardware,
        "model_manifest": candidate_model_manifest,
        "selected_layer_split_plan": candidate_config["selected_layer_split_plan"],
        "selected_pipeline_mode": candidate_config["selected_pipeline_mode"],
        "selected_microbatch_size": candidate_config["selected_microbatch_size"],
        "selected_stage_bridge": candidate_config["selected_stage_bridge"],
        "baseline_product_commands": {
            "run": baseline_run_command,
            "serve": baseline_serve_command,
        },
        "candidate_product_commands": {
            "run": candidate_run_command,
            "serve": candidate_serve_command,
        },
        "bench_command_signature": candidate_bench["command_signature"],
        "baseline_admission": baseline_admission,
        "candidate_admission": candidate_admission,
        "baseline_pipeline_cache_metrics": summarize_pipeline_cache_metrics(
            baseline_cache_metrics
        ),
        "candidate_pipeline_cache_metrics": summarize_pipeline_cache_metrics(
            candidate_cache_metrics
        ),
        "target_mode": target_mode,
        "target_pass_summary": target_pass_summary,
        "target_output_tps": target_tps,
        "fixed_public_target_tps": FIXED_PUBLIC_TARGET_TPS,
        "fixed_public_target_passed": fixed_public_target_passed,
        "same_pod_vllm_target_tps": same_pod_vllm_target_tps,
        "same_pod_vllm_target_passed": same_pod_vllm_target_passed,
        "stretch_output_tps": STRETCH_TARGET_TPS,
        "stretch_passed": candidate_max >= STRETCH_TARGET_TPS,
        "same_pod_vllm_output_tps": vllm_tps,
        "same_pod_vllm_throughput_by_concurrency": vllm_bench["throughput_by_concurrency"]
        if vllm_bench is not None
        else None,
        "same_pod_vllm_metadata": {
            "git_sha": vllm_metadata.get("git_sha"),
            "cuda_version": vllm_metadata.get("cuda_version"),
            "driver_version": vllm_metadata.get("driver_version"),
            "gpu_names": vllm_metadata.get("gpu_names"),
            "gpu_uuids": vllm_metadata.get("gpu_uuids"),
        }
        if vllm_metadata is not None
        else None,
        "baseline_max_c4_c8_c16_output_tps": baseline_max,
        "candidate_max_c4_c8_c16_output_tps": candidate_max,
        "baseline_throughput_by_concurrency": baseline_bench["throughput_by_concurrency"],
        "candidate_throughput_by_concurrency": candidate_bench["throughput_by_concurrency"],
        "baseline_correctness": baseline_correctness,
        "candidate_correctness": candidate_correctness,
        "correctness": correctness,
        "pass_line": pass_line,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "layer_split_perf_goal_gate.json", result)
    write_text(
        out_dir / "summary.md",
        "\n".join(
            [
                "# Layer Split Performance Goal Gate",
                "",
                f"- Status: pass",
                f"- Target mode: {target_mode}",
                f"- Target pass summary: {target_pass_summary}",
                f"- Target output throughput: {target_tps:.3f} tok/s",
                f"- Fixed public lower bound passed: {fixed_public_target_passed}",
                f"- Same-pod vLLM 80% passed: {same_pod_vllm_target_passed}",
                f"- Baseline max c4/c8/c16: {baseline_max:.3f} tok/s",
                f"- Candidate max c4/c8/c16: {candidate_max:.3f} tok/s",
                f"- Stretch passed: {candidate_max >= STRETCH_TARGET_TPS}",
                "",
                pass_line,
                "",
            ]
        ),
    )
    return result


def bench_report(concurrency: int, tps: float) -> dict[str, Any]:
    return {
        "concurrency": concurrency,
        "n_requests_per_run": 96,
        "completed_per_run": [96, 96, 96],
        "failed_per_run": [0, 0, 0],
        "errored_per_run": [0, 0, 0],
        "n_repeats": 3,
        "output_token_count_source": "usage",
        "output_throughput_tps": {"mean": tps, "stddev": 0.1, "ci95_hw": 0.2},
        "ttft_ms": {"p50": {"mean": 500.0, "stddev": 1.0, "ci95_hw": 2.0}},
        "tpot_ms": {"p50": {"mean": 40.0, "stddev": 0.1, "ci95_hw": 0.2}},
        "e2e_ms": {"p95": {"mean": 6000.0, "stddev": 10.0, "ci95_hw": 20.0}},
        "actual_input_tokens": {"requested": 256, "mean": 256.0, "min": 256, "max": 256},
        "actual_input_tokens_per_request": [[256] * 96 for _ in range(3)],
        "bad_output_per_run": [0, 0, 0],
        "malformed_stream_per_run": [0, 0, 0],
        "missing_done_per_run": [0, 0, 0],
        "duplicate_done_per_run": [0, 0, 0],
        "zero_output_tokens_per_run": [0, 0, 0],
        "stream_bulk_flush_per_run": [0, 0, 0],
        "http_500_per_run": [0, 0, 0],
        "panic_per_run": [0, 0, 0],
    }


def make_perf_artifact(
    root: Path,
    *,
    tps_by_c: dict[int, float],
    pipeline_mode: str,
    diagnostic: bool = False,
) -> None:
    metadata = {
        "schema_version": 1,
        "status": "pass",
        "git_sha": "abcdef0123456789abcdef0123456789abcdef01",
        "dirty_status": {"is_dirty": False, "status_short": []},
        "binary_sha256": "a" * 64,
        "model_id": "clowman/Llama-3.3-70B-Instruct-GPTQ-Int4",
        "cuda_version": "12.4",
        "driver_version": "550.54.15",
        "gpu_names": ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"],
        "gpu_uuids": ["GPU-baseline-0", "GPU-baseline-1"],
        "requested_gpu_devices": [0, 1],
        "selected_gpu_devices": [0, 1],
        "distributed_strategy": "layer_split",
        "layer_split_plan": "stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79",
        "diagnostic_only": diagnostic,
    }
    write_json(root / "metadata.json", metadata)
    gpu_rows = [
        {
            "index": 0,
            "name": metadata["gpu_names"][0],
            "uuid": metadata["gpu_uuids"][0],
            "driver_version": metadata["driver_version"],
            "memory_total_mib": 24564,
            "memory_used_mib": 20480,
            "utilization_gpu_percent": 25,
            "utilization_memory_percent": 18,
            "pcie_link_gen_current": 4,
            "pcie_link_width_current": 16,
        },
        {
            "index": 1,
            "name": metadata["gpu_names"][1],
            "uuid": metadata["gpu_uuids"][1],
            "driver_version": metadata["driver_version"],
            "memory_total_mib": 24564,
            "memory_used_mib": 20480,
            "utilization_gpu_percent": 27,
            "utilization_memory_percent": 19,
            "pcie_link_gen_current": 4,
            "pcie_link_width_current": 16,
        },
    ]
    write_json(
        root / "hardware.json",
        {
            "schema_version": 1,
            "status": "pass",
            "cuda_device_count": 2,
            "cuda_version": metadata["cuda_version"],
            "driver_version": metadata["driver_version"],
            "gpu_names": metadata["gpu_names"],
            "gpu_uuids": metadata["gpu_uuids"],
            "gpu_utilization_percent": [25, 27],
            "gpu_memory_utilization_percent": [18, 19],
            "pcie_link_gen_current": [4, 4],
            "pcie_link_width_current": [16, 16],
            "gpus": gpu_rows,
        },
    )
    for snapshot_label, util in [("before", 0), ("during", 82), ("after", 5)]:
        snapshot_rows = [
            dict(row, utilization_gpu_percent=util + idx) for idx, row in enumerate(gpu_rows)
        ]
        write_json(
            root / f"nvidia-smi.{snapshot_label}.json",
            {
                "schema_version": 1,
                "status": "pass",
                "gpus": snapshot_rows,
            },
        )
    write_text(
        root / "nvidia-smi.bench.samples.jsonl",
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "sample_phase": "bench",
                "elapsed_sec": 15.0,
                "gpus": [
                    dict(gpu_rows[0], utilization_gpu_percent=84),
                    dict(gpu_rows[1], utilization_gpu_percent=87),
                ],
            },
            sort_keys=True,
        )
        + "\n",
    )
    write_json(
        root / "model_manifest.json",
        {
            "schema_version": 2,
            "status": "pass",
            "model": metadata["model_id"],
            "model_id": metadata["model_id"],
            "model_path": "/models/clowman/Llama-3.3-70B-Instruct-GPTQ-Int4",
            "resolved_from": "hf_cache_snapshot",
            "quant_format": "gptq_int4",
            "tokenizer_path": "/models/clowman/Llama-3.3-70B-Instruct-GPTQ-Int4",
            "config_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64,
            "tokenizer_metadata_sha256": "d" * 64,
            "weight_manifest_sha256": "e" * 64,
            "weight_file_count": 2,
            "files": [
                {"path": "config.json", "size_bytes": 20, "sha256": "b" * 64},
                {"path": "tokenizer.json", "size_bytes": 30, "sha256": "c" * 64},
                {
                    "path": "model-00001-of-00002.safetensors",
                    "size_bytes": 40,
                    "sha256": "f" * 64,
                },
                {
                    "path": "model-00002-of-00002.safetensors",
                    "size_bytes": 40,
                    "sha256": "1" * 64,
                },
            ],
            "tokenizer_files": [
                {"path": "tokenizer.json", "size_bytes": 30, "sha256": "c" * 64}
            ],
        },
    )
    for subcommand in ["run", "serve"]:
        write_json(
            root / f"{subcommand}.command.json",
            {
                "status": "run",
                "cmd": [
                    "ferrum",
                    subcommand,
                    metadata["model_id"],
                    "--backend",
                    "cuda",
                    "--gpu-devices",
                    "0,1",
                    "--layer-split-pipeline-mode",
                    pipeline_mode,
                    "--effective-config-json",
                    str(root / f"{subcommand}.effective_config.json"),
                ],
                "expected_layer_split_pipeline_mode": pipeline_mode,
            },
        )
    effective = {
        "status": "pass",
        "selected_distributed_strategy": "layer_split",
        "selected_layer_split_plan": metadata["layer_split_plan"],
        "selected_pipeline_mode": pipeline_mode,
        "selected_microbatch_size": 8 if pipeline_mode == "overlapped" else 16,
        "selected_stage_bridge": "host",
        "selected_max_sequences": 16,
        "selected_max_batched_tokens": 1024,
        "selected_admission_limit": 16,
        "selected_kv_capacity": 2048,
        "selected_max_model_len": 8192,
    }
    write_json(root / "effective_config.json", effective)
    write_json(
        root / "serve.health.json",
        {
            "status": "healthy",
            "cache": {
                "prefix_cache": {
                    "position": "llama-layer-split-pipeline",
                    "stage_count": 2,
                    "stage_device_ordinals": [0, 1],
                    "transport": "host-hidden-bridge",
                    "selected_pipeline_mode": effective["selected_pipeline_mode"],
                    "selected_stage_bridge": effective["selected_stage_bridge"],
                    "pipeline_hidden": {
                        "dtype": "f32",
                        "device": "host",
                        "layout": "row_major",
                    },
                    "pipeline_decode": {
                        "calls": 0,
                        "overlapped_calls": 0,
                        "rows": 0,
                        "max_batch": 0,
                        "last_batch": 0,
                        "microbatch_count_max": 0,
                        "microbatch_count_last": 0,
                        "microbatch_size_max": 0,
                        "microbatch_size_last": 0,
                        "in_flight_stage_count_max": 0,
                        "in_flight_stage_count_last": 0,
                        "queue_depth_max": 0,
                        "queue_depth_last": 0,
                        "host_bridge_bytes_total": 0,
                        "host_bridge_bytes_last": 0,
                        "host_bridge_bytes_avg": None,
                        "stage_us_total": [0, 0],
                        "stage_us_last": [0, 0],
                        "stage_us_avg": [None, None],
                        "logits_us_total": 0,
                        "logits_us_last": 0,
                        "logits_us_avg": None,
                        "total_us_total": 0,
                        "total_us_last": 0,
                        "total_us_avg": None,
                    },
                }
            },
        },
    )
    write_json(
        root / "serve.health.after.json",
        {
            "status": "healthy",
            "admission": {
                "schema_version": 1,
                "effective_max_concurrent": effective["selected_admission_limit"],
                "queue_depth": 0,
                "active_prefill": 0,
                "active_decode": 0,
                "current_batch_size": 0,
                "rejected_requests_total": 0,
                "failed_requests_total": 0,
                "completed_requests_total": 384,
                "scheduler_policy": "active_decode_prefill_chunk:64",
            },
            "scheduler": {
                "total_requests": 384,
                "successful_requests": 384,
                "failed_requests": 0,
                "throughput_rps": 3.5,
            },
            "cache": {
                "prefix_cache": {
                    "position": "llama-layer-split-pipeline",
                    "stage_count": 2,
                    "stage_device_ordinals": [0, 1],
                    "transport": "host-hidden-bridge",
                    "selected_pipeline_mode": effective["selected_pipeline_mode"],
                    "selected_stage_bridge": effective["selected_stage_bridge"],
                    "pipeline_hidden": {
                        "dtype": "f32",
                        "device": "host",
                        "layout": "row_major",
                    },
                    "pipeline_decode": {
                        "calls": 8,
                        "overlapped_calls": 8 if pipeline_mode == "overlapped" else 0,
                        "rows": 32,
                        "max_batch": 4,
                        "last_batch": 4,
                        "microbatch_count_max": 2 if pipeline_mode == "overlapped" else 1,
                        "microbatch_count_last": 2 if pipeline_mode == "overlapped" else 1,
                        "microbatch_size_max": 2 if pipeline_mode == "overlapped" else 4,
                        "microbatch_size_last": 2 if pipeline_mode == "overlapped" else 4,
                        "in_flight_stage_count_max": 2 if pipeline_mode == "overlapped" else 1,
                        "in_flight_stage_count_last": 2 if pipeline_mode == "overlapped" else 1,
                        "queue_depth_max": 1 if pipeline_mode == "overlapped" else 0,
                        "queue_depth_last": 1 if pipeline_mode == "overlapped" else 0,
                        "host_bridge_bytes_total": 1048576,
                        "host_bridge_bytes_last": 131072,
                        "host_bridge_bytes_avg": 131072,
                        "stage_us_total": [8000, 9000],
                        "stage_us_last": [1000, 1100],
                        "stage_us_avg": [1000, 1125],
                        "logits_us_total": 2400,
                        "logits_us_last": 300,
                        "logits_us_avg": 300,
                        "total_us_total": 22000,
                        "total_us_last": 2750,
                        "total_us_avg": 2750,
                    },
                }
            },
        },
    )
    write_json(
        root / "bench-serve.command.json",
        {
            "cmd": [
                "ferrum",
                "bench-serve",
                "--base-url",
                "http://127.0.0.1:19400",
                "--model",
                metadata["model_id"],
                "--tokenizer",
                "/models/clowman/Llama-3.3-70B-Instruct-GPTQ-Int4",
                "--dataset",
                "random",
                "--fail-on-error",
                "--require-ci",
                "--seed",
                "9271",
                "--n-repeats",
                "3",
                "--concurrency-sweep",
                "1,4,8,16",
                "--random-input-len",
                "256",
                "--random-output-len",
                "128",
                "--num-prompts",
                "96",
                "--warmup-requests",
                "10",
                "--output",
                "json",
                "--out",
                str(root / "bench-serve.json"),
                "--tag",
                "cuda-llama33-70b-4bit-2x4090",
            ]
        },
    )
    write_json(
        root / "bench-serve.json",
        [bench_report(c, tps_by_c[c]) for c in sorted(REQUIRED_CONCURRENCY_CELLS)],
    )
    write_text(root / "run.stdout", "Paris is in France\n")
    write_text(root / "run.stderr", "")
    write_text(root / "serve.log", "server ready\n")
    write_text(root / "bench-serve.stdout", "bench ok\n")
    write_text(root / "bench-serve.stderr", "")
    make_correctness_artifact(root)


def make_correctness_artifact(root: Path) -> None:
    checks = {name: {"status": "pass"} for name in REQUIRED_CORRECTNESS_CHECKS}
    checks["ferrum_run_single"]["assistant_turns"] = 2
    checks["ferrum_run_multiturn"]["has_precise_recall"] = True
    checks["ferrum_serve_single"]["contains_expected_answer"] = True
    checks["ferrum_serve_multiturn"]["has_precise_recall"] = True
    checks["streaming_done"]["done_count"] = 1
    checks["streaming_usage"]["include_usage"] = True
    checks["streaming_usage"]["usage_received"] = True
    checks["streaming_usage"]["usage_chunk_count"] = 1
    checks["tool_calling"]["details"] = {
        "status": "pass",
        "checks": {
            "omitted_tool_choice": {"passed": True},
            "explicit_auto_tool_choice": {"passed": True},
            "required_tool_choice": {"passed": True},
            "tool_result_fill": {"passed": True},
        },
    }
    checks["structured_output"]["json_ok"] = True
    checks["structured_output"]["schema_ok"] = True
    checks["log_scan"]["bad_pattern_count"] = 0
    write_json(root / "correctness.json", {"status": "pass", "checks": checks})


def make_vllm_artifact(root: Path, tps: float) -> None:
    write_json(
        root / "vllm-baseline.metadata.json",
        {
            "schema_version": 1,
            "status": "pass",
            "engine": "vllm",
            "git_sha": "abcdef0123456789abcdef0123456789abcdef01",
            "dirty_status": {"is_dirty": False, "status_short": []},
            "binary_sha256": "a" * 64,
            "model_id": "clowman/Llama-3.3-70B-Instruct-GPTQ-Int4",
            "cuda_version": "12.4",
            "driver_version": "550.54.15",
            "gpu_names": ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"],
            "gpu_uuids": ["GPU-baseline-0", "GPU-baseline-1"],
            "requested_gpu_devices": [0, 1],
            "selected_gpu_devices": [0, 1],
        },
    )
    write_json(
        root / "vllm-baseline.command.json",
        {
            "status": "run",
            "server_cmd": [
                "vllm",
                "serve",
                "clowman/Llama-3.3-70B-Instruct-GPTQ-Int4",
                "--tensor-parallel-size",
                "2",
            ],
            "bench_cmd": [
                "ferrum",
                "bench-serve",
                "--base-url",
                "http://127.0.0.1:19401",
                "--model",
                "clowman/Llama-3.3-70B-Instruct-GPTQ-Int4",
                "--tokenizer",
                "/models/clowman/Llama-3.3-70B-Instruct-GPTQ-Int4",
                "--dataset",
                "random",
                "--fail-on-error",
                "--require-ci",
                "--seed",
                "9271",
                "--n-repeats",
                "3",
                "--concurrency-sweep",
                "1,4,8,16",
                "--random-input-len",
                "256",
                "--random-output-len",
                "128",
                "--num-prompts",
                "96",
                "--warmup-requests",
                "10",
                "--output",
                "json",
                "--out",
                str(root / "vllm-baseline.json"),
                "--tag",
                "vllm-llama33-70b-4bit-2x4090",
            ],
            "same_hardware_required": True,
        },
    )
    write_json(
        root / "vllm-baseline.json",
        [
            bench_report(c, tps if c in THROUGHPUT_TARGET_CELLS else tps - 2.0)
            for c in sorted(REQUIRED_CONCURRENCY_CELLS)
        ],
    )


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-layer-split-perf-goal-") as tmp:
        root = Path(tmp)
        baseline = root / "baseline"
        candidate = root / "candidate"
        correctness = root / "correctness"
        vllm = root / "vllm"
        make_perf_artifact(
            baseline,
            tps_by_c={1: 20.0, 4: 20.5, 8: 20.7, 16: 20.6},
            pipeline_mode="batch",
        )
        make_perf_artifact(
            candidate,
            tps_by_c={1: 21.0, 4: 27.8, 8: 28.4, 16: 28.1},
            pipeline_mode="overlapped",
        )
        make_correctness_artifact(correctness)
        make_vllm_artifact(vllm, 30.0)
        result = validate_perf_goal(
            out_dir=root / "out",
            baseline_artifact=baseline,
            candidate_artifact=candidate,
            correctness_artifact=correctness,
            optional_vllm_artifact=vllm,
        )
        if result["target_mode"] != "fixed_public_lower_bound":
            raise AssertionError("selftest expected fixed target mode")
        if (
            result["target_pass_summary"]
            != "fixed_public_lower_bound_and_same_pod_vllm_80pct"
        ):
            raise AssertionError("selftest expected both target summary for low vLLM")
        if result["fixed_public_target_passed"] is not True:
            raise AssertionError("selftest expected fixed public target pass")
        if result["same_pod_vllm_target_passed"] is not True:
            raise AssertionError("selftest expected same-pod vLLM target pass")

        no_vllm_result = validate_perf_goal(
            out_dir=root / "no-vllm-out",
            baseline_artifact=baseline,
            candidate_artifact=candidate,
            correctness_artifact=correctness,
            optional_vllm_artifact=None,
        )
        if no_vllm_result["target_pass_summary"] != "fixed_public_lower_bound_only":
            raise AssertionError("selftest expected fixed-only target summary without vLLM")
        if no_vllm_result["same_pod_vllm_target_passed"] is not None:
            raise AssertionError("selftest expected no same-pod target without vLLM")

        strong_candidate = root / "strong-candidate"
        strong_vllm = root / "strong-vllm"
        make_perf_artifact(
            strong_candidate,
            tps_by_c={1: 22.0, 4: 34.0, 8: 34.4, 16: 34.2},
            pipeline_mode="overlapped",
        )
        make_vllm_artifact(strong_vllm, 40.0)
        strong_result = validate_perf_goal(
            out_dir=root / "strong-out",
            baseline_artifact=baseline,
            candidate_artifact=strong_candidate,
            correctness_artifact=correctness,
            optional_vllm_artifact=strong_vllm,
        )
        if strong_result["target_mode"] != "same_pod_vllm_80pct":
            raise AssertionError("selftest expected same-pod vLLM target mode")
        if (
            strong_result["target_pass_summary"]
            != "fixed_public_lower_bound_and_same_pod_vllm_80pct"
        ):
            raise AssertionError("selftest expected both target summary")
        if strong_result["same_pod_vllm_target_passed"] is not True:
            raise AssertionError("selftest expected same-pod vLLM target pass")

        bad = root / "bad-candidate"
        make_perf_artifact(
            bad,
            tps_by_c={1: 21.0, 4: 23.0, 8: 24.0, 16: 23.5},
            pipeline_mode="overlapped",
        )
        try:
            validate_perf_goal(
                out_dir=root / "bad-out",
                baseline_artifact=baseline,
                candidate_artifact=bad,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("under-target candidate unexpectedly passed")
        except ValidationError as exc:
            if "target" not in str(exc):
                raise

        diagnostic = root / "diagnostic"
        make_perf_artifact(
            diagnostic,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
            diagnostic=True,
        )
        try:
            validate_perf_goal(
                out_dir=root / "diagnostic-out",
                baseline_artifact=baseline,
                candidate_artifact=diagnostic,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("diagnostic candidate unexpectedly passed")
        except ValidationError as exc:
            if "diagnostic" not in str(exc):
                raise

        missing_candidate_correctness = root / "missing-candidate-correctness"
        make_perf_artifact(
            missing_candidate_correctness,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        (missing_candidate_correctness / "correctness.json").unlink()
        try:
            validate_perf_goal(
                out_dir=root / "missing-candidate-correctness-out",
                baseline_artifact=baseline,
                candidate_artifact=missing_candidate_correctness,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("missing candidate correctness unexpectedly passed")
        except ValidationError as exc:
            if "candidate correctness artifact" not in str(exc):
                raise

        bad_candidate_recall = root / "bad-candidate-recall"
        make_perf_artifact(
            bad_candidate_recall,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        bad_correctness = load_json(bad_candidate_recall / "correctness.json")
        bad_correctness["checks"]["ferrum_run_multiturn"]["has_precise_recall"] = False
        write_json(bad_candidate_recall / "correctness.json", bad_correctness)
        try:
            validate_perf_goal(
                out_dir=root / "bad-candidate-recall-out",
                baseline_artifact=baseline,
                candidate_artifact=bad_candidate_recall,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("bad candidate recall unexpectedly passed")
        except ValidationError as exc:
            if "has_precise_recall" not in str(exc):
                raise

        bad_streaming_usage = root / "bad-streaming-usage"
        make_correctness_artifact(bad_streaming_usage)
        bad_correctness = load_json(bad_streaming_usage / "correctness.json")
        bad_correctness["checks"]["streaming_usage"]["usage_chunk_count"] = 2
        write_json(bad_streaming_usage / "correctness.json", bad_correctness)
        try:
            validate_perf_goal(
                out_dir=root / "bad-streaming-usage-out",
                baseline_artifact=baseline,
                candidate_artifact=candidate,
                correctness_artifact=bad_streaming_usage,
                optional_vllm_artifact=None,
            )
            raise AssertionError("bad streaming usage unexpectedly passed")
        except ValidationError as exc:
            if "usage_chunk_count" not in str(exc):
                raise

        bad_structured = root / "bad-structured"
        make_perf_artifact(
            bad_structured,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        bad_correctness = load_json(bad_structured / "correctness.json")
        bad_correctness["checks"]["structured_output"]["schema_ok"] = False
        write_json(bad_structured / "correctness.json", bad_correctness)
        try:
            validate_perf_goal(
                out_dir=root / "bad-structured-out",
                baseline_artifact=baseline,
                candidate_artifact=bad_structured,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("bad structured output unexpectedly passed")
        except ValidationError as exc:
            if "schema_ok" not in str(exc):
                raise

        bad_tool = root / "bad-tool"
        make_perf_artifact(
            bad_tool,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        bad_correctness = load_json(bad_tool / "correctness.json")
        bad_correctness["checks"]["tool_calling"]["details"]["checks"]["required_tool_choice"][
            "passed"
        ] = False
        write_json(bad_tool / "correctness.json", bad_correctness)
        try:
            validate_perf_goal(
                out_dir=root / "bad-tool-out",
                baseline_artifact=baseline,
                candidate_artifact=bad_tool,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("bad tool-call output unexpectedly passed")
        except ValidationError as exc:
            if "tool_calling" not in str(exc):
                raise

        pending_model_manifest = root / "pending-model-manifest"
        make_perf_artifact(
            pending_model_manifest,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        manifest = load_json(pending_model_manifest / "model_manifest.json")
        manifest["status"] = "pending_model_resolution"
        manifest["config_sha256"] = None
        write_json(pending_model_manifest / "model_manifest.json", manifest)
        try:
            validate_perf_goal(
                out_dir=root / "pending-model-manifest-out",
                baseline_artifact=baseline,
                candidate_artifact=pending_model_manifest,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("pending model manifest candidate unexpectedly passed")
        except ValidationError as exc:
            if "model manifest" not in str(exc):
                raise

        missing_hardware_snapshot = root / "missing-hardware-snapshot"
        make_perf_artifact(
            missing_hardware_snapshot,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        (missing_hardware_snapshot / "nvidia-smi.during.json").unlink()
        try:
            validate_perf_goal(
                out_dir=root / "missing-hardware-snapshot-out",
                baseline_artifact=baseline,
                candidate_artifact=missing_hardware_snapshot,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("missing hardware snapshot candidate unexpectedly passed")
        except ValidationError as exc:
            if "nvidia-smi" not in str(exc):
                raise

        zero_bench_gpu = root / "zero-bench-gpu"
        make_perf_artifact(
            zero_bench_gpu,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        sample = load_json(zero_bench_gpu / "nvidia-smi.during.json")
        for gpu in sample["gpus"]:
            gpu["utilization_gpu_percent"] = 0
        write_text(
            zero_bench_gpu / "nvidia-smi.bench.samples.jsonl",
            json.dumps(sample, sort_keys=True) + "\n",
        )
        try:
            validate_perf_goal(
                out_dir=root / "zero-bench-gpu-out",
                baseline_artifact=baseline,
                candidate_artifact=zero_bench_gpu,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("zero GPU utilization candidate unexpectedly passed")
        except ValidationError as exc:
            if "non-zero utilization" not in str(exc):
                raise

        missing_admission = root / "missing-admission"
        make_perf_artifact(
            missing_admission,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        health = load_json(missing_admission / "serve.health.after.json")
        health.pop("admission", None)
        write_json(missing_admission / "serve.health.after.json", health)
        try:
            validate_perf_goal(
                out_dir=root / "missing-admission-out",
                baseline_artifact=baseline,
                candidate_artifact=missing_admission,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("missing admission candidate unexpectedly passed")
        except ValidationError as exc:
            if "admission" not in str(exc):
                raise

        binary_mismatch = root / "binary-mismatch"
        make_perf_artifact(
            binary_mismatch,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        metadata = load_json(binary_mismatch / "metadata.json")
        metadata["binary_sha256"] = "b" * 64
        write_json(binary_mismatch / "metadata.json", metadata)
        try:
            validate_perf_goal(
                out_dir=root / "binary-mismatch-out",
                baseline_artifact=baseline,
                candidate_artifact=binary_mismatch,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("binary-mismatch candidate unexpectedly passed")
        except ValidationError as exc:
            if "binary SHA256" not in str(exc):
                raise

        hardware_mismatch = root / "hardware-mismatch"
        make_perf_artifact(
            hardware_mismatch,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        metadata = load_json(hardware_mismatch / "metadata.json")
        metadata["gpu_uuids"] = ["GPU-other-0", "GPU-other-1"]
        write_json(hardware_mismatch / "metadata.json", metadata)
        try:
            validate_perf_goal(
                out_dir=root / "hardware-mismatch-out",
                baseline_artifact=baseline,
                candidate_artifact=hardware_mismatch,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("hardware-mismatch candidate unexpectedly passed")
        except ValidationError as exc:
            if "gpu_uuids" not in str(exc) and "uuid" not in str(exc):
                raise

        pcie_mismatch = root / "pcie-mismatch"
        make_perf_artifact(
            pcie_mismatch,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        hardware = load_json(pcie_mismatch / "hardware.json")
        hardware["gpus"][1]["pcie_link_width_current"] = 8
        hardware["pcie_link_width_current"] = [16, 8]
        write_json(pcie_mismatch / "hardware.json", hardware)
        try:
            validate_perf_goal(
                out_dir=root / "pcie-mismatch-out",
                baseline_artifact=baseline,
                candidate_artifact=pcie_mismatch,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("pcie-mismatch candidate unexpectedly passed")
        except ValidationError as exc:
            if "pcie_link_width_current" not in str(exc):
                raise

        wrong_baseline = root / "wrong-baseline"
        make_perf_artifact(
            wrong_baseline,
            tps_by_c={1: 20.0, 4: 20.5, 8: 20.7, 16: 20.6},
            pipeline_mode="overlapped",
        )
        try:
            validate_perf_goal(
                out_dir=root / "wrong-baseline-out",
                baseline_artifact=wrong_baseline,
                candidate_artifact=candidate,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("wrong-baseline candidate unexpectedly passed")
        except ValidationError as exc:
            if "baseline selected_pipeline_mode" not in str(exc):
                raise

        vllm_hardware_mismatch = root / "vllm-hardware-mismatch"
        make_vllm_artifact(vllm_hardware_mismatch, 40.0)
        metadata = load_json(vllm_hardware_mismatch / "vllm-baseline.metadata.json")
        metadata["gpu_uuids"] = ["GPU-vllm-other-0", "GPU-vllm-other-1"]
        write_json(vllm_hardware_mismatch / "vllm-baseline.metadata.json", metadata)
        try:
            validate_perf_goal(
                out_dir=root / "vllm-hardware-mismatch-out",
                baseline_artifact=baseline,
                candidate_artifact=candidate,
                correctness_artifact=correctness,
                optional_vllm_artifact=vllm_hardware_mismatch,
            )
            raise AssertionError("vllm hardware mismatch unexpectedly passed")
        except ValidationError as exc:
            if "vllm and candidate metadata gpu_uuids differ" not in str(exc):
                raise

        vllm_missing_command = root / "vllm-missing-command"
        make_vllm_artifact(vllm_missing_command, 40.0)
        (vllm_missing_command / "vllm-baseline.command.json").unlink()
        try:
            validate_perf_goal(
                out_dir=root / "vllm-missing-command-out",
                baseline_artifact=baseline,
                candidate_artifact=candidate,
                correctness_artifact=correctness,
                optional_vllm_artifact=vllm_missing_command,
            )
            raise AssertionError("vllm missing command unexpectedly passed")
        except ValidationError as exc:
            if "vllm-baseline.command.json" not in str(exc):
                raise

        bridge_mismatch = root / "bridge-mismatch"
        make_perf_artifact(
            bridge_mismatch,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        health = load_json(bridge_mismatch / "serve.health.after.json")
        health["cache"]["prefix_cache"]["selected_stage_bridge"] = "cuda_peer"
        write_json(bridge_mismatch / "serve.health.after.json", health)
        try:
            validate_perf_goal(
                out_dir=root / "bridge-mismatch-out",
                baseline_artifact=baseline,
                candidate_artifact=bridge_mismatch,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("bridge-mismatch candidate unexpectedly passed")
        except ValidationError as exc:
            if "selected_stage_bridge" not in str(exc):
                raise

        zero_profile = root / "zero-profile"
        make_perf_artifact(
            zero_profile,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        health = load_json(zero_profile / "serve.health.after.json")
        health["cache"]["prefix_cache"]["pipeline_decode"]["calls"] = 0
        write_json(zero_profile / "serve.health.after.json", health)
        try:
            validate_perf_goal(
                out_dir=root / "zero-profile-out",
                baseline_artifact=baseline,
                candidate_artifact=zero_profile,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("zero-profile candidate unexpectedly passed")
        except ValidationError as exc:
            if "pipeline_decode.calls" not in str(exc):
                raise

        bench_mismatch = root / "bench-mismatch"
        make_perf_artifact(
            bench_mismatch,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        command = load_json(bench_mismatch / "bench-serve.command.json")
        idx = command["cmd"].index("--random-output-len")
        command["cmd"][idx + 1] = "64"
        write_json(bench_mismatch / "bench-serve.command.json", command)
        try:
            validate_perf_goal(
                out_dir=root / "bench-mismatch-out",
                baseline_artifact=baseline,
                candidate_artifact=bench_mismatch,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("bench-mismatch candidate unexpectedly passed")
        except ValidationError as exc:
            if "random-output-len" not in str(exc) and "bench command" not in str(exc):
                raise

        wrong_bench_model = root / "wrong-bench-model"
        make_perf_artifact(
            wrong_bench_model,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        command = load_json(wrong_bench_model / "bench-serve.command.json")
        idx = command["cmd"].index("--model")
        command["cmd"][idx + 1] = "other/model"
        write_json(wrong_bench_model / "bench-serve.command.json", command)
        try:
            validate_perf_goal(
                out_dir=root / "wrong-bench-model-out",
                baseline_artifact=baseline,
                candidate_artifact=wrong_bench_model,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("wrong-bench-model candidate unexpectedly passed")
        except ValidationError as exc:
            if "--model" not in str(exc):
                raise

        missing_base_url = root / "missing-base-url"
        make_perf_artifact(
            missing_base_url,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        command = load_json(missing_base_url / "bench-serve.command.json")
        idx = command["cmd"].index("--base-url")
        del command["cmd"][idx : idx + 2]
        write_json(missing_base_url / "bench-serve.command.json", command)
        try:
            validate_perf_goal(
                out_dir=root / "missing-base-url-out",
                baseline_artifact=baseline,
                candidate_artifact=missing_base_url,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("missing-base-url candidate unexpectedly passed")
        except ValidationError as exc:
            if "base-url" not in str(exc):
                raise

        missing_ci = root / "missing-ci"
        make_perf_artifact(
            missing_ci,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        reports = load_json(missing_ci / "bench-serve.json")
        reports[0]["output_throughput_tps"].pop("ci95_hw")
        write_json(missing_ci / "bench-serve.json", reports)
        try:
            validate_perf_goal(
                out_dir=root / "missing-ci-out",
                baseline_artifact=baseline,
                candidate_artifact=missing_ci,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("missing-ci candidate unexpectedly passed")
        except ValidationError as exc:
            if "ci95_hw" not in str(exc):
                raise

        incomplete_repeat = root / "incomplete-repeat"
        make_perf_artifact(
            incomplete_repeat,
            tps_by_c={1: 21.0, 4: 29.0, 8: 30.0, 16: 29.5},
            pipeline_mode="overlapped",
        )
        reports = load_json(incomplete_repeat / "bench-serve.json")
        reports[0]["completed_per_run"][0] = 95
        write_json(incomplete_repeat / "bench-serve.json", reports)
        try:
            validate_perf_goal(
                out_dir=root / "incomplete-repeat-out",
                baseline_artifact=baseline,
                candidate_artifact=incomplete_repeat,
                correctness_artifact=correctness,
                optional_vllm_artifact=None,
            )
            raise AssertionError("incomplete-repeat candidate unexpectedly passed")
        except ValidationError as exc:
            if "completed_per_run" not in str(exc):
                raise


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--baseline-artifact", type=Path)
    parser.add_argument("--candidate-artifact", type=Path)
    parser.add_argument("--correctness-artifact", type=Path)
    parser.add_argument("--optional-vllm-artifact", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        try:
            run_self_test()
        except Exception as exc:  # pragma: no cover - exercised by shell self-test
            print(f"{SELFTEST_PASS.replace(' PASS', ' FAIL')}: {exc}", file=sys.stderr)
            return 1
        print(SELFTEST_PASS)
        return 0

    missing = [
        name
        for name in ["out", "baseline_artifact", "candidate_artifact", "correctness_artifact"]
        if getattr(args, name) is None
    ]
    if missing:
        print(f"{PASS_PREFIX.replace(' PASS', ' FAIL')}: missing args {missing}", file=sys.stderr)
        return 2
    try:
        result = validate_perf_goal(
            out_dir=args.out,
            baseline_artifact=args.baseline_artifact,
            candidate_artifact=args.candidate_artifact,
            correctness_artifact=args.correctness_artifact,
            optional_vllm_artifact=args.optional_vllm_artifact,
        )
    except ValidationError as exc:
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            write_json(
                args.out / "layer_split_perf_goal_gate.json",
                {"schema_version": 1, "status": "fail", "error": str(exc), "created_at": iso_now()},
            )
        print(f"{PASS_PREFIX.replace(' PASS', ' FAIL')}: {exc}", file=sys.stderr)
        return 1
    print(result["pass_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
