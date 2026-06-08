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
    "failed_per_run",
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


def validate_metadata(path: Path, label: str) -> dict[str, Any]:
    metadata = first_json(path, ["metadata.json", "gate.manifest.json"], f"{label} metadata")
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


def validate_optional_vllm_metadata(
    optional_vllm_artifact: Path,
    candidate_metadata: dict[str, Any],
) -> dict[str, Any]:
    metadata = validate_metadata(optional_vllm_artifact, "vllm")
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


def model_identity(metadata: dict[str, Any]) -> str:
    value = metadata.get("model_id") or metadata.get("model_path") or metadata.get("model")
    if isinstance(value, dict):
        value = value.get("id") or value.get("path")
    return str(value) if value else ""


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
    raw = data.get("cmd") if isinstance(data, dict) else data
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


def require_bench_command(path: Path, label: str) -> dict[str, str]:
    cmd = command_list(path / "bench-serve.command.json", label)
    if "bench-serve" not in cmd:
        raise ValidationError(f"{label}: command is not ferrum bench-serve")
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


def list_sum(value: Any) -> int:
    if value is None:
        return 0
    if not isinstance(value, list):
        raise ValidationError(f"expected list, got {type(value).__name__}")
    return sum(int(item) for item in value)


def validate_report(report: dict[str, Any], label: str) -> tuple[int, float]:
    try:
        concurrency = int(report.get("concurrency") or report.get("c") or 0)
    except ValueError as exc:
        raise ValidationError(f"{label}: invalid concurrency") from exc
    if concurrency <= 0:
        raise ValidationError(f"{label}: missing positive concurrency")
    if report.get("diagnostic_only") is True:
        raise ValidationError(f"{label} c{concurrency}: report is diagnostic-only")
    if int(report.get("n_repeats") or 0) < 3:
        raise ValidationError(f"{label} c{concurrency}: n_repeats must be >= 3")
    if report.get("output_token_count_source") != "usage":
        raise ValidationError(
            f"{label} c{concurrency}: output_token_count_source must be usage"
        )
    completed = list_sum(report.get("completed_per_run"))
    if completed <= 0:
        raise ValidationError(f"{label} c{concurrency}: completed requests must be > 0")
    for field in BAD_COUNT_FIELDS:
        if field in report and list_sum(report.get(field)) != 0:
            raise ValidationError(f"{label} c{concurrency}: {field} must sum to 0")
    if "failed_per_run" not in report:
        raise ValidationError(f"{label} c{concurrency}: missing failed_per_run")
    if "errored_per_run" not in report:
        raise ValidationError(f"{label} c{concurrency}: missing errored_per_run")
    throughput = scalar_mean(report, "output_throughput_tps")
    if throughput is None:
        raise ValidationError(f"{label} c{concurrency}: missing positive output throughput")
    for metric, percentile in [
        ("ttft_ms", "p50"),
        ("tpot_ms", "p50"),
        ("e2e_ms", "p95"),
    ]:
        if not nested_positive(report, metric, percentile):
            raise ValidationError(f"{label} c{concurrency}: missing positive {metric}.{percentile}")
    if not isinstance(report.get("actual_input_tokens"), dict):
        raise ValidationError(f"{label} c{concurrency}: missing actual_input_tokens stats")
    if not isinstance(report.get("actual_input_tokens_per_request"), list):
        raise ValidationError(f"{label} c{concurrency}: missing actual_input_tokens_per_request")
    return concurrency, throughput


def validate_bench_artifact(path: Path, label: str) -> dict[str, Any]:
    command_signature = require_bench_command(path, label)
    reports = reports_from_json(load_json(path / "bench-serve.json"), label)
    rows: dict[int, dict[str, Any]] = {}
    throughput: dict[int, float] = {}
    for report in reports:
        concurrency, tps = validate_report(report, label)
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


def validate_correctness_artifact(path: Path) -> dict[str, Any]:
    path = require_dir(path, "correctness artifact")
    data = first_json(path, ["correctness.json", "gate.json", "gate.manifest.json"], "correctness")
    if data.get("diagnostic_only") is True:
        raise ValidationError("correctness artifact is diagnostic-only")
    checks = data.get("checks")
    if not isinstance(checks, dict):
        raise ValidationError("correctness artifact must include checks object")
    missing = sorted(REQUIRED_CORRECTNESS_CHECKS - set(checks))
    if missing:
        raise ValidationError("correctness artifact missing checks: " + ", ".join(missing))
    for name in sorted(REQUIRED_CORRECTNESS_CHECKS):
        status_pass_obj(checks[name], f"correctness check {name}")
    streaming_done = checks["streaming_done"]
    if int(streaming_done.get("done_count", 0)) != 1:
        raise ValidationError("streaming_done.done_count must be exactly 1")
    streaming_usage = checks["streaming_usage"]
    if streaming_usage.get("include_usage") is not True:
        raise ValidationError("streaming_usage.include_usage must be true")
    if streaming_usage.get("usage_received") is not True:
        raise ValidationError("streaming usage was not received")
    log_scan = checks["log_scan"]
    if int(log_scan.get("bad_pattern_count", 0)) != 0:
        raise ValidationError("log_scan.bad_pattern_count must be 0")
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

    baseline_bench = validate_bench_artifact(baseline_artifact, "baseline")
    candidate_bench = validate_bench_artifact(candidate_artifact, "candidate")
    if baseline_bench["command_signature"] != candidate_bench["command_signature"]:
        raise ValidationError("baseline and candidate bench command parameters differ")
    correctness = validate_correctness_artifact(correctness_artifact)
    scan_artifact_logs(baseline_artifact, "baseline")
    scan_artifact_logs(candidate_artifact, "candidate")

    vllm_tps = None
    vllm_metadata = None
    target_tps = FIXED_PUBLIC_TARGET_TPS
    target_mode = "fixed_public_lower_bound"
    if optional_vllm_artifact is not None:
        vllm_metadata = validate_optional_vllm_metadata(
            optional_vllm_artifact, candidate_metadata
        )
        vllm_tps = vllm_tps_from_artifact(optional_vllm_artifact)
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
        "baseline_pipeline_cache_metrics": summarize_pipeline_cache_metrics(
            baseline_cache_metrics
        ),
        "candidate_pipeline_cache_metrics": summarize_pipeline_cache_metrics(
            candidate_cache_metrics
        ),
        "target_mode": target_mode,
        "target_output_tps": target_tps,
        "stretch_output_tps": STRETCH_TARGET_TPS,
        "stretch_passed": candidate_max >= STRETCH_TARGET_TPS,
        "same_pod_vllm_output_tps": vllm_tps,
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
                f"- Target output throughput: {target_tps:.3f} tok/s",
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
        "completed_per_run": [4, 4, 4],
        "failed_per_run": [0, 0, 0],
        "errored_per_run": [0, 0, 0],
        "n_repeats": 3,
        "output_token_count_source": "usage",
        "output_throughput_tps": {"mean": tps, "stddev": 0.1, "ci95_hw": 0.2},
        "ttft_ms": {"p50": {"mean": 500.0}},
        "tpot_ms": {"p50": {"mean": 40.0}},
        "e2e_ms": {"p95": {"mean": 6000.0}},
        "actual_input_tokens": {"mean": 512.0, "min": 512, "max": 512},
        "actual_input_tokens_per_request": [[512, 512, 512, 512]] * 3,
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
                "--fail-on-error",
                "--require-ci",
                "--seed",
                "9271",
                "--n-repeats",
                "3",
                "--concurrency-sweep",
                "1,4,8,16",
                "--dataset",
                "random",
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


def make_correctness_artifact(root: Path) -> None:
    checks = {name: {"status": "pass"} for name in REQUIRED_CORRECTNESS_CHECKS}
    checks["streaming_done"]["done_count"] = 1
    checks["streaming_usage"]["include_usage"] = True
    checks["streaming_usage"]["usage_received"] = True
    checks["log_scan"]["bad_pattern_count"] = 0
    write_json(root / "correctness.json", {"status": "pass", "checks": checks})


def make_vllm_artifact(root: Path, tps: float) -> None:
    write_json(
        root / "metadata.json",
        {
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
        },
    )
    write_json(
        root / "vllm-baseline.json",
        [
            {"concurrency": c, "output_throughput": tps if c == 8 else tps - 1.0}
            for c in sorted(THROUGHPUT_TARGET_CELLS)
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
            if "gpu_uuids" not in str(exc):
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
        metadata = load_json(vllm_hardware_mismatch / "metadata.json")
        metadata["gpu_uuids"] = ["GPU-vllm-other-0", "GPU-vllm-other-1"]
        write_json(vllm_hardware_mismatch / "metadata.json", metadata)
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
