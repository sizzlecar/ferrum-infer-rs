#!/usr/bin/env python3
"""Final validator for the Llama 3.3 70B 4bit 2x4090 goal."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS_PREFIX = "LLAMA33_70B_4BIT_2X4090 GOAL PASS"
LLAMA33_SOURCE_PASS_PREFIX = "G0 SOURCE g0_cuda2x4090_llama33_70b_4bit PASS"
REQUIRED_CONCURRENCY_CELLS = {1, 4, 8, 16}
REQUIRED_LLAMA33_FILES = {
    "gate.json",
    "metadata.json",
    "effective_config.json",
    "decision_trace.jsonl",
    "hardware.json",
    "nvidia-smi.before.txt",
    "nvidia-smi.during.txt",
    "nvidia-smi.after.txt",
    "model_manifest.json",
    "run.command.json",
    "run.effective_config.json",
    "run.stdin",
    "run.stdout",
    "run.stderr",
    "serve.command.json",
    "serve.effective_config.json",
    "serve.log",
    "serve.health.json",
    "serve.models.json",
    "serve.correctness.json",
    "serve.multiturn.json",
    "serve.structured_output.json",
    "serve.tool_call.json",
    "serve.streaming.sse",
    "concurrency_quality_regression.json",
    "bench-serve.command.json",
    "bench-serve.json",
    "bench-serve.stdout",
    "bench-serve.stderr",
    "vllm-baseline.command.json",
    "vllm-baseline.json",
    "comparison.json",
}
REQUIRED_METADATA_FIELDS = {
    "git_sha",
    "dirty_status",
    "binary_sha256",
    "command_line",
    "build_features",
    "cuda_version",
    "driver_version",
    "gpu_names",
    "gpu_uuids",
    "requested_gpu_devices",
    "selected_gpu_devices",
    "quant_format",
    "distributed_strategy",
    "layer_split_plan",
    "sanitized_env",
}
REQUIRED_EFFECTIVE_CONFIG_FIELDS = {
    "backend",
    "requested_gpu_devices",
    "selected_gpu_devices",
    "cuda_device_count",
    "selected_distributed_strategy",
    "selected_layer_split_plan",
    "selected_weight_placement",
    "selected_kv_layout",
    "selected_attention_impl",
    "selected_graph_mode",
    "selected_max_sequences",
    "selected_max_model_len",
    "selected_kv_capacity",
    "selected_max_batched_tokens",
    "model_capabilities",
}


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


def require_dir(path: Path, label: str) -> Path:
    if not path.exists():
        raise ValidationError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise ValidationError(f"{label} must be a directory: {path}")
    return path


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ValidationError(f"missing {label}: {path}")
    return path


def require_nonempty_file(path: Path, label: str) -> None:
    require_file(path, label)
    if not path.read_text(errors="replace").strip():
        raise ValidationError(f"{label} is empty: {path}")


def status_pass(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValidationError(f"{path} must contain a JSON object")
    status = data.get("status")
    if status is not None and status != "pass":
        raise ValidationError(f"{path} status is not pass: {status!r}")
    if data.get("ok") is False or data.get("passed") is False:
        raise ValidationError(f"{path} is not passing")
    return data


def validate_source_gate(
    label: str,
    artifact: Path,
    *,
    run_gate_lane: str,
    source_lane: str,
    legacy_gate_file: str,
) -> dict[str, Any]:
    artifact = require_dir(artifact, f"{label} artifact")
    manifest = artifact / "gate.manifest.json"
    if manifest.is_file():
        data = status_pass(manifest)
        if data.get("schema_version") != 1:
            raise ValidationError(f"{manifest} schema_version must be 1")
        if data.get("lane") != run_gate_lane:
            raise ValidationError(f"{manifest} lane must be {run_gate_lane!r}")
        artifact_dir = data.get("artifact_dir")
        expected_child = f"G0 SOURCE {source_lane} PASS: {artifact_dir}"
        if data.get("child_pass_line") != expected_child:
            raise ValidationError(
                f"{manifest} child_pass_line {data.get('child_pass_line')!r} != {expected_child!r}"
            )
        return {
            "kind": "run_gate_manifest",
            "label": label,
            "artifact": str(artifact),
            "pass_line": data.get("pass_line"),
        }

    gate = artifact / legacy_gate_file
    data = status_pass(gate)
    return {
        "kind": "source_gate_json",
        "label": label,
        "artifact": str(artifact),
        "lane": data.get("lane"),
    }


def validate_llama33_artifact(artifact: Path) -> dict[str, Any]:
    artifact = require_dir(artifact, "cuda llama33 70b artifact")
    missing = sorted(rel for rel in REQUIRED_LLAMA33_FILES if not (artifact / rel).exists())
    if missing:
        raise ValidationError(
            "cuda llama33 artifact missing required files: " + ", ".join(missing)
        )

    gate = status_pass(artifact / "gate.json")
    pass_line = gate.get("pass_line")
    expected_pass = f"{LLAMA33_SOURCE_PASS_PREFIX}: {artifact}"
    if pass_line != expected_pass:
        raise ValidationError(f"{artifact / 'gate.json'} pass_line must be {expected_pass!r}")

    metadata = validate_metadata(load_json(artifact / "metadata.json"))
    effective = validate_effective_config(load_json(artifact / "effective_config.json"))
    validate_runtime_effective_config(load_json(artifact / "run.effective_config.json"), "run")
    validate_runtime_effective_config(load_json(artifact / "serve.effective_config.json"), "serve")
    validate_hardware(load_json(artifact / "hardware.json"))
    validate_json_status_files(artifact)
    validate_bench_reports(load_json(artifact / "bench-serve.json"))
    validate_optional_vllm_baseline(load_json(artifact / "vllm-baseline.json"))
    validate_comparison(load_json(artifact / "comparison.json"))

    for rel in ["nvidia-smi.before.txt", "nvidia-smi.during.txt", "nvidia-smi.after.txt"]:
        require_nonempty_file(artifact / rel, rel)
    require_nonempty_file(artifact / "decision_trace.jsonl", "decision_trace.jsonl")
    require_nonempty_file(artifact / "serve.streaming.sse", "serve.streaming.sse")
    scan_logs(artifact)

    return {
        "kind": "cuda_llama33_70b_4bit_2x4090",
        "artifact": str(artifact),
        "metadata": {
            "git_sha": metadata["git_sha"],
            "binary_sha256": metadata["binary_sha256"],
            "quant_format": metadata["quant_format"],
            "distributed_strategy": metadata["distributed_strategy"],
        },
        "effective_config": {
            "backend": effective["backend"],
            "selected_gpu_devices": effective["selected_gpu_devices"],
            "selected_distributed_strategy": effective["selected_distributed_strategy"],
        },
    }


def validate_metadata(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValidationError("metadata.json must contain a JSON object")
    missing = sorted(REQUIRED_METADATA_FIELDS - set(data))
    if missing:
        raise ValidationError("metadata.json missing fields: " + ", ".join(missing))
    if not isinstance(data.get("model_id") or data.get("model_path"), str):
        raise ValidationError("metadata.json must include model_id or model_path")
    if data["requested_gpu_devices"] != [0, 1]:
        raise ValidationError("metadata requested_gpu_devices must be [0, 1]")
    if data["selected_gpu_devices"] != [0, 1]:
        raise ValidationError("metadata selected_gpu_devices must be [0, 1]")
    if data["distributed_strategy"] != "layer_split":
        raise ValidationError("metadata distributed_strategy must be layer_split")
    if len(data["gpu_names"]) != 2 or len(data["gpu_uuids"]) != 2:
        raise ValidationError("metadata must record exactly two GPUs")
    digest = data["binary_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValidationError("metadata binary_sha256 must be a 64-char hex digest")
    quant = str(data["quant_format"]).lower()
    if not any(marker in quant for marker in ["gptq", "awq", "q4"]):
        raise ValidationError("metadata quant_format must identify a 4bit format")
    if not isinstance(data["sanitized_env"], dict):
        raise ValidationError("metadata sanitized_env must be an object")
    return data


def validate_effective_config(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValidationError("effective_config.json must contain a JSON object")
    missing = sorted(REQUIRED_EFFECTIVE_CONFIG_FIELDS - set(data))
    if missing:
        raise ValidationError("effective_config.json missing fields: " + ", ".join(missing))
    if data["backend"] != "cuda":
        raise ValidationError("effective_config backend must be cuda")
    if data["requested_gpu_devices"] != [0, 1]:
        raise ValidationError("effective_config requested_gpu_devices must be [0, 1]")
    if data["selected_gpu_devices"] != [0, 1]:
        raise ValidationError("effective_config selected_gpu_devices must be [0, 1]")
    if data["cuda_device_count"] != 2:
        raise ValidationError("effective_config cuda_device_count must be 2")
    if data["selected_distributed_strategy"] != "layer_split":
        raise ValidationError("effective_config selected_distributed_strategy must be layer_split")
    if not data["selected_layer_split_plan"]:
        raise ValidationError("effective_config selected_layer_split_plan must be non-empty")
    return data


def validate_runtime_effective_config(data: Any, label: str) -> None:
    if not isinstance(data, dict):
        raise ValidationError(f"{label}.effective_config.json must contain a JSON object")
    for key in ["requested_gpu_devices", "selected_gpu_devices"]:
        if data.get(key) != [0, 1]:
            raise ValidationError(f"{label}.effective_config.json {key} must be [0, 1]")


def validate_hardware(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValidationError("hardware.json must contain a JSON object")
    names = data.get("gpu_names") or data.get("gpus")
    if not isinstance(names, list) or len(names) != 2:
        raise ValidationError("hardware.json must record exactly two GPUs")


def validate_json_status_files(artifact: Path) -> None:
    for rel in [
        "serve.health.json",
        "serve.models.json",
        "serve.correctness.json",
        "serve.multiturn.json",
        "serve.structured_output.json",
        "serve.tool_call.json",
        "concurrency_quality_regression.json",
    ]:
        status_pass(artifact / rel)


def validate_bench_reports(data: Any) -> None:
    reports = data if isinstance(data, list) else [data]
    if not reports or not all(isinstance(report, dict) for report in reports):
        raise ValidationError("bench-serve.json must contain a BenchReport object or list")
    rows: dict[int, dict[str, Any]] = {}
    for report in reports:
        concurrency = int(report.get("concurrency") or 0)
        if concurrency <= 0:
            raise ValidationError("bench-serve report missing positive concurrency")
        completed = sum(int(v) for v in report.get("completed_per_run", []))
        errored = sum(int(v) for v in report.get("errored_per_run", []))
        if completed <= 0:
            raise ValidationError(f"bench-serve c{concurrency} completed must be > 0")
        if errored != 0:
            raise ValidationError(f"bench-serve c{concurrency} errored must be 0")
        if int(report.get("n_repeats") or 0) < 3:
            raise ValidationError(f"bench-serve c{concurrency} n_repeats must be >= 3")
        if report.get("output_token_count_source") != "usage":
            raise ValidationError(f"bench-serve c{concurrency} output_token_count_source must be usage")
        for metric, key in [
            ("output_throughput_tps", None),
            ("ttft_ms", "p50"),
            ("tpot_ms", "p50"),
            ("e2e_ms", "p95"),
        ]:
            if positive_metric_value(report, metric, key) is None:
                suffix = f".{key}" if key else ""
                raise ValidationError(f"bench-serve c{concurrency} {metric}{suffix} must be positive")
        for field in [
            "bad_output_per_run",
            "malformed_stream_per_run",
            "missing_done_per_run",
            "duplicate_done_per_run",
            "zero_output_tokens_per_run",
            "stream_bulk_flush_per_run",
            "http_500_per_run",
            "panic_per_run",
        ]:
            values = report.get(field)
            if not isinstance(values, list):
                raise ValidationError(f"bench-serve c{concurrency} missing {field}")
            if sum(int(v) for v in values) != 0:
                raise ValidationError(f"bench-serve c{concurrency} {field} must sum to 0")
        rows[concurrency] = report
    missing = sorted(REQUIRED_CONCURRENCY_CELLS - set(rows))
    if missing:
        raise ValidationError(f"bench-serve.json missing cells: {missing}")


def positive_metric_value(report: dict[str, Any], metric: str, percentile: str | None) -> float | None:
    value: Any = report.get(metric)
    if percentile:
        if not isinstance(value, dict):
            return None
        value = value.get(percentile)
    if isinstance(value, dict):
        value = value.get("mean")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def validate_optional_vllm_baseline(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValidationError("vllm-baseline.json must contain a JSON object")
    status = data.get("status")
    if status in (None, "pass"):
        return
    if status not in {"skipped", "diagnostic"}:
        raise ValidationError(f"vllm-baseline.json status is not accepted: {status!r}")
    if not isinstance(data.get("reason"), str) or not data["reason"].strip():
        raise ValidationError("skipped vllm-baseline.json must include a reason")


def validate_comparison(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValidationError("comparison.json must contain a JSON object")
    cells = normalize_cells(data.get("cells") or data.get("required_cells"))
    missing = sorted(REQUIRED_CONCURRENCY_CELLS - set(cells))
    if missing:
        raise ValidationError(f"comparison.json missing cells: {missing}")
    if data.get("mode") == "ferrum_only":
        validate_ferrum_only_comparison(cells)
        return
    for c in sorted(REQUIRED_CONCURRENCY_CELLS):
        cell = cells[c]
        if cell.get("status") not in (None, "pass"):
            raise ValidationError(f"comparison c{c} status is not pass")
        require_ratio_at_least(
            cell,
            [
                "output_throughput_ratio_to_vllm",
                "throughput_ratio_to_vllm",
                "ferrum_output_throughput_ratio",
            ],
            0.70,
            f"comparison c{c} throughput ratio",
        )
        require_ratio_at_most(
            cell,
            ["ttft_ratio_to_vllm", "ferrum_ttft_ratio"],
            1.50,
            f"comparison c{c} TTFT ratio",
        )
        require_ratio_at_most(
            cell,
            ["tpot_ratio_to_vllm", "ferrum_tpot_ratio"],
            1.50,
            f"comparison c{c} TPOT ratio",
        )
        if int(cell.get("bad_output_count", 0)) != 0:
            raise ValidationError(f"comparison c{c} bad_output_count must be 0")
        if int(cell.get("malformed_stream_count", 0)) != 0:
            raise ValidationError(f"comparison c{c} malformed_stream_count must be 0")


def validate_ferrum_only_comparison(cells: dict[int, dict[str, Any]]) -> None:
    for c in sorted(REQUIRED_CONCURRENCY_CELLS):
        cell = cells[c]
        if cell.get("status") not in (None, "pass"):
            raise ValidationError(f"comparison c{c} status is not pass")
        if cell.get("mode") not in (None, "ferrum_only"):
            raise ValidationError(f"comparison c{c} mode must be ferrum_only")
        for key in [
            "ferrum_output_throughput_tps",
            "ferrum_ttft_p50_ms",
            "ferrum_tpot_p50_ms",
            "p95_end_to_end_latency_ms",
        ]:
            value = first_number(cell, [key])
            if value is None or value <= 0:
                raise ValidationError(f"comparison c{c} {key} must be positive")
        if int(cell.get("bad_output_count", 0)) != 0:
            raise ValidationError(f"comparison c{c} bad_output_count must be 0")
        if int(cell.get("malformed_stream_count", 0)) != 0:
            raise ValidationError(f"comparison c{c} malformed_stream_count must be 0")


def normalize_cells(raw: Any) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                out[int(str(key).lstrip("c"))] = value
    elif isinstance(raw, list):
        for value in raw:
            if isinstance(value, dict):
                cell = value.get("concurrency", value.get("c"))
                if cell is not None:
                    out[int(cell)] = value
    return out


def require_ratio_at_least(
    cell: dict[str, Any],
    keys: list[str],
    threshold: float,
    label: str,
) -> None:
    value = first_number(cell, keys)
    if value is None:
        raise ValidationError(f"{label} missing")
    if value < threshold:
        raise ValidationError(f"{label} {value} < {threshold}")


def require_ratio_at_most(
    cell: dict[str, Any],
    keys: list[str],
    threshold: float,
    label: str,
) -> None:
    value = first_number(cell, keys)
    if value is None:
        raise ValidationError(f"{label} missing")
    if value > threshold:
        raise ValidationError(f"{label} {value} > {threshold}")


def first_number(data: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key in data:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                return None
    return None


def scan_logs(artifact: Path) -> None:
    forbidden = [
        "panic",
        "CUDA illegal memory access",
        "NCCL error",
        "OOM",
        "KV cache overflow",
        "missing tokenizer",
        "chat template render failure",
        "CPU fallback",
        "single-GPU fallback",
    ]
    for rel in ["run.stderr", "serve.log", "bench-serve.stderr"]:
        text = (artifact / rel).read_text(errors="replace")
        for pattern in forbidden:
            if pattern.lower() in text.lower():
                raise ValidationError(f"{rel} contains forbidden log pattern: {pattern}")


def validate_goal(args: argparse.Namespace) -> dict[str, Any]:
    evidence = {
        "schema_version": 1,
        "status": "pass",
        "validated_at": iso_now(),
        "artifacts": {
            "metal": validate_source_gate(
                "metal",
                args.metal_artifact,
                run_gate_lane="metal",
                source_lane="metal",
                legacy_gate_file="metal.gate.json",
            ),
            "cuda_full": validate_source_gate(
                "cuda-full",
                args.cuda_full_artifact,
                run_gate_lane="cuda-full",
                source_lane="g0_cuda4090_full",
                legacy_gate_file="g0_cuda4090_full.gate.json",
            ),
            "cuda_llama_dense": validate_source_gate(
                "cuda-llama-dense",
                args.cuda_llama_dense_artifact,
                run_gate_lane="cuda-llama-dense",
                source_lane="g0_cuda4090_llama_dense",
                legacy_gate_file="g0_cuda4090_llama_dense.gate.json",
            ),
            "cuda_llama33_70b": validate_llama33_artifact(args.cuda_llama33_70b_artifact),
        },
    }
    return evidence


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--metal-artifact", type=Path)
    parser.add_argument("--cuda-full-artifact", type=Path)
    parser.add_argument("--cuda-llama-dense-artifact", type=Path)
    parser.add_argument("--cuda-llama33-70b-artifact", type=Path)
    return parser.parse_args(argv)


def require_args(args: argparse.Namespace) -> None:
    if args.self_test:
        return
    missing = [
        name
        for name in [
            "out",
            "metal_artifact",
            "cuda_full_artifact",
            "cuda_llama_dense_artifact",
            "cuda_llama33_70b_artifact",
        ]
        if getattr(args, name) is None
    ]
    if missing:
        raise ValidationError("missing required arguments: " + ", ".join(missing))


def make_source_artifact(root: Path, gate_file: str, lane: str) -> None:
    write_json(root / gate_file, {"status": "pass", "lane": lane})


def make_llama33_artifact(root: Path) -> None:
    root.mkdir(parents=True)
    for rel in REQUIRED_LLAMA33_FILES:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".json"):
            write_json(path, {"status": "pass"})
        elif rel.endswith(".jsonl"):
            path.write_text('{"status":"pass"}\n')
        else:
            path.write_text("selftest\n")
    write_json(
        root / "gate.json",
        {
            "status": "pass",
            "pass_line": f"{LLAMA33_SOURCE_PASS_PREFIX}: {root}",
        },
    )
    metadata = {
        "git_sha": "selftest",
        "dirty_status": {"is_dirty": False, "status_short": []},
        "binary_sha256": "0" * 64,
        "command_line": ["ferrum", "selftest"],
        "build_features": ["cuda"],
        "cuda_version": "12.8",
        "driver_version": "selftest",
        "gpu_names": ["RTX 4090", "RTX 4090"],
        "gpu_uuids": ["GPU-0", "GPU-1"],
        "requested_gpu_devices": [0, 1],
        "selected_gpu_devices": [0, 1],
        "model_id": "Llama-3.3-70B-Instruct-4bit",
        "quant_format": "gptq_int4",
        "distributed_strategy": "layer_split",
        "layer_split_plan": [{"device": 0, "layers": [0, 39]}, {"device": 1, "layers": [40, 79]}],
        "sanitized_env": {},
    }
    write_json(root / "metadata.json", metadata)
    effective = {
        "backend": "cuda",
        "requested_gpu_devices": [0, 1],
        "selected_gpu_devices": [0, 1],
        "cuda_device_count": 2,
        "selected_distributed_strategy": "layer_split",
        "selected_layer_split_plan": metadata["layer_split_plan"],
        "selected_weight_placement": "layer_split",
        "selected_kv_layout": "paged",
        "selected_attention_impl": "vllm_paged_attn",
        "selected_graph_mode": "disabled",
        "selected_max_sequences": 16,
        "selected_max_model_len": 8192,
        "selected_kv_capacity": 512,
        "selected_max_batched_tokens": 1024,
        "model_capabilities": {"architecture": "llama"},
    }
    write_json(root / "effective_config.json", effective)
    write_json(root / "run.effective_config.json", effective)
    write_json(root / "serve.effective_config.json", effective)
    write_json(root / "hardware.json", {"gpu_names": ["RTX 4090", "RTX 4090"]})
    write_json(
        root / "bench-serve.json",
        [
            {
                "concurrency": c,
                "completed_per_run": [2, 2, 2],
                "errored_per_run": [0, 0, 0],
                "n_repeats": 3,
                "output_token_count_source": "usage",
                "output_throughput_tps": {"mean": 20.0},
                "ttft_ms": {"p50": {"mean": 500.0}},
                "tpot_ms": {"p50": {"mean": 50.0}},
                "e2e_ms": {"p95": {"mean": 5000.0}},
                "bad_output_per_run": [0, 0, 0],
                "malformed_stream_per_run": [0, 0, 0],
                "missing_done_per_run": [0, 0, 0],
                "duplicate_done_per_run": [0, 0, 0],
                "zero_output_tokens_per_run": [0, 0, 0],
                "stream_bulk_flush_per_run": [0, 0, 0],
                "http_500_per_run": [0, 0, 0],
                "panic_per_run": [0, 0, 0],
            }
            for c in sorted(REQUIRED_CONCURRENCY_CELLS)
        ],
    )
    write_json(
        root / "comparison.json",
        {
            "status": "pass",
            "mode": "ferrum_only",
            "baseline": "not_run",
            "reason": "selftest Ferrum-only path",
            "cells": {
                f"c{c}": {
                    "status": "pass",
                    "mode": "ferrum_only",
                    "ferrum_output_throughput_tps": 20.0,
                    "ferrum_ttft_p50_ms": 500.0,
                    "ferrum_tpot_p50_ms": 50.0,
                    "p95_end_to_end_latency_ms": 5000.0,
                    "bad_output_count": 0,
                    "malformed_stream_count": 0,
                }
                for c in sorted(REQUIRED_CONCURRENCY_CELLS)
            }
        },
    )
    write_json(
        root / "vllm-baseline.json",
        {"status": "skipped", "reason": "selftest Ferrum-only path"},
    )


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="ferrum-llama33-goal-") as tmp:
        root = Path(tmp)
        metal = root / "metal"
        cuda_full = root / "cuda-full"
        cuda_dense = root / "cuda-dense"
        llama33 = root / "llama33"
        out = root / "out"
        make_source_artifact(metal, "metal.gate.json", "metal")
        make_source_artifact(cuda_full, "g0_cuda4090_full.gate.json", "g0_cuda4090_full")
        make_source_artifact(
            cuda_dense,
            "g0_cuda4090_llama_dense.gate.json",
            "g0_cuda4090_llama_dense",
        )
        make_llama33_artifact(llama33)
        args = argparse.Namespace(
            out=out,
            metal_artifact=metal,
            cuda_full_artifact=cuda_full,
            cuda_llama_dense_artifact=cuda_dense,
            cuda_llama33_70b_artifact=llama33,
            self_test=False,
        )
        result = validate_goal(args)
        write_json(out / "llama33_70b_4bit_2x4090_goal_gate.json", result)
        print("LLAMA33_70B_4BIT_2X4090 GOAL SELFTEST PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    require_args(args)
    result = validate_goal(args)
    args.out.mkdir(parents=True, exist_ok=True)
    result["pass_line"] = f"{PASS_PREFIX}: {args.out}"
    write_json(args.out / "llama33_70b_4bit_2x4090_goal_gate.json", result)
    print(result["pass_line"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"LLAMA33_70B_4BIT_2X4090 GOAL FAIL: {exc}", flush=True)
        raise SystemExit(1)
