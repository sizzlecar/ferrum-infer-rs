#!/usr/bin/env python3
"""Collect real G00 external and frozen-Ferrum performance evidence.

The collector runs one model/backend lane at a time. Comparable lanes use the
fixed A-B-B-A-B-A-A-B outer order; blocked legacy lanes run the four external A
slots only. Every server slot is a fresh process and covers the complete fixed
cell matrix with the canonical benchmark client and resource sampler.
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
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import runtime_vnext_baseline_gate as baseline_gate
    import runtime_vnext_baseline_scenarios as scenario_runner
    import runtime_vnext_resource_sampler as resource_sampler
except ModuleNotFoundError:
    from scripts.release import runtime_vnext_baseline_gate as baseline_gate
    from scripts.release import runtime_vnext_baseline_scenarios as scenario_runner
    from scripts.release import runtime_vnext_resource_sampler as resource_sampler


REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR_PATH = Path(__file__).resolve()
COLLECTOR_RELATIVE_PATH = COLLECTOR_PATH.relative_to(REPO_ROOT).as_posix()
RESOURCE_SAMPLER_PATH = REPO_ROOT / resource_sampler.COLLECTOR_RELATIVE_PATH
SCHEMA_VERSION = 1
PASS_PREFIX = "FERRUM RUNTIME VNEXT PERFORMANCE COLLECTOR PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT PERFORMANCE COLLECTOR SELFTEST PASS"
SLOT_ORDER = tuple(baseline_gate.SLOT_ORDER)
BLOCKED_DOWNSTREAM_GOALS = {
    "m1-qwen35-4b/metal": "G08A",
    "m2-qwen35-35b-a3b/metal": "G08B",
}
SECRET_KEY_RE = re.compile(
    r"(?:^|[_-])(?:token|secret|password|api[_-]?key|credential|authorization|auth[_-]?key|bearer|cookie|private[_-]?key)(?:$|[_-])",
    re.IGNORECASE,
)
SECRET_VALUE_RE = re.compile(
    r"(?:--(?:token|secret|password|api[-_]?key|authorization|auth[-_]?key)(?=$|[=:\s])|authorization:|bearer\s)",
    re.IGNORECASE,
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
IDENTITY_PROBE_TIMEOUT_SEC = 60.0


class CollectorError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectorError(message)


def require_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be a non-empty string")
    return value


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, f"timestamp lacks timezone: {value}")
    return parsed.astimezone(timezone.utc)


def duration_seconds(started_at: str, finished_at: str) -> float:
    value = (parse_timestamp(finished_at) - parse_timestamp(started_at)).total_seconds()
    require(value > 0, "execution duration must be positive")
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectorError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def artifact_path(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), "artifact path must be relative")
    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise CollectorError(f"artifact escapes root: {relative}") from exc
    return resolved


def artifact_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise CollectorError(f"artifact is outside root: {path}") from exc


def stage_file(root: Path, source: Path, relative: str, expected_sha256: str | None = None) -> Path:
    source = source.expanduser().resolve()
    require(source.is_file(), f"source artifact missing: {source}")
    actual = file_sha256(source)
    if expected_sha256 is not None:
        require(actual == expected_sha256, f"source SHA256 mismatch for {source}")
    destination = artifact_path(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(destination.is_file() and file_sha256(destination) == actual, f"refusing to replace artifact: {destination}")
        return destination
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    require(file_sha256(temporary) == actual, f"staged artifact SHA256 mismatch: {temporary}")
    temporary.replace(destination)
    return destination


def write_artifact_json(root: Path, relative: str, value: Any) -> Path:
    destination = artifact_path(root, relative)
    if destination.exists():
        require(read_json(destination) == value, f"refusing to change existing artifact: {destination}")
        return destination
    atomic_write_json(destination, value)
    return destination


def ensure_nonempty_log(path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(f"[collector] {label}: no child output\n", encoding="utf-8")


def reject_secret_material(value: Any, label: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            require(isinstance(key, str), f"{label} contains a non-string key")
            require(not SECRET_KEY_RE.search(key), f"{label} contains forbidden secret-bearing key {key!r}")
            reject_secret_material(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_secret_material(child, f"{label}[{index}]")
    elif isinstance(value, str):
        require(not SECRET_VALUE_RE.search(value), f"{label} contains secret-bearing command material")


def sanitized_environment(overrides: Any = None) -> dict[str, str]:
    environment = scenario_runner.sanitized_child_environment()
    if overrides is not None:
        require(isinstance(overrides, dict), "environment overrides must be an object")
        for key, value in overrides.items():
            require(isinstance(key, str) and isinstance(value, str), "environment overrides must be strings")
            require(not key.startswith("FERRUM_"), f"hidden Ferrum environment is forbidden: {key}")
            require(not SECRET_KEY_RE.search(key), f"secret-bearing environment key is forbidden: {key}")
            environment[key] = value
    require(not any(key.startswith("FERRUM_") for key in environment), "sanitized environment inherited FERRUM_ controls")
    return dict(sorted(environment.items()))


def git_output(argv: list[str]) -> str:
    process = subprocess.run(["git", *argv], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    require(process.returncode == 0, f"git {' '.join(argv)} failed: {process.stderr.strip()}")
    return process.stdout.strip()


def git_bytes(argv: list[str]) -> bytes:
    process = subprocess.run(["git", *argv], cwd=REPO_ROOT, capture_output=True, check=False)
    require(process.returncode == 0, f"git {' '.join(argv)} failed: {process.stderr.decode(errors='replace').strip()}")
    return process.stdout


def locked_file_map(model: dict[str, Any], backend: str) -> dict[str, str]:
    return {
        str(row["path"]): str(row["sha256"])
        for row in model["lanes"][backend]["files"]
    }


def tokenizer_lock(model: dict[str, Any], backend: str) -> tuple[dict[str, Any], dict[str, Any]]:
    lane = model["lanes"][backend]
    source = lane.get("tokenizer_source") or lane.get("semantic_source")
    require(isinstance(source, dict), "models.lock lane lacks tokenizer source")
    rows = [row for row in source.get("files", []) if row.get("path") == "tokenizer.json"]
    require(len(rows) == 1, "tokenizer source must lock exactly one tokenizer.json")
    return source, rows[0]


def expected_cells(backend: str) -> list[tuple[str, int]]:
    return sorted(baseline_gate.expected_cells(backend))


def load_context(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    models_lock_path = root / "models.lock.json"
    legacy_path = root / "legacy-binaries.json"
    require(models_lock_path.is_file() and legacy_path.is_file(), "artifact root lacks models.lock.json or legacy-binaries.json")
    models_lock = read_json(models_lock_path)
    model_key = config.get("model_key")
    backend = config.get("backend")
    require(isinstance(model_key, str) and model_key, "config.model_key is required")
    require(backend in {"cuda", "metal"}, "config.backend must be cuda or metal")
    model_rows = [row for row in models_lock.get("models", []) if row.get("key") == model_key]
    require(len(model_rows) == 1, f"model key is not uniquely locked: {model_key}")
    model = model_rows[0]
    lane = model["lanes"][backend]
    hardware_id = lane["hardware_id"]
    hardware_rows = [row for row in models_lock.get("hardware", []) if row.get("id") == hardware_id]
    require(len(hardware_rows) == 1, f"hardware id is not uniquely locked: {hardware_id}")
    hardware = hardware_rows[0]
    legacy = read_json(legacy_path)
    binary_rows = [row for row in legacy.get("binaries", []) if row.get("backend") == backend]
    require(len(binary_rows) == 1, f"legacy binary is not uniquely locked for {backend}")
    binary = binary_rows[0]
    correctness_path = root / "correctness" / model_key / backend / "lane.json"
    require(correctness_path.is_file(), f"correctness prerequisite is missing: {correctness_path}")
    correctness = read_json(correctness_path)
    status = correctness.get("status")
    require(status in {"pass", "blocked"}, f"correctness prerequisite is not pass/blocked: {status}")
    expected_blocked = f"{model_key}/{backend}" in BLOCKED_DOWNSTREAM_GOALS
    require((status == "blocked") == expected_blocked, "correctness status disagrees with the G00 blocked-lane contract")
    return {
        "models_lock": models_lock,
        "model": model,
        "lane": lane,
        "hardware": hardware,
        "legacy": legacy,
        "binary": binary,
        "correctness_status": status,
    }


def normalize_config(root: Path, raw: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    reject_secret_material(raw)
    require(raw.get("schema_version") == SCHEMA_VERSION, "collector config schema_version must be 1")
    config = copy.deepcopy(raw)
    backend = str(config["backend"])
    model = context["model"]
    hardware = context["hardware"]
    for field in ("model_origin_path", "tokenizer_origin_path", "request_model"):
        require(isinstance(config.get(field), str) and config[field], f"config.{field} is required")
    model_origin = Path(config["model_origin_path"]).expanduser().resolve()
    tokenizer_origin = Path(config["tokenizer_origin_path"]).expanduser().resolve()
    require(model_origin.exists(), f"model origin does not exist: {model_origin}")
    require(tokenizer_origin.is_dir(), f"tokenizer origin is not a directory: {tokenizer_origin}")
    config["model_origin_path"] = str(model_origin)
    config["tokenizer_origin_path"] = str(tokenizer_origin)
    cap = config.get("typed_active_cap")
    budget = config.get("memory_budget_bytes")
    require(isinstance(cap, int) and not isinstance(cap, bool) and cap > 0, "typed_active_cap must be positive")
    require(isinstance(budget, int) and not isinstance(budget, bool) and 0 < budget <= hardware["memory_bytes"], "memory_budget_bytes exceeds locked hardware")
    if config["model_key"] == "m2-qwen35-35b-a3b" and backend == "cuda":
        require(cap >= 16, "M2 CUDA typed_active_cap must be at least 16")
    server = config.get("server")
    require(isinstance(server, dict), "config.server is required")
    require(isinstance(server.get("host"), str) and server["host"], "server.host is required")
    require(isinstance(server.get("port"), int) and 1024 <= server["port"] <= 65535, "server.port must be 1024..65535")
    server.setdefault("ready_timeout_sec", 900)
    server.setdefault("shutdown_timeout_sec", 60)
    server.setdefault("command_timeout_sec", 7200)
    for field in ("ready_timeout_sec", "shutdown_timeout_sec", "command_timeout_sec"):
        require(isinstance(server[field], (int, float)) and not isinstance(server[field], bool) and server[field] > 0, f"server.{field} must be positive")
    client = config.get("benchmark_client")
    require(isinstance(client, dict), "config.benchmark_client is required")
    for field in ("binary_path", "build_log_path", "source_git_sha"):
        require(isinstance(client.get(field), str) and client[field], f"benchmark_client.{field} is required")
    client_binary = Path(client["binary_path"]).expanduser().resolve()
    client_build_log = Path(client["build_log_path"]).expanduser().resolve()
    require(
        client_binary.is_file() and os.access(client_binary, os.X_OK),
        f"benchmark client binary is missing or not executable: {client_binary}",
    )
    require(
        client_build_log.is_file() and client_build_log.stat().st_size > 0,
        f"benchmark client build log is missing or empty: {client_build_log}",
    )
    require(GIT_SHA_RE.fullmatch(client["source_git_sha"]) is not None, "benchmark client source SHA is invalid")
    require(isinstance(client.get("cargo_features"), list) and client["cargo_features"], "benchmark client cargo_features are required")
    external = config.get("external")
    require(isinstance(external, dict), "config.external is required")
    expected_engine = "vllm" if backend == "cuda" else "llama.cpp"
    require(external.get("engine") == expected_engine, f"external.engine must be {expected_engine}")
    for field in ("binary_path", "engine_version", "engine_revision"):
        require(isinstance(external.get(field), str) and external[field], f"external.{field} is required")
    require(GIT_SHA_RE.fullmatch(external["engine_revision"]) is not None, "external.engine_revision must be a 40-hex commit")
    for field in ("server_argv", "version_argv", "revision_argv"):
        require(isinstance(external.get(field), list) and external[field] and all(isinstance(item, str) and item for item in external[field]), f"external.{field} must be argv")
    validate_active_probe(external.get("active_probe"), "external.active_probe")
    validate_external_server_command(config)
    if context["correctness_status"] == "pass":
        legacy = config.get("legacy")
        require(isinstance(legacy, dict), "config.legacy is required for a comparable lane")
        validate_active_probe(legacy.get("active_probe"), "legacy.active_probe")
        extra = legacy.setdefault("extra_serve_argv", [])
        require(isinstance(extra, list) and all(isinstance(item, str) and item for item in extra), "legacy.extra_serve_argv must be argv")
    datasets = config.setdefault("datasets", {})
    require(isinstance(datasets, dict), "config.datasets must be an object")
    required_realistic = "sharegpt" if backend == "cuda" else "real-chat"
    require(isinstance(datasets.get(required_realistic), str) and datasets[required_realistic], f"datasets.{required_realistic} is required")
    dataset_path = Path(datasets[required_realistic]).expanduser().resolve()
    require(dataset_path.is_file() and dataset_path.stat().st_size > 0, f"dataset is missing: {dataset_path}")
    datasets[required_realistic] = str(dataset_path)
    slo = config.setdefault("goodput_slo", {"ttft": 500.0, "tpot": 50.0, "e2e": 30000.0})
    require(isinstance(slo, dict) and set(slo) == {"ttft", "tpot", "e2e"}, "goodput_slo must contain ttft/tpot/e2e")
    require(all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0 for value in slo.values()), "goodput_slo values must be positive finite")
    model_files = locked_file_map(model, backend)
    if model_origin.is_file():
        require(len(model_files) == 1 and model_origin.name in model_files, "file model origin does not match models.lock")
        require(file_sha256(model_origin) == model_files[model_origin.name], "model origin SHA256 differs from models.lock")
    else:
        require(model_origin.is_dir(), "model origin must be a model file or snapshot directory")
        for relative, digest in model_files.items():
            candidate = model_origin / relative
            require(candidate.is_file(), f"locked model file is missing: {candidate}")
            require(file_sha256(candidate) == digest, f"locked model file SHA256 mismatch: {candidate}")
    _, tokenizer = tokenizer_lock(model, backend)
    tokenizer_path = tokenizer_origin / "tokenizer.json"
    require(tokenizer_path.is_file() and file_sha256(tokenizer_path) == tokenizer["sha256"], "tokenizer origin differs from models.lock")
    return config


def validate_active_probe(raw: Any, label: str) -> None:
    require(isinstance(raw, dict), f"{label} is required")
    require(raw.get("format") in {"json", "prometheus"}, f"{label}.format must be json or prometheus")
    require(isinstance(raw.get("path"), str) and raw["path"].startswith("/"), f"{label}.path must be absolute")
    require(isinstance(raw.get("selector"), str) and raw["selector"], f"{label}.selector is required")


def render_argv(template: list[str], values: dict[str, str]) -> list[str]:
    rendered: list[str] = []
    for token in template:
        try:
            value = token.format_map(values)
        except (KeyError, ValueError) as exc:
            raise CollectorError(f"invalid command template token {token!r}: {exc}") from exc
        require(value != "", f"command template rendered an empty token: {token!r}")
        rendered.append(value)
    return rendered


def validate_external_server_command(config: dict[str, Any]) -> list[str]:
    external = config["external"]
    values = {
        "model_origin_path": config["model_origin_path"],
        "request_model": config["request_model"],
        "host": config["server"]["host"],
        "port": str(config["server"]["port"]),
        "typed_active_cap": str(config["typed_active_cap"]),
        "memory_budget_bytes": str(config["memory_budget_bytes"]),
    }
    argv = render_argv(external["server_argv"], values)
    binary = Path(external["binary_path"]).expanduser().resolve()
    require(
        binary.is_file() and os.access(binary, os.X_OK),
        f"external server binary is missing or not executable: {binary}",
    )
    require(
        Path(argv[0]).expanduser().resolve() == binary,
        "external server argv[0] must equal external.binary_path",
    )
    require(
        Path(external["version_argv"][0]).expanduser().resolve() == binary,
        "external version argv[0] must equal external.binary_path",
    )
    revision_program = external["revision_argv"][0]
    if "/" in revision_program:
        revision_executable = Path(revision_program).expanduser().resolve()
        revision_exists = revision_executable.is_file() and os.access(revision_executable, os.X_OK)
    else:
        revision_exists = shutil.which(
            revision_program,
            path=sanitized_environment().get("PATH"),
        ) is not None
    require(revision_exists, f"external revision probe executable is missing: {revision_program}")
    _, options, _ = baseline_gate.parse_argv(argv, "external.server_argv")
    baseline_gate.require_option(options, "--host", values["host"], "external.server_argv")
    baseline_gate.require_option(options, "--port", values["port"], "external.server_argv")
    if external["engine"] == "vllm":
        require(Path(argv[0]).name == "vllm", "external.server_argv must execute the vllm binary")
        require(len(argv) >= 3 and argv[1] == "serve", "external.server_argv must execute vllm serve")
        require(
            argv[2] == values["model_origin_path"],
            "external.server_argv vllm positional model origin mismatch",
        )
        require("--model" not in options, "external.server_argv must use the vllm positional model")
        baseline_gate.require_option(
            options,
            "--served-model-name",
            values["request_model"],
            "external.server_argv",
        )
        baseline_gate.require_option(
            options,
            "--max-num-seqs",
            values["typed_active_cap"],
            "external.server_argv",
        )
    else:
        require(
            "llama" in Path(argv[0]).name.lower(),
            "external.server_argv must execute a llama.cpp server binary",
        )
        baseline_gate.require_option(
            options,
            "--model",
            values["model_origin_path"],
            "external.server_argv",
        )
        baseline_gate.require_option(
            options,
            "--alias",
            values["request_model"],
            "external.server_argv",
        )
        baseline_gate.require_option(
            options,
            "--parallel",
            values["typed_active_cap"],
            "external.server_argv",
        )
    return argv


def collection_fingerprint(root: Path, config: dict[str, Any]) -> str:
    correctness = root / "correctness" / config["model_key"] / config["backend"] / "lane.json"
    external_binary = Path(config["external"]["binary_path"]).expanduser().resolve()
    material = {
        "collector_sha256": file_sha256(COLLECTOR_PATH),
        "models_lock_sha256": file_sha256(root / "models.lock.json"),
        "legacy_binaries_sha256": file_sha256(root / "legacy-binaries.json"),
        "correctness_lane_sha256": file_sha256(correctness),
        "external_binary_sha256": file_sha256(external_binary),
        "config": config,
    }
    return canonical_json_sha256(material)


def prepare_plan(root: Path, config: dict[str, Any], context: dict[str, Any], resume: bool) -> tuple[Path, str]:
    fingerprint = collection_fingerprint(root, config)
    lane_dir = root / "collection" / config["model_key"] / config["backend"]
    config_path = write_artifact_json(
        root,
        f"collection/{config['model_key']}/{config['backend']}/config.normalized.json",
        config,
    )
    plan_path = lane_dir / "plan.json"
    slots = [
        {"slot": slot, "implementation": owner}
        for slot, owner in enumerate(SLOT_ORDER, start=1)
        if context["correctness_status"] == "pass" or owner == "A"
    ]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_g00_performance_collection_plan",
        "collector_path": COLLECTOR_RELATIVE_PATH,
        "collector_sha256": file_sha256(COLLECTOR_PATH),
        "config_fingerprint": fingerprint,
        "config": artifact_relative(root, config_path),
        "config_sha256": file_sha256(config_path),
        "correctness_lane_sha256": file_sha256(
            root / "correctness" / config["model_key"] / config["backend"] / "lane.json"
        ),
        "external_binary_sha256": file_sha256(
            Path(config["external"]["binary_path"]).expanduser().resolve()
        ),
        "model_key": config["model_key"],
        "backend": config["backend"],
        "correctness_status": context["correctness_status"],
        "hardware_id": context["hardware"]["id"],
        "cells": [{"dataset": dataset, "concurrency": concurrency} for dataset, concurrency in expected_cells(config["backend"])],
        "slots": slots,
        "paid_work_started_by_collector": False,
    }
    if plan_path.exists():
        existing = read_json(plan_path)
        require(resume, f"collection plan already exists; pass --resume: {plan_path}")
        require(existing == plan, "resume plan differs from the original collection inputs")
    else:
        atomic_write_json(plan_path, plan)
    return lane_dir, fingerprint


def prepare_benchmark_client(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    raw = config["benchmark_client"]
    backend = config["backend"]
    source_sha = raw["source_git_sha"]
    require(git_output(["cat-file", "-t", source_sha]) == "commit", "benchmark client source commit is unavailable")
    source_tree = git_output(["rev-parse", f"{source_sha}^{{tree}}"])
    binary = stage_file(root, Path(raw["binary_path"]), f"benchmark-client/{backend}/ferrum")
    build_log = stage_file(root, Path(raw["build_log_path"]), f"benchmark-client/{backend}/build.log")
    require(build_log.stat().st_size > 0, "benchmark client build log is empty")
    source_files = []
    for relative in baseline_gate.BENCHMARK_CLIENT_RUST_ALLOWLIST:
        source_files.append({"path": relative, "sha256": hashlib.sha256(git_bytes(["show", f"{source_sha}:{relative}"])).hexdigest()})
    identity = {
        "source_git_sha": source_sha,
        "source_tree_sha": source_tree,
        "collector_source_files": sorted(source_files, key=lambda row: row["path"]),
        "production_rust_diff": sorted(baseline_gate.BENCHMARK_CLIENT_RUST_ALLOWLIST),
    }
    client = {
        "binary_path": str(binary),
        "binary_sha256": file_sha256(binary),
        "artifact_binary": artifact_relative(root, binary),
        "source_git_sha": source_sha,
        "source_tree_sha": source_tree,
        "dirty_status": {"is_dirty": False, "status_short": []},
        "collector_source_files": source_files,
        "production_rust_diff": list(baseline_gate.BENCHMARK_CLIENT_RUST_ALLOWLIST),
        "collector_identity_sha256": canonical_json_sha256(identity),
        "cargo_features": raw["cargo_features"],
        "build_log": artifact_relative(root, build_log),
    }
    baseline_gate.validate_benchmark_client(
        root,
        client,
        "collector.benchmark_client",
        allow_synthetic=False,
    )
    return client


def validate_identity_probe(
    root: Path,
    receipt: dict[str, Any],
    *,
    name: str,
    argv: list[str],
    expected: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    require(receipt.get("name") == name, f"identity probe {name} receipt name mismatch")
    require(receipt.get("argv") == argv, f"identity probe {name} argv changed across resume")
    require(receipt.get("env") == environment, f"identity probe {name} environment changed across resume")
    require(receipt.get("expected") == expected, f"identity probe {name} expected value changed across resume")
    require(
        receipt.get("timeout_sec") == IDENTITY_PROBE_TIMEOUT_SEC,
        f"identity probe {name} timeout changed across resume",
    )
    require(receipt.get("returncode") == 0, f"identity probe {name} receipt is not successful")
    stdout_path = artifact_path(root, require_string(receipt.get("stdout"), f"identity probe {name} stdout"))
    stderr_path = artifact_path(root, require_string(receipt.get("stderr"), f"identity probe {name} stderr"))
    require(stdout_path.is_file() and stderr_path.is_file(), f"identity probe {name} logs are missing")
    require(file_sha256(stdout_path) == receipt.get("stdout_sha256"), f"identity probe {name} stdout SHA256 mismatch")
    require(file_sha256(stderr_path) == receipt.get("stderr_sha256"), f"identity probe {name} stderr SHA256 mismatch")
    observed = f"{stdout_path.read_text(encoding='utf-8', errors='replace')}\n{stderr_path.read_text(encoding='utf-8', errors='replace')}"
    require(expected in observed, f"identity probe {name} logs do not contain expected value {expected!r}")
    duration_seconds(
        require_string(receipt.get("started_at"), f"identity probe {name} started_at"),
        require_string(receipt.get("finished_at"), f"identity probe {name} finished_at"),
    )
    return receipt


def run_identity_probe(
    root: Path,
    lane_dir: Path,
    name: str,
    argv: list[str],
    expected: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    identity_dir = lane_dir / "identity"
    receipt_path = identity_dir / f"{name}.json"
    if receipt_path.exists():
        return validate_identity_probe(
            root,
            read_json(receipt_path),
            name=name,
            argv=argv,
            expected=expected,
            environment=environment,
        )
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt_dir = identity_dir / "attempts" / f"{name}-{attempt_id}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    started_at = now_iso()
    try:
        process = subprocess.run(
            argv,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=IDENTITY_PROBE_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(str(exc.stdout or "[collector] identity probe timed out without stdout\n"), encoding="utf-8")
        stderr_path.write_text(str(exc.stderr or "[collector] identity probe timed out without stderr\n"), encoding="utf-8")
        raise CollectorError(
            f"identity probe timed out after {IDENTITY_PROBE_TIMEOUT_SEC:.0f}s: {' '.join(argv)}"
        ) from exc
    finished_at = now_iso()
    stdout_path.write_text(process.stdout or "[collector] identity probe produced no stdout\n", encoding="utf-8")
    stderr_path.write_text(process.stderr or "[collector] identity probe produced no stderr\n", encoding="utf-8")
    require(process.returncode == 0, f"identity probe failed: {' '.join(argv)}")
    require(expected in f"{process.stdout}\n{process.stderr}", f"identity probe did not report expected value {expected!r}")
    receipt = {
        "name": name,
        "argv": argv,
        "env": environment,
        "expected": expected,
        "timeout_sec": IDENTITY_PROBE_TIMEOUT_SEC,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": duration_seconds(started_at, finished_at),
        "returncode": process.returncode,
        "stdout": artifact_relative(root, stdout_path),
        "stdout_sha256": file_sha256(stdout_path),
        "stderr": artifact_relative(root, stderr_path),
        "stderr_sha256": file_sha256(stderr_path),
    }
    atomic_write_json(receipt_path, receipt)
    return validate_identity_probe(
        root,
        receipt,
        name=name,
        argv=argv,
        expected=expected,
        environment=environment,
    )


def normalized_server_config(context: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    model = context["model"]
    lane = context["lane"]
    return {
        "schema_version": SCHEMA_VERSION,
        "config_source": "normalized-server-argv",
        "model_key": model["key"],
        "backend": config["backend"],
        "model_repo": lane["repo"],
        "model_revision": lane["revision"],
        "model_format": lane["format"],
        "model_files": locked_file_map(model, config["backend"]),
        "official_model_id": model["official_model_id"],
        "request_model": config["request_model"],
        "enable_thinking": False,
        "model_origin_path": config["model_origin_path"],
        "typed_active_cap": config["typed_active_cap"],
        "memory_budget_bytes": config["memory_budget_bytes"],
    }


def prepare_identities(root: Path, lane_dir: Path, config: dict[str, Any], context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized = normalized_server_config(context, config)
    identities: dict[str, dict[str, Any]] = {}
    external_env = sanitized_environment(config["external"].get("env"))
    version_probe = run_identity_probe(root, lane_dir, "external-version", config["external"]["version_argv"], config["external"]["engine_version"], external_env)
    revision_probe = run_identity_probe(root, lane_dir, "external-revision", config["external"]["revision_argv"], config["external"]["engine_revision"], external_env)
    external_binary = Path(config["external"]["binary_path"]).expanduser().resolve()
    require(external_binary.is_file(), f"external server binary is missing: {external_binary}")
    for implementation in ("A", "B"):
        if implementation == "B" and context["correctness_status"] != "pass":
            continue
        config_path = write_artifact_json(root, f"server-config/{config['model_key']}/{config['backend']}/{implementation}.json", normalized)
        identity = {
            "implementation": implementation,
            "role": "external" if implementation == "A" else "legacy",
            "engine": config["external"]["engine"] if implementation == "A" else "ferrum",
            "binary_sha256": file_sha256(external_binary) if implementation == "A" else context["binary"]["binary_sha256"],
            "effective_config": artifact_relative(root, config_path),
            "effective_config_sha256": file_sha256(config_path),
            "typed_active_cap": config["typed_active_cap"],
            "model_key": context["model"]["key"],
            "model_repo": context["lane"]["repo"],
            "model_revision": context["lane"]["revision"],
            "model_format": context["lane"]["format"],
            "model_files": locked_file_map(context["model"], config["backend"]),
            "official_model_id": context["model"]["official_model_id"],
            "request_model": config["request_model"],
            "model_origin_path": config["model_origin_path"],
            "memory_budget_bytes": config["memory_budget_bytes"],
        }
        if implementation == "A":
            identity.update({
                "engine_version": config["external"]["engine_version"],
                "engine_revision": config["external"]["engine_revision"],
                "identity_probes": {"version": version_probe, "revision": revision_probe},
            })
        else:
            identity["source_git_sha"] = baseline_gate.FROZEN_LEGACY_SHA
        identities[implementation] = identity
    return identities


def prepare_workloads(root: Path, config: dict[str, Any], context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    model = context["model"]
    backend = config["backend"]
    tokenizer_source, tokenizer_row = tokenizer_lock(model, backend)
    tokenizer = stage_file(
        root,
        Path(config["tokenizer_origin_path"]) / "tokenizer.json",
        f"workloads/{config['model_key']}/{backend}/tokenizer.json",
        tokenizer_row["sha256"],
    )
    dataset_artifacts: dict[str, tuple[Path, str]] = {}
    random_descriptor = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "random",
        "generator": "canonical-tokenizer-aware-random",
        "seed": 9271,
        "input_tokens": 256 if backend == "cuda" else 64,
        "output_tokens": 128,
    }
    random_path = write_artifact_json(root, f"workloads/{config['model_key']}/{backend}/datasets/random.json", random_descriptor)
    dataset_artifacts["random"] = (random_path, str(random_path))
    realistic = "sharegpt" if backend == "cuda" else "real-chat"
    realistic_path = stage_file(
        root,
        Path(config["datasets"][realistic]),
        f"workloads/{config['model_key']}/{backend}/datasets/{realistic}.jsonl",
    )
    dataset_artifacts[realistic] = (realistic_path, str(realistic_path))
    workloads: dict[str, dict[str, Any]] = {}
    for dataset, concurrency in expected_cells(backend):
        dataset_path, dataset_origin = dataset_artifacts[dataset]
        effective = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset,
            "dataset_sha256": file_sha256(dataset_path),
            "tokenizer_sha256": file_sha256(tokenizer),
            "model_revision": context["lane"]["revision"],
            "seed": 9271,
            "max_output_tokens": 128,
            "ignore_eos": dataset == "random",
            "enable_thinking": False,
            "requested_concurrency": concurrency,
            "typed_active_cap": config["typed_active_cap"],
            "request_model": config["request_model"],
        }
        cell_id = f"{dataset}:c{concurrency}"
        effective_path = write_artifact_json(
            root,
            f"workloads/{config['model_key']}/{backend}/cells/{dataset}-c{concurrency}.json",
            effective,
        )
        workloads[cell_id] = {
            "dataset_id": dataset,
            "dataset_artifact": artifact_relative(root, dataset_path),
            "dataset_sha256": file_sha256(dataset_path),
            "dataset_origin_path": dataset_origin,
            "tokenizer_artifact": artifact_relative(root, tokenizer),
            "tokenizer_sha256": file_sha256(tokenizer),
            "tokenizer_id": tokenizer_source["repo"],
            "tokenizer_revision": tokenizer_source["revision"],
            "tokenizer_origin_path": config["tokenizer_origin_path"],
            "effective_config": artifact_relative(root, effective_path),
            "effective_config_sha256": file_sha256(effective_path),
        }
    return workloads


def process_identity(pid: int) -> tuple[str, dict[str, str]]:
    try:
        return resource_sampler._process_start_identity(pid)
    except (OSError, resource_sampler.ResourceEvidenceError, subprocess.CalledProcessError) as exc:
        raise CollectorError(f"cannot capture process start identity for pid {pid}: {exc}") from exc


def write_process_receipt(
    root: Path,
    path: Path,
    *,
    pid: int,
    pgid: int,
    argv: list[str],
    environment: dict[str, str],
    marker: str,
    source: dict[str, str],
) -> dict[str, Any]:
    process = subprocess.run(
        ["ps", "-ww", "-p", str(pid), "-o", "pid=", "-o", "ppid=", "-o", "pgid=", "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    require(process.returncode == 0 and process.stdout.strip(), f"ps could not capture pid {pid}")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": now_iso(),
        "pid": pid,
        "pgid": pgid,
        "process_start_marker": marker,
        "process_start_source": source,
        "argv": argv,
        "argv_sha256": canonical_json_sha256(argv),
        "environment": environment,
        "environment_sha256": canonical_json_sha256(environment),
        "ps_stdout": process.stdout,
        "ps_stdout_sha256": hashlib.sha256(process.stdout.encode("utf-8")).hexdigest(),
    }
    atomic_write_json(path, receipt)
    return {"path": artifact_relative(root, path), "sha256": file_sha256(path)}


def wait_for_server(process: subprocess.Popen[Any], url: str, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    last_error = "not attempted"
    while time.monotonic() < deadline:
        returncode = process.poll()
        require(returncode is None, f"server exited before readiness with returncode {returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                body = response.read()
                if response.status == 200 and body:
                    return
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise CollectorError(f"server readiness timeout for {url}: {last_error}")


def wait_for_process_exec(
    process: subprocess.Popen[Any],
    expected_binary: Path,
    timeout_sec: float,
) -> None:
    deadline = time.monotonic() + timeout_sec
    expected_name = expected_binary.name
    while time.monotonic() < deadline:
        require(process.poll() is None, f"exec barrier child exited before product exec with {process.returncode}")
        probe = subprocess.run(
            ["ps", "-ww", "-p", str(process.pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0 and expected_name in probe.stdout and "--exec-barrier-child" not in probe.stdout:
            return
        time.sleep(0.02)
    raise CollectorError(f"exec barrier did not enter {expected_name} within {timeout_sec:.1f}s")


def collect_endpoint_probe(
    root: Path,
    attempt_dir: Path,
    name: str,
    url: str,
    environment: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    body_path = attempt_dir / f"{name}.body.json"
    receipt_path = attempt_dir / f"{name}.receipt.json"
    stdout_path = attempt_dir / f"{name}.stdout.log"
    stderr_path = attempt_dir / f"{name}.stderr.log"
    argv = [
        sys.executable,
        str(RESOURCE_SAMPLER_PATH),
        "--probe-url",
        url,
        "--probe-body-out",
        str(body_path),
        "--probe-receipt-out",
        str(receipt_path),
        "--probe-timeout-sec",
        "10",
    ]
    process = subprocess.run(argv, env=environment, capture_output=True, text=True, check=False)
    stdout_path.write_text(process.stdout or "[collector] endpoint probe produced no stdout\n", encoding="utf-8")
    stderr_path.write_text(process.stderr or "[collector] endpoint probe produced no stderr\n", encoding="utf-8")
    require(process.returncode == 0, f"endpoint probe failed for {url}: {process.stderr.strip()}")
    read_json(receipt_path)
    body = read_json(body_path)
    evidence = {
        "receipt_origin_path": str(receipt_path),
        "body_origin_path": str(body_path),
        "receipt": artifact_relative(root, receipt_path),
        "receipt_sha256": file_sha256(receipt_path),
        "body": artifact_relative(root, body_path),
        "body_sha256": file_sha256(body_path),
    }
    return evidence, body


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def signal_process_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def wait_process_group_gone(pgid: int, timeout_sec: float) -> bool:
    deadline = time.monotonic() + max(timeout_sec, 0.0)
    while process_group_exists(pgid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def terminate_process_group(process: subprocess.Popen[Any], timeout_sec: float) -> tuple[int, bool]:
    pgid = process.pid
    returncode = process.poll()
    if returncode is None:
        signal_process_group(pgid, signal.SIGINT)
        try:
            returncode = process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            signal_process_group(pgid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=min(max(timeout_sec, 1.0), 15.0))
            except subprocess.TimeoutExpired:
                signal_process_group(pgid, signal.SIGKILL)
                returncode = process.wait(timeout=10)
    if wait_process_group_gone(pgid, 1.0):
        return int(returncode), True
    signal_process_group(pgid, signal.SIGTERM)
    if not wait_process_group_gone(pgid, 2.0):
        signal_process_group(pgid, signal.SIGKILL)
    return int(returncode), wait_process_group_gone(pgid, 5.0)


def cleanup_process_group_noexcept(
    process: subprocess.Popen[Any],
    timeout_sec: float,
) -> tuple[int | None, bool, str | None]:
    try:
        returncode, group_gone = terminate_process_group(process, timeout_sec)
        return returncode, group_gone, None
    except BaseException as exc:
        signal_process_group(process.pid, signal.SIGKILL)
        group_gone = wait_process_group_gone(process.pid, 5.0)
        returncode = process.poll()
        if returncode is None:
            try:
                returncode = process.wait(timeout=1.0)
            except BaseException:
                pass
        return returncode, group_gone, f"{type(exc).__name__}: {exc}"


def server_argv(
    root: Path,
    attempt_dir: Path,
    implementation: str,
    config: dict[str, Any],
    context: dict[str, Any],
) -> tuple[list[str], Path | None, dict[str, str], dict[str, Any]]:
    host = config["server"]["host"]
    port = config["server"]["port"]
    if implementation == "A":
        argv = validate_external_server_command(config)
        environment = sanitized_environment(config["external"].get("env"))
        probe = config["external"]["active_probe"]
        return argv, None, environment, probe
    binary = root / context["binary"]["artifact_binary"]
    require(binary.is_file() and file_sha256(binary) == context["binary"]["binary_sha256"], "frozen legacy binary artifact mismatch")
    binary.chmod(binary.stat().st_mode | 0o100)
    product_config = attempt_dir / "product-effective-config.json"
    argv = [
        str(binary),
        "serve",
        config["model_origin_path"],
        "--backend",
        config["backend"],
        "--host",
        host,
        "--port",
        str(port),
        "--max-num-seqs",
        str(config["typed_active_cap"]),
        "--effective-config-json",
        str(product_config),
        *config["legacy"]["extra_serve_argv"],
    ]
    environment = sanitized_environment(config["legacy"].get("env"))
    return argv, product_config, environment, config["legacy"]["active_probe"]


def exec_barrier_launcher_argv(release_file: Path, product_argv: list[str]) -> list[str]:
    require(product_argv and all(isinstance(item, str) and item for item in product_argv), "exec barrier product argv is invalid")
    return [
        sys.executable,
        str(COLLECTOR_PATH),
        "--exec-barrier-child",
        "--release-file",
        str(release_file),
        "--",
        *product_argv,
    ]


def bench_argv(
    root: Path,
    client: dict[str, Any],
    workload: dict[str, Any],
    session: dict[str, Any],
    config: dict[str, Any],
    dataset: str,
    concurrency: int,
    raw_report: Path,
) -> list[str]:
    slo = config["goodput_slo"]
    argv = [
        client["binary_path"],
        "bench-serve",
        "--base-url",
        session["base_url"],
        "--model",
        config["request_model"],
        "--tokenizer",
        workload["tokenizer_origin_path"],
        "--concurrency",
        str(concurrency),
        "--dataset",
        "random" if dataset == "random" else "sharegpt",
        "--random-input-len",
        "256" if config["backend"] == "cuda" else "64",
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
        "--output",
        "json",
        "--out",
        str(raw_report),
        "--hw-id",
        session["hardware_id"],
        "--commit-sha",
        client["source_git_sha"],
        "--goodput",
        f"ttft:{slo['ttft']},tpot:{slo['tpot']},e2el:{slo['e2e']}",
        "--enable-thinking",
        "false",
        "--timeout",
        str(config["server"]["command_timeout_sec"]),
        "--fail-on-error",
        "--require-ci",
    ]
    if dataset == "random":
        argv.append("--ignore-eos")
    else:
        argv.extend(["--sharegpt-path", workload["dataset_origin_path"]])
    return argv


def start_resource_sampler(
    root: Path,
    attempt_dir: Path,
    session: dict[str, Any],
    config: dict[str, Any],
    probe: dict[str, Any],
    cell_id: str,
) -> dict[str, Any]:
    stem = cell_id.replace(":", "-")
    observations = attempt_dir / f"{stem}.resource-observations.jsonl"
    stop_file = attempt_dir / f"{stem}.resource-stop"
    stdout_path = attempt_dir / f"{stem}.resource-sampler.stdout.log"
    stderr_path = attempt_dir / f"{stem}.resource-sampler.stderr.log"
    require(not observations.exists(), f"resource observation already exists: {observations}")
    argv = [
        sys.executable,
        str(RESOURCE_SAMPLER_PATH),
        "--out",
        str(observations),
        "--pid",
        str(session["pid"]),
        "--pgid",
        str(session["pgid"]),
        "--session-id",
        session["session_id"],
        "--cell-id",
        cell_id,
        "--backend",
        config["backend"],
        "--hardware-id",
        session["hardware_id"],
        "--base-url",
        session["base_url"],
        "--active-probe-format",
        probe["format"],
        "--active-selector",
        probe["selector"],
        "--active-semantics",
        "scheduler-active-high-water",
        "--runtime-log",
        session["runtime_log_origin_path"],
        "--stop-file",
        str(stop_file),
        "--interval-ms",
        "250",
        "--max-duration-sec",
        "7200",
        "--active-path",
        probe["path"],
    ]
    stdout_handle = stdout_path.open("x", encoding="utf-8")
    stderr_handle = stderr_path.open("x", encoding="utf-8")
    process: subprocess.Popen[Any] | None = None
    meta: dict[str, Any] | None = None
    try:
        stdout_handle.write("[collector] resource sampler stdout follows\n")
        stderr_handle.write("[collector] resource sampler stderr follows\n")
        stdout_handle.flush()
        stderr_handle.flush()
        process = subprocess.Popen(
            argv,
            env=sanitized_environment(),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
        meta = {
            "process": process,
            "argv": argv,
            "observations": observations,
            "stop_file": stop_file,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "stdout_handle": stdout_handle,
            "stderr_handle": stderr_handle,
            "finished": False,
        }
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            require(process.poll() is None, f"resource sampler exited during startup with {process.returncode}")
            if observations.exists() and observations.stat().st_size > 0:
                lines = observations.read_text(encoding="utf-8", errors="replace").splitlines()
                if len(lines) >= 2:
                    return meta
            time.sleep(0.05)
        raise CollectorError("resource sampler did not produce its first observation")
    except BaseException:
        if process is not None:
            cleanup_process_group_noexcept(process, 5.0)
        for handle in (stdout_handle, stderr_handle):
            try:
                handle.close()
            except BaseException:
                pass
        ensure_nonempty_log(stdout_path, "resource sampler stdout")
        ensure_nonempty_log(stderr_path, "resource sampler stderr")
        raise


def close_sampler_handles(meta: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("stdout_handle", "stderr_handle"):
        handle = meta[key]
        try:
            if not handle.closed:
                handle.flush()
                handle.close()
        except BaseException as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return errors


def finish_resource_sampler(meta: dict[str, Any], *, bracket_after_measurement: bool = True) -> None:
    if meta.get("finished") is True:
        return
    process: subprocess.Popen[Any] = meta["process"]
    returncode: int | None = None
    group_gone = False
    failure: BaseException | None = None
    close_errors: list[str] = []
    try:
        if bracket_after_measurement:
            time.sleep(0.4)
            meta["stop_file"].write_text("stop\n", encoding="utf-8")
        try:
            returncode = process.wait(timeout=20)
            group_gone = wait_process_group_gone(process.pid, 2.0)
            if not group_gone:
                returncode, group_gone = terminate_process_group(process, 2.0)
        except subprocess.TimeoutExpired:
            returncode, group_gone = terminate_process_group(process, 5.0)
    except BaseException as exc:
        failure = exc
        try:
            returncode, group_gone = terminate_process_group(process, 5.0)
        except BaseException:
            group_gone = not process_group_exists(process.pid)
    finally:
        meta["finished"] = True
        close_errors = close_sampler_handles(meta)
        ensure_nonempty_log(meta["stdout_path"], "resource sampler stdout")
        ensure_nonempty_log(meta["stderr_path"], "resource sampler stderr")
    if failure is not None:
        raise failure
    require(group_gone, "resource sampler process group survived cleanup")
    require(returncode == 0, f"resource sampler failed with returncode {returncode}")
    require(not close_errors, f"resource sampler log handles failed to close: {close_errors}")


def run_bench_cell(
    root: Path,
    attempt_dir: Path,
    session: dict[str, Any],
    client: dict[str, Any],
    workload: dict[str, Any],
    config: dict[str, Any],
    probe: dict[str, Any],
    dataset: str,
    concurrency: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cell_id = f"{dataset}:c{concurrency}"
    stem = cell_id.replace(":", "-")
    raw_report = attempt_dir / f"{stem}.bench-report.json"
    stdout_path = attempt_dir / f"{stem}.bench.stdout.log"
    stderr_path = attempt_dir / f"{stem}.bench.stderr.log"
    sampler: dict[str, Any] | None = None
    process: subprocess.Popen[Any] | None = None
    try:
        sampler = start_resource_sampler(root, attempt_dir, session, config, probe, cell_id)
        argv = bench_argv(root, client, workload, session, config, dataset, concurrency, raw_report)
        environment = sanitized_environment()
        started_at = now_iso()
        with stdout_path.open("x", encoding="utf-8") as stdout_handle, stderr_path.open("x", encoding="utf-8") as stderr_handle:
            stdout_handle.write("[collector] benchmark client stdout follows\n")
            stderr_handle.write("[collector] benchmark client stderr follows\n")
            stdout_handle.flush()
            stderr_handle.flush()
            process = subprocess.Popen(
                argv,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=config["server"]["command_timeout_sec"])
                returncode, bench_group_gone = terminate_process_group(process, 2.0)
            except subprocess.TimeoutExpired:
                _, bench_group_gone = terminate_process_group(process, 5.0)
                returncode = 124
        finished_at = now_iso()
        finish_resource_sampler(sampler)
        require(bench_group_gone, f"benchmark cell {cell_id} process group survived cleanup")
        process = None
        ensure_nonempty_log(stdout_path, "benchmark stdout")
        ensure_nonempty_log(stderr_path, "benchmark stderr")
        require(returncode == 0, f"benchmark cell {cell_id} failed with returncode {returncode}")
        require(raw_report.is_file() and raw_report.stat().st_size > 0, f"benchmark cell did not write report: {raw_report}")
        report = read_json(raw_report)
        require(report.get("n_repeats") == 3, f"benchmark cell {cell_id} report lacks three repeats")
        record = {
            "session_id": session["session_id"],
            "slot": session["slot"],
            "cell_id": cell_id,
            "dataset": dataset,
            "concurrency": concurrency,
            "benchmark_client_binary_sha256": client["binary_sha256"],
            "raw_report_origin_path": str(raw_report),
            "bench_argv": argv,
            "env": environment,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_sec": duration_seconds(started_at, finished_at),
            "returncode": returncode,
            "stdout": artifact_relative(root, stdout_path),
            "stdout_sha256": file_sha256(stdout_path),
            "stderr": artifact_relative(root, stderr_path),
            "stderr_sha256": file_sha256(stderr_path),
            "raw_report": artifact_relative(root, raw_report),
            "raw_report_sha256": file_sha256(raw_report),
        }
        return record, sampler
    finally:
        if process is not None:
            cleanup_process_group_noexcept(process, 5.0)
        if sampler is not None and sampler.get("finished") is not True:
            try:
                sampler["stop_file"].write_text("stop\n", encoding="utf-8")
                finish_resource_sampler(sampler, bracket_after_measurement=False)
            except BaseException:
                cleanup_process_group_noexcept(sampler["process"], 5.0)
                close_sampler_handles(sampler)
                sampler["finished"] = True
        ensure_nonempty_log(stdout_path, "benchmark stdout")
        ensure_nonempty_log(stderr_path, "benchmark stderr")


def resource_evidence(
    root: Path,
    session: dict[str, Any],
    record: dict[str, Any],
    sampler: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    observations: Path = sampler["observations"]
    summary = resource_sampler.derive_summary(
        observations,
        session_id=session["session_id"],
        cell_id=record["cell_id"],
        backend=config["backend"],
        hardware_id=session["hardware_id"],
        pid=session["pid"],
        pgid=session["pgid"],
        process_start_marker=session["process_start_marker"],
        base_url=session["base_url"],
        session_started_at=session["started_at"],
        session_finished_at=session["finished_at"],
        measurement_started_at=record["started_at"],
        measurement_finished_at=record["finished_at"],
        memory_budget_bytes=config["memory_budget_bytes"],
        requested_concurrency=record["concurrency"],
        typed_active_cap=config["typed_active_cap"],
        runtime_log_path=session["runtime_log_origin_path"],
    )
    return {
        "collector_sha256": file_sha256(RESOURCE_SAMPLER_PATH),
        "sampler_argv": sampler["argv"],
        "observation_origin_path": str(observations),
        "observations": artifact_relative(root, observations),
        "observations_sha256": file_sha256(observations),
        "summary": summary,
    }


def load_session_bundle(path: Path, fingerprint: str, slot: int, implementation: str) -> dict[str, Any]:
    bundle = read_json(path)
    require(bundle.get("config_fingerprint") == fingerprint, f"session bundle fingerprint mismatch: {path}")
    session = bundle.get("session")
    reports = bundle.get("reports")
    require(isinstance(session, dict) and session.get("slot") == slot and session.get("implementation") == implementation, f"invalid session bundle: {path}")
    require(isinstance(reports, list) and reports, f"session bundle has no reports: {path}")
    return bundle


def collect_server_session(
    root: Path,
    lane_dir: Path,
    fingerprint: str,
    slot: int,
    implementation: str,
    config: dict[str, Any],
    context: dict[str, Any],
    client: dict[str, Any],
    identity: dict[str, Any],
    workloads: dict[str, dict[str, Any]],
    resume: bool,
) -> dict[str, Any]:
    bundle_path = lane_dir / "sessions" / f"slot-{slot}.json"
    if bundle_path.exists():
        require(resume, f"server session already collected; pass --resume: {bundle_path}")
        bundle = load_session_bundle(bundle_path, fingerprint, slot, implementation)
        session = bundle["session"]
        require(session.get("executed_binary_sha256") == identity["binary_sha256"], "resumed session binary identity changed")
        require(session.get("effective_config_sha256") == identity["effective_config_sha256"], "resumed session config identity changed")
        require(
            all(report.get("benchmark_client_binary_sha256") == client["binary_sha256"] for report in bundle["reports"]),
            "resumed session benchmark client identity changed",
        )
        return bundle
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt_dir = lane_dir / "attempts" / f"slot-{slot}-{implementation}-{attempt_id}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    runtime_log = attempt_dir / "server-runtime.log"
    command_log = lane_dir / "command-log.jsonl"
    argv, product_config, environment, probe = server_argv(root, attempt_dir, implementation, config, context)
    base_url = f"http://{config['server']['host']}:{config['server']['port']}"
    started_at = now_iso()
    process: subprocess.Popen[Any] | None = None
    runtime_handle = runtime_log.open("x", encoding="utf-8")
    runtime_handle.write(f"[collector] slot={slot} implementation={implementation} argv={json.dumps(argv)}\n")
    runtime_handle.flush()
    failure: BaseException | None = None
    try:
        process = subprocess.Popen(
            argv,
            env=environment,
            stdout=runtime_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        pid = process.pid
        pgid = os.getpgid(pid)
        require(pgid == pid, "server must own an independent process group")
        marker, marker_source = process_identity(pid)
        receipt = write_process_receipt(
            root,
            attempt_dir / "server-process-receipt.json",
            pid=pid,
            pgid=pgid,
            argv=argv,
            environment=environment,
            marker=marker,
            source=marker_source,
        )
        session: dict[str, Any] = {
            "session_id": f"g00-{config['model_key']}-{config['backend']}-s{slot}-{attempt_id}",
            "implementation": implementation,
            "slot": slot,
            "sequence": slot,
            "hardware_id": context["hardware"]["id"],
            "hardware_fingerprint": context["hardware"]["fingerprint"],
            "effective_config": identity["effective_config"],
            "effective_config_sha256": identity["effective_config_sha256"],
            "typed_active_cap": config["typed_active_cap"],
            "executed_binary_sha256": identity["binary_sha256"],
            "pid": pid,
            "pgid": pgid,
            "process_start_marker": marker,
            "process_start_source": marker_source,
            "process_receipt": receipt,
            "server_argv": argv,
            "env": environment,
            "base_url": base_url,
            "started_at": started_at,
            "model_key": config["model_key"],
            "model_revision": context["lane"]["revision"],
            "model_files": locked_file_map(context["model"], config["backend"]),
            "runtime_log": artifact_relative(root, runtime_log),
            "runtime_log_origin_path": str(runtime_log),
        }
        wait_for_server(process, f"{base_url}/v1/models", config["server"]["ready_timeout_sec"])
        ready_probe, ready_body = collect_endpoint_probe(root, attempt_dir, "ready-probe", f"{base_url}/v1/models", environment)
        ready_at = read_json(root / ready_probe["receipt"])["finished_at"]
        session["ready_probe"] = ready_probe
        session["ready_at"] = ready_at
        require(isinstance(ready_body.get("data"), list) and ready_body["data"], "ready probe observed no model")
        model_probe, model_body = collect_endpoint_probe(root, attempt_dir, "model-probe", f"{base_url}/v1/models", environment)
        observed_models = [row.get("id") for row in model_body.get("data", []) if isinstance(row, dict)]
        require(observed_models == [config["request_model"]], f"server exposed unexpected model ids: {observed_models}")
        session["model_probe"] = model_probe
        reports: list[dict[str, Any]] = []
        samplers: list[dict[str, Any]] = []
        windows: list[dict[str, Any]] = []
        for sequence, (dataset, concurrency) in enumerate(expected_cells(config["backend"]), start=1):
            cell_id = f"{dataset}:c{concurrency}"
            record, sampler = run_bench_cell(
                root,
                attempt_dir,
                session,
                client,
                workloads[cell_id],
                config,
                probe,
                dataset,
                concurrency,
            )
            reports.append(record)
            samplers.append(sampler)
            windows.append({
                "sequence": sequence,
                "cell_id": cell_id,
                "dataset": dataset,
                "concurrency": concurrency,
                "started_at": record["started_at"],
                "finished_at": record["finished_at"],
            })
        session["measurement_started_at"] = reports[0]["started_at"]
        session["measurement_finished_at"] = reports[-1]["finished_at"]
        session["cell_windows"] = windows
        session["shutdown_started_at"] = now_iso()
        returncode, group_gone = terminate_process_group(process, config["server"]["shutdown_timeout_sec"])
        session["finished_at"] = now_iso()
        session["duration_sec"] = duration_seconds(session["started_at"], session["finished_at"])
        session["returncode"] = returncode
        session["shutdown_clean"] = returncode == 0 and group_gone
        require(session["shutdown_clean"], f"server did not shut down cleanly: returncode={returncode}, group_gone={group_gone}")
        process = None
        runtime_handle.flush()
        os.fsync(runtime_handle.fileno())
        runtime_handle.close()
        runtime_handle = None
        ensure_nonempty_log(runtime_log, "server runtime")
        session["runtime_log_sha256"] = file_sha256(runtime_log)
        if implementation == "B":
            require(product_config is not None and product_config.is_file(), "frozen Ferrum did not emit effective config")
            session.update({
                "product_effective_config_origin_path": str(product_config),
                "product_effective_config": artifact_relative(root, product_config),
                "product_effective_config_sha256": file_sha256(product_config),
            })
            product = read_json(product_config)
            actual_cap = product.get("admission", {}).get("effective_max_concurrent")
            require(actual_cap == config["typed_active_cap"], f"frozen Ferrum effective active cap {actual_cap} differs from requested {config['typed_active_cap']}")
        for record, sampler in zip(reports, samplers):
            record["resources"] = resource_evidence(root, session, record, sampler, config)
        bundle = {"schema_version": SCHEMA_VERSION, "config_fingerprint": fingerprint, "session": session, "reports": reports}
        atomic_write_json(bundle_path, bundle)
        append_jsonl(command_log, {
            "event": "server-session-complete",
            "slot": slot,
            "implementation": implementation,
            "session_id": session["session_id"],
            "started_at": session["started_at"],
            "finished_at": session["finished_at"],
            "bundle": artifact_relative(root, bundle_path),
            "bundle_sha256": file_sha256(bundle_path),
        })
        return bundle
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if process is not None:
            cleanup_returncode, cleanup_gone, cleanup_error = cleanup_process_group_noexcept(process, 10.0)
        else:
            cleanup_returncode, cleanup_gone, cleanup_error = None, True, None
        runtime_close_error: str | None = None
        if runtime_handle is not None:
            try:
                runtime_handle.flush()
                runtime_handle.close()
            except BaseException as exc:
                runtime_close_error = f"{type(exc).__name__}: {exc}"
                try:
                    runtime_handle.close()
                except BaseException:
                    pass
        ensure_nonempty_log(runtime_log, "server runtime")
        if failure is not None:
            atomic_write_json(attempt_dir / "failure.json", {
                "schema_version": SCHEMA_VERSION,
                "failed_at": now_iso(),
                "error_type": type(failure).__name__,
                "error": str(failure),
                "cleanup_returncode": cleanup_returncode,
                "cleanup_process_group_gone": cleanup_gone,
                "cleanup_error": cleanup_error,
                "runtime_close_error": runtime_close_error,
            })


def implementation_summary(
    root: Path,
    bundles: list[dict[str, Any]],
    implementation: str,
    dataset: str,
    concurrency: int,
) -> dict[str, Any]:
    reports = []
    measured = completed = warmup = warmup_completed = errors = bad_outputs = 0
    for bundle in bundles:
        session = bundle["session"]
        if session["implementation"] != implementation:
            continue
        matches = [
            row for row in bundle["reports"]
            if row["dataset"] == dataset and row["concurrency"] == concurrency
        ]
        require(len(matches) == 1, f"session {session['session_id']} lacks a unique {dataset}:c{concurrency} report")
        record = copy.deepcopy(matches[0])
        report = read_json(root / record["raw_report"])
        repeats = report.get("repeat_metrics")
        require(isinstance(repeats, list) and len(repeats) == 3, "raw report repeat_metrics must contain three rows")
        for repeat in repeats:
            measured += int(repeat.get("expected_requests", 0))
            completed += int(repeat.get("completed_requests", 0))
            errors += int(repeat.get("errored_requests", 0))
            warmup += int(repeat.get("warmup_expected", 0))
            warmup_completed += int(repeat.get("warmup_completed", 0))
            quality = repeat.get("quality_issues", {})
            require(isinstance(quality, dict), "repeat quality_issues must be an object")
            bad_outputs += sum(int(value) for value in quality.values())
        reports.append(record)
    reports.sort(key=lambda row: int(row["slot"]))
    require(len(reports) == 4, f"implementation {implementation} must have four reports per cell")
    return {
        "reports": reports,
        "measured_requests": measured,
        "completed_requests": completed,
        "warmup_requests": warmup,
        "warmup_completed": warmup_completed,
        "error_count": errors,
        "bad_output_count": bad_outputs,
        "output_token_count_source": "usage",
    }


def implementation_input_mean(root: Path, implementation: dict[str, Any]) -> float:
    report_means: list[float] = []
    for record in implementation["reports"]:
        report = read_json(root / record["raw_report"])
        rows = report.get("actual_input_tokens_per_request")
        require(isinstance(rows, list) and rows, "raw report lacks actual input token vectors")
        values = [int(value) for row in rows for value in row]
        require(values and all(value > 0 for value in values), "raw report actual input tokens are invalid")
        report_means.append(sum(values) / len(values))
    return sum(report_means) / len(report_means)


def write_runtime_index(root: Path, lane_dir: Path, bundles: list[dict[str, Any]]) -> Path:
    path = lane_dir / "runtime-index.log"
    lines = [
        f"slot={bundle['session']['slot']} implementation={bundle['session']['implementation']} "
        f"session_id={bundle['session']['session_id']} runtime_log={bundle['session']['runtime_log']} "
        f"sha256={bundle['session']['runtime_log_sha256']}"
        for bundle in sorted(bundles, key=lambda row: int(row["session"]["slot"]))
    ]
    require(lines, "runtime index cannot be empty")
    payload = "\n".join(lines) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == payload, "runtime index differs from collected sessions")
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def lane_identity_base(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_key": config["model_key"],
        "backend": config["backend"],
        "hardware_id": context["hardware"]["id"],
        "model_revision": context["lane"]["revision"],
        "model_files": locked_file_map(context["model"], config["backend"]),
        "source_git_sha": baseline_gate.FROZEN_LEGACY_SHA,
        "source_tree_sha": context["legacy"]["source_tree_sha"],
        "dirty_status": {"is_dirty": False, "status_short": []},
        "binary_sha256": context["binary"]["binary_sha256"],
    }


def run_sampler_for_process(
    root: Path,
    attempt_dir: Path,
    *,
    pid: int,
    pgid: int,
    sample_id: str,
    config: dict[str, Any],
    hardware_id: str,
    stderr_path: Path,
) -> dict[str, Any]:
    observations = attempt_dir / "resource-observations.jsonl"
    stop_file = attempt_dir / "resource-stop"
    stdout_path = attempt_dir / "resource-sampler.stdout.log"
    sampler_stderr_path = attempt_dir / "resource-sampler.stderr.log"
    argv = [
        sys.executable,
        str(RESOURCE_SAMPLER_PATH),
        "--out",
        str(observations),
        "--pid",
        str(pid),
        "--pgid",
        str(pgid),
        "--session-id",
        sample_id,
        "--cell-id",
        "run:c1",
        "--backend",
        config["backend"],
        "--hardware-id",
        hardware_id,
        "--base-url",
        f"process://{sample_id}",
        "--active-probe-format",
        "process",
        "--active-selector",
        "process-alive",
        "--active-semantics",
        "process-alive",
        "--runtime-log",
        str(stderr_path),
        "--stop-file",
        str(stop_file),
        "--interval-ms",
        "250",
        "--max-duration-sec",
        "7200",
    ]
    stdout_handle = stdout_path.open("x", encoding="utf-8")
    stderr_handle = sampler_stderr_path.open("x", encoding="utf-8")
    process: subprocess.Popen[Any] | None = None
    try:
        stdout_handle.write("[collector] process resource sampler stdout follows\n")
        stderr_handle.write("[collector] process resource sampler stderr follows\n")
        stdout_handle.flush()
        stderr_handle.flush()
        process = subprocess.Popen(
            argv,
            env=sanitized_environment(),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
        meta = {
            "process": process,
            "argv": argv,
            "observations": observations,
            "stop_file": stop_file,
            "stdout_path": stdout_path,
            "stderr_path": sampler_stderr_path,
            "stdout_handle": stdout_handle,
            "stderr_handle": stderr_handle,
            "finished": False,
        }
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if observations.exists() and observations.stat().st_size > 0:
                lines = observations.read_text(encoding="utf-8", errors="replace").splitlines()
                if len(lines) >= 2:
                    return meta
            require(process.poll() is None, f"process resource sampler exited during startup with {process.returncode}")
            time.sleep(0.05)
        raise CollectorError("process resource sampler did not produce its first observation")
    except BaseException:
        if process is not None:
            cleanup_process_group_noexcept(process, 5.0)
        for handle in (stdout_handle, stderr_handle):
            try:
                handle.close()
            except BaseException:
                pass
        ensure_nonempty_log(stdout_path, "process resource sampler stdout")
        ensure_nonempty_log(sampler_stderr_path, "process resource sampler stderr")
        raise


def wait_process_sampler(meta: dict[str, Any]) -> list[dict[str, Any]]:
    finish_resource_sampler(meta, bracket_after_measurement=False)
    rows = []
    for line in meta["observations"].read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        require(isinstance(row, dict), "process resource observation row must be an object")
        rows.append(row)
    samples = [row for row in rows if row.get("record_type") == "sample"]
    require(len(samples) >= 3, "cold run ended before three resource samples were collected")
    require(rows[-1].get("record_type") == "footer" and rows[-1].get("exit_reason") == "process-exit", "cold run sampler did not terminate on process exit")
    return samples


def prepare_run_workload(root: Path, config: dict[str, Any], context: dict[str, Any], workloads: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], str]:
    prompt = "Return the word Paris and then stop."
    prompt_path = write_artifact_json(root, f"run/{config['model_key']}/{config['backend']}/prompt.json", {"prompt": prompt})
    tokenizer = workloads[next(iter(sorted(workloads)))]["tokenizer_artifact"]
    tokenizer_path = root / tokenizer
    tokenizer_source, _ = tokenizer_lock(context["model"], config["backend"])
    effective = {
        "schema_version": SCHEMA_VERSION,
        "model_revision": context["lane"]["revision"],
        "prompt_sha256": file_sha256(prompt_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "seed": 9271,
        "max_output_tokens": 128,
        "enable_thinking": False,
        "temperature": 0.0,
        "top_k": 20,
        "top_p": 0.8,
        "repeat_penalty": 1.0,
        "eos_policy": "model-metadata",
        "backend": config["backend"],
        "request_model": config["model_origin_path"],
        "memory_budget_bytes": config["memory_budget_bytes"],
    }
    effective_path = write_artifact_json(root, f"run/{config['model_key']}/{config['backend']}/effective-config.json", effective)
    workload = {
        "prompt_artifact": artifact_relative(root, prompt_path),
        "prompt_sha256": file_sha256(prompt_path),
        "tokenizer_artifact": artifact_relative(root, tokenizer_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "tokenizer_id": tokenizer_source["repo"],
        "tokenizer_revision": tokenizer_source["revision"],
        "tokenizer_origin_path": config["tokenizer_origin_path"],
        "effective_config": artifact_relative(root, effective_path),
        "effective_config_sha256": file_sha256(effective_path),
    }
    return workload, prompt


def load_run_sample_bundle(path: Path, fingerprint: str, sample_id: str) -> dict[str, Any]:
    bundle = read_json(path)
    require(bundle.get("config_fingerprint") == fingerprint, f"run sample fingerprint mismatch: {path}")
    require(bundle.get("sample", {}).get("sample_id") == sample_id, f"run sample identity mismatch: {path}")
    return bundle


def collect_run_sample(
    root: Path,
    lane_dir: Path,
    fingerprint: str,
    slot: int,
    repeat: int,
    config: dict[str, Any],
    context: dict[str, Any],
    workload: dict[str, Any],
    prompt: str,
    resume: bool,
) -> dict[str, Any]:
    sample_id = f"run-{config['model_key']}-{config['backend']}-s{slot}-r{repeat}"
    bundle_path = lane_dir / "run-samples" / f"slot-{slot}-repeat-{repeat}.json"
    if bundle_path.exists():
        require(resume, f"run sample already exists; pass --resume: {bundle_path}")
        return load_run_sample_bundle(bundle_path, fingerprint, sample_id)
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt_dir = lane_dir / "attempts" / f"{sample_id}-{attempt_id}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = attempt_dir / "run.stdout.jsonl"
    stderr_path = attempt_dir / "run.stderr.log"
    product_config = attempt_dir / "product-effective-config.json"
    binary = root / context["binary"]["artifact_binary"]
    binary.chmod(binary.stat().st_mode | 0o100)
    argv = [
        str(binary),
        "run",
        config["model_origin_path"],
        "--prompt",
        prompt,
        "--tokenizer",
        config["tokenizer_origin_path"],
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
        config["backend"],
        "--output-format",
        "jsonl",
        "--effective-config-json",
        str(product_config),
    ]
    barrier_release = attempt_dir / "exec-barrier.release"
    require(not barrier_release.exists(), f"exec barrier release already exists: {barrier_release}")
    launcher_argv = exec_barrier_launcher_argv(barrier_release, argv)
    environment = sanitized_environment(config["legacy"].get("env"))
    started_at = now_iso()
    failure: BaseException | None = None
    process: subprocess.Popen[Any] | None = None
    sampler: dict[str, Any] | None = None
    with stderr_path.open("x", encoding="utf-8") as stderr_handle:
        stderr_handle.write("[collector] frozen Ferrum run stderr follows\n")
        stderr_handle.flush()
        try:
            process = subprocess.Popen(
                launcher_argv,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                text=True,
                start_new_session=True,
            )
            pid = process.pid
            pgid = os.getpgid(pid)
            marker, marker_source = process_identity(pid)
            sampler = run_sampler_for_process(
                root,
                attempt_dir,
                pid=pid,
                pgid=pgid,
                sample_id=sample_id,
                config=config,
                hardware_id=context["hardware"]["id"],
                stderr_path=stderr_path,
            )
            barrier_release.write_text("release\n", encoding="utf-8")
            wait_for_process_exec(process, binary, 30.0)
            process_receipt = write_process_receipt(
                root,
                attempt_dir / "run-process-receipt.json",
                pid=pid,
                pgid=pgid,
                argv=argv,
                environment=environment,
                marker=marker,
                source=marker_source,
            )
            try:
                stdout, _ = process.communicate(timeout=config["server"]["command_timeout_sec"])
                returncode = process.returncode
                _, run_group_gone = terminate_process_group(process, 2.0)
            except subprocess.TimeoutExpired:
                _, run_group_gone = terminate_process_group(process, 5.0)
                stdout, _ = process.communicate(timeout=10)
                returncode = 124
            stdout_path.write_text(stdout, encoding="utf-8")
            sampler_meta = sampler
            samples = wait_process_sampler(sampler_meta)
            sampler = None
            finished_at = now_iso()
            require(run_group_gone, "frozen Ferrum run process group survived cleanup")
            process = None
            require(returncode == 0, f"frozen Ferrum run failed with returncode {returncode}")
            require(stdout_path.is_file() and stdout_path.stat().st_size > 0, "frozen Ferrum run emitted no JSONL")
            require(product_config.is_file(), "frozen Ferrum run emitted no effective config")
            output_tokens, inference_ms, output_tps = baseline_gate.validate_run_stdout(stdout_path, sample_id)
            measurement_started_at = samples[0]["sampled_at"]
            measurement_finished_at = samples[-1]["sampled_at"]
            resource_summary = resource_sampler.derive_summary(
                sampler_meta["observations"],
                session_id=sample_id,
                cell_id="run:c1",
                backend=config["backend"],
                hardware_id=context["hardware"]["id"],
                pid=pid,
                pgid=pgid,
                process_start_marker=marker,
                base_url=f"process://{sample_id}",
                session_started_at=started_at,
                session_finished_at=finished_at,
                measurement_started_at=measurement_started_at,
                measurement_finished_at=measurement_finished_at,
                memory_budget_bytes=config["memory_budget_bytes"],
                requested_concurrency=1,
                typed_active_cap=1,
                runtime_log_path=str(stderr_path),
            )
            resources = {
                "collector_sha256": file_sha256(RESOURCE_SAMPLER_PATH),
                "sampler_argv": sampler_meta["argv"],
                "observation_origin_path": str(sampler_meta["observations"]),
                "observations": artifact_relative(root, sampler_meta["observations"]),
                "observations_sha256": file_sha256(sampler_meta["observations"]),
                "summary": resource_summary,
            }
            sample = {
                "sample_id": sample_id,
                "session_id": f"g00-{config['model_key']}-{config['backend']}-run-session-{slot}",
                "slot": slot,
                "repeat": repeat,
                "binary_sha256": context["binary"]["binary_sha256"],
                "hardware_id": context["hardware"]["id"],
                "hardware_fingerprint": context["hardware"]["fingerprint"],
                "effective_config": workload["effective_config"],
                "effective_config_sha256": workload["effective_config_sha256"],
                "prompt_sha256": workload["prompt_sha256"],
                "tokenizer_sha256": workload["tokenizer_sha256"],
                "argv": argv,
                "launcher_argv": launcher_argv,
                "barrier_release_origin_path": str(barrier_release),
                "barrier_release": artifact_relative(root, barrier_release),
                "barrier_release_sha256": file_sha256(barrier_release),
                "env": environment,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_sec": duration_seconds(started_at, finished_at),
                "measurement_started_at": measurement_started_at,
                "measurement_finished_at": measurement_finished_at,
                "pid": pid,
                "pgid": pgid,
                "process_start_marker": marker,
                "process_start_source": marker_source,
                "process_receipt": process_receipt,
                "returncode": returncode,
                "stdout": artifact_relative(root, stdout_path),
                "stdout_sha256": file_sha256(stdout_path),
                "stderr": artifact_relative(root, stderr_path),
                "stderr_sha256": file_sha256(stderr_path),
                "stderr_origin_path": str(stderr_path),
                "product_effective_config_origin_path": str(product_config),
                "product_effective_config": artifact_relative(root, product_config),
                "product_effective_config_sha256": file_sha256(product_config),
                "output_tokens": output_tokens,
                "legacy_inference_e2e_ms": inference_ms,
                "legacy_inference_e2e_output_tps": output_tps,
                "cold_process_first_request": True,
                "resources": resources,
            }
            bundle = {
                "schema_version": SCHEMA_VERSION,
                "config_fingerprint": fingerprint,
                "sample": sample,
            }
            atomic_write_json(bundle_path, bundle)
            append_jsonl(lane_dir / "command-log.jsonl", {
                "event": "run-sample-complete",
                "sample_id": sample_id,
                "slot": slot,
                "repeat": repeat,
                "started_at": started_at,
                "finished_at": finished_at,
                "bundle": artifact_relative(root, bundle_path),
                "bundle_sha256": file_sha256(bundle_path),
            })
            return bundle
        except BaseException as exc:
            failure = exc
            raise
        finally:
            if process is not None:
                cleanup_process_group_noexcept(process, 10.0)
            if sampler is not None and sampler.get("finished") is not True:
                try:
                    sampler["stop_file"].write_text("stop\n", encoding="utf-8")
                    finish_resource_sampler(sampler, bracket_after_measurement=False)
                except BaseException:
                    cleanup_process_group_noexcept(sampler["process"], 5.0)
                    close_sampler_handles(sampler)
                    sampler["finished"] = True
            ensure_nonempty_log(stderr_path, "frozen Ferrum run stderr")
            if failure is not None:
                atomic_write_json(attempt_dir / "failure.json", {
                    "schema_version": SCHEMA_VERSION,
                    "failed_at": now_iso(),
                    "error_type": type(failure).__name__,
                    "error": str(failure),
                })
    raise CollectorError("unreachable run sample collector state")


def collect_run_legacy(
    root: Path,
    lane_dir: Path,
    fingerprint: str,
    config: dict[str, Any],
    context: dict[str, Any],
    workloads: dict[str, dict[str, Any]],
    resume: bool,
) -> dict[str, Any]:
    workload, prompt = prepare_run_workload(root, config, context, workloads)
    expected_pairs = [
        (slot, repeat)
        for slot, owner in enumerate(SLOT_ORDER, start=1)
        if owner == "B"
        for repeat in range(1, 4)
    ]
    existing_pairs = [
        pair for pair in expected_pairs
        if (lane_dir / "run-samples" / f"slot-{pair[0]}-repeat-{pair[1]}.json").exists()
    ]
    require(existing_pairs == expected_pairs[: len(existing_pairs)], "run-sample resume state is not a chronological prefix")
    sample_bundles: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    for slot in [slot for slot, owner in enumerate(SLOT_ORDER, start=1) if owner == "B"]:
        session_bundle_path = lane_dir / "run-sessions" / f"slot-{slot}.json"
        state_path = lane_dir / "run-sessions" / f"slot-{slot}.state.json"
        if session_bundle_path.exists():
            require(resume, f"run session already exists; pass --resume: {session_bundle_path}")
            session_bundle = read_json(session_bundle_path)
            require(session_bundle.get("config_fingerprint") == fingerprint, "run session fingerprint mismatch")
            session = session_bundle["session"]
            slot_samples = session_bundle["samples"]
            sessions.append(session)
            sample_bundles.extend({"sample": sample} for sample in slot_samples)
            continue
        if state_path.exists():
            require(resume, f"run session state already exists; pass --resume: {state_path}")
            state = read_json(state_path)
            require(state.get("config_fingerprint") == fingerprint, "run session state fingerprint mismatch")
        else:
            state = {
                "schema_version": SCHEMA_VERSION,
                "config_fingerprint": fingerprint,
                "slot": slot,
                "started_at": now_iso(),
            }
            atomic_write_json(state_path, state)
        slot_bundles = [
            collect_run_sample(
                root,
                lane_dir,
                fingerprint,
                slot,
                repeat,
                config,
                context,
                workload,
                prompt,
                resume,
            )
            for repeat in range(1, 4)
        ]
        slot_samples = [bundle["sample"] for bundle in slot_bundles]
        finished_at = now_iso()
        session = {
            "session_id": f"g00-{config['model_key']}-{config['backend']}-run-session-{slot}",
            "slot": slot,
            "sequence": slot,
            "hardware_id": context["hardware"]["id"],
            "hardware_fingerprint": context["hardware"]["fingerprint"],
            "started_at": state["started_at"],
            "measurement_started_at": slot_samples[0]["started_at"],
            "measurement_finished_at": slot_samples[-1]["finished_at"],
            "finished_at": finished_at,
            "duration_sec": duration_seconds(state["started_at"], finished_at),
            "sample_windows": [
                {
                    "repeat": sample["repeat"],
                    "sample_id": sample["sample_id"],
                    "started_at": sample["started_at"],
                    "finished_at": sample["finished_at"],
                }
                for sample in slot_samples
            ],
        }
        session_bundle = {
            "schema_version": SCHEMA_VERSION,
            "config_fingerprint": fingerprint,
            "session": session,
            "samples": slot_samples,
        }
        atomic_write_json(session_bundle_path, session_bundle)
        sessions.append(session)
        sample_bundles.extend(slot_bundles)
    samples = [bundle["sample"] for bundle in sample_bundles]
    samples.sort(key=lambda row: (int(row["slot"]), int(row["repeat"])))
    e2e_ms = [float(sample["legacy_inference_e2e_ms"]) for sample in samples]
    e2e_tps = [float(sample["legacy_inference_e2e_output_tps"]) for sample in samples]
    config_shas = {sample["product_effective_config_sha256"] for sample in samples}
    require(len(config_shas) == 1, "frozen run product effective configs differ across samples")
    return {
        "comparison_id": "g00-run-legacy",
        "workload": workload,
        "sessions": sessions,
        "samples": samples,
        "measured_samples": len(samples),
        "completed_samples": len(samples),
        "error_count": 0,
        "output_token_count_source": "generated_tokens",
        "metric_boundary": "engine.infer_e2e",
        "summary": {
            "legacy_inference_e2e_ms": {
                "median": baseline_gate.percentile_linear(e2e_ms, 0.5),
                "p95": baseline_gate.percentile_linear(e2e_ms, 0.95),
            },
            "legacy_inference_e2e_output_tps": {
                "median": baseline_gate.percentile_linear(e2e_tps, 0.5),
                "p95": baseline_gate.percentile_linear(e2e_tps, 0.95),
            },
        },
    }


def assemble_summaries(
    root: Path,
    lane_dir: Path,
    fingerprint: str,
    config: dict[str, Any],
    context: dict[str, Any],
    client: dict[str, Any],
    identities: dict[str, dict[str, Any]],
    workloads: dict[str, dict[str, Any]],
    bundles: list[dict[str, Any]],
    run_legacy: dict[str, Any] | None,
) -> tuple[Path, Path]:
    bundles = sorted(bundles, key=lambda row: int(row["session"]["slot"]))
    sessions = [copy.deepcopy(bundle["session"]) for bundle in bundles]
    external_sessions = [copy.deepcopy(row) for row in sessions if row["implementation"] == "A"]
    plan_path = lane_dir / "plan.json"
    plan = read_json(plan_path)
    require(plan.get("config_fingerprint") == fingerprint, "collection plan fingerprint changed")
    collection = {
        "collector_path": COLLECTOR_RELATIVE_PATH,
        "collector_sha256": file_sha256(COLLECTOR_PATH),
        "plan": artifact_relative(root, plan_path),
        "plan_sha256": file_sha256(plan_path),
        "config_fingerprint": fingerprint,
    }
    command_log = lane_dir / "command-log.jsonl"
    require(command_log.is_file() and command_log.stat().st_size > 0, "collection command log is empty")
    runtime_index = write_runtime_index(root, lane_dir, bundles)
    external_cells: list[dict[str, Any]] = []
    performance_cells: list[dict[str, Any]] = []
    for dataset, concurrency in expected_cells(config["backend"]):
        cell_id = f"{dataset}:c{concurrency}"
        impl_a = implementation_summary(root, bundles, "A", dataset, concurrency)
        external_cells.append({
            "dataset": dataset,
            "concurrency": concurrency,
            "workload": copy.deepcopy(workloads[cell_id]),
            "implementation": copy.deepcopy(impl_a),
        })
        if context["correctness_status"] == "pass":
            impl_b = implementation_summary(root, bundles, "B", dataset, concurrency)
            a_mean = implementation_input_mean(root, impl_a)
            b_mean = implementation_input_mean(root, impl_b)
            performance_cells.append({
                "dataset": dataset,
                "concurrency": concurrency,
                "tokenizer_input_len_diff_pct": abs(b_mean - a_mean) / a_mean * 100.0,
                "workload": copy.deepcopy(workloads[cell_id]),
                "implementations": {"A": copy.deepcopy(impl_a), "B": impl_b},
            })
    external_summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "model_key": config["model_key"],
        "backend": config["backend"],
        "hardware_id": context["hardware"]["id"],
        "hardware_fingerprint": context["hardware"]["fingerprint"],
        "model_revision": context["lane"]["revision"],
        "model_files": locked_file_map(context["model"], config["backend"]),
        "collection": copy.deepcopy(collection),
        "benchmark_client": copy.deepcopy(client),
        "server_identity": copy.deepcopy(identities["A"]),
        "sessions": external_sessions,
        "command_log": artifact_relative(root, command_log),
        "runtime_log": artifact_relative(root, runtime_index),
        "cells": external_cells,
    }
    external_path = root / "external-baselines" / config["model_key"] / config["backend"] / "summary.json"
    if external_path.exists():
        require(read_json(external_path) == external_summary, "existing external summary differs from collected evidence")
    else:
        atomic_write_json(external_path, external_summary)
    performance = lane_identity_base(config, context)
    performance["collection"] = copy.deepcopy(collection)
    if context["correctness_status"] == "blocked":
        performance.update({
            "status": "blocked",
            "comparable": False,
            "reason": "frozen Ferrum legacy correctness lane is unavailable; external baseline was collected standalone",
            "downstream_goal": BLOCKED_DOWNSTREAM_GOALS[f"{config['model_key']}/{config['backend']}"],
        })
    else:
        require(run_legacy is not None, "comparable performance summary requires run_legacy evidence")
        performance.update({
            "status": "pass",
            "comparable": True,
            "hardware_fingerprint": context["hardware"]["fingerprint"],
            "comparison_id": "g00-legacy-external",
            "slot_order": list(SLOT_ORDER),
            "benchmark_client": copy.deepcopy(client),
            "implementations": {"A": copy.deepcopy(identities["A"]), "B": copy.deepcopy(identities["B"])},
            "sessions": sessions,
            "command_log": artifact_relative(root, command_log),
            "runtime_log": artifact_relative(root, runtime_index),
            "cells": performance_cells,
            "run_legacy": run_legacy,
        })
    performance_path = root / "performance" / config["model_key"] / config["backend"] / "summary.json"
    if performance_path.exists():
        require(read_json(performance_path) == performance, "existing performance summary differs from collected evidence")
    else:
        atomic_write_json(performance_path, performance)
    return external_path, performance_path


def validate_collected_lane(root: Path, config: dict[str, Any], context: dict[str, Any]) -> None:
    hardware = {row["id"]: row for row in context["models_lock"]["hardware"]}
    binaries = {row["backend"]: row["binary_sha256"] for row in context["legacy"]["binaries"]}
    baseline_gate.validate_performance_lane(
        root,
        model=context["model"],
        backend=config["backend"],
        correctness_status=context["correctness_status"],
        hardware=hardware,
        binaries=binaries,
        allow_synthetic=False,
    )


def collect_lane(root: Path, config_path: Path, *, resume: bool, plan_only: bool) -> tuple[Path, Path] | None:
    require(root.is_dir(), f"artifact root does not exist: {root}")
    try:
        root.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise CollectorError("artifact root must stay outside the Git worktree")
    raw = read_json(config_path)
    context = load_context(root, raw)
    config = normalize_config(root, raw, context)
    lane_dir, fingerprint = prepare_plan(root, config, context, resume)
    if plan_only:
        print(f"FERRUM RUNTIME VNEXT PERFORMANCE COLLECTOR PLAN: {lane_dir / 'plan.json'}")
        return None
    client = prepare_benchmark_client(root, config)
    identities = prepare_identities(root, lane_dir, config, context)
    workloads = prepare_workloads(root, config, context)
    required_slots = [
        (slot, owner)
        for slot, owner in enumerate(SLOT_ORDER, start=1)
        if context["correctness_status"] == "pass" or owner == "A"
    ]
    existing_slots = [
        pair for pair in required_slots
        if (lane_dir / "sessions" / f"slot-{pair[0]}.json").exists()
    ]
    require(existing_slots == required_slots[: len(existing_slots)], "server-session resume state is not a chronological prefix")
    bundles = [
        collect_server_session(
            root,
            lane_dir,
            fingerprint,
            slot,
            implementation,
            config,
            context,
            client,
            identities[implementation],
            workloads,
            resume,
        )
        for slot, implementation in required_slots
    ]
    run_legacy = None
    if context["correctness_status"] == "pass":
        run_legacy = collect_run_legacy(root, lane_dir, fingerprint, config, context, workloads, resume)
    paths = assemble_summaries(
        root,
        lane_dir,
        fingerprint,
        config,
        context,
        client,
        identities,
        workloads,
        bundles,
        run_legacy,
    )
    validate_collected_lane(root, config, context)
    print(f"{PASS_PREFIX}: {config['model_key']}/{config['backend']}: {paths[1]}")
    return paths


def self_test() -> int:
    rendered = render_argv(
        ["/bin/vllm", "serve", "{model_origin_path}", "--served-model-name", "{request_model}", "--port", "{port}"],
        {"model_origin_path": "/models/qwen", "request_model": "served-qwen", "port": "8080"},
    )
    require(rendered[-1] == "8080" and "served-qwen" in rendered, "command template self-test failed")
    reject_secret_material(
        {"tokenizer_origin_path": "/models/tokenizer", "argv": ["--tokenizer"]}
    )
    try:
        reject_secret_material({"api_token": "forbidden"})
        raise CollectorError("secret-bearing config unexpectedly passed")
    except CollectorError as exc:
        require("secret-bearing" in str(exc), "secret rejection self-test failed unexpectedly")
    model = {
        "key": "m3-qwen3-30b-a3b",
        "official_model_id": "Qwen/Qwen3-30B-A3B",
        "lanes": {
            "metal": {
                "repo": "example/gguf",
                "revision": "1" * 40,
                "format": "gguf_q4_k_m",
                "files": [{"path": "model.gguf", "sha256": "2" * 64}],
            }
        },
    }
    normalized = normalized_server_config(
        {"model": model, "lane": model["lanes"]["metal"]},
        {
            "backend": "metal",
            "request_model": "model",
            "model_origin_path": "/models/model.gguf",
            "typed_active_cap": 16,
            "memory_budget_bytes": 1024,
        },
    )
    require(normalized["official_model_id"] == model["official_model_id"], "official model identity self-test failed")
    require(normalized["request_model"] == "model", "runtime request alias self-test failed")
    with tempfile.TemporaryDirectory(prefix="runtime-vnext-performance-selftest-") as temporary:
        root = Path(temporary)
        lane_dir = root / "collection" / "selftest"
        vllm_binary = root / "vllm"
        vllm_binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        vllm_binary.chmod(0o755)
        vllm_config = {
            "model_origin_path": "/models/qwen",
            "request_model": "served-qwen",
            "typed_active_cap": 32,
            "memory_budget_bytes": 1024,
            "server": {"host": "127.0.0.1", "port": 8080},
            "external": {
                "engine": "vllm",
                "binary_path": str(vllm_binary),
                "server_argv": [
                    str(vllm_binary),
                    "serve",
                    "{model_origin_path}",
                    "--served-model-name",
                    "{request_model}",
                    "--host",
                    "{host}",
                    "--port",
                    "{port}",
                    "--max-num-seqs",
                    "{typed_active_cap}",
                ],
                "version_argv": [str(vllm_binary), "--version"],
                "revision_argv": [str(vllm_binary), "--revision"],
            },
        }
        require(
            validate_external_server_command(vllm_config)[2] == "/models/qwen",
            "vllm command contract self-test failed",
        )
        invalid_vllm_config = copy.deepcopy(vllm_config)
        invalid_vllm_config["external"]["server_argv"] = [
            str(vllm_binary),
            "serve",
            "--model",
            "{model_origin_path}",
            "--served-model-name",
            "{request_model}",
            "--host",
            "{host}",
            "--port",
            "{port}",
            "--max-num-seqs",
            "{typed_active_cap}",
        ]
        try:
            validate_external_server_command(invalid_vllm_config)
            raise CollectorError("legacy vllm --model command unexpectedly passed")
        except CollectorError as exc:
            require("positional model" in str(exc), "invalid vllm command failed for the wrong reason")
        llama_binary = root / "llama-server"
        llama_binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        llama_binary.chmod(0o755)
        llama_config = copy.deepcopy(vllm_config)
        llama_config["external"] = {
            "engine": "llama.cpp",
            "binary_path": str(llama_binary),
            "server_argv": [
                str(llama_binary),
                "--model",
                "{model_origin_path}",
                "--alias",
                "{request_model}",
                "--host",
                "{host}",
                "--port",
                "{port}",
                "--parallel",
                "{typed_active_cap}",
            ],
            "version_argv": [str(llama_binary), "--version"],
            "revision_argv": [str(llama_binary), "--revision"],
        }
        require(
            "--parallel" in validate_external_server_command(llama_config),
            "llama.cpp command contract self-test failed",
        )
        probe_argv = [sys.executable, "-c", "print('collector-version-1')"]
        probe_env = sanitized_environment()
        first_probe = run_identity_probe(root, lane_dir, "selftest-version", probe_argv, "collector-version-1", probe_env)
        second_probe = run_identity_probe(root, lane_dir, "selftest-version", probe_argv, "collector-version-1", probe_env)
        require(first_probe == second_probe, "identity probe resume rewrote immutable evidence")
        require((lane_dir / "identity" / "selftest-version.json").is_file(), "identity probe receipt self-test failed")

        release_file = root / "exec-barrier.release"
        barrier_argv = exec_barrier_launcher_argv(release_file, ["/bin/sleep", "1"])
        require(
            barrier_argv[-2:] == ["/bin/sleep", "1"] and "--exec-barrier-child" in barrier_argv,
            "exec barrier launcher argv self-test failed",
        )
        barrier_process = subprocess.Popen(
            barrier_argv,
            env=probe_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            barrier_marker, _ = process_identity(barrier_process.pid)
            time.sleep(0.05)
            require(barrier_process.poll() is None, "exec barrier child did not wait for release")
            release_file.write_text("release\n", encoding="utf-8")
            wait_for_process_exec(barrier_process, Path("/bin/sleep"), 5.0)
            exec_marker, _ = process_identity(barrier_process.pid)
            require(exec_marker == barrier_marker, "exec barrier changed the product PID identity")
            barrier_stdout, barrier_stderr = barrier_process.communicate(timeout=5.0)
            require(barrier_process.returncode == 0, f"exec barrier product failed: {barrier_stderr}")
            require(barrier_stdout == "", "exec barrier product emitted unexpected stdout")
            _, barrier_group_gone = terminate_process_group(barrier_process, 2.0)
            require(barrier_group_gone, "exec barrier product process group survived completion")
        finally:
            if process_group_exists(barrier_process.pid):
                cleanup_process_group_noexcept(barrier_process, 2.0)

        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); time.sleep(60)",
            ],
            start_new_session=True,
        )
        try:
            _, group_gone = terminate_process_group(child, 2.0)
            require(group_gone, "process-group cleanup self-test left a descendant alive")
        finally:
            if process_group_exists(child.pid):
                signal_process_group(child.pid, signal.SIGKILL)
                wait_process_group_gone(child.pid, 5.0)

        path = root / "resource.jsonl"
        pid = os.getpid()
        marker, source = process_identity(pid)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows: list[dict[str, Any]] = [{
            "record_type": "header",
            "schema_version": resource_sampler.SCHEMA_VERSION,
            "collector_path": resource_sampler.COLLECTOR_RELATIVE_PATH,
            "collector_sha256": file_sha256(RESOURCE_SAMPLER_PATH),
            "session_id": "selftest-run",
            "cell_id": "run:c1",
            "backend": "metal",
            "hardware_id": "selftest-metal",
            "pid": pid,
            "pgid": os.getpgid(pid),
            "process_start_marker": marker,
            "process_start_source": source,
            "base_url": "process://selftest-run",
            "started_at": (start + timedelta(seconds=1)).isoformat().replace("+00:00", "Z"),
            "interval_ms": 250,
            "runtime_log_path": "/tmp/selftest-runtime.log",
            "active_probe": {
                "format": "process",
                "path": "",
                "url": "",
                "selector": "process-alive",
                "semantics": "process-alive",
            },
        }]
        for sequence in range(4):
            rows.append({
                "record_type": "sample",
                "sequence": sequence,
                "sampled_at": (start + timedelta(seconds=2 + sequence)).isoformat().replace("+00:00", "Z"),
                "pid": pid,
                "pgid": os.getpgid(pid),
                "process_start_marker": marker,
                "process_alive": True,
                "process_rss_bytes": 1024,
                "memory_used_bytes": 1024,
                "physical_headroom_bytes": 4 * 1024**3,
                "swap_used_bytes": 0,
                "active_requests": 1,
                "oom_count": 0,
                "admission_error_count": 0,
                "thermal_state": "nominal",
                "power_mode": "normal",
            })
        rows.append({
            "record_type": "footer",
            "finished_at": (start + timedelta(seconds=6)).isoformat().replace("+00:00", "Z"),
            "sample_count": 4,
            "exit_reason": "process-exit",
        })
        path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
        summary = resource_sampler.derive_summary(
            path,
            session_id="selftest-run",
            cell_id="run:c1",
            backend="metal",
            hardware_id="selftest-metal",
            pid=pid,
            pgid=os.getpgid(pid),
            process_start_marker=marker,
            base_url="process://selftest-run",
            session_started_at=start.isoformat().replace("+00:00", "Z"),
            session_finished_at=(start + timedelta(seconds=7)).isoformat().replace("+00:00", "Z"),
            measurement_started_at=(start + timedelta(seconds=2)).isoformat().replace("+00:00", "Z"),
            measurement_finished_at=(start + timedelta(seconds=5)).isoformat().replace("+00:00", "Z"),
            memory_budget_bytes=2048,
            requested_concurrency=1,
            typed_active_cap=1,
            runtime_log_path="/tmp/selftest-runtime.log",
        )
        require(
            summary["exit_reason"] == "process-exit"
            and summary["sample_count"] == 4
            and summary["observed_max_active"] == 1,
            "process-exit resource self-test failed",
        )
    print(SELFTEST_PASS_LINE)
    return 0


def run_exec_barrier_child(release_file: Path, raw_argv: list[str]) -> int:
    exec_argv = list(raw_argv)
    if exec_argv and exec_argv[0] == "--":
        exec_argv.pop(0)
    require(exec_argv and all(isinstance(item, str) and item for item in exec_argv), "exec barrier requires product argv after --")
    deadline = time.monotonic() + 120.0
    while not release_file.exists():
        if time.monotonic() >= deadline:
            raise CollectorError(f"exec barrier release timed out: {release_file}")
        time.sleep(0.01)
    require(release_file.read_text(encoding="utf-8") == "release\n", "exec barrier release artifact is invalid")
    os.execvpe(exec_argv[0], exec_argv, os.environ)
    raise CollectorError("exec barrier os.execvpe unexpectedly returned")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--exec-barrier-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--release-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("exec_argv", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.exec_barrier_child:
        require(args.release_file is not None, "--exec-barrier-child requires --release-file")
        return run_exec_barrier_child(args.release_file, args.exec_argv)
    if args.self_test:
        require(args.artifact_root is None and args.config is None, "--self-test cannot collect a lane")
        return self_test()
    require(args.artifact_root is not None and args.config is not None, "--artifact-root and --config are required")
    collect_lane(args.artifact_root.expanduser().resolve(), args.config.expanduser().resolve(), resume=args.resume, plan_only=args.plan_only)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CollectorError, baseline_gate.BaselineError, resource_sampler.ResourceEvidenceError, OSError, subprocess.SubprocessError) as exc:
        print(f"runtime vNext performance collector failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
