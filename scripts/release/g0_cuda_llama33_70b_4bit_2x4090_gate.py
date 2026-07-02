#!/usr/bin/env python3
"""CUDA source gate for Llama 3.3 70B 4bit on 2x RTX 4090."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openai_concurrency_quality_regression import run_concurrency_quality_regression
from openai_tool_call_regression import run_tool_call_regression


LANE = "g0_cuda2x4090_llama33_70b_4bit"
PASS_LINE_PREFIX = f"G0 SOURCE {LANE} PASS"
SMOKE_LANE = "g0_cuda2x4090_llama33_70b_4bit_smoke"
QWEN72B_LANE = "layer_split_perf_qwen72b_gptq"
QWEN72B_SMOKE_LANE = "layer_split_perf_qwen72b_gptq_smoke"
REQUIRED_GPU_DEVICES = [0, 1]
BAD_PATTERNS = [
    "panic",
    "CUDA illegal memory access",
    "NCCL error",
    "OOM",
    "KV cache overflow",
    "missing tokenizer",
    "chat template render failure",
    "CPU fallback",
    "single-GPU fallback",
    "<unk>",
    "[PAD]",
    "<|reserved_special_token_",
    "<|assistant|>",
    "<|tool|>",
]
DEFAULT_LAYER_SPLIT_PLAN = "stage0:cuda:0:layers=auto;stage1:cuda:1:layers=auto"
RECALL_MARKER = "banana"
MODEL_MANIFEST_INTERESTING_SUFFIXES = {".json", ".model", ".safetensors", ".gguf"}
TOKENIZER_METADATA_FILE_NAMES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.json",
}
GPU_QUERY_FIELDS = [
    "index",
    "name",
    "uuid",
    "driver_version",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "utilization.memory",
    "pcie.link.gen.current",
    "pcie.link.width.current",
]
BENCH_QUALITY_COUNT_FIELDS = (
    "bad_output_per_run",
    "malformed_stream_per_run",
    "missing_done_per_run",
    "duplicate_done_per_run",
    "zero_output_tokens_per_run",
    "stream_bulk_flush_per_run",
    "http_500_per_run",
    "panic_per_run",
)
REQUIRED_ARTIFACT_FILES = {
    "gate.json",
    "metadata.json",
    "effective_config.json",
    "decision_trace.jsonl",
    "hardware.json",
    "nvidia-smi.before.txt",
    "nvidia-smi.before.json",
    "nvidia-smi.during.txt",
    "nvidia-smi.during.json",
    "nvidia-smi.after.txt",
    "nvidia-smi.after.json",
    "nvidia-smi.bench.samples.jsonl",
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
    "serve.health.after.json",
    "serve.models.json",
    "serve.correctness.json",
    "serve.multiturn.json",
    "serve.structured_output.json",
    "serve.tool_call.json",
    "serve.streaming.sse",
    "correctness.json",
    "concurrency_quality_regression.json",
    "bench-serve.command.json",
    "bench-serve.json",
    "bench-serve.stdout",
    "bench-serve.stderr",
    "vllm-baseline.command.json",
    "vllm-baseline.json",
    "comparison.json",
}


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def run(
    cmd: list[str],
    *,
    cwd: Path,
    input_text: str | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(cmd, 127, "", str(exc))


def post_json(base_url: str, payload: dict[str, Any], timeout: int = 180) -> tuple[int, str]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def get_url(url: str, timeout: int = 30) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def wait_health(base_url: str, timeout_sec: int = 1200) -> None:
    deadline = time.time() + timeout_sec
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=3) as response:
                if response.status == 200:
                    return
                last = f"status={response.status}"
        except Exception as exc:
            last = str(exc)
            time.sleep(2)
    raise RuntimeError(f"server did not become healthy within {timeout_sec}s: {last}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def query_vllm_preflight(server_cmd: list[str]) -> dict[str, Any]:
    executable = server_cmd[0] if server_cmd else "vllm"
    resolved = shutil.which(executable)
    if resolved is None:
        return {
            "schema_version": 1,
            "status": "fail",
            "cmd": [executable],
            "error": "vllm executable not found on PATH",
            "binary_path": None,
            "binary_sha256": None,
        }
    digest = sha256(Path(resolved))
    if digest is None:
        return {
            "schema_version": 1,
            "status": "fail",
            "cmd": [executable],
            "error": f"vllm executable is not a readable file: {resolved}",
            "binary_path": resolved,
            "binary_sha256": None,
        }
    return {
        "schema_version": 1,
        "status": "pass",
        "cmd": [executable],
        "binary_path": resolved,
        "binary_sha256": digest,
    }


def require_vllm_preflight(root: Path, server_cmd: list[str]) -> dict[str, Any]:
    preflight = query_vllm_preflight(server_cmd)
    write_json(root / "vllm-baseline.preflight.json", preflight)
    if preflight.get("status") != "pass":
        raise RuntimeError(preflight.get("error") or "vllm preflight failed")
    return preflight


def git_output(args: list[str], repo: Path) -> str:
    proc = run(["git", *args], cwd=repo, timeout=30)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def sanitized_env_summary() -> dict[str, str]:
    out: dict[str, str] = {}
    secret_fragments = ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL")
    allowed_exact = {"CUDA_VISIBLE_DEVICES", "HF_HOME", "LD_LIBRARY_PATH", "RUST_LOG"}
    for key, value in sorted(os.environ.items()):
        if not (key.startswith("FERRUM_") or key in allowed_exact):
            continue
        if any(fragment in key.upper() for fragment in secret_fragments):
            out[key] = "<redacted>"
        elif len(value) > 512:
            out[key] = f"{value[:512]}...<truncated>"
        else:
            out[key] = value
    return out


def parse_gpu_query_int(value: str) -> int | str:
    text = value.strip()
    try:
        return int(text)
    except ValueError:
        return text


def query_gpu_snapshot(cwd: Path) -> dict[str, Any]:
    proc = run(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(GPU_QUERY_FIELDS),
            "--format=csv,noheader,nounits",
        ],
        cwd=cwd,
        timeout=30,
    )
    if proc.returncode != 0:
        return {
            "schema_version": 1,
            "status": "fail",
            "error": proc.stderr.strip() or proc.stdout.strip() or f"nvidia-smi rc={proc.returncode}",
            "query_fields": GPU_QUERY_FIELDS,
            "gpus": [],
        }

    rows: list[dict[str, Any]] = []
    reader = csv.reader(proc.stdout.splitlines())
    for row in reader:
        parts = [part.strip() for part in row]
        if len(parts) < len(GPU_QUERY_FIELDS):
            continue
        (
            index,
            name,
            uuid,
            driver_version,
            memory_total,
            memory_used,
            utilization_gpu,
            utilization_memory,
            pcie_link_gen,
            pcie_link_width,
        ) = parts[: len(GPU_QUERY_FIELDS)]
        rows.append(
            {
                "index": parse_gpu_query_int(index),
                "name": name,
                "uuid": uuid,
                "driver_version": driver_version,
                "memory_total_mib": parse_gpu_query_int(memory_total),
                "memory_used_mib": parse_gpu_query_int(memory_used),
                "utilization_gpu_percent": parse_gpu_query_int(utilization_gpu),
                "utilization_memory_percent": parse_gpu_query_int(utilization_memory),
                "pcie_link_gen_current": parse_gpu_query_int(pcie_link_gen),
                "pcie_link_width_current": parse_gpu_query_int(pcie_link_width),
            }
        )
    return {
        "schema_version": 1,
        "status": "pass" if rows else "fail",
        "query_fields": GPU_QUERY_FIELDS,
        "gpus": rows,
    }


def capture_nvidia_smi(root: Path, label: str) -> str:
    proc = run(["nvidia-smi"], cwd=root, timeout=30)
    body = proc.stdout if proc.returncode == 0 else proc.stderr
    text = body or f"nvidia-smi rc={proc.returncode}\n"
    write_text(root / f"nvidia-smi.{label}.txt", text)
    write_json(root / f"nvidia-smi.{label}.json", query_gpu_snapshot(root))
    return text


def write_gpu_bench_sample(
    root: Path,
    sample: dict[str, Any],
    *,
    samples_name: str = "nvidia-smi.bench.samples.jsonl",
    during_label: str = "during",
) -> None:
    samples_path = root / samples_name
    with samples_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sample, sort_keys=True) + "\n")
    during_text = root / f"nvidia-smi.{during_label}.txt"
    if not during_text.is_file():
        proc = run(["nvidia-smi"], cwd=root, timeout=30)
        body = proc.stdout if proc.returncode == 0 else proc.stderr
        write_text(during_text, body or f"nvidia-smi rc={proc.returncode}\n")
    write_json(root / f"nvidia-smi.{during_label}.json", sample)


def flag_value(cmd: list[str], flag: str) -> str | None:
    prefix = flag + "="
    for idx, part in enumerate(cmd):
        if part == flag and idx + 1 < len(cmd):
            return cmd[idx + 1]
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def concurrency_sweep_from_cmd(cmd: list[str]) -> list[int]:
    value = flag_value(cmd, "--concurrency-sweep")
    if not value:
        value = flag_value(cmd, "--concurrency")
    if not value:
        return []
    cells: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            cells.append(int(part))
        except ValueError:
            continue
    return cells


def active_bench_concurrency(stderr_path: Path) -> int | None:
    if not stderr_path.is_file():
        return None
    text = stderr_path.read_text(errors="replace")[-20000:]
    matches = re.findall(r"closed_loop c=(\d+)", text)
    if not matches:
        return None
    return int(matches[-1])


def run_with_gpu_samples(
    cmd: list[str],
    *,
    cwd: Path,
    root: Path,
    timeout: int,
    sample_interval_sec: int,
    samples_name: str = "nvidia-smi.bench.samples.jsonl",
    tmp_stem: str = "bench-serve",
    sample_phase: str = "bench",
    during_label: str = "during",
) -> subprocess.CompletedProcess[str]:
    start = time.time()
    sample_interval_sec = max(1, sample_interval_sec)
    stdout_tmp = root / f"{tmp_stem}.stdout.tmp"
    stderr_tmp = root / f"{tmp_stem}.stderr.tmp"
    next_sample_at = start
    concurrency_cells = concurrency_sweep_from_cmd(cmd)
    with stdout_tmp.open("w", encoding="utf-8") as stdout_file, stderr_tmp.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                text=True,
                stdout=stdout_file,
                stderr=stderr_file,
            )
        except FileNotFoundError as exc:
            sample = query_gpu_snapshot(cwd)
            sample["created_at"] = iso_now()
            sample["elapsed_sec"] = 0.0
            sample["sample_phase"] = f"{sample_phase}-start-failed"
            sample["bench_concurrency_sweep"] = concurrency_cells
            write_gpu_bench_sample(
                root,
                sample,
                samples_name=samples_name,
                during_label=during_label,
            )
            return subprocess.CompletedProcess(cmd, 127, "", str(exc))

        timed_out = False
        while proc.poll() is None:
            now = time.time()
            if now >= next_sample_at:
                sample = query_gpu_snapshot(cwd)
                sample["created_at"] = iso_now()
                sample["elapsed_sec"] = round(now - start, 3)
                sample["sample_phase"] = sample_phase
                sample["bench_concurrency_sweep"] = concurrency_cells
                current_concurrency = active_bench_concurrency(stderr_tmp)
                if current_concurrency is not None:
                    sample["bench_concurrency"] = current_concurrency
                write_gpu_bench_sample(
                    root,
                    sample,
                    samples_name=samples_name,
                    during_label=during_label,
                )
                next_sample_at = now + sample_interval_sec
            if now - start > timeout:
                timed_out = True
                proc.kill()
                proc.wait(timeout=15)
                break
            time.sleep(1)

        if not (root / samples_name).is_file():
            sample = query_gpu_snapshot(cwd)
            sample["created_at"] = iso_now()
            sample["elapsed_sec"] = round(time.time() - start, 3)
            sample["sample_phase"] = f"{sample_phase}-finished-before-first-sample"
            sample["bench_concurrency_sweep"] = concurrency_cells
            current_concurrency = active_bench_concurrency(stderr_tmp)
            if current_concurrency is not None:
                sample["bench_concurrency"] = current_concurrency
            write_gpu_bench_sample(
                root,
                sample,
                samples_name=samples_name,
                during_label=during_label,
            )

    stdout = stdout_tmp.read_text(errors="replace") if stdout_tmp.is_file() else ""
    stderr = stderr_tmp.read_text(errors="replace") if stderr_tmp.is_file() else ""
    stdout_tmp.unlink(missing_ok=True)
    stderr_tmp.unlink(missing_ok=True)
    if timed_out:
        stderr = (stderr + "\n" if stderr else "") + f"bench command timed out after {timeout}s"
        return subprocess.CompletedProcess(cmd, 124, stdout, stderr)
    return subprocess.CompletedProcess(cmd, proc.returncode or 0, stdout, stderr)


def parse_nvidia_smi_versions(text: str) -> dict[str, str]:
    driver = "unknown"
    cuda = "unknown"
    driver_match = re.search(r"Driver Version:\s*([^\s|]+)", text)
    cuda_match = re.search(r"CUDA Version:\s*([^\s|]+)", text)
    if driver_match:
        driver = driver_match.group(1)
    if cuda_match:
        cuda = cuda_match.group(1)
    return {"driver_version": driver, "cuda_version": cuda}


def query_hardware(repo: Path, nvidia_smi_text: str) -> dict[str, Any]:
    versions = parse_nvidia_smi_versions(nvidia_smi_text)
    snapshot = query_gpu_snapshot(repo)
    if snapshot.get("status") != "pass":
        return {
            "schema_version": 1,
            "status": "unavailable",
            "error": snapshot.get("error", "structured nvidia-smi query failed"),
            "cuda_device_count": 0,
            "cuda_version": versions["cuda_version"],
            "driver_version": versions["driver_version"],
            "gpu_names": [],
            "gpu_uuids": [],
            "gpus": [],
        }

    rows = snapshot["gpus"]
    driver_version = rows[0]["driver_version"] if rows else versions["driver_version"]
    return {
        "schema_version": 1,
        "status": "pass" if len(rows) == 2 else "fail",
        "cuda_device_count": len(rows),
        "cuda_version": versions["cuda_version"],
        "driver_version": driver_version,
        "gpu_names": [str(row["name"]) for row in rows],
        "gpu_uuids": [str(row["uuid"]) for row in rows],
        "gpu_utilization_percent": [row.get("utilization_gpu_percent") for row in rows],
        "gpu_memory_utilization_percent": [
            row.get("utilization_memory_percent") for row in rows
        ],
        "pcie_link_gen_current": [row.get("pcie_link_gen_current") for row in rows],
        "pcie_link_width_current": [row.get("pcie_link_width_current") for row in rows],
        "gpus": rows,
    }


def write_hardware(root: Path, repo: Path, nvidia_smi_text: str) -> dict[str, Any]:
    hardware = query_hardware(repo, nvidia_smi_text)
    write_json(root / "hardware.json", hardware)
    return hardware


def validate_config(cfg: dict[str, Any]) -> None:
    if cfg.get("gpu_devices") != REQUIRED_GPU_DEVICES:
        raise RuntimeError(f"config gpu_devices must be {REQUIRED_GPU_DEVICES}")
    if cfg.get("distributed_strategy") != "layer_split":
        raise RuntimeError("config distributed_strategy must be layer_split")
    configured_pipeline_mode_from_config(cfg)
    expected_pipeline_mode_from_config(cfg)
    model = cfg.get("model")
    if not isinstance(model, str) or not model:
        raise RuntimeError("config model must be a non-empty model id/path")
    quant = str(cfg.get("quant_format", "")).lower()
    if not any(marker in quant for marker in ("gptq", "awq", "q4")):
        raise RuntimeError("config quant_format must identify a 4bit format")
    scheduler_prefill = cfg.get("scheduler_prefill_first_until_active")
    if scheduler_prefill is not None and (
        not isinstance(scheduler_prefill, int) or scheduler_prefill <= 0
    ):
        raise RuntimeError("config scheduler_prefill_first_until_active must be a positive int")
    for key in [
        "run_max_model_len",
        "run_kv_capacity",
        "run_max_num_seqs",
        "run_max_num_batched_tokens",
    ]:
        value = cfg.get(key)
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise RuntimeError(f"config {key} must be a positive int")


def require_structured_hardware_snapshot(snapshot: dict[str, Any], label: str) -> None:
    if snapshot.get("status") != "pass":
        raise RuntimeError(f"{label}: structured GPU snapshot status must be pass")
    gpus = snapshot.get("gpus")
    if not isinstance(gpus, list) or len(gpus) != 2:
        raise RuntimeError(f"{label}: structured GPU snapshot must contain exactly two GPUs")
    for idx, gpu in enumerate(gpus):
        if not isinstance(gpu, dict):
            raise RuntimeError(f"{label}: GPU snapshot row {idx} must be an object")
        for key in [
            "memory_total_mib",
            "memory_used_mib",
            "utilization_gpu_percent",
            "utilization_memory_percent",
            "pcie_link_gen_current",
            "pcie_link_width_current",
        ]:
            if not isinstance(gpu.get(key), int) or gpu[key] < 0:
                raise RuntimeError(f"{label}: GPU {idx} missing non-negative integer {key}")
        if gpu["memory_total_mib"] <= 0:
            raise RuntimeError(f"{label}: GPU {idx} memory_total_mib must be > 0")
        if gpu["pcie_link_gen_current"] <= 0:
            raise RuntimeError(f"{label}: GPU {idx} pcie_link_gen_current must be > 0")
        if gpu["pcie_link_width_current"] <= 0:
            raise RuntimeError(f"{label}: GPU {idx} pcie_link_width_current must be > 0")


def validate_structured_hardware_evidence(
    root: Path,
    hardware: dict[str, Any],
    expected_concurrency_cells: list[int] | None = None,
) -> dict[str, Any]:
    if hardware.get("status") != "pass":
        raise RuntimeError("hardware.json status must be pass")
    gpus = hardware.get("gpus")
    if not isinstance(gpus, list) or len(gpus) != 2:
        raise RuntimeError("hardware.json must contain exactly two GPU rows")
    for idx, gpu in enumerate(gpus):
        if not isinstance(gpu, dict):
            raise RuntimeError(f"hardware.json GPU row {idx} must be an object")
        for key in [
            "memory_total_mib",
            "memory_used_mib",
            "utilization_gpu_percent",
            "utilization_memory_percent",
            "pcie_link_gen_current",
            "pcie_link_width_current",
        ]:
            if not isinstance(gpu.get(key), int) or gpu[key] < 0:
                raise RuntimeError(f"hardware.json GPU {idx} missing non-negative integer {key}")
        if gpu["memory_total_mib"] <= 0:
            raise RuntimeError(f"hardware.json GPU {idx} memory_total_mib must be > 0")
        if gpu["pcie_link_gen_current"] <= 0:
            raise RuntimeError(f"hardware.json GPU {idx} pcie_link_gen_current must be > 0")
        if gpu["pcie_link_width_current"] <= 0:
            raise RuntimeError(f"hardware.json GPU {idx} pcie_link_width_current must be > 0")
    snapshots = {}
    for label in ["before", "during", "after"]:
        snapshot = load_json(root / f"nvidia-smi.{label}.json")
        require_structured_hardware_snapshot(snapshot, f"nvidia-smi.{label}.json")
        snapshots[label] = {
            "gpu_utilization_percent": [
                gpu["utilization_gpu_percent"] for gpu in snapshot["gpus"]
            ],
            "memory_used_mib": [gpu["memory_used_mib"] for gpu in snapshot["gpus"]],
            "pcie_link_width_current": [
                gpu["pcie_link_width_current"] for gpu in snapshot["gpus"]
            ],
        }
    samples_path = root / "nvidia-smi.bench.samples.jsonl"
    if not samples_path.is_file():
        raise RuntimeError("missing nvidia-smi.bench.samples.jsonl")
    sample_count = 0
    max_gpu_utilization = [0, 0]
    expected_cells = set(expected_concurrency_cells or [])
    max_gpu_utilization_by_concurrency = {
        cell: [0, 0] for cell in sorted(expected_cells)
    }
    sample_count_by_concurrency = {cell: 0 for cell in sorted(expected_cells)}
    for line_no, line in enumerate(samples_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid nvidia-smi.bench.samples.jsonl line {line_no}: {exc}"
            ) from exc
        if not isinstance(sample, dict):
            raise RuntimeError(f"nvidia-smi.bench.samples.jsonl line {line_no} must be an object")
        if sample.get("status") != "pass":
            continue
        require_structured_hardware_snapshot(
            sample, f"nvidia-smi.bench.samples.jsonl line {line_no}"
        )
        sample_count += 1
        sample_concurrency = sample.get("bench_concurrency")
        if isinstance(sample_concurrency, str) and sample_concurrency.isdigit():
            sample_concurrency = int(sample_concurrency)
        if expected_cells and sample_concurrency in expected_cells:
            sample_count_by_concurrency[int(sample_concurrency)] += 1
        for idx, gpu in enumerate(sample["gpus"]):
            max_gpu_utilization[idx] = max(
                max_gpu_utilization[idx], gpu["utilization_gpu_percent"]
            )
            if expected_cells and sample_concurrency in expected_cells:
                max_gpu_utilization_by_concurrency[int(sample_concurrency)][idx] = max(
                    max_gpu_utilization_by_concurrency[int(sample_concurrency)][idx],
                    gpu["utilization_gpu_percent"],
                )
    if sample_count <= 0:
        raise RuntimeError("missing passing bench-period GPU samples")
    if any(value <= 0 for value in max_gpu_utilization):
        raise RuntimeError("bench-period GPU samples must show non-zero utilization on both GPUs")
    if expected_cells:
        missing_cells = [
            cell for cell, count in sorted(sample_count_by_concurrency.items()) if count <= 0
        ]
        if missing_cells:
            raise RuntimeError(
                "bench-period GPU samples missing concurrency cells "
                + ",".join(str(cell) for cell in missing_cells)
            )
        zero_cells = [
            cell
            for cell, values in sorted(max_gpu_utilization_by_concurrency.items())
            if any(value <= 0 for value in values)
        ]
        if zero_cells:
            raise RuntimeError(
                "bench-period GPU samples must show non-zero utilization on both GPUs "
                "for concurrency cells "
                + ",".join(str(cell) for cell in zero_cells)
            )
    return {
        "status": "pass",
        "pcie_link_width_current": hardware.get("pcie_link_width_current"),
        "pcie_link_gen_current": hardware.get("pcie_link_gen_current"),
        "gpu_utilization_percent": hardware.get("gpu_utilization_percent"),
        "gpu_memory_utilization_percent": hardware.get("gpu_memory_utilization_percent"),
        "snapshots": snapshots,
        "bench_sample_count": sample_count,
        "bench_max_gpu_utilization_percent": max_gpu_utilization,
        "bench_sample_count_by_concurrency": sample_count_by_concurrency,
        "bench_max_gpu_utilization_percent_by_concurrency": (
            max_gpu_utilization_by_concurrency
        ),
    }


def parse_pipeline_mode(value: Any, *, label: str) -> str:
    value = str(value).strip().lower()
    if value not in {"batch", "overlapped"}:
        raise RuntimeError(f"config {label} must be batch or overlapped")
    return value


def configured_pipeline_mode_from_config(cfg: dict[str, Any]) -> str | None:
    if cfg.get("layer_split_pipeline_mode") is None:
        return None
    return parse_pipeline_mode(cfg["layer_split_pipeline_mode"], label="layer_split_pipeline_mode")


def run_pipeline_mode_from_config(cfg: dict[str, Any]) -> str | None:
    if cfg.get("run_layer_split_pipeline_mode") is not None:
        return parse_pipeline_mode(
            cfg["run_layer_split_pipeline_mode"], label="run_layer_split_pipeline_mode"
        )
    configured = configured_pipeline_mode_from_config(cfg)
    if configured is not None:
        return configured
    if cfg.get("expected_runtime_preset") is None:
        return "overlapped"
    return None


def serve_pipeline_mode_from_config(cfg: dict[str, Any]) -> str | None:
    configured = configured_pipeline_mode_from_config(cfg)
    if configured is not None:
        return configured
    if cfg.get("expected_runtime_preset") is None:
        return "overlapped"
    return None


def expected_pipeline_mode_from_config(cfg: dict[str, Any]) -> str:
    if cfg.get("expected_layer_split_pipeline_mode") is not None:
        return parse_pipeline_mode(
            cfg["expected_layer_split_pipeline_mode"],
            label="expected_layer_split_pipeline_mode",
        )
    return configured_pipeline_mode_from_config(cfg) or "overlapped"


def even_layer_split_plan_for_layers(devices: list[int], num_layers: int) -> str:
    if not devices:
        raise RuntimeError("layer split requires at least one CUDA device")
    if num_layers < len(devices):
        raise RuntimeError(
            f"layer split requires at least as many transformer layers ({num_layers}) as CUDA devices ({len(devices)})"
        )
    base = num_layers // len(devices)
    remainder = num_layers % len(devices)
    start = 0
    stages = []
    for idx, device in enumerate(devices):
        count = base + (1 if idx < remainder else 0)
        end = start + count - 1
        stages.append(f"stage{idx}:cuda:{device}:layers={start}-{end}")
        start = end + 1
    return ";".join(stages)


def layer_split_plan_from_config(cfg: dict[str, Any]) -> str:
    if isinstance(cfg.get("layer_split_plan"), str) and cfg["layer_split_plan"].strip():
        return cfg["layer_split_plan"]
    num_layers = cfg.get("num_hidden_layers")
    if isinstance(num_layers, int):
        return even_layer_split_plan_for_layers(REQUIRED_GPU_DEVICES, num_layers)
    return DEFAULT_LAYER_SPLIT_PLAN


def latest_hf_snapshot_for_model(model: str) -> Path | None:
    repo = hf_cache_dir(model)
    if repo is None:
        return None
    snapshots = sorted(
        (repo / "snapshots").glob("*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return snapshots[0] if snapshots else None


def resolve_model_manifest_path(model: str) -> tuple[Path | None, str]:
    path = Path(model)
    if path.exists():
        return path, "local_path"
    snapshot = latest_hf_snapshot_for_model(model)
    if snapshot is not None:
        return snapshot, "hf_cache_snapshot"
    return None, "unresolved"


def tokenizer_manifest_path(cfg: dict[str, Any], model_source: Path) -> Path:
    tokenizer = cfg.get("tokenizer_path") or cfg.get("tokenizer")
    if tokenizer is not None and str(tokenizer) != "auto":
        path = Path(str(tokenizer))
        return path.parent if path.is_file() else path
    return model_source.parent if model_source.is_file() else model_source


def is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def model_manifest(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    model = str(cfg["model"])
    path, resolved_from = resolve_model_manifest_path(model)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "status": "pending_model_resolution",
        "model": model,
        "model_id": model if "/" in model and not Path(model).exists() else None,
        "model_path": str(path) if path is not None else None,
        "resolved_from": resolved_from,
        "quant_format": cfg["quant_format"],
        "tokenizer_path": None,
        "config_sha256": None,
        "tokenizer_sha256": None,
        "tokenizer_metadata_sha256": None,
        "tokenizer_files": [],
        "weight_manifest_sha256": None,
        "weight_file_count": 0,
        "files": [],
    }
    if path is None:
        write_json(root / "model_manifest.json", manifest)
        return manifest

    files: list[dict[str, Any]] = []
    candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    base = path.parent if path.is_file() else path
    manifest["tokenizer_path"] = str(tokenizer_manifest_path(cfg, path))
    for file in candidates:
        if file.suffix not in MODEL_MANIFEST_INTERESTING_SUFFIXES:
            continue
        rel = file.relative_to(base).as_posix()
        digest = sha256(file)
        files.append({"path": rel, "size_bytes": file.stat().st_size, "sha256": digest})
    manifest["files"] = files
    for item in files:
        file_name = Path(str(item["path"])).name
        if file_name == "config.json":
            manifest["config_sha256"] = item["sha256"]
        elif file_name in {"tokenizer.json", "tokenizer.model"}:
            manifest["tokenizer_sha256"] = item["sha256"]
    tokenizer_files = [
        item for item in files if Path(str(item["path"])).name in TOKENIZER_METADATA_FILE_NAMES
    ]
    manifest["tokenizer_files"] = tokenizer_files
    if tokenizer_files:
        tokenizer_payload = json.dumps(tokenizer_files, sort_keys=True).encode("utf-8")
        manifest["tokenizer_metadata_sha256"] = hashlib.sha256(tokenizer_payload).hexdigest()
    weight_items = [item for item in files if str(item["path"]).endswith((".safetensors", ".gguf"))]
    manifest["weight_file_count"] = len(weight_items)
    if weight_items:
        weight_payload = json.dumps(weight_items, sort_keys=True).encode("utf-8")
        manifest["weight_manifest_sha256"] = hashlib.sha256(weight_payload).hexdigest()
    missing = [
        key
        for key in [
            "config_sha256",
            "tokenizer_sha256",
            "tokenizer_metadata_sha256",
            "weight_manifest_sha256",
        ]
        if not is_sha256_digest(manifest.get(key))
    ]
    if not weight_items:
        missing.append("weight files")
    manifest["status"] = "pass" if not missing else "incomplete"
    if missing:
        manifest["missing_required"] = missing
    write_json(root / "model_manifest.json", manifest)
    return manifest


def require_release_model_manifest(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    manifest = model_manifest(root, cfg)
    errors: list[str] = []
    if manifest.get("status") != "pass":
        errors.append(f"status={manifest.get('status')!r}")
    if not isinstance(manifest.get("model_path"), str) or not manifest["model_path"]:
        errors.append("missing model_path")
    for key in [
        "config_sha256",
        "tokenizer_sha256",
        "tokenizer_metadata_sha256",
        "weight_manifest_sha256",
    ]:
        if not is_sha256_digest(manifest.get(key)):
            errors.append(f"missing {key}")
    if int(manifest.get("weight_file_count") or 0) <= 0:
        errors.append("missing weight files")
    if not isinstance(manifest.get("files"), list) or not manifest["files"]:
        errors.append("missing file manifest")
    if not isinstance(manifest.get("tokenizer_files"), list) or not manifest["tokenizer_files"]:
        errors.append("missing tokenizer metadata files")
    if errors:
        raise RuntimeError("model_manifest incomplete for release evidence: " + "; ".join(errors))
    return {
        "status": "pass",
        "model": manifest.get("model"),
        "model_path": manifest.get("model_path"),
        "resolved_from": manifest.get("resolved_from"),
        "file_count": len(manifest.get("files", [])),
        "weight_file_count": manifest.get("weight_file_count"),
        "config_sha256": manifest.get("config_sha256"),
        "tokenizer_sha256": manifest.get("tokenizer_sha256"),
        "tokenizer_metadata_sha256": manifest.get("tokenizer_metadata_sha256"),
        "weight_manifest_sha256": manifest.get("weight_manifest_sha256"),
    }


def build_run_command(ferrum_bin: Path, model: str, cfg: dict[str, Any], root: Path) -> list[str]:
    cmd = [
        str(ferrum_bin),
        "run",
        model,
        "--backend",
        "cuda",
        "--gpu-devices",
        ",".join(str(device) for device in REQUIRED_GPU_DEVICES),
        "--max-tokens",
        str(cfg.get("run_max_tokens", 64)),
        "--stop",
        "\n",
        "--output-format",
        "jsonl",
        "--effective-config-json",
        str(root / "run.effective_config.json"),
        "--decision-trace-jsonl",
        str(root / "run.decision_trace.jsonl"),
    ]
    run_pipeline_mode = run_pipeline_mode_from_config(cfg)
    if run_pipeline_mode is not None:
        cmd.extend(["--layer-split-pipeline-mode", run_pipeline_mode])
    if cfg.get("run_max_model_len", cfg.get("max_model_len")) is not None:
        cmd.extend(["--max-model-len", str(cfg.get("run_max_model_len", cfg.get("max_model_len")))])
    if cfg.get("run_kv_capacity", cfg.get("kv_capacity")) is not None:
        cmd.extend(["--kv-capacity", str(cfg.get("run_kv_capacity", cfg.get("kv_capacity")))])
    if cfg.get("run_kv_max_blocks", cfg.get("kv_max_blocks")) is not None:
        cmd.extend(
            [
                "--kv-max-blocks",
                str(cfg.get("run_kv_max_blocks", cfg.get("kv_max_blocks"))),
            ]
        )
    if cfg.get("run_max_num_seqs", cfg.get("max_num_seqs")) is not None:
        cmd.extend(["--max-num-seqs", str(cfg.get("run_max_num_seqs", cfg.get("max_num_seqs")))])
    if cfg.get("run_max_num_batched_tokens", cfg.get("max_num_batched_tokens")) is not None:
        cmd.extend(
            [
                "--max-num-batched-tokens",
                str(cfg.get("run_max_num_batched_tokens", cfg.get("max_num_batched_tokens"))),
            ]
        )
    return cmd


def run_cli_probe_input_text() -> str:
    return "\n".join(
        [
            f"Remember the codeword {RECALL_MARKER}. Reply exactly OK.",
            "What codeword did I ask you to remember? Answer with only the codeword.",
            "/bye",
            "",
        ]
    )


def build_serve_command(ferrum_bin: Path, model: str, cfg: dict[str, Any], root: Path) -> list[str]:
    cmd = [
        str(ferrum_bin),
        "serve",
        "--model",
        model,
        "--backend",
        "cuda",
        "--gpu-devices",
        ",".join(str(device) for device in REQUIRED_GPU_DEVICES),
        "--host",
        "127.0.0.1",
        "--port",
        str(cfg.get("port", 19400)),
        "--effective-config-json",
        str(root / "serve.effective_config.json"),
        "--decision-trace-jsonl",
        str(root / "serve.decision_trace.jsonl"),
    ]
    pipeline_mode = serve_pipeline_mode_from_config(cfg)
    if pipeline_mode is not None:
        cmd.extend(["--layer-split-pipeline-mode", pipeline_mode])
    if cfg.get("max_model_len") is not None:
        cmd.extend(["--max-model-len", str(cfg["max_model_len"])])
    if cfg.get("kv_capacity") is not None:
        cmd.extend(["--kv-capacity", str(cfg["kv_capacity"])])
    if cfg.get("max_num_seqs") is not None:
        cmd.extend(["--max-num-seqs", str(cfg["max_num_seqs"])])
    if cfg.get("max_num_batched_tokens") is not None:
        cmd.extend(["--max-num-batched-tokens", str(cfg["max_num_batched_tokens"])])
    if cfg.get("scheduler_prefill_first_until_active") is not None:
        cmd.extend(
            [
                "--scheduler-prefill-first-until-active",
                str(cfg["scheduler_prefill_first_until_active"]),
            ]
        )
    return cmd


def build_bench_serve_command(ferrum_bin: Path, cfg: dict[str, Any], root: Path) -> list[str]:
    port = int(cfg.get("port", 19400))
    cmd = [
        str(ferrum_bin),
        "bench-serve",
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--model",
        str(cfg["model"]),
        "--tokenizer",
        str(cfg.get("tokenizer_path") or cfg.get("tokenizer") or cfg["model"]),
        "--dataset",
        "random",
        "--fail-on-error",
        "--seed",
        str(cfg.get("seed", 9271)),
        "--n-repeats",
        str(cfg.get("n_repeats", 3)),
        "--num-prompts",
        str(cfg.get("num_prompts", 96)),
        "--warmup-requests",
        str(cfg.get("warmup_requests", 10)),
        "--concurrency-sweep",
        ",".join(str(cell) for cell in cfg.get("concurrency_cells", [1, 4, 8, 16])),
        "--random-input-len",
        str(cfg.get("random_input_len", 256)),
        "--random-output-len",
        str(cfg.get("random_output_len", 128)),
        "--output",
        "json",
        "--out",
        str(root / "bench-serve.json"),
        "--tag",
        "cuda-llama33-70b-4bit-2x4090",
    ]
    if cfg.get("require_ci", True):
        cmd.insert(cmd.index("--seed"), "--require-ci")
    return cmd


def build_vllm_baseline_command(cfg: dict[str, Any]) -> list[str]:
    return [
        "vllm",
        "serve",
        str(cfg["model"]),
        "--tensor-parallel-size",
        "2",
        "--port",
        str(int(cfg.get("port", 19400)) + 1),
        "--quantization",
        str(cfg.get("quant_format", "gptq_int4")).split("_")[0],
    ]


def write_planned_command_artifacts(root: Path, ferrum_bin: Path, cfg: dict[str, Any]) -> None:
    serve_cmd = build_serve_command(ferrum_bin, cfg["model"], cfg, root)
    bench_cmd = build_bench_serve_command(ferrum_bin, cfg, root)
    vllm_cmd = build_vllm_baseline_command(cfg)
    required_bench_flags = ["--fail-on-error"]
    if cfg.get("require_ci", True):
        required_bench_flags.append("--require-ci")
    required_bench_flags.extend(["--seed", "--n-repeats"])
    write_json(
        root / "serve.command.json",
        {
            "status": "planned",
            "cmd": serve_cmd,
            "expected_gpu_devices": REQUIRED_GPU_DEVICES,
            "expected_distributed_strategy": "layer_split",
            "expected_layer_split_plan": layer_split_plan_from_config(cfg),
            "scheduler_prefill_first_until_active": cfg.get(
                "scheduler_prefill_first_until_active"
            ),
        },
    )
    write_json(
        root / "bench-serve.command.json",
        {
            "status": "planned",
            "cmd": bench_cmd,
            "required_flags": required_bench_flags,
        },
    )
    write_json(
        root / "vllm-baseline.command.json",
        {
            "status": "planned",
            "cmd": vllm_cmd,
            "same_hardware_required": True,
        },
    )


def write_metadata(
    root: Path,
    repo: Path,
    ferrum_bin: Path,
    cfg: dict[str, Any],
    hardware: dict[str, Any],
    command_line: list[str],
) -> dict[str, Any]:
    metadata = {
        "schema_version": 1,
        "lane": LANE,
        "status": "running",
        "created_at": iso_now(),
        "git_sha": git_output(["rev-parse", "HEAD"], repo),
        "dirty_status": {
            "is_dirty": bool(git_output(["status", "--short"], repo)),
            "status_short": git_output(["status", "--short"], repo).splitlines(),
        },
        "binary_sha256": sha256(ferrum_bin),
        "command_line": command_line,
        "build_features": ["cuda", "vllm-moe-marlin", "vllm-paged-attn-v2"],
        "cuda_version": hardware.get("cuda_version", "unknown"),
        "driver_version": hardware.get("driver_version", "unknown"),
        "gpu_names": hardware.get("gpu_names", []),
        "gpu_uuids": hardware.get("gpu_uuids", []),
        "requested_gpu_devices": REQUIRED_GPU_DEVICES,
        "selected_gpu_devices": REQUIRED_GPU_DEVICES,
        "model_id": cfg["model"],
        "quant_format": cfg["quant_format"],
        "distributed_strategy": "layer_split",
        "layer_split_plan": layer_split_plan_from_config(cfg),
        "layer_split_pipeline_mode": expected_pipeline_mode_from_config(cfg),
        "sanitized_env": sanitized_env_summary(),
    }
    write_json(root / "metadata.json", metadata)
    return metadata


def write_vllm_metadata(
    root: Path,
    repo: Path,
    cfg: dict[str, Any],
    hardware: dict[str, Any],
    server_cmd: list[str],
    bench_cmd: list[str],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    vllm_bin = preflight.get("binary_path") or shutil.which(server_cmd[0]) or server_cmd[0]
    metadata = {
        "schema_version": 1,
        "lane": LANE,
        "status": "running",
        "created_at": iso_now(),
        "engine": "vllm",
        "git_sha": git_output(["rev-parse", "HEAD"], repo),
        "dirty_status": {
            "is_dirty": bool(git_output(["status", "--short"], repo)),
            "status_short": git_output(["status", "--short"], repo).splitlines(),
        },
        "binary_path": vllm_bin,
        "binary_sha256": preflight.get("binary_sha256") or sha256(Path(vllm_bin)),
        "server_command": server_cmd,
        "bench_command": bench_cmd,
        "preflight": preflight,
        "cuda_version": hardware.get("cuda_version", "unknown"),
        "driver_version": hardware.get("driver_version", "unknown"),
        "gpu_names": hardware.get("gpu_names", []),
        "gpu_uuids": hardware.get("gpu_uuids", []),
        "requested_gpu_devices": REQUIRED_GPU_DEVICES,
        "selected_gpu_devices": REQUIRED_GPU_DEVICES,
        "model_id": cfg["model"],
        "quant_format": cfg["quant_format"],
        "sanitized_env": sanitized_env_summary(),
    }
    write_json(root / "vllm-baseline.metadata.json", metadata)
    return metadata


def update_vllm_metadata_status(root: Path, status: str, error: str | None = None) -> None:
    path = root / "vllm-baseline.metadata.json"
    if not path.is_file():
        return
    data = load_json(path)
    data["status"] = status
    data["finished_at"] = iso_now()
    if error:
        data["error"] = error
    write_json(path, data)


def assert_no_bad_patterns(label: str, text: str) -> None:
    lower = text.lower()
    for pattern in BAD_PATTERNS:
        if pattern.lower() in lower:
            raise RuntimeError(f"forbidden pattern {pattern!r} in {label}")


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def normalized_recall(text: str) -> str:
    return strip_think(text).strip().strip("\"'`").strip()


def recall_matches_marker(text: str) -> bool:
    return normalized_recall(text).casefold() == RECALL_MARKER.casefold()


def assistant_text_from_jsonl(stdout: str) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "assistant":
            turns.append(row)
    return turns


def parsed_response(label: str, status: int, body: str) -> dict[str, Any]:
    if status != 200:
        raise RuntimeError(f"{label}: expected HTTP 200, got {status}: {body[:500]}")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label}: invalid JSON: {exc}: {body[:500]}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{label}: response must be a JSON object")
    return data


def first_message_content(data: dict[str, Any]) -> tuple[str, str | None]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"missing choices: {data}")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError(f"invalid choice: {choice!r}")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"missing message: {choice}")
    return str(message.get("content") or ""), choice.get("finish_reason")


def parse_sse(body: str) -> tuple[list[dict[str, Any]], int, int]:
    chunks: list[dict[str, Any]] = []
    done_count = 0
    malformed = 0
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ").strip()
        if not data:
            continue
        if data == "[DONE]":
            done_count += 1
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(parsed, dict):
            chunks.append(parsed)
        else:
            malformed += 1
    return chunks, done_count, malformed


def hf_cache_dir(model: str) -> Path | None:
    if "/" not in model:
        return None
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    repo_dir = hf_home / "hub" / ("models--" + model.replace("/", "--"))
    return repo_dir if repo_dir.is_dir() else None


def resolve_tokenizer(tokenizer: str, model: str) -> Path:
    if tokenizer != "auto":
        path = Path(tokenizer)
        if path.is_file() and path.name == "tokenizer.json":
            return path.parent
        if (path / "tokenizer.json").is_file():
            return path
        raise RuntimeError(f"tokenizer not found: {path}")
    model_path = Path(model)
    if model_path.is_dir() and (model_path / "tokenizer.json").is_file():
        return model_path
    repo = hf_cache_dir(model)
    if repo is not None:
        snapshots = sorted(
            (repo / "snapshots").glob("*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for snap in snapshots:
            if (snap / "tokenizer.json").is_file():
                return snap
    raise RuntimeError("could not auto-resolve tokenizer.json; pass tokenizer in gate config")


def report_list(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    reports = data if isinstance(data, list) else [data]
    if not all(isinstance(report, dict) for report in reports):
        raise RuntimeError(f"{path} must contain a BenchReport object or list")
    return reports


def scalar_mean(report: dict[str, Any], key: str) -> float:
    value = report.get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    if not isinstance(value, (int, float)) or value <= 0:
        raise RuntimeError(f"invalid {key}: {value!r}")
    return float(value)


def percentile_mean(report: dict[str, Any], metric: str, percentile: str = "p50") -> float:
    metric_obj = report.get(metric)
    if not isinstance(metric_obj, dict):
        raise RuntimeError(f"missing metric {metric}")
    value = metric_obj.get(percentile)
    if isinstance(value, dict):
        value = value.get("mean")
    if not isinstance(value, (int, float)) or value <= 0:
        raise RuntimeError(f"invalid {metric}.{percentile}: {value!r}")
    return float(value)


def validate_bench_quality(report: dict[str, Any], *, label: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for field in BENCH_QUALITY_COUNT_FIELDS:
        values = report.get(field)
        if not isinstance(values, list):
            raise RuntimeError(f"{label}: missing {field}")
        total = 0
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError(f"{label}: {field} contains non-integer {value!r}")
            total += value
        if total != 0:
            raise RuntimeError(f"{label}: {field} total={total} values={values!r}")
        counts[field.removesuffix("_per_run")] = total
    return counts


def validate_bench_reports(
    path: Path,
    cfg: dict[str, Any],
    *,
    require_ci: bool,
    label: str,
) -> dict[int, dict[str, Any]]:
    required = {int(c) for c in cfg.get("concurrency_cells", [1, 4, 8, 16])}
    rows: dict[int, dict[str, Any]] = {}
    for report in report_list(path):
        concurrency = int(report.get("concurrency") or 0)
        if concurrency <= 0:
            raise RuntimeError(f"{label}: report missing positive concurrency")
        completed = sum(int(v) for v in report.get("completed_per_run", []))
        errored = sum(int(v) for v in report.get("errored_per_run", []))
        n_repeats = int(report.get("n_repeats") or 0)
        output_source = report.get("output_token_count_source")
        if require_ci and n_repeats < 3:
            raise RuntimeError(f"{label} c={concurrency}: n_repeats < 3")
        if errored != 0 or completed <= 0:
            raise RuntimeError(f"{label} c={concurrency}: completed={completed} errored={errored}")
        if output_source != "usage":
            raise RuntimeError(
                f"{label} c={concurrency}: output_token_count_source={output_source!r}"
            )
        quality_counts = validate_bench_quality(report, label=f"{label} c={concurrency}")
        rows[concurrency] = {
            "concurrency": concurrency,
            "completed": completed,
            "errored": errored,
            "n_repeats": n_repeats,
            "output_token_count_source": output_source,
            "output_throughput_tps": scalar_mean(report, "output_throughput_tps"),
            "ttft_p50_ms": percentile_mean(report, "ttft_ms", "p50"),
            "tpot_p50_ms": percentile_mean(report, "tpot_ms", "p50"),
            "e2e_p95_ms": percentile_mean(report, "e2e_ms", "p95"),
            **quality_counts,
        }
    missing = sorted(required - set(rows))
    if missing:
        raise RuntimeError(f"{label}: missing concurrency cells {missing}")
    return rows


def compare_benchmarks(
    ferrum_rows: dict[int, dict[str, Any]],
    vllm_rows: dict[int, dict[str, Any]],
    required_cells: list[int],
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for concurrency in required_cells:
        ferrum = ferrum_rows[concurrency]
        vllm = vllm_rows[concurrency]
        throughput_ratio = ferrum["output_throughput_tps"] / vllm["output_throughput_tps"]
        ttft_ratio = ferrum["ttft_p50_ms"] / vllm["ttft_p50_ms"]
        tpot_ratio = ferrum["tpot_p50_ms"] / vllm["tpot_p50_ms"]
        passed = throughput_ratio >= 0.70 and ttft_ratio <= 1.50 and tpot_ratio <= 1.50
        cells[f"c{concurrency}"] = {
            "status": "pass" if passed else "fail",
            "concurrency": concurrency,
            "ferrum_output_throughput_tps": ferrum["output_throughput_tps"],
            "vllm_output_throughput_tps": vllm["output_throughput_tps"],
            "output_throughput_ratio_to_vllm": throughput_ratio,
            "ferrum_ttft_p50_ms": ferrum["ttft_p50_ms"],
            "vllm_ttft_p50_ms": vllm["ttft_p50_ms"],
            "ttft_ratio_to_vllm": ttft_ratio,
            "ferrum_tpot_p50_ms": ferrum["tpot_p50_ms"],
            "vllm_tpot_p50_ms": vllm["tpot_p50_ms"],
            "tpot_ratio_to_vllm": tpot_ratio,
            "p95_end_to_end_latency_ms": ferrum["e2e_p95_ms"],
            "bad_output_count": ferrum.get("bad_output", 0),
            "malformed_stream_count": ferrum.get("malformed_stream", 0),
        }
        if not passed:
            raise RuntimeError(
                f"comparison c={concurrency} failed: throughput_ratio={throughput_ratio:.3f} "
                f"ttft_ratio={ttft_ratio:.3f} tpot_ratio={tpot_ratio:.3f}"
            )
    return {"status": "pass", "cells": cells}


def ferrum_only_comparison(
    ferrum_rows: dict[int, dict[str, Any]],
    required_cells: list[int],
    reason: str,
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for concurrency in required_cells:
        ferrum = ferrum_rows[concurrency]
        cells[f"c{concurrency}"] = {
            "status": "pass",
            "mode": "ferrum_only",
            "baseline": "not_run",
            "reason": reason,
            "concurrency": concurrency,
            "ferrum_output_throughput_tps": ferrum["output_throughput_tps"],
            "ferrum_ttft_p50_ms": ferrum["ttft_p50_ms"],
            "ferrum_tpot_p50_ms": ferrum["tpot_p50_ms"],
            "p95_end_to_end_latency_ms": ferrum["e2e_p95_ms"],
            "bad_output_count": ferrum.get("bad_output", 0),
            "malformed_stream_count": ferrum.get("malformed_stream", 0),
        }
    return {
        "status": "pass",
        "mode": "ferrum_only",
        "baseline": "not_run",
        "reason": reason,
        "cells": cells,
    }


def copy_if_present(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def normalize_runtime_artifacts(root: Path) -> None:
    copy_if_present(root / "run.effective_config.json", root / "effective_config.json")
    copy_if_present(root / "run.decision_trace.jsonl", root / "decision_trace.jsonl")


def promote_serve_runtime_artifacts(root: Path) -> None:
    copy_if_present(root / "serve.effective_config.json", root / "effective_config.json")
    copy_if_present(root / "serve.decision_trace.jsonl", root / "decision_trace.jsonl")


def run_effective_config_validation_config(cfg: dict[str, Any]) -> dict[str, Any]:
    run_cfg = dict(cfg)
    for key in [
        "expected_runtime_preset",
        "expected_scheduler_prefill_first_until_active",
        "require_default_runtime_sources",
    ]:
        run_cfg.pop(key, None)
    run_expected_scalars = {
        "run_max_model_len": "expected_max_model_len",
        "run_kv_capacity": "expected_kv_capacity",
        "run_kv_max_blocks": "expected_kv_max_blocks",
        "run_max_num_seqs": "expected_max_num_seqs",
        "run_max_num_batched_tokens": "expected_max_num_batched_tokens",
    }
    for run_key, expected_key in run_expected_scalars.items():
        if cfg.get(run_key) is not None:
            run_cfg[expected_key] = cfg[run_key]
    return run_cfg


def require_effective_config(path: Path, label: str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    data = load_json(path)
    required = {
        "backend",
        "requested_gpu_devices",
        "selected_gpu_devices",
        "cuda_device_count",
        "selected_distributed_strategy",
        "selected_layer_split_plan",
        "selected_pipeline_mode",
        "selected_microbatch_size",
        "selected_stage_bridge",
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
    missing = sorted(required - set(data))
    if missing:
        raise RuntimeError(f"{label} missing effective config fields: {missing}")
    if data["backend"] != "cuda":
        raise RuntimeError(f"{label}: backend must be cuda, got {data['backend']!r}")
    if data["requested_gpu_devices"] != REQUIRED_GPU_DEVICES:
        raise RuntimeError(
            f"{label}: requested_gpu_devices must be {REQUIRED_GPU_DEVICES}, "
            f"got {data['requested_gpu_devices']!r}"
        )
    if data["selected_gpu_devices"] != REQUIRED_GPU_DEVICES:
        raise RuntimeError(
            f"{label}: selected_gpu_devices must be {REQUIRED_GPU_DEVICES}, "
            f"got {data['selected_gpu_devices']!r}"
        )
    if data["cuda_device_count"] != 2:
        raise RuntimeError(f"{label}: cuda_device_count must be 2, got {data['cuda_device_count']!r}")
    if data["selected_distributed_strategy"] != "layer_split":
        raise RuntimeError(
            f"{label}: selected_distributed_strategy must be layer_split, "
            f"got {data['selected_distributed_strategy']!r}"
        )
    expected_plan = even_layer_split_plan_for_layers(REQUIRED_GPU_DEVICES, 80)
    if data["selected_layer_split_plan"] != expected_plan:
        raise RuntimeError(
            f"{label}: selected_layer_split_plan must be {expected_plan!r}, "
            f"got {data['selected_layer_split_plan']!r}"
        )
    if data["selected_weight_placement"] != "layer_split":
        raise RuntimeError(
            f"{label}: selected_weight_placement must be layer_split, "
            f"got {data['selected_weight_placement']!r}"
        )
    expected_pipeline_mode = expected_pipeline_mode_from_config(cfg or {})
    if data["selected_pipeline_mode"] != expected_pipeline_mode:
        raise RuntimeError(
            f"{label}: selected_pipeline_mode must be {expected_pipeline_mode}, "
            f"got {data['selected_pipeline_mode']!r}"
        )
    if data["selected_stage_bridge"] != "host":
        raise RuntimeError(
            f"{label}: selected_stage_bridge must be host, got {data['selected_stage_bridge']!r}"
        )
    max_sequences = int(data["selected_max_sequences"])
    if expected_pipeline_mode == "overlapped":
        expected_microbatch = max(1, (max_sequences + 1) // 2)
    else:
        expected_microbatch = max_sequences
    if data["selected_microbatch_size"] != expected_microbatch:
        raise RuntimeError(
            f"{label}: selected_microbatch_size must be {expected_microbatch}, "
            f"got {data['selected_microbatch_size']!r}"
        )
    if cfg:
        expected_preset = cfg.get("expected_runtime_preset")
        if expected_preset is not None and data.get("preset") != expected_preset:
            raise RuntimeError(
                f"{label}: preset must be {expected_preset!r}, got {data.get('preset')!r}"
            )
        expected_scalars = {
            "expected_max_model_len": "selected_max_model_len",
            "expected_kv_capacity": "selected_kv_capacity",
            "expected_max_num_seqs": "selected_max_sequences",
            "expected_max_num_batched_tokens": "selected_max_batched_tokens",
        }
        for config_key, effective_key in expected_scalars.items():
            expected = cfg.get(config_key)
            if expected is not None and data.get(effective_key) != expected:
                raise RuntimeError(
                    f"{label}: {effective_key} must be {expected!r}, "
                    f"got {data.get(effective_key)!r}"
                )
        expected_kv_blocks = cfg.get("expected_kv_max_blocks")
        if expected_kv_blocks is not None:
            actual_kv_blocks = effective_kv_block_count(data)
            if actual_kv_blocks != expected_kv_blocks:
                raise RuntimeError(
                    f"{label}: kv_block_count must be {expected_kv_blocks!r}, "
                    f"got {actual_kv_blocks!r}"
                )
        expected_scheduler = cfg.get("expected_scheduler_prefill_first_until_active")
        if expected_scheduler is not None:
            scheduler = decision_by_selection(data, "scheduler_admission_policy")
            expected_selected = f"prefill_first_until_active:{expected_scheduler}"
            if scheduler.get("selected") != expected_selected:
                raise RuntimeError(
                    f"{label}: scheduler_admission_policy must be {expected_selected!r}, "
                    f"got {scheduler.get('selected')!r}"
                )
        if cfg.get("require_default_runtime_sources"):
            for key in [
                "FERRUM_LAYER_SPLIT_PIPELINE_MODE",
                "FERRUM_MAX_MODEL_LEN",
                "FERRUM_KV_MAX_BLOCKS",
                "FERRUM_KV_CAPACITY",
                "FERRUM_PAGED_MAX_SEQS",
                "FERRUM_MAX_BATCHED_TOKENS",
                "FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE",
            ]:
                entry = runtime_entry_by_key(data, key)
                if entry.get("source") != "default":
                    raise RuntimeError(
                        f"{label}: {key} source must be default, got {entry.get('source')!r}"
                    )
    return data


def runtime_entry_by_key(data: dict[str, Any], key: str) -> dict[str, Any]:
    for entry in data.get("entries", []):
        if isinstance(entry, dict) and entry.get("key") == key:
            return entry
    raise RuntimeError(f"effective config missing runtime entry {key}")


def decision_by_selection(data: dict[str, Any], selection: str) -> dict[str, Any]:
    for decision in data.get("decisions", []):
        if isinstance(decision, dict) and decision.get("selection") == selection:
            return decision
    raise RuntimeError(f"effective config missing decision {selection}")


def effective_kv_block_count(data: dict[str, Any]) -> Any:
    if data.get("kv_block_count") is not None:
        return data.get("kv_block_count")
    admission = data.get("admission")
    if isinstance(admission, dict) and admission.get("kv_block_count") is not None:
        return admission.get("kv_block_count")
    decision = decision_by_selection(data, "kv_block_count")
    selected = decision.get("selected")
    try:
        return int(selected)
    except (TypeError, ValueError):
        return selected


def validate_hardware_for_gate(hardware: dict[str, Any]) -> None:
    if hardware.get("cuda_device_count") != 2:
        raise RuntimeError(f"expected exactly 2 CUDA devices, got {hardware.get('cuda_device_count')!r}")
    names = [str(name).lower() for name in hardware.get("gpu_names", [])]
    if len(names) != 2:
        raise RuntimeError(f"expected two GPU names, got {hardware.get('gpu_names')!r}")
    if any("4090" not in name for name in names):
        raise RuntimeError(f"expected two RTX 4090 GPUs, got {hardware.get('gpu_names')!r}")


def ensure_failure_artifacts(root: Path, error: str) -> None:
    json_defaults = {
        "metadata.json": {"status": "fail", "error": error, "lane": LANE},
        "effective_config.json": {"status": "fail", "error": error},
        "hardware.json": {"status": "fail", "error": error, "gpu_names": [], "gpus": []},
        "nvidia-smi.before.json": {"status": "fail", "error": error, "gpus": []},
        "nvidia-smi.during.json": {"status": "fail", "error": error, "gpus": []},
        "nvidia-smi.after.json": {"status": "fail", "error": error, "gpus": []},
        "model_manifest.json": {"status": "fail", "error": error},
        "run.command.json": {"status": "fail", "error": error},
        "run.effective_config.json": {"status": "fail", "error": error},
        "serve.command.json": {"status": "not_run", "blocked_by": error},
        "serve.effective_config.json": {"status": "not_run", "blocked_by": error},
        "serve.health.json": {"status": "not_run", "blocked_by": error},
        "serve.health.after.json": {"status": "not_run", "blocked_by": error},
        "serve.models.json": {"status": "not_run", "blocked_by": error},
        "serve.correctness.json": {"status": "not_run", "blocked_by": error},
        "serve.multiturn.json": {"status": "not_run", "blocked_by": error},
        "serve.structured_output.json": {"status": "not_run", "blocked_by": error},
        "serve.tool_call.json": {"status": "not_run", "blocked_by": error},
        "correctness.json": {"status": "fail", "error": error, "checks": {}},
        "concurrency_quality_regression.json": {"status": "not_run", "blocked_by": error},
        "bench-serve.command.json": {"status": "not_run", "blocked_by": error},
        "bench-serve.json": {"status": "not_run", "blocked_by": error},
        "vllm-baseline.command.json": {"status": "not_run", "blocked_by": error},
        "vllm-baseline.json": {"status": "not_run", "blocked_by": error},
        "comparison.json": {"status": "not_run", "blocked_by": error, "cells": {}},
    }
    text_defaults = {
        "decision_trace.jsonl": json.dumps({"status": "fail", "error": error}) + "\n",
        "nvidia-smi.before.txt": f"not captured: {error}\n",
        "nvidia-smi.during.txt": f"not captured: {error}\n",
        "nvidia-smi.after.txt": f"not captured: {error}\n",
        "nvidia-smi.bench.samples.jsonl": json.dumps(
            {"status": "fail", "error": error, "gpus": []}
        )
        + "\n",
        "run.stdin": "",
        "run.stdout": "",
        "run.stderr": f"not run: {error}\n",
        "serve.log": f"not run: {error}\n",
        "serve.streaming.sse": f"not run: {error}\n",
        "bench-serve.stdout": "",
        "bench-serve.stderr": f"not run: {error}\n",
    }
    for rel, data in json_defaults.items():
        path = root / rel
        if not path.exists():
            write_json(path, data)
    for rel, text in text_defaults.items():
        path = root / rel
        if not path.exists():
            write_text(path, text)


def update_metadata_status(root: Path, status: str, error: str | None = None) -> None:
    path = root / "metadata.json"
    if not path.is_file():
        return
    try:
        data = load_json(path)
    except Exception:
        return
    data["status"] = status
    data["finished_at"] = iso_now()
    if error:
        data["error"] = error
    write_json(path, data)


def require_pass_check(checks: dict[str, Any], key: str) -> dict[str, Any]:
    value = checks.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"missing check {key}")
    if value.get("status") != "pass":
        raise RuntimeError(f"check {key} did not pass: {value!r}")
    return value


def write_goal_correctness_artifact(root: Path, checks: dict[str, Any]) -> dict[str, Any]:
    run_check = require_pass_check(checks, "run")
    serve_single = require_pass_check(checks, "serve_correctness")
    serve_multiturn_check = require_pass_check(checks, "serve_multiturn")
    structured = require_pass_check(checks, "serve_structured_output")
    tool = require_pass_check(checks, "serve_tool_call")
    streaming = require_pass_check(checks, "serve_streaming")

    done_count = int(streaming.get("done_count", 0))
    usage_chunk_count = int(streaming.get("usage_chunk_count", 0))
    artifact = {
        "schema_version": 1,
        "status": "pass",
        "source": "g0_cuda_llama33_70b_4bit_2x4090_gate",
        "created_at": iso_now(),
        "checks": {
            "ferrum_run_single": {
                "status": "pass",
                "source_check": "run",
                "assistant_turns": run_check.get("assistant_turns"),
            },
            "ferrum_run_multiturn": {
                "status": "pass",
                "source_check": "run",
                "has_precise_recall": run_check.get("has_precise_recall") is True,
            },
            "ferrum_serve_single": {
                "status": "pass",
                "source_check": "serve_correctness",
                "contains_expected_answer": serve_single.get("contains_expected_answer") is True,
            },
            "ferrum_serve_multiturn": {
                "status": "pass",
                "source_check": "serve_multiturn",
                "has_precise_recall": serve_multiturn_check.get("has_precise_recall") is True,
            },
            "streaming_done": {
                "status": "pass",
                "source_check": "serve_streaming",
                "done_count": done_count,
            },
            "streaming_usage": {
                "status": "pass",
                "source_check": "serve_streaming",
                "include_usage": True,
                "usage_received": usage_chunk_count == 1,
                "usage_chunk_count": usage_chunk_count,
            },
            "tool_calling": {
                "status": "pass",
                "source_check": "serve_tool_call",
                "details": tool,
            },
            "structured_output": {
                "status": "pass",
                "source_check": "serve_structured_output",
                "json_ok": structured.get("json_ok") is True,
                "schema_ok": structured.get("schema_ok") is True,
            },
            "log_scan": {
                "status": "pass",
                "bad_pattern_count": 0,
                "scanned_files": [
                    "run.stdout",
                    "run.stderr",
                    "serve.log",
                    "serve.streaming.sse",
                    "bench-serve.stdout",
                    "bench-serve.stderr",
                ],
            },
        },
    }
    write_json(root / "correctness.json", artifact)
    return {"status": "pass", "path": "correctness.json"}


def run_cli_probe(root: Path, repo: Path, ferrum_bin: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    input_text = run_cli_probe_input_text()
    cmd = build_run_command(ferrum_bin, cfg["model"], cfg, root)
    write_json(
        root / "run.command.json",
        {
            "cmd": cmd,
            "expected_gpu_devices": REQUIRED_GPU_DEVICES,
            "expected_distributed_strategy": "layer_split",
            "expected_layer_split_plan": layer_split_plan_from_config(cfg),
            "expected_layer_split_pipeline_mode": expected_pipeline_mode_from_config(cfg),
            "config": {
                "max_model_len": cfg.get("run_max_model_len", cfg.get("max_model_len")),
                "kv_max_blocks": cfg.get("run_kv_max_blocks", cfg.get("kv_max_blocks")),
                "kv_capacity": cfg.get("run_kv_capacity", cfg.get("kv_capacity")),
                "max_num_seqs": cfg.get("run_max_num_seqs", cfg.get("max_num_seqs")),
                "max_num_batched_tokens": cfg.get(
                    "run_max_num_batched_tokens", cfg.get("max_num_batched_tokens")
                ),
            },
        },
    )
    write_text(root / "run.stdin", input_text)
    proc = run(cmd, cwd=repo, input_text=input_text, timeout=1800)
    write_text(root / "run.stdout", proc.stdout)
    write_text(root / "run.stderr", proc.stderr)
    normalize_runtime_artifacts(root)
    assert_no_bad_patterns("ferrum run stdout/stderr", proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"ferrum run layer_split probe failed rc={proc.returncode}")
    turns = assistant_text_from_jsonl(proc.stdout)
    if len(turns) < 2:
        raise RuntimeError(f"ferrum run expected at least 2 assistant turns, got {len(turns)}")
    for turn in turns:
        content = strip_think(str(turn.get("content") or ""))
        if not content:
            raise RuntimeError(f"ferrum run assistant turn is empty: {turn}")
        if turn.get("finish_reason") == "length":
            raise RuntimeError(f"ferrum run assistant turn finished by length: {turn}")
    recall = strip_think(str(turns[-1].get("content") or ""))
    if not recall_matches_marker(recall):
        raise RuntimeError(f"ferrum run recall failed: {recall[:500]!r}")
    effective = require_effective_config(
        root / "run.effective_config.json", "run", run_effective_config_validation_config(cfg)
    )
    return {
        "status": "pass",
        "returncode": proc.returncode,
        "assistant_turns": len(turns),
        "has_precise_recall": True,
        "selected_layer_split_plan": effective["selected_layer_split_plan"],
    }


def capture_health(root: Path, base_url: str, filename: str = "serve.health.json") -> dict[str, Any]:
    status, body = get_url(base_url.rstrip("/") + "/health", timeout=30)
    write_text(root / filename, body)
    data = parsed_response("serve health", status, body)
    assert_no_bad_patterns("serve health", body)
    result = {"status": "pass", "http_status": status, "version": data.get("version")}
    write_json(root / filename, {**data, **result})
    return result


def capture_models(root: Path, base_url: str, model: str) -> dict[str, Any]:
    status, body = get_url(base_url.rstrip("/") + "/v1/models", timeout=30)
    write_text(root / "serve.models.raw.json", body)
    data = parsed_response("serve models", status, body)
    model_ids = []
    for item in data.get("data", []):
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            model_ids.append(item["id"])
    accepted_ids = {model, Path(model).name}
    if not any(model_id in accepted_ids or model_id == model for model_id in model_ids):
        raise RuntimeError(f"/v1/models does not include target model {model!r}: {model_ids!r}")
    result = {"status": "pass", "http_status": status, "model_ids": model_ids}
    write_json(root / "serve.models.json", result)
    return result


def serve_correctness(root: Path, base_url: str, model: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0,
        "seed": 9271,
        "messages": [{"role": "user", "content": "123+456 等于多少？只输出数字。"}],
        "max_tokens": 128,
    }
    write_json(root / "serve.correctness.request.json", payload)
    status, body = post_json(base_url, payload, timeout=300)
    write_text(root / "serve.correctness.response.json", body)
    data = parsed_response("serve correctness", status, body)
    content, finish_reason = first_message_content(data)
    assert_no_bad_patterns("serve correctness", content)
    if "579" not in strip_think(content):
        raise RuntimeError(f"serve correctness expected 579, got {content[:500]!r}")
    if finish_reason == "length":
        raise RuntimeError("serve correctness finished by length")
    result = {
        "status": "pass",
        "http_status": status,
        "contains_expected_answer": True,
        "finish_reason": finish_reason,
    }
    write_json(root / "serve.correctness.json", result)
    return result


def serve_multiturn(root: Path, base_url: str, model: str) -> dict[str, Any]:
    messages = serve_multiturn_probe_messages()
    payload = {
        "model": model,
        "temperature": 0,
        "seed": 9271,
        "messages": messages,
        "max_tokens": 128,
        "stop": ["\n"],
    }
    write_json(root / "serve.multiturn.request.json", payload)
    required_passes = 2
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_body: str | None = None
    for attempt in range(1, 4):
        status, body = post_json(base_url, payload, timeout=300)
        write_text(root / f"serve.multiturn.attempt{attempt}.response.json", body)
        if attempt == 1:
            write_text(root / "serve.multiturn.response.json", body)
        row: dict[str, Any] = {
            "attempt": attempt,
            "http_status": status,
            "passed": False,
        }
        try:
            data = parsed_response(f"serve multiturn attempt {attempt}", status, body)
            content, finish_reason = first_message_content(data)
            row["content"] = content
            row["finish_reason"] = finish_reason
            assert_no_bad_patterns(f"serve multiturn attempt {attempt}", content)
            if not recall_matches_marker(content):
                raise RuntimeError(f"recall failed: {content[:500]!r}")
            if finish_reason == "length":
                raise RuntimeError("finished by length")
            row["passed"] = True
            if selected is None:
                selected = row
                selected_body = body
        except Exception as exc:
            row["error"] = str(exc)
        attempts.append(row)
        if sum(1 for item in attempts if item.get("passed")) >= required_passes:
            break
    passed_attempts = sum(1 for item in attempts if item.get("passed"))
    if passed_attempts < required_passes:
        write_json(
            root / "serve.multiturn.json",
            {
                "status": "fail",
                "has_precise_recall": False,
                "required_passes": required_passes,
                "passed_attempts": passed_attempts,
                "attempts": attempts,
            },
        )
        raise RuntimeError(f"serve multiturn recall failed attempts: {attempts}")
    if selected_body is not None:
        write_text(root / "serve.multiturn.response.json", selected_body)
    result = {
        "status": "pass",
        "http_status": selected.get("http_status") if selected else None,
        "has_precise_recall": True,
        "finish_reason": selected.get("finish_reason") if selected else None,
        "required_passes": required_passes,
        "passed_attempts": passed_attempts,
        "attempts": attempts,
    }
    write_json(root / "serve.multiturn.json", result)
    return result


def serve_multiturn_probe_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": f"Remember the codeword {RECALL_MARKER}. Reply exactly OK.",
        },
        {"role": "assistant", "content": "OK"},
        {
            "role": "user",
            "content": "What codeword did I ask you to remember? Answer with only the codeword.",
        },
    ]


def serve_structured_output(root: Path, base_url: str, model: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0,
        "seed": 9271,
        "messages": [
            {
                "role": "user",
                "content": "Return JSON for the answer to 123+456. Use answer string 579.",
            }
        ],
        "max_tokens": 128,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "Answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        },
    }
    write_json(root / "serve.structured_output.request.json", payload)
    status, body = post_json(base_url, payload, timeout=300)
    write_text(root / "serve.structured_output.response.json", body)
    data = parsed_response("serve structured output", status, body)
    content, finish_reason = first_message_content(data)
    assert_no_bad_patterns("serve structured output", content)
    try:
        parsed_content = json.loads(strip_think(content))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"structured output content is not JSON: {content[:500]!r}") from exc
    if parsed_content.get("answer") != "579":
        raise RuntimeError(f"structured output answer mismatch: {parsed_content!r}")
    if finish_reason == "length":
        raise RuntimeError("serve structured output finished by length")
    result = {
        "status": "pass",
        "http_status": status,
        "json_ok": True,
        "schema_ok": True,
        "finish_reason": finish_reason,
    }
    write_json(root / "serve.structured_output.json", result)
    return result


def serve_tool_call(root: Path, base_url: str, model: str) -> dict[str, Any]:
    subdir = root / "tool-call-regression"
    result = run_tool_call_regression(base_url, model, subdir)
    write_json(root / "serve.tool_call.json", result)
    return result


def serve_streaming(root: Path, base_url: str, model: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0,
        "seed": 9271,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "用一句中文解释 Ferrum 是什么。"}],
        "max_tokens": 128,
    }
    write_json(root / "serve.streaming.request.json", payload)
    status, body = post_json(base_url, payload, timeout=300)
    write_text(root / "serve.streaming.sse", body)
    if status != 200:
        raise RuntimeError(f"serve streaming expected HTTP 200, got {status}: {body[:500]}")
    assert_no_bad_patterns("serve streaming", body)
    chunks, done_count, malformed = parse_sse(body)
    content = "".join(
        str(chunk.get("choices", [{}])[0].get("delta", {}).get("content") or "")
        for chunk in chunks
        if isinstance(chunk.get("choices"), list) and chunk.get("choices")
    )
    usage_chunks = sum(1 for chunk in chunks if chunk.get("usage") not in (None, {}))
    if done_count != 1:
        raise RuntimeError(f"serve streaming expected exactly one [DONE], got {done_count}")
    if malformed != 0:
        raise RuntimeError(f"serve streaming malformed SSE JSON count={malformed}")
    if not content.strip():
        raise RuntimeError("serve streaming emitted no content delta")
    if usage_chunks != 1:
        raise RuntimeError(f"serve streaming expected one usage chunk, got {usage_chunks}")
    return {
        "status": "pass",
        "http_status": status,
        "done_count": done_count,
        "malformed_sse_json_count": malformed,
        "usage_chunk_count": usage_chunks,
        "content_delta_nonempty": True,
    }


def run_concurrency_gate(root: Path, base_url: str, model: str, cfg: dict[str, Any]) -> dict[str, Any]:
    cells = [int(c) for c in cfg.get("concurrency_cells", [1, 4, 8, 16])]
    subdir = root / "concurrency-quality-regression"
    result = run_concurrency_quality_regression(base_url, model, subdir, cells, timeout=300)
    copy_if_present(
        subdir / "concurrency_quality_regression.json",
        root / "concurrency_quality_regression.json",
    )
    return result


def bench_command_for_base_url(
    ferrum_bin: Path,
    cfg: dict[str, Any],
    *,
    base_url: str,
    model: str,
    tokenizer_dir: Path,
    out_file: Path,
    tag: str,
) -> list[str]:
    cmd = [
        str(ferrum_bin),
        "bench-serve",
        "--base-url",
        base_url.rstrip("/"),
        "--model",
        model,
        "--tokenizer",
        str(tokenizer_dir),
        "--dataset",
        "random",
        "--random-input-len",
        str(cfg.get("random_input_len", 256)),
        "--random-output-len",
        str(cfg.get("random_output_len", 128)),
        "--num-prompts",
        str(cfg.get("num_prompts", 96)),
        "--warmup-requests",
        str(cfg.get("warmup_requests", 10)),
        "--n-repeats",
        str(cfg.get("n_repeats", 3)),
        "--concurrency-sweep",
        ",".join(str(cell) for cell in cfg.get("concurrency_cells", [1, 4, 8, 16])),
        "--fail-on-error",
        "--seed",
        str(cfg.get("seed", 9271)),
        "--output",
        "json",
        "--out",
        str(out_file),
        "--tag",
        tag,
    ]
    if cfg.get("require_ci", True):
        cmd.insert(cmd.index("--seed"), "--require-ci")
    return cmd


def run_ferrum_bench_gate(
    root: Path,
    repo: Path,
    ferrum_bin: Path,
    cfg: dict[str, Any],
    base_url: str,
    model: str,
) -> dict[str, Any]:
    tokenizer_dir = resolve_tokenizer(str(cfg.get("tokenizer", "auto")), model)
    cmd = bench_command_for_base_url(
        ferrum_bin,
        cfg,
        base_url=base_url,
        model=model,
        tokenizer_dir=tokenizer_dir,
        out_file=root / "bench-serve.json",
        tag=str(cfg.get("bench_tag", LANE)),
    )
    write_json(root / "bench-serve.command.json", {"status": "run", "cmd": cmd})
    proc = run_with_gpu_samples(
        cmd,
        cwd=repo,
        root=root,
        timeout=int(cfg.get("bench_timeout_sec", 7200)),
        sample_interval_sec=int(cfg.get("bench_gpu_sample_interval_sec", 15)),
    )
    write_text(root / "bench-serve.stdout", proc.stdout)
    write_text(root / "bench-serve.stderr", proc.stderr)
    assert_no_bad_patterns("bench-serve output", proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"bench-serve failed rc={proc.returncode}")
    rows = validate_bench_reports(
        root / "bench-serve.json",
        cfg,
        require_ci=bool(cfg.get("require_ci", True)),
        label="ferrum bench-serve",
    )
    return {"status": "pass", "rows": list(rows.values())}


def build_vllm_server_command(cfg: dict[str, Any]) -> list[str]:
    port = int(cfg.get("port", 19400)) + int(cfg.get("vllm_port_offset", 1))
    cmd = [
        "vllm",
        "serve",
        str(cfg["model"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "2",
        "--served-model-name",
        str(cfg["model"]),
    ]
    quant = str(cfg.get("quant_format", "gptq_int4")).lower()
    if "gptq" in quant:
        cmd.extend(["--quantization", "gptq"])
    elif "awq" in quant:
        cmd.extend(["--quantization", "awq"])
    if cfg.get("max_model_len") is not None:
        cmd.extend(["--max-model-len", str(cfg["max_model_len"])])
    if cfg.get("max_num_seqs") is not None:
        cmd.extend(["--max-num-seqs", str(cfg["max_num_seqs"])])
    if cfg.get("gpu_memory_utilization") is not None:
        cmd.extend(["--gpu-memory-utilization", str(cfg["gpu_memory_utilization"])])
    return cmd


def terminate_process_group(proc: subprocess.Popen[str] | None, *, sig: int = signal.SIGINT) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=15)


def run_vllm_baseline_gate(
    root: Path,
    repo: Path,
    ferrum_bin: Path,
    cfg: dict[str, Any],
    model: str,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    tokenizer_dir = resolve_tokenizer(str(cfg.get("tokenizer", "auto")), model)
    port = int(cfg.get("port", 19400)) + int(cfg.get("vllm_port_offset", 1))
    base_url = f"http://127.0.0.1:{port}"
    server_cmd = build_vllm_server_command(cfg)
    bench_cmd = bench_command_for_base_url(
        ferrum_bin,
        cfg,
        base_url=base_url,
        model=model,
        tokenizer_dir=tokenizer_dir,
        out_file=root / "vllm-baseline.json",
        tag="vllm-llama33-70b-4bit-2x4090",
    )
    write_json(
        root / "vllm-baseline.command.json",
        {
            "status": "run",
            "server_cmd": server_cmd,
            "bench_cmd": bench_cmd,
            "same_hardware_required": True,
        },
    )
    preflight = require_vllm_preflight(root, server_cmd)
    write_vllm_metadata(root, repo, cfg, hardware, server_cmd, bench_cmd, preflight)
    proc: subprocess.Popen[str] | None = None
    log_path = root / "vllm-baseline.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                server_cmd,
                cwd=repo,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                start_new_session=True,
            )
        wait_health(base_url, timeout_sec=int(cfg.get("vllm_startup_timeout_sec", 1800)))
        bench = run_with_gpu_samples(
            bench_cmd,
            cwd=repo,
            root=root,
            timeout=int(cfg.get("bench_timeout_sec", 7200)),
            sample_interval_sec=int(cfg.get("bench_gpu_sample_interval_sec", 15)),
            samples_name="vllm-nvidia-smi.bench.samples.jsonl",
            tmp_stem="vllm-baseline",
            sample_phase="vllm-bench",
            during_label="vllm-during",
        )
        write_text(root / "vllm-baseline.stdout", bench.stdout)
        write_text(root / "vllm-baseline.stderr", bench.stderr)
        assert_no_bad_patterns("vLLM bench output", bench.stdout + "\n" + bench.stderr)
        if bench.returncode != 0:
            raise RuntimeError(f"vLLM baseline bench failed rc={bench.returncode}")
        rows = validate_bench_reports(
            root / "vllm-baseline.json",
            cfg,
            require_ci=bool(cfg.get("require_ci", True)),
            label="vLLM baseline",
        )
        update_vllm_metadata_status(root, "pass")
        return {"status": "pass", "rows": list(rows.values())}
    except Exception as exc:
        update_vllm_metadata_status(root, "fail", str(exc))
        raise
    finally:
        terminate_process_group(proc)


def write_diagnostic_vllm_skip(
    root: Path,
    cfg: dict[str, Any],
    ferrum_rows: dict[int, dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    write_json(
        root / "vllm-baseline.command.json",
        {
            "status": "skipped",
            "reason": reason,
            "server_cmd": build_vllm_server_command(cfg),
            "same_hardware_required": True,
        },
    )
    result = {"status": "skipped", "reason": reason}
    write_json(root / "vllm-baseline.json", result)
    comparison = ferrum_only_comparison(
        ferrum_rows,
        [int(c) for c in cfg.get("concurrency_cells", [1, 4, 8, 16])],
        reason,
    )
    write_json(root / "comparison.json", comparison)
    return result


def gate_fail(root: Path, error: str, checks: dict[str, Any]) -> int:
    ensure_failure_artifacts(root, error)
    update_metadata_status(root, "fail", error)
    write_json(
        root / "gate.json",
        {
            "schema_version": 1,
            "status": "fail",
            "lane": LANE,
            "error": error,
            "checks": checks,
            "pass_line": None,
        },
    )
    print(f"G0 CUDA LLAMA33 70B 4BIT 2X4090 GATE FAIL: {error}", file=sys.stderr)
    return 1


def gate_pass(root: Path, checks: dict[str, Any]) -> int:
    pass_line = f"{PASS_LINE_PREFIX}: {root}"
    update_metadata_status(root, "pass")
    write_json(
        root / "gate.json",
        {
            "schema_version": 1,
            "status": "pass",
            "lane": LANE,
            "checks": checks,
            "pass_line": pass_line,
        },
    )
    print(pass_line)
    return 0


def self_test() -> int:
    cfg = {
        "model": "clowman/Llama-3.3-70B-Instruct-GPTQ-Int4",
        "quant_format": "gptq_int4",
        "gpu_devices": [0, 1],
        "distributed_strategy": "layer_split",
        "num_hidden_layers": 80,
        "max_model_len": 8192,
        "kv_max_blocks": 2048,
        "kv_capacity": 2048,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 1024,
        "scheduler_prefill_first_until_active": 8,
    }
    validate_config(cfg)
    assert expected_pipeline_mode_from_config(cfg) == "overlapped"
    with tempfile.TemporaryDirectory(prefix="ferrum-model-manifest-local-") as tmp:
        root = Path(tmp)
        model_dir = root / "model"
        model_dir.mkdir()
        write_json(model_dir / "config.json", {"model_type": "llama"})
        write_json(model_dir / "tokenizer.json", {"version": "1.0"})
        write_json(model_dir / "tokenizer_config.json", {"chat_template": "{{ messages }}"})
        write_json(model_dir / "special_tokens_map.json", {"eos_token": "</s>"})
        write_text(model_dir / "model-00001-of-00001.safetensors", "weights")
        manifest = model_manifest(root, {**cfg, "model": str(model_dir)})
        assert manifest["status"] == "pass"
        assert manifest["resolved_from"] == "local_path"
        assert is_sha256_digest(manifest["config_sha256"])
        assert is_sha256_digest(manifest["tokenizer_sha256"])
        assert is_sha256_digest(manifest["tokenizer_metadata_sha256"])
        assert is_sha256_digest(manifest["weight_manifest_sha256"])
        assert require_release_model_manifest(root, {**cfg, "model": str(model_dir)})[
            "weight_file_count"
        ] == 1
    with tempfile.TemporaryDirectory(prefix="ferrum-model-manifest-hf-") as tmp:
        root = Path(tmp)
        hf_home = root / "hf"
        snapshot = hf_home / "hub" / "models--org--repo" / "snapshots" / "abcdef"
        snapshot.mkdir(parents=True)
        write_json(snapshot / "config.json", {"model_type": "llama"})
        write_json(snapshot / "tokenizer.json", {"version": "1.0"})
        write_json(snapshot / "tokenizer_config.json", {"chat_template": "{{ messages }}"})
        write_text(snapshot / "model.safetensors", "weights")
        previous_hf_home = os.environ.get("HF_HOME")
        os.environ["HF_HOME"] = str(hf_home)
        try:
            manifest = model_manifest(root, {**cfg, "model": "org/repo"})
        finally:
            if previous_hf_home is None:
                os.environ.pop("HF_HOME", None)
            else:
                os.environ["HF_HOME"] = previous_hf_home
        assert manifest["status"] == "pass"
        assert manifest["resolved_from"] == "hf_cache_snapshot"
        assert manifest["model_path"] == str(snapshot)
    bad_mode_cfg = {**cfg, "layer_split_pipeline_mode": "serial"}
    try:
        validate_config(bad_mode_cfg)
        raise AssertionError("invalid layer_split_pipeline_mode unexpectedly passed")
    except RuntimeError as exc:
        assert "layer_split_pipeline_mode" in str(exc)
    assert (
        layer_split_plan_from_config(cfg)
        == "stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79"
    )
    cmd = build_run_command(Path("./target/release/ferrum"), cfg["model"], cfg, Path("/tmp/out"))
    assert "--gpu-devices" in cmd
    assert "0,1" in cmd
    assert "--layer-split-pipeline-mode" in cmd
    assert "overlapped" in cmd
    assert "--effective-config-json" in cmd
    assert "--max-model-len" in cmd
    assert "8192" in cmd
    assert "--kv-capacity" in cmd
    assert "2048" in cmd
    assert "--kv-max-blocks" in cmd
    assert "--max-num-seqs" in cmd
    assert "8" in cmd
    assert "--max-num-batched-tokens" in cmd
    assert "1024" in cmd
    run_probe_input = run_cli_probe_input_text()
    assert run_probe_input.count(RECALL_MARKER) == 1
    assert "Remember the codeword" in run_probe_input
    assert "What codeword did I ask you to remember" in run_probe_input
    assert recall_matches_marker(RECALL_MARKER)
    assert recall_matches_marker(f" {RECALL_MARKER.upper()} ")
    assert not recall_matches_marker(f"Remember the codeword {RECALL_MARKER}.")
    assert "inside brackets" not in run_probe_input
    serve_probe_text = "\n".join(
        message["content"] for message in serve_multiturn_probe_messages()
    )
    assert serve_probe_text.count(RECALL_MARKER) == 1
    assert "Remember the codeword" in serve_probe_text
    assert "What codeword did I ask you to remember" in serve_probe_text
    assert f"[{RECALL_MARKER}]" not in serve_probe_text
    serve_cmd = build_serve_command(
        Path("./target/release/ferrum"), cfg["model"], cfg, Path("/tmp/out")
    )
    assert serve_cmd[1] == "serve"
    assert "--gpu-devices" in serve_cmd
    assert "--layer-split-pipeline-mode" in serve_cmd
    assert "overlapped" in serve_cmd
    assert "--scheduler-prefill-first-until-active" in serve_cmd
    assert "8" in serve_cmd
    effective_doc = {
        "backend": "cuda",
        "requested_gpu_devices": REQUIRED_GPU_DEVICES,
        "selected_gpu_devices": REQUIRED_GPU_DEVICES,
        "cuda_device_count": 2,
        "selected_distributed_strategy": "layer_split",
        "selected_layer_split_plan": layer_split_plan_from_config(cfg),
        "selected_pipeline_mode": "overlapped",
        "selected_microbatch_size": 4,
        "selected_stage_bridge": "host",
        "selected_weight_placement": "layer_split",
        "selected_kv_layout": "paged",
        "selected_attention_impl": "vllm_paged_attn",
        "selected_graph_mode": "disabled",
        "kv_block_count": 2048,
        "selected_max_sequences": 8,
        "selected_max_model_len": 8192,
        "selected_kv_capacity": 2048,
        "selected_max_batched_tokens": 1024,
        "model_capabilities": {},
    }
    with tempfile.TemporaryDirectory(prefix="ferrum-llama33-effective-config-") as tmp:
        effective_path = Path(tmp) / "effective_config.json"
        write_json(effective_path, effective_doc)
        require_effective_config(effective_path, "selftest-overlapped", cfg)

        batch_cfg = {**cfg, "layer_split_pipeline_mode": "batch"}
        effective_doc["selected_pipeline_mode"] = "batch"
        effective_doc["selected_microbatch_size"] = 8
        write_json(effective_path, effective_doc)
        require_effective_config(effective_path, "selftest-batch", batch_cfg)

        preset_cfg = {
            **batch_cfg,
            "expected_runtime_preset": "qwen25_72b_gptq_int4_2x4090_layer_split",
            "expected_max_model_len": 8192,
            "expected_kv_capacity": 2048,
            "expected_kv_max_blocks": 2048,
            "expected_max_num_seqs": 8,
            "expected_max_num_batched_tokens": 1024,
            "expected_scheduler_prefill_first_until_active": 8,
            "require_default_runtime_sources": True,
            "run_max_model_len": 8192,
            "run_kv_capacity": 2048,
            "run_kv_max_blocks": 2048,
            "run_max_num_seqs": 8,
            "run_max_num_batched_tokens": 1024,
        }
        try:
            require_effective_config(effective_path, "selftest-run-raw-preset", preset_cfg)
            raise AssertionError("run effective config unexpectedly accepted serve preset expectations")
        except RuntimeError as exc:
            assert "preset must be" in str(exc)
        require_effective_config(
            effective_path,
            "selftest-run-preset-stripped",
            run_effective_config_validation_config(preset_cfg),
        )
    with tempfile.TemporaryDirectory(prefix="ferrum-runtime-artifact-promotion-") as tmp:
        root = Path(tmp)
        run_doc = {**effective_doc, "selected_max_sequences": 8}
        serve_doc = {**effective_doc, "selected_max_sequences": 16, "selected_microbatch_size": 16}
        write_json(root / "run.effective_config.json", run_doc)
        write_json(root / "serve.effective_config.json", serve_doc)
        write_text(root / "run.decision_trace.jsonl", "run\n")
        write_text(root / "serve.decision_trace.jsonl", "serve\n")
        normalize_runtime_artifacts(root)
        assert load_json(root / "effective_config.json")["selected_max_sequences"] == 8
        promote_serve_runtime_artifacts(root)
        assert load_json(root / "effective_config.json")["selected_max_sequences"] == 16
        assert (root / "decision_trace.jsonl").read_text() == "serve\n"
    with tempfile.TemporaryDirectory(prefix="ferrum-llama33-correctness-") as tmp:
        root = Path(tmp)
        correctness_checks = {
            "run": {"status": "pass", "assistant_turns": 2, "has_precise_recall": True},
            "serve_correctness": {"status": "pass", "contains_expected_answer": True},
            "serve_multiturn": {"status": "pass", "has_precise_recall": True},
            "serve_structured_output": {"status": "pass", "json_ok": True, "schema_ok": True},
            "serve_tool_call": {"status": "pass", "tool_call_ok": True},
            "serve_streaming": {"status": "pass", "done_count": 1, "usage_chunk_count": 1},
        }
        write_goal_correctness_artifact(root, correctness_checks)
        from layer_split_perf_goal_gate import validate_correctness_artifact

        validate_correctness_artifact(root)
    bench_cmd = build_bench_serve_command(Path("./target/release/ferrum"), cfg, Path("/tmp/out"))
    assert "--fail-on-error" in bench_cmd
    assert "--require-ci" in bench_cmd
    assert "--n-repeats" in bench_cmd
    smoke_cfg = {**cfg, "require_ci": False, "n_repeats": 1, "concurrency_cells": [1, 4]}
    smoke_bench_cmd = build_bench_serve_command(
        Path("./target/release/ferrum"),
        smoke_cfg,
        Path("/tmp/out"),
    )
    assert "--fail-on-error" in smoke_bench_cmd
    assert "--require-ci" not in smoke_bench_cmd
    hardware = query_hardware(Path("/tmp"), "")
    assert "gpu_names" in hardware
    with tempfile.TemporaryDirectory(prefix="ferrum-llama33-hardware-") as tmp:
        root = Path(tmp)
        gpu_rows = [
            {
                "index": 0,
                "name": "NVIDIA GeForce RTX 4090",
                "uuid": "GPU-selftest-0",
                "driver_version": "550.54.15",
                "memory_total_mib": 24564,
                "memory_used_mib": 1234,
                "utilization_gpu_percent": 10,
                "utilization_memory_percent": 8,
                "pcie_link_gen_current": 4,
                "pcie_link_width_current": 16,
            },
            {
                "index": 1,
                "name": "NVIDIA GeForce RTX 4090",
                "uuid": "GPU-selftest-1",
                "driver_version": "550.54.15",
                "memory_total_mib": 24564,
                "memory_used_mib": 2345,
                "utilization_gpu_percent": 12,
                "utilization_memory_percent": 9,
                "pcie_link_gen_current": 4,
                "pcie_link_width_current": 16,
            },
        ]
        hardware_doc = {
            "schema_version": 1,
            "status": "pass",
            "cuda_device_count": 2,
            "gpu_names": ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"],
            "gpu_uuids": ["GPU-selftest-0", "GPU-selftest-1"],
            "gpu_utilization_percent": [10, 12],
            "gpu_memory_utilization_percent": [8, 9],
            "pcie_link_gen_current": [4, 4],
            "pcie_link_width_current": [16, 16],
            "gpus": gpu_rows,
        }
        for label in ["before", "during", "after"]:
            write_json(
                root / f"nvidia-smi.{label}.json",
                {
                    "schema_version": 1,
                    "status": "pass",
                    "gpus": gpu_rows,
                },
            )
        write_text(
            root / "nvidia-smi.bench.samples.jsonl",
            "".join(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "pass",
                        "bench_concurrency": concurrency,
                        "bench_concurrency_sweep": [1, 4, 8, 16],
                        "gpus": [
                            {
                                **gpu_rows[0],
                                "utilization_gpu_percent": 80 + concurrency,
                            },
                            {
                                **gpu_rows[1],
                                "utilization_gpu_percent": 83 + concurrency,
                            },
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
                for concurrency in [1, 4, 8, 16]
            ),
        )
        summary = validate_structured_hardware_evidence(
            root, hardware_doc, [1, 4, 8, 16]
        )
        assert summary["status"] == "pass"
        assert summary["pcie_link_width_current"] == [16, 16]
        assert summary["bench_max_gpu_utilization_percent"] == [96, 99]
        assert summary["bench_sample_count_by_concurrency"] == {
            1: 1,
            4: 1,
            8: 1,
            16: 1,
        }
        write_text(
            root / "nvidia-smi.bench.samples.jsonl",
            "".join(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "pass",
                        "bench_concurrency": concurrency,
                        "bench_concurrency_sweep": [1, 4, 8, 16],
                        "gpus": [
                            {
                                **gpu_rows[0],
                                "utilization_gpu_percent": 80 + concurrency,
                            },
                            {
                                **gpu_rows[1],
                                "utilization_gpu_percent": 83 + concurrency,
                            },
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
                for concurrency in [1, 4, 8]
            ),
        )
        try:
            validate_structured_hardware_evidence(root, hardware_doc, [1, 4, 8, 16])
            raise AssertionError("missing c16 GPU sample unexpectedly passed")
        except RuntimeError as exc:
            assert "missing concurrency cells" in str(exc)
    with tempfile.TemporaryDirectory(prefix="ferrum-llama33-vllm-metadata-") as tmp:
        root = Path(tmp)
        hardware_doc = {
            "cuda_version": "12.4",
            "driver_version": "550.54.15",
            "gpu_names": ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"],
            "gpu_uuids": ["GPU-selftest-0", "GPU-selftest-1"],
        }
        preflight = require_vllm_preflight(root, [sys.executable, "--version"])
        assert preflight["status"] == "pass"
        assert preflight["binary_path"] == sys.executable
        assert isinstance(preflight["binary_sha256"], str)
        assert len(preflight["binary_sha256"]) == 64
        metadata = write_vllm_metadata(
            root,
            Path.cwd(),
            cfg,
            hardware_doc,
            [sys.executable, "--version"],
            [str(Path("./target/release/ferrum")), "bench-serve"],
            preflight,
        )
        assert metadata["engine"] == "vllm"
        assert metadata["model_id"] == cfg["model"]
        assert metadata["gpu_uuids"] == hardware_doc["gpu_uuids"]
        assert metadata["binary_sha256"] == preflight["binary_sha256"]
        assert load_json(root / "vllm-baseline.preflight.json")["status"] == "pass"
        update_vllm_metadata_status(root, "pass")
        assert load_json(root / "vllm-baseline.metadata.json")["status"] == "pass"
    with tempfile.TemporaryDirectory(prefix="ferrum-llama33-vllm-preflight-") as tmp:
        root = Path(tmp)
        previous_path = os.environ.get("PATH", "")
        empty_bin = root / "empty-bin"
        empty_bin.mkdir()
        try:
            os.environ["PATH"] = str(empty_bin)
            try:
                require_vllm_preflight(root, ["vllm", "serve"])
                raise AssertionError("missing vllm preflight unexpectedly passed")
            except RuntimeError as exc:
                assert "vllm executable not found" in str(exc)
            failed = load_json(root / "vllm-baseline.preflight.json")
            assert failed["status"] == "fail"
            assert failed["binary_sha256"] is None
        finally:
            os.environ["PATH"] = previous_path
    with tempfile.TemporaryDirectory(prefix="ferrum-llama33-source-gate-") as tmp:
        root = Path(tmp)
        ensure_failure_artifacts(root, "selftest failure")
        write_json(
            root / "gate.json",
            {
                "schema_version": 1,
                "status": "fail",
                "lane": LANE,
                "pass_line": None,
            },
        )
        missing = sorted(name for name in REQUIRED_ARTIFACT_FILES if not (root / name).exists())
        assert not missing, missing
    with tempfile.TemporaryDirectory(prefix="ferrum-llama33-bench-") as tmp:
        root = Path(tmp)
        bench_report = {
            "concurrency": 1,
            "completed_per_run": [2, 2, 2],
            "errored_per_run": [0, 0, 0],
            "n_repeats": 3,
            "output_token_count_source": "usage",
            "output_throughput_tps": {"mean": 70.0},
            "ttft_ms": {"p50": {"mean": 120.0}},
            "tpot_ms": {"p50": {"mean": 12.0}},
            "e2e_ms": {"p95": {"mean": 800.0}},
            "bad_output_per_run": [0, 0, 0],
            "malformed_stream_per_run": [0, 0, 0],
            "missing_done_per_run": [0, 0, 0],
            "duplicate_done_per_run": [0, 0, 0],
            "zero_output_tokens_per_run": [0, 0, 0],
            "stream_bulk_flush_per_run": [0, 0, 0],
            "http_500_per_run": [0, 0, 0],
            "panic_per_run": [0, 0, 0],
        }
        vllm_report = {
            **bench_report,
            "output_throughput_tps": {"mean": 100.0},
            "ttft_ms": {"p50": {"mean": 100.0}},
            "tpot_ms": {"p50": {"mean": 10.0}},
        }
        write_json(root / "ferrum.json", [bench_report])
        write_json(root / "vllm.json", [vllm_report])
        bench_cfg = {**cfg, "concurrency_cells": [1], "require_ci": True}
        ferrum_rows = validate_bench_reports(
            root / "ferrum.json",
            bench_cfg,
            require_ci=True,
            label="selftest ferrum",
        )
        vllm_rows = validate_bench_reports(
            root / "vllm.json",
            bench_cfg,
            require_ci=True,
            label="selftest vllm",
        )
        comparison = compare_benchmarks(ferrum_rows, vllm_rows, [1])
        assert comparison["cells"]["c1"]["status"] == "pass"
        ferrum_only = ferrum_only_comparison(ferrum_rows, [1], "selftest")
        assert ferrum_only["status"] == "pass"
        assert ferrum_only["mode"] == "ferrum_only"
        skip = write_diagnostic_vllm_skip(root, bench_cfg, ferrum_rows, "selftest")
        assert skip["status"] == "skipped"
    print("G0 CUDA LLAMA33 70B 4BIT 2X4090 GATE SELFTEST PASS")
    return 0


def main() -> int:
    global LANE, PASS_LINE_PREFIX

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ferrum-bin", type=Path, default=Path("./target/release/ferrum"))
    parser.add_argument(
        "--lane-name",
        default=LANE,
        choices=[LANE, SMOKE_LANE, QWEN72B_LANE, QWEN72B_SMOKE_LANE],
    )
    args = parser.parse_args()
    LANE = args.lane_name
    PASS_LINE_PREFIX = f"G0 SOURCE {LANE} PASS"
    if args.self_test:
        return self_test()
    if args.config is None or args.out is None:
        parser.error("--config and --out are required")

    repo = Path(__file__).resolve().parents[2]
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    proc: subprocess.Popen[str] | None = None
    try:
        cfg = load_json(args.config)
        validate_config(cfg)
        checks["config"] = {"status": "pass", "path": str(args.config)}
        before_smi = capture_nvidia_smi(root, "before")
        hardware = write_hardware(root, repo, before_smi)
        validate_hardware_for_gate(hardware)
        checks["hardware"] = {
            "status": "pass",
            "cuda_device_count": hardware.get("cuda_device_count"),
            "gpu_names": hardware.get("gpu_names", []),
        }
        if cfg.get("run_vllm_baseline", True):
            checks["vllm_preflight"] = require_vllm_preflight(
                root,
                build_vllm_server_command(cfg),
            )
        model_manifest(root, cfg)
        write_planned_command_artifacts(root, args.ferrum_bin, cfg)
        write_metadata(root, repo, args.ferrum_bin, cfg, hardware, sys.argv)
        checks["run"] = run_cli_probe(root, repo, args.ferrum_bin, cfg)

        model = str(cfg["model"])
        port = int(cfg.get("port", 19400))
        base_url = f"http://127.0.0.1:{port}"
        serve_cmd = build_serve_command(args.ferrum_bin, model, cfg, root)
        write_json(
            root / "serve.command.json",
            {
                "status": "run",
                "cmd": serve_cmd,
                "expected_gpu_devices": REQUIRED_GPU_DEVICES,
                "expected_distributed_strategy": "layer_split",
                "expected_layer_split_plan": layer_split_plan_from_config(cfg),
                "expected_layer_split_pipeline_mode": expected_pipeline_mode_from_config(cfg),
                "scheduler_prefill_first_until_active": cfg.get(
                    "scheduler_prefill_first_until_active"
                ),
            },
        )
        serve_log = root / "serve.log"
        with serve_log.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                serve_cmd,
                cwd=repo,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                start_new_session=True,
            )
        wait_health(base_url, timeout_sec=int(cfg.get("serve_startup_timeout_sec", 1800)))
        capture_nvidia_smi(root, "serve-ready")
        require_effective_config(root / "serve.effective_config.json", "serve", cfg)
        promote_serve_runtime_artifacts(root)
        checks["serve_health"] = capture_health(root, base_url)
        checks["serve_models"] = capture_models(root, base_url, model)
        checks["serve_correctness"] = serve_correctness(root, base_url, model)
        checks["serve_multiturn"] = serve_multiturn(root, base_url, model)
        checks["serve_structured_output"] = serve_structured_output(root, base_url, model)
        checks["serve_tool_call"] = serve_tool_call(root, base_url, model)
        checks["serve_streaming"] = serve_streaming(root, base_url, model)
        checks["concurrency_quality"] = run_concurrency_gate(root, base_url, model, cfg)
        checks["bench_serve"] = run_ferrum_bench_gate(root, repo, args.ferrum_bin, cfg, base_url, model)
        checks["serve_health_after"] = capture_health(
            root, base_url, filename="serve.health.after.json"
        )

        terminate_process_group(proc)
        proc = None
        log_text = serve_log.read_text(errors="replace")
        assert_no_bad_patterns("serve.log", log_text)
        checks["goal_correctness"] = write_goal_correctness_artifact(root, checks)
        checks["model_manifest"] = require_release_model_manifest(root, cfg)

        ferrum_rows = validate_bench_reports(
            root / "bench-serve.json",
            cfg,
            require_ci=bool(cfg.get("require_ci", True)),
            label="ferrum bench-serve",
        )
        if cfg.get("run_vllm_baseline", True):
            checks["vllm_baseline"] = run_vllm_baseline_gate(
                root,
                repo,
                args.ferrum_bin,
                cfg,
                model,
                hardware,
            )
            vllm_rows = validate_bench_reports(
                root / "vllm-baseline.json",
                cfg,
                require_ci=bool(cfg.get("require_ci", True)),
                label="vLLM baseline",
            )
            comparison = compare_benchmarks(
                ferrum_rows,
                vllm_rows,
                [int(c) for c in cfg.get("concurrency_cells", [1, 4, 8, 16])],
            )
            write_json(root / "comparison.json", comparison)
            checks["comparison"] = comparison
        else:
            reason = (
                "config sets run_vllm_baseline=false; Ferrum-only service evidence is required "
                "for this lane and no vLLM-relative performance claim is made"
            )
            checks["vllm_baseline"] = write_diagnostic_vllm_skip(root, cfg, ferrum_rows, reason)
            checks["comparison"] = load_json(root / "comparison.json")

        require_effective_config(root / "effective_config.json", "top-level", cfg)
    except Exception as exc:
        terminate_process_group(proc, sig=signal.SIGKILL)
        capture_nvidia_smi(root, "after")
        return gate_fail(root, str(exc), checks)
    finally:
        terminate_process_group(proc, sig=signal.SIGKILL)
        if not (root / "nvidia-smi.during.txt").exists():
            capture_nvidia_smi(root, "during")

    capture_nvidia_smi(root, "after")
    checks["hardware_snapshots"] = validate_structured_hardware_evidence(
        root,
        hardware,
        [int(c) for c in cfg.get("concurrency_cells", [1, 4, 8, 16])],
    )
    return gate_pass(root, checks)


if __name__ == "__main__":
    raise SystemExit(main())
