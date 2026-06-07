#!/usr/bin/env python3
"""CUDA source gate scaffold for Llama 3.3 70B 4bit on 2x RTX 4090.

This runner intentionally refuses to PASS until the product path can execute
the typed 2-GPU layer_split request and the rest of the correctness/perf matrix
is implemented. It still writes failure artifacts for the first product command
so CUDA development can move from a concrete failing point.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LANE = "g0_cuda2x4090_llama33_70b_4bit"
PASS_LINE_PREFIX = f"G0 SOURCE {LANE} PASS"
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
]
LAYER_SPLIT_PLAN = "stage0:cuda:0:layers=auto;stage1:cuda:1:layers=auto"
REQUIRED_ARTIFACT_FILES = {
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


def git_output(args: list[str], repo: Path) -> str:
    proc = run(["git", *args], cwd=repo, timeout=30)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def capture_nvidia_smi(root: Path, label: str) -> str:
    proc = run(["nvidia-smi"], cwd=root, timeout=30)
    body = proc.stdout if proc.returncode == 0 else proc.stderr
    text = body or f"nvidia-smi rc={proc.returncode}\n"
    write_text(root / f"nvidia-smi.{label}.txt", text)
    return text


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
    proc = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        cwd=repo,
        timeout=30,
    )
    if proc.returncode != 0:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "error": proc.stderr.strip() or proc.stdout.strip() or f"nvidia-smi rc={proc.returncode}",
            "cuda_device_count": 0,
            "cuda_version": versions["cuda_version"],
            "driver_version": versions["driver_version"],
            "gpu_names": [],
            "gpu_uuids": [],
            "gpus": [],
        }

    rows: list[dict[str, Any]] = []
    reader = csv.reader(proc.stdout.splitlines())
    for row in reader:
        parts = [part.strip() for part in row]
        if len(parts) < 6:
            continue
        index, name, uuid, driver_version, memory_total, memory_used = parts[:6]
        rows.append(
            {
                "index": int(index) if index.isdigit() else index,
                "name": name,
                "uuid": uuid,
                "driver_version": driver_version,
                "memory_total_mib": int(memory_total) if memory_total.isdigit() else memory_total,
                "memory_used_mib": int(memory_used) if memory_used.isdigit() else memory_used,
            }
        )

    driver_version = rows[0]["driver_version"] if rows else versions["driver_version"]
    return {
        "schema_version": 1,
        "status": "pass" if len(rows) == 2 else "fail",
        "cuda_device_count": len(rows),
        "cuda_version": versions["cuda_version"],
        "driver_version": driver_version,
        "gpu_names": [str(row["name"]) for row in rows],
        "gpu_uuids": [str(row["uuid"]) for row in rows],
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
    model = cfg.get("model")
    if not isinstance(model, str) or not model:
        raise RuntimeError("config model must be a non-empty model id/path")
    quant = str(cfg.get("quant_format", "")).lower()
    if not any(marker in quant for marker in ("gptq", "awq", "q4")):
        raise RuntimeError("config quant_format must identify a 4bit format")


def model_manifest(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    model = str(cfg["model"])
    path = Path(model)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "pending_model_resolution",
        "model": model,
        "quant_format": cfg["quant_format"],
        "config_sha256": None,
        "tokenizer_sha256": None,
        "weight_manifest_sha256": None,
        "files": [],
    }
    if not path.exists():
        write_json(root / "model_manifest.json", manifest)
        return manifest

    files: list[dict[str, Any]] = []
    interesting_suffixes = {".json", ".model", ".safetensors", ".gguf"}
    candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    base = path.parent if path.is_file() else path
    for file in candidates:
        if file.suffix not in interesting_suffixes:
            continue
        rel = file.relative_to(base).as_posix()
        digest = sha256(file)
        files.append({"path": rel, "size_bytes": file.stat().st_size, "sha256": digest})
    manifest["status"] = "pass"
    manifest["files"] = files
    for item in files:
        file_name = Path(str(item["path"])).name
        if file_name == "config.json":
            manifest["config_sha256"] = item["sha256"]
        elif file_name in {"tokenizer.json", "tokenizer.model"}:
            manifest["tokenizer_sha256"] = item["sha256"]
    weight_items = [item for item in files if str(item["path"]).endswith((".safetensors", ".gguf"))]
    weight_payload = json.dumps(weight_items, sort_keys=True).encode("utf-8")
    manifest["weight_manifest_sha256"] = hashlib.sha256(weight_payload).hexdigest()
    write_json(root / "model_manifest.json", manifest)
    return manifest


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
        "256",
        "--output-format",
        "jsonl",
        "--effective-config-json",
        str(root / "run.effective_config.json"),
        "--decision-trace-jsonl",
        str(root / "run.decision_trace.jsonl"),
    ]
    if cfg.get("max_model_len") is not None:
        # `ferrum run` does not yet expose --max-model-len; preserve the
        # intended value in run.command.json for the 70B lane artifact.
        pass
    return cmd


def build_serve_command(ferrum_bin: Path, model: str, cfg: dict[str, Any], root: Path) -> list[str]:
    return [
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


def build_bench_serve_command(ferrum_bin: Path, cfg: dict[str, Any], root: Path) -> list[str]:
    port = int(cfg.get("port", 19400))
    return [
        str(ferrum_bin),
        "bench-serve",
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--model",
        str(cfg["model"]),
        "--tokenizer",
        str(cfg.get("tokenizer_path") or cfg.get("tokenizer") or cfg["model"]),
        "--fail-on-error",
        "--require-ci",
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
    ]


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
    write_json(
        root / "serve.command.json",
        {
            "status": "planned",
            "cmd": serve_cmd,
            "expected_gpu_devices": REQUIRED_GPU_DEVICES,
            "expected_distributed_strategy": "layer_split",
        },
    )
    write_json(
        root / "bench-serve.command.json",
        {
            "status": "planned",
            "cmd": bench_cmd,
            "required_flags": ["--fail-on-error", "--require-ci", "--seed", "--n-repeats"],
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
        "build_features": ["cuda", "vllm-moe-marlin", "vllm-paged-attn-v2", "fa2-source"],
        "cuda_version": hardware.get("cuda_version", "unknown"),
        "driver_version": hardware.get("driver_version", "unknown"),
        "gpu_names": hardware.get("gpu_names", []),
        "gpu_uuids": hardware.get("gpu_uuids", []),
        "requested_gpu_devices": REQUIRED_GPU_DEVICES,
        "selected_gpu_devices": REQUIRED_GPU_DEVICES,
        "model_id": cfg["model"],
        "quant_format": cfg["quant_format"],
        "distributed_strategy": "layer_split",
        "layer_split_plan": LAYER_SPLIT_PLAN,
        "sanitized_env": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("FERRUM_") or key in {"CUDA_VISIBLE_DEVICES", "HF_HOME"}
        },
    }
    write_json(root / "metadata.json", metadata)
    return metadata


def assert_no_bad_patterns(label: str, text: str) -> None:
    lower = text.lower()
    for pattern in BAD_PATTERNS:
        if pattern.lower() in lower:
            raise RuntimeError(f"forbidden pattern {pattern!r} in {label}")


def copy_if_present(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def normalize_runtime_artifacts(root: Path) -> None:
    copy_if_present(root / "run.effective_config.json", root / "effective_config.json")
    copy_if_present(root / "run.decision_trace.jsonl", root / "decision_trace.jsonl")


def ensure_failure_artifacts(root: Path, error: str) -> None:
    json_defaults = {
        "metadata.json": {"status": "fail", "error": error, "lane": LANE},
        "effective_config.json": {"status": "fail", "error": error},
        "hardware.json": {"status": "fail", "error": error, "gpu_names": [], "gpus": []},
        "model_manifest.json": {"status": "fail", "error": error},
        "run.command.json": {"status": "fail", "error": error},
        "run.effective_config.json": {"status": "fail", "error": error},
        "serve.command.json": {"status": "not_run", "blocked_by": error},
        "serve.effective_config.json": {"status": "not_run", "blocked_by": error},
        "serve.health.json": {"status": "not_run", "blocked_by": error},
        "serve.models.json": {"status": "not_run", "blocked_by": error},
        "serve.correctness.json": {"status": "not_run", "blocked_by": error},
        "serve.multiturn.json": {"status": "not_run", "blocked_by": error},
        "serve.structured_output.json": {"status": "not_run", "blocked_by": error},
        "serve.tool_call.json": {"status": "not_run", "blocked_by": error},
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


def run_cli_probe(root: Path, repo: Path, ferrum_bin: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    input_text = "\n".join(
        [
            "请记住短语 ferrum-blue。只回答 OK。",
            "第一条用户消息里的 ferrum 开头短语是什么？只输出短语。",
            "/bye",
            "",
        ]
    )
    cmd = build_run_command(ferrum_bin, cfg["model"], cfg, root)
    write_json(
        root / "run.command.json",
        {
            "cmd": cmd,
            "expected_gpu_devices": REQUIRED_GPU_DEVICES,
            "expected_distributed_strategy": "layer_split",
            "config": {
                "max_model_len": cfg.get("max_model_len"),
                "kv_capacity": cfg.get("kv_capacity"),
                "max_num_seqs": cfg.get("max_num_seqs"),
                "max_num_batched_tokens": cfg.get("max_num_batched_tokens"),
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
    return {"status": "pass", "returncode": proc.returncode}


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
    }
    validate_config(cfg)
    cmd = build_run_command(Path("./target/release/ferrum"), cfg["model"], cfg, Path("/tmp/out"))
    assert "--gpu-devices" in cmd
    assert "0,1" in cmd
    assert "--effective-config-json" in cmd
    serve_cmd = build_serve_command(Path("./target/release/ferrum"), cfg["model"], cfg, Path("/tmp/out"))
    assert serve_cmd[1] == "serve"
    assert "--gpu-devices" in serve_cmd
    bench_cmd = build_bench_serve_command(Path("./target/release/ferrum"), cfg, Path("/tmp/out"))
    assert "--fail-on-error" in bench_cmd
    assert "--require-ci" in bench_cmd
    assert "--n-repeats" in bench_cmd
    hardware = query_hardware(Path("/tmp"), "")
    assert "gpu_names" in hardware
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
    print("G0 CUDA LLAMA33 70B 4BIT 2X4090 GATE SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ferrum-bin", type=Path, default=Path("./target/release/ferrum"))
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.config is None or args.out is None:
        parser.error("--config and --out are required")

    repo = Path(__file__).resolve().parents[2]
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    try:
        cfg = load_json(args.config)
        validate_config(cfg)
        checks["config"] = {"status": "pass", "path": str(args.config)}
        before_smi = capture_nvidia_smi(root, "before")
        hardware = write_hardware(root, repo, before_smi)
        checks["hardware"] = {
            "status": hardware.get("status"),
            "cuda_device_count": hardware.get("cuda_device_count"),
            "gpu_names": hardware.get("gpu_names", []),
        }
        model_manifest(root, cfg)
        write_planned_command_artifacts(root, args.ferrum_bin, cfg)
        write_metadata(root, repo, args.ferrum_bin, cfg, hardware, sys.argv)
        checks["run"] = run_cli_probe(root, repo, args.ferrum_bin, cfg)

        raise RuntimeError(
            "serve correctness, concurrency quality, bench-serve, and vLLM baseline checks are not implemented for this lane yet"
        )
    except Exception as exc:
        capture_nvidia_smi(root, "after")
        return gate_fail(root, str(exc), checks)
    finally:
        capture_nvidia_smi(root, "during")

    return gate_pass(root, checks)


if __name__ == "__main__":
    raise SystemExit(main())
