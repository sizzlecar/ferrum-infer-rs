#!/usr/bin/env python3
"""Run the W3 Qwen3.5 CUDA release-grade lane on a 1x RTX 4090 host.

The script composes the existing W3 gates instead of reimplementing them:

1. build the CUDA Ferrum binary while optionally prefetching the HF snapshot;
2. run real product-path correctness for `ferrum run` and `ferrum serve`;
3. package L2 from the real known-answer report;
4. start a measured `ferrum serve` instance for L4 and fixed-output L5;
5. package L5 and build the final release-grade manifest.

No step uses FERRUM_* hidden environment overrides. Runtime tuning is expressed
through typed CLI flags and every command is written into the artifact tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GOAL_DOC = "docs/goals/model-coverage-2026-06-12/W3_QWEN35_RELEASE_GRADE_GOAL.md"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
DEFAULT_QUANTIZATION = "hf-gptq-int4"
DEFAULT_BASE_CONFIG = (
    REPO_ROOT / "docs/goals/model-coverage-2026-06-12/w3_qwen35_current_evidence_config.json"
)
DEFAULT_SHAREGPT = (
    REPO_ROOT
    / "docs/goals/model-coverage-2026-06-12/artifacts/"
    / "w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl"
)
DEFAULT_DATASET_ID = "w3/ascii-sharegpt-100-seed9271"
DEFAULT_DATASET_SHA = "58d5721d8389d7ed9ec4b8b2dbd8797faa61641c6ba023dd150a1a9d93c0a01e"
CUDA_FEATURES = "cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source"
REQUIRED_CELLS = "1,4,16,32"
EXPECTED_OUTPUT_LEN = 128


class LaneError(Exception):
    pass


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def shlex_join(cmd: list[str]) -> str:
    return shlex.join([str(part) for part in cmd])


def sanitize_env_for_record(env: dict[str, str]) -> dict[str, Any]:
    return {
        "HF_HOME": env.get("HF_HOME"),
        "HF_XET_HIGH_PERFORMANCE": env.get("HF_XET_HIGH_PERFORMANCE"),
        "HF_TOKEN_present": bool(env.get("HF_TOKEN")),
        "NO_COLOR": env.get("NO_COLOR"),
        "FERRUM_hidden_env_keys_present": sorted(k for k in env if k.startswith("FERRUM_")),
    }


def child_env(args: argparse.Namespace) -> tuple[dict[str, str], list[str]]:
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["HF_HOME"] = str(args.hf_home)
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    scrubbed = sorted(k for k in env if k.startswith("FERRUM_"))
    for key in scrubbed:
        env.pop(key, None)
    return env, scrubbed


def run_capture(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        timeout=timeout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def run_logged(
    *,
    label: str,
    cmd: list[str],
    out_dir: Path,
    env: dict[str, str],
    timeout: float | None,
    cwd: Path = REPO_ROOT,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / f"{label}.command.json",
        {
            "cmd": cmd,
            "cwd": str(cwd),
            "env": sanitize_env_for_record(env),
            "started_at": iso_now(),
        },
    )
    write_text(out_dir / f"{label}.command.txt", shlex_join(cmd) + "\n")
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            input=input_text,
            timeout=timeout,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        write_text(out_dir / f"{label}.stdout.txt", exc.stdout or "")
        write_text(out_dir / f"{label}.stderr.txt", exc.stderr or "")
        write_json(
            out_dir / f"{label}.result.json",
            {"returncode": None, "timed_out": True, "finished_at": iso_now()},
        )
        raise LaneError(f"{label} timed out after {timeout}s") from exc
    write_text(out_dir / f"{label}.stdout.txt", proc.stdout)
    write_text(out_dir / f"{label}.stderr.txt", proc.stderr)
    write_json(
        out_dir / f"{label}.result.json",
        {"returncode": proc.returncode, "timed_out": False, "finished_at": iso_now()},
    )
    if proc.returncode != 0:
        raise LaneError(f"{label} failed with rc={proc.returncode}: {out_dir}")
    return proc


def git_output(args: list[str]) -> str:
    proc = run_capture(["git", *args], cwd=REPO_ROOT)
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def git_summary() -> dict[str, Any]:
    tracked = [
        line
        for line in git_output(["status", "--short", "--untracked-files=no"]).splitlines()
        if line.strip()
    ]
    untracked = [
        line
        for line in git_output(["ls-files", "--others", "--exclude-standard"]).splitlines()
        if line.strip()
    ]
    return {
        "sha": git_output(["rev-parse", "HEAD"]),
        "dirty": bool(tracked or untracked),
        "status_short": "\n".join([*tracked, *untracked]),
        "tracked_status_short": tracked,
        "untracked_count": len(untracked),
        "untracked_sample": untracked[:20],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    try:
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def http_get(url: str, timeout: float) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", "replace")


def wait_for_health(proc: subprocess.Popen[str], base_url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise LaneError(f"ferrum serve exited early with rc={proc.returncode}")
        try:
            status, _ = http_get(f"{base_url}/health", timeout=2.0)
            if 200 <= status < 300:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise LaneError(f"ferrum serve did not become healthy before {timeout_seconds}s")


def server_model_id(base_url: str, timeout: float, fallback: str) -> str:
    status, body = http_get(f"{base_url}/v1/models", timeout=timeout)
    if not 200 <= status < 300:
        return fallback
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return fallback
    models = data.get("data")
    if isinstance(models, list) and models and isinstance(models[0], dict):
        model_id = models[0].get("id")
        if isinstance(model_id, str) and model_id:
            return model_id
    return fallback


def wait_for_openai_server(
    proc: subprocess.Popen[str],
    base_url: str,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise LaneError(f"OpenAI-compatible server exited early with rc={proc.returncode}")
        try:
            status, _ = http_get(f"{base_url}/v1/models", timeout=2.0)
            if 200 <= status < 300:
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise LaneError(f"OpenAI-compatible server did not become ready before {timeout_seconds}s")


def gpu_snapshot(out_dir: Path, env: dict[str, str], require_gpu: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        if require_gpu:
            raise LaneError("nvidia-smi not found on this host")
        write_text(out_dir / "nvidia_smi_before.txt", "nvidia-smi not found\n")
        write_text(out_dir / "nvidia_smi_query.txt", "nvidia-smi not found\n")
    else:
        proc = run_capture([nvidia_smi], env=env, timeout=30)
        write_text(out_dir / "nvidia_smi_before.txt", proc.stdout + proc.stderr)
        query = run_capture(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version,cuda_version",
                "--format=csv,noheader",
            ],
            env=env,
            timeout=30,
        )
        write_text(out_dir / "nvidia_smi_query.txt", query.stdout + query.stderr)
        if require_gpu and proc.returncode != 0:
            raise LaneError(f"nvidia-smi failed with rc={proc.returncode}")
    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        proc = run_capture([nvcc, "--version"], env=env, timeout=30)
        write_text(out_dir / "nvcc_version.txt", proc.stdout + proc.stderr)
    else:
        write_text(out_dir / "nvcc_version.txt", "nvcc not found\n")
    write_json(
        out_dir / "hardware.json",
        {
            "status": "pass",
            "generated_at": iso_now(),
            "nvidia_smi": repo_path(out_dir / "nvidia_smi_before.txt"),
            "nvidia_smi_query": repo_path(out_dir / "nvidia_smi_query.txt"),
            "nvcc_version": repo_path(out_dir / "nvcc_version.txt"),
        },
    )


def model_cache_dir_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def resolve_tokenizer_path(model: str, hf_home: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not (explicit / "tokenizer.json").is_file() and explicit.name != "tokenizer.json":
            raise LaneError(f"--tokenizer must be a tokenizer.json file or snapshot dir: {explicit}")
        return explicit
    model_path = Path(model)
    if model_path.exists():
        if model_path.is_dir() and (model_path / "tokenizer.json").is_file():
            return model_path
        if model_path.is_file() and model_path.name == "tokenizer.json":
            return model_path
    snapshots = hf_home / "hub" / model_cache_dir_name(model) / "snapshots"
    candidates = sorted(snapshots.glob("*/tokenizer.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise LaneError(
            "could not find tokenizer.json; pass --tokenizer or prefetch the model under "
            f"HF_HOME={hf_home}"
        )
    return candidates[-1].parent


def runtime_flags(args: argparse.Namespace) -> list[str]:
    flags = ["--backend", args.backend]
    optional = [
        ("--gpu-devices", args.gpu_devices),
        ("--gpu-memory-utilization", args.gpu_memory_utilization),
        ("--max-model-len", args.max_model_len),
        ("--max-num-seqs", args.max_num_seqs),
        ("--max-num-batched-tokens", args.max_num_batched_tokens),
        ("--kv-capacity", args.kv_capacity),
        ("--kv-max-blocks", args.kv_max_blocks),
        ("--kv-dtype", args.kv_dtype),
    ]
    for flag, value in optional:
        if value is not None:
            flags.extend([flag, str(value)])
    return flags


def serve_runtime_flags(args: argparse.Namespace, server_dir: Path) -> list[str]:
    flags = [
        *runtime_flags(args),
        "--effective-config-json",
        str(server_dir / "effective_config.json"),
        "--decision-trace-jsonl",
        str(server_dir / "decision_trace.jsonl"),
    ]
    optional = [
        ("--runtime-preset", args.runtime_preset),
        ("--scheduler-prefill-first-until-active", args.scheduler_prefill_first_until_active),
        ("--scheduler-prefill-step-chunk", args.scheduler_prefill_step_chunk),
        ("--scheduler-active-decode-prefill-chunk", args.scheduler_active_decode_prefill_chunk),
    ]
    for flag, value in optional:
        if value is not None:
            flags.extend([flag, str(value)])
    if args.enable_prefix_caching:
        flags.append("--enable-prefix-caching")
    if args.disable_prefix_cache:
        flags.append("--disable-prefix-cache")
    for extra in args.serve_extra_arg:
        flags.append(extra)
    return flags


def product_report_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    optional = [
        ("--gpu-devices", args.gpu_devices),
        ("--gpu-memory-utilization", args.gpu_memory_utilization),
        ("--max-model-len", args.max_model_len),
        ("--max-num-seqs", args.max_num_seqs),
        ("--max-num-batched-tokens", args.max_num_batched_tokens),
        ("--kv-capacity", args.kv_capacity),
        ("--kv-max-blocks", args.kv_max_blocks),
        ("--kv-dtype", args.kv_dtype),
        ("--runtime-preset", args.runtime_preset),
        ("--scheduler-prefill-first-until-active", args.scheduler_prefill_first_until_active),
        ("--scheduler-prefill-step-chunk", args.scheduler_prefill_step_chunk),
        ("--scheduler-active-decode-prefill-chunk", args.scheduler_active_decode_prefill_chunk),
    ]
    for flag, value in optional:
        if value is not None:
            flags.extend([flag, str(value)])
    if args.enable_prefix_caching:
        flags.append("--enable-prefix-caching")
    if args.disable_prefix_cache:
        flags.append("--disable-prefix-cache")
    return flags


def prefetch_command(model: str, revision: str | None) -> list[str]:
    code = (
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "repo_id = sys.argv[1]\n"
        "revision = sys.argv[2] or None\n"
        "snapshot_download(repo_id=repo_id, revision=revision, local_files_only=False)\n"
    )
    return [sys.executable, "-c", code, model, revision or ""]


def start_prefetch(args: argparse.Namespace, out_dir: Path, env: dict[str, str]) -> subprocess.Popen[str] | None:
    if not args.prefetch:
        write_json(out_dir / "prefetch.skipped.json", {"skipped": True, "reason": "--no-prefetch"})
        return None
    cmd = prefetch_command(args.model, args.model_revision)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "prefetch.command.json",
        {
            "cmd": cmd,
            "env": sanitize_env_for_record(env),
            "hf_cache_layout": "HF_HOME/hub/models--ORG--NAME/snapshots",
            "started_at": iso_now(),
        },
    )
    write_text(out_dir / "prefetch.command.txt", shlex_join(cmd) + "\n")
    log = (out_dir / "prefetch.log").open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log.close()


def wait_prefetch(proc: subprocess.Popen[str] | None, out_dir: Path) -> None:
    if proc is None:
        return
    rc = proc.wait()
    write_json(out_dir / "prefetch.result.json", {"returncode": rc, "finished_at": iso_now()})
    if rc != 0:
        raise LaneError(f"prefetch failed with rc={rc}: {out_dir / 'prefetch.log'}")


def build_binary(args: argparse.Namespace, out_dir: Path, env: dict[str, str]) -> Path:
    if not args.build:
        ferrum_bin = args.ferrum_bin or REPO_ROOT / "target/release/ferrum"
        if not ferrum_bin.is_file():
            raise LaneError(f"missing ferrum binary: {ferrum_bin}")
        return ferrum_bin
    cmd = [
        "cargo",
        "build",
        "--release",
        "-p",
        "ferrum-cli",
        "--bin",
        "ferrum",
        "--features",
        CUDA_FEATURES,
    ]
    run_logged(
        label="build_cuda",
        cmd=cmd,
        out_dir=out_dir,
        env=env,
        timeout=args.build_timeout_seconds,
    )
    ferrum_bin = args.ferrum_bin or REPO_ROOT / "target/release/ferrum"
    if not ferrum_bin.is_file():
        raise LaneError(f"CUDA build completed but binary is missing: {ferrum_bin}")
    return ferrum_bin


def run_product_report(
    args: argparse.Namespace,
    out_dir: Path,
    ferrum_bin: Path,
    env: dict[str, str],
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/release/w3_qwen35_real_product_report.py"),
        "--out",
        str(out_dir),
        "--model",
        args.model,
        "--release-model-id",
        args.release_model_id,
        "--quantized-format",
        args.quantization,
        "--backend",
        args.backend,
        "--host",
        args.host,
        "--ferrum-bin",
        str(ferrum_bin),
        "--hf-home",
        str(args.hf_home),
        "--run-timeout-seconds",
        str(args.run_timeout_seconds),
        "--serve-startup-timeout-seconds",
        str(args.serve_startup_timeout_seconds),
        "--request-timeout-seconds",
        str(args.request_timeout_seconds),
        *product_report_flags(args),
    ]
    if args.request_model:
        cmd.extend(["--request-model", args.request_model])
    if args.require_clean_git:
        cmd.append("--require-clean-git")
    else:
        cmd.append("--no-require-clean-git")
    run_logged(
        label="w3_product_report",
        cmd=cmd,
        out_dir=out_dir.parent / "commands",
        env=env,
        timeout=args.product_timeout_seconds,
    )


def run_l2(product_dir: Path, out_dir: Path, args: argparse.Namespace, env: dict[str, str]) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/release/w3_l2_quantized_gate.py"),
        "--report",
        str(product_dir / "known_answer_report.json"),
        "--out",
        str(out_dir),
        "--model-id",
        args.release_model_id,
        "--format",
        args.quantization,
    ]
    run_logged(
        label="w3_l2_quantized",
        cmd=cmd,
        out_dir=out_dir.parent / "commands",
        env=env,
        timeout=args.short_gate_timeout_seconds,
    )


def start_server(
    args: argparse.Namespace,
    server_dir: Path,
    ferrum_bin: Path,
    env: dict[str, str],
) -> tuple[subprocess.Popen[str], str, str]:
    server_dir.mkdir(parents=True, exist_ok=True)
    port = args.port if args.port is not None else free_port()
    cmd = [
        str(ferrum_bin),
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(port),
        *serve_runtime_flags(args, server_dir),
    ]
    write_json(
        server_dir / "serve.command.json",
        {"cmd": cmd, "cwd": str(REPO_ROOT), "env": sanitize_env_for_record(env)},
    )
    write_text(server_dir / "serve.command.txt", shlex_join(cmd) + "\n")
    log = (server_dir / "serve.log").open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log.close()
    base_url = f"http://{args.host}:{port}"
    wait_for_health(proc, base_url, args.serve_startup_timeout_seconds)
    request_model = args.request_model or server_model_id(
        base_url,
        args.request_timeout_seconds,
        args.release_model_id,
    )
    write_json(
        server_dir / "server_runtime.json",
        {
            "base_url": base_url,
            "request_model": request_model,
            "started_at": iso_now(),
            "serve_command": repo_path(server_dir / "serve.command.json"),
        },
    )
    return proc, base_url, request_model


def stop_server(proc: subprocess.Popen[str], server_dir: Path) -> None:
    if proc.poll() is not None:
        write_json(server_dir / "serve.exit.json", {"returncode": proc.returncode})
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20)
    write_json(server_dir / "serve.exit.json", {"returncode": proc.returncode})


def run_l4(
    args: argparse.Namespace,
    out_dir: Path,
    base_url: str,
    request_model: str,
    env: dict[str, str],
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/release/w3_l4_agent_gate.py"),
        "--base-url",
        base_url,
        "--model",
        request_model,
        "--release-model-id",
        args.release_model_id,
        "--out",
        str(out_dir),
        "--timeout",
        str(args.request_timeout_seconds),
        "--tool-total",
        str(args.l4_tool_total),
        "--strict-total",
        str(args.l4_strict_total),
    ]
    run_logged(
        label="w3_l4_agent",
        cmd=cmd,
        out_dir=out_dir.parent / "commands",
        env=env,
        timeout=args.l4_timeout_seconds,
    )


def bench_command(
    args: argparse.Namespace,
    ferrum_bin: Path,
    base_url: str,
    request_model: str,
    tokenizer: Path,
    perf_report: Path,
) -> list[str]:
    return [
        str(ferrum_bin),
        "bench-serve",
        "--base-url",
        base_url,
        "--model",
        request_model,
        "--tokenizer",
        str(tokenizer),
        "--dataset",
        "sharegpt",
        "--sharegpt-path",
        str(args.sharegpt_path),
        "--random-output-len",
        str(EXPECTED_OUTPUT_LEN),
        "--ignore-eos",
        "--concurrency-sweep",
        REQUIRED_CELLS,
        "--num-prompts",
        str(args.num_prompts),
        "--warmup-requests",
        str(args.warmup_requests),
        "--n-repeats",
        "3",
        "--fail-on-error",
        "--require-ci",
        "--seed",
        "9271",
        "--timeout",
        str(args.request_timeout_seconds),
        "--out",
        str(perf_report),
    ]


def run_bench(
    args: argparse.Namespace,
    perf_dir: Path,
    ferrum_bin: Path,
    base_url: str,
    request_model: str,
    env: dict[str, str],
) -> Path:
    tokenizer = resolve_tokenizer_path(args.model, args.hf_home, args.tokenizer)
    perf_report = perf_dir / "bench_ferrum_sharegpt_sweep_100x3.json"
    cmd = bench_command(args, ferrum_bin, base_url, request_model, tokenizer, perf_report)
    perf_dir.mkdir(parents=True, exist_ok=True)
    write_text(perf_dir / "bench-ferrum.command.txt", shlex_join(cmd) + "\n")
    run_logged(
        label="bench_ferrum_sharegpt_sweep_100x3",
        cmd=cmd,
        out_dir=perf_dir / "commands",
        env=env,
        timeout=args.bench_process_timeout_seconds,
    )
    if not perf_report.is_file():
        raise LaneError(f"bench-serve did not produce report: {perf_report}")
    return perf_report


def run_l5(args: argparse.Namespace, perf_dir: Path, l5_dir: Path, env: dict[str, str]) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/release/w3_l5_concurrency_gate.py"),
        "--report",
        str(perf_dir / "bench_ferrum_sharegpt_sweep_100x3.json"),
        "--out",
        str(l5_dir),
        "--model-id",
        args.release_model_id,
        "--expected-output-len",
        str(EXPECTED_OUTPUT_LEN),
        "--command",
        (perf_dir / "bench-ferrum.command.txt").read_text(encoding="utf-8").strip(),
    ]
    for value in args.effective_concurrency:
        cmd.extend(["--effective-concurrency", value])
    run_logged(
        label="w3_l5_concurrency",
        cmd=cmd,
        out_dir=l5_dir.parent / "commands",
        env=env,
        timeout=args.short_gate_timeout_seconds,
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_config_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise LaneError(f"{label} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def config_args(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.base_evidence_config)
    if not isinstance(config, dict):
        raise LaneError(f"base evidence config is not a JSON object: {args.base_evidence_config}")
    raw = config.get("args", {})
    if not isinstance(raw, dict):
        raise LaneError(f"base evidence config args must be a JSON object: {args.base_evidence_config}")
    return raw


def has_flag(parts: list[str], flag: str) -> bool:
    return flag in parts or any(part.startswith(f"{flag}=") for part in parts)


def flag_values(parts: list[str], flag: str) -> list[str]:
    values: list[str] = []
    prefix = f"{flag}="
    idx = 0
    while idx < len(parts):
        part = parts[idx]
        if part.startswith(prefix):
            values.append(part[len(prefix) :])
        elif part == flag and idx + 1 < len(parts):
            values.append(parts[idx + 1])
            idx += 1
        idx += 1
    return values


def command_file_fixed_output_ok(path: Path, label: str) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if not path.is_file():
        return False, [f"{label} missing: {path}"]
    try:
        parts = shlex.split(path.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        return False, [f"{label} is not a valid shell command: {exc}"]
    if not has_flag(parts, "--ignore-eos"):
        problems.append(f"{label} missing --ignore-eos")
    values = flag_values(parts, "--random-output-len")
    if values != [str(EXPECTED_OUTPUT_LEN)]:
        problems.append(f"{label} must include --random-output-len {EXPECTED_OUTPUT_LEN}")
    if "--concurrency-sweep" not in parts:
        problems.append(f"{label} must include --concurrency-sweep {REQUIRED_CELLS}")
    elif REQUIRED_CELLS not in flag_values(parts, "--concurrency-sweep"):
        problems.append(f"{label} --concurrency-sweep must include {REQUIRED_CELLS}")
    return not problems, problems


def bench_report_fixed_output_ok(path: Path, label: str) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if not path.is_file():
        return False, [f"{label} missing: {path}"]
    data = load_json(path)
    reports = data.get("reports") if isinstance(data, dict) else data
    if not isinstance(reports, list):
        return False, [f"{label} must contain a report list"]
    seen: set[int] = set()
    for idx, report in enumerate(reports):
        if not isinstance(report, dict):
            problems.append(f"{label}[{idx}] must be a JSON object")
            continue
        concurrency = report.get("concurrency")
        if isinstance(concurrency, int):
            seen.add(concurrency)
        if report.get("output_token_count_source") != "usage":
            problems.append(f"{label}[{idx}] output_token_count_source must be usage")
        rows = report.get("output_tokens_per_request")
        if not isinstance(rows, list):
            problems.append(f"{label}[{idx}] missing output_tokens_per_request")
            continue
        for row_idx, row in enumerate(rows):
            if not isinstance(row, list):
                problems.append(f"{label}[{idx}].output_tokens_per_request[{row_idx}] invalid")
                continue
            if any(token != EXPECTED_OUTPUT_LEN for token in row):
                problems.append(
                    f"{label}[{idx}].output_tokens_per_request[{row_idx}] "
                    f"must equal {EXPECTED_OUTPUT_LEN}"
                )
    required = {1, 4, 16, 32}
    missing = sorted(required - seen)
    if missing:
        problems.append(f"{label} missing concurrency cells: {missing}")
    return not problems, problems


def materialize_command_value(value: Any, label: str, out_dir: Path) -> Path:
    if isinstance(value, str):
        candidate = Path(value)
        candidate = candidate if candidate.is_absolute() else REPO_ROOT / candidate
        if candidate.is_file():
            return candidate
        text = value
    elif isinstance(value, list) and value and all(isinstance(part, str) for part in value):
        text = shlex_join(value)
    else:
        raise LaneError(f"{label} must be a command string, command list, or path")
    path = out_dir / f"{label}.txt"
    write_text(path, text.strip() + "\n")
    return path


def configured_baseline_paths(args: argparse.Namespace, out_dir: Path) -> dict[str, Path]:
    raw = config_args(args)
    return {
        "perf_report": args.baseline_perf_report
        or resolve_config_path(raw.get("baseline_perf_report"), "baseline_perf_report"),
        "bench_command": args.baseline_bench_command
        or materialize_command_value(
            raw.get("baseline_bench_command"),
            "baseline_bench_command",
            out_dir / "configured_baseline",
        ),
        "server_command": args.baseline_server_command
        or materialize_command_value(
            raw.get("baseline_server_command"),
            "baseline_server_command",
            out_dir / "configured_baseline",
        ),
        "build_command": args.baseline_build_command
        or materialize_command_value(
            raw.get("baseline_build_command"),
            "baseline_build_command",
            out_dir / "configured_baseline",
        ),
    }


def historical_baseline_contract(
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[bool, list[str], dict[str, Path]]:
    paths = configured_baseline_paths(args, out_dir)
    problems: list[str] = []
    command_ok, command_problems = command_file_fixed_output_ok(
        paths["bench_command"],
        "historical baseline bench command",
    )
    report_ok, report_problems = bench_report_fixed_output_ok(
        paths["perf_report"],
        "historical baseline report",
    )
    problems.extend(command_problems)
    problems.extend(report_problems)
    for key in ["server_command", "build_command"]:
        if not paths[key].is_file():
            problems.append(f"historical baseline {key} missing: {paths[key]}")
    return command_ok and report_ok and not problems, problems, paths


def apply_baseline_paths(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    args.baseline_perf_report = paths["perf_report"]
    args.baseline_bench_command = paths["bench_command"]
    args.baseline_server_command = paths["server_command"]
    args.baseline_build_command = paths["build_command"]


def validate_vllm_probe_data(data: dict[str, Any]) -> str:
    if not isinstance(data.get("vllm"), str) or not data["vllm"]:
        raise LaneError(f"vLLM is not importable in selected Python: {data.get('vllm_error')}")
    if data.get("cuda_available") is not True:
        raise LaneError("torch CUDA is not visible in selected vLLM Python")
    device_count = data.get("cuda_device_count")
    if isinstance(device_count, bool) or not isinstance(device_count, int) or device_count < 1:
        raise LaneError(f"selected vLLM Python reports invalid cuda_device_count={device_count!r}")
    return str(data["vllm"])


def run_vllm_version_probe(
    args: argparse.Namespace,
    out_dir: Path,
    env: dict[str, str],
) -> str:
    code = (
        "import json\n"
        "data = {}\n"
        "try:\n"
        "    import vllm\n"
        "    data['vllm'] = getattr(vllm, '__version__', 'unknown')\n"
        "except Exception as exc:\n"
        "    data['vllm_error'] = type(exc).__name__ + ': ' + str(exc)\n"
        "try:\n"
        "    import torch\n"
        "    data['torch'] = getattr(torch, '__version__', 'unknown')\n"
        "    data['cuda_available'] = bool(torch.cuda.is_available())\n"
        "    data['cuda_device_count'] = int(torch.cuda.device_count())\n"
        "except Exception as exc:\n"
        "    data['torch_error'] = type(exc).__name__ + ': ' + str(exc)\n"
        "print(json.dumps(data, sort_keys=True))\n"
    )
    cmd = [args.vllm_python, "-c", code]
    proc = run_logged(
        label="vllm_version_probe",
        cmd=cmd,
        out_dir=out_dir / "commands",
        env=env,
        timeout=args.short_gate_timeout_seconds,
    )
    version_data = json.loads(proc.stdout.strip().splitlines()[-1])
    write_json(out_dir / "vllm_versions.json", version_data)
    write_text(out_dir / "baseline-build.command.txt", shlex_join(cmd) + "\n")
    return validate_vllm_probe_data(version_data)


def start_vllm_server(
    args: argparse.Namespace,
    out_dir: Path,
    env: dict[str, str],
) -> tuple[subprocess.Popen[str], str, str]:
    server_dir = out_dir / "server"
    server_dir.mkdir(parents=True, exist_ok=True)
    port = args.vllm_port if args.vllm_port is not None else free_port()
    served_model = args.vllm_served_model_name or args.release_model_id
    cmd = [
        args.vllm_python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--served-model-name",
        served_model,
        "--host",
        args.host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
    ]
    if args.vllm_gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(args.vllm_gpu_memory_utilization)])
    if args.model_revision:
        cmd.extend(["--revision", args.model_revision])
    cmd.extend(args.vllm_extra_arg)
    write_json(
        server_dir / "vllm_server.command.json",
        {"cmd": cmd, "cwd": str(REPO_ROOT), "env": sanitize_env_for_record(env)},
    )
    write_text(server_dir / "vllm-server.command.txt", shlex_join(cmd) + "\n")
    log = (server_dir / "vllm_server.log").open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log.close()
    base_url = f"http://{args.host}:{port}"
    wait_for_openai_server(proc, base_url, args.vllm_startup_timeout_seconds)
    return proc, base_url, served_model


def run_live_vllm_baseline(
    args: argparse.Namespace,
    out_dir: Path,
    ferrum_bin: Path,
    env: dict[str, str],
) -> dict[str, Path | str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    version = run_vllm_version_probe(args, out_dir, env)
    proc: subprocess.Popen[str] | None = None
    try:
        proc, base_url, request_model = start_vllm_server(args, out_dir, env)
        tokenizer = resolve_tokenizer_path(args.model, args.hf_home, args.tokenizer)
        perf_dir = out_dir / "perf"
        perf_dir.mkdir(parents=True, exist_ok=True)
        report = perf_dir / "bench_vllm_sharegpt_sweep_100x3.json"
        cmd = bench_command(args, ferrum_bin, base_url, request_model, tokenizer, report)
        write_text(perf_dir / "bench-vllm.command.txt", shlex_join(cmd) + "\n")
        run_logged(
            label="bench_vllm_sharegpt_sweep_100x3",
            cmd=cmd,
            out_dir=perf_dir / "commands",
            env=env,
            timeout=args.vllm_baseline_timeout_seconds,
        )
        ok, problems = bench_report_fixed_output_ok(report, "live vLLM baseline report")
        if not ok:
            raise LaneError("live vLLM baseline report failed fixed-output contract: " + "; ".join(problems))
    finally:
        if proc is not None:
            stop_server(proc, out_dir / "server")
    return {
        "perf_report": out_dir / "perf/bench_vllm_sharegpt_sweep_100x3.json",
        "bench_command": out_dir / "perf/bench-vllm.command.txt",
        "server_command": out_dir / "server/vllm-server.command.txt",
        "build_command": out_dir / "baseline-build.command.txt",
        "version": version,
    }


def prepare_baseline(
    args: argparse.Namespace,
    out_dir: Path,
    ferrum_bin: Path,
    env: dict[str, str],
) -> None:
    ok, problems, paths = historical_baseline_contract(args, out_dir)
    decision = {
        "baseline_mode": args.baseline_mode,
        "historical_ok": ok,
        "historical_paths": {key: str(value) for key, value in paths.items()},
        "historical_problems": problems,
        "generated_at": iso_now(),
    }
    if args.baseline_mode == "historical":
        if not ok:
            write_json(out_dir / "baseline_decision.json", decision)
            raise LaneError("historical baseline is not W3 fixed-output valid: " + "; ".join(problems))
        apply_baseline_paths(args, paths)
        decision["selected"] = "historical"
        write_json(out_dir / "baseline_decision.json", decision)
        return
    if args.baseline_mode == "auto" and ok:
        apply_baseline_paths(args, paths)
        decision["selected"] = "historical"
        write_json(out_dir / "baseline_decision.json", decision)
        return
    live = run_live_vllm_baseline(args, out_dir / "baseline_vllm", ferrum_bin, env)
    live_paths = {
        "perf_report": live["perf_report"],
        "bench_command": live["bench_command"],
        "server_command": live["server_command"],
        "build_command": live["build_command"],
    }
    apply_baseline_paths(args, live_paths)  # type: ignore[arg-type]
    args.baseline_version = str(live["version"])
    decision["selected"] = "live_vllm"
    decision["live_paths"] = {key: str(value) for key, value in live_paths.items()}
    decision["live_version"] = args.baseline_version
    write_json(out_dir / "baseline_decision.json", decision)


def preflight_baseline(args: argparse.Namespace, out_dir: Path) -> bool:
    if args.baseline_mode not in {"auto", "historical"}:
        return True
    ok, problems, paths = historical_baseline_contract(args, out_dir / "baseline_preflight")
    write_json(
        out_dir / "baseline_preflight.json",
        {
            "baseline_mode": args.baseline_mode,
            "historical_ok": ok,
            "historical_paths": {key: str(value) for key, value in paths.items()},
            "historical_problems": problems,
            "generated_at": iso_now(),
        },
    )
    if args.baseline_mode == "historical":
        if not ok:
            raise LaneError("historical baseline is not W3 fixed-output valid: " + "; ".join(problems))
        apply_baseline_paths(args, paths)
        return False
    if ok:
        apply_baseline_paths(args, paths)
        return False
    return True


def preflight_live_vllm_if_needed(
    args: argparse.Namespace,
    out_dir: Path,
    env: dict[str, str],
    needs_live_baseline: bool,
) -> None:
    if not needs_live_baseline:
        return
    version = run_vllm_version_probe(args, out_dir / "baseline_vllm_preflight", env)
    write_json(
        out_dir / "baseline_vllm_preflight.json",
        {
            "status": "pass",
            "version": version,
            "reason": "historical baseline is not fixed-output command-valid",
            "generated_at": iso_now(),
        },
    )


def render_manifest_config(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    ferrum_bin: Path,
    binary_sha: str,
    git: dict[str, Any],
) -> Path:
    base = load_json(args.base_evidence_config)
    if not isinstance(base, dict):
        raise LaneError(f"base evidence config is not a JSON object: {args.base_evidence_config}")
    config = dict(base)
    config["description"] = (
        "W3 Qwen3.5 CUDA release lane output generated by "
        "scripts/release/w3_qwen35_cuda_release_lane.py"
    )
    config["lane"] = "w3"
    config["out"] = str(out_dir / "manifest")
    config["effective_concurrency"] = args.effective_concurrency
    cfg_args = dict(base.get("args", {}))
    cfg_args.update(
        {
            "backend": args.backend,
            "binary_sha256": binary_sha,
            "dataset_id": args.dataset_id,
            "dataset_sha": args.dataset_sha,
            "dirty_status": {
                "dirty": bool(git["dirty"]),
                "status_short": git["status_short"],
            },
            "ferrum_bench_command": str(out_dir / "perf/bench-ferrum.command.txt"),
            "ferrum_perf_report": str(out_dir / "perf/bench_ferrum_sharegpt_sweep_100x3.json"),
            "ferrum_run": str(out_dir / "product/w3_s2_whole_model_product_path.json"),
            "ferrum_serve": str(out_dir / "product/w3_s2_whole_model_product_path.json"),
            "git_sha": git["sha"],
            "hardware": str(out_dir / "hardware/nvidia_smi_before.txt"),
            "l2_quantized": str(out_dir / "l2/w3_l2_quantized.json"),
            "l3_behavior": str(out_dir / "product/w3_l3_behavior.json"),
            "l4_agent": str(out_dir / "l4/w3_l4_agent.json"),
            "l5_concurrency": str(out_dir / "l5/w3_l5_concurrency.json"),
            "model_id": args.release_model_id,
            "product_surface": "typed_cli",
            "quantization": args.quantization,
            "runtime_snapshot": str(out_dir / "server/effective_config.json"),
            "w3_s2_product": str(out_dir / "product/w3_s2_whole_model_product_path.json"),
        }
    )
    if args.baseline_perf_report is not None:
        cfg_args["baseline_perf_report"] = str(args.baseline_perf_report)
    if args.baseline_bench_command is not None:
        cfg_args["baseline_bench_command"] = str(args.baseline_bench_command)
    if args.baseline_server_command is not None:
        cfg_args["baseline_server_command"] = str(args.baseline_server_command)
    if args.baseline_build_command is not None:
        cfg_args["baseline_build_command"] = str(args.baseline_build_command)
    if args.baseline_engine is not None:
        cfg_args["baseline_engine"] = str(args.baseline_engine)
    if args.baseline_version is not None:
        cfg_args["baseline_version"] = str(args.baseline_version)
    config["args"] = cfg_args
    path = out_dir / "manifest_config.json"
    write_json(path, config)
    write_json(out_dir / "env/dirty_status.json", cfg_args["dirty_status"])
    write_text(out_dir / "env/git_sha.txt", str(git["sha"]) + "\n")
    write_text(out_dir / "env/ferrum.sha256", binary_sha + "  " + str(ferrum_bin) + "\n")
    return path


def run_manifest(args: argparse.Namespace, config_path: Path, out_dir: Path, env: dict[str, str]) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/release/model_release_grade_manifest.py"),
        "--config",
        str(config_path),
        "--out",
        str(out_dir / "manifest"),
    ]
    if args.no_run_final_validator:
        cmd.append("--no-run-validator")
    run_logged(
        label="model_release_grade_manifest",
        cmd=cmd,
        out_dir=out_dir / "commands",
        env=env,
        timeout=args.short_gate_timeout_seconds,
    )


def write_gpu_contract(args: argparse.Namespace, out_dir: Path) -> None:
    write_json(
        out_dir / "gpu_contract.json",
        {
            "goal_doc": GOAL_DOC,
            "lane": "W3 Qwen3.5 CUDA release-grade lane on exactly 1x RTX 4090",
            "expected_runtime_cost": args.expected_runtime_cost,
            "stop_condition": (
                "stop on first build, prefetch, product correctness, L2, L4, bench, L5, "
                "or final manifest failure; otherwise stop after MODEL_RELEASE_GRADE_W3 PASS"
            ),
            "correctness_gate": (
                "w3_qwen35_real_product_report.py plus w3_l2_quantized_gate.py, "
                "w3_l4_agent_gate.py, and w3_l5_concurrency_gate.py"
            ),
            "performance_command": (
                "ferrum bench-serve --dataset sharegpt --random-output-len 128 "
                "--ignore-eos --concurrency-sweep 1,4,16,32 --num-prompts "
                f"{args.num_prompts} --n-repeats 3 --fail-on-error --require-ci --seed 9271"
            ),
            "baseline_mode": args.baseline_mode,
            "model": args.model,
            "release_model_id": args.release_model_id,
            "backend": args.backend,
            "generated_at": iso_now(),
        },
    )


def run_lane(args: argparse.Namespace) -> int:
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.hf_home = args.hf_home.resolve()
    if args.sharegpt_path is not None:
        args.sharegpt_path = args.sharegpt_path.resolve()
    if args.tokenizer is not None:
        args.tokenizer = args.tokenizer.resolve()
    env, scrubbed = child_env(args)
    git = git_summary()
    write_json(out_dir / "env/git_status.json", git)
    write_json(out_dir / "env/scrubbed_hidden_env.json", {"scrubbed_ferrum_keys": scrubbed})
    write_gpu_contract(args, out_dir)
    if args.require_clean_git and git["dirty"]:
        raise LaneError(
            "git worktree is dirty; commit or stash before collecting release-grade evidence"
        )
    if not args.sharegpt_path.is_file():
        raise LaneError(f"missing ShareGPT dataset: {args.sharegpt_path}")
    if sha256_file(args.sharegpt_path) != args.dataset_sha:
        raise LaneError(
            f"ShareGPT dataset sha mismatch for {args.sharegpt_path}; expected {args.dataset_sha}"
        )
    needs_live_baseline = preflight_baseline(args, out_dir)

    gpu_snapshot(out_dir / "hardware", env, args.require_gpu)
    prefetch = start_prefetch(args, out_dir / "prefetch", env)
    try:
        ferrum_bin = build_binary(args, out_dir / "build", env)
    except Exception:
        if prefetch is not None and prefetch.poll() is None:
            prefetch.terminate()
        raise
    wait_prefetch(prefetch, out_dir / "prefetch")
    binary_sha = sha256_file(ferrum_bin)
    write_json(
        out_dir / "env/binary.json",
        {"path": str(ferrum_bin), "sha256": binary_sha, "generated_at": iso_now()},
    )
    preflight_live_vllm_if_needed(args, out_dir, env, needs_live_baseline)

    run_product_report(args, out_dir / "product", ferrum_bin, env)
    run_l2(out_dir / "product", out_dir / "l2", args, env)

    server_proc: subprocess.Popen[str] | None = None
    try:
        server_proc, base_url, request_model = start_server(args, out_dir / "server", ferrum_bin, env)
        run_l4(args, out_dir / "l4", base_url, request_model, env)
        run_bench(args, out_dir / "perf", ferrum_bin, base_url, request_model, env)
    finally:
        if server_proc is not None:
            stop_server(server_proc, out_dir / "server")
    run_l5(args, out_dir / "perf", out_dir / "l5", env)
    prepare_baseline(args, out_dir, ferrum_bin, env)
    config_path = render_manifest_config(
        args=args,
        out_dir=out_dir,
        ferrum_bin=ferrum_bin,
        binary_sha=binary_sha,
        git=git,
    )
    run_manifest(args, config_path, out_dir, env)
    write_json(
        out_dir / "lane_summary.json",
        {
            "status": "pass",
            "pass_line": f"W3 QWEN35 CUDA RELEASE LANE PASS: {out_dir}",
            "model_release_manifest": repo_path(out_dir / "manifest/model_release_grade_manifest.json"),
            "generated_at": iso_now(),
        },
    )
    print(f"W3 QWEN35 CUDA RELEASE LANE PASS: {out_dir}")
    return 0


def fake_bench_report(concurrency: int) -> dict[str, Any]:
    zeros = [0, 0, 0]
    return {
        "scenario": "closed_loop",
        "concurrency": concurrency,
        "request_rate": None,
        "n_repeats": 3,
        "n_requests_per_run": 4,
        "output_token_count_source": "usage",
        "completed_per_run": [4, 4, 4],
        "errored_per_run": zeros,
        "output_tokens_per_request": [[EXPECTED_OUTPUT_LEN] * 4 for _ in range(3)],
        "output_throughput_tps": {"mean": 100.0, "ci95_hw": 1.0},
        "itl_ms": {"p95": {"mean": 10.0}},
        "bad_output_per_run": zeros,
        "malformed_stream_per_run": zeros,
        "missing_done_per_run": zeros,
        "duplicate_done_per_run": zeros,
        "zero_output_tokens_per_run": zeros,
        "stream_bulk_flush_per_run": zeros,
        "http_500_per_run": zeros,
        "panic_per_run": zeros,
    }


def run_selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="ferrum-w3-qwen35-lane-") as tmp:
        root = Path(tmp)
        report = root / "bench.json"
        write_json(report, [fake_bench_report(c) for c in [1, 4, 16, 32]])
        cmd = [
            "target/release/ferrum",
            "bench-serve",
            "--fail-on-error",
            "--require-ci",
            "--seed",
            "9271",
            "--n-repeats",
            "3",
            "--concurrency-sweep",
            "1,4,16,32",
            "--random-output-len",
            "128",
            "--ignore-eos",
        ]
        bench_cmd_path = root / "bench.command.txt"
        write_text(bench_cmd_path, shlex_join(cmd) + "\n")
        env = os.environ.copy()
        env["NO_COLOR"] = "1"
        env["HF_HOME"] = str(root / "hf-cache")
        env["HF_XET_HIGH_PERFORMANCE"] = "1"
        run_logged(
            label="l5_selftest",
            cmd=[
                sys.executable,
                str(REPO_ROOT / "scripts/release/w3_l5_concurrency_gate.py"),
                "--report",
                str(report),
                "--out",
                str(root / "l5"),
                "--command",
                bench_cmd_path.read_text(encoding="utf-8").strip(),
            ],
            out_dir=root / "commands",
            env=env,
            timeout=60,
        )
        lane_perf_dir = root / "lane_perf"
        write_json(
            lane_perf_dir / "bench_ferrum_sharegpt_sweep_100x3.json",
            [fake_bench_report(c) for c in [1, 4, 16, 32]],
        )
        write_text(lane_perf_dir / "bench-ferrum.command.txt", shlex_join(cmd) + "\n")
        run_l5(
            argparse.Namespace(
                release_model_id=DEFAULT_MODEL_ID,
                effective_concurrency=["16=8", "32=8"],
                short_gate_timeout_seconds=60,
            ),
            lane_perf_dir,
            root / "l5_forwarded",
            env,
        )
        forwarded_l5 = load_json(root / "l5_forwarded/w3_l5_concurrency.json")
        forwarded_cells = {
            cell["requested_concurrency"]: cell
            for cell in forwarded_l5["concurrency"]["cells"]
        }
        if forwarded_cells[16]["effective_active_concurrency"] != 8:
            raise AssertionError("lane selftest did not forward c16 effective concurrency")
        if forwarded_cells[32]["effective_active_concurrency"] != 8:
            raise AssertionError("lane selftest did not forward c32 effective concurrency")
        args = parse_args(
            [
                "--out",
                str(root / "lane"),
                "--hf-home",
                str(root / "hf-cache"),
                "--sharegpt-path",
                str(DEFAULT_SHAREGPT),
                "--no-build",
                "--no-prefetch",
                "--no-require-gpu",
                "--no-require-clean-git",
                "--no-run-final-validator",
            ]
        )
        historical_ok, historical_problems, _ = historical_baseline_contract(
            args,
            root / "historical_probe",
        )
        if historical_ok or not any("--ignore-eos" in problem for problem in historical_problems):
            raise AssertionError("selftest did not reject the checked-in historical baseline command")
        baseline_server = root / "baseline-server.command.txt"
        baseline_build = root / "baseline-build.command.txt"
        write_text(baseline_server, "python -m vllm.entrypoints.openai.api_server --model test\n")
        write_text(baseline_build, "python -c 'import vllm; print(vllm.__version__)'\n")
        args.baseline_perf_report = report
        args.baseline_bench_command = bench_cmd_path
        args.baseline_server_command = baseline_server
        args.baseline_build_command = baseline_build
        baseline_ok, baseline_problems, _ = historical_baseline_contract(
            args,
            root / "valid_baseline_probe",
        )
        if not baseline_ok:
            raise AssertionError(f"selftest valid baseline unexpectedly failed: {baseline_problems}")
        if validate_vllm_probe_data(
            {"vllm": "0.23.0", "torch": "2.11.0+cu130", "cuda_available": True, "cuda_device_count": 1}
        ) != "0.23.0":
            raise AssertionError("selftest did not accept valid vLLM/CUDA probe data")
        for bad_probe, expected in [
            ({"vllm_error": "missing", "cuda_available": True, "cuda_device_count": 1}, "vLLM"),
            ({"vllm": "0.23.0", "cuda_available": False, "cuda_device_count": 1}, "CUDA"),
            ({"vllm": "0.23.0", "cuda_available": True, "cuda_device_count": 0}, "cuda_device_count"),
        ]:
            try:
                validate_vllm_probe_data(bad_probe)
            except LaneError as exc:
                if expected not in str(exc):
                    raise AssertionError(f"unexpected vLLM probe error: {exc}") from exc
            else:
                raise AssertionError(f"bad vLLM probe unexpectedly passed: {bad_probe}")
        prefetch = prefetch_command(args.model, args.model_revision)
        if "HF_TOKEN" in " ".join(prefetch):
            raise AssertionError("prefetch command must not expose HF_TOKEN")
        flags = serve_runtime_flags(args, root / "server")
        for required in ["--backend", "--effective-config-json", "--decision-trace-jsonl"]:
            if required not in flags:
                raise AssertionError(f"serve flags missing {required}")
        config_path = render_manifest_config(
            args=args,
            out_dir=root / "lane",
            ferrum_bin=root / "ferrum",
            binary_sha="a" * 64,
            git={"sha": "b" * 40, "dirty": False, "status_short": ""},
        )
        rendered = load_json(config_path)
        if rendered["args"]["ferrum_perf_report"] != str(root / "lane/perf/bench_ferrum_sharegpt_sweep_100x3.json"):
            raise AssertionError("manifest config did not point at lane perf report")
    print("W3 QWEN35 CUDA RELEASE LANE SELFTEST PASS")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--release-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--request-model")
    parser.add_argument("--quantization", default=DEFAULT_QUANTIZATION)
    parser.add_argument("--backend", default="cuda", choices=["cuda", "auto", "metal", "cpu"])
    parser.add_argument("--base-evidence-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--hf-home", type=Path, default=Path("/workspace/hf-cache"))
    parser.add_argument("--model-revision")
    parser.add_argument("--sharegpt-path", type=Path, default=DEFAULT_SHAREGPT)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--ferrum-bin", type=Path)
    parser.add_argument("--gpu-devices")
    parser.add_argument("--gpu-memory-utilization")
    parser.add_argument("--max-model-len")
    parser.add_argument("--max-num-seqs")
    parser.add_argument("--max-num-batched-tokens")
    parser.add_argument("--kv-capacity")
    parser.add_argument("--kv-max-blocks")
    parser.add_argument("--kv-dtype")
    parser.add_argument("--runtime-preset")
    parser.add_argument("--scheduler-prefill-first-until-active")
    parser.add_argument("--scheduler-prefill-step-chunk")
    parser.add_argument("--scheduler-active-decode-prefill-chunk")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--serve-extra-arg", action="append", default=[])
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-sha", default=DEFAULT_DATASET_SHA)
    parser.add_argument("--effective-concurrency", action="append", default=[])
    parser.add_argument("--baseline-perf-report", type=Path)
    parser.add_argument("--baseline-bench-command", type=Path)
    parser.add_argument("--baseline-server-command", type=Path)
    parser.add_argument("--baseline-build-command", type=Path)
    parser.add_argument("--baseline-mode", choices=["auto", "historical", "live"], default="auto")
    parser.add_argument("--baseline-engine", default="vLLM")
    parser.add_argument("--baseline-version")
    parser.add_argument("--vllm-python", default=sys.executable)
    parser.add_argument("--vllm-port", type=int)
    parser.add_argument("--vllm-served-model-name")
    parser.add_argument("--vllm-gpu-memory-utilization")
    parser.add_argument("--vllm-extra-arg", action="append", default=[])
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--warmup-requests", type=int, default=10)
    parser.add_argument("--l4-tool-total", type=int, default=10)
    parser.add_argument("--l4-strict-total", type=int, default=20)
    parser.add_argument("--expected-runtime-cost", default="1x RTX 4090, about 1-2 hours")
    parser.add_argument("--build", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-clean-git", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-run-final-validator", action="store_true")
    parser.add_argument("--build-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--product-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--l4-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--bench-process-timeout-seconds", type=float, default=14400.0)
    parser.add_argument("--short-gate-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--run-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--serve-startup-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--vllm-startup-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--vllm-baseline-timeout-seconds", type=float, default=14400.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    if args.out is None:
        parser.error("missing required arg: --out")
    if args.enable_prefix_caching and args.disable_prefix_cache:
        parser.error("--enable-prefix-caching conflicts with --disable-prefix-cache")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            return run_selftest()
        return run_lane(args)
    except LaneError as exc:
        if getattr(args, "out", None) is not None:
            out = args.out.resolve()
            write_json(
                out / "lane_summary.json",
                {"status": "fail", "error": str(exc), "finished_at": iso_now()},
            )
        print(f"W3 QWEN35 CUDA RELEASE LANE FAIL: {exc}", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired as exc:
        print(f"W3 QWEN35 CUDA RELEASE LANE FAIL: timeout: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
