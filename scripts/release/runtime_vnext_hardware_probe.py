#!/usr/bin/env python3
"""Collect reproducible Runtime vNext hardware identity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PASS_PREFIX = "RUNTIME VNEXT HARDWARE PROBE PASS"
PROBE_ARGV = {
    "cuda": {
        "host": ["uname", "-s", "-n", "-r", "-m"],
        "gpu": ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        "toolchain": ["nvcc", "--version"],
        "memory": ["free", "-b"],
        "cpu": ["lscpu", "-J"],
    },
    "metal": {
        "host": ["uname", "-s", "-n", "-r", "-m"],
        "gpu": ["system_profiler", "SPDisplaysDataType"],
        "toolchain": ["xcrun", "metal", "-v"],
        "memory": ["sysctl", "-n", "hw.memsize"],
        "cpu": ["sysctl", "-n", "hw.logicalcpu"],
        "os": ["sw_vers", "-productVersion"],
    },
}


class ProbeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_value(source_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=source_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def run_command(out: Path, kind: str, argv: list[str]) -> dict[str, Any]:
    started_at = now_iso()
    started = time.monotonic()
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    duration = time.monotonic() - started
    finished_at = now_iso()
    stdout_path = out / "raw" / f"{kind}.stdout.txt"
    stderr_path = out / "raw" / f"{kind}.stderr.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    require(result.returncode == 0, f"hardware probe command failed: {argv}: {result.stderr.strip()}")
    return {
        "kind": kind,
        "argv": argv,
        "returncode": result.returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": max(duration, 1e-6),
        "stdout": stdout_path.relative_to(out).as_posix(),
        "stdout_sha256": sha256(stdout_path),
        "stderr": stderr_path.relative_to(out).as_posix(),
        "stderr_sha256": sha256(stderr_path),
        "_stdout": result.stdout,
        "_stderr": result.stderr,
    }


def first_match(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    require(match is not None, f"could not parse {label}")
    return match.group(1).strip()


def parse_uname(text: str) -> tuple[str, str, str, str]:
    parts = text.strip().split()
    require(len(parts) == 4, "unexpected uname output")
    system_name, host, release, arch = parts
    return system_name, host, release, arch


def normalized_from_outputs(
    backend: str,
    policy_id: str,
    outputs: dict[str, str],
    errors: dict[str, str],
) -> dict[str, Any]:
    system_name, host, release, arch = parse_uname(outputs["host"])
    toolchain = (outputs["toolchain"] or errors["toolchain"]).strip()
    require(toolchain, "toolchain output is empty")
    if backend == "metal":
        gpu = outputs["gpu"]
        memory_text = outputs["memory"].strip()
        logical_cpu_text = outputs["cpu"].strip()
        macos_version = outputs["os"].strip()
        device = first_match(r"Chipset Model:\s*(.+)$", gpu, "Apple GPU name")
        core_text = first_match(r"Total Number of Cores:\s*(\d+)", gpu, "Apple GPU core count")
        require("M1 Max" in device, f"expected M1 Max, got {device}")
        memory = int(memory_text)
        logical_cpu_count = int(logical_cpu_text)
        runtime = {"metal_toolchain": toolchain, "macos_version": macos_version}
        normalized: dict[str, Any] = {
            "schema_version": 1,
            "backend": backend,
            "policy_id": policy_id,
            "host": host,
            "device_name": device,
            "device_count": 1,
            "memory_bytes": memory,
            "runtime": runtime,
            "system": {
                "os": f"{system_name} {release}",
                "arch": arch,
                "cpu": device,
                "logical_cpu_count": logical_cpu_count,
                "host_memory_bytes": memory,
            },
            "gpu_core_count": int(core_text),
        }
        return normalized

    row = outputs["gpu"].strip().splitlines()
    require(len(row) == 1, "CUDA probe requires exactly one visible GPU")
    parts = [part.strip() for part in row[0].split(",")]
    require(len(parts) == 3, "unexpected nvidia-smi CSV output")
    name, memory_mib, driver = parts
    require("4090" in name, f"expected RTX 4090, got {name}")
    cuda_version = first_match(r"release\s+([0-9.]+)", toolchain, "CUDA release")
    try:
        cpu_rows = json.loads(outputs["cpu"])["lscpu"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ProbeError("invalid lscpu JSON") from exc
    cpu_map = {
        str(row.get("field", "")).strip().rstrip(":"): str(row.get("data", "")).strip()
        for row in cpu_rows
        if isinstance(row, dict)
    }
    cpu_name = cpu_map.get("Model name")
    logical_cpu_text = cpu_map.get("CPU(s)")
    require(bool(cpu_name) and bool(logical_cpu_text), "lscpu output lacks CPU identity")
    host_memory = int(first_match(r"^Mem:\s+(\d+)", outputs["memory"], "host memory"))
    return {
        "schema_version": 1,
        "backend": backend,
        "policy_id": policy_id,
        "host": host,
        "device_name": name,
        "device_count": 1,
        "memory_bytes": int(memory_mib) * 1024 * 1024,
        "runtime": {
            "driver_version": driver,
            "cuda_version": cuda_version,
            "nvcc_version": toolchain,
        },
        "system": {
            "os": f"{system_name} {release}",
            "arch": arch,
            "cpu": cpu_name,
            "logical_cpu_count": int(logical_cpu_text),
            "host_memory_bytes": host_memory,
        },
    }


def collect(backend: str, hardware_id: str, policy_id: str, out: Path, source_root: Path) -> dict[str, Any]:
    require(out != source_root and source_root.is_dir(), "invalid source root or output directory")
    out.mkdir(parents=True, exist_ok=True)
    argv = PROBE_ARGV[backend]
    commands = {kind: run_command(out, kind, command) for kind, command in argv.items()}
    normalized = normalized_from_outputs(
        backend,
        policy_id,
        {kind: command["_stdout"] for kind, command in commands.items()},
        {kind: command["_stderr"] for kind, command in commands.items()},
    )
    status = git_value(source_root, "status", "--short").splitlines()
    clean_commands = []
    for command in commands.values():
        clean_commands.append({key: value for key, value in command.items() if not key.startswith("_")})
    return {
        "schema_version": 1,
        "source_git_sha": git_value(source_root, "rev-parse", "HEAD"),
        "source_tree_sha": git_value(source_root, "rev-parse", "HEAD^{tree}"),
        "dirty_status": {"is_dirty": bool(status), "status_short": status},
        "collector": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "hardware_id": hardware_id,
        "normalized": normalized,
        "fingerprint": canonical_sha(normalized),
        "commands": clean_commands,
    }


def self_test() -> None:
    metal_outputs = {
        "host": "Darwin MacBookPro 24.1.0 arm64\n",
        "gpu": "Chipset Model: Apple M1 Max\nTotal Number of Cores: 24\n",
        "memory": "34359738368\n",
        "cpu": "10\n",
        "os": "15.1.1\n",
        "toolchain": "metal version 1\n",
    }
    facts = normalized_from_outputs("metal", "metal-reference-m1-max-32gb", metal_outputs, {key: "" for key in metal_outputs})
    require(facts["memory_bytes"] == 32 * 1024**3 and facts["gpu_core_count"] == 24, "Metal parser self-test failed")
    cuda_outputs = {
        "host": "Linux cuda-host 6.8.0 x86_64\n",
        "gpu": "NVIDIA GeForce RTX 4090, 24564, 555.42\n",
        "toolchain": "Cuda compilation tools, release 12.4, V12.4.99\n",
        "memory": "Mem: 68719476736 0 0 0 0 0\n",
        "cpu": json.dumps({"lscpu": [{"field": "CPU(s):", "data": "32"}, {"field": "Model name:", "data": "Test CPU"}]}),
    }
    facts = normalized_from_outputs("cuda", "cuda-g0-1x-rtx4090", cuda_outputs, {key: "" for key in cuda_outputs})
    require(facts["runtime"]["cuda_version"] == "12.4", "CUDA parser self-test failed")
    require(canonical_sha({"b": 2, "a": 1}) == canonical_sha({"a": 1, "b": 2}), "canonical fingerprint is unstable")
    print("RUNTIME VNEXT HARDWARE PROBE SELF-TEST PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cuda", "metal"))
    parser.add_argument("--hardware-id")
    parser.add_argument("--policy-id")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not all((args.backend, args.hardware_id, args.policy_id, args.out)):
        parser.error("--backend, --hardware-id, --policy-id and --out are required")
    try:
        probe = collect(args.backend, args.hardware_id, args.policy_id, args.out, args.source_root.resolve())
        path = args.out / "probe.json"
        path.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, ProbeError) as exc:
        print(f"RUNTIME VNEXT HARDWARE PROBE FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"{PASS_PREFIX}: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
