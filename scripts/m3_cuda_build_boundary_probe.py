#!/usr/bin/env python3
"""Run and validate the M3 CUDA build-boundary timing gate.

This is a measurement harness for Milestone A. It intentionally validates the
same machine-readable `[cuda-build-summary]` rows used by the build script
instead of grepping ad hoc cargo output.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from validate_cuda_build_summary import ValidationError, validate_summary
from validate_cuda_build_boundary_manifest import validate_manifest

DEFAULT_FEATURES = "cuda,marlin,vllm-paged-attn-v2,vllm-moe-marlin"
DEFAULT_KERNEL = "crates/ferrum-kernels/kernels/paged_varlen_attention_vllm.cu"
DEFAULT_CACHE_HITS = [
    "core-ptx:kernels/paged_varlen_attention_vllm.cu",
    "marlin",
    "vllm_marlin",
    "vllm_moe_marlin",
    "vllm_paged_attn",
]


class ProbeError(Exception):
    pass


def run_text(command: list[str], *, cwd: Path) -> str | None:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_snapshot(repo: Path) -> dict[str, Any]:
    return {
        "head": run_text(["git", "rev-parse", "HEAD"], cwd=repo),
        "status_short": run_text(["git", "status", "--short"], cwd=repo),
    }


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
    return ordered[rank - 1]


def touch_file(path: Path) -> None:
    now = time.time()
    os.utime(path, (now, now))


def append_content_marker(path: Path, marker: str) -> bytes:
    original = path.read_bytes()
    path.write_bytes(original + marker.encode("utf-8"))
    return original


def cargo_build_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.cargo_bin,
        "build",
        "--release",
        "-p",
        args.package,
        "--features",
        args.features,
    ]
    if args.cargo_verbose:
        command.append("-vv")
    return command


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    kernel = (repo / args.kernel).resolve()
    if not kernel.is_file():
        raise ProbeError(f"kernel input not found: {kernel}")

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    command = cargo_build_command(args)
    runs: list[dict[str, Any]] = []

    for index in range(1, args.iterations + 1):
        run_dir = out_dir / f"run_{index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        build_log = run_dir / "build.log"
        restore_bytes: bytes | None = None
        if args.mutation == "touch":
            touch_file(kernel)
        else:
            restore_bytes = append_content_marker(
                kernel,
                f"\n// ferrum build-boundary probe run {index}\n",
            )

        start = time.monotonic()
        try:
            with build_log.open("wb") as log:
                result = subprocess.run(
                    command,
                    cwd=repo,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        finally:
            if restore_bytes is not None:
                kernel.write_bytes(restore_bytes)
        elapsed_sec = time.monotonic() - start

        text = build_log.read_text(encoding="utf-8", errors="replace")
        validation: dict[str, Any] | None = None
        validation_error: str | None = None
        try:
            validation = validate_summary(
                text,
                require_cache_hit=args.require_cache_hit,
                require_built=args.require_built,
            )
        except ValidationError as exc:
            validation_error = str(exc)

        run = {
            "index": index,
            "command": command,
            "exit_code": result.returncode,
            "elapsed_sec": round(elapsed_sec, 3),
            "build_log": str(build_log),
            "summary_validation": validation,
            "summary_validation_error": validation_error,
        }
        runs.append(run)
        if result.returncode != 0:
            raise ProbeError(f"run {index} failed with exit code {result.returncode}: {build_log}")
        if validation_error:
            raise ProbeError(f"run {index} failed CUDA summary validation: {validation_error}")

    elapsed_values = [float(run["elapsed_sec"]) for run in runs]
    p50 = percentile_nearest_rank(elapsed_values, 50)
    p95 = percentile_nearest_rank(elapsed_values, 95)
    limits_pass = p50 <= args.p50_limit_sec and p95 <= args.p95_limit_sec
    manifest = {
        "schema_version": 1,
        "probe": "m3_cuda_build_boundary",
        "repo": str(repo),
        "git": git_snapshot(repo),
        "kernel": args.kernel,
        "mutation": args.mutation,
        "iterations": args.iterations,
        "features": args.features,
        "package": args.package,
        "command": command,
        "timing": {
            "elapsed_sec": elapsed_values,
            "p50_sec_nearest_rank": round(p50, 3),
            "p95_sec_nearest_rank": round(p95, 3),
            "p50_limit_sec": args.p50_limit_sec,
            "p95_limit_sec": args.p95_limit_sec,
            "limits_pass": limits_pass,
        },
        "required_cache_hit": args.require_cache_hit,
        "required_built": args.require_built,
        "runs": runs,
    }
    manifest_path = out_dir / "build_boundary_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.fail_on_limit and not limits_pass:
        raise ProbeError(
            f"timing gate failed: p50={p50:.3f}s limit={args.p50_limit_sec:.3f}s, "
            f"p95={p95:.3f}s limit={args.p95_limit_sec:.3f}s"
        )
    return manifest


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        repo = root / "repo"
        kernel_dir = repo / "crates/ferrum-kernels/kernels"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "paged_varlen_attention_vllm.cu").write_text("// fake\n", encoding="utf-8")
        fake_bin = root / "cargo"
        fake_bin.write_text(
            "#!/usr/bin/env bash\n"
            "echo '[cuda-build-summary] artifact=core-ptx:kernels/paged_varlen_attention_vllm.cu status=cache_hit reason=signature-match elapsed_ms=1 inputs_hash=fnv1a64:0123456789abcdef'\n"
            "echo '[cuda-build-summary] artifact=marlin status=cache_hit reason=signature-match elapsed_ms=1 inputs_hash=fnv1a64:0123456789abcdef'\n"
            "echo '[cuda-build-summary] artifact=vllm_marlin status=cache_hit reason=signature-match elapsed_ms=1 inputs_hash=fnv1a64:0123456789abcdef'\n"
            "echo '[cuda-build-summary] artifact=vllm_moe_marlin status=cache_hit reason=signature-match elapsed_ms=1 inputs_hash=fnv1a64:0123456789abcdef'\n"
            "echo '[cuda-build-summary] artifact=vllm_paged_attn status=cache_hit reason=signature-match elapsed_ms=1 inputs_hash=fnv1a64:0123456789abcdef'\n",
            encoding="utf-8",
        )
        fake_bin.chmod(0o755)
        args = argparse.Namespace(
            repo=repo,
            out=root / "out",
            iterations=2,
            package="ferrum-cli",
            features=DEFAULT_FEATURES,
            kernel=DEFAULT_KERNEL,
            mutation="touch",
            cargo_bin=str(fake_bin),
            cargo_verbose=True,
            require_cache_hit=DEFAULT_CACHE_HITS,
            require_built=[],
            p50_limit_sec=75.0,
            p95_limit_sec=90.0,
            fail_on_limit=True,
        )
        manifest = run_probe(args)
        validate_manifest(root / "out" / "build_boundary_manifest.json", require_limits_pass=True)
        assert manifest["iterations"] == 2
        assert manifest["timing"]["limits_pass"] is True
        assert len(manifest["runs"]) == 2
    print("m3_cuda_build_boundary_probe self-test ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("/tmp/ferrum-m3-cuda-build-boundary"))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--package", default="ferrum-cli")
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--kernel", default=DEFAULT_KERNEL)
    parser.add_argument("--mutation", choices=["touch", "content-change"], default="touch")
    parser.add_argument("--cargo-bin", default="cargo")
    parser.add_argument("--cargo-verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-cache-hit", action="append", default=None)
    parser.add_argument("--require-built", action="append", default=None)
    parser.add_argument("--p50-limit-sec", type=float, default=75.0)
    parser.add_argument("--p95-limit-sec", type=float, default=90.0)
    parser.add_argument("--fail-on-limit", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.require_cache_hit is None:
        if args.mutation == "content-change":
            args.require_cache_hit = [
                artifact
                for artifact in DEFAULT_CACHE_HITS
                if artifact != "core-ptx:kernels/paged_varlen_attention_vllm.cu"
            ]
        else:
            args.require_cache_hit = list(DEFAULT_CACHE_HITS)
    if args.require_built is None:
        args.require_built = []
        if args.mutation == "content-change":
            args.require_built.append("core-ptx:kernels/paged_varlen_attention_vllm.cu")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    manifest = run_probe(args)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        timing = manifest["timing"]
        print(
            "m3 cuda build boundary probe ok: "
            f"runs={manifest['iterations']} "
            f"p50={timing['p50_sec_nearest_rank']}s "
            f"p95={timing['p95_sec_nearest_rank']}s "
            f"manifest={Path(args.out).resolve() / 'build_boundary_manifest.json'}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProbeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
