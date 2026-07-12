#!/usr/bin/env python3
"""Collect six reproducible CUDA build-timing scenarios for Runtime vNext G00."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PASS_PREFIX = "RUNTIME VNEXT BUILD TIMING PASS"
REPAIR_PASS_PREFIX = "RUNTIME VNEXT BUILD TIMING CORE PTX REPAIR PASS"
CORE_PTX_INPUT = "crates/ferrum-kernels/kernels/add_bias.cu"
REJECTED_CORE_PTX_INPUT = "crates/ferrum-kernels/triton_ptx/add_bias_f16.ptx"
REJECTED_COLLECTOR_SHA256 = "2579e30d6c04dd8dbb502ac0dadd267f4edc0eb3e1fe574bba4fb0b42f53b13e"
SCENARIOS = (
    ("noop", "none", None),
    ("rust-model-leaf", "content-mutation", "crates/ferrum-models/src/lib.rs"),
    ("rust-runtime-leaf", "content-mutation", "crates/ferrum-engine/src/lib.rs"),
    ("core-ptx", "content-mutation", CORE_PTX_INPUT),
    ("native-tu", "content-mutation", "crates/ferrum-kernels/vllm_marlin/gptq_marlin_repack.cu"),
    ("clean-release", "cargo-clean", None),
)
SCENARIO_PACKAGES = {
    "rust-model-leaf": "ferrum-models",
    "rust-runtime-leaf": "ferrum-engine",
    "core-ptx": "ferrum-kernels",
    "native-tu": "ferrum-kernels",
}
BUILD_ARGV = [
    "cargo",
    "build",
    "--release",
    "-p",
    "ferrum-cli",
    "--bin",
    "ferrum",
    "--features",
    "cuda,vllm-moe-marlin,vllm-paged-attn-v2",
    "--message-format=json-render-diagnostics",
    "--timings",
    "-vv",
]


class BuildTimingError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildTimingError(message)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_value(source_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=source_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def artifact_ref(path: Path, artifact_root: Path) -> dict[str, Any]:
    return {"path": path.relative_to(artifact_root).as_posix(), "sha256": sha256(path)}


def parse_cargo_messages(path: Path) -> dict[str, Any]:
    messages = []
    verbose_build_script_lines = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            # `cargo -vv --message-format=json-render-diagnostics` multiplexes
            # documented, package-prefixed build-script output onto stdout.
            # Preserve those lines in the raw artifact and reject every other
            # non-JSON shape.
            if re.fullmatch(r"\[[^\]\r\n]+\] .+", line):
                verbose_build_script_lines.append(line)
                continue
            raise BuildTimingError(f"invalid Cargo JSON at {path}:{line_number}: {exc}") from exc
        require(isinstance(value, dict), f"Cargo JSON at {path}:{line_number} must be an object")
        messages.append(value)
    artifacts = [row for row in messages if row.get("reason") == "compiler-artifact"]
    require(artifacts, f"no compiler-artifact messages in {path}")
    fresh = [row for row in artifacts if row.get("fresh") is True]
    nonfresh = [row for row in artifacts if row.get("fresh") is False]
    require(len(fresh) + len(nonfresh) == len(artifacts), f"compiler artifacts in {path} must carry boolean fresh")
    finished = [row for row in messages if row.get("reason") == "build-finished"]
    require(len(finished) == 1, f"expected exactly one build-finished message in {path}")
    require(finished[0].get("success") is True, f"Cargo reported an unsuccessful build in {path}")
    packages = sorted({str(row.get("package_id", "")) for row in nonfresh if row.get("package_id")})
    custom_builds = [
        row
        for row in nonfresh
        if "custom-build" in (row.get("target", {}).get("kind", []) if isinstance(row.get("target"), dict) else [])
    ]
    return {
        "message_count": len(messages),
        "verbose_build_script_line_count": len(verbose_build_script_lines),
        "verbose_build_script_sha256": hashlib.sha256(
            ("\n".join(verbose_build_script_lines) + ("\n" if verbose_build_script_lines else "")).encode("utf-8")
        ).hexdigest(),
        "compiler_artifact_count": len(artifacts),
        "fresh_artifact_count": len(fresh),
        "nonfresh_artifact_count": len(nonfresh),
        "nonfresh_packages": packages,
        "nonfresh_custom_build_count": len(custom_builds),
        "build_finished_count": 1,
        "build_finished_success": True,
    }


def native_build_summary(log: str, input_rel: str | None) -> dict[str, Any]:
    compiled = []
    for line in log.splitlines():
        match = re.search(r"\[[^]]+\]\s+compiling\s+(\S+)\s+->\s+(\S+)", line)
        if match:
            compiled.append(match.group(1))
    log_input = input_rel
    if log_input and log_input.startswith("crates/ferrum-kernels/"):
        log_input = log_input.removeprefix("crates/ferrum-kernels/")
    return {
        "compiled_tu_paths": compiled,
        "compiled_tu_count": len(compiled),
        "static_lib_built_count": sum("static lib built:" in line for line in log.splitlines()),
        "input_path_mentions": log.count(str(log_input)) if log_input else 0,
    }


def validate_scenario_signal(
    name: str,
    cargo_summary: dict[str, Any],
    log: str,
    input_rel: str | None,
) -> dict[str, Any]:
    nonfresh_count = cargo_summary.get("nonfresh_artifact_count")
    require(isinstance(nonfresh_count, int), f"{name} Cargo summary lacks a non-fresh artifact count")
    if name == "noop":
        require(nonfresh_count == 0, "noop build unexpectedly compiled non-fresh artifacts")
    else:
        require(nonfresh_count > 0, f"{name} build did not invalidate any Cargo artifact")

    expected_package = SCENARIO_PACKAGES.get(name)
    if expected_package is not None:
        packages = cargo_summary.get("nonfresh_packages")
        require(isinstance(packages, list), f"{name} Cargo summary lacks non-fresh packages")
        require(
            any(expected_package in str(package) for package in packages),
            f"{name} build did not invalidate {expected_package}",
        )

    native_summary = native_build_summary(log, input_rel)
    if name == "core-ptx":
        log_input = str(input_rel).removeprefix("crates/ferrum-kernels/")
        require(
            f"artifact=core-ptx:{log_input} status=built" in log,
            f"core-ptx build log does not prove {log_input} was rebuilt",
        )
        require(native_summary["compiled_tu_count"] == 0, "core-ptx unexpectedly compiled a native CUDA TU")
    elif name == "native-tu":
        require(native_summary["compiled_tu_count"] > 0, "native-tu did not compile a native CUDA TU")
        require(
            any(path.endswith("vllm_marlin/gptq_marlin_repack.cu") for path in native_summary["compiled_tu_paths"]),
            "native-tu did not compile the selected CUDA TU",
        )
    return native_summary


def run_build(source_root: Path, stdout: Path, stderr: Path) -> tuple[subprocess.CompletedProcess[str], str, str, float]:
    started_at = now_iso()
    started = time.monotonic()
    result = subprocess.run(
        BUILD_ARGV,
        cwd=source_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    duration = time.monotonic() - started
    finished_at = now_iso()
    stdout.write_text(result.stdout, encoding="utf-8")
    stderr.write_text(result.stderr, encoding="utf-8")
    return result, started_at, finished_at, max(duration, 1e-6)


def setup_sample(
    source_root: Path,
    artifact_root: Path,
    sample_dir: Path,
    setup_kind: str,
    input_rel: str | None,
    mutation_id: str,
) -> tuple[dict[str, Any], tuple[bytes, int, int] | None]:
    if setup_kind == "none":
        return {"kind": "none"}, None
    if setup_kind == "cargo-clean":
        log = sample_dir / "cargo-clean.log"
        result = subprocess.run(
            ["cargo", "clean"],
            cwd=source_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write_text(result.stdout, encoding="utf-8")
        require(result.returncode == 0, f"cargo clean failed: {result.stdout[-2000:]}")
        return {
            "kind": "cargo-clean",
            "argv": ["cargo", "clean"],
            "returncode": result.returncode,
            "log": artifact_ref(log, artifact_root),
            "target_absent_after_clean": not (source_root / "target" / "release" / "ferrum").exists(),
        }, None
    require(input_rel is not None, "content mutation scenario is missing an input")
    path = source_root / input_rel
    require(path.is_file(), f"build timing input is missing: {path}")
    before = path.stat()
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    mutation = f"\n// runtime-vnext-build-timing:{mutation_id}\n".encode("ascii")
    before_copy = sample_dir / "input.before"
    mutation_copy = sample_dir / "mutation.append"
    during_copy = sample_dir / "input.during"
    before_copy.write_bytes(original)
    mutation_copy.write_bytes(mutation)
    path.write_bytes(original + mutation)
    during_copy.write_bytes(path.read_bytes())
    during = path.stat()
    during_sha = sha256(path)
    require(during_sha != digest, f"content mutation did not change source: {path}")
    return {
        "kind": "content-mutation",
        "input_path": input_rel,
        "before_sha256": digest,
        "during_sha256": during_sha,
        "mutation_kind": "append-comment",
        "mutation_sha256": hashlib.sha256(mutation).hexdigest(),
        "mutation_bytes": len(mutation),
        "before_input": artifact_ref(before_copy, artifact_root),
        "mutation_artifact": artifact_ref(mutation_copy, artifact_root),
        "during_input": artifact_ref(during_copy, artifact_root),
        "before_mtime_ns": before.st_mtime_ns,
        "during_mtime_ns": during.st_mtime_ns,
    }, (original, before.st_atime_ns, before.st_mtime_ns)


def collect_scenario(
    source_root: Path,
    artifact_root: Path,
    scenario_dir: Path,
    name: str,
    setup_kind: str,
    input_rel: str | None,
    repeats: int,
    canonical_binary_sha256: str,
) -> dict[str, Any]:
    samples = []
    for repeat in range(repeats):
        sample_dir = scenario_dir / f"sample-{repeat + 1}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        setup, restore_state = setup_sample(
            source_root,
            artifact_root,
            sample_dir,
            setup_kind,
            input_rel,
            f"{name}-{repeat + 1}",
        )
        stdout = sample_dir / "cargo-messages.jsonl"
        stderr = sample_dir / "cargo.log"
        try:
            result, sample_started, sample_finished, sample_duration = run_build(source_root, stdout, stderr)
        finally:
            if input_rel is not None and restore_state is not None:
                original, atime_ns, mtime_ns = restore_state
                (source_root / input_rel).write_bytes(original)
                os.utime(source_root / input_rel, ns=(atime_ns, mtime_ns))
        require(result.returncode == 0, f"{name} sample {repeat + 1} failed: {result.stderr[-4000:]}")
        cargo_summary = parse_cargo_messages(stdout)
        native_summary = validate_scenario_signal(name, cargo_summary, result.stderr, input_rel)
        timings_source = source_root / "target" / "cargo-timings" / "cargo-timing.html"
        require(timings_source.is_file(), f"{name} sample {repeat + 1} did not produce Cargo timings")
        timings_copy = sample_dir / "cargo-timing.html"
        shutil.copy2(timings_source, timings_copy)
        binary = source_root / "target" / "release" / "ferrum"
        require(binary.is_file(), f"{name} sample {repeat + 1} did not produce ferrum")
        copied_binary = sample_dir / "ferrum"
        shutil.copy2(binary, copied_binary)
        if setup_kind == "content-mutation":
            path = source_root / str(input_rel)
            setup["after_sha256"] = sha256(path)
            setup["after_mtime_ns"] = path.stat().st_mtime_ns
        post_status = git_value(source_root, "status", "--short").splitlines()
        require(not post_status, f"{name} sample {repeat + 1} left a dirty worktree: {post_status}")
        samples.append(
            {
                "sample_id": f"{name}-{repeat + 1}",
                "argv": BUILD_ARGV,
                "setup": setup,
                "returncode": result.returncode,
                "started_at": sample_started,
                "finished_at": sample_finished,
                "duration_sec": sample_duration,
                "cargo_messages": artifact_ref(stdout, artifact_root),
                "log": artifact_ref(stderr, artifact_root),
                "cargo_summary": cargo_summary,
                "cargo_timings": artifact_ref(timings_copy, artifact_root),
                "native_build": native_summary,
                "output_binary": artifact_ref(copied_binary, artifact_root),
                "post_git_status": post_status,
            }
        )
    restore_verification: dict[str, Any] | None = None
    if setup_kind == "content-mutation":
        restored_input = source_root / str(input_rel)
        restored_stat = restored_input.stat()
        restored_sha256 = sha256(restored_input)
        expected_restored_sha256 = str(samples[-1]["setup"]["before_sha256"])
        require(restored_sha256 == expected_restored_sha256, f"{name} input content was not restored")
        verification_mtime_ns = max(time.time_ns(), restored_stat.st_mtime_ns + 1)
        os.utime(restored_input, ns=(restored_stat.st_atime_ns, verification_mtime_ns))
        verify_dir = scenario_dir / "restore-verification"
        verify_dir.mkdir(parents=True, exist_ok=True)
        verify_stdout = verify_dir / "cargo-messages.jsonl"
        verify_stderr = verify_dir / "cargo.log"
        try:
            verify_result, verify_started, verify_finished, verify_duration = run_build(
                source_root, verify_stdout, verify_stderr
            )
        finally:
            os.utime(restored_input, ns=(restored_stat.st_atime_ns, restored_stat.st_mtime_ns))
        require(verify_result.returncode == 0, f"{name} restore verification failed: {verify_result.stderr[-4000:]}")
        verify_cargo_summary = parse_cargo_messages(verify_stdout)
        validate_scenario_signal(name, verify_cargo_summary, verify_result.stderr, input_rel)
        verify_timings_source = source_root / "target" / "cargo-timings" / "cargo-timing.html"
        verify_timings = verify_dir / "cargo-timing.html"
        shutil.copy2(verify_timings_source, verify_timings)
        verify_binary = verify_dir / "ferrum"
        shutil.copy2(source_root / "target" / "release" / "ferrum", verify_binary)
        require(
            sha256(verify_binary) == canonical_binary_sha256,
            f"{name} restore build did not reproduce the canonical binary",
        )
        restore_verification = {
            "argv": BUILD_ARGV,
            "returncode": 0,
            "started_at": verify_started,
            "finished_at": verify_finished,
            "duration_sec": verify_duration,
            "cargo_messages": artifact_ref(verify_stdout, artifact_root),
            "log": artifact_ref(verify_stderr, artifact_root),
            "cargo_summary": verify_cargo_summary,
            "cargo_timings": artifact_ref(verify_timings, artifact_root),
            "output_binary": artifact_ref(verify_binary, artifact_root),
            "restored_input": {
                "input_path": input_rel,
                "sha256": restored_sha256,
                "before_verification_mtime_ns": restored_stat.st_mtime_ns,
                "verification_mtime_ns": verification_mtime_ns,
                "after_verification_mtime_ns": restored_input.stat().st_mtime_ns,
            },
            "post_git_status": git_value(source_root, "status", "--short").splitlines(),
        }
        require(not restore_verification["post_git_status"], f"{name} restore verification left a dirty worktree")
    durations = sorted(float(row["duration_sec"]) for row in samples)
    scenario_row: dict[str, Any] = {
        "name": name,
        "command": BUILD_ARGV,
        "samples": samples,
        "p50_sec": durations[2],
        "p95_sec": durations[4],
    }
    if restore_verification is not None:
        scenario_row["restore_verification"] = restore_verification
    return scenario_row


def collect(source_root: Path, out: Path, hardware_id: str, hardware_fingerprint: str, repeats: int) -> dict[str, Any]:
    require(repeats == 5, "canonical G00 build timing requires exactly five samples")
    require(len(hardware_fingerprint) == 64, "hardware fingerprint must be SHA256")
    status = git_value(source_root, "status", "--short").splitlines()
    require(not status, f"source root must be clean before build timing: {status}")
    require(git_value(source_root, "rev-parse", "HEAD"), "source root has no git HEAD")
    artifact_root = out.parent
    out.mkdir(parents=True, exist_ok=True)

    prewarm_dir = out / "prewarm"
    prewarm_dir.mkdir(parents=True, exist_ok=True)
    prewarm_stdout = prewarm_dir / "cargo-messages.jsonl"
    prewarm_stderr = prewarm_dir / "cargo.log"
    prewarm, started_at, finished_at, duration = run_build(source_root, prewarm_stdout, prewarm_stderr)
    require(prewarm.returncode == 0, f"prewarm build failed: {prewarm.stderr[-4000:]}")
    prewarm_summary = parse_cargo_messages(prewarm_stdout)
    prewarm_timings_source = source_root / "target" / "cargo-timings" / "cargo-timing.html"
    require(prewarm_timings_source.is_file(), "prewarm did not produce Cargo timings")
    prewarm_timings = prewarm_dir / "cargo-timing.html"
    shutil.copy2(prewarm_timings_source, prewarm_timings)
    prewarm_binary = source_root / "target" / "release" / "ferrum"
    require(prewarm_binary.is_file(), "prewarm did not produce ferrum")
    canonical_binary_sha256 = sha256(prewarm_binary)
    prewarm_record = {
        "argv": BUILD_ARGV,
        "returncode": prewarm.returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": duration,
        "cargo_messages": artifact_ref(prewarm_stdout, artifact_root),
        "log": artifact_ref(prewarm_stderr, artifact_root),
        "cargo_summary": prewarm_summary,
        "cargo_timings": artifact_ref(prewarm_timings, artifact_root),
    }

    scenario_rows = []
    for name, setup_kind, input_rel in SCENARIOS:
        scenario_rows.append(
            collect_scenario(
                source_root,
                artifact_root,
                out / name,
                name,
                setup_kind,
                input_rel,
                repeats,
                canonical_binary_sha256,
            )
        )
    return {
        "schema_version": 1,
        "source_git_sha": git_value(source_root, "rev-parse", "HEAD"),
        "source_tree_sha": git_value(source_root, "rev-parse", "HEAD^{tree}"),
        "dirty_status": {"is_dirty": False, "status_short": []},
        "collector": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "hardware_id": hardware_id,
        "hardware_fingerprint": hardware_fingerprint,
        "prewarm": prewarm_record,
        "scenarios": scenario_rows,
    }


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildTimingError(f"cannot read JSON object {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


def validate_rejected_core_ptx_base(
    source_root: Path,
    base_out: Path,
    hardware_id: str,
    hardware_fingerprint: str,
) -> tuple[dict[str, Any], str]:
    require(base_out.name == "build-timings", "core PTX repair base must be the canonical build-timings directory")
    summary_path = base_out / "summary.json"
    require(summary_path.is_file() and not summary_path.is_symlink(), "core PTX repair base summary is missing")
    summary = read_json_object(summary_path)
    require(summary.get("schema_version") == 1, "core PTX repair base schema mismatch")
    require(summary.get("source_git_sha") == git_value(source_root, "rev-parse", "HEAD"), "core PTX repair source SHA mismatch")
    require(
        summary.get("source_tree_sha") == git_value(source_root, "rev-parse", "HEAD^{tree}"),
        "core PTX repair source tree mismatch",
    )
    require(summary.get("dirty_status") == {"is_dirty": False, "status_short": []}, "core PTX repair base is dirty")
    collector = summary.get("collector")
    require(isinstance(collector, dict), "core PTX repair base collector is missing")
    require(collector.get("path") == Path(__file__).resolve().relative_to(ROOT).as_posix(), "core PTX repair base collector path mismatch")
    require(collector.get("sha256") == REJECTED_COLLECTOR_SHA256, "core PTX repair base is not the reviewed rejected collector")
    require(summary.get("hardware_id") == hardware_id, "core PTX repair hardware id mismatch")
    require(summary.get("hardware_fingerprint") == hardware_fingerprint, "core PTX repair hardware fingerprint mismatch")
    rows = summary.get("scenarios")
    require(isinstance(rows, list) and len(rows) == len(SCENARIOS), "core PTX repair base scenario matrix mismatch")
    indexed = {row.get("name"): row for row in rows if isinstance(row, dict)}
    require(set(indexed) == {row[0] for row in SCENARIOS}, "core PTX repair base scenario names mismatch")
    rejected = indexed["core-ptx"]
    samples = rejected.get("samples")
    require(isinstance(samples, list) and len(samples) == 5, "rejected core PTX base must have five samples")
    for index, sample in enumerate(samples):
        require(isinstance(sample, dict), f"rejected core PTX sample {index} is malformed")
        setup = sample.get("setup")
        cargo_summary = sample.get("cargo_summary")
        require(isinstance(setup, dict) and setup.get("input_path") == REJECTED_CORE_PTX_INPUT, f"rejected core PTX sample {index} input mismatch")
        require(
            isinstance(cargo_summary, dict)
            and cargo_summary.get("nonfresh_artifact_count") == 0
            and cargo_summary.get("fresh_artifact_count") == cargo_summary.get("compiler_artifact_count"),
            f"rejected core PTX sample {index} does not reproduce the all-Fresh failure",
        )
    restore = rejected.get("restore_verification")
    require(isinstance(restore, dict), "rejected core PTX base lacks restore verification")
    restored_input = restore.get("restored_input")
    restore_summary = restore.get("cargo_summary")
    require(
        isinstance(restored_input, dict) and restored_input.get("input_path") == REJECTED_CORE_PTX_INPUT,
        "rejected core PTX restore input mismatch",
    )
    require(
        isinstance(restore_summary, dict) and restore_summary.get("nonfresh_artifact_count") == 0,
        "rejected core PTX restore does not reproduce the all-Fresh failure",
    )
    canonical_shas = {
        sample.get("output_binary", {}).get("sha256")
        for name in ("noop", "clean-release")
        for sample in indexed[name].get("samples", [])
        if isinstance(sample, dict) and isinstance(sample.get("output_binary"), dict)
    }
    require(len(canonical_shas) == 1, "core PTX repair base canonical binary identity is inconsistent")
    canonical_sha = next(iter(canonical_shas))
    require(isinstance(canonical_sha, str) and len(canonical_sha) == 64, "core PTX repair base binary SHA256 is invalid")
    return summary, canonical_sha


def collect_core_ptx_repair(
    source_root: Path,
    out: Path,
    base_out: Path,
    hardware_id: str,
    hardware_fingerprint: str,
    repeats: int,
) -> dict[str, Any]:
    require(repeats == 5, "canonical G00 core PTX repair requires exactly five samples")
    require(len(hardware_fingerprint) == 64, "hardware fingerprint must be SHA256")
    status = git_value(source_root, "status", "--short").splitlines()
    require(not status, f"source root must be clean before core PTX repair: {status}")
    base_out = base_out.resolve()
    artifact_root = base_out.parent
    expected_out = artifact_root / "build-timing-repairs" / "core-ptx"
    require(out.resolve() == expected_out.resolve(), f"core PTX repair output must be {expected_out}")
    require(not out.exists(), f"core PTX repair output already exists: {out}")
    _base_summary, canonical_binary_sha256 = validate_rejected_core_ptx_base(
        source_root,
        base_out,
        hardware_id,
        hardware_fingerprint,
    )
    binary = source_root / "target" / "release" / "ferrum"
    require(binary.is_file(), "core PTX repair requires the warmed canonical ferrum binary")
    require(sha256(binary) == canonical_binary_sha256, "warmed ferrum binary differs from the rejected base")
    out.mkdir(parents=True)
    scenario = collect_scenario(
        source_root,
        artifact_root,
        out / "samples",
        "core-ptx",
        "content-mutation",
        CORE_PTX_INPUT,
        repeats,
        canonical_binary_sha256,
    )
    return {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_build_timing_core_ptx_repair",
        "source_git_sha": git_value(source_root, "rev-parse", "HEAD"),
        "source_tree_sha": git_value(source_root, "rev-parse", "HEAD^{tree}"),
        "dirty_status": {"is_dirty": False, "status_short": []},
        "collector": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "hardware_id": hardware_id,
        "hardware_fingerprint": hardware_fingerprint,
        "base_summary": artifact_ref(base_out / "summary.json", artifact_root),
        "base_failure": "build-timings.core-ptx.samples[0] touch did not invalidate any Cargo artifact",
        "rejected_collector_sha256": REJECTED_COLLECTOR_SHA256,
        "rejected_input": REJECTED_CORE_PTX_INPUT,
        "replacement_input": CORE_PTX_INPUT,
        "canonical_binary_sha256": canonical_binary_sha256,
        "scenario": scenario,
    }


def self_test() -> None:
    with Path(__file__).open("rb") as handle:
        require(len(handle.read(32)) == 32, "collector file cannot be read")
    fixture = Path(__file__).resolve().parent / "configs" / "runtime_vnext_models.json"
    require(fixture.is_file(), "model catalog is missing")
    require(len(SCENARIOS) == 6 and len({row[0] for row in SCENARIOS}) == 6, "scenario matrix mismatch")
    require(BUILD_ARGV[0:2] == ["cargo", "build"] and "--release" in BUILD_ARGV, "build argv mismatch")
    build_script = ROOT / "crates/ferrum-kernels/build.rs"
    require(f'"{CORE_PTX_INPUT.removeprefix("crates/ferrum-kernels/")}"' in build_script.read_text(encoding="utf-8"), "core PTX timing input is not registered by ferrum-kernels/build.rs")
    with tempfile.TemporaryDirectory(prefix="runtime-vnext-build-timing-") as raw_tmp:
        root = Path(raw_tmp)
        messages = root / "cargo-messages.jsonl"
        messages.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "reason": "compiler-artifact",
                            "package_id": "ferrum-kernels 0.1.0",
                            "target": {"kind": ["lib"]},
                            "fresh": False,
                        }
                    ),
                    "[fixture 0.1.0] cargo:rerun-if-changed=build.rs",
                    json.dumps({"reason": "build-finished", "success": True}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        summary = parse_cargo_messages(messages)
        require(summary["message_count"] == 2, "mixed Cargo JSON message count mismatch")
        require(summary["verbose_build_script_line_count"] == 1, "verbose Cargo line count mismatch")
        core_log = f"[ferrum-kernels] [cuda-build-summary] artifact=core-ptx:{CORE_PTX_INPUT.removeprefix('crates/ferrum-kernels/')} status=built reason=signature-changed\n"
        signal = validate_scenario_signal("core-ptx", summary, core_log, CORE_PTX_INPUT)
        require(signal["compiled_tu_count"] == 0, "core PTX signal self-test compiled a native TU")
        fresh_summary = dict(summary)
        fresh_summary.update({"nonfresh_artifact_count": 0, "nonfresh_packages": []})
        try:
            validate_scenario_signal("core-ptx", fresh_summary, core_log, CORE_PTX_INPUT)
        except BuildTimingError as exc:
            require("did not invalidate any Cargo artifact" in str(exc), "unexpected fresh core PTX rejection")
        else:
            raise BuildTimingError("fresh core PTX sample unexpectedly passed")
        messages.write_text("not cargo JSON\n", encoding="utf-8")
        try:
            parse_cargo_messages(messages)
        except BuildTimingError as exc:
            require("invalid Cargo JSON" in str(exc), "unexpected malformed Cargo rejection")
        else:
            raise BuildTimingError("malformed non-JSON Cargo output unexpectedly passed")
    require(math.isfinite(1.0), "numeric self-test failed")
    print("RUNTIME VNEXT BUILD TIMING SELF-TEST PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--hardware-id")
    parser.add_argument("--hardware-fingerprint")
    parser.add_argument("--repair-core-ptx-from", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not all((args.out, args.hardware_id, args.hardware_fingerprint)):
        parser.error("--out, --hardware-id and --hardware-fingerprint are required")
    try:
        if args.repair_core_ptx_from is None:
            summary = collect(
                args.source_root.resolve(), args.out.resolve(), args.hardware_id, args.hardware_fingerprint, args.repeats
            )
        else:
            summary = collect_core_ptx_repair(
                args.source_root.resolve(),
                args.out.resolve(),
                args.repair_core_ptx_from.resolve(),
                args.hardware_id,
                args.hardware_fingerprint,
                args.repeats,
            )
        write_json(args.out / "summary.json", summary)
    except (BuildTimingError, OSError, ValueError) as exc:
        print(f"RUNTIME VNEXT BUILD TIMING FAIL: {exc}", file=sys.stderr)
        return 1
    if args.repair_core_ptx_from is not None:
        print(f"{REPAIR_PASS_PREFIX}: {args.out}")
    else:
        print(f"{PASS_PREFIX}: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
