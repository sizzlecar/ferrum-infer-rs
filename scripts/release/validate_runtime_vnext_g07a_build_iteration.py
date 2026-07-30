#!/usr/bin/env python3
"""Independently verify raw G07A build-iteration evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = (
    REPO_ROOT
    / "scripts/release/configs/runtime_vnext_g07a_build_iteration.json"
)
SCHEMA_VERSION = 1
SOURCE_BUILD_RECEIPT_SCHEMA_VERSION = 7
BOUNDED_RECEIPT_SCHEMA = "ferrum.bounded-command-receipt.v1"
MAX_JSON_BYTES = 64 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_SCENARIOS = (
    ("noop", "cargo_incremental", None, None, 30),
    (
        "rust-model-leaf",
        "cargo_incremental",
        "crates/ferrum-models/src/lib.rs",
        "ferrum-models",
        90,
    ),
    (
        "rust-runtime-leaf",
        "cargo_incremental",
        "crates/ferrum-engine/src/lib.rs",
        "ferrum-engine",
        90,
    ),
    (
        "core-ptx",
        "cargo_incremental",
        "crates/ferrum-kernels/kernels/add_bias.cu",
        "ferrum-kernels",
        120,
    ),
    (
        "native-tu",
        "native_source_build",
        "vllm_marlin/gptq_marlin_repack.cu",
        "ferrum.cuda.vllm_marlin",
        300,
    ),
    (
        "clean-release",
        "cargo_clean_release",
        None,
        "ferrum-cli",
        900,
    ),
)
PRODUCT_NATIVE_UNITS = {
    "marlin",
    "vllm_marlin",
    "vllm_moe_marlin",
    "vllm_paged_attn",
}
PRODUCT_NATIVE_OPERATORS = {
    "ferrum.cuda.marlin",
    "ferrum.cuda.vllm_marlin",
    "ferrum.cuda.vllm_moe_marlin",
    "ferrum.cuda.vllm_paged_attention_v2",
}
DOES_NOT_PROVE = {
    "canonical G07A PASS",
    "canonical G07B PASS",
    "G07 aggregate PASS",
    "model correctness",
    "model performance",
    "release readiness",
}
EVIDENCE_PASS_PREFIX = (
    "FERRUM RUNTIME VNEXT G07A BUILD ITERATION EVIDENCE VERIFIED"
)
SELFTEST_PASS_LINE = (
    "FERRUM RUNTIME VNEXT G07A BUILD ITERATION VALIDATOR SELFTEST PASS"
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def read_regular(path: Path, max_bytes: int, label: str) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VerificationError(f"cannot open {label} {path}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_size <= max_bytes,
            f"{label} is not a bounded regular file: {path}",
        )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        require(len(raw) <= max_bytes, f"{label} exceeds {max_bytes} bytes")
        return raw
    finally:
        os.close(descriptor)


def read_json(path: Path, label: str) -> Any:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing")
    try:
        return json.loads(read_regular(path, MAX_JSON_BYTES, label).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot parse {label} {path}: {error}") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_text(cwd: Path, command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=False,
    )
    require(
        result.returncode == 0,
        f"command failed ({result.returncode}): {command!r}: "
        f"{result.stderr[-1000:]}",
    )
    return result.stdout.strip()


def resolve_ref(
    root: Path,
    raw: Any,
    label: str,
    *,
    expected_kind: str | None = None,
) -> Path:
    row = require_dict(raw, label)
    require(
        set(row) == {"path", "sha256", "size_bytes", "kind"},
        f"{label} reference shape mismatch",
    )
    relative_raw = row.get("path")
    require(
        isinstance(relative_raw, str)
        and relative_raw
        and "\\" not in relative_raw,
        f"{label} path is invalid",
    )
    relative = Path(relative_raw)
    require(
        not relative.is_absolute()
        and relative.as_posix() == relative_raw
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"{label} path is unsafe",
    )
    path = root.joinpath(*relative.parts)
    require(
        path.is_file()
        and not path.is_symlink()
        and path.resolve().is_relative_to(root.resolve()),
        f"{label} file is missing or escapes the artifact",
    )
    require(
        isinstance(row.get("sha256"), str)
        and SHA256_RE.fullmatch(row["sha256"]) is not None
        and sha256(path) == row["sha256"],
        f"{label} SHA256 mismatch",
    )
    require(
        isinstance(row.get("size_bytes"), int)
        and row["size_bytes"] == path.stat().st_size,
        f"{label} size mismatch",
    )
    require(
        isinstance(row.get("kind"), str)
        and row["kind"]
        and (expected_kind is None or row["kind"] == expected_kind),
        f"{label} kind mismatch",
    )
    return path


def artifact_index(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"artifact contains symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "evidence.manifest.json":
            continue
        rows.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def verify_artifact_index(root: Path, manifest: dict[str, Any]) -> None:
    recorded = require_list(manifest.get("artifact_index"), "artifact index")
    recomputed = artifact_index(root)
    require(recorded == recomputed, "artifact index does not match artifact tree")
    require(
        manifest.get("artifact_index_sha256")
        == canonical_json_sha256(recomputed),
        "artifact index digest mismatch",
    )


def load_policy(path: Path) -> dict[str, Any]:
    policy = require_dict(read_json(path, "G07A policy"), "G07A policy")
    require(
        policy.get("schema_version") == SCHEMA_VERSION
        and policy.get("artifact_type")
        == "runtime_vnext_g07a_build_iteration_policy"
        and policy.get("repeats") == 5,
        "G07A policy identity mismatch",
    )
    rows = require_list(policy.get("scenarios"), "policy scenarios")
    observed = [
        (
            row.get("name"),
            row.get("kind"),
            row.get("input"),
            row.get("expected_package"),
            row.get("p95_target_seconds"),
        )
        for row in rows
        if isinstance(row, dict)
    ]
    require(observed == list(EXPECTED_SCENARIOS), "G07A policy scenario drift")
    return policy


def parse_cargo_messages(path: Path) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    verbose_lines = 0
    for number, line in enumerate(
        path.read_text(encoding="utf-8", errors="strict").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            if re.fullmatch(r"\[[^\]\r\n]+\] .+", line):
                verbose_lines += 1
                continue
            raise VerificationError(
                f"invalid Cargo JSON at {path}:{number}: {error}"
            ) from error
        require(isinstance(value, dict), f"Cargo row {number} is not an object")
        messages.append(value)
    artifacts = [
        row for row in messages if row.get("reason") == "compiler-artifact"
    ]
    finished = [
        row for row in messages if row.get("reason") == "build-finished"
    ]
    require(artifacts, "Cargo output contains no compiler artifact")
    require(
        len(finished) == 1 and finished[0].get("success") is True,
        "Cargo output contains no unique successful build-finished row",
    )
    require(
        all(isinstance(row.get("fresh"), bool) for row in artifacts),
        "Cargo artifact is missing a boolean fresh field",
    )
    nonfresh = [row for row in artifacts if row["fresh"] is False]
    fresh = [row for row in artifacts if row["fresh"] is True]
    return {
        "message_count": len(messages),
        "verbose_line_count": verbose_lines,
        "compiler_artifact_count": len(artifacts),
        "fresh_artifact_count": len(fresh),
        "nonfresh_artifact_count": len(nonfresh),
        "nonfresh_packages": sorted(
            {
                row["package_id"]
                for row in nonfresh
                if isinstance(row.get("package_id"), str)
            }
        ),
        "build_finished_success": True,
    }


def verify_bounded_step(
    root: Path,
    raw: Any,
    label: str,
    *,
    require_nonempty_logs: bool,
) -> tuple[dict[str, Any], Path, Path]:
    step = require_dict(raw, label)
    command = require_list(step.get("command"), f"{label}.command")
    require(command and all(isinstance(value, str) for value in command), f"{label} command invalid")
    receipt_path = resolve_ref(
        root,
        step.get("bounded_receipt"),
        f"{label}.bounded_receipt",
        expected_kind="bounded-receipt",
    )
    stdout_path = resolve_ref(
        root,
        step.get("stdout"),
        f"{label}.stdout",
        expected_kind="stdout-log",
    )
    stderr_path = resolve_ref(
        root,
        step.get("stderr"),
        f"{label}.stderr",
        expected_kind="stderr-log",
    )
    require(
        receipt_path.parent == stdout_path.parent == stderr_path.parent,
        f"{label} receipt/log roots differ",
    )
    plan = require_dict(
        read_json(receipt_path.parent / "plan.json", f"{label} plan"),
        f"{label} plan",
    )
    receipt = require_dict(
        read_json(receipt_path, f"{label} bounded receipt"),
        f"{label} bounded receipt",
    )
    require(
        receipt.get("schema") == BOUNDED_RECEIPT_SCHEMA
        and receipt.get("status") == "pass"
        and receipt.get("rc") == 0
        and receipt.get("reason") == "command_completed"
        and receipt.get("violation") is None
        and receipt.get("sampling_error_count") == 0
        and receipt.get("cleanup", {}).get("process_group_gone") is True,
        f"{label} bounded receipt is not a clean terminal PASS",
    )
    require(
        receipt.get("command") == command == plan.get("command")
        and receipt.get("cwd") == plan.get("cwd"),
        f"{label} command/cwd binding mismatch",
    )
    require(
        step.get("returncode") == 0
        and isinstance(step.get("duration_seconds"), (int, float))
        and math.isclose(
            float(step["duration_seconds"]),
            float(receipt.get("duration_seconds", -1)),
            abs_tol=1e-6,
        ),
        f"{label} duration/return code mismatch",
    )
    limits = require_dict(receipt.get("limits"), f"{label} limits")
    peaks = require_dict(receipt.get("peaks"), f"{label} peaks")
    require(
        peaks.get("processes", 0) <= limits.get("max_processes", -1)
        and peaks.get("group_threads", 0)
        <= limits.get("max_group_threads", -1)
        and peaks.get("per_process_threads", 0)
        <= limits.get("max_per_process_threads", -1),
        f"{label} resource peak exceeds limit",
    )
    for stream, path in (("stdout", stdout_path), ("stderr", stderr_path)):
        identity = require_dict(receipt.get(stream), f"{label} receipt {stream}")
        require(
            identity.get("sha256") == sha256(path)
            and identity.get("size_bytes") == path.stat().st_size,
            f"{label} {stream} receipt identity mismatch",
        )
    if require_nonempty_logs:
        require(
            stdout_path.stat().st_size > 0 and stderr_path.stat().st_size > 0,
            f"{label} build logs must both be non-empty",
        )
    return receipt, stdout_path, stderr_path


def verify_product_command(command: list[str], policy: dict[str, Any]) -> None:
    require(command[0] == "env" and "cargo" in command, "product command is not env + Cargo")
    cargo_index = command.index("cargo")
    env = set(command[1:cargo_index])
    product = policy["product_build"]
    required_env = {
        "NO_COLOR=1",
        f"CARGO_BUILD_JOBS={product['cargo_jobs']}",
        f"CUDA_COMPUTE_CAP={product['compute_capability']}",
        f"FERRUM_NVCC_THREADS={product['nvcc_threads']}",
        "FERRUM_CUDA_NATIVE_SOURCE_POLICY=cache-only",
    }
    require(required_env.issubset(env), "product command environment drift")
    require(
        sum(value.startswith("FERRUM_NATIVE_OPERATOR_SET_LOCK=") for value in env)
        == 1
        and sum(
            value.startswith("FERRUM_CUDA_BUILD_SUMMARY_RECEIPT=")
            for value in env
        )
        == 1
        and not any(
            value.startswith(("RUSTC_WRAPPER=", "SCCACHE_", "CCACHE_"))
            for value in env
        ),
        "product command lock/summary/cache environment is invalid",
    )
    require(
        command[cargo_index:]
        == [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--jobs",
            str(product["cargo_jobs"]),
            "-p",
            "ferrum-cli",
            "--bin",
            "ferrum",
            "--features",
            product["features"],
            "--message-format=json-render-diagnostics",
            "--timings",
            "-vv",
        ],
        "product Cargo argv drift",
    )


def verify_timing(sample: dict[str, Any], build: dict[str, Any], smoke: dict[str, Any], label: str) -> float:
    duration = sample.get("duration_seconds")
    start = sample.get("timed_monotonic_started_seconds")
    finish = sample.get("timed_monotonic_finished_seconds")
    require(
        isinstance(duration, (int, float))
        and float(duration) > 0
        and isinstance(start, (int, float))
        and isinstance(finish, (int, float))
        and math.isclose(float(finish) - float(start), float(duration), abs_tol=1e-6),
        f"{label} monotonic timing mismatch",
    )
    component = float(build["duration_seconds"]) + float(smoke["duration_seconds"])
    require(
        component <= float(duration) <= component + 15.0,
        f"{label} duration is not bounded by build+smoke receipts",
    )
    try:
        wall_start = datetime.fromisoformat(
            str(sample["timed_started_at"]).replace("Z", "+00:00")
        )
        wall_finish = datetime.fromisoformat(
            str(sample["timed_finished_at"]).replace("Z", "+00:00")
        )
    except (KeyError, ValueError) as error:
        raise VerificationError(f"{label} wall timestamps are invalid") from error
    require(
        abs((wall_finish - wall_start).total_seconds() - float(duration)) <= 15.0,
        f"{label} wall/monotonic timing mismatch",
    )
    return float(duration)


def verify_setup(
    source_root: Path,
    policy: dict[str, Any],
    setup: Any,
    scenario: tuple[Any, ...],
    *,
    mutated_plan: dict[str, Any] | None,
) -> None:
    name, _, input_path, _, _ = scenario
    row = require_dict(setup, f"{name} setup")
    if name == "noop":
        require(row == {"kind": "none"}, "noop setup drift")
        return
    if name == "clean-release":
        require(
            row == {
                "kind": "cargo-clean",
                "target_absent_before_timing": True,
            },
            "clean-release setup drift",
        )
        return
    require(
        row.get("kind") == "content-mutation"
        and row.get("input_path") == input_path
        and row.get("edit_fsync_completed_before_timing") is True
        and row.get("restored_sha256") == row.get("before_sha256")
        and row.get("restored_mtime_ns") == row.get("before_mtime_ns")
        and row.get("during_mtime_ns", -1) >= row.get("before_mtime_ns", 0),
        f"{name} mutation/restore contract mismatch",
    )
    for key in (
        "sentinel_suffix_sha256",
        "before_sha256",
        "during_sha256",
        "restored_sha256",
    ):
        require(
            isinstance(row.get(key), str)
            and SHA256_RE.fullmatch(row[key]) is not None,
            f"{name} {key} is not a SHA256",
        )
    suffix = policy["sentinel_suffix"].encode("ascii")
    require(
        row["sentinel_suffix_sha256"] == hashlib.sha256(suffix).hexdigest(),
        f"{name} sentinel identity mismatch",
    )
    if name != "native-tu":
        original = (source_root / str(input_path)).read_bytes()
        require(
            row["before_sha256"] == hashlib.sha256(original).hexdigest()
            and row["during_sha256"]
            == hashlib.sha256(original + suffix).hexdigest(),
            f"{name} mutation content identity mismatch",
        )
    else:
        require(mutated_plan is not None, "native-TU mutated plan is missing")
        units = {
            item.get("path"): item.get("sha256")
            for item in require_list(
                mutated_plan.get("translation_units"),
                "mutated translation units",
            )
            if isinstance(item, dict)
        }
        require(
            units.get(input_path) == row["during_sha256"],
            "native-TU mutated plan does not bind mutated source",
        )


def verify_cuda_build_summary(path: Path, label: str) -> dict[str, Any]:
    receipt = require_dict(read_json(path, label), label)
    rows = require_list(receipt.get("rows"), f"{label} rows")
    require(
        receipt.get("schema_version") == 1
        and receipt.get("artifact_type")
        == "ferrum_cuda_build_summary_receipt",
        f"{label} identity mismatch",
    )
    native_rows = {
        row.get("artifact"): row
        for row in rows
        if isinstance(row, dict)
        and row.get("artifact") in PRODUCT_NATIVE_UNITS
    }
    require(
        set(native_rows) == PRODUCT_NATIVE_UNITS
        and all(
            row.get("status") == "artifact"
            and row.get("reason") == "native-operator-artifact-set"
            for row in native_rows.values()
        )
        and not any(
            isinstance(row, dict) and row.get("status") == "rejected"
            for row in rows
        ),
        f"{label} does not prove a complete cache-only native artifact set",
    )
    core = sorted(
        str(row.get("artifact")).removeprefix("core-ptx:")
        for row in rows
        if isinstance(row, dict)
        and str(row.get("artifact", "")).startswith("core-ptx:")
        and row.get("status") == "built"
    )
    return {"rows": rows, "native_units": sorted(native_rows), "core_ptx": core}


def verify_product_sample(
    root: Path,
    source_root: Path,
    policy: dict[str, Any],
    source_bundle_members: list[str],
    scenario: tuple[Any, ...],
    sample: dict[str, Any],
    index: int,
) -> float:
    name, _, _, expected_package, _ = scenario
    label = f"{name} sample {index}"
    require(
        sample.get("schema_version") == SCHEMA_VERSION
        and sample.get("sample_id") == f"{name}-{index}"
        and sample.get("status") == "pass",
        f"{label} identity/status mismatch",
    )
    prewarm = sample.get("prewarm")
    if name == "clean-release":
        require(prewarm is None, "clean-release must not prewarm Cargo target")
    else:
        prewarm_receipt, _, _ = verify_bounded_step(
            root,
            prewarm,
            f"{label} prewarm",
            require_nonempty_logs=True,
        )
        verify_product_command(prewarm_receipt["command"], policy)
    build_receipt, cargo_stdout, cargo_stderr = verify_bounded_step(
        root,
        sample.get("build"),
        f"{label} build",
        require_nonempty_logs=True,
    )
    verify_product_command(build_receipt["command"], policy)
    smoke_receipt, smoke_stdout, _ = verify_bounded_step(
        root,
        sample.get("smoke"),
        f"{label} smoke",
        require_nonempty_logs=False,
    )
    require(
        len(smoke_receipt["command"]) == 2
        and smoke_receipt["command"][1] == "--version"
        and smoke_stdout.stat().st_size > 0,
        f"{label} binary smoke command/output mismatch",
    )
    build = require_dict(sample.get("build"), f"{label} build record")
    cargo_summary = parse_cargo_messages(cargo_stdout)
    require(
        build.get("cargo_summary") == cargo_summary,
        f"{label} Cargo summary is not independently reproducible",
    )
    if name == "noop":
        require(
            cargo_summary["nonfresh_artifact_count"] == 0,
            "noop rebuilt a Cargo artifact",
        )
    elif expected_package is not None:
        require(
            any(
                str(expected_package) in package
                for package in cargo_summary["nonfresh_packages"]
            ),
            f"{label} did not invalidate {expected_package}",
        )
    stderr = cargo_stderr.read_text(encoding="utf-8", errors="strict")
    require(
        not any(member in stderr for member in source_bundle_members),
        f"{label} product build referenced external native source",
    )
    summary_raw = build.get("cuda_build_summary")
    if name in {"core-ptx", "clean-release"}:
        require(summary_raw is not None, f"{label} CUDA build summary is missing")
    if summary_raw is not None:
        summary_path = resolve_ref(
            root,
            summary_raw,
            f"{label} CUDA build summary",
            expected_kind="cuda-build-summary",
        )
        summary = verify_cuda_build_summary(summary_path, f"{label} CUDA summary")
        if name == "core-ptx":
            require(
                summary["core_ptx"] == ["kernels/add_bias.cu"],
                "core-ptx did not build exactly add_bias.cu",
            )
    native_signal = require_dict(build.get("native_signal"), f"{label} native signal")
    require(
        native_signal.get("compiled_native_tu_count") == 0
        and native_signal.get("compiled_native_tu_paths") == [],
        f"{label} reports product native source compilation",
    )
    resolve_ref(
        root,
        build.get("cargo_timing"),
        f"{label} Cargo timing",
        expected_kind="cargo-timing",
    )
    output = require_dict(sample.get("output"), f"{label} output")
    binary = resolve_ref(
        root,
        require_dict(output.get("artifact"), f"{label} output artifact"),
        f"{label} output artifact",
        expected_kind="binary",
    )
    require(
        output.get("kind") == "binary"
        and output.get("sha256") == sha256(binary),
        f"{label} binary identity mismatch",
    )
    receipt_path = resolve_ref(
        root,
        build["bounded_receipt"],
        f"{label} build receipt path",
    )
    sample_path = receipt_path.parent.parent / "sample.json"
    require(
        read_json(sample_path, f"{label} sample file") == sample,
        f"{label} embedded/file sample mismatch",
    )
    verify_setup(source_root, policy, sample.get("setup"), scenario, mutated_plan=None)
    return verify_timing(sample, build_receipt, smoke_receipt, label)


def verify_native_sample(
    root: Path,
    source_root: Path,
    policy: dict[str, Any],
    scenario: tuple[Any, ...],
    sample: dict[str, Any],
    index: int,
) -> float:
    name, _, expected_input, _, _ = scenario
    label = f"{name} sample {index}"
    require(
        sample.get("sample_id") == f"{name}-{index}"
        and sample.get("status") == "pass",
        f"{label} identity/status mismatch",
    )
    prewarm_receipt, _, _ = verify_bounded_step(
        root,
        sample.get("prewarm"),
        f"{label} prewarm",
        require_nonempty_logs=False,
    )
    require(
        "source-build" in prewarm_receipt["command"],
        f"{label} prewarm is not a source build",
    )
    plan_receipt, _, _ = verify_bounded_step(
        root,
        sample.get("plan"),
        f"{label} mutated lock",
        require_nonempty_logs=False,
    )
    require(
        "lock-source" in plan_receipt["command"],
        f"{label} did not lock mutated native source",
    )
    build_receipt, _, _ = verify_bounded_step(
        root,
        sample.get("build"),
        f"{label} build",
        require_nonempty_logs=False,
    )
    smoke_receipt, smoke_stdout, _ = verify_bounded_step(
        root,
        sample.get("smoke"),
        f"{label} smoke",
        require_nonempty_logs=False,
    )
    require(
        len(smoke_receipt["command"]) == 3
        and smoke_receipt["command"][1] == "t"
        and smoke_stdout.stat().st_size > 0,
        f"{label} archive smoke mismatch",
    )
    build = require_dict(sample.get("build"), f"{label} build record")
    receipt_path = resolve_ref(
        root,
        build.get("source_build_receipt"),
        f"{label} source receipt",
        expected_kind="native-source-build-receipt",
    )
    receipt = require_dict(read_json(receipt_path, f"{label} source receipt"), f"{label} source receipt")
    require(
        receipt.get("schema_version") == SOURCE_BUILD_RECEIPT_SCHEMA_VERSION
        and receipt.get("status") == "pass"
        and receipt.get("plan_only") is False
        and receipt.get("compiled_translation_units") == [expected_input],
        f"{label} did not compile exactly one expected TU",
    )
    sample_root = receipt_path.parent.parent
    mutated_plan = require_dict(
        read_json(sample_root / "mutated.plan.json", f"{label} mutated plan"),
        f"{label} mutated plan",
    )
    planned_units = [
        row.get("path")
        for row in require_list(
            mutated_plan.get("translation_units"),
            f"{label} translation units",
        )
        if isinstance(row, dict)
    ]
    cache_hits = require_list(
        receipt.get("cache_hit_translation_units"),
        f"{label} cache hits",
    )
    require(
        sorted([expected_input, *cache_hits]) == sorted(planned_units)
        and expected_input not in cache_hits
        and len(set(planned_units)) == len(planned_units),
        f"{label} cache hit/compiled partition mismatch",
    )
    require(
        build.get("compiled_translation_units") == [expected_input]
        and build.get("cache_hit_translation_units") == cache_hits,
        f"{label} source receipt projection mismatch",
    )
    require(
        sample.get("cache", {}).get("scope")
        == "verified-base-cloned-per-sample",
        f"{label} object cache isolation policy mismatch",
    )
    output = require_dict(sample.get("output"), f"{label} output")
    archive = resolve_ref(
        root,
        output.get("artifact"),
        f"{label} native archive",
        expected_kind="native-archive",
    )
    require(
        output.get("kind") == "native-archive"
        and output.get("sha256") == sha256(archive)
        and receipt.get("archive_sha256") == sha256(archive),
        f"{label} archive identity mismatch",
    )
    require(
        read_json(sample_root / "sample.json", f"{label} sample file")
        == sample,
        f"{label} embedded/file sample mismatch",
    )
    verify_setup(
        source_root,
        policy,
        sample.get("setup"),
        scenario,
        mutated_plan=mutated_plan,
    )
    return verify_timing(sample, build_receipt, smoke_receipt, label)


def nearest_rank(values: list[float], percentile: float) -> float:
    require(values and all(value > 0 for value in values), "timings are invalid")
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def verify_dependencies(
    root: Path,
    manifest: dict[str, Any],
    source: dict[str, Any],
    *,
    require_fresh_inputs: bool,
) -> tuple[dict[str, Any], list[str]]:
    inputs = require_dict(manifest.get("inputs"), "manifest inputs")
    require(
        set(inputs)
        == {
            "policy",
            "g00f",
            "s1",
            "source_bundle",
            "native_operator_set_lock",
        },
        "manifest input set mismatch",
    )
    policy_copy = resolve_ref(root, inputs["policy"], "policy input")
    require(
        policy_copy.read_bytes() == POLICY_PATH.read_bytes(),
        "copied policy differs from checked-in policy",
    )
    policy = load_policy(policy_copy)
    g00f_path = resolve_ref(root, inputs["g00f"], "G00F input")
    g00f = require_dict(read_json(g00f_path, "G00F input"), "G00F input")
    require(
        g00f.get("schema_version") == 1
        and g00f.get("artifact_type") == "runtime_vnext_g00f_facts_manifest"
        and g00f.get("checkpoint_id") == "G00F"
        and g00f.get("lane") == "runtime-vnext-g00f"
        and g00f.get("status") == "pass"
        and str(g00f.get("pass_line", "")).startswith(
            "FERRUM RUNTIME VNEXT G00F FACTS PASS:"
        ),
        "G00F input identity/status mismatch",
    )
    s1_path = resolve_ref(root, inputs["s1"], "S1 input")
    s1 = require_dict(read_json(s1_path, "S1 input"), "S1 input")
    require(
        s1.get("schema_version") == 1
        and s1.get("artifact_type")
        == "runtime_vnext_s1_cuda_basic_slice_manifest"
        and s1.get("lane") == "runtime-vnext-s1-cuda"
        and s1.get("status") == "pass"
        and str(s1.get("pass_line", "")).startswith(
            "FERRUM RUNTIME VNEXT S1 CUDA BASIC SLICE PASS:"
        ),
        "S1 input identity/status mismatch",
    )
    if require_fresh_inputs:
        require(
            g00f.get("source", {}).get("git_sha") == source["git_sha"]
            and g00f.get("source", {}).get("git_tree_sha")
            == source["git_tree_sha"]
            and s1.get("source_git_sha") == source["git_sha"],
            "canonical G07A dependency manifest is stale",
        )
    bundle_path = resolve_ref(root, inputs["source_bundle"], "source bundle input")
    checked_bundle = (
        REPO_ROOT
        / "native-operators/cuda/source-bundles/ferrum-native-cuda-v1.json"
    )
    require(
        bundle_path.read_bytes() == checked_bundle.read_bytes(),
        "copied native source bundle manifest drifted",
    )
    bundle = require_dict(read_json(bundle_path, "source bundle input"), "source bundle input")
    members = [
        row["path"]
        for row in require_list(bundle.get("members"), "source bundle members")
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    ]
    require(len(members) == 53 and len(set(members)) == 53, "source bundle member set drift")
    lock_path = resolve_ref(
        root,
        inputs["native_operator_set_lock"],
        "native operator set lock",
    )
    lock = require_dict(read_json(lock_path, "native operator set lock"), "native operator set lock")
    artifacts = require_list(lock.get("artifacts"), "native operator set artifacts")
    operators = {
        row.get("operator")
        for row in artifacts
        if isinstance(row, dict)
    }
    require(
        lock.get("schema_version") == 5
        and isinstance(lock.get("g03_catalog_sha256"), str)
        and SHA256_RE.fullmatch(lock["g03_catalog_sha256"]) is not None
        and operators == PRODUCT_NATIVE_OPERATORS
        and len(artifacts) == len(PRODUCT_NATIVE_OPERATORS),
        "native operator set lock is incomplete",
    )
    return policy, members


def verify_crate_graph(root: Path, manifest: dict[str, Any]) -> None:
    path = resolve_ref(
        root,
        manifest.get("crate_graph"),
        "crate graph",
        expected_kind="cargo-metadata",
    )
    graph = require_dict(read_json(path, "crate graph"), "crate graph")
    packages = require_list(graph.get("packages"), "crate graph packages")
    names = {
        row.get("name")
        for row in packages
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    require(
        {
            "ferrum-cli",
            "ferrum-engine",
            "ferrum-kernels",
            "ferrum-models",
            "ferrum-native-ops",
            "ferrum-native-ops-builder",
        }.issubset(names),
        "crate graph is missing required build/runtime domains",
    )
    forbidden = (
        REPO_ROOT / "crates/ferrum-kernels/vllm_marlin",
        REPO_ROOT / "crates/ferrum-kernels/kernels/vllm_marlin_moe",
        REPO_ROOT / "crates/ferrum-kernels/kernels/vllm_attn",
    )
    require(
        not any(path.exists() for path in forbidden),
        "product repository still contains a forbidden third-party native source tree",
    )


def verify_manifest(
    root: Path,
    source_root: Path,
    *,
    require_canonical: bool,
    verify_checkout: bool,
) -> dict[str, Any]:
    root = root.resolve()
    source_root = source_root.resolve()
    require(root.is_dir(), f"artifact root is missing: {root}")
    require(not (root / "failure.json").exists(), "evidence contains failure.json")
    manifest = require_dict(
        read_json(root / "evidence.manifest.json", "evidence manifest"),
        "evidence manifest",
    )
    require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type")
        == "runtime_vnext_g07a_build_iteration_evidence"
        and manifest.get("status") == "ready"
        and manifest.get("mode") in {"diagnostic", "canonical"},
        "G07A evidence identity/status mismatch",
    )
    if require_canonical:
        require(manifest.get("mode") == "canonical", "canonical evidence is required")
    require(
        set(require_list(manifest.get("does_not_prove"), "does_not_prove"))
        == DOES_NOT_PROVE,
        "G07A evidence proof boundary mismatch",
    )
    source = require_dict(manifest.get("source"), "source identity")
    require(
        GIT_SHA_RE.fullmatch(str(source.get("git_sha"))) is not None
        and GIT_SHA_RE.fullmatch(str(source.get("git_tree_sha"))) is not None
        and source.get("dirty") is False
        and source.get("status_short") == [],
        "source identity is invalid",
    )
    if verify_checkout:
        require(
            run_text(source_root, ["git", "rev-parse", "HEAD"])
            == source["git_sha"]
            and run_text(source_root, ["git", "rev-parse", "HEAD^{tree}"])
            == source["git_tree_sha"]
            and run_text(
                source_root,
                ["git", "status", "--short", "--untracked-files=all"],
            )
            == "",
            "evidence source is stale or checkout is dirty",
        )
    hardware = require_dict(manifest.get("hardware"), "hardware")
    raw_hardware = require_dict(hardware.get("raw"), "hardware raw")
    require(
        hardware.get("gpu_count") == 1
        and "RTX 4090" in str(hardware.get("gpu_model"))
        and hardware.get("fingerprint") == canonical_json_sha256(raw_hardware)
        and set(raw_hardware)
        == {"gpu", "uname", "cpu", "nvcc", "rustc", "cargo", "tools"},
        "fixed RTX 4090 hardware/toolchain identity mismatch",
    )
    require(
        manifest.get("compiler_cache")
        == {
            "rustc_wrapper": None,
            "sccache": "disabled",
            "ccache": "disabled",
        },
        "compiler cache policy mismatch",
    )
    policy, source_members = verify_dependencies(
        root,
        manifest,
        source,
        require_fresh_inputs=require_canonical,
    )
    verify_crate_graph(root, manifest)
    verify_bounded_step(
        root,
        manifest.get("builder_setup"),
        "builder setup",
        require_nonempty_logs=True,
    )
    plan = require_dict(read_json(root / "lane-plan.json", "lane plan"), "lane plan")
    require(
        plan.get("lane") == "runtime-vnext-g07a-build-iteration"
        and plan.get("mode") == manifest["mode"]
        and plan.get("source") == source
        and plan.get("hardware_fingerprint") == hardware["fingerprint"]
        and plan.get("compiler_cache") == manifest["compiler_cache"],
        "lane plan binding mismatch",
    )
    repeats = 5 if manifest["mode"] == "canonical" else 1
    require(
        manifest.get("repeats") == repeats
        and plan.get("repeats") == repeats,
        "sample repeat count mismatch",
    )
    rows = require_list(manifest.get("scenarios"), "scenario rows")
    require(len(rows) == len(EXPECTED_SCENARIOS), "scenario row count mismatch")
    recomputed_targets: dict[str, dict[str, Any]] = {}
    native_counts: list[int] = []
    product_native_count = 0
    for scenario, row_raw in zip(EXPECTED_SCENARIOS, rows, strict=True):
        row = require_dict(row_raw, f"{scenario[0]} scenario")
        name, kind, input_path, expected_package, target = scenario
        require(
            row.get("name") == name
            and row.get("kind") == kind
            and row.get("input") == input_path
            and row.get("expected_package") == expected_package
            and row.get("p95_target_seconds") == target
            and row.get("sample_count") == repeats,
            f"{name} scenario identity/count mismatch",
        )
        samples = require_list(row.get("samples"), f"{name} samples")
        require(len(samples) == repeats, f"{name} sample list count mismatch")
        durations: list[float] = []
        for index, sample_raw in enumerate(samples, start=1):
            sample = require_dict(sample_raw, f"{name} sample {index}")
            require(
                sample.get("source_git_sha") == source["git_sha"]
                and sample.get("source_tree_sha") == source["git_tree_sha"],
                f"{name} sample {index} source mismatch",
            )
            if kind == "native_source_build":
                duration = verify_native_sample(
                    root,
                    source_root,
                    policy,
                    scenario,
                    sample,
                    index,
                )
                native_counts.append(
                    len(sample["build"]["compiled_translation_units"])
                )
            else:
                duration = verify_product_sample(
                    root,
                    source_root,
                    policy,
                    source_members,
                    scenario,
                    sample,
                    index,
                )
                product_native_count += sample["build"]["native_signal"][
                    "compiled_native_tu_count"
                ]
            durations.append(duration)
        ordered = sorted(durations)
        p50 = nearest_rank(durations, 0.50)
        p95 = nearest_rank(durations, 0.95)
        target_met = p95 <= target
        require(
            row.get("durations_seconds") == ordered
            and math.isclose(float(row.get("p50_seconds", -1)), p50, abs_tol=1e-6)
            and math.isclose(float(row.get("p95_seconds", -1)), p95, abs_tol=1e-6)
            and row.get("target_met") is target_met,
            f"{name} percentile summary mismatch",
        )
        if require_canonical:
            require(target_met, f"{name} p95 misses target: {p95:.3f}s > {target}s")
        recomputed_targets[name] = {
            "p95_seconds": p95,
            "target_seconds": target,
            "target_met": target_met,
        }
    invalidation_path = resolve_ref(
        root,
        manifest.get("invalidation_report"),
        "invalidation report",
        expected_kind="invalidation-report",
    )
    invalidation = require_dict(
        read_json(invalidation_path, "invalidation report"),
        "invalidation report",
    )
    require(
        invalidation
        == {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_g07a_invalidation_report",
            "native_product_source_compile_count": product_native_count,
            "native_tu_compiled_counts": native_counts,
            "scenario_targets": recomputed_targets,
        },
        "invalidation report is not independently reproducible",
    )
    require(product_native_count == 0, "product graph compiled native source")
    require(
        native_counts == [1] * repeats,
        "native-TU samples did not each compile exactly one TU",
    )
    verify_artifact_index(root, manifest)
    return {
        "mode": manifest["mode"],
        "source_git_sha": source["git_sha"],
        "hardware_fingerprint": hardware["fingerprint"],
        "scenario_targets": recomputed_targets,
    }


def expect_reject(action: Any, label: str) -> None:
    try:
        action()
    except VerificationError:
        return
    raise AssertionError(f"{label}: verifier accepted invalid evidence")


def self_test() -> None:
    policy = load_policy(POLICY_PATH)
    require(
        nearest_rank([1.0, 2.0, 3.0, 4.0, 5.0], 0.95) == 5.0,
        "nearest-rank p95 self-test failed",
    )
    with tempfile.TemporaryDirectory(prefix="g07a-validator-selftest-") as raw:
        root = Path(raw)
        cargo = root / "cargo.jsonl"
        cargo.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "reason": "compiler-artifact",
                            "package_id": "ferrum-models 0.8.0",
                            "fresh": False,
                        }
                    ),
                    json.dumps({"reason": "build-finished", "success": True}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        parsed = parse_cargo_messages(cargo)
        require(
            parsed["nonfresh_artifact_count"] == 1
            and parsed["nonfresh_packages"] == ["ferrum-models 0.8.0"],
            "Cargo parser self-test failed",
        )
        bad = cargo.read_text(encoding="utf-8").replace(
            '"success": true',
            '"success": false',
        )
        cargo.write_text(bad, encoding="utf-8")
        expect_reject(
            lambda: parse_cargo_messages(cargo),
            "failed Cargo build",
        )
        receipt = root / "source-build.receipt.json"
        plan = root / "mutated.plan.json"
        plan.write_text(
            json.dumps(
                {
                    "translation_units": [
                        {
                            "path": policy["native_build"]["input"],
                            "sha256": "1" * 64,
                        },
                        {"path": "other.cu", "sha256": "2" * 64},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        valid_receipt = {
            "schema_version": SOURCE_BUILD_RECEIPT_SCHEMA_VERSION,
            "status": "pass",
            "plan_only": False,
            "compiled_translation_units": [policy["native_build"]["input"]],
            "cache_hit_translation_units": ["other.cu"],
        }
        receipt.write_text(
            json.dumps(valid_receipt) + "\n",
            encoding="utf-8",
        )
        loaded = require_dict(read_json(receipt, "selftest receipt"), "selftest receipt")
        require(
            loaded["schema_version"] == SOURCE_BUILD_RECEIPT_SCHEMA_VERSION
            and sorted(
                loaded["compiled_translation_units"]
                + loaded["cache_hit_translation_units"]
            )
            == sorted(
                row["path"]
                for row in read_json(plan, "selftest plan")["translation_units"]
            ),
            "native receipt partition self-test failed",
        )
        tampered = copy.deepcopy(valid_receipt)
        tampered["schema_version"] = 2
        expect_reject(
            lambda: require(
                tampered["schema_version"]
                == SOURCE_BUILD_RECEIPT_SCHEMA_VERSION,
                "stale source receipt schema",
            ),
            "stale source receipt schema",
        )
    print(SELFTEST_PASS_LINE)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--artifact-root", type=Path)
    result.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT,
    )
    result.add_argument("--require-canonical", action="store_true")
    result.add_argument("--skip-checkout-freshness", action="store_true")
    result.add_argument("--self-test", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        try:
            self_test()
        except (AssertionError, OSError, VerificationError) as error:
            print(
                "FERRUM RUNTIME VNEXT G07A BUILD ITERATION VALIDATOR "
                f"SELFTEST REJECT: {error}",
                file=sys.stderr,
            )
            return 1
        return 0
    if args.artifact_root is None:
        print("--artifact-root is required", file=sys.stderr)
        return 2
    root = args.artifact_root.expanduser().resolve()
    try:
        summary = verify_manifest(
            root,
            args.source_root.expanduser().resolve(),
            require_canonical=args.require_canonical,
            verify_checkout=not args.skip_checkout_freshness,
        )
    except (
        OSError,
        subprocess.SubprocessError,
        VerificationError,
        ValueError,
    ) as error:
        print(
            f"{EVIDENCE_PASS_PREFIX} REJECTED: {root}: {error}",
            file=sys.stderr,
        )
        return 1
    print(
        f"{EVIDENCE_PASS_PREFIX}: {root}: "
        f"mode={summary['mode']} source={summary['source_git_sha'][:8]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
