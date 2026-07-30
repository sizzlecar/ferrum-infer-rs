#!/usr/bin/env python3
"""Collect bounded G07A CUDA build-iteration evidence on one fixed RTX 4090."""

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
from typing import Any, Sequence

import bounded_command
import native_operator_source_bundle as source_bundle


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = (
    REPO_ROOT
    / "scripts/release/configs/runtime_vnext_g07a_build_iteration.json"
)
SCHEMA_VERSION = 2
SOURCE_BUILD_RECEIPT_SCHEMA_VERSION = 7
ARTIFACT_TYPE = "runtime_vnext_g07a_build_iteration_evidence"
PASS_LINE = "FERRUM RUNTIME VNEXT G07A BUILD ITERATION EVIDENCE READY"
KEEP_LINE = "FERRUM RUNTIME VNEXT G07A BUILD ITERATION DIAGNOSTIC KEEP"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT G07A BUILD ITERATION SELFTEST PASS"
EXPECTED_SCENARIOS = (
    "noop",
    "rust-model-leaf",
    "rust-runtime-leaf",
    "core-ptx",
    "native-tu",
    "clean-release",
)
PRODUCT_NATIVE_ARTIFACTS = {
    "marlin",
    "vllm_marlin",
    "vllm_moe_marlin",
    "vllm_paged_attn",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class BuildIterationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildIterationError(message)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def ensure_distinct_roots(
    *,
    source_root: Path,
    native_source_root: Path,
    evidence_root: Path,
    worktree_root: Path,
    target_root: Path,
    object_cache: Path,
) -> None:
    roots = {
        "source": source_root,
        "native-source": native_source_root,
        "evidence": evidence_root,
        "worktree": worktree_root,
        "target": target_root,
        "object-cache": object_cache,
    }
    home = Path.home().resolve()
    for name, path in roots.items():
        resolved = path.resolve()
        require(
            resolved not in {Path("/"), home},
            f"G07A {name} root is unsafe: {resolved}",
        )
    names = list(roots)
    for index, left_name in enumerate(names):
        left = roots[left_name].resolve()
        for right_name in names[index + 1 :]:
            right = roots[right_name].resolve()
            require(
                not left.is_relative_to(right)
                and not right.is_relative_to(left),
                f"G07A roots overlap: {left_name}={left} "
                f"{right_name}={right}",
            )


def claim_managed_root(
    root: Path,
    *,
    role: str,
    source_git_sha: str,
    resume: bool,
) -> None:
    marker = root / ".ferrum-g07a-managed.json"
    expected = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_g07a_managed_root",
        "role": role,
        "created_source_git_sha": source_git_sha,
    }
    if root.exists():
        require(root.is_dir() and not root.is_symlink(), f"{role} root is unsafe")
        entries = list(root.iterdir())
        if marker.exists():
            observed = read_json(marker, f"{role} ownership marker")
            require(
                observed.get("schema_version") == SCHEMA_VERSION
                and observed.get("artifact_type")
                == "runtime_vnext_g07a_managed_root"
                and observed.get("role") == role
                and observed.get("created_source_git_sha") == source_git_sha,
                f"{role} ownership marker mismatch",
            )
            return
        require(
            not entries,
            f"{role} root is non-empty and has no G07A ownership marker: {root}",
        )
    else:
        root.mkdir(parents=True)
    require(
        resume is False or marker.exists() or not list(root.iterdir()),
        f"{role} resume root has no ownership marker",
    )
    write_json(marker, expected)


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildIterationError(f"cannot read {label} {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must contain one JSON object")
    return value


def run_text(
    cwd: Path,
    command: Sequence[str],
    *,
    timeout: int = 60,
) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    require(
        result.returncode == 0,
        f"command failed ({result.returncode}): {list(command)!r}: "
        f"{result.stderr[-2000:]}",
    )
    return result.stdout.strip()


def resolve_tool(raw: str, label: str) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        found = shutil.which(raw)
        require(found is not None, f"{label} is not on PATH: {raw}")
        resolved = Path(found).resolve()  # type: ignore[arg-type]
    require(
        resolved.is_file() and os.access(resolved, os.X_OK),
        f"{label} is not executable: {resolved}",
    )
    return resolved


def artifact_ref(root: Path, path: Path, kind: str) -> dict[str, Any]:
    path = path.resolve()
    require(path.is_file() and not path.is_symlink(), f"artifact is missing: {path}")
    try:
        relative = path.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise BuildIterationError(f"artifact escapes evidence root: {path}") from error
    return {
        "path": relative,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "kind": kind,
    }


def verify_ref(root: Path, raw: Any, label: str) -> Path:
    require(isinstance(raw, dict), f"{label} must be an artifact reference")
    require(
        set(raw) == {"path", "sha256", "size_bytes", "kind"},
        f"{label} reference shape mismatch",
    )
    relative = raw.get("path")
    require(
        isinstance(relative, str)
        and relative
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts,
        f"{label} path is invalid",
    )
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"{label} escapes evidence root")
    require(path.is_file() and not path.is_symlink(), f"{label} is missing")
    require(
        isinstance(raw.get("size_bytes"), int)
        and raw["size_bytes"] == path.stat().st_size,
        f"{label} size mismatch",
    )
    require(
        isinstance(raw.get("sha256"), str)
        and SHA256_RE.fullmatch(raw["sha256"]) is not None
        and raw["sha256"] == sha256(path),
        f"{label} SHA256 mismatch",
    )
    require(
        isinstance(raw.get("kind"), str) and raw["kind"],
        f"{label} kind is invalid",
    )
    return path


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    policy = read_json(path, "G07A policy")
    require(
        policy.get("schema_version") == SCHEMA_VERSION
        and policy.get("artifact_type")
        == "runtime_vnext_g07a_build_iteration_policy",
        "G07A policy identity mismatch",
    )
    require(policy.get("repeats") == 5, "G07A policy must require five samples")
    suffix = policy.get("sentinel_suffix")
    require(
        isinstance(suffix, str)
        and suffix.startswith("\n// ")
        and suffix.endswith("\n")
        and suffix.isascii(),
        "G07A sentinel suffix is invalid",
    )
    scenarios = policy.get("scenarios")
    require(
        isinstance(scenarios, list)
        and [row.get("name") for row in scenarios if isinstance(row, dict)]
        == list(EXPECTED_SCENARIOS),
        "G07A scenario order is invalid",
    )
    expected = {
        "noop": ("cargo_incremental", 30),
        "rust-model-leaf": ("cargo_incremental", 90),
        "rust-runtime-leaf": ("cargo_incremental", 90),
        "core-ptx": ("cargo_incremental", 120),
        "native-tu": ("native_source_build", 300),
        "clean-release": ("cargo_clean_release", 900),
    }
    for row in scenarios:
        require(
            isinstance(row, dict)
            and set(row)
            == {
                "deadline_seconds",
                "expected_package",
                "input",
                "kind",
                "name",
                "p95_target_seconds",
            },
            "G07A scenario shape mismatch",
        )
        kind, target = expected[row["name"]]
        require(
            row["kind"] == kind
            and row["p95_target_seconds"] == target
            and isinstance(row["deadline_seconds"], int)
            and row["deadline_seconds"] > target,
            f"G07A scenario policy drift: {row['name']}",
        )
    product = policy.get("product_build")
    require(
        isinstance(product, dict)
        and set(product)
        == {
            "bootstrap_source_policy",
            "cargo_jobs",
            "compute_capability",
            "core_ptx_inputs",
            "core_ptx_source_policy",
            "default_source_policy",
            "features",
            "nvcc_threads",
            "profile",
        },
        "G07A product build policy shape drift",
    )
    core_ptx_inputs = product["core_ptx_inputs"]
    require(
        product["bootstrap_source_policy"] == "allow"
        and product["cargo_jobs"] == 4
        and product["compute_capability"] == "89"
        and product["core_ptx_source_policy"] == "allow"
        and product["default_source_policy"] == "cache-only"
        and product["features"]
        == "cuda,vllm-moe-marlin,vllm-paged-attn-v2"
        and product["nvcc_threads"] == 4
        and product["profile"] == "release"
        and isinstance(core_ptx_inputs, list)
        and len(core_ptx_inputs) == 40
        and len(set(core_ptx_inputs)) == len(core_ptx_inputs)
        and all(
            isinstance(path, str)
            and path.startswith("kernels/")
            and path.endswith(".cu")
            for path in core_ptx_inputs
        )
        and "kernels/add_bias.cu" in core_ptx_inputs,
        "G07A product build policy drift",
    )
    native = policy.get("native_build")
    require(
        isinstance(native, dict)
        and native.get("compute_capability") == "sm_89"
        and native.get("nvcc_threads") == 4
        and native.get("operator") == "ferrum.cuda.vllm_marlin",
        "G07A native build policy drift",
    )
    return policy


def source_identity(source_root: Path) -> dict[str, Any]:
    git_sha = run_text(source_root, ["git", "rev-parse", "HEAD"])
    tree_sha = run_text(source_root, ["git", "rev-parse", "HEAD^{tree}"])
    status = run_text(
        source_root,
        ["git", "status", "--short", "--untracked-files=all"],
    ).splitlines()
    require(GIT_SHA_RE.fullmatch(git_sha) is not None, "invalid source Git SHA")
    require(GIT_SHA_RE.fullmatch(tree_sha) is not None, "invalid source tree SHA")
    require(not status, f"G07A requires clean source: {status}")
    return {
        "git_sha": git_sha,
        "git_tree_sha": tree_sha,
        "dirty": False,
        "status_short": [],
    }


def tool_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def hardware_identity(
    source_root: Path,
    tools: dict[str, Path],
) -> dict[str, Any]:
    names = run_text(
        source_root,
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    ).splitlines()
    require(len(names) == 1, f"G07A requires exactly one GPU, found {names}")
    require("RTX 4090" in names[0], f"G07A requires RTX 4090, found {names[0]!r}")
    raw = {
        "gpu": run_text(
            source_root,
            [
                "nvidia-smi",
                (
                    "--query-gpu=index,name,uuid,memory.total,driver_version,"
                    "persistence_mode,power.limit"
                ),
                "--format=csv,noheader",
            ],
        ),
        "uname": run_text(source_root, ["uname", "-a"]),
        "cpu": run_text(source_root, ["lscpu", "--json"]),
        "nvcc": run_text(source_root, [str(tools["nvcc"]), "--version"]),
        "rustc": run_text(source_root, ["rustc", "-vV"]),
        "cargo": run_text(source_root, ["cargo", "-V"]),
        "tools": {name: tool_identity(path) for name, path in tools.items()},
    }
    return {
        "gpu_count": 1,
        "gpu_model": names[0].strip(),
        "fingerprint": canonical_json_sha256(raw),
        "raw": raw,
    }


def validate_no_hidden_compiler_cache() -> dict[str, Any]:
    relevant = {
        key: value
        for key, value in os.environ.items()
        if key == "RUSTC_WRAPPER"
        or key.startswith("SCCACHE_")
        or key.startswith("CCACHE_")
    }
    require(
        not relevant,
        f"G07A canonical timing forbids undeclared compiler wrappers/caches: {relevant}",
    )
    return {
        "rustc_wrapper": None,
        "sccache": "disabled",
        "ccache": "disabled",
    }


def run_bounded_step(
    *,
    evidence_root: Path,
    step_root: Path,
    cwd: Path,
    command: Sequence[str],
    expected_seconds: int,
    deadline_seconds: int,
    progress_signal: str,
    lane_deadline: float,
    max_processes: int = 32,
    max_group_threads: int = 128,
    max_per_process_threads: int = 48,
) -> dict[str, Any]:
    remaining = int(lane_deadline - time.monotonic())
    require(
        remaining >= expected_seconds,
        f"lane deadline cannot admit {step_root.name}: "
        f"remaining={remaining}s expected={expected_seconds}s",
    )
    deadline_seconds = min(deadline_seconds, remaining)
    require(not step_root.exists(), f"step output already exists: {step_root}")
    step_root.mkdir(parents=True)
    write_json(
        step_root / "plan.json",
        {
            "schema_version": SCHEMA_VERSION,
            "command": list(command),
            "cwd": str(cwd),
            "expected_duration_seconds": expected_seconds,
            "hard_deadline_seconds": deadline_seconds,
            "progress_signal": progress_signal,
            "started_at": now_iso(),
        },
    )
    wrapper_rc, receipt = bounded_command.run_bounded_command(
        command=list(command),
        cwd=cwd,
        receipt_path=step_root / "bounded.receipt.json",
        stdout_path=step_root / "stdout.log",
        stderr_path=step_root / "stderr.log",
        limits=bounded_command.Limits(
            wall_timeout_seconds=float(deadline_seconds),
            max_processes=max_processes,
            max_group_threads=max_group_threads,
            max_per_process_threads=max_per_process_threads,
            sample_interval_seconds=0.2,
            max_sampling_errors=3,
            term_grace_seconds=3.0,
        ),
    )
    require(
        wrapper_rc == 0
        and receipt.get("schema") == "ferrum.bounded-command-receipt.v1"
        and receipt.get("status") == "pass"
        and receipt.get("rc") == 0
        and receipt.get("command") == list(command)
        and Path(str(receipt.get("cwd"))).resolve() == cwd.resolve()
        and receipt.get("violation") is None
        and receipt.get("sampling_error_count") == 0
        and receipt.get("cleanup", {}).get("process_group_gone") is True,
        f"bounded step failed: {step_root / 'bounded.receipt.json'}",
    )
    limits = receipt.get("limits")
    require(
        isinstance(limits, dict)
        and limits.get("wall_timeout_seconds") == float(deadline_seconds)
        and limits.get("max_processes") == max_processes
        and limits.get("max_group_threads") == max_group_threads
        and limits.get("max_per_process_threads") == max_per_process_threads,
        f"bounded step limits drifted: {step_root / 'bounded.receipt.json'}",
    )
    return {
        "command": list(command),
        "bounded_receipt": artifact_ref(
            evidence_root,
            step_root / "bounded.receipt.json",
            "bounded-receipt",
        ),
        "stdout": artifact_ref(
            evidence_root,
            step_root / "stdout.log",
            "stdout-log",
        ),
        "stderr": artifact_ref(
            evidence_root,
            step_root / "stderr.log",
            "stderr-log",
        ),
        "returncode": 0,
        "duration_seconds": receipt["duration_seconds"],
    }


def cargo_build_command(
    *,
    policy: dict[str, Any],
    target_dir: Path,
    native_operator_set_lock: Path,
    native_build_cache: Path,
    build_summary_receipt: Path,
    source_policy: str,
) -> list[str]:
    product = policy["product_build"]
    require(
        source_policy in {"allow", "cache-only"},
        f"invalid CUDA source policy: {source_policy}",
    )
    return [
        "env",
        "NO_COLOR=1",
        f"CARGO_TARGET_DIR={target_dir}",
        f"CARGO_BUILD_JOBS={product['cargo_jobs']}",
        f"CUDA_COMPUTE_CAP={product['compute_capability']}",
        f"FERRUM_NVCC_THREADS={product['nvcc_threads']}",
        (
            "FERRUM_NATIVE_OPERATOR_SET_LOCK="
            f"{native_operator_set_lock}"
        ),
        (
            "FERRUM_CUDA_NATIVE_SOURCE_POLICY="
            f"{source_policy}"
        ),
        f"FERRUM_CUDA_NATIVE_BUILD_CACHE={native_build_cache}",
        f"FERRUM_CUDA_BUILD_SUMMARY_RECEIPT={build_summary_receipt}",
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
    ]


def parse_cargo_messages(path: Path) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    verbose_lines: list[str] = []
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
                verbose_lines.append(line)
                continue
            raise BuildIterationError(
                f"invalid Cargo JSON at {path}:{number}: {error}"
            ) from error
        require(isinstance(value, dict), f"Cargo row {number} is not an object")
        messages.append(value)
    artifacts = [
        row for row in messages if row.get("reason") == "compiler-artifact"
    ]
    require(artifacts, f"Cargo emitted no compiler artifacts: {path}")
    finished = [row for row in messages if row.get("reason") == "build-finished"]
    require(
        len(finished) == 1 and finished[0].get("success") is True,
        f"Cargo did not emit one successful build-finished row: {path}",
    )
    fresh = [row for row in artifacts if row.get("fresh") is True]
    nonfresh = [row for row in artifacts if row.get("fresh") is False]
    require(
        len(fresh) + len(nonfresh) == len(artifacts),
        "Cargo compiler artifacts must carry boolean fresh",
    )
    return {
        "message_count": len(messages),
        "verbose_line_count": len(verbose_lines),
        "compiler_artifact_count": len(artifacts),
        "fresh_artifact_count": len(fresh),
        "nonfresh_artifact_count": len(nonfresh),
        "nonfresh_packages": sorted(
            {
                str(row["package_id"])
                for row in nonfresh
                if isinstance(row.get("package_id"), str)
            }
        ),
        "build_finished_success": True,
    }


def parse_product_native_signal(
    build_stderr: Path,
    build_summary_receipt: Path,
) -> dict[str, Any]:
    log = build_stderr.read_text(encoding="utf-8", errors="strict")
    compiled_paths = [
        match.group(1)
        for match in re.finditer(
            r"\[[^]]+\]\s+compiling\s+(\S+)\s+->\s+(\S+)",
            log,
        )
    ]
    summaries: list[dict[str, Any]] = []
    if build_summary_receipt.is_file():
        receipt = read_json(build_summary_receipt, "CUDA build summary receipt")
        require(
            receipt.get("schema_version") == 1
            and receipt.get("artifact_type")
            == "ferrum_cuda_build_summary_receipt"
            and isinstance(receipt.get("rows"), list),
            "CUDA build summary receipt identity mismatch",
        )
        summaries = receipt["rows"]
    artifact_rows = {
        row.get("artifact"): row
        for row in summaries
        if isinstance(row, dict)
        and row.get("artifact") in PRODUCT_NATIVE_ARTIFACTS
    }
    require(
        all(
            row.get("status") == "artifact"
            and row.get("reason") == "native-operator-artifact-set"
            for row in artifact_rows.values()
        ),
        f"product build used a non-artifact native operator: {artifact_rows}",
    )
    require(
        not any(
            isinstance(row, dict) and row.get("status") == "rejected"
            for row in summaries
        ),
        "product CUDA build summary contains a rejection",
    )
    core_ptx_built = sorted(
        str(row.get("artifact")).removeprefix("core-ptx:")
        for row in summaries
        if isinstance(row, dict)
        and str(row.get("artifact", "")).startswith("core-ptx:")
        and row.get("status") == "built"
    )
    core_ptx_rows = {
        str(row.get("artifact")).removeprefix("core-ptx:"): row
        for row in summaries
        if isinstance(row, dict)
        and str(row.get("artifact", "")).startswith("core-ptx:")
    }
    return {
        "compiled_native_tu_paths": compiled_paths,
        "compiled_native_tu_count": len(compiled_paths),
        "core_ptx_built_paths": core_ptx_built,
        "core_ptx_rows": core_ptx_rows,
        "artifact_build_units": sorted(artifact_rows),
        "build_summary_present": build_summary_receipt.is_file(),
        "build_summaries": summaries,
    }


def validate_product_cache_bootstrap_signal(
    policy: dict[str, Any],
    native: dict[str, Any],
) -> None:
    expected_core = set(policy["product_build"]["core_ptx_inputs"])
    require(
        native["compiled_native_tu_count"] == 0,
        "core PTX cache bootstrap compiled external native operator source",
    )
    require(
        set(native["artifact_build_units"]) == PRODUCT_NATIVE_ARTIFACTS,
        "core PTX cache bootstrap did not resolve the complete native artifact set",
    )
    require(
        set(native["core_ptx_rows"]) == expected_core
        and all(
            row.get("status") in {"built", "cache_hit"}
            for row in native["core_ptx_rows"].values()
        ),
        "core PTX cache bootstrap did not materialize every configured PTX",
    )


def fsync_replace(path: Path, payload: bytes) -> tuple[int, int]:
    before = path.stat()
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    after = path.stat()
    require(after.st_mtime_ns >= before.st_mtime_ns, "mutation mtime moved backwards")
    return before.st_mtime_ns, after.st_mtime_ns


def prepare_mutation(
    path: Path,
    input_label: str,
    sentinel_suffix: str,
) -> tuple[dict[str, Any], bytes, int, int]:
    require(path.is_file() and not path.is_symlink(), f"mutation input missing: {path}")
    original = path.read_bytes()
    mutated = original + sentinel_suffix.encode("ascii")
    require(mutated != original, "sentinel mutation did not change content")
    before_mtime, during_mtime = fsync_replace(path, mutated)
    setup = {
        "kind": "content-mutation",
        "input_path": input_label,
        "sentinel_suffix_sha256": hashlib.sha256(
            sentinel_suffix.encode("ascii")
        ).hexdigest(),
        "before_sha256": hashlib.sha256(original).hexdigest(),
        "during_sha256": hashlib.sha256(mutated).hexdigest(),
        "before_mtime_ns": before_mtime,
        "during_mtime_ns": during_mtime,
        "edit_fsync_completed_before_timing": True,
    }
    return setup, original, before_mtime, path.stat().st_atime_ns


def restore_mutation(
    path: Path,
    setup: dict[str, Any],
    original: bytes,
    original_mtime_ns: int,
    original_atime_ns: int,
) -> None:
    with path.open("wb") as handle:
        handle.write(original)
        handle.flush()
        os.fsync(handle.fileno())
    os.utime(path, ns=(original_atime_ns, original_mtime_ns))
    setup["restored_sha256"] = sha256(path)
    setup["restored_mtime_ns"] = path.stat().st_mtime_ns
    require(
        setup["restored_sha256"] == setup["before_sha256"]
        and setup["restored_mtime_ns"] == setup["before_mtime_ns"],
        f"mutation input was not restored: {path}",
    )


def reset_worktree(
    source_root: Path,
    worktree: Path,
    git_sha: str,
) -> None:
    if worktree.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=source_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=False,
        )
        if worktree.exists():
            shutil.rmtree(worktree)
    run_text(source_root, ["git", "worktree", "prune"], timeout=120)
    run_text(
        source_root,
        ["git", "worktree", "add", "--detach", str(worktree), git_sha],
        timeout=180,
    )
    require(
        run_text(worktree, ["git", "status", "--short"]) == "",
        "fresh timing worktree is dirty",
    )


def remove_worktree(source_root: Path, worktree: Path) -> None:
    if not worktree.exists():
        return
    status = run_text(worktree, ["git", "status", "--short"]).splitlines()
    require(not status, f"timing worktree is dirty at removal: {status}")
    run_text(
        source_root,
        ["git", "worktree", "remove", "--force", str(worktree)],
        timeout=180,
    )
    run_text(source_root, ["git", "worktree", "prune"], timeout=120)


def store_blob(evidence_root: Path, path: Path, kind: str) -> dict[str, Any]:
    digest = sha256(path)
    blob = evidence_root / "blobs" / digest
    if not blob.exists():
        blob.parent.mkdir(parents=True, exist_ok=True)
        temporary = blob.with_name(f".{blob.name}.{os.getpid()}.tmp")
        shutil.copy2(path, temporary)
        require(sha256(temporary) == digest, "copied blob SHA256 mismatch")
        temporary.replace(blob)
    require(sha256(blob) == digest, "existing content-addressed blob is corrupt")
    return artifact_ref(evidence_root, blob, kind)


def copy_cargo_timing(
    evidence_root: Path,
    target_dir: Path,
    sample_root: Path,
) -> dict[str, Any]:
    source = target_dir / "cargo-timings/cargo-timing.html"
    require(source.is_file(), f"Cargo timing report is missing: {source}")
    target = sample_root / "cargo-timing.html"
    shutil.copy2(source, target)
    return artifact_ref(evidence_root, target, "cargo-timing")


def validate_product_sample_signal(
    scenario: dict[str, Any],
    cargo: dict[str, Any],
    native: dict[str, Any],
) -> None:
    name = scenario["name"]
    require(
        native["compiled_native_tu_count"] == 0,
        f"{name} compiled vendored/native operator source in the product graph",
    )
    if native["build_summary_present"]:
        require(
            set(native["artifact_build_units"]) == PRODUCT_NATIVE_ARTIFACTS,
            f"{name} did not resolve the complete native artifact set",
        )
    if name in {"core-ptx", "clean-release"}:
        require(
            native["build_summary_present"]
            and set(native["artifact_build_units"])
            == PRODUCT_NATIVE_ARTIFACTS,
            f"{name} is missing complete cache-only native artifact evidence",
        )
    if name == "noop":
        require(
            cargo["nonfresh_artifact_count"] == 0,
            "noop sample unexpectedly rebuilt Cargo artifacts",
        )
    elif name in {"rust-model-leaf", "rust-runtime-leaf", "core-ptx"}:
        expected_package = scenario["expected_package"]
        require(
            any(
                expected_package in package
                for package in cargo["nonfresh_packages"]
            ),
            f"{name} did not invalidate {expected_package}",
        )
    if name == "core-ptx":
        require(
            native["core_ptx_built_paths"] == ["kernels/add_bias.cu"],
            "core-ptx sample did not rebuild exactly add_bias.cu",
        )


def build_product_core_ptx_cache(
    *,
    source_root: Path,
    evidence_root: Path,
    source: dict[str, Any],
    policy: dict[str, Any],
    target_root: Path,
    native_operator_set_lock: Path,
    native_build_cache: Path,
    lane_deadline: float,
    resume: bool,
) -> dict[str, Any]:
    setup_root = evidence_root / "setup/product-core-ptx-cache-bootstrap"
    record_path = setup_root / "bootstrap.json"
    if resume and record_path.is_file():
        record = read_json(record_path, "product core PTX cache bootstrap")
        require(
            record.get("schema_version") == SCHEMA_VERSION
            and record.get("status") == "pass"
            and record.get("source_git_sha") == source["git_sha"]
            and record.get("native_build_cache") == str(native_build_cache)
            and native_build_cache.is_dir()
            and any(native_build_cache.iterdir()),
            "resumed product core PTX cache bootstrap is stale",
        )
        for label, raw in (
            ("bootstrap bounded receipt", record["build"]["bounded_receipt"]),
            ("bootstrap stdout", record["build"]["stdout"]),
            ("bootstrap stderr", record["build"]["stderr"]),
            ("bootstrap CUDA summary", record["build"]["cuda_build_summary"]),
            ("bootstrap smoke receipt", record["smoke"]["bounded_receipt"]),
            ("bootstrap binary", record["output"]["artifact"]),
        ):
            verify_ref(evidence_root, raw, label)
        return record
    if setup_root.exists():
        require(resume, f"bootstrap output already exists: {setup_root}")
        shutil.rmtree(setup_root)

    target_dir = target_root / "product-cache-bootstrap"
    summary_path = setup_root / "cuda-build-summary.receipt.json"
    command = cargo_build_command(
        policy=policy,
        target_dir=target_dir,
        native_operator_set_lock=native_operator_set_lock,
        native_build_cache=native_build_cache,
        build_summary_receipt=summary_path,
        source_policy=policy["product_build"]["bootstrap_source_policy"],
    )
    build = run_bounded_step(
        evidence_root=evidence_root,
        step_root=setup_root / "build",
        cwd=source_root,
        command=command,
        expected_seconds=300,
        deadline_seconds=1200,
        progress_signal=(
            "Cargo log growth, explicit core PTX cache publish/hit rows, "
            "and product binary creation"
        ),
        lane_deadline=lane_deadline,
    )
    binary = target_dir / "release/ferrum"
    require(
        binary.is_file() and os.access(binary, os.X_OK),
        "core PTX cache bootstrap binary is missing",
    )
    smoke = run_bounded_step(
        evidence_root=evidence_root,
        step_root=setup_root / "smoke",
        cwd=source_root,
        command=[str(binary), "--version"],
        expected_seconds=1,
        deadline_seconds=30,
        progress_signal="bootstrap ferrum version output",
        lane_deadline=lane_deadline,
        max_processes=8,
        max_group_threads=32,
        max_per_process_threads=16,
    )
    cargo_stdout = verify_ref(
        evidence_root,
        build["stdout"],
        "core PTX cache bootstrap Cargo stdout",
    )
    cargo_stderr = verify_ref(
        evidence_root,
        build["stderr"],
        "core PTX cache bootstrap Cargo stderr",
    )
    cargo_summary = parse_cargo_messages(cargo_stdout)
    native_signal = parse_product_native_signal(cargo_stderr, summary_path)
    validate_product_cache_bootstrap_signal(policy, native_signal)
    summary_ref = artifact_ref(
        evidence_root,
        summary_path,
        "cuda-build-summary",
    )
    binary_ref = store_blob(evidence_root, binary, "binary")
    record = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_g07a_product_core_ptx_cache_bootstrap",
        "status": "pass",
        "source_git_sha": source["git_sha"],
        "source_tree_sha": source["git_tree_sha"],
        "source_policy": policy["product_build"]["bootstrap_source_policy"],
        "native_build_cache": str(native_build_cache),
        "build": {
            **build,
            "cargo_summary": cargo_summary,
            "native_signal": native_signal,
            "cuda_build_summary": summary_ref,
        },
        "smoke": smoke,
        "output": {
            "kind": "binary",
            "artifact": binary_ref,
            "sha256": binary_ref["sha256"],
        },
    }
    write_json(record_path, record)
    return record


def product_sample(
    *,
    source_root: Path,
    evidence_root: Path,
    source: dict[str, Any],
    policy: dict[str, Any],
    scenario: dict[str, Any],
    sample_index: int,
    worktree: Path,
    target_dir: Path,
    native_operator_set_lock: Path,
    native_build_cache: Path,
    lane_deadline: float,
) -> dict[str, Any]:
    name = scenario["name"]
    source_policy = (
        policy["product_build"]["core_ptx_source_policy"]
        if name == "core-ptx"
        else policy["product_build"]["default_source_policy"]
    )
    sample_id = f"{name}-{sample_index}"
    sample_root = evidence_root / "build-timings" / name / f"sample-{sample_index}"
    require(not sample_root.exists(), f"sample output already exists: {sample_root}")
    sample_root.mkdir(parents=True)
    reset_worktree(source_root, worktree, source["git_sha"])
    clean_before = run_text(worktree, ["git", "status", "--short"]) == ""
    summary_path = target_dir / "g07a-build-summary.receipt.json"

    if scenario["kind"] == "cargo_clean_release":
        if target_dir.exists():
            shutil.rmtree(target_dir)
        setup: dict[str, Any] = {
            "kind": "cargo-clean",
            "target_absent_before_timing": not target_dir.exists(),
        }
        prewarm = None
    else:
        summary_path.unlink(missing_ok=True)
        prewarm = run_bounded_step(
            evidence_root=evidence_root,
            step_root=sample_root / "prewarm",
            cwd=worktree,
            command=cargo_build_command(
                policy=policy,
                target_dir=target_dir,
                native_operator_set_lock=native_operator_set_lock,
                native_build_cache=native_build_cache,
                build_summary_receipt=summary_path,
                source_policy=source_policy,
            ),
            expected_seconds=60,
            deadline_seconds=1200,
            progress_signal="Cargo log growth, rustc/linker activity, and binary creation",
            lane_deadline=lane_deadline,
        )
        if scenario["input"] is None:
            setup = {"kind": "none"}
        else:
            mutation_path = worktree / scenario["input"]
            setup, original, original_mtime, original_atime = prepare_mutation(
                mutation_path,
                scenario["input"],
                policy["sentinel_suffix"],
            )

    summary_path.unlink(missing_ok=True)
    timed_started_at = now_iso()
    timed_started = time.monotonic()
    try:
        build = run_bounded_step(
            evidence_root=evidence_root,
            step_root=sample_root / "build",
            cwd=worktree,
            command=cargo_build_command(
                policy=policy,
                target_dir=target_dir,
                native_operator_set_lock=native_operator_set_lock,
                native_build_cache=native_build_cache,
                build_summary_receipt=summary_path,
                source_policy=source_policy,
            ),
            expected_seconds=min(60, scenario["p95_target_seconds"]),
            deadline_seconds=scenario["deadline_seconds"],
            progress_signal="Cargo log growth and rustc/PTX/linker activity",
            lane_deadline=lane_deadline,
        )
        binary = target_dir / "release/ferrum"
        require(binary.is_file() and os.access(binary, os.X_OK), "ferrum binary is missing")
        smoke = run_bounded_step(
            evidence_root=evidence_root,
            step_root=sample_root / "smoke",
            cwd=worktree,
            command=[str(binary), "--version"],
            expected_seconds=1,
            deadline_seconds=30,
            progress_signal="ferrum version output",
            lane_deadline=lane_deadline,
            max_processes=8,
            max_group_threads=32,
            max_per_process_threads=16,
        )
        duration = time.monotonic() - timed_started
        timed_finished_at = now_iso()
        cargo_stdout = verify_ref(
            evidence_root,
            build["stdout"],
            f"{sample_id} Cargo stdout",
        )
        cargo_stderr = verify_ref(
            evidence_root,
            build["stderr"],
            f"{sample_id} Cargo stderr",
        )
        cargo_summary = parse_cargo_messages(cargo_stdout)
        native_signal = parse_product_native_signal(
            cargo_stderr,
            summary_path,
        )
        validate_product_sample_signal(
            scenario,
            cargo_summary,
            native_signal,
        )
        binary_ref = store_blob(evidence_root, binary, "binary")
        timing_ref = copy_cargo_timing(
            evidence_root,
            target_dir,
            sample_root,
        )
        build_summary_ref = (
            artifact_ref(
                evidence_root,
                summary_path,
                "cuda-build-summary",
            )
            if summary_path.is_file()
            else None
        )
    finally:
        if scenario.get("input") is not None and "original" in locals():
            restore_mutation(
                worktree / scenario["input"],
                setup,
                original,
                original_mtime,
                original_atime,
            )
    clean_after = run_text(worktree, ["git", "status", "--short"]) == ""
    require(clean_before and clean_after, f"{sample_id} worktree cleanliness failed")
    remove_worktree(source_root, worktree)
    record = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "status": "pass",
        "source_git_sha": source["git_sha"],
        "source_tree_sha": source["git_tree_sha"],
        "timed_started_at": timed_started_at,
        "timed_finished_at": timed_finished_at,
        "timed_monotonic_started_seconds": timed_started,
        "timed_monotonic_finished_seconds": timed_started + duration,
        "duration_seconds": duration,
        "worktree": {
            "path": str(worktree),
            "clean_before": clean_before,
            "clean_after": clean_after,
            "stable_recreated_path": True,
        },
        "cache": {
            "cargo_target": str(target_dir),
            "scope": (
                "fresh-per-sample"
                if scenario["kind"] == "cargo_clean_release"
                else "declared-shared-incremental"
            ),
            "native_operator_set_lock_sha256": sha256(
                native_operator_set_lock
            ),
            "native_build_cache": str(native_build_cache),
            "source_policy": source_policy,
        },
        "setup": setup,
        "prewarm": prewarm,
        "build": {
            **build,
            "cargo_summary": cargo_summary,
            "native_signal": native_signal,
            "cuda_build_summary": build_summary_ref,
            "cargo_timing": timing_ref,
        },
        "smoke": smoke,
        "output": {
            "kind": "binary",
            "artifact": binary_ref,
            "sha256": binary_ref["sha256"],
        },
    }
    write_json(sample_root / "sample.json", record)
    return record


def native_source_build_command(
    *,
    builder: Path,
    plan: Path,
    native_source_root: Path,
    output: Path,
    object_cache: Path,
    source: dict[str, Any],
    tools: dict[str, Path],
    cuda_toolkit_root: Path,
    policy: dict[str, Any],
) -> list[str]:
    native = policy["native_build"]
    return [
        str(builder),
        "source-build",
        "--plan",
        str(plan),
        "--source-root",
        str(native_source_root),
        "--compute-capability",
        native["compute_capability"],
        "--builder-sha",
        source["git_sha"],
        "--nvcc",
        str(tools["nvcc"]),
        "--cuda-toolkit-root",
        str(cuda_toolkit_root),
        "--ccbin",
        str(tools["ccbin"]),
        "--ar",
        str(tools["ar"]),
        "--nvcc-threads",
        str(native["nvcc_threads"]),
        "--object-cache",
        str(object_cache),
        "--out",
        str(output),
    ]


def validate_native_receipt(
    path: Path,
    expected_input: str,
) -> dict[str, Any]:
    receipt = read_json(path, "native source-build receipt")
    require(
        receipt.get("schema_version") == SOURCE_BUILD_RECEIPT_SCHEMA_VERSION
        and receipt.get("status") == "pass"
        and receipt.get("plan_only") is False,
        "native source-build receipt identity/status mismatch",
    )
    compiled = receipt.get("compiled_translation_units")
    cache_hits = receipt.get("cache_hit_translation_units")
    commands = receipt.get("commands")
    require(
        compiled == [expected_input],
        f"native-TU sample must compile exactly {expected_input}: {compiled}",
    )
    require(
        isinstance(cache_hits, list)
        and expected_input not in cache_hits
        and isinstance(commands, list),
        "native-TU cache/command evidence is invalid",
    )
    archive = receipt.get("archive_file")
    require(
        isinstance(archive, str)
        and archive.endswith(".a")
        and "/" not in archive,
        "native source-build archive filename is invalid",
    )
    return receipt


def copy_verified_native_source(
    canonical_root: Path,
    timing_root: Path,
    bundle_manifest: dict[str, Any],
) -> None:
    source_bundle.verify_materialized_tree(bundle_manifest, canonical_root)
    if timing_root.exists():
        shutil.rmtree(timing_root)
    shutil.copytree(canonical_root, timing_root)
    source_bundle.verify_materialized_tree(bundle_manifest, timing_root)


def native_sample(
    *,
    source_root: Path,
    canonical_native_source_root: Path,
    evidence_root: Path,
    source: dict[str, Any],
    policy: dict[str, Any],
    scenario: dict[str, Any],
    sample_index: int,
    timing_native_source_root: Path,
    builder: Path,
    object_cache: Path,
    tools: dict[str, Path],
    cuda_toolkit_root: Path,
    bundle_manifest: dict[str, Any],
    lane_deadline: float,
) -> dict[str, Any]:
    name = scenario["name"]
    sample_id = f"{name}-{sample_index}"
    sample_root = evidence_root / "build-timings" / name / f"sample-{sample_index}"
    require(not sample_root.exists(), f"sample output already exists: {sample_root}")
    sample_root.mkdir(parents=True)
    copy_verified_native_source(
        canonical_native_source_root,
        timing_native_source_root,
        bundle_manifest,
    )
    canonical_plan = source_root / policy["native_build"]["plan"]
    prewarm_output = sample_root / "prewarm-output"
    base_object_cache = object_cache / "native-base"
    sample_object_cache = (
        object_cache / "native-samples" / sample_id
    )
    prewarm = run_bounded_step(
        evidence_root=evidence_root,
        step_root=sample_root / "prewarm",
        cwd=source_root,
        command=native_source_build_command(
            builder=builder,
            plan=canonical_plan,
            native_source_root=timing_native_source_root,
            output=prewarm_output,
            object_cache=base_object_cache,
            source=source,
            tools=tools,
            cuda_toolkit_root=cuda_toolkit_root,
            policy=policy,
        ),
        expected_seconds=60,
        deadline_seconds=1200,
        progress_signal="source-build logs, nvcc activity, or object-cache growth",
        lane_deadline=lane_deadline,
        max_processes=48,
        max_group_threads=192,
        max_per_process_threads=64,
    )
    if sample_object_cache.exists():
        shutil.rmtree(sample_object_cache)
    sample_object_cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_object_cache, sample_object_cache)
    input_path = timing_native_source_root / scenario["input"]
    setup, original, original_mtime, original_atime = prepare_mutation(
        input_path,
        scenario["input"],
        policy["sentinel_suffix"],
    )
    mutated_plan = sample_root / "mutated.plan.json"
    try:
        plan_step = run_bounded_step(
            evidence_root=evidence_root,
            step_root=sample_root / "lock-mutated-source",
            cwd=source_root,
            command=[
                str(builder),
                "lock-source",
                "--definition",
                str(source_root / policy["native_build"]["definition"]),
                "--source-root",
                str(timing_native_source_root),
                "--out",
                str(mutated_plan),
            ],
            expected_seconds=2,
            deadline_seconds=60,
            progress_signal="mutated source plan creation",
            lane_deadline=lane_deadline,
        )
        timed_output = sample_root / "timed-output"
        timed_started_at = now_iso()
        timed_started = time.monotonic()
        build = run_bounded_step(
            evidence_root=evidence_root,
            step_root=sample_root / "build",
            cwd=source_root,
            command=native_source_build_command(
                builder=builder,
                plan=mutated_plan,
                native_source_root=timing_native_source_root,
                output=timed_output,
                object_cache=sample_object_cache,
                source=source,
                tools=tools,
                cuda_toolkit_root=cuda_toolkit_root,
                policy=policy,
            ),
            expected_seconds=60,
            deadline_seconds=scenario["deadline_seconds"],
            progress_signal="exactly one nvcc TU plus archive creation",
            lane_deadline=lane_deadline,
            max_processes=48,
            max_group_threads=192,
            max_per_process_threads=64,
        )
        receipt_path = timed_output / "source-build.receipt.json"
        receipt = validate_native_receipt(receipt_path, scenario["input"])
        archive = timed_output / receipt["archive_file"]
        require(archive.is_file(), f"native source archive is missing: {archive}")
        smoke = run_bounded_step(
            evidence_root=evidence_root,
            step_root=sample_root / "smoke",
            cwd=source_root,
            command=[str(tools["ar"]), "t", str(archive)],
            expected_seconds=1,
            deadline_seconds=30,
            progress_signal="archive member listing",
            lane_deadline=lane_deadline,
            max_processes=8,
            max_group_threads=32,
            max_per_process_threads=16,
        )
        duration = time.monotonic() - timed_started
        timed_finished_at = now_iso()
        receipt_ref = artifact_ref(
            evidence_root,
            receipt_path,
            "native-source-build-receipt",
        )
        archive_ref = store_blob(evidence_root, archive, "native-archive")
    finally:
        restore_mutation(
            input_path,
            setup,
            original,
            original_mtime,
            original_atime,
        )
        if sample_object_cache.exists():
            shutil.rmtree(sample_object_cache)
    source_bundle.verify_materialized_tree(
        bundle_manifest,
        timing_native_source_root,
    )
    shutil.rmtree(timing_native_source_root)
    record = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "status": "pass",
        "source_git_sha": source["git_sha"],
        "source_tree_sha": source["git_tree_sha"],
        "timed_started_at": timed_started_at,
        "timed_finished_at": timed_finished_at,
        "timed_monotonic_started_seconds": timed_started,
        "timed_monotonic_finished_seconds": timed_started + duration,
        "duration_seconds": duration,
        "worktree": {
            "path": str(timing_native_source_root),
            "clean_before": True,
            "clean_after": True,
            "stable_recreated_path": True,
        },
        "cache": {
            "base_object_cache": str(base_object_cache),
            "sample_object_cache": str(sample_object_cache),
            "scope": "verified-base-cloned-per-sample",
        },
        "setup": setup,
        "prewarm": prewarm,
        "plan": plan_step,
        "build": {
            **build,
            "source_build_receipt": receipt_ref,
            "compiled_translation_units": receipt[
                "compiled_translation_units"
            ],
            "cache_hit_translation_units": receipt[
                "cache_hit_translation_units"
            ],
        },
        "smoke": smoke,
        "output": {
            "kind": "native-archive",
            "artifact": archive_ref,
            "sha256": archive_ref["sha256"],
        },
    }
    write_json(sample_root / "sample.json", record)
    return record


def sample_refs_valid(evidence_root: Path, value: Any) -> None:
    if isinstance(value, dict):
        if set(value) == {"path", "sha256", "size_bytes", "kind"}:
            verify_ref(evidence_root, value, "resumed sample artifact")
            return
        for child in value.values():
            sample_refs_valid(evidence_root, child)
    elif isinstance(value, list):
        for child in value:
            sample_refs_valid(evidence_root, child)


def resumed_sample(
    evidence_root: Path,
    path: Path,
    source: dict[str, Any],
    expected_id: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        record = read_json(path, "resumed sample")
        require(
            record.get("status") == "pass"
            and record.get("sample_id") == expected_id
            and record.get("source_git_sha") == source["git_sha"]
            and record.get("source_tree_sha") == source["git_tree_sha"],
            "resumed sample identity mismatch",
        )
        sample_refs_valid(evidence_root, record)
        return record
    except BuildIterationError:
        shutil.rmtree(path.parent)
        return None


def scenario_summary(
    scenario: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    durations = sorted(float(sample["duration_seconds"]) for sample in samples)
    require(durations and all(value > 0 for value in durations), "sample durations are invalid")
    p50_index = math.ceil(0.50 * len(durations)) - 1
    p95_index = math.ceil(0.95 * len(durations)) - 1
    return {
        "name": scenario["name"],
        "kind": scenario["kind"],
        "input": scenario["input"],
        "expected_package": scenario["expected_package"],
        "p95_target_seconds": scenario["p95_target_seconds"],
        "sample_count": len(samples),
        "durations_seconds": durations,
        "p50_seconds": durations[p50_index],
        "p95_seconds": durations[p95_index],
        "target_met": durations[p95_index]
        <= scenario["p95_target_seconds"],
        "samples": samples,
    }


def artifact_index(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"evidence tree contains a symlink: {path}")
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


def copy_input(
    evidence_root: Path,
    source: Path,
    name: str,
    *,
    resume: bool,
) -> dict[str, Any]:
    require(source.is_file() and not source.is_symlink(), f"input is missing: {source}")
    destination = evidence_root / "inputs" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(
            resume
            and destination.is_file()
            and not destination.is_symlink()
            and sha256(destination) == sha256(source),
            f"resumed G07A input changed: {name}",
        )
    else:
        shutil.copy2(source, destination)
    return artifact_ref(evidence_root, destination, "input-manifest")


def build_builder(
    *,
    source_root: Path,
    evidence_root: Path,
    target_root: Path,
    lane_deadline: float,
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    target = target_root / "builder"
    step_root = evidence_root / "setup/builder-build"
    command = [
        "env",
        f"CARGO_TARGET_DIR={target}",
        "CARGO_BUILD_JOBS=4",
        "cargo",
        "build",
        "--release",
        "--locked",
        "--jobs",
        "4",
        "-p",
        "ferrum-native-ops-builder",
        "--bin",
        "ferrum-native-ops-builder",
        "--message-format=json-render-diagnostics",
    ]
    builder = target / "release/ferrum-native-ops-builder"
    if resume and step_root.is_dir() and builder.is_file():
        try:
            plan = read_json(step_root / "plan.json", "builder build plan")
            receipt = read_json(
                step_root / "bounded.receipt.json",
                "builder bounded receipt",
            )
            require(plan.get("command") == command, "builder resume command drift")
            require(
                receipt.get("schema")
                == "ferrum.bounded-command-receipt.v1"
                and receipt.get("status") == "pass"
                and receipt.get("rc") == 0
                and receipt.get("cleanup", {}).get("process_group_gone")
                is True,
                "builder resume receipt is not a terminal PASS",
            )
            return builder, {
                "command": command,
                "bounded_receipt": artifact_ref(
                    evidence_root,
                    step_root / "bounded.receipt.json",
                    "bounded-receipt",
                ),
                "stdout": artifact_ref(
                    evidence_root,
                    step_root / "stdout.log",
                    "stdout-log",
                ),
                "stderr": artifact_ref(
                    evidence_root,
                    step_root / "stderr.log",
                    "stderr-log",
                ),
                "returncode": 0,
                "duration_seconds": receipt["duration_seconds"],
                "resumed": True,
            }
        except BuildIterationError:
            shutil.rmtree(step_root)
    elif resume and step_root.exists():
        shutil.rmtree(step_root)
    step = run_bounded_step(
        evidence_root=evidence_root,
        step_root=step_root,
        cwd=source_root,
        command=command,
        expected_seconds=60,
        deadline_seconds=900,
        progress_signal="Cargo log growth and builder binary creation",
        lane_deadline=lane_deadline,
    )
    require(builder.is_file() and os.access(builder, os.X_OK), "native builder is missing")
    return builder, step


def collect(args: argparse.Namespace) -> Path:
    source_root = args.source_root.expanduser().resolve()
    native_source_root = args.native_source_root.expanduser().resolve()
    native_operator_set_lock = args.native_operator_set_lock.expanduser().resolve()
    g00f = args.g00f.expanduser().resolve()
    s1 = args.s1_manifest.expanduser().resolve()
    evidence_root = args.out.expanduser().resolve()
    worktree_root = args.worktree_root.expanduser().resolve()
    target_root = args.target_root.expanduser().resolve()
    object_cache = args.object_cache.expanduser().resolve()
    product_native_build_cache = object_cache / "product-core-ptx"
    cuda_toolkit_root = args.cuda_toolkit_root.expanduser().resolve()
    policy = load_policy()
    canonical = args.mode == "canonical"
    repeats = policy["repeats"] if canonical else args.diagnostic_repeats
    require(
        canonical or repeats == 1,
        "diagnostic mode requires --diagnostic-repeats 1",
    )
    require(source_root.is_dir(), f"source root is missing: {source_root}")
    require(
        native_source_root.is_dir()
        and not native_source_root.is_relative_to(source_root),
        "native source root must be an external directory",
    )
    require(
        native_operator_set_lock.is_file()
        and not native_operator_set_lock.is_symlink(),
        "native operator set lock must be a regular file",
    )
    require(g00f.is_file() and s1.is_file(), "G00F and S1 manifests are required")
    require(
        not evidence_root.is_relative_to(source_root)
        and not worktree_root.is_relative_to(source_root)
        and not target_root.is_relative_to(source_root)
        and not object_cache.is_relative_to(source_root),
        "G07A outputs and caches must be outside the source tree",
    )
    ensure_distinct_roots(
        source_root=source_root,
        native_source_root=native_source_root,
        evidence_root=evidence_root,
        worktree_root=worktree_root,
        target_root=target_root,
        object_cache=object_cache,
    )
    require(
        args.hard_timeout_seconds >= args.expected_runtime_seconds,
        "hard timeout must be at least expected runtime",
    )
    if args.resume:
        require(evidence_root.is_dir(), "--resume requires an existing evidence root")
    else:
        require(not evidence_root.exists(), f"evidence root already exists: {evidence_root}")
        evidence_root.mkdir(parents=True)
    source = source_identity(source_root)
    claim_managed_root(
        worktree_root,
        role="worktree-root",
        source_git_sha=source["git_sha"],
        resume=args.resume,
    )
    claim_managed_root(
        target_root,
        role="target-root",
        source_git_sha=source["git_sha"],
        resume=args.resume,
    )
    claim_managed_root(
        object_cache,
        role="object-cache-root",
        source_git_sha=source["git_sha"],
        resume=args.resume,
    )
    compiler_cache = validate_no_hidden_compiler_cache()
    tools = {
        "nvcc": resolve_tool(args.nvcc, "nvcc"),
        "ccbin": resolve_tool(args.ccbin, "CUDA host compiler"),
        "ar": resolve_tool(args.ar, "archiver"),
    }
    require(cuda_toolkit_root.is_dir(), "CUDA toolkit root is missing")
    hardware_before = hardware_identity(source_root, tools)
    bundle_manifest_path = source_root / policy["source_bundle_manifest"]
    bundle_manifest = source_bundle.validate_manifest(
        source_bundle.read_json(bundle_manifest_path, "native source bundle")
    )
    source_bundle.verify_materialized_tree(bundle_manifest, native_source_root)
    lane_deadline = time.monotonic() + args.hard_timeout_seconds
    input_refs = {
        "policy": copy_input(
            evidence_root,
            POLICY_PATH,
            "g07a-policy.json",
            resume=args.resume,
        ),
        "g00f": copy_input(
            evidence_root,
            g00f,
            "g00f.manifest.json",
            resume=args.resume,
        ),
        "s1": copy_input(
            evidence_root,
            s1,
            "s1.manifest.json",
            resume=args.resume,
        ),
        "source_bundle": copy_input(
            evidence_root,
            bundle_manifest_path,
            "native-source-bundle.json",
            resume=args.resume,
        ),
        "native_operator_set_lock": copy_input(
            evidence_root,
            native_operator_set_lock,
            "native-operator-set.lock.json",
            resume=args.resume,
        ),
    }
    crate_graph_path = evidence_root / "crate-graph.json"
    metadata = subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        cwd=source_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
        check=False,
    )
    require(metadata.returncode == 0, f"cargo metadata failed: {metadata.stderr}")
    if crate_graph_path.exists():
        require(
            args.resume
            and crate_graph_path.read_text(encoding="utf-8")
            == metadata.stdout,
            "resumed G07A crate graph changed",
        )
    else:
        crate_graph_path.write_text(metadata.stdout, encoding="utf-8")
    lane_plan = {
        "schema_version": SCHEMA_VERSION,
        "lane": "runtime-vnext-g07a-build-iteration",
        "mode": args.mode,
        "source": source,
        "hardware_fingerprint": hardware_before["fingerprint"],
        "expected_runtime_seconds": args.expected_runtime_seconds,
        "hard_deadline_seconds": args.hard_timeout_seconds,
        "hard_stop": "first bounded failure, source drift, hidden compiler cache, native source fallback, or threshold miss",
        "correctness_gate": "workspace source gate and release/correctness semantic-plan equivalence are external G07A inputs",
        "performance_command": "the six policy scenarios, each with nearest-rank p95 over five samples in canonical mode",
        "progress_signal": "per-step receipts/log bytes, Cargo or nvcc activity, and completed sample.json files",
        "repeats": repeats,
        "compiler_cache": compiler_cache,
        "paths": {
            "worktree_root": str(worktree_root),
            "target_root": str(target_root),
            "object_cache": str(object_cache),
            "product_native_build_cache": str(product_native_build_cache),
        },
        "inputs": input_refs,
    }
    plan_path = evidence_root / "lane-plan.json"
    if args.resume:
        existing_plan = read_json(plan_path, "resumed lane plan")
        require(
            existing_plan == lane_plan,
            "resumed G07A lane plan differs from the original plan",
        )
    else:
        write_json(plan_path, lane_plan)
    builder, builder_setup = build_builder(
        source_root=source_root,
        evidence_root=evidence_root,
        target_root=target_root,
        lane_deadline=lane_deadline,
        resume=args.resume,
    )
    product_cache_bootstrap = build_product_core_ptx_cache(
        source_root=source_root,
        evidence_root=evidence_root,
        source=source,
        policy=policy,
        target_root=target_root,
        native_operator_set_lock=native_operator_set_lock,
        native_build_cache=product_native_build_cache,
        lane_deadline=lane_deadline,
        resume=args.resume,
    )
    worktree = worktree_root / "product-timing-worktree"
    timing_native_source = worktree_root / "native-source-timing"
    scenario_rows = []
    for scenario in policy["scenarios"]:
        samples = []
        for index in range(1, repeats + 1):
            sample_json = (
                evidence_root
                / "build-timings"
                / scenario["name"]
                / f"sample-{index}"
                / "sample.json"
            )
            if args.resume and sample_json.parent.exists() and not sample_json.exists():
                shutil.rmtree(sample_json.parent)
            resumed = resumed_sample(
                evidence_root,
                sample_json,
                source,
                f"{scenario['name']}-{index}",
            )
            if resumed is not None:
                samples.append(resumed)
                continue
            if scenario["kind"] == "native_source_build":
                sample = native_sample(
                    source_root=source_root,
                    canonical_native_source_root=native_source_root,
                    evidence_root=evidence_root,
                    source=source,
                    policy=policy,
                    scenario=scenario,
                    sample_index=index,
                    timing_native_source_root=timing_native_source,
                    builder=builder,
                    object_cache=object_cache,
                    tools=tools,
                    cuda_toolkit_root=cuda_toolkit_root,
                    bundle_manifest=bundle_manifest,
                    lane_deadline=lane_deadline,
                )
            else:
                target = (
                    target_root / "clean-release"
                    if scenario["kind"] == "cargo_clean_release"
                    else target_root / "product-incremental"
                )
                sample = product_sample(
                    source_root=source_root,
                    evidence_root=evidence_root,
                    source=source,
                    policy=policy,
                    scenario=scenario,
                    sample_index=index,
                    worktree=worktree,
                    target_dir=target,
                    native_operator_set_lock=native_operator_set_lock,
                    native_build_cache=product_native_build_cache,
                    lane_deadline=lane_deadline,
                )
            samples.append(sample)
        row = scenario_summary(scenario, samples)
        require(
            not canonical or row["target_met"],
            f"canonical G07A threshold miss: {row['name']} "
            f"p95={row['p95_seconds']:.3f}s "
            f"target={row['p95_target_seconds']}s",
        )
        scenario_rows.append(row)
    hardware_after = hardware_identity(source_root, tools)
    require(
        hardware_after == hardware_before,
        "hardware/toolchain identity changed during G07A timing",
    )
    require(
        source_identity(source_root) == source,
        "source identity changed during G07A timing",
    )
    invalidation = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_g07a_invalidation_report",
        "native_product_source_compile_count": sum(
            sample["build"]["native_signal"]["compiled_native_tu_count"]
            for row in scenario_rows
            if row["kind"] != "native_source_build"
            for sample in row["samples"]
        ),
        "native_tu_compiled_counts": [
            len(sample["build"]["compiled_translation_units"])
            for row in scenario_rows
            if row["kind"] == "native_source_build"
            for sample in row["samples"]
        ],
        "scenario_targets": {
            row["name"]: {
                "p95_seconds": row["p95_seconds"],
                "target_seconds": row["p95_target_seconds"],
                "target_met": row["target_met"],
            }
            for row in scenario_rows
        },
    }
    invalidation_path = evidence_root / "invalidation-report.json"
    write_json(invalidation_path, invalidation)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "ready",
        "mode": args.mode,
        "created_at": now_iso(),
        "source": source,
        "hardware": hardware_before,
        "compiler_cache": compiler_cache,
        "policy": input_refs["policy"],
        "inputs": input_refs,
        "crate_graph": artifact_ref(
            evidence_root,
            crate_graph_path,
            "cargo-metadata",
        ),
        "builder_setup": builder_setup,
        "product_cache_bootstrap": product_cache_bootstrap,
        "repeats": repeats,
        "scenarios": scenario_rows,
        "invalidation_report": artifact_ref(
            evidence_root,
            invalidation_path,
            "invalidation-report",
        ),
        "does_not_prove": [
            "canonical G07A PASS",
            "canonical G07B PASS",
            "G07 aggregate PASS",
            "model correctness",
            "model performance",
            "release readiness",
        ],
    }
    manifest["artifact_index"] = artifact_index(evidence_root)
    manifest["artifact_index_sha256"] = canonical_json_sha256(
        manifest["artifact_index"]
    )
    write_json(evidence_root / "evidence.manifest.json", manifest)
    return evidence_root


def write_plan(args: argparse.Namespace) -> Path:
    source_root = args.source_root.expanduser().resolve()
    source = source_identity(source_root)
    policy = load_policy()
    out = args.out.expanduser().resolve()
    require(not out.exists(), f"plan output already exists: {out}")
    out.mkdir(parents=True)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_g07a_build_iteration_plan",
        "status": "plan",
        "source": source,
        "policy_path": str(POLICY_PATH),
        "policy_sha256": sha256(POLICY_PATH),
        "mode": args.mode,
        "repeats": (
            policy["repeats"]
            if args.mode == "canonical"
            else args.diagnostic_repeats
        ),
        "scenarios": policy["scenarios"],
        "required_inputs": {
            "g00f": str(args.g00f.expanduser().resolve()),
            "s1_manifest": str(args.s1_manifest.expanduser().resolve()),
            "native_source_root": str(
                args.native_source_root.expanduser().resolve()
            ),
            "native_operator_set_lock": str(
                args.native_operator_set_lock.expanduser().resolve()
            ),
        },
        "expected_runtime_seconds": args.expected_runtime_seconds,
        "hard_timeout_seconds": args.hard_timeout_seconds,
    }
    write_json(out / "plan.json", plan)
    print(f"FERRUM RUNTIME VNEXT G07A BUILD ITERATION PLAN READY: {out}")
    return out


def self_test() -> None:
    policy = load_policy()
    require(
        [row["p95_target_seconds"] for row in policy["scenarios"]]
        == [30, 90, 90, 120, 300, 900],
        "G07A threshold vector drifted",
    )
    with tempfile.TemporaryDirectory(prefix="ferrum-g07a-selftest-") as raw:
        root = Path(raw)
        messages = root / "cargo.jsonl"
        messages.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "reason": "compiler-artifact",
                            "package_id": "ferrum-models 0.7.7",
                            "fresh": False,
                        }
                    ),
                    "[ferrum-kernels 0.7.7] cargo:rerun-if-changed=build.rs",
                    json.dumps({"reason": "build-finished", "success": True}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        cargo = parse_cargo_messages(messages)
        require(
            cargo["nonfresh_artifact_count"] == 1
            and cargo["nonfresh_packages"] == ["ferrum-models 0.7.7"],
            "Cargo parser self-test failed",
        )
        source = root / "input.rs"
        source.write_text("pub fn fixture() {}\n", encoding="ascii")
        setup, original, mtime, atime = prepare_mutation(
            source,
            "input.rs",
            policy["sentinel_suffix"],
        )
        require(
            setup["before_sha256"] != setup["during_sha256"],
            "sentinel self-test did not change content",
        )
        restore_mutation(source, setup, original, mtime, atime)
        require(
            setup["restored_sha256"] == setup["before_sha256"],
            "sentinel self-test did not restore content",
        )
        receipt = root / "source-build.receipt.json"
        write_json(
            receipt,
            {
                "schema_version": SOURCE_BUILD_RECEIPT_SCHEMA_VERSION,
                "status": "pass",
                "plan_only": False,
                "compiled_translation_units": [
                    policy["native_build"]["input"]
                ],
                "cache_hit_translation_units": ["other.cu"],
                "commands": [{"kind": "translation_unit"}],
                "archive_file": "libfixture.a",
            },
        )
        validate_native_receipt(receipt, policy["native_build"]["input"])
        scenario = scenario_summary(
            policy["scenarios"][0],
            [
                {"duration_seconds": value}
                for value in (1.0, 2.0, 3.0, 4.0, 5.0)
            ],
        )
        require(
            scenario["p50_seconds"] == 3.0
            and scenario["p95_seconds"] == 5.0,
            "nearest-rank percentile self-test failed",
        )
        lock = root / "native.lock.json"
        lock.write_text("{}\n", encoding="utf-8")
        command = cargo_build_command(
            policy=policy,
            target_dir=root / "target",
            native_operator_set_lock=lock,
            native_build_cache=root / "native-cache",
            build_summary_receipt=root / "summary.json",
            source_policy=policy["product_build"]["default_source_policy"],
        )
        require(
            command[0] == "env"
            and f"FERRUM_NATIVE_OPERATOR_SET_LOCK={lock}" in command
            and "FERRUM_CUDA_NATIVE_SOURCE_POLICY=cache-only" in command
            and f"FERRUM_CUDA_NATIVE_BUILD_CACHE={root / 'native-cache'}"
            in command
            and command[-1] == "-vv",
            "canonical Cargo command self-test failed",
        )
    print(SELFTEST_PASS_LINE)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--self-test", action="store_true")
    subparsers = result.add_subparsers(dest="command")
    for name in ("plan", "run"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--source-root",
            type=Path,
            default=REPO_ROOT,
        )
        command.add_argument("--native-source-root", type=Path, required=True)
        command.add_argument(
            "--native-operator-set-lock",
            type=Path,
            required=True,
        )
        command.add_argument("--g00f", type=Path, required=True)
        command.add_argument("--s1-manifest", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument(
            "--mode",
            choices=("diagnostic", "canonical"),
            default="diagnostic",
        )
        command.add_argument("--diagnostic-repeats", type=int, default=1)
        command.add_argument(
            "--expected-runtime-seconds",
            type=int,
            default=5400,
        )
        command.add_argument(
            "--hard-timeout-seconds",
            type=int,
            default=10800,
        )
        command.add_argument(
            "--worktree-root",
            type=Path,
            required=True,
        )
        command.add_argument("--target-root", type=Path, required=True)
        command.add_argument("--object-cache", type=Path, required=True)
        command.add_argument(
            "--cuda-toolkit-root",
            type=Path,
            default=Path("/usr/local/cuda"),
        )
        command.add_argument("--nvcc", default="nvcc")
        command.add_argument("--ccbin", default="g++")
        command.add_argument("--ar", default="ar")
        command.add_argument("--resume", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.command is None:
        parser().error("choose --self-test, plan, or run")
    try:
        if args.command == "plan":
            write_plan(args)
        else:
            result = collect(args)
            prefix = PASS_LINE if args.mode == "canonical" else KEEP_LINE
            print(f"{prefix}: {result}")
    except (
        BuildIterationError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(
            f"FERRUM RUNTIME VNEXT G07A BUILD ITERATION REJECT: {error}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
