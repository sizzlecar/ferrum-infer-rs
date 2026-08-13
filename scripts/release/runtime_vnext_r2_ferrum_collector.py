#!/usr/bin/env python3
"""Collect one formal R2 Ferrum-only model/backend performance lane.

The collector intentionally has no external-engine, legacy-binary, or ABBA
execution path.  One Ferrum server process per epoch executes the remaining
backend cell suffix in order; a validated completed-cell prefix may be resumed
in an explicitly recorded later epoch.  Three independent ``ferrum run``
processes follow the server matrix.
Collection PASS means the immutable raw lane evidence is complete; the R2
aggregate validator remains responsible for baseline and threshold decisions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import selectors
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import runtime_vnext_performance_collector as collector_support
except ModuleNotFoundError:
    from scripts.release import runtime_vnext_performance_collector as collector_support


REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR_PATH = Path(__file__).resolve()
COLLECTOR_RELATIVE_PATH = COLLECTOR_PATH.relative_to(REPO_ROOT).as_posix()
SUPPORT_PATH = Path(collector_support.__file__).resolve()
RESOURCE_SAMPLER_PATH = Path(collector_support.RESOURCE_SAMPLER_PATH).resolve()
COLLECTION_EPOCH_SOURCE_PATHS = {
    "collector_sha256": COLLECTOR_PATH.relative_to(REPO_ROOT).as_posix(),
    "support_sha256": SUPPORT_PATH.relative_to(REPO_ROOT).as_posix(),
    "resource_sampler_sha256": RESOURCE_SAMPLER_PATH.relative_to(REPO_ROOT).as_posix(),
}
_REVIEWED_GIT_BLOB_SHA256S: dict[str, frozenset[str]] = {}
SCHEMA_VERSION = 1
CONTRACT = "ferrum.runtime-vnext.r2.ferrum-collector.v1"
CELL_CHECKPOINT_CONTRACT = "ferrum.runtime-vnext.r2.completed-cell-checkpoint.v1"
R1_CORRECTNESS_ARTIFACT_TYPE = (
    "runtime_vnext_r1_product_correctness_manifest"
)
PASS_PREFIX = "FERRUM RUNTIME VNEXT R2 FERRUM COLLECTOR PASS"
PLAN_PREFIX = "FERRUM RUNTIME VNEXT R2 FERRUM COLLECTOR PLAN"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT R2 FERRUM COLLECTOR SELFTEST PASS"
TEMPLATE_PREFIX = "FERRUM RUNTIME VNEXT R2 FERRUM COLLECTOR CONFIG TEMPLATE"
MODEL_KEYS = {
    "m1-qwen35-4b",
    "m2-qwen35-35b-a3b",
    "m3-qwen3-30b-a3b",
}
TYPED_ACTIVE_CAP_FLOORS = {
    ("m1-qwen35-4b", "cuda"): 32,
    ("m2-qwen35-35b-a3b", "cuda"): 16,
    ("m3-qwen3-30b-a3b", "cuda"): 32,
    ("m1-qwen35-4b", "metal"): 16,
    ("m2-qwen35-35b-a3b", "metal"): 4,
    ("m3-qwen3-30b-a3b", "metal"): 16,
}
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_PROMPT = (
    "Write the integers from 1 through 100 in ascending order, separated only "
    "by single spaces. Do not add commentary."
)
RUN_SAMPLE_COUNT = 3
RUN_MAX_TOKENS = 128
WARMUP_REQUESTS = 10
SEED = 9271
ACTIVE_PROBE = {
    "format": "json",
    "path": "/health",
    "selector": "engine.active_requests",
}
CUDA_COMPUTE_QUERY = (
    "--query-compute-apps=pid,used_gpu_memory",
    "--format=csv,noheader,nounits",
)
CUDA_GPU_UUID_QUERY = (
    "--query-gpu=uuid",
    "--format=csv,noheader,nounits",
)
CUDA_PID_NAMESPACE_BRIDGE_CONTRACT = (
    "ferrum.runtime-vnext.r2.cuda-pid-namespace-bridge.v1"
)
RESERVED_EXTRA_OPTIONS = {
    "--backend",
    "--host",
    "--port",
    "--max-num-seqs",
    "--runtime-memory-budget-bytes",
    "--effective-config-json",
    "--semantic-source",
    "--tokenizer-source",
    "--served-model-name",
    "--profile-detail",
    "--profile-jsonl",
    "--memory-profile-jsonl",
    "--scheduler-trace-jsonl",
    "--prompt",
    "--max-tokens",
    "--seed",
    "--temperature",
    "--top-k",
    "--top-p",
    "--repeat-penalty",
    "--output-format",
    "--enable-thinking",
    "--disable-thinking",
}


class R2CollectorError(RuntimeError):
    pass


def validate_typed_active_cap(model_key: str, backend: str, value: Any) -> int:
    floor = TYPED_ACTIVE_CAP_FLOORS[(model_key, backend)]
    require(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= floor,
        f"typed_active_cap must meet the {model_key}/{backend} active floor {floor}",
    )
    return floor


class CollectorInterrupted(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = signal.Signals(signum)
        super().__init__(self.signum.name)


def run_interruptibly(action: Callable[[], int], *, report: bool = True) -> int:
    """Turn TERM/INT into an exception so nested process cleanup can unwind."""
    prior_handlers: dict[signal.Signals, Any] = {}
    interrupted = False

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        if interrupted:
            return
        interrupted = True
        for managed_signal in prior_handlers:
            signal.signal(managed_signal, signal.SIG_IGN)
        raise CollectorInterrupted(signum)

    try:
        for managed_signal in (signal.SIGINT, signal.SIGTERM):
            prior_handlers[managed_signal] = signal.getsignal(managed_signal)
            signal.signal(managed_signal, handle_signal)
        return action()
    except CollectorInterrupted as exc:
        if report:
            print(
                f"runtime vNext R2 Ferrum collector interrupted by {exc.signum.name}; child cleanup completed",
                file=sys.stderr,
            )
        return 128 + int(exc.signum)
    finally:
        for managed_signal, previous in prior_handlers.items():
            signal.signal(managed_signal, previous)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R2CollectorError(message)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def config_template() -> dict[str, Any]:
    """Return the complete minimum operator-authored config shape.

    The template defaults to the locked Metal lane.  For CUDA, operators change
    ``backend``, the backend-specific candidate/hardware values, and provide the
    ``sharegpt`` path; both realistic dataset keys remain visible so that one
    template documents the complete two-backend schema.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "model_key": "m1-qwen35-4b",
        "backend": "metal",
        "request_model": "m1-qwen35-4b",
        "models_lock_path": "/absolute/path/to/models.lock.json",
        "correctness_manifest_path": "/absolute/path/to/r1-aggregate-manifest.json",
        "model_origin_path": "/absolute/path/to/locked-model",
        "semantic_source_root": "/absolute/path/to/semantic-source",
        "tokenizer_source_root": "/absolute/path/to/tokenizer-source",
        "candidate": {
            "binary_path": "/absolute/path/to/ferrum",
            "build_log_path": "/absolute/path/to/clean-build.log",
            "build_receipt_path": "/absolute/path/to/clean-build-receipt.json",
            "source_git_sha": "0000000000000000000000000000000000000000",
            "dirty_status": {"is_dirty": False, "status_short": []},
            "cargo_features": ["metal"],
            "env": {},
        },
        "hardware": {
            "id": "m1-max-24gpu-32g",
            "fingerprint": "replace-with-immutable-hardware-fingerprint",
            "accelerator_model": "Apple M1 Max",
            "accelerator_count": 1,
            "gpu_core_count": 24,
            "memory_bytes": 34359738368,
        },
        "typed_active_cap": 16,
        "memory_budget_bytes": 30064771072,
        "server": {
            "host": "127.0.0.1",
            "port": 18080,
            "ready_timeout_sec": 900,
            "shutdown_timeout_sec": 60,
            "command_timeout_sec": 7200,
            "extra_serve_argv": [],
        },
        "run": {"extra_argv": []},
        "datasets": {
            "real-chat": "/absolute/path/to/real-chat.jsonl",
            "sharegpt": "/absolute/path/to/sharegpt.jsonl",
        },
        "goodput_slo": {"ttft": 500.0, "tpot": 50.0, "e2e": 30000.0},
    }


def write_config_template(path: Path) -> Path:
    output = path.expanduser().resolve()
    require(not output.exists(), f"refusing to overwrite config template: {output}")
    atomic_write_json(output, config_template())
    print(f"{TEMPLATE_PREFIX}: {output}")
    return output


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R2CollectorError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def file_sha256(path: Path) -> str:
    return collector_support.file_sha256(path)


PROCESS_COLLECTION_EPOCH = {
    "collector_sha256": file_sha256(COLLECTOR_PATH),
    "support_sha256": file_sha256(SUPPORT_PATH),
    "resource_sampler_sha256": file_sha256(RESOURCE_SAMPLER_PATH),
}


def process_collection_epoch() -> dict[str, str]:
    return copy.deepcopy(PROCESS_COLLECTION_EPOCH)


def canonical_json_sha256(value: Any) -> str:
    return collector_support.canonical_json_sha256(value)


def artifact_relative(root: Path, path: Path) -> str:
    return collector_support.artifact_relative(root, path)


def artifact_ref(root: Path, path: Path, *, kind: str) -> dict[str, Any]:
    require(path.is_file(), f"artifact is missing: {path}")
    return {
        "kind": kind,
        "path": artifact_relative(root, path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def atomic_write_json(path: Path, value: Any) -> None:
    collector_support.atomic_write_json(path, value)


def atomic_write_text(path: Path, value: str) -> None:
    collector_support.atomic_write_text(path, value)


def parse_cuda_compute_rows(raw: str, label: str) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = [field.strip() for field in stripped.split(",")]
        require(
            len(fields) == 2 and all(field.isdigit() for field in fields),
            f"{label} row {line_number} is not numeric pid,memory evidence",
        )
        pid = int(fields[0])
        memory_mib = int(fields[1])
        require(pid > 0 and memory_mib > 0, f"{label} row {line_number} is not positive")
        rows.append({"pid": pid, "used_gpu_memory_mib": memory_mib})
    return rows


def process_group_pids(pgid: int) -> set[int]:
    process = subprocess.run(
        ["ps", "-eo", "pid=,pgid="],
        capture_output=True,
        text=True,
        check=False,
    )
    require(process.returncode == 0, "ps failed while binding CUDA process evidence")
    pids: set[int] = set()
    for line in process.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2 and all(field.isdigit() for field in fields):
            pid, observed_pgid = (int(field) for field in fields)
            if observed_pgid == pgid:
                pids.add(pid)
    return pids


def normalize_cuda_compute_rows(
    rows: list[dict[str, int]],
    *,
    server_pid: int,
    group_pids: set[int],
    preflight_rows: list[dict[str, int]],
    proc_exists: Any = None,
) -> tuple[list[dict[str, int]], str]:
    """Bind a host-PID NVIDIA row to a container-local server group.

    Native PID matches always pass through.  The namespace fallback is only
    allowed for a dedicated, initially idle one-GPU lane with exactly one new
    compute application whose host PID is absent from the container /proc.
    """

    exists = proc_exists or (lambda pid: Path(f"/proc/{pid}").exists())
    require(server_pid in group_pids, "CUDA bridge server PID left its process group")
    native = [row for row in rows if row["pid"] in group_pids]
    if native:
        return rows, "native-process-group-pid"
    require(not preflight_rows, "CUDA namespace fallback requires an idle preflight")
    if not rows:
        return [], "idle-before-device-allocation"
    require(len(rows) == 1, "CUDA namespace fallback requires exactly one compute application")
    host_pid = rows[0]["pid"]
    require(host_pid not in group_pids, "CUDA namespace fallback received an ambiguous local PID")
    require(not exists(host_pid), "CUDA namespace fallback host PID is visible in container /proc")
    return (
        [
            {
                "pid": server_pid,
                "used_gpu_memory_mib": rows[0]["used_gpu_memory_mib"],
            }
        ],
        "single-new-host-pid-mapped-to-container-server",
    )


def append_bridge_audit(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def cuda_pid_namespace_bridge(args: argparse.Namespace) -> int:
    child_argv = list(args.exec_argv)
    if child_argv and child_argv[0] == "--":
        child_argv = child_argv[1:]
    audit_path = args.bridge_audit_log.expanduser().resolve()
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": CUDA_PID_NAMESPACE_BRIDGE_CONTRACT,
        "observed_at": now_iso(),
        "collector_path": COLLECTOR_RELATIVE_PATH,
        "collector_sha256": args.bridge_collector_sha256,
        "server_pid": args.bridge_server_pid,
        "server_pgid": args.bridge_server_pgid,
        "nvidia_smi_argv": child_argv,
        "status": "reject",
    }
    try:
        require(
            args.bridge_collector_sha256 == file_sha256(COLLECTOR_PATH),
            "CUDA bridge collector source changed after the parent process started",
        )
        real_binary = args.real_nvidia_smi.expanduser().resolve()
        require(real_binary.is_file(), f"real nvidia-smi is missing: {real_binary}")
        preflight = read_json(args.bridge_preflight.expanduser().resolve())
        require(
            preflight.get("contract") == CUDA_PID_NAMESPACE_BRIDGE_CONTRACT
            and preflight.get("collector_sha256") == args.bridge_collector_sha256
            and preflight.get("real_nvidia_smi_path") == str(real_binary)
            and preflight.get("real_nvidia_smi_sha256") == file_sha256(real_binary)
            and preflight.get("gpu_count") == 1
            and isinstance(preflight.get("compute_apps"), list),
            "CUDA bridge preflight identity is invalid",
        )
        process = subprocess.run(
            [str(real_binary), *child_argv],
            capture_output=True,
            text=True,
            check=False,
        )
        audit.update(
            {
                "real_nvidia_smi_path": str(real_binary),
                "real_returncode": process.returncode,
                "raw_stdout": process.stdout,
                "raw_stderr": process.stderr,
            }
        )
        if process.returncode != 0:
            sys.stdout.write(process.stdout)
            sys.stderr.write(process.stderr)
            audit["error"] = "real nvidia-smi returned non-zero"
            return process.returncode
        is_compute_query = all(option in child_argv for option in CUDA_COMPUTE_QUERY)
        if not is_compute_query:
            sys.stdout.write(process.stdout)
            sys.stderr.write(process.stderr)
            audit.update({"status": "pass", "strategy": "transparent-passthrough"})
            return 0
        raw_rows = parse_cuda_compute_rows(process.stdout, "CUDA compute query")
        group_pids = process_group_pids(args.bridge_server_pgid)
        normalized, strategy = normalize_cuda_compute_rows(
            raw_rows,
            server_pid=args.bridge_server_pid,
            group_pids=group_pids,
            preflight_rows=preflight["compute_apps"],
        )
        normalized_stdout = "".join(
            f"{row['pid']}, {row['used_gpu_memory_mib']}\n" for row in normalized
        )
        sys.stdout.write(normalized_stdout)
        sys.stderr.write(process.stderr)
        audit.update(
            {
                "status": "pass",
                "strategy": strategy,
                "group_pids": sorted(group_pids),
                "raw_compute_apps": raw_rows,
                "normalized_compute_apps": normalized,
                "normalized_stdout": normalized_stdout,
            }
        )
        return 0
    except (OSError, R2CollectorError, subprocess.SubprocessError) as exc:
        audit["error"] = f"{type(exc).__name__}: {exc}"
        print(f"CUDA PID namespace bridge failed: {exc}", file=sys.stderr)
        return 1
    finally:
        append_bridge_audit(audit_path, audit)


def capture_cuda_bridge_preflight(attempt_dir: Path) -> dict[str, Any]:
    resolved = shutil.which("nvidia-smi")
    require(resolved is not None, "CUDA resource evidence requires nvidia-smi")
    real_binary = Path(resolved).resolve()
    compute = subprocess.run(
        [str(real_binary), *CUDA_COMPUTE_QUERY],
        capture_output=True,
        text=True,
        check=False,
    )
    require(compute.returncode == 0, "CUDA process preflight nvidia-smi query failed")
    compute_rows = parse_cuda_compute_rows(compute.stdout, "CUDA process preflight")
    require(not compute_rows, "CUDA performance lane did not start from an idle accelerator")
    gpu = subprocess.run(
        [str(real_binary), *CUDA_GPU_UUID_QUERY],
        capture_output=True,
        text=True,
        check=False,
    )
    gpu_rows = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    require(gpu.returncode == 0 and len(gpu_rows) == 1, "CUDA bridge requires exactly one visible GPU")
    path = attempt_dir / "cuda-pid-namespace-preflight.json"
    document = {
        "schema_version": SCHEMA_VERSION,
        "contract": CUDA_PID_NAMESPACE_BRIDGE_CONTRACT,
        "artifact_type": "runtime_vnext_r2_cuda_pid_namespace_preflight",
        "captured_at": now_iso(),
        "collector_path": COLLECTOR_RELATIVE_PATH,
        "collector_sha256": PROCESS_COLLECTION_EPOCH["collector_sha256"],
        "real_nvidia_smi_path": str(real_binary),
        "real_nvidia_smi_sha256": file_sha256(real_binary),
        "compute_query": [str(real_binary), *CUDA_COMPUTE_QUERY],
        "compute_stdout": compute.stdout,
        "compute_stderr": compute.stderr,
        "compute_apps": compute_rows,
        "gpu_query": [str(real_binary), *CUDA_GPU_UUID_QUERY],
        "gpu_stdout": gpu.stdout,
        "gpu_stderr": gpu.stderr,
        "gpu_count": len(gpu_rows),
        "gpu_uuids": gpu_rows,
    }
    atomic_write_json(path, document)
    return {"path": path, "document": document, "real_binary": real_binary}


def wait_for_cuda_device_allocation(
    process: subprocess.Popen[Any],
    *,
    pid: int,
    pgid: int,
    preflight: dict[str, Any],
    timeout_sec: float = 30.0,
    query_compute_rows: Any = None,
    group_pids_fn: Any = process_group_pids,
) -> None:
    if query_compute_rows is None:
        def query_compute_rows() -> list[dict[str, int]]:
            query = subprocess.run(
                [str(preflight["real_binary"]), *CUDA_COMPUTE_QUERY],
                capture_output=True,
                text=True,
                check=False,
            )
            require(query.returncode == 0, "CUDA allocation nvidia-smi query failed")
            return parse_cuda_compute_rows(query.stdout, "CUDA allocation query")

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        require(process.poll() is None, f"run process exited before CUDA allocation with {process.returncode}")
        normalized, _ = normalize_cuda_compute_rows(
            query_compute_rows(),
            server_pid=pid,
            group_pids=group_pids_fn(pgid),
            preflight_rows=preflight["document"]["compute_apps"],
        )
        if normalized:
            return
        time.sleep(0.05)
    raise R2CollectorError("run process did not allocate CUDA device memory before sampler startup")


def prepare_cuda_bridge(
    attempt_dir: Path,
    stem: str,
    *,
    pid: int,
    pgid: int,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    bridge_dir = attempt_dir / f"{stem}.cuda-pid-bridge-bin"
    bridge_dir.mkdir(parents=True, exist_ok=False)
    wrapper = bridge_dir / "nvidia-smi"
    audit = attempt_dir / f"{stem}.cuda-pid-bridge-audit.jsonl"
    argv = [
        sys.executable,
        str(COLLECTOR_PATH),
        "--cuda-pid-namespace-bridge",
        "--bridge-collector-sha256",
        PROCESS_COLLECTION_EPOCH["collector_sha256"],
        "--real-nvidia-smi",
        str(preflight["real_binary"]),
        "--bridge-server-pid",
        str(pid),
        "--bridge-server-pgid",
        str(pgid),
        "--bridge-preflight",
        str(preflight["path"]),
        "--bridge-audit-log",
        str(audit),
        "--",
        '"$@"',
    ]
    script = "#!/bin/sh\nexec " + " ".join(
        '"$@"' if value == '"$@"' else shlex.quote(value) for value in argv
    ) + "\n"
    atomic_write_text(wrapper, script)
    wrapper.chmod(0o755)
    environment = cuda_bridge_sampler_environment(wrapper)
    return {
        "dir": bridge_dir,
        "wrapper": wrapper,
        "audit": audit,
        "preflight": preflight["path"],
        "real_binary": preflight["real_binary"],
        "server_pid": pid,
        "server_pgid": pgid,
        "environment": environment,
        "collector_sha256": PROCESS_COLLECTION_EPOCH["collector_sha256"],
    }


def cuda_bridge_sampler_environment(wrapper: Path) -> dict[str, str]:
    environment = collector_support.sanitized_environment()
    environment["PATH"] = (
        f"{wrapper.resolve().parent}{os.pathsep}{environment.get('PATH', '')}"
    )
    return dict(sorted(environment.items()))


def cuda_bridge_evidence(root: Path, sampler: dict[str, Any]) -> dict[str, Any] | None:
    bridge = sampler.get("cuda_pid_namespace_bridge")
    if bridge is None:
        return None
    audit_path: Path = bridge["audit"]
    require(audit_path.is_file() and audit_path.stat().st_size > 0, "CUDA PID bridge audit is missing")
    real_binary_sha256 = bridge.get("real_binary_sha256")
    if real_binary_sha256 is None:
        real_binary_sha256 = file_sha256(bridge["real_binary"])
    require(
        isinstance(real_binary_sha256, str)
        and SHA256_RE.fullmatch(real_binary_sha256) is not None,
        "CUDA PID bridge real binary identity is invalid",
    )
    return {
        "contract": CUDA_PID_NAMESPACE_BRIDGE_CONTRACT,
        "bridge_source_path": COLLECTOR_RELATIVE_PATH,
        "bridge_source_sha256": bridge["collector_sha256"],
        "wrapper": artifact_ref(root, bridge["wrapper"], kind="cuda-pid-namespace-wrapper"),
        "preflight": artifact_ref(root, bridge["preflight"], kind="cuda-pid-namespace-preflight"),
        "audit": artifact_ref(root, audit_path, kind="cuda-pid-namespace-audit"),
        "real_nvidia_smi_path": str(bridge["real_binary"]),
        "real_nvidia_smi_sha256": real_binary_sha256,
        "server_pid": bridge["server_pid"],
        "server_pgid": bridge["server_pgid"],
        "sampler_environment_sha256": canonical_json_sha256(bridge["environment"]),
        "product_environment_unchanged": True,
    }


def validate_cuda_bridge_evidence(
    root: Path,
    resources: dict[str, Any],
    *,
    backend: str,
    label: str,
    expected_collector_sha256: str | None = None,
) -> None:
    evidence = resources.get("cuda_pid_namespace_bridge")
    if backend != "cuda":
        require(evidence is None, f"{label} unexpectedly contains a CUDA PID bridge")
        return
    require(isinstance(evidence, dict), f"{label} CUDA PID bridge evidence is missing")
    collector_sha = expected_collector_sha256 or PROCESS_COLLECTION_EPOCH["collector_sha256"]
    require(
        isinstance(collector_sha, str) and SHA256_RE.fullmatch(collector_sha) is not None,
        f"{label} expected CUDA PID bridge collector identity is invalid",
    )
    require(
        evidence.get("contract") == CUDA_PID_NAMESPACE_BRIDGE_CONTRACT
        and evidence.get("bridge_source_path") == COLLECTOR_RELATIVE_PATH
        and evidence.get("bridge_source_sha256") == collector_sha
        and evidence.get("product_environment_unchanged") is True
        and isinstance(evidence.get("sampler_environment_sha256"), str)
        and SHA256_RE.fullmatch(evidence["sampler_environment_sha256"]) is not None,
        f"{label} CUDA PID bridge identity is invalid",
    )
    validate_artifact_ref(root, evidence.get("wrapper"), f"{label}.bridge.wrapper")
    preflight_path = validate_artifact_ref(
        root, evidence.get("preflight"), f"{label}.bridge.preflight"
    )
    audit_path = validate_artifact_ref(root, evidence.get("audit"), f"{label}.bridge.audit")
    preflight = read_json(preflight_path)
    require(
        preflight.get("contract") == CUDA_PID_NAMESPACE_BRIDGE_CONTRACT
        and preflight.get("collector_sha256") == collector_sha
        and preflight.get("compute_apps") == []
        and preflight.get("gpu_count") == 1
        and preflight.get("real_nvidia_smi_path") == evidence.get("real_nvidia_smi_path")
        and preflight.get("real_nvidia_smi_sha256") == evidence.get("real_nvidia_smi_sha256"),
        f"{label} CUDA PID bridge preflight is invalid",
    )
    audit_rows = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(audit_rows, f"{label} CUDA PID bridge audit is empty")
    require(
        all(
            isinstance(row, dict)
            and row.get("contract") == CUDA_PID_NAMESPACE_BRIDGE_CONTRACT
            and row.get("collector_sha256") == collector_sha
            and row.get("server_pid") == evidence.get("server_pid")
            and row.get("server_pgid") == evidence.get("server_pgid")
            and row.get("real_nvidia_smi_path") == evidence.get("real_nvidia_smi_path")
            and row.get("status") == "pass"
            and row.get("real_returncode") == 0
            for row in audit_rows
        ),
        f"{label} CUDA PID bridge audit contains a rejection or identity mismatch",
    )
    compute_rows = [
        row
        for row in audit_rows
        if all(option in row.get("nvidia_smi_argv", []) for option in CUDA_COMPUTE_QUERY)
    ]
    mapped_compute_rows = [
        row for row in compute_rows if row.get("strategy") != "idle-before-device-allocation"
    ]
    require(len(mapped_compute_rows) >= 3, f"{label} CUDA PID bridge lacks three mapped compute samples")
    for row in compute_rows:
        normalized = row.get("normalized_compute_apps")
        if row.get("strategy") == "idle-before-device-allocation":
            require(
                row.get("raw_compute_apps") == [] and normalized == [],
                f"{label} CUDA PID bridge idle sample is invalid",
            )
            continue
        require(
            row.get("strategy")
            in {
                "native-process-group-pid",
                "single-new-host-pid-mapped-to-container-server",
            }
            and isinstance(normalized, list)
            and len(normalized) >= 1
            and all(
                isinstance(app, dict)
                and isinstance(app.get("pid"), int)
                and isinstance(app.get("used_gpu_memory_mib"), int)
                and app["used_gpu_memory_mib"] > 0
                for app in normalized
            ),
            f"{label} CUDA PID bridge compute mapping is invalid",
        )
        if row["strategy"] == "single-new-host-pid-mapped-to-container-server":
            require(
                normalized == [
                    {
                        "pid": evidence["server_pid"],
                        "used_gpu_memory_mib": row["raw_compute_apps"][0]["used_gpu_memory_mib"],
                    }
                ]
                and len(row.get("raw_compute_apps", [])) == 1
                and row["raw_compute_apps"][0]["pid"] != evidence["server_pid"],
                f"{label} CUDA namespace mapping is not one-to-one",
            )


def duration_seconds(started_at: str, finished_at: str) -> float:
    return collector_support.duration_seconds(started_at, finished_at)


def expected_cells(backend: str) -> tuple[dict[str, Any], ...]:
    require(backend in {"cuda", "metal"}, f"unsupported backend: {backend}")
    random_input = 256 if backend == "cuda" else 64
    random_concurrency = (1, 4, 16, 32) if backend == "cuda" else (1, 4, 16)
    realistic = "sharegpt" if backend == "cuda" else "real-chat"
    top = 32 if backend == "cuda" else 16
    cells = [
        {
            "sequence": sequence,
            "dataset": "random",
            "concurrency": concurrency,
            "input_tokens": random_input,
            "output_tokens": 128,
            "num_prompts": 100,
            "n_repeats": 3,
            "warmup_requests": WARMUP_REQUESTS,
        }
        for sequence, concurrency in enumerate(random_concurrency, start=1)
    ]
    for concurrency in (1, top):
        cells.append(
            {
                "sequence": len(cells) + 1,
                "dataset": realistic,
                "concurrency": concurrency,
                "input_tokens": random_input,
                "output_tokens": 128,
                "num_prompts": 30,
                "n_repeats": 3,
                "warmup_requests": WARMUP_REQUESTS,
            }
        )
    return tuple(cells)


def run_parity_cell(backend: str) -> dict[str, Any]:
    require(backend in {"cuda", "metal"}, f"unsupported backend: {backend}")
    return {
        "sequence": len(expected_cells(backend)) + 1,
        "dataset": "run-parity",
        "concurrency": 1,
        "input_tokens": 256 if backend == "cuda" else 64,
        "output_tokens": RUN_MAX_TOKENS,
        "num_prompts": RUN_SAMPLE_COUNT,
        "n_repeats": 3,
        "warmup_requests": WARMUP_REQUESTS,
        "formal_matrix_cell": False,
        "purpose": "same-prompt-output serve-c1 denominator for ferrum run parity",
    }


def cell_id(cell: dict[str, Any]) -> str:
    return f"{cell['dataset']}:c{cell['concurrency']}"


def parse_extra_argv(raw: Any, label: str) -> list[str]:
    require(isinstance(raw, list), f"{label} must be an argv list")
    require(all(isinstance(value, str) and value for value in raw), f"{label} contains an invalid token")
    for token in raw:
        option = token.split("=", 1)[0]
        require(option not in RESERVED_EXTRA_OPTIONS, f"{label} overrides collector-owned option {option}")
    collector_support.reject_secret_material(raw, label)
    return list(raw)


def resolve_file(raw: Any, label: str, *, executable: bool = False) -> Path:
    require(isinstance(raw, str) and raw, f"{label} is required")
    path = Path(raw).expanduser().resolve()
    require(path.is_file(), f"{label} is not a file: {path}")
    if executable:
        require(os.access(path, os.X_OK), f"{label} is not executable: {path}")
    return path


def resolve_directory(raw: Any, label: str) -> Path:
    require(isinstance(raw, str) and raw, f"{label} is required")
    path = Path(raw).expanduser().resolve()
    require(path.is_dir(), f"{label} is not a directory: {path}")
    return path


def git_output(argv: list[str]) -> str:
    process = subprocess.run(
        ["git", *argv],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    require(process.returncode == 0, f"git {' '.join(argv)} failed: {process.stderr.strip()}")
    return process.stdout.strip()


def reviewed_git_blob_sha256s(relative_path: str) -> frozenset[str]:
    cached = _REVIEWED_GIT_BLOB_SHA256S.get(relative_path)
    if cached is not None:
        return cached
    commits = git_output(["log", "--format=%H", "--", relative_path]).splitlines()
    require(commits, f"collection epoch source has no Git history: {relative_path}")
    digests: set[str] = set()
    for commit in commits:
        process = subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        if process.returncode == 0:
            digests.add(hashlib.sha256(process.stdout).hexdigest())
    require(digests, f"collection epoch source has no reviewed Git blobs: {relative_path}")
    result = frozenset(digests)
    _REVIEWED_GIT_BLOB_SHA256S[relative_path] = result
    return result


def require_reviewed_native_collection_epoch(collection_epoch: dict[str, str], label: str) -> None:
    for identity_key, relative_path in COLLECTION_EPOCH_SOURCE_PATHS.items():
        digest = collection_epoch[identity_key]
        require(
            digest in reviewed_git_blob_sha256s(relative_path),
            f"{label} {identity_key} is not a reviewed Git-history source",
        )


def load_locked_model(models_lock: dict[str, Any], model_key: str, backend: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [row for row in models_lock.get("models", []) if isinstance(row, dict) and row.get("key") == model_key]
    require(len(rows) == 1, f"models lock does not contain one {model_key} row")
    model = rows[0]
    lanes = model.get("lanes")
    require(isinstance(lanes, dict) and isinstance(lanes.get(backend), dict), f"models lock lacks {model_key}/{backend}")
    return model, lanes[backend]


def locked_model_files(lane: dict[str, Any]) -> dict[str, str]:
    rows = lane.get("files")
    require(isinstance(rows, list) and rows, "model lane files must be non-empty")
    result: dict[str, str] = {}
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"model lane files[{index}] must be an object")
        path = row.get("path")
        digest = row.get("sha256")
        require(isinstance(path, str) and path and not Path(path).is_absolute(), f"model lane files[{index}].path is invalid")
        require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None, f"model lane files[{index}].sha256 is invalid")
        require(path not in result, f"duplicate locked model path: {path}")
        result[path] = digest
    return result


def tokenizer_lock(lane: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = lane.get("tokenizer_source") or lane.get("semantic_source")
    require(isinstance(source, dict), "model lane lacks tokenizer/semantic source")
    rows = [row for row in source.get("files", []) if isinstance(row, dict) and row.get("path") == "tokenizer.json"]
    require(len(rows) == 1, "model lane must lock exactly one tokenizer.json")
    digest = rows[0].get("sha256")
    require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None, "tokenizer SHA256 is invalid")
    return source, rows[0]


def verify_model_origin(origin: Path, locked: dict[str, str]) -> dict[str, str]:
    if origin.is_file():
        require(len(locked) == 1 and origin.name in locked, "model file origin does not match the locked filename")
        actual = file_sha256(origin)
        require(actual == locked[origin.name], f"model SHA256 mismatch: {origin}")
        return {origin.name: actual}
    require(origin.is_dir(), f"model origin is neither file nor directory: {origin}")
    actual_files: dict[str, str] = {}
    for relative, expected in sorted(locked.items()):
        path = origin / relative
        require(path.is_file(), f"locked model file is missing: {path}")
        actual = file_sha256(path)
        require(actual == expected, f"model SHA256 mismatch: {path}")
        actual_files[relative] = actual
    return actual_files


def validate_r1_correctness_authority(
    correctness: dict[str, Any],
    *,
    backend: str,
    candidate_source: dict[str, Any],
    candidate_binary_sha256: str,
) -> None:
    """Fail before model startup when R1 did not authorize this candidate."""

    require(
        correctness.get("artifact_type") == R1_CORRECTNESS_ARTIFACT_TYPE
        and correctness.get("checkpoint_id") == "R1"
        and correctness.get("status") == "pass"
        and correctness.get("canonical") is True,
        "correctness manifest is not a canonical R1 PASS",
    )
    require(
        correctness.get("source") == candidate_source,
        "candidate source differs from the R1 correctness authority",
    )
    acceptance = correctness.get("acceptance")
    binaries = (
        acceptance.get("backend_binary_sha256")
        if isinstance(acceptance, dict)
        else None
    )
    require(
        isinstance(binaries, dict)
        and binaries.get(backend) == candidate_binary_sha256,
        f"candidate binary differs from the R1 correctness authority for {backend}",
    )


def normalize_config(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    collector_support.reject_secret_material(raw)
    require(raw.get("schema_version") == SCHEMA_VERSION, "config.schema_version must be 1")
    config = copy.deepcopy(raw)
    model_key = config.get("model_key")
    backend = config.get("backend")
    require(model_key in MODEL_KEYS, f"config.model_key must be one of {sorted(MODEL_KEYS)}")
    require(backend in {"cuda", "metal"}, "config.backend must be cuda or metal")
    require(isinstance(config.get("request_model"), str) and config["request_model"], "config.request_model is required")

    models_lock_path = resolve_file(config.get("models_lock_path"), "config.models_lock_path")
    correctness_path = resolve_file(config.get("correctness_manifest_path"), "config.correctness_manifest_path")
    models_lock = read_json(models_lock_path)
    correctness = read_json(correctness_path)
    model, lane = load_locked_model(models_lock, str(model_key), str(backend))
    require(isinstance(lane.get("revision"), str) and GIT_SHA_RE.fullmatch(lane["revision"]) is not None, "locked model revision is invalid")

    model_origin_raw = config.get("model_origin_path")
    require(isinstance(model_origin_raw, str) and model_origin_raw, "config.model_origin_path is required")
    # Hugging Face snapshot files are normally logical-name symlinks into the
    # blob store.  Keep the final path component intact for lock matching and
    # for the product argv; file hashing below intentionally follows it.
    model_origin = Path(os.path.abspath(os.path.expanduser(model_origin_raw)))
    model_files = verify_model_origin(model_origin, locked_model_files(lane))
    semantic_root = resolve_directory(config.get("semantic_source_root"), "config.semantic_source_root")
    tokenizer_root = resolve_directory(config.get("tokenizer_source_root", str(semantic_root)), "config.tokenizer_source_root")
    tokenizer_source, tokenizer_row = tokenizer_lock(lane)
    tokenizer_path = tokenizer_root / "tokenizer.json"
    require(tokenizer_path.is_file(), f"locked tokenizer is missing: {tokenizer_path}")
    require(file_sha256(tokenizer_path) == tokenizer_row["sha256"], "tokenizer SHA256 differs from models lock")

    candidate = config.get("candidate")
    require(isinstance(candidate, dict), "config.candidate is required")
    binary_path = resolve_file(candidate.get("binary_path"), "config.candidate.binary_path", executable=True)
    require(binary_path.name == "ferrum", "candidate binary must be named ferrum")
    build_log_path = resolve_file(candidate.get("build_log_path"), "config.candidate.build_log_path")
    build_receipt_path = resolve_file(candidate.get("build_receipt_path"), "config.candidate.build_receipt_path")
    source_sha = candidate.get("source_git_sha")
    require(isinstance(source_sha, str) and GIT_SHA_RE.fullmatch(source_sha) is not None, "candidate.source_git_sha is invalid")
    require(git_output(["cat-file", "-t", source_sha]) == "commit", "candidate source commit is unavailable")
    source_tree_sha = git_output(["rev-parse", f"{source_sha}^{{tree}}"])
    dirty = candidate.get("dirty_status")
    require(
        isinstance(dirty, dict)
        and dirty.get("is_dirty") is False
        and dirty.get("status_short") == [],
        "candidate.dirty_status must prove a clean build",
    )
    features = candidate.get("cargo_features")
    require(isinstance(features, list) and features and all(isinstance(item, str) and item for item in features), "candidate.cargo_features is required")
    require(backend in features, f"candidate.cargo_features must include {backend}")
    candidate_env = collector_support.sanitized_environment(candidate.get("env"))
    candidate_binary_sha256 = file_sha256(binary_path)
    validate_r1_correctness_authority(
        correctness,
        backend=str(backend),
        candidate_source={
            "git_sha": source_sha,
            "git_tree_sha": source_tree_sha,
            "dirty": False,
        },
        candidate_binary_sha256=candidate_binary_sha256,
    )

    hardware = config.get("hardware")
    require(isinstance(hardware, dict), "config.hardware is required")
    for field in ("id", "fingerprint", "accelerator_model"):
        require(isinstance(hardware.get(field), str) and hardware[field], f"hardware.{field} is required")
    require(hardware.get("accelerator_count") == 1, "R2 requires exactly one accelerator")
    memory_bytes = hardware.get("memory_bytes")
    require(isinstance(memory_bytes, int) and not isinstance(memory_bytes, bool) and memory_bytes > 0, "hardware.memory_bytes must be positive")
    if backend == "cuda":
        require("4090" in hardware["accelerator_model"], "CUDA R2 hardware must be one RTX 4090")
    else:
        require(hardware["accelerator_model"] == "Apple M1 Max", "Metal R2 hardware must be Apple M1 Max")
        require(hardware.get("gpu_core_count") == 24, "Metal R2 hardware must have 24 GPU cores")
        require(memory_bytes == 32 * 1024**3, "Metal R2 hardware must have 32 GiB unified memory")

    typed_active_cap = config.get("typed_active_cap")
    memory_budget_bytes = config.get("memory_budget_bytes")
    validate_typed_active_cap(str(model_key), str(backend), typed_active_cap)
    require(
        isinstance(memory_budget_bytes, int)
        and not isinstance(memory_budget_bytes, bool)
        and 0 < memory_budget_bytes <= memory_bytes,
        "memory_budget_bytes must fit locked hardware",
    )

    server = config.get("server")
    require(isinstance(server, dict), "config.server is required")
    require(isinstance(server.get("host"), str) and server["host"], "server.host is required")
    require(isinstance(server.get("port"), int) and 1024 <= server["port"] <= 65535, "server.port must be 1024..65535")
    for field, default in (("ready_timeout_sec", 900), ("shutdown_timeout_sec", 60), ("command_timeout_sec", 7200)):
        server.setdefault(field, default)
        require(isinstance(server[field], (int, float)) and not isinstance(server[field], bool) and server[field] > 0, f"server.{field} must be positive")
    server["extra_serve_argv"] = parse_extra_argv(server.get("extra_serve_argv", []), "server.extra_serve_argv")
    run = config.setdefault("run", {})
    require(isinstance(run, dict), "config.run must be an object")
    run["extra_argv"] = parse_extra_argv(run.get("extra_argv", []), "run.extra_argv")

    datasets = config.get("datasets")
    require(isinstance(datasets, dict), "config.datasets is required")
    realistic = "sharegpt" if backend == "cuda" else "real-chat"
    dataset_path = resolve_file(datasets.get(realistic), f"config.datasets.{realistic}")
    slo = config.setdefault("goodput_slo", {"ttft": 500.0, "tpot": 50.0, "e2e": 30000.0})
    require(isinstance(slo, dict) and set(slo) == {"ttft", "tpot", "e2e"}, "goodput_slo must contain ttft/tpot/e2e")
    require(
        all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0 for value in slo.values()),
        "goodput_slo values must be positive finite",
    )

    config.update(
        {
            "models_lock_path": str(models_lock_path),
            "correctness_manifest_path": str(correctness_path),
            "model_origin_path": str(model_origin),
            "semantic_source_root": str(semantic_root),
            "tokenizer_source_root": str(tokenizer_root),
        }
    )
    candidate.update(
        {
            "binary_path": str(binary_path),
            "build_log_path": str(build_log_path),
            "build_receipt_path": str(build_receipt_path),
            "source_tree_sha": source_tree_sha,
            "env": candidate_env,
        }
    )
    datasets[realistic] = str(dataset_path)
    context = {
        "models_lock": models_lock,
        "models_lock_path": models_lock_path,
        "correctness": correctness,
        "correctness_path": correctness_path,
        "model": model,
        "lane": lane,
        "model_files": model_files,
        "tokenizer_source": tokenizer_source,
        "tokenizer_row": tokenizer_row,
        "tokenizer_path": tokenizer_path,
        "dataset_path": dataset_path,
        "binary_path": binary_path,
        "build_log_path": build_log_path,
        "build_receipt_path": build_receipt_path,
    }
    return config, context


def collection_fingerprint(
    config: dict[str, Any],
    context: dict[str, Any],
    *,
    collector_sha256: str | None = None,
    support_sha256: str | None = None,
    resource_sampler_sha256: str | None = None,
) -> str:
    material = {
        "contract": CONTRACT,
        "collector_sha256": collector_sha256 or PROCESS_COLLECTION_EPOCH["collector_sha256"],
        "support_sha256": support_sha256 or PROCESS_COLLECTION_EPOCH["support_sha256"],
        "resource_sampler_sha256": resource_sampler_sha256 or PROCESS_COLLECTION_EPOCH["resource_sampler_sha256"],
        "models_lock_sha256": file_sha256(context["models_lock_path"]),
        "correctness_manifest_sha256": file_sha256(context["correctness_path"]),
        "candidate_binary_sha256": file_sha256(context["binary_path"]),
        "candidate_build_log_sha256": file_sha256(context["build_log_path"]),
        "candidate_build_receipt_sha256": file_sha256(context["build_receipt_path"]),
        "model_files": context["model_files"],
        "tokenizer_sha256": file_sha256(context["tokenizer_path"]),
        "realistic_dataset_sha256": file_sha256(context["dataset_path"]),
        "config": config,
    }
    return canonical_json_sha256(material)


def lane_dir(root: Path, config: dict[str, Any]) -> Path:
    return root / "r2-ferrum" / config["model_key"] / config["backend"]


def prepare_plan(root: Path, config: dict[str, Any], context: dict[str, Any], *, resume: bool) -> tuple[Path, str]:
    lane = lane_dir(root, config)
    lane.mkdir(parents=True, exist_ok=True)
    fingerprint = collection_fingerprint(config, context)
    normalized_path = lane / "config.normalized.json"
    plan_path = lane / "plan.json"
    if normalized_path.exists():
        require(resume, f"normalized config already exists; pass --resume: {normalized_path}")
        require(read_json(normalized_path) == config, "resume config differs from frozen normalized config")
    else:
        atomic_write_json(normalized_path, config)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "artifact_type": "runtime_vnext_r2_ferrum_collection_plan",
        "collector": {
            "path": COLLECTOR_RELATIVE_PATH,
            "sha256": PROCESS_COLLECTION_EPOCH["collector_sha256"],
            "support_path": SUPPORT_PATH.relative_to(REPO_ROOT).as_posix(),
            "support_sha256": PROCESS_COLLECTION_EPOCH["support_sha256"],
            "resource_sampler_path": RESOURCE_SAMPLER_PATH.relative_to(REPO_ROOT).as_posix(),
            "resource_sampler_sha256": PROCESS_COLLECTION_EPOCH["resource_sampler_sha256"],
        },
        "config_fingerprint": fingerprint,
        "config": artifact_ref(root, normalized_path, kind="normalized-config"),
        "model_key": config["model_key"],
        "backend": config["backend"],
        "hardware": copy.deepcopy(config["hardware"]),
        "profile_detail": "off",
        "server_process_count_per_epoch": 1,
        "server_epoch_policy": "one fresh server process resumes after the validated completed-cell prefix",
        "server_cell_order": [copy.deepcopy(cell) for cell in expected_cells(config["backend"])],
        "run_serve_parity_probe": copy.deepcopy(run_parity_cell(config["backend"])),
        "run_process_count": RUN_SAMPLE_COUNT,
        "run_policy": {
            "prompt_sha256": hashlib.sha256(RUN_PROMPT.encode("utf-8")).hexdigest(),
            "max_tokens": RUN_MAX_TOKENS,
            "seed": SEED,
            "temperature": 0.0,
            "top_k": 20,
            "top_p": 0.8,
            "repeat_penalty": 1.0,
            "enable_thinking": False,
            "profile_detail": "off",
        },
        "external_engine": None,
        "legacy_binary": None,
        "abba_order": None,
    }
    if plan_path.exists():
        require(resume, f"collection plan already exists; pass --resume: {plan_path}")
        frozen_plan = read_json(plan_path)
        frozen_collector = frozen_plan.get("collector")
        require(isinstance(frozen_collector, dict), "frozen plan collector binding is missing")
        frozen_collector_sha = frozen_collector.get("sha256")
        frozen_support_sha = frozen_collector.get("support_sha256")
        frozen_sampler_sha = frozen_collector.get("resource_sampler_sha256")
        require(
            isinstance(frozen_collector_sha, str)
            and SHA256_RE.fullmatch(frozen_collector_sha) is not None,
            "frozen plan collector SHA256 is invalid",
        )
        require(
            isinstance(frozen_support_sha, str)
            and SHA256_RE.fullmatch(frozen_support_sha) is not None
            and isinstance(frozen_sampler_sha, str)
            and SHA256_RE.fullmatch(frozen_sampler_sha) is not None,
            "frozen plan support/sampler SHA256 is invalid",
        )
        frozen_fingerprint = collection_fingerprint(
            config,
            context,
            collector_sha256=frozen_collector_sha,
            support_sha256=frozen_support_sha,
            resource_sampler_sha256=frozen_sampler_sha,
        )
        frozen_expected = copy.deepcopy(plan)
        frozen_expected["collector"] = copy.deepcopy(frozen_collector)
        frozen_expected["config_fingerprint"] = frozen_fingerprint
        if "server_process_count" in frozen_plan and "server_process_count_per_epoch" not in frozen_plan:
            require(frozen_plan.get("server_process_count") == 1, "legacy frozen plan server process count is invalid")
            frozen_expected.pop("server_process_count_per_epoch")
            frozen_expected.pop("server_epoch_policy")
            frozen_expected["server_process_count"] = 1
        require(frozen_plan == frozen_expected, "resume plan differs from frozen plan")
        fingerprint = frozen_fingerprint
    else:
        atomic_write_json(plan_path, plan)
    return lane, fingerprint


def stage_inputs(root: Path, lane: Path, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    prefix = artifact_relative(root, lane / "inputs")
    binary = collector_support.stage_file(root, context["binary_path"], f"{prefix}/candidate/ferrum")
    binary.chmod(binary.stat().st_mode | 0o100)
    build_log = collector_support.stage_file(root, context["build_log_path"], f"{prefix}/candidate/build.log")
    build_receipt = collector_support.stage_file(root, context["build_receipt_path"], f"{prefix}/candidate/build-receipt.json")
    models_lock = collector_support.stage_file(root, context["models_lock_path"], f"{prefix}/models.lock.json")
    correctness = collector_support.stage_file(root, context["correctness_path"], f"{prefix}/correctness-manifest.json")
    tokenizer = collector_support.stage_file(
        root,
        context["tokenizer_path"],
        f"{prefix}/tokenizer/tokenizer.json",
        context["tokenizer_row"]["sha256"],
    )
    realistic = "sharegpt" if config["backend"] == "cuda" else "real-chat"
    dataset = collector_support.stage_file(root, context["dataset_path"], f"{prefix}/datasets/{realistic}.jsonl")
    parity_dataset = lane / "inputs" / "datasets" / "run-parity.jsonl"
    parity_payload = "".join(
        json.dumps({"input": RUN_PROMPT}, sort_keys=True, separators=(",", ":")) + "\n"
        for _ in range(RUN_SAMPLE_COUNT)
    )
    if parity_dataset.exists():
        require(parity_dataset.read_text(encoding="utf-8") == parity_payload, "run parity dataset changed during resume")
    else:
        atomic_write_text(parity_dataset, parity_payload)
    run_prompt = lane / "inputs" / "run-prompt.json"
    prompt_document = {
        "schema_version": SCHEMA_VERSION,
        "prompt": RUN_PROMPT,
        "prompt_sha256": hashlib.sha256(RUN_PROMPT.encode("utf-8")).hexdigest(),
        "max_tokens": RUN_MAX_TOKENS,
        "eos_policy": "model-metadata",
    }
    if run_prompt.exists():
        require(read_json(run_prompt) == prompt_document, "run prompt changed during resume")
    else:
        atomic_write_json(run_prompt, prompt_document)
    return {
        "binary_path": binary,
        "binary": artifact_ref(root, binary, kind="candidate-binary"),
        "build_log": artifact_ref(root, build_log, kind="build-log"),
        "build_receipt": artifact_ref(root, build_receipt, kind="build-receipt"),
        "models_lock": artifact_ref(root, models_lock, kind="models-lock"),
        "correctness_manifest": artifact_ref(root, correctness, kind="correctness-manifest"),
        "tokenizer_path": tokenizer,
        "tokenizer": artifact_ref(root, tokenizer, kind="tokenizer"),
        "realistic_dataset_path": dataset,
        "realistic_dataset": artifact_ref(root, dataset, kind="realistic-dataset"),
        "run_parity_dataset_path": parity_dataset,
        "run_parity_dataset": artifact_ref(root, parity_dataset, kind="run-parity-dataset"),
        "run_prompt": artifact_ref(root, run_prompt, kind="run-prompt"),
    }


def server_argv(binary: Path, attempt_dir: Path, config: dict[str, Any]) -> tuple[list[str], Path, Path]:
    effective = attempt_dir / "server-effective-config.json"
    scheduler_trace = attempt_dir / "server-scheduler-trace.jsonl"
    argv = [
        str(binary),
        "serve",
        config["model_origin_path"],
        "--backend",
        config["backend"],
        "--host",
        config["server"]["host"],
        "--port",
        str(config["server"]["port"]),
        "--max-num-seqs",
        str(config["typed_active_cap"]),
        "--runtime-memory-budget-bytes",
        str(config["memory_budget_bytes"]),
        "--semantic-source",
        config["semantic_source_root"],
        "--tokenizer-source",
        config["tokenizer_source_root"],
        "--served-model-name",
        config["request_model"],
        "--profile-detail",
        "off",
        "--scheduler-trace-jsonl",
        str(scheduler_trace),
        "--effective-config-json",
        str(effective),
        *config["server"]["extra_serve_argv"],
    ]
    return argv, effective, scheduler_trace


def bench_argv(
    binary: Path,
    raw_report: Path,
    config: dict[str, Any],
    inputs: dict[str, Any],
    cell: dict[str, Any],
) -> list[str]:
    slo = config["goodput_slo"]
    dataset = cell["dataset"]
    argv = [
        str(binary),
        "bench-serve",
        "--base-url",
        f"http://{config['server']['host']}:{config['server']['port']}",
        "--model",
        config["request_model"],
        "--tokenizer",
        str(inputs["tokenizer_path"].parent),
        "--target-backend",
        config["backend"],
        "--http-connection-mode",
        "fresh",
        "--concurrency",
        str(cell["concurrency"]),
        "--dataset",
        "random" if dataset == "random" else "sharegpt",
        "--random-input-len",
        str(cell["input_tokens"]),
        "--random-output-len",
        str(cell["output_tokens"]),
        "--num-prompts",
        str(cell["num_prompts"]),
        "--warmup-requests",
        str(cell["warmup_requests"]),
        "--n-repeats",
        str(cell["n_repeats"]),
        "--seed",
        str(SEED),
        "--output",
        "json",
        "--out",
        str(raw_report),
        "--hw-id",
        config["hardware"]["id"],
        "--commit-sha",
        config["candidate"]["source_git_sha"],
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
        dataset_path = (
            inputs["run_parity_dataset_path"]
            if dataset == "run-parity"
            else inputs["realistic_dataset_path"]
        )
        argv.extend(["--sharegpt-path", str(dataset_path)])
    return argv


def run_argv(binary: Path, effective: Path, config: dict[str, Any]) -> list[str]:
    return [
        str(binary),
        "run",
        config["model_origin_path"],
        "--prompt",
        RUN_PROMPT,
        "--semantic-source",
        config["semantic_source_root"],
        "--tokenizer-source",
        config["tokenizer_source_root"],
        "--max-tokens",
        str(RUN_MAX_TOKENS),
        "--disable-thinking",
        "--seed",
        str(SEED),
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
        "--runtime-memory-budget-bytes",
        str(config["memory_budget_bytes"]),
        "--profile-detail",
        "off",
        "--output-format",
        "jsonl",
        "--effective-config-json",
        str(effective),
        *config["run"]["extra_argv"],
    ]


def cell_sampler_argv(
    attempt_dir: Path,
    session: dict[str, Any],
    config: dict[str, Any],
    cell: dict[str, Any],
) -> list[str]:
    identifier = cell_id(cell)
    stem = identifier.replace(":", "-")
    observations = attempt_dir / f"{stem}.resource-observations.jsonl"
    stop_file = attempt_dir / f"{stem}.resource-stop"
    max_duration = int(math.ceil(float(config["server"]["command_timeout_sec"]))) + 120
    return [
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
        identifier,
        "--backend",
        config["backend"],
        "--hardware-id",
        config["hardware"]["id"],
        "--base-url",
        session["base_url"],
        "--active-probe-format",
        ACTIVE_PROBE["format"],
        "--active-selector",
        ACTIVE_PROBE["selector"],
        "--active-semantics",
        "scheduler-active-high-water",
        "--runtime-log",
        session["runtime_log_origin_path"],
        "--stop-file",
        str(stop_file),
        "--interval-ms",
        "250",
        "--max-duration-sec",
        str(max_duration),
        "--active-probe-timeout-ms",
        str(collector_support.resource_sampler.ACTIVE_PROBE_TIMEOUT_MS),
        "--active-probe-max-attempts",
        str(collector_support.resource_sampler.ACTIVE_PROBE_MAX_ATTEMPTS),
        "--active-path",
        ACTIVE_PROBE["path"],
    ]


def start_cell_sampler(
    root: Path,
    attempt_dir: Path,
    session: dict[str, Any],
    config: dict[str, Any],
    cell: dict[str, Any],
    cuda_preflight: dict[str, Any] | None,
) -> dict[str, Any]:
    """Start the shared sampler with a cell deadline longer than the benchmark."""
    identifier = cell_id(cell)
    stem = identifier.replace(":", "-")
    observations = attempt_dir / f"{stem}.resource-observations.jsonl"
    stop_file = attempt_dir / f"{stem}.resource-stop"
    stdout_path = attempt_dir / f"{stem}.resource-sampler.stdout.log"
    stderr_path = attempt_dir / f"{stem}.resource-sampler.stderr.log"
    require(not observations.exists(), f"resource observation already exists: {observations}")
    bridge = None
    sampler_environment = collector_support.sanitized_environment()
    if config["backend"] == "cuda":
        require(cuda_preflight is not None, "CUDA sampler requires an idle process preflight")
        bridge = prepare_cuda_bridge(
            attempt_dir,
            stem,
            pid=session["pid"],
            pgid=session["pgid"],
            preflight=cuda_preflight,
        )
        sampler_environment = bridge["environment"]
    argv = cell_sampler_argv(attempt_dir, session, config, cell)
    stdout_handle = stdout_path.open("x", encoding="utf-8")
    stderr_handle = stderr_path.open("x", encoding="utf-8")
    process: subprocess.Popen[Any] | None = None
    try:
        stdout_handle.write("[r2-collector] resource sampler stdout follows\n")
        stderr_handle.write("[r2-collector] resource sampler stderr follows\n")
        stdout_handle.flush()
        stderr_handle.flush()
        process = subprocess.Popen(
            argv,
            env=sampler_environment,
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
            "cuda_pid_namespace_bridge": bridge,
        }
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            require(process.poll() is None, f"resource sampler exited during startup with {process.returncode}")
            if observations.exists() and observations.stat().st_size > 0:
                if len(observations.read_text(encoding="utf-8", errors="replace").splitlines()) >= 2:
                    return meta
            time.sleep(0.05)
        raise R2CollectorError("resource sampler did not produce its first observation")
    except BaseException:
        if process is not None:
            collector_support.cleanup_process_group_noexcept(process, 5.0)
        for handle in (stdout_handle, stderr_handle):
            try:
                handle.close()
            except BaseException:
                pass
        collector_support.ensure_nonempty_log(stdout_path, "resource sampler stdout")
        collector_support.ensure_nonempty_log(stderr_path, "resource sampler stderr")
        raise


def start_run_resource_sampler(
    root: Path,
    attempt_dir: Path,
    *,
    pid: int,
    pgid: int,
    sample_id: str,
    config: dict[str, Any],
    stderr_path: Path,
    cuda_preflight: dict[str, Any] | None,
) -> dict[str, Any]:
    observations = attempt_dir / "resource-observations.jsonl"
    stop_file = attempt_dir / "resource-stop"
    stdout_path = attempt_dir / "resource-sampler.stdout.log"
    sampler_stderr_path = attempt_dir / "resource-sampler.stderr.log"
    bridge = None
    sampler_environment = collector_support.sanitized_environment()
    if config["backend"] == "cuda":
        require(cuda_preflight is not None, "CUDA run sampler requires an idle process preflight")
        bridge = prepare_cuda_bridge(
            attempt_dir,
            "run-c1",
            pid=pid,
            pgid=pgid,
            preflight=cuda_preflight,
        )
        sampler_environment = bridge["environment"]
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
        config["hardware"]["id"],
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
        str(int(math.ceil(float(config["server"]["command_timeout_sec"]))) + 120),
    ]
    stdout_handle = stdout_path.open("x", encoding="utf-8")
    stderr_handle = sampler_stderr_path.open("x", encoding="utf-8")
    process: subprocess.Popen[Any] | None = None
    try:
        stdout_handle.write("[r2-collector] process resource sampler stdout follows\n")
        stderr_handle.write("[r2-collector] process resource sampler stderr follows\n")
        stdout_handle.flush()
        stderr_handle.flush()
        process = subprocess.Popen(
            argv,
            env=sampler_environment,
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
            "cuda_pid_namespace_bridge": bridge,
        }
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            require(
                process.poll() is None,
                f"process resource sampler exited during startup with {process.returncode}",
            )
            if observations.exists() and observations.stat().st_size > 0:
                if len(observations.read_text(encoding="utf-8", errors="replace").splitlines()) >= 2:
                    return meta
            time.sleep(0.05)
        raise R2CollectorError("process resource sampler did not produce its first observation")
    except BaseException:
        if process is not None:
            collector_support.cleanup_process_group_noexcept(process, 5.0)
        for handle in (stdout_handle, stderr_handle):
            try:
                handle.close()
            except BaseException:
                pass
        collector_support.ensure_nonempty_log(stdout_path, "process resource sampler stdout")
        collector_support.ensure_nonempty_log(sampler_stderr_path, "process resource sampler stderr")
        raise


def validate_bench_report(report: dict[str, Any], config: dict[str, Any], cell: dict[str, Any]) -> None:
    label = cell_id(cell)
    expected = cell["num_prompts"]
    require(report.get("model") == config["request_model"], f"{label} report model mismatch")
    require(report.get("backend") == config["backend"], f"{label} report backend mismatch")
    require(report.get("scenario") == "closed_loop", f"{label} report scenario must be closed_loop")
    require(report.get("concurrency") == cell["concurrency"], f"{label} report concurrency mismatch")
    require(report.get("n_gen") == 128, f"{label} report output length mismatch")
    require(report.get("n_repeats") == 3, f"{label} report must contain three repeats")
    require(report.get("n_requests_per_run") == expected, f"{label} report request count mismatch")
    require(report.get("warmup_requests") == WARMUP_REQUESTS, f"{label} report warmup mismatch")
    require(report.get("output_token_count_source") == "usage", f"{label} output tokens must come from usage")
    repeats = report.get("repeat_metrics")
    require(isinstance(repeats, list) and len(repeats) == 3, f"{label} repeat_metrics must have length 3")
    quality_names = {
        "bad_output",
        "malformed_stream",
        "missing_done",
        "duplicate_done",
        "zero_output_tokens",
        "stream_bulk_flush",
        "http_500",
        "panic",
    }
    for index, row in enumerate(repeats, start=1):
        require(isinstance(row, dict), f"{label} repeat {index} is not an object")
        require(row.get("repeat") == index, f"{label} repeat numbering mismatch")
        require(row.get("expected_requests") == expected, f"{label} repeat {index} expected count mismatch")
        require(row.get("completed_requests") == expected, f"{label} repeat {index} completion mismatch")
        require(row.get("errored_requests") == 0, f"{label} repeat {index} has errors")
        require(row.get("warmup_expected") == WARMUP_REQUESTS, f"{label} repeat {index} warmup expected mismatch")
        require(row.get("warmup_completed") == WARMUP_REQUESTS, f"{label} repeat {index} warmup completion mismatch")
        require(row.get("warmup_errored") == 0, f"{label} repeat {index} warmup errors")
        require(row.get("output_token_count_source") == "usage", f"{label} repeat {index} output token source mismatch")
        for field in ("quality_issues", "warmup_quality_issues"):
            quality = row.get(field)
            require(isinstance(quality, dict) and quality_names <= set(quality), f"{label} repeat {index} lacks {field}")
            require(all(quality[name] == 0 for name in quality_names), f"{label} repeat {index} has {field}")
    for field in ("completed_per_run", "errored_per_run"):
        values = report.get(field)
        expected_values = [expected] * 3 if field == "completed_per_run" else [0, 0, 0]
        require(values == expected_values, f"{label} {field} mismatch")
    for field in (
        "bad_output_per_run",
        "malformed_stream_per_run",
        "missing_done_per_run",
        "duplicate_done_per_run",
        "zero_output_tokens_per_run",
        "stream_bulk_flush_per_run",
        "http_500_per_run",
        "panic_per_run",
    ):
        require(report.get(field) == [0, 0, 0], f"{label} {field} is non-zero")
    for field in ("actual_input_tokens_per_request", "output_tokens_per_request", "itl_evidence_per_request"):
        rows = report.get(field)
        require(
            isinstance(rows, list)
            and len(rows) == 3
            and all(isinstance(row, list) and len(row) == expected for row in rows),
            f"{label} {field} must be a 3x{expected} matrix",
        )
    if cell["dataset"] == "run-parity":
        outputs = report["output_tokens_per_request"]
        require(
            all(value == RUN_MAX_TOKENS for row in outputs for value in row),
            f"{label} must use the same fixed {RUN_MAX_TOKENS}-token output boundary as ferrum run",
        )


def write_bench_request_sidecar(
    root: Path,
    path: Path,
    raw_report: Path,
    report: dict[str, Any],
    cell: dict[str, Any],
) -> dict[str, Any]:
    per_repeat: list[dict[str, Any]] = []
    inputs = report["actual_input_tokens_per_request"]
    outputs = report["output_tokens_per_request"]
    itl = report["itl_evidence_per_request"]
    repeats = report["repeat_metrics"]
    for repeat_index in range(3):
        requests = [
            {
                "request_ordinal": request_index + 1,
                "actual_input_tokens": inputs[repeat_index][request_index],
                "usage_output_tokens": outputs[repeat_index][request_index],
                "itl_evidence": itl[repeat_index][request_index],
                "output_token_count_source": "usage",
            }
            for request_index in range(cell["num_prompts"])
        ]
        per_repeat.append(
            {
                "repeat": repeat_index + 1,
                "requests": requests,
                "aggregate_metrics": copy.deepcopy(repeats[repeat_index]),
            }
        )
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_r2_bench_request_evidence_sidecar",
        "source_report": artifact_ref(root, raw_report, kind="raw-bench-report"),
        "cell": copy.deepcopy(cell),
        "output_token_count_source": "usage",
        "per_request_token_and_itl_evidence_complete": True,
        "per_request_latency_samples_exposed_by_b72_report": False,
        "latency_evidence_scope": "three immutable repeat-level percentile sets from BenchReport",
        "repeats": per_repeat,
    }
    atomic_write_json(path, sidecar)
    return artifact_ref(root, path, kind="bench-request-evidence-sidecar")


def parity_metrics(report: dict[str, Any]) -> dict[str, Any]:
    tpot_p50_ms = [float(row["tpot_ms"]["p50"]) for row in report["repeat_metrics"]]
    require(all(math.isfinite(value) and value > 0 for value in tpot_p50_ms), "run parity TPOT must be positive")
    steady_tps = [1000.0 / value for value in tpot_p50_ms]
    return {
        "metric_definition": "1000 / per-repeat request TPOT p50 milliseconds",
        "tpot_p50_ms_per_repeat": tpot_p50_ms,
        "steady_decode_tps_per_repeat": steady_tps,
        "steady_decode_tps_median": statistics.median(steady_tps),
        "engine_infer_e2e_ms_p50_per_repeat": [
            float(row["e2e_ms"]["p50"]) for row in report["repeat_metrics"]
        ],
    }


def run_bench_cell(
    root: Path,
    attempt_dir: Path,
    session: dict[str, Any],
    binary: Path,
    config: dict[str, Any],
    inputs: dict[str, Any],
    cell: dict[str, Any],
    cuda_preflight: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identifier = cell_id(cell)
    stem = identifier.replace(":", "-")
    raw_report = attempt_dir / f"{stem}.bench-report.json"
    stdout_path = attempt_dir / f"{stem}.bench.stdout.log"
    stderr_path = attempt_dir / f"{stem}.bench.stderr.log"
    sampler: dict[str, Any] | None = None
    process: subprocess.Popen[Any] | None = None
    try:
        sampler = start_cell_sampler(
            root,
            attempt_dir,
            session,
            config,
            cell,
            cuda_preflight,
        )
        argv = bench_argv(binary, raw_report, config, inputs, cell)
        environment = collector_support.sanitized_environment()
        started_at = now_iso()
        with stdout_path.open("x", encoding="utf-8") as stdout_handle, stderr_path.open("x", encoding="utf-8") as stderr_handle:
            stdout_handle.write("[r2-collector] benchmark client stdout follows\n")
            stderr_handle.write("[r2-collector] benchmark client stderr follows\n")
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
                returncode, group_gone = collector_support.terminate_process_group(process, 2.0)
            except subprocess.TimeoutExpired:
                _, group_gone = collector_support.terminate_process_group(process, 5.0)
                returncode = 124
        finished_at = now_iso()
        collector_support.finish_resource_sampler(sampler)
        require(group_gone, f"benchmark cell {identifier} process group survived cleanup")
        process = None
        collector_support.ensure_nonempty_log(stdout_path, "benchmark stdout")
        collector_support.ensure_nonempty_log(stderr_path, "benchmark stderr")
        require(returncode == 0, f"benchmark cell {identifier} failed with returncode {returncode}")
        require(raw_report.is_file() and raw_report.stat().st_size > 0, f"benchmark cell did not write report: {raw_report}")
        report = read_json(raw_report)
        validate_bench_report(report, config, cell)
        request_sidecar = write_bench_request_sidecar(
            root,
            attempt_dir / f"{stem}.bench-request-evidence.json",
            raw_report,
            report,
            cell,
        )
        record = {
            "sequence": cell["sequence"],
            "cell_id": identifier,
            "dataset": cell["dataset"],
            "concurrency": cell["concurrency"],
            "num_prompts": cell["num_prompts"],
            "n_repeats": 3,
            "warmup_requests": WARMUP_REQUESTS,
            "formal_matrix_cell": cell.get("formal_matrix_cell", True),
            "session_id": session["session_id"],
            "server_pid": session["pid"],
            "candidate_binary_sha256": file_sha256(binary),
            "bench_argv": argv,
            "bench_argv_sha256": canonical_json_sha256(argv),
            "environment": environment,
            "environment_sha256": canonical_json_sha256(environment),
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_sec": duration_seconds(started_at, finished_at),
            "returncode": returncode,
            "stdout": artifact_ref(root, stdout_path, kind="bench-stdout"),
            "stderr": artifact_ref(root, stderr_path, kind="bench-stderr"),
            "raw_report": artifact_ref(root, raw_report, kind="raw-bench-report"),
            "raw_request_evidence": request_sidecar,
        }
        if cell["dataset"] == "run-parity":
            record["serve_c1_parity_metrics"] = parity_metrics(report)
        return record, sampler
    finally:
        if process is not None:
            collector_support.cleanup_process_group_noexcept(process, 5.0)
        if sampler is not None and sampler.get("finished") is not True:
            try:
                sampler["stop_file"].write_text("stop\n", encoding="utf-8")
                collector_support.finish_resource_sampler(sampler, bracket_after_measurement=False)
            except BaseException:
                collector_support.cleanup_process_group_noexcept(sampler["process"], 5.0)
                collector_support.close_sampler_handles(sampler)
                sampler["finished"] = True
        collector_support.ensure_nonempty_log(stdout_path, "benchmark stdout")
        collector_support.ensure_nonempty_log(stderr_path, "benchmark stderr")


def cell_resource_evidence(
    root: Path,
    session: dict[str, Any],
    record: dict[str, Any],
    sampler: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    observations: Path = sampler["observations"]
    summary = collector_support.resource_sampler.derive_summary(
        observations,
        session_id=session["session_id"],
        cell_id=record["cell_id"],
        backend=config["backend"],
        hardware_id=config["hardware"]["id"],
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
        runtime_log_evidence_path=Path(
            session.get("runtime_log_evidence_path", session["runtime_log_origin_path"])
        ),
    )
    interval_path = observations.with_name(
        observations.name.replace(".resource-observations.jsonl", ".active-intervals.json")
    )
    interval_rows: list[dict[str, Any]] = []
    raw_rows = [json.loads(line) for line in observations.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = [row for row in raw_rows if row.get("record_type") == "sample"]
    measurement_start = collector_support.parse_timestamp(record["started_at"])
    measurement_finish = collector_support.parse_timestamp(record["finished_at"])
    for left, right in zip(samples, samples[1:]):
        left_at = collector_support.parse_timestamp(left["sampled_at"])
        right_at = collector_support.parse_timestamp(right["sampled_at"])
        clipped_start = max(left_at, measurement_start)
        clipped_finish = min(right_at, measurement_finish)
        duration_ms = (clipped_finish - clipped_start).total_seconds() * 1000.0
        if duration_ms <= 0:
            continue
        errors = [*left.get("active_probe_errors", []), *right.get("active_probe_errors", [])]
        eligible = (
            left.get("process_alive") is True
            and right.get("process_alive") is True
            and not errors
            and isinstance(left.get("active_requests"), int)
            and isinstance(right.get("active_requests"), int)
        )
        interval_rows.append(
            {
                "sequence": len(interval_rows) + 1,
                "started_at": clipped_start.isoformat().replace("+00:00", "Z"),
                "finished_at": clipped_finish.isoformat().replace("+00:00", "Z"),
                "duration_ms": duration_ms,
                "eligible": eligible,
                "active_requests_conservative": (
                    min(left["active_requests"], right["active_requests"])
                    if eligible
                    else None
                ),
                "left_sample_sequence": left.get("sequence"),
                "right_sample_sequence": right.get("sequence"),
                "probe_errors": errors,
            }
        )
    eligible_ms = 0.0
    total_interval_duration_ms = 0.0
    for row in interval_rows:
        total_interval_duration_ms += row["duration_ms"]
        if row["eligible"]:
            eligible_ms += row["duration_ms"]
    require(interval_rows and eligible_ms > 0, f"{record['cell_id']} has no eligible active intervals")
    interval_document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_r2_active_interval_sidecar",
        "source_observations": artifact_ref(root, observations, kind="resource-observations"),
        "cell_id": record["cell_id"],
        "measurement_started_at": record["started_at"],
        "measurement_finished_at": record["finished_at"],
        "definition": "adjacent 250ms samples clipped to the measured window; eligible only when both probes are live and error-free; active count is conservative interval minimum",
        "eligible_duration_ms": eligible_ms,
        "total_interval_duration_ms": total_interval_duration_ms,
        "intervals": interval_rows,
    }
    atomic_write_json(interval_path, interval_document)
    return {
        "collector": artifact_ref(root, RESOURCE_SAMPLER_PATH, kind="resource-sampler-source")
        if RESOURCE_SAMPLER_PATH.is_relative_to(root)
        else {
            "kind": "resource-sampler-source",
            "path": RESOURCE_SAMPLER_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": file_sha256(RESOURCE_SAMPLER_PATH),
            "size_bytes": RESOURCE_SAMPLER_PATH.stat().st_size,
        },
        "sampler_argv": sampler["argv"],
        "sampler_argv_sha256": canonical_json_sha256(sampler["argv"]),
        "observations": artifact_ref(root, observations, kind="resource-observations"),
        "active_intervals": artifact_ref(root, interval_path, kind="active-interval-sidecar"),
        "summary": summary,
        "cuda_pid_namespace_bridge": cuda_bridge_evidence(root, sampler),
    }


def validate_artifact_ref(root: Path, raw: Any, label: str) -> Path:
    require(isinstance(raw, dict), f"{label} must be an artifact reference")
    relative = raw.get("path")
    digest = raw.get("sha256")
    size = raw.get("size_bytes")
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), f"{label}.path is invalid")
    require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None, f"{label}.sha256 is invalid")
    require(isinstance(size, int) and not isinstance(size, bool) and size > 0, f"{label}.size_bytes is invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise R2CollectorError(f"{label} escapes artifact root") from exc
    require(path.is_file(), f"{label} is missing: {path}")
    require(path.stat().st_size == size, f"{label} size changed")
    require(file_sha256(path) == digest, f"{label} SHA256 changed")
    return path


def wait_for_quiescence(
    root: Path,
    attempt_dir: Path,
    name: str,
    base_url: str,
    environment: dict[str, str],
    timeout_sec: float = 60.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    sequence = 0
    while time.monotonic() < deadline:
        sequence += 1
        evidence, body = collector_support.collect_endpoint_probe(
            root,
            attempt_dir,
            f"{name}-{sequence:03d}",
            f"{base_url}/health",
            environment,
        )
        engine = body.get("engine")
        admission = body.get("admission")
        active = engine.get("active_requests") if isinstance(engine, dict) else None
        queued = engine.get("queued_requests", 0) if isinstance(engine, dict) else None
        admission_queue = admission.get("queue_depth", 0) if isinstance(admission, dict) else 0
        if active == 0 and queued == 0 and admission_queue == 0:
            return {
                "status": "quiescent",
                "attempts": sequence,
                "probe": evidence,
                "active_requests": active,
                "queued_requests": queued,
                "admission_queue_depth": admission_queue,
            }
        time.sleep(0.25)
    raise R2CollectorError(f"server failed to become quiescent after {name}")


def checkpoint_path(lane: Path, sequence: int) -> Path:
    return lane / "cell-checkpoints" / f"cell-{sequence:02d}.json"


def require_chronological_prefix(sequences: list[int], label: str) -> None:
    require(sequences == list(range(1, len(sequences) + 1)), f"{label} is not a chronological prefix")


def frozen_collection_epoch(plan: dict[str, Any]) -> dict[str, str]:
    collector = plan.get("collector")
    require(isinstance(collector, dict), "frozen collector identity is missing")
    epoch = {
        "collector_sha256": collector.get("sha256"),
        "support_sha256": collector.get("support_sha256"),
        "resource_sampler_sha256": collector.get("resource_sampler_sha256"),
    }
    require(
        all(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None for value in epoch.values()),
        "frozen collector epoch identity is invalid",
    )
    return epoch


def require_legacy_finalized_server_shape(
    session: Any, reports: Any, parity: Any, backend: str
) -> None:
    require(
        isinstance(session, dict)
        and session.get("server_process_ordinal") == 1
        and session.get("shutdown_clean") is True,
        "legacy server bundle is not a clean single-process final bundle",
    )
    require(
        isinstance(reports, list)
        and len(reports) == len(expected_cells(backend))
        and [record.get("sequence") for record in reports]
        == list(range(1, len(expected_cells(backend)) + 1))
        and isinstance(parity, dict)
        and parity.get("sequence") == len(expected_cells(backend)) + 1,
        "legacy server bundle is incomplete",
    )


def portable_artifact_argv(argv: Any, label: str) -> list[str]:
    require(isinstance(argv, list) and all(isinstance(value, str) for value in argv), f"{label} argv is invalid")
    marker = f"{os.sep}r2-ferrum{os.sep}"
    return [f"<artifact-root>{marker}{value.split(marker, 1)[1]}" if marker in value else value for value in argv]


def portable_sampler_argv(argv: Any, label: str) -> list[str]:
    normalized = portable_artifact_argv(argv, label)
    sampler_suffix = ("scripts", "release", RESOURCE_SAMPLER_PATH.name)
    if (
        len(normalized) >= 2
        and Path(normalized[0]).is_absolute()
        and re.fullmatch(r"python3(?:\.\d+)?", Path(normalized[0]).name) is not None
        and Path(normalized[1]).is_absolute()
        and tuple(Path(normalized[1]).parts[-len(sampler_suffix) :]) == sampler_suffix
    ):
        normalized[0] = "<python3>"
        normalized[1] = "<source-root>/scripts/release/" + RESOURCE_SAMPLER_PATH.name
    return normalized


def observation_active_envelope(
    observations: Path,
    *,
    session_id: str,
    identifier: str,
    resource_sampler_sha256: str | None = None,
) -> tuple[str, str]:
    """Validate a legacy sampler stream and return its conservative active envelope."""
    try:
        rows = [json.loads(line) for line in observations.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise R2CollectorError(f"cannot read checkpoint observations {observations}: {exc}") from exc
    require(len(rows) >= 4, f"{identifier} checkpoint observations are incomplete")
    header = rows[0]
    footer = rows[-1]
    require(header.get("record_type") == "header", f"{identifier} checkpoint header is missing")
    require(footer.get("record_type") == "footer", f"{identifier} checkpoint footer is missing")
    require(
        sum(row.get("record_type") == "header" for row in rows) == 1
        and sum(row.get("record_type") == "footer" for row in rows) == 1,
        f"{identifier} checkpoint header/footer is not unique",
    )
    require(
        header.get("session_id") == session_id and header.get("cell_id") == identifier,
        f"{identifier} checkpoint observation identity mismatch",
    )
    if resource_sampler_sha256 is not None:
        require(
            header.get("collector_sha256") == resource_sampler_sha256,
            f"{identifier} checkpoint sampler identity mismatch",
        )
    samples = [row for row in rows[1:-1] if row.get("record_type") == "sample"]
    require(len(samples) == len(rows) - 2, f"{identifier} checkpoint has records after/between its footer")
    require(footer.get("exit_reason") == "stop-file", f"{identifier} checkpoint sampler did not stop cleanly")
    require(footer.get("sample_count") == len(samples), f"{identifier} checkpoint footer sample count mismatch")
    require(
        [row.get("sequence") for row in samples] == list(range(len(samples))),
        f"{identifier} checkpoint sample sequence is not contiguous",
    )
    active_indexes = [
        index
        for index, row in enumerate(samples)
        if isinstance(row.get("active_requests"), int) and row["active_requests"] > 0
    ]
    require(active_indexes, f"{identifier} checkpoint has no observed active measurement envelope")
    first_active = active_indexes[0]
    last_active = active_indexes[-1]
    require(
        first_active > 0 and last_active + 1 < len(samples),
        f"{identifier} checkpoint active envelope lacks idle brackets",
    )
    before = samples[first_active - 1]
    after = samples[last_active + 1]
    require(
        before.get("active_requests") == 0 and after.get("active_requests") == 0,
        f"{identifier} checkpoint active envelope is not bracketed by idle samples",
    )
    active = [samples[index] for index in active_indexes]
    require(
        all(row.get("process_alive") is True and not row.get("active_probe_errors") for row in active),
        f"{identifier} checkpoint active envelope is not healthy",
    )
    started_at = before.get("sampled_at")
    finished_at = after.get("sampled_at")
    require(
        isinstance(started_at, str)
        and isinstance(finished_at, str)
        and duration_seconds(started_at, finished_at) > 0,
        f"{identifier} checkpoint active envelope timestamps are invalid",
    )
    return started_at, finished_at


def checkpoint_probe_refs(root: Path, probe: dict[str, Any], label: str) -> dict[str, Any]:
    receipt_origin = probe.get("receipt_origin_path")
    body_origin = probe.get("body_origin_path")
    require(isinstance(receipt_origin, str) and isinstance(body_origin, str), f"{label} probe origins are missing")
    receipt_path = Path(receipt_origin).resolve()
    body_path = Path(body_origin).resolve()
    receipt = read_json(receipt_path)
    body = read_json(body_path)
    require(receipt.get("returncode") == 0 and receipt.get("http_status") == 200, f"{label} probe failed")
    require(
        receipt.get("body_sha256") == file_sha256(body_path)
        and receipt.get("body_size_bytes") == body_path.stat().st_size,
        f"{label} probe body binding mismatch",
    )
    engine = body.get("engine")
    admission = body.get("admission")
    require(isinstance(engine, dict), f"{label} probe engine state is missing")
    require(
        engine.get("active_requests") == 0
        and engine.get("queued_requests", 0) == 0
        and (admission.get("queue_depth", 0) if isinstance(admission, dict) else 0) == 0,
        f"{label} post-cell probe is active",
    )
    return {
        "status": "quiescent",
        "receipt": artifact_ref(root, receipt_path, kind="post-cell-idle-receipt"),
        "body": artifact_ref(root, body_path, kind="post-cell-idle-body"),
    }


def require_completed_attempt_cleanup(attempt_dir: Path, label: str) -> dict[str, Any]:
    failure_path = attempt_dir / "failure.json"
    require(failure_path.is_file(), f"{label} attempt has no finalized cleanup receipt")
    failure = read_json(failure_path)
    require(failure.get("cleanup_process_group_gone") is True, f"{label} attempt process group survived cleanup")
    require(failure.get("cleanup_error") is None, f"{label} attempt cleanup reported an error")
    require(
        failure.get("cleanup_returncode") == 0,
        f"{label} attempt cleanup returncode is not zero",
    )
    return failure


def checkpoint_cuda_bridge(root: Path, bridge: dict[str, Any] | None) -> dict[str, Any] | None:
    if bridge is None:
        return None
    real_binary = Path(bridge["real_binary"]).resolve()
    return {
        "wrapper": artifact_ref(root, Path(bridge["wrapper"]), kind="cuda-pid-namespace-wrapper"),
        "preflight": artifact_ref(root, Path(bridge["preflight"]), kind="cuda-pid-namespace-preflight"),
        "audit": artifact_ref(root, Path(bridge["audit"]), kind="cuda-pid-namespace-audit"),
        "real_nvidia_smi_path": str(real_binary),
        "real_nvidia_smi_sha256": file_sha256(real_binary),
        "server_pid": bridge["server_pid"],
        "server_pgid": bridge["server_pgid"],
    }


def restore_checkpoint_cuda_bridge(root: Path, raw: Any, label: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    require(isinstance(raw, dict), f"{label} CUDA bridge checkpoint is invalid")
    wrapper = validate_artifact_ref(root, raw.get("wrapper"), f"{label}.cuda_bridge.wrapper")
    preflight = validate_artifact_ref(root, raw.get("preflight"), f"{label}.cuda_bridge.preflight")
    preflight_document = read_json(preflight)
    collector_sha256 = preflight_document.get("collector_sha256")
    require(
        isinstance(collector_sha256, str) and SHA256_RE.fullmatch(collector_sha256) is not None,
        f"{label} CUDA bridge collector identity is invalid",
    )
    audit = validate_artifact_ref(root, raw.get("audit"), f"{label}.cuda_bridge.audit")
    real_binary_raw = raw.get("real_nvidia_smi_path")
    require(
        isinstance(real_binary_raw, str) and Path(real_binary_raw).is_absolute(),
        f"{label} CUDA bridge binary path is invalid",
    )
    real_binary_sha256 = raw.get("real_nvidia_smi_sha256")
    require(
        isinstance(real_binary_sha256, str)
        and SHA256_RE.fullmatch(real_binary_sha256) is not None,
        f"{label} CUDA bridge binary identity is invalid",
    )
    require(
        preflight_document.get("contract") == CUDA_PID_NAMESPACE_BRIDGE_CONTRACT
        and preflight_document.get("collector_path") == COLLECTOR_RELATIVE_PATH
        and preflight_document.get("real_nvidia_smi_path") == real_binary_raw
        and preflight_document.get("real_nvidia_smi_sha256") == real_binary_sha256
        and preflight_document.get("compute_apps") == []
        and preflight_document.get("gpu_count") == 1,
        f"{label} CUDA bridge preflight identity is invalid",
    )
    real_binary = Path(real_binary_raw)
    return {
        "wrapper": wrapper,
        "preflight": preflight,
        "audit": audit,
        "real_binary": real_binary,
        "real_binary_sha256": real_binary_sha256,
        "server_pid": raw.get("server_pid"),
        "server_pgid": raw.get("server_pgid"),
        "environment": None,
        "collector_sha256": collector_sha256,
    }


def restore_checkpoint_cuda_bridge_environment(
    bridge: dict[str, Any] | None,
    *,
    raw_bridge: Any,
    sampler_argv: Any,
    observations_ref: Any,
    base_environment: Any,
    label: str,
) -> dict[str, Any] | None:
    if bridge is None:
        return None
    require(isinstance(raw_bridge, dict), f"{label} CUDA bridge checkpoint is invalid")
    require(
        isinstance(base_environment, dict)
        and all(isinstance(key, str) and isinstance(value, str) for key, value in base_environment.items())
        and isinstance(base_environment.get("PATH"), str),
        f"{label} CUDA bridge base environment is invalid",
    )
    argv = portable_artifact_argv(sampler_argv, f"{label} sampler")
    out_indexes = [index for index, value in enumerate(argv) if value == "--out"]
    require(
        len(out_indexes) == 1 and out_indexes[0] + 1 < len(argv),
        f"{label} CUDA bridge sampler output path is missing",
    )
    recorded_out = sampler_argv[out_indexes[0] + 1]
    require(
        isinstance(recorded_out, str) and Path(recorded_out).is_absolute(),
        f"{label} CUDA bridge sampler output path is invalid",
    )
    require(isinstance(observations_ref, dict), f"{label} CUDA bridge observations reference is invalid")
    observations_relative = observations_ref.get("path")
    require(
        isinstance(observations_relative, str)
        and observations_relative
        and not Path(observations_relative).is_absolute(),
        f"{label} CUDA bridge observations path is invalid",
    )
    recorded_out_posix = Path(recorded_out).as_posix()
    observations_suffix = "/" + Path(observations_relative).as_posix()
    require(
        recorded_out_posix.endswith(observations_suffix),
        f"{label} CUDA bridge sampler output origin mismatch",
    )
    recorded_root = recorded_out_posix[: -len(observations_suffix)] or "/"
    wrapper_ref = raw_bridge.get("wrapper")
    require(isinstance(wrapper_ref, dict), f"{label} CUDA bridge wrapper reference is invalid")
    wrapper_relative = wrapper_ref.get("path")
    require(
        isinstance(wrapper_relative, str)
        and wrapper_relative
        and not Path(wrapper_relative).is_absolute(),
        f"{label} CUDA bridge wrapper path is invalid",
    )
    recorded_wrapper = Path(recorded_root) / wrapper_relative
    environment = copy.deepcopy(base_environment)
    environment["PATH"] = (
        f"{recorded_wrapper.parent.as_posix()}{os.pathsep}{environment['PATH']}"
    )
    bridge["environment"] = dict(sorted(environment.items()))
    return bridge


def make_completed_cell_checkpoint(
    root: Path,
    lane: Path,
    fingerprint: str,
    config: dict[str, Any],
    inputs: dict[str, Any],
    attempt_dir: Path,
    session: dict[str, Any],
    record: dict[str, Any],
    sampler: dict[str, Any],
    quiescence: dict[str, Any],
    *,
    provenance: str,
    collection_epoch: dict[str, str] | None = None,
) -> dict[str, Any]:
    sequence = record["sequence"]
    path = checkpoint_path(lane, sequence)
    probe = checkpoint_probe_refs(root, quiescence["probe"], f"cell {sequence}")
    identities = collection_epoch or process_collection_epoch()
    require(
        set(identities) == {"collector_sha256", "support_sha256", "resource_sampler_sha256"}
        and all(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None for value in identities.values()),
        "completed cell collection epoch identity is invalid",
    )
    epoch = {
        key: copy.deepcopy(session[key])
        for key in (
            "session_id",
            "pid",
            "pgid",
            "process_start_marker",
            "process_start_source",
            "base_url",
            "started_at",
            "runtime_log_origin_path",
        )
    }
    if provenance == "legacy-active-envelope":
        epoch["runtime_log_evidence"] = artifact_ref(
            root,
            Path(session.get("runtime_log_evidence_path", session["runtime_log_origin_path"])),
            kind="server-runtime-log",
        )
        epoch["scheduler_trace"] = artifact_ref(
            root, attempt_dir / "server-scheduler-trace.jsonl", kind="scheduler-trace"
        )
        epoch["product_effective_config"] = artifact_ref(
            root, attempt_dir / "server-effective-config.json", kind="product-effective-config"
        )
    document = {
        "schema_version": SCHEMA_VERSION,
        "contract": CELL_CHECKPOINT_CONTRACT,
        "artifact_type": "runtime_vnext_r2_completed_cell_checkpoint",
        "config_fingerprint": fingerprint,
        "normalized_config_sha256": file_sha256(lane / "config.normalized.json"),
        "source_git_sha": config["candidate"]["source_git_sha"],
        "source_tree_sha": config["candidate"]["source_tree_sha"],
        "candidate_binary_sha256": inputs["binary"]["sha256"],
        "collection_epoch": copy.deepcopy(identities),
        "sequence": sequence,
        "cell_id": record["cell_id"],
        "completion_provenance": provenance,
        "attempt_dir": artifact_relative(root, attempt_dir),
        "epoch": epoch,
        "process_receipt": artifact_ref(root, attempt_dir / "server-process-receipt.json", kind="server-process-receipt"),
        "attempt_cleanup": (
            artifact_ref(root, attempt_dir / "failure.json", kind="attempt-cleanup-receipt")
            if (attempt_dir / "failure.json").is_file()
            else None
        ),
        "record": copy.deepcopy(record),
        "sampler": {
            "argv": copy.deepcopy(sampler["argv"]),
            "observations": artifact_ref(root, sampler["observations"], kind="resource-observations"),
            "stdout": artifact_ref(root, sampler["stdout_path"], kind="resource-sampler-stdout"),
            "stderr": artifact_ref(root, sampler["stderr_path"], kind="resource-sampler-stderr"),
            "cuda_pid_namespace_bridge": checkpoint_cuda_bridge(
                root, sampler.get("cuda_pid_namespace_bridge")
            ),
        },
        "post_cell_idle": probe,
    }
    if path.exists():
        require(read_json(path) == document, f"completed cell checkpoint changed: {path}")
    else:
        atomic_write_json(path, document)
    return document


def finalize_attempt_cell_checkpoints(root: Path, lane: Path, attempt_dir: Path) -> None:
    """Bind mutable epoch logs only after the epoch process is gone."""
    for path in sorted((lane / "cell-checkpoints").glob("cell-*.json")):
        checkpoint = read_json(path)
        if checkpoint.get("attempt_dir") != artifact_relative(root, attempt_dir):
            continue
        epoch = checkpoint.get("epoch")
        require(isinstance(epoch, dict), f"checkpoint epoch is missing: {path}")
        if "runtime_log_evidence" in epoch:
            continue
        epoch["runtime_log_evidence"] = artifact_ref(
            root, attempt_dir / "server-runtime.log", kind="server-runtime-log"
        )
        epoch["scheduler_trace"] = artifact_ref(
            root, attempt_dir / "server-scheduler-trace.jsonl", kind="scheduler-trace"
        )
        epoch["product_effective_config"] = artifact_ref(
            root, attempt_dir / "server-effective-config.json", kind="product-effective-config"
        )
        failure_path = attempt_dir / "failure.json"
        checkpoint["attempt_cleanup"] = (
            artifact_ref(root, failure_path, kind="attempt-cleanup-receipt")
            if failure_path.is_file()
            else None
        )
        atomic_write_json(path, checkpoint)


def legacy_probe(attempt_dir: Path, cell: dict[str, Any]) -> dict[str, Any] | None:
    prefix = f"post-{cell['sequence']:02d}-{cell['dataset']}-c{cell['concurrency']}-"
    receipts = sorted(attempt_dir.glob(f"{prefix}*.receipt.json"))
    if len(receipts) != 1:
        return None
    receipt = read_json(receipts[0])
    body = receipts[0].with_name(receipts[0].name.replace(".receipt.json", ".body.json"))
    if not body.is_file():
        return None
    return {"receipt_origin_path": str(receipts[0]), "body_origin_path": str(body), "receipt": receipt}


def legacy_cell_has_completion_envelope(attempt_dir: Path, cell: dict[str, Any]) -> bool:
    stem = cell_id(cell).replace(":", "-")
    required = [
        attempt_dir / f"{stem}.bench-report.json",
        attempt_dir / f"{stem}.bench-request-evidence.json",
        attempt_dir / f"{stem}.bench.stdout.log",
        attempt_dir / f"{stem}.bench.stderr.log",
        attempt_dir / f"{stem}.resource-observations.jsonl",
        attempt_dir / f"{stem}.resource-sampler.stdout.log",
        attempt_dir / f"{stem}.resource-sampler.stderr.log",
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        last = json.loads(required[4].read_text(encoding="utf-8").splitlines()[-1])
    except (OSError, IndexError, json.JSONDecodeError):
        return False
    return last.get("record_type") == "footer" and legacy_probe(attempt_dir, cell) is not None


def recover_legacy_cell_checkpoint(
    root: Path,
    lane: Path,
    fingerprint: str,
    config: dict[str, Any],
    inputs: dict[str, Any],
    attempt_dir: Path,
    cell: dict[str, Any],
) -> dict[str, Any] | None:
    require(
        config["backend"] == "metal",
        "legacy completed-cell import is only defined for the audited Metal active-envelope artifact",
    )
    identifier = cell_id(cell)
    stem = identifier.replace(":", "-")
    paths = {
        "report": attempt_dir / f"{stem}.bench-report.json",
        "request": attempt_dir / f"{stem}.bench-request-evidence.json",
        "stdout": attempt_dir / f"{stem}.bench.stdout.log",
        "stderr": attempt_dir / f"{stem}.bench.stderr.log",
        "observations": attempt_dir / f"{stem}.resource-observations.jsonl",
        "sampler_stdout": attempt_dir / f"{stem}.resource-sampler.stdout.log",
        "sampler_stderr": attempt_dir / f"{stem}.resource-sampler.stderr.log",
    }
    probe = legacy_probe(attempt_dir, cell)
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()) or probe is None:
        return None
    failure = require_completed_attempt_cleanup(attempt_dir, identifier)
    receipt = read_json(attempt_dir / "server-process-receipt.json")
    observation_header = json.loads(paths["observations"].read_text(encoding="utf-8").splitlines()[0])
    session_id = observation_header.get("session_id")
    require(isinstance(session_id, str) and session_id, f"{identifier} legacy session id is missing")
    frozen_collector = read_json(lane / "plan.json").get("collector")
    require(isinstance(frozen_collector, dict), f"{identifier} frozen collector identity is missing")
    legacy_epoch = {
        "collector_sha256": frozen_collector.get("sha256"),
        "support_sha256": frozen_collector.get("support_sha256"),
        "resource_sampler_sha256": frozen_collector.get("resource_sampler_sha256"),
    }
    started_at, finished_at = observation_active_envelope(
        paths["observations"],
        session_id=session_id,
        identifier=identifier,
        resource_sampler_sha256=legacy_epoch["resource_sampler_sha256"],
    )
    report = read_json(paths["report"])
    validate_bench_report(report, config, cell)
    request = read_json(paths["request"])
    require(request.get("cell") == cell, f"{identifier} request sidecar cell mismatch")
    validate_artifact_ref(root, request.get("source_report"), f"{identifier} request sidecar source")
    session = {
        "session_id": session_id,
        "pid": receipt.get("pid"),
        "pgid": receipt.get("pgid"),
        "process_start_marker": receipt.get("process_start_marker"),
        "process_start_source": receipt.get("process_start_source"),
        "base_url": f"http://{config['server']['host']}:{config['server']['port']}",
        "started_at": receipt.get("captured_at"),
        "finished_at": failure.get("failed_at"),
        "runtime_log_origin_path": observation_header.get("runtime_log_path"),
        "runtime_log_evidence_path": str(attempt_dir / "server-runtime.log"),
    }
    require(
        isinstance(session["pid"], int)
        and isinstance(session["pgid"], int)
        and session["pid"] == session["pgid"],
        f"{identifier} legacy process receipt is invalid",
    )
    bench = bench_argv(inputs["binary_path"], paths["report"], config, inputs, cell)
    record = {
        "sequence": cell["sequence"],
        "cell_id": identifier,
        "dataset": cell["dataset"],
        "concurrency": cell["concurrency"],
        "num_prompts": cell["num_prompts"],
        "n_repeats": 3,
        "warmup_requests": WARMUP_REQUESTS,
        "formal_matrix_cell": cell.get("formal_matrix_cell", True),
        "session_id": session_id,
        "server_pid": session["pid"],
        "candidate_binary_sha256": inputs["binary"]["sha256"],
        "bench_argv": bench,
        "bench_argv_sha256": canonical_json_sha256(bench),
        "environment": copy.deepcopy(config["candidate"]["env"]),
        "environment_sha256": canonical_json_sha256(config["candidate"]["env"]),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": duration_seconds(started_at, finished_at),
        "returncode": 0,
        "measurement_window_source": "legacy-active-envelope",
        "measurement_window_method": "last idle sample before first active through first idle sample after last active",
        "stdout": artifact_ref(root, paths["stdout"], kind="bench-stdout"),
        "stderr": artifact_ref(root, paths["stderr"], kind="bench-stderr"),
        "raw_report": artifact_ref(root, paths["report"], kind="raw-bench-report"),
        "raw_request_evidence": artifact_ref(root, paths["request"], kind="bench-request-evidence-sidecar"),
    }
    sampler = {
        "argv": cell_sampler_argv(attempt_dir, session, config, cell),
        "observations": paths["observations"],
        "stdout_path": paths["sampler_stdout"],
        "stderr_path": paths["sampler_stderr"],
        "cuda_pid_namespace_bridge": None,
    }
    quiescence = {"probe": probe}
    was_present = checkpoint_path(lane, cell["sequence"]).exists()
    checkpoint = make_completed_cell_checkpoint(
        root,
        lane,
        fingerprint,
        config,
        inputs,
        attempt_dir,
        session,
        record,
        sampler,
        quiescence,
        provenance="legacy-active-envelope",
        collection_epoch=legacy_epoch,
    )
    if not was_present:
        checkpoint_file = checkpoint_path(lane, cell["sequence"])
        collector_support.append_jsonl(
            lane / "command-log.jsonl",
            {
                "event": "legacy-completed-cell-import",
                "imported_at": now_iso(),
                "cell_id": identifier,
                "sequence": cell["sequence"],
                "completion_provenance": "legacy-active-envelope",
                "checkpoint": artifact_relative(root, checkpoint_file),
                "checkpoint_sha256": file_sha256(checkpoint_file),
                "source_attempt": artifact_relative(root, attempt_dir),
                "collection_epoch": legacy_epoch,
            },
        )
    return checkpoint


def validate_completed_cell_checkpoint(
    root: Path,
    lane: Path,
    fingerprint: str,
    config: dict[str, Any],
    inputs: dict[str, Any],
    checkpoint: dict[str, Any],
    cell: dict[str, Any],
    *,
    finalized_session: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    identifier = cell_id(cell)
    require(checkpoint.get("contract") == CELL_CHECKPOINT_CONTRACT, f"{identifier} checkpoint contract mismatch")
    require(checkpoint.get("config_fingerprint") == fingerprint, f"{identifier} checkpoint fingerprint mismatch")
    require(
        checkpoint.get("normalized_config_sha256") == file_sha256(lane / "config.normalized.json")
        and checkpoint.get("source_git_sha") == config["candidate"]["source_git_sha"]
        and checkpoint.get("source_tree_sha") == config["candidate"]["source_tree_sha"]
        and checkpoint.get("candidate_binary_sha256") == inputs["binary"]["sha256"],
        f"{identifier} checkpoint source/config/binary binding mismatch",
    )
    require(
        checkpoint.get("sequence") == cell["sequence"] and checkpoint.get("cell_id") == identifier,
        f"{identifier} checkpoint order mismatch",
    )
    collection_epoch = checkpoint.get("collection_epoch")
    require(
        isinstance(collection_epoch, dict)
        and set(collection_epoch) == {"collector_sha256", "support_sha256", "resource_sampler_sha256"}
        and all(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None for value in collection_epoch.values()),
        f"{identifier} checkpoint collection epoch identity is invalid",
    )
    provenance = checkpoint.get("completion_provenance")
    if provenance == "legacy-active-envelope":
        frozen_collector = read_json(lane / "plan.json").get("collector")
        expected_epoch = {
            "collector_sha256": frozen_collector.get("sha256") if isinstance(frozen_collector, dict) else None,
            "support_sha256": frozen_collector.get("support_sha256") if isinstance(frozen_collector, dict) else None,
            "resource_sampler_sha256": (
                frozen_collector.get("resource_sampler_sha256") if isinstance(frozen_collector, dict) else None
            ),
        }
        require(collection_epoch == expected_epoch, f"{identifier} checkpoint collection epoch binding mismatch")
    else:
        require(provenance == "native-completed-cell", f"{identifier} checkpoint provenance is invalid")
        require_reviewed_native_collection_epoch(collection_epoch, f"{identifier} checkpoint collection epoch")
    attempt_relative = checkpoint.get("attempt_dir")
    require(isinstance(attempt_relative, str) and not Path(attempt_relative).is_absolute(), f"{identifier} checkpoint attempt path is invalid")
    attempt_dir = (root / attempt_relative).resolve()
    cleanup_ref = checkpoint.get("attempt_cleanup")
    if cleanup_ref is not None:
        cleanup_path = validate_artifact_ref(root, cleanup_ref, f"{identifier}.attempt_cleanup")
        require(cleanup_path == attempt_dir / "failure.json", f"{identifier} cleanup receipt path mismatch")
        failure = require_completed_attempt_cleanup(attempt_dir, identifier)
    else:
        require(not (attempt_dir / "failure.json").exists(), f"{identifier} cleanup receipt is unbound")
        failure = None
    process_receipt_path = validate_artifact_ref(
        root, checkpoint.get("process_receipt"), f"{identifier}.process_receipt"
    )
    process_receipt = read_json(process_receipt_path)
    record = checkpoint.get("record")
    epoch = checkpoint.get("epoch")
    sampler_document = checkpoint.get("sampler")
    require(isinstance(record, dict) and isinstance(epoch, dict) and isinstance(sampler_document, dict), f"{identifier} checkpoint payload is incomplete")
    expected_server_argv, _, _ = server_argv(inputs["binary_path"], attempt_dir, config)
    received_server_argv = process_receipt.get("argv")
    require(
        portable_artifact_argv(received_server_argv, f"{identifier} server")
        == portable_artifact_argv(expected_server_argv, f"{identifier} expected server")
        and process_receipt.get("argv_sha256") == canonical_json_sha256(received_server_argv)
        and process_receipt.get("environment") == config["candidate"]["env"]
        and process_receipt.get("environment_sha256") == canonical_json_sha256(config["candidate"]["env"]),
        f"{identifier} checkpoint server process binding mismatch",
    )
    require(
        epoch.get("pid") == process_receipt.get("pid")
        and epoch.get("pgid") == process_receipt.get("pgid")
        and epoch.get("process_start_marker") == process_receipt.get("process_start_marker")
        and epoch.get("pid") == epoch.get("pgid"),
        f"{identifier} checkpoint epoch/process receipt mismatch",
    )
    require(record.get("sequence") == cell["sequence"] and record.get("cell_id") == identifier, f"{identifier} checkpoint record mismatch")
    report_path = validate_artifact_ref(root, record.get("raw_report"), f"{identifier}.raw_report")
    validate_bench_report(read_json(report_path), config, cell)
    expected_bench = bench_argv(inputs["binary_path"], report_path, config, inputs, cell)
    received_bench = record.get("bench_argv")
    require(
        portable_artifact_argv(received_bench, f"{identifier} benchmark")
        == portable_artifact_argv(expected_bench, f"{identifier} expected benchmark")
        and record.get("bench_argv_sha256") == canonical_json_sha256(received_bench)
        and record.get("candidate_binary_sha256") == inputs["binary"]["sha256"],
        f"{identifier} checkpoint benchmark binding mismatch",
    )
    record_environment = record.get("environment")
    require(
        isinstance(record_environment, dict)
        and all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in record_environment.items()
        )
        and record.get("environment_sha256")
        == canonical_json_sha256(record_environment),
        f"{identifier} checkpoint benchmark environment identity mismatch",
    )
    for key in ("stdout", "stderr"):
        validate_artifact_ref(root, record.get(key), f"{identifier}.{key}")
    request_path = validate_artifact_ref(
        root, record.get("raw_request_evidence"), f"{identifier}.raw_request_evidence"
    )
    request = read_json(request_path)
    require(request.get("cell") == cell, f"{identifier} checkpoint request evidence cell mismatch")
    require(
        validate_artifact_ref(root, request.get("source_report"), f"{identifier}.request.source_report")
        == report_path,
        f"{identifier} checkpoint request evidence report mismatch",
    )
    observations = validate_artifact_ref(root, sampler_document.get("observations"), f"{identifier}.observations")
    require(
        portable_sampler_argv(sampler_document.get("argv"), f"{identifier} sampler")
        == portable_sampler_argv(
            cell_sampler_argv(attempt_dir, epoch, config, cell),
            f"{identifier} expected sampler",
        ),
        f"{identifier} checkpoint sampler argv mismatch",
    )
    for key in ("stdout", "stderr"):
        validate_artifact_ref(root, sampler_document.get(key), f"{identifier}.sampler.{key}")
    observation_active_envelope(
        observations,
        session_id=epoch.get("session_id"),
        identifier=identifier,
        resource_sampler_sha256=collection_epoch["resource_sampler_sha256"],
    )
    post = checkpoint.get("post_cell_idle")
    require(isinstance(post, dict) and post.get("status") == "quiescent", f"{identifier} checkpoint post-cell idle proof is missing")
    receipt_path = validate_artifact_ref(root, post.get("receipt"), f"{identifier}.post_idle.receipt")
    body_path = validate_artifact_ref(root, post.get("body"), f"{identifier}.post_idle.body")
    checkpoint_probe_refs(
        root,
        {"receipt_origin_path": str(receipt_path), "body_origin_path": str(body_path)},
        identifier,
    )
    epoch_session = copy.deepcopy(epoch)
    epoch_session["runtime_log_evidence_path"] = str(
        validate_artifact_ref(
            root,
            epoch.get("runtime_log_evidence"),
            f"{identifier}.epoch.runtime_log_evidence",
        )
    )
    validate_artifact_ref(root, epoch.get("scheduler_trace"), f"{identifier}.epoch.scheduler_trace")
    product_path = validate_artifact_ref(
        root, epoch.get("product_effective_config"), f"{identifier}.epoch.product_effective_config"
    )
    product = read_json(product_path)
    require(
        isinstance(product.get("admission"), dict)
        and product["admission"].get("effective_max_concurrent") == config["typed_active_cap"],
        f"{identifier} checkpoint effective active cap mismatch",
    )
    if failure is not None:
        epoch_session["finished_at"] = failure["failed_at"]
    else:
        require(
            isinstance(finalized_session, dict)
            and finalized_session.get("session_id") == epoch.get("session_id")
            and finalized_session.get("shutdown_clean") is True
            and isinstance(finalized_session.get("returncode"), int),
            f"{identifier} checkpoint has neither failed-attempt cleanup nor a clean finalized epoch",
        )
        epoch_session["finished_at"] = finalized_session["finished_at"]
    raw_cuda_bridge = sampler_document.get("cuda_pid_namespace_bridge")
    restored_cuda_bridge = restore_checkpoint_cuda_bridge(
        root, raw_cuda_bridge, identifier
    )
    restored_cuda_bridge = restore_checkpoint_cuda_bridge_environment(
        restored_cuda_bridge,
        raw_bridge=raw_cuda_bridge,
        sampler_argv=sampler_document.get("argv"),
        observations_ref=sampler_document.get("observations"),
        base_environment=record_environment,
        label=identifier,
    )
    sampler = {
        "argv": copy.deepcopy(sampler_document.get("argv")),
        "observations": observations,
        "cuda_pid_namespace_bridge": restored_cuda_bridge,
    }
    record = copy.deepcopy(record)
    record["resources"] = cell_resource_evidence(root, epoch_session, record, sampler, config)
    return record, checkpoint, epoch_session


def load_completed_cell_prefix(
    root: Path,
    lane: Path,
    fingerprint: str,
    config: dict[str, Any],
    inputs: dict[str, Any],
    *,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cells = [*expected_cells(config["backend"]), run_parity_cell(config["backend"])]
    checkpoint_dir = lane / "cell-checkpoints"
    existing_paths = sorted(checkpoint_dir.glob("cell-*.json")) if checkpoint_dir.is_dir() else []
    existing_sequences = []
    for path in existing_paths:
        match = re.fullmatch(r"cell-(\d+)\.json", path.name)
        require(match is not None, f"invalid completed-cell checkpoint name: {path}")
        existing_sequences.append(int(match.group(1)))
    require_chronological_prefix(existing_sequences, "completed-cell resume state")
    if resume and not existing_paths:
        legacy_completed: list[tuple[Path, dict[str, Any]]] = []
        for attempt_dir in sorted((lane / "attempts").glob("server-*")) if (lane / "attempts").is_dir() else []:
            for cell in cells:
                if legacy_cell_has_completion_envelope(attempt_dir, cell):
                    legacy_completed.append((attempt_dir, cell))
        sequences = sorted({cell["sequence"] for _, cell in legacy_completed})
        require_chronological_prefix(sequences, "legacy completed-cell state")
        for expected_sequence in sequences:
            matches = [(attempt, cell) for attempt, cell in legacy_completed if cell["sequence"] == expected_sequence]
            require(len(matches) == 1, f"legacy cell {expected_sequence} has ambiguous completed attempts")
            attempt, cell = matches[0]
            checkpoint = recover_legacy_cell_checkpoint(root, lane, fingerprint, config, inputs, attempt, cell)
            require(checkpoint is not None, f"legacy cell {expected_sequence} is not safely recoverable")
        existing_paths = [checkpoint_path(lane, sequence) for sequence in sequences]
    require(not existing_paths or resume, "completed cell checkpoints require --resume")
    records: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    epochs: list[dict[str, Any]] = []
    for path, cell in zip(existing_paths, cells):
        record, checkpoint, epoch = validate_completed_cell_checkpoint(
            root, lane, fingerprint, config, inputs, read_json(path), cell
        )
        records.append(record)
        checkpoints.append(checkpoint)
        epochs.append(epoch)
    return records, checkpoints, epochs


def recovered_session_epoch_rows(
    root: Path,
    lane: Path,
    checkpoints: list[dict[str, Any]],
    epochs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for checkpoint, epoch in zip(checkpoints, epochs):
        session_id = epoch["session_id"]
        row = grouped.setdefault(
            session_id,
            {
                "kind": "recovered-completed-cell",
                "session_id": session_id,
                "collection_epoch": copy.deepcopy(checkpoint["collection_epoch"]),
                "completed_cell_sequences": [],
                "checkpoints": [],
            },
        )
        require(
            row["collection_epoch"] == checkpoint["collection_epoch"],
            f"recovered session {session_id} changes collection identity",
        )
        row["completed_cell_sequences"].append(checkpoint["sequence"])
        row["checkpoints"].append(
            artifact_ref(
                root,
                checkpoint_path(lane, checkpoint["sequence"]),
                kind="completed-cell-checkpoint",
            )
        )
    return list(grouped.values())


def finalize_recovered_only_server_bundle(
    root: Path,
    lane: Path,
    fingerprint: str,
    config: dict[str, Any],
    inputs: dict[str, Any],
    records: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    epochs: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped_epochs = recovered_session_epoch_rows(root, lane, checkpoints, epochs)
    require(grouped_epochs, "recovered-only server bundle has no epochs")
    last_epoch = epochs[-1]
    last_checkpoint = checkpoints[-1]
    attempt_dir = (root / last_checkpoint["attempt_dir"]).resolve()
    cleanup = require_completed_attempt_cleanup(attempt_dir, "recovered-only server epoch")
    receipt = read_json(
        validate_artifact_ref(
            root, last_checkpoint["process_receipt"], "recovered-only server process receipt"
        )
    )
    session = {
        **copy.deepcopy(last_epoch),
        "server_process_ordinal": len(grouped_epochs),
        "hardware": copy.deepcopy(config["hardware"]),
        "candidate_binary_sha256": inputs["binary"]["sha256"],
        "source_git_sha": config["candidate"]["source_git_sha"],
        "source_tree_sha": config["candidate"]["source_tree_sha"],
        "dirty_status": copy.deepcopy(config["candidate"]["dirty_status"]),
        "profile_detail": "off",
        "environment": copy.deepcopy(receipt["environment"]),
        "environment_sha256": receipt["environment_sha256"],
        "server_argv": copy.deepcopy(receipt["argv"]),
        "server_argv_sha256": receipt["argv_sha256"],
        "runtime_log": copy.deepcopy(last_epoch["runtime_log_evidence"]),
        "scheduler_trace": copy.deepcopy(last_epoch["scheduler_trace"]),
        "product_effective_config": copy.deepcopy(last_epoch["product_effective_config"]),
        "finished_at": cleanup["failed_at"],
        "duration_sec": duration_seconds(last_epoch["started_at"], cleanup["failed_at"]),
        "formal_measurement_started_at": records[0]["started_at"],
        "formal_measurement_finished_at": records[len(expected_cells(config["backend"])) - 1]["finished_at"],
        "parity_measurement_started_at": records[-1]["started_at"],
        "parity_measurement_finished_at": records[-1]["finished_at"],
        "returncode": cleanup["cleanup_returncode"],
        "shutdown_clean": True,
        "shutdown_provenance": "interrupted-after-all-cells-with-process-group-gone",
        "cell_quiescence": [
            {"status": "checkpointed-quiescent", "cell_id": record["cell_id"]}
            for record in records
        ],
    }
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "config_fingerprint": fingerprint,
        "session": session,
        "session_epochs": grouped_epochs,
        "completed_cell_checkpoints": [
            artifact_ref(
                root,
                checkpoint_path(lane, checkpoint["sequence"]),
                kind="completed-cell-checkpoint",
            )
            for checkpoint in checkpoints
        ],
        "formal_reports": records[: len(expected_cells(config["backend"]))],
        "run_serve_parity_report": records[len(expected_cells(config["backend"]))],
    }
    path = lane / "server-session.json"
    atomic_write_json(path, bundle)
    validate_server_bundle(root, bundle, fingerprint, config, inputs)
    return bundle


def validate_server_bundle(
    root: Path,
    bundle: dict[str, Any],
    fingerprint: str,
    config: dict[str, Any],
    inputs: dict[str, Any],
) -> None:
    require(bundle.get("schema_version") == SCHEMA_VERSION, "server bundle schema mismatch")
    require(bundle.get("config_fingerprint") == fingerprint, "server bundle fingerprint mismatch")
    session = bundle.get("session")
    epochs = bundle.get("session_epochs")
    checkpoints = bundle.get("completed_cell_checkpoints")
    reports = bundle.get("formal_reports")
    parity = bundle.get("run_serve_parity_report")
    require(isinstance(session, dict), "server bundle session is missing")
    legacy_single_epoch = epochs is None and checkpoints is None
    legacy_epoch = frozen_collection_epoch(read_json(lane_dir(root, config) / "plan.json")) if legacy_single_epoch else None
    if legacy_single_epoch:
        require_legacy_finalized_server_shape(session, reports, parity, config["backend"])
        epochs = [{"session_id": session.get("session_id")}]
        checkpoints = []
    else:
        require(isinstance(epochs, list) and epochs, "server bundle session epochs are missing")
        require(
            isinstance(checkpoints, list) and len(checkpoints) == len(expected_cells(config["backend"])) + 1,
            "server bundle completed-cell checkpoint set is incomplete",
        )
    checkpoint_documents: list[dict[str, Any]] = []
    for index, ref in enumerate(checkpoints, start=1):
        checkpoint_file = validate_artifact_ref(root, ref, f"completed_cell_checkpoints[{index}]")
        checkpoint_document = read_json(checkpoint_file)
        require(checkpoint_document.get("sequence") == index, "server bundle checkpoint order mismatch")
        checkpoint_documents.append(checkpoint_document)
    epoch_session_ids = [epoch.get("session_id") for epoch in epochs if isinstance(epoch, dict)]
    require(
        len(epoch_session_ids) == len(epochs)
        and all(isinstance(value, str) and value for value in epoch_session_ids),
        "server bundle epoch session identity is invalid",
    )
    actual_process_count = len(set(epoch_session_ids))
    require(
        session.get("server_process_ordinal") == actual_process_count,
        "server bundle process ordinal does not match explicit epoch count",
    )
    require(isinstance(reports, list) and len(reports) == len(expected_cells(config["backend"])), "server bundle formal cell count mismatch")
    require(isinstance(parity, dict), "server bundle lacks run/serve parity probe")
    if not legacy_single_epoch:
        all_records = [*reports, parity]
        all_cells = [*expected_cells(config["backend"]), run_parity_cell(config["backend"])]
        for checkpoint_document, cell, expected_record in zip(checkpoint_documents, all_cells, all_records):
            checkpoint_record, _, _ = validate_completed_cell_checkpoint(
                root,
                lane_dir(root, config),
                fingerprint,
                config,
                inputs,
                checkpoint_document,
                cell,
                finalized_session=session,
            )
            require(checkpoint_record == expected_record, f"{cell_id(cell)} finalized checkpoint record mismatch")
    expected = list(expected_cells(config["backend"]))
    for index, (record, cell) in enumerate(zip(reports, expected), start=1):
        require(record.get("sequence") == index and record.get("cell_id") == cell_id(cell), "server bundle cell order mismatch")
        require(record.get("session_id") in epoch_session_ids, "server bundle formal report has an unknown epoch")
        report_path = validate_artifact_ref(root, record.get("raw_report"), f"formal_reports[{index}].raw_report")
        validate_bench_report(read_json(report_path), config, cell)
        for key in ("stdout", "stderr"):
            validate_artifact_ref(root, record.get(key), f"formal_reports[{index}].{key}")
        validate_artifact_ref(root, record.get("raw_request_evidence"), f"formal_reports[{index}].raw_request_evidence")
        resources = record.get("resources")
        require(isinstance(resources, dict), f"formal_reports[{index}].resources is missing")
        validate_artifact_ref(root, resources.get("observations"), f"formal_reports[{index}].resources.observations")
        validate_artifact_ref(root, resources.get("active_intervals"), f"formal_reports[{index}].resources.active_intervals")
        validate_cuda_bridge_evidence(
            root,
            resources,
            backend=config["backend"],
            label=f"formal_reports[{index}].resources",
            expected_collector_sha256=(
                (legacy_epoch or {}).get("collector_sha256")
                if legacy_single_epoch
                else checkpoint_documents[index - 1]["collection_epoch"]["collector_sha256"]
            ),
        )
    parity_path = validate_artifact_ref(root, parity.get("raw_report"), "run_serve_parity_report.raw_report")
    require(parity.get("session_id") in epoch_session_ids, "run/serve parity report has an unknown epoch")
    validate_bench_report(read_json(parity_path), config, run_parity_cell(config["backend"]))
    for key in ("stdout", "stderr"):
        validate_artifact_ref(root, parity.get(key), f"run_serve_parity_report.{key}")
    validate_artifact_ref(root, parity.get("raw_request_evidence"), "run_serve_parity_report.raw_request_evidence")
    validate_artifact_ref(root, parity.get("resources", {}).get("observations"), "run_serve_parity_report.resources.observations")
    validate_artifact_ref(root, parity.get("resources", {}).get("active_intervals"), "run_serve_parity_report.resources.active_intervals")
    validate_cuda_bridge_evidence(
        root,
        parity.get("resources", {}),
        backend=config["backend"],
        label="run_serve_parity_report.resources",
        expected_collector_sha256=(
            (legacy_epoch or {}).get("collector_sha256")
            if legacy_single_epoch
            else checkpoint_documents[-1]["collection_epoch"]["collector_sha256"]
        ),
    )
    for key in ("runtime_log", "scheduler_trace", "product_effective_config"):
        validate_artifact_ref(root, session.get(key), f"session.{key}")
    require(session.get("shutdown_clean") is True, "server bundle is not a clean-shutdown session")


def collect_server_session(
    root: Path,
    lane: Path,
    fingerprint: str,
    config: dict[str, Any],
    inputs: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    bundle_path = lane / "server-session.json"
    if bundle_path.exists():
        require(resume, f"server session already exists; pass --resume: {bundle_path}")
        bundle = read_json(bundle_path)
        validate_server_bundle(root, bundle, fingerprint, config, inputs)
        return bundle

    recovered_records, recovered_checkpoints, recovered_epochs = load_completed_cell_prefix(
        root,
        lane,
        fingerprint,
        config,
        inputs,
        resume=resume,
    )
    all_cells = [*expected_cells(config["backend"]), run_parity_cell(config["backend"])]
    if len(recovered_records) == len(all_cells):
        return finalize_recovered_only_server_bundle(
            root,
            lane,
            fingerprint,
            config,
            inputs,
            recovered_records,
            recovered_checkpoints,
            recovered_epochs,
        )

    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt_dir = lane / "attempts" / f"server-{attempt_id}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    runtime_log = attempt_dir / "server-runtime.log"
    argv, product_config, scheduler_trace = server_argv(inputs["binary_path"], attempt_dir, config)
    environment = dict(config["candidate"]["env"])
    base_url = f"http://{config['server']['host']}:{config['server']['port']}"
    started_at = now_iso()
    runtime_handle = runtime_log.open("x", encoding="utf-8")
    runtime_handle.write(f"[r2-collector] server argv={json.dumps(argv)}\n")
    runtime_handle.flush()
    process: subprocess.Popen[Any] | None = None
    failure: BaseException | None = None
    cuda_preflight: dict[str, Any] | None = None
    completed_cleanup_returncode: int | None = None
    completed_cleanup_gone = False
    try:
        if config["backend"] == "cuda":
            cuda_preflight = capture_cuda_bridge_preflight(attempt_dir)
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
        marker, marker_source = collector_support.process_identity(pid)
        receipt = collector_support.write_process_receipt(
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
            "session_id": f"r2-{config['model_key']}-{config['backend']}-{attempt_id}",
            "server_process_ordinal": len({epoch["session_id"] for epoch in recovered_epochs}) + 1,
            "hardware": copy.deepcopy(config["hardware"]),
            "candidate_binary_sha256": inputs["binary"]["sha256"],
            "source_git_sha": config["candidate"]["source_git_sha"],
            "source_tree_sha": config["candidate"]["source_tree_sha"],
            "dirty_status": copy.deepcopy(config["candidate"]["dirty_status"]),
            "profile_detail": "off",
            "pid": pid,
            "pgid": pgid,
            "process_start_marker": marker,
            "process_start_source": marker_source,
            "process_receipt": receipt,
            "server_argv": argv,
            "server_argv_sha256": canonical_json_sha256(argv),
            "environment": environment,
            "environment_sha256": canonical_json_sha256(environment),
            "base_url": base_url,
            "started_at": started_at,
            "runtime_log_origin_path": str(runtime_log),
        }
        collector_support.wait_for_server(process, f"{base_url}/v1/models", config["server"]["ready_timeout_sec"])
        ready_probe, ready_body = collector_support.collect_endpoint_probe(
            root, attempt_dir, "ready-probe", f"{base_url}/v1/models", environment
        )
        observed_models = [row.get("id") for row in ready_body.get("data", []) if isinstance(row, dict)]
        require(observed_models == [config["request_model"]], f"server exposed unexpected model ids: {observed_models}")
        session["ready_at"] = now_iso()
        session["ready_probe"] = ready_probe
        session["pre_measurement_quiescence"] = wait_for_quiescence(
            root, attempt_dir, "pre-measurement", base_url, environment
        )

        records: list[dict[str, Any]] = list(recovered_records)
        samplers: list[dict[str, Any]] = []
        new_records: list[dict[str, Any]] = []
        quiescence: list[dict[str, Any]] = [
            {"status": "checkpointed-quiescent", "cell_id": record["cell_id"]}
            for record in recovered_records
        ]
        checkpoints: list[dict[str, Any]] = list(recovered_checkpoints)
        for cell in all_cells[len(recovered_records) :]:
            record, sampler = run_bench_cell(
                root,
                attempt_dir,
                session,
                inputs["binary_path"],
                config,
                inputs,
                cell,
                cuda_preflight,
            )
            records.append(record)
            new_records.append(record)
            samplers.append(sampler)
            post_idle = wait_for_quiescence(
                root,
                attempt_dir,
                f"post-{cell['sequence']:02d}-{cell['dataset']}-c{cell['concurrency']}",
                base_url,
                environment,
            )
            quiescence.append(post_idle)
            checkpoints.append(
                make_completed_cell_checkpoint(
                    root,
                    lane,
                    fingerprint,
                    config,
                    inputs,
                    attempt_dir,
                    session,
                    record,
                    sampler,
                    post_idle,
                    provenance="native-completed-cell",
                )
            )
        session["formal_measurement_started_at"] = records[0]["started_at"]
        session["formal_measurement_finished_at"] = records[len(expected_cells(config["backend"])) - 1]["finished_at"]
        session["parity_measurement_started_at"] = records[-1]["started_at"]
        session["parity_measurement_finished_at"] = records[-1]["finished_at"]
        session["cell_quiescence"] = quiescence
        session["shutdown_started_at"] = now_iso()
        returncode, group_gone = collector_support.terminate_process_group(
            process, config["server"]["shutdown_timeout_sec"]
        )
        completed_cleanup_returncode = returncode
        completed_cleanup_gone = group_gone
        session["finished_at"] = now_iso()
        session["duration_sec"] = duration_seconds(started_at, session["finished_at"])
        session["returncode"] = returncode
        session["shutdown_clean"] = returncode == 0 and group_gone
        require(session["shutdown_clean"], f"server shutdown failed: returncode={returncode}, group_gone={group_gone}")
        process = None
        runtime_handle.flush()
        os.fsync(runtime_handle.fileno())
        runtime_handle.close()
        runtime_handle = None
        collector_support.ensure_nonempty_log(runtime_log, "server runtime")
        require(product_config.is_file(), "server did not write effective config")
        require(scheduler_trace.is_file() and scheduler_trace.stat().st_size > 0, "server did not write scheduler trace")
        product = read_json(product_config)
        admission = product.get("admission")
        require(
            isinstance(admission, dict) and admission.get("effective_max_concurrent") == config["typed_active_cap"],
            "server effective active cap differs from typed_active_cap",
        )
        session["runtime_log"] = artifact_ref(root, runtime_log, kind="server-runtime-log")
        session["scheduler_trace"] = artifact_ref(root, scheduler_trace, kind="scheduler-trace")
        session["product_effective_config"] = artifact_ref(root, product_config, kind="product-effective-config")
        finalize_attempt_cell_checkpoints(root, lane, attempt_dir)
        checkpoints = [read_json(checkpoint_path(lane, row["sequence"])) for row in records]
        for record, sampler in zip(new_records, samplers):
            record["resources"] = cell_resource_evidence(root, session, record, sampler, config)
        formal_count = len(expected_cells(config["backend"]))
        bundle = {
            "schema_version": SCHEMA_VERSION,
            "contract": CONTRACT,
            "config_fingerprint": fingerprint,
            "session": session,
            "session_epochs": recovered_session_epoch_rows(
                root, lane, recovered_checkpoints, recovered_epochs
            )
            + [
                {
                    "kind": "current-server-session",
                    "session_id": session["session_id"],
                    "collection_epoch": process_collection_epoch(),
                    "completed_cell_sequences": [record["sequence"] for record in new_records],
                }
            ],
            "completed_cell_checkpoints": [
                artifact_ref(
                    root,
                    checkpoint_path(lane, checkpoint["sequence"]),
                    kind="completed-cell-checkpoint",
                )
                for checkpoint in checkpoints
            ],
            "formal_reports": records[:formal_count],
            "run_serve_parity_report": records[formal_count],
        }
        atomic_write_json(bundle_path, bundle)
        collector_support.append_jsonl(
            lane / "command-log.jsonl",
            {
                "event": "server-session-complete",
                "session_id": session["session_id"],
                "started_at": started_at,
                "finished_at": session["finished_at"],
                "bundle": artifact_relative(root, bundle_path),
                "bundle_sha256": file_sha256(bundle_path),
            },
        )
        validate_server_bundle(root, bundle, fingerprint, config, inputs)
        return bundle
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if process is not None:
            cleanup_returncode, cleanup_gone, cleanup_error = collector_support.cleanup_process_group_noexcept(process, 10.0)
        else:
            cleanup_returncode = completed_cleanup_returncode
            cleanup_gone = completed_cleanup_gone or failure is None
            cleanup_error = None
        if runtime_handle is not None:
            try:
                runtime_handle.flush()
                runtime_handle.close()
            except BaseException:
                pass
        collector_support.ensure_nonempty_log(runtime_log, "server runtime")
        if failure is not None:
            atomic_write_json(
                attempt_dir / "failure.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "failed_at": now_iso(),
                    "error_type": type(failure).__name__,
                    "error": str(failure),
                    "cleanup_returncode": cleanup_returncode,
                    "cleanup_process_group_gone": cleanup_gone,
                    "cleanup_error": cleanup_error,
                    "resume_policy": "recover only the validated chronological completed-cell prefix; discard the partial suffix",
                },
            )
            finalize_attempt_cell_checkpoints(root, lane, attempt_dir)


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


def run_exec_barrier_child(release_file: Path, raw_argv: list[str]) -> int:
    argv = list(raw_argv)
    if argv and argv[0] == "--":
        argv.pop(0)
    require(argv and all(isinstance(item, str) and item for item in argv), "exec barrier requires product argv")
    deadline = time.monotonic() + 120.0
    while not release_file.exists():
        if time.monotonic() >= deadline:
            raise R2CollectorError(f"exec barrier release timed out: {release_file}")
        time.sleep(0.01)
    require(release_file.read_text(encoding="utf-8") == "release\n", "exec barrier release artifact is invalid")
    os.execvpe(argv[0], argv, os.environ)
    raise R2CollectorError("exec barrier os.execvpe unexpectedly returned")


def collect_jsonl_stream(
    process: subprocess.Popen[Any],
    stdout_path: Path,
    arrival_path: Path,
    timeout_sec: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, bool]:
    require(process.stdout is not None, "run stdout pipe is missing")
    fd = process.stdout.fileno()
    os.set_blocking(fd, False)
    selector = selectors.DefaultSelector()
    selector.register(fd, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout_sec
    pending = b""
    raw = bytearray()
    events: list[dict[str, Any]] = []
    arrivals: list[dict[str, Any]] = []
    chunk_sequence = 0
    eof = False
    timed_out = False
    try:
        while not eof:
            if time.monotonic() >= deadline:
                timed_out = True
                break
            ready = selector.select(timeout=min(0.25, max(0.0, deadline - time.monotonic())))
            if not ready:
                if process.poll() is not None:
                    try:
                        payload = os.read(fd, 65536)
                    except BlockingIOError:
                        payload = b""
                    if not payload:
                        eof = True
                        continue
                continue
            for _, _ in ready:
                try:
                    payload = os.read(fd, 65536)
                except BlockingIOError:
                    continue
                if not payload:
                    eof = True
                    break
                chunk_sequence += 1
                observed_at = now_iso()
                observed_monotonic_ns = time.monotonic_ns()
                raw.extend(payload)
                pending += payload
                complete = pending.split(b"\n")
                pending = complete.pop()
                line_count = len([line for line in complete if line.strip()])
                for line in complete:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line.decode("utf-8", errors="strict"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise R2CollectorError(f"invalid run JSONL: {exc}") from exc
                    require(isinstance(event, dict), "run JSONL event must be an object")
                    events.append(event)
                    arrivals.append(
                        {
                            "event_ordinal": len(events),
                            "event": event.get("event"),
                            "assistant_delta_index": event.get("index") if event.get("event") == "assistant_delta" else None,
                            "token_id": event.get("token_id") if event.get("event") == "assistant_delta" else None,
                            "observed_at": observed_at,
                            "observed_monotonic_ns": observed_monotonic_ns,
                            "read_chunk_sequence": chunk_sequence,
                            "read_chunk_jsonl_line_count": line_count,
                        }
                    )
        if pending.strip() and not timed_out:
            raise R2CollectorError("run JSONL ended with an unterminated record")
    finally:
        selector.close()
    stdout_path.write_bytes(bytes(raw))
    arrival_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in arrivals),
        encoding="utf-8",
    )
    if timed_out:
        return events, arrivals, 124, False
    try:
        returncode = process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        returncode = 124
    returncode, group_gone = collector_support.terminate_process_group(process, 2.0)
    return events, arrivals, returncode, group_gone


def validate_run_events(
    events: list[dict[str, Any]],
    arrivals: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    assistant = [event for event in events if event.get("event") == "assistant"]
    require(len(assistant) == 1, f"{label} must contain exactly one assistant event")
    row = assistant[0]
    output_tokens = row.get("n_tokens")
    require(output_tokens == RUN_MAX_TOKENS, f"{label} must produce exactly {RUN_MAX_TOKENS} output tokens")
    usage = row.get("usage")
    require(isinstance(usage, dict) and usage.get("completion_tokens") == RUN_MAX_TOKENS, f"{label} usage output count mismatch")
    require(row.get("finish_reason") == "length", f"{label} must use the fixed max-token boundary")
    content = row.get("content")
    require(isinstance(content, str) and content, f"{label} assistant content is empty")
    forbidden = ("<unk>", "[PAD", "<pad>", "<|reserved_special_token", "\ufffd", "Ã©", "Â©", "â€")
    require(not any(marker in content for marker in forbidden), f"{label} output contains reserved-token/UTF-8 corruption")
    e2e_ms = row.get("ms")
    require(isinstance(e2e_ms, (int, float)) and not isinstance(e2e_ms, bool) and math.isfinite(e2e_ms) and e2e_ms > 0, f"{label} engine.infer E2E is invalid")
    deltas = [entry for entry in arrivals if entry.get("event") == "assistant_delta"]
    require(len(deltas) >= 2, f"{label} lacks two token arrivals for steady decode")
    require(all(isinstance(entry.get("token_id"), int) for entry in deltas), f"{label} token arrival lacks token_id")
    delta_indexes = [entry.get("assistant_delta_index") for entry in deltas]
    require(delta_indexes == list(range(len(deltas))), f"{label} assistant delta indexes are not contiguous")
    first_ns = deltas[0]["observed_monotonic_ns"]
    last_ns = deltas[-1]["observed_monotonic_ns"]
    require(isinstance(first_ns, int) and isinstance(last_ns, int) and last_ns > first_ns, f"{label} steady decode interval is not positive")
    interval_sec = (last_ns - first_ns) / 1_000_000_000.0
    steady_tokens = len(deltas) - 1
    steady_tps = steady_tokens / interval_sec
    require(math.isfinite(steady_tps) and steady_tps > 0, f"{label} steady decode TPS is invalid")
    return {
        "output_tokens": output_tokens,
        "visible_token_arrivals": len(deltas),
        "steady_decode_interval_tokens": steady_tokens,
        "steady_decode_interval_ms": interval_sec * 1000.0,
        "steady_decode_tps": steady_tps,
        "steady_decode_definition": "visible token intervals from first through last flushed assistant_delta arrival; first-token interval excluded",
        "engine_infer_e2e_ms": float(e2e_ms),
        "engine_infer_e2e_output_tps": output_tokens * 1000.0 / float(e2e_ms),
        "finish_reason": row["finish_reason"],
        "usage_output_tokens": usage["completion_tokens"],
    }


def validate_run_bundle(
    root: Path,
    bundle: dict[str, Any],
    fingerprint: str,
    sample_ordinal: int,
    *,
    legacy_collection_epoch: dict[str, str] | None = None,
) -> None:
    require(bundle.get("schema_version") == SCHEMA_VERSION, "run bundle schema mismatch")
    require(bundle.get("config_fingerprint") == fingerprint, "run bundle fingerprint mismatch")
    collection_epoch = bundle.get("collection_epoch")
    require(
        (collection_epoch is None and legacy_collection_epoch is not None)
        or (
            isinstance(collection_epoch, dict)
            and set(collection_epoch) == {"collector_sha256", "support_sha256", "resource_sampler_sha256"}
            and all(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None for value in collection_epoch.values())
        ),
        "run bundle collection epoch differs",
    )
    sample = bundle.get("sample")
    require(isinstance(sample, dict) and sample.get("sample_ordinal") == sample_ordinal, "run sample ordinal mismatch")
    stdout = validate_artifact_ref(root, sample.get("stdout"), f"run sample {sample_ordinal}.stdout")
    arrival = validate_artifact_ref(root, sample.get("arrival_timeline"), f"run sample {sample_ordinal}.arrival_timeline")
    for key in ("stderr", "product_effective_config"):
        validate_artifact_ref(root, sample.get(key), f"run sample {sample_ordinal}.{key}")
    resources = sample.get("resources", {})
    observations = validate_artifact_ref(
        root, resources.get("observations"), f"run sample {sample_ordinal}.resources.observations"
    )
    header = json.loads(observations.read_text(encoding="utf-8").splitlines()[0])
    effective_epoch = collection_epoch or legacy_collection_epoch
    require(
        isinstance(effective_epoch, dict)
        and header.get("collector_sha256") == effective_epoch["resource_sampler_sha256"],
        f"run sample {sample_ordinal} sampler epoch differs",
    )
    argv = sample.get("argv")
    require(isinstance(argv, list) and "--backend" in argv, f"run sample {sample_ordinal} lacks backend argv")
    backend_index = argv.index("--backend")
    require(backend_index + 1 < len(argv), f"run sample {sample_ordinal} backend argv is incomplete")
    validate_cuda_bridge_evidence(
        root,
        resources,
        backend=argv[backend_index + 1],
        label=f"run sample {sample_ordinal}.resources",
        expected_collector_sha256=(
            legacy_collection_epoch or collection_epoch or {}
        ).get("collector_sha256"),
    )
    events = [json.loads(line) for line in stdout.read_text(encoding="utf-8").splitlines() if line.strip()]
    arrivals = [json.loads(line) for line in arrival.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = validate_run_events(events, arrivals, f"run sample {sample_ordinal}")
    require(metrics == sample.get("metrics"), f"run sample {sample_ordinal} metrics no longer match raw evidence")


def collect_run_sample(
    root: Path,
    lane: Path,
    fingerprint: str,
    sample_ordinal: int,
    config: dict[str, Any],
    inputs: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    bundle_path = lane / "run-samples" / f"sample-{sample_ordinal}.json"
    if bundle_path.exists():
        require(resume, f"run sample already exists; pass --resume: {bundle_path}")
        bundle = read_json(bundle_path)
        validate_run_bundle(
            root,
            bundle,
            fingerprint,
            sample_ordinal,
            legacy_collection_epoch=frozen_collection_epoch(read_json(lane / "plan.json")),
        )
        return bundle
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    sample_id = f"r2-{config['model_key']}-{config['backend']}-run-{sample_ordinal}"
    attempt_dir = lane / "attempts" / f"run-{sample_ordinal}-{attempt_id}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = attempt_dir / "run.stdout.jsonl"
    stderr_path = attempt_dir / "run.stderr.log"
    arrival_path = attempt_dir / "run.arrival-timeline.jsonl"
    effective = attempt_dir / "run-effective-config.json"
    barrier_release = attempt_dir / "exec-barrier.release"
    argv = run_argv(inputs["binary_path"], effective, config)
    launcher = exec_barrier_launcher_argv(barrier_release, argv)
    environment = dict(config["candidate"]["env"])
    started_at = now_iso()
    process: subprocess.Popen[Any] | None = None
    sampler: dict[str, Any] | None = None
    failure: BaseException | None = None
    with stderr_path.open("x", encoding="utf-8") as stderr_handle:
        stderr_handle.write("[r2-collector] ferrum run stderr follows\n")
        stderr_handle.flush()
        try:
            process = subprocess.Popen(
                launcher,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                text=False,
                bufsize=0,
                start_new_session=True,
            )
            pid = process.pid
            pgid = os.getpgid(pid)
            require(pgid == pid, "run process must own an independent process group")
            marker, marker_source = collector_support.process_identity(pid)
            cuda_preflight = (
                capture_cuda_bridge_preflight(attempt_dir)
                if config["backend"] == "cuda"
                else None
            )
            if config["backend"] == "cuda":
                atomic_write_text(barrier_release, "release\n")
                collector_support.wait_for_process_exec(process, inputs["binary_path"], 30.0)
                require(cuda_preflight is not None, "CUDA run requires an idle process preflight")
                wait_for_cuda_device_allocation(
                    process,
                    pid=pid,
                    pgid=pgid,
                    preflight=cuda_preflight,
                )
            sampler = start_run_resource_sampler(
                root,
                attempt_dir,
                pid=pid,
                pgid=pgid,
                sample_id=sample_id,
                config=config,
                stderr_path=stderr_path,
                cuda_preflight=cuda_preflight,
            )
            if config["backend"] != "cuda":
                atomic_write_text(barrier_release, "release\n")
                collector_support.wait_for_process_exec(process, inputs["binary_path"], 30.0)
            receipt = collector_support.write_process_receipt(
                root,
                attempt_dir / "run-process-receipt.json",
                pid=pid,
                pgid=pgid,
                argv=argv,
                environment=environment,
                marker=marker,
                source=marker_source,
            )
            events, arrivals, returncode, group_gone = collect_jsonl_stream(
                process,
                stdout_path,
                arrival_path,
                float(config["server"]["command_timeout_sec"]),
            )
            if returncode == 124:
                _, group_gone = collector_support.terminate_process_group(process, 5.0)
            samples = collector_support.wait_process_sampler(sampler)
            finished_at = now_iso()
            sampler_meta = sampler
            sampler = None
            require(group_gone, f"run sample {sample_ordinal} process group survived cleanup")
            process = None
            require(returncode == 0, f"run sample {sample_ordinal} failed with returncode {returncode}")
            require(effective.is_file(), f"run sample {sample_ordinal} did not write effective config")
            metrics = validate_run_events(events, arrivals, f"run sample {sample_ordinal}")
            measurement_started_at = samples[0]["sampled_at"]
            measurement_finished_at = samples[-1]["sampled_at"]
            resource_summary = collector_support.resource_sampler.derive_summary(
                sampler_meta["observations"],
                session_id=sample_id,
                cell_id="run:c1",
                backend=config["backend"],
                hardware_id=config["hardware"]["id"],
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
                "sampler_argv": sampler_meta["argv"],
                "sampler_argv_sha256": canonical_json_sha256(sampler_meta["argv"]),
                "observations": artifact_ref(root, sampler_meta["observations"], kind="run-resource-observations"),
                "summary": resource_summary,
                "cuda_pid_namespace_bridge": cuda_bridge_evidence(root, sampler_meta),
            }
            sample = {
                "sample_id": sample_id,
                "sample_ordinal": sample_ordinal,
                "independent_process": True,
                "pid": pid,
                "pgid": pgid,
                "process_start_marker": marker,
                "process_start_source": marker_source,
                "process_receipt": receipt,
                "candidate_binary_sha256": inputs["binary"]["sha256"],
                "source_git_sha": config["candidate"]["source_git_sha"],
                "hardware": copy.deepcopy(config["hardware"]),
                "prompt": copy.deepcopy(inputs["run_prompt"]),
                "profile_detail": "off",
                "eos_policy": "model-metadata",
                "max_tokens": RUN_MAX_TOKENS,
                "argv": argv,
                "argv_sha256": canonical_json_sha256(argv),
                "launcher_argv": launcher,
                "environment": environment,
                "environment_sha256": canonical_json_sha256(environment),
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_sec": duration_seconds(started_at, finished_at),
                "returncode": returncode,
                "stdout": artifact_ref(root, stdout_path, kind="run-stdout-jsonl"),
                "stderr": artifact_ref(root, stderr_path, kind="run-stderr"),
                "arrival_timeline": artifact_ref(root, arrival_path, kind="run-token-arrival-timeline"),
                "product_effective_config": artifact_ref(root, effective, kind="run-effective-config"),
                "metrics": metrics,
                "resources": resources,
            }
            bundle = {
                "schema_version": SCHEMA_VERSION,
                "contract": CONTRACT,
                "config_fingerprint": fingerprint,
                "collection_epoch": process_collection_epoch(),
                "sample": sample,
            }
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(bundle_path, bundle)
            collector_support.append_jsonl(
                lane / "command-log.jsonl",
                {
                    "event": "run-sample-complete",
                    "sample_id": sample_id,
                    "sample_ordinal": sample_ordinal,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "bundle": artifact_relative(root, bundle_path),
                    "bundle_sha256": file_sha256(bundle_path),
                },
            )
            validate_run_bundle(root, bundle, fingerprint, sample_ordinal)
            return bundle
        except BaseException as exc:
            failure = exc
            raise
        finally:
            if process is not None:
                collector_support.cleanup_process_group_noexcept(process, 10.0)
            if sampler is not None and sampler.get("finished") is not True:
                try:
                    sampler["stop_file"].write_text("stop\n", encoding="utf-8")
                    collector_support.finish_resource_sampler(sampler, bracket_after_measurement=False)
                except BaseException:
                    collector_support.cleanup_process_group_noexcept(sampler["process"], 5.0)
                    collector_support.close_sampler_handles(sampler)
                    sampler["finished"] = True
            collector_support.ensure_nonempty_log(stderr_path, "run stderr")
            if failure is not None:
                atomic_write_json(
                    attempt_dir / "failure.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "failed_at": now_iso(),
                        "error_type": type(failure).__name__,
                        "error": str(failure),
                    },
                )


def run_summary(run_bundles: list[dict[str, Any]], parity_record: dict[str, Any]) -> dict[str, Any]:
    require(len(run_bundles) == RUN_SAMPLE_COUNT, "run summary requires exactly three process samples")
    metrics = [bundle["sample"]["metrics"] for bundle in run_bundles]
    steady = [float(row["steady_decode_tps"]) for row in metrics]
    e2e_ms = [float(row["engine_infer_e2e_ms"]) for row in metrics]
    e2e_tps = [float(row["engine_infer_e2e_output_tps"]) for row in metrics]
    parity = parity_record.get("serve_c1_parity_metrics")
    require(isinstance(parity, dict), "run/serve parity metrics are missing")
    serve_steady = float(parity["steady_decode_tps_median"])
    require(math.isfinite(serve_steady) and serve_steady > 0, "serve parity steady decode is invalid")
    run_steady = statistics.median(steady)
    return {
        "sample_count": RUN_SAMPLE_COUNT,
        "independent_process_count": len({bundle["sample"]["pid"] for bundle in run_bundles}),
        "prompt_sha256": hashlib.sha256(RUN_PROMPT.encode("utf-8")).hexdigest(),
        "max_tokens": RUN_MAX_TOKENS,
        "eos_policy": "model-metadata",
        "steady_decode_tps_per_process": steady,
        "steady_decode_tps_median": run_steady,
        "engine_infer_e2e_ms_per_process": e2e_ms,
        "engine_infer_e2e_ms_median": statistics.median(e2e_ms),
        "engine_infer_e2e_output_tps_per_process": e2e_tps,
        "engine_infer_e2e_output_tps_median": statistics.median(e2e_tps),
        "serve_c1_same_prompt_steady_decode_tps_median": serve_steady,
        "run_to_serve_c1_steady_decode_ratio": run_steady / serve_steady,
        "ratio_threshold_evaluated_by_aggregate": 0.90,
        "ratio_status": "unjudged-by-collector",
    }


def collect_artifact_refs(root: Path, value: Any, output: dict[str, dict[str, Any]]) -> None:
    if isinstance(value, dict):
        if {"kind", "path", "sha256", "size_bytes"} <= set(value):
            relative = value.get("path")
            if isinstance(relative, str) and not Path(relative).is_absolute():
                path = (root / relative).resolve()
                try:
                    path.relative_to(root.resolve())
                except ValueError:
                    path = Path("/__outside_artifact_root__")
                if path.is_file():
                    validated = validate_artifact_ref(root, value, f"artifact index candidate {relative}")
                    output[relative] = {
                        "kind": value["kind"],
                        "path": relative,
                        "sha256": file_sha256(validated),
                        "size_bytes": validated.stat().st_size,
                    }
        for child in value.values():
            collect_artifact_refs(root, child, output)
    elif isinstance(value, list):
        for child in value:
            collect_artifact_refs(root, child, output)


def write_artifact_index(
    root: Path,
    lane: Path,
    plan_path: Path,
    inputs: dict[str, Any],
    server_bundle: dict[str, Any],
    server_bundle_path: Path,
    run_bundles: list[dict[str, Any]],
    run_bundle_paths: list[Path],
) -> Path:
    refs: dict[str, dict[str, Any]] = {}
    collect_artifact_refs(root, inputs, refs)
    collect_artifact_refs(root, server_bundle, refs)
    for index, checkpoint_ref in enumerate(server_bundle.get("completed_cell_checkpoints", []), start=1):
        checkpoint_file = validate_artifact_ref(
            root, checkpoint_ref, f"raw index completed_cell_checkpoints[{index}]"
        )
        collect_artifact_refs(root, read_json(checkpoint_file), refs)
    collect_artifact_refs(root, run_bundles, refs)
    for path, kind in [
        (plan_path, "collection-plan"),
        (lane / "config.normalized.json", "normalized-config"),
        (server_bundle_path, "server-session-bundle"),
        *((path, "run-sample-bundle") for path in run_bundle_paths),
        (lane / "command-log.jsonl", "command-log"),
    ]:
        ref = artifact_ref(root, path, kind=kind)
        refs[ref["path"]] = ref
    document = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "artifact_type": "runtime_vnext_r2_ferrum_raw_artifact_index",
        "selected_evidence_only": True,
        "completed_cell_dependencies_included": True,
        "incomplete_failed_suffix_excluded": True,
        "artifact_count": len(refs),
        "artifacts": [refs[key] for key in sorted(refs)],
    }
    path = lane / "raw-artifact-index.json"
    if path.exists():
        require(read_json(path) == document, "raw artifact index changed during resume")
    else:
        atomic_write_json(path, document)
    return path


def validate_final_manifest(root: Path, manifest: dict[str, Any], fingerprint: str) -> None:
    require(manifest.get("schema_version") == SCHEMA_VERSION, "final manifest schema mismatch")
    require(manifest.get("contract") == CONTRACT, "final manifest contract mismatch")
    require(manifest.get("status") == "pass", "final manifest status is not pass")
    require(manifest.get("config_fingerprint") == fingerprint, "final manifest fingerprint mismatch")
    epochs = manifest.get("collection_epochs")
    require(epochs is None or (isinstance(epochs, list) and epochs), "final manifest collection epochs are invalid")
    for index, epoch in enumerate(epochs or [], start=1):
        require(
            isinstance(epoch, dict)
            and isinstance(epoch.get("collection_epoch"), dict)
            and all(
                isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
                for value in epoch["collection_epoch"].values()
            ),
            f"final manifest collection epoch {index} is invalid",
        )
    for key in ("plan", "server_session", "raw_artifact_index"):
        validate_artifact_ref(root, manifest.get(key), f"manifest.{key}")
    runs = manifest.get("run_samples")
    require(isinstance(runs, list) and len(runs) == RUN_SAMPLE_COUNT, "manifest must reference three run samples")
    for index, ref in enumerate(runs, start=1):
        validate_artifact_ref(root, ref, f"manifest.run_samples[{index}]")
    raw_index_path = validate_artifact_ref(root, manifest["raw_artifact_index"], "manifest.raw_artifact_index")
    raw_index = read_json(raw_index_path)
    artifacts = raw_index.get("artifacts")
    require(isinstance(artifacts, list) and len(artifacts) == raw_index.get("artifact_count"), "raw artifact index count mismatch")
    for index, ref in enumerate(artifacts, start=1):
        validate_artifact_ref(root, ref, f"raw artifact index[{index}]")


def collect_lane(root: Path, config_path: Path, *, resume: bool, plan_only: bool) -> Path | None:
    require(root.is_dir(), f"artifact root does not exist: {root}")
    try:
        root.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise R2CollectorError("artifact root must stay outside the Git worktree")
    raw = read_json(config_path)
    config, context = normalize_config(raw)
    frozen_config_path = lane_dir(root, config) / "config.normalized.json"
    if resume and frozen_config_path.is_file():
        frozen_config = read_json(frozen_config_path)
        current_without_env = copy.deepcopy(config)
        frozen_without_env = copy.deepcopy(frozen_config)
        current_without_env.get("candidate", {}).pop("env", None)
        frozen_without_env.get("candidate", {}).pop("env", None)
        require(
            current_without_env == frozen_without_env,
            "resume config differs from frozen normalized config",
        )
        # Sanitized inherited shell variables (for example TERM) are ambient.
        # Resume executes with the exact frozen environment, not today's shell.
        config = frozen_config
    lane, fingerprint = prepare_plan(root, config, context, resume=resume)
    plan_path = lane / "plan.json"
    if plan_only:
        print(f"{PLAN_PREFIX}: {plan_path}")
        return None

    inputs = stage_inputs(root, lane, config, context)
    server_bundle = collect_server_session(root, lane, fingerprint, config, inputs, resume=resume)
    existing = [
        ordinal
        for ordinal in range(1, RUN_SAMPLE_COUNT + 1)
        if (lane / "run-samples" / f"sample-{ordinal}.json").exists()
    ]
    require(existing == list(range(1, len(existing) + 1)), "run resume state is not a chronological prefix")
    run_bundles = [
        collect_run_sample(root, lane, fingerprint, ordinal, config, inputs, resume=resume)
        for ordinal in range(1, RUN_SAMPLE_COUNT + 1)
    ]
    server_bundle_path = lane / "server-session.json"
    run_bundle_paths = [lane / "run-samples" / f"sample-{ordinal}.json" for ordinal in range(1, RUN_SAMPLE_COUNT + 1)]
    index_path = write_artifact_index(
        root,
        lane,
        plan_path,
        inputs,
        server_bundle,
        server_bundle_path,
        run_bundles,
        run_bundle_paths,
    )
    summary = run_summary(run_bundles, server_bundle["run_serve_parity_report"])
    manifest_path = lane / "manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "artifact_type": "runtime_vnext_r2_ferrum_lane_manifest",
        "status": "pass",
        "formal_r2_aggregate_status": "not-evaluated",
        "model_key": config["model_key"],
        "backend": config["backend"],
        "hardware": copy.deepcopy(config["hardware"]),
        "config_fingerprint": fingerprint,
        "profile_detail": "off",
        "source_git_sha": config["candidate"]["source_git_sha"],
        "source_tree_sha": config["candidate"]["source_tree_sha"],
        "dirty_status": copy.deepcopy(config["candidate"]["dirty_status"]),
        "candidate_binary_sha256": inputs["binary"]["sha256"],
        "model_revision": context["lane"]["revision"],
        "model_files": copy.deepcopy(context["model_files"]),
        "plan": artifact_ref(root, plan_path, kind="collection-plan"),
        "inputs": {key: copy.deepcopy(value) for key, value in inputs.items() if isinstance(value, dict)},
        "server_session": artifact_ref(root, server_bundle_path, kind="server-session-bundle"),
        "formal_http_cell_count": len(expected_cells(config["backend"])),
        "formal_http_cells": [cell_id(cell) for cell in expected_cells(config["backend"])],
        "run_serve_parity_probe": copy.deepcopy(server_bundle["run_serve_parity_report"]),
        "run_samples": [artifact_ref(root, path, kind="run-sample-bundle") for path in run_bundle_paths],
        "run_performance": summary,
        "raw_artifact_index": artifact_ref(root, index_path, kind="raw-artifact-index"),
        "collection_epochs": copy.deepcopy(server_bundle["session_epochs"])
        + [
            {
                "kind": "current-run-process",
                "sample_id": bundle["sample"]["sample_id"],
                "collection_epoch": copy.deepcopy(bundle["collection_epoch"]),
            }
            for bundle in run_bundles
        ],
        "pass_line": f"{PASS_PREFIX}: {config['model_key']}/{config['backend']}: {manifest_path}",
    }
    if manifest_path.exists():
        require(resume, f"final manifest already exists; pass --resume: {manifest_path}")
        require(read_json(manifest_path) == manifest, "final manifest changed during resume")
    else:
        atomic_write_json(manifest_path, manifest)
    validate_server_bundle(root, server_bundle, fingerprint, config, inputs)
    for ordinal, bundle in enumerate(run_bundles, start=1):
        validate_run_bundle(root, bundle, fingerprint, ordinal)
    validate_final_manifest(root, manifest, fingerprint)
    print(manifest["pass_line"])
    return manifest_path


def synthetic_bench_report(config: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    count = cell["num_prompts"]
    quality = {
        "bad_output": 0,
        "malformed_stream": 0,
        "missing_done": 0,
        "duplicate_done": 0,
        "zero_output_tokens": 0,
        "stream_bulk_flush": 0,
        "http_500": 0,
        "panic": 0,
    }
    repeats = [
        {
            "repeat": repeat,
            "expected_requests": count,
            "completed_requests": count,
            "errored_requests": 0,
            "warmup_expected": WARMUP_REQUESTS,
            "warmup_completed": WARMUP_REQUESTS,
            "warmup_errored": 0,
            "output_token_count_source": "usage",
            "quality_issues": copy.deepcopy(quality),
            "warmup_quality_issues": copy.deepcopy(quality),
            "tpot_ms": {"p50": 10.0 + repeat, "p75": 11.0 + repeat, "p95": 12.0 + repeat, "p99": 13.0 + repeat},
            "ttft_ms": {"p50": 20.0, "p75": 21.0, "p95": 22.0, "p99": 23.0},
            "e2e_ms": {"p50": 1300.0, "p75": 1310.0, "p95": 1320.0, "p99": 1330.0},
            "output_throughput_tps": 100.0,
            "actual_input_tokens": count * cell["input_tokens"],
            "output_tokens": count * cell["output_tokens"],
        }
        for repeat in range(1, 4)
    ]
    output_len = RUN_MAX_TOKENS if cell["dataset"] in {"random", "run-parity"} else 120
    report = {
        "model": config["request_model"],
        "backend": config["backend"],
        "scenario": "closed_loop",
        "concurrency": cell["concurrency"],
        "n_prompt": cell["input_tokens"],
        "n_gen": 128,
        "n_repeats": 3,
        "n_requests_per_run": count,
        "warmup_requests": WARMUP_REQUESTS,
        "output_token_count_source": "usage",
        "repeat_metrics": repeats,
        "completed_per_run": [count, count, count],
        "errored_per_run": [0, 0, 0],
        "actual_input_tokens_per_request": [[cell["input_tokens"]] * count for _ in range(3)],
        "output_tokens_per_request": [[output_len] * count for _ in range(3)],
        "itl_evidence_per_request": [[{"eligible": True}] * count for _ in range(3)],
    }
    for name in quality:
        report[f"{name}_per_run"] = [0, 0, 0]
    return report


def self_test() -> int:
    template = config_template()
    captured_process_epoch = process_collection_epoch()
    require(
        captured_process_epoch == PROCESS_COLLECTION_EPOCH
        and captured_process_epoch is not PROCESS_COLLECTION_EPOCH,
        "process collection epoch is not frozen and copy-safe",
    )
    remote_sampler_argv = [
        "/usr/bin/python3",
        "/workspace/ferrum-infer-rs/scripts/release/runtime_vnext_resource_sampler.py",
        "--out",
        "/workspace/artifacts/runtime-vnext/r2-ferrum/m2/cuda/random-c1.resource-observations.jsonl",
        "--pid",
        "123",
        "--max-duration-sec",
        "7320",
    ]
    local_sampler_argv = [
        sys.executable,
        str(RESOURCE_SAMPLER_PATH),
        "--out",
        "/tmp/runtime-vnext/r2-ferrum/m2/cuda/random-c1.resource-observations.jsonl",
        "--pid",
        "123",
        "--max-duration-sec",
        "7320",
    ]
    require(
        portable_sampler_argv(remote_sampler_argv, "remote sampler")
        == portable_sampler_argv(local_sampler_argv, "local sampler"),
        "relocated sampler argv normalization self-test failed",
    )
    for index, replacement in (
        (0, "/usr/bin/bash"),
        (1, "/workspace/ferrum-infer-rs/scripts/release/other_sampler.py"),
        (3, "/workspace/artifacts/runtime-vnext/r2-ferrum/m2/cuda/other.jsonl"),
        (5, "124"),
        (7, "7319"),
    ):
        rejected_sampler_argv = copy.deepcopy(remote_sampler_argv)
        rejected_sampler_argv[index] = replacement
        require(
            portable_sampler_argv(rejected_sampler_argv, "rejected sampler")
            != portable_sampler_argv(local_sampler_argv, "local sampler"),
            "relocated sampler argv negative self-test failed",
        )
    reviewed_epoch = {
        identity_key: next(iter(reviewed_git_blob_sha256s(relative_path)))
        for identity_key, relative_path in COLLECTION_EPOCH_SOURCE_PATHS.items()
    }
    require_reviewed_native_collection_epoch(reviewed_epoch, "self-test native collection epoch")
    rejected_epoch = copy.deepcopy(reviewed_epoch)
    rejected_digest = next(
        candidate
        for candidate in ("0" * 64, "f" * 64)
        if candidate not in reviewed_git_blob_sha256s(COLLECTION_EPOCH_SOURCE_PATHS["collector_sha256"])
    )
    rejected_epoch["collector_sha256"] = rejected_digest
    try:
        require_reviewed_native_collection_epoch(rejected_epoch, "self-test native collection epoch")
        raise R2CollectorError("unreviewed native collection epoch unexpectedly passed")
    except R2CollectorError as exc:
        require(
            "is not a reviewed Git-history source" in str(exc),
            "unreviewed native collection epoch failed for the wrong reason",
        )
    expected_active_cap_floors = {
        ("m1-qwen35-4b", "cuda"): 32,
        ("m2-qwen35-35b-a3b", "cuda"): 16,
        ("m3-qwen3-30b-a3b", "cuda"): 32,
        ("m1-qwen35-4b", "metal"): 16,
        ("m2-qwen35-35b-a3b", "metal"): 4,
        ("m3-qwen3-30b-a3b", "metal"): 16,
    }
    require(
        TYPED_ACTIVE_CAP_FLOORS == expected_active_cap_floors,
        "typed active-cap floor matrix self-test failed",
    )
    for (model_key, backend), floor in expected_active_cap_floors.items():
        require(
            validate_typed_active_cap(model_key, backend, floor) == floor,
            "typed active-cap floor acceptance self-test failed",
        )
        try:
            validate_typed_active_cap(model_key, backend, floor - 1)
            raise R2CollectorError("below-floor typed active cap unexpectedly passed")
        except R2CollectorError as exc:
            require(
                f"active floor {floor}" in str(exc),
                "typed active-cap rejection self-test failed for the wrong reason",
            )
    authority_source = {
        "git_sha": "1" * 40,
        "git_tree_sha": "2" * 40,
        "dirty": False,
    }
    authority_binary = "3" * 64
    authority = {
        "artifact_type": R1_CORRECTNESS_ARTIFACT_TYPE,
        "checkpoint_id": "R1",
        "status": "pass",
        "canonical": True,
        "source": authority_source,
        "acceptance": {
            "backend_binary_sha256": {
                "cuda": authority_binary,
                "metal": "4" * 64,
            }
        },
    }
    validate_r1_correctness_authority(
        authority,
        backend="cuda",
        candidate_source=authority_source,
        candidate_binary_sha256=authority_binary,
    )
    for candidate_source, candidate_binary, expected_error in (
        (
            {**authority_source, "git_sha": "5" * 40},
            authority_binary,
            "candidate source differs",
        ),
        (authority_source, "6" * 64, "candidate binary differs"),
    ):
        try:
            validate_r1_correctness_authority(
                authority,
                backend="cuda",
                candidate_source=candidate_source,
                candidate_binary_sha256=candidate_binary,
            )
            raise R2CollectorError("R1 authority mismatch unexpectedly passed")
        except R2CollectorError as exc:
            require(
                expected_error in str(exc),
                "R1 authority mismatch failed for the wrong reason",
            )
    native_rows = [{"pid": 101, "used_gpu_memory_mib": 2048}]
    normalized, strategy = normalize_cuda_compute_rows(
        native_rows,
        server_pid=101,
        group_pids={101, 102},
        preflight_rows=[],
        proc_exists=lambda _pid: True,
    )
    require(normalized == native_rows and strategy == "native-process-group-pid", "native CUDA PID binding self-test failed")
    host_rows = [{"pid": 900001, "used_gpu_memory_mib": 10294}]
    normalized, strategy = normalize_cuda_compute_rows(
        host_rows,
        server_pid=101,
        group_pids={101, 102},
        preflight_rows=[],
        proc_exists=lambda _pid: False,
    )
    require(
        normalized == [{"pid": 101, "used_gpu_memory_mib": 10294}]
        and strategy == "single-new-host-pid-mapped-to-container-server",
        "container CUDA PID namespace binding self-test failed",
    )
    normalized, strategy = normalize_cuda_compute_rows(
        [],
        server_pid=101,
        group_pids={101, 102},
        preflight_rows=[],
        proc_exists=lambda _pid: False,
    )
    require(
        normalized == [] and strategy == "idle-before-device-allocation",
        "CUDA PID bridge pre-allocation idle self-test failed",
    )
    allocation_rows = iter([[], native_rows])

    class FakeRunningProcess:
        returncode = None

        @staticmethod
        def poll() -> None:
            return None

    wait_for_cuda_device_allocation(
        FakeRunningProcess(),
        pid=101,
        pgid=101,
        preflight={"document": {"compute_apps": []}},
        timeout_sec=0.2,
        query_compute_rows=lambda: next(allocation_rows),
        group_pids_fn=lambda _pgid: {101},
    )
    for rows, preflight_rows, visible, expected_error in (
        (host_rows, [{"pid": 8, "used_gpu_memory_mib": 1}], False, "idle preflight"),
        (host_rows * 2, [], False, "exactly one compute application"),
        (host_rows, [], True, "visible in container /proc"),
    ):
        try:
            normalize_cuda_compute_rows(
                rows,
                server_pid=101,
                group_pids={101},
                preflight_rows=preflight_rows,
                proc_exists=lambda _pid, value=visible: value,
            )
            raise R2CollectorError("unsafe CUDA PID namespace input unexpectedly passed")
        except R2CollectorError as exc:
            require(expected_error in str(exc), "CUDA PID namespace rejection self-test failed")
    require(
        set(template)
        == {
            "schema_version",
            "model_key",
            "backend",
            "request_model",
            "models_lock_path",
            "correctness_manifest_path",
            "model_origin_path",
            "semantic_source_root",
            "tokenizer_source_root",
            "candidate",
            "hardware",
            "typed_active_cap",
            "memory_budget_bytes",
            "server",
            "run",
            "datasets",
            "goodput_slo",
        },
        "config template top-level schema self-test failed",
    )
    require(
        set(template["candidate"])
        == {
            "binary_path",
            "build_log_path",
            "build_receipt_path",
            "source_git_sha",
            "dirty_status",
            "cargo_features",
            "env",
        },
        "config template candidate schema self-test failed",
    )
    require(
        {"real-chat", "sharegpt"} == set(template["datasets"]),
        "config template dataset schema self-test failed",
    )
    cuda = expected_cells("cuda")
    metal = expected_cells("metal")
    require(
        [(row["dataset"], row["concurrency"], row["num_prompts"]) for row in cuda]
        == [
            ("random", 1, 100),
            ("random", 4, 100),
            ("random", 16, 100),
            ("random", 32, 100),
            ("sharegpt", 1, 30),
            ("sharegpt", 32, 30),
        ],
        "CUDA formal matrix self-test failed",
    )
    require(
        [(row["dataset"], row["concurrency"], row["num_prompts"]) for row in metal]
        == [
            ("random", 1, 100),
            ("random", 4, 100),
            ("random", 16, 100),
            ("real-chat", 1, 30),
            ("real-chat", 16, 30),
        ],
        "Metal formal matrix self-test failed",
    )
    for backend, cells in (("cuda", cuda), ("metal", metal)):
        config = {
            "backend": backend,
            "request_model": "selftest-model",
            "server": {"host": "127.0.0.1", "port": 18080, "command_timeout_sec": 7200},
            "hardware": {"id": f"selftest-{backend}"},
            "candidate": {"source_git_sha": "1" * 40},
            "goodput_slo": {"ttft": 500.0, "tpot": 50.0, "e2e": 30000.0},
        }
        inputs = {
            "tokenizer_path": Path("/tmp/r2-selftest/tokenizer/tokenizer.json"),
            "realistic_dataset_path": Path("/tmp/r2-selftest/real.jsonl"),
            "run_parity_dataset_path": Path("/tmp/r2-selftest/parity.jsonl"),
        }
        for cell in [*cells, run_parity_cell(backend)]:
            argv = bench_argv(
                Path("/tmp/r2-selftest/ferrum"),
                Path("/tmp/r2-selftest/report.json"),
                config,
                inputs,
                cell,
            )
            _, options, switches = collector_support.baseline_gate.parse_argv(argv, "selftest.bench")
            require(options["--concurrency"] == str(cell["concurrency"]), "bench concurrency self-test failed")
            require(options["--num-prompts"] == str(cell["num_prompts"]), "bench prompt count self-test failed")
            require(options["--n-repeats"] == "3" and options["--seed"] == "9271", "bench repeat/seed self-test failed")
            require("--fail-on-error" in switches and "--require-ci" in switches, "formal bench switches self-test failed")
            require(("--ignore-eos" in switches) == (cell["dataset"] == "random"), "EOS policy self-test failed")
            report = synthetic_bench_report(config, cell)
            validate_bench_report(report, config, cell)
        parity_argv = bench_argv(
            Path("/tmp/r2-selftest/ferrum"),
            Path("/tmp/r2-selftest/report.json"),
            config,
            inputs,
            run_parity_cell(backend),
        )
        require(str(inputs["run_parity_dataset_path"]) in parity_argv, "run parity dataset self-test failed")

    events = [
        {
            "event": "assistant",
            "n_tokens": RUN_MAX_TOKENS,
            "usage": {"completion_tokens": RUN_MAX_TOKENS},
            "finish_reason": "length",
            "content": "1 2 3",
            "ms": 1280.0,
        }
    ]
    arrivals = [
        {
            "event": "assistant_delta",
            "assistant_delta_index": index,
            "token_id": index + 1,
            "observed_monotonic_ns": 1_000_000_000 + index * 10_000_000,
        }
        for index in range(RUN_MAX_TOKENS)
    ]
    run_metrics = validate_run_events(events, arrivals, "selftest.run")
    require(abs(run_metrics["steady_decode_tps"] - 100.0) < 1e-9, "steady decode self-test failed")
    require(abs(run_metrics["engine_infer_e2e_output_tps"] - 100.0) < 1e-9, "engine.infer E2E self-test failed")

    try:
        parse_extra_argv(["--profile-detail", "basic"], "selftest.extra")
        raise R2CollectorError("reserved profile override unexpectedly passed")
    except R2CollectorError as exc:
        require("collector-owned" in str(exc), "reserved option failed for the wrong reason")

    with tempfile.TemporaryDirectory(prefix="runtime-vnext-r2-ferrum-selftest-") as temporary:
        root = Path(temporary)
        blob = root / "blob"
        blob.write_bytes(b"locked-model-bytes")
        logical_model = root / "Qwen3.5-4B-Q4_K_M.gguf"
        logical_model.symlink_to(blob)
        lexical_model = Path(os.path.abspath(os.path.expanduser(str(logical_model))))
        require(
            lexical_model.name == logical_model.name and lexical_model.is_symlink(),
            "logical Hugging Face model path was dereferenced",
        )
        require(
            verify_model_origin(
                lexical_model,
                {logical_model.name: file_sha256(blob)},
            )
            == {logical_model.name: file_sha256(blob)},
            "logical Hugging Face model symlink lock self-test failed",
        )
        template_path = root / "nested" / "config.json"
        write_config_template(template_path)
        require(read_json(template_path) == template, "written config template self-test failed")
        raw = root / "raw.json"
        raw.write_text("{}\n", encoding="utf-8")
        ref = artifact_ref(root, raw, kind="selftest")
        validate_artifact_ref(root, ref, "selftest.ref")
        raw.write_text('{"changed":true}\n', encoding="utf-8")
        try:
            validate_artifact_ref(root, ref, "selftest.tampered")
            raise R2CollectorError("tampered artifact unexpectedly passed")
        except R2CollectorError as exc:
            require("changed" in str(exc), "tamper self-test failed for the wrong reason")

        resume_dir = root / "resume"
        resume_dir.mkdir()
        bridge_dir = resume_dir / "cuda-bridge"
        bridge_dir.mkdir()
        bridge_wrapper = bridge_dir / "nvidia-smi"
        bridge_preflight = bridge_dir / "preflight.json"
        bridge_audit = bridge_dir / "audit.jsonl"
        bridge_observations = resume_dir / "bridge-observations.jsonl"
        atomic_write_text(bridge_wrapper, "#!/bin/sh\nexit 0\n")
        atomic_write_json(
            bridge_preflight,
            {
                "contract": CUDA_PID_NAMESPACE_BRIDGE_CONTRACT,
                "collector_path": COLLECTOR_RELATIVE_PATH,
                "collector_sha256": PROCESS_COLLECTION_EPOCH["collector_sha256"],
                "real_nvidia_smi_path": "/remote-host/usr/bin/nvidia-smi",
                "real_nvidia_smi_sha256": "a" * 64,
                "compute_apps": [],
                "gpu_count": 1,
            },
        )
        atomic_write_text(bridge_audit, '{"status":"mapped"}\n')
        atomic_write_text(bridge_observations, "{}\n")
        real_nvidia_smi = Path("/remote-host/usr/bin/nvidia-smi")
        require(not real_nvidia_smi.exists(), "relocated CUDA bridge fixture unexpectedly exists locally")
        bridge_checkpoint = {
            "wrapper": artifact_ref(
                root, bridge_wrapper, kind="cuda-pid-namespace-wrapper"
            ),
            "preflight": artifact_ref(
                root, bridge_preflight, kind="cuda-pid-namespace-preflight"
            ),
            "audit": artifact_ref(
                root, bridge_audit, kind="cuda-pid-namespace-audit"
            ),
            "real_nvidia_smi_path": str(real_nvidia_smi),
            "real_nvidia_smi_sha256": "a" * 64,
            "server_pid": 101,
            "server_pgid": 101,
        }
        restored_bridge = restore_checkpoint_cuda_bridge(
            root,
            bridge_checkpoint,
            "selftest CUDA bridge",
        )
        bridge_observations_ref = artifact_ref(
            root, bridge_observations, kind="resource-observations"
        )
        bridge_sampler_argv = [
            "/usr/bin/python3",
            "/remote-source/scripts/release/runtime_vnext_resource_sampler.py",
            "--out",
            "/remote-artifacts/" + bridge_observations_ref["path"],
        ]
        bridge_base_environment = {"PATH": "/usr/bin", "TERM": "screen"}
        restored_bridge = restore_checkpoint_cuda_bridge_environment(
            restored_bridge,
            raw_bridge=bridge_checkpoint,
            sampler_argv=bridge_sampler_argv,
            observations_ref=bridge_observations_ref,
            base_environment=bridge_base_environment,
            label="selftest CUDA bridge",
        )
        require(
            restored_bridge is not None
            and restored_bridge["environment"]
            == {
                "PATH": "/remote-artifacts/resume/cuda-bridge:/usr/bin",
                "TERM": "screen",
            },
            "restored CUDA bridge sampler environment self-test failed",
        )
        restored_evidence = cuda_bridge_evidence(
            root, {"cuda_pid_namespace_bridge": restored_bridge}
        )
        require(
            restored_evidence is not None
            and restored_evidence["real_nvidia_smi_path"] == str(real_nvidia_smi)
            and restored_evidence["real_nvidia_smi_sha256"] == "a" * 64,
            "relocated CUDA bridge binary identity self-test failed",
        )
        for field, replacement in (
            ("real_nvidia_smi_path", "/other-host/usr/bin/nvidia-smi"),
            ("real_nvidia_smi_sha256", "b" * 64),
        ):
            rejected_bridge = copy.deepcopy(bridge_checkpoint)
            rejected_bridge[field] = replacement
            try:
                restore_checkpoint_cuda_bridge(root, rejected_bridge, "rejected CUDA bridge")
                raise R2CollectorError("mismatched CUDA bridge identity unexpectedly passed")
            except R2CollectorError as exc:
                require(
                    "preflight identity" in str(exc),
                    "mismatched CUDA bridge failed for the wrong reason",
                )
        observations = resume_dir / "cell.resource-observations.jsonl"
        observation_rows = [
            {
                "record_type": "header",
                "session_id": "resume-session",
                "cell_id": "random:c1",
                "collector_sha256": "7" * 64,
            },
            {
                "record_type": "sample",
                "sequence": 0,
                "sampled_at": "2026-01-01T00:00:00Z",
                "active_requests": 0,
                "process_alive": True,
                "active_probe_errors": [],
            },
            {
                "record_type": "sample",
                "sequence": 1,
                "sampled_at": "2026-01-01T00:00:01Z",
                "active_requests": 1,
                "process_alive": True,
                "active_probe_errors": [],
            },
            {
                "record_type": "sample",
                "sequence": 2,
                "sampled_at": "2026-01-01T00:00:02Z",
                "active_requests": 0,
                "process_alive": True,
                "active_probe_errors": [],
            },
            {
                "record_type": "sample",
                "sequence": 3,
                "sampled_at": "2026-01-01T00:00:03Z",
                "active_requests": 1,
                "process_alive": True,
                "active_probe_errors": [],
            },
            {
                "record_type": "sample",
                "sequence": 4,
                "sampled_at": "2026-01-01T00:00:04Z",
                "active_requests": 0,
                "process_alive": True,
                "active_probe_errors": [],
            },
            {
                "record_type": "footer",
                "exit_reason": "stop-file",
                "sample_count": 5,
            },
        ]
        atomic_write_text(
            observations,
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in observation_rows),
        )
        require(
            observation_active_envelope(
                observations,
                session_id="resume-session",
                identifier="random:c1",
                resource_sampler_sha256="7" * 64,
            )
            == ("2026-01-01T00:00:00Z", "2026-01-01T00:00:04Z"),
            "completed-cell active-envelope positive self-test failed",
        )
        bad_footer = resume_dir / "bad-footer.jsonl"
        bad_rows = copy.deepcopy(observation_rows)
        bad_rows[-1]["sample_count"] = 4
        atomic_write_text(
            bad_footer,
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in bad_rows),
        )
        try:
            observation_active_envelope(
                bad_footer,
                session_id="resume-session",
                identifier="random:c1",
                resource_sampler_sha256="7" * 64,
            )
            raise R2CollectorError("bad checkpoint footer unexpectedly passed")
        except R2CollectorError as exc:
            require("footer sample count" in str(exc), "checkpoint footer self-test failed for the wrong reason")

        idle_body = resume_dir / "post.body.json"
        idle_receipt = resume_dir / "post.receipt.json"
        atomic_write_json(idle_body, {"engine": {"active_requests": 0, "queued_requests": 0}})
        atomic_write_json(
            idle_receipt,
            {
                "returncode": 0,
                "http_status": 200,
                "body_sha256": file_sha256(idle_body),
                "body_size_bytes": idle_body.stat().st_size,
            },
        )
        checkpoint_probe_refs(
            root,
            {"receipt_origin_path": str(idle_receipt), "body_origin_path": str(idle_body)},
            "selftest idle",
        )
        atomic_write_json(idle_body, {"engine": {"active_requests": 1, "queued_requests": 0}})
        atomic_write_json(
            idle_receipt,
            {
                "returncode": 0,
                "http_status": 200,
                "body_sha256": file_sha256(idle_body),
                "body_size_bytes": idle_body.stat().st_size,
            },
        )
        try:
            checkpoint_probe_refs(
                root,
                {"receipt_origin_path": str(idle_receipt), "body_origin_path": str(idle_body)},
                "selftest post-active",
            )
            raise R2CollectorError("post-active checkpoint unexpectedly passed")
        except R2CollectorError as exc:
            require("post-cell probe is active" in str(exc), "post-active self-test failed for the wrong reason")

        try:
            require_chronological_prefix([1, 3], "selftest completed-cell state")
            raise R2CollectorError("noncontiguous checkpoints unexpectedly passed")
        except R2CollectorError as exc:
            require("chronological prefix" in str(exc), "noncontiguous checkpoint self-test failed for the wrong reason")

        cleanup_attempt = resume_dir / "attempt"
        cleanup_attempt.mkdir()
        atomic_write_json(
            cleanup_attempt / "failure.json",
            {
                "cleanup_process_group_gone": True,
                "cleanup_error": None,
                "cleanup_returncode": 0,
            },
        )
        require_completed_attempt_cleanup(cleanup_attempt, "selftest cleanup")
        atomic_write_json(
            cleanup_attempt / "failure.json",
            {
                "cleanup_process_group_gone": False,
                "cleanup_error": None,
                "cleanup_returncode": 0,
            },
        )
        try:
            require_completed_attempt_cleanup(cleanup_attempt, "selftest cleanup")
            raise R2CollectorError("unclean checkpoint attempt unexpectedly passed")
        except R2CollectorError as exc:
            require("survived cleanup" in str(exc), "cleanup checkpoint self-test failed for the wrong reason")

    for signum in (signal.SIGINT, signal.SIGTERM):
        cleanup_state = {"completed": False}

        def interrupted_action(current_signal: signal.Signals = signum) -> int:
            try:
                signal.raise_signal(current_signal)
                return 0
            finally:
                cleanup_state["completed"] = True

        returncode = run_interruptibly(interrupted_action, report=False)
        require(
            returncode == 128 + int(signum),
            f"{signum.name} interrupt returncode self-test failed",
        )
        require(
            cleanup_state["completed"] is True,
            f"{signum.name} did not unwind through child cleanup",
        )

    print(SELFTEST_PASS_LINE)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--write-config-template",
        type=Path,
        metavar="PATH",
        help="write the complete minimum operator config shape and exit",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--exec-barrier-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--release-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cuda-pid-namespace-bridge", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--bridge-collector-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--real-nvidia-smi", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--bridge-server-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--bridge-server-pgid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--bridge-preflight", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--bridge-audit-log", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("exec_argv", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.cuda_pid_namespace_bridge:
        require(
            args.real_nvidia_smi is not None
            and isinstance(args.bridge_collector_sha256, str)
            and SHA256_RE.fullmatch(args.bridge_collector_sha256) is not None
            and isinstance(args.bridge_server_pid, int)
            and args.bridge_server_pid > 0
            and isinstance(args.bridge_server_pgid, int)
            and args.bridge_server_pgid > 0
            and args.bridge_preflight is not None
            and args.bridge_audit_log is not None
            and bool(args.exec_argv),
            "CUDA PID namespace bridge arguments are incomplete",
        )
        return cuda_pid_namespace_bridge(args)
    if args.exec_barrier_child:
        require(args.release_file is not None, "--exec-barrier-child requires --release-file")
        return run_exec_barrier_child(args.release_file, args.exec_argv)
    if args.self_test:
        require(
            args.artifact_root is None
            and args.config is None
            and args.write_config_template is None
            and not args.resume
            and not args.plan_only,
            "--self-test cannot collect or plan a lane",
        )
        return self_test()
    if args.write_config_template is not None:
        require(
            args.artifact_root is None and args.config is None and not args.resume and not args.plan_only,
            "--write-config-template cannot collect or plan a lane",
        )
        write_config_template(args.write_config_template)
        return 0
    require(args.artifact_root is not None and args.config is not None, "--artifact-root and --config are required")
    def collect_action() -> int:
        collect_lane(
            args.artifact_root.expanduser().resolve(),
            args.config.expanduser().resolve(),
            resume=args.resume,
            plan_only=args.plan_only,
        )
        return 0

    return run_interruptibly(collect_action)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        R2CollectorError,
        collector_support.CollectorError,
        collector_support.baseline_gate.BaselineError,
        collector_support.resource_sampler.ResourceEvidenceError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"runtime vNext R2 Ferrum collector failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
