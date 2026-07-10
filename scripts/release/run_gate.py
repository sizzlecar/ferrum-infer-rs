#!/usr/bin/env python3
"""Unified release gate runner.

This is the product-facing release entrypoint. It delegates to the existing
source, binary, and summary validators, then writes one normalized
`gate.manifest.json` and prints the unified PASS line.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_LANES = {
    "unit": "unit",
    "metal": "metal",
    "cuda-smoke": "cuda-smoke",
    "cuda-full": "cuda-full",
    "cuda-llama-dense": "cuda-llama-dense",
    "cuda-llama33-70b-4bit-2x4090-smoke": "cuda-llama33-70b-4bit-2x4090-smoke",
    "cuda-llama33-70b-4bit-2x4090": "cuda-llama33-70b-4bit-2x4090",
}
BINARY_LANES = {
    "metal-tarball",
    "cuda-tarball",
    "homebrew-metal",
    "homebrew-cuda-fetch",
}
LANES = (
    "vnext-g00a",
    "vnext-g00",
    "unit",
    "metal",
    "cuda-smoke",
    "cuda-full",
    "cuda-llama-dense",
    "cuda-llama33-70b-4bit-2x4090-smoke",
    "cuda-llama33-70b-4bit-2x4090",
    "metal-tarball",
    "cuda-tarball",
    "homebrew-metal",
    "homebrew-cuda-fetch",
    "release-summary",
    "release-complete",
)
ENV_ALLOW_PREFIXES = ("FERRUM_",)
ENV_ALLOW_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "LD_LIBRARY_PATH",
    "RUST_LOG",
)
SECRET_KEY_FRAGMENTS = ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
VNEXT_FROZEN_LEGACY_SHA = "cff4c47765ef3259b8a04890187d99c60da86394"
VNEXT_G00_FULL_SELFTEST_PASS = (
    "FERRUM RUNTIME VNEXT G00 BASELINE FULL SELFTEST PASS"
)
VNEXT_G00_SELFTEST_SUMMARY_PREFIX = (
    "FERRUM RUNTIME VNEXT G00 BASELINE SELFTEST SUMMARY:"
)
VNEXT_G00_REDTEAM_MUTATION_COUNT = 95
VNEXT_G00_REDTEAM_MUTATION_MATRIX_SHA256 = "c70d1af0c8b49c63147d2a23e3bd5421179079ca7b4bc3eab5922d8cb53bab88"
VNEXT_G00_REDTEAM_MUTATION_NAMES = (
    "dirty",
    "stale",
    "preset-policy-drift",
    "expectations-lock-sha",
    "preset-semantic-chain-forgery",
    "chat-template-chain-forgery",
    "generation-config-chain-forgery",
    "llama-official-chain-forgery",
    "model-resolution-revision",
    "model-resolution-resolver-sha",
    "model-resolution-shard-index",
    "model-resolution-expected-sha-non-lfs",
    "hardware-derived-fingerprint",
    "hardware-probe-command",
    "hardware-probe-raw-derivation",
    "historical-identical-mutation",
    "historical-missing-signature",
    "historical-success-returncode",
    "model-sha",
    "cuda-primary-blocked",
    "scenario-no-ferrum-argv",
    "scenario-missing-tools",
    "scenario-missing-schema",
    "scenario-missing-utf8",
    "scenario-missing-thinking",
    "scenario-missing-cancel",
    "scenario-artifact-sha",
    "scenario-fake-pass",
    "cross-hardware",
    "repeat-count",
    "expected-request-accounting",
    "expected-requests-absolute",
    "expected-requests-missing",
    "errors",
    "usage",
    "ab-identity-swap",
    "duplicate-server-session",
    "server-session-same-lane-overlap",
    "cross-lane-session-id-conflict",
    "server-cell-window-overlap",
    "report-outside-cell-window",
    "server-process-start-marker",
    "ready-probe-returncode",
    "loaded-model-probe",
    "server-effective-config-model",
    "server-product-config-cap",
    "server-effective-config-argv",
    "benchmark-client-tree-binding",
    "benchmark-client-rust-allowlist",
    "bench-canonical-argv",
    "dataset-sha",
    "tokenizer-sha",
    "config-sha",
    "hardware-fingerprint",
    "active-cap",
    "observed-active",
    "zero-observed-active",
    "resource-observation-pid",
    "resource-observation-process-start",
    "resource-summary-forgery",
    "resource-http-process-probe",
    "raw-report-sha",
    "raw-report-metric",
    "raw-report-usage",
    "raw-report-quality",
    "itl-evidence-missing",
    "itl-evidence-cardinality",
    "itl-source-forged",
    "itl-usage-event-claimed-eligible",
    "itl-interval-claimed-eligible",
    "itl-coalesced-claimed-eligible",
    "itl-repeat-counts-forged",
    "itl-repeat-intervals-forged",
    "itl-ineligible-partial-percentiles",
    "swap-growth",
    "duplicate-repeat-ordinal",
    "warmup-error",
    "bench-thinking-payload",
    "bench-env-hash",
    "run-real-command",
    "run-session-global-overlap",
    "run-command-window-binding",
    "inventory-source-coverage",
    "inventory-review-binding",
    "artifact-index-empty-file",
    "artifact-index-symlink",
    "build-real-command",
    "build-raw-summary",
    "build-finished-failure",
    "build-content-evidence",
    "build-native-log-derivation",
    "build-restore-fresh",
    "build-restore-binary",
    "build-restore-mtime",
    "malformed-artifact-type",
)
VNEXT_G00A_DOES_NOT_PROVE = {"G00", "G01B", "model_migration", "performance", "release"}
VNEXT_G00A_CONTRACT_PATHS = {
    "docs/goals/runtime-vnext-0.8.0-2026-07-10/G00_BASELINE.md",
    "docs/goals/runtime-vnext-0.8.0-2026-07-10/G01_CORE_CONTRACTS.md",
    "docs/goals/runtime-vnext-0.8.0-2026-07-10/GOAL.md",
    "docs/goals/runtime-vnext-0.8.0-2026-07-10/MODEL_MATRIX.md",
    "scripts/release/configs/runtime_vnext_generation_presets.json",
    "scripts/release/configs/runtime_vnext_historical_bugs.json",
    "scripts/release/configs/runtime_vnext_inventory_review.json",
    "scripts/release/configs/runtime_vnext_models.json",
    "scripts/release/runtime_vnext_g00a_checkpoint.py",
    "scripts/release/runtime_vnext_inventory.py",
    "scripts/release/runtime_vnext_model_resolver.py",
}
VNEXT_LEGACY_EXPECTATIONS_PATH = (
    "scripts/release/configs/runtime_vnext_legacy_correctness_expectations.json"
)
VNEXT_LEGACY_EXPECTATIONS_SNAPSHOT = "legacy-correctness-expectations.json"
VNEXT_PRIMARY_MODELS = {
    "m1-qwen35-4b": "Qwen/Qwen3.5-4B",
    "m2-qwen35-35b-a3b": "Qwen/Qwen3.5-35B-A3B",
    "m3-qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
}
VNEXT_SUPPLEMENTAL_MODELS = {
    "qwen3-coder-30b-a3b": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "deepseek-r1-qwen3-8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "llama31-8b-compat": "meta-llama/Llama-3.1-8B-Instruct",
}
VNEXT_RESOLUTION_MODEL_IDS = {
    "m1-qwen35-4b": "M1",
    "m2-qwen35-35b-a3b": "M2",
    "m3-qwen3-30b-a3b": "M3",
    "qwen3-coder-30b-a3b": "Qwen3-Coder-30B-A3B-Instruct",
    "deepseek-r1-qwen3-8b": "DeepSeek-R1-0528-Qwen3-8B",
    "llama31-8b-compat": "Llama-3.1-8B-Instruct",
}
CHILD_INDEX_EXCLUDED = {
    "manifest.json",
    "gate.manifest.json",
    "run_gate.child.stdout",
    "run_gate.child.stderr",
    "run_gate.child.command.json",
}


@dataclass(frozen=True)
class LaneCommand:
    cmd: list[str]
    binary_path: Path | None = None
    model: str | None = None
    expected_child_pass_line: str | None = None
    child_manifest_path: Path | None = None
    expected_source_git_sha: str | None = None
    provenance_kind: str | None = None


class GateError(Exception):
    pass


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def command_line() -> list[str]:
    return [sys.executable, *sys.argv]


def git_output(args: list[str], *, default: str = "unknown") -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return default
    if proc.returncode != 0:
        return default
    return proc.stdout.strip()


def git_sha() -> str:
    return git_output(["rev-parse", "HEAD"])


def git_dirty_status() -> dict[str, Any]:
    text = git_output(["status", "--short"], default="")
    lines = [line for line in text.splitlines() if line.strip()]
    return {
        "is_dirty": bool(lines),
        "status_short": lines,
    }


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def require_external_vnext_g00_output(path: Path) -> None:
    require_gate(
        not is_within(path, REPO_ROOT),
        f"vnext-g00 --out must resolve outside the Git source tree: {path}",
    )


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitized_env_summary() -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        if not (key in ENV_ALLOW_KEYS or any(key.startswith(prefix) for prefix in ENV_ALLOW_PREFIXES)):
            continue
        if any(fragment in key.upper() for fragment in SECRET_KEY_FRAGMENTS):
            out[key] = "<redacted>"
        elif len(value) > 512:
            out[key] = f"{value[:512]}...<truncated>"
        else:
            out[key] = value
    return out


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def build_lane_command(args: argparse.Namespace, out_dir: Path) -> LaneCommand:
    lane = args.lane
    if lane == "vnext-g00a":
        if args.coupling_inventory is None:
            raise GateError("vnext-g00a requires --coupling-inventory")
        if args.model_resolution is None:
            raise GateError("vnext-g00a requires --model-resolution")
        return LaneCommand(
            cmd=[
                sys.executable,
                "scripts/release/runtime_vnext_g00a_checkpoint.py",
                "--coupling-inventory",
                str(args.coupling_inventory),
                "--model-resolution",
                str(args.model_resolution),
                "--out",
                str(out_dir),
            ],
            expected_child_pass_line=f"FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT PASS: {out_dir}",
            child_manifest_path=out_dir / "manifest.json",
            provenance_kind="vnext-g00a",
        )
    if lane == "vnext-g00":
        return LaneCommand(
            cmd=[
                sys.executable,
                "scripts/release/runtime_vnext_baseline_gate.py",
                "--out",
                str(out_dir),
                "--require-full-self-test",
            ],
            expected_child_pass_line=f"FERRUM RUNTIME VNEXT G00 BASELINE PASS: {out_dir}",
            child_manifest_path=out_dir / "manifest.json",
            expected_source_git_sha=VNEXT_FROZEN_LEGACY_SHA,
            provenance_kind="vnext-g00",
        )
    if lane in SOURCE_LANES:
        return LaneCommand(
            cmd=[
                "scripts/release/g0_source_gate.sh",
                SOURCE_LANES[lane],
                str(out_dir),
            ],
            binary_path=Path("target/release/ferrum")
            if lane.startswith("cuda") or lane == "metal"
            else None,
            model=model_for_source_lane(lane),
            expected_child_pass_line=source_pass_line(lane, out_dir),
        )
    if lane in BINARY_LANES:
        if not args.version:
            raise GateError(f"{lane} requires --version")
        cmd = [
            sys.executable,
            "scripts/release/release_binary_gate.py",
            lane,
            "--version",
            args.version,
            "--out",
            str(out_dir),
        ]
        if args.asset_path is not None:
            cmd.extend(["--asset-path", str(args.asset_path)])
        if args.sha256 is not None:
            cmd.extend(["--sha256", args.sha256])
        if args.model is not None:
            cmd.extend(["--model", args.model])
        if args.model_name is not None:
            cmd.extend(["--model-name", args.model_name])
        if args.port is not None:
            cmd.extend(["--port", str(args.port)])
        return LaneCommand(
            cmd=cmd,
            model=args.model,
            expected_child_pass_line=binary_pass_line(lane, out_dir),
        )
    if lane == "release-summary":
        release_root = args.release_root or out_dir
        return LaneCommand(
            cmd=[sys.executable, "scripts/release/g0_release_summary.py", str(release_root)],
            expected_child_pass_line=f"G0 RELEASE PASS: {release_root}",
        )
    if lane == "release-complete":
        if args.completion_manifest is None:
            raise GateError("release-complete requires --completion-manifest")
        return LaneCommand(
            cmd=[
                sys.executable,
                "scripts/release/validate_release_completion_manifest.py",
                "--manifest",
                str(args.completion_manifest),
                "--out",
                str(out_dir),
            ],
            expected_child_pass_line=f"FERRUM RELEASE COMPLETION PASS: {out_dir}",
        )
    raise GateError(f"unknown lane: {lane}")


def model_for_source_lane(lane: str) -> str | None:
    if lane in {"cuda-smoke", "cuda-full"}:
        return "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
    if lane == "cuda-llama-dense":
        return "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
    if lane in {
        "cuda-llama33-70b-4bit-2x4090-smoke",
        "cuda-llama33-70b-4bit-2x4090",
    }:
        return "clowman/Llama-3.3-70B-Instruct-GPTQ-Int4"
    return None


def source_pass_line(lane: str, out_dir: Path) -> str:
    delegated = {
        "unit": "unit",
        "metal": "metal",
        "cuda-smoke": "g0_cuda4090_smoke",
        "cuda-full": "g0_cuda4090_full",
        "cuda-llama-dense": "g0_cuda4090_llama_dense",
        "cuda-llama33-70b-4bit-2x4090-smoke": "g0_cuda2x4090_llama33_70b_4bit_smoke",
        "cuda-llama33-70b-4bit-2x4090": "g0_cuda2x4090_llama33_70b_4bit",
    }[lane]
    return f"G0 SOURCE {delegated} PASS: {out_dir}"


def binary_pass_line(lane: str, out_dir: Path) -> str:
    delegated = {
        "metal-tarball": "METAL TARBALL GATE",
        "cuda-tarball": "CUDA TARBALL GATE",
        "homebrew-metal": "HOMEBREW METAL GATE",
        "homebrew-cuda-fetch": "HOMEBREW CUDA FETCH GATE",
    }[lane]
    return f"{delegated} PASS: {out_dir}"


def require_gate(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def require_object(value: Any, label: str) -> dict[str, Any]:
    require_gate(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require_gate(isinstance(value, list), f"{label} must be a JSON array")
    return value


def require_string(value: Any, label: str) -> str:
    require_gate(isinstance(value, str) and value.strip(), f"{label} must be a non-empty string")
    return value.strip()


def require_sha256(value: Any, label: str) -> str:
    digest = require_string(value, label).lower()
    require_gate(SHA256_RE.fullmatch(digest) is not None, f"{label} must be a lowercase SHA256")
    return digest


def require_git_sha(value: Any, label: str) -> str:
    value = require_string(value, label).lower()
    require_gate(GIT_SHA_RE.fullmatch(value) is not None, f"{label} must be a 40-character git SHA")
    return value


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pretty_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def strict_json_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def strict_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=strict_json_object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON number: {value}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GateError(f"invalid {label}: {exc}") from exc


def decoded_request_body(request: dict[str, Any], label: str) -> Any:
    encoded = require_string(request.get("response_body_base64"), f"{label}.response_body_base64")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise GateError(f"invalid {label} response body base64: {exc}") from exc
    require_gate(len(payload) == request.get("response_bytes"), f"{label} response body size mismatch")
    require_gate(hashlib.sha256(payload).hexdigest() == request.get("response_sha256"), f"{label} response body SHA256 mismatch")
    return strict_json_bytes(payload, f"{label} response body JSON")


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GateError(f"invalid {label} {path}: {exc}") from exc
    return require_object(value, label)


def child_artifact_path(root: Path, raw: Any, label: str) -> tuple[Path, str]:
    relative = require_string(raw, label)
    rel_path = Path(relative)
    require_gate(not rel_path.is_absolute(), f"{label} must be relative to the child artifact root")
    resolved = (root / rel_path).resolve()
    try:
        normalized = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise GateError(f"{label} escapes the child artifact root: {relative}") from exc
    require_gate(normalized == rel_path.as_posix(), f"{label} must be a normalized relative path")
    require_gate(resolved.is_file() and not resolved.is_symlink(), f"{label} missing regular artifact: {normalized}")
    return resolved, normalized


def validate_child_artifact_index(
    root: Path,
    child_manifest: dict[str, Any],
    *,
    role_from_top_level_path: bool = True,
) -> dict[str, dict[str, Any]]:
    rows = require_list(child_manifest.get("artifact_index"), "delegated manifest artifact_index")
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = require_object(raw, f"delegated manifest artifact_index[{index}]")
        path, relative = child_artifact_path(root, row.get("path"), f"artifact_index[{index}].path")
        require_gate(relative not in CHILD_INDEX_EXCLUDED, f"artifact_index contains excluded control artifact: {relative}")
        require_gate(relative not in indexed, f"artifact_index contains duplicate path: {relative}")
        digest = require_sha256(row.get("sha256"), f"artifact_index[{relative}].sha256")
        size = row.get("size_bytes")
        require_gate(isinstance(size, int) and not isinstance(size, bool) and size > 0, f"artifact_index[{relative}].size_bytes must be positive")
        require_gate(path.stat().st_size == size, f"artifact_index[{relative}] size mismatch")
        require_gate(sha256(path) == digest, f"artifact_index[{relative}] SHA256 mismatch")
        if role_from_top_level_path:
            expected_role = relative.split("/", 1)[0] if "/" in relative else "root-manifest"
            require_gate(row.get("role") == expected_role, f"artifact_index[{relative}] role mismatch")
        else:
            require_string(row.get("role"), f"artifact_index[{relative}].role")
        indexed[relative] = row
    actual: set[str] = set()
    for path in sorted(root.rglob("*")):
        require_gate(not path.is_symlink(), f"child artifact tree contains forbidden symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative not in CHILD_INDEX_EXCLUDED:
            actual.add(relative)
    require_gate(set(indexed) == actual, f"delegated artifact_index coverage mismatch: missing={sorted(actual - set(indexed))} extra={sorted(set(indexed) - actual)}")
    require_gate(child_manifest.get("artifact_count") == len(indexed), "delegated manifest artifact_count mismatch")
    return indexed


def require_indexed_artifact(
    root: Path,
    index: dict[str, dict[str, Any]],
    raw_path: Any,
    raw_sha256: Any,
    label: str,
) -> tuple[Path, str, str]:
    path, relative = child_artifact_path(root, raw_path, f"{label}.path")
    digest = require_sha256(raw_sha256, f"{label}.sha256")
    require_gate(relative in index, f"{label} is absent from delegated artifact_index: {relative}")
    require_gate(index[relative].get("sha256") == digest, f"{label} SHA256 differs from delegated artifact_index")
    require_gate(sha256(path) == digest, f"{label} artifact SHA256 mismatch")
    return path, relative, digest


def validate_vnext_g00_expectations_snapshot(
    root: Path,
    artifact_index: dict[str, dict[str, Any]],
    raw: Any,
    *,
    source_path: Path,
    source_sha256: str,
    label: str,
) -> dict[str, Any]:
    ref = require_object(raw, label)
    require_gate(
        set(ref) == {"kind", "path", "sha256"},
        f"{label} must be a canonical artifact snapshot reference",
    )
    require_gate(ref.get("kind") == "raw-json", f"{label}.kind must be raw-json")
    snapshot_path, snapshot_relative, snapshot_sha256 = require_indexed_artifact(
        root,
        artifact_index,
        ref.get("path"),
        ref.get("sha256"),
        label,
    )
    require_gate(
        snapshot_relative == VNEXT_LEGACY_EXPECTATIONS_SNAPSHOT,
        f"{label}.path must be {VNEXT_LEGACY_EXPECTATIONS_SNAPSHOT}",
    )
    require_gate(
        snapshot_sha256 == source_sha256,
        f"{label} SHA256 differs from models.lock source contract",
    )
    require_gate(
        snapshot_path.read_bytes() == source_path.read_bytes(),
        f"{label} bytes differ from the checked-in expectations catalog",
    )
    return {
        "kind": "raw-json",
        "path": snapshot_relative,
        "sha256": snapshot_sha256,
    }


def validate_vnext_g00_runner_identity(
    raw: Any,
    *,
    scenario_runner_path: str,
    scenario_runner_sha256: str,
    validator_git_sha: str,
    contract_by_path: dict[str, dict[str, Any]],
    label: str,
    verify_checkout: bool,
) -> dict[str, Any]:
    runner = require_object(raw, label)
    expected_fields = {
        "path",
        "sha256",
        "git_sha",
        "source_tree_sha",
        "git_blob_sha",
        "dirty_status",
    }
    require_gate(set(runner) == expected_fields, f"{label} field set mismatch")
    require_gate(runner.get("path") == scenario_runner_path, f"{label}.path mismatch")
    runner_sha256 = require_sha256(runner.get("sha256"), f"{label}.sha256")
    require_gate(runner_sha256 == scenario_runner_sha256, f"{label}.sha256 mismatch")
    runner_git_sha = require_git_sha(runner.get("git_sha"), f"{label}.git_sha")
    runner_tree_sha = require_git_sha(runner.get("source_tree_sha"), f"{label}.source_tree_sha")
    runner_blob_sha = require_git_sha(runner.get("git_blob_sha"), f"{label}.git_blob_sha")
    require_gate(runner_git_sha == validator_git_sha, f"{label}.git_sha differs from delegated validator")
    dirty = require_object(runner.get("dirty_status"), f"{label}.dirty_status")
    require_gate(
        dirty == {"is_dirty": False, "status_short": []},
        f"{label}.dirty_status must prove a clean runner checkout",
    )
    runner_contract = require_object(
        contract_by_path.get(scenario_runner_path),
        f"{label} delegated contract",
    )
    require_gate(
        require_sha256(runner_contract.get("sha256"), f"{label} delegated contract SHA256")
        == runner_sha256,
        f"{label} differs from delegated contract",
    )
    if verify_checkout:
        require_gate(git_sha() == validator_git_sha, f"{label} validator SHA is stale against current HEAD")
        require_gate(
            git_output(["rev-parse", "HEAD^{tree}"]) == runner_tree_sha,
            f"{label}.source_tree_sha is stale against current HEAD",
        )
        require_gate(
            git_output(["rev-parse", f"HEAD:{scenario_runner_path}"]) == runner_blob_sha,
            f"{label}.git_blob_sha is stale against current HEAD",
        )
        require_gate(not git_dirty_status()["is_dirty"], f"{label} current checkout is dirty")
        runner_path = REPO_ROOT / scenario_runner_path
        require_gate(
            runner_path.is_file()
            and not runner_path.is_symlink()
            and sha256(runner_path) == runner_sha256,
            f"{label} current runner file differs from its identity",
        )
    return {
        "path": scenario_runner_path,
        "sha256": runner_sha256,
        "git_sha": runner_git_sha,
        "source_tree_sha": runner_tree_sha,
        "git_blob_sha": runner_blob_sha,
        "dirty_status": {"is_dirty": False, "status_short": []},
    }


def normalized_file_locks(raw: Any, label: str) -> list[dict[str, Any]]:
    rows = require_list(raw, label)
    require_gate(rows, f"{label} must not be empty")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item_raw in enumerate(rows):
        item = require_object(item_raw, f"{label}[{index}]")
        path = require_string(item.get("path"), f"{label}[{index}].path")
        require_gate(path not in seen, f"{label} duplicate file path: {path}")
        seen.add(path)
        size = item.get("size_bytes")
        require_gate(isinstance(size, int) and not isinstance(size, bool) and size > 0, f"{label}[{index}].size_bytes must be positive")
        normalized.append(
            {
                "path": path,
                "sha256": require_sha256(item.get("sha256"), f"{label}[{index}].sha256"),
                "size_bytes": size,
            }
        )
    return sorted(normalized, key=lambda row: row["path"])


def normalized_model_source(raw: Any, label: str) -> dict[str, Any]:
    source = require_object(raw, label)
    return {
        "repo": require_string(source.get("repo"), f"{label}.repo"),
        "revision": require_git_sha(source.get("revision"), f"{label}.revision"),
        "files": normalized_file_locks(source.get("files"), f"{label}.files"),
    }


def require_exact_string_set(raw: Any, expected: set[str], label: str) -> set[str]:
    values = require_list(raw, label)
    normalized = [require_string(value, f"{label}[{index}]") for index, value in enumerate(values)]
    require_gate(len(normalized) == len(set(normalized)), f"{label} contains duplicates")
    actual = set(normalized)
    require_gate(actual == expected, f"{label} mismatch: expected={sorted(expected)} actual={sorted(actual)}")
    return actual


def validate_vnext_catalog_expected_weight_facts(
    raw_catalog: Any,
    resolved_by_id: dict[str, dict[str, Any]],
) -> int:
    catalog = require_object(raw_catalog, "vnext-g00a models catalog")
    rows = require_list(catalog.get("models"), "vnext-g00a models catalog.models")
    require_gate(len(rows) == 12, "vnext-g00a models catalog must contain 12 lanes")
    catalog_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row = require_object(raw_row, f"vnext-g00a models catalog.models[{index}]")
        lane_id = require_string(row.get("id"), f"vnext-g00a models catalog.models[{index}].id")
        require_gate(lane_id not in catalog_by_id, f"vnext-g00a duplicate catalog lane id: {lane_id}")
        catalog_by_id[lane_id] = row
    require_gate(
        set(catalog_by_id) == set(resolved_by_id),
        "vnext-g00a models catalog/resolution lane set mismatch",
    )

    assertion_count = 0
    for lane_id in sorted(catalog_by_id):
        catalog_lane = catalog_by_id[lane_id]
        resolved_lane = require_object(resolved_by_id[lane_id], f"vnext-g00a resolved lane {lane_id}")
        weight_source = require_object(
            resolved_lane.get("weight_source"),
            f"vnext-g00a resolved lane {lane_id}.weight_source",
        )
        weight_files: dict[str, dict[str, Any]] = {}
        for file_index, raw_file in enumerate(
            require_list(weight_source.get("files"), f"vnext-g00a resolved lane {lane_id}.weight_source.files")
        ):
            file_row = require_object(raw_file, f"vnext-g00a resolved lane {lane_id}.weight_source.files[{file_index}]")
            path = require_string(file_row.get("path"), f"vnext-g00a resolved lane {lane_id}.weight_source.files[{file_index}].path")
            require_gate(path not in weight_files, f"vnext-g00a duplicate resolved weight path: {lane_id}/{path}")
            weight_files[path] = file_row

        selectors = require_list(catalog_lane.get("files"), f"vnext-g00a models catalog {lane_id}.files")
        for selector_index, raw_selector in enumerate(selectors):
            selector = require_object(raw_selector, f"vnext-g00a models catalog {lane_id}.files[{selector_index}]")
            expected_size = selector.get("expected_size_bytes")
            expected_sha_raw = selector.get("expected_sha256")
            if expected_size is None and expected_sha_raw is None:
                continue
            require_gate(
                "path" in selector and "glob" not in selector,
                f"vnext-g00a catalog expected weight identity requires an exact path: {lane_id}",
            )
            path = require_string(selector.get("path"), f"vnext-g00a models catalog {lane_id}.files[{selector_index}].path")
            file_row = require_object(weight_files.get(path), f"vnext-g00a resolved expected weight {lane_id}/{path}")
            if expected_size is not None:
                require_gate(
                    isinstance(expected_size, int) and not isinstance(expected_size, bool) and expected_size > 0,
                    f"vnext-g00a catalog expected size is invalid: {lane_id}/{path}",
                )
                require_gate(
                    file_row.get("size_bytes") == expected_size,
                    f"vnext-g00a catalog expected size mismatch: {lane_id}/{path}",
                )
            if expected_sha_raw is not None:
                require_gate(
                    selector.get("required") is True,
                    f"vnext-g00a catalog expected SHA256 requires required=true: {lane_id}/{path}",
                )
                expected_sha = require_sha256(
                    expected_sha_raw,
                    f"vnext-g00a models catalog {lane_id}.files[{selector_index}].expected_sha256",
                )
                require_gate(
                    file_row.get("sha256") == expected_sha,
                    f"vnext-g00a catalog expected SHA256 mismatch: {lane_id}/{path}",
                )
                require_gate(
                    file_row.get("sha256_source") == "hugging_face_lfs_oid",
                    f"vnext-g00a catalog expected SHA256 lacks Hugging Face LFS identity: {lane_id}/{path}",
                )
                require_gate(
                    file_row.get("lfs_oid") == expected_sha,
                    f"vnext-g00a catalog expected SHA256 differs from LFS OID: {lane_id}/{path}",
                )
            assertion_count += 1
    require_gate(assertion_count > 0, "vnext-g00a models catalog has no expected weight identity assertions")
    return assertion_count


def validate_vnext_g00a_provenance(
    lane_command: LaneCommand,
    child_manifest: dict[str, Any],
    child_manifest_sha256: str,
    *,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    manifest_path = lane_command.child_manifest_path
    require_gate(manifest_path is not None, "vnext-g00a delegated manifest path is missing")
    root = manifest_path.parent.resolve()
    require_gate(manifest_path.resolve() == root / "manifest.json", "vnext-g00a manifest path mismatch")
    require_gate(child_manifest.get("schema_version") == 1, "vnext-g00a manifest schema_version mismatch")
    require_gate(child_manifest.get("artifact_type") == "runtime_vnext_g00a_fact_checkpoint_manifest", "vnext-g00a manifest artifact_type mismatch")
    require_gate(child_manifest.get("lane") == "runtime-vnext-g00a", "vnext-g00a manifest lane mismatch")
    require_gate(child_manifest.get("checkpoint_id") == "G00a", "vnext-g00a checkpoint_id mismatch")
    require_gate(child_manifest.get("canonical") is True, "vnext-g00a manifest must be canonical")
    require_gate(child_manifest.get("dirty") is False, "vnext-g00a manifest must be clean")
    artifact_dir = Path(require_string(child_manifest.get("artifact_dir"), "vnext-g00a artifact_dir"))
    require_gate(artifact_dir.resolve() == root, "vnext-g00a artifact_dir mismatch")
    require_exact_string_set(child_manifest.get("unlocks"), {"G01A"}, "vnext-g00a unlocks")
    require_exact_string_set(child_manifest.get("does_not_prove"), VNEXT_G00A_DOES_NOT_PROVE, "vnext-g00a does_not_prove")

    freshness = require_object(child_manifest.get("freshness"), "vnext-g00a freshness")
    expected_freshness = {
        "catalogs_match_collector_head",
        "collector_checkout_clean",
        "inventory_candidate_matches_current_analyzer_recomputation",
        "inventory_frozen_source_clean",
        "model_resolution_catalog_matches_current_head",
        "model_resolution_input_matches_live_facts",
        "model_resolution_live_recomputed",
        "model_resolution_resolver_matches_current_head",
        "model_resolution_source_matches_collector_head",
    }
    require_gate(set(freshness) == expected_freshness, "vnext-g00a freshness field set mismatch")
    require_gate(all(value is True for value in freshness.values()), "vnext-g00a freshness contains a non-true value")

    index = validate_child_artifact_index(root, child_manifest, role_from_top_level_path=False)
    expected_artifacts = {
        "coupling-inventory.json",
        "generation-presets.catalog.json",
        "historical-bugs.catalog.json",
        "inventory-review.catalog.json",
        "model-facts.lock.json",
        "model-resolution.input.json",
        "model-resolution.json",
        "models.catalog.json",
    }
    require_gate(set(index) == expected_artifacts, "vnext-g00a artifact set mismatch")
    require_gate(child_manifest.get("artifact_count") == len(expected_artifacts), "vnext-g00a artifact_count mismatch")
    fact_sources = require_object(child_manifest.get("fact_source_artifacts"), "vnext-g00a fact_source_artifacts")
    expected_fact_sources = {
        "coupling_inventory": "coupling-inventory.json",
        "model_resolution_input": "model-resolution.input.json",
        "model_resolution_live": "model-resolution.json",
    }
    require_gate(set(fact_sources) == set(expected_fact_sources), "vnext-g00a fact source field set mismatch")
    for source_name, artifact_name in expected_fact_sources.items():
        source_ref = require_object(fact_sources.get(source_name), f"vnext-g00a fact source {source_name}")
        require_gate(source_ref == index[artifact_name], f"vnext-g00a fact source/index mismatch: {source_name}")

    collector = require_object(child_manifest.get("collector"), "vnext-g00a collector")
    collector_sha = require_git_sha(collector.get("git_sha"), "vnext-g00a collector.git_sha")
    collector_tree = require_git_sha(collector.get("git_tree_sha"), "vnext-g00a collector.git_tree_sha")
    require_gate(collector.get("dirty") is False and collector.get("status_short") == [], "vnext-g00a collector dirty state mismatch")
    require_gate(child_manifest.get("git_sha") == collector_sha, "vnext-g00a manifest/collector git SHA mismatch")
    require_gate(child_manifest.get("git_tree_sha") == collector_tree, "vnext-g00a manifest/collector tree mismatch")
    contract_rows = require_list(collector.get("contracts"), "vnext-g00a collector.contracts")
    require_gate(pretty_json_sha256(contract_rows) == require_sha256(collector.get("contracts_sha256"), "vnext-g00a collector.contracts_sha256"), "vnext-g00a collector contract-list digest mismatch")
    contracts: dict[str, dict[str, Any]] = {}
    for row_index, raw in enumerate(contract_rows):
        row = require_object(raw, f"vnext-g00a collector.contracts[{row_index}]")
        relative = require_string(row.get("path"), f"vnext-g00a collector.contracts[{row_index}].path")
        rel_path = Path(relative)
        require_gate(not rel_path.is_absolute() and rel_path.as_posix() == relative and ".." not in rel_path.parts, f"invalid vnext-g00a contract path: {relative}")
        require_gate(relative not in contracts, f"duplicate vnext-g00a contract path: {relative}")
        digest = require_sha256(row.get("sha256"), f"vnext-g00a collector contract {relative}.sha256")
        size = row.get("size_bytes")
        require_gate(isinstance(size, int) and not isinstance(size, bool) and size > 0, f"vnext-g00a collector contract {relative}.size_bytes invalid")
        git_blob = require_string(row.get("git_blob"), f"vnext-g00a collector contract {relative}.git_blob")
        require_gate(re.fullmatch(r"[0-9a-f]{40,64}", git_blob) is not None, f"vnext-g00a collector contract {relative}.git_blob invalid")
        contracts[relative] = {"sha256": digest, "size_bytes": size, "git_blob": git_blob}
    require_gate(set(contracts) == VNEXT_G00A_CONTRACT_PATHS, "vnext-g00a collector contract path set mismatch")
    catalog_contract_copies = {
        "generation-presets.catalog.json": "scripts/release/configs/runtime_vnext_generation_presets.json",
        "historical-bugs.catalog.json": "scripts/release/configs/runtime_vnext_historical_bugs.json",
        "inventory-review.catalog.json": "scripts/release/configs/runtime_vnext_inventory_review.json",
        "models.catalog.json": "scripts/release/configs/runtime_vnext_models.json",
    }
    for artifact_name, contract_path in catalog_contract_copies.items():
        require_gate(
            index[artifact_name].get("sha256") == contracts[contract_path]["sha256"],
            f"vnext-g00a copied catalog differs from collector contract: {artifact_name}",
        )
        require_gate(
            index[artifact_name].get("size_bytes") == contracts[contract_path]["size_bytes"],
            f"vnext-g00a copied catalog size differs from collector contract: {artifact_name}",
        )

    if verify_checkout:
        require_gate(git_sha() == collector_sha, "vnext-g00a collector SHA is stale against current HEAD")
        require_gate(git_output(["rev-parse", "HEAD^{tree}"]) == collector_tree, "vnext-g00a collector tree is stale against current HEAD")
        require_gate(not git_dirty_status()["is_dirty"], "vnext-g00a run_gate checkout must remain clean")
        for relative, identity in contracts.items():
            path = REPO_ROOT / relative
            require_gate(path.is_file() and not path.is_symlink(), f"vnext-g00a current contract missing: {relative}")
            require_gate(path.stat().st_size == identity["size_bytes"], f"vnext-g00a current contract size mismatch: {relative}")
            require_gate(sha256(path) == identity["sha256"], f"vnext-g00a current contract SHA256 mismatch: {relative}")
            require_gate(git_output(["rev-parse", f"HEAD:{relative}"]) == identity["git_blob"], f"vnext-g00a current contract Git blob mismatch: {relative}")

    frozen = require_object(child_manifest.get("frozen_source"), "vnext-g00a frozen_source")
    require_gate(require_git_sha(frozen.get("git_sha"), "vnext-g00a frozen_source.git_sha") == VNEXT_FROZEN_LEGACY_SHA, "vnext-g00a frozen legacy SHA mismatch")
    frozen_tree = require_git_sha(frozen.get("git_tree_sha"), "vnext-g00a frozen_source.git_tree_sha")
    if verify_checkout:
        require_gate(
            git_output(["rev-parse", f"{VNEXT_FROZEN_LEGACY_SHA}^{{tree}}"])
            == frozen_tree,
            "vnext-g00a frozen legacy tree mismatch",
        )

    lock_ref = require_object(child_manifest.get("model_facts_lock"), "vnext-g00a model_facts_lock")
    lock_path, lock_rel, lock_digest = require_indexed_artifact(
        root,
        index,
        lock_ref.get("path"),
        lock_ref.get("sha256"),
        "vnext-g00a model_facts_lock",
    )
    require_gate(lock_ref.get("size_bytes") == lock_path.stat().st_size, "vnext-g00a model_facts_lock size mismatch")
    lock = read_json_object(lock_path, "vnext-g00a model facts lock")
    require_gate(lock.get("schema_version") == 1, "vnext-g00a model facts lock schema mismatch")
    require_gate(lock.get("artifact_type") == "runtime_vnext_g00a_model_facts_lock", "vnext-g00a model facts lock artifact_type mismatch")
    require_gate(lock.get("checkpoint_id") == "G00a", "vnext-g00a model facts lock checkpoint mismatch")
    lock_scope = require_object(lock.get("scope"), "vnext-g00a model facts lock scope")
    require_exact_string_set(lock_scope.get("unlocks"), {"G01A"}, "vnext-g00a lock scope.unlocks")
    require_exact_string_set(lock_scope.get("does_not_prove"), VNEXT_G00A_DOES_NOT_PROVE, "vnext-g00a lock scope.does_not_prove")
    require_gate(lock_scope.get("historical_evidence") == "catalog_only", "vnext-g00a historical scope must be catalog_only")
    require_gate(lock.get("collector") == {"contracts_sha256": collector["contracts_sha256"], "git_sha": collector_sha, "git_tree_sha": collector_tree}, "vnext-g00a lock collector mismatch")
    require_gate(lock.get("frozen_legacy_source") == frozen, "vnext-g00a lock frozen source mismatch")

    def indexed_digest(relative: str) -> str:
        require_gate(relative in index, f"vnext-g00a missing indexed artifact: {relative}")
        return require_sha256(index[relative].get("sha256"), f"vnext-g00a index {relative}.sha256")

    model_catalog = require_object(lock.get("model_catalog"), "vnext-g00a lock model_catalog")
    require_gate(model_catalog.get("lane_count") == 12, "vnext-g00a model catalog lane_count must be 12")
    require_gate(model_catalog.get("catalog_sha256") == indexed_digest("models.catalog.json"), "vnext-g00a model catalog copy mismatch")
    presets = require_object(lock.get("generation_presets"), "vnext-g00a lock generation_presets")
    require_gate(presets.get("catalog_sha256") == indexed_digest("generation-presets.catalog.json"), "vnext-g00a generation preset catalog copy mismatch")
    preset_facts = require_object(presets.get("facts"), "vnext-g00a generation preset facts")
    preset_models = require_object(preset_facts.get("models"), "vnext-g00a generation preset models")
    require_gate(set(preset_models) == set(VNEXT_PRIMARY_MODELS), "vnext-g00a generation presets must cover exactly the three primary models")
    expected_presets = {"P_DETERMINISTIC", "P_NO_THINKING", "P_THINKING", "P_OFFICIAL_DEFAULT"}
    for model_key, raw in preset_models.items():
        model_presets = require_object(require_object(raw, f"vnext-g00a generation preset {model_key}").get("presets"), f"vnext-g00a generation preset {model_key}.presets")
        require_gate(set(model_presets) == expected_presets, f"vnext-g00a generation preset matrix mismatch: {model_key}")

    history = require_object(lock.get("historical_bug_catalog"), "vnext-g00a lock historical_bug_catalog")
    require_gate(history.get("catalog_sha256") == indexed_digest("historical-bugs.catalog.json"), "vnext-g00a historical catalog copy mismatch")
    history_facts = require_object(history.get("facts"), "vnext-g00a historical facts")
    require_gate(history_facts.get("catalog_scope") == "catalog_only" and history_facts.get("full_historical_corpus_complete") is False, "vnext-g00a historical facts overclaim corpus completion")
    require_gate(history_facts.get("family_count") == 15 and history_facts.get("concrete_case_count") == 28, "vnext-g00a historical fact counts mismatch")
    families = require_list(history_facts.get("families"), "vnext-g00a historical families")
    require_gate(len(families) == 15 and sum(len(require_list(require_object(family, "vnext-g00a historical family").get("cases"), "vnext-g00a historical family cases")) for family in families) == 28, "vnext-g00a historical family/case matrix mismatch")

    inventory = require_object(lock.get("inventory"), "vnext-g00a lock inventory")
    inventory_document = read_json_object(root / "coupling-inventory.json", "vnext-g00a coupling inventory")
    normalized_inventory = dict(inventory_document)
    normalized_inventory.pop("root", None)
    require_gate(pretty_json_sha256(normalized_inventory) == inventory.get("normalized_inventory_sha256"), "vnext-g00a normalized inventory digest mismatch")
    analyzer_contract = require_object(inventory.get("analyzer_contract"), "vnext-g00a inventory analyzer_contract")
    expected_analyzer_contract = next(row for row in contract_rows if row.get("path") == "scripts/release/runtime_vnext_inventory.py")
    require_gate(analyzer_contract == expected_analyzer_contract, "vnext-g00a inventory analyzer contract mismatch")
    analyzer = require_object(inventory_document.get("analyzer"), "vnext-g00a coupling inventory analyzer")
    require_gate(analyzer.get("path") == "scripts/release/runtime_vnext_inventory.py", "vnext-g00a coupling inventory analyzer path mismatch")
    review = require_object(inventory.get("review"), "vnext-g00a lock inventory review")
    require_gate(review.get("sha256") == indexed_digest("inventory-review.catalog.json"), "vnext-g00a inventory review copy mismatch")
    require_gate(review.get("unresolved_count") == 0, "vnext-g00a inventory review has unresolved classifications")

    resolution_ref = require_object(lock.get("model_resolution"), "vnext-g00a lock model_resolution")
    require_gate(resolution_ref.get("live_recomputed") is True, "vnext-g00a model resolution was not live-recomputed")
    require_sha256(resolution_ref.get("live_facts_sha256"), "vnext-g00a live model facts SHA256")
    resolution_path = root / "model-resolution.json"
    resolution = read_json_object(resolution_path, "vnext-g00a model resolution")
    require_gate(resolution.get("schema_version") == 1 and resolution.get("artifact_type") == "runtime_vnext_model_resolution", "vnext-g00a model resolution schema mismatch")
    require_gate(resolution.get("catalog_sha256") == indexed_digest("models.catalog.json"), "vnext-g00a model resolution catalog mismatch")
    require_gate(resolution.get("source") == resolution_ref.get("source"), "vnext-g00a model resolution source lock mismatch")
    require_gate(resolution.get("resolver") == resolution_ref.get("resolver"), "vnext-g00a model resolution resolver lock mismatch")
    resolution_source = require_object(resolution.get("source"), "vnext-g00a model resolution source")
    require_gate(resolution_source.get("git_sha") == collector_sha and resolution_source.get("dirty") is False and resolution_source.get("status_short") == [], "vnext-g00a model resolution source is stale or dirty")
    resolution_resolver = require_object(resolution.get("resolver"), "vnext-g00a model resolution resolver")
    require_gate(resolution_resolver.get("path") == "scripts/release/runtime_vnext_model_resolver.py", "vnext-g00a resolver path mismatch")
    require_gate(resolution_resolver.get("sha256") == contracts["scripts/release/runtime_vnext_model_resolver.py"]["sha256"], "vnext-g00a resolver contract SHA mismatch")
    resolution_policy = require_object(resolution.get("policy"), "vnext-g00a model resolution policy")
    require_gate(resolution_policy.get("transport") == "network_huggingface_https", "vnext-g00a model resolution transport mismatch")
    require_gate(resolution_policy.get("raw_response_body_kinds") == ["model-info", "repo-tree"], "vnext-g00a raw response body policy mismatch")
    requests = require_list(resolution.get("requests"), "vnext-g00a model resolution requests")
    require_gate(requests, "vnext-g00a model resolution has no live request provenance")
    request_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for request_index, raw in enumerate(requests):
        request = require_object(raw, f"vnext-g00a model resolution requests[{request_index}]")
        require_gate(request.get("method") == "GET", f"vnext-g00a request method mismatch at {request_index}")
        kind = require_string(request.get("kind"), f"vnext-g00a request[{request_index}].kind")
        url = require_string(request.get("url"), f"vnext-g00a request[{request_index}].url")
        require_gate(url.startswith("https://huggingface.co/"), f"vnext-g00a request URL mismatch at {request_index}")
        status = request.get("status")
        require_gate(isinstance(status, int) and not isinstance(status, bool) and 200 <= status < 300, f"vnext-g00a request status mismatch at {request_index}")
        response_bytes = request.get("response_bytes")
        require_gate(isinstance(response_bytes, int) and not isinstance(response_bytes, bool) and response_bytes > 0, f"vnext-g00a request size mismatch at {request_index}")
        require_sha256(request.get("response_sha256"), f"vnext-g00a request[{request_index}].response_sha256")
        key = (kind, url)
        require_gate(key not in request_lookup, f"vnext-g00a duplicate request provenance: {kind} {url}")
        request_lookup[key] = request
        if kind in {"model-info", "repo-tree"}:
            decoded_request_body(request, f"vnext-g00a request[{request_index}]")
        else:
            require_gate("response_body_base64" not in request, f"vnext-g00a metadata request unexpectedly embeds its body: {url}")

    def validate_live_source_response(raw_source: Any, label: str) -> None:
        source = require_object(raw_source, label)
        model_url = require_string(source.get("model_request_url"), f"{label}.model_request_url")
        model_request = require_object(request_lookup.get(("model-info", model_url)), f"{label} model request")
        model_body = require_object(decoded_request_body(model_request, f"{label} model request"), f"{label} model response")
        require_gate(model_body.get("sha") == source.get("revision"), f"{label} model response revision mismatch")
        tree_entries: dict[str, dict[str, Any]] = {}
        for tree_url_raw in require_list(source.get("tree_request_urls"), f"{label}.tree_request_urls"):
            tree_url = require_string(tree_url_raw, f"{label}.tree_request_url")
            tree_request = require_object(request_lookup.get(("repo-tree", tree_url)), f"{label} tree request")
            tree_body = require_list(decoded_request_body(tree_request, f"{label} tree request"), f"{label} tree response")
            for entry_index, raw_entry in enumerate(tree_body):
                entry = require_object(raw_entry, f"{label} tree response[{entry_index}]")
                if entry.get("type") not in {"file", None}:
                    continue
                path = require_string(entry.get("path"), f"{label} tree response[{entry_index}].path")
                require_gate(not Path(path).is_absolute() and ".." not in Path(path).parts, f"{label} unsafe tree path: {path}")
                require_gate(path not in tree_entries, f"{label} duplicate tree path: {path}")
                tree_entries[path] = entry
        license_files = require_list(require_object(source.get("license"), f"{label}.license").get("files"), f"{label}.license.files")
        for file_index, raw_file in enumerate([*require_list(source.get("files"), f"{label}.files"), *license_files]):
            file_row = require_object(raw_file, f"{label}.file[{file_index}]")
            path = require_string(file_row.get("path"), f"{label}.file[{file_index}].path")
            tree_entry = require_object(tree_entries.get(path), f"{label} tree fact {path}")
            require_gate(tree_entry.get("oid") == file_row.get("git_oid"), f"{label} tree Git OID mismatch: {path}")
            require_gate(tree_entry.get("size") == file_row.get("size_bytes"), f"{label} tree size mismatch: {path}")
            if file_row.get("sha256_source") == "hugging_face_lfs_oid":
                lfs = require_object(tree_entry.get("lfs"), f"{label} tree LFS fact {path}")
                lfs_oid = require_string(lfs.get("oid"), f"{label} tree LFS OID {path}").lower()
                if lfs_oid.startswith("sha256:"):
                    lfs_oid = lfs_oid.removeprefix("sha256:")
                require_gate(lfs_oid == file_row.get("sha256"), f"{label} tree LFS SHA256 mismatch: {path}")
                require_gate(lfs.get("size") == file_row.get("size_bytes"), f"{label} tree LFS size mismatch: {path}")
            else:
                content_url = require_string(file_row.get("content_request_url"), f"{label} metadata URL {path}")
                content_request = require_object(request_lookup.get(("metadata-file", content_url)), f"{label} metadata request {path}")
                require_gate(content_request.get("response_sha256") == file_row.get("sha256"), f"{label} metadata SHA256 mismatch: {path}")
                require_gate(content_request.get("response_bytes") == file_row.get("size_bytes"), f"{label} metadata size mismatch: {path}")

    resolved_lanes = require_list(resolution.get("lanes"), "vnext-g00a model resolution lanes")
    input_resolution = read_json_object(root / "model-resolution.input.json", "vnext-g00a input model resolution")
    require_gate(input_resolution.get("schema_version") == 1 and input_resolution.get("artifact_type") == "runtime_vnext_model_resolution", "vnext-g00a input model resolution schema mismatch")
    require_gate(input_resolution.get("catalog_sha256") == resolution.get("catalog_sha256"), "vnext-g00a input/live model catalog mismatch")
    input_source = require_object(input_resolution.get("source"), "vnext-g00a input model resolution source")
    require_gate(input_source.get("git_sha") == collector_sha and input_source.get("dirty") is False and input_source.get("status_short") == [], "vnext-g00a input model resolution source is stale or dirty")
    input_resolver = require_object(input_resolution.get("resolver"), "vnext-g00a input model resolution resolver")
    require_gate(input_resolver == resolution_resolver, "vnext-g00a input/live resolver identity mismatch")
    input_lanes = require_list(input_resolution.get("lanes"), "vnext-g00a input model resolution lanes")
    locked_lanes = require_list(lock.get("models"), "vnext-g00a locked model lanes")
    require_gate(len(input_lanes) == 12 and len(resolved_lanes) == 12 and len(locked_lanes) == 12, "vnext-g00a model lane count mismatch")
    require_gate(pretty_json_sha256(locked_lanes) == resolution_ref.get("live_facts_sha256"), "vnext-g00a live model facts digest mismatch")
    input_by_id = {require_string(require_object(row, "vnext-g00a input lane").get("catalog_lane_id"), "vnext-g00a input lane id"): row for row in input_lanes}
    resolved_by_id = {require_string(require_object(row, "vnext-g00a resolved lane").get("catalog_lane_id"), "vnext-g00a resolved lane id"): row for row in resolved_lanes}
    locked_by_id = {require_string(require_object(row, "vnext-g00a locked lane").get("catalog_lane_id"), "vnext-g00a locked lane id"): row for row in locked_lanes}
    require_gate(len(input_by_id) == 12 and len(resolved_by_id) == 12 and set(input_by_id) == set(resolved_by_id) == set(locked_by_id), "vnext-g00a model lane identity mismatch")
    expected_weight_identity_count = validate_vnext_catalog_expected_weight_facts(
        read_json_object(root / "models.catalog.json", "vnext-g00a models catalog"),
        resolved_by_id,
    )
    expected_pairs = {(model_id, backend) for model_id in VNEXT_RESOLUTION_MODEL_IDS.values() for backend in ("cuda", "metal")}
    actual_pairs: set[tuple[str, str]] = set()
    for lane_id in sorted(locked_by_id):
        locked_lane = require_object(locked_by_id[lane_id], f"vnext-g00a locked lane {lane_id}")
        resolved_lane = require_object(resolved_by_id[lane_id], f"vnext-g00a resolved lane {lane_id}")
        input_lane = require_object(input_by_id[lane_id], f"vnext-g00a input lane {lane_id}")
        for source_name in ("weight_source", "semantic_source", "tokenizer_source"):
            if resolved_lane.get(source_name) is not None:
                validate_live_source_response(
                    resolved_lane.get(source_name),
                    f"vnext-g00a live {lane_id}.{source_name}",
                )
        pair = (
            require_string(locked_lane.get("model_id"), f"vnext-g00a locked lane {lane_id}.model_id"),
            require_string(locked_lane.get("backend"), f"vnext-g00a locked lane {lane_id}.backend"),
        )
        actual_pairs.add(pair)
        require_gate(pair == (resolved_lane.get("model_id"), resolved_lane.get("backend")), f"vnext-g00a model/backend drift: {lane_id}")
        require_gate(pair == (input_lane.get("model_id"), input_lane.get("backend")), f"vnext-g00a input/live model/backend drift: {lane_id}")
        require_gate(locked_lane.get("format") == resolved_lane.get("format"), f"vnext-g00a format drift: {lane_id}")
        require_gate(input_lane.get("format") == resolved_lane.get("format"), f"vnext-g00a input/live format drift: {lane_id}")
        for source_name in ("weight_source", "semantic_source"):
            require_gate(normalized_model_source(locked_lane.get(source_name), f"vnext-g00a locked {lane_id}.{source_name}") == normalized_model_source(resolved_lane.get(source_name), f"vnext-g00a resolved {lane_id}.{source_name}"), f"vnext-g00a source drift: {lane_id}.{source_name}")
            require_gate(normalized_model_source(input_lane.get(source_name), f"vnext-g00a input {lane_id}.{source_name}") == normalized_model_source(resolved_lane.get(source_name), f"vnext-g00a live {lane_id}.{source_name}"), f"vnext-g00a input/live source drift: {lane_id}.{source_name}")
        if locked_lane.get("tokenizer_source") is None or resolved_lane.get("tokenizer_source") is None:
            require_gate(locked_lane.get("tokenizer_source") is None and resolved_lane.get("tokenizer_source") is None, f"vnext-g00a tokenizer source nullability drift: {lane_id}")
        else:
            require_gate(normalized_model_source(locked_lane.get("tokenizer_source"), f"vnext-g00a locked {lane_id}.tokenizer_source") == normalized_model_source(resolved_lane.get("tokenizer_source"), f"vnext-g00a resolved {lane_id}.tokenizer_source"), f"vnext-g00a tokenizer source drift: {lane_id}")
        if input_lane.get("tokenizer_source") is None or resolved_lane.get("tokenizer_source") is None:
            require_gate(input_lane.get("tokenizer_source") is None and resolved_lane.get("tokenizer_source") is None, f"vnext-g00a input/live tokenizer source nullability drift: {lane_id}")
        else:
            require_gate(normalized_model_source(input_lane.get("tokenizer_source"), f"vnext-g00a input {lane_id}.tokenizer_source") == normalized_model_source(resolved_lane.get("tokenizer_source"), f"vnext-g00a live {lane_id}.tokenizer_source"), f"vnext-g00a input/live tokenizer source drift: {lane_id}")
        for field in ("chat_template", "generation_config", "official_upstream"):
            require_gate(locked_lane.get(field) == resolved_lane.get(field), f"vnext-g00a resolved field drift: {lane_id}.{field}")
            require_gate(input_lane.get(field) == resolved_lane.get(field), f"vnext-g00a input/live field drift: {lane_id}.{field}")
    require_gate(actual_pairs == expected_pairs, "vnext-g00a exact six-model CUDA/Metal matrix mismatch")

    return {
        "kind": "vnext-g00a",
        "child_manifest": {"path": str(manifest_path), "sha256": require_sha256(child_manifest_sha256, "vnext-g00a delegated manifest SHA256")},
        "checkpoint": {"id": "G00a", "unlocks": ["G01A"], "does_not_prove": sorted(VNEXT_G00A_DOES_NOT_PROVE)},
        "collector": {"git_sha": collector_sha, "git_tree_sha": collector_tree, "contracts_sha256": collector["contracts_sha256"]},
        "model_facts_lock": {"path": lock_rel, "sha256": lock_digest},
        "model_lane_count": len(locked_lanes),
        "catalog_expected_weight_identity_count": expected_weight_identity_count,
        "historical_bug_counts": {"families": 15, "cases": 28},
        "artifact_index_sha256": canonical_json_sha256(child_manifest["artifact_index"]),
    }


def validate_vnext_g00_full_redteam(
    lane_command: LaneCommand,
    stdout: str,
) -> dict[str, Any]:
    require_gate(
        "--require-full-self-test" in lane_command.cmd,
        "vnext-g00 delegated command is missing --require-full-self-test",
    )
    lines = stdout.splitlines()
    require_gate(
        VNEXT_G00_FULL_SELFTEST_PASS in lines,
        f"vnext-g00 delegated command did not print exact FULL self-test PASS line: {VNEXT_G00_FULL_SELFTEST_PASS}",
    )
    summary_lines = [
        line.removeprefix(VNEXT_G00_SELFTEST_SUMMARY_PREFIX).strip()
        for line in lines
        if line.startswith(VNEXT_G00_SELFTEST_SUMMARY_PREFIX)
    ]
    require_gate(
        len(summary_lines) == 1,
        "vnext-g00 delegated command must print exactly one full-redteam summary",
    )
    try:
        summary_raw = json.loads(summary_lines[0])
    except json.JSONDecodeError as exc:
        raise GateError(f"vnext-g00 full-redteam summary is invalid JSON: {exc}") from exc
    summary = require_object(summary_raw, "vnext-g00 full-redteam summary")
    require_gate(summary.get("schema_version") == 1, "vnext-g00 full-redteam summary schema mismatch")
    require_gate(summary.get("mode") == "full", "vnext-g00 full-redteam summary mode must be full")
    mutation_count = summary.get("mutation_assertion_count")
    require_gate(
        mutation_count == VNEXT_G00_REDTEAM_MUTATION_COUNT,
        f"vnext-g00 full-redteam mutation count must be {VNEXT_G00_REDTEAM_MUTATION_COUNT}",
    )
    require_gate(
        summary.get("expected_mutation_assertion_count") == VNEXT_G00_REDTEAM_MUTATION_COUNT,
        "vnext-g00 full-redteam locked mutation count mismatch",
    )
    mutation_names = require_list(
        summary.get("mutation_names"),
        "vnext-g00 full-redteam mutation_names",
    )
    require_gate(
        len(mutation_names) == mutation_count
        and all(isinstance(name, str) and name for name in mutation_names),
        "vnext-g00 full-redteam mutation_names are incomplete or malformed",
    )
    require_gate(
        len(set(mutation_names)) == mutation_count,
        "vnext-g00 full-redteam mutation_names contain duplicates",
    )
    mutation_matrix_sha256 = canonical_json_sha256(mutation_names)
    require_gate(
        mutation_matrix_sha256 == VNEXT_G00_REDTEAM_MUTATION_MATRIX_SHA256,
        "vnext-g00 full-redteam mutation matrix SHA256 mismatch",
    )
    validator_counts = require_object(
        summary.get("validator_counts"),
        "vnext-g00 full-redteam validator_counts",
    )
    require_gate(
        validator_counts == {"full-root": mutation_count},
        "vnext-g00 full-redteam validator_counts.full-root must equal mutation count",
    )
    return {
        "pass_line": VNEXT_G00_FULL_SELFTEST_PASS,
        "summary": summary,
        "summary_sha256": canonical_json_sha256(summary),
        "mutation_matrix_sha256": mutation_matrix_sha256,
    }


def validate_vnext_g00_provenance(
    lane_command: LaneCommand,
    child_manifest: dict[str, Any],
    child_manifest_sha256: str,
    *,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    manifest_path = lane_command.child_manifest_path
    require_gate(manifest_path is not None, "vnext-g00 delegated manifest path is missing")
    root = manifest_path.parent.resolve()
    require_gate(manifest_path.resolve() == root / "manifest.json", "vnext-g00 delegated manifest must be <artifact_root>/manifest.json")
    manifest_digest = require_sha256(child_manifest_sha256, "vnext-g00 delegated manifest SHA256")
    artifact_dir = Path(require_string(child_manifest.get("artifact_dir"), "delegated manifest artifact_dir"))
    require_gate(artifact_dir.resolve() == root, "delegated manifest artifact_dir mismatch")
    require_gate(child_manifest.get("schema_version") == 1, "delegated manifest schema_version mismatch")
    require_gate(child_manifest.get("waiver_count") == 0, "delegated manifest waiver_count must be zero")
    validator_git_sha = require_git_sha(
        child_manifest.get("validator_git_sha"),
        "delegated manifest validator_git_sha",
    )
    require_gate(child_manifest.get("validator_dirty_status") == [], "delegated manifest validator must be clean")

    artifact_index = validate_child_artifact_index(root, child_manifest)
    models_lock_path, models_lock_rel, models_lock_digest = require_indexed_artifact(
        root,
        artifact_index,
        "models.lock.json",
        child_manifest.get("models_lock_sha256"),
        "models.lock",
    )
    models_lock = read_json_object(models_lock_path, "models.lock")
    require_gate(models_lock.get("schema_version") == 1, "models.lock schema_version mismatch")
    require_gate(models_lock.get("source_git_sha") == VNEXT_FROZEN_LEGACY_SHA, "models.lock frozen source SHA mismatch")
    expectations_binding = require_object(
        models_lock.get("expectations_catalog"),
        "models.lock.expectations_catalog",
    )
    expectations_path = require_string(
        expectations_binding.get("path"),
        "models.lock.expectations_catalog.path",
    )
    expectations_sha = require_sha256(
        expectations_binding.get("sha256"),
        "models.lock.expectations_catalog.sha256",
    )
    require_gate(
        expectations_path == VNEXT_LEGACY_EXPECTATIONS_PATH,
        "models.lock expectations catalog path mismatch",
    )
    expectations_file = REPO_ROOT / expectations_path
    require_gate(
        expectations_file.is_file() and sha256(expectations_file) == expectations_sha,
        "models.lock expectations catalog differs from the clean checkout",
    )
    require_gate(
        child_manifest.get("expectations_catalog") == expectations_binding,
        "delegated manifest expectations catalog mismatch",
    )
    contract_files = require_list(child_manifest.get("contract_files"), "delegated manifest contract_files")
    contract_by_path: dict[str, dict[str, Any]] = {}
    for contract_index, raw_contract in enumerate(contract_files):
        contract = require_object(raw_contract, f"delegated manifest contract_files[{contract_index}]")
        relative = require_string(contract.get("path"), f"delegated manifest contract_files[{contract_index}].path")
        require_gate(relative not in contract_by_path, f"duplicate delegated contract path: {relative}")
        contract_by_path[relative] = contract
    require_gate(
        require_sha256(
            require_object(
                contract_by_path.get(expectations_path),
                "delegated expectations catalog contract",
            ).get("sha256"),
            "delegated expectations catalog contract SHA256",
        )
        == expectations_sha,
        "delegated expectations catalog contract mismatch",
    )

    resolution_ref = require_object(models_lock.get("model_resolution"), "models.lock.model_resolution")
    resolution_path, resolution_rel, resolution_digest = require_indexed_artifact(
        root,
        artifact_index,
        resolution_ref.get("path"),
        resolution_ref.get("sha256"),
        "model-resolution",
    )
    require_gate(resolution_rel == "model-resolution.json", "model-resolution path must be model-resolution.json")
    resolution = read_json_object(resolution_path, "model-resolution")
    require_gate(resolution.get("schema_version") == 1, "model-resolution schema_version mismatch")
    require_gate(resolution.get("artifact_type") == "runtime_vnext_model_resolution", "model-resolution artifact_type mismatch")
    resolver_raw = require_object(resolution.get("resolver"), "model-resolution.resolver")
    resolver_identity = {
        "path": require_string(resolver_raw.get("path"), "model-resolution.resolver.path"),
        "sha256": require_sha256(resolver_raw.get("sha256"), "model-resolution.resolver.sha256"),
    }
    resolution_rows = require_list(resolution.get("lanes"), "model-resolution.lanes")
    resolution_lanes: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(resolution_rows):
        row = require_object(raw, f"model-resolution.lanes[{index}]")
        lane_id = require_string(row.get("catalog_lane_id"), f"model-resolution.lanes[{index}].catalog_lane_id")
        require_gate(lane_id not in resolution_lanes, f"duplicate model-resolution lane {lane_id}")
        resolution_lanes[lane_id] = row

    model_rows = require_list(models_lock.get("models"), "models.lock.models")
    expected_models = {**VNEXT_PRIMARY_MODELS, **VNEXT_SUPPLEMENTAL_MODELS}
    models: dict[str, dict[str, Any]] = {}
    model_identities: list[dict[str, Any]] = []
    expected_resolution_ids: set[str] = set()
    for index, raw in enumerate(model_rows):
        model = require_object(raw, f"models.lock.models[{index}]")
        key = require_string(model.get("key"), f"models.lock.models[{index}].key")
        require_gate(key in expected_models and key not in models, f"unknown or duplicate locked model {key}")
        role = require_string(model.get("role"), f"models[{key}].role")
        expected_role = "primary" if key in VNEXT_PRIMARY_MODELS else "supplemental"
        require_gate(role == expected_role, f"models[{key}].role mismatch")
        official_model_id = require_string(model.get("official_model_id"), f"models[{key}].official_model_id")
        require_gate(official_model_id == expected_models[key], f"models[{key}].official_model_id mismatch")
        lanes = require_object(model.get("lanes"), f"models[{key}].lanes")
        require_gate(set(lanes) == {"cuda", "metal"}, f"models[{key}] must contain CUDA and Metal lanes")
        normalized_lanes: dict[str, Any] = {}
        for backend in ("cuda", "metal"):
            lane = require_object(lanes[backend], f"models[{key}].lanes.{backend}")
            lane_id = require_string(lane.get("catalog_lane_id"), f"models[{key}].lanes.{backend}.catalog_lane_id")
            require_gate(lane_id in resolution_lanes, f"models[{key}].lanes.{backend} missing model-resolution lane")
            expected_resolution_ids.add(lane_id)
            resolved = resolution_lanes[lane_id]
            require_gate(resolved.get("backend") == backend, f"models[{key}].lanes.{backend} resolution backend mismatch")
            require_gate(resolved.get("model_id") == VNEXT_RESOLUTION_MODEL_IDS[key], f"models[{key}].lanes.{backend} resolution model id mismatch")
            require_gate(resolved.get("format") == lane.get("format"), f"models[{key}].lanes.{backend} resolution format mismatch")
            locked_weight = {
                "repo": require_string(lane.get("repo"), f"models[{key}].lanes.{backend}.repo"),
                "revision": require_git_sha(lane.get("revision"), f"models[{key}].lanes.{backend}.revision"),
                "files": normalized_file_locks(lane.get("files"), f"models[{key}].lanes.{backend}.files"),
            }
            require_gate(locked_weight == normalized_model_source(resolved.get("weight_source"), f"model-resolution.{lane_id}.weight_source"), f"models[{key}].lanes.{backend} weight identity differs from model-resolution")
            locked_semantic = normalized_model_source(lane.get("semantic_source"), f"models[{key}].lanes.{backend}.semantic_source")
            require_gate(locked_semantic == normalized_model_source(resolved.get("semantic_source"), f"model-resolution.{lane_id}.semantic_source"), f"models[{key}].lanes.{backend} semantic identity differs from model-resolution")
            locked_tokenizer = lane.get("tokenizer_source")
            resolved_tokenizer = resolved.get("tokenizer_source")
            require_gate((locked_tokenizer is None) == (resolved_tokenizer is None), f"models[{key}].lanes.{backend} tokenizer resolution presence mismatch")
            normalized_tokenizer = None
            if locked_tokenizer is not None:
                normalized_tokenizer = normalized_model_source(locked_tokenizer, f"models[{key}].lanes.{backend}.tokenizer_source")
                require_gate(normalized_tokenizer == normalized_model_source(resolved_tokenizer, f"model-resolution.{lane_id}.tokenizer_source"), f"models[{key}].lanes.{backend} tokenizer identity differs from model-resolution")
            lane_identity = {
                "catalog_lane_id": lane_id,
                "backend": backend,
                "format": require_string(lane.get("format"), f"models[{key}].lanes.{backend}.format"),
                "hardware_id": require_string(lane.get("hardware_id"), f"models[{key}].lanes.{backend}.hardware_id"),
                "weight_source": locked_weight,
                "semantic_source": locked_semantic,
                "tokenizer_source": normalized_tokenizer,
            }
            lane_identity["identity_sha256"] = canonical_json_sha256(lane_identity)
            normalized_lanes[backend] = lane_identity
        identity: dict[str, Any] = {
            "key": key,
            "official_model_id": official_model_id,
            "role": role,
            "lanes": normalized_lanes,
        }
        if role == "primary":
            presets = require_object(model.get("generation_presets"), f"models[{key}].generation_presets")
            require_gate(presets, f"models[{key}].generation_presets must not be empty")
            identity["generation_presets_sha256"] = canonical_json_sha256(presets)
        identity["identity_sha256"] = canonical_json_sha256(identity)
        model_identities.append(identity)
        models[key] = model
    require_gate(set(models) == set(expected_models), "models.lock model matrix is incomplete")
    require_gate(set(resolution_lanes) == expected_resolution_ids, "model-resolution lane matrix differs from models.lock")
    require_gate(child_manifest.get("primary_models") == sorted(VNEXT_PRIMARY_MODELS), "delegated manifest primary_models mismatch")
    require_gate(child_manifest.get("supplemental_models") == sorted(VNEXT_SUPPLEMENTAL_MODELS), "delegated manifest supplemental_models mismatch")

    binaries_path, binaries_rel, binaries_digest = require_indexed_artifact(
        root,
        artifact_index,
        "legacy-binaries.json",
        child_manifest.get("legacy_binaries_sha256"),
        "legacy-binaries",
    )
    binaries_doc = read_json_object(binaries_path, "legacy-binaries")
    require_gate(binaries_doc.get("source_git_sha") == VNEXT_FROZEN_LEGACY_SHA, "legacy-binaries frozen source SHA mismatch")
    binary_rows = require_list(binaries_doc.get("binaries"), "legacy-binaries.binaries")
    binary_identities: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(binary_rows):
        binary = require_object(raw, f"legacy-binaries.binaries[{index}]")
        backend = require_string(binary.get("backend"), f"legacy-binaries.binaries[{index}].backend")
        require_gate(backend in {"cuda", "metal"} and backend not in binary_identities, f"invalid or duplicate legacy binary backend {backend}")
        artifact, artifact_rel, digest = require_indexed_artifact(
            root,
            artifact_index,
            binary.get("artifact_binary"),
            binary.get("binary_sha256"),
            f"legacy-binaries.{backend}.artifact_binary",
        )
        require_gate(sha256(artifact) == digest, f"legacy-binaries.{backend} artifact digest mismatch")
        features = require_list(binary.get("cargo_features"), f"legacy-binaries.{backend}.cargo_features")
        require_gate(features and all(isinstance(item, str) and item for item in features), f"legacy-binaries.{backend}.cargo_features invalid")
        identity = {
            "backend": backend,
            "hardware_id": require_string(binary.get("hardware_id"), f"legacy-binaries.{backend}.hardware_id"),
            "artifact_binary": artifact_rel,
            "binary_sha256": digest,
            "cargo_features": features,
            "build_command": require_list(binary.get("build_command"), f"legacy-binaries.{backend}.build_command"),
        }
        identity["identity_sha256"] = canonical_json_sha256(identity)
        binary_identities[backend] = identity
    require_gate(set(binary_identities) == {"cuda", "metal"}, "legacy-binaries must contain CUDA and Metal identities")

    correctness = require_object(child_manifest.get("correctness_lanes"), "delegated manifest correctness_lanes")
    expected_correctness = {f"{key}/{backend}" for key in VNEXT_PRIMARY_MODELS for backend in ("cuda", "metal")}
    require_gate(set(correctness) == expected_correctness, "delegated correctness lane matrix mismatch")
    require_gate(all(status in {"pass", "blocked"} for status in correctness.values()), "delegated correctness lane status invalid")
    expected_correctness_status = {
        "m1-qwen35-4b/cuda": "pass",
        "m1-qwen35-4b/metal": "blocked",
        "m2-qwen35-35b-a3b/cuda": "pass",
        "m2-qwen35-35b-a3b/metal": "blocked",
        "m3-qwen3-30b-a3b/cuda": "pass",
        "m3-qwen3-30b-a3b/metal": "pass",
    }
    require_gate(correctness == expected_correctness_status, "delegated correctness status matrix mismatch")

    required_scenario_configs: set[str] = set()
    correctness_invocations: dict[str, dict[str, Any]] = {}
    scenario_runner_path = "scripts/release/runtime_vnext_baseline_scenarios.py"
    scenario_runner_sha = sha256(REPO_ROOT / scenario_runner_path)
    require_gate(scenario_runner_sha is not None, "scenario runner is missing from the clean checkout")
    for lane_key, status in sorted(correctness.items()):
        model_key, backend = lane_key.split("/", 1)
        lane_path, _, _ = require_indexed_artifact(
            root,
            artifact_index,
            f"correctness/{model_key}/{backend}/lane.json",
            artifact_index.get(f"correctness/{model_key}/{backend}/lane.json", {}).get("sha256"),
            f"correctness.{model_key}.{backend}.lane",
        )
        lane = read_json_object(lane_path, f"correctness.{model_key}.{backend}.lane")
        require_gate(lane.get("status") == status, f"correctness.{model_key}.{backend} status mismatch")
        model_lane = require_object(require_object(models[model_key]["lanes"], f"models[{model_key}].lanes")[backend], f"models[{model_key}].lanes.{backend}")
        expected_files = {row["path"]: row["sha256"] for row in normalized_file_locks(model_lane.get("files"), f"models[{model_key}].lanes.{backend}.files")}
        require_gate(lane.get("model_key") == model_key and lane.get("backend") == backend, f"correctness.{model_key}.{backend} model/backend identity mismatch")
        require_gate(lane.get("model_revision") == model_lane.get("revision"), f"correctness.{model_key}.{backend} model revision mismatch")
        require_gate(lane.get("model_files") == expected_files, f"correctness.{model_key}.{backend} model files mismatch")
        require_gate(lane.get("hardware_id") == model_lane.get("hardware_id"), f"correctness.{model_key}.{backend} hardware identity mismatch")
        require_gate(lane.get("binary_sha256") == binary_identities[backend]["binary_sha256"], f"correctness.{model_key}.{backend} binary identity mismatch")
        if status == "blocked":
            require_gate(lane.get("current_support") is False and lane.get("comparable") is False and lane.get("waiver") is False, f"correctness.{model_key}.{backend} blocked policy mismatch")
            for field in (
                "failure_class",
                "reason",
                "first_failure",
                "downstream_goal",
                "implementation_path",
                "acceptance_path",
                "downstream_acceptance_pass_line",
            ):
                require_string(lane.get(field), f"correctness.{model_key}.{backend}.{field}")
            attempted = require_list(lane.get("attempted_command"), f"correctness.{model_key}.{backend}.attempted_command")
            require_gate(attempted and Path(str(attempted[0])).name == "ferrum" and any(part in {"run", "serve"} for part in attempted[1:]), f"correctness.{model_key}.{backend} blocked attempt is not a product entrypoint")
            require_gate(isinstance(lane.get("attempted_returncode"), int) and lane.get("attempted_returncode") != 0, f"correctness.{model_key}.{backend} blocked returncode mismatch")
            failure_log = require_string(lane.get("failure_log"), f"correctness.{model_key}.{backend}.failure_log")
            require_gate(failure_log in artifact_index and (root / failure_log).stat().st_size > 0, f"correctness.{model_key}.{backend} blocked failure log missing")
            require_gate("scenario_report" not in lane and "pass_line" not in lane, f"correctness.{model_key}.{backend} blocked lane contains pass evidence")
            continue
        report_ref = require_object(lane.get("scenario_report"), f"correctness.{model_key}.{backend}.scenario_report")
        report_path, _, _ = require_indexed_artifact(
            root,
            artifact_index,
            report_ref.get("path"),
            report_ref.get("sha256"),
            f"correctness.{model_key}.{backend}.scenario_report",
        )
        report = read_json_object(report_path, f"correctness.{model_key}.{backend}.scenario_report")
        for field, expected in {
            "model_key": model_key,
            "backend": backend,
            "model_revision": model_lane.get("revision"),
            "model_files": expected_files,
            "hardware_id": model_lane.get("hardware_id"),
            "binary_sha256": binary_identities[backend]["binary_sha256"],
            "models_lock_sha256": models_lock_digest,
        }.items():
            require_gate(report.get(field) == expected, f"correctness.{model_key}.{backend}.scenario_report {field} mismatch")
        report_expectations = validate_vnext_g00_expectations_snapshot(
            root,
            artifact_index,
            report.get("expectations_catalog"),
            source_path=expectations_file,
            source_sha256=expectations_sha,
            label=f"correctness.{model_key}.{backend}.scenario_report.expectations_catalog",
        )
        require_gate(
            report.get("expectations_catalog_sha256") == report_expectations["sha256"],
            f"correctness.{model_key}.{backend} report expectations SHA mismatch",
        )
        validate_vnext_g00_runner_identity(
            report.get("runner"),
            scenario_runner_path=scenario_runner_path,
            scenario_runner_sha256=scenario_runner_sha,
            validator_git_sha=validator_git_sha,
            contract_by_path=contract_by_path,
            label=f"correctness.{model_key}.{backend}.scenario_report.runner",
            verify_checkout=verify_checkout,
        )
        invocation_ref = require_object(report.get("executor_invocation"), f"correctness.{model_key}.{backend}.executor_invocation")
        invocation_path, invocation_rel, invocation_digest = require_indexed_artifact(
            root,
            artifact_index,
            invocation_ref.get("path"),
            invocation_ref.get("sha256"),
            f"correctness.{model_key}.{backend}.executor_invocation",
        )
        invocation = read_json_object(invocation_path, f"correctness.{model_key}.{backend}.executor_invocation")
        require_gate(invocation.get("mode") == "canonical", f"correctness.{model_key}.{backend} executor mode mismatch")
        require_gate(invocation.get("runner_path") == scenario_runner_path and invocation.get("runner_sha256") == scenario_runner_sha, f"correctness.{model_key}.{backend} executor runner identity mismatch")
        correctness_invocations[lane_key] = {
            "path": invocation_rel,
            "sha256": invocation_digest,
            "runner_path": scenario_runner_path,
            "runner_sha256": scenario_runner_sha,
            "mode": "canonical",
        }
        config_ref = require_object(report.get("effective_config"), f"correctness.{model_key}.{backend}.effective_config")
        _, config_rel, _ = require_indexed_artifact(
            root,
            artifact_index,
            config_ref.get("path"),
            config_ref.get("sha256"),
            f"correctness.{model_key}.{backend}.effective_config",
        )
        required_scenario_configs.add(config_rel)

    require_gate(
        child_manifest.get("correctness_executor_invocations")
        == correctness_invocations,
        "delegated correctness executor invocation matrix mismatch",
    )

    config_refs: dict[str, dict[str, Any]] = {}

    def register_config(raw_path: Any, raw_digest: Any, referenced_by: str) -> None:
        path, relative, digest = require_indexed_artifact(
            root,
            artifact_index,
            raw_path,
            raw_digest,
            f"config reference from {referenced_by}",
        )
        require_gate(path.suffix.lower() == ".json", f"effective config must be JSON: {relative}")
        parsed = json.loads(path.read_text(encoding="utf-8"))
        require_gate(isinstance(parsed, (dict, list)) and bool(parsed), f"effective config is empty: {relative}")
        existing = config_refs.setdefault(relative, {"path": relative, "sha256": digest, "referenced_by": []})
        require_gate(existing["sha256"] == digest, f"effective config has conflicting hashes: {relative}")
        if referenced_by not in existing["referenced_by"]:
            existing["referenced_by"].append(referenced_by)

    def discover_config_refs(value: Any, referenced_by: str) -> None:
        if isinstance(value, dict):
            config = value.get("effective_config")
            if isinstance(config, str) and "effective_config_sha256" in value:
                register_config(config, value.get("effective_config_sha256"), referenced_by)
            elif isinstance(config, dict) and "path" in config and "sha256" in config:
                register_config(config.get("path"), config.get("sha256"), referenced_by)
            for child in value.values():
                discover_config_refs(child, referenced_by)
        elif isinstance(value, list):
            for child in value:
                discover_config_refs(child, referenced_by)

    for relative in sorted(artifact_index):
        if not relative.endswith(".json"):
            continue
        path = root / relative
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GateError(f"indexed JSON artifact is invalid: {relative}: {exc}") from exc
        discover_config_refs(value, relative)
    require_gate(required_scenario_configs <= set(config_refs), "scenario effective config hashes are absent from aggregated config provenance")
    require_gate(config_refs, "vnext-g00 config provenance is empty")

    config_identities = sorted(config_refs.values(), key=lambda row: row["path"])
    for row in config_identities:
        row["referenced_by"].sort()
    model_identities.sort(key=lambda row: row["key"])
    return {
        "kind": "vnext-g00",
        "child_manifest": {
            "path": "manifest.json",
            "sha256": manifest_digest,
            "artifact_count": len(artifact_index),
            "contract_sha256": require_sha256(child_manifest.get("contract_sha256"), "delegated manifest contract_sha256"),
        },
        "models_lock": {
            "path": models_lock_rel,
            "sha256": models_lock_digest,
            "catalog_sha256": require_sha256(models_lock.get("catalog_sha256"), "models.lock.catalog_sha256"),
            "preset_catalog_sha256": require_sha256(models_lock.get("preset_catalog_sha256"), "models.lock.preset_catalog_sha256"),
        },
        "expectations_catalog": {
            "path": expectations_path,
            "sha256": expectations_sha,
        },
        "model_resolution": {
            "path": resolution_rel,
            "sha256": resolution_digest,
            "lane_count": len(resolution_lanes),
            "resolver": resolver_identity,
        },
        "legacy_binaries": {
            "path": binaries_rel,
            "sha256": binaries_digest,
            "identities": [binary_identities[key] for key in sorted(binary_identities)],
        },
        "model_identities": model_identities,
        "config_artifacts": config_identities,
        "correctness_lanes": dict(sorted(correctness.items())),
        "correctness_executor_invocations": correctness_invocations,
        "artifact_index_sha256": canonical_json_sha256(child_manifest["artifact_index"]),
    }


def verify_child_pass_line(
    lane_command: LaneCommand,
    stdout: str,
    *,
    verify_checkout: bool = True,
) -> dict[str, Any] | None:
    expected = lane_command.expected_child_pass_line
    if expected is None:
        return None
    if expected not in stdout.splitlines():
        raise GateError(f"delegated command did not print required PASS line: {expected}")
    full_redteam = None
    if lane_command.provenance_kind == "vnext-g00":
        full_redteam = validate_vnext_g00_full_redteam(lane_command, stdout)
    if lane_command.child_manifest_path is None:
        return None
    try:
        child_manifest_bytes = lane_command.child_manifest_path.read_bytes()
        child_manifest = json.loads(child_manifest_bytes.decode("utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GateError(f"invalid delegated manifest {lane_command.child_manifest_path}: {exc}") from exc
    if not isinstance(child_manifest, dict):
        raise GateError(f"delegated manifest must be a JSON object: {lane_command.child_manifest_path}")
    if child_manifest.get("status") != "pass":
        raise GateError(f"delegated manifest status is not pass: {lane_command.child_manifest_path}")
    if child_manifest.get("pass_line") != expected:
        raise GateError(f"delegated manifest pass_line mismatch: {lane_command.child_manifest_path}")
    if (
        lane_command.expected_source_git_sha is not None
        and child_manifest.get("source_git_sha") != lane_command.expected_source_git_sha
    ):
        raise GateError(f"delegated manifest source_git_sha mismatch: {lane_command.child_manifest_path}")
    child_manifest_digest = hashlib.sha256(child_manifest_bytes).hexdigest()
    if lane_command.provenance_kind == "vnext-g00a":
        return validate_vnext_g00a_provenance(
            lane_command,
            child_manifest,
            child_manifest_digest,
            verify_checkout=verify_checkout,
        )
    if lane_command.provenance_kind == "vnext-g00":
        provenance = validate_vnext_g00_provenance(
            lane_command,
            child_manifest,
            child_manifest_digest,
            verify_checkout=verify_checkout,
        )
        require_gate(full_redteam is not None, "vnext-g00 full-redteam provenance is missing")
        provenance["full_redteam"] = full_redteam
        return provenance
    return {
        "kind": "delegated-manifest",
        "child_manifest": {
            "path": str(lane_command.child_manifest_path),
            "sha256": child_manifest_digest,
        },
    }


def run_child(
    cmd: list[str],
    out_dir: Path,
    timeout: int | None,
    *,
    prepare_out_dir: bool = True,
) -> subprocess.CompletedProcess[str]:
    if prepare_out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        if out_dir.exists():
            raise GateError(f"delegated command requires a fresh --out directory: {out_dir}")
    started = time.monotonic()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    duration = time.monotonic() - started
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_gate.child.stdout").write_text(proc.stdout, errors="replace")
    (out_dir / "run_gate.child.stderr").write_text(proc.stderr, errors="replace")
    (out_dir / "run_gate.child.command.json").write_text(
        json.dumps(
            {
                "cmd": cmd,
                "duration_sec": duration,
                "env_overrides": {"PYTHONDONTWRITEBYTECODE": "1"},
            },
            indent=2,
        )
        + "\n"
    )
    return proc


def pass_line(lane: str, out_dir: Path) -> str:
    return f"FERRUM GATE {lane} PASS: {out_dir}"


def child_execution_artifacts(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in (
        "run_gate.child.command.json",
        "run_gate.child.stdout",
        "run_gate.child.stderr",
    ):
        path = out_dir / name
        digest = sha256(path)
        if digest is None:
            continue
        rows.append(
            {
                "path": name,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def manifest(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    lane_command: LaneCommand | None,
    status: str,
    started_at: str,
    finished_at: str,
    duration_sec: float,
    child_returncode: int | None,
    child_pass_line: str | None,
    child_artifacts: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    binary_path = lane_command.binary_path if lane_command else None
    binary_sha = sha256(binary_path) if binary_path else None
    return {
        "schema_version": 1,
        "lane": args.lane,
        "status": status,
        "command_line": command_line(),
        "delegated_command_line": lane_command.cmd if lane_command else None,
        "child_returncode": child_returncode,
        "child_pass_line": child_pass_line,
        "child_artifacts": child_artifacts,
        "child_execution_artifacts": child_execution_artifacts(out_dir),
        "git_sha": git_sha(),
        "dirty_status": git_dirty_status(),
        "artifact_dir": str(out_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": duration_sec,
        "binary": {
            "path": str(binary_path) if binary_path else None,
            "sha256": binary_sha,
        },
        "model": lane_command.model if lane_command else args.model,
        "sanitized_env": sanitized_env_summary(),
        "pass_line": pass_line(args.lane, out_dir) if status == "pass" else None,
        "error": error,
    }


def list_lanes() -> None:
    for lane in LANES:
        print(lane)


def require_selftest(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_selftest_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    return subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)


def write_selftest_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def selftest_file_lock(seed: str, path: str) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": hashlib.sha256(seed.encode("utf-8")).hexdigest(),
        "size_bytes": 1024,
    }


def selftest_artifact_index(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in CHILD_INDEX_EXCLUDED:
            continue
        digest = sha256(path)
        require_selftest(digest is not None, f"missing selftest artifact digest: {relative}")
        rows.append(
            {
                "path": relative,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "role": relative.split("/", 1)[0] if "/" in relative else "root-manifest",
            }
        )
    return rows


def make_selftest_vnext_g00_artifact(root: Path) -> LaneCommand:
    root.mkdir(parents=True, exist_ok=True)
    models: list[dict[str, Any]] = []
    resolution_lanes: list[dict[str, Any]] = []
    all_models = {**VNEXT_PRIMARY_MODELS, **VNEXT_SUPPLEMENTAL_MODELS}
    for key, official_model_id in all_models.items():
        lanes: dict[str, Any] = {}
        for backend in ("cuda", "metal"):
            lane_id = f"{key}-{backend}"
            weight = {
                "repo": f"org/{key}-{backend}",
                "revision": hashlib.sha1(f"weight-{lane_id}".encode("utf-8")).hexdigest(),
                "files": [selftest_file_lock(f"weight-file-{lane_id}", f"{key}-{backend}.weights")],
            }
            semantic = {
                "repo": f"org/{key}-semantic",
                "revision": hashlib.sha1(f"semantic-{lane_id}".encode("utf-8")).hexdigest(),
                "files": [selftest_file_lock(f"semantic-file-{lane_id}", "config.json")],
            }
            lanes[backend] = {
                "catalog_lane_id": lane_id,
                "repo": weight["repo"],
                "revision": weight["revision"],
                "format": "safetensors" if backend == "cuda" else "gguf",
                "hardware_id": f"{backend}-selftest",
                "files": weight["files"],
                "semantic_source": semantic,
            }
            resolution_lanes.append(
                {
                    "catalog_lane_id": lane_id,
                    "backend": backend,
                    "model_id": VNEXT_RESOLUTION_MODEL_IDS[key],
                    "format": lanes[backend]["format"],
                    "weight_source": weight,
                    "semantic_source": semantic,
                }
            )
        model: dict[str, Any] = {
            "key": key,
            "official_model_id": official_model_id,
            "role": "primary" if key in VNEXT_PRIMARY_MODELS else "supplemental",
            "lanes": lanes,
        }
        if key in VNEXT_PRIMARY_MODELS:
            model["generation_presets"] = {
                "P_DETERMINISTIC": {"temperature": 0, "seed": 9271},
                "P_NO_THINKING": {"temperature": 0.7, "seed": 9271},
                "P_THINKING": {"temperature": 0.7, "seed": 9271},
                "P_OFFICIAL_DEFAULT": {"temperature": 0.7, "seed": 9271},
            }
        models.append(model)

    resolution_path = root / "model-resolution.json"
    write_selftest_json(
        resolution_path,
        {
            "schema_version": 1,
            "artifact_type": "runtime_vnext_model_resolution",
            "resolver": {
                "path": "scripts/release/runtime_vnext_model_resolver.py",
                "sha256": hashlib.sha256(b"selftest-resolver").hexdigest(),
            },
            "lanes": resolution_lanes,
        },
    )
    resolution_digest = sha256(resolution_path)
    require_selftest(resolution_digest is not None, "selftest model-resolution digest missing")
    expectations_file = REPO_ROOT / VNEXT_LEGACY_EXPECTATIONS_PATH
    expectations_digest = sha256(expectations_file)
    require_selftest(expectations_digest is not None, "selftest expectations catalog digest missing")
    expectations_binding = {
        "path": VNEXT_LEGACY_EXPECTATIONS_PATH,
        "sha256": expectations_digest,
    }
    expectations_snapshot_path = root / VNEXT_LEGACY_EXPECTATIONS_SNAPSHOT
    expectations_snapshot_path.write_bytes(expectations_file.read_bytes())
    expectations_snapshot = {
        "kind": "raw-json",
        "path": VNEXT_LEGACY_EXPECTATIONS_SNAPSHOT,
        "sha256": expectations_digest,
    }
    scenario_runner_relative = "scripts/release/runtime_vnext_baseline_scenarios.py"
    scenario_runner_path = REPO_ROOT / scenario_runner_relative
    scenario_runner_digest = sha256(scenario_runner_path)
    require_selftest(scenario_runner_digest is not None, "selftest scenario runner digest missing")
    selftest_validator_git_sha = "1" * 40
    selftest_runner_identity = {
        "path": scenario_runner_relative,
        "sha256": scenario_runner_digest,
        "git_sha": selftest_validator_git_sha,
        "source_tree_sha": "2" * 40,
        "git_blob_sha": "3" * 40,
        "dirty_status": {"is_dirty": False, "status_short": []},
    }
    models_lock_path = root / "models.lock.json"
    write_selftest_json(
        models_lock_path,
        {
            "schema_version": 1,
            "source_git_sha": VNEXT_FROZEN_LEGACY_SHA,
            "catalog_sha256": hashlib.sha256(b"selftest-model-catalog").hexdigest(),
            "preset_catalog_sha256": hashlib.sha256(b"selftest-preset-catalog").hexdigest(),
            "expectations_catalog": expectations_binding,
            "model_resolution": {"path": "model-resolution.json", "sha256": resolution_digest},
            "models": models,
        },
    )

    binary_rows: list[dict[str, Any]] = []
    binary_sha_by_backend: dict[str, str] = {}
    for backend in ("cuda", "metal"):
        binary_rel = f"binaries/{backend}/ferrum"
        binary_path = root / binary_rel
        binary_path.parent.mkdir(parents=True, exist_ok=True)
        binary_path.write_text(f"{backend} selftest binary\n", encoding="utf-8")
        digest = sha256(binary_path)
        require_selftest(digest is not None, f"selftest {backend} binary digest missing")
        binary_sha_by_backend[backend] = digest
        binary_rows.append(
            {
                "backend": backend,
                "hardware_id": f"{backend}-selftest",
                "artifact_binary": binary_rel,
                "binary_sha256": digest,
                "cargo_features": [backend],
                "build_command": ["cargo", "build", "--release", "--features", backend],
            }
        )
    binaries_path = root / "legacy-binaries.json"
    write_selftest_json(
        binaries_path,
        {
            "schema_version": 1,
            "source_git_sha": VNEXT_FROZEN_LEGACY_SHA,
            "binaries": binary_rows,
        },
    )

    model_by_key = {row["key"]: row for row in models}
    correctness: dict[str, str] = {}
    executor_invocations: dict[str, dict[str, Any]] = {}
    models_lock_digest = sha256(models_lock_path)
    require_selftest(models_lock_digest is not None, "selftest models.lock digest missing")
    for model_key in VNEXT_PRIMARY_MODELS:
        for backend in ("cuda", "metal"):
            lane = model_by_key[model_key]["lanes"][backend]
            correctness[f"{model_key}/{backend}"] = "pass"
            config_rel = f"correctness/{model_key}/{backend}/effective-config.json"
            write_selftest_json(
                root / config_rel,
                {
                    "schema_version": 1,
                    "model_key": model_key,
                    "backend": backend,
                    "typed_effective_config": {"max_tokens": 128},
                },
            )
            config_digest = sha256(root / config_rel)
            require_selftest(config_digest is not None, "selftest correctness config digest missing")
            invocation_rel = f"correctness/{model_key}/{backend}/executor-invocation.json"
            write_selftest_json(
                root / invocation_rel,
                {
                    "schema_version": 1,
                    "mode": "canonical",
                    "runner_path": scenario_runner_relative,
                    "runner_sha256": scenario_runner_digest,
                },
            )
            invocation_digest = sha256(root / invocation_rel)
            require_selftest(invocation_digest is not None, "selftest executor invocation digest missing")
            invocation_identity = {
                "path": invocation_rel,
                "sha256": invocation_digest,
                "runner_path": scenario_runner_relative,
                "runner_sha256": scenario_runner_digest,
                "mode": "canonical",
            }
            executor_invocations[f"{model_key}/{backend}"] = invocation_identity
            report_rel = f"correctness/{model_key}/{backend}/scenario-report.json"
            model_files = {row["path"]: row["sha256"] for row in lane["files"]}
            write_selftest_json(
                root / report_rel,
                {
                    "schema_version": 1,
                    "status": "pass",
                    "model_key": model_key,
                    "backend": backend,
                    "model_revision": lane["revision"],
                    "model_files": model_files,
                    "hardware_id": lane["hardware_id"],
                    "binary_sha256": binary_sha_by_backend[backend],
                    "models_lock_sha256": models_lock_digest,
                    "runner": copy.deepcopy(selftest_runner_identity),
                    "expectations_catalog": copy.deepcopy(expectations_snapshot),
                    "expectations_catalog_sha256": expectations_digest,
                    "executor_invocation": {
                        "path": invocation_rel,
                        "sha256": invocation_digest,
                    },
                    "effective_config": {
                        "path": config_rel,
                        "sha256": config_digest,
                    },
                },
            )
            report_digest = sha256(root / report_rel)
            require_selftest(report_digest is not None, "selftest scenario report digest missing")
            write_selftest_json(
                root / f"correctness/{model_key}/{backend}/lane.json",
                {
                    "schema_version": 1,
                    "status": "pass",
                    "model_key": model_key,
                    "backend": backend,
                    "model_revision": lane["revision"],
                    "model_files": model_files,
                    "hardware_id": lane["hardware_id"],
                    "binary_sha256": binary_sha_by_backend[backend],
                    "scenario_report": {"path": report_rel, "sha256": report_digest},
                },
            )
            if backend == "metal" and model_key in {
                "m1-qwen35-4b",
                "m2-qwen35-35b-a3b",
            }:
                correctness[f"{model_key}/{backend}"] = "blocked"
                executor_invocations.pop(f"{model_key}/{backend}")
                failure_log_rel = f"correctness/{model_key}/{backend}/blocked.log"
                (root / failure_log_rel).write_text("legacy model/backend unsupported\n")
                write_selftest_json(
                    root / f"correctness/{model_key}/{backend}/lane.json",
                    {
                        "schema_version": 1,
                        "status": "blocked",
                        "model_key": model_key,
                        "backend": backend,
                        "model_revision": lane["revision"],
                        "model_files": model_files,
                        "hardware_id": lane["hardware_id"],
                        "binary_sha256": binary_sha_by_backend[backend],
                        "current_support": False,
                        "comparable": False,
                        "waiver": False,
                        "failure_class": "legacy-model-backend-unsupported",
                        "reason": "selftest frozen unsupported lane",
                        "first_failure": "model load rejected before inference",
                        "downstream_goal": "G08A" if model_key == "m1-qwen35-4b" else "G08B",
                        "implementation_path": "vNext model migration",
                        "acceptance_path": "runtime-vNext model lane gate",
                        "downstream_acceptance_pass_line": "FERRUM RUNTIME VNEXT MODEL PASS: fixture",
                        "attempted_command": ["ferrum", "run", "--model", model_key],
                        "attempted_returncode": 1,
                        "failure_log": failure_log_rel,
                    },
                )

    workload_config_rel = "workloads/m3-qwen3-30b-a3b/cuda/random/c1.config.json"
    write_selftest_json(root / workload_config_rel, {"schema_version": 1, "seed": 9271, "concurrency": 1})
    workload_config_digest = sha256(root / workload_config_rel)
    require_selftest(workload_config_digest is not None, "selftest workload config digest missing")
    write_selftest_json(
        root / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
        {
            "schema_version": 1,
            "workload": {
                "effective_config": workload_config_rel,
                "effective_config_sha256": workload_config_digest,
            },
        },
    )

    artifact_index = selftest_artifact_index(root)
    pass_line_value = f"FERRUM RUNTIME VNEXT G00 BASELINE PASS: {root}"
    write_selftest_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "status": "pass",
            "source_git_sha": VNEXT_FROZEN_LEGACY_SHA,
            "validator_git_sha": selftest_validator_git_sha,
            "validator_dirty_status": [],
            "artifact_dir": str(root),
            "contract_sha256": hashlib.sha256(b"selftest-contract").hexdigest(),
            "contract_files": [
                {
                    "path": VNEXT_LEGACY_EXPECTATIONS_PATH,
                    "sha256": expectations_digest,
                    "size_bytes": expectations_file.stat().st_size,
                },
                {
                    "path": scenario_runner_relative,
                    "sha256": scenario_runner_digest,
                    "size_bytes": scenario_runner_path.stat().st_size,
                },
            ],
            "artifact_index": artifact_index,
            "artifact_count": len(artifact_index),
            "models_lock_sha256": models_lock_digest,
            "expectations_catalog": expectations_binding,
            "correctness_executor_invocations": executor_invocations,
            "legacy_binaries_sha256": sha256(binaries_path),
            "primary_models": sorted(VNEXT_PRIMARY_MODELS),
            "supplemental_models": sorted(VNEXT_SUPPLEMENTAL_MODELS),
            "correctness_lanes": correctness,
            "waiver_count": 0,
            "pass_line": pass_line_value,
        },
    )
    return LaneCommand(
        ["selftest", "--require-full-self-test"],
        expected_child_pass_line=pass_line_value,
        child_manifest_path=root / "manifest.json",
        expected_source_git_sha=VNEXT_FROZEN_LEGACY_SHA,
        provenance_kind="vnext-g00",
    )


def refresh_selftest_vnext_manifest(root: Path, *, sync_models_lock: bool = False, sync_binaries: bool = False) -> None:
    manifest_path = root / "manifest.json"
    doc = read_json_object(manifest_path, "selftest delegated manifest")
    doc["artifact_dir"] = str(root)
    doc["pass_line"] = f"FERRUM RUNTIME VNEXT G00 BASELINE PASS: {root}"
    if sync_models_lock:
        doc["models_lock_sha256"] = sha256(root / "models.lock.json")
    if sync_binaries:
        doc["legacy_binaries_sha256"] = sha256(root / "legacy-binaries.json")
    doc["artifact_index"] = selftest_artifact_index(root)
    doc["artifact_count"] = len(doc["artifact_index"])
    write_selftest_json(manifest_path, doc)


def selftest_vnext_lane(root: Path) -> LaneCommand:
    pass_line_value = f"FERRUM RUNTIME VNEXT G00 BASELINE PASS: {root}"
    return LaneCommand(
        ["selftest", "--require-full-self-test"],
        expected_child_pass_line=pass_line_value,
        child_manifest_path=root / "manifest.json",
        expected_source_git_sha=VNEXT_FROZEN_LEGACY_SHA,
        provenance_kind="vnext-g00",
    )


def selftest_vnext_full_summary(**updates: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": 1,
        "mode": "full",
        "mutation_assertion_count": VNEXT_G00_REDTEAM_MUTATION_COUNT,
        "expected_mutation_assertion_count": VNEXT_G00_REDTEAM_MUTATION_COUNT,
        "mutation_names": list(VNEXT_G00_REDTEAM_MUTATION_NAMES),
        "validator_counts": {"full-root": VNEXT_G00_REDTEAM_MUTATION_COUNT},
        "valid_fixture_assertion_count": 2,
    }
    summary.update(updates)
    return summary


def selftest_vnext_stdout(
    lane: LaneCommand,
    *,
    full_pass_line: str = VNEXT_G00_FULL_SELFTEST_PASS,
    summary: dict[str, Any] | None = None,
) -> str:
    require_selftest(lane.expected_child_pass_line is not None, "selftest vnext lane lacks PASS line")
    summary = summary or selftest_vnext_full_summary()
    return "\n".join(
        [
            f"{VNEXT_G00_SELFTEST_SUMMARY_PREFIX} {json.dumps(summary, sort_keys=True, separators=(',', ':'))}",
            full_pass_line,
            lane.expected_child_pass_line,
            "",
        ]
    )


def expect_vnext_provenance_reject(
    valid_root: Path,
    name: str,
    mutate: Any,
    marker: str,
) -> None:
    case = valid_root.parent / f"vnext-reject-{name}"
    shutil.copytree(valid_root, case)
    refresh_selftest_vnext_manifest(case)
    mutate(case)
    lane = selftest_vnext_lane(case)
    try:
        verify_child_pass_line(
            lane,
            selftest_vnext_stdout(lane),
            verify_checkout=False,
        )
    except GateError as exc:
        require_selftest(marker.lower() in str(exc).lower(), f"{name} rejected for unexpected reason: {exc}")
        return
    raise AssertionError(f"vnext provenance mutation {name} unexpectedly passed")


def mutate_selftest_json(path: Path, update: Any) -> None:
    doc = read_json_object(path, "selftest mutation input")
    update(doc)
    write_selftest_json(path, doc)


def tamper_selftest_vnext_invocation_mode(root: Path) -> None:
    lane_key = "m3-qwen3-30b-a3b/cuda"
    base = root / "correctness/m3-qwen3-30b-a3b/cuda"
    invocation_path = base / "executor-invocation.json"
    mutate_selftest_json(invocation_path, lambda data: data.update({"mode": "discover"}))
    invocation_digest = sha256(invocation_path)
    require_selftest(invocation_digest is not None, "tampered invocation digest missing")
    report_path = base / "scenario-report.json"
    mutate_selftest_json(
        report_path,
        lambda data: data["executor_invocation"].update({"sha256": invocation_digest}),
    )
    report_digest = sha256(report_path)
    require_selftest(report_digest is not None, "tampered scenario report digest missing")
    mutate_selftest_json(
        base / "lane.json",
        lambda data: data["scenario_report"].update({"sha256": report_digest}),
    )
    mutate_selftest_json(
        root / "manifest.json",
        lambda data: data["correctness_executor_invocations"][lane_key].update(
            {"sha256": invocation_digest, "mode": "discover"}
        ),
    )
    refresh_selftest_vnext_manifest(root)


def update_selftest_vnext_report(root: Path, lane_key: str, update: Any) -> None:
    model_key, backend = lane_key.split("/", 1)
    base = root / "correctness" / model_key / backend
    report_path = base / "scenario-report.json"
    mutate_selftest_json(report_path, update)
    report_digest = sha256(report_path)
    require_selftest(report_digest is not None, "updated scenario report digest missing")
    mutate_selftest_json(
        base / "lane.json",
        lambda data: data["scenario_report"].update({"sha256": report_digest}),
    )


def minimize_selftest_vnext_runner_identity(root: Path) -> None:
    lane_key = "m3-qwen3-30b-a3b/cuda"
    update_selftest_vnext_report(
        root,
        lane_key,
        lambda data: data.update(
            {
                "runner": {
                    "path": data["runner"]["path"],
                    "sha256": data["runner"]["sha256"],
                }
            }
        ),
    )
    refresh_selftest_vnext_manifest(root)


def tamper_selftest_vnext_runner_git_sha(root: Path) -> None:
    lane_key = "m3-qwen3-30b-a3b/cuda"
    update_selftest_vnext_report(
        root,
        lane_key,
        lambda data: data["runner"].update({"git_sha": "4" * 40}),
    )
    refresh_selftest_vnext_manifest(root)


def tamper_selftest_vnext_expectations_kind(root: Path) -> None:
    lane_key = "m3-qwen3-30b-a3b/cuda"
    update_selftest_vnext_report(
        root,
        lane_key,
        lambda data: data["expectations_catalog"].update({"kind": "source-contract"}),
    )
    refresh_selftest_vnext_manifest(root)


def tamper_selftest_vnext_expectations_snapshot(root: Path) -> None:
    snapshot_path = root / VNEXT_LEGACY_EXPECTATIONS_SNAPSHOT
    write_selftest_json(snapshot_path, {"schema_version": 1, "tampered": True})
    snapshot_digest = sha256(snapshot_path)
    require_selftest(snapshot_digest is not None, "tampered expectations snapshot digest missing")
    manifest = read_json_object(root / "manifest.json", "selftest delegated manifest")
    correctness = require_object(manifest.get("correctness_lanes"), "selftest correctness lanes")
    for lane_key, status in correctness.items():
        if status != "pass":
            continue
        update_selftest_vnext_report(
            root,
            lane_key,
            lambda data, digest=snapshot_digest: (
                data["expectations_catalog"].update({"sha256": digest}),
                data.update({"expectations_catalog_sha256": digest}),
            ),
        )
    refresh_selftest_vnext_manifest(root)


def selftest_g00a_response_request(kind: str, url: str, body: Any) -> dict[str, Any]:
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "kind": kind,
        "method": "GET",
        "response_body_base64": base64.b64encode(payload).decode("ascii"),
        "response_bytes": len(payload),
        "response_sha256": hashlib.sha256(payload).hexdigest(),
        "status": 200,
        "url": url,
    }


def set_selftest_g00a_response_body(request: dict[str, Any], body: Any) -> None:
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    request["response_body_base64"] = base64.b64encode(payload).decode("ascii")
    request["response_bytes"] = len(payload)
    request["response_sha256"] = hashlib.sha256(payload).hexdigest()


def refresh_selftest_g00a_manifest(root: Path) -> None:
    manifest_path = root / "manifest.json"
    doc = read_json_object(manifest_path, "G00a selftest manifest")
    doc["artifact_dir"] = str(root)
    doc["pass_line"] = f"FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT PASS: {root}"
    rows = selftest_artifact_index(root)
    indexed = {row["path"]: row for row in rows}
    doc["artifact_index"] = rows
    doc["artifact_count"] = len(rows)
    doc["fact_source_artifacts"] = {
        "coupling_inventory": copy.deepcopy(indexed["coupling-inventory.json"]),
        "model_resolution_input": copy.deepcopy(indexed["model-resolution.input.json"]),
        "model_resolution_live": copy.deepcopy(indexed["model-resolution.json"]),
    }
    lock_row = indexed["model-facts.lock.json"]
    doc["model_facts_lock"] = {
        "path": "model-facts.lock.json",
        "sha256": lock_row["sha256"],
        "size_bytes": lock_row["size_bytes"],
    }
    write_selftest_json(manifest_path, doc)


def selftest_g00a_lane(root: Path) -> LaneCommand:
    return LaneCommand(
        ["selftest"],
        expected_child_pass_line=f"FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT PASS: {root}",
        child_manifest_path=root / "manifest.json",
        provenance_kind="vnext-g00a",
    )


def make_selftest_vnext_g00a_artifact(root: Path) -> LaneCommand:
    root.mkdir(parents=True, exist_ok=True)
    selftest_catalog_lanes: list[dict[str, Any]] = []
    expected_identity_lane = "m2-qwen35-35b-a3b-metal"
    for model_key in {**VNEXT_PRIMARY_MODELS, **VNEXT_SUPPLEMENTAL_MODELS}:
        for backend in ("cuda", "metal"):
            lane_id = f"{model_key}-{backend}"
            seed = f"{lane_id}:weight_source"
            selector: dict[str, Any] = {"path": "weight_source.bin", "required": True}
            if lane_id == expected_identity_lane:
                selector.update(
                    {
                        "expected_size_bytes": 1024 + len(seed),
                        "expected_sha256": hashlib.sha256(f"file:{seed}".encode("utf-8")).hexdigest(),
                    }
                )
            selftest_catalog_lanes.append({"files": [selector], "id": lane_id})
    selftest_catalog_lanes.sort(key=lambda row: row["id"])
    require_selftest(len(selftest_catalog_lanes) == 12, "G00a selftest catalog must contain 12 lanes")
    catalog_payloads = {
        "generation-presets.catalog.json": {
            "catalog_id": "g00a-selftest-generation-presets",
            "model_catalog_id": "g00a-selftest-models",
            "schema_version": 1,
        },
        "historical-bugs.catalog.json": {
            "catalog_id": "g00a-selftest-history",
            "concrete_case_count": 28,
            "family_count": 15,
            "schema_version": 1,
        },
        "inventory-review.catalog.json": {
            "candidate_count": 1,
            "schema_version": 1,
            "unresolved_count": 0,
        },
        "models.catalog.json": {
            "catalog_id": "g00a-selftest-models",
            "lane_count": 12,
            "models": selftest_catalog_lanes,
            "schema_version": 1,
        },
    }
    for name, payload in catalog_payloads.items():
        write_selftest_json(root / name, payload)

    artifact_to_contract = {
        "generation-presets.catalog.json": "scripts/release/configs/runtime_vnext_generation_presets.json",
        "historical-bugs.catalog.json": "scripts/release/configs/runtime_vnext_historical_bugs.json",
        "inventory-review.catalog.json": "scripts/release/configs/runtime_vnext_inventory_review.json",
        "models.catalog.json": "scripts/release/configs/runtime_vnext_models.json",
    }
    contract_payload_by_path = {
        contract_path: (root / artifact_name).read_bytes()
        for artifact_name, contract_path in artifact_to_contract.items()
    }
    for contract_path in VNEXT_G00A_CONTRACT_PATHS - set(contract_payload_by_path):
        contract_payload_by_path[contract_path] = f"G00a selftest contract: {contract_path}\n".encode("utf-8")
    contract_rows = []
    for contract_path in sorted(VNEXT_G00A_CONTRACT_PATHS):
        payload = contract_payload_by_path[contract_path]
        contract_rows.append(
            {
                "git_blob": hashlib.sha1(payload).hexdigest(),
                "path": contract_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    contracts = {row["path"]: row for row in contract_rows}
    collector_sha = "1" * 40
    collector_tree = "2" * 40
    collector = {
        "contracts": contract_rows,
        "contracts_sha256": pretty_json_sha256(contract_rows),
        "dirty": False,
        "git_sha": collector_sha,
        "git_tree_sha": collector_tree,
        "status_short": [],
    }
    frozen = {
        "git_sha": VNEXT_FROZEN_LEGACY_SHA,
        "git_tree_sha": "3" * 40,
    }

    requests: dict[tuple[str, str], dict[str, Any]] = {}

    def model_source(lane_id: str, source_name: str) -> dict[str, Any]:
        seed = f"{lane_id}:{source_name}"
        repo = f"fixture/{lane_id.lower()}-{source_name.replace('_', '-')}"
        revision = hashlib.sha1(f"revision:{seed}".encode("utf-8")).hexdigest()
        file_path = f"{source_name}.bin"
        file_sha = hashlib.sha256(f"file:{seed}".encode("utf-8")).hexdigest()
        file_size = 1024 + len(seed)
        git_oid = hashlib.sha1(f"git-oid:{seed}".encode("utf-8")).hexdigest()
        file_row = {
            "git_oid": git_oid,
            "lfs_oid": file_sha,
            "path": file_path,
            "sha256": file_sha,
            "sha256_source": "hugging_face_lfs_oid",
            "size_bytes": file_size,
        }
        model_url = f"https://huggingface.co/api/models/{repo}/revision/{revision}"
        tree_url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&expand=true"
        requests[("model-info", model_url)] = selftest_g00a_response_request(
            "model-info",
            model_url,
            {"id": repo, "sha": revision},
        )
        requests[("repo-tree", tree_url)] = selftest_g00a_response_request(
            "repo-tree",
            tree_url,
            [
                {
                    "lfs": {"oid": f"sha256:{file_sha}", "size": file_size},
                    "oid": git_oid,
                    "path": file_path,
                    "size": file_size,
                    "type": "file",
                }
            ],
        )
        return {
            "files": [file_row],
            "gated": False,
            "license": {"files": [], "hugging_face_id": "apache-2.0"},
            "model_request_url": model_url,
            "repo": repo,
            "requested_revision": {"status": "pinned", "value": revision},
            "revision": revision,
            "tree_request_urls": [tree_url],
        }

    lanes: list[dict[str, Any]] = []
    for model_key, model_id in {**VNEXT_PRIMARY_MODELS, **VNEXT_SUPPLEMENTAL_MODELS}.items():
        for backend in ("cuda", "metal"):
            lane_id = f"{model_key}-{backend}"
            weight = model_source(lane_id, "weight_source")
            semantic = model_source(lane_id, "semantic_source")
            tokenizer = (
                model_source(lane_id, "tokenizer_source")
                if model_key == "llama31-8b-compat"
                else None
            )
            lanes.append(
                {
                    "backend": backend,
                    "catalog_lane_id": lane_id,
                    "chat_template": {
                        "content_sha256": hashlib.sha256(f"chat:{lane_id}".encode("utf-8")).hexdigest(),
                        "source": "tokenizer_source" if tokenizer is not None else "semantic_source",
                    },
                    "format": "safetensors" if backend == "cuda" else "gguf",
                    "generation_config": {
                        "policy": "fixture",
                        "sha256": hashlib.sha256(f"generation:{lane_id}".encode("utf-8")).hexdigest(),
                    },
                    "hardware_policy": f"{backend}-selftest",
                    "model_id": VNEXT_RESOLUTION_MODEL_IDS[model_key],
                    "official_upstream": (
                        {
                            "repo": "meta-llama/Llama-3.1-8B-Instruct",
                            "revision": "4" * 40,
                            "verification_method": "selftest",
                        }
                        if model_key == "llama31-8b-compat"
                        else None
                    ),
                    "role": "primary" if model_key in VNEXT_PRIMARY_MODELS else "supplemental",
                    "semantic_source": semantic,
                    "tokenizer_source": tokenizer,
                    "weight_source": weight,
                }
            )
    lanes.sort(key=lambda row: row["catalog_lane_id"])
    require_selftest(len(lanes) == 12, "G00a selftest lane matrix must contain 12 lanes")

    models_catalog_sha = sha256(root / "models.catalog.json")
    resolver_identity = {
        "path": "scripts/release/runtime_vnext_model_resolver.py",
        "sha256": contracts["scripts/release/runtime_vnext_model_resolver.py"]["sha256"],
    }
    resolution_source = {
        "dirty": False,
        "git_sha": collector_sha,
        "status_short": [],
    }
    live_resolution = {
        "artifact_type": "runtime_vnext_model_resolution",
        "catalog_id": "g00a-selftest-models",
        "catalog_sha256": models_catalog_sha,
        "lanes": copy.deepcopy(lanes),
        "policy": {
            "raw_response_body_kinds": ["model-info", "repo-tree"],
            "transport": "network_huggingface_https",
        },
        "requests": [requests[key] for key in sorted(requests)],
        "resolver": resolver_identity,
        "schema_version": 1,
        "source": resolution_source,
    }
    input_resolution = copy.deepcopy(live_resolution)
    write_selftest_json(root / "model-resolution.input.json", input_resolution)
    write_selftest_json(root / "model-resolution.json", live_resolution)

    preset_names = {
        "P_DETERMINISTIC",
        "P_NO_THINKING",
        "P_OFFICIAL_DEFAULT",
        "P_THINKING",
    }
    preset_facts = {
        "models": {
            model_key: {
                "presets": {
                    name: {"seed": 9271, "source": "G00a outer selftest"}
                    for name in sorted(preset_names)
                }
            }
            for model_key in sorted(VNEXT_PRIMARY_MODELS)
        }
    }
    history_families = []
    for family_index in range(1, 16):
        case_count = 2 if family_index <= 13 else 1
        history_families.append(
            {
                "cases": [
                    {"evidence_status": "bound", "id": f"H{family_index:02d}.{case_index}"}
                    for case_index in range(1, case_count + 1)
                ],
                "id": f"H{family_index:02d}",
            }
        )
    require_selftest(
        sum(len(family["cases"]) for family in history_families) == 28,
        "G00a selftest history matrix must contain 28 cases",
    )
    inventory_document = {
        "analyzer": {
            "identity_key": "sha256",
            "path": "scripts/release/runtime_vnext_inventory.py",
        },
        "git": {
            "dirty": False,
            "sha": VNEXT_FROZEN_LEGACY_SHA,
            "status_short": [],
            "tree_sha": frozen["git_tree_sha"],
        },
        "root": "/fixture/frozen-cff4",
        "schema_version": 1,
        "summary": {"coupling_finding_count": 1, "file_count": 1},
    }
    write_selftest_json(root / "coupling-inventory.json", inventory_document)
    normalized_inventory = dict(inventory_document)
    normalized_inventory.pop("root")
    lock = {
        "artifact_type": "runtime_vnext_g00a_model_facts_lock",
        "checkpoint_id": "G00a",
        "collector": {
            "contracts_sha256": collector["contracts_sha256"],
            "git_sha": collector_sha,
            "git_tree_sha": collector_tree,
        },
        "frozen_legacy_source": frozen,
        "generation_presets": {
            "catalog_sha256": sha256(root / "generation-presets.catalog.json"),
            "facts": preset_facts,
        },
        "historical_bug_catalog": {
            "catalog_sha256": sha256(root / "historical-bugs.catalog.json"),
            "facts": {
                "catalog_scope": "catalog_only",
                "concrete_case_count": 28,
                "families": history_families,
                "family_count": 15,
                "full_historical_corpus_complete": False,
            },
        },
        "inventory": {
            "analyzer_contract": contracts["scripts/release/runtime_vnext_inventory.py"],
            "normalized_inventory_sha256": pretty_json_sha256(normalized_inventory),
            "review": {
                "sha256": sha256(root / "inventory-review.catalog.json"),
                "unresolved_count": 0,
            },
        },
        "model_catalog": {
            "catalog_id": "g00a-selftest-models",
            "catalog_sha256": models_catalog_sha,
            "lane_count": 12,
        },
        "model_resolution": {
            "artifact_sha256": sha256(root / "model-resolution.json"),
            "artifact_size_bytes": (root / "model-resolution.json").stat().st_size,
            "input_artifact_sha256": sha256(root / "model-resolution.input.json"),
            "input_artifact_size_bytes": (root / "model-resolution.input.json").stat().st_size,
            "live_facts_sha256": pretty_json_sha256(lanes),
            "live_recomputed": True,
            "resolver": resolver_identity,
            "source": resolution_source,
        },
        "models": copy.deepcopy(lanes),
        "schema_version": 1,
        "scope": {
            "does_not_prove": sorted(VNEXT_G00A_DOES_NOT_PROVE),
            "historical_evidence": "catalog_only",
            "unlocks": ["G01A"],
        },
    }
    write_selftest_json(root / "model-facts.lock.json", lock)

    manifest = {
        "artifact_count": 0,
        "artifact_dir": str(root),
        "artifact_index": [],
        "artifact_index_policy": {
            "manifest_self_digest": "excluded-to-avoid-recursive-digest",
            "non_manifest_artifacts_indexed": True,
        },
        "artifact_type": "runtime_vnext_g00a_fact_checkpoint_manifest",
        "canonical": True,
        "checkpoint_id": "G00a",
        "collector": collector,
        "dirty": False,
        "does_not_prove": sorted(VNEXT_G00A_DOES_NOT_PROVE),
        "fact_source_artifacts": {},
        "freshness": {
            "catalogs_match_collector_head": True,
            "collector_checkout_clean": True,
            "inventory_candidate_matches_current_analyzer_recomputation": True,
            "inventory_frozen_source_clean": True,
            "model_resolution_catalog_matches_current_head": True,
            "model_resolution_input_matches_live_facts": True,
            "model_resolution_live_recomputed": True,
            "model_resolution_resolver_matches_current_head": True,
            "model_resolution_source_matches_collector_head": True,
        },
        "frozen_source": frozen,
        "git_sha": collector_sha,
        "git_tree_sha": collector_tree,
        "lane": "runtime-vnext-g00a",
        "model_facts_lock": {},
        "pass_line": f"FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT PASS: {root}",
        "schema_version": 1,
        "status": "pass",
        "unlocks": ["G01A"],
    }
    write_selftest_json(root / "manifest.json", manifest)
    refresh_selftest_g00a_manifest(root)
    return selftest_g00a_lane(root)


def expect_g00a_provenance_reject(
    valid_root: Path,
    name: str,
    mutate: Any,
    marker: str,
    *,
    refresh_after_mutation: bool = True,
) -> None:
    case = valid_root.parent / f"g00a-reject-{name}"
    shutil.copytree(valid_root, case)
    refresh_selftest_g00a_manifest(case)
    mutate(case)
    if refresh_after_mutation:
        refresh_selftest_g00a_manifest(case)
    manifest_path = case / "manifest.json"
    manifest = read_json_object(manifest_path, f"G00a selftest mutation {name}")
    lane = selftest_g00a_lane(case)
    digest = sha256(manifest_path)
    require_selftest(digest is not None, f"G00a selftest mutation {name} manifest digest missing")
    try:
        validate_vnext_g00a_provenance(
            lane,
            manifest,
            digest,
            verify_checkout=False,
        )
    except GateError as exc:
        require_selftest(marker.lower() in str(exc).lower(), f"G00a {name} rejected for unexpected reason: {exc}")
        return
    raise AssertionError(f"G00a provenance mutation {name} unexpectedly passed")


def mutate_g00a_live_lfs(root: Path) -> None:
    path = root / "model-resolution.json"
    doc = read_json_object(path, "G00a live LFS mutation")
    lane = doc["lanes"][0]
    source = lane["weight_source"]
    file_row = source["files"][0]
    replacement = "f" * 64
    file_row["sha256"] = replacement
    file_row["lfs_oid"] = replacement
    tree_url = source["tree_request_urls"][0]
    request = next(row for row in doc["requests"] if row["kind"] == "repo-tree" and row["url"] == tree_url)
    tree_body = decoded_request_body(request, "G00a live LFS mutation tree")
    tree_body[0]["lfs"]["oid"] = f"sha256:{replacement}"
    set_selftest_g00a_response_body(request, tree_body)
    write_selftest_json(path, doc)


def mutate_g00a_expected_sha_coherently(root: Path) -> None:
    lane_id = "m2-qwen35-35b-a3b-metal"
    replacement = "a" * 64
    resolution_documents: dict[str, dict[str, Any]] = {}
    for name in ("model-resolution.input.json", "model-resolution.json"):
        path = root / name
        doc = read_json_object(path, f"G00a coherent expected-SHA mutation {name}")
        lane = next(row for row in doc["lanes"] if row["catalog_lane_id"] == lane_id)
        source = lane["weight_source"]
        file_row = source["files"][0]
        file_row["sha256"] = replacement
        file_row["sha256_source"] = "hugging_face_lfs_oid"
        file_row["lfs_oid"] = replacement
        tree_url = source["tree_request_urls"][0]
        request = next(
            row
            for row in doc["requests"]
            if row["kind"] == "repo-tree" and row["url"] == tree_url
        )
        tree_body = decoded_request_body(request, f"G00a coherent expected-SHA mutation tree {name}")
        tree_body[0]["lfs"]["oid"] = f"sha256:{replacement}"
        set_selftest_g00a_response_body(request, tree_body)
        write_selftest_json(path, doc)
        resolution_documents[name] = doc

    lock_path = root / "model-facts.lock.json"
    lock = read_json_object(lock_path, "G00a coherent expected-SHA mutation lock")
    locked_lane = next(row for row in lock["models"] if row["catalog_lane_id"] == lane_id)
    locked_file = locked_lane["weight_source"]["files"][0]
    locked_file["sha256"] = replacement
    locked_file["sha256_source"] = "hugging_face_lfs_oid"
    locked_file["lfs_oid"] = replacement
    model_resolution = lock["model_resolution"]
    model_resolution["live_facts_sha256"] = pretty_json_sha256(lock["models"])
    model_resolution["artifact_sha256"] = sha256(root / "model-resolution.json")
    model_resolution["artifact_size_bytes"] = (root / "model-resolution.json").stat().st_size
    model_resolution["input_artifact_sha256"] = sha256(root / "model-resolution.input.json")
    model_resolution["input_artifact_size_bytes"] = (root / "model-resolution.input.json").stat().st_size
    require_selftest(
        resolution_documents["model-resolution.input.json"]["lanes"]
        == resolution_documents["model-resolution.json"]["lanes"]
        == lock["models"],
        "G00a coherent expected-SHA mutation did not keep input/live/lock facts aligned",
    )
    write_selftest_json(lock_path, lock)


def mutate_g00a_raw_tree_body(root: Path) -> None:
    path = root / "model-resolution.json"
    doc = read_json_object(path, "G00a raw tree mutation")
    request = next(row for row in doc["requests"] if row["kind"] == "repo-tree")
    tree_body = decoded_request_body(request, "G00a raw tree mutation body")
    tree_body[0]["lfs"]["oid"] = f"sha256:{'e' * 64}"
    set_selftest_g00a_response_body(request, tree_body)
    write_selftest_json(path, doc)


def make_selftest_release_summary_artifact(root: Path) -> None:
    for rel in [
        "source-unit/unit.gate.json",
        "source-metal/metal.gate.json",
        "source-cuda-full/g0_cuda4090_full.gate.json",
        "source-cuda-llama-dense/g0_cuda4090_llama_dense.gate.json",
        "metal-tarball/gate.json",
        "cuda-tarball/gate.json",
        "homebrew-metal/gate.json",
        "homebrew-cuda-fetch/gate.json",
    ]:
        write_selftest_json(root / rel, {"status": "pass"})


def make_selftest_completion_manifest(path: Path) -> None:
    root = path.parent
    artifacts = {}
    for name in [
        "metal_source_gate_artifact",
        "cuda_full_source_gate_artifact",
        "cuda_dense_source_gate_artifact",
        "metal_tarball_gate_artifact",
        "cuda_tarball_gate_artifact",
        "homebrew_metal_gate_artifact",
        "homebrew_cuda_fetch_gate_artifact",
    ]:
        artifact = root / "artifacts" / name
        artifact.mkdir(parents=True)
        artifacts[name] = str(artifact)
    write_selftest_json(
        path,
        {
            "git_sha": "selftest",
            "dirty_status": {"is_dirty": False, "status_short": []},
            "tag": "v0.0.0-selftest",
            "github_release_url": "https://example.invalid/selftest",
            "release_assets": [
                {
                    "name": "ferrum-selftest.tar.gz",
                    "sha256": "0" * 64,
                }
            ],
            "cargo_workspace_crates": [
                {
                    "name": "ferrum-cli",
                    "version": "0.0.0-selftest",
                    "crates_io_visible": True,
                }
            ],
            **artifacts,
        },
    )


def self_test() -> int:
    this_script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="ferrum-run-gate-selftest-") as tmp:
        root = Path(tmp)

        listed = run_selftest_command([sys.executable, str(this_script), "--list-lanes"])
        require_selftest(listed.returncode == 0, listed.stderr or listed.stdout)
        require_selftest(listed.stdout.splitlines() == list(LANES), listed.stdout)

        dry_out = root / "unit-dry-run"
        dry = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "unit",
                "--out",
                str(dry_out),
                "--dry-run",
            ]
        )
        require_selftest(dry.returncode == 0, dry.stderr or dry.stdout)
        dry_manifest = json.loads((dry_out / "gate.manifest.json").read_text())
        require_selftest(dry_manifest["status"] == "dry-run", dry_manifest)
        require_selftest(dry_manifest["lane"] == "unit", dry_manifest)
        require_selftest(
            dry_manifest["delegated_command_line"][0] == "scripts/release/g0_source_gate.sh",
            dry_manifest,
        )
        require_selftest(
            dry_manifest["child_pass_line"] == source_pass_line("unit", dry_out),
            dry_manifest,
        )
        in_tree_vnext_out = REPO_ROOT / (
            f".run-gate-vnext-g00-selftest-{os.getpid()}-{time.monotonic_ns()}"
        )
        in_tree_vnext = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "vnext-g00",
                "--out",
                str(in_tree_vnext_out),
                "--dry-run",
            ]
        )
        require_selftest(in_tree_vnext.returncode != 0, "in-tree vnext-g00 --out unexpectedly passed")
        require_selftest(
            "must resolve outside the Git source tree" in in_tree_vnext.stderr,
            in_tree_vnext.stderr or in_tree_vnext.stdout,
        )
        require_selftest(
            not in_tree_vnext_out.exists(),
            "rejected in-tree vnext-g00 --out created an artifact directory",
        )
        vnext_out = (root / "vnext-g00-dry-run").resolve()
        vnext = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "vnext-g00",
                "--out",
                str(vnext_out),
                "--dry-run",
            ]
        )
        require_selftest(vnext.returncode == 0, vnext.stderr or vnext.stdout)
        vnext_manifest = json.loads((vnext_out / "gate.manifest.json").read_text())
        require_selftest(vnext_manifest["status"] == "dry-run", vnext_manifest)
        require_selftest(vnext_manifest["lane"] == "vnext-g00", vnext_manifest)
        require_selftest(
            vnext_manifest["delegated_command_line"][1]
            == "scripts/release/runtime_vnext_baseline_gate.py",
            vnext_manifest,
        )
        require_selftest(
            "--require-full-self-test" in vnext_manifest["delegated_command_line"],
            vnext_manifest,
        )
        require_selftest(
            vnext_manifest["child_pass_line"]
            == f"FERRUM RUNTIME VNEXT G00 BASELINE PASS: {vnext_out}",
            vnext_manifest,
        )

        g00a_out = (root / "vnext-g00a-dry-run").resolve()
        g00a_inventory = root / "g00a-coupling-inventory.json"
        g00a_resolution = root / "g00a-model-resolution.json"
        g00a = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "vnext-g00a",
                "--coupling-inventory",
                str(g00a_inventory),
                "--model-resolution",
                str(g00a_resolution),
                "--out",
                str(g00a_out),
                "--dry-run",
            ]
        )
        require_selftest(g00a.returncode == 0, g00a.stderr or g00a.stdout)
        g00a_manifest = json.loads((g00a_out / "gate.manifest.json").read_text())
        require_selftest(g00a_manifest["status"] == "dry-run" and g00a_manifest["lane"] == "vnext-g00a", g00a_manifest)
        require_selftest(
            g00a_manifest["delegated_command_line"][1]
            == "scripts/release/runtime_vnext_g00a_checkpoint.py",
            g00a_manifest,
        )
        require_selftest(
            g00a_manifest["child_pass_line"]
            == f"FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT PASS: {g00a_out}",
            g00a_manifest,
        )

        g00a_provenance_root = root / "vnext-g00a-provenance"
        g00a_lane = make_selftest_vnext_g00a_artifact(g00a_provenance_root)
        g00a_child_manifest = read_json_object(
            g00a_provenance_root / "manifest.json",
            "G00a selftest child manifest",
        )
        g00a_manifest_digest = sha256(g00a_provenance_root / "manifest.json")
        require_selftest(g00a_manifest_digest is not None, "G00a selftest manifest digest missing")
        g00a_provenance = validate_vnext_g00a_provenance(
            g00a_lane,
            g00a_child_manifest,
            g00a_manifest_digest,
            verify_checkout=False,
        )
        require_selftest(g00a_provenance["kind"] == "vnext-g00a", str(g00a_provenance))
        require_selftest(g00a_provenance["model_lane_count"] == 12, str(g00a_provenance))
        require_selftest(
            g00a_provenance["catalog_expected_weight_identity_count"] == 1,
            str(g00a_provenance),
        )
        require_selftest(
            g00a_provenance["historical_bug_counts"] == {"families": 15, "cases": 28},
            str(g00a_provenance),
        )
        require_selftest(
            g00a_provenance["child_manifest"]["sha256"] == g00a_manifest_digest,
            str(g00a_provenance),
        )

        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "input-lfs",
            lambda case: mutate_selftest_json(
                case / "model-resolution.input.json",
                lambda data: data["lanes"][0]["weight_source"]["files"][0].update(
                    {"lfs_oid": "d" * 64, "sha256": "d" * 64}
                ),
            ),
            "input/live source drift",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "live-lfs",
            mutate_g00a_live_lfs,
            "source drift",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "catalog-expected-sha-coherent-forgery",
            mutate_g00a_expected_sha_coherently,
            "catalog expected sha256 mismatch",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "raw-tree-body",
            mutate_g00a_raw_tree_body,
            "tree lfs sha256 mismatch",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "lock-live-facts",
            lambda case: mutate_selftest_json(
                case / "model-facts.lock.json",
                lambda data: data["model_resolution"].update(
                    {"live_facts_sha256": "c" * 64}
                ),
            ),
            "live model facts digest mismatch",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "copied-catalog",
            lambda case: mutate_selftest_json(
                case / "generation-presets.catalog.json",
                lambda data: data.update({"tampered": True}),
            ),
            "copied catalog differs from collector contract",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "inventory-normalized-digest",
            lambda case: mutate_selftest_json(
                case / "model-facts.lock.json",
                lambda data: data["inventory"].update(
                    {"normalized_inventory_sha256": "b" * 64}
                ),
            ),
            "normalized inventory digest mismatch",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "manifest-scope",
            lambda case: mutate_selftest_json(
                case / "manifest.json",
                lambda data: data.update({"unlocks": ["G01B"]}),
            ),
            "unlocks",
            refresh_after_mutation=False,
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "manifest-index",
            lambda case: mutate_selftest_json(
                case / "manifest.json",
                lambda data: data["artifact_index"][0].update(
                    {"sha256": "a" * 64}
                ),
            ),
            "artifact_index",
            refresh_after_mutation=False,
        )

        vnext_provenance_root = root / "vnext-g00-provenance"
        vnext_lane = make_selftest_vnext_g00_artifact(vnext_provenance_root)
        provenance = verify_child_pass_line(
            vnext_lane,
            selftest_vnext_stdout(vnext_lane),
            verify_checkout=False,
        )
        require_selftest(provenance is not None and provenance["kind"] == "vnext-g00", str(provenance))
        require_selftest(
            provenance["child_manifest"]["sha256"] == sha256(vnext_provenance_root / "manifest.json"),
            str(provenance),
        )
        require_selftest(len(provenance["model_identities"]) == 6, str(provenance))
        require_selftest(len(provenance["legacy_binaries"]["identities"]) == 2, str(provenance))
        require_selftest(provenance["model_resolution"]["lane_count"] == 12, str(provenance))
        require_selftest(len(provenance["config_artifacts"]) >= 7, str(provenance))
        require_selftest(
            provenance["full_redteam"]["summary"]["mutation_assertion_count"]
            == VNEXT_G00_REDTEAM_MUTATION_COUNT,
            str(provenance),
        )
        valid_vnext_stdout = selftest_vnext_stdout(vnext_lane)
        invalid_redteam_outputs = [
            (
                "missing-full-line",
                valid_vnext_stdout.replace(VNEXT_G00_FULL_SELFTEST_PASS + "\n", ""),
                "exact FULL self-test PASS line",
            ),
            (
                "forged-full-line",
                valid_vnext_stdout.replace(
                    VNEXT_G00_FULL_SELFTEST_PASS,
                    VNEXT_G00_FULL_SELFTEST_PASS + " FORGED",
                ),
                "exact FULL self-test PASS line",
            ),
            (
                "missing-summary",
                "\n".join(
                    line
                    for line in valid_vnext_stdout.splitlines()
                    if not line.startswith(VNEXT_G00_SELFTEST_SUMMARY_PREFIX)
                )
                + "\n",
                "exactly one full-redteam summary",
            ),
            (
                "forged-summary-mode",
                selftest_vnext_stdout(
                    vnext_lane,
                    summary=selftest_vnext_full_summary(mode="fast"),
                ),
                "mode must be full",
            ),
            (
                "forged-summary-count",
                selftest_vnext_stdout(
                    vnext_lane,
                    summary=selftest_vnext_full_summary(
                        mutation_assertion_count=VNEXT_G00_REDTEAM_MUTATION_COUNT - 1
                    ),
                ),
                "mutation count must be",
            ),
            (
                "forged-summary-matrix",
                selftest_vnext_stdout(
                    vnext_lane,
                    summary=selftest_vnext_full_summary(
                        mutation_names=[
                            *VNEXT_G00_REDTEAM_MUTATION_NAMES[:-1],
                            "replacement-mutation-with-valid-count",
                        ]
                    ),
                ),
                "mutation matrix SHA256 mismatch",
            ),
        ]
        for name, bad_stdout, marker in invalid_redteam_outputs:
            try:
                verify_child_pass_line(
                    vnext_lane,
                    bad_stdout,
                    verify_checkout=False,
                )
                raise AssertionError(f"vnext-g00 {name} unexpectedly passed")
            except GateError as exc:
                require_selftest(marker in str(exc), f"{name}: {exc}")
        missing_flag_lane = LaneCommand(
            ["selftest"],
            expected_child_pass_line=vnext_lane.expected_child_pass_line,
            child_manifest_path=vnext_lane.child_manifest_path,
            expected_source_git_sha=vnext_lane.expected_source_git_sha,
            provenance_kind="vnext-g00",
        )
        try:
            verify_child_pass_line(
                missing_flag_lane,
                valid_vnext_stdout,
                verify_checkout=False,
            )
            raise AssertionError("vnext-g00 missing formal full-self-test flag unexpectedly passed")
        except GateError as exc:
            require_selftest("missing --require-full-self-test" in str(exc), str(exc))
        top_doc = manifest(
            args=argparse.Namespace(lane="vnext-g00", model=None),
            out_dir=vnext_provenance_root,
            lane_command=vnext_lane,
            status="pass",
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            duration_sec=1.0,
            child_returncode=0,
            child_pass_line=vnext_lane.expected_child_pass_line,
            child_artifacts=provenance,
            error=None,
        )
        require_selftest(top_doc["child_artifacts"] == provenance, str(top_doc))

        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "stdout-only-pass",
            lambda case: (case / "manifest.json").unlink(),
            "invalid delegated manifest",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "shallow-pass-manifest",
            lambda case: write_selftest_json(
                case / "manifest.json",
                {
                    "schema_version": 1,
                    "status": "pass",
                    "source_git_sha": VNEXT_FROZEN_LEGACY_SHA,
                    "validator_git_sha": "1" * 40,
                    "validator_dirty_status": [],
                    "artifact_dir": str(case),
                    "waiver_count": 0,
                    "pass_line": f"FERRUM RUNTIME VNEXT G00 BASELINE PASS: {case}",
                },
            ),
            "artifact_index",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "artifact-index-sha",
            lambda case: mutate_selftest_json(
                case / "manifest.json",
                lambda data: data["artifact_index"][0].update({"sha256": "0" * 64}),
            ),
            "artifact_index",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "models-lock-sha",
            lambda case: mutate_selftest_json(
                case / "manifest.json",
                lambda data: data.update({"models_lock_sha256": "0" * 64}),
            ),
            "models.lock",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "expectations-catalog-sha",
            lambda case: mutate_selftest_json(
                case / "models.lock.json",
                lambda data: data["expectations_catalog"].update({"sha256": "0" * 64}),
            ),
            "models.lock",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "minimal-runner-identity",
            minimize_selftest_vnext_runner_identity,
            "runner field set mismatch",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "runner-validator-sha",
            tamper_selftest_vnext_runner_git_sha,
            "git_sha differs from delegated validator",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "expectations-snapshot-kind",
            tamper_selftest_vnext_expectations_kind,
            "kind must be raw-json",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "expectations-snapshot-bytes",
            tamper_selftest_vnext_expectations_snapshot,
            "sha256 differs from models.lock source contract",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "executor-invocation-mode",
            tamper_selftest_vnext_invocation_mode,
            "executor mode mismatch",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "cuda-primary-blocked",
            lambda case: mutate_selftest_json(
                case / "manifest.json",
                lambda data: data["correctness_lanes"].update(
                    {"m1-qwen35-4b/cuda": "blocked"}
                ),
            ),
            "correctness status matrix",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "model-resolution-sha",
            lambda case: (
                mutate_selftest_json(
                    case / "model-resolution.json",
                    lambda data: data["lanes"][0]["weight_source"].update({"revision": "2" * 40}),
                ),
                refresh_selftest_vnext_manifest(case),
            ),
            "model-resolution",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "config-sha",
            lambda case: (
                mutate_selftest_json(
                    case / "correctness/m3-qwen3-30b-a3b/cuda/effective-config.json",
                    lambda data: data.update({"tampered": True}),
                ),
                refresh_selftest_vnext_manifest(case),
            ),
            "effective_config",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "binary-sha",
            lambda case: (
                (case / "binaries/cuda/ferrum").write_text("tampered CUDA binary\n", encoding="utf-8"),
                refresh_selftest_vnext_manifest(case),
            ),
            "legacy-binaries.cuda.artifact_binary",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "model-identity",
            lambda case: (
                mutate_selftest_json(
                    case / "models.lock.json",
                    lambda data: data["models"][0]["lanes"]["cuda"].update({"revision": "3" * 40}),
                ),
                refresh_selftest_vnext_manifest(case, sync_models_lock=True),
            ),
            "weight identity differs",
        )
        try:
            verify_child_pass_line(
                LaneCommand(["selftest"], expected_child_pass_line="SELFTEST PASS"),
                "no pass line here\n",
            )
            raise AssertionError("missing delegated PASS line unexpectedly passed")
        except GateError as exc:
            require_selftest("SELFTEST PASS" in str(exc), str(exc))

        delegated_manifest_path = root / "delegated-manifest.json"
        expected_delegated_pass = "DELEGATED GATE PASS: selftest"
        expected_source_sha = "1" * 40
        delegated_lane = LaneCommand(
            ["selftest"],
            expected_child_pass_line=expected_delegated_pass,
            child_manifest_path=delegated_manifest_path,
            expected_source_git_sha=expected_source_sha,
        )
        valid_delegated_manifest = {
            "status": "pass",
            "pass_line": expected_delegated_pass,
            "source_git_sha": expected_source_sha,
        }
        write_selftest_json(delegated_manifest_path, valid_delegated_manifest)
        verify_child_pass_line(delegated_lane, expected_delegated_pass + "\n")
        for field, value, marker in [
            ("status", "fail", "status is not pass"),
            ("pass_line", "WRONG PASS", "pass_line mismatch"),
            ("source_git_sha", "2" * 40, "source_git_sha mismatch"),
        ]:
            bad_manifest = dict(valid_delegated_manifest)
            bad_manifest[field] = value
            write_selftest_json(delegated_manifest_path, bad_manifest)
            try:
                verify_child_pass_line(delegated_lane, expected_delegated_pass + "\n")
                raise AssertionError(f"bad delegated manifest field {field} unexpectedly passed")
            except GateError as exc:
                require_selftest(marker in str(exc), str(exc))

        release_root = root / "release-root"
        make_selftest_release_summary_artifact(release_root)
        summary_out = root / "release-summary"
        summary = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "release-summary",
                "--release-root",
                str(release_root),
                "--out",
                str(summary_out),
            ]
        )
        require_selftest(summary.returncode == 0, summary.stderr or summary.stdout)
        require_selftest(
            f"FERRUM GATE release-summary PASS: {summary_out}" in summary.stdout,
            summary.stdout,
        )
        summary_manifest = json.loads((summary_out / "gate.manifest.json").read_text())
        require_selftest(summary_manifest["status"] == "pass", summary_manifest)
        require_selftest(summary_manifest["pass_line"], summary_manifest)
        execution_paths = {row["path"] for row in summary_manifest["child_execution_artifacts"]}
        require_selftest(
            execution_paths
            == {
                "run_gate.child.command.json",
                "run_gate.child.stdout",
                "run_gate.child.stderr",
            },
            summary_manifest,
        )
        for row in summary_manifest["child_execution_artifacts"]:
            require_selftest(row["sha256"] == sha256(summary_out / row["path"]), row)
        require_selftest(
            summary_manifest["child_pass_line"] == f"G0 RELEASE PASS: {release_root}",
            summary_manifest,
        )

        completion_manifest_path = root / "completion-manifest.json"
        make_selftest_completion_manifest(completion_manifest_path)
        completion_out = root / "release-complete"
        complete = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "release-complete",
                "--completion-manifest",
                str(completion_manifest_path),
                "--out",
                str(completion_out),
            ]
        )
        require_selftest(complete.returncode == 0, complete.stderr or complete.stdout)
        require_selftest(
            f"FERRUM GATE release-complete PASS: {completion_out}" in complete.stdout,
            complete.stdout,
        )
        complete_manifest = json.loads((completion_out / "gate.manifest.json").read_text())
        require_selftest(complete_manifest["status"] == "pass", complete_manifest)
        require_selftest(
            complete_manifest["child_pass_line"]
            == f"FERRUM RELEASE COMPLETION PASS: {completion_out}",
            complete_manifest,
        )
        require_selftest(
            (completion_out / "release_completion_gate.json").is_file(),
            "missing completion validator artifact",
        )
    print("FERRUM RUN GATE SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lane", nargs="?", choices=LANES)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--list-lanes", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--version")
    parser.add_argument("--asset-path", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--model")
    parser.add_argument("--model-name")
    parser.add_argument("--port", type=int)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--completion-manifest", type=Path)
    parser.add_argument("--coupling-inventory", type=Path)
    parser.add_argument("--model-resolution", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if args.list_lanes:
        list_lanes()
        return 0
    if args.lane is None:
        parser.error("lane is required unless --list-lanes is set")
    if args.out is None:
        parser.error("--out is required")

    out_dir = args.out.resolve() if args.lane.startswith("vnext-") else args.out
    if args.lane == "vnext-g00":
        try:
            require_external_vnext_g00_output(out_dir)
        except GateError as exc:
            print(f"FERRUM GATE {args.lane} FAIL: {out_dir}: {exc}", file=sys.stderr)
            return 1
    started_at = iso_now()
    start = time.monotonic()
    lane_command: LaneCommand | None = None
    child_returncode: int | None = None
    child_pass_line: str | None = None
    child_artifacts: dict[str, Any] | None = None
    status = "fail"
    error: str | None = None
    try:
        lane_command = build_lane_command(args, out_dir)
        if args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                out_dir / "gate.manifest.json",
                manifest(
                    args=args,
                    out_dir=out_dir,
                    lane_command=lane_command,
                    status="dry-run",
                    started_at=started_at,
                    finished_at=iso_now(),
                    duration_sec=time.monotonic() - start,
                    child_returncode=None,
                    child_pass_line=lane_command.expected_child_pass_line,
                    child_artifacts=None,
                    error=None,
                ),
            )
            print(" ".join(shlex.quote(part) for part in lane_command.cmd))
            return 0
        proc = run_child(
            lane_command.cmd,
            out_dir,
            args.timeout,
            prepare_out_dir=lane_command.provenance_kind != "vnext-g00a",
        )
        child_returncode = proc.returncode
        if proc.returncode != 0:
            error = f"delegated command failed rc={proc.returncode}"
            status = "fail"
        else:
            child_artifacts = verify_child_pass_line(lane_command, proc.stdout)
            child_pass_line = lane_command.expected_child_pass_line
            status = "pass"
    except (GateError, subprocess.TimeoutExpired) as exc:
        error = str(exc)
        status = "fail"
    finished_at = iso_now()
    doc = manifest(
        args=args,
        out_dir=out_dir,
        lane_command=lane_command,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=time.monotonic() - start,
        child_returncode=child_returncode,
        child_pass_line=child_pass_line,
        child_artifacts=child_artifacts,
        error=error,
    )
    write_json(out_dir / "gate.manifest.json", doc)
    if status == "pass":
        if args.lane.startswith("vnext-") and child_pass_line is not None:
            print(child_pass_line)
        print(doc["pass_line"])
        return 0
    print(f"FERRUM GATE {args.lane} FAIL: {out_dir}: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
