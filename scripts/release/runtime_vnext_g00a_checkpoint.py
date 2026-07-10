#!/usr/bin/env python3
"""Create the immutable Runtime vNext G00a fact checkpoint.

G00a is intentionally narrower than the full G00 baseline. It proves that the
frozen legacy inventory, model resolution, generation presets, and historical
bug catalog are mutually bound to checked-in collectors and catalogs. It only
unlocks G01A contract design work.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import fnmatch
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
INVENTORY_ANALYZER_PATH = REPO_ROOT / "scripts/release/runtime_vnext_inventory.py"
MODEL_RESOLVER_PATH = REPO_ROOT / "scripts/release/runtime_vnext_model_resolver.py"
MODELS_CATALOG_PATH = REPO_ROOT / "scripts/release/configs/runtime_vnext_models.json"
PRESETS_CATALOG_PATH = REPO_ROOT / "scripts/release/configs/runtime_vnext_generation_presets.json"
BUG_CATALOG_PATH = REPO_ROOT / "scripts/release/configs/runtime_vnext_historical_bugs.json"
INVENTORY_REVIEW_PATH = REPO_ROOT / "scripts/release/configs/runtime_vnext_inventory_review.json"
GOAL_ROOT = REPO_ROOT / "docs/goals/runtime-vnext-0.8.0-2026-07-10"

FROZEN_LEGACY_SHA = "cff4c47765ef3259b8a04890187d99c60da86394"
CHECKPOINT_ID = "G00a"
SCHEMA_VERSION = 1
PASS_PREFIX = "FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT SELFTEST PASS"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FAMILY_RE = re.compile(r"^H(0[1-9]|1[0-5])$")
CASE_RE = re.compile(r"^(H(?:0[1-9]|1[0-5]))\.([1-9][0-9]*)$")
SAFETENSORS_SHARD_RE = re.compile(
    r"-(\d{5,6})-of-(\d{5,6})\.safetensors$"
)

EXPECTED_MODEL_IDS = {
    "M1",
    "M2",
    "M3",
    "Qwen3-Coder-30B-A3B-Instruct",
    "DeepSeek-R1-0528-Qwen3-8B",
    "Llama-3.1-8B-Instruct",
}
PRIMARY_MODEL_KEYS = {
    "M1": "m1-qwen35-4b",
    "M2": "m2-qwen35-35b-a3b",
    "M3": "m3-qwen3-30b-a3b",
}
REQUIRED_PRESETS = {
    "P_DETERMINISTIC",
    "P_NO_THINKING",
    "P_THINKING",
    "P_OFFICIAL_DEFAULT",
}
PRESET_FIELDS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
    "seed",
    "max_tokens",
    "stop",
    "eos_token_ids",
    "enable_thinking",
    "template_kwargs",
    "source",
}
CONTRACT_PATHS = (
    SCRIPT_PATH,
    INVENTORY_ANALYZER_PATH,
    MODEL_RESOLVER_PATH,
    MODELS_CATALOG_PATH,
    PRESETS_CATALOG_PATH,
    BUG_CATALOG_PATH,
    INVENTORY_REVIEW_PATH,
    GOAL_ROOT / "GOAL.md",
    GOAL_ROOT / "G00_BASELINE.md",
    GOAL_ROOT / "G01_CORE_CONTRACTS.md",
    GOAL_ROOT / "MODEL_MATRIX.md",
)


class CheckpointError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointError(message)


def validate_safetensors_shard_paths(paths: set[str], label: str) -> bool:
    shards = sorted(path for path in paths if path.endswith(".safetensors"))
    matches = [(path, SAFETENSORS_SHARD_RE.search(path)) for path in shards]
    sharded = len(shards) > 1 or any(match is not None for _, match in matches)
    if not sharded:
        return False
    require(shards, f"{label} sharded safetensors set is empty")
    require(
        all(match is not None for _, match in matches),
        f"{label} sharded safetensors path lacks canonical numbering",
    )
    numbered = [
        (int(match.group(1)), int(match.group(2)), len(match.group(1)), len(match.group(2)))
        for _, match in matches
        if match is not None
    ]
    require(
        len({number_width for _, _, number_width, _ in numbered}) == 1,
        f"{label} sharded safetensors number width differs",
    )
    require(
        len({total_width for _, _, _, total_width in numbered}) == 1,
        f"{label} sharded safetensors total width differs",
    )
    totals = {total for _, total, _, _ in numbered}
    require(len(totals) == 1, f"{label} safetensors shards disagree on total count")
    total = totals.pop()
    require(total == len(numbered), f"{label} safetensors shard count differs from numbered total")
    require(
        {number for number, _, _, _ in numbered} == set(range(1, total + 1)),
        f"{label} safetensors shard numbering is incomplete",
    )
    return True


def require_object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a JSON array")
    return value


def require_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be a non-empty string")
    return value


def require_sha256(value: Any, label: str) -> str:
    digest = require_string(value, label).lower()
    require(SHA256_RE.fullmatch(digest) is not None, f"{label} must be a lowercase SHA256")
    return digest


def require_git_sha(value: Any, label: str) -> str:
    sha = require_string(value, label).lower()
    require(GIT_SHA_RE.fullmatch(sha) is not None, f"{label} must be a full lowercase Git SHA")
    return sha


def require_positive_int(value: Any, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"{label} must be a positive integer")
    return value


def require_safe_relative_path(value: Any, label: str) -> str:
    text = require_string(value, label)
    path = Path(text)
    require(not path.is_absolute(), f"{label} must be repository-relative")
    require(".." not in path.parts and "." not in path.parts, f"{label} escapes its root")
    require(path.as_posix() == text, f"{label} must use normalized POSIX separators")
    return text


def reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def strict_json_loads(payload: str) -> Any:
    return json.loads(
        payload,
        object_pairs_hook=unique_json_object,
        parse_constant=reject_json_constant,
    )


def request_json_body(request: dict[str, Any], label: str) -> Any:
    encoded = require_string(request.get("response_body_base64"), f"{label}.response_body_base64")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise CheckpointError(f"{label} response body is not canonical base64: {exc}") from exc
    require(len(payload) == request.get("response_bytes"), f"{label} response body size mismatch")
    require(bytes_sha256(payload) == request.get("response_sha256"), f"{label} response body SHA256 mismatch")
    try:
        return strict_json_loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CheckpointError(f"{label} response body is invalid JSON: {exc}") from exc


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = strict_json_loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CheckpointError(f"missing JSON artifact: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CheckpointError(f"cannot read JSON artifact {path}: {exc}") from exc
    return require_object(value, str(path))


def read_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CheckpointError(f"cannot parse JSON artifact {label}: {exc}") from exc
    return require_object(value, label)


def snapshot_regular_file(source: Path, destination: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise CheckpointError(f"cannot open {label} as a regular non-symlink file: {source}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file: {source}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    try:
        path_after = source.stat(follow_symlinks=False)
    except OSError as exc:
        raise CheckpointError(f"{label} path changed while it was being snapshotted: {source}: {exc}") from exc
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        f"{label} changed while it was being snapshotted",
    )
    require(
        not stat.S_ISLNK(path_after.st_mode)
        and (path_after.st_dev, path_after.st_ino) == (after.st_dev, after.st_ino),
        f"{label} path identity changed while it was being snapshotted",
    )
    require(len(payload) == after.st_size and payload, f"{label} snapshot is empty or truncated")
    destination.write_bytes(payload)
    require(destination.read_bytes() == payload, f"{label} snapshot write verification failed")
    return payload


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value))


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_fingerprint(value: Any) -> str:
    return bytes_sha256(canonical_bytes(value))


GIT_OVERRIDE_ENV_KEYS = {
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_DIR",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_WORK_TREE",
}


def clean_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key in GIT_OVERRIDE_ENV_KEYS or key == "GIT_CONFIG_COUNT" or key.startswith("GIT_CONFIG_KEY_") or key.startswith("GIT_CONFIG_VALUE_"):
            env.pop(key, None)
    for key in ("PYTHONHOME", "PYTHONINSPECT", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"):
        env.pop(key, None)
    return env


def run_command(argv: list[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        env=clean_subprocess_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def run_git_command(args: list[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return run_command(["git", "-C", str(cwd), *args], cwd=REPO_ROOT)


def git_text(args: list[str], *, cwd: Path = REPO_ROOT) -> str:
    proc = run_git_command(args, cwd=cwd)
    require(proc.returncode == 0, f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def git_bytes(args: list[str], *, cwd: Path = REPO_ROOT) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(cwd), *args],
        cwd=REPO_ROOT,
        env=clean_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(
        proc.returncode == 0,
        f"git {' '.join(args)} failed: {proc.stderr.decode('utf-8', errors='replace').strip()}",
    )
    return proc.stdout


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def require_external_input(path: Path, label: str) -> Path:
    require(path.exists(), f"{label} does not exist: {path}")
    require(not path.is_symlink(), f"{label} must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    require(resolved.is_file(), f"{label} must be a regular file: {resolved}")
    require(not is_within(resolved, REPO_ROOT.resolve()), f"{label} must be outside the Git source tree")
    return resolved


def require_external_output(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    require(not is_within(resolved, REPO_ROOT.resolve()), "--out must resolve outside the Git source tree")
    require(not resolved.exists(), f"--out already exists; use a fresh artifact directory: {resolved}")
    require(resolved != resolved.parent, "--out cannot be a filesystem root")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    require(not resolved.parent.is_symlink(), "--out parent must not be a symlink")
    return resolved


def first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            return f"{path}: key sets differ; only-left={sorted(left_keys - right_keys)}, only-right={sorted(right_keys - left_keys)}"
        for key in sorted(left):
            result = first_difference(left[key], right[key], f"{path}.{key}")
            if result is not None:
                return result
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            result = first_difference(left_item, right_item, f"{path}[{index}]")
            if result is not None:
                return result
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def collector_identity() -> dict[str, Any]:
    require(
        Path(git_text(["rev-parse", "--show-toplevel"])).resolve() == REPO_ROOT.resolve(),
        "collector Git top-level differs from the repository root",
    )
    status = git_text(["status", "--short", "--untracked-files=all"])
    require(not status, "collector checkout must be clean; commit every G00a contract before canonical collection")
    head = require_git_sha(git_text(["rev-parse", "HEAD"]), "collector HEAD")
    tree = require_git_sha(git_text(["rev-parse", "HEAD^{tree}"]), "collector tree")
    rows: list[dict[str, Any]] = []
    for path in sorted(CONTRACT_PATHS):
        require(path.is_file() and not path.is_symlink(), f"missing or symlinked G00a contract: {path}")
        relative = path.relative_to(REPO_ROOT).as_posix()
        tracked = run_git_command(["ls-files", "--error-unmatch", "--", relative])
        require(tracked.returncode == 0, f"G00a contract is not checked in at HEAD: {relative}")
        current = path.read_bytes()
        committed = git_bytes(["show", f"HEAD:{relative}"])
        require(current == committed, f"G00a contract differs from HEAD: {relative}")
        blob = git_text(["rev-parse", f"HEAD:{relative}"])
        require(re.fullmatch(r"[0-9a-f]{40,64}", blob) is not None, f"invalid Git blob identity for {relative}")
        rows.append(
            {
                "git_blob": blob,
                "path": relative,
                "sha256": bytes_sha256(current),
                "size_bytes": len(current),
            }
        )
    return {
        "git_sha": head,
        "git_tree_sha": tree,
        "dirty": False,
        "status_short": [],
        "contracts": rows,
        "contracts_sha256": json_fingerprint(rows),
    }


def snapshot_collector_contract(
    collector: dict[str, Any],
    relative: str,
    destination: Path,
) -> bytes:
    rows: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(require_list(collector.get("contracts"), "collector.contracts")):
        row = require_object(raw, f"collector.contracts[{index}]")
        path = require_string(row.get("path"), f"collector.contracts[{index}].path")
        require(path not in rows, f"duplicate collector contract path: {path}")
        rows[path] = row
    require(relative in rows, f"collector contract is not frozen: {relative}")
    identity = rows[relative]
    payload = git_bytes(["show", f"{collector['git_sha']}:{relative}"])
    require(bytes_sha256(payload) == identity.get("sha256"), f"collector contract SHA256 mismatch: {relative}")
    require(len(payload) == identity.get("size_bytes") and payload, f"collector contract size mismatch: {relative}")
    require(
        git_text(["rev-parse", f"{collector['git_sha']}:{relative}"]) == identity.get("git_blob"),
        f"collector contract Git blob mismatch: {relative}",
    )
    destination.write_bytes(payload)
    require(destination.read_bytes() == payload, f"collector contract snapshot write failed: {relative}")
    return payload


def frozen_source_identity() -> dict[str, str]:
    object_type = git_text(["cat-file", "-t", FROZEN_LEGACY_SHA])
    require(object_type == "commit", f"frozen legacy object is not a commit: {FROZEN_LEGACY_SHA}")
    tree = require_git_sha(git_text(["rev-parse", f"{FROZEN_LEGACY_SHA}^{{tree}}"]), "frozen tree")
    return {"git_sha": FROZEN_LEGACY_SHA, "git_tree_sha": tree}


def normalize_inventory(value: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(value)
    normalized.pop("root", None)
    return normalized


def validate_inventory_shape(data: dict[str, Any], frozen: dict[str, str]) -> None:
    require(data.get("schema_version") == 1, "coupling inventory schema_version must be 1")
    analyzer = require_object(data.get("analyzer"), "coupling-inventory.analyzer")
    require(analyzer.get("path") == "scripts/release/runtime_vnext_inventory.py", "coupling inventory analyzer path mismatch")
    require(analyzer.get("identity_key") == "sha256", "coupling inventory identity key must be sha256")
    require_string(data.get("root"), "coupling-inventory.root")
    source = require_object(data.get("git"), "coupling-inventory.git")
    require(source.get("sha") == frozen["git_sha"], "coupling inventory source SHA is not frozen cff4")
    require(source.get("tree_sha") == frozen["git_tree_sha"], "coupling inventory source tree mismatch")
    require(source.get("dirty") is False and source.get("status_short") == [], "coupling inventory source must be clean")
    scope = require_object(data.get("scope"), "coupling-inventory.scope")
    require(scope.get("scan_roots") == ["crates", "scripts"], "coupling inventory scan roots mismatch")
    require(scope.get("coverage_ratio") == 1.0, "coupling inventory coverage must be 1.0")
    require(scope.get("discovered_file_count") == scope.get("inventoried_file_count"), "coupling inventory has uncovered files")
    files = require_list(data.get("files"), "coupling-inventory.files")
    require(files, "coupling inventory contains no files")
    seen: set[str] = set()
    for index, raw in enumerate(files):
        row = require_object(raw, f"coupling-inventory.files[{index}]")
        path = require_safe_relative_path(row.get("path"), f"coupling-inventory.files[{index}].path")
        require(path not in seen, f"duplicate coupling inventory path: {path}")
        seen.add(path)
        digest = require_sha256(row.get("sha256"), f"coupling-inventory.files[{index}].sha256")
        require(row.get("content_id") == f"sha256:{digest}", f"coupling-inventory.files[{index}].content_id mismatch")
        require(isinstance(row.get("size_bytes"), int) and row["size_bytes"] >= 0, f"coupling-inventory.files[{index}].size_bytes invalid")
        require(isinstance(row.get("logical_loc"), int) and row["logical_loc"] >= 0, f"coupling-inventory.files[{index}].logical_loc invalid")
    require(len(files) == scope.get("inventoried_file_count"), "coupling inventory file count mismatch")
    summary = require_object(data.get("summary"), "coupling-inventory.summary")
    require(summary.get("file_count") == len(files), "coupling inventory summary file count mismatch")
    findings = require_list(require_object(data.get("coupling"), "coupling-inventory.coupling").get("findings"), "coupling-inventory.coupling.findings")
    require(summary.get("coupling_finding_count") == len(findings), "coupling inventory finding count mismatch")
    native = require_list(data.get("large_native_source_trees"), "coupling-inventory.large_native_source_trees")
    require(summary.get("native_source_tree_count") == len(native), "coupling inventory native source count mismatch")


def recompute_inventory(frozen: dict[str, str], collector: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ferrum-g00a-inventory-") as raw_tmp:
        tmp = Path(raw_tmp)
        collector_worktree = tmp / "collector-source"
        worktree = tmp / "frozen-source"
        output = tmp / "recomputed-inventory.json"
        added: list[Path] = []
        try:
            for path, sha, label in (
                (collector_worktree, collector["git_sha"], "collector"),
                (worktree, frozen["git_sha"], "frozen cff4"),
            ):
                add = run_git_command(["worktree", "add", "--detach", str(path), sha])
                require(add.returncode == 0, f"cannot create {label} worktree: {add.stderr.strip()}")
                added.append(path)
            proc = run_command(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    str(collector_worktree / "scripts/release/runtime_vnext_inventory.py"),
                    "--root",
                    str(worktree),
                    "--out",
                    str(output),
                ]
            )
            require(proc.returncode == 0, f"current inventory analyzer failed on frozen cff4: {proc.stderr.strip()}\n{proc.stdout.strip()}")
            require(proc.stdout.strip().endswith(f"RUNTIME VNEXT INVENTORY PASS: {output}"), "inventory analyzer did not print its exact PASS line")
            recomputed = read_json(output)
            validate_inventory_shape(recomputed, frozen)
            return recomputed
        finally:
            for path in reversed(added):
                remove = run_git_command(["worktree", "remove", "--force", str(path)])
                if remove.returncode != 0:
                    run_git_command(["worktree", "prune"])


def compare_inventory(candidate: dict[str, Any], recomputed: dict[str, Any]) -> str:
    left = normalize_inventory(candidate)
    right = normalize_inventory(recomputed)
    difference = first_difference(left, right)
    require(difference is None, f"coupling inventory differs from current-analyzer recomputation: {difference}")
    return json_fingerprint(left)


def recompute_model_resolution(output: Path, collector: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ferrum-g00a-model-resolution-") as raw_tmp:
        worktree = Path(raw_tmp) / "collector-source"
        added = False
        try:
            add = run_git_command(["worktree", "add", "--detach", str(worktree), collector["git_sha"]])
            require(add.returncode == 0, f"cannot create model resolver collector worktree: {add.stderr.strip()}")
            added = True
            proc = run_command(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    str(worktree / "scripts/release/runtime_vnext_model_resolver.py"),
                    "--out",
                    str(output),
                ],
                cwd=worktree,
            )
            require(
                proc.returncode == 0,
                f"live model resolver failed: {proc.stderr.strip()}\n{proc.stdout.strip()}",
            )
            require(
                proc.stdout.strip().endswith(f"RUNTIME VNEXT MODEL RESOLUTION PASS: {output}"),
                "live model resolver did not print its exact PASS line",
            )
            return read_json(output)
        finally:
            if added:
                remove = run_git_command(["worktree", "remove", "--force", str(worktree)])
                if remove.returncode != 0:
                    run_git_command(["worktree", "prune"])


def compare_model_resolution_facts(
    candidate_lanes: list[dict[str, Any]],
    live_lanes: list[dict[str, Any]],
) -> str:
    difference = first_difference(candidate_lanes, live_lanes)
    require(difference is None, f"model resolution input differs from live resolver facts: {difference}")
    return json_fingerprint(live_lanes)


def validate_inventory_review(
    inventory: dict[str, Any],
    review_path: Path = INVENTORY_REVIEW_PATH,
) -> dict[str, Any]:
    review = read_json(review_path)
    require(review.get("schema_version") == 1, "inventory review schema_version must be 1")
    require(review.get("reviewed_at_git_sha") == FROZEN_LEGACY_SHA, "inventory review frozen SHA mismatch")
    require(review.get("candidate_identity") == "path+symbol", "inventory review candidate identity mismatch")
    require(review.get("unresolved_count") == 0, "inventory review has unresolved scaffolding candidates")
    decisions = require_list(review.get("decisions"), "inventory-review.decisions")
    expected_decisions = {
        "scaffolding-owned",
        "excluded-math",
        "excluded-parser",
        "excluded-weights",
        "excluded-other",
    }
    require(set(decisions) == expected_decisions, "inventory review decision vocabulary mismatch")

    findings = require_list(require_object(inventory.get("coupling"), "coupling-inventory.coupling").get("findings"), "coupling-inventory.coupling.findings")
    candidates: dict[tuple[str, str], set[int]] = defaultdict(set)
    for raw in findings:
        if isinstance(raw, dict) and raw.get("category") == "model_scaffolding_candidate":
            path = require_safe_relative_path(raw.get("path"), "model scaffolding candidate.path")
            symbol = require_string(raw.get("symbol"), "model scaffolding candidate.symbol")
            line = require_positive_int(raw.get("line"), "model scaffolding candidate.line")
            candidates[(path, symbol)].add(line)

    reviewed: dict[tuple[str, str], set[int]] = {}
    counts: Counter[str] = Counter()
    for index, raw in enumerate(require_list(review.get("reviews"), "inventory-review.reviews")):
        row = require_object(raw, f"inventory-review.reviews[{index}]")
        key = (
            require_safe_relative_path(row.get("path"), f"inventory-review.reviews[{index}].path"),
            require_string(row.get("symbol"), f"inventory-review.reviews[{index}].symbol"),
        )
        require(key not in reviewed, f"duplicate inventory review row: {key}")
        line_hints = require_list(row.get("line_hints"), f"inventory-review.reviews[{index}].line_hints")
        require(line_hints and all(isinstance(line, int) and line > 0 for line in line_hints), f"inventory-review.reviews[{index}].line_hints invalid")
        reviewed[key] = set(line_hints)
        classification = require_string(row.get("classification"), f"inventory-review.reviews[{index}].classification")
        require(classification in expected_decisions, f"invalid inventory review classification: {classification}")
        counts[classification] += 1
        require_string(row.get("reason"), f"inventory-review.reviews[{index}].reason")
        require_string(row.get("owner"), f"inventory-review.reviews[{index}].owner")
        require(row.get("reviewed_at_git_sha") == FROZEN_LEGACY_SHA, f"inventory-review.reviews[{index}] SHA mismatch")
    require(reviewed == candidates, "inventory review does not exactly match current analyzer candidates")
    require(review.get("candidate_count") == len(candidates), "inventory review candidate_count mismatch")
    require(review.get("reviewed_count") == len(reviewed), "inventory review reviewed_count mismatch")
    expected_counts = {key: counts.get(key, 0) for key in expected_decisions}
    require(review.get("classification_counts") == expected_counts, "inventory review classification_counts mismatch")

    native = require_list(inventory.get("large_native_source_trees"), "coupling-inventory.large_native_source_trees")
    native_keys = {
        (row.get("tree_key"), row.get("content_root_sha256"))
        for row in native
        if isinstance(row, dict) and row.get("is_large") is True
    }
    reviewed_native: set[tuple[Any, Any]] = set()
    for index, raw in enumerate(require_list(review.get("large_native_content_roots"), "inventory-review.large_native_content_roots")):
        row = require_object(raw, f"inventory-review.large_native_content_roots[{index}]")
        key = (row.get("tree_key"), row.get("content_root_sha256"))
        require(key not in reviewed_native, f"duplicate reviewed native content root: {key}")
        reviewed_native.add(key)
        require(row.get("counted_build_input_count") == 1, f"native review {key} must count one content root")
        require_string(row.get("aggregation_decision"), f"native review {key}.aggregation_decision")
        require(row.get("reviewed_at_git_sha") == FROZEN_LEGACY_SHA, f"native review {key} SHA mismatch")
    require(reviewed_native == native_keys, "inventory native-root review is stale")
    require(review.get("large_native_content_root_count") == len(native_keys), "inventory native-root review count mismatch")
    require(review.get("large_native_content_root_reviewed_count") == len(reviewed_native), "inventory native-root reviewed count mismatch")
    require(review.get("large_native_content_root_unresolved_count") == 0, "inventory native-root review has unresolved rows")
    return review


def validate_hash_map(value: Any, expected_paths: set[str], label: str) -> dict[str, str]:
    mapping = require_object(value, label)
    require(set(mapping) == expected_paths, f"{label} must cover exactly {sorted(expected_paths)}")
    return {path: require_sha256(digest, f"{label}.{path}") for path, digest in mapping.items()}


def validate_models_catalog(
    catalog_path: Path = MODELS_CATALOG_PATH,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    catalog = read_json(catalog_path)
    require(catalog.get("schema_version") == 1, "models catalog schema_version must be 1")
    require(catalog.get("generated_from_git_sha") == FROZEN_LEGACY_SHA, "models catalog frozen SHA mismatch")
    require(
        catalog.get("source_document")
        == "docs/goals/runtime-vnext-0.8.0-2026-07-10/MODEL_MATRIX.md",
        "models catalog source document mismatch",
    )
    require_string(catalog.get("catalog_id"), "models catalog catalog_id")
    hardware = require_object(catalog.get("hardware_policies"), "models catalog hardware_policies")
    require(hardware, "models catalog hardware policy map is empty")
    rows = require_list(catalog.get("models"), "models catalog models")
    require(len(rows) == 12, "models catalog must contain exactly 12 backend lanes")
    indexed: dict[str, dict[str, Any]] = {}
    pairs: set[tuple[str, str]] = set()
    for index, raw in enumerate(rows):
        row = require_object(raw, f"models catalog models[{index}]")
        lane_id = require_string(row.get("id"), f"models catalog models[{index}].id")
        require(lane_id not in indexed, f"duplicate models catalog lane id: {lane_id}")
        model_id = require_string(row.get("model_id"), f"models catalog {lane_id}.model_id")
        require(model_id in EXPECTED_MODEL_IDS, f"models catalog {lane_id} has unknown model_id {model_id}")
        backend = require_string(row.get("backend"), f"models catalog {lane_id}.backend")
        require(backend in {"cuda", "metal"}, f"models catalog {lane_id} backend must be cuda or metal")
        require((model_id, backend) not in pairs, f"duplicate model/backend lane: {model_id}/{backend}")
        pairs.add((model_id, backend))
        require_string(row.get("repo"), f"models catalog {lane_id}.repo")
        require_string(row.get("format"), f"models catalog {lane_id}.format")
        hardware_policy = require_string(row.get("hardware_policy"), f"models catalog {lane_id}.hardware_policy")
        require(hardware_policy in hardware, f"models catalog {lane_id} references unknown hardware policy")

        revision = require_object(row.get("revision"), f"models catalog {lane_id}.revision")
        require(revision.get("status") in {"pinned", "resolution_required"}, f"models catalog {lane_id} revision status invalid")
        if revision["status"] == "pinned":
            require_git_sha(revision.get("value"), f"models catalog {lane_id}.revision.value")
        else:
            require(revision.get("value") is None, f"models catalog {lane_id} resolution_required value must be null")

        selectors = require_list(row.get("files"), f"models catalog {lane_id}.files")
        require(selectors, f"models catalog {lane_id} has no file selectors")
        for selector_index, selector_raw in enumerate(selectors):
            selector = require_object(selector_raw, f"models catalog {lane_id}.files[{selector_index}]")
            require(("path" in selector) != ("glob" in selector), f"models catalog {lane_id}.files[{selector_index}] needs exactly one path or glob")
            key = "path" if "path" in selector else "glob"
            require_safe_relative_path(selector.get(key), f"models catalog {lane_id}.files[{selector_index}].{key}")
            require(selector.get("required") is True or selector.get("required_if_sharded") is True, f"models catalog {lane_id}.files[{selector_index}] is not required")
            if "expected_size_bytes" in selector:
                require_positive_int(selector.get("expected_size_bytes"), f"models catalog {lane_id}.files[{selector_index}].expected_size_bytes")
            if "expected_sha256" in selector:
                require(key == "path", f"models catalog {lane_id}.files[{selector_index}].expected_sha256 requires an exact path selector")
                require(selector.get("required") is True, f"models catalog {lane_id}.files[{selector_index}].expected_sha256 requires required=true")
                require_sha256(selector.get("expected_sha256"), f"models catalog {lane_id}.files[{selector_index}].expected_sha256")

        reference = require_object(row.get("reference"), f"models catalog {lane_id}.reference")
        semantic_repo = require_string(reference.get("semantic_repo"), f"models catalog {lane_id}.reference.semantic_repo")
        semantic_revision = require_object(reference.get("semantic_revision"), f"models catalog {lane_id}.reference.semantic_revision")
        require(semantic_revision.get("status") in {"pinned", "same_as_weight_revision"}, f"models catalog {lane_id} semantic revision status invalid")
        if semantic_revision["status"] == "pinned":
            require_git_sha(semantic_revision.get("value"), f"models catalog {lane_id}.reference.semantic_revision.value")
        else:
            require(semantic_repo == row["repo"], f"models catalog {lane_id} same_as_weight_revision must use the weight repo")
            require(revision["status"] == "pinned", f"models catalog {lane_id} same_as_weight_revision needs pinned weights")
        semantic_paths = [
            require_safe_relative_path(path, f"models catalog {lane_id}.reference.required_semantic_files")
            for path in require_list(reference.get("required_semantic_files"), f"models catalog {lane_id}.reference.required_semantic_files")
        ]
        require(len(semantic_paths) == len(set(semantic_paths)), f"models catalog {lane_id} semantic files contain duplicates")
        semantic_hashes = validate_hash_map(
            reference.get("semantic_file_sha256"),
            set(semantic_paths),
            f"models catalog {lane_id}.reference.semantic_file_sha256",
        )

        tokenizer_hashes: dict[str, str] | None = None
        tokenizer_paths: list[str] = []
        if reference.get("tokenizer_repo") is not None:
            require_string(reference.get("tokenizer_repo"), f"models catalog {lane_id}.reference.tokenizer_repo")
            tokenizer_revision = require_object(reference.get("tokenizer_revision"), f"models catalog {lane_id}.reference.tokenizer_revision")
            require(tokenizer_revision.get("status") == "pinned", f"models catalog {lane_id} tokenizer revision must be pinned")
            require_git_sha(tokenizer_revision.get("value"), f"models catalog {lane_id}.reference.tokenizer_revision.value")
            tokenizer_paths = [
                require_safe_relative_path(path, f"models catalog {lane_id}.reference.required_tokenizer_files")
                for path in require_list(reference.get("required_tokenizer_files"), f"models catalog {lane_id}.reference.required_tokenizer_files")
            ]
            tokenizer_hashes = validate_hash_map(
                reference.get("tokenizer_file_sha256"),
                set(tokenizer_paths),
                f"models catalog {lane_id}.reference.tokenizer_file_sha256",
            )

        generation = require_object(reference.get("generation_config_source"), f"models catalog {lane_id}.reference.generation_config_source")
        require(generation.get("source") == "semantic_source", f"models catalog {lane_id} generation source must be semantic_source")
        generation_path = require_safe_relative_path(generation.get("path"), f"models catalog {lane_id}.reference.generation_config_source.path")
        require(generation.get("policy") in {"required", "absent"}, f"models catalog {lane_id} generation policy invalid")
        if generation["policy"] == "required":
            expected_hash = require_sha256(generation.get("content_sha256"), f"models catalog {lane_id}.reference.generation_config_source.content_sha256")
            require(semantic_hashes.get(generation_path) == expected_hash, f"models catalog {lane_id} generation hash is not semantically bound")
        else:
            require(generation_path not in semantic_hashes and "content_sha256" not in generation, f"models catalog {lane_id} absent generation config is contradictory")

        chat = require_object(reference.get("chat_template_source"), f"models catalog {lane_id}.reference.chat_template_source")
        chat_source = chat.get("source")
        require(chat_source in {"semantic_source", "tokenizer_source"}, f"models catalog {lane_id} chat template source invalid")
        chat_path = require_safe_relative_path(chat.get("path"), f"models catalog {lane_id}.reference.chat_template_source.path")
        require(chat.get("json_pointer") == "/chat_template", f"models catalog {lane_id} chat template pointer mismatch")
        source_hashes = semantic_hashes if chat_source == "semantic_source" else tokenizer_hashes
        source_paths = semantic_paths if chat_source == "semantic_source" else tokenizer_paths
        require(source_hashes is not None and chat_path in source_paths, f"models catalog {lane_id} chat template source file is not bound")
        require(require_sha256(chat.get("container_sha256"), f"models catalog {lane_id}.reference.chat_template_source.container_sha256") == source_hashes[chat_path], f"models catalog {lane_id} chat template container hash mismatch")
        require_sha256(chat.get("content_sha256"), f"models catalog {lane_id}.reference.chat_template_source.content_sha256")

        official_raw = reference.get("official_upstream")
        if model_id == "Llama-3.1-8B-Instruct":
            official = require_object(official_raw, f"models catalog {lane_id}.reference.official_upstream")
            require_string(official.get("repo"), f"models catalog {lane_id}.reference.official_upstream.repo")
            official_revision = require_object(official.get("revision"), f"models catalog {lane_id}.reference.official_upstream.revision")
            require(official_revision.get("status") == "pinned", f"models catalog {lane_id} official upstream revision must be pinned")
            require_git_sha(official_revision.get("value"), f"models catalog {lane_id}.reference.official_upstream.revision.value")
            require(official.get("required_gated") is True, f"models catalog {lane_id} official upstream must require gated provenance")
            match_paths = [
                require_safe_relative_path(path, f"models catalog {lane_id}.reference.official_upstream.blob_oid_match_files")
                for path in require_list(official.get("blob_oid_match_files"), f"models catalog {lane_id}.reference.official_upstream.blob_oid_match_files")
            ]
            require(len(match_paths) == len(set(match_paths)), f"models catalog {lane_id} official match paths contain duplicates")
            oid_map = require_object(official.get("expected_git_oids"), f"models catalog {lane_id}.reference.official_upstream.expected_git_oids")
            require(set(oid_map) == set(match_paths), f"models catalog {lane_id} official Git OID map mismatch")
            for path, oid in oid_map.items():
                require_git_sha(oid, f"models catalog {lane_id}.reference.official_upstream.expected_git_oids.{path}")
            validate_hash_map(
                official.get("expected_content_sha256"),
                set(match_paths),
                f"models catalog {lane_id}.reference.official_upstream.expected_content_sha256",
            )
            size_map = require_object(official.get("expected_size_bytes"), f"models catalog {lane_id}.reference.official_upstream.expected_size_bytes")
            require(set(size_map) == set(match_paths), f"models catalog {lane_id} official size map mismatch")
            for path, size in size_map.items():
                require_positive_int(size, f"models catalog {lane_id}.reference.official_upstream.expected_size_bytes.{path}")
            require_string(official.get("access_note"), f"models catalog {lane_id}.reference.official_upstream.access_note")
        else:
            require(official_raw is None, f"models catalog {lane_id} has unexpected official_upstream evidence")
        indexed[lane_id] = row

    expected_pairs = {(model_id, backend) for model_id in EXPECTED_MODEL_IDS for backend in ("cuda", "metal")}
    require(pairs == expected_pairs, "models catalog does not contain the exact six-model CUDA/Metal matrix")
    return catalog, indexed


def normalized_file(raw: Any, label: str) -> dict[str, Any]:
    row = require_object(raw, label)
    path = require_safe_relative_path(row.get("path"), f"{label}.path")
    digest = require_sha256(row.get("sha256"), f"{label}.sha256")
    size = require_positive_int(row.get("size_bytes"), f"{label}.size_bytes")
    source = require_string(row.get("sha256_source"), f"{label}.sha256_source")
    require(source in {"downloaded_content", "hugging_face_lfs_oid"}, f"{label}.sha256_source is not canonical")
    git_oid = require_git_sha(row.get("git_oid"), f"{label}.git_oid")
    normalized: dict[str, Any] = {
        "git_oid": git_oid,
        "path": path,
        "sha256": digest,
        "sha256_source": source,
        "size_bytes": size,
    }
    if source == "hugging_face_lfs_oid":
        require(row.get("lfs_oid") == digest, f"{label}.lfs_oid must equal its SHA256")
        normalized["lfs_oid"] = digest
        if path.endswith(".safetensors.index.json"):
            require(size <= 32 * 1024 * 1024, f"{label} LFS index exceeds metadata limit")
            url = require_string(row.get("content_request_url"), f"{label}.content_request_url")
            require(url.startswith("https://huggingface.co/"), f"{label}.content_request_url must use Hugging Face HTTPS")
            require(row.get("lfs_metadata_downloaded") is True, f"{label}.lfs_metadata_downloaded must be true")
            normalized["content_request_url"] = url
            normalized["lfs_metadata_downloaded"] = True
        else:
            require(
                "content_request_url" not in row and "lfs_metadata_downloaded" not in row,
                f"{label} non-index LFS file must not contain download evidence",
            )
    else:
        require(
            "lfs_oid" not in row and "lfs_metadata_downloaded" not in row,
            f"{label} downloaded metadata must not contain LFS identity or download flags",
        )
        url = require_string(row.get("content_request_url"), f"{label}.content_request_url")
        require(url.startswith("https://huggingface.co/"), f"{label}.content_request_url must use Hugging Face HTTPS")
        normalized["content_request_url"] = url
    return normalized


def normalized_source(
    raw: Any,
    *,
    label: str,
    expected_repo: str,
    expected_revision: str,
    expected_requested_revision: dict[str, Any],
    required_paths: set[str] | None = None,
    expected_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    source = require_object(raw, label)
    require(source.get("repo") == expected_repo, f"{label}.repo mismatch")
    require(source.get("revision") == expected_revision, f"{label}.revision mismatch")
    require(source.get("requested_revision") == expected_requested_revision, f"{label}.requested_revision mismatch")
    require(isinstance(source.get("gated"), bool), f"{label}.gated must be boolean")
    model_url = require_string(source.get("model_request_url"), f"{label}.model_request_url")
    expected_model_url = f"https://huggingface.co/api/models/{expected_repo}"
    if expected_requested_revision.get("status") != "resolution_required":
        expected_model_url += f"/revision/{expected_revision}"
    require(model_url == expected_model_url, f"{label}.model_request_url is not bound to repo/revision")
    tree_urls = require_list(source.get("tree_request_urls"), f"{label}.tree_request_urls")
    tree_prefix = f"https://huggingface.co/api/models/{expected_repo}/tree/{expected_revision}?"
    require(tree_urls and all(isinstance(url, str) and url.startswith(tree_prefix) for url in tree_urls), f"{label}.tree_request_urls are not bound to repo/revision")
    files = [normalized_file(item, f"{label}.files[{index}]") for index, item in enumerate(require_list(source.get("files"), f"{label}.files"))]
    require(files, f"{label}.files must not be empty")
    files.sort(key=lambda item: item["path"])
    paths = [item["path"] for item in files]
    require(len(paths) == len(set(paths)), f"{label}.files contains duplicate paths")
    indexed = {item["path"]: item for item in files}
    if required_paths is not None:
        require(set(paths) == required_paths, f"{label}.files must cover exactly {sorted(required_paths)}")
    if expected_hashes is not None:
        require(all(indexed.get(path, {}).get("sha256") == digest for path, digest in expected_hashes.items()), f"{label}.files differ from catalog SHA256 locks")

    license_raw = require_object(source.get("license"), f"{label}.license")
    license_files = [
        normalized_file(item, f"{label}.license.files[{index}]")
        for index, item in enumerate(require_list(license_raw.get("files"), f"{label}.license.files"))
    ]
    license_files.sort(key=lambda item: item["path"])
    license_paths = {item["path"] for item in license_files}
    require(
        len(license_paths) == len(license_files),
        f"{label}.license.files contains duplicate paths",
    )
    require(
        set(paths).isdisjoint(license_paths),
        f"{label}.files and license.files contain duplicate paths",
    )
    allowed_license_basenames = {
        "copying",
        "license",
        "license.md",
        "license.txt",
        "notice",
        "notice.txt",
    }
    require(
        all(Path(path).name.lower() in allowed_license_basenames for path in license_paths),
        f"{label}.license.files contains a non-license path",
    )
    license_id = license_raw.get("hugging_face_id")
    require(license_id is None or isinstance(license_id, str), f"{label}.license.hugging_face_id invalid")
    return {
        "files": files,
        "gated": source["gated"],
        "license": {"files": license_files, "hugging_face_id": license_id},
        "model_request_url": model_url,
        "repo": expected_repo,
        "requested_revision": copy.deepcopy(expected_requested_revision),
        "revision": expected_revision,
        "tree_request_urls": tree_urls,
    }


def require_catalog_weight_files(catalog_lane: dict[str, Any], source: dict[str, Any], label: str) -> None:
    files = {item["path"]: item for item in source["files"]}
    paths = set(files)
    sharded = validate_safetensors_shard_paths(paths, label)
    selectors = require_list(catalog_lane.get("files"), f"{label}.catalog.files")
    allowed_lfs_metadata_paths = {
        str(selector["path"])
        for selector in selectors
        if isinstance(selector, dict)
        and selector.get("required_if_sharded") is True
        and "path" in selector
        and str(selector["path"]).endswith(".safetensors.index.json")
    }
    conditional_paths = {
        str(selector["path"])
        for selector in selectors
        if isinstance(selector, dict)
        and selector.get("required_if_sharded") is True
        and "path" in selector
    }
    require(
        sharded or paths.isdisjoint(conditional_paths),
        f"{label} unsharded model contains a conditional weight file",
    )
    active_selectors: list[dict[str, Any]] = []
    for index, raw in enumerate(selectors):
        selector = require_object(raw, f"{label}.catalog.files[{index}]")
        required = selector.get("required") is True or (selector.get("required_if_sharded") is True and sharded)
        if not required:
            continue
        active_selectors.append(selector)
        if "path" in selector:
            path = str(selector["path"])
            require(path in paths, f"{label} is missing required weight file {path}")
            if "expected_size_bytes" in selector:
                require(files[path]["size_bytes"] == selector["expected_size_bytes"], f"{label} size mismatch for {path}")
            if "expected_sha256" in selector:
                expected_sha256 = require_sha256(
                    selector.get("expected_sha256"),
                    f"{label}.catalog.files[{index}].expected_sha256",
                )
                require(files[path]["sha256"] == expected_sha256, f"{label} expected SHA256 mismatch for {path}")
                require(
                    files[path]["sha256_source"] == "hugging_face_lfs_oid",
                    f"{label} expected SHA256 for {path} requires Hugging Face LFS identity",
                )
                require(
                    files[path].get("lfs_oid") == expected_sha256,
                    f"{label} expected SHA256 for {path} differs from its Hugging Face LFS OID",
                )
        else:
            pattern = str(selector["glob"])
            require(any(fnmatch.fnmatchcase(path, pattern) for path in paths), f"{label} is missing required weight glob {pattern}")
    for path in sorted(paths):
        file_row = files[path]
        if file_row.get("lfs_metadata_downloaded") is True:
            require(
                sharded and path in allowed_lfs_metadata_paths,
                f"{label} LFS metadata download is not bound to an exact required_if_sharded selector: {path}",
            )
        if (
            sharded
            and path in allowed_lfs_metadata_paths
            and file_row.get("sha256_source") == "hugging_face_lfs_oid"
        ):
            require(
                file_row.get("lfs_metadata_downloaded") is True,
                f"{label} LFS safetensors index lacks download evidence: {path}",
            )
        selected = any(
            ("path" in selector and selector["path"] == path)
            or ("glob" in selector and fnmatch.fnmatchcase(path, str(selector["glob"])))
            for selector in active_selectors
        )
        require(selected, f"{label} contains an unselected weight file: {path}")


def require_no_lfs_metadata_downloads(source: dict[str, Any], label: str) -> None:
    rows = [*source["files"], *source["license"]["files"]]
    require(
        all(row.get("lfs_metadata_downloaded") is not True for row in rows),
        f"{label} must not carry LFS metadata download evidence",
    )


def validate_official_upstream(
    lane: dict[str, Any],
    reference: dict[str, Any],
    semantic: dict[str, Any],
    label: str,
) -> dict[str, Any] | None:
    rule_raw = reference.get("official_upstream")
    actual_raw = lane.get("official_upstream")
    if rule_raw is None:
        require(actual_raw is None, f"{label}.official_upstream is unexpected")
        return None
    rule = require_object(rule_raw, f"{label}.catalog.reference.official_upstream")
    actual = require_object(actual_raw, f"{label}.official_upstream")
    repo = require_string(rule.get("repo"), f"{label}.catalog.reference.official_upstream.repo")
    revision = str(require_object(rule.get("revision"), f"{label}.catalog.reference.official_upstream.revision")["value"])
    require(actual.get("repo") == repo and actual.get("revision") == revision, f"{label}.official_upstream repo/revision mismatch")
    require(actual.get("gated") == "manual" and rule.get("required_gated") is True, f"{label}.official_upstream gated provenance mismatch")
    require(actual.get("access_note") == rule.get("access_note"), f"{label}.official_upstream access note mismatch")
    require(actual.get("mirror_repo") == semantic["repo"] and actual.get("mirror_revision") == semantic["revision"], f"{label}.official_upstream mirror source mismatch")
    require(actual.get("verification_method") == "mirror_content_sha256_and_official_git_blob_oid", f"{label}.official_upstream verification method mismatch")
    expected_model_url = f"https://huggingface.co/api/models/{repo}/revision/{revision}"
    require(actual.get("model_request_url") == expected_model_url, f"{label}.official_upstream model URL mismatch")
    tree_urls = require_list(actual.get("tree_request_urls"), f"{label}.official_upstream.tree_request_urls")
    tree_prefix = f"https://huggingface.co/api/models/{repo}/tree/{revision}?"
    require(tree_urls and all(isinstance(url, str) and url.startswith(tree_prefix) for url in tree_urls), f"{label}.official_upstream tree URLs mismatch")
    paths = require_list(rule.get("blob_oid_match_files"), f"{label}.catalog.reference.official_upstream.blob_oid_match_files")
    oid_map = require_object(rule.get("expected_git_oids"), f"{label}.catalog.reference.official_upstream.expected_git_oids")
    hash_map = require_object(rule.get("expected_content_sha256"), f"{label}.catalog.reference.official_upstream.expected_content_sha256")
    size_map = require_object(rule.get("expected_size_bytes"), f"{label}.catalog.reference.official_upstream.expected_size_bytes")
    matches = require_list(actual.get("mirror_blob_oid_matches"), f"{label}.official_upstream.mirror_blob_oid_matches")
    require([item.get("path") if isinstance(item, dict) else None for item in matches] == paths, f"{label}.official_upstream match path order mismatch")
    semantic_files = {item["path"]: item for item in semantic["files"]}
    normalized_matches: list[dict[str, Any]] = []
    for index, raw in enumerate(matches):
        item = require_object(raw, f"{label}.official_upstream.mirror_blob_oid_matches[{index}]")
        path = str(paths[index])
        require(item.get("git_oid") == oid_map[path], f"{label}.official_upstream Git OID mismatch for {path}")
        require(item.get("content_sha256") == hash_map[path], f"{label}.official_upstream content SHA256 mismatch for {path}")
        require(item.get("size_bytes") == size_map[path], f"{label}.official_upstream size mismatch for {path}")
        require(semantic_files.get(path, {}).get("sha256") == hash_map[path], f"{label}.official_upstream mirror content differs for {path}")
        normalized_matches.append(
            {
                "content_sha256": hash_map[path],
                "git_oid": oid_map[path],
                "path": path,
                "size_bytes": size_map[path],
            }
        )
    return {
        "access_note": actual["access_note"],
        "gated": "manual",
        "mirror_blob_oid_matches": normalized_matches,
        "mirror_repo": semantic["repo"],
        "mirror_revision": semantic["revision"],
        "model_request_url": expected_model_url,
        "repo": repo,
        "revision": revision,
        "tree_request_urls": tree_urls,
        "verification_method": "mirror_content_sha256_and_official_git_blob_oid",
    }


def validate_resolution_matrix(
    catalog: dict[str, Any],
    indexed_catalog: dict[str, dict[str, Any]],
    resolution: dict[str, Any],
    *,
    expected_lane_count: int,
) -> list[dict[str, Any]]:
    lanes = require_list(resolution.get("lanes"), "model-resolution.lanes")
    require(len(lanes) == expected_lane_count, f"model resolution must contain exactly {expected_lane_count} lanes")
    normalized_lanes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(lanes):
        lane = require_object(raw, f"model-resolution.lanes[{index}]")
        lane_id = require_string(lane.get("catalog_lane_id"), f"model-resolution.lanes[{index}].catalog_lane_id")
        require(lane_id not in seen, f"duplicate model resolution lane: {lane_id}")
        seen.add(lane_id)
        catalog_lane = indexed_catalog.get(lane_id)
        require(catalog_lane is not None, f"model resolution contains unknown catalog lane: {lane_id}")
        label = f"model-resolution.{lane_id}"
        for field in ("model_id", "backend", "format"):
            require(lane.get(field) == catalog_lane.get(field), f"{label}.{field} differs from catalog")

        catalog_revision = require_object(catalog_lane.get("revision"), f"models catalog {lane_id}.revision")
        requested_weight = {"status": catalog_revision["status"], "value": catalog_revision.get("value")}
        weight_raw = require_object(lane.get("weight_source"), f"{label}.weight_source")
        weight_revision = require_git_sha(weight_raw.get("revision"), f"{label}.weight_source.revision")
        if catalog_revision["status"] == "pinned":
            require(weight_revision == catalog_revision["value"], f"{label}.weight_source revision differs from pinned catalog")
        weight = normalized_source(
            weight_raw,
            label=f"{label}.weight_source",
            expected_repo=str(catalog_lane["repo"]),
            expected_revision=weight_revision,
            expected_requested_revision=requested_weight,
        )
        require_catalog_weight_files(catalog_lane, weight, label)

        reference = require_object(catalog_lane.get("reference"), f"models catalog {lane_id}.reference")
        semantic_revision_spec = require_object(reference.get("semantic_revision"), f"models catalog {lane_id}.reference.semantic_revision")
        semantic_revision = weight_revision if semantic_revision_spec["status"] == "same_as_weight_revision" else str(semantic_revision_spec["value"])
        semantic_paths = set(require_list(reference.get("required_semantic_files"), f"models catalog {lane_id}.reference.required_semantic_files"))
        semantic_hashes = require_object(reference.get("semantic_file_sha256"), f"models catalog {lane_id}.reference.semantic_file_sha256")
        semantic = normalized_source(
            lane.get("semantic_source"),
            label=f"{label}.semantic_source",
            expected_repo=str(reference["semantic_repo"]),
            expected_revision=semantic_revision,
            expected_requested_revision={"status": "pinned", "value": semantic_revision},
            required_paths=semantic_paths,
            expected_hashes={str(path): str(digest) for path, digest in semantic_hashes.items()},
        )
        require_no_lfs_metadata_downloads(semantic, f"{label}.semantic_source")

        tokenizer: dict[str, Any] | None = None
        if reference.get("tokenizer_repo") is not None:
            tokenizer_revision = str(require_object(reference.get("tokenizer_revision"), f"models catalog {lane_id}.reference.tokenizer_revision")["value"])
            tokenizer_paths = set(require_list(reference.get("required_tokenizer_files"), f"models catalog {lane_id}.reference.required_tokenizer_files"))
            tokenizer_hashes = require_object(reference.get("tokenizer_file_sha256"), f"models catalog {lane_id}.reference.tokenizer_file_sha256")
            tokenizer = normalized_source(
                lane.get("tokenizer_source"),
                label=f"{label}.tokenizer_source",
                expected_repo=str(reference["tokenizer_repo"]),
                expected_revision=tokenizer_revision,
                expected_requested_revision={"status": "pinned", "value": tokenizer_revision},
                required_paths=tokenizer_paths,
                expected_hashes={str(path): str(digest) for path, digest in tokenizer_hashes.items()},
            )
            require_no_lfs_metadata_downloads(tokenizer, f"{label}.tokenizer_source")
        else:
            require(lane.get("tokenizer_source") is None, f"{label}.tokenizer_source must be null")

        generation_spec = require_object(reference.get("generation_config_source"), f"models catalog {lane_id}.reference.generation_config_source")
        generation = require_object(lane.get("generation_config"), f"{label}.generation_config")
        expected_generation_base = {
            "path": generation_spec["path"],
            "policy": generation_spec["policy"],
            "repo": semantic["repo"],
            "revision": semantic["revision"],
            "source": "semantic_source",
        }
        require(all(generation.get(key) == value for key, value in expected_generation_base.items()), f"{label}.generation_config source contract mismatch")
        if generation_spec["policy"] == "required":
            require(generation.get("present") is True, f"{label}.generation_config must be present")
            generation_file = normalized_file(generation.get("file"), f"{label}.generation_config.file")
            require(generation_file["path"] == generation_spec["path"], f"{label}.generation_config path mismatch")
            require(generation_file["sha256"] == generation_spec["content_sha256"], f"{label}.generation_config SHA256 mismatch")
        else:
            require(generation.get("present") is False and "file" not in generation, f"{label}.generation_config must be absent")

        chat_spec = require_object(reference.get("chat_template_source"), f"models catalog {lane_id}.reference.chat_template_source")
        chat = require_object(lane.get("chat_template"), f"{label}.chat_template")
        selected_source = semantic if chat_spec["source"] == "semantic_source" else tokenizer
        require(selected_source is not None, f"{label}.chat_template selects a missing source")
        expected_chat = {
            "container_sha256": chat_spec["container_sha256"],
            "content_sha256": chat_spec["content_sha256"],
            "json_pointer": "/chat_template",
            "path": chat_spec["path"],
            "repo": selected_source["repo"],
            "revision": selected_source["revision"],
            "source": chat_spec["source"],
        }
        require(all(chat.get(key) == value for key, value in expected_chat.items()), f"{label}.chat_template contract mismatch")
        require_positive_int(chat.get("content_bytes"), f"{label}.chat_template.content_bytes")
        selected_files = {item["path"]: item for item in selected_source["files"]}
        require(selected_files.get(str(chat_spec["path"]), {}).get("sha256") == chat_spec["container_sha256"], f"{label}.chat_template container is not bound to source file")
        official_upstream = validate_official_upstream(lane, reference, semantic, label)

        normalized_lanes.append(
            {
                "backend": lane["backend"],
                "catalog_lane_id": lane_id,
                "chat_template": copy.deepcopy(chat),
                "format": lane["format"],
                "generation_config": copy.deepcopy(generation),
                "hardware_policy": catalog_lane["hardware_policy"],
                "model_id": lane["model_id"],
                "official_upstream": official_upstream,
                "role": catalog_lane.get("role"),
                "semantic_source": semantic,
                "tokenizer_source": tokenizer,
                "weight_source": weight,
            }
        )
    require(seen == set(indexed_catalog), "model resolution does not exactly cover catalog lane ids")
    normalized_lanes.sort(key=lambda item: item["catalog_lane_id"])
    return normalized_lanes


def validate_resolution_provenance(resolution: dict[str, Any], lanes: list[dict[str, Any]]) -> None:
    request_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    expected_request_keys: set[tuple[str, str]] = set()
    expected_metadata_urls: set[str] = set()
    for index, raw in enumerate(require_list(resolution.get("requests"), "model-resolution.requests")):
        request = require_object(raw, f"model-resolution.requests[{index}]")
        key = (str(request.get("kind")), str(request.get("url")))
        require(key not in request_lookup, f"duplicate model resolution request provenance: {key[0]} {key[1]}")
        request_lookup[key] = request

    for lane in lanes:
        lane_id = str(lane["catalog_lane_id"])
        for source_name in ("weight_source", "semantic_source", "tokenizer_source"):
            source = lane.get(source_name)
            if source is None:
                continue
            label = f"model-resolution.{lane_id}.{source_name}"
            model_url = str(source["model_request_url"])
            expected_request_keys.add(("model-info", model_url))
            model_request = request_lookup.get(("model-info", model_url))
            require(model_request is not None, f"{label} model request is absent from provenance")
            model_body = require_object(request_json_body(model_request, f"{label} model request"), f"{label} model response")
            require(model_body.get("sha") == source["revision"], f"{label} model response revision mismatch")
            tree_entries: dict[str, dict[str, Any]] = {}
            for tree_url in source["tree_request_urls"]:
                expected_request_keys.add(("repo-tree", tree_url))
                tree_request = request_lookup.get(("repo-tree", tree_url))
                require(tree_request is not None, f"{label} tree request is absent from provenance")
                tree_body = require_list(request_json_body(tree_request, f"{label} tree request"), f"{label} tree response")
                for tree_index, raw_entry in enumerate(tree_body):
                    entry = require_object(raw_entry, f"{label} tree response[{tree_index}]")
                    if entry.get("type") not in {"file", None}:
                        continue
                    path = require_safe_relative_path(entry.get("path"), f"{label} tree response[{tree_index}].path")
                    require(path not in tree_entries, f"{label} tree response duplicates path {path}")
                    tree_entries[path] = entry
            for file_row in [*source["files"], *source["license"]["files"]]:
                tree_entry = require_object(tree_entries.get(file_row["path"]), f"{label} tree fact {file_row['path']}")
                require(tree_entry.get("oid") == file_row["git_oid"], f"{label} tree Git OID mismatch: {file_row['path']}")
                require(tree_entry.get("size") == file_row["size_bytes"], f"{label} tree size mismatch: {file_row['path']}")
                if file_row["sha256_source"] == "hugging_face_lfs_oid":
                    lfs = require_object(tree_entry.get("lfs"), f"{label} tree LFS fact {file_row['path']}")
                    raw_lfs_oid = require_string(lfs.get("oid"), f"{label} tree LFS OID {file_row['path']}").lower()
                    if raw_lfs_oid.startswith("sha256:"):
                        raw_lfs_oid = raw_lfs_oid.removeprefix("sha256:")
                    require(raw_lfs_oid == file_row["sha256"], f"{label} tree LFS SHA256 mismatch: {file_row['path']}")
                    require(lfs.get("size") == file_row["size_bytes"], f"{label} tree LFS size mismatch: {file_row['path']}")
                    if file_row["path"].endswith(".safetensors.index.json"):
                        require(file_row["size_bytes"] <= 32 * 1024 * 1024, f"{label} LFS index exceeds metadata limit: {file_row['path']}")
                        expected_url = (
                            f"https://huggingface.co/{source['repo']}/resolve/"
                            f"{source['revision']}/{file_row['path']}"
                        )
                        require(file_row.get("lfs_metadata_downloaded") is True, f"{label} LFS index lacks downloaded metadata evidence: {file_row['path']}")
                        require(file_row.get("content_request_url") == expected_url, f"{label} LFS index URL is not bound to repo/revision/path: {file_row['path']}")
                        expected_metadata_urls.add(expected_url)
                        expected_request_keys.add(("metadata-file", expected_url))
                        request = request_lookup.get(("metadata-file", expected_url))
                        require(request is not None, f"{label} LFS index request is absent from provenance: {file_row['path']}")
                        require(request.get("response_sha256") == file_row["sha256"], f"{label} LFS index request SHA256 mismatch: {file_row['path']}")
                        require(request.get("response_bytes") == file_row["size_bytes"], f"{label} LFS index request size mismatch: {file_row['path']}")
                    else:
                        require("content_request_url" not in file_row and "lfs_metadata_downloaded" not in file_row, f"{label} non-index LFS file has download evidence: {file_row['path']}")
                if file_row["sha256_source"] != "downloaded_content":
                    continue
                require(
                    tree_entry.get("lfs") is None,
                    f"{label} downloaded metadata is LFS-backed in the authoritative tree: {file_row['path']}",
                )
                expected_url = (
                    f"https://huggingface.co/{source['repo']}/resolve/"
                    f"{source['revision']}/{file_row['path']}"
                )
                require(file_row.get("content_request_url") == expected_url, f"{label} metadata URL is not bound to repo/revision/path: {file_row['path']}")
                expected_metadata_urls.add(expected_url)
                expected_request_keys.add(("metadata-file", expected_url))
                request = request_lookup.get(("metadata-file", expected_url))
                require(request is not None, f"{label} metadata request is absent from provenance: {file_row['path']}")
                require(request.get("response_sha256") == file_row["sha256"], f"{label} request SHA256 mismatch: {file_row['path']}")
                require(request.get("response_bytes") == file_row["size_bytes"], f"{label} request size mismatch: {file_row['path']}")
        official = lane.get("official_upstream")
        if official is not None:
            label = f"model-resolution.{lane_id}.official_upstream"
            expected_request_keys.add(("model-info", official["model_request_url"]))
            model_request = request_lookup.get(("model-info", official["model_request_url"]))
            require(model_request is not None, f"{label} model request is absent from provenance")
            model_body = require_object(request_json_body(model_request, f"{label} model request"), f"{label} model response")
            require(model_body.get("sha") == official["revision"], f"{label} model response revision mismatch")
            official_tree_entries: dict[str, dict[str, Any]] = {}
            for tree_url in official["tree_request_urls"]:
                expected_request_keys.add(("repo-tree", tree_url))
                require(("repo-tree", tree_url) in request_lookup, f"{label} tree request is absent from provenance")
                tree_body = require_list(
                    request_json_body(request_lookup[("repo-tree", tree_url)], f"{label} tree request"),
                    f"{label} tree response",
                )
                for tree_index, tree_raw in enumerate(tree_body):
                    tree_row = require_object(tree_raw, f"{label} tree response[{tree_index}]")
                    if tree_row.get("type") not in {"file", None}:
                        continue
                    tree_path = require_safe_relative_path(
                        tree_row.get("path"),
                        f"{label} tree response[{tree_index}].path",
                    )
                    require(tree_path not in official_tree_entries, f"{label} tree duplicates {tree_path}")
                    official_tree_entries[tree_path] = tree_row
            for match in official["mirror_blob_oid_matches"]:
                match_path = str(match["path"])
                tree_row = require_object(
                    official_tree_entries.get(match_path),
                    f"{label} tree fact {match_path}",
                )
                require(
                    tree_row.get("oid") == match["git_oid"]
                    and tree_row.get("size") == match["size_bytes"],
                    f"{label} tree identity mismatch for {match_path}",
                )
    actual_metadata_urls = {
        url for kind, url in request_lookup if kind == "metadata-file"
    }
    require(
        actual_metadata_urls == expected_metadata_urls,
        "metadata request provenance differs from the exact selected metadata file set",
    )
    require(
        set(request_lookup) == expected_request_keys,
        "network request provenance differs from the exact selected source request set",
    )


def validate_model_resolution(
    path: Path,
    collector: dict[str, Any],
    catalog: dict[str, Any],
    indexed_catalog: dict[str, dict[str, Any]],
    *,
    catalog_path: Path = MODELS_CATALOG_PATH,
    resolver_path: Path = MODEL_RESOLVER_PATH,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolution = read_json(path)
    require(resolution.get("schema_version") == 1, "model resolution schema_version must be 1")
    require(resolution.get("artifact_type") == "runtime_vnext_model_resolution", "model resolution artifact_type mismatch")
    require(resolution.get("catalog_id") == catalog.get("catalog_id"), "model resolution catalog_id mismatch")
    require(resolution.get("catalog_path") == "scripts/release/configs/runtime_vnext_models.json", "model resolution catalog_path mismatch")
    require(resolution.get("catalog_sha256") == file_sha256(catalog_path), "model resolution is stale against current models catalog")
    require_string(resolution.get("generated_at"), "model-resolution.generated_at")

    source = require_object(resolution.get("source"), "model-resolution.source")
    require(source.get("git_sha") == collector["git_sha"], "model resolution source SHA must equal the clean collector HEAD")
    require(source.get("dirty") is False and source.get("status_short") == [], "model resolution must come from a clean checkout")
    resolver = require_object(resolution.get("resolver"), "model-resolution.resolver")
    require(resolver.get("path") == "scripts/release/runtime_vnext_model_resolver.py", "model resolution resolver path mismatch")
    require(resolver.get("sha256") == file_sha256(resolver_path), "model resolution resolver SHA256 is stale")
    policy = require_object(resolution.get("policy"), "model-resolution.policy")
    require(policy.get("transport") == "network_huggingface_https", "model resolution must use the live Hugging Face HTTPS transport")
    require(policy.get("revision") == "full_hugging_face_commit", "model resolution revision policy mismatch")
    require(policy.get("large_weight_downloaded") is False, "model resolution must not download large weights")
    require(policy.get("lfs_sha256_source") == "Hugging Face tree lfs.oid", "model resolution LFS identity policy mismatch")
    require(policy.get("non_lfs_max_download_bytes") == 32 * 1024 * 1024, "model resolution metadata download limit mismatch")
    require(
        policy.get("lfs_metadata_download")
        == {
            "allowed_suffixes": [".safetensors.index.json"],
            "max_bytes": 32 * 1024 * 1024,
            "selector_requirement": "weight_source_exact_path_required_if_sharded",
            "sha256_must_match_lfs_oid": True,
        },
        "model resolution LFS metadata download policy mismatch",
    )
    require(
        policy.get("raw_response_body_kinds") == ["model-info", "repo-tree"],
        "model resolution raw response body policy mismatch",
    )

    request_kinds: set[str] = set()
    request_keys: set[tuple[str, str]] = set()
    requests = require_list(resolution.get("requests"), "model-resolution.requests")
    require(requests, "model resolution has no network provenance requests")
    for index, raw in enumerate(requests):
        request = require_object(raw, f"model-resolution.requests[{index}]")
        require(request.get("method") == "GET", f"model-resolution.requests[{index}] method mismatch")
        kind = require_string(request.get("kind"), f"model-resolution.requests[{index}].kind")
        request_kinds.add(kind)
        url = require_string(request.get("url"), f"model-resolution.requests[{index}].url")
        require(url.startswith("https://huggingface.co/"), f"model-resolution.requests[{index}] must use Hugging Face HTTPS")
        status = request.get("status")
        require(isinstance(status, int) and 200 <= status < 300, f"model-resolution.requests[{index}] HTTP status is not successful")
        require_positive_int(request.get("response_bytes"), f"model-resolution.requests[{index}].response_bytes")
        if kind == "metadata-file":
            require(
                request["response_bytes"] <= 32 * 1024 * 1024,
                f"model-resolution.requests[{index}] metadata response exceeds download limit",
            )
        require_sha256(request.get("response_sha256"), f"model-resolution.requests[{index}].response_sha256")
        if kind in {"model-info", "repo-tree"}:
            request_json_body(request, f"model-resolution.requests[{index}]")
        else:
            require("response_body_base64" not in request, f"model-resolution.requests[{index}] unexpectedly embeds metadata body")
        key = (kind, url)
        require(key not in request_keys, f"duplicate model-resolution request provenance: {kind} {url}")
        request_keys.add(key)
    require(request_kinds == {"model-info", "repo-tree", "metadata-file"}, "model resolution network provenance kind matrix is incomplete")

    normalized_lanes = validate_resolution_matrix(
        catalog,
        indexed_catalog,
        resolution,
        expected_lane_count=12,
    )
    validate_resolution_provenance(resolution, normalized_lanes)
    return resolution, normalized_lanes


def validate_presets_catalog(
    catalog: dict[str, Any],
    indexed_catalog: dict[str, dict[str, Any]],
    lanes: list[dict[str, Any]],
    presets_path: Path = PRESETS_CATALOG_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    presets = read_json(presets_path)
    require(presets.get("schema_version") == 1, "generation presets schema_version must be 1")
    require(presets.get("model_catalog_id") == catalog.get("catalog_id"), "generation presets model_catalog_id mismatch")
    require(presets.get("seed") == 9271, "generation presets seed must be 9271")
    models = require_object(presets.get("models"), "generation presets models")
    require(set(models) == set(PRIMARY_MODEL_KEYS.values()), "generation presets must cover exactly M1-M3")
    resolved_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lane in lanes:
        resolved_by_model[str(lane["model_id"])].append(lane)

    normalized_models: dict[str, Any] = {}
    for model_id, model_key in PRIMARY_MODEL_KEYS.items():
        raw = require_object(models.get(model_key), f"generation presets {model_key}")
        metadata_repo = require_string(raw.get("metadata_repo"), f"generation presets {model_key}.metadata_repo")
        metadata_revision = require_git_sha(raw.get("metadata_revision"), f"generation presets {model_key}.metadata_revision")
        evidence = require_object(raw.get("evidence"), f"generation presets {model_key}.evidence")
        require("README.md" in evidence, f"generation presets {model_key} must bind README.md")
        evidence_hashes = {require_safe_relative_path(path, f"generation presets {model_key}.evidence path"): require_sha256(digest, f"generation presets {model_key}.evidence.{path}") for path, digest in evidence.items()}
        model_lanes = resolved_by_model.get(model_id, [])
        require(len(model_lanes) == 2, f"generation presets {model_key} needs CUDA and Metal model facts")
        for lane in model_lanes:
            semantic = lane["semantic_source"]
            require(semantic["repo"] == metadata_repo and semantic["revision"] == metadata_revision, f"generation presets {model_key} metadata source differs from resolved lane {lane['catalog_lane_id']}")
            semantic_hashes = {item["path"]: item["sha256"] for item in semantic["files"]}
            require(all(semantic_hashes.get(path) == digest for path, digest in evidence_hashes.items()), f"generation presets {model_key} evidence differs from resolved semantic source")
            catalog_lane = indexed_catalog[str(lane["catalog_lane_id"])]
            require(catalog_lane["model_id"] == model_id, f"generation presets {model_key} catalog model mismatch")

        raw_presets = require_object(raw.get("presets"), f"generation presets {model_key}.presets")
        require(set(raw_presets) == REQUIRED_PRESETS, f"generation presets {model_key} preset matrix mismatch")
        normalized_presets: dict[str, Any] = {}
        for preset_name in sorted(REQUIRED_PRESETS):
            preset = require_object(raw_presets.get(preset_name), f"generation presets {model_key}.{preset_name}")
            require(set(preset) == PRESET_FIELDS, f"generation presets {model_key}.{preset_name} field set mismatch")
            require(preset.get("seed") == 9271, f"generation presets {model_key}.{preset_name} seed mismatch")
            require(isinstance(preset.get("stop"), list), f"generation presets {model_key}.{preset_name}.stop must be an array")
            eos_ids = require_list(preset.get("eos_token_ids"), f"generation presets {model_key}.{preset_name}.eos_token_ids")
            require(eos_ids and all(isinstance(item, int) and item >= 0 for item in eos_ids), f"generation presets {model_key}.{preset_name}.eos_token_ids invalid")
            require(isinstance(preset.get("template_kwargs"), dict), f"generation presets {model_key}.{preset_name}.template_kwargs must be an object")
            require_string(preset.get("source"), f"generation presets {model_key}.{preset_name}.source")
            if preset_name == "P_DETERMINISTIC":
                require(preset.get("temperature") == 0.0 and preset.get("enable_thinking") is False, f"generation presets {model_key}.P_DETERMINISTIC contract mismatch")
                require(preset.get("template_kwargs") == {"enable_thinking": False}, f"generation presets {model_key}.P_DETERMINISTIC template kwargs mismatch")
            elif preset_name == "P_NO_THINKING":
                require(preset.get("enable_thinking") is False, f"generation presets {model_key}.P_NO_THINKING must disable thinking")
                require(preset.get("template_kwargs") == {"enable_thinking": False}, f"generation presets {model_key}.P_NO_THINKING template kwargs mismatch")
            elif preset_name == "P_THINKING":
                require(preset.get("enable_thinking") is True, f"generation presets {model_key}.P_THINKING must enable thinking")
                require(preset.get("template_kwargs") == {"enable_thinking": True}, f"generation presets {model_key}.P_THINKING template kwargs mismatch")
            else:
                require(preset.get("enable_thinking") == "model-default", f"generation presets {model_key}.P_OFFICIAL_DEFAULT must preserve model default")
                require(preset.get("template_kwargs") == {}, f"generation presets {model_key}.P_OFFICIAL_DEFAULT template kwargs must be empty")
            normalized_presets[preset_name] = copy.deepcopy(preset)
        normalized_models[model_key] = {
            "evidence": dict(sorted(evidence_hashes.items())),
            "metadata_repo": metadata_repo,
            "metadata_revision": metadata_revision,
            "presets": normalized_presets,
        }
    normalized = {
        "catalog_id": presets.get("catalog_id"),
        "model_catalog_id": presets.get("model_catalog_id"),
        "models": normalized_models,
        "schema_version": 1,
        "seed": 9271,
    }
    return presets, normalized


def repository_evidence_identity(
    raw_path: Any,
    label: str,
    collector_sha: str,
) -> dict[str, Any]:
    relative = require_safe_relative_path(raw_path, label)
    object_type = git_text(["cat-file", "-t", f"{collector_sha}:{relative}"])
    require(object_type == "blob", f"{label} is not a file in collector commit {collector_sha}: {relative}")
    payload = git_bytes(["show", f"{collector_sha}:{relative}"])
    require(payload, f"{label} is empty in collector commit: {relative}")
    return {
        "path": relative,
        "sha256": bytes_sha256(payload),
        "size_bytes": len(payload),
    }


def validate_historical_catalog(
    catalog_path: Path = BUG_CATALOG_PATH,
    collector_sha: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence_commit = collector_sha or require_git_sha(git_text(["rev-parse", "HEAD"]), "historical evidence commit")
    catalog = read_json(catalog_path)
    require(catalog.get("schema_version") == 1, "historical bug catalog schema_version must be 1")
    require(catalog.get("baseline_git_sha") == FROZEN_LEGACY_SHA, "historical bug catalog baseline SHA mismatch")
    require(catalog.get("family_count") == 15, "historical bug catalog family_count must be 15")
    require(catalog.get("concrete_case_count") == 28, "historical bug catalog concrete_case_count must be 28")
    vocabulary = require_object(catalog.get("evidence_status_vocabulary"), "historical bug catalog evidence_status_vocabulary")
    require(set(vocabulary) == {"bound", "partial", "gap"}, "historical bug catalog evidence status vocabulary mismatch")
    for status, description in vocabulary.items():
        require_string(description, f"historical bug catalog evidence status {status}")
    families = require_list(catalog.get("families"), "historical bug catalog families")
    require(len(families) == 15, "historical bug catalog must contain 15 families")
    seen_families: set[str] = set()
    seen_cases: set[str] = set()
    status_counts: Counter[str] = Counter()
    normalized_families: list[dict[str, Any]] = []
    verified_commits: set[str] = set()
    for family_index, family_raw in enumerate(families):
        family = require_object(family_raw, f"historical bug catalog families[{family_index}]")
        family_id = require_string(family.get("id"), f"historical bug catalog families[{family_index}].id")
        require(FAMILY_RE.fullmatch(family_id) is not None, f"invalid historical bug family id: {family_id}")
        require(family_id not in seen_families, f"duplicate historical bug family id: {family_id}")
        seen_families.add(family_id)
        title = require_string(family.get("title"), f"historical bug catalog {family_id}.title")
        cases = require_list(family.get("cases"), f"historical bug catalog {family_id}.cases")
        require(cases, f"historical bug catalog {family_id} has no cases")
        normalized_cases: list[dict[str, Any]] = []
        for case_index, case_raw in enumerate(cases):
            case = require_object(case_raw, f"historical bug catalog {family_id}.cases[{case_index}]")
            case_id = require_string(case.get("id"), f"historical bug catalog {family_id}.cases[{case_index}].id")
            case_match = CASE_RE.fullmatch(case_id)
            require(case_match is not None and case_match.group(1) == family_id, f"historical case {case_id} does not belong to {family_id}")
            require(case_id not in seen_cases, f"duplicate historical bug case id: {case_id}")
            seen_cases.add(case_id)
            status = require_string(case.get("evidence_status"), f"historical bug catalog {case_id}.evidence_status")
            require(status in vocabulary, f"historical bug catalog {case_id} evidence status is undefined")
            status_counts[status] += 1
            commits: list[dict[str, str]] = []
            for commit_index, commit_raw in enumerate(require_list(case.get("commits"), f"historical bug catalog {case_id}.commits")):
                commit = require_object(commit_raw, f"historical bug catalog {case_id}.commits[{commit_index}]")
                sha = require_git_sha(commit.get("sha"), f"historical bug catalog {case_id}.commits[{commit_index}].sha")
                relation = require_string(commit.get("relation"), f"historical bug catalog {case_id}.commits[{commit_index}].relation")
                if sha not in verified_commits:
                    require(git_text(["cat-file", "-t", sha]) == "commit", f"historical bug catalog commit is unavailable: {sha}")
                    verified_commits.add(sha)
                commits.append({"relation": relation, "sha": sha})
            require(commits or status == "gap", f"historical bug catalog {case_id} lacks a commit but is not marked gap")
            historical_artifacts = [
                repository_evidence_identity(path, f"historical bug catalog {case_id}.historical_artifacts", evidence_commit)
                for path in require_list(case.get("historical_artifacts"), f"historical bug catalog {case_id}.historical_artifacts")
            ]
            reproducer_paths = [
                repository_evidence_identity(path, f"historical bug catalog {case_id}.reproducer_paths", evidence_commit)
                for path in require_list(case.get("reproducer_paths"), f"historical bug catalog {case_id}.reproducer_paths")
            ]
            require(historical_artifacts or reproducer_paths or status == "gap", f"historical bug catalog {case_id} lacks path evidence but is not marked gap")
            entrypoints = [require_string(item, f"historical bug catalog {case_id}.entrypoints") for item in require_list(case.get("entrypoints"), f"historical bug catalog {case_id}.entrypoints")]
            backends = [require_string(item, f"historical bug catalog {case_id}.backends") for item in require_list(case.get("backends"), f"historical bug catalog {case_id}.backends")]
            downstream = [require_string(item, f"historical bug catalog {case_id}.downstream_goals") for item in require_list(case.get("downstream_goals"), f"historical bug catalog {case_id}.downstream_goals")]
            require(entrypoints and backends and downstream, f"historical bug catalog {case_id} classification matrix is incomplete")
            normalized_cases.append(
                {
                    "backends": backends,
                    "commits": commits,
                    "downstream_goals": downstream,
                    "entrypoints": entrypoints,
                    "evidence_status": status,
                    "failure_class": require_string(case.get("failure_class"), f"historical bug catalog {case_id}.failure_class"),
                    "historical_artifacts": historical_artifacts,
                    "id": case_id,
                    "reproducer_paths": reproducer_paths,
                    "title": require_string(case.get("title"), f"historical bug catalog {case_id}.title"),
                }
            )
        normalized_cases.sort(key=lambda item: item["id"])
        normalized_families.append({"cases": normalized_cases, "id": family_id, "title": title})
    require(seen_families == {f"H{index:02d}" for index in range(1, 16)}, "historical bug catalog family ids are incomplete")
    require(len(seen_cases) == 28, "historical bug catalog must contain 28 unique concrete cases")
    require(status_counts == Counter({"bound": 13, "partial": 10, "gap": 5}), "historical bug evidence status distribution changed without an explicit catalog update")
    normalized_families.sort(key=lambda item: item["id"])
    normalized = {
        "catalog_id": catalog.get("catalog_id"),
        "catalog_scope": "catalog_only",
        "concrete_case_count": 28,
        "evidence_status_counts": dict(sorted(status_counts.items())),
        "evidence_status_vocabulary": copy.deepcopy(vocabulary),
        "families": normalized_families,
        "family_count": 15,
        "full_historical_corpus_complete": False,
    }
    return catalog, normalized


def artifact_row(path: Path, root: Path, role: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"artifact is missing or symlinked: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "role": role,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def validate_artifact_index(root: Path, rows: list[dict[str, Any]], expected_paths: set[str]) -> None:
    require({row.get("path") for row in rows} == expected_paths, "artifact index path set is incomplete")
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        row = require_object(raw, f"artifact_index[{index}]")
        relative = require_safe_relative_path(row.get("path"), f"artifact_index[{index}].path")
        require(relative not in seen, f"duplicate artifact index path: {relative}")
        seen.add(relative)
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"indexed artifact is missing or symlinked: {relative}")
        require(row.get("sha256") == file_sha256(path), f"indexed artifact SHA256 mismatch: {relative}")
        require(row.get("size_bytes") == path.stat().st_size and path.stat().st_size > 0, f"indexed artifact size mismatch: {relative}")
        require_string(row.get("role"), f"artifact_index[{index}].role")


def contract_row(collector: dict[str, Any], relative: str) -> dict[str, Any]:
    for row in collector["contracts"]:
        if row["path"] == relative:
            return row
    raise CheckpointError(f"collector contract identity missing: {relative}")


def build_model_facts_lock(
    *,
    collector: dict[str, Any],
    frozen: dict[str, str],
    inventory: dict[str, Any],
    inventory_fingerprint: str,
    review: dict[str, Any],
    resolution_facts_sha256: str,
    resolution: dict[str, Any],
    lanes: list[dict[str, Any]],
    model_catalog: dict[str, Any],
    models_catalog_path: Path,
    presets_catalog_path: Path,
    bug_catalog_path: Path,
    inventory_review_path: Path,
    normalized_presets: dict[str, Any],
    normalized_history: dict[str, Any],
) -> dict[str, Any]:
    summary = require_object(inventory.get("summary"), "coupling-inventory.summary")
    review_counts = require_object(review.get("classification_counts"), "inventory-review.classification_counts")
    return {
        "artifact_type": "runtime_vnext_g00a_model_facts_lock",
        "checkpoint_id": CHECKPOINT_ID,
        "collector": {
            "contracts_sha256": collector["contracts_sha256"],
            "git_sha": collector["git_sha"],
            "git_tree_sha": collector["git_tree_sha"],
        },
        "frozen_legacy_source": copy.deepcopy(frozen),
        "generation_presets": {
            "catalog_sha256": file_sha256(presets_catalog_path),
            "facts": normalized_presets,
        },
        "historical_bug_catalog": {
            "catalog_sha256": file_sha256(bug_catalog_path),
            "facts": normalized_history,
        },
        "inventory": {
            "analyzer": copy.deepcopy(inventory["analyzer"]),
            "analyzer_contract": contract_row(collector, "scripts/release/runtime_vnext_inventory.py"),
            "normalized_inventory_sha256": inventory_fingerprint,
            "review": {
                "candidate_count": review.get("candidate_count"),
                "classification_counts": dict(sorted(review_counts.items())),
                "large_native_content_root_count": review.get("large_native_content_root_count"),
                "sha256": file_sha256(inventory_review_path),
                "unresolved_count": review.get("unresolved_count"),
            },
            "summary": copy.deepcopy(summary),
        },
        "model_catalog": {
            "catalog_id": model_catalog.get("catalog_id"),
            "catalog_sha256": file_sha256(models_catalog_path),
            "lane_count": 12,
        },
        "model_resolution": {
            "catalog_sha256": resolution.get("catalog_sha256"),
            "live_facts_sha256": resolution_facts_sha256,
            "live_recomputed": True,
            "resolver": copy.deepcopy(resolution.get("resolver")),
            "source": copy.deepcopy(resolution.get("source")),
        },
        "models": copy.deepcopy(lanes),
        "scope": {
            "does_not_prove": [
                "G00",
                "G01B",
                "model_migration",
                "performance",
                "release",
            ],
            "historical_evidence": "catalog_only",
            "unlocks": ["G01A"],
        },
        "schema_version": SCHEMA_VERSION,
    }


def create_checkpoint(inventory_arg: Path, resolution_arg: Path, out_arg: Path) -> Path:
    inventory_input = require_external_input(inventory_arg, "--coupling-inventory")
    resolution_input = require_external_input(resolution_arg, "--model-resolution")
    require(inventory_input != resolution_input, "inventory and model resolution inputs must be distinct files")
    out = require_external_output(out_arg)

    collector = collector_identity()
    frozen = frozen_source_identity()
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))
    try:
        snapshot_regular_file(inventory_input, staging / "coupling-inventory.json", "--coupling-inventory")
        snapshot_regular_file(resolution_input, staging / "model-resolution.input.json", "--model-resolution")
        contract_copies = {
            "generation-presets.catalog.json": PRESETS_CATALOG_PATH,
            "historical-bugs.catalog.json": BUG_CATALOG_PATH,
            "inventory-review.catalog.json": INVENTORY_REVIEW_PATH,
            "models.catalog.json": MODELS_CATALOG_PATH,
        }
        for name, source in contract_copies.items():
            snapshot_collector_contract(
                collector,
                source.relative_to(REPO_ROOT).as_posix(),
                staging / name,
            )

        inventory_path = staging / "coupling-inventory.json"
        resolution_input_path = staging / "model-resolution.input.json"
        resolution_path = staging / "model-resolution.json"
        models_catalog_path = staging / "models.catalog.json"
        presets_catalog_path = staging / "generation-presets.catalog.json"
        bug_catalog_path = staging / "historical-bugs.catalog.json"
        inventory_review_path = staging / "inventory-review.catalog.json"

        inventory = read_json(inventory_path)
        validate_inventory_shape(inventory, frozen)
        recomputed_inventory = recompute_inventory(frozen, collector)
        inventory_fingerprint = compare_inventory(inventory, recomputed_inventory)
        review = validate_inventory_review(inventory, inventory_review_path)
        model_catalog, indexed_catalog = validate_models_catalog(models_catalog_path)
        input_resolution, input_lanes = validate_model_resolution(
            resolution_input_path,
            collector,
            model_catalog,
            indexed_catalog,
            catalog_path=models_catalog_path,
        )
        live_resolution = recompute_model_resolution(resolution_path, collector)
        resolution, lanes = validate_model_resolution(
            resolution_path,
            collector,
            model_catalog,
            indexed_catalog,
            catalog_path=models_catalog_path,
        )
        require(live_resolution == resolution, "live model resolution changed between write and validation")
        resolution_facts_sha256 = compare_model_resolution_facts(input_lanes, lanes)
        require(
            input_resolution.get("catalog_sha256") == resolution.get("catalog_sha256"),
            "model resolution input/live catalog identity mismatch",
        )
        _, normalized_presets = validate_presets_catalog(
            model_catalog,
            indexed_catalog,
            lanes,
            presets_catalog_path,
        )
        _, normalized_history = validate_historical_catalog(
            bug_catalog_path,
            collector["git_sha"],
        )

        lock = build_model_facts_lock(
            collector=collector,
            frozen=frozen,
            inventory=inventory,
            inventory_fingerprint=inventory_fingerprint,
            review=review,
            resolution_facts_sha256=resolution_facts_sha256,
            resolution=resolution,
            lanes=lanes,
            model_catalog=model_catalog,
            models_catalog_path=models_catalog_path,
            presets_catalog_path=presets_catalog_path,
            bug_catalog_path=bug_catalog_path,
            inventory_review_path=inventory_review_path,
            normalized_presets=normalized_presets,
            normalized_history=normalized_history,
        )
        write_json(staging / "model-facts.lock.json", lock)

        roles = {
            "coupling-inventory.json": "frozen-cff4-current-analyzer-inventory",
            "generation-presets.catalog.json": "generation-preset-catalog",
            "historical-bugs.catalog.json": "historical-bug-catalog-only",
            "inventory-review.catalog.json": "inventory-human-review-catalog",
            "model-facts.lock.json": "deterministic-g00a-fact-lock",
            "model-resolution.input.json": "caller-supplied-resolution-snapshot",
            "model-resolution.json": "checkpoint-live-hugging-face-recheck",
            "models.catalog.json": "model-lane-catalog",
        }
        artifact_index = [artifact_row(staging / name, staging, role) for name, role in sorted(roles.items())]
        validate_artifact_index(staging, artifact_index, set(roles))
        indexed_artifacts = {row["path"]: row for row in artifact_index}
        pass_line = f"{PASS_PREFIX}: {out}"
        manifest = {
            "artifact_count": len(artifact_index),
            "artifact_dir": str(out),
            "artifact_index": artifact_index,
            "artifact_index_policy": {
                "manifest_self_digest": "excluded-to-avoid-recursive-digest",
                "non_manifest_artifacts_indexed": True,
            },
            "artifact_type": "runtime_vnext_g00a_fact_checkpoint_manifest",
            "canonical": True,
            "checkpoint_id": CHECKPOINT_ID,
            "collector": collector,
            "dirty": False,
            "does_not_prove": ["G00", "G01B", "model_migration", "performance", "release"],
            "fact_source_artifacts": {
                "coupling_inventory": copy.deepcopy(indexed_artifacts["coupling-inventory.json"]),
                "model_resolution_input": copy.deepcopy(indexed_artifacts["model-resolution.input.json"]),
                "model_resolution_live": copy.deepcopy(indexed_artifacts["model-resolution.json"]),
            },
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
            "git_sha": collector["git_sha"],
            "git_tree_sha": collector["git_tree_sha"],
            "lane": "runtime-vnext-g00a",
            "model_facts_lock": {
                "path": "model-facts.lock.json",
                "sha256": file_sha256(staging / "model-facts.lock.json"),
                "size_bytes": (staging / "model-facts.lock.json").stat().st_size,
            },
            "pass_line": pass_line,
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "unlocks": ["G01A"],
        }
        write_json(staging / "manifest.json", manifest)
        require(read_json(staging / "manifest.json") == manifest, "manifest round-trip verification failed")
        validate_artifact_index(staging, artifact_index, set(roles))
        require(collector_identity() == collector, "collector identity changed during checkpoint collection")
        os.replace(staging, out)
        staging = Path()
    finally:
        if staging != Path() and staging.exists():
            shutil.rmtree(staging)
    return out


def expect_rejected(label: str, action: Any, *, marker: str | None = None) -> None:
    try:
        action()
    except CheckpointError as exc:
        if marker is not None and marker not in str(exc):
            raise CheckpointError(
                f"self-test {label} rejected for an unexpected reason: {exc}"
            ) from exc
        return
    raise CheckpointError(f"self-test expected rejection: {label}")


def fixture_file(path: str, digest: str, size: int = 7) -> dict[str, Any]:
    return {
        "content_request_url": f"https://huggingface.co/acme/model/resolve/{'a' * 40}/{path}",
        "git_oid": "b" * 40,
        "path": path,
        "sha256": digest,
        "sha256_source": "downloaded_content",
        "size_bytes": size,
    }


def fixture_lfs_file(path: str, digest: str, size: int = 7) -> dict[str, Any]:
    return {
        "git_oid": "b" * 40,
        "lfs_oid": digest,
        "path": path,
        "sha256": digest,
        "sha256_source": "hugging_face_lfs_oid",
        "size_bytes": size,
    }


def fixture_json_request(kind: str, url: str, body: Any) -> dict[str, Any]:
    payload = canonical_bytes(body)
    return {
        "kind": kind,
        "method": "GET",
        "response_body_base64": base64.b64encode(payload).decode("ascii"),
        "response_bytes": len(payload),
        "response_sha256": bytes_sha256(payload),
        "status": 200,
        "url": url,
    }


def fixture_source(repo: str, revision: str, requested: dict[str, Any], files: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "files": files,
        "gated": False,
        "license": {"files": [], "hugging_face_id": "apache-2.0"},
        "model_request_url": f"https://huggingface.co/api/models/{repo}/revision/{revision}",
        "repo": repo,
        "requested_revision": requested,
        "revision": revision,
        "tree_request_urls": [f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true"],
    }


def resolution_selftest_fixture() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    revision = "a" * 40
    config_sha = "1" * 64
    weight_sha = "2" * 64
    readme_sha = "3" * 64
    tokenizer_sha = "4" * 64
    lane = {
        "backend": "cuda",
        "catalog_lane_id": "T-CUDA",
        "chat_template": {
            "container_sha256": tokenizer_sha,
            "content_bytes": 20,
            "content_sha256": "5" * 64,
            "json_pointer": "/chat_template",
            "path": "tokenizer_config.json",
            "repo": "acme/model",
            "revision": revision,
            "source": "semantic_source",
        },
        "format": "test",
        "generation_config": {
            "path": "generation_config.json",
            "policy": "absent",
            "present": False,
            "repo": "acme/model",
            "revision": revision,
            "source": "semantic_source",
        },
        "model_id": "T",
        "semantic_source": fixture_source(
            "acme/model",
            revision,
            {"status": "pinned", "value": revision},
            [
                fixture_file("README.md", readme_sha),
                fixture_file("tokenizer_config.json", tokenizer_sha),
            ],
        ),
        "tokenizer_source": None,
        "weight_source": fixture_source(
            "acme/model",
            revision,
            {"status": "pinned", "value": revision},
            [
                fixture_file("config.json", config_sha),
                fixture_lfs_file("model.safetensors", weight_sha),
            ],
        ),
    }
    catalog_lane = {
        "backend": "cuda",
        "files": [
            {"path": "config.json", "required": True},
            {
                "path": "model.safetensors",
                "required": True,
                "expected_size_bytes": 7,
                "expected_sha256": weight_sha,
            },
        ],
        "format": "test",
        "hardware_policy": "fixture",
        "id": "T-CUDA",
        "model_id": "T",
        "reference": {
            "chat_template_source": {
                "container_sha256": tokenizer_sha,
                "content_sha256": "5" * 64,
                "json_pointer": "/chat_template",
                "path": "tokenizer_config.json",
                "source": "semantic_source",
            },
            "generation_config_source": {
                "path": "generation_config.json",
                "policy": "absent",
                "source": "semantic_source",
            },
            "required_semantic_files": ["README.md", "tokenizer_config.json"],
            "semantic_file_sha256": {
                "README.md": readme_sha,
                "tokenizer_config.json": tokenizer_sha,
            },
            "semantic_repo": "acme/model",
            "semantic_revision": {"status": "same_as_weight_revision"},
        },
        "repo": "acme/model",
        "revision": {"status": "pinned", "value": revision},
        "role": "fixture",
    }
    catalog = {"catalog_id": "fixture", "models": [catalog_lane]}
    return catalog, {"T-CUDA": catalog_lane}, {"lanes": [lane]}


def run_self_test() -> None:
    require(
        validate_safetensors_shard_paths(
            {
                "model-00001-of-000002.safetensors",
                "model-00002-of-000002.safetensors",
                "model.safetensors.index.json",
            },
            "valid shard fixture",
        ),
        "valid 5/6-width shard fixture was not detected as sharded",
    )
    expect_rejected(
        "shard total differs from file count",
        lambda: validate_safetensors_shard_paths(
            {
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors",
            },
            "incomplete shard fixture",
        ),
    )
    expect_rejected(
        "multiple non-canonical safetensors files",
        lambda: validate_safetensors_shard_paths(
            {"model-a.safetensors", "model-b.safetensors"},
            "non-canonical shard fixture",
        ),
    )
    expect_rejected(
        "duplicate JSON keys",
        lambda: read_json_bytes(b'{"fact":1,"fact":2}', "duplicate-key fixture"),
    )
    expect_rejected(
        "non-finite JSON number",
        lambda: read_json_bytes(b'{"fact":NaN}', "non-finite fixture"),
    )
    lfs_index = fixture_lfs_file("model.safetensors.index.json", "6" * 64)
    lfs_index["content_request_url"] = (
        f"https://huggingface.co/acme/model/resolve/{'a' * 40}/model.safetensors.index.json"
    )
    lfs_index["lfs_metadata_downloaded"] = True
    normalized_lfs_index = normalized_file(lfs_index, "LFS index fixture")
    require(
        normalized_lfs_index.get("content_request_url") == lfs_index["content_request_url"]
        and normalized_lfs_index.get("lfs_metadata_downloaded") is True,
        "LFS index fixture lost download provenance during normalization",
    )
    missing_lfs_index_provenance = copy.deepcopy(lfs_index)
    missing_lfs_index_provenance.pop("content_request_url")
    expect_rejected(
        "LFS index missing download provenance",
        lambda: normalized_file(missing_lfs_index_provenance, "missing LFS index provenance"),
    )
    forbidden_weight_download = fixture_lfs_file("model.safetensors", "7" * 64)
    forbidden_weight_download["content_request_url"] = (
        f"https://huggingface.co/acme/model/resolve/{'a' * 40}/model.safetensors"
    )
    forbidden_weight_download["lfs_metadata_downloaded"] = True
    expect_rejected(
        "weight LFS download provenance",
        lambda: normalized_file(forbidden_weight_download, "weight LFS download"),
    )
    forged_downloaded_lfs = fixture_file("config.json", "8" * 64)
    forged_downloaded_lfs["lfs_metadata_downloaded"] = True
    expect_rejected(
        "downloaded metadata with LFS flag",
        lambda: normalized_file(forged_downloaded_lfs, "downloaded metadata LFS flag"),
        marker="must not contain LFS identity or download flags",
    )
    request_payload = canonical_bytes({"sha": "a" * 40})
    request_fixture = {
        "response_body_base64": base64.b64encode(request_payload).decode("ascii"),
        "response_bytes": len(request_payload),
        "response_sha256": bytes_sha256(request_payload),
    }
    require(
        request_json_body(request_fixture, "self-test request") == {"sha": "a" * 40},
        "request body replay mismatch",
    )
    tampered_request = copy.deepcopy(request_fixture)
    tampered_request["response_sha256"] = "0" * 64
    expect_rejected(
        "request response body digest",
        lambda: request_json_body(tampered_request, "tampered self-test request"),
    )
    inventory_a = {
        "root": "/a",
        "schema_version": 1,
        "summary": {"file_count": 1},
        "files": [{"path": "crates/a.rs", "sha256": "1" * 64}],
    }
    inventory_b = copy.deepcopy(inventory_a)
    inventory_b["root"] = "/b"
    compare_inventory(inventory_a, inventory_b)
    tampered_candidate = copy.deepcopy(inventory_a)
    tampered_candidate["summary"]["file_count"] = 2
    expect_rejected("candidate-side inventory tamper", lambda: compare_inventory(tampered_candidate, inventory_b))
    tampered_recompute = copy.deepcopy(inventory_b)
    tampered_recompute["files"][0]["sha256"] = "2" * 64
    expect_rejected("analyzer-side inventory tamper", lambda: compare_inventory(inventory_a, tampered_recompute))

    catalog, indexed, resolution = resolution_selftest_fixture()
    facts = validate_resolution_matrix(catalog, indexed, resolution, expected_lane_count=1)
    require(len(facts) == 1, "self-test resolution fixture did not produce one fact")
    provenance_requests: dict[tuple[str, str], dict[str, Any]] = {}
    provenance_tree_rows: dict[str, dict[str, Any]] = {}
    for source_name in ("weight_source", "semantic_source"):
        source = facts[0][source_name]
        model_url = source["model_request_url"]
        tree_url = source["tree_request_urls"][0]
        provenance_requests[("model-info", model_url)] = fixture_json_request(
            "model-info",
            model_url,
            {"sha": source["revision"]},
        )
        for file_row in source["files"]:
            tree_row: dict[str, Any] = {
                "oid": file_row["git_oid"],
                "path": file_row["path"],
                "size": file_row["size_bytes"],
                "type": "file",
            }
            if file_row["sha256_source"] == "hugging_face_lfs_oid":
                tree_row["lfs"] = {
                    "oid": f"sha256:{file_row['sha256']}",
                    "size": file_row["size_bytes"],
                }
            else:
                content_url = file_row["content_request_url"]
                provenance_requests[("metadata-file", content_url)] = {
                    "kind": "metadata-file",
                    "method": "GET",
                    "response_bytes": file_row["size_bytes"],
                    "response_sha256": file_row["sha256"],
                    "status": 200,
                    "url": content_url,
                }
            provenance_tree_rows[file_row["path"]] = tree_row
        provenance_requests[("repo-tree", tree_url)] = fixture_json_request(
            "repo-tree",
            tree_url,
            sorted(provenance_tree_rows.values(), key=lambda row: row["path"]),
        )
    provenance_fixture = {"requests": list(provenance_requests.values())}
    validate_resolution_provenance(provenance_fixture, facts)
    forged_tree_resolution = copy.deepcopy(provenance_fixture)
    forged_tree_request = next(
        row
        for row in forged_tree_resolution["requests"]
        if row["kind"] == "repo-tree"
    )
    forged_tree_body = request_json_body(forged_tree_request, "forged tree fixture")
    forged_config = next(row for row in forged_tree_body if row["path"] == "config.json")
    forged_config["lfs"] = {"oid": f"sha256:{'1' * 64}", "size": 7}
    forged_tree_request.update(
        fixture_json_request(
            "repo-tree",
            forged_tree_request["url"],
            forged_tree_body,
        )
    )
    expect_rejected(
        "downloaded metadata backed by authoritative LFS tree",
        lambda: validate_resolution_provenance(forged_tree_resolution, facts),
        marker="downloaded metadata is LFS-backed in the authoritative tree",
    )
    duplicate_license_resolution = copy.deepcopy(resolution)
    duplicate_weight = duplicate_license_resolution["lanes"][0]["weight_source"]
    duplicate_weight["license"]["files"].append(
        copy.deepcopy(duplicate_weight["files"][1])
    )
    expect_rejected(
        "source/license duplicate path",
        lambda: validate_resolution_matrix(
            catalog,
            indexed,
            duplicate_license_resolution,
            expected_lane_count=1,
        ),
        marker="files and license.files contain duplicate paths",
    )
    unsharded_conditional_catalog = copy.deepcopy(catalog)
    unsharded_conditional_lane = unsharded_conditional_catalog["models"][0]
    unsharded_conditional_lane["files"].append({"glob": "*", "required": True})
    unsharded_conditional_lane["files"].append(
        {"path": "model.safetensors.index.json", "required_if_sharded": True}
    )
    unsharded_conditional_resolution = copy.deepcopy(resolution)
    unsharded_conditional_resolution["lanes"][0]["weight_source"]["files"].append(
        fixture_file("model.safetensors.index.json", "8" * 64)
    )
    expect_rejected(
        "unsharded conditional index",
        lambda: validate_resolution_matrix(
            unsharded_conditional_catalog,
            {"T-CUDA": unsharded_conditional_lane},
            unsharded_conditional_resolution,
            expected_lane_count=1,
        ),
        marker="unsharded model contains a conditional weight file",
    )
    compare_model_resolution_facts(facts, copy.deepcopy(facts))
    forged_lfs_fact = copy.deepcopy(facts)
    forged_lfs_fact[0]["weight_source"]["files"][1]["sha256"] = "9" * 64
    expect_rejected(
        "caller-forged LFS fact versus live re-resolution",
        lambda: compare_model_resolution_facts(forged_lfs_fact, facts),
    )
    coherent_expected_sha_mismatch = copy.deepcopy(resolution)
    coherent_weight = coherent_expected_sha_mismatch["lanes"][0]["weight_source"]["files"][1]
    coherent_weight["sha256"] = "9" * 64
    coherent_weight["lfs_oid"] = "9" * 64
    expect_rejected(
        "coherent resolved LFS identity differs from catalog expected SHA256",
        lambda: validate_resolution_matrix(
            catalog,
            indexed,
            coherent_expected_sha_mismatch,
            expected_lane_count=1,
        ),
        marker="expected SHA256 mismatch for model.safetensors",
    )
    redacted_expected_lfs = copy.deepcopy(resolution)
    redacted_expected_lfs["lanes"][0]["weight_source"]["files"][1]["lfs_oid"] = "*" * 64
    expect_rejected(
        "redacted LFS identity for catalog expected SHA256",
        lambda: validate_resolution_matrix(
            catalog,
            indexed,
            redacted_expected_lfs,
            expected_lane_count=1,
        ),
        marker="lfs_oid must equal its SHA256",
    )
    non_lfs_expected_sha = copy.deepcopy(resolution)
    non_lfs_weight = non_lfs_expected_sha["lanes"][0]["weight_source"]["files"][1]
    non_lfs_weight.pop("lfs_oid")
    non_lfs_weight["sha256_source"] = "downloaded_content"
    non_lfs_weight["content_request_url"] = (
        f"https://huggingface.co/acme/model/resolve/{'a' * 40}/model.safetensors"
    )
    expect_rejected(
        "non-LFS source for catalog expected SHA256",
        lambda: validate_resolution_matrix(
            catalog,
            indexed,
            non_lfs_expected_sha,
            expected_lane_count=1,
        ),
        marker="requires Hugging Face LFS identity",
    )
    missing_weight_fact = copy.deepcopy(facts)
    missing_weight_fact[0]["weight_source"]["files"].clear()
    expect_rejected(
        "caller-omitted shard versus live re-resolution",
        lambda: compare_model_resolution_facts(missing_weight_fact, facts),
    )
    tampered_resolution = copy.deepcopy(resolution)
    tampered_resolution["lanes"][0]["weight_source"]["repo"] = "acme/tampered"
    expect_rejected(
        "resolution-side model tamper",
        lambda: validate_resolution_matrix(catalog, indexed, tampered_resolution, expected_lane_count=1),
    )
    tampered_catalog = copy.deepcopy(indexed)
    tampered_catalog["T-CUDA"]["repo"] = "acme/catalog-tamper"
    expect_rejected(
        "catalog-side model tamper",
        lambda: validate_resolution_matrix(catalog, tampered_catalog, resolution, expected_lane_count=1),
    )

    validate_models_catalog()
    validate_historical_catalog()
    deterministic = {"facts": facts, "scope": {"unlocks": ["G01A"]}}
    require(canonical_bytes(deterministic) == canonical_bytes(copy.deepcopy(deterministic)), "model fact serialization is not deterministic")

    with tempfile.TemporaryDirectory(prefix="ferrum-g00a-index-selftest-") as raw_tmp:
        root = Path(raw_tmp)
        snapshot_source = root / "snapshot-source.json"
        snapshot_destination = root / "snapshot-destination.json"
        snapshot_payload = canonical_bytes({"fact": "validated-and-archived"})
        snapshot_source.write_bytes(snapshot_payload)
        require(
            snapshot_regular_file(snapshot_source, snapshot_destination, "self-test snapshot")
            == snapshot_payload,
            "snapshot payload mismatch",
        )
        snapshot_source.write_bytes(canonical_bytes({"fact": "changed-after-snapshot"}))
        require(
            snapshot_destination.read_bytes() == snapshot_payload,
            "snapshot destination changed with its source",
        )
        symlink_source = root / "snapshot-symlink.json"
        symlink_source.symlink_to(snapshot_source)
        expect_rejected(
            "symlink snapshot source",
            lambda: snapshot_regular_file(symlink_source, root / "symlink-copy.json", "self-test symlink"),
        )
        artifact = root / "fact.json"
        write_json(artifact, {"fact": "clean"})
        rows = [artifact_row(artifact, root, "fixture")]
        validate_artifact_index(root, rows, {"fact.json"})
        write_json(artifact, {"fact": "tampered"})
        expect_rejected("artifact-index tamper", lambda: validate_artifact_index(root, rows, {"fact.json"}))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coupling-inventory",
        "--inventory",
        dest="coupling_inventory",
        type=Path,
        help="clean frozen-cff4 coupling-inventory.json produced outside the source tree",
    )
    parser.add_argument(
        "--model-resolution",
        type=Path,
        help="clean current-HEAD live model-resolution.json produced outside the source tree",
    )
    parser.add_argument("--out", type=Path, help="fresh output directory outside the Git source tree")
    parser.add_argument("--self-test", action="store_true", help="run deterministic bilateral tamper tests")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.self_test:
            require(args.coupling_inventory is None and args.model_resolution is None and args.out is None, "--self-test does not accept canonical collection arguments")
            run_self_test()
            print(SELFTEST_PASS_LINE)
            return 0
        require(args.coupling_inventory is not None, "--coupling-inventory is required")
        require(args.model_resolution is not None, "--model-resolution is required")
        require(args.out is not None, "--out is required")
        out = create_checkpoint(args.coupling_inventory, args.model_resolution, args.out)
        print(f"{PASS_PREFIX}: {out}")
        return 0
    except CheckpointError as exc:
        print(f"G00a checkpoint ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
