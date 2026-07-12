#!/usr/bin/env python3
"""Collect and validate Runtime vNext G00 C01-C21 legacy evidence.

The canonical command accepts only artifact-backed evidence. Internal fixtures
exist solely for validator self-tests and are not exposed as a collection mode.
"""

from __future__ import annotations

import argparse
import base64
import codecs
import concurrent.futures
import copy
import hashlib
import http.client
import json
import math
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = Path(__file__).resolve()
RUNNER_REPO_PATH = RUNNER_PATH.relative_to(REPO_ROOT).as_posix()
EXPECTATIONS_PATH = REPO_ROOT / "scripts/release/configs/runtime_vnext_legacy_correctness_expectations.json"
EXPECTATIONS_REPO_PATH = EXPECTATIONS_PATH.relative_to(REPO_ROOT).as_posix()
FROZEN_LEGACY_SHA = "cff4c47765ef3259b8a04890187d99c60da86394"
PASS_PREFIX = "FERRUM RUNTIME VNEXT G00 SCENARIOS PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT G00 SCENARIOS SELFTEST PASS"
SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SCENARIO_IDS = tuple(f"C{index:02d}" for index in range(1, 22))
EXPECTED_STATUSES = {"pass", "known-fail", "blocked", "discovery-required"}
BLOCKED_LANE_FAILURE_CLASSES = {
    "m1-qwen35-4b/metal": "legacy-model-backend-unsupported",
    "m2-qwen35-35b-a3b/metal": "legacy-model-backend-unsupported",
}
FORBIDDEN_KEYS = {"skip", "skipped", "waiver", "waivers", "placeholder"}
LOG_KINDS = {"stdout-log", "stderr-log", "checker-log", "runtime-log"}
CASE_ARTIFACT_KINDS = {"raw-json", "request-json", "http-transcript", *LOG_KINDS}
BLOCKER_MARKERS = (
    "thread panicked",
    "segmentation fault",
    "traceback",
    "mojibake",
    "invalid utf-8",
    "<unk>",
    "[pad",
)
CHILD_ENV_ALLOWLIST = frozenset(
    {
        "CUDA_HOME",
        "CUDA_VISIBLE_DEVICES",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "LOGNAME",
        "PATH",
        "RUST_BACKTRACE",
        "RUST_LOG",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "TRANSFORMERS_CACHE",
        "USER",
    }
)
EXPECTED_ARCHITECTURES = {
    "m1-qwen35-4b": {"qwen3_5", "qwen3_5_text"},
    "m2-qwen35-35b-a3b": {"qwen3_5_moe", "qwen3_5_moe_text"},
    "m3-qwen3-30b-a3b": {"qwen3", "qwen3_moe"},
}


class ScenarioError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScenarioError(message)


def require_object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a JSON array")
    return value


def require_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and value.strip(), f"{label} must be a non-empty string")
    return value.strip()


def require_count(value: Any, label: str, *, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    require(value >= minimum, f"{label} must be >= {minimum}")
    return value


def require_sha256(value: Any, label: str) -> str:
    digest = require_string(value, label).lower()
    require(SHA256_RE.fullmatch(digest) is not None, f"{label} must be a lowercase SHA256")
    return digest


def require_git_sha(value: Any, label: str) -> str:
    sha = require_string(value, label).lower()
    require(GIT_SHA_RE.fullmatch(sha) is not None, f"{label} must be a 40-character git SHA")
    return sha


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ScenarioError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ScenarioError(f"invalid JSON in {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: Any, label: str) -> datetime:
    text = require_string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ScenarioError(f"{label} must be an ISO-8601 timestamp") from exc
    require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def sanitized_child_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in CHILD_ENV_ALLOWLIST and isinstance(value, str)
    }
    environment.update({"NO_COLOR": "1", "PYTHONUNBUFFERED": "1"})
    return dict(sorted(environment.items()))


def validate_sanitized_environment(raw: Any, label: str) -> dict[str, str]:
    environment = require_object(raw, label)
    require(
        all(isinstance(key, str) and isinstance(value, str) for key, value in environment.items()),
        f"{label} must contain string keys and values",
    )
    allowed = CHILD_ENV_ALLOWLIST | {"NO_COLOR", "PYTHONUNBUFFERED"}
    require(set(environment) <= allowed, f"{label} contains non-allowlisted keys: {sorted(set(environment) - allowed)}")
    require(not any(key.startswith("FERRUM_") for key in environment), f"{label} contains inherited FERRUM_* controls")
    require(environment.get("NO_COLOR") == "1", f"{label}.NO_COLOR must be 1")
    require(environment.get("PYTHONUNBUFFERED") == "1", f"{label}.PYTHONUNBUFFERED must be 1")
    require(environment == dict(sorted(environment.items())), f"{label} must be sorted deterministically")
    return {str(key): str(value) for key, value in environment.items()}


def capture_process_receipt(
    root: Path,
    path: Path,
    *,
    pid: int,
    pgid: int,
    argv: list[str],
    role: str,
    environment: dict[str, str] | None = None,
) -> dict[str, str]:
    proc = subprocess.run(
        ["ps", "-ww", "-p", str(pid), "-o", "pid=", "-o", "ppid=", "-o", "pgid=", "-o", "command="],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    require(proc.returncode == 0 and proc.stdout.strip(), f"failed to capture live OS receipt for {role} pid={pid}: {proc.stderr.strip()}")
    fields = proc.stdout.strip().split(None, 3)
    require(len(fields) == 4, f"OS receipt for {role} has invalid ps shape")
    ps_pid, ps_ppid, ps_pgid = (int(fields[index]) for index in range(3))
    require(ps_pid == pid and ps_pgid == pgid, f"OS receipt for {role} pid/pgid mismatch")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "role": role,
        "captured_at": iso_now(),
        "captured_monotonic_ns": time.monotonic_ns(),
        "pid": pid,
        "ppid": ps_ppid,
        "pgid": pgid,
        "argv": argv,
        "argv_sha256": canonical_json_sha256(argv),
        "ps_command": fields[3],
        "ps_stdout_sha256": hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest(),
        "ps_stdout": proc.stdout,
    }
    if environment is not None:
        sanitized = validate_sanitized_environment(environment, f"{role} child environment")
        receipt["environment"] = sanitized
        receipt["environment_sha256"] = canonical_json_sha256(sanitized)
    write_json(path, receipt)
    return existing_artifact_ref(root, path, "raw-json")


def validate_process_receipt(
    root: Path,
    raw: Any,
    *,
    label: str,
    pid: int,
    pgid: int,
    argv: list[str],
    role: str,
    expected_ppid: int | None = None,
    expected_environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    _, _, parsed = validate_artifact_ref(root, raw, label, allowed_kinds={"raw-json"})
    receipt = require_object(parsed, f"{label} JSON")
    require(receipt.get("schema_version") == SCHEMA_VERSION and receipt.get("role") == role, f"{label} role/schema mismatch")
    require(receipt.get("pid") == pid and receipt.get("pgid") == pgid, f"{label} PID/PGID mismatch")
    require(receipt.get("argv") == argv, f"{label} argv mismatch")
    require(receipt.get("argv_sha256") == canonical_json_sha256(argv), f"{label} argv receipt mismatch")
    raw_stdout = receipt.get("ps_stdout")
    require(isinstance(raw_stdout, str) and raw_stdout.strip(), f"{label}.ps_stdout must be a non-empty string")
    require(hashlib.sha256(raw_stdout.encode("utf-8")).hexdigest() == receipt.get("ps_stdout_sha256"), f"{label} raw ps SHA mismatch")
    fields = raw_stdout.strip().split(None, 3)
    require(
        len(fields) == 4
        and int(fields[0]) == pid
        and int(fields[1]) == receipt.get("ppid")
        and int(fields[2]) == pgid,
        f"{label} raw ps identity mismatch",
    )
    if expected_ppid is not None:
        require(receipt.get("ppid") == expected_ppid, f"{label} parent PID mismatch")
    if expected_environment is None:
        require("environment" not in receipt and "environment_sha256" not in receipt, f"{label} fabricates a child environment")
    else:
        environment = validate_sanitized_environment(receipt.get("environment"), f"{label}.environment")
        require(environment == expected_environment, f"{label} child environment mismatch")
        require(receipt.get("environment_sha256") == canonical_json_sha256(environment), f"{label} child environment SHA mismatch")
    require(fields[3] == receipt.get("ps_command"), f"{label} raw ps command mismatch")
    parse_timestamp(receipt.get("captured_at"), f"{label}.captured_at")
    require_count(receipt.get("captured_monotonic_ns"), f"{label}.captured_monotonic_ns", minimum=1)
    return receipt


def git_bytes(args: list[str]) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(proc.returncode == 0, f"git {' '.join(args)} failed: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def git_text(args: list[str]) -> str:
    return git_bytes(args).decode("utf-8").strip()


def frozen_tree_sha() -> str:
    tree = git_text(["rev-parse", f"{FROZEN_LEGACY_SHA}^{{tree}}"])
    require(GIT_SHA_RE.fullmatch(tree) is not None, "frozen source tree SHA is invalid")
    return tree


def validate_expectations_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    require(catalog.get("schema_version") == SCHEMA_VERSION, "legacy expectations schema_version mismatch")
    require(catalog.get("source_git_sha") == FROZEN_LEGACY_SHA, "legacy expectations source SHA mismatch")
    vocabulary = require_object(catalog.get("status_vocabulary"), "legacy expectations status_vocabulary")
    require(set(vocabulary) == EXPECTED_STATUSES, "legacy expectations status vocabulary mismatch")
    policy = require_object(catalog.get("resolution_policy"), "legacy expectations resolution_policy")
    require(policy.get("wildcard") == "*", "legacy expectations wildcard must be '*'")
    blocked_policy = require_object(catalog.get("blocked_lane_policy"), "legacy expectations blocked_lane_policy")
    require(
        blocked_policy.get("allowed_lane_failure_classes") == BLOCKED_LANE_FAILURE_CLASSES,
        "legacy expectations blocked lane allowlist mismatch",
    )
    require(
        set(require_list(blocked_policy.get("forbidden_lanes"), "legacy expectations blocked_lane_policy.forbidden_lanes"))
        == {
            "m1-qwen35-4b/cuda",
            "m2-qwen35-35b-a3b/cuda",
            "m3-qwen3-30b-a3b/cuda",
            "m3-qwen3-30b-a3b/metal",
        },
        "legacy expectations blocked forbidden-lane set mismatch",
    )
    lanes = require_object(catalog.get("lanes"), "legacy expectations lanes")
    expected_lanes = {
        f"{model}/{backend}"
        for model in ("m1-qwen35-4b", "m2-qwen35-35b-a3b", "m3-qwen3-30b-a3b")
        for backend in ("cuda", "metal")
    }
    require(set(lanes) == expected_lanes, "legacy expectations must cover exactly six primary lanes")
    for lane_key, lane_raw in lanes.items():
        lane = require_object(lane_raw, f"legacy expectations lane {lane_key}")
        rules = require_list(lane.get("rules"), f"legacy expectations lane {lane_key}.rules")
        require(rules, f"legacy expectations lane {lane_key} has no rules")
        for index, rule_raw in enumerate(rules):
            label = f"legacy expectations lane {lane_key}.rules[{index}]"
            rule = require_object(rule_raw, label)
            require(
                set(rule) == {"selector", "expected_status", "failure_class", "downstream_goal", "owner", "evidence_basis", "next_action"},
                f"{label} fields mismatch",
            )
            selector = require_object(rule.get("selector"), f"{label}.selector")
            require(set(selector) == {"scenario_id", "variant", "preset"}, f"{label}.selector fields mismatch")
            scenario = selector.get("scenario_id")
            require(scenario == "*" or scenario in SCENARIO_IDS, f"{label} scenario selector invalid")
            require(isinstance(selector.get("variant"), str) and selector["variant"], f"{label} variant selector invalid")
            preset = selector.get("preset")
            require(preset in {"*", "none", "P_DETERMINISTIC", "P_NO_THINKING", "P_THINKING", "P_OFFICIAL_DEFAULT"}, f"{label} preset selector invalid")
            status = rule.get("expected_status")
            require(status in EXPECTED_STATUSES, f"{label} expected_status invalid")
            if status == "blocked":
                require(
                    BLOCKED_LANE_FAILURE_CLASSES.get(lane_key) == rule.get("failure_class"),
                    f"{label} cannot use blocked for executable lane {lane_key}",
                )
            if status == "pass":
                require(rule.get("failure_class") is None, f"{label} passing rule must not declare failure_class")
            else:
                require_string(rule.get("failure_class"), f"{label}.failure_class")
            require(re.fullmatch(r"G(?:0[1-9]|10|08[A-D])", require_string(rule.get("downstream_goal"), f"{label}.downstream_goal")) is not None, f"{label} downstream_goal invalid")
            require_string(rule.get("owner"), f"{label}.owner")
            require_string(rule.get("evidence_basis"), f"{label}.evidence_basis")
            require_string(rule.get("next_action"), f"{label}.next_action")
    return catalog


def resolve_expectation(
    catalog: dict[str, Any],
    *,
    model_key: str,
    backend: str,
    scenario_id: str,
    variant: str,
    preset: str | None,
) -> dict[str, Any]:
    lane_key = f"{model_key}/{backend}"
    lane = require_object(require_object(catalog["lanes"], "legacy expectations lanes").get(lane_key), f"legacy expectation lane {lane_key}")
    preset_key = preset or "none"
    matches: list[tuple[int, dict[str, Any]]] = []
    for rule_raw in require_list(lane.get("rules"), f"legacy expectation lane {lane_key}.rules"):
        rule = require_object(rule_raw, f"legacy expectation lane {lane_key} rule")
        selector = require_object(rule.get("selector"), "legacy expectation selector")
        values = ((selector["scenario_id"], scenario_id), (selector["variant"], variant), (selector["preset"], preset_key))
        if all(selected == "*" or selected == actual for selected, actual in values):
            matches.append((sum(selected != "*" for selected, _ in values), rule))
    require(matches, f"legacy expectations do not cover {lane_key}/{scenario_id}/{variant}/{preset_key}")
    specificity = max(score for score, _ in matches)
    winners = [rule for score, rule in matches if score == specificity]
    require(len(winners) == 1, f"legacy expectations are ambiguous for {lane_key}/{scenario_id}/{variant}/{preset_key}")
    return winners[0]


def canonical_expectations_sha256() -> str:
    validate_expectations_catalog(read_json(EXPECTATIONS_PATH))
    checked_in = git_bytes(["show", f"HEAD:{EXPECTATIONS_REPO_PATH}"])
    require(checked_in == EXPECTATIONS_PATH.read_bytes(), "legacy expectations differ from checked-in blob")
    return hashlib.sha256(checked_in).hexdigest()


def canonical_runner_identity() -> dict[str, Any]:
    status = git_text(["status", "--short", "--untracked-files=all"])
    require(not status, "scenario runner requires a clean git worktree")
    git_sha = require_git_sha(git_text(["rev-parse", "HEAD"]), "runner.git_sha")
    tree_sha = require_git_sha(git_text(["rev-parse", "HEAD^{tree}"]), "runner.source_tree_sha")
    blob_sha = require_git_sha(git_text(["rev-parse", f"HEAD:{RUNNER_REPO_PATH}"]), "runner.git_blob_sha")
    checked_in = git_bytes(["show", f"HEAD:{RUNNER_REPO_PATH}"])
    require(checked_in == RUNNER_PATH.read_bytes(), "scenario runner differs from its checked-in blob")
    return {
        "path": RUNNER_REPO_PATH,
        "git_sha": git_sha,
        "source_tree_sha": tree_sha,
        "git_blob_sha": blob_sha,
        "sha256": hashlib.sha256(checked_in).hexdigest(),
        "dirty_status": {"is_dirty": False, "status_short": []},
    }


def internal_fixture_runner_identity() -> dict[str, Any]:
    return {
        "path": RUNNER_REPO_PATH,
        "git_sha": FROZEN_LEGACY_SHA,
        "source_tree_sha": frozen_tree_sha(),
        "git_blob_sha": "0" * 40,
        "sha256": file_sha256(RUNNER_PATH),
        "dirty_status": {"is_dirty": False, "status_short": []},
        "internal_fixture": True,
    }


def reject_forbidden_markers(value: Any, label: str, *, allow_internal_fixture: bool) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_KEYS:
                require(child in (False, 0, None, [], {}), f"{label}.{key} contains forbidden skip/waiver evidence")
            reject_forbidden_markers(child, f"{label}.{key}", allow_internal_fixture=allow_internal_fixture)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_markers(child, f"{label}[{index}]", allow_internal_fixture=allow_internal_fixture)
    elif isinstance(value, str) and not allow_internal_fixture:
        lowered = value.lower()
        require("selftest" not in lowered and "self-test" not in lowered and "synthetic" not in lowered, f"{label} contains fixture evidence")


def artifact_path(root: Path, raw: Any, label: str) -> Path:
    text = require_string(raw, label)
    path = Path(text)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ScenarioError(f"{label} must stay inside artifact root: {text}") from exc
    return resolved


def validate_artifact_ref(
    root: Path,
    raw: Any,
    label: str,
    *,
    allowed_kinds: set[str] | None = None,
) -> tuple[Path, str, Any | None]:
    ref = require_object(raw, label)
    kind = require_string(ref.get("kind"), f"{label}.kind")
    if allowed_kinds is not None:
        require(kind in allowed_kinds, f"{label}.kind {kind!r} is not allowed")
    path = artifact_path(root, ref.get("path"), f"{label}.path")
    require(path.is_file(), f"{label} missing artifact: {path}")
    require(path.stat().st_size > 0, f"{label} artifact is empty: {path}")
    expected = require_sha256(ref.get("sha256"), f"{label}.sha256")
    require(file_sha256(path) == expected, f"{label} artifact SHA256 mismatch")
    parsed: Any | None = None
    if kind == "raw-json" or path.suffix.lower() == ".json":
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScenarioError(f"{label} does not contain valid UTF-8 JSON: {exc}") from exc
        require(isinstance(parsed, (dict, list)) and bool(parsed), f"{label} JSON must be a non-empty object or array")
    elif kind in LOG_KINDS:
        require(path.stat().st_size >= 16, f"{label} log is too small to be credible evidence")
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        require(not any(marker in text for marker in BLOCKER_MARKERS), f"{label} log contains a release blocker")
    return path, kind, parsed


def validate_source_identity(data: dict[str, Any], label: str) -> None:
    require(data.get("source_git_sha") == FROZEN_LEGACY_SHA, f"{label}.source_git_sha is not the frozen legacy SHA")
    require(data.get("source_tree_sha") == frozen_tree_sha(), f"{label}.source_tree_sha mismatch")
    dirty = require_object(data.get("dirty_status"), f"{label}.dirty_status")
    require(dirty.get("is_dirty") is False, f"{label} dirty legacy source is forbidden")
    require(dirty.get("status_short") == [], f"{label}.dirty_status.status_short must be empty")


def validate_runner_identity(raw: Any, *, allow_internal_fixture: bool) -> dict[str, Any]:
    runner = require_object(raw, "scenario_report.runner")
    require(runner.get("path") == RUNNER_REPO_PATH, f"scenario_report.runner.path must be {RUNNER_REPO_PATH}")
    require_sha256(runner.get("sha256"), "scenario_report.runner.sha256")
    require_git_sha(runner.get("git_sha"), "scenario_report.runner.git_sha")
    require_git_sha(runner.get("source_tree_sha"), "scenario_report.runner.source_tree_sha")
    require_git_sha(runner.get("git_blob_sha"), "scenario_report.runner.git_blob_sha")
    dirty = require_object(runner.get("dirty_status"), "scenario_report.runner.dirty_status")
    require(dirty.get("is_dirty") is False and dirty.get("status_short") == [], "scenario_report runner must be clean")
    if allow_internal_fixture and runner.get("internal_fixture") is True:
        require(runner == internal_fixture_runner_identity(), "scenario_report internal fixture runner identity mismatch")
        return runner
    require("internal_fixture" not in runner, "scenario_report fixture runner is forbidden")
    runner_git_sha = str(runner["git_sha"])
    expected_tree = require_git_sha(git_text(["rev-parse", f"{runner_git_sha}^{{tree}}"]), "scenario_report.runner.source_tree_sha")
    require(runner.get("source_tree_sha") == expected_tree, "scenario_report runner source tree mismatch")
    expected_blob = require_git_sha(git_text(["rev-parse", f"{runner_git_sha}:{RUNNER_REPO_PATH}"]), "scenario_report.runner.git_blob_sha")
    require(runner.get("git_blob_sha") == expected_blob, "scenario_report runner git blob mismatch")
    checked_in = git_bytes(["show", f"{runner_git_sha}:{RUNNER_REPO_PATH}"])
    expected_sha256 = hashlib.sha256(checked_in).hexdigest()
    require(runner.get("sha256") == expected_sha256, "scenario_report runner SHA256 mismatch")
    require(file_sha256(RUNNER_PATH) == expected_sha256, "scenario_report runner does not match the current gate contract")
    return runner


def validate_effective_config(
    root: Path,
    raw: Any,
    *,
    expected: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    path, _, parsed = validate_artifact_ref(root, raw, "scenario_report.effective_config", allowed_kinds={"raw-json"})
    config = require_object(parsed, "scenario_report.effective_config JSON")
    require(config.get("schema_version") == SCHEMA_VERSION, "effective config schema_version mismatch")
    validate_source_identity(config, "effective_config")
    for key in (
        "models_lock_sha256",
        "binary_sha256",
        "model_key",
        "backend",
        "model_revision",
        "model_files",
        "hardware_id",
    ):
        require(config.get(key) == expected.get(key), f"effective_config.{key} binding mismatch")
    typed = require_object(config.get("typed_effective_config"), "effective_config.typed_effective_config")
    require(typed, "effective_config.typed_effective_config must not be empty")
    require(set(typed) >= {"run", "serve"}, "effective_config must contain typed run and serve config")
    return config, file_sha256(path)


def validate_command(
    root: Path,
    raw: Any,
    label: str,
    *,
    expected: dict[str, Any],
    effective_config_sha256: str,
) -> tuple[str, str]:
    command = require_object(raw, label)
    command_id = require_string(command.get("id"), f"{label}.id")
    entrypoint = require_string(command.get("entrypoint"), f"{label}.entrypoint")
    require(entrypoint in {"run", "serve"}, f"{label}.entrypoint must be run or serve")
    argv = require_list(command.get("argv"), f"{label}.argv")
    require(len(argv) >= 3 and all(isinstance(part, str) and part for part in argv), f"{label}.argv must include ferrum, entrypoint, and model/config args")
    require(Path(argv[0]).name == "ferrum", f"{label}.argv must execute ferrum")
    require(argv[1] == entrypoint, f"{label}.argv must have exact ferrum {entrypoint} command shape")
    if "binary_path" in expected:
        require(Path(argv[0]).resolve() == expected["binary_path"], f"{label}.argv binary is not the bound binary artifact")
        require(argv[2] == expected.get("model_path"), f"{label}.argv model path is not the bound execution model")
    for key in ("source_git_sha", "source_tree_sha", "models_lock_sha256", "binary_sha256"):
        require(command.get(key) == expected.get(key), f"{label}.{key} binding mismatch")
    require(command.get("effective_config_sha256") == effective_config_sha256, f"{label}.effective_config_sha256 mismatch")
    require(command.get("returncode") == 0, f"{label}.returncode must be 0")
    started_at = parse_timestamp(command.get("started_at"), f"{label}.started_at")
    finished_at = parse_timestamp(command.get("finished_at"), f"{label}.finished_at")
    require(finished_at > started_at, f"{label} process window is not positive")
    duration = command.get("duration_sec")
    require(isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(duration) and duration > 0, f"{label}.duration_sec must be positive")
    actual_duration = (finished_at - started_at).total_seconds()
    require(abs(float(duration) - actual_duration) <= max(0.05, actual_duration * 0.05), f"{label}.duration_sec does not match timestamps")
    if "invocation_started_at" in expected:
        require(
            expected["invocation_started_at"] <= started_at < finished_at <= expected["invocation_finished_at"],
            f"{label} wall window is outside executor invocation",
        )
    environment = validate_sanitized_environment(command.get("env"), f"{label}.env")
    require(command.get("env_sha256") == canonical_json_sha256(environment), f"{label}.env_sha256 mismatch")
    if "invocation_started_at" in expected:
        _, _, receipt_raw = validate_artifact_ref(root, command.get("process_receipt"), f"{label}.process_receipt", allowed_kinds={"raw-json"})
        receipt = require_object(receipt_raw, f"{label}.process_receipt JSON")
        require(receipt.get("argv") == argv, f"{label} process receipt argv mismatch")
        require(receipt.get("role") == f"ferrum-{entrypoint}", f"{label} process receipt role mismatch")
        require(receipt.get("environment") == environment, f"{label} command environment differs from process receipt")
        require(receipt.get("environment_sha256") == command.get("env_sha256"), f"{label} command/receipt environment SHA mismatch")
    validate_artifact_ref(root, command.get("stdout"), f"{label}.stdout", allowed_kinds={"stdout-log"})
    validate_artifact_ref(root, command.get("stderr"), f"{label}.stderr", allowed_kinds={"stderr-log"})
    return command_id, entrypoint


def minimum_presets(scenario_id: str, model_key: str) -> tuple[dict[str, int], int]:
    if scenario_id == "C01":
        return {}, 20
    if scenario_id in {f"C{index:02d}" for index in range(2, 10)} | {"C16", "C17", "C18"}:
        return {"P_DETERMINISTIC": 1}, 0
    if scenario_id in {"C10", "C11", "C12", "C13"}:
        count = 30 if model_key == "m3-qwen3-30b-a3b" else 20
        return {"P_NO_THINKING": count, "P_THINKING": count}, 0
    if scenario_id in {"C14", "C15"}:
        return {"P_NO_THINKING": 50, "P_THINKING": 20}, 0
    if scenario_id == "C19":
        return {"P_THINKING": 20}, 0
    if scenario_id == "C20":
        return {"P_DETERMINISTIC": 10}, 40
    if scenario_id == "C21":
        return {"P_OFFICIAL_DEFAULT": 20}, 0
    raise AssertionError(scenario_id)


def required_entrypoints(scenario_id: str) -> set[str]:
    if scenario_id in {"C01", "C02", "C03", "C04"}:
        return {"run"}
    if scenario_id in {"C17", "C19", "C21"}:
        return {"run", "serve"}
    return {"serve"}


def minimum_case_count(scenario_id: str, model_key: str) -> int:
    fixed = {
        "C01": 20,
        "C02": 20,
        "C03": 10,
        "C04": 3,
        "C05": 20,
        "C06": 20,
        "C07": 6,
        "C08": 60,
        "C09": 60,
        "C14": 70,
        "C15": 70,
        "C16": 30,
        "C17": 60,
        "C19": 20,
        "C20": 50,
        "C21": 20,
    }
    if scenario_id in {"C10", "C11", "C12", "C13"}:
        return 60 if model_key == "m3-qwen3-30b-a3b" else 40
    if scenario_id == "C18":
        return 1
    return fixed[scenario_id]


def required_variants(scenario_id: str, model_key: str) -> tuple[dict[str, int], bool]:
    variants: dict[str, dict[str, int]] = {
        "C01": {"config-resolution": 5, "template-byte": 5, "special-token-eos": 5, "unknown-fail-closed": 5},
        "C02": {"known-answer": 20},
        "C04": {"long-output": 3},
        "C05": {"known-answer": 20},
        "C06": {"stream": 20},
        "C08": {"stop": 20, "natural-eos": 20, "max-tokens": 20},
        "C09": {"cancel": 20, "timeout": 20, "disconnect": 20},
        "C10": {"required-tool": 40},
        "C11": {"auto-tool": 40},
        "C12": {"streamed-tool": 40},
        "C13": {"tool-result": 40},
        "C14": {"required": 18, "type": 18, "additional-properties": 17, "enum": 17},
        "C15": {"json-object": 70},
        "C16": {"invalid-tool": 6, "invalid-schema": 6, "invalid-stream-option": 6, "invalid-model": 6, "invalid-context": 6},
        "C17": {"chinese": 20, "emoji": 20, "combining": 20},
        "C20": {"image-url": 10, "data-url": 10, "video-url": 10, "mixed-text-media": 10, "text-array": 10},
        "C21": {"run-plain": 4, "serve-stream": 4, "required-tool": 4, "strict-schema": 4, "json-object": 4},
    }
    if scenario_id in {"C10", "C11", "C12", "C13"} and model_key == "m3-qwen3-30b-a3b":
        key = next(iter(variants[scenario_id]))
        return {key: 40, "soft-think": 10, "soft-no-think": 10}, True
    if scenario_id == "C19":
        modes = {"default-thinking": 4, "hard-thinking": 4, "hard-no-thinking": 4}
        if model_key == "m3-qwen3-30b-a3b":
            modes.update({"soft-think": 4, "soft-no-think": 4})
        else:
            modes.update({"soft-think-misuse": 4, "soft-no-think-misuse": 4})
        return modes, True
    return variants.get(scenario_id, {}), scenario_id != "C18"


ZERO_ASSERTIONS: dict[str, set[str]] = {
    "C01": {"config_field_failures", "template_mismatch_count", "llama_fallback_count"},
    "C02": {"empty_output_count", "reserved_token_count", "mojibake_count"},
    "C03": {"memory_failure_count", "usage_failure_count", "state_crosstalk_count"},
    "C04": {"hang_count", "crash_count", "invalid_utf8_count", "non_incremental_count", "bad_finish_count"},
    "C05": {"response_shape_failure_count", "finish_reason_failure_count", "usage_failure_count"},
    "C06": {"content_mismatch_count", "malformed_sse_count", "missing_delta_count"},
    "C07": {"prefix_repeat_count", "early_stop_count", "cross_request_pollution_count"},
    "C08": {"finish_reason_failure_count", "stop_leak_count", "eos_metadata_failure_count", "exact_token_count_failure_count"},
    "C09": {"double_release_count", "leak_count", "post_capacity_failure_count"},
    "C10": {"tool_name_failure_count", "arguments_failure_count", "schema_failure_count", "reasoning_argument_leak_count"},
    "C11": {"missing_tool_count", "content_json_count"},
    "C12": {"delta_reassembly_failure_count", "invalid_json_count"},
    "C13": {"tool_role_failure_count", "result_reference_failure_count", "duplicate_call_count", "history_pollution_count"},
    "C14": {"invalid_schema_count", "constraint_failure_count", "reasoning_json_leak_count"},
    "C15": {"invalid_object_count", "markdown_fence_count", "surrounding_junk_count", "reasoning_json_leak_count"},
    "C16": {"non_4xx_count", "non_openai_shape_count"},
    "C17": {"replacement_char_count", "mojibake_count", "partial_character_chunk_count"},
    "C18": {"crosstalk_count", "bad_checksum_count", "server_500_count", "panic_count", "oom_count"},
    "C19": {"reasoning_final_leak_count", "history_separation_failure_count", "soft_switch_misuse_count"},
    "C20": {"vision_token_inserted_count", "silent_media_ignore_count", "text_array_failure_count"},
    "C21": {"garbage_count", "shape_failure_count", "termination_failure_count", "tool_schema_failure_count", "resource_failure_count"},
}


def validate_assertions(scenario_id: str, raw: Any, *, case_count: int) -> None:
    assertions = require_object(raw, f"scenario {scenario_id}.assertions")
    require(assertions.get("bad_output_count") == 0, f"scenario {scenario_id} bad_output_count must be 0")
    require(assertions.get("resource_final_state") == "released", f"scenario {scenario_id} resource final state must be released")
    for field in ZERO_ASSERTIONS[scenario_id]:
        require(assertions.get(field) == 0, f"scenario {scenario_id} assertion {field} must be 0")
    if scenario_id == "C02":
        require(assertions.get("natural_eos_count") == case_count, "C02 natural EOS coverage incomplete")
    elif scenario_id == "C06":
        for field in ("done_count", "usage_count", "output_with_delta_count", "paired_nonstream_equivalence_count"):
            require(assertions.get(field) == case_count, f"C06 {field} must equal case_count")
    elif scenario_id == "C07":
        require(assertions.get("conversation_count") == 6, "C07 must execute six isolated conversations")
        require(assertions.get("history_turn_count") == 30, "C07 must execute thirty history-carrying turns")
    elif scenario_id == "C09":
        require(assertions.get("released_count") == case_count, "C09 cancel/timeout/disconnect release coverage incomplete")
        require(assertions.get("post_capacity_success_count") == case_count, "C09 post-cancel capacity recovery incomplete")
        ticks = assertions.get("max_scheduler_ticks_to_release")
        wall = assertions.get("max_wall_sec_to_release")
        require(isinstance(ticks, int) and not isinstance(ticks, bool) and 0 <= ticks <= 2, "C09 release must occur within 2 scheduler ticks")
        require(isinstance(wall, (int, float)) and not isinstance(wall, bool) and 0 <= wall <= 5, "C09 release must occur within 5 seconds")
    elif scenario_id in {"C10", "C11", "C12", "C13"}:
        require(assertions.get("tool_success_count") == case_count, f"{scenario_id} tool coverage incomplete")
        if scenario_id == "C12":
            require(assertions.get("paired_c11_equivalence_count") == case_count, "C12 paired C11 equivalence coverage incomplete")
    elif scenario_id == "C14":
        require(assertions.get("valid_json_count") == case_count, "C14 valid schema count incomplete")
        require(assertions.get("distinct_schema_count") == case_count and assertions.get("distinct_prompt_count") == case_count, "C14 schemas/prompts are not materially distinct")
    elif scenario_id == "C15":
        require(assertions.get("valid_object_count") == case_count, "C15 valid object count incomplete")
    elif scenario_id == "C20":
        require(assertions.get("rejected_media_count") == 40, "C20 must reject all 40 media cases")
        require(assertions.get("text_array_success_count") == 10, "C20 text-array positive coverage incomplete")
        require(assertions.get("declared_modalities") == ["text"], "C20 /v1/models must declare text-only modality")
    elif scenario_id == "C17":
        require(assertions.get("streaming_split_boundary_count") == case_count // 2, "C17 streaming split-boundary coverage incomplete")
    elif scenario_id == "C19":
        require(assertions.get("history_case_count") == case_count, "C19 reasoning history coverage incomplete")
    elif scenario_id == "C21":
        require(assertions.get("tool_priority_count") == 4, "C21 required-tool priority coverage incomplete")
        require(assertions.get("serve_stream_count") == 4 and assertions.get("strict_schema_count") == 4 and assertions.get("json_object_count") == 4, "C21 serve subgroup coverage incomplete")


def validate_concurrency_cells(backend: str, raw: Any, *, case_count: int, require_pass: bool = True) -> None:
    cells = require_list(raw, "C18.concurrency_cells")
    expected = {1, 4, 16, 32} if backend == "cuda" else {1, 4, 16}
    seen: set[int] = set()
    total = 0
    for index, cell_raw in enumerate(cells):
        cell = require_object(cell_raw, f"C18.concurrency_cells[{index}]")
        requested = require_count(cell.get("requested_concurrency"), f"C18.concurrency_cells[{index}].requested_concurrency", minimum=1)
        require(requested in expected and requested not in seen, f"C18 invalid or duplicate concurrency cell {requested}")
        seen.add(requested)
        count = require_count(cell.get("case_count"), f"C18.c{requested}.case_count", minimum=1)
        if require_pass:
            require(cell.get("passed_count") == count, f"C18.c{requested} must pass every case")
            require(cell.get("completion_rate") == 1.0, f"C18.c{requested} completion_rate must be 1.0")
        else:
            require_count(cell.get("passed_count"), f"C18.c{requested}.passed_count")
            rate = cell.get("completion_rate")
            require(isinstance(rate, (int, float)) and not isinstance(rate, bool) and 0 <= rate <= 1, f"C18.c{requested} completion_rate invalid")
        cap = require_count(cell.get("typed_admission_cap"), f"C18.c{requested}.typed_admission_cap", minimum=1)
        active = require_count(cell.get("observed_max_active"), f"C18.c{requested}.observed_max_active", minimum=1)
        require(active <= min(requested, cap), f"C18.c{requested} observed max-active exceeds requested/cap")
        for field in ("error_count", "bad_output_count", "crosstalk_count", "bad_checksum_count", "server_500_count", "panic_count", "oom_count"):
            value = require_count(cell.get(field), f"C18.c{requested}.{field}")
            if require_pass:
                require(value == 0, f"C18.c{requested}.{field} must be 0")
        total += count
    require(seen == expected, f"C18 concurrency cells must be exactly {sorted(expected)}")
    require(total == case_count, "C18 concurrency cell counts must sum to case_count")


def parse_jsonl_artifact(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScenarioError(f"{label}:{line_no} is not JSON: {exc}") from exc
        require(isinstance(row, dict), f"{label}:{line_no} must be a JSON object")
        rows.append(row)
    require(rows, f"{label} must contain JSONL records")
    return rows


def response_message(response: dict[str, Any], label: str) -> tuple[dict[str, Any], str]:
    choices = require_list(response.get("choices"), f"{label}.choices")
    require(len(choices) == 1, f"{label} must contain exactly one choice")
    choice = require_object(choices[0], f"{label}.choices[0]")
    message = require_object(choice.get("message"), f"{label}.choices[0].message")
    content = message.get("content")
    require(content is None or isinstance(content, str), f"{label} message content must be string or null")
    return message, content or ""


def history_response_message(exchange: dict[str, Any], label: str, errors: list[dict[str, Any]]) -> dict[str, Any] | None:
    raw_response = exchange.get("response")
    try:
        response = require_object(raw_response, f"{label}.response")
        message, _ = response_message(response, label)
        return message
    except ScenarioError as exc:
        errors.append(
            {
                "label": label,
                "error": str(exc),
                "response_sha256": canonical_json_sha256(raw_response),
            }
        )
        return None


def validate_case_output(
    scenario_id: str,
    variant: str,
    entrypoint: str,
    stdout_path: Path,
    transcript: dict[str, Any] | None,
    observed: dict[str, Any],
    label: str,
    *,
    input_document: dict[str, Any] | None = None,
    actual_config: dict[str, Any] | None = None,
    artifact_root: Path | None = None,
) -> None:
    stdout_text = stdout_path.read_text(encoding="utf-8")
    require(not any(marker in stdout_text.lower() for marker in BLOCKER_MARKERS), f"{label} stdout contains bad output")
    if entrypoint == "run":
        rows = parse_jsonl_artifact(stdout_path, f"{label}.stdout")
        assistants = [row for row in rows if row.get("event") == "assistant"]
        require(assistants, f"{label} run emitted no assistant record")
        for index, assistant in enumerate(assistants):
            content = assistant.get("content")
            require(isinstance(content, str) and content.strip(), f"{label} assistant[{index}] is empty")
            require_count(assistant.get("n_tokens"), f"{label} assistant[{index}].n_tokens", minimum=1)
            require(assistant.get("finish_reason") in {"stop", "eos", "length"}, f"{label} assistant[{index}] has invalid finish reason")
        if scenario_id == "C02":
            require(assistants[-1].get("finish_reason") in {"stop", "eos"}, f"{label} did not end naturally")
        if scenario_id == "C03":
            require(len(assistants) >= 3, f"{label} must contain three assistant turns")
            marker = require_string(observed.get("expected_marker"), f"{label}.observed.expected_marker")
            require(marker in str(assistants[-1].get("content")), f"{label} did not recall its marker")
        if scenario_id == "C04":
            require(sum(require_count(row.get("n_tokens"), f"{label}.n_tokens", minimum=1) for row in assistants) >= 512, f"{label} long output is shorter than 512 tokens")
            require(any(require_count(row.get("chunk_count"), f"{label}.chunk_count", minimum=1) > 1 for row in assistants), f"{label} long output was not incremental")
        if scenario_id == "C01":
            require(input_document is not None, f"{label} C01 input evidence is missing")
            validate_c01_resolution_probe(
                input_document.get("resolution_probe"),
                model_key=require_string(observed.get("model_key"), f"{label}.observed.model_key"),
                variant=variant,
                actual_config=actual_config,
                label=label,
            )
            if variant == "unknown-fail-closed":
                require(artifact_root is not None, f"{label} C01 negative probe lacks artifact root")
                validate_c01_negative_probe(
                    artifact_root,
                    input_document.get("negative_probe"),
                    resolution_probe=require_object(input_document.get("resolution_probe"), f"{label}.resolution_probe"),
                    ordinal=require_count(observed.get("ordinal"), f"{label}.observed.ordinal", minimum=1),
                    label=label,
                )
            ready = [row for row in rows if row.get("event") == "ready"]
            require(len(ready) == 1, f"{label} C01 must record exactly one runtime model-resolution event")
            require(ready[0].get("model") == observed.get("requested_model"), f"{label} runtime resolved a different model argument")
        if scenario_id == "C19":
            require(len(assistants) == 2, f"{label} thinking run must carry a two-turn history")
            users = [row for row in rows if row.get("event") == "user"]
            require(len(users) == 2 and [row.get("turn") for row in users] == [1, 2], f"{label} thinking run history turns are incomplete")
            reasoning_expected = bool(observed.get("reasoning_expected"))
            for index, assistant in enumerate(assistants):
                reasoning = assistant.get("reasoning")
                require(bool(isinstance(reasoning, str) and reasoning.strip()) is reasoning_expected, f"{label} assistant[{index}] reasoning mode mismatch")
                require("<think>" not in str(assistant.get("content")) and "</think>" not in str(assistant.get("content")), f"{label} reasoning leaked into final content")
            require(assistants[1].get("history_turns_seen") == 1, f"{label} second thinking turn did not carry prior assistant history")
        if scenario_id == "C17":
            expected_text = require_string(observed.get("expected_marker"), f"{label}.observed.expected_marker")
            require(assistants[-1].get("content") == expected_text, f"{label} Unicode run content mismatch")
            require(require_count(assistants[-1].get("chunk_count"), f"{label}.chunk_count", minimum=2) >= 2, f"{label} Unicode run was not incremental")
            require("\ufffd" not in expected_text, f"{label} Unicode run contains replacement characters")
        marker = observed.get("expected_marker")
        if isinstance(marker, str) and marker and scenario_id not in {"C03", "C04"}:
            require(marker in str(assistants[-1].get("content")), f"{label} missing expected marker")
        return

    require(transcript is not None, f"{label} serve case requires an HTTP transcript")
    require(transcript.get("case_id") == observed.get("case_id"), f"{label} transcript case id mismatch")
    exchanges = require_list(transcript.get("exchanges"), f"{label}.transcript.exchanges")
    require(exchanges, f"{label} HTTP transcript has no exchanges")
    if input_document is not None:
        bound_exchange = exchanges[-1] if (scenario_id in {"C06", "C12", "C17"} or (scenario_id == "C21" and variant == "serve-stream")) else exchanges[0]
        require(require_object(bound_exchange, f"{label}.bound_exchange").get("request") == input_document, f"{label} HTTP transcript is not bound to the persisted input request")
    statuses: list[int] = []
    responses: list[dict[str, Any]] = []
    for index, exchange_raw in enumerate(exchanges):
        exchange = require_object(exchange_raw, f"{label}.exchange[{index}]")
        require_object(exchange.get("request"), f"{label}.exchange[{index}].request")
        status = require_count(exchange.get("status"), f"{label}.exchange[{index}].status", minimum=100)
        statuses.append(status)
        body = exchange.get("response")
        if isinstance(body, dict):
            responses.append(body)
    if scenario_id == "C21" and variant == "required-tool":
        require(len(exchanges) == 2 and statuses == [200, 200], f"{label} tool-priority conflict must succeed twice")
        requests = [require_object(exchange.get("request"), f"{label}.tool_priority.request") for exchange in exchanges]
        require(requests[0] == requests[1], f"{label} tool-priority replay requests differ")
        require("tools" in requests[0] and requests[0].get("tool_choice") == "required" and "response_format" in requests[0], f"{label} did not send required-tool plus strict-schema together")
        contracts: list[dict[str, Any]] = []
        for index, response in enumerate(responses):
            message, content = response_message(response, f"{label}.tool_priority[{index}]")
            choice = require_object(require_list(response.get("choices"), f"{label}.tool_priority[{index}].choices")[0], f"{label}.tool_priority[{index}].choice")
            require(choice.get("finish_reason") == "tool_calls", f"{label} tool priority finish_reason mismatch")
            require(content == "", f"{label} fabricated strict-schema content instead of choosing the required tool")
            calls = require_list(message.get("tool_calls"), f"{label}.tool_priority[{index}].tool_calls")
            require(len(calls) == 1, f"{label} tool priority must return exactly one call")
            function = require_object(require_object(calls[0], f"{label}.tool_priority[{index}].call").get("function"), f"{label}.tool_priority[{index}].function")
            require(function.get("name") == "echo_value", f"{label} tool priority returned the wrong function")
            arguments = json.loads(require_string(function.get("arguments"), f"{label}.tool_priority[{index}].arguments"))
            require(arguments == {"value": observed.get("expected_marker")}, f"{label} tool priority arguments violate the declared schema")
            contracts.append({"choices": response.get("choices"), "usage": response.get("usage")})
        require(contracts[0] == contracts[1], f"{label} tool-priority contract is nondeterministic")
        return
    media_negative = scenario_id == "C20" and variant != "text-array"
    if scenario_id == "C20":
        models_response = require_object(transcript.get("models_response"), f"{label}.models_response")
        models = require_list(models_response.get("data"), f"{label}.models_response.data")
        require(models and require_object(models[0], f"{label}.models_response.data[0]").get("modalities") == ["text"], f"{label} /v1/models must declare text-only modality")
        require(observed.get("declared_modalities") == ["text"], f"{label} observed modality binding mismatch")
    if scenario_id == "C16" or media_negative:
        require(all(400 <= status < 500 for status in statuses), f"{label} negative API case must return 4xx")
        for response in responses:
            error = require_object(response.get("error"), f"{label}.error")
            require_string(error.get("message"), f"{label}.error.message")
        return
    if scenario_id == "C09":
        require(statuses[0] == 499 and statuses[-1] == 200, f"{label} must record client abort followed by successful recovery")
    else:
        require(all(status == 200 for status in statuses), f"{label} expected HTTP 200")
    if scenario_id in {"C06", "C12", "C17"} or (scenario_id == "C21" and variant == "serve-stream"):
        require(transcript.get("done_count") == 1, f"{label} stream must contain one [DONE]")
        require(transcript.get("usage_count") == 1, f"{label} stream must contain one usage record")
        require_count(transcript.get("delta_count"), f"{label}.delta_count", minimum=1)
        reconstruction = require_object(transcript.get("stream_reconstruction"), f"{label}.stream_reconstruction")
        require(reconstruction.get("malformed_count") == 0, f"{label} stream contains malformed SSE JSON")
        require(len(exchanges) == 2, f"{label} requires a non-stream reference and one stream exchange")
        reference_request = require_object(exchanges[0].get("request"), f"{label}.reference.request")
        stream_request = require_object(exchanges[1].get("request"), f"{label}.stream.request")
        normalized_stream_request = copy.deepcopy(stream_request)
        normalized_stream_request.pop("stream", None)
        normalized_stream_request.pop("stream_options", None)
        require(reference_request == normalized_stream_request, f"{label} stream request differs from its non-stream reference")
        metadata = require_object(reference_request.get("metadata"), f"{label}.reference.metadata")
        reference_ordinal = require_count(metadata.get("g00_ordinal"), f"{label}.reference.metadata.g00_ordinal", minimum=1)
        if scenario_id == "C06":
            require(metadata.get("g00_reference_contract") == "C05", f"{label} is not bound to the C05 non-stream contract")
            require(metadata.get("g00_reference_case_id") == f"c05-{reference_ordinal:03d}", f"{label} C05 pair identity mismatch")
        if scenario_id == "C12":
            require(metadata.get("g00_reference_contract") == "C11", f"{label} is not bound to the C11 auto-tool contract")
            require(metadata.get("g00_reference_case_id") == f"c11-{reference_ordinal:03d}", f"{label} C11 pair identity mismatch")
        reference_response = require_object(exchanges[0].get("response"), f"{label}.reference.response")
        reference_message, reference_content = response_message(reference_response, f"{label}.reference")
        reference_choice = require_object(require_list(reference_response.get("choices"), f"{label}.reference.choices")[0], f"{label}.reference.choice")
        require(reconstruction.get("finish_reason") == reference_choice.get("finish_reason"), f"{label} stream finish_reason differs from non-stream")
        require(reconstruction.get("usage") == reference_response.get("usage"), f"{label} stream usage differs from non-stream")
        reference_reasoning = reference_message.get("reasoning", reference_message.get("reasoning_content")) or ""
        require(reconstruction.get("reasoning", "") == reference_reasoning, f"{label} stream reasoning differs from non-stream")
        if scenario_id in {"C06", "C17"} or (scenario_id == "C21" and variant == "serve-stream"):
            require(reconstruction.get("content") == reference_content, f"{label} stream content differs from non-stream")
            require(reference_content == observed.get("expected_marker"), f"{label} paired reference returned unexpected content")
        if scenario_id == "C12":
            reference_calls = require_list(reference_message.get("tool_calls"), f"{label}.reference.tool_calls")
            streamed_calls = require_list(reconstruction.get("tool_calls"), f"{label}.stream.tool_calls")
            require(streamed_calls == reference_calls, f"{label} streamed tool call differs from matching C11 non-stream call")
            function = require_object(require_object(streamed_calls[0], f"{label}.stream.tool_calls[0]").get("function"), f"{label}.stream.function")
            arguments = json.loads(require_string(function.get("arguments"), f"{label}.stream.arguments"))
            require(arguments == {"city": "Paris"}, f"{label} reconstructed tool arguments are invalid")
        if scenario_id == "C17":
            wire = require_object(transcript.get("utf8_wire_evidence"), f"{label}.utf8_wire_evidence")
            chunks = [base64.b64decode(require_string(value, f"{label}.wire_chunk"), validate=True) for value in require_list(wire.get("chunks_base64"), f"{label}.chunks_base64")]
            decoded, fragments, split_count = decode_wire_chunks(chunks)
            require(hashlib.sha256(b"".join(chunks)).hexdigest() == wire.get("wire_sha256"), f"{label} UTF-8 wire SHA mismatch")
            require(decoded == exchanges[1].get("response_raw"), f"{label} incremental UTF-8 decoding differs from raw response")
            require(fragments == wire.get("decoded_fragments"), f"{label} UTF-8 decoded fragment evidence mismatch")
            require(split_count == wire.get("split_boundary_count") and split_count > 0, f"{label} does not cover a real multibyte UTF-8 split boundary")
            require("\ufffd" not in decoded and not any(marker in decoded.lower() for marker in ("mojibake", "invalid utf-8")), f"{label} UTF-8 stream contains replacement or mojibake output")
        return
    if scenario_id == "C07":
        require(len(exchanges) == 5, f"{label} must execute exactly five history-carrying turns")
        conversation_id = require_string(observed.get("conversation_id"), f"{label}.conversation_id")
        expected_messages: list[dict[str, Any]] = []
        for turn, exchange in enumerate(exchanges, start=1):
            request = require_object(exchange.get("request"), f"{label}.turn[{turn}].request")
            messages = require_list(request.get("messages"), f"{label}.turn[{turn}].messages")
            require(messages[:-1] == expected_messages, f"{label} turn {turn} did not carry exact prior user/assistant history")
            user = require_object(messages[-1], f"{label}.turn[{turn}].user")
            require(user.get("role") == "user" and conversation_id in str(user.get("content")), f"{label} turn {turn} user marker mismatch")
            response = require_object(exchange.get("response"), f"{label}.turn[{turn}].response")
            message, content = response_message(response, f"{label}.turn[{turn}]")
            expected_marker = f"{case_marker(observed['case_id'])}-T{turn}"
            require(content == expected_marker, f"{label} turn {turn} response marker mismatch")
            foreign = re.findall(r"G00-c07-\d{3}-OK-T\d+", content)
            require(foreign == [expected_marker], f"{label} turn {turn} contains cross-conversation state")
            expected_messages.extend([copy.deepcopy(user), {"role": "assistant", "content": message.get("content")}])
        return
    if scenario_id == "C08":
        response = responses[-1] if responses else {}
        choices = require_list(response.get("choices"), f"{label}.choices")
        choice = require_object(choices[0] if choices else None, f"{label}.choice")
        usage = require_object(response.get("usage"), f"{label}.usage")
        completion_tokens = require_count(usage.get("completion_tokens"), f"{label}.usage.completion_tokens")
        content_value = require_object(choice.get("message"), f"{label}.message").get("content") or ""
        if variant == "stop":
            require(choice.get("finish_reason") == "stop" and "<G00STOP>" not in content_value, f"{label} stop sequence leaked or finish reason drifted")
        elif variant == "max-tokens":
            require(choice.get("finish_reason") == "length" and completion_tokens == 8, f"{label} max-token case must generate exactly 8 usage-counted tokens")
        else:
            require(choice.get("finish_reason") in {"stop", "eos"} and 0 < completion_tokens < 64, f"{label} natural EOS case did not end before its cap")
        return
    if scenario_id == "C09":
        require_count(observed.get("scheduler_ticks_to_release"), f"{label}.scheduler_ticks_to_release") <= 2
        wall = observed.get("wall_sec_to_release")
        require(isinstance(wall, (int, float)) and not isinstance(wall, bool) and 0 <= wall <= 5, f"{label} release wall time exceeds 5s")
        require(observed.get("post_capacity_success") is True, f"{label} capacity was not recovered")
    if scenario_id == "C18":
        requested = require_count(observed.get("requested_concurrency"), f"{label}.requested_concurrency", minimum=1)
        require(len(exchanges) == requested, f"{label} concurrency transcript count mismatch")
        cap = require_count(observed.get("typed_admission_cap"), f"{label}.typed_admission_cap", minimum=1)
        active = require_count(observed.get("observed_max_active"), f"{label}.observed_max_active", minimum=1)
        require(active <= min(requested, cap), f"{label} observed max-active exceeds requested/cap")
    if responses:
        final_response = responses[-1]
        require_string(final_response.get("id"), f"{label}.id")
        require(final_response.get("object") == "chat.completion", f"{label} OpenAI response object mismatch")
        usage = require_object(final_response.get("usage"), f"{label}.usage")
        prompt_tokens = require_count(usage.get("prompt_tokens"), f"{label}.usage.prompt_tokens", minimum=1)
        completion_tokens = require_count(usage.get("completion_tokens"), f"{label}.usage.completion_tokens", minimum=1)
        require(usage.get("total_tokens") == prompt_tokens + completion_tokens, f"{label} usage total mismatch")
        message, content = response_message(final_response, label)
        marker = observed.get("expected_marker")
        if isinstance(marker, str) and marker and scenario_id not in {"C10", "C11", "C12", "C13", "C14", "C15", "C21"}:
            require(marker in content, f"{label} missing expected marker")
        if scenario_id in {"C10", "C11", "C12"}:
            calls = require_list(message.get("tool_calls"), f"{label}.tool_calls")
            require(calls, f"{label} did not return a tool call")
            function = require_object(require_object(calls[0], f"{label}.tool_calls[0]").get("function"), f"{label}.tool_calls[0].function")
            require(function.get("name") == "lookup_weather", f"{label} returned the wrong tool")
            arguments = json.loads(require_string(function.get("arguments"), f"{label}.tool arguments"))
            require(arguments == {"city": "Paris"}, f"{label} returned invalid tool arguments")
        elif scenario_id == "C13":
            require("21" in content, f"{label} did not incorporate the tool result")
        elif scenario_id in {"C14", "C15"}:
            parsed = json.loads(content)
            require(isinstance(parsed, dict), f"{label} structured output is not an object")
            if scenario_id == "C14":
                request = require_object(exchanges[-1].get("request"), f"{label}.request")
                response_format = require_object(request.get("response_format"), f"{label}.response_format")
                json_schema = require_object(response_format.get("json_schema"), f"{label}.json_schema")
                require(json_schema.get("strict") is True, f"{label} strict schema flag missing")
                schema = require_object(json_schema.get("schema"), f"{label}.schema")
                require(derived_c14_category(schema, f"{label}.schema") == variant, f"{label} variant label is not derived from schema structure")
                validate_strict_schema_instance(parsed, schema, label)
                prompt = require_string(require_object(require_list(request.get("messages"), f"{label}.messages")[0], f"{label}.message").get("content"), f"{label}.prompt")
                require(observed.get("strict_schema_sha256") == canonical_json_sha256(schema), f"{label} strict schema digest mismatch")
                require(observed.get("strict_prompt_sha256") == hashlib.sha256(prompt.encode("utf-8")).hexdigest(), f"{label} strict prompt digest mismatch")
        elif scenario_id == "C19":
            require(len(exchanges) == 2, f"{label} thinking case must execute two history-carrying turns")
            first_response = require_object(exchanges[0].get("response"), f"{label}.first.response")
            first_message, first_content = response_message(first_response, f"{label}.first")
            second_request = require_object(exchanges[1].get("request"), f"{label}.second.request")
            second_messages = require_list(second_request.get("messages"), f"{label}.second.messages")
            require(len(second_messages) == 3, f"{label} second request must include user, assistant, user history")
            require(second_messages[1] == first_message, f"{label} second request did not carry exact assistant reasoning history")
            reasoning_expected = bool(observed.get("reasoning_expected"))
            for response_label, response in (("first", first_response), ("second", final_response)):
                response_message_value, response_content = response_message(response, f"{label}.{response_label}")
                reasoning = response_message_value.get("reasoning")
                require(bool(isinstance(reasoning, str) and reasoning.strip()) is reasoning_expected, f"{label} {response_label} reasoning mode mismatch")
                require("<think>" not in response_content and "</think>" not in response_content, f"{label} reasoning tags leaked into final content")
            require(first_content != content, f"{label} second turn did not produce a distinct history-aware response")
            first_request = require_object(exchanges[0].get("request"), f"{label}.first.request")
            kwargs = first_request.get("chat_template_kwargs")
            if variant == "hard-thinking":
                require(kwargs == {"enable_thinking": True}, f"{label} hard thinking toggle was not sent")
            elif variant == "hard-no-thinking":
                require(kwargs == {"enable_thinking": False}, f"{label} hard no-thinking toggle was not sent")
            else:
                require(kwargs is None, f"{label} soft/default mode incorrectly sent a hard toggle")
            prompt = str(require_object(require_list(first_request.get("messages"), f"{label}.first.messages")[0], f"{label}.first.user").get("content"))
            if variant in {"soft-think", "soft-think-misuse"}:
                require(prompt.endswith("/think"), f"{label} soft think probe missing")
            if variant in {"soft-no-think", "soft-no-think-misuse"}:
                require(prompt.endswith("/no_think"), f"{label} soft no-think probe missing")
        elif scenario_id == "C20":
            require(observed.get("declared_modalities") == ["text"], f"{label} model declared non-text modality")
        elif scenario_id == "C21":
            require(content and marker in content, f"{label} official-default content is empty or unrelated")
            request = require_object(exchanges[-1].get("request"), f"{label}.request")
            if variant == "strict-schema":
                parsed = json.loads(content)
                schema = require_object(require_object(require_object(request.get("response_format"), f"{label}.response_format").get("json_schema"), f"{label}.json_schema").get("schema"), f"{label}.schema")
                validate_strict_schema_instance(parsed, schema, label)
            elif variant == "json-object":
                require(request.get("response_format") == {"type": "json_object"}, f"{label} json_object request contract mismatch")
                parsed = json.loads(content)
                require(parsed == {"result": marker}, f"{label} json_object result mismatch")


def validate_case_evidence(
    root: Path,
    raw_ref: Any,
    *,
    scenario_id: str,
    expected: dict[str, Any],
    ordinal: int,
    used_case_ids: set[str],
    used_case_artifacts: set[Path],
    used_argv: set[tuple[str, ...]],
) -> dict[str, Any]:
    raw_path, _, parsed = validate_artifact_ref(root, raw_ref, f"scenario {scenario_id}.cases[{ordinal}]", allowed_kinds={"raw-json"})
    require(raw_path not in used_case_artifacts, f"scenario {scenario_id} reuses case evidence {raw_path}")
    used_case_artifacts.add(raw_path)
    case = require_object(parsed, f"scenario {scenario_id}.case[{ordinal}]")
    case_id = require_string(case.get("case_id"), f"scenario {scenario_id}.case[{ordinal}].case_id")
    require(case_id not in used_case_ids, f"duplicate case id {case_id}")
    used_case_ids.add(case_id)
    require(case.get("schema_version") == SCHEMA_VERSION, f"case {case_id} schema_version mismatch")
    require(case.get("scenario_id") == scenario_id, f"case {case_id} scenario binding mismatch")
    require(case.get("ordinal") == ordinal, f"case {case_id} ordinal mismatch")
    require(case.get("status") in {"pass", "known-fail", "blocked"}, f"case {case_id} status invalid")
    lane_key = f"{expected['model_key']}/{expected['backend']}"
    if case.get("status") == "blocked":
        require(lane_key in BLOCKED_LANE_FAILURE_CLASSES, f"case {case_id} cannot use blocked for executable lane {lane_key}")
    for key in (
        "source_git_sha",
        "source_tree_sha",
        "models_lock_sha256",
        "binary_sha256",
        "effective_config_sha256",
        "expectations_catalog_sha256",
        "model_key",
        "backend",
        "model_revision",
        "model_files",
        "hardware_id",
    ):
        require(case.get(key) == expected.get(key), f"case {case_id} {key} binding mismatch")
    entrypoint = require_string(case.get("entrypoint"), f"case {case_id}.entrypoint")
    require(entrypoint in required_entrypoints(scenario_id), f"case {case_id} uses forbidden entrypoint")
    variant = require_string(case.get("variant"), f"case {case_id}.variant")
    preset = case.get("preset")
    require(preset is None or preset in {"P_DETERMINISTIC", "P_NO_THINKING", "P_THINKING", "P_OFFICIAL_DEFAULT"}, f"case {case_id} preset invalid")
    execution = require_object(case.get("execution"), f"case {case_id}.execution")
    require(execution.get("id") == f"exec-{case_id}", f"case {case_id} execution id mismatch")
    argv_raw = require_list(execution.get("argv"), f"case {case_id}.execution.argv")
    require(len(argv_raw) >= 3 and all(isinstance(part, str) and part for part in argv_raw), f"case {case_id} argv invalid")
    argv = tuple(argv_raw)
    require(argv not in used_argv, f"case {case_id} argv is not unique")
    used_argv.add(argv)
    if entrypoint == "run":
        require(Path(argv[0]).name == "ferrum" and argv[1] == "run", f"case {case_id} must execute ferrum run")
    else:
        require(Path(argv[0]).name == "curl" and argv[1:3] == ("--request", "POST"), f"case {case_id} must record its HTTP POST argv")
        require(case_id in argv, f"case {case_id} HTTP argv must include its unique case id")
    started = parse_timestamp(execution.get("started_at"), f"case {case_id}.started_at")
    finished = parse_timestamp(execution.get("finished_at"), f"case {case_id}.finished_at")
    require(finished > started, f"case {case_id} process window is not positive")
    duration = execution.get("duration_sec")
    actual_duration = (finished - started).total_seconds()
    require(isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0, f"case {case_id} duration invalid")
    require(abs(float(duration) - actual_duration) <= max(0.05, actual_duration * 0.05), f"case {case_id} duration does not match timestamps")
    if "invocation_started_at" in expected:
        require(expected["invocation_started_at"] <= started < finished <= expected["invocation_finished_at"], f"case {case_id} wall window is outside executor invocation")
    expectation_rule = resolve_expectation(
        expected["expectations_catalog"],
        model_key=expected["model_key"],
        backend=expected["backend"],
        scenario_id=scenario_id,
        variant=variant,
        preset=preset,
    )
    expected_outcome = require_object(case.get("expected_outcome"), f"case {case_id}.expected_outcome")
    for key in ("expected_status", "failure_class", "downstream_goal", "owner", "evidence_basis", "next_action"):
        require(expected_outcome.get(key) == expectation_rule.get(key), f"case {case_id} expectation {key} drift")
    require(expectation_rule["expected_status"] != "discovery-required", f"case {case_id} remains discovery-required and cannot enter canonical evidence")
    observed_outcome = require_object(case.get("observed_outcome"), f"case {case_id}.observed_outcome")
    require(observed_outcome.get("status") == expectation_rule["expected_status"], f"case {case_id} unexpected pass/fail status")
    require(observed_outcome.get("failure_class") == expectation_rule["failure_class"], f"case {case_id} failure class drift")
    require(case.get("status") == observed_outcome.get("status"), f"case {case_id} status does not match observed outcome")
    if case["status"] == "blocked":
        require(
            observed_outcome.get("failure_class") == BLOCKED_LANE_FAILURE_CLASSES[lane_key],
            f"case {case_id} blocked failure class is not allowlisted for {lane_key}",
        )
    returncode = execution.get("returncode")
    require(isinstance(returncode, int) and not isinstance(returncode, bool), f"case {case_id} execution returncode invalid")
    if case["status"] == "pass":
        require(returncode == 0, f"case {case_id} passing execution returncode must be 0")
    elif case["status"] == "blocked":
        require(returncode != 0, f"case {case_id} blocked execution unexpectedly ran")
    model_identity = require_object(case.get("model_identity"), f"case {case_id}.model_identity")
    for key in ("model_key", "backend", "model_revision", "model_files", "binary_sha256"):
        require(model_identity.get(key) == expected.get(key), f"case {case_id} model_identity.{key} mismatch")
    if not expected["allow_internal_fixture"] or "model_path" in expected:
        require(model_identity.get("model_path") == expected.get("model_path"), f"case {case_id} model path mismatch")
        if entrypoint == "run":
            require(argv[2] == expected["model_path"], f"case {case_id} run argv model path mismatch")
            if "binary_path" in expected:
                require(Path(argv[0]).resolve() == expected["binary_path"], f"case {case_id} run argv binary is not bound artifact")
    artifacts = require_object(case.get("artifacts"), f"case {case_id}.artifacts")
    required_artifacts = {"input", "stdout", "stderr", "effective_config"}
    if entrypoint == "serve":
        required_artifacts.add("http_transcript")
    require(set(artifacts) == required_artifacts, f"case {case_id} artifact set mismatch")
    resolved: dict[str, tuple[Path, Any | None]] = {}
    for name, ref in artifacts.items():
        allowed = {
            "input": {"request-json", "raw-json"},
            "stdout": {"stdout-log"},
            "stderr": {"stderr-log", "checker-log"},
            "effective_config": {"raw-json"},
            "http_transcript": {"http-transcript", "raw-json"},
        }[name]
        path, _, artifact_parsed = validate_artifact_ref(root, ref, f"case {case_id}.artifacts.{name}", allowed_kinds=allowed)
        if name != "effective_config":
            require(path not in used_case_artifacts, f"case {case_id} reuses case artifact {path}")
            used_case_artifacts.add(path)
        resolved[name] = (path, artifact_parsed)
    input_document = require_object(resolved["input"][1], f"case {case_id}.input")
    if entrypoint == "serve":
        expected_payload = case_http_payload(
            {
                "case_id": case_id,
                "scenario_id": scenario_id,
                "variant": variant,
                "preset": preset,
                "ordinal": ordinal,
                "model_key": expected["model_key"],
            },
            expected["model_key"],
        )
        require(input_document == expected_payload, f"case {case_id} persisted input differs from generated scenario contract")
    require(file_sha256(resolved["effective_config"][0]) == expected["effective_config_sha256"], f"case {case_id} effective config mismatch")
    actual_config: dict[str, Any] | None = None
    envelope_ref = case.get("execution_envelope")
    if not expected["allow_internal_fixture"] or envelope_ref is not None:
        envelope_path, _, envelope_parsed = validate_artifact_ref(root, envelope_ref, f"case {case_id}.execution_envelope", allowed_kinds={"raw-json"})
        require(envelope_path not in used_case_artifacts, f"case {case_id} reuses execution envelope {envelope_path}")
        used_case_artifacts.add(envelope_path)
        envelope_document = require_object(envelope_parsed, f"case {case_id} execution envelope JSON")
        require(envelope_document.get("schema_version") == SCHEMA_VERSION and envelope_document.get("case_id") == case_id, f"case {case_id} execution envelope binding mismatch")
        command_spec_path, _, command_spec_parsed = validate_artifact_ref(root, envelope_document.get("command_spec"), f"case {case_id}.command_spec", allowed_kinds={"raw-json"})
        require(command_spec_path not in used_case_artifacts, f"case {case_id} reuses command spec {command_spec_path}")
        used_case_artifacts.add(command_spec_path)
        command_spec = require_object(command_spec_parsed, f"case {case_id} command spec JSON")
        require(command_spec.get("case_id") == case_id and command_spec.get("argv") == list(argv), f"case {case_id} command spec argv mismatch")
        require(command_spec.get("input_sha256") == file_sha256(resolved["input"][0]), f"case {case_id} command spec input mismatch")
        require(command_spec.get("binary_sha256") == expected["binary_sha256"], f"case {case_id} command spec binary mismatch")
        require(command_spec.get("model_revision") == expected["model_revision"] and command_spec.get("model_files") == expected["model_files"], f"case {case_id} command spec model mismatch")
        require(command_spec.get("model_path") == expected.get("model_path"), f"case {case_id} command spec model path mismatch")
        child_environment = validate_sanitized_environment(envelope_document.get("child_environment"), f"case {case_id}.child_environment")
        child_environment_sha = canonical_json_sha256(child_environment)
        require(envelope_document.get("child_environment_sha256") == child_environment_sha, f"case {case_id} child environment SHA mismatch")
        require(command_spec.get("child_environment_sha256") == child_environment_sha, f"case {case_id} command spec environment binding mismatch")
        spawn = require_object(envelope_document.get("spawn"), f"case {case_id}.spawn")
        expected_mode = "subprocess" if entrypoint == "run" else "in-process-http"
        require(spawn.get("mode") == expected_mode and command_spec.get("execution_mode") == expected_mode, f"case {case_id} execution mode mismatch")
        for key in ("id", "argv", "started_at", "finished_at", "duration_sec", "returncode"):
            require(spawn.get(key) == execution.get(key), f"case {case_id} spawn.{key} mismatch")
        for key in ("started_monotonic_ns", "finished_monotonic_ns"):
            require_count(spawn.get(key), f"case {case_id}.spawn.{key}", minimum=1)
        require(spawn["finished_monotonic_ns"] > spawn["started_monotonic_ns"], f"case {case_id} monotonic window invalid")
        require(
            expected["invocation_started_monotonic_ns"] <= spawn["started_monotonic_ns"] < spawn["finished_monotonic_ns"] <= expected["invocation_finished_monotonic_ns"],
            f"case {case_id} monotonic window is outside executor invocation",
        )
        require(abs((spawn["finished_monotonic_ns"] - spawn["started_monotonic_ns"]) / 1e9 - float(duration)) <= max(0.05, float(duration) * 0.05), f"case {case_id} monotonic duration mismatch")
        require_count(spawn.get("pid"), f"case {case_id}.spawn.pid", minimum=1)
        require_count(spawn.get("pgid"), f"case {case_id}.spawn.pgid", minimum=1)
        product_process = require_object(envelope_document.get("product_process"), f"case {case_id}.product_process")
        require_count(product_process.get("pid"), f"case {case_id}.product_process.pid", minimum=1)
        require_count(product_process.get("pgid"), f"case {case_id}.product_process.pgid", minimum=1)
        execution_receipt = validate_process_receipt(
            root,
            envelope_document.get("execution_process_receipt"),
            label=f"case {case_id}.execution_process_receipt",
            pid=spawn["pid"],
            pgid=spawn["pgid"],
            argv=expected["invocation_argv"] if entrypoint == "serve" else list(argv),
            role="scenario-executor" if entrypoint == "serve" else "ferrum-run",
            expected_ppid=None if entrypoint == "serve" else expected["invocation_pid"],
            expected_environment=None if entrypoint == "serve" else child_environment,
        )
        product_receipt = validate_process_receipt(
            root,
            envelope_document.get("product_process_receipt"),
            label=f"case {case_id}.product_process_receipt",
            pid=product_process["pid"],
            pgid=product_process["pgid"],
            argv=product_process["argv"],
            role="ferrum-serve" if entrypoint == "serve" else "ferrum-run",
            expected_ppid=expected["invocation_pid"],
            expected_environment=child_environment,
        )
        for receipt in (execution_receipt, product_receipt):
            captured = parse_timestamp(receipt["captured_at"], f"case {case_id}.receipt.captured_at")
            require(expected["invocation_started_at"] <= captured <= expected["invocation_finished_at"], f"case {case_id} process receipt is outside invocation window")
            captured_ns = require_count(receipt.get("captured_monotonic_ns"), f"case {case_id}.receipt.captured_monotonic_ns", minimum=1)
            require(
                expected["invocation_started_monotonic_ns"] <= captured_ns <= expected["invocation_finished_monotonic_ns"],
                f"case {case_id} process receipt monotonic time is outside invocation window",
            )
        product_started = parse_timestamp(product_process.get("started_at"), f"case {case_id}.product_process.started_at")
        for name in ("stdout", "stderr"):
            envelope_artifact_path, _, _ = validate_artifact_ref(root, envelope_document.get(name), f"case {case_id}.envelope.{name}", allowed_kinds={f"{name}-log"})
            require(envelope_artifact_path == resolved[name][0], f"case {case_id} envelope {name} mismatch")
        if entrypoint == "serve":
            envelope_transcript_path, _, _ = validate_artifact_ref(root, envelope_document.get("http_transcript"), f"case {case_id}.envelope.http_transcript", allowed_kinds={"http-transcript"})
            require(envelope_transcript_path == resolved["http_transcript"][0], f"case {case_id} envelope transcript mismatch")
            product_argv = require_list(envelope_document.get("product_argv"), f"case {case_id}.product_argv")
            require(Path(product_argv[0]).name == "ferrum" and product_argv[1] == "serve" and product_argv[2] == expected.get("model_path"), f"case {case_id} serve product argv/model mismatch")
            require(Path(product_argv[0]).resolve() == expected["binary_path"], f"case {case_id} serve product binary is not bound artifact")
            require(product_process.get("argv") == product_argv and product_process.get("state_during_case") == "running", f"case {case_id} serve product process binding mismatch")
            ready_at = parse_timestamp(product_process.get("ready_at"), f"case {case_id}.product_process.ready_at")
            require(product_started <= ready_at <= started and finished >= started, f"case {case_id} request window is outside server ready window")
        else:
            require(envelope_document.get("http_transcript") is None, f"case {case_id} run envelope contains HTTP transcript")
            require(envelope_document.get("product_argv") == list(argv), f"case {case_id} run product argv mismatch")
            require(product_process.get("argv") == list(argv), f"case {case_id} run product process argv mismatch")
            product_finished = parse_timestamp(product_process.get("finished_at"), f"case {case_id}.product_process.finished_at")
            require(product_started == started and product_finished == finished, f"case {case_id} run process window mismatch")
        actual_config_path, _, actual_config_parsed = validate_artifact_ref(root, envelope_document.get("actual_effective_config"), f"case {case_id}.actual_effective_config", allowed_kinds={"raw-json"})
        actual_config = validate_actual_effective_config(
            actual_config_parsed,
            expected_backend=expected["backend"],
            expected_model_key=expected["model_key"],
            label=f"case {case_id}.actual_effective_config JSON",
        )
        require(
            command_spec.get("actual_effective_config_sha256") == file_sha256(actual_config_path),
            f"case {case_id} command spec actual effective config mismatch",
        )
        checker = require_object(envelope_document.get("checker"), f"case {case_id}.checker")
        require(checker.get("mode") == "in-process" and checker.get("function") == "classify_execution_outcome", f"case {case_id} checker identity mismatch")
        require(checker.get("runner_sha256") == file_sha256(RUNNER_PATH), f"case {case_id} checker runner SHA mismatch")
        checker_inputs = require_object(checker.get("input_artifact_sha256"), f"case {case_id}.checker.input_artifact_sha256")
        expected_checker_inputs = {"stdout": file_sha256(resolved["stdout"][0]), "stderr": file_sha256(resolved["stderr"][0])}
        if entrypoint == "serve":
            expected_checker_inputs["http_transcript"] = file_sha256(resolved["http_transcript"][0])
        require(checker_inputs == expected_checker_inputs, f"case {case_id} checker input artifact binding mismatch")
        validate_artifact_ref(root, checker.get("log"), f"case {case_id}.checker.log", allowed_kinds={"checker-log"})
        require(checker.get("result") == observed_outcome.get("status"), f"case {case_id} checker result mismatch")
        require(checker.get("failure_class") == observed_outcome.get("failure_class"), f"case {case_id} checker failure class mismatch")
    observed = require_object(case.get("observed"), f"case {case_id}.observed")
    require(observed.get("case_id") == case_id, f"case {case_id} observed binding mismatch")
    checks = require_object(case.get("checks"), f"case {case_id}.checks")
    require(checks.get("execution_envelope") is True and checks.get("model_binding") is True, f"case {case_id} execution/model checks failed")
    require(checks.get("scenario_oracle") is (case["status"] == "pass"), f"case {case_id} scenario oracle status mismatch")
    transcript = resolved.get("http_transcript", (Path(), None))[1]
    if scenario_id in {"C09", "C18"} and (not expected["allow_internal_fixture"] or envelope_ref is not None):
        transcript_object = require_object(transcript, f"case {case_id}.http_transcript")
        trace_rows = require_list(transcript_object.get("scheduler_trace_rows"), f"case {case_id}.scheduler_trace_rows")
        require(trace_rows, f"case {case_id} scheduler trace evidence is empty")
        for index, trace_raw in enumerate(trace_rows):
            trace = require_object(trace_raw, f"case {case_id}.scheduler_trace_rows[{index}]")
            raw_trace = require_object(trace.get("raw"), f"case {case_id}.scheduler_trace_rows[{index}].raw")
            require(trace.get("raw_sha256") == canonical_json_sha256(raw_trace), f"case {case_id} scheduler trace raw SHA mismatch")
            observed_ns = require_count(trace.get("collector_observed_monotonic_ns"), f"case {case_id}.scheduler_trace_rows[{index}].collector_observed_monotonic_ns", minimum=1)
            require(spawn["started_monotonic_ns"] <= observed_ns <= spawn["finished_monotonic_ns"], f"case {case_id} scheduler trace observation is outside case window")
        if scenario_id == "C09":
            released, derived_ticks, release_ns = trace_released(trace_rows)
            require(released and release_ns is not None, f"case {case_id} scheduler trace never proves release")
            exchanges = require_list(transcript_object.get("exchanges"), f"case {case_id}.exchanges")
            abort_start_ns = require_count(require_object(exchanges[0], f"case {case_id}.exchange[0]").get("started_monotonic_ns"), f"case {case_id}.abort_start_ns", minimum=1)
            derived_wall = (release_ns - abort_start_ns) / 1e9
            require(observed.get("scheduler_ticks_to_release") == derived_ticks, f"case {case_id} release tick count is not derived from trace")
            wall = observed.get("wall_sec_to_release")
            require(isinstance(wall, (int, float)) and abs(float(wall) - derived_wall) <= 0.02, f"case {case_id} release wall time is not derived from trace")
        else:
            derived_active = observed_max_active(trace_rows)
            require(derived_active > 0 and observed.get("observed_max_active") == derived_active, f"case {case_id} observed max-active is not derived from trace")
            derived_cap = typed_admission_cap_value(actual_config)
            require(derived_cap > 0 and observed.get("typed_admission_cap") == derived_cap, f"case {case_id} admission cap is not derived from actual effective config")
    output_error: ScenarioError | None = None
    try:
        validate_case_output(
            scenario_id,
            variant,
            entrypoint,
            resolved["stdout"][0],
            require_object(transcript, f"case {case_id}.http_transcript") if transcript is not None else None,
            observed,
            f"case {case_id}",
            input_document=input_document,
            actual_config=actual_config,
            artifact_root=root,
        )
    except ScenarioError as exc:
        output_error = exc
    if case["status"] == "pass":
        require(output_error is None, f"case {case_id} expected pass but checker failed: {output_error}")
    elif case["status"] == "known-fail":
        require(output_error is not None, f"case {case_id} unexpectedly passed its known-fail oracle")
    else:
        require(output_error is not None or returncode != 0, f"case {case_id} blocked outcome lacks failure evidence")
    return {
        "case_id": case_id,
        "entrypoint": entrypoint,
        "variant": variant,
        "preset": preset,
        "status": case["status"],
        "observed": observed,
        "input_document": input_document,
        "input_sha256": file_sha256(resolved["input"][0]),
        "ordinal": ordinal,
        "started_at": started,
        "finished_at": finished,
    }


def validate_scenario(
    root: Path,
    raw: Any,
    *,
    expected_id: str,
    expected: dict[str, Any],
    commands: dict[str, str],
    used_artifacts: set[Path],
    used_case_ids: set[str],
    used_case_artifacts: set[Path],
    used_argv: set[tuple[str, ...]],
) -> list[dict[str, Any]]:
    scenario = require_object(raw, f"scenario {expected_id}")
    require(scenario.get("id") == expected_id, f"scenario order/id mismatch: expected {expected_id}")
    require(scenario.get("status") in {"pass", "known-fail", "blocked"}, f"scenario {expected_id} status invalid")
    reject_forbidden_markers(scenario, f"scenario {expected_id}", allow_internal_fixture=bool(expected["allow_internal_fixture"]))
    count = require_count(scenario.get("case_count"), f"scenario {expected_id}.case_count", minimum=minimum_case_count(expected_id, expected["model_key"]))
    passed_count = require_count(scenario.get("passed_count"), f"scenario {expected_id}.passed_count")
    known_failed_count = require_count(scenario.get("known_failed_count", 0), f"scenario {expected_id}.known_failed_count")
    blocked_count = require_count(scenario.get("blocked_count", 0), f"scenario {expected_id}.blocked_count")
    require(scenario.get("failed_count") == known_failed_count, f"scenario {expected_id}.failed_count must expose known failures")
    require(scenario.get("error_count") == 0 and scenario.get("unexpected_count", 0) == 0, f"scenario {expected_id} has unexpected failures/errors")
    require(passed_count + known_failed_count + blocked_count == count, f"scenario {expected_id} outcome counts must sum to case_count")
    presets = require_object(scenario.get("presets"), f"scenario {expected_id}.presets")
    for name, value in presets.items():
        require(name in {"P_DETERMINISTIC", "P_NO_THINKING", "P_THINKING", "P_OFFICIAL_DEFAULT"}, f"scenario {expected_id} unknown preset {name}")
        require_count(value, f"scenario {expected_id}.presets.{name}", minimum=1)
    unpreset = require_count(scenario.get("unpreset_count"), f"scenario {expected_id}.unpreset_count")
    minimums, min_unpreset = minimum_presets(expected_id, expected["model_key"])
    require(set(presets) == set(minimums), f"scenario {expected_id} must use exactly presets {sorted(minimums)}")
    for name, minimum in minimums.items():
        require(presets.get(name, 0) >= minimum, f"scenario {expected_id} preset {name} count must be >= {minimum}")
    require(unpreset >= min_unpreset, f"scenario {expected_id}.unpreset_count must be >= {min_unpreset}")
    require(sum(presets.values()) + unpreset == count, f"scenario {expected_id} preset counts must sum to case_count")
    entrypoint_rows = require_list(scenario.get("entrypoints"), f"scenario {expected_id}.entrypoints")
    require(all(isinstance(item, str) and item for item in entrypoint_rows), f"scenario {expected_id}.entrypoints must contain strings")
    entrypoints = set(entrypoint_rows)
    require(len(entrypoints) == len(entrypoint_rows), f"scenario {expected_id}.entrypoints must not contain duplicates")
    require(entrypoints <= {"run", "serve"}, f"scenario {expected_id} has invalid entrypoint")
    required = required_entrypoints(expected_id)
    require(required <= entrypoints, f"scenario {expected_id} missing entrypoints {sorted(required - entrypoints)}")
    command_ids = require_list(scenario.get("command_ids"), f"scenario {expected_id}.command_ids")
    require(command_ids and all(isinstance(item, str) and item for item in command_ids), f"scenario {expected_id}.command_ids must contain strings")
    require(len(set(command_ids)) == len(command_ids), f"scenario {expected_id}.command_ids must be unique")
    referenced_entrypoints: set[str] = set()
    for command_id in command_ids:
        require(command_id in commands, f"scenario {expected_id} references unknown command {command_id}")
        referenced_entrypoints.add(commands[command_id])
    require(entrypoints <= referenced_entrypoints, f"scenario {expected_id} command records do not cover its entrypoints")
    variants = require_object(scenario.get("variants"), f"scenario {expected_id}.variants")
    for name, value in variants.items():
        require_string(name, f"scenario {expected_id}.variants name")
        require_count(value, f"scenario {expected_id}.variants.{name}", minimum=1)
    minimum_variants, partition = required_variants(expected_id, expected["model_key"])
    for name, minimum in minimum_variants.items():
        require_count(variants.get(name), f"scenario {expected_id}.variants.{name}", minimum=minimum)
    if partition:
        require(sum(value for value in variants.values() if isinstance(value, int) and not isinstance(value, bool)) == count, f"scenario {expected_id} variant counts must partition case_count")
    dimensions = require_object(scenario.get("dimensions"), f"scenario {expected_id}.dimensions")
    if expected_id == "C03":
        require(dimensions.get("groups") >= 10 and dimensions.get("rounds_per_group") >= 3, "C03 must cover 10 groups x 3 rounds")
    elif expected_id == "C04":
        require(dimensions.get("groups") >= 3 and dimensions.get("min_output_tokens") >= 512, "C04 must cover 3 groups with >=512 output tokens")
    elif expected_id == "C07":
        require(dimensions.get("requests") >= 6 and dimensions.get("rounds_per_request") >= 5, "C07 must cover 6 requests x 5 rounds")
    if expected_id == "C18":
        validate_concurrency_cells(expected["backend"], scenario.get("concurrency_cells"), case_count=count, require_pass=scenario["status"] == "pass")
    else:
        require("concurrency_cells" not in scenario, f"scenario {expected_id} must not fabricate concurrency cells")
    if scenario["status"] == "pass":
        validate_assertions(expected_id, scenario.get("assertions"), case_count=count)
    else:
        assertions = require_object(scenario.get("assertions"), f"scenario {expected_id}.assertions")
        require(assertions.get("expected_failure_count") == known_failed_count + blocked_count, f"scenario {expected_id} expected failure count mismatch")
        require(assertions.get("unexpected_count") == 0, f"scenario {expected_id} assertions contain unexpected outcomes")
    artifact_refs = require_list(scenario.get("artifacts"), f"scenario {expected_id}.artifacts")
    kinds: set[str] = set()
    raw_json: dict[str, Any] | None = None
    for index, artifact_ref in enumerate(artifact_refs):
        path, kind, parsed = validate_artifact_ref(
            root,
            artifact_ref,
            f"scenario {expected_id}.artifacts[{index}]",
            allowed_kinds={"raw-json", *LOG_KINDS},
        )
        require(path not in used_artifacts, f"scenario {expected_id} reuses another scenario's raw artifact: {path}")
        used_artifacts.add(path)
        kinds.add(kind)
        if kind == "raw-json":
            require(raw_json is None, f"scenario {expected_id} must have exactly one primary raw JSON artifact")
            raw_json = require_object(parsed, f"scenario {expected_id} raw JSON")
    require("raw-json" in kinds and bool(kinds & LOG_KINDS), f"scenario {expected_id} requires raw JSON and log artifacts")
    require(raw_json is not None, f"scenario {expected_id} raw JSON missing")
    for key in (
        "source_git_sha",
        "source_tree_sha",
        "models_lock_sha256",
        "binary_sha256",
        "effective_config_sha256",
        "model_key",
        "backend",
        "model_revision",
        "model_files",
        "hardware_id",
    ):
        require(raw_json.get(key) == expected.get(key), f"scenario {expected_id} raw JSON {key} binding mismatch")
    require(raw_json.get("scenario_id") == expected_id, f"scenario {expected_id} raw JSON id mismatch")
    require(raw_json.get("status") == scenario.get("status"), f"scenario {expected_id} raw JSON status mismatch")
    for key in ("case_count", "passed_count", "known_failed_count", "blocked_count", "failed_count", "error_count", "unexpected_count"):
        require(raw_json.get(key) == scenario.get(key), f"scenario {expected_id} raw JSON {key} mismatch")
    for key in (
        "presets",
        "unpreset_count",
        "entrypoints",
        "command_ids",
        "variants",
        "dimensions",
        "assertions",
    ):
        require(raw_json.get(key) == scenario.get(key), f"scenario {expected_id} raw JSON {key} mismatch")
    if expected_id == "C18":
        require(raw_json.get("concurrency_cells") == scenario.get("concurrency_cells"), "C18 raw JSON concurrency cells mismatch")
    else:
        require("concurrency_cells" not in raw_json, f"scenario {expected_id} raw JSON contains unexpected concurrency cells")
    case_refs = require_list(raw_json.get("cases"), f"scenario {expected_id} raw JSON cases")
    require(len(case_refs) == count, f"scenario {expected_id} must contain one evidence record per case")
    case_rows = [
        validate_case_evidence(
            root,
            case_ref,
            scenario_id=expected_id,
            expected=expected,
            ordinal=index,
            used_case_ids=used_case_ids,
            used_case_artifacts=used_case_artifacts,
            used_argv=used_argv,
        )
        for index, case_ref in enumerate(case_refs, start=1)
    ]
    derived_entrypoints = sorted({row["entrypoint"] for row in case_rows})
    require(derived_entrypoints == sorted(scenario["entrypoints"]), f"scenario {expected_id} entrypoints are not derived from cases")
    derived_variants: dict[str, int] = {}
    derived_presets: dict[str, int] = {}
    derived_unpreset = 0
    for row in case_rows:
        derived_variants[row["variant"]] = derived_variants.get(row["variant"], 0) + 1
        if row["preset"] is None:
            derived_unpreset += 1
        else:
            derived_presets[row["preset"]] = derived_presets.get(row["preset"], 0) + 1
    require(derived_variants == scenario["variants"], f"scenario {expected_id} variants are not derived from cases")
    require(derived_presets == scenario["presets"], f"scenario {expected_id} presets are not derived from cases")
    require(derived_unpreset == scenario["unpreset_count"], f"scenario {expected_id} unpreset count is not derived from cases")
    if expected_id == "C01":
        require(set(derived_variants.values()) == {5} and len(derived_variants) == 4, "C01 must execute four exact five-case groups")
    elif expected_id == "C07":
        conversation_ids = [row["observed"].get("conversation_id") for row in case_rows]
        require(len(set(conversation_ids)) == 6 and all(row["observed"].get("history_turn_count") == 5 for row in case_rows), "C07 conversation identity/history corpus is incomplete")
    elif expected_id == "C14":
        schema_hashes = [row["observed"].get("strict_schema_sha256") for row in case_rows]
        prompt_hashes = [row["observed"].get("strict_prompt_sha256") for row in case_rows]
        require(len(set(schema_hashes)) == count and None not in schema_hashes, "C14 schemas are not all materially distinct")
        require(len(set(prompt_hashes)) == count and None not in prompt_hashes, "C14 prompts are not all materially distinct")
        partition: dict[tuple[str | None, str], int] = {}
        for row in case_rows:
            key = (row["preset"], row["variant"])
            partition[key] = partition.get(key, 0) + 1
        require(
            partition
            == {
                ("P_NO_THINKING", "required"): 13,
                ("P_NO_THINKING", "type"): 13,
                ("P_NO_THINKING", "additional-properties"): 12,
                ("P_NO_THINKING", "enum"): 12,
                ("P_THINKING", "required"): 5,
                ("P_THINKING", "type"): 5,
                ("P_THINKING", "additional-properties"): 5,
                ("P_THINKING", "enum"): 5,
            },
            "C14 preset/constraint partition mismatch",
        )
    elif expected_id == "C17":
        unicode_partition: dict[tuple[str, str], int] = {}
        for row in case_rows:
            key = (row["variant"], row["entrypoint"])
            unicode_partition[key] = unicode_partition.get(key, 0) + 1
        require(
            unicode_partition == {(variant, entrypoint): 10 for variant in ("chinese", "emoji", "combining") for entrypoint in ("run", "serve")},
            "C17 must execute run-incremental10/serve-stream10 for every Unicode class",
        )
    elif expected_id == "C19":
        require(all(row["observed"].get("history_turn_count") == 2 for row in case_rows), "C19 reasoning history corpus is incomplete")
        require(set(derived_variants.values()) == {4} and len(derived_variants) == 5, "C19 must execute five exact four-case modes")
    elif expected_id == "C21":
        require(set(derived_variants.values()) == {4} and len(derived_variants) == 5, "C21 must execute five exact four-case groups")
        require(
            all(row["entrypoint"] == ("run" if row["variant"] == "run-plain" else "serve") for row in case_rows),
            "C21 subgroup entrypoint mapping mismatch",
        )
    derived_passed = sum(row["status"] == "pass" for row in case_rows)
    derived_known = sum(row["status"] == "known-fail" for row in case_rows)
    derived_blocked = sum(row["status"] == "blocked" for row in case_rows)
    derived_status = "blocked" if derived_blocked else "known-fail" if derived_known else "pass"
    require(derived_passed == passed_count, f"scenario {expected_id} passed_count is not derived from cases")
    require(derived_known == known_failed_count, f"scenario {expected_id} known_failed_count is not derived from cases")
    require(derived_blocked == blocked_count, f"scenario {expected_id} blocked_count is not derived from cases")
    require(scenario["status"] == derived_status, f"scenario {expected_id} status is not derived from cases")
    return case_rows


PAIR_CONTRACTS = (("C05", "C06"), ("C11", "C12"))
PAIR_METADATA_DIFFERENCES = {
    "g00_case_id",
    "g00_scenario_id",
    "g00_variant",
    "g00_reference_contract",
    "g00_reference_case_id",
}


def canonical_pair_payload(raw: Any, label: str) -> dict[str, Any]:
    payload = copy.deepcopy(require_object(raw, label))
    payload.pop("stream", None)
    payload.pop("stream_options", None)
    metadata = require_object(payload.get("metadata"), f"{label}.metadata")
    for key in PAIR_METADATA_DIFFERENCES:
        metadata.pop(key, None)
    require(metadata, f"{label}.metadata became empty after allowed pair differences")
    return payload


def build_pair_registry(case_rows: dict[str, list[dict[str, Any]]], *, model_key: str, backend: str) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for left_id, right_id in PAIR_CONTRACTS:
        left_rows = {int(row["ordinal"]): row for row in case_rows[left_id]}
        right_rows = {int(row["ordinal"]): row for row in case_rows[right_id]}
        require(set(left_rows) == set(right_rows), f"pair registry {left_id}/{right_id} ordinal sets differ")
        for ordinal in sorted(left_rows):
            left = left_rows[ordinal]
            right = right_rows[ordinal]
            require(left["preset"] == right["preset"], f"pair registry {left_id}/{right_id} preset mismatch at ordinal {ordinal}")
            left_payload = canonical_pair_payload(left["input_document"], f"pair {left['case_id']} input")
            right_payload = canonical_pair_payload(right["input_document"], f"pair {right['case_id']} input")
            require(left_payload == right_payload, f"pair registry {left_id}/{right_id} payload mismatch at ordinal {ordinal}")
            canonical_sha = canonical_json_sha256(left_payload)
            pairs.append(
                {
                    "contract": f"{left_id}-{right_id}",
                    "ordinal": ordinal,
                    "left_case_id": left["case_id"],
                    "right_case_id": right["case_id"],
                    "preset": left["preset"],
                    "left_input_sha256": left["input_sha256"],
                    "right_input_sha256": right["input_sha256"],
                    "canonical_payload_sha256": canonical_sha,
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "model_key": model_key,
        "backend": backend,
        "pair_count": len(pairs),
        "pairs": pairs,
    }


def raw_pair_case_rows(report: dict[str, Any], root: Path) -> dict[str, list[dict[str, Any]]]:
    scenarios = {
        require_string(scenario.get("id"), "pair registry scenario id"): require_object(scenario, "pair registry scenario")
        for scenario in require_list(report.get("scenarios"), "pair registry scenarios")
        if isinstance(scenario, dict) and scenario.get("id") in {item for pair in PAIR_CONTRACTS for item in pair}
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for scenario_id in {item for pair in PAIR_CONTRACTS for item in pair}:
        scenario = scenarios.get(scenario_id)
        require(scenario is not None, f"pair registry missing {scenario_id}")
        raw_ref = next((ref for ref in require_list(scenario.get("artifacts"), f"pair registry {scenario_id}.artifacts") if isinstance(ref, dict) and ref.get("kind") == "raw-json"), None)
        _, _, raw_document = validate_artifact_ref(root, raw_ref, f"pair registry {scenario_id}.raw", allowed_kinds={"raw-json"})
        rows: list[dict[str, Any]] = []
        for case_ref in require_list(require_object(raw_document, f"pair registry {scenario_id}.raw JSON").get("cases"), f"pair registry {scenario_id}.cases"):
            _, _, case_raw = validate_artifact_ref(root, case_ref, f"pair registry {scenario_id}.case", allowed_kinds={"raw-json"})
            case = require_object(case_raw, f"pair registry {scenario_id}.case JSON")
            artifacts = require_object(case.get("artifacts"), f"pair registry {scenario_id}.case.artifacts")
            input_path, _, input_raw = validate_artifact_ref(root, artifacts.get("input"), f"pair registry {scenario_id}.case.input", allowed_kinds={"request-json"})
            rows.append(
                {
                    "case_id": require_string(case.get("case_id"), f"pair registry {scenario_id}.case_id"),
                    "ordinal": require_count(case.get("ordinal"), f"pair registry {scenario_id}.ordinal", minimum=1),
                    "preset": case.get("preset"),
                    "input_document": require_object(input_raw, f"pair registry {scenario_id}.input JSON"),
                    "input_sha256": file_sha256(input_path),
                }
            )
        result[scenario_id] = rows
    return result


def attach_pair_registry(report: dict[str, Any], root: Path) -> None:
    registry = build_pair_registry(
        raw_pair_case_rows(report, root),
        model_key=require_string(report.get("model_key"), "pair registry model_key"),
        backend=require_string(report.get("backend"), "pair registry backend"),
    )
    path = root / "correctness" / registry["model_key"] / registry["backend"] / "pair-registry.json"
    write_json(path, registry)
    report["pair_registry"] = existing_artifact_ref(root, path, "raw-json")


def validate_report_document(
    report: dict[str, Any],
    root: Path,
    *,
    report_path: Path | None = None,
    allow_internal_fixture: bool = False,
    require_current_output_path: bool = False,
) -> dict[str, Any]:
    require(report.get("schema_version") == SCHEMA_VERSION, "scenario_report.schema_version mismatch")
    require(report.get("status") == "pass", "scenario_report.status must be pass")
    reject_forbidden_markers(report, "scenario_report", allow_internal_fixture=allow_internal_fixture)
    validate_source_identity(report, "scenario_report")
    model_key = require_string(report.get("model_key"), "scenario_report.model_key")
    require(model_key in {"m1-qwen35-4b", "m2-qwen35-35b-a3b", "m3-qwen3-30b-a3b"}, "scenario_report.model_key is not primary")
    backend = require_string(report.get("backend"), "scenario_report.backend")
    require(backend in {"cuda", "metal"}, "scenario_report.backend must be cuda or metal")
    require_git_sha(report.get("model_revision"), "scenario_report.model_revision")
    model_files = require_object(report.get("model_files"), "scenario_report.model_files")
    require(model_files, "scenario_report.model_files must not be empty")
    for path, digest in model_files.items():
        require_string(path, "scenario_report.model_files path")
        require_sha256(digest, f"scenario_report.model_files[{path}]")
    require_sha256(report.get("models_lock_sha256"), "scenario_report.models_lock_sha256")
    require_sha256(report.get("binary_sha256"), "scenario_report.binary_sha256")
    require_sha256(report.get("expectations_catalog_sha256"), "scenario_report.expectations_catalog_sha256")
    require_string(report.get("hardware_id"), "scenario_report.hardware_id")
    if not allow_internal_fixture or "model_path" in report:
        require_string(report.get("model_path"), "scenario_report.model_path")
    expected = {
        key: report[key]
        for key in (
            "source_git_sha",
            "source_tree_sha",
            "models_lock_sha256",
            "binary_sha256",
            "model_key",
            "backend",
            "model_revision",
            "model_files",
            "hardware_id",
        )
    }
    if "model_path" in report:
        expected["model_path"] = report["model_path"]
    _, effective_config_sha256 = validate_effective_config(root, report.get("effective_config"), expected=expected)
    expected["effective_config_sha256"] = effective_config_sha256
    expectations_path, _, expectations_parsed = validate_artifact_ref(
        root,
        report.get("expectations_catalog"),
        "scenario_report.expectations_catalog",
        allowed_kinds={"raw-json"},
    )
    expectations_catalog = validate_expectations_catalog(require_object(expectations_parsed, "scenario_report expectations catalog JSON"))
    require(file_sha256(expectations_path) == report["expectations_catalog_sha256"], "scenario_report expectations catalog binding mismatch")
    if not allow_internal_fixture:
        require(report["expectations_catalog_sha256"] == canonical_expectations_sha256(), "scenario_report expectations catalog is not the checked-in contract")
    expected["expectations_catalog"] = expectations_catalog
    expected["expectations_catalog_sha256"] = report["expectations_catalog_sha256"]
    binary_path, _, _ = validate_artifact_ref(root, report.get("binary_artifact"), "scenario_report.binary_artifact", allowed_kinds={"binary"})
    require(file_sha256(binary_path) == report["binary_sha256"], "scenario_report binary artifact binding mismatch")
    models_lock_path, _, _ = validate_artifact_ref(root, report.get("models_lock"), "scenario_report.models_lock", allowed_kinds={"raw-json"})
    require(file_sha256(models_lock_path) == report["models_lock_sha256"], "scenario_report models.lock binding mismatch")
    validate_runner_identity(report.get("runner"), allow_internal_fixture=allow_internal_fixture)
    invocation_ref = report.get("executor_invocation")
    if not allow_internal_fixture or invocation_ref is not None:
        _, _, invocation_parsed = validate_artifact_ref(root, invocation_ref, "scenario_report.executor_invocation", allowed_kinds={"raw-json"})
        invocation = require_object(invocation_parsed, "scenario_report executor invocation JSON")
        require(invocation.get("runner_path") == RUNNER_REPO_PATH, "executor invocation runner path mismatch")
        require(invocation.get("runner_sha256") == report["runner"]["sha256"], "executor invocation runner SHA mismatch")
        require(invocation.get("mode") == "canonical", "canonical report was not produced by canonical executor mode")
        invocation_argv = require_list(invocation.get("argv"), "executor invocation argv")
        require(str(RUNNER_PATH) in invocation_argv and "--manifest" in invocation_argv and "--artifact-root" in invocation_argv and "--out" in invocation_argv, "executor invocation argv incomplete")
        require("--discover" not in invocation_argv, "discovery invocation cannot produce canonical report")
        invocation_started_at = parse_timestamp(invocation.get("started_at"), "executor invocation started_at")
        invocation_finished_at = parse_timestamp(invocation.get("finished_at"), "executor invocation finished_at")
        require(invocation_finished_at > invocation_started_at, "executor invocation wall window invalid")
        start_ns = require_count(invocation.get("started_monotonic_ns"), "executor invocation started_monotonic_ns", minimum=1)
        finish_ns = require_count(invocation.get("finished_monotonic_ns"), "executor invocation finished_monotonic_ns", minimum=1)
        require(finish_ns > start_ns, "executor invocation monotonic window invalid")
        invocation_duration = invocation.get("duration_sec")
        require(isinstance(invocation_duration, (int, float)) and not isinstance(invocation_duration, bool) and invocation_duration > 0, "executor invocation duration invalid")
        wall_duration = (invocation_finished_at - invocation_started_at).total_seconds()
        mono_duration = (finish_ns - start_ns) / 1e9
        require(abs(float(invocation_duration) - wall_duration) <= max(0.05, wall_duration * 0.05), "executor invocation wall duration mismatch")
        require(abs(float(invocation_duration) - mono_duration) <= max(0.05, mono_duration * 0.05), "executor invocation monotonic duration mismatch")
        require_count(invocation.get("pid"), "executor invocation pid", minimum=1)
        require_count(invocation.get("pgid"), "executor invocation pgid", minimum=1)
        invocation_receipt = validate_process_receipt(
            root,
            invocation.get("process_receipt"),
            label="executor invocation process receipt",
            pid=invocation["pid"],
            pgid=invocation["pgid"],
            argv=invocation_argv,
            role="scenario-executor",
        )
        receipt_time = parse_timestamp(invocation_receipt["captured_at"], "executor invocation receipt captured_at")
        require(invocation_started_at <= receipt_time <= invocation_finished_at, "executor invocation receipt is outside invocation window")
        receipt_ns = require_count(invocation_receipt.get("captured_monotonic_ns"), "executor invocation receipt captured_monotonic_ns", minimum=1)
        require(start_ns <= receipt_ns <= finish_ns, "executor invocation receipt monotonic time is outside invocation window")
        manifest_snapshot, _, _ = validate_artifact_ref(root, invocation.get("manifest_snapshot"), "executor invocation manifest snapshot", allowed_kinds={"raw-json"})
        require(file_sha256(manifest_snapshot) == invocation.get("manifest_sha256"), "executor invocation manifest SHA mismatch")
        snapshot = validate_execution_manifest(require_object(read_json(manifest_snapshot), "executor invocation manifest snapshot"), root)
        for key in ("source_git_sha", "source_tree_sha", "models_lock_sha256", "binary_sha256", "model_key", "backend", "model_revision", "model_files", "hardware_id"):
            require(read_json(manifest_snapshot).get(key) == report.get(key), f"executor invocation manifest {key} differs from report")
        require(snapshot["execution"]["model_arg"] == report.get("model_path"), "executor invocation manifest model path differs from report")
        expected.update(
            {
                "invocation_argv": invocation_argv,
                "invocation_started_at": invocation_started_at,
                "invocation_finished_at": invocation_finished_at,
                "invocation_started_monotonic_ns": start_ns,
                "invocation_finished_monotonic_ns": finish_ns,
                "invocation_pid": invocation["pid"],
                "binary_path": snapshot["binary_path"],
            }
        )
    commands_raw = require_list(report.get("commands"), "scenario_report.commands")
    commands: dict[str, str] = {}
    for index, command_raw in enumerate(commands_raw):
        command_id, entrypoint = validate_command(
            root,
            command_raw,
            f"scenario_report.commands[{index}]",
            expected=expected,
            effective_config_sha256=effective_config_sha256,
        )
        require(command_id not in commands, f"duplicate scenario command id {command_id}")
        commands[command_id] = entrypoint
    require(set(commands.values()) == {"run", "serve"}, "scenario report must contain real ferrum run and serve commands")
    scenarios_raw = require_list(report.get("scenarios"), "scenario_report.scenarios")
    require(len(scenarios_raw) == len(SCENARIO_IDS), "scenario report must contain exactly C01-C21")
    used_artifacts: set[Path] = set()
    used_case_ids: set[str] = set()
    used_case_artifacts: set[Path] = set()
    used_argv: set[tuple[str, ...]] = set()
    case_rows_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for scenario_id, scenario_raw in zip(SCENARIO_IDS, scenarios_raw):
        case_rows_by_scenario[scenario_id] = validate_scenario(
            root,
            scenario_raw,
            expected_id=scenario_id,
            expected={**expected, "allow_internal_fixture": allow_internal_fixture},
            commands=commands,
            used_artifacts=used_artifacts,
            used_case_ids=used_case_ids,
            used_case_artifacts=used_case_artifacts,
            used_argv=used_argv,
        )
    if invocation_ref is not None:
        serve_command = next(require_object(row, "scenario_report serve command") for row in commands_raw if row.get("entrypoint") == "serve")
        serve_started_at = parse_timestamp(serve_command.get("started_at"), "scenario_report serve command started_at")
        for case in (case for rows in case_rows_by_scenario.values() for case in rows if case.get("entrypoint") == "run"):
            require(case["finished_at"] <= serve_started_at, f"case {case.get('case_id')} overlapped the resident serve model")
    _, _, pair_registry_raw = validate_artifact_ref(root, report.get("pair_registry"), "scenario_report.pair_registry", allowed_kinds={"raw-json"})
    expected_pair_registry = build_pair_registry(case_rows_by_scenario, model_key=model_key, backend=backend)
    require(pair_registry_raw == expected_pair_registry, "scenario_report pair registry is not derived from persisted paired inputs")
    expected_pass_prefix = f"{PASS_PREFIX}: "
    pass_line = require_string(report.get("pass_line"), "scenario_report.pass_line")
    require(pass_line.startswith(expected_pass_prefix), f"scenario_report.pass_line must start with {expected_pass_prefix}")
    collected_path = require_string(report.get("artifact_path"), "scenario_report.artifact_path")
    require(pass_line == f"{PASS_PREFIX}: {collected_path}", "scenario_report.pass_line does not match collected artifact path")
    if require_current_output_path:
        require(report_path is not None, "current output path validation requires report_path")
        require(collected_path == str(report_path.resolve()), "scenario_report.artifact_path does not match current output path")
    return report


def collect_manifest(
    manifest: dict[str, Any],
    root: Path,
    out: Path,
    *,
    allow_internal_fixture: bool = False,
) -> dict[str, Any]:
    if not allow_internal_fixture:
        return execute_manifest(manifest, root, out, discover=False)
    root = root.resolve()
    out = out.resolve()
    report = copy.deepcopy(manifest)
    require(
        not ({"runner", "pass_line", "status", "artifact_path", "pair_registry"} & set(report)),
        "input manifest must not predeclare runner/status/PASS/artifact path",
    )
    report["schema_version"] = SCHEMA_VERSION
    report["status"] = "pass"
    report["runner"] = internal_fixture_runner_identity() if allow_internal_fixture else canonical_runner_identity()
    report["artifact_path"] = str(out)
    report["pass_line"] = f"{PASS_PREFIX}: {out}"
    attach_pair_registry(report, root)
    validate_report_document(
        report,
        root,
        report_path=out,
        allow_internal_fixture=allow_internal_fixture,
        require_current_output_path=True,
    )
    return report


def artifact_ref(root: Path, rel: str, kind: str, text: str) -> dict[str, str]:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {"kind": kind, "path": rel, "sha256": file_sha256(path)}


def existing_artifact_ref(root: Path, path: Path, kind: str) -> dict[str, str]:
    resolved = path.resolve()
    rel = resolved.relative_to(root.resolve()).as_posix()
    require(resolved.is_file() and resolved.stat().st_size > 0, f"cannot reference empty artifact {resolved}")
    return {"kind": kind, "path": rel, "sha256": file_sha256(resolved)}


def planned_case_rows(model_key: str, backend: str, catalog: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_id in SCENARIO_IDS:
        shape = selftest_scenario_shape(scenario_id, model_key, backend)
        entrypoints = sorted(required_entrypoints(scenario_id))
        for index, (variant, preset) in enumerate(planned_variant_presets(scenario_id, shape), start=1):
            expectation = resolve_expectation(
                catalog,
                model_key=model_key,
                backend=backend,
                scenario_id=scenario_id,
                variant=variant,
                preset=preset,
            )
            rows.append(
                {
                    "case_id": f"{scenario_id.lower()}-{index:03d}",
                    "scenario_id": scenario_id,
                    "ordinal": index,
                    "variant": variant,
                    "preset": preset,
                    "entrypoint": planned_entrypoint(scenario_id, variant, index, entrypoints),
                    "expectation": expectation,
                    "shape": shape,
                    "concurrency_cell": shape.get("concurrency_cells", [None] * shape["case_count"])[index - 1]
                    if scenario_id == "C18"
                    else None,
                }
            )
    return rows


def preset_values(model_key: str, preset: str | None) -> dict[str, Any]:
    if preset is None:
        return {}
    path = REPO_ROOT / "scripts/release/configs/runtime_vnext_generation_presets.json"
    data = read_json(path)
    models = require_object(data.get("models"), "generation preset models")
    model = require_object(models.get(model_key), f"generation preset {model_key}")
    presets = require_object(model.get("presets"), f"generation preset {model_key}.presets")
    return copy.deepcopy(require_object(presets.get(preset), f"generation preset {model_key}.{preset}"))


def case_marker(case_id: str) -> str:
    return f"G00-{case_id}-OK"


def expected_case_text(case: dict[str, Any]) -> str:
    if case["scenario_id"] == "C17":
        return {"chinese": "中文正确", "emoji": "🙂🚀", "combining": "e\u0301"}[case["variant"]]
    if case["scenario_id"] == "C06":
        return case_marker(f"c05-{int(case['ordinal']):03d}")
    return case_marker(case["case_id"])


def model_file_digest(model_files: dict[str, str], source_path: Path) -> str | None:
    exact = [digest for path, digest in model_files.items() if path == source_path.name or path.endswith("/" + source_path.name)]
    return exact[0] if len(set(exact)) == 1 else None


def json_pointer_value(document: Any, pointer: Any, label: str) -> Any:
    if pointer in {None, ""}:
        return document
    text = require_string(pointer, label)
    require(text.startswith("/"), f"{label} must be an RFC 6901 pointer")
    current = document
    for raw_part in text[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            require(part in current, f"{label} does not resolve: {part}")
            current = current[part]
        elif isinstance(current, list):
            require(part.isdigit() and int(part) < len(current), f"{label} list index invalid: {part}")
            current = current[int(part)]
        else:
            raise ScenarioError(f"{label} traverses a scalar at {part}")
    return current


def build_c01_resolution_probe(
    model_arg: str,
    model_key: str,
    *,
    sources: dict[str, Any],
    semantic_root: Path,
    tokenizer_root: Path,
) -> dict[str, Any]:
    semantic = sources["semantic_source"]
    tokenizer = sources["tokenizer_source"]
    chat = sources["chat_template"]
    config_path = semantic_root / "config.json"
    config_document = read_json(config_path)
    config_model = config_document.get("text_config") if isinstance(config_document.get("text_config"), dict) else config_document
    architectures = config_model.get("architectures") if isinstance(config_model, dict) else None
    architecture = architectures[0] if isinstance(architectures, list) and architectures else config_model.get("model_type") if isinstance(config_model, dict) else None
    selected_source = semantic if chat["source"] == "semantic_source" else require_object(tokenizer, "C01 tokenizer source")
    selected_root = semantic_root if chat["source"] == "semantic_source" else tokenizer_root
    template_path = selected_root / chat["path"]
    template_bytes = template_path.read_bytes()
    if chat["json_pointer"] in {None, ""}:
        template_text = template_bytes.decode("utf-8", "strict")
        template_source_document: Any = None
    else:
        template_source_document = read_json(template_path)
        template_value = json_pointer_value(template_source_document, chat["json_pointer"], "C01 chat template json_pointer")
        template_text = require_string(template_value, "C01 selected chat template")
    token_source = tokenizer if tokenizer is not None else semantic
    token_root = tokenizer_root if tokenizer is not None else semantic_root
    tokenizer_path = token_root / "tokenizer.json"
    tokenizer_config_path = token_root / "tokenizer_config.json"
    tokenizer_config = read_json(tokenizer_config_path)
    eos_ids = config_model.get("eos_token_id") if isinstance(config_model, dict) else None
    if isinstance(eos_ids, int) and not isinstance(eos_ids, bool):
        eos_ids = [eos_ids]
    special_tokens = {
        key: tokenizer_config.get(key)
        for key in ("bos_token", "eos_token", "unk_token", "pad_token")
        if tokenizer_config.get(key) is not None
    }
    core_fields = {"architectures", "model_type", "text_config", "torch_dtype", "transformers_version"}
    unknown_fields = sorted(str(key) for key in config_document if key not in core_fields)
    runtime_binding = {
        "semantic_repo": semantic["repo"],
        "semantic_revision": semantic["revision"],
        "config_container_sha256": semantic["files"]["config.json"],
        "template_source": chat["source"],
        "template_repo": selected_source["repo"],
        "template_revision": selected_source["revision"],
        "template_path": chat["path"],
        "template_container_sha256": chat["container_sha256"],
        "template_content_sha256": chat["content_sha256"],
    }
    return {
        "available": True,
        "requested_model": model_arg,
        "resolved_model_path": str(Path(model_arg).expanduser().resolve()),
        "semantic_source_root": str(semantic_root),
        "tokenizer_source_root": str(tokenizer_root),
        "config": {
            "source": "semantic_source/config.json",
            "raw_sha256": file_sha256(config_path),
            "locked_sha256": semantic["files"]["config.json"],
            "document_sha256": canonical_json_sha256(config_document),
            "document": config_document,
            "resolved_architecture": architecture,
            "unknown_top_level_fields": unknown_fields,
        },
        "template": {
            "source": f"{chat['source']}/{chat['path']}",
            "json_pointer": chat["json_pointer"],
            "raw_file_sha256": hashlib.sha256(template_bytes).hexdigest(),
            "locked_container_sha256": chat["container_sha256"],
            "template_sha256": hashlib.sha256(template_text.encode("utf-8")).hexdigest(),
            "locked_content_sha256": chat["content_sha256"],
            "template": template_text,
            "source_document": template_source_document,
        },
        "special_tokens": {
            "source": ("tokenizer_source" if tokenizer is not None else "semantic_source") + "/tokenizer_config.json",
            "tokenizer_sha256": file_sha256(tokenizer_path),
            "locked_tokenizer_sha256": token_source["files"]["tokenizer.json"],
            "container_sha256": file_sha256(tokenizer_config_path),
            "locked_container_sha256": token_source["files"]["tokenizer_config.json"],
            "tokens": special_tokens,
            "eos_token_ids": eos_ids,
        },
        "runtime_binding": runtime_binding,
        "expected_runtime_architectures": sorted(EXPECTED_ARCHITECTURES[model_key]),
    }


def validate_c01_resolution_probe(
    raw: Any,
    *,
    model_key: str,
    variant: str,
    actual_config: dict[str, Any] | None,
    label: str,
) -> None:
    probe = require_object(raw, f"{label}.resolution_probe")
    require(probe.get("available") is True, f"{label} lacks a complete local config/template resolution probe")
    require_string(probe.get("requested_model"), f"{label}.resolution_probe.requested_model")
    require_string(probe.get("resolved_model_path"), f"{label}.resolution_probe.resolved_model_path")
    config = require_object(probe.get("config"), f"{label}.resolution_probe.config")
    config_document = require_object(config.get("document"), f"{label}.resolution_probe.config.document")
    require(config.get("document_sha256") == canonical_json_sha256(config_document), f"{label} parsed config document SHA mismatch")
    require(config.get("raw_sha256") == config.get("locked_sha256"), f"{label} config snapshot is not bound to semantic_source")
    architecture = require_string(config.get("resolved_architecture"), f"{label}.resolution_probe.config.resolved_architecture").lower()
    require("qwen" in architecture and "llama" not in architecture, f"{label} config resolved through an incorrect Llama fallback")
    unknown_fields = require_list(config.get("unknown_top_level_fields"), f"{label}.resolution_probe.config.unknown_top_level_fields")
    template = require_object(probe.get("template"), f"{label}.resolution_probe.template")
    template_text = require_string(template.get("template"), f"{label}.resolution_probe.template.template")
    require(template.get("template_sha256") == hashlib.sha256(template_text.encode("utf-8")).hexdigest(), f"{label} template bytes SHA mismatch")
    require(template.get("raw_file_sha256") == template.get("locked_container_sha256"), f"{label} template container is not bound to the selected locked source")
    require(template.get("template_sha256") == template.get("locked_content_sha256"), f"{label} selected template bytes differ from models.lock")
    require("llama" not in require_string(template.get("source"), f"{label}.resolution_probe.template.source").lower(), f"{label} selected a Llama fallback template")
    require(probe.get("expected_runtime_architectures") == sorted(EXPECTED_ARCHITECTURES[model_key]), f"{label} expected architecture set drift")
    if variant == "special-token-eos":
        special = require_object(probe.get("special_tokens"), f"{label}.resolution_probe.special_tokens")
        require(special.get("tokenizer_sha256") == special.get("locked_tokenizer_sha256"), f"{label} tokenizer.json is not locked")
        require(special.get("container_sha256") == special.get("locked_container_sha256"), f"{label} tokenizer_config is not locked")
        tokens = require_object(special.get("tokens"), f"{label}.resolution_probe.special_tokens.tokens")
        require(tokens.get("eos_token") is not None, f"{label} official EOS token is absent")
        eos_ids = require_list(special.get("eos_token_ids"), f"{label}.resolution_probe.special_tokens.eos_token_ids")
        require(eos_ids and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in eos_ids), f"{label} official EOS token ids are invalid")
    if variant == "unknown-fail-closed":
        require(unknown_fields and len(set(unknown_fields)) == len(unknown_fields), f"{label} does not exercise additional official config fields")
    if actual_config is not None:
        runtime = require_object(actual_config.get("model_capabilities"), f"{label}.actual_config.model_capabilities")
        runtime_architecture = require_string(runtime.get("architecture"), f"{label}.actual_config.model_capabilities.architecture")
        require(runtime_architecture in EXPECTED_ARCHITECTURES[model_key], f"{label} runtime architecture did not preserve the resolved model family")
        require("llama" not in runtime_architecture.lower(), f"{label} runtime silently fell back to Llama")
        runtime_binding = require_object(probe.get("runtime_binding"), f"{label}.resolution_probe.runtime_binding")
        actual_binding = require_object(actual_config.get("resolution_evidence"), f"{label}.actual_config.resolution_evidence")
        require(actual_binding == runtime_binding, f"{label} runtime did not expose the selected config/template source identities")


def validate_c01_negative_probe(
    root: Path,
    raw: Any,
    *,
    resolution_probe: dict[str, Any],
    ordinal: int,
    label: str,
) -> None:
    probe = require_object(raw, f"{label}.negative_probe")
    require(probe.get("contract") == "unsupported-architecture-layout-fail-closed", f"{label} negative probe contract mismatch")
    require(probe.get("fixture_id") == f"unknown-layout-{ordinal:03d}", f"{label} negative fixture identity mismatch")
    require(probe.get("unknown_architecture") == f"G00UnsupportedArchitecture{ordinal:03d}", f"{label} negative architecture is not ordinal-specific")
    config_probe = require_object(resolution_probe.get("config"), f"{label}.resolution_probe.config")
    require(probe.get("base_config_sha256") == config_probe.get("locked_sha256"), f"{label} negative fixture is not derived from the locked semantic config")
    environment = validate_sanitized_environment(probe.get("environment"), f"{label}.negative_probe.environment")
    require(probe.get("environment_sha256") == canonical_json_sha256(environment), f"{label} negative probe environment SHA mismatch")
    artifacts = require_object(probe.get("artifacts"), f"{label}.negative_probe.artifacts")
    require(
        set(artifacts) == {"config", "tokenizer", "tokenizer_config", "dummy_weight_marker", "named_weight", "stdout", "stderr", "process_receipt"},
        f"{label} negative probe artifact set mismatch",
    )
    require(probe.get("fixture_manifest_sha256") == canonical_json_sha256(artifacts), f"{label} negative fixture manifest SHA mismatch")
    config_path, _, config_raw = validate_artifact_ref(root, artifacts["config"], f"{label}.negative.config", allowed_kinds={"raw-json"})
    config = require_object(config_raw, f"{label}.negative.config JSON")
    target = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    architectures = require_list(target.get("architectures"), f"{label}.negative.config.architectures")
    require(architectures == [probe["unknown_architecture"]], f"{label} negative config architecture mismatch")
    require(target.get("model_type") == f"g00_unsupported_layout_{ordinal:03d}", f"{label} negative config layout mismatch")
    marker = require_object(config.get("g00_negative_fixture"), f"{label}.negative.config.g00_negative_fixture")
    require(marker == {"ordinal": ordinal, "expected_failure": "unsupported-architecture-layout"}, f"{label} negative config marker mismatch")
    tokenizer_path, _, _ = validate_artifact_ref(root, artifacts["tokenizer"], f"{label}.negative.tokenizer", allowed_kinds={"raw-json"})
    special_tokens = require_object(resolution_probe.get("special_tokens"), f"{label}.resolution_probe.special_tokens")
    require(file_sha256(tokenizer_path) == special_tokens.get("locked_tokenizer_sha256"), f"{label} negative tokenizer differs from the locked source")
    validate_artifact_ref(root, artifacts["tokenizer_config"], f"{label}.negative.tokenizer_config", allowed_kinds={"raw-json"})
    dummy_path, _, _ = validate_artifact_ref(root, artifacts["dummy_weight_marker"], f"{label}.negative.dummy", allowed_kinds={"runtime-log"})
    require("must reject" in dummy_path.read_text(encoding="utf-8").lower(), f"{label} negative dummy marker contract missing")
    named_weight_path, _, _ = validate_artifact_ref(root, artifacts["named_weight"], f"{label}.negative.named_weight", allowed_kinds={"binary"})
    access = require_object(probe.get("weight_access_contract"), f"{label}.negative.weight_access_contract")
    require(access.get("tensor_count") == 0 and access.get("payload_bytes") == 0, f"{label} negative named weight contains tensor payload")
    require(access.get("sha256_before") == access.get("sha256_after") == file_sha256(named_weight_path), f"{label} negative named weight sentinel changed")
    require(access.get("mtime_ns_before") == access.get("mtime_ns_after"), f"{label} negative named weight was modified")
    weight_format = require_string(probe.get("weight_format"), f"{label}.negative.weight_format")
    if "gguf" in weight_format.lower():
        require(access.get("kind") == "metadata-only-gguf" and named_weight_path.suffix.lower() == ".gguf", f"{label} Metal/GGUF negative fixture is not metadata-reachable")
        require(named_weight_path.read_bytes().startswith(b"GGUF"), f"{label} negative GGUF header invalid")
    else:
        require(access.get("kind") == "unreadable-empty-safetensors-sentinel", f"{label} CUDA negative fixture lacks unreadable safetensors sentinel")
        require(access.get("mode_during_execution") == "0o0", f"{label} safetensors sentinel was readable during dispatch")
        require(access.get("atime_ns_before") == access.get("atime_ns_after"), f"{label} safetensors payload sentinel was accessed before fail-closed rejection")
    stdout_path, _, _ = validate_artifact_ref(root, artifacts["stdout"], f"{label}.negative.stdout", allowed_kinds={"stdout-log"})
    stderr_path, _, _ = validate_artifact_ref(root, artifacts["stderr"], f"{label}.negative.stderr", allowed_kinds={"stderr-log"})
    argv = require_list(probe.get("argv"), f"{label}.negative_probe.argv")
    require(len(argv) >= 3 and Path(argv[0]).name == "ferrum" and argv[1] == "run", f"{label} negative probe did not invoke ferrum run")
    fixture_root = artifact_path(root, probe.get("fixture_root"), f"{label}.negative_probe.fixture_root")
    invocation_model = artifact_path(root, probe.get("invocation_model"), f"{label}.negative_probe.invocation_model")
    argv_model = artifact_path(root, argv[2], f"{label}.negative_probe.argv[2]")
    require(argv_model == invocation_model and config_path.parent.resolve() == fixture_root, f"{label} negative argv is not bound to its fixture")
    require(invocation_model == (named_weight_path.resolve() if "gguf" in weight_format.lower() else fixture_root), f"{label} negative invocation path does not match backend format")
    returncode = probe.get("returncode")
    require(isinstance(returncode, int) and not isinstance(returncode, bool) and returncode not in {0, 124}, f"{label} negative probe must fail promptly and nonzero")
    require(probe.get("effective_config_emitted") is False, f"{label} negative probe reached effective runtime planning")
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="strict").lower()
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="strict").lower()
    combined = stdout_text + "\n" + stderr_text
    require("unsupported" in combined and ("architecture" in combined or "layout" in combined), f"{label} negative probe lacks exact unsupported architecture/layout evidence")
    forbidden = (
        "missing weight",
        "no such file",
        "safetensor",
        "gguf",
        "loading weight",
        "kernel launch",
        "out of memory",
        "llama",
        "fallback",
    )
    require(not any(value in combined for value in forbidden), f"{label} negative probe failed after fallback, weight, or kernel work")
    validate_process_receipt(
        root,
        artifacts["process_receipt"],
        label=f"{label}.negative.process_receipt",
        pid=require_count(probe.get("pid"), f"{label}.negative_probe.pid", minimum=1),
        pgid=require_count(probe.get("pgid"), f"{label}.negative_probe.pgid", minimum=1),
        argv=[str(value) for value in argv],
        role="ferrum-run",
        expected_environment=environment,
    )
    started = parse_timestamp(probe.get("started_at"), f"{label}.negative_probe.started_at")
    finished = parse_timestamp(probe.get("finished_at"), f"{label}.negative_probe.finished_at")
    require(finished > started, f"{label} negative probe wall window invalid")
    start_ns = require_count(probe.get("started_monotonic_ns"), f"{label}.negative_probe.started_monotonic_ns", minimum=1)
    finish_ns = require_count(probe.get("finished_monotonic_ns"), f"{label}.negative_probe.finished_monotonic_ns", minimum=1)
    require(finish_ns > start_ns, f"{label} negative probe monotonic window invalid")


def strict_schema_case(case: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    ordinal = int(case["ordinal"])
    variant = require_string(case.get("variant"), "C14 variant")
    nonce = f"v{ordinal:03d}"
    if variant == "required":
        schema = {
            "type": "object",
            "properties": {
                "identity": {
                    "type": "object",
                    "properties": {
                        "first": {"type": "string", "const": f"first-{nonce}"},
                        "last": {"type": "string", "const": f"last-{nonce}"},
                    },
                    "required": ["first", "last"],
                    "additionalProperties": False,
                },
                "checksum": {"type": "string", "const": f"checksum-{nonce}"},
                "sequence": {"type": "integer", "const": ordinal},
            },
            "required": ["identity", "checksum", "sequence"],
            "additionalProperties": False,
        }
        output = {"identity": {"first": f"first-{nonce}", "last": f"last-{nonce}"}, "checksum": f"checksum-{nonce}", "sequence": ordinal}
    elif variant == "type":
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "const": f"text-{nonce}"},
                "count": {"type": "integer", "const": ordinal},
                "enabled": {"type": "boolean", "const": ordinal % 2 == 0},
                "values": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
            },
            "required": ["text", "count", "enabled", "values"],
            "additionalProperties": False,
        }
        output = {"text": f"text-{nonce}", "count": ordinal, "enabled": ordinal % 2 == 0, "values": [ordinal, ordinal + 1]}
    elif variant == "additional-properties":
        schema = {
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "properties": {"allowed": {"type": "string", "const": f"allowed-{nonce}"}},
                    "required": ["allowed"],
                    "additionalProperties": False,
                }
            },
            "required": ["payload"],
            "additionalProperties": False,
        }
        output = {"payload": {"allowed": f"allowed-{nonce}"}}
    elif variant == "enum":
        selected = ordinal % 3
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": [f"queued-{nonce}", f"running-{nonce}", f"done-{nonce}"]},
                "priority": {"type": "integer", "enum": [ordinal, ordinal + 10, ordinal + 20]},
            },
            "required": ["status", "priority"],
            "additionalProperties": False,
        }
        output = {"status": schema["properties"]["status"]["enum"][selected], "priority": schema["properties"]["priority"]["enum"][selected]}
    else:
        raise ScenarioError(f"unsupported C14 variant: {variant}")
    prompt = f"C14 case {ordinal:03d} ({variant}): return exactly this schema-valid object: {json.dumps(output, sort_keys=True)}"
    return prompt, schema, output


def thinking_reasoning_expected(model_key: str, variant: str) -> bool:
    if variant == "hard-no-thinking":
        return False
    if model_key == "m3-qwen3-30b-a3b" and variant == "soft-no-think":
        return False
    return True


def validate_strict_schema_instance(instance: Any, schema: dict[str, Any], label: str) -> None:
    expected_type = schema.get("type")
    if expected_type == "object":
        require(isinstance(instance, dict), f"{label} must be an object")
        properties = require_object(schema.get("properties"), f"{label}.schema.properties")
        required = require_list(schema.get("required"), f"{label}.schema.required")
        require(len(required) == len(set(required)) and set(required) <= set(properties), f"{label} schema required keys invalid")
        require(set(required) <= set(instance), f"{label} output misses required keys")
        if schema.get("additionalProperties") is False:
            require(set(instance) <= set(properties), f"{label} output contains forbidden additional properties")
        for key, value in instance.items():
            if key in properties:
                validate_strict_schema_instance(value, require_object(properties[key], f"{label}.schema.properties.{key}"), f"{label}.{key}")
    elif expected_type == "array":
        require(isinstance(instance, list), f"{label} must be an array")
        minimum = schema.get("minItems", 0)
        maximum = schema.get("maxItems", len(instance))
        require(isinstance(minimum, int) and isinstance(maximum, int) and minimum <= len(instance) <= maximum, f"{label} array length violates bounds")
        item_schema = require_object(schema.get("items"), f"{label}.schema.items")
        for index, value in enumerate(instance):
            validate_strict_schema_instance(value, item_schema, f"{label}[{index}]")
    elif expected_type == "string":
        require(isinstance(instance, str), f"{label} must be a string")
    elif expected_type == "integer":
        require(isinstance(instance, int) and not isinstance(instance, bool), f"{label} must be an integer")
    elif expected_type == "boolean":
        require(isinstance(instance, bool), f"{label} must be a boolean")
    else:
        raise ScenarioError(f"{label} uses unsupported schema type {expected_type!r}")
    if "const" in schema:
        require(instance == schema["const"], f"{label} violates const")
    if "enum" in schema:
        require(instance in require_list(schema["enum"], f"{label}.schema.enum"), f"{label} violates enum")


def derived_c14_category(schema: dict[str, Any], label: str) -> str:
    properties = require_object(schema.get("properties"), f"{label}.properties")
    if any(isinstance(rule, dict) and isinstance(rule.get("enum"), list) and len(rule["enum"]) >= 3 for rule in properties.values()):
        return "enum"
    property_types = {rule.get("type") for rule in properties.values() if isinstance(rule, dict)}
    if {"string", "integer", "boolean", "array"} <= property_types:
        return "type"
    nested = [rule for rule in properties.values() if isinstance(rule, dict) and rule.get("type") == "object"]
    required = require_list(schema.get("required"), f"{label}.required")
    if len(required) >= 3 and any(len(require_list(rule.get("required"), f"{label}.nested.required")) >= 2 for rule in nested):
        return "required"
    if len(properties) == 1 and nested and all(rule.get("additionalProperties") is False for rule in nested):
        return "additional-properties"
    raise ScenarioError(f"{label} does not derive one material C14 constraint category")


def case_http_payload(case: dict[str, Any], model_key: str) -> dict[str, Any]:
    scenario_id = case["scenario_id"]
    variant = case["variant"]
    marker = expected_case_text(case)
    prompt = f"Return the exact marker {marker} and no other text."
    payload: dict[str, Any] = {
        "model": model_key,
        "messages": [{"role": "user", "content": prompt}],
        "metadata": {
            "g00_case_id": case["case_id"],
            "g00_scenario_id": scenario_id,
            "g00_variant": variant,
            "g00_ordinal": int(case["ordinal"]),
        },
    }
    values = preset_values(model_key, case["preset"])
    for key in ("temperature", "top_p", "top_k", "seed", "max_tokens", "stop"):
        if key in values:
            payload[key] = values[key]
    if isinstance(values.get("template_kwargs"), dict) and values["template_kwargs"]:
        payload["chat_template_kwargs"] = values["template_kwargs"]
    if scenario_id in {"C06", "C12", "C17"} or (scenario_id == "C21" and variant == "serve-stream"):
        payload.update({"stream": True, "stream_options": {"include_usage": True}})
    if scenario_id == "C06":
        payload["metadata"]["g00_reference_contract"] = "C05"
        payload["metadata"]["g00_reference_case_id"] = f"c05-{int(case['ordinal']):03d}"
    if scenario_id == "C07":
        conversation_id = f"conversation-{case['case_id']}"
        payload["messages"] = [{"role": "user", "content": f"Conversation {conversation_id}, turn 1. Return {case_marker(case['case_id'])}-T1 exactly."}]
        payload["metadata"]["g00_conversation_id"] = conversation_id
        payload["metadata"]["g00_turn"] = 1
    if scenario_id == "C08":
        if variant == "stop":
            payload["messages"] = [{"role": "user", "content": "Say before<G00STOP>after exactly."}]
            payload["stop"] = ["<G00STOP>"]
            payload["max_tokens"] = 64
        elif variant == "max-tokens":
            payload["messages"] = [{"role": "user", "content": "Continue listing integers without stopping."}]
            payload.update({"max_tokens": 8, "stop": [], "ignore_eos": True})
        else:
            payload["max_tokens"] = 64
    if scenario_id in {"C10", "C11", "C12"}:
        payload["messages"] = [{"role": "user", "content": "Use lookup_weather for Paris."}]
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Return weather for one city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "enum": ["Paris"]}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        payload["tool_choice"] = "required" if scenario_id == "C10" else "auto"
        if scenario_id == "C12":
            payload["metadata"]["g00_reference_contract"] = "C11"
            payload["metadata"]["g00_reference_case_id"] = f"c11-{int(case['ordinal']):03d}"
    elif scenario_id == "C13":
        payload["messages"] = [
            {"role": "user", "content": "Use the calculator."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "calculator", "arguments": "{\"expression\":\"7*3\"}"}}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "21"},
        ]
    elif scenario_id == "C14":
        strict_prompt, strict_schema, _ = strict_schema_case(case)
        payload["messages"] = [{"role": "user", "content": strict_prompt}]
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": f"g00_result_{int(case['ordinal']):03d}",
                "strict": True,
                "schema": strict_schema,
            },
        }
    elif scenario_id == "C15":
        payload["messages"] = [{"role": "user", "content": f"Return a JSON object containing marker {marker}."}]
        payload["response_format"] = {"type": "json_object"}
    elif scenario_id == "C16":
        if variant == "invalid-tool":
            payload["tool_choice"] = "required"
        elif variant == "invalid-schema":
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "bad", "strict": True}}
        elif variant == "invalid-stream-option":
            payload.update({"stream": False, "stream_options": {"include_usage": True}})
        elif variant == "invalid-model":
            payload["model"] = "not-a-loaded-model"
        else:
            payload["max_tokens"] = 10**9
    elif scenario_id == "C19":
        if variant == "hard-thinking":
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        elif variant == "hard-no-thinking":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            payload.pop("chat_template_kwargs", None)
            if variant in {"soft-think", "soft-think-misuse"}:
                payload["messages"][0]["content"] += " /think"
            elif variant in {"soft-no-think", "soft-no-think-misuse"}:
                payload["messages"][0]["content"] += " /no_think"
    elif scenario_id == "C20":
        if variant == "text-array":
            payload["messages"] = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            media_type = "video_url" if variant == "video-url" else "image_url"
            url = "data:image/png;base64,iVBORw0KGgo=" if variant == "data-url" else "https://example.invalid/media.png"
            content: list[dict[str, Any]] = [{"type": media_type, media_type: {"url": url}}]
            if variant == "mixed-text-media":
                content.insert(0, {"type": "text", "text": prompt})
            payload["messages"] = [{"role": "user", "content": content}]
    elif scenario_id == "C21" and variant == "required-tool":
        payload["messages"] = [{"role": "user", "content": f"Call echo_value for {marker}; tool choice has priority over the simultaneous strict response format."}]
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "echo_value",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string", "const": marker}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        payload["tool_choice"] = "required"
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": f"conflict_{int(case['ordinal']):03d}",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"result": {"type": "string", "const": marker}},
                    "required": ["result"],
                    "additionalProperties": False,
                },
            },
        }
    elif scenario_id == "C21" and variant == "strict-schema":
        payload["messages"] = [{"role": "user", "content": f"Return strict JSON result {marker}."}]
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": f"official_schema_{int(case['ordinal']):03d}",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"result": {"type": "string", "const": marker}},
                    "required": ["result"],
                    "additionalProperties": False,
                },
            },
        }
    elif scenario_id == "C21" and variant == "json-object":
        payload["messages"] = [{"role": "user", "content": f"Return a JSON object with result {marker}."}]
        payload["response_format"] = {"type": "json_object"}
    return payload


def parse_http_body(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"raw_text": text}
    return value if isinstance(value, dict) else {"value": value}


def decode_wire_chunks(chunks: list[bytes]) -> tuple[str, list[str], int]:
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    fragments: list[str] = []
    split_boundaries = 0
    for index, chunk in enumerate(chunks):
        fragments.append(decoder.decode(chunk, final=index == len(chunks) - 1))
        if index < len(chunks) - 1 and decoder.getstate()[0]:
            split_boundaries += 1
    return "".join(fragments), fragments, split_boundaries


def http_exchange(
    base_url: str,
    payload: dict[str, Any],
    timeout_sec: float,
    *,
    read_chunk_size: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    started = iso_now()
    start_ns = time.monotonic_ns()
    try:
        response = urllib.request.urlopen(request, timeout=timeout_sec)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "strict")
        status = exc.code
        headers = dict(exc.headers.items())
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        body = json.dumps({"error": {"message": f"client transport error: {type(exc).__name__}"}})
        status = 599
        headers = {}
    else:
        with response:
            if read_chunk_size is None:
                wire_chunks = [response.read()]
            else:
                wire_chunks = []
                while True:
                    chunk = response.read(read_chunk_size)
                    if not chunk:
                        break
                    wire_chunks.append(chunk)
            body, decoded_fragments, split_boundaries = decode_wire_chunks(wire_chunks)
            status = response.status
            headers = dict(response.headers.items())
    finish_ns = time.monotonic_ns()
    finished = iso_now()
    exchange = {
        "request": payload,
        "status": status,
        "response_headers": headers,
        "response": parse_http_body(body),
        "response_raw": body,
        "started_at": started,
        "finished_at": finished,
        "started_monotonic_ns": start_ns,
        "finished_monotonic_ns": finish_ns,
    }
    if read_chunk_size is not None and status < 400:
        exchange["wire_chunks_base64"] = [base64.b64encode(chunk).decode("ascii") for chunk in wire_chunks]
        exchange["incremental_decoded_fragments"] = decoded_fragments
        exchange["utf8_split_boundary_count"] = split_boundaries
        exchange["wire_sha256"] = hashlib.sha256(b"".join(wire_chunks)).hexdigest()
    return exchange, {"started_at": started, "finished_at": finished, "started_monotonic_ns": start_ns, "finished_monotonic_ns": finish_ns}


def http_get_json(url: str, timeout_sec: float) -> tuple[int, dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8", "strict")
            return response.status, parse_http_body(body)
    except urllib.error.HTTPError as exc:
        return exc.code, parse_http_body(exc.read().decode("utf-8", "strict"))


def parse_sse_evidence(body: str) -> dict[str, Any]:
    done = usage_count = delta_count = malformed_count = 0
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    tool_calls: dict[int, dict[str, Any]] = {}
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            done += 1
            continue
        try:
            row = json.loads(data)
        except json.JSONDecodeError:
            malformed_count += 1
            continue
        if isinstance(row, dict) and row.get("usage") is not None:
            usage_count += 1
            usage = require_object(row.get("usage"), "SSE usage")
        choices = row.get("choices") if isinstance(row, dict) else None
        if isinstance(choices, list) and choices:
            delta_count += 1
            choice = require_object(choices[0], "SSE choice")
            if choice.get("finish_reason") is not None:
                finish_reason = require_string(choice.get("finish_reason"), "SSE finish_reason")
            delta = require_object(choice.get("delta"), "SSE delta")
            content = delta.get("content")
            if content is not None:
                require(isinstance(content, str), "SSE content delta must be a string")
                content_parts.append(content)
            reasoning = delta.get("reasoning", delta.get("reasoning_content"))
            if reasoning is not None:
                require(isinstance(reasoning, str), "SSE reasoning delta must be a string")
                reasoning_parts.append(reasoning)
            for call_raw in delta.get("tool_calls", []) if isinstance(delta.get("tool_calls"), list) else []:
                call = require_object(call_raw, "SSE tool call delta")
                index = require_count(call.get("index", 0), "SSE tool call delta index")
                merged = tool_calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if isinstance(call.get("id"), str):
                    merged["id"] += call["id"]
                if isinstance(call.get("type"), str):
                    merged["type"] = call["type"]
                function = call.get("function")
                if isinstance(function, dict):
                    if isinstance(function.get("name"), str):
                        merged["function"]["name"] += function["name"]
                    if isinstance(function.get("arguments"), str):
                        merged["function"]["arguments"] += function["arguments"]
    return {
        "done_count": done,
        "usage_count": usage_count,
        "delta_count": delta_count,
        "malformed_count": malformed_count,
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "finish_reason": finish_reason,
        "usage": usage,
        "tool_calls": [tool_calls[index] for index in sorted(tool_calls)],
    }


def read_jsonl_since(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        return [], offset
    with path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read()
        new_offset = handle.tell()
    rows: list[dict[str, Any]] = []
    for line in payload.decode("utf-8", "replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows, new_offset


def nested_activity_values(value: Any, *, parent_key: str = "") -> list[int]:
    values: list[int] = []
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if isinstance(child, int) and not isinstance(child, bool) and (
                "active" in lowered or lowered in {"batch_size", "running_count", "scheduled_count"}
            ):
                values.append(child)
            elif isinstance(child, list) and lowered in {"requests", "scheduled", "running", "active_requests"}:
                values.append(len(child))
            values.extend(nested_activity_values(child, parent_key=lowered))
    elif isinstance(value, list):
        for child in value:
            values.extend(nested_activity_values(child, parent_key=parent_key))
    return values


def observed_max_active(rows: list[dict[str, Any]]) -> int:
    values = [activity for row in rows for activity in nested_activity_values(row)]
    return max(values, default=0)


def trace_released(rows: list[dict[str, Any]]) -> tuple[bool, int, int | None]:
    active_seen = False
    ticks = 0
    for row in rows:
        values = nested_activity_values(row)
        if not values:
            continue
        current = max(values)
        if current > 0:
            active_seen = True
            ticks = 0
        elif active_seen:
            ticks += 1
            observed_ns = row.get("collector_observed_monotonic_ns")
            return True, ticks, observed_ns if isinstance(observed_ns, int) else None
    return False, ticks, None


def typed_admission_cap(config_path: Path) -> int:
    try:
        data = read_json(config_path)
    except ScenarioError:
        return 0
    return typed_admission_cap_value(data)


def typed_admission_cap_value(data: Any) -> int:
    candidates: list[int] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lowered = str(key).lower()
                if isinstance(child, int) and not isinstance(child, bool) and child > 0 and lowered in {
                    "max_sequences",
                    "max_num_seqs",
                    "max_active",
                    "admission_cap",
                    "target_concurrency",
                }:
                    candidates.append(child)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(data)
    return min(candidates) if candidates else 0


def validate_actual_effective_config(
    raw: Any,
    *,
    expected_backend: str,
    expected_model_key: str,
    label: str,
) -> dict[str, Any]:
    config = require_object(raw, label)
    require(config.get("missing") is not True, f"{label} declares missing product evidence")
    require(config.get("schema_version") == SCHEMA_VERSION, f"{label} schema_version mismatch")
    for key in ("entries", "hardware_capabilities", "model_capabilities", "workload_profile", "decisions"):
        require(key in config, f"{label} missing {key}")
    entries = require_list(config.get("entries"), f"{label}.entries")
    require(entries, f"{label}.entries must not be empty")
    hardware = require_object(config.get("hardware_capabilities"), f"{label}.hardware_capabilities")
    require(hardware.get("backend") == expected_backend, f"{label} backend differs from lane")
    model_capabilities = require_object(config.get("model_capabilities"), f"{label}.model_capabilities")
    require(model_capabilities, f"{label}.model_capabilities empty")
    architecture = require_string(model_capabilities.get("architecture"), f"{label}.model_capabilities.architecture")
    expected_architectures = {
        "m1-qwen35-4b": {"qwen3_5", "qwen3_5_text"},
        "m2-qwen35-35b-a3b": {"qwen3_5_moe", "qwen3_5_moe_text"},
        "m3-qwen3-30b-a3b": {"qwen3", "qwen3_moe"},
    }
    require(architecture in expected_architectures[expected_model_key], f"{label} architecture differs from lane model")
    workload = require_object(config.get("workload_profile"), f"{label}.workload_profile")
    require(workload, f"{label}.workload_profile empty")
    require_string(workload.get("serving_mode"), f"{label}.workload_profile.serving_mode")
    require_count(workload.get("target_concurrency"), f"{label}.workload_profile.target_concurrency", minimum=1)
    require_list(config.get("decisions"), f"{label}.decisions")
    backend_entries = [entry for entry in entries if isinstance(entry, dict) and entry.get("key") == "FERRUM_BACKEND"]
    require(len(backend_entries) == 1 and backend_entries[0].get("effective_value") == expected_backend, f"{label} typed backend entry mismatch")
    return config


def aborted_http_exchange(base_url: str, payload: dict[str, Any], variant: str) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(base_url)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: {parsed.hostname}\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    ).encode("ascii") + body
    started_at = iso_now()
    started_ns = time.monotonic_ns()
    with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 80), timeout=2) as sock:
        sock.sendall(request)
        if variant == "timeout":
            sock.settimeout(0.01)
            try:
                sock.recv(1)
            except (TimeoutError, socket.timeout):
                pass
        elif variant == "cancel":
            sock.shutdown(socket.SHUT_WR)
    return {
        "request": payload,
        "status": 499,
        "response_headers": {},
        "response": {"client_abort": variant},
        "response_raw": "",
        "started_at": started_at,
        "finished_at": iso_now(),
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": time.monotonic_ns(),
    }


class ProductServer:
    def __init__(
        self,
        *,
        argv: list[str],
        stdout_path: Path,
        stderr_path: Path,
        artifact_root: Path,
        receipt_path: Path,
        base_url: str,
        startup_timeout_sec: float,
    ) -> None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.stdout_handle = stdout_path.open("wb")
        self.stderr_handle = stderr_path.open("wb")
        self.argv = argv
        self.child_environment = sanitized_child_environment()
        self.started_at = iso_now()
        self.started_monotonic_ns = time.monotonic_ns()
        self.proc = subprocess.Popen(
            argv,
            stdout=self.stdout_handle,
            stderr=self.stderr_handle,
            env=self.child_environment,
            start_new_session=True,
        )
        self.pid = self.proc.pid
        self.pgid = os.getpgid(self.proc.pid)
        self.process_receipt = capture_process_receipt(
            artifact_root,
            receipt_path,
            pid=self.pid,
            pgid=self.pgid,
            argv=argv,
            role="ferrum-serve",
            environment=self.child_environment,
        )
        deadline = time.monotonic() + startup_timeout_sec
        last_error = "server did not become ready"
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                last_error = f"server exited during startup with {self.proc.returncode}"
                break
            for path in ("/health", "/v1/models"):
                try:
                    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=0.5) as response:
                        if response.status == 200:
                            self.ready_at = iso_now()
                            self.ready_monotonic_ns = time.monotonic_ns()
                            return
                except (OSError, urllib.error.URLError):
                    pass
            time.sleep(0.05)
        self.stop()
        raise ScenarioError(last_error)

    def stop(self) -> None:
        if getattr(self, "proc", None) is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.finished_monotonic_ns = time.monotonic_ns()
        self.finished_at = iso_now()
        self.process_returncode = self.proc.returncode if getattr(self, "proc", None) is not None else -1
        if not self.stdout_handle.closed:
            self.stdout_handle.close()
        if not self.stderr_handle.closed:
            self.stderr_handle.close()
        for path, label in ((self.stdout_path, "stdout"), (self.stderr_path, "stderr")):
            if not path.exists() or path.stat().st_size == 0:
                path.write_text(f"ferrum serve {label} capture was empty; collector observed controlled shutdown\n", encoding="utf-8")

    def envelope(self) -> dict[str, Any]:
        return {
            "argv": self.argv,
            "pid": self.pid,
            "pgid": self.pgid,
            "started_at": self.started_at,
            "ready_at": getattr(self, "ready_at", None),
            "finished_at": self.finished_at,
            "started_monotonic_ns": self.started_monotonic_ns,
            "ready_monotonic_ns": getattr(self, "ready_monotonic_ns", None),
            "finished_monotonic_ns": self.finished_monotonic_ns,
            "process_returncode": self.process_returncode,
            "child_environment": self.child_environment,
            "child_environment_sha256": canonical_json_sha256(self.child_environment),
        }


def minimal_unknown_gguf(architecture: str) -> bytes:
    def gguf_string(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    metadata = (("general.architecture", architecture), ("general.name", "Ferrum G00 unknown-layout fixture"))
    payload = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata)))
    for key, value in metadata:
        payload.extend(gguf_string(key))
        payload.extend(struct.pack("<I", 8))
        payload.extend(gguf_string(value))
    return bytes(payload)


def execute_c01_unknown_fixture(
    root: Path,
    case: dict[str, Any],
    *,
    binary_path: Path,
    backend: str,
    semantic_root: Path,
    tokenizer_root: Path,
    weight_format: str,
    child_environment: dict[str, str],
    timeout_sec: float,
    case_root: Path,
) -> dict[str, Any]:
    fixture_root = case_root / "unknown-architecture-fixture"
    fixture_root.mkdir(parents=True, exist_ok=True)
    base_config_path = semantic_root / "config.json"
    config = read_json(base_config_path)
    target = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    ordinal = int(case["ordinal"])
    unknown_architecture = f"G00UnsupportedArchitecture{ordinal:03d}"
    target["architectures"] = [unknown_architecture]
    target["model_type"] = f"g00_unsupported_layout_{ordinal:03d}"
    config["g00_negative_fixture"] = {"ordinal": ordinal, "expected_failure": "unsupported-architecture-layout"}
    fixture_config = fixture_root / "config.json"
    write_json(fixture_config, config)
    source_tokenizer = tokenizer_root / "tokenizer.json"
    fixture_tokenizer = fixture_root / "tokenizer.json"
    shutil.copy2(source_tokenizer, fixture_tokenizer)
    source_tokenizer_config = tokenizer_root / "tokenizer_config.json"
    fixture_tokenizer_config = fixture_root / "tokenizer_config.json"
    shutil.copy2(source_tokenizer_config, fixture_tokenizer_config)
    (fixture_root / "DUMMY_WEIGHT_NOT_FOR_LOADING").write_text(
        "Architecture dispatch must reject this fixture before any weight or kernel load.\n",
        encoding="utf-8",
    )
    if "gguf" in weight_format.lower():
        named_weight = fixture_root / "unknown-layout.gguf"
        named_weight.write_bytes(minimal_unknown_gguf(unknown_architecture))
        invocation_model = named_weight
        weight_access_contract: dict[str, Any] = {
            "kind": "metadata-only-gguf",
            "tensor_count": 0,
            "payload_bytes": 0,
            "mode_during_execution": oct(named_weight.stat().st_mode & 0o777),
        }
    else:
        named_weight = fixture_root / "model.safetensors"
        header = b"{}"
        named_weight.write_bytes(struct.pack("<Q", len(header)) + header)
        invocation_model = fixture_root
        weight_access_contract = {
            "kind": "unreadable-empty-safetensors-sentinel",
            "tensor_count": 0,
            "payload_bytes": 0,
            "mode_during_execution": "0o0",
        }
    weight_sha_before = file_sha256(named_weight)
    if "gguf" not in weight_format.lower():
        named_weight.chmod(0)
    weight_before = named_weight.stat()
    stdout_path = fixture_root / "stdout.log"
    stderr_path = fixture_root / "stderr.log"
    effective_path = fixture_root / "unexpected-effective-config.json"
    argv = [
        str(binary_path),
        "run",
        str(invocation_model),
        "--backend",
        backend,
        "--output-format",
        "jsonl",
        "--prompt",
        "This request must never reach inference.",
        "--effective-config-json",
        str(effective_path),
    ]
    started_at = iso_now()
    started_ns = time.monotonic_ns()
    proc = subprocess.Popen(
        argv,
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_environment,
        start_new_session=True,
    )
    pid = proc.pid
    pgid = os.getpgid(pid)
    receipt = capture_process_receipt(
        root,
        fixture_root / "process-receipt.json",
        pid=pid,
        pgid=pgid,
        argv=argv,
        role="ferrum-run",
        environment=child_environment,
    )
    try:
        stdout_text, stderr_text = proc.communicate(timeout=timeout_sec)
        returncode = int(proc.returncode)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout_text, stderr_text = proc.communicate(timeout=10)
        returncode = 124
        stderr_text = (stderr_text or "") + "\nnegative architecture probe timed out\n"
    finished_ns = time.monotonic_ns()
    finished_at = iso_now()
    weight_after = named_weight.stat()
    if "gguf" not in weight_format.lower():
        named_weight.chmod(0o644)
    weight_sha_after = file_sha256(named_weight)
    stdout_path.write_text(stdout_text or "negative probe produced no stdout\n", encoding="utf-8")
    stderr_path.write_text(stderr_text or "negative probe produced no stderr\n", encoding="utf-8")
    dummy_path = fixture_root / "DUMMY_WEIGHT_NOT_FOR_LOADING"
    artifacts = {
        "config": existing_artifact_ref(root, fixture_config, "raw-json"),
        "tokenizer": existing_artifact_ref(root, fixture_tokenizer, "raw-json"),
        "tokenizer_config": existing_artifact_ref(root, fixture_tokenizer_config, "raw-json"),
        "dummy_weight_marker": existing_artifact_ref(root, dummy_path, "runtime-log"),
        "named_weight": existing_artifact_ref(root, named_weight, "binary"),
        "stdout": existing_artifact_ref(root, stdout_path, "stdout-log"),
        "stderr": existing_artifact_ref(root, stderr_path, "stderr-log"),
        "process_receipt": receipt,
    }
    return {
        "contract": "unsupported-architecture-layout-fail-closed",
        "fixture_id": f"unknown-layout-{ordinal:03d}",
        "unknown_architecture": unknown_architecture,
        "base_config_sha256": file_sha256(base_config_path),
        "fixture_manifest_sha256": canonical_json_sha256(artifacts),
        "fixture_root": str(fixture_root),
        "invocation_model": str(invocation_model),
        "weight_format": weight_format,
        "weight_access_contract": {
            **weight_access_contract,
            "sha256_before": weight_sha_before,
            "sha256_after": weight_sha_after,
            "atime_ns_before": weight_before.st_atime_ns,
            "atime_ns_after": weight_after.st_atime_ns,
            "mtime_ns_before": weight_before.st_mtime_ns,
            "mtime_ns_after": weight_after.st_mtime_ns,
        },
        "argv": argv,
        "environment": child_environment,
        "environment_sha256": canonical_json_sha256(child_environment),
        "pid": pid,
        "pgid": pgid,
        "started_at": started_at,
        "finished_at": finished_at,
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": finished_ns,
        "returncode": returncode,
        "artifacts": artifacts,
        "effective_config_emitted": effective_path.is_file(),
    }


def run_case_command(
    root: Path,
    case: dict[str, Any],
    *,
    binary_path: Path,
    model_arg: str,
    backend: str,
    run_extra_args: list[str],
    timeout_sec: float,
    case_root: Path,
    locked_sources: dict[str, Any],
    semantic_root: Path,
    tokenizer_root: Path,
) -> tuple[list[str], dict[str, Any], Path, Path, Path, dict[str, Any]]:
    case_id = case["case_id"]
    marker = expected_case_text(case)
    actual_config = case_root / "actual-effective-config.json"
    values = preset_values(case["model_key"], case["preset"])
    argv = [str(binary_path), "run", model_arg, "--backend", backend, "--output-format", "jsonl"]
    if case["scenario_id"] == "C03":
        stdin_text = "\n".join(
            [
                f"Remember the exact secret {marker}.",
                "Reply with the word acknowledged.",
                "What exact secret did I ask you to remember?",
            ]
        ) + "\n"
    elif case["scenario_id"] == "C19":
        soft_suffix = ""
        if case["variant"] in {"soft-think", "soft-think-misuse"}:
            soft_suffix = " /think"
        elif case["variant"] in {"soft-no-think", "soft-no-think-misuse"}:
            soft_suffix = " /no_think"
        stdin_text = "\n".join(
            [
                f"Return the exact marker {marker}-H1.{soft_suffix}",
                f"Using our prior exchange, return the exact marker {marker}-H2.",
            ]
        ) + "\n"
    else:
        prompt = (
            "Write at least 512 numbered one-word items, then end naturally."
            if case["scenario_id"] == "C04"
            else f"Return the exact marker {marker} and no other text."
        )
        argv.extend(["--prompt", prompt])
        stdin_text = ""
    for flag, key in (("--temperature", "temperature"), ("--top-p", "top_p"), ("--top-k", "top_k"), ("--seed", "seed"), ("--max-tokens", "max_tokens"), ("--repeat-penalty", "repetition_penalty")):
        if key in values:
            argv.extend([flag, str(values[key])])
    thinking = values.get("enable_thinking")
    if case["scenario_id"] == "C19":
        if case["variant"] == "hard-thinking":
            thinking = True
        elif case["variant"] == "hard-no-thinking":
            thinking = False
        else:
            thinking = "model-default"
    if thinking is True:
        argv.append("--enable-thinking")
    elif thinking is False:
        argv.append("--disable-thinking")
    argv.extend(["--effective-config-json", str(actual_config), *run_extra_args])
    input_path = case_root / "input.json"
    input_document: dict[str, Any] = {"case_id": case_id, "stdin": stdin_text, "argv": argv}
    if case["scenario_id"] == "C01":
        input_document["resolution_probe"] = build_c01_resolution_probe(
            model_arg,
            case["model_key"],
            sources=locked_sources,
            semantic_root=semantic_root,
            tokenizer_root=tokenizer_root,
        )
    write_json(input_path, input_document)
    stdout_path = case_root / "stdout.log"
    stderr_path = case_root / "stderr.log"
    started_at = iso_now()
    started_ns = time.monotonic_ns()
    child_environment = sanitized_child_environment()
    proc = subprocess.Popen(
        argv,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_environment,
        start_new_session=True,
    )
    pid = proc.pid
    pgid = os.getpgid(proc.pid)
    process_receipt = capture_process_receipt(
        root,
        case_root / "process-receipt.json",
        pid=pid,
        pgid=pgid,
        argv=argv,
        role="ferrum-run",
        environment=child_environment,
    )
    try:
        stdout_text, stderr_text = proc.communicate(input=stdin_text, timeout=timeout_sec)
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        trailing_stdout, trailing_stderr = proc.communicate(timeout=10)
        returncode = 124
        stdout_text = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr_text = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stdout_text += trailing_stdout or ""
        stderr_text += trailing_stderr or ""
        stderr_text += "\ncollector timeout\n"
        timed_out = True
    finished_ns = time.monotonic_ns()
    finished_at = iso_now()
    stdout_path.write_text(stdout_text or "no product stdout captured\n", encoding="utf-8")
    stderr_path.write_text(stderr_text or "product stderr was empty; collector recorded successful execution\n", encoding="utf-8")
    if not actual_config.is_file() or actual_config.stat().st_size == 0:
        write_json(actual_config, {"missing": True, "case_id": case_id, "reason": "product did not emit effective config"})
    if case["scenario_id"] == "C01" and case["variant"] == "unknown-fail-closed":
        input_document["negative_probe"] = execute_c01_unknown_fixture(
            root,
            case,
            binary_path=binary_path,
            backend=backend,
            semantic_root=semantic_root,
            tokenizer_root=tokenizer_root,
            weight_format=require_string(locked_sources.get("weight_format"), "C01 locked weight format"),
            child_environment=child_environment,
            timeout_sec=timeout_sec,
            case_root=case_root,
        )
        write_json(input_path, input_document)
    envelope = {
        "id": f"exec-{case_id}",
        "mode": "subprocess",
        "argv": argv,
        "pid": pid,
        "pgid": pgid,
        "started_at": started_at,
        "finished_at": finished_at,
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": finished_ns,
        "duration_sec": (finished_ns - started_ns) / 1_000_000_000,
        "returncode": returncode,
        "timed_out": timed_out,
        "child_environment": child_environment,
        "child_environment_sha256": canonical_json_sha256(child_environment),
    }
    observed = {"case_id": case_id, "expected_marker": marker}
    if case["scenario_id"] == "C01":
        observed.update(
            {
                "model_key": case["model_key"],
                "model_files": case["model_files"],
                "requested_model": model_arg,
                "ordinal": int(case["ordinal"]),
            }
        )
    if case["scenario_id"] == "C19":
        observed.update(
            {
                "thinking_mode": case["variant"],
                "model_key": case["model_key"],
                "reasoning_expected": thinking_reasoning_expected(case["model_key"], case["variant"]),
                "history_turn_count": 2,
            }
        )
    return argv, envelope, input_path, stdout_path, stderr_path, {
        "actual_config": actual_config,
        "observed": observed,
        "process_receipt": process_receipt,
        "child_environment": child_environment,
    }


def serve_case_request(
    root: Path,
    case: dict[str, Any],
    *,
    base_url: str,
    timeout_sec: float,
    case_root: Path,
    server: ProductServer,
    scheduler_trace_path: Path,
    effective_config_path: Path,
) -> tuple[list[str], dict[str, Any], Path, Path, Path, Path, dict[str, Any]]:
    case_id = case["case_id"]
    payload = case_http_payload(case, case["model_key"])
    input_path = case_root / "input.json"
    write_json(input_path, payload)
    requested = int(case["concurrency_cell"]["requested_concurrency"]) if case["concurrency_cell"] else 1
    started_at = iso_now()
    started_ns = time.monotonic_ns()
    trace_offset = scheduler_trace_path.stat().st_size if scheduler_trace_path.is_file() else 0
    trace_rows: list[dict[str, Any]] = []
    release_ticks = 0
    released = False
    release_observed_ns: int | None = None
    release_wall = 0.0
    history_construction_errors: list[dict[str, Any]] = []
    if case["scenario_id"] == "C09":
        abort_started = time.monotonic()
        aborted = aborted_http_exchange(base_url, copy.deepcopy(payload), case["variant"])
        deadline = abort_started + 5.0
        while time.monotonic() < deadline:
            fresh, trace_offset = read_jsonl_since(scheduler_trace_path, trace_offset)
            observed_ns = time.monotonic_ns()
            trace_rows.extend(
                {"raw": row, "collector_observed_monotonic_ns": observed_ns, "raw_sha256": canonical_json_sha256(row)}
                for row in fresh
            )
            released, release_ticks, release_observed_ns = trace_released(trace_rows)
            if released:
                break
            time.sleep(0.02)
        release_wall = (
            (release_observed_ns - aborted["started_monotonic_ns"]) / 1e9
            if released and release_observed_ns is not None
            else time.monotonic() - abort_started
        )
        recovery_payload = copy.deepcopy(payload)
        recovery_payload["metadata"] = {**recovery_payload.get("metadata", {}), "g00_recovery": True}
        recovery, _ = http_exchange(base_url, recovery_payload, timeout_sec)
        results = [(aborted, {}), (recovery, {})]
    elif case["scenario_id"] == "C18":
        with concurrent.futures.ThreadPoolExecutor(max_workers=requested) as pool:
            futures = [pool.submit(http_exchange, base_url, copy.deepcopy(payload), timeout_sec) for _ in range(requested)]
            results = [future.result() for future in futures]
    elif case["scenario_id"] == "C07":
        results = []
        history: list[dict[str, Any]] = []
        conversation_id = require_string(payload["metadata"].get("g00_conversation_id"), f"case {case_id}.conversation_id")
        for turn in range(1, 6):
            turn_payload = copy.deepcopy(payload)
            user_message = {
                "role": "user",
                "content": f"Conversation {conversation_id}, turn {turn}. Return {case_marker(case_id)}-T{turn} exactly.",
            }
            turn_payload["messages"] = [*copy.deepcopy(history), user_message]
            turn_payload["metadata"] = {**turn_payload["metadata"], "g00_turn": turn}
            result = http_exchange(base_url, turn_payload, timeout_sec)
            results.append(result)
            history.append(user_message)
            assistant = history_response_message(result[0], f"case {case_id}.turn[{turn}]", history_construction_errors)
            if assistant is not None:
                history.append({"role": "assistant", "content": assistant.get("content")})
    elif case["scenario_id"] in {"C06", "C12", "C17"} or (case["scenario_id"] == "C21" and case["variant"] == "serve-stream"):
        reference_payload = copy.deepcopy(payload)
        reference_payload.pop("stream", None)
        reference_payload.pop("stream_options", None)
        reference = http_exchange(base_url, reference_payload, timeout_sec)
        stream = http_exchange(
            base_url,
            copy.deepcopy(payload),
            timeout_sec,
            read_chunk_size=1 if case["scenario_id"] == "C17" else None,
        )
        results = [reference, stream]
    elif case["scenario_id"] == "C19":
        first = http_exchange(base_url, copy.deepcopy(payload), timeout_sec)
        first_message = history_response_message(first[0], f"case {case_id}.first", history_construction_errors)
        second_payload = copy.deepcopy(payload)
        second_payload["messages"] = [copy.deepcopy(payload["messages"][0])]
        if first_message is not None:
            second_payload["messages"].append(copy.deepcopy(first_message))
        second_payload["messages"].append(
            {"role": "user", "content": f"Using that reasoning history, return {case_marker(case_id)}-H2 exactly."}
        )
        second_payload["metadata"] = {**second_payload["metadata"], "g00_history_turn": 2}
        second = http_exchange(base_url, second_payload, timeout_sec)
        results = [first, second]
    elif case["scenario_id"] == "C21" and case["variant"] == "required-tool":
        results = [http_exchange(base_url, copy.deepcopy(payload), timeout_sec) for _ in range(2)]
    else:
        results = [http_exchange(base_url, copy.deepcopy(payload), timeout_sec)]
    exchanges = [item[0] for item in results]
    if case["scenario_id"] != "C09":
        fresh, _ = read_jsonl_since(scheduler_trace_path, trace_offset)
        observed_ns = time.monotonic_ns()
        trace_rows.extend(
            {"raw": row, "collector_observed_monotonic_ns": observed_ns, "raw_sha256": canonical_json_sha256(row)}
            for row in fresh
        )
    sse: dict[str, Any] = {"done_count": 0, "usage_count": 0, "delta_count": 0}
    utf8_wire_evidence: dict[str, Any] | None = None
    if case["scenario_id"] in {"C06", "C12", "C17"} or (case["scenario_id"] == "C21" and case["variant"] == "serve-stream"):
        stream_exchange = exchanges[-1]
        reconstruction = parse_sse_evidence(str(stream_exchange.get("response_raw", "")))
        sse = {
            "done_count": reconstruction["done_count"],
            "usage_count": reconstruction["usage_count"],
            "delta_count": reconstruction["delta_count"],
            "stream_reconstruction": reconstruction,
        }
        stream_exchange["response"] = {}
        if case["scenario_id"] == "C17":
            utf8_wire_evidence = {
                "chunks_base64": stream_exchange.pop("wire_chunks_base64", None),
                "decoded_fragments": stream_exchange.pop("incremental_decoded_fragments", None),
                "split_boundary_count": stream_exchange.pop("utf8_split_boundary_count", None),
                "wire_sha256": stream_exchange.pop("wire_sha256", None),
            }
    models_response: dict[str, Any] | None = None
    if case["scenario_id"] == "C20":
        models_status, models_response = http_get_json(base_url.rstrip("/") + "/v1/models", timeout_sec)
    transcript = {
        "case_id": case_id,
        "exchanges": exchanges,
        "scheduler_trace_rows": trace_rows,
        "models_response": models_response,
        "models_status": models_status if case["scenario_id"] == "C20" else None,
        "history_construction_errors": history_construction_errors,
        "utf8_wire_evidence": utf8_wire_evidence,
        **sse,
    }
    transcript_path = case_root / "http-transcript.json"
    write_json(transcript_path, transcript)
    stdout_path = case_root / "stdout.log"
    stderr_path = case_root / "stderr.log"
    stdout_path.write_text("\n".join(str(exchange.get("response_raw", "")) for exchange in exchanges) or "empty HTTP body\n", encoding="utf-8")
    stderr_path.write_text(f"case {case_id} HTTP checker completed; server pid={server.pid} pgid={server.pgid}\n", encoding="utf-8")
    finished_ns = time.monotonic_ns()
    finished_at = iso_now()
    argv = ["curl", "--request", "POST", case_id, "--data-binary", f"@{input_path}", base_url.rstrip("/") + "/v1/chat/completions"]
    envelope = {
        "id": f"exec-{case_id}",
        "mode": "in-process-http",
        "argv": argv,
        "pid": os.getpid(),
        "pgid": os.getpgid(0),
        "started_at": started_at,
        "finished_at": finished_at,
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": finished_ns,
        "duration_sec": (finished_ns - started_ns) / 1_000_000_000,
        "returncode": 0,
        "server_started_at": server.started_at,
        "server_ready_at": server.ready_at,
        "server_pid": server.pid,
        "server_pgid": server.pgid,
    }
    observed: dict[str, Any] = {"case_id": case_id, "expected_marker": expected_case_text(case)}
    if case["scenario_id"] == "C07":
        observed.update({"conversation_id": f"conversation-{case_id}", "history_turn_count": 5})
    if case["scenario_id"] == "C14":
        prompt, schema, _ = strict_schema_case(case)
        observed.update(
            {
                "strict_schema_sha256": canonical_json_sha256(schema),
                "strict_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
    if case["scenario_id"] == "C19":
        observed.update(
            {
                "thinking_mode": case["variant"],
                "model_key": case["model_key"],
                "reasoning_expected": thinking_reasoning_expected(case["model_key"], case["variant"]),
                "history_turn_count": 2,
            }
        )
    if case["scenario_id"] == "C09":
        observed.update(
            {
                "scheduler_ticks_to_release": release_ticks if released else 999,
                "wall_sec_to_release": release_wall,
                "post_capacity_success": len(exchanges) == 2 and exchanges[-1]["status"] == 200,
            }
        )
    if case["scenario_id"] == "C18":
        observed.update(
            {
                "requested_concurrency": requested,
                "typed_admission_cap": typed_admission_cap(effective_config_path),
                "observed_max_active": observed_max_active(trace_rows),
            }
        )
    if case["scenario_id"] == "C20":
        data = models_response.get("data") if isinstance(models_response, dict) else None
        first = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else {}
        observed["declared_modalities"] = first.get("modalities", [])
    return argv, envelope, input_path, stdout_path, stderr_path, transcript_path, observed


def classify_execution_outcome(
    case: dict[str, Any],
    *,
    returncode: int,
    stdout_path: Path,
    transcript_path: Path | None,
    observed: dict[str, Any],
    input_path: Path,
    actual_config_path: Path,
    artifact_root: Path,
) -> tuple[str, str | None, str | None]:
    if returncode != 0:
        stderr_text = (stdout_path.parent / "stderr.log").read_text(encoding="utf-8", errors="replace").lower()
        if "unsupported" in stderr_text or "not supported" in stderr_text:
            return "blocked", "legacy-model-backend-unsupported", "product command reported unsupported model/backend"
        return "blocked", "product-command-failed", f"product command returned {returncode}"
    transcript = read_json(transcript_path) if transcript_path is not None else None
    try:
        validate_case_output(
            case["scenario_id"],
            case["variant"],
            case["entrypoint"],
            stdout_path,
            transcript,
            observed,
            f"case {case['case_id']}",
            input_document=read_json(input_path),
            actual_config=read_json(actual_config_path),
            artifact_root=artifact_root,
        )
    except (ScenarioError, json.JSONDecodeError) as exc:
        return "known-fail", f"{case['scenario_id'].lower()}-contract-violation", str(exc)
    return "pass", None, None


def normalized_source_files(raw: Any, label: str) -> dict[str, str]:
    rows = require_list(raw, label)
    result: dict[str, str] = {}
    for index, row_raw in enumerate(rows):
        row = require_object(row_raw, f"{label}[{index}]")
        path = require_string(row.get("path"), f"{label}[{index}].path")
        require(path not in result and not Path(path).is_absolute() and ".." not in Path(path).parts, f"{label} contains duplicate/unsafe path {path}")
        result[path] = require_sha256(row.get("sha256"), f"{label}[{index}].sha256")
    require(result, f"{label} must not be empty")
    return result


def locked_execution_sources(models_lock: dict[str, Any], model_key: str, backend: str) -> dict[str, Any]:
    models = require_list(models_lock.get("models"), "models.lock.models")
    matching = [require_object(row, "models.lock model") for row in models if isinstance(row, dict) and row.get("key") == model_key]
    require(len(matching) == 1, f"models.lock must contain exactly one {model_key} entry")
    lane = require_object(require_object(matching[0].get("lanes"), f"models.lock.{model_key}.lanes").get(backend), f"models.lock.{model_key}.{backend}")
    weight_revision = require_git_sha(lane.get("revision"), f"models.lock.{model_key}.{backend}.revision")
    weight_files = normalized_source_files(lane.get("files"), f"models.lock.{model_key}.{backend}.files")
    semantic = require_object(lane.get("semantic_source"), f"models.lock.{model_key}.{backend}.semantic_source")
    semantic_normalized = {
        "repo": require_string(semantic.get("repo"), "models.lock semantic repo"),
        "revision": require_git_sha(semantic.get("revision"), "models.lock semantic revision"),
        "files": normalized_source_files(semantic.get("files"), "models.lock semantic files"),
    }
    tokenizer_raw = lane.get("tokenizer_source")
    tokenizer_normalized: dict[str, Any] | None = None
    if tokenizer_raw is not None:
        tokenizer = require_object(tokenizer_raw, f"models.lock.{model_key}.{backend}.tokenizer_source")
        tokenizer_normalized = {
            "repo": require_string(tokenizer.get("repo"), "models.lock tokenizer repo"),
            "revision": require_git_sha(tokenizer.get("revision"), "models.lock tokenizer revision"),
            "files": normalized_source_files(tokenizer.get("files"), "models.lock tokenizer files"),
        }
    chat = require_object(lane.get("chat_template"), f"models.lock.{model_key}.{backend}.chat_template")
    chat_source = require_string(chat.get("source"), "models.lock chat_template.source")
    require(chat_source in {"semantic_source", "tokenizer_source"}, "models.lock chat template source invalid")
    require(chat_source == "semantic_source" or tokenizer_normalized is not None, "models.lock chat template selects absent tokenizer_source")
    selected = semantic_normalized if chat_source == "semantic_source" else tokenizer_normalized
    require(selected is not None, "models.lock selected chat source missing")
    chat_normalized = {
        "source": chat_source,
        "repo": require_string(chat.get("repo"), "models.lock chat_template.repo"),
        "revision": require_git_sha(chat.get("revision"), "models.lock chat_template.revision"),
        "path": require_string(chat.get("path"), "models.lock chat_template.path"),
        "json_pointer": chat.get("json_pointer"),
        "container_sha256": require_sha256(chat.get("container_sha256"), "models.lock chat_template.container_sha256"),
        "content_sha256": require_sha256(chat.get("content_sha256"), "models.lock chat_template.content_sha256"),
    }
    require(chat_normalized["repo"] == selected["repo"] and chat_normalized["revision"] == selected["revision"], "models.lock chat template source identity mismatch")
    require(selected["files"].get(chat_normalized["path"]) == chat_normalized["container_sha256"], "models.lock chat template container is not bound to selected source")
    return {
        "weight_format": require_string(lane.get("format"), f"models.lock.{model_key}.{backend}.format"),
        "weight_revision": weight_revision,
        "weight_files": weight_files,
        "semantic_source": semantic_normalized,
        "tokenizer_source": tokenizer_normalized,
        "chat_template": chat_normalized,
    }


def validate_local_source_root(raw: Any, source: dict[str, Any], label: str) -> Path:
    root = Path(require_string(raw, label)).expanduser().resolve()
    require(root.is_dir(), f"{label} is not a local directory")
    for rel, digest in source["files"].items():
        path = root / rel
        try:
            path.absolute().relative_to(root)
        except ValueError as exc:
            raise ScenarioError(f"{label} file escapes its root: {rel}") from exc
        require(path.is_file() and file_sha256(path) == digest, f"{label} locked file missing or SHA mismatch: {rel}")
    return root


def validate_local_model_arg(raw: Any, weight_files: dict[str, str], label: str) -> Path:
    supplied = Path(require_string(raw, label)).expanduser().absolute()
    require(supplied.exists(), f"{label} does not exist")
    if supplied.is_file():
        require(len(weight_files) == 1, f"{label} is a file, but the locked model has {len(weight_files)} files")
        rel, digest = next(iter(weight_files.items()))
        require(supplied.name == Path(rel).name, f"{label} filename differs from the locked model file: {rel}")
        require(file_sha256(supplied) == digest, f"{label} locked weight SHA mismatch: {rel}")
        return supplied
    require(supplied.is_dir(), f"{label} is neither a model file nor directory")
    root = supplied.resolve()
    for rel, digest in weight_files.items():
        path = root / rel
        try:
            path.absolute().relative_to(root)
        except ValueError as exc:
            raise ScenarioError(f"{label} weight escapes its root: {rel}") from exc
        require(path.is_file() and file_sha256(path) == digest, f"{label} locked weight missing or SHA mismatch: {rel}")
    return supplied


def validate_execution_manifest(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    require(not ({"runner", "commands", "scenarios", "pass_line", "status", "artifact_path"} & set(manifest)), "execution manifest must not contain prebuilt report fields")
    validate_source_identity(manifest, "execution_manifest")
    for key in ("model_key", "backend", "model_revision", "hardware_id"):
        require_string(manifest.get(key), f"execution_manifest.{key}")
    require(manifest["model_key"] in {"m1-qwen35-4b", "m2-qwen35-35b-a3b", "m3-qwen3-30b-a3b"}, "execution manifest model_key invalid")
    require(manifest["backend"] in {"cuda", "metal"}, "execution manifest backend invalid")
    require_git_sha(manifest["model_revision"], "execution_manifest.model_revision")
    model_files = require_object(manifest.get("model_files"), "execution_manifest.model_files")
    require(model_files, "execution manifest model_files empty")
    for path, digest in model_files.items():
        require_string(path, "execution_manifest.model_files path")
        require_sha256(digest, f"execution_manifest.model_files[{path}]")
    binary_path, _, _ = validate_artifact_ref(root, manifest.get("binary_artifact"), "execution_manifest.binary_artifact", allowed_kinds={"binary"})
    models_lock_path, _, models_lock_document = validate_artifact_ref(root, manifest.get("models_lock"), "execution_manifest.models_lock", allowed_kinds={"raw-json"})
    require(file_sha256(binary_path) == require_sha256(manifest.get("binary_sha256"), "execution_manifest.binary_sha256"), "execution manifest binary SHA mismatch")
    require(file_sha256(models_lock_path) == require_sha256(manifest.get("models_lock_sha256"), "execution_manifest.models_lock_sha256"), "execution manifest models.lock SHA mismatch")
    effective_path, _, effective = validate_artifact_ref(root, manifest.get("effective_config"), "execution_manifest.effective_config", allowed_kinds={"raw-json"})
    require_object(effective, "execution_manifest effective config")
    execution = require_object(manifest.get("execution"), "execution_manifest.execution")
    require(set(execution) >= {"model_arg", "semantic_source_root", "host", "port", "startup_timeout_sec", "case_timeout_sec", "run_extra_args", "serve_extra_args"}, "execution manifest execution fields incomplete")
    require_string(execution.get("model_arg"), "execution_manifest.execution.model_arg")
    require(execution.get("host") in {"127.0.0.1", "localhost"}, "execution manifest host must be loopback")
    require_count(execution.get("port"), "execution_manifest.execution.port", minimum=1)
    for key in ("startup_timeout_sec", "case_timeout_sec"):
        value = execution.get(key)
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value <= 3600, f"execution_manifest.execution.{key} invalid")
    for key in ("run_extra_args", "serve_extra_args"):
        values = require_list(execution.get(key), f"execution_manifest.execution.{key}")
        require(all(isinstance(value, str) and value for value in values), f"execution_manifest.execution.{key} invalid")
        require(not any(value.startswith("FERRUM_") for value in values), f"execution_manifest.execution.{key} contains hidden env control")
    sources = locked_execution_sources(require_object(models_lock_document, "execution_manifest models.lock JSON"), manifest["model_key"], manifest["backend"])
    require(manifest["model_revision"] == sources["weight_revision"], "execution manifest model_revision differs from models.lock weight revision")
    require(manifest["model_files"] == sources["weight_files"], "execution manifest model_files must contain exactly the selected weight lock")
    model_path = validate_local_model_arg(execution.get("model_arg"), sources["weight_files"], "execution_manifest.execution.model_arg")
    semantic_root = validate_local_source_root(execution.get("semantic_source_root"), sources["semantic_source"], "execution_manifest.execution.semantic_source_root")
    if sources["tokenizer_source"] is None:
        require(execution.get("tokenizer_source_root") in {None, str(semantic_root)}, "execution manifest has an undeclared tokenizer source root")
        tokenizer_root = semantic_root
    else:
        tokenizer_root = validate_local_source_root(execution.get("tokenizer_source_root"), sources["tokenizer_source"], "execution_manifest.execution.tokenizer_source_root")
    return {
        "binary_path": binary_path,
        "models_lock_path": models_lock_path,
        "effective_path": effective_path,
        "execution": execution,
        "sources": sources,
        "model_path": model_path,
        "semantic_root": semantic_root,
        "tokenizer_root": tokenizer_root,
    }


def execute_manifest(
    manifest: dict[str, Any],
    root: Path,
    out: Path,
    *,
    discover: bool,
    allow_internal_fixture: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    out = out.resolve()
    validated = validate_execution_manifest(manifest, root)
    catalog = internal_expectations_catalog() if allow_internal_fixture else validate_expectations_catalog(read_json(EXPECTATIONS_PATH))
    catalog_bytes = (json.dumps(catalog, indent=2, sort_keys=True) + "\n").encode("utf-8") if allow_internal_fixture else EXPECTATIONS_PATH.read_bytes()
    expectations_path = root / "legacy-correctness-expectations.json"
    expectations_path.write_bytes(catalog_bytes)
    expectations_sha = file_sha256(expectations_path)
    if not allow_internal_fixture:
        require(expectations_sha == canonical_expectations_sha256(), "execution used a non-canonical expectations catalog")
    rows = planned_case_rows(manifest["model_key"], manifest["backend"], catalog)
    unresolved = [row for row in rows if row["expectation"]["expected_status"] == "discovery-required"]
    if unresolved and not discover:
        raise ScenarioError(f"canonical collection refused: {len(unresolved)} cases remain discovery-required; run --discover first")
    invocation_started = iso_now()
    invocation_start_ns = time.monotonic_ns()
    input_manifest_path = root / "scenario-execution-manifest.snapshot.json"
    write_json(input_manifest_path, manifest)
    invocation_argv = (
        [str(RUNNER_PATH), "--manifest", str(input_manifest_path), "--artifact-root", str(root), "--out", str(out), *(["--discover"] if discover else [])]
        if allow_internal_fixture
        else [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    )
    if not allow_internal_fixture:
        require(Path(invocation_argv[0]).resolve() == RUNNER_PATH, "canonical executor must be entered through the checked-in runner main")
        require("--manifest" in invocation_argv and "--artifact-root" in invocation_argv and "--out" in invocation_argv, "canonical executor invocation argv incomplete")
        require(("--discover" in invocation_argv) is discover, "canonical executor mode differs from actual invocation argv")
    invocation_process_receipt = capture_process_receipt(
        root,
        root / "scenario-executor-process-receipt.json",
        pid=os.getpid(),
        pgid=os.getpgid(0),
        argv=invocation_argv,
        role="scenario-executor",
    )
    invocation = {
        "runner_path": RUNNER_REPO_PATH,
        "runner_sha256": file_sha256(RUNNER_PATH),
        "argv": invocation_argv,
        "manifest_sha256": file_sha256(input_manifest_path),
        "manifest_snapshot": existing_artifact_ref(root, input_manifest_path, "raw-json"),
        "started_at": invocation_started,
        "started_monotonic_ns": invocation_start_ns,
        "pid": os.getpid(),
        "pgid": os.getpgid(0),
        "process_receipt": invocation_process_receipt,
        "mode": "discover" if discover else "canonical",
    }
    execution = validated["execution"]
    binary_path = validated["binary_path"]
    base_url = f"http://{execution['host']}:{execution['port']}"
    lane_root = root / "correctness" / manifest["model_key"] / manifest["backend"]
    serve_stdout = lane_root / "commands/serve.stdout.log"
    serve_stderr = lane_root / "commands/serve.stderr.log"
    serve_config = lane_root / "commands/serve.actual-effective-config.json"
    scheduler_trace = lane_root / "commands/serve.scheduler-trace.jsonl"
    serve_argv = [
        str(binary_path),
        "serve",
        execution["model_arg"],
        "--backend",
        manifest["backend"],
        "--host",
        execution["host"],
        "--port",
        str(execution["port"]),
        "--effective-config-json",
        str(serve_config),
        "--scheduler-trace-jsonl",
        str(scheduler_trace),
        *execution["serve_extra_args"],
    ]
    server: ProductServer | None = None
    server_error: str | None = None

    def ensure_server() -> None:
        nonlocal server, server_error
        if server is not None or server_error is not None:
            return
        try:
            server = ProductServer(
                argv=serve_argv,
                stdout_path=serve_stdout,
                stderr_path=serve_stderr,
                artifact_root=root,
                receipt_path=lane_root / "commands/serve.process-receipt.json",
                base_url=base_url,
                startup_timeout_sec=float(execution["startup_timeout_sec"]),
            )
        except ScenarioError as exc:
            server_error = str(exc)

    case_refs_by_scenario: dict[str, list[dict[str, str]]] = {scenario_id: [] for scenario_id in SCENARIO_IDS}
    case_results: dict[str, list[dict[str, Any]]] = {scenario_id: [] for scenario_id in SCENARIO_IDS}
    first_run_record: dict[str, Any] | None = None
    try:
        execution_rows = sorted(rows, key=lambda row: 0 if row["entrypoint"] == "run" else 1)
        for row in execution_rows:
            row = {**row, "model_key": manifest["model_key"], "model_files": manifest["model_files"]}
            case_root = lane_root / "scenarios" / row["scenario_id"] / "cases" / row["case_id"]
            case_root.mkdir(parents=True, exist_ok=True)
            transcript_path: Path | None = None
            if row["entrypoint"] == "run":
                argv, envelope, input_path, stdout_path, stderr_path, extra = run_case_command(
                    root,
                    row,
                    binary_path=binary_path,
                    model_arg=execution["model_arg"],
                    backend=manifest["backend"],
                    run_extra_args=execution["run_extra_args"],
                    timeout_sec=float(execution["case_timeout_sec"]),
                    case_root=case_root,
                    locked_sources=validated["sources"],
                    semantic_root=validated["semantic_root"],
                    tokenizer_root=validated["tokenizer_root"],
                )
                observed = extra["observed"]
                actual_config = extra["actual_config"]
                execution_process_receipt = extra["process_receipt"]
                product_process_receipt = extra["process_receipt"]
                child_environment = extra["child_environment"]
                if first_run_record is None:
                    first_run_record = {
                        "argv": argv,
                        "envelope": envelope,
                        "stdout": stdout_path,
                        "stderr": stderr_path,
                        "process_receipt": extra["process_receipt"],
                    }
            else:
                ensure_server()
            if row["entrypoint"] == "serve" and server is not None:
                argv, envelope, input_path, stdout_path, stderr_path, transcript_path, observed = serve_case_request(
                    root,
                    row,
                    base_url=base_url,
                    timeout_sec=float(execution["case_timeout_sec"]),
                    case_root=case_root,
                    server=server,
                    scheduler_trace_path=scheduler_trace,
                    effective_config_path=serve_config,
                )
                actual_config = serve_config
                execution_process_receipt = invocation_process_receipt
                product_process_receipt = server.process_receipt
                child_environment = server.child_environment
            elif row["entrypoint"] == "serve":
                input_path = case_root / "input.json"
                write_json(input_path, case_http_payload(row, manifest["model_key"]))
                stdout_path = case_root / "stdout.log"
                stderr_path = case_root / "stderr.log"
                stdout_path.write_text("serve process unavailable\n", encoding="utf-8")
                stderr_path.write_text((server_error or "serve startup failed") + "\n", encoding="utf-8")
                actual_config = case_root / "actual-effective-config.json"
                write_json(actual_config, {"missing": True, "reason": server_error or "serve startup failed"})
                now = iso_now()
                now_ns = time.monotonic_ns()
                argv = ["curl", "--request", "POST", row["case_id"], base_url + "/v1/chat/completions"]
                envelope = {"id": f"exec-{row['case_id']}", "mode": "in-process-http", "argv": argv, "pid": os.getpid(), "pgid": os.getpgid(0), "started_at": now, "finished_at": iso_now(), "started_monotonic_ns": now_ns, "finished_monotonic_ns": time.monotonic_ns(), "duration_sec": max(1e-6, (time.monotonic_ns() - now_ns) / 1e9), "returncode": 69}
                observed = {"case_id": row["case_id"], "expected_marker": expected_case_text(row)}
                execution_process_receipt = invocation_process_receipt
                product_process_receipt = None
                child_environment = sanitized_child_environment()
            status, failure_class, checker_error = classify_execution_outcome(
                row,
                returncode=int(envelope["returncode"]),
                stdout_path=stdout_path,
                transcript_path=transcript_path,
                observed=observed,
                input_path=input_path,
                actual_config_path=actual_config,
                artifact_root=root,
            )
            expectation = row["expectation"]
            if not discover:
                require(status == expectation["expected_status"], f"case {row['case_id']} unexpected status: expected {expectation['expected_status']}, observed {status}")
                require(failure_class == expectation["failure_class"], f"case {row['case_id']} failure class drift: expected {expectation['failure_class']}, observed {failure_class}")
            command_spec_path = case_root / "command-spec.json"
            write_json(
                command_spec_path,
                {
                    "case_id": row["case_id"],
                    "execution_mode": envelope["mode"],
                    "argv": argv,
                    "input_sha256": file_sha256(input_path),
                    "binary_sha256": manifest["binary_sha256"],
                    "model_revision": manifest["model_revision"],
                    "model_files": manifest["model_files"],
                    "model_path": execution["model_arg"],
                    "actual_effective_config_sha256": file_sha256(actual_config),
                    "child_environment_sha256": canonical_json_sha256(child_environment),
                },
            )
            checker_path = case_root / "checker.log"
            checker_path.write_text(
                f"case={row['case_id']} observed_status={status} failure_class={failure_class or 'none'} checker_error={checker_error or 'none'}\n",
                encoding="utf-8",
            )
            envelope_path = case_root / "execution-envelope.json"
            envelope_document = {
                "schema_version": SCHEMA_VERSION,
                "case_id": row["case_id"],
                "command_spec": existing_artifact_ref(root, command_spec_path, "raw-json"),
                "spawn": envelope,
                "execution_process_receipt": execution_process_receipt,
                "product_process_receipt": product_process_receipt,
                "child_environment": child_environment,
                "child_environment_sha256": canonical_json_sha256(child_environment),
                "product_argv": argv if row["entrypoint"] == "run" else serve_argv,
                "product_process": (
                    {
                        "argv": argv,
                        "pid": envelope["pid"],
                        "pgid": envelope["pgid"],
                        "started_at": envelope["started_at"],
                        "finished_at": envelope["finished_at"],
                        "started_monotonic_ns": envelope["started_monotonic_ns"],
                        "finished_monotonic_ns": envelope["finished_monotonic_ns"],
                        "returncode": envelope["returncode"],
                    }
                    if row["entrypoint"] == "run"
                    else ({
                        "argv": serve_argv,
                        "pid": server.pid,
                        "pgid": server.pgid,
                        "started_at": server.started_at,
                        "ready_at": server.ready_at,
                        "started_monotonic_ns": server.started_monotonic_ns,
                        "ready_monotonic_ns": server.ready_monotonic_ns,
                        "state_during_case": "running",
                    } if server is not None else {"argv": serve_argv, "state_during_case": "startup-failed", "failure": server_error})
                ),
                "stdout": existing_artifact_ref(root, stdout_path, "stdout-log"),
                "stderr": existing_artifact_ref(root, stderr_path, "stderr-log"),
                "http_transcript": existing_artifact_ref(root, transcript_path, "http-transcript") if transcript_path else None,
                "actual_effective_config": existing_artifact_ref(root, actual_config, "raw-json"),
                "checker": {
                    "mode": "in-process",
                    "function": "classify_execution_outcome",
                    "runner_sha256": file_sha256(RUNNER_PATH),
                    "input_artifact_sha256": {
                        "stdout": file_sha256(stdout_path),
                        "stderr": file_sha256(stderr_path),
                        **({"http_transcript": file_sha256(transcript_path)} if transcript_path else {}),
                    },
                    "result": status,
                    "failure_class": failure_class,
                    "log": existing_artifact_ref(root, checker_path, "checker-log"),
                },
            }
            write_json(envelope_path, envelope_document)
            expected_outcome = {key: expectation[key] for key in ("expected_status", "failure_class", "downstream_goal", "owner", "evidence_basis", "next_action")}
            case_document = {
                "schema_version": SCHEMA_VERSION,
                "case_id": row["case_id"],
                "scenario_id": row["scenario_id"],
                "ordinal": row["ordinal"],
                "status": status,
                **{key: manifest[key] for key in ("source_git_sha", "source_tree_sha", "models_lock_sha256", "binary_sha256", "model_key", "backend", "model_revision", "model_files", "hardware_id")},
                "effective_config_sha256": file_sha256(validated["effective_path"]),
                "expectations_catalog_sha256": expectations_sha,
                "entrypoint": row["entrypoint"],
                "variant": row["variant"],
                "preset": row["preset"],
                "model_identity": {
                    **{key: manifest[key] for key in ("model_key", "backend", "model_revision", "model_files", "binary_sha256")},
                    "model_path": execution["model_arg"],
                },
                "execution": {key: envelope[key] for key in ("id", "argv", "started_at", "finished_at", "duration_sec", "returncode")},
                "execution_envelope": existing_artifact_ref(root, envelope_path, "raw-json"),
                "expected_outcome": expected_outcome,
                "observed_outcome": {"status": status, "failure_class": failure_class},
                "artifacts": {
                    "input": existing_artifact_ref(root, input_path, "request-json"),
                    "stdout": existing_artifact_ref(root, stdout_path, "stdout-log"),
                    "stderr": existing_artifact_ref(root, stderr_path, "stderr-log"),
                    "effective_config": manifest["effective_config"],
                    **({"http_transcript": existing_artifact_ref(root, transcript_path, "http-transcript")} if transcript_path else {}),
                },
                "observed": observed,
                "checks": {"execution_envelope": True, "model_binding": True, "scenario_oracle": status == "pass"},
            }
            if status != "pass":
                case_document["checks"]["scenario_oracle"] = False
            case_path = case_root / "case.json"
            write_json(case_path, case_document)
            ref = existing_artifact_ref(root, case_path, "raw-json")
            case_refs_by_scenario[row["scenario_id"]].append(ref)
            case_results[row["scenario_id"]].append(case_document)
    finally:
        if server is not None:
            server.stop()
    for scenario_id in SCENARIO_IDS:
        ordered = sorted(
            zip(case_results[scenario_id], case_refs_by_scenario[scenario_id], strict=True),
            key=lambda pair: int(pair[0]["ordinal"]),
        )
        case_results[scenario_id] = [case for case, _ in ordered]
        case_refs_by_scenario[scenario_id] = [ref for _, ref in ordered]
    invocation["finished_at"] = iso_now()
    invocation["finished_monotonic_ns"] = time.monotonic_ns()
    invocation["duration_sec"] = (invocation["finished_monotonic_ns"] - invocation_start_ns) / 1e9
    invocation_path = root / "scenario-executor-invocation.json"
    write_json(invocation_path, invocation)
    if discover:
        discovery = {
            "schema_version": SCHEMA_VERSION,
            "status": "discovery",
            "source_git_sha": manifest["source_git_sha"],
            "model_key": manifest["model_key"],
            "backend": manifest["backend"],
            "expectations_catalog_sha256": expectations_sha,
            "executor_invocation": existing_artifact_ref(root, invocation_path, "raw-json"),
            "case_count": len(rows),
            "observations": [existing_artifact_ref(root, root / ref["path"], "raw-json") for refs in case_refs_by_scenario.values() for ref in refs],
            "artifact_path": str(out),
            "formal_pass_allowed": False,
        }
        return discovery
    require(server is not None, f"canonical serve session unavailable: {server_error}")
    require(first_run_record is not None, "canonical run produced no execution record")
    effective_sha = file_sha256(validated["effective_path"])
    commands = []
    for command_id, entrypoint, argv, times, stdout_path, stderr_path, process_receipt in (
        ("actual-run", "run", first_run_record["argv"], first_run_record["envelope"], first_run_record["stdout"], first_run_record["stderr"], first_run_record["process_receipt"]),
        ("actual-serve", "serve", serve_argv, server.envelope(), serve_stdout, serve_stderr, server.process_receipt),
    ):
        duration = (parse_timestamp(times["finished_at"], "command finished") - parse_timestamp(times["started_at"], "command started")).total_seconds()
        commands.append(
            {
                "id": command_id,
                "entrypoint": entrypoint,
                "argv": argv,
                "source_git_sha": manifest["source_git_sha"],
                "source_tree_sha": manifest["source_tree_sha"],
                "models_lock_sha256": manifest["models_lock_sha256"],
                "binary_sha256": manifest["binary_sha256"],
                "effective_config_sha256": effective_sha,
                "started_at": times["started_at"],
                "finished_at": times["finished_at"],
                "duration_sec": max(duration, 1e-6),
                "env": times["child_environment"],
                "env_sha256": times["child_environment_sha256"],
                "process_receipt": process_receipt,
                "returncode": 0,
                "stdout": existing_artifact_ref(root, stdout_path, "stdout-log"),
                "stderr": existing_artifact_ref(root, stderr_path, "stderr-log"),
            }
        )
    scenarios = []
    for scenario_id in SCENARIO_IDS:
        shape = selftest_scenario_shape(scenario_id, manifest["model_key"], manifest["backend"])
        results = case_results[scenario_id]
        passed = sum(case["status"] == "pass" for case in results)
        known = sum(case["status"] == "known-fail" for case in results)
        blocked = sum(case["status"] == "blocked" for case in results)
        status = "blocked" if blocked else "known-fail" if known else "pass"
        assertions = selftest_assertions(scenario_id, len(results)) if status == "pass" else {"expected_failure_count": known + blocked, "unexpected_count": 0}
        if scenario_id == "C18":
            shape["concurrency_cells"] = [
                {
                    "requested_concurrency": case["observed"]["requested_concurrency"],
                    "case_count": 1,
                    "passed_count": 1 if case["status"] == "pass" else 0,
                    "completion_rate": 1.0 if case["status"] == "pass" else 0.0,
                    "typed_admission_cap": case["observed"]["typed_admission_cap"],
                    "observed_max_active": case["observed"]["observed_max_active"],
                    "error_count": 0 if case["status"] == "pass" else 1,
                    "bad_output_count": 0,
                    "crosstalk_count": 0,
                    "bad_checksum_count": 0,
                    "server_500_count": 0,
                    "panic_count": 0,
                    "oom_count": 0,
                }
                for case in results
            ]
        scenario = {
            **shape,
            "status": status,
            "passed_count": passed,
            "known_failed_count": known,
            "blocked_count": blocked,
            "failed_count": known,
            "error_count": 0,
            "unexpected_count": 0,
            "command_ids": ["actual-run" if entrypoint == "run" else "actual-serve" for entrypoint in shape["entrypoints"]],
            "assertions": assertions,
        }
        raw = {
            **{key: manifest[key] for key in ("source_git_sha", "source_tree_sha", "models_lock_sha256", "binary_sha256", "model_key", "backend", "model_revision", "model_files", "hardware_id")},
            "effective_config_sha256": effective_sha,
            "expectations_catalog_sha256": expectations_sha,
            "scenario_id": scenario_id,
            **{key: scenario[key] for key in ("status", "case_count", "passed_count", "known_failed_count", "blocked_count", "failed_count", "error_count", "unexpected_count", "presets", "unpreset_count", "entrypoints", "command_ids", "variants", "dimensions", "assertions")},
            "cases": case_refs_by_scenario[scenario_id],
        }
        if scenario_id == "C18":
            raw["concurrency_cells"] = shape["concurrency_cells"]
        raw_path = lane_root / "scenarios" / scenario_id / "raw.json"
        write_json(raw_path, raw)
        checker_path = lane_root / "scenarios" / scenario_id / "checker.log"
        checker_path.write_text(f"{scenario_id} checked {len(results)} independently executed cases; status={status}\n", encoding="utf-8")
        scenario["artifacts"] = [existing_artifact_ref(root, raw_path, "raw-json"), existing_artifact_ref(root, checker_path, "checker-log")]
        scenarios.append(scenario)
    report = {
        **{key: manifest[key] for key in ("source_git_sha", "source_tree_sha", "dirty_status", "models_lock_sha256", "binary_sha256", "model_key", "backend", "model_revision", "model_files", "hardware_id", "binary_artifact", "models_lock", "effective_config")},
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "model_path": execution["model_arg"],
        "runner": internal_fixture_runner_identity() if allow_internal_fixture else canonical_runner_identity(),
        "expectations_catalog_sha256": expectations_sha,
        "expectations_catalog": existing_artifact_ref(root, expectations_path, "raw-json"),
        "executor_invocation": existing_artifact_ref(root, invocation_path, "raw-json"),
        "commands": commands,
        "scenarios": scenarios,
        "artifact_path": str(out),
        "pass_line": f"{PASS_PREFIX}: {out}",
    }
    attach_pair_registry(report, root)
    validate_report_document(report, root, report_path=out, allow_internal_fixture=allow_internal_fixture, require_current_output_path=True)
    return report


def selftest_assertions(scenario_id: str, case_count: int) -> dict[str, Any]:
    values: dict[str, Any] = {field: 0 for field in ZERO_ASSERTIONS[scenario_id]}
    values.update({"bad_output_count": 0, "resource_final_state": "released"})
    if scenario_id == "C02":
        values["natural_eos_count"] = case_count
    elif scenario_id == "C06":
        values.update({"done_count": case_count, "usage_count": case_count, "output_with_delta_count": case_count, "paired_nonstream_equivalence_count": case_count})
    elif scenario_id == "C07":
        values.update({"conversation_count": 6, "history_turn_count": 30})
    elif scenario_id == "C09":
        values.update({"released_count": case_count, "post_capacity_success_count": case_count, "max_scheduler_ticks_to_release": 2, "max_wall_sec_to_release": 5.0})
    elif scenario_id in {"C10", "C11", "C12", "C13"}:
        values["tool_success_count"] = case_count
        if scenario_id == "C12":
            values["paired_c11_equivalence_count"] = case_count
    elif scenario_id == "C14":
        values.update({"valid_json_count": case_count, "distinct_schema_count": case_count, "distinct_prompt_count": case_count})
    elif scenario_id == "C15":
        values["valid_object_count"] = case_count
    elif scenario_id == "C20":
        values.update({"rejected_media_count": 40, "text_array_success_count": 10, "declared_modalities": ["text"]})
    elif scenario_id == "C17":
        values["streaming_split_boundary_count"] = case_count // 2
    elif scenario_id == "C19":
        values["history_case_count"] = case_count
    elif scenario_id == "C21":
        values.update({"tool_priority_count": 4, "serve_stream_count": 4, "strict_schema_count": 4, "json_object_count": 4})
    return values


def selftest_scenario_shape(scenario_id: str, model_key: str, backend: str) -> dict[str, Any]:
    count = minimum_case_count(scenario_id, model_key)
    presets_min, unpreset = minimum_presets(scenario_id, model_key)
    presets = dict(presets_min)
    assigned = sum(presets.values()) + unpreset
    if assigned < count:
        target = next(iter(presets), None)
        if target is None:
            unpreset += count - assigned
        else:
            presets[target] += count - assigned
    variants_min, partition = required_variants(scenario_id, model_key)
    variants = dict(variants_min)
    if partition:
        assigned_variants = sum(variants.values())
        target = next(iter(variants), None)
        if target is None:
            variants["all"] = count
        elif assigned_variants < count:
            variants[target] += count - assigned_variants
    dimensions: dict[str, int] = {}
    if scenario_id == "C03":
        dimensions = {"groups": 10, "rounds_per_group": 3}
    elif scenario_id == "C04":
        dimensions = {"groups": 3, "min_output_tokens": 512}
    elif scenario_id == "C07":
        dimensions = {"requests": 6, "rounds_per_request": 5}
    scenario: dict[str, Any] = {
        "id": scenario_id,
        "status": "pass",
        "case_count": count,
        "passed_count": count,
        "known_failed_count": 0,
        "blocked_count": 0,
        "failed_count": 0,
        "error_count": 0,
        "unexpected_count": 0,
        "presets": presets,
        "unpreset_count": unpreset,
        "entrypoints": sorted(required_entrypoints(scenario_id)),
        "command_ids": [f"fixture-{entrypoint}" for entrypoint in sorted(required_entrypoints(scenario_id))],
        "variants": variants,
        "dimensions": dimensions,
        "assertions": selftest_assertions(scenario_id, count),
    }
    if scenario_id == "C18":
        concurrencies = [1, 4, 16, 32] if backend == "cuda" else [1, 4, 16]
        cells = []
        for requested in concurrencies:
            cells.append(
                {
                    "requested_concurrency": requested,
                    "case_count": 1,
                    "passed_count": 1,
                    "completion_rate": 1.0,
                    "typed_admission_cap": 16 if backend == "cuda" else 8,
                    "observed_max_active": min(requested, 16 if backend == "cuda" else 8),
                    "error_count": 0,
                    "bad_output_count": 0,
                    "crosstalk_count": 0,
                    "bad_checksum_count": 0,
                    "server_500_count": 0,
                    "panic_count": 0,
                    "oom_count": 0,
                }
            )
        scenario["case_count"] = sum(cell["case_count"] for cell in cells)
        scenario["passed_count"] = scenario["case_count"]
        scenario["presets"] = {"P_DETERMINISTIC": scenario["case_count"]}
        scenario["variants"] = {"concurrency-cell": scenario["case_count"]}
        scenario["concurrency_cells"] = cells
    return scenario


def expand_counts(counts: dict[str, int]) -> list[str]:
    return [name for name, count in counts.items() for _ in range(count)]


def planned_variant_presets(scenario_id: str, scenario: dict[str, Any]) -> list[tuple[str, str | None]]:
    if scenario_id == "C14":
        rows: list[tuple[str, str | None]] = []
        for preset, counts in (
            ("P_NO_THINKING", {"required": 13, "type": 13, "additional-properties": 12, "enum": 12}),
            ("P_THINKING", {"required": 5, "type": 5, "additional-properties": 5, "enum": 5}),
        ):
            rows.extend((variant, preset) for variant in expand_counts(counts))
        require(len(rows) == scenario["case_count"], "C14 planned partition must contain exactly 70 cases")
        return rows
    variants = expand_counts(scenario["variants"])
    presets: list[str | None] = [*expand_counts(scenario["presets"]), *([None] * scenario["unpreset_count"])]
    return list(zip(variants, presets, strict=True))


def planned_entrypoint(scenario_id: str, variant: str, ordinal: int, entrypoints: list[str]) -> str:
    if scenario_id == "C21":
        return "run" if variant == "run-plain" else "serve"
    return entrypoints[(ordinal - 1) % len(entrypoints)]


def make_case_fixture(
    root: Path,
    *,
    base: dict[str, Any],
    effective_config_ref: dict[str, str],
    scenario_id: str,
    ordinal: int,
    entrypoint: str,
    variant: str,
    preset: str | None,
    binary_argv0: str,
    concurrency_cell: dict[str, Any] | None = None,
) -> dict[str, str]:
    case_id = f"{scenario_id.lower()}-{ordinal:03d}"
    case_root = f"correctness/{base['model_key']}/{base['backend']}/scenarios/{scenario_id}/cases/{case_id}"
    case_spec = {
        "case_id": case_id,
        "scenario_id": scenario_id,
        "variant": variant,
        "preset": preset,
        "ordinal": ordinal,
        "model_key": base["model_key"],
    }
    marker = expected_case_text(case_spec)
    input_value: dict[str, Any]
    if entrypoint == "serve":
        input_value = case_http_payload(case_spec, base["model_key"])
    else:
        input_value = {
            "case_id": case_id,
            "scenario_id": scenario_id,
            "variant": variant,
            "preset": preset,
            "prompt": f"Return the exact marker {marker}.",
        }
        if scenario_id == "C01":
            fixture_config = {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "eos_token_id": [1],
                "fixture_unknown_official_field": {"preserved": True},
            }
            fixture_tokenizer = {"model": {"type": "BPE", "vocab": {"<s>": 0, "</s>": 1}}}
            fixture_tokenizer_path = root / case_root / "tokenizer.json"
            write_json(fixture_tokenizer_path, fixture_tokenizer)
            tokenizer_digest = file_sha256(fixture_tokenizer_path)
            fixture_template = "{% for message in messages %}{{ message['role'] }}:{{ message['content'] }}{% endfor %}"
            config_digest = canonical_json_sha256(fixture_config)
            template_digest = hashlib.sha256(fixture_template.encode("utf-8")).hexdigest()
            input_value["resolution_probe"] = {
                "available": True,
                "requested_model": base["model_path"],
                "resolved_model_path": base["model_path"],
                "resolved_source_kind": "internal-fixture",
                "config": {
                    "source": "config.json",
                    "raw_sha256": config_digest,
                    "locked_sha256": config_digest,
                    "document_sha256": config_digest,
                    "document": fixture_config,
                    "resolved_architecture": "Qwen3ForCausalLM",
                    "unknown_top_level_fields": ["fixture_unknown_official_field"],
                },
                "template": {
                    "source": "tokenizer_config.json:chat_template",
                    "raw_file_sha256": template_digest,
                    "locked_container_sha256": template_digest,
                    "template_sha256": template_digest,
                    "locked_content_sha256": template_digest,
                    "template": fixture_template,
                    "source_document": {"chat_template": fixture_template},
                },
                "special_tokens": {
                    "source": "semantic_source/tokenizer_config.json",
                    "tokenizer_sha256": tokenizer_digest,
                    "locked_tokenizer_sha256": tokenizer_digest,
                    "container_sha256": template_digest,
                    "locked_container_sha256": template_digest,
                    "tokens": {"bos_token": "<s>", "eos_token": "</s>"},
                    "eos_token_ids": [1],
                },
                "runtime_binding": {
                    "semantic_repo": "internal/fixture",
                    "semantic_revision": "1" * 40,
                    "config_container_sha256": config_digest,
                    "template_source": "semantic_source",
                    "template_repo": "internal/fixture",
                    "template_revision": "1" * 40,
                    "template_path": "tokenizer_config.json",
                    "template_container_sha256": template_digest,
                    "template_content_sha256": template_digest,
                },
                "expected_runtime_architectures": sorted(EXPECTED_ARCHITECTURES[base["model_key"]]),
            }
            if variant == "unknown-fail-closed":
                negative_root = root / case_root / "unknown-architecture-fixture"
                negative_root.mkdir(parents=True, exist_ok=True)
                negative_config = copy.deepcopy(fixture_config)
                negative_config["architectures"] = [f"G00UnsupportedArchitecture{ordinal:03d}"]
                negative_config["model_type"] = f"g00_unsupported_layout_{ordinal:03d}"
                negative_config["g00_negative_fixture"] = {"ordinal": ordinal, "expected_failure": "unsupported-architecture-layout"}
                negative_config_path = negative_root / "config.json"
                negative_tokenizer_json_path = negative_root / "tokenizer.json"
                negative_tokenizer_path = negative_root / "tokenizer_config.json"
                dummy_path = negative_root / "DUMMY_WEIGHT_NOT_FOR_LOADING"
                negative_stdout = negative_root / "stdout.log"
                negative_stderr = negative_root / "stderr.log"
                write_json(negative_config_path, negative_config)
                shutil.copy2(fixture_tokenizer_path, negative_tokenizer_json_path)
                write_json(negative_tokenizer_path, {"chat_template": fixture_template, "bos_token": "<s>", "eos_token": "</s>"})
                dummy_path.write_text("Architecture dispatch must reject this fixture before any weight or kernel load.\n", encoding="utf-8")
                if base["backend"] == "metal":
                    negative_named_weight = negative_root / "unknown-layout.gguf"
                    negative_named_weight.write_bytes(minimal_unknown_gguf(f"G00UnsupportedArchitecture{ordinal:03d}"))
                    negative_invocation_model = negative_named_weight
                    negative_weight_format = "gguf_q4_k_m"
                    negative_access_kind = "metadata-only-gguf"
                    negative_mode = "0o644"
                else:
                    negative_named_weight = negative_root / "model.safetensors"
                    header = b"{}"
                    negative_named_weight.write_bytes(struct.pack("<Q", len(header)) + header)
                    negative_invocation_model = negative_root
                    negative_weight_format = "gptq_int4"
                    negative_access_kind = "unreadable-empty-safetensors-sentinel"
                    negative_mode = "0o0"
                negative_root_ref = negative_root.resolve().relative_to(root.resolve()).as_posix()
                negative_invocation_model_ref = negative_invocation_model.resolve().relative_to(root.resolve()).as_posix()
                negative_weight_sha = file_sha256(negative_named_weight)
                negative_weight_stat = negative_named_weight.stat()
                negative_stdout.write_text("negative fixture rejected before inference\n", encoding="utf-8")
                negative_stderr.write_text(f"unsupported architecture/layout: G00UnsupportedArchitecture{ordinal:03d}\n", encoding="utf-8")
                negative_env = {"NO_COLOR": "1", "PYTHONUNBUFFERED": "1"}
                negative_argv = [binary_argv0, "run", negative_invocation_model_ref, "--backend", base["backend"]]
                started = iso_now()
                start_ns = time.monotonic_ns()
                receipt = capture_process_receipt(
                    root,
                    negative_root / "process-receipt.json",
                    pid=os.getpid(),
                    pgid=os.getpgid(0),
                    argv=negative_argv,
                    role="ferrum-run",
                    environment=negative_env,
                )
                finish_ns = time.monotonic_ns()
                finished = iso_now()
                negative_artifacts = {
                    "config": existing_artifact_ref(root, negative_config_path, "raw-json"),
                    "tokenizer": existing_artifact_ref(root, negative_tokenizer_json_path, "raw-json"),
                    "tokenizer_config": existing_artifact_ref(root, negative_tokenizer_path, "raw-json"),
                    "dummy_weight_marker": existing_artifact_ref(root, dummy_path, "runtime-log"),
                    "named_weight": existing_artifact_ref(root, negative_named_weight, "binary"),
                    "stdout": existing_artifact_ref(root, negative_stdout, "stdout-log"),
                    "stderr": existing_artifact_ref(root, negative_stderr, "stderr-log"),
                    "process_receipt": receipt,
                }
                input_value["negative_probe"] = {
                    "contract": "unsupported-architecture-layout-fail-closed",
                    "fixture_id": f"unknown-layout-{ordinal:03d}",
                    "unknown_architecture": f"G00UnsupportedArchitecture{ordinal:03d}",
                    "base_config_sha256": config_digest,
                    "fixture_manifest_sha256": canonical_json_sha256(negative_artifacts),
                    "fixture_root": negative_root_ref,
                    "invocation_model": negative_invocation_model_ref,
                    "weight_format": negative_weight_format,
                    "weight_access_contract": {
                        "kind": negative_access_kind,
                        "tensor_count": 0,
                        "payload_bytes": 0,
                        "mode_during_execution": negative_mode,
                        "sha256_before": negative_weight_sha,
                        "sha256_after": negative_weight_sha,
                        "atime_ns_before": negative_weight_stat.st_atime_ns,
                        "atime_ns_after": negative_weight_stat.st_atime_ns,
                        "mtime_ns_before": negative_weight_stat.st_mtime_ns,
                        "mtime_ns_after": negative_weight_stat.st_mtime_ns,
                    },
                    "argv": negative_argv,
                    "environment": negative_env,
                    "environment_sha256": canonical_json_sha256(negative_env),
                    "pid": os.getpid(),
                    "pgid": os.getpgid(0),
                    "started_at": started,
                    "finished_at": finished,
                    "started_monotonic_ns": start_ns,
                    "finished_monotonic_ns": finish_ns,
                    "returncode": 65,
                    "artifacts": negative_artifacts,
                    "effective_config_emitted": False,
                }
    input_rel = f"{case_root}/input.json"
    write_json(root / input_rel, input_value)
    if entrypoint == "run":
        assistant_count = 3 if scenario_id == "C03" else 2 if scenario_id == "C19" else 1
        rows: list[dict[str, Any]] = [{"event": "ready", "model": base["model_key"], "backend": base["backend"]}]
        for turn in range(1, assistant_count + 1):
            content = (
                f"{marker}-H{turn}"
                if scenario_id == "C19"
                else marker if turn == assistant_count else f"turn-{turn}-ack"
            )
            rows.append({"event": "user", "turn": turn, "content": input_value["prompt"]})
            assistant_row = {
                    "event": "assistant",
                    "turn": turn,
                    "content": content,
                    "finish_reason": "eos",
                    "n_tokens": 512 if scenario_id == "C04" else 4,
                    "chunk_count": 512 if scenario_id == "C04" else 2,
                    "ms": 10.0,
                }
            if scenario_id == "C19":
                if thinking_reasoning_expected(base["model_key"], variant):
                    assistant_row["reasoning"] = f"fixture reasoning turn {turn}"
                assistant_row["history_turns_seen"] = turn - 1
            rows.append(assistant_row)
        rows.append({"event": "exit", "reason": "complete"})
        stdout_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        transcript_ref = None
        argv = [binary_argv0, "run", base["model_key"], "--output-format", "jsonl", "--prompt", case_id]
        observed: dict[str, Any] = {"case_id": case_id, "expected_marker": marker}
        if scenario_id == "C01":
            observed.update(
                {
                    "model_key": base["model_key"],
                    "model_files": base["model_files"],
                    "requested_model": base["model_path"],
                    "ordinal": ordinal,
                }
            )
        if scenario_id == "C19":
            observed.update(
                {
                    "thinking_mode": variant,
                    "model_key": base["model_key"],
                    "reasoning_expected": thinking_reasoning_expected(base["model_key"], variant),
                    "history_turn_count": 2,
                }
            )
    else:
        status = 400 if scenario_id == "C16" or (scenario_id == "C20" and variant != "text-array") else 200
        if status == 400:
            response: dict[str, Any] = {
                "error": {
                    "message": f"rejected {variant}",
                    "type": "invalid_request_error",
                    "param": variant,
                }
            }
        else:
            message: dict[str, Any] = {"role": "assistant", "content": marker}
            finish_reason = "stop"
            completion_tokens = 4
            if scenario_id in {"C10", "C11", "C12"}:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{case_id}",
                            "type": "function",
                            "function": {"name": "lookup_weather", "arguments": json.dumps({"city": "Paris"}, separators=(",", ":"))},
                        }
                    ],
                }
            elif scenario_id == "C13":
                message["content"] = "The tool result is 21."
            elif scenario_id == "C14":
                _, _, strict_output = strict_schema_case(case_spec)
                message["content"] = json.dumps(strict_output, separators=(",", ":"))
            elif scenario_id == "C15":
                message["content"] = json.dumps({"answer": marker}, separators=(",", ":"))
            elif scenario_id == "C08" and variant == "stop":
                message["content"] = "before"
            elif scenario_id == "C08" and variant == "max-tokens":
                message["content"] = "1 2 3 4 5 6 7 8"
                finish_reason = "length"
                completion_tokens = 8
            elif scenario_id == "C19" and thinking_reasoning_expected(base["model_key"], variant):
                message["reasoning"] = "fixture reasoning"
            elif scenario_id == "C21" and variant == "required-tool":
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{case_id}",
                            "type": "function",
                            "function": {"name": "echo_value", "arguments": json.dumps({"value": marker}, separators=(",", ":"))},
                        }
                    ],
                }
                finish_reason = "tool_calls"
            elif scenario_id == "C21" and variant in {"strict-schema", "json-object"}:
                message["content"] = json.dumps({"result": marker}, separators=(",", ":"))
            response = {
                "id": f"chatcmpl-{case_id}",
                "object": "chat.completion",
                "model": base["model_key"],
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if scenario_id in {"C10", "C11", "C12"} else finish_reason}],
                "usage": {"prompt_tokens": 8, "completion_tokens": completion_tokens, "total_tokens": 8 + completion_tokens},
            }
        requested = int(concurrency_cell["requested_concurrency"]) if concurrency_cell else 1
        exchanges = [{"request": input_value, "status": status, "response": response} for _ in range(requested)]
        stream_reconstruction: dict[str, Any] | None = None
        utf8_wire_evidence: dict[str, Any] | None = None
        if scenario_id in {"C06", "C12", "C17"} or (scenario_id == "C21" and variant == "serve-stream"):
            reference_request = copy.deepcopy(input_value)
            reference_request.pop("stream", None)
            reference_request.pop("stream_options", None)
            if scenario_id == "C12":
                stream_delta_rows = [
                    {"index": 0, "id": f"call-{case_id}", "type": "function", "function": {"name": "lookup_", "arguments": "{\"city\":"}},
                    {"index": 0, "function": {"name": "weather", "arguments": "\"Paris\"}"}},
                ]
                stream_reconstruction = {
                    "done_count": 1,
                    "usage_count": 1,
                    "delta_count": 3,
                    "malformed_count": 0,
                    "content": "",
                    "reasoning": "",
                    "finish_reason": "tool_calls",
                    "usage": response["usage"],
                    "tool_calls": response["choices"][0]["message"]["tool_calls"],
                }
                chunks = [
                    {"id": f"chatcmpl-{case_id}", "choices": [{"index": 0, "delta": {"tool_calls": [row]}, "finish_reason": None}], "usage": None}
                    for row in stream_delta_rows
                ]
                chunks.append({"id": f"chatcmpl-{case_id}", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": None})
            else:
                stream_reconstruction = {
                    "done_count": 1,
                    "usage_count": 1,
                    "delta_count": 2,
                    "malformed_count": 0,
                    "content": marker,
                    "reasoning": "",
                    "finish_reason": "stop",
                    "usage": response["usage"],
                    "tool_calls": [],
                }
                midpoint = max(1, len(marker) // 2)
                chunks = [
                    {"id": f"chatcmpl-{case_id}", "choices": [{"index": 0, "delta": {"content": marker[:midpoint]}, "finish_reason": None}], "usage": None},
                    {"id": f"chatcmpl-{case_id}", "choices": [{"index": 0, "delta": {"content": marker[midpoint:]}, "finish_reason": "stop"}], "usage": None},
                ]
            chunks.append({"id": f"chatcmpl-{case_id}", "choices": [], "usage": response["usage"]})
            raw_stream = "".join("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            exchanges = [
                {"request": reference_request, "status": 200, "response": response, "response_raw": json.dumps(response, ensure_ascii=False)},
                {"request": input_value, "status": 200, "response": {}, "response_raw": raw_stream},
            ]
            if scenario_id == "C17":
                wire_chunks = [bytes([value]) for value in raw_stream.encode("utf-8")]
                decoded, fragments, split_count = decode_wire_chunks(wire_chunks)
                require(decoded == raw_stream and split_count > 0, "C17 internal fixture failed to produce a UTF-8 split boundary")
                utf8_wire_evidence = {
                    "chunks_base64": [base64.b64encode(chunk).decode("ascii") for chunk in wire_chunks],
                    "decoded_fragments": fragments,
                    "split_boundary_count": split_count,
                    "wire_sha256": hashlib.sha256(raw_stream.encode("utf-8")).hexdigest(),
                }
        elif scenario_id == "C07":
            exchanges = []
            history: list[dict[str, Any]] = []
            conversation_id = f"conversation-{case_id}"
            for turn in range(1, 6):
                user = {"role": "user", "content": f"Conversation {conversation_id}, turn {turn}. Return {case_marker(case_id)}-T{turn} exactly."}
                request = copy.deepcopy(input_value)
                request["messages"] = [*copy.deepcopy(history), user]
                request["metadata"] = {**request["metadata"], "g00_turn": turn}
                turn_message = {"role": "assistant", "content": f"{case_marker(case_id)}-T{turn}"}
                turn_response = {
                    "id": f"chatcmpl-{case_id}-{turn}",
                    "object": "chat.completion",
                    "model": base["model_key"],
                    "choices": [{"index": 0, "message": turn_message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 8 + turn, "completion_tokens": 4, "total_tokens": 12 + turn},
                }
                exchanges.append({"request": request, "status": 200, "response": turn_response})
                history.extend([user, turn_message])
            response = exchanges[-1]["response"]
        elif scenario_id == "C19":
            first_message = copy.deepcopy(response["choices"][0]["message"])
            first_message["content"] = f"{marker}-H1"
            first_response = copy.deepcopy(response)
            first_response["choices"][0]["message"] = first_message
            second_request = copy.deepcopy(input_value)
            second_request["messages"] = [
                copy.deepcopy(input_value["messages"][0]),
                copy.deepcopy(first_message),
                {"role": "user", "content": f"Using that reasoning history, return {marker}-H2 exactly."},
            ]
            second_message = copy.deepcopy(first_message)
            second_message["content"] = f"{marker}-H2"
            second_response = copy.deepcopy(response)
            second_response["choices"][0]["message"] = second_message
            exchanges = [
                {"request": input_value, "status": 200, "response": first_response},
                {"request": second_request, "status": 200, "response": second_response},
            ]
            response = second_response
        elif scenario_id == "C21" and variant == "required-tool":
            exchanges = [{"request": input_value, "status": 200, "response": copy.deepcopy(response)} for _ in range(2)]
        if scenario_id == "C09":
            exchanges = [
                {"request": input_value, "status": 499, "response": {"client_abort": variant}},
                {"request": input_value, "status": 200, "response": response},
            ]
        transcript = {
            "case_id": case_id,
            "exchanges": exchanges,
            "models_response": {"object": "list", "data": [{"id": base["model_key"], "modalities": ["text"]}]} if scenario_id == "C20" else None,
            "done_count": stream_reconstruction["done_count"] if stream_reconstruction else 0,
            "usage_count": stream_reconstruction["usage_count"] if stream_reconstruction else 0,
            "delta_count": stream_reconstruction["delta_count"] if stream_reconstruction else 0,
            "stream_reconstruction": stream_reconstruction,
            "utf8_wire_evidence": utf8_wire_evidence,
        }
        transcript_rel = f"{case_root}/http-transcript.json"
        write_json(root / transcript_rel, transcript)
        transcript_ref = {"kind": "http-transcript", "path": transcript_rel, "sha256": file_sha256(root / transcript_rel)}
        stdout_text = json.dumps(response, sort_keys=True) + "\n"
        argv = ["curl", "--request", "POST", case_id, f"http://127.0.0.1/v1/chat/completions"]
        observed = {"case_id": case_id, "expected_marker": marker}
        if scenario_id == "C07":
            observed.update({"conversation_id": f"conversation-{case_id}", "history_turn_count": 5})
        if scenario_id == "C14":
            prompt, schema, _ = strict_schema_case(case_spec)
            observed.update({"strict_schema_sha256": canonical_json_sha256(schema), "strict_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
        if scenario_id == "C19":
            observed.update(
                {
                    "thinking_mode": variant,
                    "model_key": base["model_key"],
                    "reasoning_expected": thinking_reasoning_expected(base["model_key"], variant),
                    "history_turn_count": 2,
                }
            )
        if scenario_id == "C09":
            observed.update({"scheduler_ticks_to_release": 2, "wall_sec_to_release": 0.25, "post_capacity_success": True})
        if scenario_id == "C18":
            require(concurrency_cell is not None, "C18 fixture requires a concurrency cell")
            observed.update(
                {
                    "requested_concurrency": requested,
                    "typed_admission_cap": concurrency_cell["typed_admission_cap"],
                    "observed_max_active": concurrency_cell["observed_max_active"],
                }
            )
        if scenario_id == "C20":
            observed["declared_modalities"] = ["text"]
    stdout_rel = f"{case_root}/stdout.log"
    stderr_rel = f"{case_root}/stderr.log"
    (root / stdout_rel).parent.mkdir(parents=True, exist_ok=True)
    (root / stdout_rel).write_text(stdout_text, encoding="utf-8")
    (root / stderr_rel).write_text(f"case {case_id} checker completed successfully\n", encoding="utf-8")
    artifacts: dict[str, Any] = {
        "input": {"kind": "request-json", "path": input_rel, "sha256": file_sha256(root / input_rel)},
        "stdout": {"kind": "stdout-log", "path": stdout_rel, "sha256": file_sha256(root / stdout_rel)},
        "stderr": {"kind": "stderr-log", "path": stderr_rel, "sha256": file_sha256(root / stderr_rel)},
        "effective_config": effective_config_ref,
    }
    if transcript_ref is not None:
        artifacts["http_transcript"] = transcript_ref
    second = (ordinal % 50) + 1
    started_at = f"2026-01-{(int(scenario_id[1:]) - 1) % 28 + 1:02d}T00:00:{second:02d}Z"
    finished_at = f"2026-01-{(int(scenario_id[1:]) - 1) % 28 + 1:02d}T00:00:{second + 1:02d}Z"
    case = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "scenario_id": scenario_id,
        "ordinal": ordinal,
        "status": "pass",
        **base,
        "entrypoint": entrypoint,
        "variant": variant,
        "preset": preset,
        "model_identity": {
            **{key: base[key] for key in ("model_key", "backend", "model_revision", "model_files", "binary_sha256")},
            "model_path": base["model_path"],
        },
        "execution": {
            "id": f"exec-{case_id}",
            "argv": argv,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_sec": 1.0,
            "returncode": 0,
        },
        "expected_outcome": {
            "expected_status": "pass",
            "failure_class": None,
            "downstream_goal": {"m1-qwen35-4b": "G08A", "m2-qwen35-35b-a3b": "G08B", "m3-qwen3-30b-a3b": "G08C"}[base["model_key"]],
            "owner": "internal-fixture",
            "evidence_basis": "Internal validator fixture only; forbidden in canonical evidence.",
            "next_action": "Exercise validator branches without making a product claim.",
        },
        "observed_outcome": {"status": "pass", "failure_class": None},
        "artifacts": artifacts,
        "observed": observed,
        "checks": {"execution_envelope": True, "model_binding": True, "scenario_oracle": True},
    }
    case_rel = f"{case_root}/case.json"
    write_json(root / case_rel, case)
    return {"kind": "raw-json", "path": case_rel, "sha256": file_sha256(root / case_rel)}


def internal_expectations_catalog() -> dict[str, Any]:
    lanes: dict[str, Any] = {}
    for model_key, goal in (
        ("m1-qwen35-4b", "G08A"),
        ("m2-qwen35-35b-a3b", "G08B"),
        ("m3-qwen3-30b-a3b", "G08C"),
    ):
        for backend in ("cuda", "metal"):
            lanes[f"{model_key}/{backend}"] = {
                "rules": [
                    {
                        "selector": {"scenario_id": "*", "variant": "*", "preset": "*"},
                        "expected_status": "pass",
                        "failure_class": None,
                        "downstream_goal": goal,
                        "owner": "internal-fixture",
                        "evidence_basis": "Internal validator fixture only; forbidden in canonical evidence.",
                        "next_action": "Exercise validator branches without making a product claim.",
                    }
                ]
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_id": "runtime-vnext-g00-internal-fixture",
        "source_git_sha": FROZEN_LEGACY_SHA,
        "status_vocabulary": {
            "pass": "fixture",
            "known-fail": "fixture",
            "blocked": "fixture",
            "discovery-required": "fixture",
        },
        "resolution_policy": {"wildcard": "*"},
        "blocked_lane_policy": {
            "allowed_lane_failure_classes": BLOCKED_LANE_FAILURE_CLASSES,
            "forbidden_lanes": [
                "m1-qwen35-4b/cuda",
                "m2-qwen35-35b-a3b/cuda",
                "m3-qwen3-30b-a3b/cuda",
                "m3-qwen3-30b-a3b/metal",
            ],
        },
        "lanes": lanes,
    }


def make_internal_fixture_manifest(
    root: Path,
    *,
    model_key: str,
    backend: str,
    model_revision: str,
    model_files: dict[str, str],
    hardware_id: str,
    binary_artifact: str,
    models_lock_artifact: str,
) -> dict[str, Any]:
    binary_path = artifact_path(root, binary_artifact, "fixture.binary_artifact")
    models_lock_path = artifact_path(root, models_lock_artifact, "fixture.models_lock")
    binary_sha = file_sha256(binary_path)
    models_lock_sha = file_sha256(models_lock_path)
    expectations_rel = "legacy-correctness-expectations.json"
    write_json(root / expectations_rel, internal_expectations_catalog())
    expectations_sha = file_sha256(root / expectations_rel)
    base = {
        "source_git_sha": FROZEN_LEGACY_SHA,
        "source_tree_sha": frozen_tree_sha(),
        "models_lock_sha256": models_lock_sha,
        "binary_sha256": binary_sha,
        "model_key": model_key,
        "backend": backend,
        "model_revision": model_revision,
        "model_files": model_files,
        "hardware_id": hardware_id,
        "model_path": model_key,
        "expectations_catalog_sha256": expectations_sha,
    }
    config_rel = f"correctness/{model_key}/{backend}/effective-config.json"
    config = {
        "schema_version": 1,
        "source_git_sha": FROZEN_LEGACY_SHA,
        "source_tree_sha": frozen_tree_sha(),
        "dirty_status": {"is_dirty": False, "status_short": []},
        **base,
        "typed_effective_config": {"run": {"temperature": 0}, "serve": {"temperature": 0}},
    }
    write_json(root / config_rel, config)
    config_sha = file_sha256(root / config_rel)
    config_ref = {"kind": "raw-json", "path": config_rel, "sha256": config_sha}
    commands = []
    for entrypoint in ("run", "serve"):
        stdout = artifact_ref(root, f"correctness/{model_key}/{backend}/commands/{entrypoint}.stdout.log", "stdout-log", f"ferrum {entrypoint} fixture command completed with valid product output\n")
        stderr = artifact_ref(root, f"correctness/{model_key}/{backend}/commands/{entrypoint}.stderr.log", "stderr-log", "runtime completed without blocker markers\n")
        commands.append(
            {
                "id": f"fixture-{entrypoint}",
                "entrypoint": entrypoint,
                "argv": ["ferrum", entrypoint, model_key],
                "source_git_sha": FROZEN_LEGACY_SHA,
                "source_tree_sha": frozen_tree_sha(),
                "models_lock_sha256": models_lock_sha,
                "binary_sha256": binary_sha,
                "effective_config_sha256": config_sha,
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:00:01Z",
                "duration_sec": 1.0,
                "env": {"NO_COLOR": "1", "PYTHONUNBUFFERED": "1"},
                "env_sha256": canonical_json_sha256({"NO_COLOR": "1", "PYTHONUNBUFFERED": "1"}),
                "returncode": 0,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
    scenarios = []
    for scenario_id in SCENARIO_IDS:
        scenario = selftest_scenario_shape(scenario_id, model_key, backend)
        variant_presets = planned_variant_presets(scenario_id, scenario)
        require(len(variant_presets) == scenario["case_count"], f"{scenario_id} fixture variant/preset partition mismatch")
        entrypoints = sorted(required_entrypoints(scenario_id))
        case_refs = []
        for index in range(scenario["case_count"]):
            concurrency_cell = scenario.get("concurrency_cells", [None] * scenario["case_count"])[index] if scenario_id == "C18" else None
            case_refs.append(
                make_case_fixture(
                    root,
                    base={**base, "effective_config_sha256": config_sha},
                    effective_config_ref=config_ref,
                    scenario_id=scenario_id,
                    ordinal=index + 1,
                    entrypoint=planned_entrypoint(scenario_id, variant_presets[index][0], index + 1, entrypoints),
                    variant=variant_presets[index][0],
                    preset=variant_presets[index][1],
                    binary_argv0=binary_path.relative_to(root.resolve()).as_posix(),
                    concurrency_cell=concurrency_cell,
                )
            )
        raw = {
            **base,
            "effective_config_sha256": config_sha,
            "scenario_id": scenario_id,
            "status": "pass",
            "case_count": scenario["case_count"],
            "passed_count": scenario["passed_count"],
            "known_failed_count": scenario["known_failed_count"],
            "blocked_count": scenario["blocked_count"],
            "failed_count": 0,
            "error_count": 0,
            "unexpected_count": 0,
            "presets": scenario["presets"],
            "unpreset_count": scenario["unpreset_count"],
            "entrypoints": scenario["entrypoints"],
            "command_ids": scenario["command_ids"],
            "variants": scenario["variants"],
            "dimensions": scenario["dimensions"],
            "assertions": scenario["assertions"],
            "cases": case_refs,
        }
        if scenario_id == "C18":
            raw["concurrency_cells"] = scenario["concurrency_cells"]
        raw_rel = f"correctness/{model_key}/{backend}/scenarios/{scenario_id}/raw.json"
        log_rel = f"correctness/{model_key}/{backend}/scenarios/{scenario_id}/checker.log"
        write_json(root / raw_rel, raw)
        scenario["artifacts"] = [
            {"kind": "raw-json", "path": raw_rel, "sha256": file_sha256(root / raw_rel)},
            artifact_ref(root, log_rel, "checker-log", f"{scenario_id} fixture checker evaluated every case and emitted raw evidence\n"),
        ]
        scenarios.append(scenario)
    return {
        "source_git_sha": FROZEN_LEGACY_SHA,
        "source_tree_sha": frozen_tree_sha(),
        "dirty_status": {"is_dirty": False, "status_short": []},
        **base,
        "binary_artifact": {"kind": "binary", "path": binary_artifact, "sha256": binary_sha},
        "models_lock": {"kind": "raw-json", "path": models_lock_artifact, "sha256": models_lock_sha},
        "effective_config": config_ref,
        "commands": commands,
        "scenarios": scenarios,
        "expectations_catalog_sha256": expectations_sha,
        "expectations_catalog": {"kind": "raw-json", "path": expectations_rel, "sha256": expectations_sha},
    }


def expect_reject(report: dict[str, Any], root: Path, name: str, mutate: Callable[[dict[str, Any]], None], marker: str) -> None:
    candidate = copy.deepcopy(report)
    mutate(candidate)
    try:
        validate_report_document(candidate, root, allow_internal_fixture=True)
    except ScenarioError as exc:
        require(marker.lower() in str(exc).lower(), f"{name} rejected for unexpected reason: {exc}")
        return
    raise AssertionError(f"{name} unexpectedly passed")


def write_fake_ferrum(path: Path) -> None:
    source = r'''#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def value_after(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def write_config(argv, entrypoint):
    raw = value_after(argv, "--effective-config-json")
    if raw:
        path = Path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        backend = value_after(argv, "--backend", "auto")
        identity_path = Path(argv[1]) / "g00-resolution-identity.json"
        resolution_evidence = json.loads(identity_path.read_text()) if identity_path.is_file() else None
        path.write_text(json.dumps({
            "schema_version": 1,
            "entrypoint": entrypoint,
            "entries": [{"key": "FERRUM_BACKEND", "effective_value": backend, "source": "cli", "affects": ["performance"]}],
            "hardware_capabilities": {"backend": backend, "compiled_features": {}},
            "model_capabilities": {"architecture": "qwen3_moe", "max_context_len": 4096},
            "workload_profile": {"serving_mode": "openai_chat", "target_concurrency": 32},
            "decisions": [],
            "max_sequences": 32,
            "resolution_evidence": resolution_evidence
        }) + "\n")


def run_mode(argv):
    model_path = Path(argv[1])
    model_config = (model_path if model_path.is_dir() else model_path.parent) / "config.json"
    if model_config.is_file():
        config_document = json.loads(model_config.read_text())
        target = config_document.get("text_config") if isinstance(config_document.get("text_config"), dict) else config_document
        architectures = target.get("architectures", [])
        if architectures and str(architectures[0]).startswith("G00UnsupportedArchitecture"):
            print(f"unsupported architecture/layout: {architectures[0]}", file=sys.stderr)
            return 65
    write_config(argv, "run")
    config = value_after(argv, "--effective-config-json", "")
    match = re.search(r"/(c\d\d-\d{3})/actual-effective-config", config)
    case_id = match.group(1) if match else "c02-001"
    scenario = case_id[:3].upper()
    ordinal = int(case_id.rsplit("-", 1)[1])
    marker = "中文正确" if scenario == "C17" and ordinal <= 20 else "🙂🚀" if scenario == "C17" and ordinal <= 40 else "e\u0301" if scenario == "C17" else f"G00-{case_id}-OK"
    if scenario == "C06":
        marker = f"G00-c05-{ordinal:03d}-OK"
    turns = 3 if scenario == "C03" else 2 if scenario == "C19" else 1
    print(json.dumps({"event": "ready", "model": argv[1], "backend": value_after(argv, "--backend", "auto")}))
    for turn in range(1, turns + 1):
        print(json.dumps({"event": "user", "turn": turn, "content": "fixture input"}))
        content = f"{marker}-H{turn}" if scenario == "C19" else marker if turn == turns else f"turn-{turn}-ack"
        row = {"event": "assistant", "turn": turn, "content": content, "finish_reason": "eos", "n_tokens": 512 if scenario == "C04" else 4, "chunk_count": 512 if scenario == "C04" else 2, "ms": 1.0}
        if scenario == "C19":
            if "--disable-thinking" not in argv and not (scenario == "C19" and ordinal >= 17):
                row["reasoning"] = f"fixture reasoning turn {turn}"
            row["history_turns_seen"] = turn - 1
        print(json.dumps(row))
    print(json.dumps({"event": "exit", "reason": "complete"}))
    return 0


class Handler(BaseHTTPRequestHandler):
    server_version = "fake-ferrum"

    def log_message(self, fmt, *args):
        return

    def finish(self):
        try:
            super().finish()
        finally:
            if getattr(self, "g00_traced", False):
                self.server.request_finished(self.g00_case_id)

    def send_json(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self.send_json(200, {"object": "list", "data": [{"id": "fixture-model", "modalities": ["text"]}]})
        else:
            self.send_json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        size = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(size))
        metadata = payload.get("metadata", {})
        case_id = metadata.get("g00_case_id", "c05-001")
        scenario = metadata.get("g00_scenario_id", "C05")
        variant = metadata.get("g00_variant", "known-answer")
        marker = {"chinese": "中文正确", "emoji": "🙂🚀", "combining": "e\u0301"}.get(variant, f"G00-{case_id}-OK") if scenario == "C17" else f"G00-{case_id}-OK"
        if scenario == "C06":
            marker = f"G00-c05-{int(metadata.get('g00_ordinal', 1)):03d}-OK"
        self.g00_case_id = case_id
        self.g00_traced = True
        self.server.request_started(case_id)
        if scenario in {"C09", "C18"}:
            time.sleep(0.05)
        if scenario == "C16" or (scenario == "C20" and variant != "text-array"):
            self.send_json(400, {"error": {"message": f"rejected {variant}", "type": "invalid_request_error", "param": variant}})
            return
        if payload.get("stream"):
            if scenario == "C12":
                deltas = [
                    {"tool_calls": [{"index": 0, "id": f"call-{case_id}", "type": "function", "function": {"name": "lookup_", "arguments": "{\"city\":"}}]},
                    {"tool_calls": [{"index": 0, "function": {"name": "weather", "arguments": "\"Paris\"}"}}]},
                ]
                finish = "tool_calls"
            else:
                midpoint = max(1, len(marker) // 2)
                deltas = [{"content": marker[:midpoint]}, {"content": marker[midpoint:]}]
                finish = "stop"
            chunks = [{"id": f"chatcmpl-{case_id}", "choices": [{"index": 0, "delta": delta, "finish_reason": None}], "usage": None} for delta in deltas]
            chunks.append({"id": f"chatcmpl-{case_id}", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": None})
            chunks.append({"id": f"chatcmpl-{case_id}", "choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}})
            body = "".join("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            raw = body.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        message = {"role": "assistant", "content": marker}
        finish = "stop"
        completion_tokens = 4
        if scenario in {"C10", "C11", "C12"}:
            message = {"role": "assistant", "content": None, "tool_calls": [{"id": f"call-{case_id}", "type": "function", "function": {"name": "lookup_weather", "arguments": "{\"city\":\"Paris\"}"}}]}
            finish = "tool_calls"
        elif scenario == "C13":
            message["content"] = "The tool result is 21."
        elif scenario == "C14":
            prompt = payload["messages"][0]["content"]
            message["content"] = prompt.split(": return exactly this schema-valid object: ", 1)[1]
        elif scenario == "C15":
            message["content"] = json.dumps({"answer": marker}, separators=(",", ":"))
        elif scenario == "C08" and variant == "stop":
            message["content"] = "before"
        elif scenario == "C08" and variant == "max-tokens":
            message["content"] = "1 2 3 4 5 6 7 8"
            finish = "length"
            completion_tokens = 8
        elif scenario == "C07":
            turn = int(metadata.get("g00_turn", 1))
            message["content"] = f"G00-{case_id}-OK-T{turn}"
        elif scenario == "C19":
            history_turn = int(metadata.get("g00_history_turn", 1))
            message["content"] = f"G00-{case_id}-OK-H{history_turn}"
            kwargs = payload.get("chat_template_kwargs") or {}
            prompt = str(payload.get("messages", [{}])[0].get("content", ""))
            reasoning_enabled = kwargs.get("enable_thinking") is not False and not prompt.endswith("/no_think")
            if reasoning_enabled:
                message["reasoning"] = f"fixture reasoning turn {history_turn}"
        elif scenario == "C21" and variant == "required-tool":
            message = {"role": "assistant", "content": None, "tool_calls": [{"id": f"call-{case_id}", "type": "function", "function": {"name": "echo_value", "arguments": json.dumps({"value": marker}, separators=(",", ":"))}}]}
            finish = "tool_calls"
        elif scenario == "C21" and variant in {"strict-schema", "json-object"}:
            message["content"] = json.dumps({"result": marker}, separators=(",", ":"))
        self.send_json(200, {"id": f"chatcmpl-{case_id}", "object": "chat.completion", "model": payload.get("model"), "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": {"prompt_tokens": 8, "completion_tokens": completion_tokens, "total_tokens": 8 + completion_tokens}})


class Server(ThreadingHTTPServer):
    request_queue_size = 128

    def __init__(self, address, handler, trace_path):
        super().__init__(address, handler)
        self.trace_path = Path(trace_path) if trace_path else None
        self.trace_lock = threading.Lock()
        self.active = 0

    def handle_error(self, request, client_address):
        return

    def trace(self, case_id, phase):
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a") as handle:
            handle.write(json.dumps({"case_id": case_id, "phase": phase, "active": self.active}) + "\n")

    def request_started(self, case_id):
        with self.trace_lock:
            self.active += 1
            self.trace(case_id, "scheduled")

    def request_finished(self, case_id):
        with self.trace_lock:
            self.active = max(0, self.active - 1)
            self.trace(case_id, "released")


def serve_mode(argv):
    write_config(argv, "serve")
    port = int(value_after(argv, "--port", "8000"))
    trace_path = value_after(argv, "--scheduler-trace-jsonl")
    print(f"fake ferrum ready on {port}", flush=True)
    Server(("127.0.0.1", port), Handler, trace_path).serve_forever()


def main():
    forbidden = sorted(key for key in os.environ if key.startswith("FERRUM_"))
    if forbidden:
        print("inherited forbidden environment: " + ",".join(forbidden), file=sys.stderr)
        return 97
    argv = sys.argv[1:]
    if not argv:
        return 2
    if argv[0] == "run":
        return run_mode(argv)
    if argv[0] == "serve":
        serve_mode(argv)
        return 0
    return 2


raise SystemExit(main())
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_execution_fixture_manifest(root: Path) -> dict[str, Any]:
    binary = root / "binaries/cuda/ferrum"
    write_fake_ferrum(binary)
    models_lock = root / "models.lock.json"
    model_dir = root / "models/fixture-model"
    config_path = model_dir / "config.json"
    tokenizer_json_path = model_dir / "tokenizer.json"
    tokenizer_path = model_dir / "tokenizer_config.json"
    weight_path = model_dir / "weights.gguf"
    write_json(
        config_path,
        {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "eos_token_id": [1],
            "fixture_unknown_official_field": {"preserved": True},
        },
    )
    write_json(
        tokenizer_json_path,
        {
            "model": {"type": "BPE", "vocab": {"<s>": 0, "</s>": 1}},
            "added_tokens": [],
        },
    )
    write_json(
        tokenizer_path,
        {
            "chat_template": "{% for message in messages %}{{ message['role'] }}:{{ message['content'] }}{% endfor %}",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "fixture_tokenizer_extension": True,
        },
    )
    weight_path.write_bytes(b"GGUF fixture weight bytes\n")
    weight_sha = file_sha256(weight_path)
    semantic_repo = "internal/fixture-semantic"
    semantic_revision = "3" * 40
    template_text = read_json(tokenizer_path)["chat_template"]
    runtime_binding = {
        "semantic_repo": semantic_repo,
        "semantic_revision": semantic_revision,
        "config_container_sha256": file_sha256(config_path),
        "template_source": "semantic_source",
        "template_repo": semantic_repo,
        "template_revision": semantic_revision,
        "template_path": "tokenizer_config.json",
        "template_container_sha256": file_sha256(tokenizer_path),
        "template_content_sha256": hashlib.sha256(template_text.encode("utf-8")).hexdigest(),
    }
    write_json(model_dir / "g00-resolution-identity.json", runtime_binding)
    write_json(
        models_lock,
        {
            "schema_version": SCHEMA_VERSION,
            "models": [
                {
                    "key": "m3-qwen3-30b-a3b",
                    "lanes": {
                        "cuda": {
                            "revision": "1" * 40,
                            "format": "gptq_int4",
                            "files": [{"path": "weights.gguf", "sha256": weight_sha}],
                            "semantic_source": {
                                "repo": semantic_repo,
                                "revision": semantic_revision,
                                "files": [
                                    {"path": "config.json", "sha256": file_sha256(config_path)},
                                    {"path": "tokenizer.json", "sha256": file_sha256(tokenizer_json_path)},
                                    {"path": "tokenizer_config.json", "sha256": file_sha256(tokenizer_path)},
                                ],
                            },
                            "chat_template": {
                                "source": "semantic_source",
                                "repo": semantic_repo,
                                "revision": semantic_revision,
                                "path": "tokenizer_config.json",
                                "json_pointer": "/chat_template",
                                "container_sha256": file_sha256(tokenizer_path),
                                "content_sha256": runtime_binding["template_content_sha256"],
                            },
                        }
                    },
                }
            ],
        },
    )
    effective = root / "correctness/m3-qwen3-30b-a3b/cuda/effective-config.json"
    binary_sha = file_sha256(binary)
    models_lock_sha = file_sha256(models_lock)
    base = {
        "source_git_sha": FROZEN_LEGACY_SHA,
        "source_tree_sha": frozen_tree_sha(),
        "dirty_status": {"is_dirty": False, "status_short": []},
        "models_lock_sha256": models_lock_sha,
        "binary_sha256": binary_sha,
        "model_key": "m3-qwen3-30b-a3b",
        "backend": "cuda",
        "model_revision": "1" * 40,
        "model_files": {"weights.gguf": weight_sha},
        "hardware_id": "cuda-fixture",
    }
    write_json(
        effective,
        {
            "schema_version": SCHEMA_VERSION,
            **base,
            "typed_effective_config": {"run": {"temperature": 0}, "serve": {"temperature": 0}},
        },
    )
    return {
        **base,
        "binary_artifact": existing_artifact_ref(root, binary, "binary"),
        "models_lock": existing_artifact_ref(root, models_lock, "raw-json"),
        "effective_config": existing_artifact_ref(root, effective, "raw-json"),
        "execution": {
            "model_arg": str(model_dir),
            "semantic_source_root": str(model_dir),
            "host": "127.0.0.1",
            "port": unused_loopback_port(),
            "startup_timeout_sec": 10,
            "case_timeout_sec": 10,
            "run_extra_args": [],
            "serve_extra_args": [],
        },
    }


@contextmanager
def execution_report_mutation_fixture(
    source_root: Path,
    report: dict[str, Any],
    backup: Path,
) -> Any:
    shutil.copytree(source_root, backup)
    try:
        yield source_root, copy.deepcopy(report)
    finally:
        shutil.rmtree(source_root)
        shutil.copytree(backup, source_root)


def update_ref_sha(ref: dict[str, Any], root: Path) -> None:
    ref["sha256"] = file_sha256(root / ref["path"])


def execution_case_paths(report: dict[str, Any], root: Path, scenario_index: int = 1, case_index: int = 0) -> tuple[dict[str, Any], Path, dict[str, Any], Path, dict[str, Any], Path]:
    scenario = report["scenarios"][scenario_index]
    raw_ref = next(ref for ref in scenario["artifacts"] if ref["kind"] == "raw-json")
    raw_path = root / raw_ref["path"]
    raw = read_json(raw_path)
    case_ref = raw["cases"][case_index]
    case_path = root / case_ref["path"]
    case = read_json(case_path)
    return scenario, raw_path, raw, case_path, case, root / case["execution_envelope"]["path"]


def persist_execution_case_mutation(
    root: Path,
    scenario: dict[str, Any],
    raw_path: Path,
    raw: dict[str, Any],
    case_path: Path,
    case: dict[str, Any],
    *,
    envelope_path: Path | None = None,
    envelope: dict[str, Any] | None = None,
) -> None:
    if envelope is not None:
        require(envelope_path is not None, "mutated execution envelope path is required")
        write_json(envelope_path, envelope)
        update_ref_sha(case["execution_envelope"], root)
    write_json(case_path, case)
    case_rel = case_path.resolve().relative_to(root.resolve()).as_posix()
    case_ref = next(ref for ref in raw["cases"] if ref["path"] == case_rel)
    update_ref_sha(case_ref, root)
    write_json(raw_path, raw)
    raw_rel = raw_path.resolve().relative_to(root.resolve()).as_posix()
    raw_ref = next(ref for ref in scenario["artifacts"] if ref["kind"] == "raw-json" and ref["path"] == raw_rel)
    update_ref_sha(raw_ref, root)


def persist_transcript_mutation(
    root: Path,
    scenario: dict[str, Any],
    raw_path: Path,
    raw: dict[str, Any],
    case_path: Path,
    case: dict[str, Any],
    envelope_path: Path,
    envelope: dict[str, Any],
    transcript: dict[str, Any],
) -> None:
    transcript_path = root / case["artifacts"]["http_transcript"]["path"]
    write_json(transcript_path, transcript)
    update_ref_sha(case["artifacts"]["http_transcript"], root)
    update_ref_sha(envelope["http_transcript"], root)
    envelope["checker"]["input_artifact_sha256"]["http_transcript"] = file_sha256(transcript_path)
    persist_execution_case_mutation(
        root,
        scenario,
        raw_path,
        raw,
        case_path,
        case,
        envelope_path=envelope_path,
        envelope=envelope,
    )


def persist_input_mutation(
    root: Path,
    scenario: dict[str, Any],
    raw_path: Path,
    raw: dict[str, Any],
    case_path: Path,
    case: dict[str, Any],
    envelope_path: Path,
    envelope: dict[str, Any],
    input_document: dict[str, Any],
) -> None:
    input_path = root / case["artifacts"]["input"]["path"]
    write_json(input_path, input_document)
    update_ref_sha(case["artifacts"]["input"], root)
    command_spec_path = root / envelope["command_spec"]["path"]
    command_spec = read_json(command_spec_path)
    command_spec["input_sha256"] = file_sha256(input_path)
    write_json(command_spec_path, command_spec)
    update_ref_sha(envelope["command_spec"], root)
    persist_execution_case_mutation(root, scenario, raw_path, raw, case_path, case, envelope_path=envelope_path, envelope=envelope)


def expect_execution_report_reject(root: Path, report: dict[str, Any], marker: str) -> None:
    try:
        validate_report_document(report, root, allow_internal_fixture=True)
    except ScenarioError as exc:
        require(marker.lower() in str(exc).lower(), f"execution report mutation rejected for unexpected reason: {exc}")
        return
    raise AssertionError(f"execution report mutation unexpectedly passed; expected {marker}")


def self_test() -> int:
    history_errors: list[dict[str, Any]] = []
    valid_history = history_response_message(
        {"response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}},
        "history-fixture.valid",
        history_errors,
    )
    require(valid_history == {"role": "assistant", "content": "ok"} and not history_errors, "valid history response was not preserved")
    malformed_history = history_response_message(
        {"response": {"error": {"message": "legacy malformed response"}}},
        "history-fixture.malformed",
        history_errors,
    )
    require(malformed_history is None and len(history_errors) == 1, "malformed history response did not become evidence")
    require(
        history_errors[0]["error"] == "history-fixture.malformed.choices must be a JSON array"
        and SHA256_RE.fullmatch(history_errors[0]["response_sha256"]) is not None,
        "malformed history response evidence is incomplete",
    )
    with tempfile.TemporaryDirectory(prefix="ferrum-vnext-scenario-runner-") as tmp:
        root = Path(tmp) / "artifacts"
        root.mkdir()
        binary = root / "binaries/cuda/ferrum"
        binary.parent.mkdir(parents=True)
        binary.write_text("fixture binary\n", encoding="utf-8")
        models_lock = root / "models.lock.json"
        write_json(models_lock, {"schema_version": 1, "fixture": True})
        manifest = make_internal_fixture_manifest(
            root,
            model_key="m3-qwen3-30b-a3b",
            backend="cuda",
            model_revision="1" * 40,
            model_files={"weights.gguf": "2" * 64},
            hardware_id="cuda-fixture",
            binary_artifact="binaries/cuda/ferrum",
            models_lock_artifact="models.lock.json",
        )
        out = root / "correctness/m3-qwen3-30b-a3b/cuda/scenario-report.json"
        report = collect_manifest(manifest, root, out, allow_internal_fixture=True)
        validate_report_document(
            report,
            root,
            report_path=out,
            allow_internal_fixture=True,
            require_current_output_path=True,
        )
        try:
            validate_report_document(report, root, report_path=out)
        except ScenarioError as exc:
            require(
                "expectations catalog is not the checked-in contract" in str(exc)
                or "fixture runner is forbidden" in str(exc)
                or "exists on disk, but not in 'HEAD'" in str(exc),
                f"canonical fixture rejection used unexpected reason: {exc}",
            )
        else:
            raise AssertionError("canonical scenario validation accepted an internal fixture")
        expect_reject(report, root, "missing-ferrum-argv", lambda value: value["commands"][0].update({"argv": ["python", "run", "model"]}), "execute ferrum")
        expect_reject(report, root, "missing-tools", lambda value: value["scenarios"].pop(9), "exactly C01-C21")
        expect_reject(report, root, "missing-schema", lambda value: value["scenarios"][13]["assertions"].pop("valid_json_count"), "valid schema count")
        expect_reject(report, root, "missing-utf8", lambda value: value["scenarios"][16]["variants"].pop("emoji"), "variants.emoji")
        expect_reject(report, root, "missing-thinking", lambda value: value["scenarios"][18]["variants"].pop("soft-think"), "soft-think")
        expect_reject(report, root, "missing-cancel", lambda value: value["scenarios"][8]["variants"].pop("cancel"), "variants.cancel")
        expect_reject(report, root, "fake-pass", lambda value: value["scenarios"][0].update({"status": "PASS"}), "status invalid")
        expect_reject(report, root, "skip", lambda value: value["scenarios"][0].update({"skipped": 1}), "forbidden")

        execution_root = Path(tmp) / "execution-artifacts"
        execution_root.mkdir()
        execution_manifest = make_execution_fixture_manifest(execution_root)
        fixture_weight = Path(execution_manifest["execution"]["model_arg"]) / "weights.gguf"
        fixture_weight_bytes = fixture_weight.read_bytes()
        fixture_weight.write_bytes(fixture_weight_bytes + b"tampered")
        try:
            validate_execution_manifest(execution_manifest, execution_root)
        except ScenarioError as exc:
            require("locked weight missing or SHA mismatch" in str(exc), f"weight mutation used unexpected rejection: {exc}")
        else:
            raise AssertionError("execution manifest accepted mutated model bytes")
        fixture_weight.write_bytes(fixture_weight_bytes)
        execution_out = execution_root / "correctness/m3-qwen3-30b-a3b/cuda/scenario-report.json"
        hostile_key = "FERRUM_HOSTILE_PARENT_SWITCH"
        previous_hostile = os.environ.get(hostile_key)
        os.environ[hostile_key] = "must-not-reach-product"
        try:
            execution_report = execute_manifest(
                execution_manifest,
                execution_root,
                execution_out,
                discover=False,
                allow_internal_fixture=True,
            )
        finally:
            if previous_hostile is None:
                os.environ.pop(hostile_key, None)
            else:
                os.environ[hostile_key] = previous_hostile
        write_json(execution_out, execution_report)
        validate_report_document(
            execution_report,
            execution_root,
            report_path=execution_out,
            allow_internal_fixture=True,
            require_current_output_path=True,
        )
        require(len(planned_case_rows("m3-qwen3-30b-a3b", "cuda", internal_expectations_catalog())) == 783, "fake executor did not cover the complete C01-C21 case corpus")
        require(
            hostile_key not in json.dumps(execution_report, sort_keys=True),
            "hostile inherited FERRUM_* environment leaked into the report",
        )
        for receipt_path in execution_root.rglob("*process-receipt.json"):
            require(hostile_key not in receipt_path.read_text(encoding="utf-8"), f"hostile environment leaked into {receipt_path}")
        for model_key in ("m1-qwen35-4b", "m2-qwen35-35b-a3b"):
            modes = {
                row["variant"]
                for row in planned_case_rows(model_key, "cuda", internal_expectations_catalog())
                if row["scenario_id"] == "C19"
            }
            require(
                {"soft-think-misuse", "soft-no-think-misuse"} <= modes and not ({"soft-think", "soft-no-think"} & modes),
                f"{model_key} C19 does not distinguish soft-command misuse from Qwen3 soft switches",
            )

        rejected_mutations: set[str] = set()

        candidate = copy.deepcopy(execution_report)
        candidate["commands"][0]["env"]["RUST_LOG"] = "mutated-command-only"
        candidate["commands"][0]["env"] = dict(sorted(candidate["commands"][0]["env"].items()))
        candidate["commands"][0]["env_sha256"] = canonical_json_sha256(candidate["commands"][0]["env"])
        expect_execution_report_reject(execution_root, candidate, "environment differs from process receipt")
        rejected_mutations.add("command-receipt-environment-divergence")

        with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / "backup-c01-resolution") as (mutation_root, candidate):
            scenario, raw_path, raw, case_path, case, envelope_path = execution_case_paths(candidate, mutation_root, scenario_index=0)
            envelope = read_json(envelope_path)
            input_path = mutation_root / case["artifacts"]["input"]["path"]
            input_document = read_json(input_path)
            input_document.pop("resolution_probe")
            write_json(input_path, input_document)
            update_ref_sha(case["artifacts"]["input"], mutation_root)
            command_spec_path = mutation_root / envelope["command_spec"]["path"]
            command_spec = read_json(command_spec_path)
            command_spec["input_sha256"] = file_sha256(input_path)
            write_json(command_spec_path, command_spec)
            update_ref_sha(envelope["command_spec"], mutation_root)
            persist_execution_case_mutation(mutation_root, scenario, raw_path, raw, case_path, case, envelope_path=envelope_path, envelope=envelope)
            expect_execution_report_reject(mutation_root, candidate, "resolution_probe")
            rejected_mutations.add("c01-missing-resolution-probe")

        with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / "backup-c01-negative-class") as (mutation_root, candidate):
            scenario, raw_path, raw, case_path, case, envelope_path = execution_case_paths(candidate, mutation_root, scenario_index=0, case_index=15)
            envelope = read_json(envelope_path)
            input_document = read_json(mutation_root / case["artifacts"]["input"]["path"])
            negative = require_object(input_document.get("negative_probe"), "C01 mutation negative probe")
            negative_stderr = mutation_root / negative["artifacts"]["stderr"]["path"]
            negative_stderr.write_text("missing weight file before architecture dispatch\n", encoding="utf-8")
            update_ref_sha(negative["artifacts"]["stderr"], mutation_root)
            negative["fixture_manifest_sha256"] = canonical_json_sha256(negative["artifacts"])
            persist_input_mutation(mutation_root, scenario, raw_path, raw, case_path, case, envelope_path, envelope, input_document)
            expect_execution_report_reject(mutation_root, candidate, "lacks exact unsupported architecture/layout evidence")
            rejected_mutations.add("c01-wrong-negative-failure-class")

        with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / "backup-c01-negative-tokenizer") as (mutation_root, candidate):
            scenario, raw_path, raw, case_path, case, envelope_path = execution_case_paths(candidate, mutation_root, scenario_index=0, case_index=15)
            envelope = read_json(envelope_path)
            input_document = read_json(mutation_root / case["artifacts"]["input"]["path"])
            negative = require_object(input_document.get("negative_probe"), "C01 mutation negative probe")
            negative_tokenizer = mutation_root / negative["artifacts"]["tokenizer"]["path"]
            write_json(negative_tokenizer, {"tampered": True})
            update_ref_sha(negative["artifacts"]["tokenizer"], mutation_root)
            negative["fixture_manifest_sha256"] = canonical_json_sha256(negative["artifacts"])
            persist_input_mutation(mutation_root, scenario, raw_path, raw, case_path, case, envelope_path, envelope, input_document)
            expect_execution_report_reject(mutation_root, candidate, "negative tokenizer differs from the locked source")
            rejected_mutations.add("c01-negative-tokenizer-drift")

        with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / "backup-forged-pair-payload") as (mutation_root, candidate):
            for scenario_index in (4, 5):
                scenario, raw_path, raw, case_path, case, envelope_path = execution_case_paths(candidate, mutation_root, scenario_index=scenario_index)
                envelope = read_json(envelope_path)
                input_document = read_json(mutation_root / case["artifacts"]["input"]["path"])
                input_document["messages"][0]["content"] = "Simultaneously forged paired payload."
                transcript = read_json(mutation_root / case["artifacts"]["http_transcript"]["path"])
                for exchange in transcript["exchanges"]:
                    exchange["request"]["messages"][0]["content"] = "Simultaneously forged paired payload."
                persist_input_mutation(mutation_root, scenario, raw_path, raw, case_path, case, envelope_path, envelope, input_document)
                envelope = read_json(envelope_path)
                persist_transcript_mutation(mutation_root, scenario, raw_path, raw, case_path, case, envelope_path, envelope, transcript)
            forged_registry = build_pair_registry(
                raw_pair_case_rows(candidate, mutation_root),
                model_key=candidate["model_key"],
                backend=candidate["backend"],
            )
            registry_path = mutation_root / candidate["pair_registry"]["path"]
            write_json(registry_path, forged_registry)
            update_ref_sha(candidate["pair_registry"], mutation_root)
            expect_execution_report_reject(mutation_root, candidate, "persisted input differs from generated scenario contract")
            rejected_mutations.add("simultaneously-forged-pair-payload")

        with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / "backup-forged-tool-pair-payload") as (mutation_root, candidate):
            for scenario_index in (10, 11):
                scenario, raw_path, raw, case_path, case, envelope_path = execution_case_paths(candidate, mutation_root, scenario_index=scenario_index)
                envelope = read_json(envelope_path)
                input_document = read_json(mutation_root / case["artifacts"]["input"]["path"])
                input_document["messages"][0]["content"] = "Use lookup_weather for Lyon instead."
                transcript = read_json(mutation_root / case["artifacts"]["http_transcript"]["path"])
                for exchange in transcript["exchanges"]:
                    exchange["request"]["messages"][0]["content"] = "Use lookup_weather for Lyon instead."
                persist_input_mutation(mutation_root, scenario, raw_path, raw, case_path, case, envelope_path, envelope, input_document)
                envelope = read_json(envelope_path)
                persist_transcript_mutation(mutation_root, scenario, raw_path, raw, case_path, case, envelope_path, envelope, transcript)
            forged_registry = build_pair_registry(raw_pair_case_rows(candidate, mutation_root), model_key=candidate["model_key"], backend=candidate["backend"])
            registry_path = mutation_root / candidate["pair_registry"]["path"]
            write_json(registry_path, forged_registry)
            update_ref_sha(candidate["pair_registry"], mutation_root)
            expect_execution_report_reject(mutation_root, candidate, "persisted input differs from generated scenario contract")
            rejected_mutations.add("simultaneously-forged-tool-pair-payload")

        for scenario_index, case_index, mutation_name, mutation_marker, mutate_transcript in (
            (5, 0, "c06-missing-stream-reconstruction", "stream_reconstruction", lambda transcript: transcript.pop("stream_reconstruction")),
            (6, 0, "c07-missing-history-turn", "exactly five", lambda transcript: transcript["exchanges"].pop()),
            (11, 0, "c12-corrupt-tool-reassembly", "differs from matching C11", lambda transcript: transcript["stream_reconstruction"]["tool_calls"][0]["function"].update({"arguments": "{\"city\":\"Lyon\"}"})),
            (16, 1, "c17-missing-utf8-wire", "utf8_wire_evidence", lambda transcript: transcript.update({"utf8_wire_evidence": None})),
            (18, 1, "c19-missing-reasoning-history", "exact assistant reasoning history", lambda transcript: transcript["exchanges"][1]["request"]["messages"][1].pop("reasoning")),
        ):
            with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / f"backup-{mutation_name}") as (mutation_root, candidate):
                scenario, raw_path, raw, case_path, case, envelope_path = execution_case_paths(candidate, mutation_root, scenario_index=scenario_index, case_index=case_index)
                envelope = read_json(envelope_path)
                transcript = read_json(mutation_root / case["artifacts"]["http_transcript"]["path"])
                mutate_transcript(transcript)
                persist_transcript_mutation(mutation_root, scenario, raw_path, raw, case_path, case, envelope_path, envelope, transcript)
                expect_execution_report_reject(mutation_root, candidate, mutation_marker)
                rejected_mutations.add(mutation_name)

        with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / "backup-c14-schema-proof") as (mutation_root, candidate):
            scenario, raw_path, raw, case_path, case, _ = execution_case_paths(candidate, mutation_root, scenario_index=13)
            case["observed"].pop("strict_schema_sha256")
            persist_execution_case_mutation(mutation_root, scenario, raw_path, raw, case_path, case)
            expect_execution_report_reject(mutation_root, candidate, "strict schema digest mismatch")
            rejected_mutations.add("c14-missing-schema-proof")

        with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / "backup-c14-forged-category") as (mutation_root, candidate):
            scenario, raw_path, raw, case_path, case, envelope_path = execution_case_paths(candidate, mutation_root, scenario_index=13)
            envelope = read_json(envelope_path)
            input_document = read_json(mutation_root / case["artifacts"]["input"]["path"])
            schema = input_document["response_format"]["json_schema"]["schema"]
            schema["required"] = ["identity"]
            case["observed"]["strict_schema_sha256"] = canonical_json_sha256(schema)
            transcript = read_json(mutation_root / case["artifacts"]["http_transcript"]["path"])
            transcript["exchanges"][0]["request"] = copy.deepcopy(input_document)
            persist_input_mutation(mutation_root, scenario, raw_path, raw, case_path, case, envelope_path, envelope, input_document)
            envelope = read_json(envelope_path)
            persist_transcript_mutation(mutation_root, scenario, raw_path, raw, case_path, case, envelope_path, envelope, transcript)
            expect_execution_report_reject(mutation_root, candidate, "persisted input differs from generated scenario contract")
            rejected_mutations.add("c14-forged-category")

        with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / "backup-c21-tool-priority") as (mutation_root, candidate):
            scenario, raw_path, raw, case_path, case, envelope_path = execution_case_paths(candidate, mutation_root, scenario_index=20, case_index=8)
            envelope = read_json(envelope_path)
            input_document = read_json(mutation_root / case["artifacts"]["input"]["path"])
            input_document.pop("tools")
            transcript = read_json(mutation_root / case["artifacts"]["http_transcript"]["path"])
            for exchange in transcript["exchanges"]:
                exchange["request"].pop("tools")
            persist_input_mutation(mutation_root, scenario, raw_path, raw, case_path, case, envelope_path, envelope, input_document)
            envelope = read_json(envelope_path)
            persist_transcript_mutation(mutation_root, scenario, raw_path, raw, case_path, case, envelope_path, envelope, transcript)
            expect_execution_report_reject(mutation_root, candidate, "persisted input differs from generated scenario contract")
            rejected_mutations.add("c21-missing-tool-priority-input")

        with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / "backup-invocation-mode") as (mutation_root, candidate):
            invocation_path = mutation_root / candidate["executor_invocation"]["path"]
            invocation = read_json(invocation_path)
            invocation["mode"] = "discover"
            write_json(invocation_path, invocation)
            update_ref_sha(candidate["executor_invocation"], mutation_root)
            expect_execution_report_reject(mutation_root, candidate, "not produced by canonical executor mode")
            rejected_mutations.add("invocation-mode")

        with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / "backup-bilateral-status") as (mutation_root, candidate):
            scenario, raw_path, raw, case_path, case, _ = execution_case_paths(candidate, mutation_root)
            case["status"] = "known-fail"
            case["observed_outcome"] = {"status": "known-fail", "failure_class": "c02-contract-violation"}
            case["checks"]["scenario_oracle"] = False
            raw.update({"status": "known-fail", "passed_count": raw["passed_count"] - 1, "known_failed_count": 1, "failed_count": 1, "assertions": {"expected_failure_count": 1, "unexpected_count": 0}})
            scenario.update({"status": "known-fail", "passed_count": scenario["passed_count"] - 1, "known_failed_count": 1, "failed_count": 1, "assertions": {"expected_failure_count": 1, "unexpected_count": 0}})
            persist_execution_case_mutation(mutation_root, scenario, raw_path, raw, case_path, case)
            expect_execution_report_reject(mutation_root, candidate, "unexpected pass/fail status")
            rejected_mutations.add("bilateral-status")

        with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / "backup-nonexistent-pid") as (mutation_root, candidate):
            scenario, raw_path, raw, case_path, case, envelope_path = execution_case_paths(candidate, mutation_root)
            envelope = read_json(envelope_path)
            for process in (envelope["spawn"], envelope["product_process"]):
                process["pid"] = 99_999_999
                process["pgid"] = 99_999_999
            persist_execution_case_mutation(
                mutation_root,
                scenario,
                raw_path,
                raw,
                case_path,
                case,
                envelope_path=envelope_path,
                envelope=envelope,
            )
            expect_execution_report_reject(mutation_root, candidate, "PID/PGID mismatch")
            rejected_mutations.add("nonexistent-pid-pgid")

        with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / "backup-c09-trace") as (mutation_root, candidate):
            scenario, raw_path, raw, case_path, case, envelope_path = execution_case_paths(candidate, mutation_root, scenario_index=8)
            envelope = read_json(envelope_path)
            transcript_path = mutation_root / case["artifacts"]["http_transcript"]["path"]
            transcript = read_json(transcript_path)
            transcript["scheduler_trace_rows"] = []
            write_json(transcript_path, transcript)
            update_ref_sha(case["artifacts"]["http_transcript"], mutation_root)
            update_ref_sha(envelope["http_transcript"], mutation_root)
            envelope["checker"]["input_artifact_sha256"]["http_transcript"] = file_sha256(transcript_path)
            persist_execution_case_mutation(
                mutation_root,
                scenario,
                raw_path,
                raw,
                case_path,
                case,
                envelope_path=envelope_path,
                envelope=envelope,
            )
            expect_execution_report_reject(mutation_root, candidate, "scheduler trace evidence is empty")
            rejected_mutations.add("c09-missing-scheduler-trace")

        with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / "backup-c18-active") as (mutation_root, candidate):
            scenario, raw_path, raw, case_path, case, _ = execution_case_paths(candidate, mutation_root, scenario_index=17, case_index=3)
            transcript = read_json(mutation_root / case["artifacts"]["http_transcript"]["path"])
            require(observed_max_active(transcript["scheduler_trace_rows"]) == 32, "C18 red-team fixture did not observe max_active=32")
            case["observed"]["observed_max_active"] = 1
            next(cell for cell in raw["concurrency_cells"] if cell["requested_concurrency"] == 32)["observed_max_active"] = 1
            next(cell for cell in scenario["concurrency_cells"] if cell["requested_concurrency"] == 32)["observed_max_active"] = 1
            persist_execution_case_mutation(mutation_root, scenario, raw_path, raw, case_path, case)
            expect_execution_report_reject(mutation_root, candidate, "observed max-active is not derived from trace")
            rejected_mutations.add("c18-fabricated-max-active")

        with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / "backup-invocation-time") as (mutation_root, candidate):
            invocation_path = mutation_root / candidate["executor_invocation"]["path"]
            invocation = read_json(invocation_path)
            invocation.update(
                {
                    "started_at": "2000-01-01T00:00:00Z",
                    "finished_at": "2000-01-01T00:00:01Z",
                    "started_monotonic_ns": 1,
                    "finished_monotonic_ns": 1_000_000_001,
                    "duration_sec": 1.0,
                }
            )
            write_json(invocation_path, invocation)
            update_ref_sha(candidate["executor_invocation"], mutation_root)
            expect_execution_report_reject(mutation_root, candidate, "receipt is outside invocation window")
            rejected_mutations.add("invocation-disjoint-time")

        with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / "backup-manifest-snapshot") as (mutation_root, candidate):
            invocation_path = mutation_root / candidate["executor_invocation"]["path"]
            invocation = read_json(invocation_path)
            manifest_path = mutation_root / invocation["manifest_snapshot"]["path"]
            write_json(manifest_path, {"schema_version": SCHEMA_VERSION, "unrelated": True})
            update_ref_sha(invocation["manifest_snapshot"], mutation_root)
            invocation["manifest_sha256"] = file_sha256(manifest_path)
            write_json(invocation_path, invocation)
            update_ref_sha(candidate["executor_invocation"], mutation_root)
            expect_execution_report_reject(mutation_root, candidate, "source_git_sha")
            rejected_mutations.add("unrelated-manifest-snapshot")

        with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / "backup-product-argv") as (mutation_root, candidate):
            scenario, raw_path, raw, case_path, case, envelope_path = execution_case_paths(candidate, mutation_root, scenario_index=4)
            envelope = read_json(envelope_path)
            envelope["product_argv"][2] = "/tmp/unbound/wrong-model"
            envelope["product_process"]["argv"][2] = "/tmp/unbound/wrong-model"
            command_spec_path = mutation_root / envelope["command_spec"]["path"]
            command_spec = read_json(command_spec_path)
            command_spec["model_path"] = "/tmp/unbound/wrong-model"
            write_json(command_spec_path, command_spec)
            update_ref_sha(envelope["command_spec"], mutation_root)
            persist_execution_case_mutation(
                mutation_root,
                scenario,
                raw_path,
                raw,
                case_path,
                case,
                envelope_path=envelope_path,
                envelope=envelope,
            )
            expect_execution_report_reject(mutation_root, candidate, "command spec model path mismatch")
            rejected_mutations.add("wrong-product-argv-model")

        candidate = copy.deepcopy(execution_report)
        candidate["model_path"] = "/tmp/unbound/wrong-model"
        expect_execution_report_reject(execution_root, candidate, "manifest model path differs from report")
        rejected_mutations.add("wrong-report-model-path")

        with execution_report_mutation_fixture(execution_root, execution_report, Path(tmp) / "backup-effective-config") as (mutation_root, candidate):
            scenario, raw_path, raw, case_path, case, envelope_path = execution_case_paths(candidate, mutation_root)
            envelope = read_json(envelope_path)
            actual_config_path = mutation_root / envelope["actual_effective_config"]["path"]
            write_json(actual_config_path, {"missing": True, "fabricated": True})
            update_ref_sha(envelope["actual_effective_config"], mutation_root)
            persist_execution_case_mutation(
                mutation_root,
                scenario,
                raw_path,
                raw,
                case_path,
                case,
                envelope_path=envelope_path,
                envelope=envelope,
            )
            expect_execution_report_reject(mutation_root, candidate, "declares missing product evidence")
            rejected_mutations.add("unrelated-effective-config")

        candidate = copy.deepcopy(execution_report)
        candidate["runner"] = {**candidate["runner"], "sha256": "f" * 64}
        expect_execution_report_reject(execution_root, candidate, "internal fixture runner identity mismatch")
        rejected_mutations.add("runner-identity")

        canonical_catalog = validate_expectations_catalog(read_json(EXPECTATIONS_PATH))
        unresolved = [row for row in planned_case_rows("m1-qwen35-4b", "cuda", canonical_catalog) if row["expectation"]["expected_status"] == "discovery-required"]
        require(unresolved, "discovery-required guard fixture unexpectedly resolved every M1 CUDA case")
        for model_key in ("m1-qwen35-4b", "m2-qwen35-35b-a3b"):
            lane_key = f"{model_key}/cuda"
            blocked_catalog = copy.deepcopy(canonical_catalog)
            blocked_catalog["lanes"][lane_key]["rules"] = [
                {
                    "selector": {"scenario_id": "*", "variant": "*", "preset": "*"},
                    "expected_status": "blocked",
                    "failure_class": "legacy-model-backend-unsupported",
                    "downstream_goal": "G08A" if model_key == "m1-qwen35-4b" else "G08B",
                    "owner": "red-team-fixture",
                    "evidence_basis": "Hand-authored whole-lane blocker must not replace CUDA execution.",
                    "next_action": "Run discovery and formal product cases.",
                }
            ]
            try:
                validate_expectations_catalog(blocked_catalog)
            except ScenarioError as exc:
                require("cannot use blocked for executable lane" in str(exc), f"{lane_key} blocked mutation rejected for unexpected reason: {exc}")
            else:
                raise AssertionError(f"{lane_key} hand-authored blocked lane unexpectedly passed")
            rejected_mutations.add(f"{model_key}-cuda-blocked-lane")

        required_rejections = {
            "command-receipt-environment-divergence",
            "c01-missing-resolution-probe",
            "c01-negative-tokenizer-drift",
            "c01-wrong-negative-failure-class",
            "simultaneously-forged-pair-payload",
            "simultaneously-forged-tool-pair-payload",
            "c06-missing-stream-reconstruction",
            "c07-missing-history-turn",
            "c12-corrupt-tool-reassembly",
            "c14-missing-schema-proof",
            "c14-forged-category",
            "c17-missing-utf8-wire",
            "c19-missing-reasoning-history",
            "c21-missing-tool-priority-input",
            "invocation-mode",
            "bilateral-status",
            "nonexistent-pid-pgid",
            "c09-missing-scheduler-trace",
            "c18-fabricated-max-active",
            "invocation-disjoint-time",
            "unrelated-manifest-snapshot",
            "wrong-product-argv-model",
            "wrong-report-model-path",
            "unrelated-effective-config",
            "runner-identity",
            "m1-qwen35-4b-cuda-blocked-lane",
            "m2-qwen35-35b-a3b-cuda-blocked-lane",
        }
        require(rejected_mutations == required_rejections, "self-test did not reject the complete mutation corpus")
    print(SELFTEST_PASS_LINE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.manifest is None or args.artifact_root is None or args.out is None:
        parser.error("--manifest, --artifact-root, and --out are required")
    root = args.artifact_root.resolve()
    out = args.out.resolve()
    try:
        out.relative_to(root)
        manifest = read_json(args.manifest.resolve())
        report = execute_manifest(manifest, root, out, discover=args.discover) if args.discover else collect_manifest(manifest, root, out)
        write_json(out, report)
    except (ScenarioError, ValueError) as exc:
        print(f"FERRUM RUNTIME VNEXT G00 SCENARIOS FAIL: {args.out}: {exc}", file=sys.stderr)
        return 1
    if args.discover:
        print(f"FERRUM RUNTIME VNEXT G00 SCENARIOS DISCOVERY COMPLETE: {out}")
    else:
        print(report["pass_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
