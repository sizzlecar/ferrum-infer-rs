#!/usr/bin/env python3
"""Assemble and validate the Runtime vNext G00 historical bug corpus.

The production path consumes evidence that was already captured under an
external G00 artifact root. It does not execute a reproducer, download data, or
turn current positive tests into historical failure evidence. Missing receipts
produce an explicit INCOMPLETE artifact and a non-zero exit status.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
POLICY_PATH = REPO_ROOT / "scripts/release/configs/runtime_vnext_historical_corpus.json"
SELFTEST_FIXTURE_PATH = REPO_ROOT / "scripts/release/fixtures/runtime_vnext_historical_corpus_selftest.json"
SCHEMA_VERSION = 1
ARTIFACT_TYPE = "runtime_vnext_historical_bug_corpus"
RECEIPT_TYPE = "runtime_vnext_historical_case_evidence"
PASS_PREFIX = "FERRUM RUNTIME VNEXT G00 HISTORICAL CORPUS PASS"
INCOMPLETE_PREFIX = "FERRUM RUNTIME VNEXT G00 HISTORICAL CORPUS INCOMPLETE"
FAIL_PREFIX = "FERRUM RUNTIME VNEXT G00 HISTORICAL CORPUS FAIL"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT G00 HISTORICAL CORPUS SELFTEST PASS"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
FAMILY_RE = re.compile(r"H(0[1-9]|1[0-5])")
CASE_RE = re.compile(r"(H(?:0[1-9]|1[0-5]))\.([1-9][0-9]*)")


class CorpusError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CorpusError(message)


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_json_object,
            parse_constant=reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise CorpusError(f"cannot read strict JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CorpusError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write(path, canonical_json_bytes(value))


def require_object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_list(value: Any, label: str, *, nonempty: bool = False) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    if nonempty:
        require(bool(value), f"{label} must not be empty")
    return value


def require_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be a non-empty string")
    return value.strip()


def require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    if minimum is not None:
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


def require_exact_keys(value: dict[str, Any], required: set[str], optional: set[str], label: str) -> None:
    missing = required - set(value)
    extra = set(value) - required - optional
    require(not missing, f"{label} missing fields: {sorted(missing)}")
    require(not extra, f"{label} has unknown fields: {sorted(extra)}")


def parse_timestamp(value: Any, label: str) -> datetime:
    text = require_string(value, label)
    require(text.endswith("Z"), f"{label} must use UTC Z notation")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise CorpusError(f"{label} is not a valid RFC3339 timestamp: {text}") from exc
    require(parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0), f"{label} must be UTC")
    return parsed


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def safe_relative(raw: Any, label: str) -> Path:
    text = require_string(raw, label)
    path = Path(text)
    require(not path.is_absolute() and ".." not in path.parts, f"{label} must stay relative: {text}")
    require(path.as_posix() == text, f"{label} must use canonical POSIX separators: {text}")
    return path


def regular_file(root: Path, raw: Any, label: str) -> Path:
    relative = safe_relative(raw, label)
    candidate = root / relative
    root_resolved = root.resolve()
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise CorpusError(f"{label} escapes its root: {relative}") from exc
    require(candidate.is_file(), f"{label} is missing: {relative}")
    require(not candidate.is_symlink(), f"{label} must not be a symlink: {relative}")
    return candidate


def repo_file(repo_root: Path, raw: Any, label: str) -> Path:
    return regular_file(repo_root, raw, label)


def artifact_file(artifact_root: Path, raw: Any, label: str) -> Path:
    return regular_file(artifact_root, raw, label)


def file_ref(path: Path, root: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"file reference is not a regular file: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def verify_file_ref(path: Path, raw: dict[str, Any], label: str) -> None:
    digest = require_sha256(raw.get("sha256"), f"{label}.sha256")
    size = require_int(raw.get("size_bytes"), f"{label}.size_bytes", minimum=1)
    require(path.stat().st_size == size, f"{label} size mismatch")
    require(sha256_file(path) == digest, f"{label} SHA256 mismatch")


def external_artifact_root(path: Path) -> Path:
    root = path.expanduser().resolve(strict=False)
    try:
        root.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return root
    raise CorpusError(f"G00 artifact root must be outside the source worktree: {root}")


def run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


_VERIFIED_COMMITS: set[tuple[Path, str]] = set()


def verify_commit(repo_root: Path, sha: str, label: str) -> None:
    key = (repo_root.resolve(), sha)
    if key in _VERIFIED_COMMITS:
        return
    require(run_git(repo_root, "cat-file", "-t", sha) == "commit", f"{label} does not resolve to a commit: {sha}")
    _VERIFIED_COMMITS.add(key)


@dataclass(frozen=True)
class Context:
    repo_root: Path
    policy_path: Path
    policy: dict[str, Any]
    catalog_path: Path
    catalog: dict[str, Any]
    cases: dict[str, dict[str, Any]]
    family_cases: dict[str, tuple[str, ...]]
    baseline_tree_sha: str
    allow_selftest: bool


def locked_repo_file(repo_root: Path, raw: Any, label: str) -> Path:
    lock = require_object(raw, label)
    require_exact_keys(lock, {"path", "sha256"}, set(), label)
    path = repo_file(repo_root, lock.get("path"), f"{label}.path")
    expected = require_sha256(lock.get("sha256"), f"{label}.sha256")
    require(sha256_file(path) == expected, f"{label} content is stale: {lock.get('path')}")
    return path


def load_context(
    *,
    repo_root: Path = REPO_ROOT,
    policy_path: Path = POLICY_PATH,
    allow_selftest: bool = False,
) -> Context:
    policy = read_json(policy_path)
    require_exact_keys(
        policy,
        {
            "schema_version",
            "artifact_type",
            "baseline_git_sha",
            "catalog",
            "goal_documents",
            "expected_family_ids",
            "expected_concrete_case_count",
            "artifact_file",
            "receipt_glob",
            "receipt_template",
            "required_source_evidence_kinds",
            "historical_artifact_role",
            "historical_artifact_roots",
            "allowed_failure_layers",
            "allowed_mutation_kinds",
            "allowed_runner_basenames",
            "freshness",
        },
        set(),
        "historical corpus policy",
    )
    require(policy.get("schema_version") == SCHEMA_VERSION, "historical corpus policy schema_version mismatch")
    require(policy.get("artifact_type") == "runtime_vnext_historical_corpus_policy", "historical corpus policy artifact_type mismatch")
    baseline_sha = require_git_sha(policy.get("baseline_git_sha"), "historical corpus policy baseline_git_sha")
    verify_commit(repo_root, baseline_sha, "historical corpus baseline")
    baseline_tree_sha = run_git(repo_root, "rev-parse", f"{baseline_sha}^{{tree}}")
    require(GIT_SHA_RE.fullmatch(baseline_tree_sha) is not None, "historical corpus baseline tree SHA is invalid")

    catalog_path = locked_repo_file(repo_root, policy.get("catalog"), "historical corpus policy.catalog")
    goals = require_list(policy.get("goal_documents"), "historical corpus policy.goal_documents", nonempty=True)
    seen_goals: set[str] = set()
    for index, raw in enumerate(goals):
        goal = require_object(raw, f"historical corpus policy.goal_documents[{index}]")
        path = locked_repo_file(repo_root, goal, f"historical corpus policy.goal_documents[{index}]")
        relative = path.relative_to(repo_root).as_posix()
        require(relative not in seen_goals, f"duplicate historical corpus goal lock: {relative}")
        seen_goals.add(relative)

    expected_family_ids = require_list(policy.get("expected_family_ids"), "historical corpus policy.expected_family_ids", nonempty=True)
    require(all(isinstance(item, str) and FAMILY_RE.fullmatch(item) is not None for item in expected_family_ids), "historical corpus policy has invalid family ids")
    require(len(expected_family_ids) == len(set(expected_family_ids)) == 15, "historical corpus policy must contain 15 unique family ids")
    require(expected_family_ids == [f"H{index:02d}" for index in range(1, 16)], "historical corpus policy family order must be H01-H15")
    require_int(policy.get("expected_concrete_case_count"), "historical corpus policy.expected_concrete_case_count", minimum=1)
    require(policy.get("artifact_file") == "historical-bug-corpus.json", "historical corpus artifact_file mismatch")
    require(policy.get("receipt_glob") == "historical-bugs/**/evidence.json", "historical corpus receipt_glob mismatch")
    require(policy.get("receipt_template") == "historical-bugs/{case_id}/evidence.json", "historical corpus receipt_template mismatch")
    required_kinds = require_list(policy.get("required_source_evidence_kinds"), "historical corpus required evidence kinds", nonempty=True)
    require(set(required_kinds) == {"commit", "artifact"}, "historical corpus must require commit and artifact evidence")
    require(policy.get("historical_artifact_role") == "historical_failure", "historical corpus artifact role mismatch")
    historical_roots = require_list(policy.get("historical_artifact_roots"), "historical corpus artifact roots", nonempty=True)
    require(
        historical_roots == ["docs/bench", "docs/goals", "docs/release", "docs/status"],
        "historical corpus artifact roots mismatch",
    )
    require(set(require_list(policy.get("allowed_failure_layers"), "historical corpus failure layers")) == {f"L{index}" for index in range(6)}, "historical corpus failure layers must be L0-L5")
    require(set(require_list(policy.get("allowed_mutation_kinds"), "historical corpus mutation kinds")) == {"revert_patch", "fault_injection", "frozen_bad_input"}, "historical corpus mutation kinds mismatch")
    require(set(require_list(policy.get("allowed_runner_basenames"), "historical corpus runner basenames")) == {"cargo", "ferrum", "python", "python3"}, "historical corpus runner allowlist mismatch")
    freshness = require_object(policy.get("freshness"), "historical corpus policy.freshness")
    require_exact_keys(freshness, {"mode", "max_clock_skew_seconds", "require_empty_invalidated_by"}, set(), "historical corpus policy.freshness")
    require(freshness.get("mode") == "content_addressed", "historical corpus freshness mode mismatch")
    require_int(freshness.get("max_clock_skew_seconds"), "historical corpus max clock skew", minimum=0)
    require(freshness.get("require_empty_invalidated_by") is True, "historical corpus must reject invalidated evidence")

    catalog = read_json(catalog_path)
    require(catalog.get("schema_version") == SCHEMA_VERSION, "historical bug catalog schema_version mismatch")
    require(catalog.get("baseline_git_sha") == baseline_sha, "historical bug catalog baseline SHA mismatch")
    require(catalog.get("family_count") == 15, "historical bug catalog family_count must be 15")
    require(catalog.get("concrete_case_count") == policy.get("expected_concrete_case_count"), "historical bug catalog concrete case denominator is stale")
    catalog_families = require_list(catalog.get("families"), "historical bug catalog.families", nonempty=True)
    require(len(catalog_families) == 15, "historical bug catalog must have 15 family rows")

    cases: dict[str, dict[str, Any]] = {}
    family_cases: dict[str, tuple[str, ...]] = {}
    seen_families: set[str] = set()
    for family_index, family_raw in enumerate(catalog_families):
        family = require_object(family_raw, f"historical bug catalog.families[{family_index}]")
        family_id = require_string(family.get("id"), f"historical bug catalog.families[{family_index}].id")
        require(FAMILY_RE.fullmatch(family_id) is not None, f"invalid historical bug family id: {family_id}")
        require(family_id not in seen_families, f"duplicate historical bug family: {family_id}")
        seen_families.add(family_id)
        family_case_ids: list[str] = []
        for case_index, case_raw in enumerate(require_list(family.get("cases"), f"historical bug catalog.{family_id}.cases", nonempty=True)):
            case = require_object(case_raw, f"historical bug catalog.{family_id}.cases[{case_index}]")
            case_id = require_string(case.get("id"), f"historical bug catalog.{family_id}.cases[{case_index}].id")
            match = CASE_RE.fullmatch(case_id)
            require(match is not None and match.group(1) == family_id, f"historical bug case id is invalid or in the wrong family: {case_id}")
            require(case_id not in cases, f"duplicate historical bug catalog case: {case_id}")
            require_string(case.get("failure_class"), f"historical bug catalog.{case_id}.failure_class")
            require(case.get("evidence_status") in {"bound", "partial", "gap"}, f"historical bug catalog.{case_id}.evidence_status is invalid")
            entrypoints = require_list(case.get("entrypoints"), f"historical bug catalog.{case_id}.entrypoints", nonempty=True)
            backends = require_list(case.get("backends"), f"historical bug catalog.{case_id}.backends", nonempty=True)
            require(all(isinstance(item, str) and item for item in entrypoints), f"historical bug catalog.{case_id}.entrypoints is invalid")
            require(all(isinstance(item, str) and item for item in backends), f"historical bug catalog.{case_id}.backends is invalid")
            for commit_index, commit_raw in enumerate(require_list(case.get("commits", []), f"historical bug catalog.{case_id}.commits")):
                commit = require_object(commit_raw, f"historical bug catalog.{case_id}.commits[{commit_index}]")
                sha = require_git_sha(commit.get("sha"), f"historical bug catalog.{case_id}.commits[{commit_index}].sha")
                require_string(commit.get("relation"), f"historical bug catalog.{case_id}.commits[{commit_index}].relation")
                verify_commit(repo_root, sha, f"historical bug catalog.{case_id} commit")
            for field in ("historical_artifacts", "reproducer_paths"):
                for ref_index, ref in enumerate(require_list(case.get(field, []), f"historical bug catalog.{case_id}.{field}")):
                    repo_file(repo_root, ref, f"historical bug catalog.{case_id}.{field}[{ref_index}]")
            cases[case_id] = case
            family_case_ids.append(case_id)
        family_cases[family_id] = tuple(family_case_ids)

    require(list(family_cases) == expected_family_ids, "historical bug catalog family coverage/order mismatch")
    require(len(cases) == policy.get("expected_concrete_case_count"), "historical bug catalog concrete case coverage mismatch")
    return Context(
        repo_root=repo_root,
        policy_path=policy_path,
        policy=policy,
        catalog_path=catalog_path,
        catalog=catalog,
        cases=cases,
        family_cases=family_cases,
        baseline_tree_sha=baseline_tree_sha,
        allow_selftest=allow_selftest,
    )


def receipt_binding_sha256(receipt: dict[str, Any]) -> str:
    bound = copy.deepcopy(receipt)
    freshness = require_object(bound.get("freshness"), "evidence receipt.freshness")
    freshness.pop("binding_sha256", None)
    return canonical_json_sha256(bound)


def collector_identity(context: Context) -> dict[str, Any]:
    status = run_git(context.repo_root, "status", "--short").splitlines()
    return {
        "git_sha": run_git(context.repo_root, "rev-parse", "HEAD"),
        "tree_sha": run_git(context.repo_root, "rev-parse", "HEAD^{tree}"),
        "dirty_status": {"is_dirty": bool(status), "status_short": status},
    }


def catalog_evidence_inventory(context: Context, case: dict[str, Any]) -> dict[str, Any]:
    commits = [
        {"sha": commit["sha"], "relation": commit["relation"]}
        for commit in case.get("commits", [])
    ]
    artifacts = [
        file_ref(repo_file(context.repo_root, path, "historical catalog artifact"), context.repo_root)
        for path in case.get("historical_artifacts", [])
    ]
    candidates = [
        file_ref(repo_file(context.repo_root, path, "historical catalog reproducer candidate"), context.repo_root)
        for path in case.get("reproducer_paths", [])
    ]
    return {
        "catalog_status": case["evidence_status"],
        "source_commits": commits,
        "historical_artifacts": artifacts,
        "reproducer_candidates": candidates,
        "gap": case.get("gap"),
        "candidate_paths_are_not_frozen_reproducer_evidence": True,
    }


def scan_receipts(context: Context, artifact_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    rows: dict[str, tuple[Path, dict[str, Any]]] = {}
    evidence_root = artifact_root / "historical-bugs"
    if evidence_root.exists():
        require(evidence_root.is_dir() and not evidence_root.is_symlink(), f"historical evidence root must be a regular directory: {evidence_root}")
        for candidate in evidence_root.rglob("*"):
            require(not candidate.is_symlink(), f"historical evidence tree contains a forbidden symlink: {candidate}")
    glob_pattern = require_string(context.policy.get("receipt_glob"), "historical corpus receipt_glob")
    for path in sorted(artifact_root.glob(glob_pattern)):
        require(path.is_file() and not path.is_symlink(), f"evidence receipt must be a regular file: {path}")
        receipt = read_json(path)
        case_id = require_string(receipt.get("case_id"), f"evidence receipt {path}.case_id")
        require(case_id not in rows, f"duplicate evidence receipt for {case_id}: {rows.get(case_id, (None,))[0]} and {path}")
        require(case_id in context.cases, f"orphan evidence receipt for unknown case {case_id}: {path}")
        rows[case_id] = (path, receipt)
    return rows


def validate_source_evidence(
    context: Context,
    case_id: str,
    catalog_case: dict[str, Any],
    raw: Any,
) -> list[dict[str, Any]]:
    rows = require_list(raw, f"historical evidence {case_id}.source_evidence", nonempty=True)
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    kinds: set[str] = set()
    has_historical_artifact = False
    catalog_commits = {
        (commit["sha"], commit["relation"])
        for commit in catalog_case.get("commits", [])
    }
    for index, item_raw in enumerate(rows):
        item = require_object(item_raw, f"historical evidence {case_id}.source_evidence[{index}]")
        kind = require_string(item.get("kind"), f"historical evidence {case_id}.source_evidence[{index}].kind")
        ref = require_string(item.get("ref"), f"historical evidence {case_id}.source_evidence[{index}].ref")
        pair = (kind, ref)
        require(pair not in seen, f"duplicate source evidence for {case_id}: {kind}:{ref}")
        seen.add(pair)
        kinds.add(kind)
        if kind == "commit":
            require_exact_keys(item, {"kind", "ref", "relation"}, set(), f"historical evidence {case_id}.source_evidence[{index}]")
            sha = require_git_sha(ref, f"historical evidence {case_id}.source_evidence[{index}].ref")
            relation = require_string(item.get("relation"), f"historical evidence {case_id}.source_evidence[{index}].relation")
            verify_commit(context.repo_root, sha, f"historical evidence {case_id} commit")
            if not context.allow_selftest:
                require(
                    (sha, relation) in catalog_commits,
                    f"historical evidence {case_id} source commit is not bound by the reviewed catalog",
                )
            normalized.append({"kind": "commit", "ref": sha, "relation": relation})
        elif kind == "artifact":
            require_exact_keys(
                item,
                {"kind", "ref", "sha256", "size_bytes", "evidence_role"},
                set(),
                f"historical evidence {case_id}.source_evidence[{index}]",
            )
            path = repo_file(context.repo_root, ref, f"historical evidence {case_id}.source_evidence[{index}].ref")
            verify_file_ref(path, item, f"historical evidence {case_id} source artifact")
            role = require_string(item.get("evidence_role"), f"historical evidence {case_id}.source_evidence[{index}].evidence_role")
            if not context.allow_selftest:
                require(
                    not ref.startswith("scripts/release/fixtures/") and "selftest" not in ref.lower() and "synthetic" not in ref.lower(),
                    f"historical evidence {case_id} uses fixture or synthetic source evidence",
                )
                require(
                    ref in catalog_case.get("historical_artifacts", []),
                    f"historical evidence {case_id} source artifact is not bound by the reviewed catalog",
                )
            if role == context.policy.get("historical_artifact_role"):
                if not context.allow_selftest:
                    require(
                        any(ref == root or ref.startswith(f"{root}/") for root in context.policy["historical_artifact_roots"]),
                        f"historical evidence {case_id} historical_failure artifact is outside reviewed evidence roots",
                    )
                has_historical_artifact = True
            normalized.append(
                {
                    "kind": "artifact",
                    "ref": ref,
                    "sha256": item["sha256"],
                    "size_bytes": item["size_bytes"],
                    "evidence_role": role,
                }
            )
        else:
            raise CorpusError(f"historical evidence {case_id} has unsupported source evidence kind: {kind}")

    require(kinds == set(context.policy.get("required_source_evidence_kinds", [])), f"historical evidence {case_id} must contain commit and artifact evidence")
    require(has_historical_artifact, f"historical evidence {case_id} lacks an artifact explicitly classified as historical_failure")
    for commit in catalog_case.get("commits", []):
        require(("commit", commit["sha"]) in seen, f"historical evidence {case_id} omits catalog commit {commit['sha']}")
    for artifact in catalog_case.get("historical_artifacts", []):
        require(("artifact", artifact) in seen, f"historical evidence {case_id} omits catalog artifact {artifact}")
    return normalized


def validate_reproducer(
    context: Context,
    artifact_root: Path,
    case_id: str,
    raw: Any,
) -> dict[str, Any]:
    reproducer = require_object(raw, f"historical evidence {case_id}.reproducer")
    require_exact_keys(
        reproducer,
        {
            "input",
            "mutation",
            "mutation_kind",
            "failure_log",
            "failure_signature",
            "command",
            "returncode",
            "started_at",
            "finished_at",
            "duration_sec",
            "expected_invariant",
        },
        set(),
        f"historical evidence {case_id}.reproducer",
    )

    checked: dict[str, tuple[Path, dict[str, Any]]] = {}
    for key, display in (("input", "reproducer input"), ("mutation", "reproducer mutation"), ("failure_log", "reproducer failure log")):
        ref = require_object(reproducer.get(key), f"historical evidence {case_id}.reproducer.{key}")
        require_exact_keys(ref, {"path", "sha256", "size_bytes"}, set(), f"historical evidence {case_id}.reproducer.{key}")
        path = artifact_file(artifact_root, ref.get("path"), f"historical evidence {case_id} {display}")
        verify_file_ref(path, ref, f"historical evidence {case_id} {display}")
        checked[key] = (path, ref)

    input_path, input_ref = checked["input"]
    mutation_path, mutation_ref = checked["mutation"]
    failure_path, failure_ref = checked["failure_log"]
    require(input_path != mutation_path, f"historical evidence {case_id} mutation must use a distinct file")
    require(input_ref["sha256"] != mutation_ref["sha256"], f"historical evidence {case_id} mutation must differ from input")
    require(reproducer.get("mutation_kind") in context.policy.get("allowed_mutation_kinds", []), f"historical evidence {case_id} mutation_kind is invalid")
    failure_signature = require_string(reproducer.get("failure_signature"), f"historical evidence {case_id}.reproducer.failure_signature")
    try:
        failure_text = failure_path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise CorpusError(f"historical evidence {case_id} failure log is not strict UTF-8: {exc}") from exc
    require(failure_signature in failure_text, f"historical evidence {case_id} failure signature is absent from failure log")

    command = require_list(reproducer.get("command"), f"historical evidence {case_id}.reproducer.command", nonempty=True)
    require(all(isinstance(part, str) and part for part in command), f"historical evidence {case_id} reproducer command must be argv")
    require(Path(command[0]).name in context.policy.get("allowed_runner_basenames", []), f"historical evidence {case_id} reproducer runner is not allowed")
    require(case_id in command, f"historical evidence {case_id} reproducer command does not bind case id")
    require(input_ref["path"] in command, f"historical evidence {case_id} reproducer command does not bind input")
    require(mutation_ref["path"] in command, f"historical evidence {case_id} reproducer command does not bind mutation")
    returncode = require_int(reproducer.get("returncode"), f"historical evidence {case_id}.reproducer.returncode")
    require(returncode != 0, f"historical evidence {case_id} reproducer returncode must be non-zero")
    started = parse_timestamp(reproducer.get("started_at"), f"historical evidence {case_id}.reproducer.started_at")
    finished = parse_timestamp(reproducer.get("finished_at"), f"historical evidence {case_id}.reproducer.finished_at")
    require(finished >= started, f"historical evidence {case_id} reproducer execution window is reversed")
    duration = reproducer.get("duration_sec")
    require(isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration >= 0, f"historical evidence {case_id}.reproducer.duration_sec is invalid")
    require(abs(float(duration) - (finished - started).total_seconds()) <= 0.01, f"historical evidence {case_id} reproducer duration does not match execution window")
    expected_invariant = require_string(reproducer.get("expected_invariant"), f"historical evidence {case_id}.reproducer.expected_invariant")
    return {
        "input_path": input_ref["path"],
        "input_sha256": input_ref["sha256"],
        "mutation_path": mutation_ref["path"],
        "mutation_sha256": mutation_ref["sha256"],
        "mutation_kind": reproducer["mutation_kind"],
        "failure_log": failure_ref["path"],
        "failure_log_sha256": failure_ref["sha256"],
        "failure_signature": failure_signature,
        "command": command,
        "returncode": returncode,
        "started_at": reproducer["started_at"],
        "finished_at": reproducer["finished_at"],
        "duration_sec": float(duration),
        "expected_invariant": expected_invariant,
    }


def validate_receipt(
    context: Context,
    artifact_root: Path,
    case_id: str,
    receipt_path: Path,
    receipt: dict[str, Any],
    *,
    validation_time: datetime,
) -> dict[str, Any]:
    require_exact_keys(
        receipt,
        {
            "schema_version",
            "artifact_type",
            "case_id",
            "family_id",
            "catalog_id",
            "catalog_sha256",
            "baseline_git_sha",
            "captured_at",
            "expected_failure_layer",
            "source_evidence",
            "reproducer",
            "freshness",
        },
        {"fixture_mode"},
        f"historical evidence {case_id}",
    )
    require(receipt.get("schema_version") == SCHEMA_VERSION, f"historical evidence {case_id} schema_version mismatch")
    require(receipt.get("artifact_type") == RECEIPT_TYPE, f"historical evidence {case_id} artifact_type mismatch")
    require(receipt.get("case_id") == case_id, f"historical evidence {case_id} case_id mismatch")
    family_id = CASE_RE.fullmatch(case_id).group(1)  # type: ignore[union-attr]
    require(receipt.get("family_id") == family_id, f"historical evidence {case_id} family_id mismatch")
    expected_relative = context.policy["receipt_template"].format(case_id=case_id)
    require(receipt_path.relative_to(artifact_root).as_posix() == expected_relative, f"historical evidence {case_id} receipt path mismatch")
    require(receipt.get("catalog_id") == context.catalog.get("catalog_id"), f"historical evidence {case_id} catalog id is stale")
    current_catalog_sha = sha256_file(context.catalog_path)
    require(receipt.get("catalog_sha256") == current_catalog_sha, f"historical evidence {case_id} catalog SHA256 is stale")
    require(receipt.get("baseline_git_sha") == context.policy.get("baseline_git_sha"), f"historical evidence {case_id} baseline SHA is stale")
    if receipt.get("fixture_mode") is True:
        require(context.allow_selftest, f"historical evidence {case_id} fixture evidence is forbidden in production")
    elif "fixture_mode" in receipt:
        require(receipt.get("fixture_mode") is False, f"historical evidence {case_id}.fixture_mode must be boolean")

    failure_layer = require_string(receipt.get("expected_failure_layer"), f"historical evidence {case_id}.expected_failure_layer")
    require(failure_layer in context.policy.get("allowed_failure_layers", []), f"historical evidence {case_id} failure layer is invalid")
    catalog_case = context.cases[case_id]
    if not context.allow_selftest:
        require(
            catalog_case.get("evidence_status") == "bound",
            f"historical evidence {case_id} catalog case must be reviewed and upgraded to bound before collection",
        )
    source_evidence = validate_source_evidence(context, case_id, catalog_case, receipt.get("source_evidence"))
    reproducer = validate_reproducer(context, artifact_root, case_id, receipt.get("reproducer"))

    captured = parse_timestamp(receipt.get("captured_at"), f"historical evidence {case_id}.captured_at")
    finished = parse_timestamp(reproducer.get("finished_at"), f"historical evidence {case_id}.reproducer.finished_at")
    require(captured >= finished, f"historical evidence {case_id} captured_at precedes reproducer completion")
    skew = int(context.policy["freshness"]["max_clock_skew_seconds"])
    require(captured <= validation_time + timedelta(seconds=skew), f"historical evidence {case_id} captured_at is in the future")
    freshness = require_object(receipt.get("freshness"), f"historical evidence {case_id}.freshness")
    require_exact_keys(
        freshness,
        {"mode", "catalog_sha256", "baseline_git_sha", "binding_sha256", "invalidated_by"},
        set(),
        f"historical evidence {case_id}.freshness",
    )
    require(freshness.get("mode") == "content_addressed", f"historical evidence {case_id} freshness mode mismatch")
    require(freshness.get("catalog_sha256") == current_catalog_sha, f"historical evidence {case_id} freshness catalog SHA256 is stale")
    require(freshness.get("baseline_git_sha") == context.policy.get("baseline_git_sha"), f"historical evidence {case_id} freshness baseline SHA is stale")
    require(freshness.get("invalidated_by") == [], f"historical evidence {case_id} has been invalidated")
    binding = require_sha256(freshness.get("binding_sha256"), f"historical evidence {case_id}.freshness.binding_sha256")
    require(binding == receipt_binding_sha256(receipt), f"historical evidence {case_id} freshness binding SHA256 mismatch")

    return {
        "id": case_id,
        "failure_class": catalog_case["failure_class"],
        "status": "frozen",
        "entrypoints": catalog_case["entrypoints"],
        "backends": catalog_case["backends"],
        "catalog_evidence": catalog_evidence_inventory(context, catalog_case),
        "expected_failure_layer": failure_layer,
        "source_evidence": source_evidence,
        "reproducer": reproducer,
        "evidence_receipt": {
            "path": receipt_path.relative_to(artifact_root).as_posix(),
            "sha256": sha256_file(receipt_path),
            "size_bytes": receipt_path.stat().st_size,
            "binding_sha256": binding,
        },
    }


def corpus_content_sha256(document: dict[str, Any]) -> str:
    value = copy.deepcopy(document)
    value.pop("content_sha256", None)
    return canonical_json_sha256(value)


def validate_corpus_partition(context: Context, document: dict[str, Any]) -> None:
    families = require_list(document.get("families"), "historical corpus.families")
    seen_families: set[str] = set()
    seen_cases: set[str] = set()
    for family_index, family_raw in enumerate(families):
        family = require_object(family_raw, f"historical corpus.families[{family_index}]")
        family_id = require_string(family.get("id"), f"historical corpus.families[{family_index}].id")
        require(family_id not in seen_families, f"duplicate historical corpus family: {family_id}")
        require(family_id in context.family_cases, f"orphan historical corpus family: {family_id}")
        seen_families.add(family_id)
        family_case_ids: set[str] = set()
        for case_index, case_raw in enumerate(require_list(family.get("cases"), f"historical corpus.{family_id}.cases")):
            case = require_object(case_raw, f"historical corpus.{family_id}.cases[{case_index}]")
            case_id = require_string(case.get("id"), f"historical corpus.{family_id}.cases[{case_index}].id")
            require(case_id not in seen_cases, f"duplicate historical corpus case: {case_id}")
            require(case_id in context.cases, f"orphan historical corpus case: {case_id}")
            require(CASE_RE.fullmatch(case_id).group(1) == family_id, f"historical corpus case is in the wrong family: {case_id}")  # type: ignore[union-attr]
            seen_cases.add(case_id)
            family_case_ids.add(case_id)
        require(
            family_case_ids == set(context.family_cases[family_id]),
            f"historical corpus {family_id} concrete case coverage mismatch",
        )
    require(seen_families == set(context.family_cases), "historical corpus family coverage mismatch")
    require(seen_cases == set(context.cases), "historical corpus concrete case coverage mismatch")


def build_corpus(
    context: Context,
    artifact_root: Path,
    *,
    assembled_at: str | None = None,
) -> dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    timestamp = assembled_at or now_iso()
    validation_time = datetime.now(timezone.utc)
    assembled_time = parse_timestamp(timestamp, "historical corpus assembled_at")
    skew = int(context.policy["freshness"]["max_clock_skew_seconds"])
    require(assembled_time <= validation_time + timedelta(seconds=skew), "historical corpus assembled_at is in the future")
    receipts = scan_receipts(context, artifact_root)
    collector = collector_identity(context)
    families: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []
    if collector["dirty_status"]["is_dirty"] and not context.allow_selftest:
        blockers.append(
            {
                "case_id": "collector",
                "code": "dirty_collector",
                "message": "collector worktree is dirty; complete evidence must be assembled from a clean committed tree",
            }
        )
    complete_case_count = 0
    complete_family_count = 0
    for family_id, case_ids in context.family_cases.items():
        family_rows: list[dict[str, Any]] = []
        family_complete = True
        for case_id in case_ids:
            case_blockers: list[str] = []
            if context.cases[case_id]["evidence_status"] != "bound" and not context.allow_selftest:
                message = (
                    f"catalog case remains {context.cases[case_id]['evidence_status']}; "
                    "review real evidence and upgrade it to bound before collection"
                )
                blockers.append({"case_id": case_id, "code": "catalog_not_bound", "message": message})
                case_blockers.append(message)
            receipt_row = receipts.get(case_id)
            if receipt_row is None:
                expected_path = context.policy["receipt_template"].format(case_id=case_id)
                message = f"missing evidence receipt: {expected_path}"
                blockers.append({"case_id": case_id, "code": "missing_receipt", "message": message})
                case_blockers.append(message)
                family_rows.append(
                    {
                        "id": case_id,
                        "failure_class": context.cases[case_id]["failure_class"],
                        "status": "incomplete",
                        "entrypoints": context.cases[case_id]["entrypoints"],
                        "backends": context.cases[case_id]["backends"],
                        "catalog_evidence": catalog_evidence_inventory(context, context.cases[case_id]),
                        "evidence_receipt": {"path": expected_path, "state": "missing"},
                        "blockers": case_blockers,
                    }
                )
                family_complete = False
                continue
            receipt_path, receipt = receipt_row
            family_rows.append(
                validate_receipt(
                    context,
                    artifact_root,
                    case_id,
                    receipt_path,
                    receipt,
                    validation_time=validation_time,
                )
            )
            complete_case_count += 1
        if family_complete:
            complete_family_count += 1
        families.append({"id": family_id, "status": "complete" if family_complete else "incomplete", "cases": family_rows})

    family_count = len(context.family_cases)
    concrete_case_count = len(context.cases)
    catalog_status_counts = {
        status_name: sum(1 for case in context.cases.values() if case["evidence_status"] == status_name)
        for status_name in ("bound", "partial", "gap")
    }
    blocker_counts_by_code = {
        code: sum(1 for blocker in blockers if blocker["code"] == code)
        for code in sorted({blocker["code"] for blocker in blockers})
    }
    status = "complete" if not blockers else "incomplete"
    input_refs = [
        file_ref(context.policy_path, context.repo_root),
        file_ref(context.catalog_path, context.repo_root),
        file_ref(SCRIPT_PATH, context.repo_root),
    ]
    for goal in context.policy["goal_documents"]:
        input_refs.append(file_ref(repo_file(context.repo_root, goal["path"], "historical corpus goal"), context.repo_root))
    input_refs = sorted(input_refs, key=lambda row: row["path"])
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": status,
        "assembled_at": timestamp,
        "source_git_sha": context.policy["baseline_git_sha"],
        "source_tree_sha": context.baseline_tree_sha,
        "dirty_status": {"is_dirty": False, "status_short": []},
        "collector": collector,
        "catalog_id": context.catalog["catalog_id"],
        "catalog_sha256": sha256_file(context.catalog_path),
        "policy_sha256": sha256_file(context.policy_path),
        "assembler": file_ref(SCRIPT_PATH, context.repo_root),
        "freshness": {
            "mode": "content_addressed",
            "inputs": input_refs,
            "inputs_fingerprint": canonical_json_sha256(input_refs),
            "invalidated_by": [],
        },
        "family_count": family_count,
        "complete_family_count": complete_family_count,
        "incomplete_family_count": family_count - complete_family_count,
        "concrete_case_count": concrete_case_count,
        "catalog_status_counts": catalog_status_counts,
        "complete_case_count": complete_case_count,
        "incomplete_case_count": concrete_case_count - complete_case_count,
        "orphan_case_count": 0,
        "duplicate_case_count": 0,
        "blocker_count": len(blockers),
        "blocker_counts_by_code": blocker_counts_by_code,
        "blockers": blockers,
        "families": families,
    }
    document["content_sha256"] = corpus_content_sha256(document)
    return document


def validate_corpus_document(context: Context, artifact_root: Path, document: dict[str, Any]) -> dict[str, Any]:
    require(document.get("schema_version") == SCHEMA_VERSION, "historical corpus schema_version mismatch")
    require(document.get("artifact_type") == ARTIFACT_TYPE, "historical corpus artifact_type mismatch")
    digest = require_sha256(document.get("content_sha256"), "historical corpus content_sha256")
    require(digest == corpus_content_sha256(document), "historical corpus content SHA256 mismatch")
    validate_corpus_partition(context, document)
    assembled_at = require_string(document.get("assembled_at"), "historical corpus assembled_at")
    expected = build_corpus(context, artifact_root, assembled_at=assembled_at)
    require(document == expected, "historical corpus does not match current catalog, receipts, hashes, or freshness inputs")
    require(document.get("family_count") == 15, "historical corpus family denominator must be 15")
    require(document.get("concrete_case_count") == context.policy.get("expected_concrete_case_count"), "historical corpus concrete case denominator mismatch")
    require(document.get("orphan_case_count") == 0, "historical corpus contains orphan cases")
    require(document.get("duplicate_case_count") == 0, "historical corpus contains duplicate cases")
    if document.get("status") == "complete":
        require(document.get("complete_family_count") == 15, "historical corpus complete family coverage must be 15/15")
        require(document.get("complete_case_count") == document.get("concrete_case_count"), "historical corpus concrete case coverage must be M/M")
        require(document.get("blocker_count") == 0 and document.get("blockers") == [], "complete historical corpus must not contain blockers")
    else:
        require(document.get("status") == "incomplete", "historical corpus status must be complete or incomplete")
        require_int(document.get("blocker_count"), "historical corpus blocker_count", minimum=1)
    return document


def output_path(context: Context, artifact_root: Path) -> Path:
    return artifact_root / require_string(context.policy.get("artifact_file"), "historical corpus artifact_file")


def assemble(context: Context, artifact_root: Path) -> dict[str, Any]:
    document = build_corpus(context, artifact_root)
    path = output_path(context, artifact_root)
    atomic_write_json(path, document)
    return validate_corpus_document(context, artifact_root, read_json(path))


def validate_existing(context: Context, artifact_root: Path) -> dict[str, Any]:
    path = output_path(context, artifact_root)
    require(path.is_file() and not path.is_symlink(), f"historical corpus artifact is missing: {path}")
    return validate_corpus_document(context, artifact_root, read_json(path))


def fixture_receipt(
    context: Context,
    artifact_root: Path,
    case_id: str,
    captured: datetime,
) -> dict[str, Any]:
    case = context.cases[case_id]
    case_dir = artifact_root / "historical-bugs" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    input_path = case_dir / "input.json"
    mutation_path = case_dir / "mutation.patch"
    failure_path = case_dir / "failure.log"
    input_path.write_text(json.dumps({"case_id": case_id, "fixture": True}, sort_keys=True) + "\n", encoding="utf-8")
    mutation_path.write_text(f"fixture mutation for {case_id}\n", encoding="utf-8")
    signature = f"fixture-{case_id}-failure"
    failure_path.write_text(signature + "\n", encoding="utf-8")
    source_evidence: list[dict[str, Any]] = []
    for commit in case.get("commits", []):
        source_evidence.append({"kind": "commit", "ref": commit["sha"], "relation": commit["relation"]})
    if not case.get("commits"):
        fallback = next(
            commit
            for catalog_case in context.cases.values()
            for commit in catalog_case.get("commits", [])
        )
        source_evidence.append({"kind": "commit", "ref": fallback["sha"], "relation": "fixture_only"})
    for artifact in case.get("historical_artifacts", []):
        path = repo_file(context.repo_root, artifact, f"self-test catalog artifact {case_id}")
        source_evidence.append(
            {
                "kind": "artifact",
                "ref": artifact,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "evidence_role": "catalog_reference",
            }
        )
    fixture_rel = SELFTEST_FIXTURE_PATH.relative_to(context.repo_root).as_posix()
    source_evidence.append(
        {
            "kind": "artifact",
            "ref": fixture_rel,
            "sha256": sha256_file(SELFTEST_FIXTURE_PATH),
            "size_bytes": SELFTEST_FIXTURE_PATH.stat().st_size,
            "evidence_role": "historical_failure",
        }
    )
    start = captured - timedelta(seconds=2)
    finish = captured - timedelta(seconds=1)
    input_ref = file_ref(input_path, artifact_root)
    mutation_ref = file_ref(mutation_path, artifact_root)
    failure_ref = file_ref(failure_path, artifact_root)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": RECEIPT_TYPE,
        "case_id": case_id,
        "family_id": CASE_RE.fullmatch(case_id).group(1),  # type: ignore[union-attr]
        "catalog_id": context.catalog["catalog_id"],
        "catalog_sha256": sha256_file(context.catalog_path),
        "baseline_git_sha": context.policy["baseline_git_sha"],
        "captured_at": captured.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "expected_failure_layer": "L0",
        "source_evidence": source_evidence,
        "reproducer": {
            "input": input_ref,
            "mutation": mutation_ref,
            "mutation_kind": "fault_injection",
            "failure_log": failure_ref,
            "failure_signature": signature,
            "command": ["python3", "fixture_runner.py", case_id, input_ref["path"], mutation_ref["path"]],
            "returncode": 1,
            "started_at": start.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "finished_at": finish.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "duration_sec": 1.0,
            "expected_invariant": "self-test fixture must be rejected by the fault injection",
        },
        "freshness": {
            "mode": "content_addressed",
            "catalog_sha256": sha256_file(context.catalog_path),
            "baseline_git_sha": context.policy["baseline_git_sha"],
            "binding_sha256": "0" * 64,
            "invalidated_by": [],
        },
        "fixture_mode": True,
    }
    receipt["freshness"]["binding_sha256"] = receipt_binding_sha256(receipt)
    return receipt


def create_selftest_root(context: Context, artifact_root: Path) -> None:
    captured = datetime.now(timezone.utc)
    for case_id in context.cases:
        receipt = fixture_receipt(context, artifact_root, case_id, captured)
        atomic_write_json(artifact_root / context.policy["receipt_template"].format(case_id=case_id), receipt)


def rewrite_receipt(path: Path, update: Any, *, recompute_binding: bool = True) -> None:
    receipt = read_json(path)
    update(receipt)
    if recompute_binding:
        receipt["freshness"]["binding_sha256"] = receipt_binding_sha256(receipt)
    atomic_write_json(path, receipt)


def apply_selftest_mutation(mutation_id: str, context: Context, artifact_root: Path) -> None:
    case_id = next(iter(context.cases))
    receipt_path = artifact_root / context.policy["receipt_template"].format(case_id=case_id)
    if mutation_id == "missing-receipt":
        receipt_path.unlink()
    elif mutation_id == "duplicate-case":
        duplicate = artifact_root / "historical-bugs" / "duplicate" / "evidence.json"
        duplicate.parent.mkdir(parents=True)
        shutil.copy2(receipt_path, duplicate)
    elif mutation_id == "orphan-case":
        orphan = artifact_root / "historical-bugs" / "H99.1" / "evidence.json"
        orphan.parent.mkdir(parents=True)
        receipt = read_json(receipt_path)
        receipt["case_id"] = "H99.1"
        receipt["family_id"] = "H99"
        receipt["freshness"]["binding_sha256"] = receipt_binding_sha256(receipt)
        atomic_write_json(orphan, receipt)
    elif mutation_id == "source-artifact-hash":
        def mutate_source(receipt: dict[str, Any]) -> None:
            row = next(item for item in receipt["source_evidence"] if item["kind"] == "artifact" and item["evidence_role"] == "historical_failure")
            row["sha256"] = "0" * 64

        rewrite_receipt(receipt_path, mutate_source)
    elif mutation_id == "reproducer-input-hash":
        receipt = read_json(receipt_path)
        path = artifact_root / receipt["reproducer"]["input"]["path"]
        payload = path.read_bytes()
        require(bool(payload), "self-test reproducer input is empty")
        replacement = b"X" if payload[:1] != b"X" else b"Y"
        path.write_bytes(replacement + payload[1:])
    elif mutation_id == "identical-mutation":
        receipt = read_json(receipt_path)
        input_path = artifact_root / receipt["reproducer"]["input"]["path"]
        mutation_path = artifact_root / receipt["reproducer"]["mutation"]["path"]
        mutation_path.write_bytes(input_path.read_bytes())

        def update_mutation(row: dict[str, Any]) -> None:
            row["reproducer"]["mutation"]["sha256"] = sha256_file(mutation_path)
            row["reproducer"]["mutation"]["size_bytes"] = mutation_path.stat().st_size

        rewrite_receipt(receipt_path, update_mutation)
    elif mutation_id == "missing-failure-signature":
        rewrite_receipt(receipt_path, lambda row: row["reproducer"].update({"failure_signature": "absent-signature"}))
    elif mutation_id == "success-returncode":
        rewrite_receipt(receipt_path, lambda row: row["reproducer"].update({"returncode": 0}))
    elif mutation_id == "stale-catalog":
        rewrite_receipt(receipt_path, lambda row: row.update({"catalog_sha256": "0" * 64}))
    elif mutation_id == "stale-binding":
        rewrite_receipt(receipt_path, lambda row: row["freshness"].update({"binding_sha256": "0" * 64}), recompute_binding=False)
    elif mutation_id == "future-timestamp":
        rewrite_receipt(receipt_path, lambda row: row.update({"captured_at": "2999-01-01T00:00:00Z"}))
    elif mutation_id == "duplicate-json-key":
        payload = receipt_path.read_text(encoding="utf-8")
        needle = '  "schema_version": 1,\n'
        require(needle in payload, "self-test receipt lacks schema_version line")
        receipt_path.write_text(payload.replace(needle, needle + needle, 1), encoding="utf-8")
    else:
        raise CorpusError(f"unknown self-test mutation: {mutation_id}")


def apply_selftest_corpus_mutation(mutation_id: str, context: Context, artifact_root: Path) -> None:
    path = output_path(context, artifact_root)
    document = read_json(path)
    if mutation_id == "corpus-missing-family":
        document["families"].pop()
    elif mutation_id == "corpus-duplicate-family":
        document["families"].append(copy.deepcopy(document["families"][0]))
    elif mutation_id == "corpus-missing-case":
        document["families"][0]["cases"].pop()
    elif mutation_id == "corpus-duplicate-case":
        document["families"][0]["cases"].append(copy.deepcopy(document["families"][0]["cases"][0]))
    elif mutation_id == "corpus-orphan-case":
        orphan = copy.deepcopy(document["families"][0]["cases"][0])
        orphan["id"] = "H99.1"
        document["families"][0]["cases"].append(orphan)
    else:
        raise CorpusError(f"unknown corpus self-test mutation: {mutation_id}")
    document["content_sha256"] = corpus_content_sha256(document)
    atomic_write_json(path, document)


def self_test() -> None:
    fixture = read_json(SELFTEST_FIXTURE_PATH)
    require(fixture.get("schema_version") == SCHEMA_VERSION, "historical corpus self-test fixture schema mismatch")
    require(fixture.get("artifact_type") == "runtime_vnext_historical_corpus_selftest_fixture", "historical corpus self-test fixture type mismatch")
    mutations = require_list(fixture.get("mutations"), "historical corpus self-test fixture.mutations", nonempty=True)
    expected_ids = {
        "missing-receipt",
        "duplicate-case",
        "orphan-case",
        "source-artifact-hash",
        "reproducer-input-hash",
        "identical-mutation",
        "missing-failure-signature",
        "success-returncode",
        "stale-catalog",
        "stale-binding",
        "future-timestamp",
        "duplicate-json-key",
        "corpus-missing-family",
        "corpus-duplicate-family",
        "corpus-missing-case",
        "corpus-duplicate-case",
        "corpus-orphan-case",
    }
    mutation_ids = [require_string(require_object(row, "self-test mutation").get("id"), "self-test mutation.id") for row in mutations]
    require(len(mutation_ids) == len(set(mutation_ids)) and set(mutation_ids) == expected_ids, "historical corpus self-test mutation registry is stale")
    context = load_context(allow_selftest=True)
    with tempfile.TemporaryDirectory(prefix="ferrum-vnext-historical-corpus-") as temporary:
        temporary_root = Path(temporary)
        base = temporary_root / "base"
        create_selftest_root(context, base)
        complete = assemble(context, base)
        require(complete.get("status") == "complete", "complete self-test corpus was not accepted")
        require(complete.get("complete_family_count") == 15, "self-test corpus did not cover 15/15 families")
        require(complete.get("complete_case_count") == len(context.cases), "self-test corpus did not cover M/M cases")
        production_context = load_context(allow_selftest=False)
        try:
            build_corpus(production_context, base)
        except CorpusError as exc:
            require("fixture evidence is forbidden" in str(exc), f"production fixture rejection returned the wrong error: {exc}")
        else:
            raise CorpusError("production mode accepted self-test evidence")

        for index, mutation_raw in enumerate(mutations):
            mutation = require_object(mutation_raw, f"self-test mutation[{index}]")
            mutation_id = require_string(mutation.get("id"), f"self-test mutation[{index}].id")
            expected_outcome = require_string(mutation.get("expected_outcome"), f"self-test mutation[{index}].expected_outcome")
            expected_message = require_string(mutation.get("expected_message"), f"self-test mutation[{index}].expected_message")
            case_root = temporary_root / f"case-{index:02d}-{mutation_id}"
            shutil.copytree(base, case_root)
            corpus_mutation = mutation_id.startswith("corpus-")
            if corpus_mutation:
                apply_selftest_corpus_mutation(mutation_id, context, case_root)
            else:
                apply_selftest_mutation(mutation_id, context, case_root)
            if expected_outcome == "incomplete":
                result = build_corpus(context, case_root)
                require(result.get("status") == "incomplete", f"self-test mutation {mutation_id} did not produce INCOMPLETE")
                messages = "\n".join(item["message"] for item in result.get("blockers", []))
                require(expected_message in messages, f"self-test mutation {mutation_id} returned the wrong blocker: {messages}")
                continue
            require(expected_outcome == "reject", f"self-test mutation {mutation_id} has invalid expected_outcome")
            try:
                if corpus_mutation:
                    validate_existing(context, case_root)
                else:
                    build_corpus(context, case_root)
            except CorpusError as exc:
                require(expected_message in str(exc), f"self-test mutation {mutation_id} returned the wrong error: {exc}")
            else:
                raise CorpusError(f"self-test mutation {mutation_id} was accepted")
    print(SELFTEST_PASS_LINE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, help="external G00 artifact root")
    parser.add_argument("--validate-only", action="store_true", help="validate without rewriting historical-bug-corpus.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.artifact_root is not None or args.validate_only:
            parser.error("--self-test does not accept --artifact-root or --validate-only")
    elif args.artifact_root is None:
        parser.error("--artifact-root is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            self_test()
            return 0
        artifact_root = external_artifact_root(args.artifact_root)
        context = load_context()
        document = validate_existing(context, artifact_root) if args.validate_only else assemble(context, artifact_root)
    except (OSError, ValueError, CorpusError) as exc:
        print(f"{FAIL_PREFIX}: {exc}", file=sys.stderr)
        return 1
    if document["status"] == "complete":
        print(f"{PASS_PREFIX}: {artifact_root}")
        return 0
    print(
        f"{INCOMPLETE_PREFIX}: {artifact_root}: "
        f"families={document['complete_family_count']}/{document['family_count']}; "
        f"cases={document['complete_case_count']}/{document['concrete_case_count']}; "
        f"blockers={document['blocker_count']} "
        f"{json.dumps(document['blocker_counts_by_code'], sort_keys=True, separators=(',', ':'))}; "
        f"artifact={output_path(context, artifact_root)}"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
