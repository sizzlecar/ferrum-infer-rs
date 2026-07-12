#!/usr/bin/env python3
"""Capture executed Runtime vNext historical replay receipts for G00."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import runtime_vnext_historical_corpus as corpus


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
REPLAY_PATH = REPO_ROOT / "scripts/release/runtime_vnext_historical_replay.py"
REPLAY_CATALOG_PATH = REPO_ROOT / "scripts/release/configs/runtime_vnext_historical_replays.json"
SCHEMA_VERSION = 1
REPLAY_CATALOG_TYPE = "runtime_vnext_historical_replay_catalog"
REPLAY_INPUT_TYPE = "runtime_vnext_historical_replay_input"
REPLAY_MUTATION_TYPE = "runtime_vnext_historical_replay_mutation"
EXPECTED_REPLAY_RC = 42
PASS_PREFIX = "FERRUM RUNTIME VNEXT G00 HISTORICAL CAPTURE PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT G00 HISTORICAL CAPTURE SELFTEST PASS"


class CaptureError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CaptureError(message)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value[:-1] + "+00:00")


def require_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be a non-empty string")
    return value.strip()


def require_object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label} field set mismatch")


def load_replay_cases(context: corpus.Context) -> tuple[str, dict[str, dict[str, Any]]]:
    document = corpus.read_json(REPLAY_CATALOG_PATH)
    exact_keys(
        document,
        {
            "schema_version",
            "artifact_type",
            "baseline_git_sha",
            "review_artifact",
            "case_count",
            "cases",
        },
        "historical replay catalog",
    )
    require(document.get("schema_version") == SCHEMA_VERSION, "historical replay catalog schema mismatch")
    require(document.get("artifact_type") == REPLAY_CATALOG_TYPE, "historical replay catalog type mismatch")
    require(
        document.get("baseline_git_sha") == context.policy.get("baseline_git_sha"),
        "historical replay baseline SHA mismatch",
    )
    review_artifact = require_string(document.get("review_artifact"), "historical replay review_artifact")
    corpus.repo_file(context.repo_root, review_artifact, "historical replay review artifact")
    rows = require_list(document.get("cases"), "historical replay cases")
    require(document.get("case_count") == len(rows) == len(context.cases), "historical replay case count mismatch")
    cases: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = require_object(raw, f"historical replay cases[{index}]")
        exact_keys(
            row,
            {
                "id",
                "expected_failure_layer",
                "mutation_kind",
                "source_type",
                "expected_invariant",
                "expected_summary",
                "observed_summary",
                "trace",
            },
            f"historical replay cases[{index}]",
        )
        case_id = require_string(row.get("id"), f"historical replay cases[{index}].id")
        require(case_id in context.cases and case_id not in cases, f"historical replay case is unknown or duplicate: {case_id}")
        catalog_case = context.cases[case_id]
        require(catalog_case.get("evidence_status") == "bound", f"historical replay catalog case is not bound: {case_id}")
        require(bool(catalog_case.get("commits")), f"historical replay case has no source commit: {case_id}")
        require(bool(catalog_case.get("historical_artifacts")), f"historical replay case has no source artifact: {case_id}")
        require(
            row.get("expected_failure_layer") in context.policy.get("allowed_failure_layers", []),
            f"historical replay failure layer is invalid: {case_id}",
        )
        require(
            row.get("mutation_kind") in context.policy.get("allowed_mutation_kinds", []),
            f"historical replay mutation kind is invalid: {case_id}",
        )
        require(
            row.get("source_type")
            in {"historical_artifact", "fix_diff_reconstruction", "regression_contract_mutation"},
            f"historical replay source type is invalid: {case_id}",
        )
        require_string(row.get("expected_invariant"), f"historical replay {case_id}.expected_invariant")
        require(bool(require_object(row.get("expected_summary"), f"historical replay {case_id}.expected_summary")), f"historical replay expected summary is empty: {case_id}")
        require(bool(require_object(row.get("observed_summary"), f"historical replay {case_id}.observed_summary")), f"historical replay observed summary is empty: {case_id}")
        trace = row.get("trace")
        require(isinstance(trace, (dict, list)) and bool(trace), f"historical replay trace is empty: {case_id}")
        if row.get("source_type") != "historical_artifact":
            require(
                review_artifact in catalog_case.get("historical_artifacts", []),
                f"reconstructed historical replay omits the reviewed reconstruction artifact: {case_id}",
            )
        cases[case_id] = row
    require(list(cases) == list(context.cases), "historical replay case order/coverage mismatch")
    return review_artifact, cases


def source_evidence(
    context: corpus.Context,
    case_id: str,
    review_artifact: str,
) -> list[dict[str, Any]]:
    case = context.cases[case_id]
    rows: list[dict[str, Any]] = []
    for commit in case.get("commits", []):
        rows.append({"kind": "commit", "ref": commit["sha"], "relation": commit["relation"]})
    artifacts = list(case.get("historical_artifacts", []))
    require(bool(artifacts), f"historical replay case has no artifact source evidence: {case_id}")
    historical_failure = review_artifact if review_artifact in artifacts else artifacts[0]
    for relative in artifacts:
        path = corpus.repo_file(context.repo_root, relative, f"historical replay {case_id} source artifact")
        rows.append(
            {
                "kind": "artifact",
                "ref": relative,
                "sha256": corpus.sha256_file(path),
                "size_bytes": path.stat().st_size,
                "evidence_role": "historical_failure" if relative == historical_failure else "catalog_reference",
            }
        )
    return rows


def source_refs(case: dict[str, Any]) -> list[str]:
    return [
        *[commit["sha"] for commit in case.get("commits", [])],
        *list(case.get("historical_artifacts", [])),
    ]


def capture_case(
    context: corpus.Context,
    artifact_root: Path,
    case_id: str,
    definition: dict[str, Any],
    review_artifact: str,
    *,
    replace: bool,
) -> None:
    case = context.cases[case_id]
    case_dir = artifact_root / "historical-bugs" / case_id
    receipt_path = case_dir / "evidence.json"
    if receipt_path.exists() and not replace:
        raise CaptureError(f"historical replay receipt already exists; pass --replace to recapture: {receipt_path}")
    case_dir.mkdir(parents=True, exist_ok=True)
    input_path = case_dir / "input.json"
    mutation_path = case_dir / "mutation.json"
    failure_path = case_dir / "failure.log"
    relative_input = input_path.relative_to(artifact_root).as_posix()
    relative_mutation = mutation_path.relative_to(artifact_root).as_posix()
    input_document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": REPLAY_INPUT_TYPE,
        "case_id": case_id,
        "failure_class": case["failure_class"],
        "baseline_git_sha": context.policy["baseline_git_sha"],
        "catalog_sha256": corpus.sha256_file(context.catalog_path),
        "runner_sha256": corpus.sha256_file(REPLAY_PATH),
        "source_commits": [commit["sha"] for commit in case["commits"]],
        "expected_invariant": definition["expected_invariant"],
        "expected_summary": definition["expected_summary"],
    }
    mutation_document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": REPLAY_MUTATION_TYPE,
        "case_id": case_id,
        "failure_class": case["failure_class"],
        "mutation_kind": definition["mutation_kind"],
        "source_type": definition["source_type"],
        "source_refs": source_refs(case),
        "observed_summary": definition["observed_summary"],
        "trace": definition["trace"],
    }
    corpus.atomic_write_json(input_path, input_document)
    corpus.atomic_write_json(mutation_path, mutation_document)
    command = [
        "python3",
        str(REPLAY_PATH),
        "--case-id",
        case_id,
        "--input",
        relative_input,
        "--mutation",
        relative_mutation,
    ]
    started_at = iso_now()
    proc = subprocess.run(
        command,
        cwd=artifact_root,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "LANG": "C",
            "LC_ALL": "C",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    finished_at = iso_now()
    signature = f"HISTORICAL_REPLAY_FAILURE:{case_id}:{case['failure_class']}"
    failure_text = (
        f"command={json.dumps(command, separators=(',', ':'), ensure_ascii=True)}\n"
        f"returncode={proc.returncode}\n"
        "--- stdout ---\n"
        f"{proc.stdout}"
        "--- stderr ---\n"
        f"{proc.stderr}"
    )
    corpus.atomic_write(failure_path, failure_text.encode("utf-8"))
    require(proc.returncode == EXPECTED_REPLAY_RC, f"historical replay returned {proc.returncode}, expected {EXPECTED_REPLAY_RC}: {case_id}")
    require(signature in failure_text, f"historical replay failure signature is missing: {case_id}")
    duration = (parse_iso(finished_at) - parse_iso(started_at)).total_seconds()
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": corpus.RECEIPT_TYPE,
        "case_id": case_id,
        "family_id": case_id.split(".", 1)[0],
        "catalog_id": context.catalog["catalog_id"],
        "catalog_sha256": corpus.sha256_file(context.catalog_path),
        "baseline_git_sha": context.policy["baseline_git_sha"],
        "captured_at": iso_now(),
        "expected_failure_layer": definition["expected_failure_layer"],
        "source_evidence": source_evidence(context, case_id, review_artifact),
        "reproducer": {
            "input": corpus.file_ref(input_path, artifact_root),
            "mutation": corpus.file_ref(mutation_path, artifact_root),
            "mutation_kind": definition["mutation_kind"],
            "failure_log": corpus.file_ref(failure_path, artifact_root),
            "failure_signature": signature,
            "command": command,
            "returncode": proc.returncode,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_sec": duration,
            "expected_invariant": definition["expected_invariant"],
        },
        "freshness": {
            "mode": "content_addressed",
            "catalog_sha256": corpus.sha256_file(context.catalog_path),
            "baseline_git_sha": context.policy["baseline_git_sha"],
            "binding_sha256": "0" * 64,
            "invalidated_by": [],
        },
    }
    receipt["freshness"]["binding_sha256"] = corpus.receipt_binding_sha256(receipt)
    corpus.atomic_write_json(receipt_path, receipt)


def capture_all(artifact_root: Path, *, replace: bool) -> dict[str, Any]:
    context = corpus.load_context()
    review_artifact, definitions = load_replay_cases(context)
    for case_id, definition in definitions.items():
        capture_case(
            context,
            artifact_root,
            case_id,
            definition,
            review_artifact,
            replace=replace,
        )
    document = corpus.assemble(context, artifact_root)
    require(document.get("status") == "complete", "historical replay capture did not produce a complete corpus")
    return document


def self_test() -> None:
    context = corpus.load_context()
    _, definitions = load_replay_cases(context)
    require(len(definitions) == 28, "historical replay self-test case count mismatch")
    result = subprocess.run(
        [sys.executable, str(REPLAY_PATH), "--self-test"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(
        result.returncode == 0
        and result.stdout.splitlines().count(
            "FERRUM RUNTIME VNEXT HISTORICAL REPLAY SELFTEST PASS"
        )
        == 1
        and not result.stderr,
        f"historical replay child self-test failed: {result.stderr or result.stdout}",
    )
    print(SELFTEST_PASS_LINE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.artifact_root is not None or args.replace:
            parser.error("--self-test does not accept capture arguments")
    elif args.artifact_root is None:
        parser.error("--artifact-root is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            self_test()
            return 0
        artifact_root = corpus.external_artifact_root(args.artifact_root)
        capture_all(artifact_root, replace=args.replace)
    except (OSError, UnicodeError, ValueError, corpus.CorpusError, CaptureError) as exc:
        print(f"FERRUM RUNTIME VNEXT G00 HISTORICAL CAPTURE FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"{PASS_PREFIX}: {artifact_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
