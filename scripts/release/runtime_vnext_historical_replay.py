#!/usr/bin/env python3
"""Execute one content-addressed Runtime vNext historical failure replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
INPUT_TYPE = "runtime_vnext_historical_replay_input"
MUTATION_TYPE = "runtime_vnext_historical_replay_mutation"
EXPECTED_RC = 42
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT HISTORICAL REPLAY SELFTEST PASS"
ALLOWED_MUTATION_KINDS = {"revert_patch", "fault_injection", "frozen_bad_input"}
ALLOWED_SOURCE_TYPES = {
    "historical_artifact",
    "fix_diff_reconstruction",
    "regression_contract_mutation",
}


class ReplayError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} must be a regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {raw}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ReplayError(f"invalid {label}: {exc}") from exc
    require(isinstance(value, dict), f"{label} must contain one JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label} field set mismatch")


def nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be a non-empty string")
    return value.strip()


def validate_summary(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict) and bool(value), f"{label} must be a non-empty object")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    require(len(encoded.encode("utf-8")) <= 64 * 1024, f"{label} exceeds 64 KiB")
    return value


def diff_values(expected: Any, observed: Any, path: str = "$") -> list[dict[str, Any]]:
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            return [{"path": path, "expected": expected, "observed": observed}]
        differences: list[dict[str, Any]] = []
        for key in sorted(expected):
            child = f"{path}.{key}"
            if key not in observed:
                differences.append({"path": child, "expected": expected[key], "observed": "<missing>"})
            else:
                differences.extend(diff_values(expected[key], observed[key], child))
        for key in sorted(set(observed) - set(expected)):
            differences.append({"path": f"{path}.{key}", "expected": "<absent>", "observed": observed[key]})
        return differences
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            return [{"path": path, "expected": expected, "observed": observed}]
        differences: list[dict[str, Any]] = []
        for index, expected_item in enumerate(expected):
            differences.extend(diff_values(expected_item, observed[index], f"{path}[{index}]"))
        return differences
    return [] if expected == observed else [{"path": path, "expected": expected, "observed": observed}]


def validate_documents(
    case_id: str,
    input_document: dict[str, Any],
    mutation_document: dict[str, Any],
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    exact_keys(
        input_document,
        {
            "schema_version",
            "artifact_type",
            "case_id",
            "failure_class",
            "baseline_git_sha",
            "catalog_sha256",
            "runner_sha256",
            "source_commits",
            "expected_invariant",
            "expected_summary",
        },
        "historical replay input",
    )
    exact_keys(
        mutation_document,
        {
            "schema_version",
            "artifact_type",
            "case_id",
            "failure_class",
            "mutation_kind",
            "source_type",
            "source_refs",
            "observed_summary",
            "trace",
        },
        "historical replay mutation",
    )
    require(
        input_document.get("schema_version") == mutation_document.get("schema_version") == SCHEMA_VERSION,
        "historical replay schema_version mismatch",
    )
    require(input_document.get("artifact_type") == INPUT_TYPE, "historical replay input type mismatch")
    require(mutation_document.get("artifact_type") == MUTATION_TYPE, "historical replay mutation type mismatch")
    require(
        input_document.get("case_id") == mutation_document.get("case_id") == case_id,
        "historical replay case id mismatch",
    )
    failure_class = nonempty_string(input_document.get("failure_class"), "historical replay failure_class")
    require(mutation_document.get("failure_class") == failure_class, "historical replay failure class mismatch")
    baseline = nonempty_string(input_document.get("baseline_git_sha"), "historical replay baseline_git_sha")
    require(len(baseline) == 40 and all(char in "0123456789abcdef" for char in baseline), "historical replay baseline SHA is invalid")
    for field in ("catalog_sha256", "runner_sha256"):
        digest = nonempty_string(input_document.get(field), f"historical replay {field}")
        require(len(digest) == 64 and all(char in "0123456789abcdef" for char in digest), f"historical replay {field} is invalid")
    require(input_document["runner_sha256"] == sha256_file(Path(__file__)), "historical replay runner SHA256 mismatch")
    commits = input_document.get("source_commits")
    require(isinstance(commits, list) and bool(commits), "historical replay source_commits must not be empty")
    require(
        all(isinstance(item, str) and len(item) == 40 for item in commits),
        "historical replay source_commits are invalid",
    )
    nonempty_string(input_document.get("expected_invariant"), "historical replay expected_invariant")
    mutation_kind = mutation_document.get("mutation_kind")
    require(mutation_kind in ALLOWED_MUTATION_KINDS, "historical replay mutation kind is invalid")
    require(mutation_document.get("source_type") in ALLOWED_SOURCE_TYPES, "historical replay source type is invalid")
    source_refs = mutation_document.get("source_refs")
    require(
        isinstance(source_refs, list)
        and bool(source_refs)
        and all(isinstance(item, str) and item for item in source_refs),
        "historical replay source_refs must not be empty",
    )
    trace = mutation_document.get("trace")
    require(isinstance(trace, (dict, list)) and bool(trace), "historical replay trace must not be empty")
    return (
        failure_class,
        mutation_kind,
        validate_summary(input_document.get("expected_summary"), "historical replay expected_summary"),
        validate_summary(mutation_document.get("observed_summary"), "historical replay observed_summary"),
    )


def run_replay(case_id: str, input_path: Path, mutation_path: Path) -> int:
    require(input_path.resolve() != mutation_path.resolve(), "historical replay input and mutation must be distinct")
    input_document = read_json(input_path, "historical replay input")
    mutation_document = read_json(mutation_path, "historical replay mutation")
    failure_class, mutation_kind, expected, observed = validate_documents(
        case_id, input_document, mutation_document
    )
    differences = diff_values(expected, observed)
    if not differences:
        print(f"HISTORICAL REPLAY NO FAILURE:{case_id}:{failure_class}")
        return 0
    signature = f"HISTORICAL_REPLAY_FAILURE:{case_id}:{failure_class}"
    print(signature)
    print(
        json.dumps(
            {
                "case_id": case_id,
                "failure_class": failure_class,
                "mutation_kind": mutation_kind,
                "difference_count": len(differences),
                "differences": differences,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return EXPECTED_RC


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-historical-replay-selftest-") as raw_root:
        root = Path(raw_root)
        input_path = root / "input.json"
        mutation_path = root / "mutation.json"
        input_document = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": INPUT_TYPE,
            "case_id": "H01.1",
            "failure_class": "selftest_failure",
            "baseline_git_sha": "1" * 40,
            "catalog_sha256": "2" * 64,
            "runner_sha256": sha256_file(Path(__file__)),
            "source_commits": ["3" * 40],
            "expected_invariant": "one terminal event is required",
            "expected_summary": {"terminal_count": 1},
        }
        mutation_document = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": MUTATION_TYPE,
            "case_id": "H01.1",
            "failure_class": "selftest_failure",
            "mutation_kind": "fault_injection",
            "source_type": "regression_contract_mutation",
            "source_refs": ["selftest"],
            "observed_summary": {"terminal_count": 0},
            "trace": [{"event": "output"}],
        }
        write_json(input_path, input_document)
        write_json(mutation_path, mutation_document)
        require(run_replay("H01.1", input_path, mutation_path) == EXPECTED_RC, "self-test mutation was not reproduced")
        mutation_document["observed_summary"] = {"terminal_count": 1}
        write_json(mutation_path, mutation_document)
        require(run_replay("H01.1", input_path, mutation_path) == 0, "self-test clean replay failed")
        input_document["runner_sha256"] = "0" * 64
        write_json(input_path, input_document)
        try:
            run_replay("H01.1", input_path, mutation_path)
        except ReplayError as exc:
            require("runner SHA256 mismatch" in str(exc), "self-test runner mutation failed for the wrong reason")
        else:
            raise ReplayError("self-test runner SHA mutation unexpectedly passed")
    print(SELFTEST_PASS_LINE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--mutation", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.case_id is not None or args.input is not None or args.mutation is not None:
            parser.error("--self-test does not accept replay arguments")
    elif args.case_id is None or args.input is None or args.mutation is None:
        parser.error("--case-id, --input, and --mutation are required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            self_test()
            return 0
        return run_replay(args.case_id, args.input, args.mutation)
    except (OSError, UnicodeError, ValueError, ReplayError) as exc:
        print(f"HISTORICAL REPLAY ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
