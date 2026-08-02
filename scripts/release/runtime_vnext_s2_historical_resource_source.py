#!/usr/bin/env python3
"""Collect the focused S2 historical-resource source evidence.

This gate deliberately reuses the content-addressed G00 historical corpus. It
executes only the seven current-source tests selected for H02.1/H12.1-H12.4;
actual Qwen3.5-4B CUDA product evidence is consumed by the later S2 aggregate.
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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import resource_invariant_gate as resource_gate
import runtime_vnext_historical_corpus as historical_corpus


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_PATH = (
    REPO_ROOT
    / "scripts/release/configs/runtime_vnext_s2_historical_resources.json"
)
BUG_CATALOG_PATH = (
    REPO_ROOT / "scripts/release/configs/runtime_vnext_historical_bugs.json"
)
REPLAY_CATALOG_PATH = (
    REPO_ROOT / "scripts/release/configs/runtime_vnext_historical_replays.json"
)
REPLAY_PATH = REPO_ROOT / "scripts/release/runtime_vnext_historical_replay.py"
RESOURCE_GATE_PATH = REPO_ROOT / "scripts/release/resource_invariant_gate.py"
BOUNDED_COMMAND_PATH = REPO_ROOT / "scripts/release/bounded_command.py"

CHECKPOINT_ID = "runtime-vnext-s2-historical-resource-source"
PASS_PREFIX = "FERRUM RUNTIME VNEXT S2 HISTORICAL RESOURCE SOURCE PASS"
FAIL_PREFIX = "FERRUM RUNTIME VNEXT S2 HISTORICAL RESOURCE SOURCE FAIL"
SELFTEST_PASS_LINE = (
    "FERRUM RUNTIME VNEXT S2 HISTORICAL RESOURCE SOURCE SELFTEST PASS"
)
MODEL = "Qwen/Qwen3.5-4B"
BACKEND = "cuda"
CASE_IDS = ("H02.1", "H12.1", "H12.2", "H12.3", "H12.4")
EXPECTED_PRODUCT_EVIDENCE = {
    "s1_cuda_capacity_pressure": (
        "FERRUM RUNTIME VNEXT S1 CUDA CAPACITY PRESSURE PASS"
    ),
    "s1_cuda_decode_capacity": (
        "FERRUM RUNTIME VNEXT S1 CUDA DECODE CAPACITY PASS"
    ),
    "s2_abort_release_matrix": (
        "FERRUM RUNTIME VNEXT S2 STREAM DISCONNECT PASS"
    ),
    "s2_multiturn_default_budget": (
        "FERRUM RUNTIME VNEXT S2 MULTITURN CONCURRENCY PASS"
    ),
}
EXPECTED_REPLAY_RC = 42
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(
        path.is_file() and not path.is_symlink(),
        f"{label} must be a regular file: {path}",
    )
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {raw}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read {label}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(
        result.returncode == 0,
        f"git {' '.join(args)} failed: {result.stderr.strip()}",
    )
    return result.stdout.strip()


def source_identity() -> dict[str, Any]:
    status = [line for line in git_text("status", "--short").splitlines() if line]
    require(not status, f"source checkout must be clean: {status}")
    git_sha = git_text("rev-parse", "HEAD")
    tree_sha = git_text("rev-parse", "HEAD^{tree}")
    require(GIT_SHA_RE.fullmatch(git_sha) is not None, "source git SHA is invalid")
    require(GIT_SHA_RE.fullmatch(tree_sha) is not None, "source tree SHA is invalid")
    return {
        "git_sha": git_sha,
        "git_tree_sha": tree_sha,
        "dirty_status": {"is_dirty": False, "status_short": []},
    }


def flatten_bug_cases(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    families = document.get("families")
    require(isinstance(families, list), "historical bug families missing")
    result: dict[str, dict[str, Any]] = {}
    for family in families:
        require(isinstance(family, dict), "historical bug family is invalid")
        cases = family.get("cases")
        require(isinstance(cases, list), "historical bug family cases missing")
        for case in cases:
            require(isinstance(case, dict), "historical bug case is invalid")
            case_id = case.get("id")
            require(
                isinstance(case_id, str) and case_id not in result,
                f"historical bug case is duplicate or invalid: {case_id}",
            )
            result[case_id] = case
    return result


def replay_cases(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = document.get("cases")
    require(isinstance(rows, list), "historical replay cases missing")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        require(isinstance(row, dict), "historical replay case is invalid")
        case_id = row.get("id")
        require(
            isinstance(case_id, str) and case_id not in result,
            f"historical replay case is duplicate or invalid: {case_id}",
        )
        result[case_id] = row
    return result


def validate_config_document(document: dict[str, Any]) -> dict[str, Any]:
    require(
        set(document)
        == {
            "schema_version",
            "checkpoint_id",
            "model",
            "backend",
            "case_count",
            "cases",
            "source_tests",
            "product_evidence",
        },
        "focused historical-resource config fields mismatch",
    )
    require(
        document.get("schema_version") == 1
        and document.get("checkpoint_id") == CHECKPOINT_ID
        and document.get("model") == MODEL
        and document.get("backend") == BACKEND,
        "focused historical-resource config identity mismatch",
    )
    cases = document.get("cases")
    tests = document.get("source_tests")
    require(
        isinstance(cases, list)
        and document.get("case_count") == len(cases) == len(CASE_IDS),
        "focused historical-resource case count mismatch",
    )
    require(isinstance(tests, list) and len(tests) == 7, "source test count mismatch")

    test_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(tests):
        require(
            isinstance(row, dict)
            and set(row)
            == {
                "id",
                "package",
                "target",
                "source_path",
                "test_name",
                "covers",
            },
            f"source test {index} fields mismatch",
        )
        test_id = row.get("id")
        require(
            isinstance(test_id, str) and test_id and test_id not in test_by_id,
            f"source test id is duplicate or invalid: {test_id}",
        )
        require(
            isinstance(row.get("package"), str)
            and row.get("target") == "lib"
            and isinstance(row.get("test_name"), str),
            f"source test identity is invalid: {test_id}",
        )
        covers = row.get("covers")
        require(
            isinstance(covers, list)
            and covers
            and all(case_id in CASE_IDS for case_id in covers),
            f"source test coverage is invalid: {test_id}",
        )
        source_path = row.get("source_path")
        require(
            isinstance(source_path, str)
            and source_path.startswith("crates/")
            and ".." not in Path(source_path).parts,
            f"source test path is invalid: {test_id}",
        )
        source = REPO_ROOT / source_path
        source_text = source.read_text(encoding="utf-8")
        function_name = str(row["test_name"]).rsplit("::", 1)[-1]
        require(
            re.search(rf"\bfn\s+{re.escape(function_name)}\s*\(", source_text)
            is not None,
            f"source test function is missing: {test_id}",
        )
        test_by_id[test_id] = row

    bug_by_id = flatten_bug_cases(read_json(BUG_CATALOG_PATH, "bug catalog"))
    replay_by_id = replay_cases(read_json(REPLAY_CATALOG_PATH, "replay catalog"))
    case_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(cases):
        require(
            isinstance(row, dict)
            and set(row)
            == {
                "id",
                "source_tests",
                "resource_invariant_scenarios",
                "product_evidence",
            },
            f"focused case {index} fields mismatch",
        )
        case_id = row.get("id")
        require(
            case_id == CASE_IDS[index] and case_id not in case_by_id,
            f"focused case identity/order mismatch at {index}",
        )
        require(case_id in bug_by_id and case_id in replay_by_id, f"unknown case {case_id}")
        bug = bug_by_id[case_id]
        require(bug.get("evidence_status") == "bound", f"case is not bound: {case_id}")
        require("cuda" in bug.get("backends", []), f"case lacks CUDA scope: {case_id}")
        source_test_ids = row.get("source_tests")
        require(
            isinstance(source_test_ids, list)
            and source_test_ids
            and len(source_test_ids) == len(set(source_test_ids))
            and all(test_id in test_by_id for test_id in source_test_ids),
            f"case source-test mapping is invalid: {case_id}",
        )
        reverse = [
            test_id
            for test_id, test in test_by_id.items()
            if case_id in test["covers"]
        ]
        require(source_test_ids == reverse, f"case/test coverage is asymmetric: {case_id}")
        resource_scenarios = row.get("resource_invariant_scenarios")
        require(
            isinstance(resource_scenarios, list)
            and resource_scenarios
            and len(resource_scenarios) == len(set(resource_scenarios))
            and set(resource_scenarios) <= resource_gate.REQUIRED_PASS_SCENARIOS,
            f"resource invariant mapping is invalid: {case_id}",
        )
        product_evidence = row.get("product_evidence")
        require(
            isinstance(product_evidence, list)
            and product_evidence
            and len(product_evidence) == len(set(product_evidence))
            and set(product_evidence) <= set(EXPECTED_PRODUCT_EVIDENCE),
            f"product evidence mapping is invalid: {case_id}",
        )
        case_by_id[case_id] = {
            **row,
            "failure_class": bug.get("failure_class"),
            "entrypoints": bug.get("entrypoints"),
            "backends": bug.get("backends"),
            "expected_failure_layer": replay_by_id[case_id].get(
                "expected_failure_layer"
            ),
            "mutation_kind": replay_by_id[case_id].get("mutation_kind"),
            "expected_invariant": replay_by_id[case_id].get("expected_invariant"),
        }

    require(
        document.get("product_evidence") == EXPECTED_PRODUCT_EVIDENCE,
        "product evidence PASS-prefix registry mismatch",
    )
    require(
        case_by_id["H02.1"]["product_evidence"]
        == ["s2_multiturn_default_budget"],
        "H02.1 must bind the actual default-budget product checkpoint",
    )
    require(
        "s2_abort_release_matrix" in case_by_id["H12.4"]["product_evidence"],
        "H12.4 must bind the actual abort/release product checkpoint",
    )
    return {
        "document": document,
        "cases": case_by_id,
        "tests": test_by_id,
    }


def validate_config() -> dict[str, Any]:
    return validate_config_document(read_json(CONFIG_PATH, "focused config"))


def copy_regular(source: Path, destination: Path) -> None:
    require(
        source.is_file() and not source.is_symlink(),
        f"historical evidence is not a regular file: {source}",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def validate_historical_cases(
    historical_root: Path,
    out: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    historical_root = historical_root.resolve(strict=True)
    require(historical_root.is_dir() and not historical_root.is_symlink(), "historical corpus root is invalid")
    corpus_path = historical_root / "historical-bug-corpus.json"
    corpus_document = read_json(corpus_path, "historical corpus")
    require(corpus_document.get("status") == "complete", "historical corpus is incomplete")
    require(
        corpus_document.get("catalog_sha256") == sha256_file(BUG_CATALOG_PATH),
        "historical corpus catalog binding is stale",
    )
    require(
        corpus_document.get("concrete_case_count") == 28
        and corpus_document.get("complete_case_count") == 28
        and corpus_document.get("incomplete_case_count") == 0,
        "historical corpus denominator is incomplete",
    )
    collector = corpus_document.get("collector")
    require(
        isinstance(collector, dict)
        and collector.get("dirty_status")
        == {"is_dirty": False, "status_short": []},
        "historical corpus was not captured from a clean checkout",
    )
    assembler = corpus_document.get("assembler")
    require(
        isinstance(assembler, dict)
        and assembler.get("path")
        == "scripts/release/runtime_vnext_historical_corpus.py"
        and assembler.get("sha256")
        == sha256_file(REPO_ROOT / assembler["path"]),
        "historical corpus assembler binding is stale",
    )
    freshness = corpus_document.get("freshness")
    require(isinstance(freshness, dict), "historical corpus freshness is missing")
    stale_full_inputs: list[dict[str, str]] = []
    for raw in freshness.get("inputs", []):
        require(isinstance(raw, dict), "historical corpus freshness input is invalid")
        relative = raw.get("path")
        require(
            isinstance(relative, str)
            and relative
            and ".." not in Path(relative).parts,
            "historical corpus freshness path is invalid",
        )
        path = REPO_ROOT / relative
        current_sha = sha256_file(path)
        if current_sha != raw.get("sha256"):
            stale_full_inputs.append(
                {
                    "path": relative,
                    "recorded_sha256": str(raw.get("sha256")),
                    "current_sha256": current_sha,
                }
            )

    frozen_cases: dict[str, dict[str, Any]] = {}
    families = corpus_document.get("families")
    require(isinstance(families, list), "historical corpus families are missing")
    for family in families:
        require(isinstance(family, dict), "historical corpus family is invalid")
        for case in family.get("cases", []):
            require(isinstance(case, dict), "historical corpus case is invalid")
            case_id = case.get("id")
            require(
                isinstance(case_id, str) and case_id not in frozen_cases,
                f"historical corpus case is duplicate or invalid: {case_id}",
            )
            frozen_cases[case_id] = case

    rows: list[dict[str, Any]] = []
    for case_id in CASE_IDS:
        source_dir = historical_root / "historical-bugs" / case_id
        target_dir = out / "historical" / case_id
        for filename in ("evidence.json", "input.json", "mutation.json", "failure.log"):
            copy_regular(source_dir / filename, target_dir / filename)
        evidence = read_json(target_dir / "evidence.json", f"{case_id} evidence")
        replay_input = read_json(target_dir / "input.json", f"{case_id} input")
        mutation = read_json(target_dir / "mutation.json", f"{case_id} mutation")
        expected = config["cases"][case_id]
        frozen = frozen_cases.get(case_id)
        require(
            isinstance(frozen, dict)
            and frozen.get("status") == "frozen"
            and frozen.get("failure_class") == expected["failure_class"]
            and frozen.get("expected_failure_layer")
            == expected["expected_failure_layer"],
            f"historical corpus case contract mismatch: {case_id}",
        )
        require(
            evidence.get("case_id")
            == replay_input.get("case_id")
            == mutation.get("case_id")
            == case_id,
            f"historical case identity mismatch: {case_id}",
        )
        require(
            replay_input.get("failure_class")
            == mutation.get("failure_class")
            == expected["failure_class"],
            f"historical failure class mismatch: {case_id}",
        )
        require(
            replay_input.get("catalog_sha256") == sha256_file(BUG_CATALOG_PATH)
            and replay_input.get("runner_sha256") == sha256_file(REPLAY_PATH),
            f"historical replay source binding is stale: {case_id}",
        )
        require(
            evidence.get("expected_failure_layer")
            == expected["expected_failure_layer"]
            and evidence.get("reproducer", {}).get("mutation_kind")
            == expected["mutation_kind"],
            f"historical replay contract mismatch: {case_id}",
        )
        require(
            evidence.get("catalog_sha256") == sha256_file(BUG_CATALOG_PATH)
            and evidence.get("freshness", {}).get("catalog_sha256")
            == sha256_file(BUG_CATALOG_PATH)
            and evidence.get("freshness", {}).get("mode")
            == "content_addressed"
            and evidence.get("freshness", {}).get("invalidated_by") == []
            and evidence.get("freshness", {}).get("binding_sha256")
            == historical_corpus.receipt_binding_sha256(evidence),
            f"historical receipt binding is stale: {case_id}",
        )
        frozen_receipt = frozen.get("evidence_receipt")
        require(
            isinstance(frozen_receipt, dict)
            and frozen_receipt.get("path")
            == f"historical-bugs/{case_id}/evidence.json"
            and frozen_receipt.get("sha256")
            == sha256_file(target_dir / "evidence.json")
            and frozen_receipt.get("binding_sha256")
            == evidence["freshness"]["binding_sha256"],
            f"historical corpus/receipt binding mismatch: {case_id}",
        )
        reproducer = evidence.get("reproducer")
        require(isinstance(reproducer, dict), f"historical reproducer is missing: {case_id}")
        for field, filename in (
            ("input", "input.json"),
            ("mutation", "mutation.json"),
            ("failure_log", "failure.log"),
        ):
            ref = reproducer.get(field)
            require(
                isinstance(ref, dict)
                and ref.get("path") == f"historical-bugs/{case_id}/{filename}"
                and ref.get("sha256") == sha256_file(target_dir / filename)
                and ref.get("size_bytes") == (target_dir / filename).stat().st_size,
                f"historical reproducer file binding mismatch: {case_id}/{field}",
            )
        for source in evidence.get("source_evidence", []):
            require(isinstance(source, dict), f"historical source evidence is invalid: {case_id}")
            if source.get("kind") == "commit":
                commit = source.get("ref")
                require(
                    isinstance(commit, str)
                    and GIT_SHA_RE.fullmatch(commit) is not None
                    and subprocess.run(
                        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                        cwd=REPO_ROOT,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    ).returncode
                    == 0,
                    f"historical fix commit is missing: {case_id}",
                )
            elif source.get("kind") == "artifact":
                relative = source.get("ref")
                require(
                    isinstance(relative, str)
                    and ".." not in Path(relative).parts,
                    f"historical source artifact path is invalid: {case_id}",
                )
                path = REPO_ROOT / relative
                require(
                    source.get("sha256") == sha256_file(path)
                    and source.get("size_bytes") == path.stat().st_size,
                    f"historical source artifact binding is stale: {case_id}",
                )
            else:
                raise GateError(f"historical source evidence kind is invalid: {case_id}")

        command = [
            sys.executable,
            str(REPLAY_PATH),
            "--case-id",
            case_id,
            "--input",
            str(target_dir / "input.json"),
            "--mutation",
            str(target_dir / "mutation.json"),
        ]
        started_at = iso_now()
        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
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
        stdout_path = target_dir / "current-replay.stdout.log"
        stderr_path = target_dir / "current-replay.stderr.log"
        write_text(stdout_path, result.stdout)
        write_text(stderr_path, result.stderr)
        signature = f"HISTORICAL_REPLAY_FAILURE:{case_id}:{expected['failure_class']}"
        require(
            result.returncode == EXPECTED_REPLAY_RC
            and result.stdout.splitlines().count(signature) == 1
            and not result.stderr,
            f"historical bad input was not killed as expected: {case_id}",
        )
        rows.append(
            {
                "case_id": case_id,
                "failure_class": expected["failure_class"],
                "expected_failure_layer": expected["expected_failure_layer"],
                "mutation_kind": expected["mutation_kind"],
                "failure_signature": signature,
                "returncode": result.returncode,
                "started_at": started_at,
                "finished_at": iso_now(),
                "duration_sec": time.monotonic() - started,
                "evidence": file_ref(target_dir / "evidence.json", out),
                "input": file_ref(target_dir / "input.json", out),
                "mutation": file_ref(target_dir / "mutation.json", out),
                "failure_log": file_ref(target_dir / "failure.log", out),
                "current_replay_stdout": file_ref(stdout_path, out),
                "current_replay_stderr": file_ref(stderr_path, out),
            }
        )
    return {
        "source_root": str(historical_root),
        "corpus": file_ref(corpus_path, historical_root),
        "full_corpus_policy_current": not stale_full_inputs,
        "full_corpus_stale_inputs": stale_full_inputs,
        "reuse_scope": list(CASE_IDS),
        "reuse_mode": "selected_content_addressed_receipts",
        "case_count": len(rows),
        "cases": rows,
    }


def run_resource_invariants(out: Path, expected_git_sha: str) -> dict[str, Any]:
    root = out / "resource-invariants"
    command = [sys.executable, str(RESOURCE_GATE_PATH), "--out", str(root)]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "RUST_TEST_THREADS": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path = out / "logs/resource-invariants.runner.stdout.log"
    stderr_path = out / "logs/resource-invariants.runner.stderr.log"
    write_text(stdout_path, result.stdout)
    write_text(stderr_path, result.stderr)
    expected_pass = f"{resource_gate.PASS_LINE}: {root}"
    require(
        result.returncode == 0
        and result.stdout.splitlines().count(expected_pass) == 1
        and not result.stderr,
        "resource invariant gate failed",
    )
    manifest = read_json(root / "gate.manifest.json", "resource invariant manifest")
    report = read_json(root / "invariant_report.json", "resource invariant report")
    require(
        manifest.get("status") == "pass"
        and manifest.get("pass_line") == expected_pass
        and manifest.get("git_sha") == expected_git_sha
        and manifest.get("git_dirty") is False
        and manifest.get("dirty_files") == [],
        "resource invariant source identity mismatch",
    )
    require(
        report.get("status") == "pass"
        and report.get("leaked_resources") == 0
        and report.get("underflow_count") == 0
        and report.get("silent_oom_count") == 0
        and report.get("panic_count") == 0,
        "resource invariant report contains a failure",
    )
    required = report.get("fixture_summary", {}).get("required_scenarios")
    require(
        required == sorted(resource_gate.REQUIRED_PASS_SCENARIOS),
        "resource invariant scenario denominator mismatch",
    )
    return {
        "pass_line": expected_pass,
        "manifest": file_ref(root / "gate.manifest.json", out),
        "report": file_ref(root / "invariant_report.json", out),
        "runner_stdout": file_ref(stdout_path, out),
        "runner_stderr": file_ref(stderr_path, out),
        "required_scenarios": required,
    }


def cargo_test_command(test: dict[str, Any]) -> list[str]:
    return [
        "cargo",
        "test",
        "-p",
        test["package"],
        "--lib",
        test["test_name"],
        "--",
        "--exact",
        "--test-threads=1",
        "--nocapture",
    ]


def run_source_test(test: dict[str, Any], out: Path) -> dict[str, Any]:
    test_id = test["id"]
    logs = out / "logs"
    receipt_path = logs / f"{test_id}.receipt.json"
    stdout_path = logs / f"{test_id}.stdout.log"
    stderr_path = logs / f"{test_id}.stderr.log"
    runner_stdout_path = logs / f"{test_id}.runner.stdout.log"
    runner_stderr_path = logs / f"{test_id}.runner.stderr.log"
    cargo_command = cargo_test_command(test)
    command = [
        sys.executable,
        str(BOUNDED_COMMAND_PATH),
        "--receipt",
        str(receipt_path),
        "--stdout-log",
        str(stdout_path),
        "--stderr-log",
        str(stderr_path),
        "--cwd",
        str(REPO_ROOT),
        "--wall-timeout-seconds",
        "900",
        "--max-processes",
        "24",
        "--max-group-threads",
        "96",
        "--max-per-process-threads",
        "32",
        "--sample-interval-seconds",
        "0.1",
        "--",
        *cargo_command,
    ]
    started_at = iso_now()
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "CARGO_BUILD_JOBS": "4",
            "RUST_TEST_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    write_text(runner_stdout_path, result.stdout)
    write_text(runner_stderr_path, result.stderr)
    receipt = read_json(receipt_path, f"{test_id} bounded receipt")
    require(
        result.returncode == 0
        and receipt.get("schema") == "ferrum.bounded-command-receipt.v1"
        and receipt.get("status") == "pass"
        and receipt.get("reason") == "command_completed"
        and receipt.get("rc") == 0
        and receipt.get("violation") is None
        and receipt.get("sampling_errors") == []
        and receipt.get("termination") == {"signals": [], "errors": []}
        and receipt.get("cleanup") == {"process_group_gone": True},
        f"bounded source test failed: {test_id}",
    )
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    require(
        stdout.splitlines().count(f"test {test['test_name']} ... ok") == 1,
        f"source test did not execute exactly once: {test_id}",
    )
    require(
        re.search(
            r"test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out;",
            stdout,
        )
        is not None,
        f"source test summary mismatch: {test_id}",
    )
    source_path = REPO_ROOT / test["source_path"]
    return {
        "id": test_id,
        "package": test["package"],
        "target": test["target"],
        "test_name": test["test_name"],
        "covers": test["covers"],
        "cargo_command": cargo_command,
        "started_at": started_at,
        "finished_at": iso_now(),
        "duration_sec": time.monotonic() - started,
        "source": {
            "path": test["source_path"],
            "sha256": sha256_file(source_path),
        },
        "receipt": file_ref(receipt_path, out),
        "stdout": file_ref(stdout_path, out),
        "stderr": file_ref(stderr_path, out),
        "runner_stdout": file_ref(runner_stdout_path, out),
        "runner_stderr": file_ref(runner_stderr_path, out),
        "peaks": receipt.get("peaks"),
        "cleanup": receipt.get("cleanup"),
    }


def artifact_tree(out: Path) -> dict[str, Any]:
    files = []
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name == "artifact_tree.json":
            continue
        files.append(file_ref(path, out))
    return {"schema_version": 1, "file_count": len(files), "files": files}


def prepare_output(out: Path) -> Path:
    out = out.resolve(strict=False)
    repo = REPO_ROOT.resolve(strict=True)
    require(out != repo and repo not in out.parents, "artifact output must be outside the repository")
    require(not out.exists(), f"artifact output already exists: {out}")
    out.mkdir(parents=True)
    return out


def collect(historical_root: Path, out: Path) -> int:
    started_at = iso_now()
    started = time.monotonic()
    try:
        out = prepare_output(out)
        config = validate_config()
        identity = source_identity()
        historical = validate_historical_cases(historical_root, out, config)
        resource = run_resource_invariants(out, identity["git_sha"])
        source_tests = [
            run_source_test(test, out) for test in config["tests"].values()
        ]
        require(source_identity() == identity, "source identity changed during collection")

        historical_by_id = {row["case_id"]: row for row in historical["cases"]}
        tests_by_id = {row["id"]: row for row in source_tests}
        case_rows = []
        for case_id in CASE_IDS:
            row = config["cases"][case_id]
            case_rows.append(
                {
                    **row,
                    "historical_replay": historical_by_id[case_id],
                    "source_test_receipts": [
                        tests_by_id[test_id] for test_id in row["source_tests"]
                    ],
                    "product_evidence_status": "pending_s2_aggregate",
                }
            )
        pass_line = f"{PASS_PREFIX}: {out}"
        manifest = {
            "schema_version": 1,
            "checkpoint_id": CHECKPOINT_ID,
            "status": "pass",
            "scope": list(CASE_IDS),
            "full_s2": False,
            "product_evidence_complete": False,
            "model": MODEL,
            "backend": BACKEND,
            "artifact_dir": str(out),
            "started_at": started_at,
            "finished_at": iso_now(),
            "duration_sec": time.monotonic() - started,
            "source_identity": identity,
            "inputs": {
                "collector": {
                    "path": SCRIPT_PATH.relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256_file(SCRIPT_PATH),
                },
                "config": {
                    "path": CONFIG_PATH.relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256_file(CONFIG_PATH),
                },
                "bug_catalog": {
                    "path": BUG_CATALOG_PATH.relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256_file(BUG_CATALOG_PATH),
                },
                "replay_catalog": {
                    "path": REPLAY_CATALOG_PATH.relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256_file(REPLAY_CATALOG_PATH),
                },
                "replay_runner": {
                    "path": REPLAY_PATH.relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256_file(REPLAY_PATH),
                },
                "resource_gate": {
                    "path": RESOURCE_GATE_PATH.relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256_file(RESOURCE_GATE_PATH),
                },
                "bounded_command": {
                    "path": BOUNDED_COMMAND_PATH.relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256_file(BOUNDED_COMMAND_PATH),
                },
            },
            "historical_corpus": historical,
            "resource_invariants": resource,
            "source_tests": source_tests,
            "cases": case_rows,
            "product_evidence_requirements": EXPECTED_PRODUCT_EVIDENCE,
            "does_not_prove": [
                "actual Qwen3.5-4B CUDA product behavior",
                "full S2",
                "full G02 historical matrix",
                "Metal",
                "performance",
                "release readiness",
            ],
            "pass_line": pass_line,
        }
        write_json(out / "manifest.json", manifest)
        write_text(out / "pass_line.txt", pass_line + "\n")
        write_json(out / "artifact_tree.json", artifact_tree(out))
        print(pass_line)
        return 0
    except (GateError, historical_corpus.CorpusError, OSError, UnicodeError, ValueError) as error:
        print(f"{FAIL_PREFIX}: {out.resolve(strict=False)}: {error}", file=sys.stderr)
        return 1


def expect_config_reject(document: dict[str, Any], expected: str) -> None:
    try:
        validate_config_document(document)
    except GateError:
        return
    raise GateError(f"hostile config unexpectedly passed: {expected}")


def self_test() -> None:
    config = validate_config()
    require(tuple(config["cases"]) == CASE_IDS, "self-test case order mismatch")
    require(len(config["tests"]) == 7, "self-test source-test count mismatch")
    fixture = resource_gate.run_fixture_selftest(resource_gate.DEFAULT_FIXTURES)
    require(
        fixture.get("scenario_count") == len(resource_gate.REQUIRED_PASS_SCENARIOS),
        "resource fixture denominator mismatch",
    )
    document = config["document"]

    missing_case = copy.deepcopy(document)
    missing_case["cases"].pop()
    missing_case["case_count"] -= 1
    expect_config_reject(missing_case, "missing case")

    asymmetric = copy.deepcopy(document)
    asymmetric["cases"][2]["source_tests"].pop()
    expect_config_reject(asymmetric, "asymmetric source coverage")

    stale_test = copy.deepcopy(document)
    stale_test["source_tests"][0]["test_name"] += "_missing"
    expect_config_reject(stale_test, "missing source test")

    wrong_h02_product = copy.deepcopy(document)
    wrong_h02_product["cases"][0]["product_evidence"] = [
        "s1_cuda_capacity_pressure"
    ]
    expect_config_reject(wrong_h02_product, "wrong H02 product evidence")

    wrong_prefix = copy.deepcopy(document)
    wrong_prefix["product_evidence"]["s2_abort_release_matrix"] = "FAKE PASS"
    expect_config_reject(wrong_prefix, "wrong product PASS prefix")
    print(SELFTEST_PASS_LINE)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-corpus", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.historical_corpus is not None or args.out is not None:
            parser.error("--self-test does not accept collection arguments")
    elif args.historical_corpus is None or args.out is None:
        parser.error("--historical-corpus and --out are required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        try:
            self_test()
            return 0
        except (GateError, OSError, UnicodeError, ValueError) as error:
            print(
                f"{SELFTEST_PASS_LINE.replace(' PASS', ' FAIL')}: {error}",
                file=sys.stderr,
            )
            return 1
    return collect(args.historical_corpus, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
