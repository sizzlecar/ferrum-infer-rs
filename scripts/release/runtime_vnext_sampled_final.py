#!/usr/bin/env python3
"""Package the frozen v0.8.0 sampled-final release evidence.

This is deliberately a producer, not a model runner.  It accepts evidence made
by the exact staged binary, revalidates its byte/source/model bindings, and
emits the three small manifests consumed by ``runtime_vnext_goal_gate.py``.

A focused C17 report remains a diagnostic KEEP artifact.  The producer never
changes that report.  Its derived correctness PASS means only that the
checked-in sampled-final plan explicitly selected the complete C17 partition
and that every referenced raw case passed the sampled contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import re
import statistics
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import runtime_vnext_goal_gate as goal


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
VERSION = "0.8.0"
COLLECTION_SCOPE = "sampled_final_regression"
G08_EXECUTION_CONTRACT = "g08-model-matrix-v1"
RECEIPT_TYPE = "runtime_vnext_candidate_build_receipt"
MODEL_KEYS = {
    "m1-qwen35-4b",
    "m2-qwen35-35b-a3b",
    "m3-qwen3-30b-a3b",
}
LLAMA_MODEL_KEY = "llama31-8b-compat"
BACKENDS = {"metal", "cuda"}
LLAMA_ENV_ALLOWLIST = {
    "PATH",
    "HOME",
    "TMPDIR",
    "HF_HOME",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "RUST_LOG",
    "RUST_BACKTRACE",
}
R2_FLOOR_CATALOG = (
    REPO_ROOT / "scripts/release/configs/runtime_vnext_r2_ferrum_floors.json"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
BLOCKER_RE = re.compile(
    r"(?:panic(?:ked)?|out of memory|\boom\b|<unk>|\[pad\]|"
    r"invalid utf-?8|mojibake|missing[^\n]*\[done\]|duplicate[^\n]*\[done\])",
    re.IGNORECASE,
)

PASS_PREFIXES = {
    "correctness": "FERRUM RUNTIME VNEXT SAMPLED FINAL CORRECTNESS PASS",
    "performance": "FERRUM RUNTIME VNEXT SAMPLED FINAL PERFORMANCE PASS",
    "llama-supplemental": (
        "FERRUM RUNTIME VNEXT SAMPLED LLAMA DENSE SUPPLEMENTAL PASS"
    ),
}


class SampledFinalError(RuntimeError):
    """The supplied raw evidence does not satisfy the frozen sample."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise SampledFinalError(message)


def require_object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return value


def require_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and value.strip() == value and value, f"{label} must be a non-empty trimmed string")
    return value


def require_sha256(value: Any, label: str) -> str:
    text = require_string(value, label)
    require(SHA256_RE.fullmatch(text) is not None, f"{label} must be a lowercase SHA256")
    return text


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SampledFinalError(f"cannot read {label} {path}: {error}") from error


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def artifact_ref(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    require(
        resolved.is_file() and not resolved.is_symlink(),
        f"artifact is not a regular non-symlink file: {resolved}",
    )
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def kind_ref(path: Path, root: Path, kind: str = "raw-json") -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "kind": kind,
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": file_sha256(resolved),
    }


def resolve_ref(
    raw: Any,
    label: str,
    *,
    root: Path,
    require_within_root: bool = False,
) -> tuple[dict[str, Any], Path]:
    ref = require_object(raw, label)
    require(
        set(ref)
        in (
            {"path", "sha256"},
            {"path", "sha256", "size_bytes"},
            {"kind", "path", "sha256"},
        ),
        f"{label} reference fields differ",
    )
    raw_path = Path(require_string(ref.get("path"), f"{label}.path")).expanduser()
    path = raw_path if raw_path.is_absolute() else root / raw_path
    path = path.resolve()
    if require_within_root:
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise SampledFinalError(f"{label} escapes its artifact root") from error
    require(
        path.is_file() and not path.is_symlink(),
        f"{label} is not a regular non-symlink file: {path}",
    )
    digest = require_sha256(ref.get("sha256"), f"{label}.sha256")
    require(file_sha256(path) == digest, f"{label} SHA256 mismatch")
    if "size_bytes" in ref:
        require(
            type(ref.get("size_bytes")) is int
            and ref["size_bytes"] == path.stat().st_size,
            f"{label}.size_bytes mismatch",
        )
    return artifact_ref(path), path


def normalize_source(raw: Any, label: str) -> dict[str, Any]:
    source = require_object(raw, label)
    require(
        set(source) == {"git_sha", "git_tree_sha", "dirty"}
        and re.fullmatch(r"[0-9a-f]{40}", str(source.get("git_sha", "")))
        and re.fullmatch(r"[0-9a-f]{40}", str(source.get("git_tree_sha", "")))
        and source.get("dirty") is False,
        f"{label} is not an exact clean source identity",
    )
    return copy.deepcopy(source)


def flat_source(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    dirty = value.get("dirty_status")
    require(
        dirty == {"is_dirty": False, "status_short": []},
        f"{label} dirty status differs",
    )
    return normalize_source(
        {
            "git_sha": value.get("source_git_sha"),
            "git_tree_sha": value.get("source_tree_sha"),
            "dirty": False,
        },
        label,
    )


def input_manifest(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "manifest.json"
    require(candidate.is_file(), f"manifest is missing: {candidate}")
    return candidate


def checked_sample_plan() -> dict[str, Any]:
    try:
        return goal.validate_sample_plan(REPO_ROOT)
    except Exception as error:
        raise SampledFinalError(f"checked-in sampled plan rejected: {error}") from error


def correctness_plan_selection(
    plan: dict[str, Any], model_key: str, backend: str
) -> dict[str, Any]:
    """Return a C17 selection after replaying the real G08 planner denominator."""

    require(model_key in MODEL_KEYS, f"unsupported C17 sampled model {model_key}")
    try:
        import runtime_vnext_baseline_scenarios as baseline

        profile = baseline.matrix_profile(G08_EXECUTION_CONTRACT, model_key)
        planner_case_count = baseline.minimum_case_count("C17", model_key, profile)
    except Exception as error:
        raise SampledFinalError(
            f"cannot derive {model_key} C17 denominator from the G08 planner: {error}"
        ) from error
    row = require_object(
        require_object(
            plan["manifest"].get("correctness"), "sample plan correctness"
        ).get(model_key),
        f"sample plan correctness {model_key}",
    ).get(backend)
    selection = require_object(row, f"sample plan correctness {model_key}/{backend}")
    require(
        selection
        == {
            "scenario_ids": ["C17"],
            "entrypoints": ["run", "serve"],
            "case_count": planner_case_count,
            "checks_per_case": 5,
            "comparison_count": planner_case_count * 5,
            "producer": "g08-focused-c17-v1",
            "raw_decision": "KEEP",
            "raw_formal_pass_allowed": False,
            "sample_selection_status": "pass",
        },
        f"sample plan {model_key}/{backend} does not match G08 profile {profile}",
    )
    require(
        planner_case_count % 6 == 0,
        f"G08 C17 denominator {planner_case_count} cannot form a 3x2 partition",
    )
    return {**copy.deepcopy(selection), "matrix_profile": profile}


def staged_context(path: Path, backend: str) -> dict[str, Any]:
    require(backend in BACKENDS, f"unsupported sampled backend {backend}")
    try:
        staged = goal.validate_staged_assets_manifest(path)
    except Exception as error:
        raise SampledFinalError(f"staged assets rejected: {error}") from error
    require(
        staged["manifest"].get("publish_release") is False,
        "sampled evidence must use publish_release=false staged assets",
    )
    return staged


def validate_receipt(
    path: Path,
    *,
    staged: dict[str, Any],
    backend: str,
) -> dict[str, Any]:
    receipt_path = input_manifest(path)
    receipt = require_object(read_json(receipt_path, "bind-staged receipt"), "bind-staged receipt")
    source = staged["release_candidate"]
    row = staged["assets"][backend]
    require(
        receipt.get("schema_version") == SCHEMA_VERSION
        and receipt.get("artifact_type") == RECEIPT_TYPE
        and receipt.get("status") == "pass"
        and receipt.get("execution_contract") == G08_EXECUTION_CONTRACT
        and receipt.get("build_mode") == "staged-release-asset"
        and receipt.get("release_version") == VERSION
        and receipt.get("source_git_sha") == source["git_sha"]
        and receipt.get("source_tree_sha") == source["git_tree_sha"]
        and receipt.get("dirty_status")
        == {"is_dirty": False, "status_short": []}
        and receipt.get("backend") == backend
        and receipt.get("binary_sha256") == row["binary"]["sha256"]
        and receipt.get("selected_staged_asset") == row,
        "bind-staged receipt source/backend/byte identity differs",
    )
    observations = require_object(
        receipt.get("source_observations"), "receipt source observations"
    )
    require(
        set(observations) == {"before", "after"}
        and observations["before"] == observations["after"]
        and flat_source(observations["before"], "receipt before source") == source,
        "bind-staged source changed while binding",
    )
    artifact_root = Path(
        require_string(receipt.get("artifact_root"), "receipt artifact_root")
    ).expanduser().resolve()
    require(artifact_root.is_dir(), "receipt artifact_root is missing")
    receipt_ref, _ = resolve_ref(
        receipt.get("staged_assets_manifest"),
        "receipt staged assets manifest",
        root=artifact_root,
    )
    require(
        receipt_ref["sha256"] == staged["ref"]["sha256"],
        "bind-staged receipt points to different staged assets",
    )
    binary_ref, binary_path = resolve_ref(
        receipt.get("binary_artifact"),
        "receipt staged binary",
        root=artifact_root,
        require_within_root=True,
    )
    require(
        binary_ref["sha256"] == row["binary"]["sha256"]
        and binary_path.stat().st_size == row["binary"]["size_bytes"],
        "extracted staged binary differs from tarball identity",
    )
    return {
        "path": receipt_path,
        "ref": artifact_ref(receipt_path),
        "manifest": receipt,
        "artifact_root": artifact_root,
        "binary_path": binary_path,
        "binary_ref": binary_ref,
    }


def raw_effective_config(
    path: Path,
    *,
    model_key: str,
    backend: str,
    staged: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    config = require_object(read_json(config_path, "typed effective config"), "typed effective config")
    source = (
        normalize_source(config.get("source"), "effective config source")
        if "source" in config
        else flat_source(config, "effective config source")
    )
    row = staged["assets"][backend]
    typed = require_object(
        config.get("typed_effective_config"), "typed_effective_config"
    )
    model_files = require_object(config.get("model_files"), "effective config model_files")
    require(
        source == staged["release_candidate"]
        and config.get("model_key") == model_key
        and config.get("backend") == backend
        and config.get("binary_sha256") == row["binary"]["sha256"]
        and typed
        and model_files
        and config.get("hardware_id") == receipt["manifest"].get("hardware_id"),
        "typed effective config source/model/backend/binary identity differs",
    )
    return {
        "path": config_path,
        "ref": artifact_ref(config_path),
        "manifest": config,
        "source": source,
        "typed": copy.deepcopy(typed),
        "model_files": copy.deepcopy(model_files),
        "model_files_sha256": canonical_json_sha256(model_files),
    }


def write_effective_wrapper(
    out: Path,
    *,
    model_key: str,
    backend: str,
    staged: dict[str, Any],
    effective: dict[str, Any],
) -> Path:
    path = out / "effective-config.json"
    write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_sampled_typed_effective_config",
            "status": "pass",
            "collection_scope": COLLECTION_SCOPE,
            "source": copy.deepcopy(staged["release_candidate"]),
            "model_key": model_key,
            "backend": backend,
            "binary_sha256": staged["assets"][backend]["binary"]["sha256"],
            "model_files": copy.deepcopy(effective["model_files"]),
            "typed_effective_config": copy.deepcopy(effective["typed"]),
            "raw_effective_config": effective["ref"],
        },
    )
    return path


def validate_case_artifact_refs(case: dict[str, Any], *, root: Path, label: str) -> None:
    artifacts = require_object(case.get("artifacts"), f"{label}.artifacts")
    require("effective_config" in artifacts, f"{label} lacks effective config evidence")
    for name, raw in artifacts.items():
        resolve_ref(
            raw,
            f"{label}.artifacts.{name}",
            root=root,
            require_within_root=True,
        )
    if "execution_envelope" in case:
        resolve_ref(
            case["execution_envelope"],
            f"{label}.execution_envelope",
            root=root,
            require_within_root=True,
        )


def validate_focused_c17(
    path: Path,
    *,
    model_key: str,
    backend: str,
    staged: dict[str, Any],
    receipt: dict[str, Any],
    effective: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    report_path = path.expanduser().resolve()
    report = require_object(read_json(report_path, "focused C17 report"), "focused C17 report")
    root = receipt["artifact_root"]
    try:
        report_path.relative_to(root)
    except ValueError as error:
        raise SampledFinalError("focused C17 report is outside bind-staged artifact_root") from error
    source = flat_source(report, "focused C17 source")
    row = staged["assets"][backend]
    scope = require_object(report.get("scope"), "focused C17 scope")
    cases = require_list(report.get("cases"), "focused C17 cases")
    selected_ids = require_list(
        scope.get("selected_case_ids"), "focused C17 selected case ids"
    )
    selection = correctness_plan_selection(plan, model_key, backend)
    expected_case_count = selection["case_count"]
    require(
        report.get("schema_version") == SCHEMA_VERSION
        and report.get("artifact_kind") == "runtime-vnext-focused-diagnostic"
        and report.get("execution_contract") == G08_EXECUTION_CONTRACT
        and report.get("status") == "pass"
        and report.get("decision") == "KEEP"
        and report.get("formal_pass_allowed") is False
        and source == staged["release_candidate"]
        and report.get("model_key") == model_key
        and report.get("backend") == backend
        and report.get("binary_sha256") == row["binary"]["sha256"]
        and report.get("model_files") == effective["model_files"]
        and report.get("hardware_id") == receipt["manifest"].get("hardware_id")
        and scope.get("kind") == "focused-diagnostic"
        and scope.get("requested_case_ids") == []
        and scope.get("requested_scenario_ids") == ["C17"]
        and scope.get("selected_scenario_ids") == ["C17"]
        and scope.get("case_count") == expected_case_count
        and type(scope.get("canonical_case_count")) is int
        and scope["canonical_case_count"] >= expected_case_count
        and len(cases) == len(selected_ids) == expected_case_count
        and len(set(selected_ids)) == expected_case_count
        and report.get("observed_status_counts") == {"pass": expected_case_count}
        and Path(str(report.get("artifact_path", ""))).resolve() == report_path
        and report.get("pass_line")
        == f"FERRUM RUNTIME VNEXT FOCUSED DIAGNOSTIC KEEP: {report_path}",
        "focused report is not the complete raw C17 KEEP partition",
    )
    report_effective, _ = resolve_ref(
        report.get("effective_config"),
        "focused report effective config",
        root=root,
        require_within_root=True,
    )
    report_receipt, _ = resolve_ref(
        report.get("binary_build_receipt"),
        "focused report bind-staged receipt",
        root=root,
        require_within_root=True,
    )
    require(
        report_effective["sha256"] == effective["ref"]["sha256"]
        and report_receipt["sha256"] == receipt["ref"]["sha256"],
        "focused report typed-config/receipt binding differs",
    )
    invocation_ref, invocation_path = resolve_ref(
        report.get("executor_invocation"),
        "focused executor invocation",
        root=root,
        require_within_root=True,
    )
    invocation = require_object(
        read_json(invocation_path, "focused executor invocation"),
        "focused executor invocation",
    )
    command_line = require_list(
        invocation.get("argv"), "focused executor command line"
    )
    require(
        invocation.get("mode") == "focused-diagnostic"
        and "--focus-scenario" in command_line
        and command_line[command_line.index("--focus-scenario") + 1] == "C17",
        "focused executor invocation is not explicitly scoped to C17",
    )

    partition: dict[tuple[str, str], int] = {}
    observed_ids: list[str] = []
    for ordinal, raw_ref in enumerate(cases, start=1):
        _, case_path = resolve_ref(
            raw_ref,
            f"focused C17 case[{ordinal}]",
            root=root,
            require_within_root=True,
        )
        case = require_object(read_json(case_path, f"C17 case {ordinal}"), f"C17 case {ordinal}")
        case_id = require_string(case.get("case_id"), f"C17 case {ordinal}.case_id")
        observed_ids.append(case_id)
        entrypoint = require_string(case.get("entrypoint"), f"{case_id}.entrypoint")
        variant = require_string(case.get("variant"), f"{case_id}.variant")
        observed = require_object(case.get("observed"), f"{case_id}.observed")
        observed_outcome = require_object(
            case.get("observed_outcome"), f"{case_id}.observed_outcome"
        )
        expected_outcome = require_object(
            case.get("expected_outcome"), f"{case_id}.expected_outcome"
        )
        checks = require_object(case.get("checks"), f"{case_id}.checks")
        require(
            case.get("schema_version") == SCHEMA_VERSION
            and case.get("scenario_id") == "C17"
            and case.get("ordinal") == ordinal
            and case.get("status") == "pass"
            and case.get("source_git_sha") == source["git_sha"]
            and case.get("source_tree_sha") == source["git_tree_sha"]
            and case.get("binary_sha256") == row["binary"]["sha256"]
            and case.get("model_key") == model_key
            and case.get("backend") == backend
            and case.get("model_files") == effective["model_files"]
            and entrypoint in {"run", "serve"}
            and variant in {"chinese", "emoji", "combining"}
            and observed_outcome == {"status": "pass", "failure_class": None}
            and expected_outcome.get("expected_status") == "pass"
            and checks
            and all(value is True for value in checks.values())
            and require_string(observed.get("expected_marker"), f"{case_id}.expected_marker")
            and all(
                observed.get(field, 0) == 0
                for field in (
                    "replacement_char_count",
                    "mojibake_count",
                    "partial_character_chunk_count",
                    "error_count",
                    "bad_output_count",
                )
            ),
            f"{case_id} failed its raw Unicode/source/model contract",
        )
        validate_case_artifact_refs(case, root=root, label=case_id)
        case_effective, _ = resolve_ref(
            case["artifacts"]["effective_config"],
            f"{case_id}.effective_config",
            root=root,
            require_within_root=True,
        )
        require(
            case_effective["sha256"] == effective["ref"]["sha256"],
            f"{case_id} used a different typed effective config",
        )
        partition[(variant, entrypoint)] = partition.get((variant, entrypoint), 0) + 1
    require(
        observed_ids == selected_ids
        and partition
        == {
            (variant, entrypoint): expected_case_count // 6
            for variant in ("chinese", "emoji", "combining")
            for entrypoint in ("run", "serve")
        },
        "focused C17 raw cases do not form the planner-derived 3x2 partition",
    )
    return {
        "path": report_path,
        "ref": artifact_ref(report_path),
        "manifest": report,
        "invocation": invocation_ref,
        "command_line": copy.deepcopy(command_line),
        "sample_count": expected_case_count,
        "comparison_count": selection["comparison_count"],
        "selection": selection,
    }


def command_parts(path: Path) -> tuple[dict[str, Any], list[str]]:
    command_path = path.expanduser().resolve()
    raw = read_json(command_path, "bench command")
    if isinstance(raw, list):
        parts = raw
        document: dict[str, Any] = {"command_line": copy.deepcopy(raw)}
    else:
        document = require_object(raw, "bench command")
        parts = document.get("command_line", document.get("cmd"))
    values = require_list(parts, "bench command line")
    require(
        len(values) >= 2 and all(isinstance(value, str) and value for value in values),
        "bench command line is malformed",
    )
    return {"path": command_path, "ref": artifact_ref(command_path), "manifest": document}, list(values)


def flag_value(parts: list[str], flag: str) -> str | None:
    indexes = [index for index, part in enumerate(parts) if part == flag]
    require(len(indexes) <= 1, f"bench command repeats {flag}")
    if not indexes:
        return None
    index = indexes[0]
    require(index + 1 < len(parts), f"bench command lacks a value for {flag}")
    return parts[index + 1]


def positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    require(
        type(value) is int and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return value


def validate_bench(
    report_path: Path,
    command_path: Path,
    *,
    planned: dict[str, Any],
    backend: str,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    path = report_path.expanduser().resolve()
    raw = read_json(path, "single-cell bench report")
    if isinstance(raw, list):
        require(len(raw) == 1, "bench report must contain exactly one selected cell")
        report = require_object(raw[0], "single bench cell")
    else:
        report = require_object(raw, "single bench cell")
    command, parts = command_parts(command_path)
    binary_path = Path(parts[0]).expanduser().resolve()
    require(
        binary_path == receipt["binary_path"]
        and parts[1] == "bench-serve"
        and "--fail-on-error" in parts
        and "--require-ci" in parts
        and "--request-rate" not in parts
        and not any(re.match(r"FERRUM_[A-Z0-9_]+=", part) for part in parts),
        "bench command did not use the exact staged ferrum bench-serve path/flags",
    )
    concurrency_text = flag_value(parts, "--concurrency")
    sweep_text = flag_value(parts, "--concurrency-sweep")
    require(
        (concurrency_text is None) != (sweep_text is None),
        "bench command must select one concurrency flag",
    )
    selected_text = concurrency_text if concurrency_text is not None else sweep_text
    require(
        selected_text == str(planned["concurrency"])
        and flag_value(parts, "--dataset") == planned["dataset"]
        and flag_value(parts, "--n-repeats") == str(planned["repeats"])
        and flag_value(parts, "--seed") == "9271",
        "bench command differs from the frozen sample cell",
    )
    if flag_value(parts, "--target-backend") is not None:
        require(
            flag_value(parts, "--target-backend") == backend,
            "bench command target backend differs",
        )
    requests = positive_int(report.get("n_requests_per_run"), "bench requests per run")
    completed = require_list(report.get("completed_per_run"), "completed_per_run")
    errored = require_list(report.get("errored_per_run"), "errored_per_run")
    repeats = require_list(report.get("repeat_metrics"), "repeat_metrics")
    require(
        report.get("scenario") == "closed_loop"
        and report.get("backend") == backend
        and report.get("concurrency") == planned["concurrency"]
        and report.get("n_repeats") == 3
        and report.get("output_token_count_source") == "usage"
        and completed == [requests, requests, requests]
        and errored == [0, 0, 0]
        and len(repeats) == 3,
        "bench report cell/repeat/completion contract differs",
    )
    if flag_value(parts, "--num-prompts") is not None:
        require(
            flag_value(parts, "--num-prompts") == str(requests),
            "bench command/report request denominator differs",
        )
    model = require_string(report.get("model"), "bench report model")
    if flag_value(parts, "--model") is not None:
        require(flag_value(parts, "--model") == model, "bench command/report model differs")
    n_gen = positive_int(report.get("n_gen"), "bench output length")
    if flag_value(parts, "--random-output-len") is not None:
        require(
            flag_value(parts, "--random-output-len") == str(n_gen),
            "bench command/report output length differs",
        )
    error_fields = (
        "bad_output_per_run",
        "malformed_stream_per_run",
        "missing_done_per_run",
        "duplicate_done_per_run",
        "zero_output_tokens_per_run",
        "stream_bulk_flush_per_run",
        "http_500_per_run",
        "panic_per_run",
    )
    require(
        all(report.get(field) == [0, 0, 0] for field in error_fields),
        "bench report contains a quality/error failure",
    )
    for index, repeat in enumerate(repeats, start=1):
        row = require_object(repeat, f"bench repeat {index}")
        require(
            row.get("repeat") == index
            and row.get("expected_requests") == requests
            and row.get("completed_requests") == requests
            and row.get("errored_requests") == 0
            and row.get("output_token_count_source") == "usage",
            f"bench repeat {index} completion/token-source differs",
        )
        for field in ("quality_issues", "warmup_quality_issues"):
            quality = require_object(row.get(field), f"bench repeat {index} {field}")
            require(quality and all(value == 0 for value in quality.values()), f"bench repeat {index} has {field}")
    for field in ("actual_input_tokens_per_request", "output_tokens_per_request"):
        matrix = require_list(report.get(field), f"bench {field}")
        require(
            len(matrix) == 3
            and all(
                isinstance(row, list)
                and len(row) == requests
                and all(type(value) is int and value > 0 for value in row)
                for row in matrix
            ),
            f"bench {field} denominator differs",
        )
    throughput = report.get("output_throughput_tps")
    throughput_value = throughput.get("mean") if isinstance(throughput, dict) else throughput
    require(
        isinstance(throughput_value, (int, float))
        and not isinstance(throughput_value, bool)
        and throughput_value > 0,
        "bench output throughput is not positive",
    )
    return {
        "path": path,
        "ref": artifact_ref(path),
        "manifest": report,
        "command": command,
        "command_line": parts,
        "request_count": requests * 3,
    }


def validate_log_ref(raw: Any, *, label: str, root: Path) -> dict[str, Any]:
    ref, path = resolve_ref(raw, label, root=root)
    text = path.read_text(encoding="utf-8", errors="replace")
    require(BLOCKER_RE.search(text) is None, f"{label} contains a release blocker marker")
    return ref


def validate_run_parity(
    path: Path,
    *,
    model_key: str,
    backend: str,
    staged: dict[str, Any],
    receipt: dict[str, Any],
    effective: dict[str, Any],
) -> dict[str, Any]:
    parity_path = path.expanduser().resolve()
    value = require_object(read_json(parity_path, "run parity raw evidence"), "run parity raw evidence")
    expected_fields = {
        "schema_version",
        "artifact_type",
        "status",
        "collection_scope",
        "source",
        "model_key",
        "backend",
        "binary_sha256",
        "model_files",
        "entrypoints",
        "run_parity_status",
        "failure_count",
        "hardware",
        "run",
        "serve",
        "comparison",
    }
    source = normalize_source(value.get("source"), "run parity source")
    row = staged["assets"][backend]
    hardware = require_object(value.get("hardware"), "run parity hardware")
    require(
        set(value) == expected_fields
        and value.get("schema_version") == SCHEMA_VERSION
        and value.get("artifact_type")
        == "runtime_vnext_sampled_run_parity_raw_evidence"
        and value.get("status") == "pass"
        and value.get("collection_scope") == COLLECTION_SCOPE
        and source == staged["release_candidate"]
        and value.get("model_key") == model_key
        and value.get("backend") == backend
        and value.get("binary_sha256") == row["binary"]["sha256"]
        and value.get("model_files") == effective["model_files"]
        and value.get("entrypoints") == ["run", "serve"]
        and value.get("run_parity_status") == "pass"
        and value.get("failure_count") == 0
        and hardware,
        "run parity source/model/backend/binary contract differs",
    )
    root = parity_path.parent
    records: dict[str, dict[str, Any]] = {}
    for entrypoint in ("run", "serve"):
        record = require_object(value.get(entrypoint), f"run parity {entrypoint}")
        require(
            set(record)
            == {
                "status",
                "command_line",
                "returncode",
                "output_token_count",
                "error_count",
                "stdout",
                "stderr",
                "effective_config",
            },
            f"run parity {entrypoint} fields differ",
        )
        command_line = require_list(
            record.get("command_line"), f"run parity {entrypoint} command"
        )
        require(
            len(command_line) >= 2
            and Path(str(command_line[0])).expanduser().resolve()
            == receipt["binary_path"]
            and command_line[1] == entrypoint
            and record.get("status") == "pass"
            and record.get("returncode") == 0
            and positive_int(
                record.get("output_token_count"),
                f"run parity {entrypoint} output token count",
            )
            and record.get("error_count") == 0
            and not any(
                re.match(r"FERRUM_[A-Z0-9_]+=", str(part))
                for part in command_line
            ),
            f"run parity {entrypoint} did not use the exact staged product path",
        )
        stdout = validate_log_ref(
            record.get("stdout"), label=f"run parity {entrypoint} stdout", root=root
        )
        stderr = validate_log_ref(
            record.get("stderr"), label=f"run parity {entrypoint} stderr", root=root
        )
        config_ref, _ = resolve_ref(
            record.get("effective_config"),
            f"run parity {entrypoint} effective config",
            root=root,
        )
        records[entrypoint] = {
            "command_line": copy.deepcopy(command_line),
            "stdout": stdout,
            "stderr": stderr,
            "effective_config": config_ref,
            "output_token_count": record["output_token_count"],
        }
    comparison = require_object(value.get("comparison"), "run parity comparison")
    require(
        set(comparison)
        == {
            "contract",
            "status",
            "prompt_sha256",
            "sample_count",
            "failure_count",
        }
        and comparison.get("contract") == "same-prompt-generation-options-v1"
        and comparison.get("status") == "pass"
        and SHA256_RE.fullmatch(str(comparison.get("prompt_sha256", "")))
        and positive_int(comparison.get("sample_count"), "run parity sample count")
        and comparison.get("failure_count") == 0,
        "run parity comparison contract differs",
    )
    return {
        "path": parity_path,
        "ref": artifact_ref(parity_path),
        "manifest": value,
        "hardware": copy.deepcopy(hardware),
        "records": records,
        "sample_count": comparison["sample_count"],
    }


def correctness_evidence_document(
    *,
    model_key: str,
    backend: str,
    staged: dict[str, Any],
    plan: dict[str, Any],
    focused: dict[str, Any],
) -> dict[str, Any]:
    selection = correctness_plan_selection(plan, model_key, backend)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_sampled_c17_authorized_evidence",
        # Preserve the raw diagnostic outcome.  Only the checked-in selection
        # is a PASS; this document is never a formal full-matrix PASS.
        "status": "keep",
        "raw_status": "keep",
        "raw_decision": "KEEP",
        "raw_formal_pass_allowed": False,
        "sample_selection_status": "pass",
        "collection_scope": COLLECTION_SCOPE,
        "authorization": {
            "contract": "checked-in-sample-plan-authorizes-focused-c17-v1",
            "sample_plan": plan["ref"],
            "selection": copy.deepcopy(selection),
            "raw_diagnostic_decision": "KEEP",
            "raw_formal_pass_allowed": False,
            "full_matrix_claim": False,
            "unselected_status": "not_evaluated",
        },
        "source": copy.deepcopy(staged["release_candidate"]),
        "model_key": model_key,
        "backend": backend,
        "binary_sha256": staged["assets"][backend]["binary"]["sha256"],
        "entrypoints": ["run", "serve"],
        "scenario_ids": ["C17"],
        "comparison_contract": {
            "case_count": selection["case_count"],
            "checks_per_case": 5,
            "checks": [
                "source-and-staged-byte-binding",
                "product-entrypoint-execution",
                "unicode-oracle",
                "zero-error-counters",
                "typed-effective-config-binding",
            ],
        },
        "comparison_count": focused["comparison_count"],
        "sample_count": focused["sample_count"],
        "failure_count": 0,
        "command_line": copy.deepcopy(focused["command_line"]),
        "focused_diagnostic": focused["ref"],
        "executor_invocation": focused["invocation"],
    }


def performance_evidence_document(
    *,
    model_key: str,
    backend: str,
    staged: dict[str, Any],
    plan: dict[str, Any],
    bench: dict[str, Any],
    parity: dict[str, Any],
) -> dict[str, Any]:
    selected = copy.deepcopy(plan["manifest"]["performance"][model_key][backend])
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_sampled_single_cell_performance_evidence",
        "status": "pass",
        "collection_scope": COLLECTION_SCOPE,
        "authorization": {
            "contract": "checked-in-sample-plan-selects-one-cell-v1",
            "sample_plan": plan["ref"],
            "full_matrix_claim": False,
            "unselected_status": "not_evaluated",
        },
        "source": copy.deepcopy(staged["release_candidate"]),
        "model_key": model_key,
        "backend": backend,
        "binary_sha256": staged["assets"][backend]["binary"]["sha256"],
        "selected_cell": selected,
        "repeat_count": 3,
        "run_parity_status": "pass",
        "entrypoints": ["run", "serve"],
        "sample_count": bench["request_count"] + parity["sample_count"],
        "failure_count": 0,
        "error_count": 0,
        "command_line": copy.deepcopy(bench["command_line"]),
        "hardware": copy.deepcopy(parity["hardware"]),
        "raw_bench_report": bench["ref"],
        "raw_bench_command": bench["command"]["ref"],
        "raw_run_parity": parity["ref"],
    }


def validate_common_inputs(args: argparse.Namespace, model_key: str) -> dict[str, Any]:
    backend = args.backend
    require(backend in BACKENDS, "--backend must be metal or cuda")
    plan = checked_sample_plan()
    require(model_key in plan["manifest"]["correctness"], "model is absent from sampled plan")
    staged = staged_context(args.staged_assets, backend)
    receipt = validate_receipt(
        args.binary_build_receipt, staged=staged, backend=backend
    )
    effective = raw_effective_config(
        args.effective_config,
        model_key=model_key,
        backend=backend,
        staged=staged,
        receipt=receipt,
    )
    return {
        "backend": backend,
        "plan": plan,
        "staged": staged,
        "receipt": receipt,
        "effective": effective,
    }


def ensure_fresh_out(path: Path) -> Path:
    out = path.expanduser().resolve()
    require(not out.exists(), f"--out must be a fresh path: {out}")
    out.mkdir(parents=True)
    return out


def build_correctness(args: argparse.Namespace) -> Path:
    model_key = require_string(args.model_key, "--model-key")
    require(model_key in MODEL_KEYS, "correctness mode accepts only M1/M2/M3")
    context = validate_common_inputs(args, model_key)
    focused = validate_focused_c17(
        args.focused_report,
        model_key=model_key,
        backend=context["backend"],
        staged=context["staged"],
        receipt=context["receipt"],
        effective=context["effective"],
        plan=context["plan"],
    )
    out = ensure_fresh_out(args.out)
    effective_path = write_effective_wrapper(
        out,
        model_key=model_key,
        backend=context["backend"],
        staged=context["staged"],
        effective=context["effective"],
    )
    evidence_path = out / "scenario-evidence.json"
    write_json(
        evidence_path,
        correctness_evidence_document(
            model_key=model_key,
            backend=context["backend"],
            staged=context["staged"],
            plan=context["plan"],
            focused=focused,
        ),
    )
    row = context["staged"]["assets"][context["backend"]]
    pass_line = f"{PASS_PREFIXES['correctness']}: {out}"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_sampled_final_correctness_manifest",
        "status": "pass",
        "canonical": True,
        "collection_scope": COLLECTION_SCOPE,
        "model_key": model_key,
        "backend": context["backend"],
        "source": copy.deepcopy(context["staged"]["release_candidate"]),
        "binary_sha256": row["binary"]["sha256"],
        "tarball_sha256": row["tarball"]["sha256"],
        "entrypoints": ["run", "serve"],
        "scenario_ids": ["C17"],
        "comparison_count": focused["comparison_count"],
        "outcome": "keep",
        "raw_status": "keep",
        "raw_decision": "KEEP",
        "raw_formal_pass_allowed": False,
        "sample_selection_status": "pass",
        "sample_count": focused["sample_count"],
        "failure_count": 0,
        "full_matrix_claim": False,
        "scenario_report": artifact_ref(evidence_path),
        "sample_plan": context["plan"]["ref"],
        "focused_diagnostic": focused["ref"],
        "binary_build_receipt": context["receipt"]["ref"],
        "raw_effective_config": context["effective"]["ref"],
        "effective_config": artifact_ref(effective_path),
        "model_files_sha256": context["effective"]["model_files_sha256"],
        "pass_line": pass_line,
    }
    write_json(out / "manifest.json", manifest)
    print(pass_line)
    return out


def validate_correctness_manifest(
    path: Path,
    *,
    model_key: str,
    backend: str,
    staged: dict[str, Any],
    expected_sample_plan_sha256: str,
) -> dict[str, Any]:
    """Replay every raw C17 reference behind a sampled correctness manifest."""

    manifest_path = input_manifest(path)
    manifest = require_object(
        read_json(manifest_path, "sampled correctness manifest"),
        "sampled correctness manifest",
    )
    expected_fields = {
        "schema_version",
        "artifact_type",
        "status",
        "canonical",
        "collection_scope",
        "model_key",
        "backend",
        "source",
        "binary_sha256",
        "tarball_sha256",
        "entrypoints",
        "scenario_ids",
        "comparison_count",
        "outcome",
        "raw_status",
        "raw_decision",
        "raw_formal_pass_allowed",
        "sample_selection_status",
        "sample_count",
        "failure_count",
        "full_matrix_claim",
        "scenario_report",
        "sample_plan",
        "focused_diagnostic",
        "binary_build_receipt",
        "raw_effective_config",
        "effective_config",
        "model_files_sha256",
        "pass_line",
    }
    row = staged["assets"][backend]
    plan = checked_sample_plan()
    selection = correctness_plan_selection(plan, model_key, backend)
    require(
        expected_sample_plan_sha256 == plan["ref"]["sha256"],
        "consumer sample-plan SHA differs from the checked-in plan",
    )
    require(
        set(manifest) == expected_fields
        and manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type")
        == "runtime_vnext_sampled_final_correctness_manifest"
        and manifest.get("status") == "pass"
        and manifest.get("canonical") is True
        and manifest.get("collection_scope") == COLLECTION_SCOPE
        and manifest.get("model_key") == model_key
        and manifest.get("backend") == backend
        and normalize_source(manifest.get("source"), "sampled correctness source")
        == staged["release_candidate"]
        and manifest.get("binary_sha256") == row["binary"]["sha256"]
        and manifest.get("tarball_sha256") == row["tarball"]["sha256"]
        and manifest.get("entrypoints") == ["run", "serve"]
        and manifest.get("scenario_ids") == ["C17"]
        and manifest.get("comparison_count") == selection["comparison_count"]
        and manifest.get("sample_count") == selection["case_count"]
        and manifest.get("outcome") == "keep"
        and manifest.get("raw_status") == "keep"
        and manifest.get("raw_decision") == "KEEP"
        and manifest.get("raw_formal_pass_allowed") is False
        and manifest.get("sample_selection_status") == "pass"
        and manifest.get("failure_count") == 0
        and manifest.get("full_matrix_claim") is False
        and manifest.get("pass_line")
        == f"{PASS_PREFIXES['correctness']}: {manifest_path.parent.resolve()}",
        "sampled correctness manifest identity/result differs",
    )
    plan_ref, _ = resolve_ref(
        manifest.get("sample_plan"), "sampled correctness sample plan", root=manifest_path.parent
    )
    require(
        plan_ref["sha256"] == expected_sample_plan_sha256,
        "sampled correctness manifest binds a different sample plan",
    )
    receipt_ref, receipt_path = resolve_ref(
        manifest.get("binary_build_receipt"),
        "sampled correctness build receipt",
        root=manifest_path.parent,
    )
    receipt = validate_receipt(receipt_path, staged=staged, backend=backend)
    raw_config_ref, raw_config_path = resolve_ref(
        manifest.get("raw_effective_config"),
        "sampled correctness raw effective config",
        root=manifest_path.parent,
    )
    effective = raw_effective_config(
        raw_config_path,
        model_key=model_key,
        backend=backend,
        staged=staged,
        receipt=receipt,
    )
    require(
        raw_config_ref["sha256"] == effective["ref"]["sha256"]
        and receipt_ref["sha256"] == receipt["ref"]["sha256"],
        "sampled correctness raw config/receipt refs changed",
    )
    focused_ref, focused_path = resolve_ref(
        manifest.get("focused_diagnostic"),
        "sampled correctness focused diagnostic",
        root=manifest_path.parent,
    )
    focused = validate_focused_c17(
        focused_path,
        model_key=model_key,
        backend=backend,
        staged=staged,
        receipt=receipt,
        effective=effective,
        plan=plan,
    )
    require(
        focused_ref["sha256"] == focused["ref"]["sha256"],
        "sampled correctness focused diagnostic ref changed",
    )
    scenario_ref, scenario_path = resolve_ref(
        manifest.get("scenario_report"),
        "sampled correctness derived evidence",
        root=manifest_path.parent,
    )
    scenario = require_object(
        read_json(scenario_path, "sampled correctness derived evidence"),
        "sampled correctness derived evidence",
    )
    expected_scenario = correctness_evidence_document(
        model_key=model_key,
        backend=backend,
        staged=staged,
        plan=plan,
        focused=focused,
    )
    require(
        scenario == expected_scenario,
        "sampled correctness derived evidence does not replay from raw C17 cases",
    )
    wrapper_ref, wrapper_path = resolve_ref(
        manifest.get("effective_config"),
        "sampled correctness effective config wrapper",
        root=manifest_path.parent,
    )
    wrapper = require_object(
        read_json(wrapper_path, "sampled correctness effective config wrapper"),
        "sampled correctness effective config wrapper",
    )
    require(
        wrapper.get("source") == staged["release_candidate"]
        and wrapper.get("model_key") == model_key
        and wrapper.get("backend") == backend
        and wrapper.get("binary_sha256") == row["binary"]["sha256"]
        and wrapper.get("model_files") == effective["model_files"]
        and wrapper.get("typed_effective_config") == effective["typed"]
        and wrapper.get("raw_effective_config") == effective["ref"]
        and manifest.get("model_files_sha256") == effective["model_files_sha256"],
        "sampled correctness typed effective-config wrapper differs",
    )
    return {
        "manifest": manifest,
        "manifest_ref": artifact_ref(manifest_path),
        "scenario_ref": scenario_ref,
        "receipt_ref": receipt_ref,
        "typed_config_ref": wrapper_ref,
        "sample_plan_ref": plan_ref,
        "selection": selection,
    }


def collector_root(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    plan_ref = require_object(manifest.get("plan"), "collector plan ref")
    for ancestor in manifest_path.resolve().parents:
        candidate = (ancestor / str(plan_ref.get("path", ""))).resolve()
        if (
            candidate.is_file()
            and not candidate.is_symlink()
            and file_sha256(candidate) == plan_ref.get("sha256")
        ):
            return ancestor
    raise SampledFinalError("collector artifact root is not resolvable from its plan ref")


def finite_positive(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0,
        f"{label} must be finite and positive",
    )
    return float(value)


def validate_r3_collector_performance(
    path: Path,
    *,
    model_key: str,
    backend: str,
    staged: dict[str, Any],
    expected_sample_plan_sha256: str,
    expected_g10a_sha256: str,
    expected_g08_sha256: str,
    expected_correctness_sha256: str,
    expected_build_receipt_sha256: str,
) -> dict[str, Any]:
    """Strictly replay an exact-staged collector and judge the selected cell."""

    require(model_key in MODEL_KEYS and backend in BACKENDS, "invalid performance lane")
    manifest_path = input_manifest(path)
    manifest = require_object(
        read_json(manifest_path, "R3 exact-staged collector manifest"),
        "R3 exact-staged collector manifest",
    )
    plan = checked_sample_plan()
    require(
        plan["ref"]["sha256"] == expected_sample_plan_sha256,
        "R3 performance consumer sample-plan SHA differs",
    )
    selected_plan = require_object(
        plan["manifest"]["performance"][model_key][backend],
        f"sample performance plan {model_key}/{backend}",
    )
    selected_id = f"{selected_plan['dataset']}:c{selected_plan['concurrency']}"
    row = staged["assets"][backend]
    require(
        manifest.get("artifact_type")
        == "runtime_vnext_r3_exact_staged_ferrum_lane_manifest"
        and manifest.get("contract")
        == "ferrum.runtime-vnext.r3.exact-staged-ferrum-collector.v1"
        and manifest.get("status") == "pass"
        and manifest.get("authority_mode") == "g08-rc-staged"
        and manifest.get("sample_contract") == "selected-performance-evidence-v1"
        and manifest.get("selected_cells") == [selected_id]
        and manifest.get("formal_http_cells") == [selected_id]
        and manifest.get("formal_http_cell_count") == 1
        and manifest.get("full_matrix_claim") is False
        and manifest.get("unselected_cells") == "not_evaluated"
        and manifest.get("formal_r3_aggregate_status") == "not-evaluated"
        and manifest.get("release_version") == VERSION
        and manifest.get("release_candidate") == staged["release_candidate"]
        and manifest.get("source_git_sha") == staged["release_candidate"]["git_sha"]
        and manifest.get("source_tree_sha")
        == staged["release_candidate"]["git_tree_sha"]
        and manifest.get("dirty_status") == {"is_dirty": False, "status_short": []}
        and manifest.get("model_key") == model_key
        and manifest.get("backend") == backend
        and manifest.get("candidate_binary_sha256")
        == manifest.get("staged_binary_sha256")
        == row["binary"]["sha256"]
        and manifest.get("staged_tarball_sha256") == row["tarball"]["sha256"],
        f"R3 collector {model_key}/{backend} identity/sample/staged bytes differ",
    )
    root = collector_root(manifest_path, manifest)
    normalized_path = manifest_path.parent / "config.normalized.json"
    require(
        normalized_path.is_file() and not normalized_path.is_symlink(),
        "collector normalized config is missing",
    )
    normalized = require_object(
        read_json(normalized_path, "collector normalized config"),
        "collector normalized config",
    )
    require(
        normalized.get("selected_cells") == [selected_id]
        and normalized.get("model_key") == model_key
        and normalized.get("backend") == backend
        and normalized.get("authority_mode") == "g08-rc-staged"
        and normalized.get("candidate", {}).get("source_git_sha")
        == staged["release_candidate"]["git_sha"]
        and normalized.get("candidate", {}).get("source_tree_sha")
        == staged["release_candidate"]["git_tree_sha"],
        "collector normalized model/backend/source/sample selection differs",
    )
    try:
        import runtime_vnext_r2_ferrum_collector as collector

        fingerprint = require_sha256(
            manifest.get("config_fingerprint"), "collector config fingerprint"
        )
        collector.validate_final_manifest(root, manifest, fingerprint, normalized)
        raw_inputs = require_object(manifest.get("inputs"), "collector inputs")
        inputs = copy.deepcopy(raw_inputs)
        for ref_name, path_name in (
            ("binary", "binary_path"),
            ("tokenizer", "tokenizer_path"),
            ("realistic_dataset", "realistic_dataset_path"),
            ("run_parity_dataset", "run_parity_dataset_path"),
        ):
            inputs[path_name] = collector.validate_artifact_ref(
                root, raw_inputs.get(ref_name), f"collector.inputs.{ref_name}"
            )
        server_path = collector.validate_artifact_ref(
            root, manifest.get("server_session"), "collector.server_session"
        )
        server_bundle = collector.read_json(server_path)
        # Formal R3 refuses the older receipt-light compatibility shape.
        require(
            isinstance(server_bundle.get("session_epochs"), list)
            and server_bundle["session_epochs"]
            and isinstance(server_bundle.get("completed_cell_checkpoints"), list),
            "collector server bundle lacks explicit process epochs/checkpoints",
        )
        collector.validate_server_bundle(
            root, server_bundle, fingerprint, normalized, inputs
        )
        run_bundles: list[dict[str, Any]] = []
        run_receipts: list[dict[str, Any]] = []
        for ordinal, raw_ref in enumerate(
            require_list(manifest.get("run_samples"), "collector run samples"),
            start=1,
        ):
            run_path = collector.validate_artifact_ref(
                root, raw_ref, f"collector.run_samples[{ordinal}]"
            )
            bundle = collector.read_json(run_path)
            collector.validate_run_bundle(root, bundle, fingerprint, ordinal)
            sample = require_object(bundle.get("sample"), f"run sample {ordinal}")
            receipt_path = collector.validate_artifact_ref(
                root,
                sample.get("process_receipt"),
                f"run sample {ordinal}.process_receipt",
            )
            process_receipt = collector.read_json(receipt_path)
            effective_path = collector.validate_artifact_ref(
                root,
                sample.get("product_effective_config"),
                f"run sample {ordinal}.product_effective_config",
            )
            expected_argv = collector.run_argv(
                inputs["binary_path"], effective_path, normalized
            )
            require(
                sample.get("independent_process") is True
                and sample.get("candidate_binary_sha256") == row["binary"]["sha256"]
                and sample.get("source_git_sha")
                == staged["release_candidate"]["git_sha"]
                and sample.get("hardware") == manifest.get("hardware")
                and sample.get("prompt") == raw_inputs.get("run_prompt")
                and sample.get("argv") == expected_argv
                and process_receipt.get("argv") == expected_argv
                and process_receipt.get("argv_sha256")
                == canonical_json_sha256(expected_argv)
                and process_receipt.get("environment") == sample.get("environment")
                and process_receipt.get("environment_sha256")
                == canonical_json_sha256(sample.get("environment"))
                and process_receipt.get("pid") == sample.get("pid")
                and process_receipt.get("pgid") == sample.get("pgid")
                and process_receipt.get("process_start_marker")
                == sample.get("process_start_marker")
                and sample.get("returncode") == 0,
                f"run sample {ordinal} process/transcript receipt binding differs",
            )
            run_bundles.append(bundle)
            run_receipts.append(artifact_ref(receipt_path))
        recomputed_run = collector.run_summary(
            run_bundles, server_bundle["run_serve_parity_report"]
        )
        require(
            manifest.get("run_performance") == recomputed_run,
            "collector run/serve parity summary does not replay from process transcripts",
        )
    except SampledFinalError:
        raise
    except Exception as error:
        raise SampledFinalError(
            f"strict R3 collector replay rejected {model_key}/{backend}: {error}"
        ) from error

    refs = require_object(manifest.get("inputs"), "collector authority inputs")
    authority_digests: dict[str, str] = {}
    for key in (
        "build_receipt",
        "correctness_manifest",
        "release_freeze_manifest",
        "authority",
        "staged_assets_manifest",
    ):
        authority_digests[key] = file_sha256(
            collector.validate_artifact_ref(root, refs.get(key), f"collector.inputs.{key}")
        )
    require(
        authority_digests
        == {
            "build_receipt": expected_build_receipt_sha256,
            "correctness_manifest": expected_correctness_sha256,
            "release_freeze_manifest": expected_g10a_sha256,
            "authority": expected_g08_sha256,
            "staged_assets_manifest": staged["ref"]["sha256"],
        },
        "collector authority/G10A/G08/correctness/staged refs differ",
    )

    hardware = require_object(manifest.get("hardware"), "collector hardware")
    try:
        import runtime_vnext_r2_performance_build_profile as r2_profile

        floor_rows, _ = r2_profile.validate_floor_catalog(
            R2_FLOOR_CATALOG, require_checked_in=True
        )
    except Exception as error:
        raise SampledFinalError(f"frozen R2 floor catalog rejected: {error}") from error
    floor_key = (
        model_key,
        backend,
        selected_plan["dataset"],
        selected_plan["concurrency"],
        "throughput",
    )
    floor = require_object(floor_rows.get(floor_key), f"R2 floor {floor_key}")
    require(
        floor.get("hardware_id") == hardware.get("id")
        and floor.get("hardware_sha256") == canonical_json_sha256(hardware),
        "selected performance cell is not on the frozen R2 hardware identity",
    )
    formal_reports = require_list(
        server_bundle.get("formal_reports"), "collector formal reports"
    )
    selected_records = [row for row in formal_reports if row.get("cell_id") == selected_id]
    require(len(selected_records) == 1, "collector selected report denominator differs")
    report_path = collector.validate_artifact_ref(
        root, selected_records[0].get("raw_report"), "collector selected raw report"
    )
    raw_report = collector.read_json(report_path)
    raw_throughput = raw_report.get("output_throughput_tps")
    actual_throughput = finite_positive(
        raw_throughput.get("mean") if isinstance(raw_throughput, dict) else raw_throughput,
        "selected output throughput",
    )
    floor_value = finite_positive(floor.get("value"), "frozen R2 throughput floor")
    throughput_ratio = actual_throughput / floor_value
    run_ratio = finite_positive(
        recomputed_run.get("run_to_serve_c1_steady_decode_ratio"),
        "run/serve steady-decode ratio",
    )
    require(
        throughput_ratio >= float(selected_plan["floor_ratio"]),
        f"selected throughput ratio {throughput_ratio:.6f} is below "
        f"{selected_plan['floor_ratio']:.6f}",
    )
    require(
        run_ratio >= float(selected_plan["run_serve_ratio_floor"]),
        f"run/serve ratio {run_ratio:.6f} is below "
        f"{selected_plan['run_serve_ratio_floor']:.6f}",
    )
    return {
        "status": "pass",
        "manifest": manifest,
        "manifest_ref": artifact_ref(manifest_path),
        "sample_plan_ref": plan["ref"],
        "selected_cell": copy.deepcopy(selected_plan),
        "hardware": copy.deepcopy(hardware),
        "hardware_sha256": canonical_json_sha256(hardware),
        "model_files_sha256": canonical_json_sha256(
            require_object(manifest.get("model_files"), "collector model files")
        ),
        "server_session": artifact_ref(server_path),
        "raw_bench_report": artifact_ref(report_path),
        "run_process_receipts": run_receipts,
        "actual_throughput_tps": actual_throughput,
        "frozen_floor_tps": floor_value,
        "throughput_floor_ratio": throughput_ratio,
        "run_serve_ratio": run_ratio,
    }


def parse_timestamp(value: Any, label: str) -> datetime:
    text = require_string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise SampledFinalError(f"{label} is not an ISO-8601 timestamp") from error
    require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return parsed


def validate_llama_hardware(value: Any, backend: str, expected_contract: str) -> dict[str, Any]:
    hardware = require_object(value, "Llama hardware")
    require(
        hardware.get("contract") == expected_contract
        and isinstance(hardware.get("id"), str)
        and hardware["id"]
        and hardware.get("accelerator_count") == 1
        and type(hardware.get("memory_bytes")) is int,
        "Llama hardware identity/contract is incomplete",
    )
    if backend == "cuda":
        require(
            hardware.get("accelerator_model") == "NVIDIA GeForce RTX 4090"
            and 23 * 1024**3 <= hardware["memory_bytes"] <= 25 * 1024**3,
            "Llama CUDA floor requires one 24-GiB RTX 4090",
        )
    else:
        require(
            hardware.get("accelerator_model") == "Apple M1 Max"
            and hardware.get("gpu_core_count") == 24
            and hardware["memory_bytes"] == 32 * 1024**3,
            "Llama Metal floor requires the 24-core/32-GiB Apple M1 Max",
        )
    return copy.deepcopy(hardware)


def validate_llama_process_receipt(
    raw_ref: Any,
    *,
    root: Path,
    role: str,
    source: dict[str, Any],
    binary_path: Path,
    binary_sha256: str,
    model_id: str,
    backend: str,
    typed_config_sha256: str | None,
) -> dict[str, Any]:
    ref, path = resolve_ref(
        raw_ref, f"Llama {role} process receipt", root=root, require_within_root=True
    )
    value = require_object(
        read_json(path, f"Llama {role} process receipt"),
        f"Llama {role} process receipt",
    )
    expected_fields = {
        "schema_version",
        "artifact_type",
        "role",
        "argv",
        "argv_sha256",
        "environment",
        "environment_sha256",
        "source",
        "binary_sha256",
        "model_id",
        "typed_config_sha256",
        "pid",
        "pgid",
        "started_at",
        "finished_at",
        "returncode",
        "shutdown_clean",
        "stdout",
        "stderr",
        "artifacts",
        "server_process_receipt_sha256",
    }
    argv = require_list(value.get("argv"), f"Llama {role} argv")
    environment = require_object(
        value.get("environment"), f"Llama {role} environment"
    )
    require(
        set(value) == expected_fields
        and value.get("schema_version") == SCHEMA_VERSION
        and value.get("artifact_type") == "runtime_vnext_g0_process_receipt"
        and value.get("role") == role
        and len(argv) >= 2
        and Path(str(argv[0])).expanduser().resolve() == binary_path
        and value.get("argv_sha256") == canonical_json_sha256(argv)
        and all(isinstance(key, str) and isinstance(item, str) for key, item in environment.items())
        and set(environment) <= LLAMA_ENV_ALLOWLIST
        and value.get("environment_sha256") == canonical_json_sha256(environment)
        and not any(
            marker in key.upper()
            for key in environment
            for marker in ("TOKEN", "SECRET", "PASSWORD", "AUTH", "API_KEY")
        )
        and normalize_source(value.get("source"), f"Llama {role} source") == source
        and value.get("binary_sha256") == binary_sha256
        and value.get("model_id") == model_id
        and (
            value.get("typed_config_sha256") == typed_config_sha256
            if typed_config_sha256 is not None
            else SHA256_RE.fullmatch(str(value.get("typed_config_sha256", "")))
            is not None
        )
        and type(value.get("pid")) is int
        and value["pid"] > 0
        and value.get("pgid") == value["pid"]
        and (
            value.get("returncode") == 0
            if role != "ferrum-serve"
            else value.get("returncode") in {0, -2, -15}
        )
        and value.get("shutdown_clean") is True,
        f"Llama {role} process identity/argv/result differs",
    )
    subcommand = {"ferrum-serve": "serve", "ferrum-bench-serve": "bench-serve", "ferrum-run": "run"}[role]
    require(argv[1] == subcommand, f"Llama {role} argv subcommand differs")
    backend_flag = "--target-backend" if role == "ferrum-bench-serve" else "--backend"
    require(
        argv.count(backend_flag) == 1
        and argv.index(backend_flag) + 1 < len(argv)
        and argv[argv.index(backend_flag) + 1] == backend,
        f"Llama {role} argv does not explicitly select {backend}",
    )
    if role == "ferrum-run":
        require(
            argv[1:] == [
                "run",
                model_id,
                "--backend",
                backend,
                "--temperature",
                "0",
                "--max-tokens",
                "128",
                "--output-format",
                "jsonl",
                "--effective-config-json",
                flag_value(argv, "--effective-config-json"),
            ],
            "Llama run argv differs from the deterministic parity contract",
        )
    elif role == "ferrum-serve":
        port = flag_value(argv, "--port")
        require(
            flag_value(argv, "--model") == model_id
            and flag_value(argv, "--host") == "127.0.0.1"
            and port is not None
            and port.isdigit()
            and 1024 <= int(port) <= 65535
            and flag_value(argv, "--effective-config-json") is not None,
            "Llama serve argv differs from the loopback typed-config contract",
        )
    else:
        require(
            flag_value(argv, "--model") == model_id
            and flag_value(argv, "--dataset") == "random"
            and flag_value(argv, "--random-input-len") == "64"
            and flag_value(argv, "--random-output-len") == "128"
            and flag_value(argv, "--num-prompts") == "8"
            and flag_value(argv, "--warmup-requests") == "1"
            and flag_value(argv, "--n-repeats") == "3"
            and flag_value(argv, "--concurrency") == "1"
            and flag_value(argv, "--seed") == "9271"
            and flag_value(argv, "--output") == "json"
            and flag_value(argv, "--out") is not None
            and "--fail-on-error" in argv
            and "--require-ci" in argv
            and "--request-rate" not in argv,
            "Llama bench argv differs from the frozen c1/three-repeat contract",
        )
    started = parse_timestamp(value.get("started_at"), f"Llama {role}.started_at")
    finished = parse_timestamp(value.get("finished_at"), f"Llama {role}.finished_at")
    require(finished > started, f"Llama {role} process window is not positive")
    logs = {}
    for name in ("stdout", "stderr"):
        log_ref, log_path = resolve_ref(
            value.get(name),
            f"Llama {role}.{name}",
            root=root,
            require_within_root=True,
        )
        require(
            BLOCKER_RE.search(log_path.read_text(encoding="utf-8", errors="replace"))
            is None,
            f"Llama {role}.{name} contains a blocker marker",
        )
        logs[name] = log_ref
    artifacts = require_object(value.get("artifacts"), f"Llama {role} artifacts")
    expected_artifact_names = {
        "ferrum-serve": {"effective_config"},
        "ferrum-bench-serve": {"report"},
        "ferrum-run": {"raw_effective_config", "typed_effective_config"},
    }[role]
    require(
        set(artifacts) == expected_artifact_names,
        f"Llama {role} process artifact denominator differs",
    )
    artifact_refs = {
        name: resolve_ref(
            item,
            f"Llama {role}.artifacts.{name}",
            root=root,
            require_within_root=True,
        )[0]
        for name, item in artifacts.items()
    }
    return {
        "path": path,
        "ref": ref,
        "manifest": value,
        "argv": copy.deepcopy(argv),
        "started_at": started,
        "finished_at": finished,
        "logs": logs,
        "artifacts": artifact_refs,
    }


def validate_llama_transcript(
    raw_ref: Any,
    *,
    root: Path,
    scenario_id: str,
    entrypoint: str,
    process_receipt_sha256: str,
    source: dict[str, Any],
    binary_sha256: str,
    model_id: str,
    typed_config_sha256: str,
) -> dict[str, Any]:
    ref, path = resolve_ref(
        raw_ref,
        f"Llama {scenario_id} transcript",
        root=root,
        require_within_root=True,
    )
    value = require_object(
        read_json(path, f"Llama {scenario_id} transcript"),
        f"Llama {scenario_id} transcript",
    )
    expected_fields = {
        "schema_version",
        "artifact_type",
        "status",
        "scenario_id",
        "entrypoint",
        "source",
        "binary_sha256",
        "model_id",
        "typed_config_sha256",
        "process_receipt_sha256",
        "prompt_sha256",
        "max_tokens",
        "output_token_count",
        "elapsed_ms",
        "finish_reason",
        "error_count",
        "stream_done_count",
        "stdout",
        "stderr",
    }
    elapsed_ms = finite_positive(
        value.get("elapsed_ms"), f"Llama {scenario_id} elapsed_ms"
    )
    require(
        set(value) == expected_fields
        and value.get("schema_version") == SCHEMA_VERSION
        and value.get("artifact_type") == "runtime_vnext_g0_generation_transcript"
        and value.get("status") == "pass"
        and value.get("scenario_id") == scenario_id
        and value.get("entrypoint") == entrypoint
        and normalize_source(value.get("source"), f"Llama {scenario_id} source") == source
        and value.get("binary_sha256") == binary_sha256
        and value.get("model_id") == model_id
        and value.get("typed_config_sha256") == typed_config_sha256
        and value.get("process_receipt_sha256") == process_receipt_sha256
        and SHA256_RE.fullmatch(str(value.get("prompt_sha256", ""))) is not None
        and type(value.get("max_tokens")) is int
        and value["max_tokens"] > 0
        and type(value.get("output_token_count")) is int
        and value["output_token_count"] > 0
        and isinstance(value.get("finish_reason"), str)
        and value["finish_reason"]
        and value.get("error_count") == 0
        and value.get("stream_done_count")
        == (1 if scenario_id == "serve-stream" else 0),
        f"Llama {scenario_id} transcript identity/token/result differs",
    )
    for name in ("stdout", "stderr"):
        _, log_path = resolve_ref(
            value.get(name),
            f"Llama {scenario_id}.{name}",
            root=root,
            require_within_root=True,
        )
        require(
            BLOCKER_RE.search(log_path.read_text(encoding="utf-8", errors="replace"))
            is None,
            f"Llama {scenario_id}.{name} contains a blocker marker",
        )
    output_tps = value["output_token_count"] * 1000.0 / elapsed_ms
    return {"ref": ref, "manifest": value, "output_tps": output_tps}


def validate_llama_bench_report(
    report_path: Path,
    *,
    backend: str,
    model_id: str,
    planned: dict[str, Any],
) -> dict[str, Any]:
    report = require_object(read_json(report_path, "Llama bench report"), "Llama bench report")
    requests = positive_int(report.get("n_requests_per_run"), "Llama bench requests")
    repeats = require_list(report.get("repeat_metrics"), "Llama bench repeats")
    require(
        report.get("model") == model_id
        and report.get("backend") == backend
        and report.get("scenario") == "closed_loop"
        and report.get("concurrency") == planned["concurrency"]
        and report.get("n_repeats") == planned["repeats"] == 3
        and report.get("n_gen") == 128
        and report.get("warmup_requests") == 1
        and requests == 8
        and report.get("output_token_count_source") == "usage"
        and report.get("completed_per_run") == [requests] * 3
        and report.get("errored_per_run") == [0, 0, 0]
        and len(repeats) == 3,
        "Llama bench report cell/repeat/completion identity differs",
    )
    for index, repeat in enumerate(repeats, start=1):
        row = require_object(repeat, f"Llama bench repeat {index}")
        require(
            row.get("repeat") == index
            and row.get("expected_requests") == requests
            and row.get("completed_requests") == requests
            and row.get("errored_requests") == 0
            and row.get("warmup_expected") == 1
            and row.get("warmup_completed") == 1
            and row.get("warmup_errored") == 0
            and row.get("output_token_count_source") == "usage",
            f"Llama bench repeat {index} differs",
        )
        for field in ("quality_issues", "warmup_quality_issues"):
            quality = require_object(
                row.get(field), f"Llama bench repeat {index} {field}"
            )
            require(
                quality and all(value == 0 for value in quality.values()),
                f"Llama bench repeat {index} has {field}",
            )
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
        require(report.get(field) == [0, 0, 0], f"Llama bench {field} is non-zero")
    for field in ("actual_input_tokens_per_request", "output_tokens_per_request"):
        matrix = require_list(report.get(field), f"Llama bench {field}")
        require(
            len(matrix) == 3
            and all(
                isinstance(row, list)
                and len(row) == requests
                and all(type(item) is int and item > 0 for item in row)
                for row in matrix
            ),
            f"Llama bench {field} denominator differs",
        )
    raw = report.get("output_throughput_tps")
    throughput = finite_positive(raw.get("mean") if isinstance(raw, dict) else raw, "Llama throughput")
    return {"manifest": report, "throughput": throughput, "request_count": requests * 3}


def validate_g0_llama_execution_receipt(
    path: Path,
    *,
    backend: str,
    staged: dict[str, Any],
    expected_sample_plan_sha256: str,
) -> dict[str, Any]:
    """Validate the receipt that the G0 Llama runners must emit for R3."""

    receipt_path = input_manifest(path)
    root = receipt_path.parent.resolve()
    value = require_object(
        read_json(receipt_path, "G0 Llama sampled execution receipt"),
        "G0 Llama sampled execution receipt",
    )
    expected_fields = {
        "schema_version",
        "artifact_type",
        "status",
        "producer",
        "source",
        "model_key",
        "model_id",
        "backend",
        "hardware",
        "binary_artifact",
        "binary_sha256",
        "model_files",
        "model_files_sha256",
        "typed_effective_config",
        "staged_assets_manifest",
        "sample_plan",
        "g0_gate_manifest",
        "server_process",
        "bench_process",
        "run_processes",
        "run_parity_transcripts",
        "correctness_transcripts",
        "serve_parity_transcripts",
        "bench_report",
    }
    plan = checked_sample_plan()
    planned_correctness = plan["manifest"]["correctness"][LLAMA_MODEL_KEY][backend]
    planned_performance = plan["manifest"]["performance"][LLAMA_MODEL_KEY][backend]
    source = staged["release_candidate"]
    row = staged["assets"][backend]
    model_id = require_string(value.get("model_id"), "Llama model id")
    require(
        set(value) == expected_fields
        and value.get("schema_version") == SCHEMA_VERSION
        and value.get("artifact_type")
        == "runtime_vnext_g0_llama_dense_execution_receipt"
        and value.get("status") == "pass"
        and value.get("producer") == "g0-llama-dense-execution-binding-v1"
        and normalize_source(value.get("source"), "Llama receipt source") == source
        and value.get("model_key") == LLAMA_MODEL_KEY
        and value.get("backend") == backend
        and value.get("binary_sha256") == row["binary"]["sha256"],
        "G0 Llama receipt source/model/backend/staged binary differs",
    )
    require(
        expected_sample_plan_sha256 == plan["ref"]["sha256"],
        "Llama consumer sample-plan SHA differs",
    )
    sample_ref, _ = resolve_ref(value.get("sample_plan"), "Llama sample plan", root=root)
    staged_ref, _ = resolve_ref(
        value.get("staged_assets_manifest"), "Llama staged assets", root=root
    )
    require(
        sample_ref["sha256"] == expected_sample_plan_sha256
        and staged_ref["sha256"] == staged["ref"]["sha256"],
        "G0 Llama receipt sample-plan/staged manifest binding differs",
    )
    binary_ref, binary_path = resolve_ref(
        value.get("binary_artifact"),
        "Llama staged binary",
        root=root,
        require_within_root=True,
    )
    require(
        binary_ref["sha256"] == row["binary"]["sha256"]
        and binary_path.stat().st_size == row["binary"]["size_bytes"],
        "G0 Llama executed binary is not the staged binary byte identity",
    )
    model_files = require_object(value.get("model_files"), "Llama model files")
    require(
        model_files
        and all(
            isinstance(name, str)
            and name
            and SHA256_RE.fullmatch(str(digest)) is not None
            for name, digest in model_files.items()
        )
        and value.get("model_files_sha256") == canonical_json_sha256(model_files),
        "G0 Llama model file identity differs",
    )
    typed_ref, typed_path = resolve_ref(
        value.get("typed_effective_config"),
        "Llama typed effective config",
        root=root,
        require_within_root=True,
    )
    typed = require_object(read_json(typed_path, "Llama typed effective config"), "Llama typed effective config")
    require(
        typed.get("source") == source
        and typed.get("model_key") == LLAMA_MODEL_KEY
        and typed.get("model_id") == model_id
        and typed.get("backend") == backend
        and typed.get("binary_sha256") == row["binary"]["sha256"]
        and typed.get("model_files") == model_files
        and isinstance(typed.get("typed_effective_config"), dict)
        and typed["typed_effective_config"],
        "G0 Llama typed effective config identity differs",
    )
    hardware = validate_llama_hardware(
        value.get("hardware"), backend, planned_performance["floor"]["hardware_contract"]
    )
    gate_ref, gate_path = resolve_ref(
        value.get("g0_gate_manifest"), "Llama G0 gate manifest", root=root
    )
    gate = require_object(read_json(gate_path, "Llama G0 gate manifest"), "Llama G0 gate manifest")
    gate_started = parse_timestamp(gate.get("started_at"), "Llama G0 gate started_at")
    gate_finished = parse_timestamp(gate.get("finished_at"), "Llama G0 gate finished_at")
    require(
        gate.get("status") == "pass"
        and gate.get("git_sha") == source["git_sha"]
        and gate.get("child_returncode") == 0
        and gate.get("binary", {}).get("sha256") == row["binary"]["sha256"]
        and str(gate.get("pass_line", "")).startswith("FERRUM GATE ")
        and gate_finished > gate_started,
        "Llama receipt does not bind a passing same-source/same-binary G0 gate",
    )
    common_process = {
        "root": root,
        "source": source,
        "binary_path": binary_path,
        "binary_sha256": row["binary"]["sha256"],
        "model_id": model_id,
        "backend": backend,
    }
    server = validate_llama_process_receipt(
        value.get("server_process"),
        role="ferrum-serve",
        typed_config_sha256=typed_ref["sha256"],
        **common_process,
    )
    bench = validate_llama_process_receipt(
        value.get("bench_process"),
        role="ferrum-bench-serve",
        typed_config_sha256=typed_ref["sha256"],
        **common_process,
    )
    _, raw_effective_path = resolve_ref(
        server["manifest"]["artifacts"]["effective_config"],
        "Llama serve raw effective config",
        root=root,
        require_within_root=True,
    )
    server_raw_effective = require_object(
        read_json(raw_effective_path, "Llama raw effective config"),
        "Llama raw effective config",
    )
    require(
        Path(flag_value(server["argv"], "--effective-config-json") or "")
        .expanduser()
        .resolve()
        == raw_effective_path
        and server_raw_effective == typed["typed_effective_config"],
        "Llama serve process does not bind the captured typed effective config",
    )
    server_port = flag_value(server["argv"], "--port")
    bench_report_argv = Path(flag_value(bench["argv"], "--out") or "").expanduser().resolve()
    _, bound_bench_report_path = resolve_ref(
        bench["manifest"]["artifacts"]["report"],
        "Llama bench process report",
        root=root,
        require_within_root=True,
    )
    tokenizer_dir = Path(
        flag_value(bench["argv"], "--tokenizer") or ""
    ).expanduser().resolve()
    try:
        tokenizer_dir.relative_to(root)
    except ValueError as error:
        raise SampledFinalError("Llama bench tokenizer escapes its artifact root") from error
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    require(
        bench["manifest"].get("server_process_receipt_sha256") == server["ref"]["sha256"]
        and flag_value(bench["argv"], "--base-url")
        == f"http://127.0.0.1:{server_port}"
        and bench_report_argv == bound_bench_report_path
        and tokenizer_json.is_file()
        and not tokenizer_json.is_symlink()
        and model_files.get("tokenizer.json") == file_sha256(tokenizer_json)
        and server["started_at"] <= bench["started_at"]
        < bench["finished_at"] <= server["finished_at"],
        "Llama bench is not bound to the server window/report/tokenizer",
    )
    run_refs = require_list(value.get("run_processes"), "Llama run processes")
    require(len(run_refs) == 3, "Llama run parity requires three independent processes")
    runs = [
        validate_llama_process_receipt(
            raw,
            role="ferrum-run",
            typed_config_sha256=None,
            **common_process,
        )
        for raw in run_refs
    ]
    run_raw_paths: list[Path] = []
    for ordinal, run in enumerate(runs, start=1):
        raw_ref, raw_path = resolve_ref(
            run["manifest"]["artifacts"]["raw_effective_config"],
            f"Llama run {ordinal} raw effective config",
            root=root,
            require_within_root=True,
        )
        run_typed_ref, run_typed_path = resolve_ref(
            run["manifest"]["artifacts"]["typed_effective_config"],
            f"Llama run {ordinal} typed effective config",
            root=root,
            require_within_root=True,
        )
        run_raw = require_object(
            read_json(raw_path, f"Llama run {ordinal} raw effective config"),
            f"Llama run {ordinal} raw effective config",
        )
        run_typed = require_object(
            read_json(run_typed_path, f"Llama run {ordinal} typed effective config"),
            f"Llama run {ordinal} typed effective config",
        )
        require(
            set(run_typed)
            == {
                "source",
                "model_key",
                "model_id",
                "backend",
                "binary_sha256",
                "model_files",
                "entrypoint",
                "raw_effective_config",
                "typed_effective_config",
            }
            and run_typed.get("source") == source
            and run_typed.get("model_key") == LLAMA_MODEL_KEY
            and run_typed.get("model_id") == model_id
            and run_typed.get("backend") == backend
            and run_typed.get("binary_sha256") == row["binary"]["sha256"]
            and run_typed.get("model_files") == model_files
            and run_typed.get("entrypoint") == "run"
            and run_typed.get("raw_effective_config") == raw_ref
            and run_typed.get("typed_effective_config") == run_raw
            and run["manifest"].get("typed_config_sha256")
            == run_typed_ref["sha256"]
            and Path(flag_value(run["argv"], "--effective-config-json") or "")
            .expanduser()
            .resolve()
            == raw_path,
            f"Llama run {ordinal} process/config identity differs",
        )
        for field in ("schema_version", "backend", "preset", "model_capabilities"):
            if field in server_raw_effective or field in run_raw:
                require(
                    run_raw.get(field) == server_raw_effective.get(field),
                    f"Llama run {ordinal} core typed field {field} differs from serve",
                )
        run["typed_config_ref"] = run_typed_ref
        run["raw_config_ref"] = raw_ref
        run_raw_paths.append(raw_path)
    require(
        len({run["manifest"]["pid"] for run in runs}) == 3
        and len(set(run_raw_paths)) == 3
        and all(run["manifest"].get("server_process_receipt_sha256") is None for run in runs),
        "Llama run samples are not three independent product processes/configs",
    )
    all_processes = [server, bench, *runs]
    require(
        all(gate_started <= item["started_at"] < item["finished_at"] <= gate_finished for item in all_processes),
        "Llama process window escapes the G0 gate window",
    )
    correctness_raw = require_object(
        value.get("correctness_transcripts"), "Llama correctness transcripts"
    )
    expected_scenarios = {
        "run-multiturn": (
            "run",
            runs[0]["ref"]["sha256"],
            runs[0]["typed_config_ref"]["sha256"],
        ),
        "serve-multiturn": (
            "serve",
            server["ref"]["sha256"],
            typed_ref["sha256"],
        ),
        "serve-stream": (
            "serve",
            server["ref"]["sha256"],
            typed_ref["sha256"],
        ),
    }
    require(
        list(correctness_raw) == planned_correctness["scenario_ids"],
        "Llama correctness transcript denominator/order differs",
    )
    correctness = {
        scenario: validate_llama_transcript(
            correctness_raw[scenario],
            root=root,
            scenario_id=scenario,
            entrypoint=entrypoint,
            process_receipt_sha256=process_sha,
            source=source,
            binary_sha256=row["binary"]["sha256"],
            model_id=model_id,
            typed_config_sha256=typed_sha,
        )
        for scenario, (entrypoint, process_sha, typed_sha) in expected_scenarios.items()
    }
    parity_raw = require_list(
        value.get("serve_parity_transcripts"), "Llama serve parity transcripts"
    )
    require(len(parity_raw) == 3, "Llama serve parity requires three transcripts")
    serve_parity = [
        validate_llama_transcript(
            raw,
            root=root,
            scenario_id="serve-parity",
            entrypoint="serve",
            process_receipt_sha256=server["ref"]["sha256"],
            source=source,
            binary_sha256=row["binary"]["sha256"],
            model_id=model_id,
            typed_config_sha256=typed_ref["sha256"],
        )
        for raw in parity_raw
    ]
    run_transcripts = []
    run_transcript_refs = require_list(
        value.get("run_parity_transcripts"), "Llama run parity transcripts"
    )
    require(len(run_transcript_refs) == 3, "Llama run parity transcript denominator differs")
    for index, (run, transcript_ref) in enumerate(
        zip(runs, run_transcript_refs), start=1
    ):
        run_transcripts.append(
            validate_llama_transcript(
                transcript_ref,
                root=root,
                scenario_id="run-parity",
                entrypoint="run",
                process_receipt_sha256=run["ref"]["sha256"],
                source=source,
                binary_sha256=row["binary"]["sha256"],
                model_id=model_id,
                typed_config_sha256=run["typed_config_ref"]["sha256"],
            )
        )
    prompt_hashes = {
        item["manifest"]["prompt_sha256"] for item in [*run_transcripts, *serve_parity]
    }
    max_tokens = {item["manifest"]["max_tokens"] for item in [*run_transcripts, *serve_parity]}
    require(
        len(prompt_hashes) == len(max_tokens) == 1,
        "Llama run/serve parity does not use the same prompt/options",
    )
    run_median = statistics.median(item["output_tps"] for item in run_transcripts)
    serve_median = statistics.median(item["output_tps"] for item in serve_parity)
    run_serve_ratio = run_median / serve_median
    report_ref, report_path = resolve_ref(
        value.get("bench_report"),
        "Llama bench report",
        root=root,
        require_within_root=True,
    )
    require(
        bench["artifacts"].get("report", {}).get("sha256") == report_ref["sha256"],
        "Llama bench process receipt does not bind its raw report",
    )
    report = validate_llama_bench_report(
        report_path, backend=backend, model_id=model_id, planned=planned_performance
    )
    floor = planned_performance["floor"]
    floor_path = (REPO_ROOT / floor["artifact_path"]).resolve()
    require(
        floor_path.is_file()
        and file_sha256(floor_path) == floor["artifact_sha256"],
        "Llama frozen G0 floor artifact changed",
    )
    floor_value = finite_positive(floor.get("value"), "Llama frozen floor")
    throughput_ratio = report["throughput"] / floor_value
    require(
        throughput_ratio >= float(planned_performance["floor_ratio"]),
        f"Llama throughput ratio {throughput_ratio:.6f} is below {planned_performance['floor_ratio']:.6f}",
    )
    require(
        run_serve_ratio >= float(planned_performance["run_serve_ratio_floor"]),
        f"Llama run/serve ratio {run_serve_ratio:.6f} is below {planned_performance['run_serve_ratio_floor']:.6f}",
    )
    return {
        "receipt": value,
        "receipt_ref": artifact_ref(receipt_path),
        "sample_plan_ref": sample_ref,
        "g0_gate_ref": gate_ref,
        "source": source,
        "model_id": model_id,
        "backend": backend,
        "hardware": hardware,
        "binary_ref": binary_ref,
        "model_files_sha256": value["model_files_sha256"],
        "typed_config_ref": typed_ref,
        "server_process_ref": server["ref"],
        "bench_process_ref": bench["ref"],
        "run_process_refs": [item["ref"] for item in runs],
        "correctness_refs": {key: item["ref"] for key, item in correctness.items()},
        "serve_parity_refs": [item["ref"] for item in serve_parity],
        "bench_report_ref": report_ref,
        "actual_throughput_tps": report["throughput"],
        "frozen_floor_tps": floor_value,
        "throughput_floor_ratio": throughput_ratio,
        "run_serve_ratio": run_serve_ratio,
        "sample_count": report["request_count"] + 6,
    }


def build_performance(args: argparse.Namespace) -> Path:
    raise SampledFinalError(
        "performance mode no longer accepts standalone bench/parity declarations; "
        "run runtime_vnext_r2_ferrum_collector.py in exact-staged R3 mode and "
        "pass that manifest directly to the G09 goal consumer"
    )


def build_llama_supplemental(args: argparse.Namespace) -> Path:
    backend = require_string(args.backend, "--backend")
    staged = staged_context(args.staged_assets, backend)
    plan = checked_sample_plan()
    validated = validate_g0_llama_execution_receipt(
        args.execution_receipt,
        backend=backend,
        staged=staged,
        expected_sample_plan_sha256=plan["ref"]["sha256"],
    )
    out = ensure_fresh_out(args.out)
    correctness_path = out / "correctness-evidence.json"
    performance_path = out / "performance-evidence.json"
    correctness, performance = llama_evidence_documents(
        validated=validated, staged=staged, plan=plan
    )
    write_json(correctness_path, correctness)
    write_json(performance_path, performance)
    row = staged["assets"][backend]
    pass_line = f"{PASS_PREFIXES['llama-supplemental']}: {out}"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_sampled_llama_dense_supplemental_manifest",
        "status": "pass",
        "canonical": True,
        "collection_scope": COLLECTION_SCOPE,
        "model_key": LLAMA_MODEL_KEY,
        "model_id": validated["model_id"],
        "backend": backend,
        "source": copy.deepcopy(staged["release_candidate"]),
        "binary_sha256": row["binary"]["sha256"],
        "tarball_sha256": row["tarball"]["sha256"],
        "entrypoints": ["run", "serve"],
        "correctness": artifact_ref(correctness_path),
        "performance": artifact_ref(performance_path),
        "execution_receipt": validated["receipt_ref"],
        "sample_plan": validated["sample_plan_ref"],
        "g0_gate_manifest": validated["g0_gate_ref"],
        "typed_effective_config": validated["typed_config_ref"],
        "server_process": validated["server_process_ref"],
        "bench_process": validated["bench_process_ref"],
        "run_processes": validated["run_process_refs"],
        "model_files_sha256": validated["model_files_sha256"],
        "full_matrix_claim": False,
        "pass_line": pass_line,
    }
    write_json(out / "manifest.json", manifest)
    print(pass_line)
    return out


def llama_evidence_documents(
    *,
    validated: dict[str, Any],
    staged: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    backend = validated["backend"]
    row = staged["assets"][backend]
    correctness_selection = copy.deepcopy(
        plan["manifest"]["correctness"][LLAMA_MODEL_KEY][backend]
    )
    performance_selection = copy.deepcopy(
        plan["manifest"]["performance"][LLAMA_MODEL_KEY][backend]
    )
    common = {
        "schema_version": SCHEMA_VERSION,
        "collection_scope": COLLECTION_SCOPE,
        "source": copy.deepcopy(staged["release_candidate"]),
        "model_key": LLAMA_MODEL_KEY,
        "model_id": validated["model_id"],
        "backend": backend,
        "binary_sha256": row["binary"]["sha256"],
        "entrypoints": ["run", "serve"],
        "sample_plan": validated["sample_plan_ref"],
        "execution_receipt": validated["receipt_ref"],
        "g0_gate_manifest": validated["g0_gate_ref"],
        "server_process": validated["server_process_ref"],
        "failure_count": 0,
        "full_matrix_claim": False,
    }
    correctness = {
        **common,
        "artifact_type": "runtime_vnext_sampled_llama_correctness_evidence",
        "status": "pass",
        "raw_status": "pass",
        "sample_selection_status": "pass",
        "selection": correctness_selection,
        "scenario_ids": correctness_selection["scenario_ids"],
        "sample_count": len(validated["correctness_refs"]),
        "transcripts": copy.deepcopy(validated["correctness_refs"]),
    }
    performance = {
        **common,
        "artifact_type": "runtime_vnext_sampled_llama_performance_evidence",
        "status": "pass",
        "selection": performance_selection,
        "sample_count": validated["sample_count"],
        "hardware": copy.deepcopy(validated["hardware"]),
        "bench_process": validated["bench_process_ref"],
        "run_processes": copy.deepcopy(validated["run_process_refs"]),
        "serve_parity_transcripts": copy.deepcopy(validated["serve_parity_refs"]),
        "raw_bench_report": validated["bench_report_ref"],
        "actual_throughput_tps": validated["actual_throughput_tps"],
        "frozen_floor_tps": validated["frozen_floor_tps"],
        "throughput_floor_ratio": validated["throughput_floor_ratio"],
        "run_serve_ratio": validated["run_serve_ratio"],
        "repeat_count": 3,
        "run_parity_status": "pass",
    }
    return correctness, performance


def validate_llama_supplemental_manifest(
    path: Path,
    *,
    backend: str,
    staged: dict[str, Any],
    expected_sample_plan_sha256: str,
) -> dict[str, Any]:
    manifest_path = input_manifest(path)
    manifest = require_object(
        read_json(manifest_path, "sampled Llama supplemental manifest"),
        "sampled Llama supplemental manifest",
    )
    expected_fields = {
        "schema_version",
        "artifact_type",
        "status",
        "canonical",
        "collection_scope",
        "model_key",
        "model_id",
        "backend",
        "source",
        "binary_sha256",
        "tarball_sha256",
        "entrypoints",
        "correctness",
        "performance",
        "execution_receipt",
        "sample_plan",
        "g0_gate_manifest",
        "typed_effective_config",
        "server_process",
        "bench_process",
        "run_processes",
        "model_files_sha256",
        "full_matrix_claim",
        "pass_line",
    }
    row = staged["assets"][backend]
    require(
        set(manifest) == expected_fields
        and manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type")
        == "runtime_vnext_sampled_llama_dense_supplemental_manifest"
        and manifest.get("status") == "pass"
        and manifest.get("canonical") is True
        and manifest.get("collection_scope") == COLLECTION_SCOPE
        and manifest.get("model_key") == LLAMA_MODEL_KEY
        and manifest.get("backend") == backend
        and normalize_source(manifest.get("source"), "sampled Llama source")
        == staged["release_candidate"]
        and manifest.get("binary_sha256") == row["binary"]["sha256"]
        and manifest.get("tarball_sha256") == row["tarball"]["sha256"]
        and manifest.get("entrypoints") == ["run", "serve"]
        and manifest.get("full_matrix_claim") is False
        and manifest.get("pass_line")
        == f"{PASS_PREFIXES['llama-supplemental']}: {manifest_path.parent.resolve()}",
        "sampled Llama supplemental manifest identity differs",
    )
    receipt_ref, receipt_path = resolve_ref(
        manifest.get("execution_receipt"),
        "sampled Llama execution receipt",
        root=manifest_path.parent,
    )
    validated = validate_g0_llama_execution_receipt(
        receipt_path,
        backend=backend,
        staged=staged,
        expected_sample_plan_sha256=expected_sample_plan_sha256,
    )
    require(
        receipt_ref["sha256"] == validated["receipt_ref"]["sha256"]
        and manifest.get("model_id") == validated["model_id"]
        and manifest.get("sample_plan") == validated["sample_plan_ref"]
        and manifest.get("g0_gate_manifest") == validated["g0_gate_ref"]
        and manifest.get("typed_effective_config") == validated["typed_config_ref"]
        and manifest.get("server_process") == validated["server_process_ref"]
        and manifest.get("bench_process") == validated["bench_process_ref"]
        and manifest.get("run_processes") == validated["run_process_refs"]
        and manifest.get("model_files_sha256") == validated["model_files_sha256"],
        "sampled Llama manifest execution/config/process refs differ",
    )
    plan = checked_sample_plan()
    expected_correctness, expected_performance = llama_evidence_documents(
        validated=validated, staged=staged, plan=plan
    )
    evidence_refs: dict[str, dict[str, Any]] = {}
    for name, expected in (
        ("correctness", expected_correctness),
        ("performance", expected_performance),
    ):
        ref, evidence_path = resolve_ref(
            manifest.get(name), f"sampled Llama {name}", root=manifest_path.parent
        )
        require(
            read_json(evidence_path, f"sampled Llama {name}") == expected,
            f"sampled Llama {name} does not replay from raw G0 receipts",
        )
        evidence_refs[name] = ref
    return {
        "manifest": manifest,
        "manifest_ref": artifact_ref(manifest_path),
        "correctness_ref": evidence_refs["correctness"],
        "performance_ref": evidence_refs["performance"],
        "receipt_ref": receipt_ref,
        "sample_plan_ref": validated["sample_plan_ref"],
        "typed_config_ref": validated["typed_config_ref"],
        "model_files_sha256": validated["model_files_sha256"],
    }


def make_tarball(path: Path, payload: bytes) -> dict[str, Any]:
    info = tarfile.TarInfo("ferrum")
    info.size = len(payload)
    info.mode = 0o755
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))
    return {
        "archive_path": "ferrum",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def make_staged_fixture(root: Path, source: dict[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True)
    assets: dict[str, Any] = {}
    tag = "v0.8.0-rc.1"
    for index, backend in enumerate(("cpu", "metal", "cuda"), start=1):
        directory = root / "assets" / backend
        directory.mkdir(parents=True)
        workflow_run_id = 201 if backend in {"cpu", "metal"} else 202
        workflow_run_attempt = 2
        workflow_path = (
            ".github/workflows/release-cuda.yml"
            if backend == "cuda"
            else ".github/workflows/release.yml"
        )
        staging_label = "v0.8.0-rc"
        payload = f"fixture-{backend}-binary\n".encode()
        tarball = directory / f"{backend}.tar.gz"
        binary = make_tarball(tarball, payload)
        checksum = directory / f"{backend}.tar.gz.sha256"
        checksum.write_text(f"{file_sha256(tarball)}  {tarball.name}\n", encoding="utf-8")
        version = directory / "version.json"
        write_json(version, {"version": VERSION})
        artifact_archive = directory / "github-artifact.zip"
        artifact_archive.write_bytes(f"fixture-{backend}-artifact\n".encode())
        artifact_archive_ref = artifact_ref(artifact_archive)
        artifact = {
            "id": 100 + index,
            "name": (
                f"{goal.ASSET_NAMES[backend].removesuffix('.tar.gz')}-"
                f"{staging_label}-{source['git_sha']}"
            ),
            "digest": f"sha256:{artifact_archive_ref['sha256']}",
        }
        artifact_manifest = directory / "artifact.json"
        write_json(
            artifact_manifest,
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "runtime_vnext_github_workflow_artifact_manifest",
                "status": "pass",
                "repository": goal.GITHUB_REPOSITORY,
                "workflow_run_id": workflow_run_id,
                "workflow_run": {
                    "id": workflow_run_id,
                    "attempt": workflow_run_attempt,
                    "path": workflow_path,
                    "event": "workflow_dispatch",
                    "head_sha": source["git_sha"],
                    "status": "completed",
                    "conclusion": "success",
                },
                "artifact": artifact,
                "archive": artifact_archive_ref,
                "workflow_inputs": {
                    "release_candidate_sha": source["git_sha"],
                    "release_candidate_tag": tag,
                    "staging_label": staging_label,
                    "publish_release": False,
                },
                "release_candidate": source,
                "release_candidate_tag": tag,
                "publish_release": False,
            },
        )
        dependency = directory / "dependency.json"
        write_json(
            dependency,
            {
                "release_candidate": source,
                "release_candidate_tag": tag,
                "binary_sha256": binary["sha256"],
                "tarball_sha256": file_sha256(tarball),
            },
        )
        row = {
            "backend": backend,
            "workflow_run_id": workflow_run_id,
            "artifact": artifact,
            "artifact_manifest": artifact_ref(artifact_manifest),
            "tarball": artifact_ref(tarball),
            "sha256_file": artifact_ref(checksum),
            "version_manifest": artifact_ref(version),
            "dependency_abi_manifest": artifact_ref(dependency),
            "binary": binary,
        }
        if backend == "cuda":
            row["target_sm"] = "89"
        assets[backend] = row
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_staged_assets_manifest",
        "status": "pass",
        "canonical": True,
        "version": VERSION,
        "publish_release": False,
        "release_candidate": source,
        "release_candidate_tag": tag,
        "artifact_dir": str(root.resolve()),
        "assets": assets,
        "created_at": "2026-08-14T00:00:00Z",
        "pass_line": f"FERRUM RUNTIME VNEXT STAGED ASSETS PASS: {root.resolve()}",
    }
    path = root / "manifest.json"
    write_json(path, manifest)
    return {"path": path, "manifest": manifest, "assets": assets}


def make_common_fixture(root: Path, *, model_key: str, backend: str) -> dict[str, Path]:
    source = {"git_sha": "1" * 40, "git_tree_sha": "2" * 40, "dirty": False}
    staged = make_staged_fixture(root / "staged", source)
    artifact_root = root / "lane"
    binary_path = artifact_root / "build/candidate/ferrum"
    binary_path.parent.mkdir(parents=True)
    binary_payload = f"fixture-{backend}-binary\n".encode()
    binary_path.write_bytes(binary_payload)
    binary_path.chmod(0o755)
    receipt_path = artifact_root / "build/candidate/candidate-build-receipt.json"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": RECEIPT_TYPE,
        "status": "pass",
        "execution_contract": G08_EXECUTION_CONTRACT,
        "source_git_sha": source["git_sha"],
        "source_tree_sha": source["git_tree_sha"],
        "dirty_status": {"is_dirty": False, "status_short": []},
        "hardware_id": f"fixture-{backend}",
        "backend": backend,
        "artifact_root": str(artifact_root.resolve()),
        "repository_root": str(REPO_ROOT),
        "build_mode": "staged-release-asset",
        "bound_at": "2026-08-14T00:00:00Z",
        "source_observations": {
            phase: {
                "source_git_sha": source["git_sha"],
                "source_tree_sha": source["git_tree_sha"],
                "dirty_status": {"is_dirty": False, "status_short": []},
            }
            for phase in ("before", "after")
        },
        "release_version": VERSION,
        "staged_assets_manifest": {
            "kind": "raw-json",
            "path": str(staged["path"].resolve()),
            "sha256": file_sha256(staged["path"]),
        },
        "selected_staged_asset": staged["assets"][backend],
        "staged_metadata_artifacts": {},
        "binary_artifact": kind_ref(binary_path, artifact_root, "binary"),
        "binary_sha256": file_sha256(binary_path),
    }
    write_json(receipt_path, receipt)
    model_files = {"weights.gguf": "3" * 64}
    effective_path = artifact_root / f"correctness/{model_key}/{backend}/effective-config.json"
    write_json(
        effective_path,
        {
            "schema_version": SCHEMA_VERSION,
            "execution_contract": G08_EXECUTION_CONTRACT,
            "source_git_sha": source["git_sha"],
            "source_tree_sha": source["git_tree_sha"],
            "dirty_status": {"is_dirty": False, "status_short": []},
            "binary_sha256": file_sha256(binary_path),
            "model_key": model_key,
            "backend": backend,
            "model_files": model_files,
            "hardware_id": f"fixture-{backend}",
            "typed_effective_config": {"run": {"temperature": 0}, "serve": {"temperature": 0}},
        },
    )
    invocation_path = artifact_root / f"correctness/{model_key}/{backend}/commands/invocation.json"
    write_json(
        invocation_path,
        {
            "mode": "focused-diagnostic",
            "argv": [
                "python3",
                "scripts/release/runtime_vnext_baseline_scenarios.py",
                "--focus-scenario",
                "C17",
            ],
        },
    )
    plan_document = checked_sample_plan()["manifest"]
    correctness_plan = require_object(
        plan_document["correctness"][model_key][backend],
        "fixture correctness plan",
    )
    case_count = positive_int(
        correctness_plan.get("case_count"), "fixture C17 case count"
    )
    require(case_count % 6 == 0, "fixture C17 case count must partition by 6")
    cases_per_partition = case_count // 6
    case_refs = []
    selected_ids = []
    ordinal = 0
    for variant in ("chinese", "emoji", "combining"):
        for entrypoint in ("run", "serve"):
            for _ in range(cases_per_partition):
                ordinal += 1
                case_id = f"c17-{ordinal:03d}"
                selected_ids.append(case_id)
                case_dir = artifact_root / f"correctness/{model_key}/{backend}/scenarios/C17/cases/{case_id}"
                case_dir.mkdir(parents=True)
                stdout = case_dir / "stdout.log"
                stderr = case_dir / "stderr.log"
                stdout.write_text("valid unicode output\n", encoding="utf-8")
                stderr.write_text("completed\n", encoding="utf-8")
                case_path = case_dir / "case.json"
                write_json(
                    case_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "case_id": case_id,
                        "scenario_id": "C17",
                        "ordinal": ordinal,
                        "status": "pass",
                        "source_git_sha": source["git_sha"],
                        "source_tree_sha": source["git_tree_sha"],
                        "binary_sha256": file_sha256(binary_path),
                        "model_key": model_key,
                        "backend": backend,
                        "model_files": model_files,
                        "entrypoint": entrypoint,
                        "variant": variant,
                        "expected_outcome": {"expected_status": "pass"},
                        "observed_outcome": {"status": "pass", "failure_class": None},
                        "observed": {
                            "expected_marker": {"chinese": "中文", "emoji": "🙂", "combining": "x́"}[variant],
                            "replacement_char_count": 0,
                            "mojibake_count": 0,
                            "partial_character_chunk_count": 0,
                            "error_count": 0,
                            "bad_output_count": 0,
                        },
                        "checks": {"execution": True, "oracle": True},
                        "artifacts": {
                            "stdout": kind_ref(stdout, artifact_root, "stdout-log"),
                            "stderr": kind_ref(stderr, artifact_root, "stderr-log"),
                            "effective_config": kind_ref(effective_path, artifact_root),
                        },
                    },
                )
                case_refs.append(kind_ref(case_path, artifact_root))
    focused_path = artifact_root / f"correctness/{model_key}/{backend}/focused-report.json"
    write_json(
        focused_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "runtime-vnext-focused-diagnostic",
            "execution_contract": G08_EXECUTION_CONTRACT,
            "status": "pass",
            "decision": "KEEP",
            "formal_pass_allowed": False,
            "source_git_sha": source["git_sha"],
            "source_tree_sha": source["git_tree_sha"],
            "dirty_status": {"is_dirty": False, "status_short": []},
            "binary_sha256": file_sha256(binary_path),
            "model_key": model_key,
            "backend": backend,
            "model_files": model_files,
            "hardware_id": f"fixture-{backend}",
            "effective_config": kind_ref(effective_path, artifact_root),
            "binary_build_receipt": kind_ref(receipt_path, artifact_root),
            "executor_invocation": kind_ref(invocation_path, artifact_root),
            "scope": {
                "kind": "focused-diagnostic",
                "requested_case_ids": [],
                "requested_scenario_ids": ["C17"],
                "selected_case_ids": selected_ids,
                "selected_scenario_ids": ["C17"],
                "case_count": case_count,
                "canonical_case_count": 783,
            },
            "observed_status_counts": {"pass": case_count},
            "cases": case_refs,
            "artifact_path": str(focused_path.resolve()),
            "pass_line": f"FERRUM RUNTIME VNEXT FOCUSED DIAGNOSTIC KEEP: {focused_path.resolve()}",
        },
    )
    plan = plan_document["performance"][model_key][backend]
    requests = 2
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
    bench_path = root / "bench.json"
    report = {
        "model": model_key,
        "backend": backend,
        "scenario": "closed_loop",
        "concurrency": plan["concurrency"],
        "n_gen": 8,
        "n_repeats": 3,
        "n_requests_per_run": requests,
        "output_token_count_source": "usage",
        "completed_per_run": [requests] * 3,
        "errored_per_run": [0, 0, 0],
        "repeat_metrics": [
            {
                "repeat": index,
                "expected_requests": requests,
                "completed_requests": requests,
                "errored_requests": 0,
                "output_token_count_source": "usage",
                "quality_issues": quality,
                "warmup_quality_issues": quality,
            }
            for index in range(1, 4)
        ],
        "actual_input_tokens_per_request": [[8] * requests for _ in range(3)],
        "output_tokens_per_request": [[8] * requests for _ in range(3)],
        "output_throughput_tps": {"mean": 10.0},
    }
    for name in quality:
        report[f"{name}_per_run"] = [0, 0, 0]
    write_json(bench_path, report)
    command_path = root / "bench-command.json"
    write_json(
        command_path,
        [
            str(binary_path.resolve()),
            "bench-serve",
            "--model",
            model_key,
            "--dataset",
            "random",
            "--concurrency-sweep",
            str(plan["concurrency"]),
            "--num-prompts",
            str(requests),
            "--random-output-len",
            "8",
            "--n-repeats",
            "3",
            "--seed",
            "9271",
            "--fail-on-error",
            "--require-ci",
        ],
    )
    parity_root = root / "parity"
    parity_root.mkdir()
    records = {}
    for entrypoint in ("run", "serve"):
        stdout = parity_root / f"{entrypoint}.stdout"
        stderr = parity_root / f"{entrypoint}.stderr"
        runtime_config = parity_root / f"{entrypoint}-effective.json"
        stdout.write_text("valid generated output\n", encoding="utf-8")
        stderr.write_text("completed cleanly\n", encoding="utf-8")
        write_json(runtime_config, {"typed": True})
        records[entrypoint] = {
            "status": "pass",
            "command_line": [str(binary_path.resolve()), entrypoint, model_key],
            "returncode": 0,
            "output_token_count": 8,
            "error_count": 0,
            "stdout": artifact_ref(stdout),
            "stderr": artifact_ref(stderr),
            "effective_config": artifact_ref(runtime_config),
        }
    parity_path = parity_root / "parity.json"
    write_json(
        parity_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_sampled_run_parity_raw_evidence",
            "status": "pass",
            "collection_scope": COLLECTION_SCOPE,
            "source": source,
            "model_key": model_key,
            "backend": backend,
            "binary_sha256": file_sha256(binary_path),
            "model_files": model_files,
            "entrypoints": ["run", "serve"],
            "run_parity_status": "pass",
            "failure_count": 0,
            "hardware": {"kind": f"fixture-{backend}"},
            "run": records["run"],
            "serve": records["serve"],
            "comparison": {
                "contract": "same-prompt-generation-options-v1",
                "status": "pass",
                "prompt_sha256": "4" * 64,
                "sample_count": 2,
                "failure_count": 0,
            },
        },
    )
    return {
        "staged": staged["path"],
        "receipt": receipt_path,
        "effective": effective_path,
        "focused": focused_path,
        "bench": bench_path,
        "command": command_path,
        "parity": parity_path,
    }


def make_llama_execution_fixture(root: Path, *, backend: str) -> dict[str, Path]:
    import runtime_vnext_g0_llama_sampled_execution as execution

    source = {"git_sha": "1" * 40, "git_tree_sha": "2" * 40, "dirty": False}
    staged = make_staged_fixture(root / "staged", source)
    lane = root / "execution"
    lane.mkdir(parents=True)
    binary = lane / "inputs/ferrum"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(f"fixture-{backend}-binary\n".encode())
    binary.chmod(0o755)
    row = staged["assets"][backend]
    require(file_sha256(binary) == row["binary"]["sha256"], "fixture staged binary differs")
    model_id = f"fixture-llama-{backend}"
    tokenizer_dir = lane / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer_json.write_text('{"fixture":true}\n', encoding="utf-8")
    model_files = {
        "model.bin": "5" * 64,
        "tokenizer.json": file_sha256(tokenizer_json),
    }
    raw_typed = {
        "schema_version": 1,
        "backend": backend,
        "preset": "fixture-llama",
        "model_capabilities": {"architecture": "llama"},
        "workload_profile": {"serving_mode": "serve"},
    }
    raw_typed_path = lane / "serve-effective-config.raw.json"
    write_json(raw_typed_path, raw_typed)
    typed_path = lane / "typed-effective-config.json"
    write_json(
        typed_path,
        {
            "source": source,
            "model_key": LLAMA_MODEL_KEY,
            "model_id": model_id,
            "backend": backend,
            "binary_sha256": row["binary"]["sha256"],
            "model_files": model_files,
            "typed_effective_config": raw_typed,
        },
    )
    typed_ref = artifact_ref(typed_path)
    environment = {"PATH": "/usr/bin"}

    def observation(name: str, argv: list[str], start: int, finish: int) -> dict[str, Any]:
        stdout = lane / f"{name}.stdout"
        stderr = lane / f"{name}.stderr"
        stdout.write_text("valid generated output\n", encoding="utf-8")
        stderr.write_text("completed cleanly\n", encoding="utf-8")
        return {
            "argv": argv,
            "pid": 1000 + start,
            "pgid": 1000 + start,
            "started_at": f"2026-08-14T00:{start:02d}:00+00:00",
            "finished_at": f"2026-08-14T00:{finish:02d}:00+00:00",
            "returncode": 0,
            "stdout_path": stdout,
            "stderr_path": stderr,
        }

    report_path = lane / "bench-c1.json"
    floor = checked_sample_plan()["manifest"]["performance"][LLAMA_MODEL_KEY][backend]["floor"]["value"]
    requests = 8
    report = {
        "model": model_id,
        "backend": backend,
        "scenario": "closed_loop",
        "concurrency": 1,
        "n_repeats": 3,
        "n_gen": 128,
        "warmup_requests": 1,
        "n_requests_per_run": requests,
        "output_token_count_source": "usage",
        "completed_per_run": [requests] * 3,
        "errored_per_run": [0, 0, 0],
        "repeat_metrics": [
            {
                "repeat": index,
                "expected_requests": requests,
                "completed_requests": requests,
                "errored_requests": 0,
                "warmup_expected": 1,
                "warmup_completed": 1,
                "warmup_errored": 0,
                "output_token_count_source": "usage",
                "quality_issues": {"bad_output": 0},
                "warmup_quality_issues": {"bad_output": 0},
            }
            for index in range(1, 4)
        ],
        "output_throughput_tps": {"mean": float(floor) * 1.01},
        "actual_input_tokens_per_request": [[64] * requests for _ in range(3)],
        "output_tokens_per_request": [[128] * requests for _ in range(3)],
    }
    for name in (
        "bad_output",
        "malformed_stream",
        "missing_done",
        "duplicate_done",
        "zero_output_tokens",
        "stream_bulk_flush",
        "http_500",
        "panic",
    ):
        report[f"{name}_per_run"] = [0, 0, 0]
    write_json(report_path, report)
    server_observation = observation(
        "server",
        execution.command_for_serve(
            binary, model_id, backend, 19300, raw_typed_path, []
        ),
        1,
        10,
    )
    server_ref = execution.process_receipt(
        lane / "server-process-receipt.json",
        observation=server_observation,
        role="ferrum-serve",
        source=source,
        binary_sha256=row["binary"]["sha256"],
        model_id=model_id,
        typed_config_sha256=typed_ref["sha256"],
        environment=environment,
        artifacts={"effective_config": artifact_ref(raw_typed_path)},
        server_sha256=None,
    )
    bench_observation = observation(
        "bench",
        execution.command_for_bench(
            binary, model_id, tokenizer_dir, 19300, report_path, backend
        ),
        2,
        3,
    )
    bench_ref = execution.process_receipt(
        lane / "bench-process-receipt.json",
        observation=bench_observation,
        role="ferrum-bench-serve",
        source=source,
        binary_sha256=row["binary"]["sha256"],
        model_id=model_id,
        typed_config_sha256=typed_ref["sha256"],
        environment=environment,
        artifacts={"report": artifact_ref(report_path)},
        server_sha256=server_ref["sha256"],
    )
    run_raw_paths = [
        lane / f"run-effective-config-{ordinal}.raw.json"
        for ordinal in range(1, 4)
    ]
    run_raw_refs: list[dict[str, Any]] = []
    run_typed_refs: list[dict[str, Any]] = []
    for ordinal, raw_path in enumerate(run_raw_paths, start=1):
        raw = {
            "schema_version": 1,
            "backend": backend,
            "preset": "fixture-llama",
            "model_capabilities": {"architecture": "llama"},
            "workload_profile": {"serving_mode": "run", "ordinal": ordinal},
        }
        write_json(raw_path, raw)
        raw_ref = artifact_ref(raw_path)
        typed_run_path = lane / f"run-typed-effective-config-{ordinal}.json"
        write_json(
            typed_run_path,
            {
                "source": source,
                "model_key": LLAMA_MODEL_KEY,
                "model_id": model_id,
                "backend": backend,
                "binary_sha256": row["binary"]["sha256"],
                "model_files": model_files,
                "entrypoint": "run",
                "raw_effective_config": raw_ref,
                "typed_effective_config": raw,
            },
        )
        run_raw_refs.append(raw_ref)
        run_typed_refs.append(artifact_ref(typed_run_path))
    run_observations = [
        observation(
            f"run-{ordinal}",
            execution.command_for_run(
                binary, model_id, backend, run_raw_paths[ordinal - 1]
            ),
            3 + ordinal,
            4 + ordinal,
        )
        for ordinal in range(1, 4)
    ]
    run_refs = [
        execution.process_receipt(
            lane / f"run-process-receipt-{ordinal}.json",
            observation=item,
            role="ferrum-run",
            source=source,
            binary_sha256=row["binary"]["sha256"],
            model_id=model_id,
            typed_config_sha256=run_typed_refs[ordinal - 1]["sha256"],
            environment=environment,
            artifacts={
                "raw_effective_config": run_raw_refs[ordinal - 1],
                "typed_effective_config": run_typed_refs[ordinal - 1],
            },
            server_sha256=None,
        )
        for ordinal, item in enumerate(run_observations, start=1)
    ]
    parity_prompt = "7" * 64

    def transcript_ref(
        name: str,
        scenario_id: str,
        entrypoint: str,
        process_sha: str,
        typed_config_sha: str,
        stdout: Path,
        stderr: Path,
        *,
        done: int = 0,
    ) -> dict[str, Any]:
        return execution.transcript(
            lane / f"{name}.json",
            scenario_id=scenario_id,
            entrypoint=entrypoint,
            source=source,
            binary_sha256=row["binary"]["sha256"],
            model_id=model_id,
            typed_config_sha256=typed_config_sha,
            process_sha256=process_sha,
            prompt_sha256=parity_prompt,
            output_tokens=10,
            elapsed_ms=100.0,
            finish_reason="stop",
            done=done,
            stdout=stdout,
            stderr=stderr,
        )

    run_transcripts = [
        transcript_ref(
            f"run-parity-{ordinal}",
            "run-parity",
            "run",
            run_refs[ordinal - 1]["sha256"],
            run_typed_refs[ordinal - 1]["sha256"],
            run_observations[ordinal - 1]["stdout_path"],
            run_observations[ordinal - 1]["stderr_path"],
        )
        for ordinal in range(1, 4)
    ]
    serve_transcripts = [
        transcript_ref(
            f"serve-parity-{ordinal}",
            "serve-parity",
            "serve",
            server_ref["sha256"],
            typed_ref["sha256"],
            server_observation["stdout_path"],
            server_observation["stderr_path"],
        )
        for ordinal in range(1, 4)
    ]
    correctness = {
        "run-multiturn": transcript_ref(
            "correctness-run",
            "run-multiturn",
            "run",
            run_refs[0]["sha256"],
            run_typed_refs[0]["sha256"],
            run_observations[0]["stdout_path"],
            run_observations[0]["stderr_path"],
        ),
        "serve-multiturn": transcript_ref(
            "correctness-serve",
            "serve-multiturn",
            "serve",
            server_ref["sha256"],
            typed_ref["sha256"],
            server_observation["stdout_path"],
            server_observation["stderr_path"],
        ),
        "serve-stream": transcript_ref(
            "correctness-stream",
            "serve-stream",
            "serve",
            server_ref["sha256"],
            typed_ref["sha256"],
            server_observation["stdout_path"],
            server_observation["stderr_path"],
            done=1,
        ),
    }
    hardware = (
        {
            "contract": "one-rtx-4090-24gb",
            "id": "fixture-rtx4090",
            "accelerator_count": 1,
            "accelerator_model": "NVIDIA GeForce RTX 4090",
            "memory_bytes": 24 * 1024**3,
        }
        if backend == "cuda"
        else {
            "contract": "apple-m1-max-24core-32gib",
            "id": "fixture-m1max",
            "accelerator_count": 1,
            "accelerator_model": "Apple M1 Max",
            "gpu_core_count": 24,
            "memory_bytes": 32 * 1024**3,
        }
    )
    gate_path = lane / "gate.manifest.json"
    write_json(
        gate_path,
        {
            "schema_version": 1,
            "status": "pass",
            "lane": f"g0-{backend}-llama-dense-sampled",
            "git_sha": source["git_sha"],
            "binary": {"path": str(binary), "sha256": row["binary"]["sha256"]},
            "model": model_id,
            "started_at": "2026-08-14T00:00:00+00:00",
            "finished_at": "2026-08-14T00:20:00+00:00",
            "child_returncode": 0,
            "pass_line": f"FERRUM GATE {backend}-llama-dense-sampled PASS: {lane}",
        },
    )
    receipt_path = lane / "manifest.json"
    plan = checked_sample_plan()
    write_json(
        receipt_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_g0_llama_dense_execution_receipt",
            "status": "pass",
            "producer": "g0-llama-dense-execution-binding-v1",
            "source": source,
            "model_key": LLAMA_MODEL_KEY,
            "model_id": model_id,
            "backend": backend,
            "hardware": hardware,
            "binary_artifact": artifact_ref(binary),
            "binary_sha256": row["binary"]["sha256"],
            "model_files": model_files,
            "model_files_sha256": canonical_json_sha256(model_files),
            "typed_effective_config": typed_ref,
            "staged_assets_manifest": artifact_ref(staged["path"]),
            "sample_plan": plan["ref"],
            "g0_gate_manifest": artifact_ref(gate_path),
            "server_process": server_ref,
            "bench_process": bench_ref,
            "run_processes": run_refs,
            "run_parity_transcripts": run_transcripts,
            "correctness_transcripts": correctness,
            "serve_parity_transcripts": serve_transcripts,
            "bench_report": artifact_ref(report_path),
        },
    )
    return {"staged": staged["path"], "receipt": receipt_path}


def expect_reject(label: str, callback: Callable[[], Any]) -> None:
    try:
        callback()
    except (SampledFinalError, goal.GoalGateError, KeyError, OSError, TypeError, ValueError):
        return
    raise SampledFinalError(f"hostile fixture {label} unexpectedly passed")


def self_test() -> int:
    import runtime_vnext_g0_llama_sampled_execution as llama_execution

    require(llama_execution.self_test() == 0, "G0 Llama execution CLI self-test failed")
    with tempfile.TemporaryDirectory(prefix="ferrum-sampled-final-") as temporary:
        root = Path(temporary)
        lane_outputs: dict[tuple[str, str], Path] = {}
        lane_fixtures: dict[tuple[str, str], dict[str, Path]] = {}
        # Six real planner profiles: M1 is 60/300, M2/M3 are 6/30.
        for model_key in sorted(MODEL_KEYS):
            for backend in sorted(BACKENDS):
                fixture = make_common_fixture(
                    root / f"{model_key}-{backend}",
                    model_key=model_key,
                    backend=backend,
                )
                out = build_correctness(
                    argparse.Namespace(
                        backend=backend,
                        staged_assets=fixture["staged"],
                        binary_build_receipt=fixture["receipt"],
                        effective_config=fixture["effective"],
                        model_key=model_key,
                        focused_report=fixture["focused"],
                        out=root / f"out-{model_key}-{backend}",
                    )
                )
                staged = goal.validate_staged_assets_manifest(fixture["staged"])
                lane_key = f"{model_key.split('-', 1)[0]}_{backend}"
                accepted = goal.correctness_lane_input(
                    out,
                    lane_key=lane_key,
                    model_key=model_key,
                    backend=backend,
                    release_candidate=staged["release_candidate"],
                    staged=staged,
                )
                expected_count = 60 if model_key == "m1-qwen35-4b" else 6
                require(
                    accepted["sample_count"] == expected_count
                    and accepted["comparison_count"] == expected_count * 5
                    and accepted["raw_status"] == "keep"
                    and accepted["sample_selection_status"] == "pass",
                    f"{model_key}/{backend} self-test denominator/status differs",
                )
                lane_outputs[(model_key, backend)] = out
                lane_fixtures[(model_key, backend)] = fixture

        # Both G0 Llama hardware lanes consume the real execution-receipt CLI
        # schema, including 3 run processes and the same server/bench window.
        llama_outputs: dict[str, Path] = {}
        llama_fixtures: dict[str, dict[str, Path]] = {}
        for backend in sorted(BACKENDS):
            fixture = make_llama_execution_fixture(root / f"llama-{backend}", backend=backend)
            out = build_llama_supplemental(
                argparse.Namespace(
                    backend=backend,
                    staged_assets=fixture["staged"],
                    execution_receipt=fixture["receipt"],
                    out=root / f"out-llama-{backend}",
                )
            )
            staged = goal.validate_staged_assets_manifest(fixture["staged"])
            accepted = goal.llama_supplemental_input(
                out,
                backend=backend,
                release_candidate=staged["release_candidate"],
                staged=staged,
            )
            require(
                accepted["correctness_status"] == "pass"
                and accepted["performance_status"] == "pass"
                and accepted["sample_plan_sha256"]
                == checked_sample_plan()["ref"]["sha256"],
                f"Llama {backend} strict goal consumer differs",
            )
            llama_outputs[backend] = out
            llama_fixtures[backend] = fixture

        # Hostile C17 source mutation is rejected after recursively opening raw cases.
        fixture = lane_fixtures[("m2-qwen35-35b-a3b", "cuda")]
        staged = staged_context(fixture["staged"], "cuda")
        receipt = validate_receipt(fixture["receipt"], staged=staged, backend="cuda")
        effective = raw_effective_config(
            fixture["effective"],
            model_key="m2-qwen35-35b-a3b",
            backend="cuda",
            staged=staged,
            receipt=receipt,
        )
        hostile_source = receipt["artifact_root"] / "hostile-c17-source.json"
        source_doc = read_json(fixture["focused"], "focused fixture")
        source_doc["source_git_sha"] = "9" * 40
        write_json(hostile_source, source_doc)
        expect_reject(
            "C17 source",
            lambda: validate_focused_c17(
                hostile_source,
                model_key="m2-qwen35-35b-a3b",
                backend="cuda",
                staged=staged,
                receipt=receipt,
                effective=effective,
                plan=checked_sample_plan(),
            ),
        )

        # Hostile Llama receipt cannot swap the bound server process or claim a
        # two-repeat benchmark; both attacks are below the derived manifest.
        llama = llama_fixtures["cuda"]
        llama_staged = staged_context(llama["staged"], "cuda")
        hostile_receipt = root / "hostile-llama-server.json"
        receipt_doc = read_json(llama["receipt"], "Llama receipt fixture")
        receipt_doc["bench_process"] = copy.deepcopy(receipt_doc["bench_process"])
        receipt_doc["bench_process"]["sha256"] = "a" * 64
        write_json(hostile_receipt, receipt_doc)
        expect_reject(
            "Llama server/bench receipt binding",
            lambda: validate_g0_llama_execution_receipt(
                hostile_receipt,
                backend="cuda",
                staged=llama_staged,
                expected_sample_plan_sha256=checked_sample_plan()["ref"]["sha256"],
            ),
        )

        # A run process cannot omit its independently emitted config even if
        # the attacker refreshes the process reference in the parent receipt.
        missing_config = llama_fixtures["metal"]
        missing_staged = staged_context(missing_config["staged"], "metal")
        missing_parent = read_json(
            missing_config["receipt"], "missing run config parent fixture"
        )
        _, missing_process_path = resolve_ref(
            missing_parent["run_processes"][0],
            "missing run config process fixture",
            root=missing_config["receipt"].parent,
        )
        missing_process = read_json(
            missing_process_path, "missing run config process fixture"
        )
        del missing_process["artifacts"]["raw_effective_config"]
        write_json(missing_process_path, missing_process)
        missing_parent["run_processes"][0] = artifact_ref(missing_process_path)
        missing_parent_path = root / "hostile-llama-missing-run-config.json"
        write_json(missing_parent_path, missing_parent)
        expect_reject(
            "Llama missing run effective config",
            lambda: validate_g0_llama_execution_receipt(
                missing_parent_path,
                backend="metal",
                staged=missing_staged,
                expected_sample_plan_sha256=checked_sample_plan()["ref"]["sha256"],
            ),
        )

        # A fully rehashed run config chain still fails when its core backend
        # identity differs from the same staged serve execution.
        wrong_config = make_llama_execution_fixture(
            root / "hostile-wrong-run-config", backend="metal"
        )
        wrong_staged = staged_context(wrong_config["staged"], "metal")
        wrong_parent = read_json(
            wrong_config["receipt"], "wrong run config parent fixture"
        )
        _, wrong_process_path = resolve_ref(
            wrong_parent["run_processes"][0],
            "wrong run config process fixture",
            root=wrong_config["receipt"].parent,
        )
        wrong_process = read_json(
            wrong_process_path, "wrong run config process fixture"
        )
        _, wrong_raw_path = resolve_ref(
            wrong_process["artifacts"]["raw_effective_config"],
            "wrong run raw config fixture",
            root=wrong_config["receipt"].parent,
        )
        wrong_raw = read_json(wrong_raw_path, "wrong run raw config fixture")
        wrong_raw["backend"] = "cuda"
        write_json(wrong_raw_path, wrong_raw)
        wrong_raw_ref = artifact_ref(wrong_raw_path)
        _, wrong_typed_path = resolve_ref(
            wrong_process["artifacts"]["typed_effective_config"],
            "wrong run typed config fixture",
            root=wrong_config["receipt"].parent,
        )
        wrong_typed = read_json(wrong_typed_path, "wrong run typed config fixture")
        wrong_typed["raw_effective_config"] = wrong_raw_ref
        wrong_typed["typed_effective_config"] = wrong_raw
        write_json(wrong_typed_path, wrong_typed)
        wrong_typed_ref = artifact_ref(wrong_typed_path)
        wrong_process["artifacts"] = {
            "raw_effective_config": wrong_raw_ref,
            "typed_effective_config": wrong_typed_ref,
        }
        wrong_process["typed_config_sha256"] = wrong_typed_ref["sha256"]
        write_json(wrong_process_path, wrong_process)
        wrong_parent["run_processes"][0] = artifact_ref(wrong_process_path)
        wrong_parent_path = root / "hostile-llama-wrong-run-config.json"
        write_json(wrong_parent_path, wrong_parent)
        expect_reject(
            "Llama wrong run effective config",
            lambda: validate_g0_llama_execution_receipt(
                wrong_parent_path,
                backend="metal",
                staged=wrong_staged,
                expected_sample_plan_sha256=checked_sample_plan()["ref"]["sha256"],
            ),
        )

        raw = read_json(llama["receipt"], "Llama receipt fixture")
        _, bench_path = resolve_ref(
            raw["bench_report"], "Llama bench fixture", root=llama["receipt"].parent
        )
        bench_doc = read_json(bench_path, "Llama bench fixture")
        bench_doc["n_repeats"] = 2
        write_json(bench_path, bench_doc)
        expect_reject(
            "Llama two repeats",
            lambda: validate_g0_llama_execution_receipt(
                llama["receipt"],
                backend="cuda",
                staged=llama_staged,
                expected_sample_plan_sha256=checked_sample_plan()["ref"]["sha256"],
            ),
        )
    print("FERRUM RUNTIME VNEXT SAMPLED FINAL SELFTEST PASS")
    return 0


def required(args: argparse.Namespace, names: Iterable[str]) -> None:
    missing = [f"--{name.replace('_', '-')}" for name in names if getattr(args, name) is None]
    require(not missing, f"{args.mode} requires {', '.join(missing)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("correctness", "performance", "llama-supplemental")
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model-key")
    parser.add_argument("--backend", choices=sorted(BACKENDS))
    parser.add_argument("--staged-assets", type=Path)
    parser.add_argument("--binary-build-receipt", type=Path)
    parser.add_argument("--effective-config", type=Path)
    parser.add_argument("--focused-report", type=Path)
    parser.add_argument("--bench-report", type=Path)
    parser.add_argument("--bench-command", type=Path)
    parser.add_argument("--run-parity-report", type=Path)
    parser.add_argument("--execution-receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    require(args.mode is not None, "mode is required")
    common = (
        "out",
        "backend",
        "staged_assets",
        "binary_build_receipt",
        "effective_config",
    )
    requirements = {
        "correctness": (*common, "model_key", "focused_report"),
        "performance": (
            *common,
            "model_key",
            "bench_report",
            "bench_command",
            "run_parity_report",
        ),
        "llama-supplemental": (
            "out",
            "backend",
            "staged_assets",
            "execution_receipt",
        ),
    }
    required(args, requirements[args.mode])
    builders = {
        "correctness": build_correctness,
        "performance": build_performance,
        "llama-supplemental": build_llama_supplemental,
    }
    builders[args.mode](args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SampledFinalError, goal.GoalGateError) as error:
        print(f"FERRUM RUNTIME VNEXT SAMPLED FINAL FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
