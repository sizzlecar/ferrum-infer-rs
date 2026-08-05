#!/usr/bin/env python3
"""Validate the G08A Qwen3.5 source, ownership, and legacy-removal contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import runtime_vnext_inventory as inventory  # noqa: E402


CONFIG_PATH = SCRIPT_DIR / "configs/runtime_vnext_g08a_source_contract.json"
INVENTORY_REVIEW_PATH = SCRIPT_DIR / "configs/runtime_vnext_inventory_review.json"
BOUNDED_COMMAND = SCRIPT_DIR / "bounded_command.py"
PASS_PREFIX = "FERRUM RUNTIME VNEXT G08A SOURCE OWNERSHIP PASS"
FAIL_PREFIX = "FERRUM RUNTIME VNEXT G08A SOURCE OWNERSHIP FAIL"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT G08A SOURCE OWNERSHIP SELFTEST PASS"
PROOF_PREFIX = "FERRUM G04 SYNTHETIC LIFECYCLE KEEP: "
LIFECYCLE_CATEGORIES = [
    "setup",
    "admission",
    "state_transition",
    "finalize",
    "cleanup",
]
LIFECYCLE_IMPLEMENTATION_OWNER = "shared.execution_runtime"
FORBIDDEN_EXCLUDED_PROVIDER_FRAGMENTS = [
    "PlanRuntimeResources",
    "ExecutionBatchParticipants",
    "StepResourceLease",
    "CompletionReaper",
    "RequestResourceAdmission",
    "AdmissionPressureAction",
    "PlanRuntimeCloseOutcome",
    "ResourceTransaction",
    "try_begin_step",
    "prepare_invocation",
    "finalize_participants",
]
DOES_NOT_PROVE = [
    "M1 CUDA or Metal C01-C21 model-matrix correctness",
    "M1 product-binary legacy selection count",
    "G08A complete Metal numerical reference",
    "G08A historical production mutation kill",
    "G08A CUDA or Metal performance smoke",
    "G08A final PASS",
]
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_INDEX_EXCLUDED = {
    "manifest.json",
    "gate.manifest.json",
    "run_gate.child.stdout",
    "run_gate.child.stderr",
    "run_gate.child.command.json",
}


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read {label} {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file() and not resolved.is_symlink(), f"missing regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def git_text(*args: str) -> str:
    process = subprocess.run(
        [
            "git",
            "-c",
            "core.preloadindex=false",
            "-c",
            "index.threads=1",
            *args,
        ],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    require(
        process.returncode == 0,
        process.stderr.strip() or f"git {' '.join(args)} failed",
    )
    return process.stdout.strip()


def clean_source() -> dict[str, Any]:
    status = [line for line in git_text("status", "--short", "--untracked-files=all").splitlines() if line]
    require(not status, f"G08A source gate requires a clean checkout: {status}")
    source = {
        "git_sha": git_text("rev-parse", "HEAD"),
        "git_tree_sha": git_text("rev-parse", "HEAD^{tree}"),
        "dirty": False,
        "status_short": [],
    }
    require(
        GIT_SHA_RE.fullmatch(source["git_sha"]) is not None
        and GIT_SHA_RE.fullmatch(source["git_tree_sha"]) is not None,
        "current source identity is invalid",
    )
    return source


def validate_config(config: dict[str, Any]) -> None:
    require(config.get("schema_version") == 1 and config.get("goal") == "G08A", "source contract identity differs")
    baseline = config.get("frozen_baseline")
    require(isinstance(baseline, dict), "frozen_baseline must be an object")
    require(
        GIT_SHA_RE.fullmatch(str(baseline.get("git_sha"))) is not None
        and GIT_SHA_RE.fullmatch(str(baseline.get("git_tree_sha"))) is not None
        and SHA256_RE.fullmatch(str(baseline.get("inventory_sha256"))) is not None
        and SHA256_RE.fullmatch(str(baseline.get("inventory_review_sha256"))) is not None,
        "frozen baseline identities are invalid",
    )
    limits = config.get("limits")
    require(isinstance(limits, dict), "limits must be an object")
    require(
        limits.get("minimum_scaffolding_reduction_ratio") == 0.6
        and limits.get("maximum_provider_file_count") == 8
        and limits.get("maximum_provider_glue_production_loc") == 1500
        and limits.get("maximum_family_file_production_loc") == 5000,
        "G08A source limits differ from the goal",
    )


def validate_baseline_inputs(
    baseline_path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = config["frozen_baseline"]
    require(sha256_file(baseline_path) == frozen["inventory_sha256"], "baseline inventory SHA256 differs")
    require(
        sha256_file(INVENTORY_REVIEW_PATH) == frozen["inventory_review_sha256"],
        "baseline inventory review SHA256 differs",
    )
    baseline = read_json(baseline_path, "baseline inventory")
    require(baseline.get("schema_version") == 1, "baseline inventory schema differs")
    require(
        baseline.get("git")
        == {
            "dirty": False,
            "sha": frozen["git_sha"],
            "status_short": [],
            "tree_sha": frozen["git_tree_sha"],
        },
        "baseline inventory source identity differs",
    )
    require(
        baseline.get("analyzer", {}).get("path")
        == "scripts/release/runtime_vnext_inventory.py",
        "baseline inventory analyzer differs",
    )
    review = read_json(INVENTORY_REVIEW_PATH, "baseline inventory review")
    require(
        review.get("schema_version") == 1
        and review.get("reviewed_at_git_sha") == frozen["git_sha"]
        and review.get("unresolved_count") == 0,
        "baseline inventory review is not closed over the frozen source",
    )
    return baseline, review


def git_file_at(revision: str, path: str) -> bytes:
    process = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(process.returncode == 0, f"cannot read frozen source {revision}:{path}")
    return process.stdout


def validate_reviewed_source(path: str, revision: str, label: str) -> None:
    require(GIT_SHA_RE.fullmatch(revision) is not None, f"{label} review SHA is invalid")
    require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0,
        f"{label} review SHA is not an ancestor of HEAD",
    )
    require(
        git_file_at(revision, path) == (REPO_ROOT / path).read_bytes(),
        f"{label} changed after its recorded review SHA",
    )


def span_for_review(
    spans: list[dict[str, Any]],
    symbol: str,
    line_hints: list[int],
    label: str,
) -> dict[str, Any]:
    matches = [
        span
        for span in spans
        if span["name"] == symbol
        and span["logical_loc_by_classification"].get("production", 0) > 0
        and any(span["start_line"] <= line <= span["end_line"] for line in line_hints)
    ]
    require(len(matches) == 1, f"{label} did not resolve to exactly one production function")
    return matches[0]


def baseline_scaffolding(
    baseline: dict[str, Any],
    review: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    frozen_sha = config["frozen_baseline"]["git_sha"]
    inventory_files = {row["path"]: row for row in baseline["files"]}
    rows = [
        row
        for row in review["reviews"]
        if row.get("classification") == "scaffolding-owned"
        and "qwen35" in str(row.get("path", "")).lower()
    ]
    require(rows, "frozen Qwen3.5 scaffolding review is empty")
    source_cache: dict[str, tuple[bytes, list[dict[str, Any]]]] = {}
    functions = []
    for row in rows:
        path = row["path"]
        if path not in source_cache:
            source = git_file_at(frozen_sha, path)
            require(path in inventory_files, f"frozen reviewed path is absent from baseline inventory: {path}")
            require(
                sha256_bytes(source) == inventory_files[path]["sha256"],
                f"frozen reviewed source identity differs: {path}",
            )
            source_cache[path] = (
                source,
                inventory.rust_function_spans(source.decode("utf-8")),
            )
        span = span_for_review(
            source_cache[path][1],
            row["symbol"],
            row["line_hints"],
            f"frozen {path}:{row['symbol']}",
        )
        functions.append(
            {
                "path": path,
                "symbol": row["symbol"],
                "start_line": span["start_line"],
                "end_line": span["end_line"],
                "production_loc": span["logical_loc_by_classification"]["production"],
            }
        )
    return {
        "function_count": len(functions),
        "file_count": len({row["path"] for row in functions}),
        "production_loc": sum(row["production_loc"] for row in functions),
        "functions": functions,
    }


def production_line_numbers(text: str) -> set[int]:
    _, _, code_lines, classifications = inventory.logical_loc("rust", text, "production")
    return {
        line_number
        for line_number, (code, classification) in enumerate(
            zip(code_lines, classifications), start=1
        )
        if classification == "production" and code.strip()
    }


def span_production_line_numbers(
    span: dict[str, Any], production_lines: set[int]
) -> set[int]:
    return {
        line_number
        for line_number in range(span["start_line"], span["end_line"] + 1)
        if line_number in production_lines
    }


def reviewed_scaffolding_metric(
    path: str,
    spans: list[dict[str, Any]],
    names: set[str],
) -> dict[str, Any]:
    functions = [
        {
            "path": path,
            "symbol": span["name"],
            "start_line": span["start_line"],
            "end_line": span["end_line"],
            "production_loc": span["logical_loc_by_classification"]["production"],
        }
        for span in spans
        if span["name"] in names
    ]
    return {
        "definition": "exhaustive-reviewed-execution-scaffolding",
        "function_count": len(functions),
        "file_count": len({row["path"] for row in functions}),
        "production_loc": sum(row["production_loc"] for row in functions),
        "functions": functions,
    }


def validate_provider_review(config: dict[str, Any]) -> dict[str, Any]:
    review = config["provider_review"]
    classifications = {
        "counted-execution-scaffolding",
        "counted-provider-glue",
        "excluded-parser",
        "excluded-weights",
        "excluded-math-program",
    }
    require(
        set(review)
        == {
            "path",
            "owner",
            "reviewed_at_git_sha",
            "classification_reasons",
            "forbidden_excluded_fragments",
            *classifications,
        },
        "provider review categories differ",
    )
    require(isinstance(review["owner"], str) and review["owner"], "provider review owner is missing")
    reasons = review["classification_reasons"]
    require(
        isinstance(reasons, dict)
        and set(reasons) == classifications
        and all(isinstance(reason, str) and reason for reason in reasons.values()),
        "provider classification reasons are incomplete",
    )
    require(
        review["forbidden_excluded_fragments"]
        == FORBIDDEN_EXCLUDED_PROVIDER_FRAGMENTS,
        "provider excluded-function authority boundary differs",
    )
    validate_reviewed_source(
        review["path"],
        review["reviewed_at_git_sha"],
        "Qwen3.5 provider",
    )
    path = REPO_ROOT / review["path"]
    text = path.read_text(encoding="utf-8")
    spans = [
        span
        for span in inventory.rust_function_spans(text)
        if span["logical_loc_by_classification"].get("production", 0) > 0
    ]
    groups = {key: set(review[key]) for key in classifications}
    seen: set[str] = set()
    for classification, names in groups.items():
        require(all(isinstance(name, str) and name for name in names), f"{classification} contains an invalid symbol")
        overlap = seen & names
        require(not overlap, f"provider review duplicates symbols: {sorted(overlap)}")
        seen |= names
    actual = {span["name"] for span in spans}
    require(actual == seen, f"provider review is not exhaustive: missing={sorted(actual-seen)} extra={sorted(seen-actual)}")
    production_lines = production_line_numbers(text)
    excluded_categories = {
        "excluded-parser",
        "excluded-weights",
        "excluded-math-program",
    }
    excluded_lines: set[int] = set()
    excluded_authority_hits = []
    code_lines = inventory.code_lines_for("rust", text)
    for span in spans:
        if any(span["name"] in groups[category] for category in excluded_categories):
            excluded_lines |= span_production_line_numbers(span, production_lines)
            body = "\n".join(
                inventory.mask_strings(line)
                for line in code_lines[span["start_line"] - 1 : span["end_line"]]
            )
            for fragment in FORBIDDEN_EXCLUDED_PROVIDER_FRAGMENTS:
                if fragment in body:
                    excluded_authority_hits.append(
                        {"symbol": span["name"], "fragment": fragment}
                    )
    require(
        not excluded_authority_hits,
        "provider excluded functions contain runtime lifecycle authority: "
        f"{excluded_authority_hits}",
    )
    provider_lines = production_lines - excluded_lines
    scaffolding = reviewed_scaffolding_metric(
        review["path"], spans, groups["counted-execution-scaffolding"]
    )
    by_classification = {
        classification: {
            "function_count": sum(span["name"] in names for span in spans),
            "production_loc": sum(
                span["logical_loc_by_classification"]["production"]
                for span in spans
                if span["name"] in names
            ),
        }
        for classification, names in groups.items()
    }
    return {
        "path": review["path"],
        "owner": review["owner"],
        "reviewed_at_git_sha": review["reviewed_at_git_sha"],
        "reviewed_function_name_count": len(actual),
        "reviewed_function_span_count": len(spans),
        "provider_file_count": 1,
        "full_file_production_loc": len(production_lines),
        "excluded_function_production_loc": len(excluded_lines),
        "excluded_runtime_authority_hit_count": 0,
        "provider_glue_production_loc": len(provider_lines),
        "top_level_or_type_provider_glue_production_loc": (
            len(provider_lines)
            - by_classification["counted-provider-glue"]["production_loc"]
            - scaffolding["production_loc"]
        ),
        "execution_scaffolding": scaffolding,
        "by_classification": by_classification,
    }


def validate_family_inventory(candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    files = {row["path"]: row for row in candidate["files"]}
    family_declared = {row["path"] for row in config["family_sources"]}
    boundary_declared = {row["path"] for row in config["module_boundary_sources"]}
    declared = family_declared | boundary_declared
    discovered = {
        row["path"]
        for row in candidate["coupling"]["findings"]
        if row.get("category") == "qwen35_symbol"
        and row.get("line_classification") == "production"
        and str(row.get("path", "")).startswith("crates/ferrum-models/src/")
    }
    require(
        discovered == declared,
        "Qwen3.5 reviewed source closure differs: "
        f"unreviewed={sorted(discovered-declared)} stale={sorted(declared-discovered)}",
    )
    boundary_rows = []
    for declaration in config["module_boundary_sources"]:
        require(
            set(declaration)
            == {"path", "owner", "reviewed_at_git_sha", "reason"}
            and isinstance(declaration["owner"], str)
            and declaration["owner"]
            and isinstance(declaration["reason"], str)
            and declaration["reason"],
            "module boundary review row is incomplete",
        )
        validate_reviewed_source(
            declaration["path"],
            declaration["reviewed_at_git_sha"],
            f"module boundary {declaration['path']}",
        )
        entry = files[declaration["path"]]
        boundary_rows.append(
            {
                **declaration,
                "sha256": entry["sha256"],
                "production_loc": entry["logical_loc_by_classification"].get(
                    "production", 0
                ),
            }
        )
    rows = []
    forbidden_terms = config["provider_trait"]["forbidden_lifecycle_terms"]
    forbidden_authority = config["lifecycle_ownership"][
        "forbidden_family_authority_fragments"
    ]
    lifecycle_function_hits = []
    lifecycle_authority_hits = []
    for declaration in config["family_sources"]:
        require(
            set(declaration)
            == {"path", "classification", "owner", "reviewed_at_git_sha", "reason"}
            and isinstance(declaration["owner"], str)
            and declaration["owner"]
            and isinstance(declaration["reason"], str)
            and declaration["reason"],
            "family source review row is incomplete",
        )
        path = declaration["path"]
        validate_reviewed_source(
            path,
            declaration["reviewed_at_git_sha"],
            f"family source {path}",
        )
        entry = files[path]
        production_loc = entry["logical_loc_by_classification"].get("production", 0)
        require(production_loc > 0, f"family source has no production LOC: {path}")
        rows.append(
            {
                "path": path,
                "classification": declaration["classification"],
                "owner": declaration["owner"],
                "reviewed_at_git_sha": declaration["reviewed_at_git_sha"],
                "reason": declaration["reason"],
                "sha256": entry["sha256"],
                "production_loc": production_loc,
            }
        )
        if path.endswith(".rs"):
            text = (REPO_ROOT / path).read_text(encoding="utf-8")
            _, _, code_lines, line_classes = inventory.logical_loc(
                "rust", text, "production"
            )
            for line_number, (code, classification) in enumerate(
                zip(code_lines, line_classes), start=1
            ):
                if classification != "production":
                    continue
                for fragment in forbidden_authority:
                    if fragment in code:
                        lifecycle_authority_hits.append(
                            {
                                "path": path,
                                "line": line_number,
                                "fragment": fragment,
                            }
                        )
            for span in inventory.rust_function_spans(
                text
            ):
                if span["logical_loc_by_classification"].get("production", 0) == 0:
                    continue
                terms = [term for term in forbidden_terms if term in span["name"].lower()]
                if terms:
                    lifecycle_function_hits.append(
                        {
                            "path": path,
                            "symbol": span["name"],
                            "terms": terms,
                            "start_line": span["start_line"],
                        }
                    )
    require(
        not lifecycle_function_hits,
        f"Qwen3.5 family regained product lifecycle functions: {lifecycle_function_hits}",
    )
    require(
        not lifecycle_authority_hits,
        "Qwen3.5 family regained shared runtime authority types: "
        f"{lifecycle_authority_hits}",
    )
    novel = set(config["novel_operation_sources"])
    require(all(path in files for path in novel), "reviewed novel operation source is missing")
    return {
        "family_file_count": len(rows),
        "provider_file_count": sum(
            row["classification"] == "reviewed-provider" for row in rows
        ),
        "family_production_loc": sum(row["production_loc"] for row in rows),
        "maximum_family_file_production_loc": max(row["production_loc"] for row in rows),
        "forbidden_lifecycle_function_count": 0,
        "forbidden_lifecycle_authority_count": 0,
        "reviewed_source_closure": sorted(discovered),
        "module_boundaries": boundary_rows,
        "files": rows,
        "novel_operation_sources": [
            {
                "path": path,
                "sha256": files[path]["sha256"],
                "production_loc": files[path]["logical_loc_by_classification"].get("production", 0),
            }
            for path in sorted(novel)
        ],
    }


def block_method_names(text: str, header: re.Pattern[str], label: str) -> list[str]:
    lines = inventory.code_lines_for("rust", text)
    depth = 0
    pending_base: int | None = None
    active_base: int | None = None
    methods: list[str] = []
    found_header = False
    for line in lines:
        masked = inventory.mask_strings(line)
        depth_before = depth
        if active_base is None and pending_base is None and header.search(masked):
            pending_base = depth_before
            found_header = True
        if pending_base is not None and "{" in masked:
            active_base = pending_base
            pending_base = None
        if active_base is not None and depth_before == active_base + 1:
            match = inventory.FUNCTION_RE.search(masked)
            if match is not None:
                methods.append(match.group(1))
        depth += masked.count("{") - masked.count("}")
        if active_base is not None and depth <= active_base:
            active_base = None
            break
    require(found_header and active_base is None, f"{label} block was not resolved")
    return methods


def validate_provider_boundary(config: dict[str, Any]) -> dict[str, Any]:
    contract = config["provider_trait"]
    require(
        set(contract)
        == {
            "path",
            "name",
            "owner",
            "reviewed_at_git_sha",
            "required_methods",
            "forbidden_lifecycle_terms",
        },
        "provider trait contract fields differ",
    )
    validate_reviewed_source(
        contract["path"],
        contract["reviewed_at_git_sha"],
        "ModelFamilyProvider trait",
    )
    trait_path = REPO_ROOT / contract["path"]
    trait_methods = block_method_names(
        trait_path.read_text(encoding="utf-8"),
        re.compile(r"\btrait\s+ModelFamilyProvider\b"),
        "ModelFamilyProvider trait",
    )
    provider_path = REPO_ROOT / config["provider_review"]["path"]
    implementation_methods = block_method_names(
        provider_path.read_text(encoding="utf-8"),
        re.compile(r"\bimpl\s+ModelFamilyProvider\s+for\s+Qwen35FamilyProvider\b"),
        "Qwen35 ModelFamilyProvider implementation",
    )
    required = contract["required_methods"]
    require(trait_methods == required, f"ModelFamilyProvider method boundary differs: {trait_methods}")
    require(implementation_methods == required, f"Qwen35 provider method boundary differs: {implementation_methods}")
    forbidden = contract["forbidden_lifecycle_terms"]
    violations = [
        method
        for method in trait_methods + implementation_methods
        if any(term in method.lower() for term in forbidden)
    ]
    require(not violations, f"model provider regained lifecycle hooks: {violations}")
    return {
        "trait_path": contract["path"],
        "trait_owner": contract["owner"],
        "trait_reviewed_at_git_sha": contract["reviewed_at_git_sha"],
        "trait_methods": trait_methods,
        "implementation_path": config["provider_review"]["path"],
        "implementation_methods": implementation_methods,
        "forbidden_lifecycle_hook_count": 0,
    }


def validate_reviewed_function(
    contract: dict[str, Any], label: str
) -> dict[str, Any]:
    required = {
        "path",
        "symbol",
        "line_hint",
        "owner",
        "reviewed_at_git_sha",
        "required_fragments",
    }
    require(required <= set(contract), f"{label} function contract fields differ")
    require(
        isinstance(contract["line_hint"], int)
        and not isinstance(contract["line_hint"], bool)
        and contract["line_hint"] > 0,
        f"{label} line_hint is invalid",
    )
    require(
        isinstance(contract["owner"], str) and contract["owner"],
        f"{label} owner is missing",
    )
    fragments = contract["required_fragments"]
    require(
        isinstance(fragments, list)
        and fragments
        and all(isinstance(fragment, str) and fragment for fragment in fragments),
        f"{label} required fragments are invalid",
    )
    validate_reviewed_source(
        contract["path"], contract["reviewed_at_git_sha"], label
    )
    text = (REPO_ROOT / contract["path"]).read_text(encoding="utf-8")
    span = span_for_review(
        inventory.rust_function_spans(text),
        contract["symbol"],
        [contract["line_hint"]],
        label,
    )
    lines = inventory.code_lines_for("rust", text)
    body = "\n".join(
        inventory.mask_strings(line)
        for line in lines[span["start_line"] - 1 : span["end_line"]]
    )
    missing = [fragment for fragment in fragments if fragment not in body]
    require(not missing, f"{label} is missing required call/owner fragments: {missing}")
    return {
        "path": contract["path"],
        "symbol": contract["symbol"],
        "line_hint": contract["line_hint"],
        "start_line": span["start_line"],
        "end_line": span["end_line"],
        "owner": contract["owner"],
        "reviewed_at_git_sha": contract["reviewed_at_git_sha"],
        "required_fragments": fragments,
        "production_loc": span["logical_loc_by_classification"].get(
            "production", 0
        ),
    }


def validate_product_route(config: dict[str, Any]) -> dict[str, Any]:
    contract = config["product_route"]
    require(
        set(contract)
        == {
            "family_registration",
            "registration_resolution",
            "factory_selection",
            "shared_composition",
            "product_selection",
            "shared_executor_call_count",
            "external_metadata_ids",
            "gguf_architectures",
        },
        "Qwen3.5 product route fields differ",
    )
    functions = [
        validate_reviewed_function(contract[key], f"product route {key}")
        for key in (
            "family_registration",
            "registration_resolution",
            "factory_selection",
            "shared_composition",
            "product_selection",
        )
    ]
    registry_path = REPO_ROOT / contract["family_registration"]["path"]
    registry = registry_path.read_text(encoding="utf-8")
    loaders_start = registry.find("const MODEL_LOADERS")
    legacy_start = registry.find("const LEGACY_MODELS")
    legacy_end = registry.find("pub enum ProductionExecutionKind", legacy_start)
    require(
        0 <= loaders_start < legacy_start < legacy_end,
        "product model-loader and legacy registry blocks were not resolved",
    )
    loaders = registry[loaders_start:legacy_start]
    legacy = registry[legacy_start:legacy_end]
    metadata_ids = contract["external_metadata_ids"]
    architectures = contract["gguf_architectures"]
    require(
        len(metadata_ids) == 2
        and len(set(metadata_ids)) == 2
        and all(loaders.count(identifier) == 1 for identifier in metadata_ids)
        and all(identifier not in legacy for identifier in metadata_ids),
        "Qwen3.5 metadata ids are not uniquely registered on the vNext route",
    )
    require(
        len(architectures) == 2
        and len(set(architectures)) == 2
        and all(loaders.count(f'"{architecture}"') == 1 for architecture in architectures)
        and all(f'"{architecture}"' not in legacy for architecture in architectures),
        "Qwen3.5 GGUF architectures are not uniquely registered on the vNext route",
    )
    selection = next(
        row for row in functions if row["symbol"] == "create_registered_vnext_executor"
    )
    selection_source = inventory.code_lines_for(
        "rust", (REPO_ROOT / selection["path"]).read_text(encoding="utf-8")
    )
    selection_body = "\n".join(
        inventory.mask_strings(line)
        for line in selection_source[
            selection["start_line"] - 1 : selection["end_line"]
        ]
    )
    executor_call_count = selection_body.count(
        "product_composition::create_vnext_executor("
    )
    require(
        contract["shared_executor_call_count"] == 2
        and executor_call_count == contract["shared_executor_call_count"],
        "CUDA and Metal do not converge on one shared executor composition",
    )
    return {
        "route_function_count": len(functions),
        "route_functions": functions,
        "external_metadata_ids": metadata_ids,
        "gguf_architectures": architectures,
        "legacy_registry_match_count": 0,
        "shared_executor_call_count": executor_call_count,
        "backend_compositions": ["cuda", "metal"],
        "terminal_executor_type": "VNextModelExecutor<R>",
    }


def validate_lifecycle_ownership(config: dict[str, Any]) -> dict[str, Any]:
    contract = config["lifecycle_ownership"]
    expected_categories = config["lifecycle_proof"]["ownership_categories"]
    require(
        expected_categories == LIFECYCLE_CATEGORIES,
        "lifecycle proof categories differ from the fixed G08A contract",
    )
    require(
        contract["implementation_owner"] == LIFECYCLE_IMPLEMENTATION_OWNER,
        "lifecycle implementation owner differs from the shared runtime",
    )
    owners = contract["owner_functions"]
    require(
        isinstance(owners, list) and len(owners) == len(expected_categories),
        "lifecycle owner map must contain one exact function per category",
    )
    require(
        [row.get("category") for row in owners] == expected_categories,
        "lifecycle owner categories differ",
    )
    resolved = []
    for row in owners:
        require(
            set(row)
            == {
                "category",
                "path",
                "self_type",
                "symbol",
                "line_hint",
                "owner",
                "reviewed_at_git_sha",
                "required_fragments",
            }
            and isinstance(row["self_type"], str)
            and row["self_type"],
            f"lifecycle owner row is incomplete: {row.get('category')}",
        )
        function = validate_reviewed_function(
            row, f"lifecycle {row['category']} owner"
        )
        source = (REPO_ROOT / row["path"]).read_text(encoding="utf-8")
        impl_header = re.compile(
            r"\bimpl(?:\s*<[^>{}]*>)?\s+"
            + re.escape(row["self_type"])
            + r"(?=\s*(?:where\b|\{|$))"
        )
        impl_methods = block_method_names(
            source,
            impl_header,
            f"lifecycle {row['category']} implementation",
        )
        require(
            row["symbol"] in impl_methods,
            f"lifecycle {row['category']} function is not owned by {row['self_type']}",
        )
        span_text = "\n".join(
            source.splitlines()[function["start_line"] - 1 : function["end_line"]]
        )
        require(
            "qwen35" not in span_text.lower(),
            f"lifecycle {row['category']} owner is model-specific",
        )
        resolved.append(
            {
                **function,
                "category": row["category"],
                "self_type": row["self_type"],
            }
        )
    implementation_owners = {row["owner"] for row in resolved}
    require(
        implementation_owners == {LIFECYCLE_IMPLEMENTATION_OWNER},
        f"lifecycle categories have multiple implementation owners: {implementation_owners}",
    )
    return {
        "implementation_owner_count": len(implementation_owners),
        "implementation_owner": contract["implementation_owner"],
        "ownership_category_count": len(resolved),
        "ownership_categories": [row["category"] for row in resolved],
        "owner_functions": resolved,
    }


def production_fragment_hits(fragments: list[str]) -> list[dict[str, Any]]:
    hits = []
    for path in sorted((REPO_ROOT / "crates").rglob("*.rs")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        _, _, code_lines, line_classes = inventory.logical_loc("rust", text, inventory.classify_path(relative))
        original = text.splitlines()
        for line_no, (code, classification) in enumerate(zip(code_lines, line_classes), start=1):
            if classification != "production":
                continue
            for fragment in fragments:
                if fragment in code:
                    hits.append(
                        {
                            "path": relative,
                            "line": line_no,
                            "fragment": fragment,
                            "text": original[line_no - 1].strip()[:240],
                        }
                    )
    return hits


def validate_legacy_removal(config: dict[str, Any]) -> dict[str, Any]:
    contract = config["legacy_removal"]
    deletion_commit = contract["deletion_commit"]
    require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", deletion_commit, "HEAD"],
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0,
        "Qwen3.5 legacy deletion commit is not an ancestor of HEAD",
    )
    present = [path for path in contract["forbidden_paths"] if (REPO_ROOT / path).exists()]
    require(not present, f"Qwen3.5 legacy production paths returned: {present}")
    hits = production_fragment_hits(contract["forbidden_production_fragments"])
    require(not hits, f"Qwen3.5 legacy production fragments returned: {hits[:8]}")
    control_paths = sorted(
        {
            path.resolve()
            for pattern in contract["control_file_globs"]
            for path in REPO_ROOT.glob(pattern)
            if path.is_file() and not path.is_symlink()
        }
    )
    require(control_paths, "Qwen3.5 legacy control-file inventory is empty")
    control_hits = []
    for path in control_paths:
        text = path.read_text(encoding="utf-8")
        for fragment in contract["forbidden_control_fragments"]:
            if fragment in text:
                control_hits.append(
                    {
                        "path": path.relative_to(REPO_ROOT).as_posix(),
                        "fragment": fragment,
                    }
                )
    require(
        not control_hits,
        f"Qwen3.5 legacy Cargo/config controls returned: {control_hits}",
    )
    return {
        "deletion_commit": deletion_commit,
        "forbidden_path_count": len(contract["forbidden_paths"]),
        "present_forbidden_path_count": 0,
        "forbidden_production_fragment_count": len(contract["forbidden_production_fragments"]),
        "production_fragment_hit_count": 0,
        "control_file_count": len(control_paths),
        "forbidden_control_fragment_count": len(
            contract["forbidden_control_fragments"]
        ),
        "control_fragment_hit_count": 0,
        "source_product_legacy_selection_count": 0,
    }


def validate_lifecycle_payload(payload: Any, contract: dict[str, Any]) -> dict[str, Any]:
    analyses = payload
    require(isinstance(analyses, list) and len(analyses) == 3, "lifecycle proof must contain three profiles")
    require([row.get("profile") for row in analyses] == contract["profiles"], "lifecycle profiles differ")
    for row in analyses:
        snapshots = row.get("snapshots")
        require(isinstance(snapshots, list), "lifecycle snapshots must be an array")
        require([snapshot.get("stage") for snapshot in snapshots] == contract["stages"], f"{row.get('profile')} lifecycle stages differ")
        empty = snapshots[-1].get("occupancy")
        require(
            isinstance(empty, dict)
            and empty
            and all(isinstance(value, int) and not isinstance(value, bool) and value == 0 for value in empty.values()),
            f"{row.get('profile')} lifecycle did not clean up every scope",
        )
        require(
            isinstance(row.get("per_child_sequence_claims"), int)
            and row["per_child_sequence_claims"] > 0
            and isinstance(row.get("per_child_sequence_bytes"), int)
            and row["per_child_sequence_bytes"] > 0,
            f"{row.get('profile')} sequence ownership proof is incomplete",
        )
    expected_invocation = {"dense": (0, 0), "moe": (1, 48), "hybrid": (1, 48)}
    require(
        {
            row["profile"]: (row.get("invocation_peak_claims"), row.get("invocation_peak_bytes"))
            for row in analyses
        }
        == expected_invocation,
        "lifecycle invocation ownership differs",
    )
    return {
        "profile_count": 3,
        "profiles": analyses,
    }


def extract_lifecycle_payload(stdout: str) -> Any:
    proof_lines = [line for line in stdout.splitlines() if PROOF_PREFIX in line]
    require(len(proof_lines) == 1, "lifecycle test did not print exactly one proof line")
    encoded = proof_lines[0].split(PROOF_PREFIX, 1)[1]
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise GateError(f"lifecycle proof JSON is invalid: {error}") from error
    require(isinstance(payload, list), "lifecycle proof payload must be an array")
    return payload


def run_lifecycle_test(out: Path, config: dict[str, Any]) -> dict[str, Any]:
    contract = config["lifecycle_proof"]
    receipt = out / "lifecycle/receipt.json"
    stdout_log = out / "lifecycle/stdout.log"
    stderr_log = out / "lifecycle/stderr.log"
    test_command = [
        "cargo",
        "test",
        "-p",
        "ferrum-interfaces",
        "--test",
        contract["test_target"],
        contract["test_name"],
        "--",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]
    command = [
        sys.executable,
        str(BOUNDED_COMMAND),
        "--receipt",
        str(receipt),
        "--stdout-log",
        str(stdout_log),
        "--stderr-log",
        str(stderr_log),
        "--cwd",
        str(REPO_ROOT),
        "--wall-timeout-seconds",
        "600",
        "--max-processes",
        "16",
        "--max-group-threads",
        "64",
        "--max-per-process-threads",
        "16",
        "--sample-interval-seconds",
        "0.25",
        "--",
        *test_command,
    ]
    write_json(
        out / "lifecycle/command.json",
        {
            "command": command,
            "expected_duration_seconds": 120,
            "hard_deadline_seconds": 600,
            "progress_signal": str(stdout_log),
            "environment": {"CARGO_BUILD_JOBS": "4", "RUST_TEST_THREADS": "1"},
        },
    )
    environment = os.environ.copy()
    environment.update({"CARGO_BUILD_JOBS": "4", "RUST_TEST_THREADS": "1"})
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    require(process.returncode == 0, process.stderr.strip() or process.stdout.strip() or "bounded lifecycle test failed")
    bounded = read_json(receipt, "bounded lifecycle receipt")
    stdout_ref = file_ref(stdout_log)
    stderr_ref = file_ref(stderr_log)
    require(
        bounded.get("schema") == "ferrum.bounded-command-receipt.v1"
        and bounded.get("status") == "pass"
        and bounded.get("rc") == 0
        and bounded.get("command") == test_command
        and bounded.get("cwd") == str(REPO_ROOT)
        and bounded.get("cleanup", {}).get("process_group_gone") is True
        and bounded.get("limits", {}).get("wall_timeout_seconds") == 600.0
        and bounded.get("limits", {}).get("max_processes") == 16
        and bounded.get("limits", {}).get("max_group_threads") == 64
        and bounded.get("limits", {}).get("max_per_process_threads") == 16
        and bounded.get("stdout") == stdout_ref
        and bounded.get("stderr") == stderr_ref,
        "bounded lifecycle receipt is not a clean PASS",
    )
    stdout = stdout_log.read_text(encoding="utf-8")
    payload = extract_lifecycle_payload(stdout)
    proof = validate_lifecycle_payload(payload, contract)
    require(
        re.search(r"test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out;", stdout) is not None,
        "focused lifecycle libtest summary differs",
    )
    return {
        **proof,
        "test_path": contract["test_path"],
        "test_name": contract["test_name"],
        "bounded_receipt": file_ref(receipt),
        "stdout": stdout_ref,
        "stderr": stderr_ref,
        "command": file_ref(out / "lifecycle/command.json"),
    }


def artifact_index(out: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(out.rglob("*")):
        require(not path.is_symlink(), f"artifact tree contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(out).as_posix()
        if relative in ARTIFACT_INDEX_EXCLUDED:
            continue
        rows.append(
            {
                "path": relative,
                "role": relative.split("/", 1)[0]
                if "/" in relative
                else relative.rsplit(".", 1)[0],
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def validate_file_ref(
    raw: Any,
    label: str,
    *,
    artifact_root: Path | None = None,
    indexed: dict[str, dict[str, Any]] | None = None,
) -> Path:
    require(isinstance(raw, dict), f"{label} must be a file reference")
    require(
        set(raw) == {"path", "sha256", "size_bytes"},
        f"{label} file-reference fields differ",
    )
    path = Path(str(raw["path"])).resolve()
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    require(
        SHA256_RE.fullmatch(str(raw["sha256"])) is not None
        and sha256_file(path) == raw["sha256"],
        f"{label} SHA256 differs",
    )
    require(
        isinstance(raw["size_bytes"], int)
        and not isinstance(raw["size_bytes"], bool)
        and raw["size_bytes"] > 0
        and path.stat().st_size == raw["size_bytes"],
        f"{label} size differs",
    )
    if artifact_root is not None:
        require(indexed is not None, f"{label} indexed validation is unavailable")
        try:
            relative = path.relative_to(artifact_root).as_posix()
        except ValueError as error:
            raise GateError(f"{label} escapes the artifact root") from error
        require(relative in indexed, f"{label} is absent from artifact_index")
        require(
            indexed[relative]["sha256"] == raw["sha256"]
            and indexed[relative]["size_bytes"] == raw["size_bytes"],
            f"{label} differs from artifact_index",
        )
    return path


def verify_manifest(manifest_path: Path, *, verify_checkout: bool) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path, "G08A source manifest")
    expected_fields = {
        "schema_version",
        "artifact_type",
        "lane",
        "status",
        "canonical",
        "source_git_sha",
        "source_tree_sha",
        "dirty",
        "artifact_dir",
        "inputs",
        "validation",
        "summary",
        "does_not_prove",
        "artifact_index",
        "artifact_count",
        "pass_line",
    }
    require(set(manifest) == expected_fields, "G08A source manifest fields differ")
    root = manifest_path.parent.resolve()
    require(
        manifest["schema_version"] == 1
        and manifest["artifact_type"]
        == "runtime_vnext_g08a_source_ownership_manifest"
        and manifest["lane"] == "runtime-vnext-g08a-source-ownership"
        and manifest["status"] == "pass"
        and manifest["canonical"] is True
        and manifest["dirty"] is False
        and Path(manifest["artifact_dir"]).resolve() == root,
        "G08A source manifest identity differs",
    )
    require(
        manifest["pass_line"] == f"{PASS_PREFIX}: {root}",
        "G08A source manifest pass line differs",
    )
    require(
        GIT_SHA_RE.fullmatch(str(manifest["source_git_sha"])) is not None
        and GIT_SHA_RE.fullmatch(str(manifest["source_tree_sha"])) is not None,
        "G08A source manifest source identity is invalid",
    )
    rows = manifest["artifact_index"]
    require(isinstance(rows, list) and rows, "G08A artifact_index is empty")
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        require(
            isinstance(row, dict)
            and set(row) == {"path", "role", "sha256", "size_bytes"},
            f"artifact_index[{index}] fields differ",
        )
        relative = Path(str(row["path"]))
        require(
            not relative.is_absolute()
            and relative.as_posix() == str(row["path"])
            and ".." not in relative.parts
            and row["path"] not in ARTIFACT_INDEX_EXCLUDED
            and row["path"] not in indexed,
            f"artifact_index[{index}] path is invalid",
        )
        require(
            isinstance(row["role"], str) and row["role"],
            f"artifact_index[{index}] role is invalid",
        )
        path = (root / relative).resolve()
        require(path.is_relative_to(root), f"artifact_index[{index}] escapes root")
        validate_file_ref(
            {
                "path": str(path),
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
            },
            f"artifact_index[{row['path']}]",
        )
        indexed[row["path"]] = row
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in ARTIFACT_INDEX_EXCLUDED
    }
    require(
        set(indexed) == actual,
        "G08A artifact_index coverage differs: "
        f"missing={sorted(actual-set(indexed))} extra={sorted(set(indexed)-actual)}",
    )
    require(
        manifest["artifact_count"] == len(indexed),
        "G08A artifact_count differs",
    )
    inputs = manifest["inputs"]
    require(
        isinstance(inputs, dict)
        and set(inputs)
        == {
            "contract",
            "inventory_analyzer",
            "baseline_inventory",
            "baseline_inventory_review",
            "candidate_inventory",
        },
        "G08A manifest inputs differ",
    )
    config_path = validate_file_ref(inputs["contract"], "contract")
    require(config_path == CONFIG_PATH.resolve(), "G08A contract path differs")
    analyzer_path = validate_file_ref(inputs["inventory_analyzer"], "inventory analyzer")
    require(
        analyzer_path == (SCRIPT_DIR / "runtime_vnext_inventory.py").resolve(),
        "G08A inventory analyzer path differs",
    )
    config = read_json(config_path, "G08A source contract")
    validate_config(config)
    baseline_path = validate_file_ref(inputs["baseline_inventory"], "baseline inventory")
    require(
        inputs["baseline_inventory"]["sha256"]
        == config["frozen_baseline"]["inventory_sha256"],
        "G08A baseline inventory binding differs",
    )
    review_path = validate_file_ref(
        inputs["baseline_inventory_review"], "baseline inventory review"
    )
    require(
        review_path == INVENTORY_REVIEW_PATH.resolve()
        and inputs["baseline_inventory_review"]["sha256"]
        == config["frozen_baseline"]["inventory_review_sha256"],
        "G08A baseline review binding differs",
    )
    candidate_path = validate_file_ref(
        inputs["candidate_inventory"],
        "candidate inventory",
        artifact_root=root,
        indexed=indexed,
    )
    candidate = read_json(candidate_path, "candidate inventory")
    require(
        candidate.get("git")
        == {
            "sha": manifest["source_git_sha"],
            "tree_sha": manifest["source_tree_sha"],
            "dirty": False,
            "status_short": [],
        },
        "candidate inventory source differs from manifest",
    )
    validation_path = validate_file_ref(
        manifest["validation"],
        "validation",
        artifact_root=root,
        indexed=indexed,
    )
    validation = read_json(validation_path, "G08A source validation")
    require(
        set(validation)
        == {
            "schema_version",
            "artifact_type",
            "status",
            "validated_at",
            "source",
            "baseline",
            "candidate_scaffolding",
            "scaffolding_reduction_ratio",
            "provider",
            "provider_boundary",
            "product_route",
            "family_surface_diagnostic",
            "legacy_removal",
            "lifecycle",
            "limits",
            "pass_line",
        },
        "G08A validation fields differ",
    )
    require(
        validation.get("schema_version") == 1
        and validation.get("artifact_type")
        == "runtime_vnext_g08a_source_ownership_validation"
        and validation.get("status") == "pass"
        and validation.get("pass_line") == manifest["pass_line"]
        and validation.get("source", {}).get("git_sha")
        == manifest["source_git_sha"]
        and validation.get("source", {}).get("git_tree_sha")
        == manifest["source_tree_sha"]
        and validation.get("source", {}).get("dirty") is False,
        "G08A validation identity differs from manifest",
    )
    summary = manifest["summary"]
    require(
        isinstance(summary, dict)
        and set(summary)
        == {
            "baseline_scaffolding_production_loc",
            "candidate_scaffolding_production_loc",
            "scaffolding_reduction_ratio",
            "provider_file_count",
            "provider_glue_production_loc",
            "full_family_file_count_diagnostic",
            "full_family_production_loc_diagnostic",
            "lifecycle_implementation_owner_count",
            "lifecycle_ownership_categories",
            "legacy_source_selection_count",
        }
        and summary["baseline_scaffolding_production_loc"]
        == validation["baseline"]["production_loc"]
        and summary["candidate_scaffolding_production_loc"]
        == validation["candidate_scaffolding"]["production_loc"]
        and summary["scaffolding_reduction_ratio"]
        == validation["scaffolding_reduction_ratio"]
        and summary["provider_file_count"]
        == validation["family_surface_diagnostic"]["provider_file_count"]
        and summary["provider_glue_production_loc"]
        == validation["provider"]["provider_glue_production_loc"]
        and summary["full_family_file_count_diagnostic"]
        == validation["family_surface_diagnostic"]["family_file_count"]
        and summary["full_family_production_loc_diagnostic"]
        == validation["family_surface_diagnostic"]["family_production_loc"]
        and summary["lifecycle_implementation_owner_count"]
        == validation["lifecycle"]["ownership"]["implementation_owner_count"]
        and summary["lifecycle_ownership_categories"]
        == validation["lifecycle"]["ownership"]["ownership_category_count"]
        and summary["legacy_source_selection_count"]
        == validation["legacy_removal"]["source_product_legacy_selection_count"],
        "G08A manifest summary differs from validation",
    )
    require(
        manifest["does_not_prove"] == DOES_NOT_PROVE,
        "G08A source manifest scope disclaimer differs",
    )
    behavior = validation["lifecycle"]["behavior"]
    for key in ("bounded_receipt", "stdout", "stderr", "command"):
        validate_file_ref(
            behavior[key],
            f"lifecycle {key}",
            artifact_root=root,
            indexed=indexed,
        )
    receipt = read_json(
        Path(behavior["bounded_receipt"]["path"]), "bounded lifecycle receipt"
    )
    lifecycle_contract = config["lifecycle_proof"]
    expected_test_command = [
        "cargo",
        "test",
        "-p",
        "ferrum-interfaces",
        "--test",
        lifecycle_contract["test_target"],
        lifecycle_contract["test_name"],
        "--",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]
    require(
        receipt.get("schema") == "ferrum.bounded-command-receipt.v1"
        and receipt.get("status") == "pass"
        and receipt.get("rc") == 0
        and receipt.get("command") == expected_test_command
        and receipt.get("cwd") == str(REPO_ROOT)
        and receipt.get("cleanup", {}).get("process_group_gone") is True
        and receipt.get("stdout") == behavior["stdout"]
        and receipt.get("stderr") == behavior["stderr"]
        and receipt.get("limits", {}).get("wall_timeout_seconds") == 600.0
        and receipt.get("limits", {}).get("max_processes") == 16
        and receipt.get("limits", {}).get("max_group_threads") == 64
        and receipt.get("limits", {}).get("max_per_process_threads") == 16,
        "bounded lifecycle receipt is not a clean PASS",
    )
    stdout = Path(behavior["stdout"]["path"]).read_text(encoding="utf-8")
    expected_behavior = validate_lifecycle_payload(
        extract_lifecycle_payload(stdout), lifecycle_contract
    )
    require(
        all(behavior.get(key) == value for key, value in expected_behavior.items())
        and behavior.get("test_path") == lifecycle_contract["test_path"]
        and behavior.get("test_name") == lifecycle_contract["test_name"]
        and re.search(
            r"test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out;",
            stdout,
        )
        is not None,
        "lifecycle behavior does not reproduce from bounded stdout",
    )
    command_doc = read_json(Path(behavior["command"]["path"]), "lifecycle command")
    require(
        command_doc.get("expected_duration_seconds") == 120
        and command_doc.get("hard_deadline_seconds") == 600
        and command_doc.get("progress_signal") == behavior["stdout"]["path"]
        and command_doc.get("environment")
        == {"CARGO_BUILD_JOBS": "4", "RUST_TEST_THREADS": "1"},
        "lifecycle command declaration differs",
    )
    if verify_checkout:
        require(not git_text("status", "--short"), "checkout became dirty after G08A collection")
        require(
            git_text("rev-parse", "HEAD") == manifest["source_git_sha"]
            and git_text("rev-parse", "HEAD^{tree}") == manifest["source_tree_sha"],
            "checkout source differs from G08A manifest",
        )
        baseline, baseline_review = validate_baseline_inputs(baseline_path, config)
        expected_family = validate_family_inventory(candidate, config)
        expected_provider = validate_provider_review(config)
        expected_baseline = baseline_scaffolding(
            baseline, baseline_review, config
        )
        expected_reduction = 1.0 - (
            expected_provider["execution_scaffolding"]["production_loc"]
            / expected_baseline["production_loc"]
        )
        require(
            validation["baseline"] == expected_baseline
            and validation["candidate_scaffolding"]
            == expected_provider["execution_scaffolding"]
            and validation["scaffolding_reduction_ratio"] == expected_reduction
            and validation["provider"] == expected_provider
            and validation["provider_boundary"] == validate_provider_boundary(config)
            and validation["product_route"] == validate_product_route(config)
            and validation["family_surface_diagnostic"] == expected_family
            and validation["legacy_removal"] == validate_legacy_removal(config)
            and validation["lifecycle"]["ownership"]
            == validate_lifecycle_ownership(config)
            and validation["limits"] == config["limits"],
            "G08A validation does not reproduce from the checked-out source",
        )
    return manifest


def collect(baseline_path: Path, out: Path) -> dict[str, Any]:
    source = clean_source()
    config = read_json(CONFIG_PATH, "G08A source contract")
    validate_config(config)
    baseline, review = validate_baseline_inputs(baseline_path, config)
    out = out.resolve()
    require(not out.is_relative_to(REPO_ROOT.resolve()), "G08A artifacts must be outside the source checkout")
    require(not out.exists(), f"G08A source output must be fresh: {out}")
    out.mkdir(parents=True)

    candidate_path = out / "candidate-inventory.json"
    candidate = inventory.build_inventory(REPO_ROOT, candidate_path, baseline_path)
    inventory.write_inventory(candidate_path, candidate)
    require(
        candidate["git"]
        == {
            "sha": source["git_sha"],
            "tree_sha": source["git_tree_sha"],
            "dirty": False,
            "status_short": [],
        },
        "candidate inventory source identity differs",
    )

    family = validate_family_inventory(candidate, config)
    provider = validate_provider_review(config)
    require(
        family["provider_file_count"] == provider["provider_file_count"],
        "reviewed provider file count differs from the family inventory",
    )
    provider_boundary = validate_provider_boundary(config)
    product_route = validate_product_route(config)
    lifecycle_ownership = validate_lifecycle_ownership(config)
    baseline_metric = baseline_scaffolding(baseline, review, config)
    candidate_metric = provider["execution_scaffolding"]
    require(baseline_metric["production_loc"] > 0, "frozen scaffolding LOC denominator is zero")
    reduction = 1.0 - candidate_metric["production_loc"] / baseline_metric["production_loc"]
    limits = config["limits"]
    require(reduction >= limits["minimum_scaffolding_reduction_ratio"], "Qwen3.5 scaffolding reduction is below 60%")
    require(family["provider_file_count"] <= limits["maximum_provider_file_count"], "Qwen3.5 provider file count exceeds eight")
    require(provider["provider_glue_production_loc"] <= limits["maximum_provider_glue_production_loc"], "Qwen3.5 provider glue exceeds 1500 production LOC")
    require(family["maximum_family_file_production_loc"] <= limits["maximum_family_file_production_loc"], "a Qwen3.5 family source file exceeds 5000 production LOC")
    legacy = validate_legacy_removal(config)
    lifecycle_behavior = run_lifecycle_test(out, config)

    pass_line = f"{PASS_PREFIX}: {out}"
    validation = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_g08a_source_ownership_validation",
        "status": "pass",
        "validated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "source": source,
        "baseline": baseline_metric,
        "candidate_scaffolding": candidate_metric,
        "scaffolding_reduction_ratio": reduction,
        "provider": provider,
        "provider_boundary": provider_boundary,
        "product_route": product_route,
        "family_surface_diagnostic": family,
        "legacy_removal": legacy,
        "lifecycle": {
            "ownership": lifecycle_ownership,
            "behavior": lifecycle_behavior,
        },
        "limits": limits,
        "pass_line": pass_line,
    }
    validation_path = out / "validation.json"
    write_json(validation_path, validation)
    indexed_artifacts = artifact_index(out)
    manifest = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_g08a_source_ownership_manifest",
        "lane": "runtime-vnext-g08a-source-ownership",
        "status": "pass",
        "canonical": True,
        "source_git_sha": source["git_sha"],
        "source_tree_sha": source["git_tree_sha"],
        "dirty": False,
        "artifact_dir": str(out),
        "inputs": {
            "contract": file_ref(CONFIG_PATH),
            "inventory_analyzer": file_ref(SCRIPT_DIR / "runtime_vnext_inventory.py"),
            "baseline_inventory": file_ref(baseline_path),
            "baseline_inventory_review": file_ref(INVENTORY_REVIEW_PATH),
            "candidate_inventory": file_ref(candidate_path),
        },
        "validation": file_ref(validation_path),
        "summary": {
            "baseline_scaffolding_production_loc": baseline_metric["production_loc"],
            "candidate_scaffolding_production_loc": candidate_metric["production_loc"],
            "scaffolding_reduction_ratio": reduction,
            "provider_file_count": family["provider_file_count"],
            "provider_glue_production_loc": provider["provider_glue_production_loc"],
            "full_family_file_count_diagnostic": family["family_file_count"],
            "full_family_production_loc_diagnostic": family["family_production_loc"],
            "lifecycle_implementation_owner_count": lifecycle_ownership[
                "implementation_owner_count"
            ],
            "lifecycle_ownership_categories": lifecycle_ownership[
                "ownership_category_count"
            ],
            "legacy_source_selection_count": legacy["source_product_legacy_selection_count"],
        },
        "does_not_prove": DOES_NOT_PROVE,
        "artifact_index": indexed_artifacts,
        "artifact_count": len(indexed_artifacts),
        "pass_line": pass_line,
    }
    manifest_path = out / "manifest.json"
    write_json(manifest_path, manifest)
    verify_manifest(manifest_path, verify_checkout=True)
    print(pass_line)
    return manifest


def lifecycle_fixture(config: dict[str, Any]) -> list[dict[str, Any]]:
    analyses = []
    for profile in config["lifecycle_proof"]["profiles"]:
        snapshots = []
        for stage in config["lifecycle_proof"]["stages"]:
            occupancy = {"total": 0, "sequence": 0, "invocation": 0}
            if stage != "empty":
                occupancy["total"] = 1
            snapshots.append({"stage": stage, "occupancy": occupancy})
        claims, size = ((0, 0) if profile == "dense" else (1, 48))
        analyses.append(
            {
                "profile": profile,
                "snapshots": snapshots,
                "per_child_sequence_claims": 1,
                "per_child_sequence_bytes": 4,
                "invocation_peak_claims": claims,
                "invocation_peak_bytes": size,
            }
        )
    return analyses


def self_test() -> None:
    config = read_json(CONFIG_PATH, "G08A source contract")
    validate_config(config)
    provider = validate_provider_review(config)
    require(provider["provider_glue_production_loc"] > 0, "provider glue self-test denominator is zero")
    require(
        provider["provider_glue_production_loc"]
        == provider["full_file_production_loc"]
        - provider["excluded_function_production_loc"]
        and provider["top_level_or_type_provider_glue_production_loc"] > 0,
        "provider whole-file subtraction metric differs",
    )
    require(
        provider["execution_scaffolding"]["production_loc"] == 0,
        "reviewed Qwen3.5 provider unexpectedly owns execution scaffolding",
    )
    validate_provider_boundary(config)
    validate_product_route(config)
    validate_lifecycle_ownership(config)
    validate_legacy_removal(config)
    payload = lifecycle_fixture(config)
    validate_lifecycle_payload(payload, config["lifecycle_proof"])
    require(
        extract_lifecycle_payload(
            "test focused ... " + PROOF_PREFIX + json.dumps(payload, separators=(",", ":"))
        )
        == payload,
        "interleaved libtest proof parsing differs",
    )

    mutations = []
    missing_review = copy.deepcopy(config)
    missing_review["provider_review"]["counted-provider-glue"].remove("new")
    mutations.append(lambda: validate_provider_review(missing_review))
    bad_owner = copy.deepcopy(config)
    bad_owner["lifecycle_ownership"]["owner_functions"][-1]["owner"] = (
        "qwen35.provider-cleanup"
    )
    mutations.append(lambda: validate_lifecycle_ownership(bad_owner))
    missing_category = copy.deepcopy(config)
    missing_category["lifecycle_ownership"]["owner_functions"].pop()
    mutations.append(lambda: validate_lifecycle_ownership(missing_category))
    wrong_self_type = copy.deepcopy(config)
    wrong_self_type["lifecycle_ownership"]["owner_functions"][0]["self_type"] = (
        "Qwen35FamilyProvider"
    )
    mutations.append(lambda: validate_lifecycle_ownership(wrong_self_type))
    bad_route = copy.deepcopy(config)
    bad_route["product_route"]["shared_composition"]["required_fragments"].append(
        "Qwen35LegacyExecutor"
    )
    mutations.append(lambda: validate_product_route(bad_route))
    missing_backend_route = copy.deepcopy(config)
    missing_backend_route["product_route"]["shared_executor_call_count"] = 1
    mutations.append(lambda: validate_product_route(missing_backend_route))
    bad_stage = copy.deepcopy(payload)
    bad_stage[0]["snapshots"].pop()
    mutations.append(lambda: validate_lifecycle_payload(bad_stage, config["lifecycle_proof"]))
    for mutation in mutations:
        try:
            mutation()
        except GateError:
            pass
        else:
            raise GateError("G08A source-contract mutation unexpectedly passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-inventory", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.baseline_inventory is not None or args.out is not None:
            parser.error("--self-test cannot be combined with collection arguments")
    elif args.baseline_inventory is None or args.out is None:
        parser.error("--baseline-inventory and --out are required")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            self_test()
            print(SELFTEST_PASS_LINE)
            return 0
        collect(args.baseline_inventory.resolve(), args.out)
        return 0
    except (GateError, inventory.InventoryError, OSError) as error:
        target = args.out if args.out is not None else "self-test"
        print(f"{FAIL_PREFIX}: {target}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
