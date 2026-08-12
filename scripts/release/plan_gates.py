#!/usr/bin/env python3
"""Plan release gates from changed files and checked-in impact rules."""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import fnmatch
import io
import json
import os
import re
import subprocess
import sys
import time
import tokenize
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GOAL = "release-regression-hardening-2026-06-28"
DEFAULT_RULES = REPO_ROOT / "scripts/release/change_impact_rules.json"
DEFAULT_FIXTURES = REPO_ROOT / "scripts/release/fixtures/change_impact/planner_fixtures.json"
PRODUCT_SCENARIOS = REPO_ROOT / "scripts/release/scenarios/product_regression.json"
PASS_LINE = "CHANGE IMPACT GATE PLAN PASS"
SELFTEST_PASS_LINE = "CHANGE IMPACT GATE PLAN SELFTEST PASS"
FINAL_STAGE_GATES = {
    "actual_model_regression",
    "model_contract",
    "native_operator",
    "observability_profile",
    "product_sentinel",
    "resource_invariant",
    "support_matrix_contract",
}
SECRET_ENV_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "AUTH", "CREDENTIAL", "KEY")
SAFE_ENV_NAMES = {"CI", "CARGO_HOME", "HF_HOME", "HOME", "PATH", "RUSTFLAGS", "RUST_BACKTRACE", "RUST_LOG"}
SAFE_ENV_PREFIXES = ("CARGO_", "FERRUM_", "HF_", "RUST_")


class PlannerError(RuntimeError):
    pass


def run_git(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise PlannerError(f"git {' '.join(args)} failed rc={proc.returncode}\n{proc.stderr}")
    return proc.stdout


def git_value(args: list[str], default: str = "unknown") -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return default
    return proc.stdout.strip() or default


def resolve_git_rev(value: str) -> str:
    if not value:
        return value
    resolved = git_value(["rev-parse", value], default="")
    return resolved or value


def git_changed_files(base: str, head: str) -> list[str]:
    out = run_git(["diff", "--name-only", f"{base}..{head}"])
    return sorted(line.strip() for line in out.splitlines() if line.strip())


def git_dirty() -> bool:
    return bool(run_git(["status", "--short"]).strip())


def git_dirty_files() -> list[str]:
    return [line for line in run_git(["status", "--short"]).splitlines() if line.strip()]


def sanitized_env() -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in os.environ.items():
        if any(marker in key.upper() for marker in SECRET_ENV_MARKERS):
            continue
        if key in SAFE_ENV_NAMES or any(key.startswith(prefix) for prefix in SAFE_ENV_PREFIXES):
            safe[key] = value
    return dict(sorted(safe.items()))


def normalize_changed_files(files: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    repo_root = REPO_ROOT.resolve()
    for item in files:
        raw = item.strip()
        if not raw:
            continue
        path = PurePosixPath(raw)
        if any(part == ".." for part in path.parts):
            raise PlannerError(f"changed file path escapes repository: {raw}")
        if path.is_absolute():
            try:
                rel = Path(path.as_posix()).resolve().relative_to(repo_root).as_posix()
            except ValueError as error:
                raise PlannerError(f"changed file path is outside repository: {raw}") from error
        else:
            rel = path.as_posix()
        if rel != "." and rel not in seen:
            seen.add(rel)
            out.append(rel)
    return sorted(out)


def load_rule_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise PlannerError(f"{path}: schema_version must be 1")
    rules = data.get("rules")
    if not isinstance(rules, list) or not rules:
        raise PlannerError(f"{path}: rules must be a non-empty list")
    required = {
        "id",
        "path_globs",
        "domains",
        "required_gates",
        "release_invalidation",
        "exceptions",
        "owner",
        "reason",
    }
    scenario_manifest = json.loads(PRODUCT_SCENARIOS.read_text(encoding="utf-8"))
    available_scenarios = {
        str(scenario.get("name"))
        for scenario in scenario_manifest.get("scenarios", [])
        if isinstance(scenario, dict) and scenario.get("name")
    }
    if not available_scenarios:
        raise PlannerError(f"{PRODUCT_SCENARIOS}: scenarios must be a non-empty list")
    profiles = data.get("qualification_profiles", {})
    if not isinstance(profiles, dict):
        raise PlannerError(f"{path}: qualification_profiles must be an object")
    for profile_id, profile in profiles.items():
        if not isinstance(profile_id, str) or not profile_id:
            raise PlannerError(f"{path}: qualification profile ids must be non-empty strings")
        if not isinstance(profile, dict):
            raise PlannerError(f"{path}: qualification_profiles.{profile_id} must be an object")
        for key in ("domains", "qualified_scopes", "required_checks", "required_gates"):
            values = profile.get(key)
            if not isinstance(values, list) or (key != "required_gates" and not values):
                raise PlannerError(
                    f"{path}: qualification_profiles.{profile_id}.{key} must be a list"
                    + ("" if key == "required_gates" else " with at least one item")
                )
        if not all(isinstance(item, str) and item for item in profile["domains"]):
            raise PlannerError(
                f"{path}: qualification_profiles.{profile_id}.domains must contain non-empty strings"
            )
        if not all(isinstance(item, str) and item for item in profile["required_checks"]):
            raise PlannerError(
                f"{path}: qualification_profiles.{profile_id}.required_checks must contain non-empty strings"
            )
        if not all(isinstance(item, str) and item for item in profile["required_gates"]):
            raise PlannerError(
                f"{path}: qualification_profiles.{profile_id}.required_gates must contain non-empty strings"
            )
        for scope_index, scope in enumerate(profile["qualified_scopes"]):
            if not isinstance(scope, dict):
                raise PlannerError(
                    f"{path}: qualification_profiles.{profile_id}.qualified_scopes[{scope_index}] "
                    "must be an object"
                )
            for key in ("backend", "entrypoint", "profile_detail"):
                if not isinstance(scope.get(key), str) or not scope[key]:
                    raise PlannerError(
                        f"{path}: qualification_profiles.{profile_id}.qualified_scopes"
                        f"[{scope_index}].{key} must be a non-empty string"
                    )
        selector = profile.get("selector")
        selector_kind = selector.get("kind") if isinstance(selector, dict) else None
        if selector_kind not in {"structured_rewrites", "symbol_contracts"}:
            raise PlannerError(
                f"{path}: qualification_profiles.{profile_id}.selector.kind must be "
                "structured_rewrites or symbol_contracts"
            )
        if selector_kind == "structured_rewrites" and not isinstance(
            selector.get("allow_test_changes", False), bool
        ):
            raise PlannerError(
                f"{path}: qualification_profiles.{profile_id}.selector.allow_test_changes "
                "must be boolean"
            )
        selector_files = selector.get("files")
        if not isinstance(selector_files, list) or not selector_files:
            raise PlannerError(
                f"{path}: qualification_profiles.{profile_id}.selector.files must be a non-empty list"
            )
        for file_index, file_selector in enumerate(selector_files):
            if not isinstance(file_selector, dict):
                raise PlannerError(
                    f"{path}: qualification_profiles.{profile_id}.selector.files[{file_index}] "
                    "must be an object"
                )
            selected_path = file_selector.get("path")
            if not isinstance(selected_path, str) or not selected_path:
                raise PlannerError(
                    f"{path}: qualification_profiles.{profile_id}.selector.files"
                    f"[{file_index}].path must be a non-empty string"
                )
            if selector_kind == "symbol_contracts":
                language = file_selector.get("language")
                if language not in {"json", "python", "rust"}:
                    raise PlannerError(
                        f"{path}: qualification_profiles.{profile_id}.selector.files"
                        f"[{file_index}].language must be json, python, or rust"
                    )
                for key in ("allow_test_changes", "test_only"):
                    if not isinstance(file_selector.get(key, False), bool):
                        raise PlannerError(
                            f"{path}: qualification_profiles.{profile_id}.selector.files"
                            f"[{file_index}].{key} must be boolean"
                        )
                contracts = file_selector.get("contracts", [])
                test_only = bool(file_selector.get("test_only", False))
                if not isinstance(contracts, list) or (not test_only and not contracts):
                    raise PlannerError(
                        f"{path}: qualification_profiles.{profile_id}.selector.files"
                        f"[{file_index}].contracts must be a non-empty list unless test_only"
                    )
                contract_ids: set[str] = set()
                rust_kinds = {
                    "rust_enum_variant",
                    "rust_const",
                    "rust_function",
                    "rust_impl_method",
                    "rust_struct_field",
                    "rust_type",
                    "rust_use",
                }
                python_kinds = {"python_import", "python_symbol"}
                json_kinds = {"json_value"}
                semantic_policy = file_selector.get("semantic_policy")
                behavioral_kinds = {
                    "python_symbol",
                    "rust_const",
                    "rust_enum_variant",
                    "rust_function",
                    "rust_impl_method",
                    "rust_struct_field",
                    "rust_type",
                }
                if not test_only and any(
                    isinstance(contract, dict) and contract.get("kind") in behavioral_kinds
                    for contract in contracts
                ):
                    if not isinstance(semantic_policy, dict) or semantic_policy.get("kind") != "timing_observability":
                        raise PlannerError(
                            f"{path}: qualification_profiles.{profile_id}.selector.files"
                            f"[{file_index}].semantic_policy.kind must be timing_observability"
                        )
                    for key in (
                        "allowed_identifiers",
                        "allowed_identifier_patterns",
                        "required_identifiers",
                        "required_identifier_patterns",
                        "allowed_literals",
                        "allowed_literal_patterns",
                        "allowed_operators",
                    ):
                        values = semantic_policy.get(key, [])
                        if not isinstance(values, list) or not all(
                            isinstance(value, str) and value for value in values
                        ):
                            raise PlannerError(
                                f"{path}: qualification_profiles.{profile_id}.selector.files"
                                f"[{file_index}].semantic_policy.{key} must be a string list"
                            )
                    for key in (
                        "allowed_identifier_patterns",
                        "required_identifier_patterns",
                        "allowed_literal_patterns",
                    ):
                        try:
                            for value in semantic_policy.get(key, []):
                                re.compile(value)
                        except re.error as error:
                            raise PlannerError(
                                f"{path}: qualification_profiles.{profile_id}.selector.files"
                                f"[{file_index}].semantic_policy.{key} contains invalid regex"
                            ) from error
                for contract_index, contract in enumerate(contracts):
                    if not isinstance(contract, dict):
                        raise PlannerError(
                            f"{path}: qualification_profiles.{profile_id}.selector.files"
                            f"[{file_index}].contracts[{contract_index}] must be an object"
                        )
                    contract_id = contract.get("id")
                    contract_kind = contract.get("kind")
                    symbol = contract.get("symbol")
                    if not all(isinstance(value, str) and value for value in (contract_id, contract_kind, symbol)):
                        raise PlannerError(
                            f"{path}: qualification_profiles.{profile_id}.selector.files"
                            f"[{file_index}].contracts[{contract_index}] requires non-empty id, kind, symbol"
                        )
                    if contract_id in contract_ids:
                        raise PlannerError(
                            f"{path}: qualification_profiles.{profile_id}.selector.files"
                            f"[{file_index}].contracts contains duplicate id {contract_id!r}"
                        )
                    contract_ids.add(contract_id)
                    supported = (
                        rust_kinds
                        if language == "rust"
                        else python_kinds
                        if language == "python"
                        else json_kinds
                    )
                    if contract_kind not in supported:
                        raise PlannerError(
                            f"{path}: qualification_profiles.{profile_id}.selector.files"
                            f"[{file_index}].contracts[{contract_index}].kind is unsupported for {language}"
                        )
                    if contract_kind == "rust_use":
                        for key in ("allowed_added", "allowed_removed"):
                            values = contract.get(key, [])
                            if not isinstance(values, list) or not all(
                                isinstance(value, str) and value for value in values
                            ):
                                raise PlannerError(
                                    f"{path}: qualification_profiles.{profile_id}.selector.files"
                                    f"[{file_index}].contracts[{contract_index}].{key} must be a string list"
                                )
                    for key in ("allowed_identifiers", "allowed_numeric_literals"):
                        values = contract.get(key, [])
                        if not isinstance(values, list) or not all(
                            isinstance(value, str) and value for value in values
                        ):
                            raise PlannerError(
                                f"{path}: qualification_profiles.{profile_id}.selector.files"
                                f"[{file_index}].contracts[{contract_index}].{key} "
                                "must be a string list"
                            )
                    semantic_rewrites = contract.get("semantic_rewrites", [])
                    if not isinstance(semantic_rewrites, list):
                        raise PlannerError(
                            f"{path}: qualification_profiles.{profile_id}.selector.files"
                            f"[{file_index}].contracts[{contract_index}].semantic_rewrites "
                            "must be a list"
                        )
                    rewrite_ids: set[str] = set()
                    for rewrite_index, rewrite in enumerate(semantic_rewrites):
                        if not isinstance(rewrite, dict):
                            raise PlannerError(
                                f"{path}: qualification_profiles.{profile_id}.selector.files"
                                f"[{file_index}].contracts[{contract_index}].semantic_rewrites"
                                f"[{rewrite_index}] must be an object"
                            )
                        rewrite_id = rewrite.get("id")
                        before_rewrite = rewrite.get("before")
                        after_rewrite = rewrite.get("after")
                        if not isinstance(rewrite_id, str) or not rewrite_id:
                            raise PlannerError(
                                f"{path}: qualification_profiles.{profile_id}.selector.files"
                                f"[{file_index}].contracts[{contract_index}].semantic_rewrites"
                                f"[{rewrite_index}].id must be a non-empty string"
                            )
                        if rewrite_id in rewrite_ids:
                            raise PlannerError(
                                f"{path}: qualification_profiles.{profile_id}.selector.files"
                                f"[{file_index}].contracts[{contract_index}].semantic_rewrites "
                                f"contains duplicate id {rewrite_id!r}"
                            )
                        rewrite_ids.add(rewrite_id)
                        if not isinstance(before_rewrite, str) or not isinstance(after_rewrite, str):
                            raise PlannerError(
                                f"{path}: qualification_profiles.{profile_id}.selector.files"
                                f"[{file_index}].contracts[{contract_index}].semantic_rewrites"
                                f"[{rewrite_index}] requires string before and after values"
                            )
                        if before_rewrite == after_rewrite:
                            raise PlannerError(
                                f"{path}: qualification_profiles.{profile_id}.selector.files"
                                f"[{file_index}].contracts[{contract_index}].semantic_rewrites"
                                f"[{rewrite_index}] must change the semantic source"
                            )
                continue
            rewrites = file_selector.get("rewrites")
            if not isinstance(rewrites, list) or not rewrites:
                raise PlannerError(
                    f"{path}: qualification_profiles.{profile_id}.selector.files"
                    f"[{file_index}].rewrites must be a non-empty list"
                )
            rewrite_ids: set[str] = set()
            for rewrite_index, rewrite in enumerate(rewrites):
                if not isinstance(rewrite, dict):
                    raise PlannerError(
                        f"{path}: qualification_profiles.{profile_id}.selector.files"
                        f"[{file_index}].rewrites[{rewrite_index}] must be an object"
                    )
                for key in ("id", "before", "after"):
                    if not isinstance(rewrite.get(key), str) or not rewrite[key]:
                        raise PlannerError(
                            f"{path}: qualification_profiles.{profile_id}.selector.files"
                            f"[{file_index}].rewrites[{rewrite_index}].{key} must be a non-empty string"
                        )
                rewrite_id = str(rewrite["id"])
                if rewrite_id in rewrite_ids:
                    raise PlannerError(
                        f"{path}: qualification_profiles.{profile_id}.selector.files"
                        f"[{file_index}].rewrites contains duplicate id {rewrite_id!r}"
                    )
                rewrite_ids.add(rewrite_id)
                if rewrite["before"] == rewrite["after"]:
                    raise PlannerError(
                        f"{path}: qualification_profiles.{profile_id}.selector.files"
                        f"[{file_index}].rewrites[{rewrite_index}] must change its snippet"
                    )
    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise PlannerError(f"{path}: rules[{idx}] must be an object")
        missing = sorted(required - set(rule))
        if missing:
            raise PlannerError(f"{path}: rules[{idx}] missing {', '.join(missing)}")
        for key in ("path_globs", "domains", "required_gates", "release_invalidation"):
            if not isinstance(rule[key], list):
                raise PlannerError(f"{path}: rules[{idx}].{key} must be a list")
        required_scenarios = rule.get("required_scenarios", [])
        if not isinstance(required_scenarios, list):
            raise PlannerError(f"{path}: rules[{idx}].required_scenarios must be a list")
        if not all(isinstance(item, str) and item for item in required_scenarios):
            raise PlannerError(
                f"{path}: rules[{idx}].required_scenarios must contain non-empty strings"
            )
        unknown_scenarios = sorted(set(required_scenarios) - available_scenarios)
        if unknown_scenarios:
            raise PlannerError(
                f"{path}: rules[{idx}].required_scenarios are absent from "
                f"{PRODUCT_SCENARIOS}: {unknown_scenarios}"
            )
        if not isinstance(rule["exceptions"], list):
            raise PlannerError(f"{path}: rules[{idx}].exceptions must be a list")
        if not isinstance(rule.get("exclusive", False), bool):
            raise PlannerError(f"{path}: rules[{idx}].exclusive must be boolean")
        profile_refs = rule.get("qualification_profiles", [])
        if not isinstance(profile_refs, list) or not all(
            isinstance(item, str) and item for item in profile_refs
        ):
            raise PlannerError(f"{path}: rules[{idx}].qualification_profiles must be a string list")
        unknown_profiles = sorted(set(profile_refs) - set(profiles))
        if unknown_profiles:
            raise PlannerError(
                f"{path}: rules[{idx}].qualification_profiles references unknown profiles: "
                f"{unknown_profiles}"
            )
    return {"rules": rules, "qualification_profiles": profiles}


def load_rules(path: Path) -> list[dict[str, Any]]:
    """Compatibility helper for callers that only need broad path rules."""

    return load_rule_config(path)["rules"]


def git_file_text(revision: str, path: str) -> str | None:
    proc = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


def git_file_versions(base: str, head: str, changed_files: list[str]) -> dict[str, dict[str, str]]:
    versions: dict[str, dict[str, str]] = {}
    for path in changed_files:
        before = git_file_text(base, path)
        after = git_file_text(head, path)
        if before is not None and after is not None:
            versions[path] = {"before": before, "after": after}
    return versions


def rust_block_end(text: str, opening_brace: int) -> int | None:
    """Return the end of a Rust block, conservatively ignoring comments and strings."""

    if opening_brace >= len(text) or text[opening_brace] != "{":
        return None
    depth = 0
    index = opening_brace
    block_comment_depth = 0
    while index < len(text):
        if block_comment_depth:
            if text.startswith("/*", index):
                block_comment_depth += 1
                index += 2
            elif text.startswith("*/", index):
                block_comment_depth -= 1
                index += 2
            else:
                index += 1
            continue
        if text.startswith("//", index):
            newline = text.find("\n", index + 2)
            index = len(text) if newline < 0 else newline + 1
            continue
        if text.startswith("/*", index):
            block_comment_depth = 1
            index += 2
            continue

        raw_match = re.match(r"(?:br|rb|r)(?P<hashes>#{0,255})\"", text[index:])
        if raw_match:
            terminator = '"' + raw_match.group("hashes")
            raw_end = text.find(terminator, index + raw_match.end())
            if raw_end < 0:
                return None
            index = raw_end + len(terminator)
            continue
        if text[index] == '"':
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                elif text[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            else:
                return None
            continue
        if text[index] == "'":
            char_end = index + 1
            escaped = False
            while char_end < len(text) and char_end - index <= 32:
                char = text[char_end]
                if char == "\n":
                    break
                if char == "'" and not escaped:
                    index = char_end + 1
                    break
                if char == "\\" and not escaped:
                    escaped = True
                else:
                    escaped = False
                char_end += 1
            if index == char_end + 1:
                continue
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
            if depth < 0:
                return None
        index += 1
    return None


def production_text(text: str, allow_test_changes: bool) -> tuple[str | None, str | None]:
    if not allow_test_changes:
        return text, None
    marker = "#[cfg(test)]\nmod tests {"
    if text.count(marker) != 1:
        return None, "test_module_marker_missing_or_ambiguous"
    marker_start = text.index(marker)
    opening_brace = marker_start + len(marker) - 1
    module_end = rust_block_end(text, opening_brace)
    if module_end is None:
        return None, "test_module_block_is_invalid"
    if text[module_end:].strip():
        return None, "test_module_is_not_final_top_level_item"
    return text[:marker_start], None


def structured_rewrites_match(
    *,
    before: str,
    after: str,
    file_selector: dict[str, Any],
    allow_test_changes: bool,
) -> tuple[bool, list[str], str | None]:
    """Accept exactly the ordered, uniquely matched production rewrites in the rule."""

    expected, before_error = production_text(before, allow_test_changes)
    actual, after_error = production_text(after, allow_test_changes)
    if expected is None:
        return False, [], f"base_{before_error}"
    if actual is None:
        return False, [], f"head_{after_error}"
    applied: list[str] = []
    for rewrite in file_selector["rewrites"]:
        before_snippet = str(rewrite["before"])
        after_snippet = str(rewrite["after"])
        occurrences = expected.count(before_snippet)
        if occurrences != 1:
            return (
                False,
                applied,
                f"rewrite_{rewrite['id']}_before_occurrences_{occurrences}",
            )
        expected = expected.replace(before_snippet, after_snippet, 1)
        applied.append(str(rewrite["id"]))
    if expected != actual:
        return False, applied, "production_diff_exceeds_structured_rewrites"
    return True, applied, None


def normalized_rust_source(text: str) -> str:
    """Ignore Rust layout/comments while preserving literal bytes and punctuation."""

    tokens: list[str] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        if text.startswith("//", index):
            newline = text.find("\n", index + 2)
            index = len(text) if newline < 0 else newline + 1
            continue
        if text.startswith("/*", index):
            depth = 1
            cursor = index + 2
            while cursor < len(text) and depth:
                if text.startswith("/*", cursor):
                    depth += 1
                    cursor += 2
                elif text.startswith("*/", cursor):
                    depth -= 1
                    cursor += 2
                else:
                    cursor += 1
            if depth:
                return "invalid-rust:block-comment:" + text
            index = cursor
            continue
        raw_match = re.match(r"(?:br|rb|r)(?P<hashes>#{0,255})\"", text[index:])
        if raw_match:
            terminator = '"' + raw_match.group("hashes")
            end = text.find(terminator, index + raw_match.end())
            if end < 0:
                return "invalid-rust:raw-string:" + text
            end += len(terminator)
            tokens.append(text[index:end])
            index = end
            continue
        if text[index] == '"':
            cursor = index + 1
            while cursor < len(text):
                if text[cursor] == "\\":
                    cursor += 2
                elif text[cursor] == '"':
                    cursor += 1
                    break
                else:
                    cursor += 1
            if cursor > len(text) or text[cursor - 1 : cursor] != '"':
                return "invalid-rust:string:" + text
            tokens.append(text[index:cursor])
            index = cursor
            continue
        if text[index] == "'":
            char_match = re.match(r"'(?:\\.|[^\\'\n])'", text[index:])
            if char_match:
                tokens.append(char_match.group(0))
                index += len(char_match.group(0))
                continue
        if text[index] == ",":
            cursor = index + 1
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor < len(text) and text[cursor] == "}":
                index += 1
                continue
        tokens.append(text[index])
        index += 1
    return "".join(tokens)


def normalized_contract_source(text: str, *, language: str) -> str:
    if language == "rust":
        return normalized_rust_source(text)
    try:
        return ast.dump(ast.parse(text), include_attributes=False)
    except SyntaxError:
        return "invalid-python:" + text


SEMANTIC_NEUTRAL_IDENTIFIERS = {
    "as",
    "async",
    "await",
    "bool",
    "break",
    "class",
    "const",
    "continue",
    "def",
    "dict",
    "else",
    "enum",
    "Err",
    "False",
    "false",
    "fn",
    "for",
    "from",
    "if",
    "impl",
    "in",
    "int",
    "is",
    "let",
    "list",
    "match",
    "mut",
    "None",
    "not",
    "Ok",
    "Option",
    "or",
    "pass",
    "pub",
    "ref",
    "return",
    "Self",
    "self",
    "set",
    "Some",
    "str",
    "struct",
    "super",
    "True",
    "true",
    "tuple",
    "u128",
    "u64",
    "usize",
    "use",
    "Vec",
    "where",
    "while",
}
SEMANTIC_NEUTRAL_OPERATORS = {
    "!",
    "#",
    "&",
    "(",
    ")",
    "*",
    ",",
    ".",
    ":",
    "::",
    ";",
    "=",
    "=>",
    "?",
    "@",
    "[",
    "]",
    "_",
    "{",
    "}",
}
SEMANTIC_OBSERVABILITY_IDENTIFIERS = {
    "Any",
    "Clone",
    "Copy",
    "Debug",
    "Deserialize",
    "Eq",
    "ExecutionEventDetail",
    "ExecutionEventKind",
    "ExecutionEventSinkError",
    "Instant",
    "PartialEq",
    "Path",
    "RequestAccepted",
    "RequestId",
    "Result",
    "Sequence",
    "Serialize",
    "String",
    "SystemTime",
    "UNIX_EPOCH",
    "UnvalidatedExecutionEventDetail",
    "UnvalidatedExecutionEventDetailWire",
    "all",
    "any",
    "as_nanos",
    "as_ref",
    "checked_duration_since",
    "derive",
    "duration_since",
    "dumps",
    "enumerate",
    "format",
    "get",
    "is_empty",
    "is_some",
    "is_some_and",
    "isinstance",
    "iter",
    "json",
    "len",
    "loads",
    "map",
    "map_err",
    "map_or",
    "map_or_else",
    "matches",
    "max",
    "min",
    "new",
    "now",
    "require",
    "saturating_add",
    "saturating_sub",
    "serde",
    "serde_json",
    "sorted",
    "startswith",
    "then",
    "to_string",
    "transpose",
    "trim",
    "try_from",
    "type",
    "unwrap_or",
    "unwrap_or_else",
    "values",
    "windows",
}
TIMING_OBSERVABILITY_IDENTIFIER = re.compile(
    r"(?i).*(?:anchor|clock|coverage|duration|elapsed|instant|interval|monotonic|nanos|profile|stage|timing|wall).*"
)
TIMING_OBSERVABILITY_LITERAL = re.compile(
    r'''(?is)[rubf]*(?:"|').*(?:anchor|clock|coverage|decode|instant|interval|monotonic|nanos|profile|stage|timing|unix|wall).*(?:"|')'''
)
PROTECTED_PRODUCT_IDENTIFIER = re.compile(
    r"(?i)(?:(?:^|_)(?:admission|capacity|kv(?:_cache)?|logits?|max_tokens|min_tokens|"
    r"penalty|sampl(?:e|er|ing)|temperature|threshold|token_ids?|top_k|top_p)(?:$|_)|"
    r"(?:^|_)tokens?$)"
)


def rust_semantic_tokens(text: str) -> list[tuple[str, str]]:
    """Tokenize Rust enough to police changed operations, excluding layout/comments."""

    tokens: list[tuple[str, str]] = []
    index = 0
    multi_operators = (
        "<<=",
        ">>=",
        "..=",
        "::",
        "->",
        "=>",
        "==",
        "!=",
        "<=",
        ">=",
        "&&",
        "||",
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
        "&=",
        "|=",
        "^=",
        "<<",
        ">>",
        "..",
    )
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        if text.startswith("//", index):
            newline = text.find("\n", index + 2)
            index = len(text) if newline < 0 else newline + 1
            continue
        if text.startswith("/*", index):
            depth = 1
            cursor = index + 2
            while cursor < len(text) and depth:
                if text.startswith("/*", cursor):
                    depth += 1
                    cursor += 2
                elif text.startswith("*/", cursor):
                    depth -= 1
                    cursor += 2
                else:
                    cursor += 1
            if depth:
                raise PlannerError("invalid Rust block comment in semantic contract")
            index = cursor
            continue
        raw_match = re.match(r'(?:br|rb|r)(?P<hashes>#{0,255})"', text[index:])
        if raw_match:
            terminator = '"' + raw_match.group("hashes")
            end = text.find(terminator, index + raw_match.end())
            if end < 0:
                raise PlannerError("invalid Rust raw string in semantic contract")
            end += len(terminator)
            tokens.append(("literal", text[index:end]))
            index = end
            continue
        if text[index] == '"':
            cursor = index + 1
            while cursor < len(text):
                if text[cursor] == "\\":
                    cursor += 2
                elif text[cursor] == '"':
                    cursor += 1
                    break
                else:
                    cursor += 1
            if cursor > len(text) or text[cursor - 1 : cursor] != '"':
                raise PlannerError("invalid Rust string in semantic contract")
            tokens.append(("literal", text[index:cursor]))
            index = cursor
            continue
        char_match = re.match(r"'(?:\\.|[^\\'\n])'", text[index:])
        if char_match:
            tokens.append(("literal", char_match.group(0)))
            index += len(char_match.group(0))
            continue
        identifier = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[index:])
        if identifier:
            tokens.append(("identifier", identifier.group(0)))
            index += len(identifier.group(0))
            continue
        number = re.match(
            r"(?:0[xX][0-9A-Fa-f_]+|0[bB][01_]+|0[oO][0-7_]+|"
            r"[0-9][0-9_]*(?:\.[0-9_]+)?(?:[eE][+-]?[0-9_]+)?)(?:[A-Za-z][A-Za-z0-9_]*)?",
            text[index:],
        )
        if number:
            tokens.append(("literal", number.group(0)))
            index += len(number.group(0))
            continue
        operator = next((item for item in multi_operators if text.startswith(item, index)), None)
        if operator is None:
            operator = text[index]
        tokens.append(("operator", operator))
        index += len(operator)
    return tokens


def python_semantic_tokens(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    try:
        stream = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in stream:
            if token.type == tokenize.NAME:
                tokens.append(("identifier", token.string))
            elif token.type in {tokenize.NUMBER, tokenize.STRING}:
                tokens.append(("literal", token.string))
            elif token.type == tokenize.OP:
                tokens.append(("operator", token.string))
    except (IndentationError, tokenize.TokenError) as error:
        raise PlannerError("invalid Python source in semantic contract") from error
    return tokens


def semantic_tokens(text: str, *, language: str) -> list[tuple[str, str]]:
    if language == "rust":
        return rust_semantic_tokens(text)
    if language == "python":
        return python_semantic_tokens(text)
    raise PlannerError(f"semantic change policy does not support {language}")


def semantic_change_tokens(
    before: str, after: str, *, language: str
) -> list[tuple[str, str]]:
    before_tokens = semantic_tokens(before, language=language)
    after_tokens = semantic_tokens(after, language=language)
    changed: list[tuple[str, str]] = []
    matcher = difflib.SequenceMatcher(a=before_tokens, b=after_tokens, autojunk=False)
    for operation, before_start, before_end, after_start, after_end in matcher.get_opcodes():
        if operation == "equal":
            continue
        changed.extend(before_tokens[before_start:before_end])
        changed.extend(after_tokens[after_start:after_end])
    return changed


def semantic_rewrites_match(
    *,
    before_source: str,
    after_source: str,
    language: str,
    contract_id: str,
    rewrites: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    """Project the base contract through exact semantic-token rewrites.

    Unlike an identifier allowlist, this proves that every semantic token in the
    resulting contract is accounted for by a reviewed rewrite. Layout and
    comments may change, but an otherwise innocuous statement made only from
    allowlisted identifiers cannot hitchhike on an observability edit.
    """

    expected = semantic_tokens(before_source, language=language)
    for rewrite in rewrites:
        rewrite_id = str(rewrite["id"])
        before_tokens = semantic_tokens(str(rewrite["before"]), language=language)
        after_tokens = semantic_tokens(str(rewrite["after"]), language=language)
        if not before_tokens:
            if expected or len(rewrites) != 1:
                return (
                    False,
                    f"contract_{contract_id}_semantic_rewrite_{rewrite_id}_empty_before_ambiguous",
                )
            expected = after_tokens
            continue
        occurrences = [
            index
            for index in range(len(expected) - len(before_tokens) + 1)
            if expected[index : index + len(before_tokens)] == before_tokens
        ]
        if len(occurrences) != 1:
            return (
                False,
                f"contract_{contract_id}_semantic_rewrite_{rewrite_id}_before_occurrences_"
                f"{len(occurrences)}",
            )
        start = occurrences[0]
        expected[start : start + len(before_tokens)] = after_tokens
    if expected != semantic_tokens(after_source, language=language):
        return False, f"contract_{contract_id}_semantic_diff_exceeds_rewrites"
    return True, None


def semantic_policy_match(
    *,
    before_source: str,
    after_source: str,
    language: str,
    contract_id: str,
    policy: dict[str, Any],
    contract_allowed_identifiers: set[str],
    allowed_numeric_literals: set[str],
) -> tuple[bool, str | None]:
    if before_source and not after_source:
        return False, f"contract_{contract_id}_removal_not_allowed"
    changed = semantic_change_tokens(before_source, after_source, language=language)
    allowed_identifiers = set(policy.get("allowed_identifiers", [])) | contract_allowed_identifiers
    identifier_patterns = [re.compile(pattern) for pattern in policy.get("allowed_identifier_patterns", [])]
    required_identifiers = set(policy.get("required_identifiers", []))
    required_patterns = [re.compile(pattern) for pattern in policy.get("required_identifier_patterns", [])]
    allowed_literals = set(policy.get("allowed_literals", []))
    literal_patterns = [re.compile(pattern) for pattern in policy.get("allowed_literal_patterns", [])]
    allowed_operators = set(policy.get("allowed_operators", [])) | SEMANTIC_NEUTRAL_OPERATORS
    changed_identifiers = {value for kind, value in changed if kind == "identifier"}
    protected = sorted(
        identifier for identifier in changed_identifiers if PROTECTED_PRODUCT_IDENTIFIER.search(identifier)
    )
    if protected:
        return False, f"contract_{contract_id}_protected_product_identifier_changed"
    unexpected_identifiers = sorted(
        identifier
        for identifier in changed_identifiers
        if identifier not in SEMANTIC_NEUTRAL_IDENTIFIERS
        and identifier not in SEMANTIC_OBSERVABILITY_IDENTIFIERS
        and not TIMING_OBSERVABILITY_IDENTIFIER.fullmatch(identifier)
        and identifier not in allowed_identifiers
        and not any(pattern.fullmatch(identifier) for pattern in identifier_patterns)
    )
    if unexpected_identifiers:
        return False, f"contract_{contract_id}_identifier_delta_exceeded"
    changed_literals = {value for kind, value in changed if kind == "literal"}
    unexpected_literals = sorted(
        literal
        for literal in changed_literals
        if (
            re.fullmatch(r"[0-9].*", literal) is not None
            and literal not in allowed_numeric_literals
        )
        or (
            re.fullmatch(r"[0-9].*", literal) is None
            and literal not in allowed_literals
            and not TIMING_OBSERVABILITY_LITERAL.fullmatch(literal)
            and not any(pattern.fullmatch(literal) for pattern in literal_patterns)
        )
    )
    if unexpected_literals:
        return False, f"contract_{contract_id}_literal_delta_exceeded"
    changed_operators = {value for kind, value in changed if kind == "operator"}
    if changed_operators - allowed_operators:
        return False, f"contract_{contract_id}_operator_delta_exceeded"
    has_required_marker = (
        any(TIMING_OBSERVABILITY_IDENTIFIER.fullmatch(identifier) for identifier in changed_identifiers)
        or bool(changed_identifiers & required_identifiers)
        or any(
        pattern.fullmatch(identifier)
        for pattern in required_patterns
        for identifier in changed_identifiers
        )
    )
    if not has_required_marker:
        return False, f"contract_{contract_id}_lacks_required_semantic_marker"
    return True, None


def declaration_start(text: str, start: int) -> int:
    """Include contiguous Rust attributes and doc comments owned by an item."""

    start = text.rfind("\n", 0, start) + 1
    cursor = start
    while cursor > 0:
        previous_end = cursor - 1
        previous_start = text.rfind("\n", 0, previous_end) + 1
        line = text[previous_start:previous_end].strip()
        if line.startswith("#[") or line.startswith("///") or line.startswith("//!"):
            start = previous_start
            cursor = previous_start
            continue
        break
    return start


def rust_braced_span(text: str, match: re.Match[str]) -> tuple[int, int] | None:
    opening = text.find("{", match.end())
    semicolon = text.find(";", match.end())
    if opening < 0 or (semicolon >= 0 and semicolon < opening):
        return None
    end = rust_block_end(text, opening)
    if end is None:
        return None
    return declaration_start(text, match.start()), end


def rust_top_level_item_span(
    text: str, *, symbol: str, kinds: str
) -> tuple[int, int] | None:
    pattern = re.compile(
        rf"(?m)^(?:pub(?:\([^\n)]*\))?\s+)?(?:unsafe\s+)?(?:async\s+)?"
        rf"(?:const\s+)?(?:{kinds})\s+{re.escape(symbol)}\b"
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        return None
    return rust_braced_span(text, matches[0])


def rust_impl_blocks(text: str, wanted: str) -> list[tuple[int, int, int]]:
    canonical_wanted = re.sub(r"\s+", "", wanted)
    blocks: list[tuple[int, int, int]] = []
    for match in re.finditer(r"(?m)^impl\b", text):
        opening = text.find("{", match.end())
        if opening < 0:
            continue
        header = re.sub(r"\s+", "", text[match.start() + len("impl") : opening])
        if header != canonical_wanted:
            continue
        end = rust_block_end(text, opening)
        if end is not None:
            blocks.append((match.start(), opening, end))
    return blocks


def rust_impl_method_span(
    text: str, *, impl_name: str, method: str
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    pattern = re.compile(
        rf"(?m)^    (?:pub(?:\([^\n)]*\))?\s+)?(?:unsafe\s+)?(?:async\s+)?"
        rf"(?:const\s+)?fn\s+{re.escape(method)}\b"
    )
    for _, opening, impl_end in rust_impl_blocks(text, impl_name):
        for match in pattern.finditer(text, opening + 1, impl_end - 1):
            span = rust_braced_span(text, match)
            if span is not None and span[1] <= impl_end:
                candidates.append(span)
    return candidates[0] if len(candidates) == 1 else None


def rust_struct_field_span(
    text: str, *, struct_name: str, field: str
) -> tuple[int, int] | None:
    struct_span = rust_top_level_item_span(text, symbol=struct_name, kinds="struct")
    if struct_span is None:
        return None
    pattern = re.compile(
        rf"(?m)^    (?:pub(?:\([^\n)]*\))?\s+)?{re.escape(field)}\s*:"
    )
    matches = list(pattern.finditer(text, struct_span[0], struct_span[1]))
    if len(matches) != 1:
        return None
    end = text.find(",", matches[0].end(), struct_span[1])
    if end < 0:
        return None
    return declaration_start(text, matches[0].start()), end + 1


def rust_enum_variant_span(
    text: str, *, enum_name: str, variant: str
) -> tuple[int, int] | None:
    enum_span = rust_top_level_item_span(text, symbol=enum_name, kinds="enum")
    if enum_span is None:
        return None
    pattern = re.compile(rf"(?m)^    {re.escape(variant)}\b")
    matches = list(pattern.finditer(text, enum_span[0], enum_span[1]))
    if len(matches) != 1:
        return None
    opening = text.find("{", matches[0].end(), enum_span[1])
    comma = text.find(",", matches[0].end(), enum_span[1])
    if opening >= 0 and (comma < 0 or opening < comma):
        end = rust_block_end(text, opening)
        if end is None:
            return None
        if end < len(text) and text[end] == ",":
            end += 1
    elif comma >= 0:
        end = comma + 1
    else:
        return None
    return declaration_start(text, matches[0].start()), end


def rust_use_span(text: str, root: str) -> tuple[tuple[int, int], set[str]] | None:
    pattern = re.compile(rf"(?m)^use\s+{re.escape(root)}(?=::|\s*;)")
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        return None
    end = text.find(";", matches[0].end())
    if end < 0:
        return None
    source = text[matches[0].start() : end + 1]
    suffix = source[len("use ") + len(root) :].strip().rstrip(";")
    suffix = suffix.removeprefix("::").strip().strip("{}")
    names = {
        part.strip().split(" as ", 1)[0].strip().rsplit("::", 1)[-1]
        for part in suffix.split(",")
        if part.strip()
    }
    return (matches[0].start(), end + 1), names


def rust_contract_span(
    text: str, contract: dict[str, Any]
) -> tuple[tuple[int, int], set[str] | None] | None:
    kind = str(contract["kind"])
    symbol = str(contract["symbol"])
    if kind == "rust_const":
        pattern = re.compile(
            rf"(?m)^(?:pub(?:\([^\n)]*\))?\s+)?const\s+{re.escape(symbol)}\b"
        )
        matches = list(pattern.finditer(text))
        if len(matches) != 1:
            return None
        end = text.find(";", matches[0].end())
        span = (
            declaration_start(text, matches[0].start()),
            end + 1,
        ) if end >= 0 else None
    elif kind == "rust_type":
        span = rust_top_level_item_span(text, symbol=symbol, kinds="struct|enum")
    elif kind == "rust_function":
        span = rust_top_level_item_span(text, symbol=symbol, kinds="fn")
    elif kind == "rust_impl_method":
        if "::" not in symbol:
            return None
        impl_name, method = symbol.rsplit("::", 1)
        span = rust_impl_method_span(text, impl_name=impl_name, method=method)
    elif kind == "rust_struct_field":
        if "::" not in symbol:
            return None
        struct_name, field = symbol.rsplit("::", 1)
        span = rust_struct_field_span(text, struct_name=struct_name, field=field)
    elif kind == "rust_enum_variant":
        if "::" not in symbol:
            return None
        enum_name, variant = symbol.rsplit("::", 1)
        span = rust_enum_variant_span(text, enum_name=enum_name, variant=variant)
    elif kind == "rust_use":
        return rust_use_span(text, symbol)
    else:
        return None
    return (span, None) if span is not None else None


def remove_contract_spans(text: str, spans: list[tuple[int, int]]) -> str | None:
    ordered = sorted(spans)
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        return None
    projected = text
    for start, end in reversed(ordered):
        projected = projected[:start] + projected[end:]
    return projected


def rust_symbol_contracts_match(
    *, before: str, after: str, file_selector: dict[str, Any]
) -> tuple[bool, list[str], str | None]:
    allow_tests = bool(file_selector.get("allow_test_changes", False))
    before_product, before_error = production_text(before, allow_tests)
    after_product, after_error = production_text(after, allow_tests)
    if before_product is None:
        return False, [], f"base_{before_error}"
    if after_product is None:
        return False, [], f"head_{after_error}"
    before_spans: list[tuple[int, int]] = []
    after_spans: list[tuple[int, int]] = []
    changed: list[str] = []
    for contract in file_selector.get("contracts", []):
        contract_id = str(contract["id"])
        before_match = rust_contract_span(before_product, contract)
        after_match = rust_contract_span(after_product, contract)
        if before_match is None and after_match is None:
            continue
        before_span, before_names = before_match or (None, None)
        after_span, after_names = after_match or (None, None)
        if contract["kind"] == "rust_use":
            old_names = before_names or set()
            new_names = after_names or set()
            unexpected_added = new_names - old_names - set(contract.get("allowed_added", []))
            unexpected_removed = old_names - new_names - set(contract.get("allowed_removed", []))
            if unexpected_added or unexpected_removed:
                return False, changed, f"contract_{contract_id}_import_delta_exceeded"
        before_source = "" if before_span is None else before_product[slice(*before_span)]
        after_source = "" if after_span is None else after_product[slice(*after_span)]
        if normalized_contract_source(before_source, language="rust") != normalized_contract_source(
            after_source, language="rust"
        ):
            if contract["kind"] != "rust_use":
                semantic_rewrites = contract.get("semantic_rewrites", [])
                if semantic_rewrites:
                    rewrites_ok, rewrites_error = semantic_rewrites_match(
                        before_source=before_source,
                        after_source=after_source,
                        language="rust",
                        contract_id=contract_id,
                        rewrites=semantic_rewrites,
                    )
                    if not rewrites_ok:
                        return False, changed, rewrites_error
                policy_ok, policy_error = semantic_policy_match(
                    before_source=before_source,
                    after_source=after_source,
                    language="rust",
                    contract_id=contract_id,
                    policy=file_selector["semantic_policy"],
                    contract_allowed_identifiers=set(contract.get("allowed_identifiers", [])),
                    allowed_numeric_literals=set(contract.get("allowed_numeric_literals", [])),
                )
                if not policy_ok:
                    return False, changed, policy_error
            changed.append(contract_id)
        if before_span is not None:
            before_spans.append(before_span)
        if after_span is not None:
            after_spans.append(after_span)
    before_projection = remove_contract_spans(before_product, before_spans)
    after_projection = remove_contract_spans(after_product, after_spans)
    if before_projection is None or after_projection is None:
        return False, changed, "overlapping_symbol_contracts"
    if normalized_contract_source(before_projection, language="rust") != normalized_contract_source(
        after_projection, language="rust"
    ):
        return False, changed, "production_diff_exceeds_symbol_contracts"
    if allow_tests and before != after and before_product == after_product:
        changed.append("test_only")
    if not changed:
        return False, [], "no_symbol_contract_changed"
    return True, sorted(set(changed)), None


def python_symbol_spans(text: str) -> dict[str, tuple[int, int]]:
    tree = ast.parse(text)
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    spans: dict[str, tuple[int, int]] = {}
    for node in tree.body:
        name: str | None = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            first_line = min(
                [node.lineno, *(decorator.lineno for decorator in node.decorator_list)]
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            if len(names) != 1:
                continue
            name = names[0]
            first_line = node.lineno
        else:
            continue
        if name in spans or node.end_lineno is None:
            raise PlannerError(f"Python qualification symbol is ambiguous: {name}")
        spans[name] = (offsets[first_line - 1], offsets[node.end_lineno])
    return spans


def python_import_spans(text: str) -> dict[str, tuple[int, int]]:
    tree = ast.parse(text)
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    spans: dict[str, tuple[int, int]] = {}
    for node in tree.body:
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules = [node.module]
        for module in modules:
            if module in spans or node.end_lineno is None:
                raise PlannerError(f"Python qualification import is ambiguous: {module}")
            spans[module] = (offsets[node.lineno - 1], offsets[node.end_lineno])
    return spans


def python_symbol_contracts_match(
    *, before: str, after: str, file_selector: dict[str, Any]
) -> tuple[bool, list[str], str | None]:
    try:
        before_symbols = python_symbol_spans(before)
        after_symbols = python_symbol_spans(after)
        before_imports = python_import_spans(before)
        after_imports = python_import_spans(after)
    except (SyntaxError, PlannerError):
        return False, [], "python_symbol_parse_failed"
    before_spans: list[tuple[int, int]] = []
    after_spans: list[tuple[int, int]] = []
    changed: list[str] = []
    for contract in file_selector.get("contracts", []):
        contract_id = str(contract["id"])
        symbol = str(contract["symbol"])
        if contract["kind"] == "python_import":
            before_span = before_imports.get(symbol)
            after_span = after_imports.get(symbol)
        else:
            before_span = before_symbols.get(symbol)
            after_span = after_symbols.get(symbol)
        if before_span is None and after_span is None:
            continue
        before_source = "" if before_span is None else before[slice(*before_span)]
        after_source = "" if after_span is None else after[slice(*after_span)]
        if normalized_contract_source(before_source, language="python") != normalized_contract_source(
            after_source, language="python"
        ):
            semantic_rewrites = contract.get("semantic_rewrites", [])
            if semantic_rewrites:
                rewrites_ok, rewrites_error = semantic_rewrites_match(
                    before_source=before_source,
                    after_source=after_source,
                    language="python",
                    contract_id=contract_id,
                    rewrites=semantic_rewrites,
                )
                if not rewrites_ok:
                    return False, changed, rewrites_error
            else:
                policy_ok, policy_error = semantic_policy_match(
                    before_source=before_source,
                    after_source=after_source,
                    language="python",
                    contract_id=contract_id,
                    policy=file_selector["semantic_policy"],
                    contract_allowed_identifiers=set(contract.get("allowed_identifiers", [])),
                    allowed_numeric_literals=set(contract.get("allowed_numeric_literals", [])),
                )
                if not policy_ok:
                    return False, changed, policy_error
            changed.append(contract_id)
        if before_span is not None:
            before_spans.append(before_span)
        if after_span is not None:
            after_spans.append(after_span)
    before_projection = remove_contract_spans(before, before_spans)
    after_projection = remove_contract_spans(after, after_spans)
    if before_projection is None or after_projection is None:
        return False, changed, "overlapping_symbol_contracts"
    if normalized_contract_source(before_projection, language="python") != normalized_contract_source(
        after_projection, language="python"
    ):
        return False, changed, "production_diff_exceeds_symbol_contracts"
    if not changed:
        return False, [], "no_symbol_contract_changed"
    return True, sorted(set(changed)), None


def json_path_parts(path: str) -> list[tuple[str, tuple[str, str] | None]] | None:
    parts: list[tuple[str, tuple[str, str] | None]] = []
    for raw in path.split("."):
        match = re.fullmatch(
            r"(?P<key>[A-Za-z0-9_-]+)(?:\[(?P<select_key>[A-Za-z0-9_-]+)=(?P<select_value>[^\]]+)\])?",
            raw,
        )
        if match is None:
            return None
        selector = None
        if match.group("select_key") is not None:
            selector = (match.group("select_key"), match.group("select_value"))
        parts.append((match.group("key"), selector))
    return parts


def json_contract_value(document: Any, path: str) -> tuple[bool, Any]:
    parts = json_path_parts(path)
    if parts is None:
        return False, None
    current = document
    for key, selector in parts:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
        if selector is not None:
            selector_key, selector_value = selector
            if not isinstance(current, list):
                return False, None
            matches = [
                item
                for item in current
                if isinstance(item, dict)
                and str(item.get(selector_key)) == selector_value
            ]
            if len(matches) != 1:
                return False, None
            current = matches[0]
    return True, current


def remove_json_contract(document: Any, path: str) -> bool:
    parts = json_path_parts(path)
    if not parts:
        return False
    current = document
    for key, selector in parts[:-1]:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
        if selector is not None:
            selector_key, selector_value = selector
            if not isinstance(current, list):
                return False
            matches = [
                item
                for item in current
                if isinstance(item, dict)
                and str(item.get(selector_key)) == selector_value
            ]
            if len(matches) != 1:
                return False
            current = matches[0]
    key, selector = parts[-1]
    if not isinstance(current, dict) or key not in current:
        return False
    if selector is None:
        del current[key]
        return True
    value = current[key]
    selector_key, selector_value = selector
    if not isinstance(value, list):
        return False
    matches = [
        index
        for index, item in enumerate(value)
        if isinstance(item, dict) and str(item.get(selector_key)) == selector_value
    ]
    if len(matches) != 1:
        return False
    del value[matches[0]]
    return True


def json_symbol_contracts_match(
    *, before: str, after: str, file_selector: dict[str, Any]
) -> tuple[bool, list[str], str | None]:
    try:
        before_document = json.loads(before)
        after_document = json.loads(after)
    except json.JSONDecodeError:
        return False, [], "json_symbol_parse_failed"
    before_projection = copy.deepcopy(before_document)
    after_projection = copy.deepcopy(after_document)
    changed: list[str] = []
    for contract in file_selector.get("contracts", []):
        contract_id = str(contract["id"])
        symbol = str(contract["symbol"])
        before_found, before_value = json_contract_value(before_document, symbol)
        after_found, after_value = json_contract_value(after_document, symbol)
        if not before_found and not after_found:
            continue
        if before_found != after_found or before_value != after_value:
            changed.append(contract_id)
        if before_found and not remove_json_contract(before_projection, symbol):
            return False, changed, f"contract_{contract_id}_base_projection_failed"
        if after_found and not remove_json_contract(after_projection, symbol):
            return False, changed, f"contract_{contract_id}_head_projection_failed"
    if before_projection != after_projection:
        return False, changed, "production_diff_exceeds_json_contracts"
    if not changed:
        return False, [], "no_json_contract_changed"
    return True, sorted(set(changed)), None


def symbol_contracts_match(
    *, before: str, after: str, file_selector: dict[str, Any]
) -> tuple[bool, list[str], str | None]:
    if file_selector.get("test_only") is True:
        if before == after:
            return False, [], "test_only_file_did_not_change"
        if file_selector.get("language") == "rust" and file_selector.get(
            "allow_test_changes"
        ):
            before_product, before_error = production_text(before, True)
            after_product, after_error = production_text(after, True)
            if before_product is None:
                return False, [], f"base_{before_error}"
            if after_product is None:
                return False, [], f"head_{after_error}"
            if normalized_contract_source(
                before_product, language="rust"
            ) != normalized_contract_source(after_product, language="rust"):
                return False, [], "test_only_file_changed_production"
        return True, ["test_only"], None
    if file_selector.get("language") == "rust":
        return rust_symbol_contracts_match(
            before=before, after=after, file_selector=file_selector
        )
    if file_selector.get("language") == "python":
        return python_symbol_contracts_match(
            before=before, after=after, file_selector=file_selector
        )
    return json_symbol_contracts_match(
        before=before, after=after, file_selector=file_selector
    )


def qualification_for_path(
    path: str,
    profile_id: str,
    profile: dict[str, Any],
    file_versions: dict[str, dict[str, str]],
) -> dict[str, Any] | None:
    selector = profile["selector"]
    file_selector = next(
        (item for item in selector["files"] if item.get("path") == path),
        None,
    )
    versions = file_versions.get(path)
    if file_selector is None or versions is None:
        return None
    selector_kind = str(selector["kind"])
    if selector_kind == "structured_rewrites":
        matched, applied_rewrites, mismatch_reason = structured_rewrites_match(
            before=str(versions.get("before", "")),
            after=str(versions.get("after", "")),
            file_selector=file_selector,
            allow_test_changes=bool(selector.get("allow_test_changes", False)),
        )
    else:
        matched, applied_rewrites, mismatch_reason = symbol_contracts_match(
            before=str(versions.get("before", "")),
            after=str(versions.get("after", "")),
            file_selector=file_selector,
        )
    if not matched:
        return None
    return {
        "profile_id": profile_id,
        "path": path,
        "selector_kind": selector_kind,
        "applied_rewrites": applied_rewrites,
        "changed_regions": applied_rewrites,
        "qualified_scopes": profile["qualified_scopes"],
        "required_checks": profile["required_checks"],
        "mismatch_reason": mismatch_reason,
    }


def matches_any(path: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in globs)


def apply_exceptions(
    path: str,
    rule: dict[str, Any],
    required_gates: set[str],
    decision_log: list[dict[str, Any]],
) -> None:
    for exception in rule.get("exceptions", []):
        if not isinstance(exception, dict):
            continue
        globs = exception.get("path_globs")
        if not isinstance(globs, list) or not matches_any(path, globs):
            continue
        removed = sorted(set(exception.get("remove_required_gates") or []) & required_gates)
        required_gates.difference_update(removed)
        decision_log.append(
            {
                "path": path,
                "rule_id": rule["id"],
                "exception_id": exception.get("id"),
                "removed_required_gates": removed,
                "reason": exception.get("reason"),
            }
        )


def artifact_id(artifact: dict[str, Any], index: int) -> str:
    raw = artifact.get("id") or artifact.get("gate") or artifact.get("artifact_dir") or f"artifact-{index}"
    return str(raw)


def stage_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise PlannerError("--stage-artifact entries must use GATE=ARTIFACT_DIR")
    gate, raw_path = value.split("=", 1)
    gate = gate.strip()
    if not gate:
        raise PlannerError("--stage-artifact gate name must be non-empty")
    if gate not in FINAL_STAGE_GATES:
        raise PlannerError(f"--stage-artifact gate must be one of {sorted(FINAL_STAGE_GATES)}: {gate}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return gate, path


def read_stage_manifest(artifact_dir: Path) -> dict[str, Any]:
    manifest_path = artifact_dir / "gate.manifest.json"
    if not manifest_path.exists():
        raise PlannerError(f"{artifact_dir}: missing gate.manifest.json")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PlannerError(f"{manifest_path}: expected JSON object")
    return data


def normalize_stage_artifact(value: str, head_sha: str) -> dict[str, Any]:
    gate, artifact_dir = stage_spec(value)
    if not artifact_dir.is_dir():
        raise PlannerError(f"{artifact_dir}: stage artifact directory does not exist")
    manifest = read_stage_manifest(artifact_dir)
    pass_line = str(manifest.get("pass_line") or "")
    if " PASS:" not in pass_line:
        raise PlannerError(f"{artifact_dir}: gate.manifest.json pass_line must contain ' PASS:'")
    git_sha = str(manifest.get("git_sha") or "")
    artifact_dirty = bool(manifest.get("git_dirty"))
    artifact = {
        "id": gate,
        "gate": gate,
        "artifact_dir": str(artifact_dir),
        "pass_line": pass_line,
        "git_sha": git_sha,
        "git_dirty": artifact_dirty,
        "impact_domains": list(manifest.get("impact_domains") or []),
        "strict_current": True,
        "manifest": str(artifact_dir / "gate.manifest.json"),
    }
    if git_sha != head_sha:
        artifact["stale_reason"] = "stage artifact git_sha does not match planned head"
    if artifact_dirty:
        artifact["stale_reason"] = "stage artifact was produced from a dirty tree"
    return artifact


def stale_artifact_invalidations(
    previous_artifacts: list[dict[str, Any]],
    impact_domains: set[str],
    head_sha: str,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    invalidated: list[str] = []
    stale: list[dict[str, Any]] = []
    satisfied: list[dict[str, Any]] = []
    for index, artifact in enumerate(previous_artifacts):
        if not isinstance(artifact, dict):
            continue
        artifact_domains = set(str(item) for item in artifact.get("impact_domains", []) if item)
        intersects = bool(artifact_domains & impact_domains)
        artifact_sha = str(artifact.get("git_sha") or "")
        artifact_dirty = bool(artifact.get("git_dirty"))
        strict_current = bool(artifact.get("strict_current"))
        aid = artifact_id(artifact, index)
        if (strict_current and artifact_sha != head_sha) or (strict_current and artifact_dirty):
            invalidated.append(f"artifact:{aid}")
            reason = artifact.get("stale_reason") or "strict stage artifact is not current"
            stale.append({**artifact, "id": aid, "stale_reason": reason})
        elif intersects and artifact_sha != head_sha:
            invalidated.append(f"artifact:{aid}")
            stale.append({**artifact, "id": aid, "stale_reason": "impact domain changed after artifact"})
        else:
            satisfied.append({**artifact, "id": aid})
    return invalidated, stale, satisfied


def plan_from_files(
    *,
    changed_files: list[str],
    base_sha: str,
    head_sha: str,
    dirty: bool,
    rules: list[dict[str, Any]],
    qualification_profiles: dict[str, dict[str, Any]] | None = None,
    file_versions: dict[str, dict[str, str]] | None = None,
    previous_artifacts: list[dict[str, Any]] | None = None,
    required_gate_overrides: set[str] | None = None,
) -> dict[str, Any]:
    changed_files = normalize_changed_files(changed_files)
    impact_domains: set[str] = set()
    required_gates: set[str] = set()
    required_product_scenarios: set[str] = set()
    invalidated: set[str] = set()
    optional_diagnostic_gates: set[str] = set()
    required_checks: set[str] = set()
    qualified_scopes_by_key: dict[str, dict[str, str]] = {}
    qualification_matches: list[dict[str, Any]] = []
    unknown_files: list[str] = []
    decision_log: list[dict[str, Any]] = []
    qualification_profiles = qualification_profiles or {}
    file_versions = file_versions or {}

    for changed in changed_files:
        matched = False
        path_qualification_matches: dict[str, dict[str, Any]] = {}
        path_rules = [rule for rule in rules if matches_any(changed, rule["path_globs"])]
        exclusive_rules = [rule for rule in path_rules if rule.get("exclusive") is True]
        if exclusive_rules:
            path_rules = exclusive_rules
        for rule in path_rules:
            for profile_id in rule.get("qualification_profiles", []):
                if profile_id in path_qualification_matches:
                    continue
                match = qualification_for_path(
                    changed,
                    profile_id,
                    qualification_profiles[profile_id],
                    file_versions,
                )
                if match is not None:
                    path_qualification_matches[profile_id] = match
        for rule in path_rules:
            matched = True
            matched_profile_ids = sorted(
                set(rule.get("qualification_profiles", [])) & set(path_qualification_matches)
            )
            if matched_profile_ids:
                decision_log.append(
                    {
                        "path": changed,
                        "rule_id": rule["id"],
                        "qualification_profiles": matched_profile_ids,
                        "broad_rule_replaced": True,
                        "reason": "production diff matched the profile selector and semantic-operation policy",
                    }
                )
                continue
            impact_domains.update(str(domain) for domain in rule["domains"])
            before_gates = set(required_gates)
            before_scenarios = set(required_product_scenarios)
            required_gates.update(str(gate) for gate in rule["required_gates"])
            required_product_scenarios.update(
                str(scenario) for scenario in rule.get("required_scenarios", [])
            )
            invalidated.update(str(gate) for gate in rule["release_invalidation"])
            apply_exceptions(changed, rule, required_gates, decision_log)
            decision_log.append(
                {
                    "path": changed,
                    "rule_id": rule["id"],
                    "domains": rule["domains"],
                    "required_gates_added": sorted(required_gates - before_gates),
                    "required_product_scenarios_added": sorted(
                        required_product_scenarios - before_scenarios
                    ),
                    "release_invalidation": rule["release_invalidation"],
                    "owner": rule["owner"],
                    "reason": rule["reason"],
                }
            )
        for profile_id, match in sorted(path_qualification_matches.items()):
            profile = qualification_profiles[profile_id]
            impact_domains.update(str(domain) for domain in profile["domains"])
            required_gates.update(str(gate) for gate in profile.get("required_gates", []))
            required_checks.update(str(check) for check in profile["required_checks"])
            for scope in profile["qualified_scopes"]:
                scope_key = json.dumps(scope, sort_keys=True, separators=(",", ":"))
                qualified_scopes_by_key[scope_key] = dict(scope)
            qualification_matches.append(match)
            decision_log.append(
                {
                    "path": changed,
                    "rule_id": None,
                    "qualification_profile": profile_id,
                    "domains": profile["domains"],
                    "required_checks_added": profile["required_checks"],
                    "required_gates_added": profile.get("required_gates", []),
                    "qualified_scopes": profile["qualified_scopes"],
                    "applied_rewrites": match["applied_rewrites"],
                    "reason": profile.get("reason"),
                }
            )
        if not matched:
            unknown_files.append(changed)
            decision_log.append(
                {
                    "path": changed,
                    "rule_id": None,
                    "domains": ["unknown"],
                    "reason": "no change-impact rule matched this path",
                }
            )

    artifact_invalidations, stale_artifacts, satisfied_artifacts = stale_artifact_invalidations(
        previous_artifacts or [],
        impact_domains,
        head_sha,
    )
    required_gates.update(required_gate_overrides or set())
    invalidated.update(artifact_invalidations)
    satisfied_gate_names = {
        str(artifact.get("gate"))
        for artifact in satisfied_artifacts
        if isinstance(artifact.get("gate"), str) and artifact.get("gate")
    }
    invalidated.difference_update(satisfied_gate_names)
    status = "fail" if unknown_files else "pass"
    return {
        "schema_version": 1,
        "status": status,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "dirty": dirty,
        "changed_files": changed_files,
        "impact_domains": sorted(impact_domains),
        "required_gates": sorted(required_gates),
        "required_product_scenarios": sorted(required_product_scenarios),
        "required_checks": sorted(required_checks),
        "qualified_scopes": [qualified_scopes_by_key[key] for key in sorted(qualified_scopes_by_key)],
        "qualification_matches": qualification_matches,
        "optional_diagnostic_gates": sorted(optional_diagnostic_gates),
        "invalidated_previous_gates": sorted(invalidated),
        "unknown_files": unknown_files,
        "decision_log": decision_log,
        "previous_artifacts": previous_artifacts or [],
        "satisfied_artifacts": satisfied_artifacts,
        "stale_artifacts": stale_artifacts,
    }


def markdown_plan(plan: dict[str, Any]) -> str:
    lines = [
        "# Gate Plan",
        "",
        f"- status: `{plan['status']}`",
        f"- base_sha: `{plan['base_sha']}`",
        f"- head_sha: `{plan['head_sha']}`",
        f"- dirty: `{plan['dirty']}`",
        f"- impact domains: {', '.join(plan['impact_domains']) or '(none)'}",
        f"- required gates: {', '.join(plan['required_gates']) or '(none)'}",
        f"- required qualified checks: {', '.join(plan['required_checks']) or '(none)'}",
        "- required product scenarios: "
        f"{', '.join(plan['required_product_scenarios']) or '(none)'}",
        f"- qualified scopes: {json.dumps(plan['qualified_scopes'], sort_keys=True)}",
        f"- invalidated gates: {', '.join(plan['invalidated_previous_gates']) or '(none)'}",
        "",
        "## Changed Files",
        "",
    ]
    lines.extend(f"- `{path}`" for path in plan["changed_files"])
    if plan["unknown_files"]:
        lines.extend(["", "## Unknown Files", ""])
        lines.extend(f"- `{path}`" for path in plan["unknown_files"])
    lines.extend(["", "## Decisions", "", "| path | rule | domains | reason |", "|---|---|---|---|"])
    for decision in plan["decision_log"]:
        lines.append(
            "| {path} | {rule} | {domains} | {reason} |".format(
                path=decision.get("path"),
                rule=decision.get("rule_id"),
                domains=", ".join(decision.get("domains") or []),
                reason=str(decision.get("reason", "")).replace("|", "\\|"),
            )
        )
    return "\n".join(lines) + "\n"


def release_candidate_manifest(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "base_sha": plan["base_sha"],
        "head_sha": plan["head_sha"],
        "dirty": plan["dirty"],
        "changed_files": plan["changed_files"],
        "impact_domains": plan["impact_domains"],
        "required_gates": plan["required_gates"],
        "required_product_scenarios": plan["required_product_scenarios"],
        "required_checks": plan["required_checks"],
        "qualified_scopes": plan["qualified_scopes"],
        "qualification_matches": plan["qualification_matches"],
        "satisfied_gates": [
            artifact.get("gate") for artifact in plan["satisfied_artifacts"] if artifact.get("gate")
        ],
        "invalidated_gates": plan["invalidated_previous_gates"],
        "invalidation_reason": "derived from change-impact rules and stale artifact impact domains",
        "artifact_paths": [
            artifact.get("artifact_dir")
            for artifact in plan["satisfied_artifacts"]
            if artifact.get("artifact_dir")
        ],
        "pass_lines": [
            artifact.get("pass_line") for artifact in plan["satisfied_artifacts"] if artifact.get("pass_line")
        ],
        "stale_artifacts": plan["stale_artifacts"],
    }


def write_outputs(out: Path, plan: dict[str, Any], selfcheck: dict[str, Any] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    (out / "gate_plan.md").write_text(markdown_plan(plan))
    (out / "changed_files.json").write_text(
        json.dumps({"schema_version": 1, "changed_files": plan["changed_files"]}, indent=2, sort_keys=True)
        + "\n"
    )
    (out / "release_candidate_manifest.json").write_text(
        json.dumps(release_candidate_manifest(plan), indent=2, sort_keys=True) + "\n"
    )
    (out / "planner_selfcheck.json").write_text(
        json.dumps(selfcheck or {"schema_version": 1, "status": "not_run"}, indent=2, sort_keys=True)
        + "\n"
    )


def write_standard_artifact_files(
    out: Path,
    *,
    plan: dict[str, Any],
    selfcheck: dict[str, Any] | None,
    started_at: int,
    ended_at: int,
    pass_line: str,
    command: list[str],
    rules: Path,
    fixtures: Path,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "failures").mkdir(exist_ok=True)
    (out / "diagnostics").mkdir(exist_ok=True)
    dirty_files = git_dirty_files()
    (out / "pass_line.txt").write_text(pass_line + "\n", encoding="utf-8")
    (out / "command.log").write_text(" ".join(command) + "\n", encoding="utf-8")
    (out / "git_status.txt").write_text(run_git(["status", "--short"]), encoding="utf-8")
    (out / "sanitized_env.json").write_text(
        json.dumps(sanitized_env(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "goal": GOAL,
        "phase": "change_impact",
        "status": plan["status"],
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_sec": ended_at - started_at,
        "repo_root": str(REPO_ROOT),
        "git_sha": plan["head_sha"],
        "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "command": command,
        "artifact_dir": str(out),
        "pass_line": pass_line,
        "inputs": {
            "rules": str(rules),
            "fixtures": str(fixtures),
            "product_scenarios": str(PRODUCT_SCENARIOS),
            "base_sha": plan["base_sha"],
            "head_sha": plan["head_sha"],
            "changed_files": plan["changed_files"],
            "previous_artifact_count": len(plan["previous_artifacts"]),
        },
        "outputs": {
            "gate_plan": str(out / "gate_plan.json"),
            "gate_plan_markdown": str(out / "gate_plan.md"),
            "release_candidate_manifest": str(out / "release_candidate_manifest.json"),
            "planner_selfcheck": str(out / "planner_selfcheck.json"),
            "changed_files": str(out / "changed_files.json"),
        },
        "validation_summary": {
            "impact_domains": plan["impact_domains"],
            "required_gates": plan["required_gates"],
            "required_product_scenarios": plan["required_product_scenarios"],
            "required_checks": plan["required_checks"],
            "qualified_scopes": plan["qualified_scopes"],
            "qualification_match_count": len(plan["qualification_matches"]),
            "unknown_file_count": len(plan["unknown_files"]),
            "stale_artifact_count": len(plan["stale_artifacts"]),
            "satisfied_artifact_count": len(plan["satisfied_artifacts"]),
            "planner_selfcheck_status": (selfcheck or {}).get("status", "not_run"),
        },
    }
    (out / "gate.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def assert_contains(label: str, actual: list[str], expected: list[str]) -> list[str]:
    missing = sorted(set(expected) - set(actual))
    return [f"{label} missing expected values: {missing}"] if missing else []


def assert_not_contains(label: str, actual: list[str], forbidden: list[str]) -> list[str]:
    present = sorted(set(forbidden) & set(actual))
    return [f"{label} unexpectedly contained values: {present}"] if present else []


def load_fixture_data(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise PlannerError(f"{path}: schema_version must be 1")
    fixtures = data.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise PlannerError(f"{path}: fixtures must be a non-empty list")
    return fixtures


def run_selftest(
    rules: list[dict[str, Any]],
    qualification_profiles: dict[str, dict[str, Any]],
    fixtures_path: Path,
) -> dict[str, Any]:
    fixture_results: list[dict[str, Any]] = []
    failures: list[str] = []
    fixtures = load_fixture_data(fixtures_path)
    safe_product, safe_error = production_text(
        'fn product() { let sample = r#"{ignored}"#; }\n\n#[cfg(test)]\nmod tests {\n'
        '    #[test]\n    fn test_only() { assert_eq!("}", "}"); }\n}\n',
        True,
    )
    if safe_error is not None or safe_product != 'fn product() { let sample = r#"{ignored}"#; }\n\n':
        failures.append(f"final test module was not isolated safely: {safe_error}")
    unsafe_product, unsafe_error = production_text(
        "fn product() {}\n\n#[cfg(test)]\nmod tests {\n}\n\nfn production_tail() {}\n",
        True,
    )
    if unsafe_product is not None or unsafe_error != "test_module_is_not_final_top_level_item":
        failures.append("production item after test module was not rejected")
    for fixture in fixtures:
        fid = str(fixture.get("id"))
        expected_error = fixture.get("expect_error_contains")
        try:
            plan = plan_from_files(
                changed_files=list(fixture.get("changed_files") or []),
                base_sha="fixture-base",
                head_sha="fixture-head",
                dirty=False,
                rules=rules,
                qualification_profiles=qualification_profiles,
                file_versions=dict(fixture.get("file_versions") or {}),
                previous_artifacts=list(fixture.get("previous_artifacts") or []),
                required_gate_overrides=FINAL_STAGE_GATES
                if fixture.get("require_final_stage_gates")
                else set(),
            )
        except PlannerError as error:
            message = str(error)
            fixture_failures = []
            if not isinstance(expected_error, str) or expected_error not in message:
                fixture_failures.append(f"unexpected planner error: {message}")
            if fixture_failures:
                failures.extend(f"{fid}: {failure}" for failure in fixture_failures)
            fixture_results.append(
                {
                    "id": fid,
                    "status": "pass" if not fixture_failures else "fail",
                    "plan_status": "error",
                    "impact_domains": [],
                    "required_gates": [],
                    "required_product_scenarios": [],
                    "unknown_files": [],
                    "invalidated_previous_gates": [],
                    "error": message,
                    "failures": fixture_failures,
                }
            )
            continue
        release_candidate = release_candidate_manifest(plan)
        fixture_failures: list[str] = []
        if expected_error is not None:
            fixture_failures.append(f"expected planner error containing {expected_error!r}")
        expected_status = fixture.get("expect_status", "pass")
        if plan["status"] != expected_status:
            fixture_failures.append(f"status {plan['status']!r} != {expected_status!r}")
        fixture_failures += assert_contains(
            "impact_domains", plan["impact_domains"], list(fixture.get("expect_domains") or [])
        )
        fixture_failures += assert_not_contains(
            "impact_domains", plan["impact_domains"], list(fixture.get("forbid_domains") or [])
        )
        if "expect_domains_exact" in fixture and plan["impact_domains"] != fixture["expect_domains_exact"]:
            fixture_failures.append(
                f"impact_domains {plan['impact_domains']!r} != {fixture['expect_domains_exact']!r}"
            )
        fixture_failures += assert_contains(
            "required_gates", plan["required_gates"], list(fixture.get("expect_required_gates") or [])
        )
        fixture_failures += assert_contains(
            "required_product_scenarios",
            plan["required_product_scenarios"],
            list(fixture.get("expect_required_scenarios") or []),
        )
        fixture_failures += assert_not_contains(
            "required_gates", plan["required_gates"], list(fixture.get("forbid_required_gates") or [])
        )
        if "expect_required_gates_exact" in fixture and plan["required_gates"] != fixture["expect_required_gates_exact"]:
            fixture_failures.append(
                f"required_gates {plan['required_gates']!r} != {fixture['expect_required_gates_exact']!r}"
            )
        fixture_failures += assert_contains(
            "required_checks", plan["required_checks"], list(fixture.get("expect_required_checks") or [])
        )
        fixture_failures += assert_not_contains(
            "required_checks", plan["required_checks"], list(fixture.get("forbid_required_checks") or [])
        )
        if "expect_required_checks_exact" in fixture and plan["required_checks"] != fixture["expect_required_checks_exact"]:
            fixture_failures.append(
                f"required_checks {plan['required_checks']!r} != {fixture['expect_required_checks_exact']!r}"
            )
        actual_profile_ids = sorted(
            str(match.get("profile_id")) for match in plan["qualification_matches"]
        )
        fixture_failures += assert_contains(
            "qualification_profile_ids",
            actual_profile_ids,
            list(fixture.get("expect_qualification_profile_ids") or []),
        )
        fixture_failures += assert_not_contains(
            "qualification_profile_ids",
            actual_profile_ids,
            list(fixture.get("forbid_qualification_profile_ids") or []),
        )
        if "expect_qualified_scopes" in fixture and plan["qualified_scopes"] != fixture["expect_qualified_scopes"]:
            fixture_failures.append(
                f"qualified_scopes {plan['qualified_scopes']!r} != {fixture['expect_qualified_scopes']!r}"
            )
        if "expect_unknown_files" in fixture and plan["unknown_files"] != fixture["expect_unknown_files"]:
            fixture_failures.append(
                f"unknown_files {plan['unknown_files']!r} != {fixture['expect_unknown_files']!r}"
            )
        if "expect_changed_files" in fixture and plan["changed_files"] != fixture["expect_changed_files"]:
            fixture_failures.append(
                f"changed_files {plan['changed_files']!r} != {fixture['expect_changed_files']!r}"
            )
        if "expect_invalidated" in fixture and plan["invalidated_previous_gates"] != fixture["expect_invalidated"]:
            fixture_failures.append(
                "invalidated_previous_gates "
                f"{plan['invalidated_previous_gates']!r} != {fixture['expect_invalidated']!r}"
            )
        fixture_failures += assert_contains(
            "invalidated_previous_gates",
            plan["invalidated_previous_gates"],
            list(fixture.get("expect_invalidated_contains") or []),
        )
        fixture_failures += assert_contains(
            "satisfied_gates",
            release_candidate["satisfied_gates"],
            list(fixture.get("expect_satisfied_gates") or []),
        )
        if "expect_artifact_path_count" in fixture:
            actual_count = len(release_candidate["artifact_paths"])
            expected_count = int(fixture["expect_artifact_path_count"])
            if actual_count != expected_count:
                fixture_failures.append(f"artifact_paths count {actual_count} != {expected_count}")
        if fixture_failures:
            failures.extend(f"{fid}: {failure}" for failure in fixture_failures)
        fixture_results.append(
            {
                "id": fid,
                "status": "pass" if not fixture_failures else "fail",
                "plan_status": plan["status"],
                "impact_domains": plan["impact_domains"],
                "required_gates": plan["required_gates"],
                "required_product_scenarios": plan["required_product_scenarios"],
                "required_checks": plan["required_checks"],
                "qualified_scopes": plan["qualified_scopes"],
                "qualification_profile_ids": actual_profile_ids,
                "unknown_files": plan["unknown_files"],
                "invalidated_previous_gates": plan["invalidated_previous_gates"],
                "failures": fixture_failures,
            }
        )
    return {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "fixture_count": len(fixtures),
        "fixtures": fixture_results,
        "failures": failures,
    }


def load_previous_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            artifacts.append(data)
        else:
            raise PlannerError(f"{path}: previous artifact manifest must be a JSON object")
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--previous-artifact", action="append", type=Path, default=[])
    parser.add_argument(
        "--stage-artifact",
        action="append",
        default=[],
        help="final goal stage evidence as GATE=ARTIFACT_DIR; reads ARTIFACT_DIR/gate.manifest.json",
    )
    parser.add_argument(
        "--require-final-stage-gates",
        action="store_true",
        help="force the final goal stage gates into required_gates for release candidate aggregation",
    )
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    started_at = int(time.time())
    args = parse_args()
    rule_config = load_rule_config(args.rules)
    rules = rule_config["rules"]
    qualification_profiles = rule_config["qualification_profiles"]
    if args.self_test:
        selfcheck = run_selftest(rules, qualification_profiles, args.fixtures)
        if args.out:
            plan = plan_from_files(
                changed_files=[],
                base_sha="self-test",
                head_sha="self-test",
                dirty=False,
                rules=rules,
                qualification_profiles=qualification_profiles,
            )
            write_outputs(
                args.out,
                plan,
                selfcheck,
            )
            write_standard_artifact_files(
                args.out,
                plan=plan,
                selfcheck=selfcheck,
                started_at=started_at,
                ended_at=int(time.time()),
                pass_line=f"{SELFTEST_PASS_LINE}: {args.out}",
                command=sys.argv,
                rules=args.rules,
                fixtures=args.fixtures,
            )
        if selfcheck["status"] != "pass":
            raise PlannerError("\n".join(selfcheck["failures"]))
        suffix = f": {args.out}" if args.out else ""
        print(f"{SELFTEST_PASS_LINE}{suffix}")
        return 0

    if args.changed_file:
        changed_files = normalize_changed_files(args.changed_file)
        base_sha = resolve_git_rev(args.base) if args.base else "manual-base"
        head_sha = resolve_git_rev(args.head) if args.head else "manual-head"
        file_versions: dict[str, dict[str, str]] = {}
    else:
        if not args.base or not args.head:
            raise PlannerError("provide --base and --head, or one or more --changed-file entries")
        changed_files = git_changed_files(args.base, args.head)
        base_sha = resolve_git_rev(args.base)
        head_sha = resolve_git_rev(args.head)
        file_versions = git_file_versions(base_sha, head_sha, changed_files)
    previous_artifacts = load_previous_artifacts(args.previous_artifact)
    previous_artifacts.extend(normalize_stage_artifact(value, head_sha) for value in args.stage_artifact)
    plan = plan_from_files(
        changed_files=changed_files,
        base_sha=base_sha,
        head_sha=head_sha,
        dirty=git_dirty(),
        rules=rules,
        qualification_profiles=qualification_profiles,
        file_versions=file_versions,
        previous_artifacts=previous_artifacts,
        required_gate_overrides=FINAL_STAGE_GATES if args.require_final_stage_gates else set(),
    )
    if args.out is None:
        raise PlannerError("--out is required for non-self-test runs")
    write_outputs(args.out, plan)
    pass_line = f"{PASS_LINE}: {args.out}"
    write_standard_artifact_files(
        args.out,
        plan=plan,
        selfcheck=None,
        started_at=started_at,
        ended_at=int(time.time()),
        pass_line=pass_line,
        command=sys.argv,
        rules=args.rules,
        fixtures=args.fixtures,
    )
    if plan["status"] != "pass":
        raise PlannerError(f"gate plan failed: unknown_files={plan['unknown_files']}")
    print(pass_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
