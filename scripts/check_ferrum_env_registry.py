#!/usr/bin/env python3
"""Audit Ferrum runtime environment knobs and compare them to a registry.

The scanner is intentionally static and conservative: it records every
uppercase Ferrum-looking token in product/build/bench sources, plus every
direct environment read call. The optional registry check is the first step
toward making new knobs visible before they land in hot paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SCAN_ROOTS = (
    "crates",
    "scripts",
    "Cargo.toml",
    "ferrum.toml",
    ".github",
)
DEFAULT_IGNORED_NAMES = "docs/runtime-env-registry-ignore.txt"
DEFAULT_MISSING_BASELINE = "docs/runtime-env-registry-missing-baseline.txt"

HOT_ROOTS = (
    "crates/ferrum-engine/src",
    "crates/ferrum-models/src",
    "crates/ferrum-kernels/src",
    "crates/ferrum-kernels/kernels",
)

SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".metal",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}

EXCLUDED_REL_PATHS = {
    # Keep the checker from counting its own regex strings.
    "scripts/check_ferrum_env_registry.py",
}

FERRUM_TOKEN_RE = re.compile(r"FERRUM_[A-Z0-9_]+")
DIRECT_ENV_READ_RE = re.compile(
    r"(?P<call>std::env::var_os|std::env::var|env::var_os|env::var|std::getenv|getenv)\s*\("
)
PROCESS_ENV_WRITE_RE = re.compile(
    r"(?P<call>std::env::set_var|env::set_var|std::env::remove_var|env::remove_var)\s*\("
)

HOT_DIRECT_ENV_READ_CLASSIFICATIONS = (
    {
        "path": "crates/ferrum-kernels/kernels/marlin_cuda_kernel.cu",
        "call": "std::getenv",
        "env_name": "FERRUM_MARLIN_TILE",
        "expected_count": 2,
        "classification": "diagnostic_tile_override",
        "read_phase": "request",
        "reason": (
            "legacy Marlin tile-tuning override; the CUDA vLLM-MoE hot path "
            "uses typed runtime config instead"
        ),
    },
    {
        "path": "crates/ferrum-kernels/kernels/vllm_marlin_moe/ops.cu",
        "call": "std::getenv",
        "line_contains": "return std::getenv(name);",
        "expected_count": 1,
        "classification": "process_static_native_config_helper",
        "read_phase": "startup",
        "reason": (
            "single helper backing process-static vLLM-MoE native diagnostic "
            "config and the backwards-compatible profile metadata fallback"
        ),
    },
    {
        "path": "crates/ferrum-kernels/src/backend/metal/q4_k_moe_id_gemv_batched.rs",
        "call": "std::env::var",
        "env_name": "MTL_CAPTURE_ENABLED",
        "expected_count": 1,
        "classification": "ignored_manual_metal_capture_test",
        "read_phase": "test-only",
        "reason": "ignored manual GPU capture test guard, not a product runtime Ferrum knob",
    },
)

PROCESS_ENV_WRITE_CLASSIFICATIONS = (
    {
        "path": "crates/ferrum-cli/src/runtime_env.rs",
        "call": "std::env::set_var",
        "expected_count": 2,
        "classification": "runtime_config_compatibility_bridge",
        "reason": (
            "centralized CLI bridge materializing typed RuntimeConfigEntry "
            "defaults/effective startup config for backend paths that still "
            "consume process env"
        ),
    },
    {
        "path": "crates/ferrum-bench-core/src/profile.rs",
        "call": "std::env::set_var",
        "expected_count": 6,
        "classification": "profile_sink_compatibility_bridge",
        "reason": (
            "backwards-compatible FERRUM_PROFILE_* process-env materialization; "
            "normal server profile wiring uses typed --profile-* arguments"
        ),
    },
    {
        "path": "crates/ferrum-bench-core/src/profile.rs",
        "call": "std::env::remove_var",
        "expected_count": 1,
        "classification": "profile_sink_compatibility_bridge",
        "reason": (
            "backwards-compatible FERRUM_PROFILE_* process-env cleanup; normal "
            "server profile wiring uses typed --profile-* arguments"
        ),
    },
)

REQUIRED_REGISTRY_FIELDS = (
    "name",
    "type",
    "default",
    "owner",
    "scope",
    "stability",
    "read_phase",
    "replacement",
    "sunset",
)

VALID_SCOPES = {"runtime", "benchmark", "debug", "build", "test"}
VALID_STABILITY = {"default", "experimental", "diagnostic", "deprecated"}
VALID_READ_PHASES = {"build", "startup", "request", "test-only"}
VALID_TYPES = {"bool", "enum", "float", "integer", "json", "path", "string", "tri-state", "url"}
SUNSET_REQUIRED_STABILITY = {"experimental", "diagnostic", "deprecated"}
EMPTY_SENTINELS = {"", "none", "unset", "n/a", "na"}

PRODUCT_TOML_RUNTIME_KEYS = {
    # Keep the checked-in sample config product-facing. Advanced M3/FA2/debug
    # selectors may be accepted by the schema for artifacts, but should not be
    # advertised as normal user settings in ferrum.toml.
    "preset",
    "kv_dtype",
    "kv_max_blocks",
    "paged_max_seqs",
    "max_batched_tokens",
    "prefix_cache",
    "moe_graph",
}

TOML_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
TOML_KEY_RE = re.compile(r"^\s*(#\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def rel_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def iter_source_files(roots: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        path = REPO_ROOT / root
        if not path.exists():
            continue
        if path.is_file():
            candidates = [path]
        else:
            candidates = [p for p in path.rglob("*") if p.is_file()]
        for candidate in candidates:
            rel = rel_path(candidate)
            if rel in EXCLUDED_REL_PATHS:
                continue
            if candidate.suffix not in SOURCE_SUFFIXES:
                continue
            files.append(candidate)
    return sorted(files)


def is_embedded_token(text: str, start: int, end: int) -> bool:
    prev = text[start - 1] if start > 0 else ""
    next_char = text[end] if end < len(text) else ""
    return (prev.isalnum() or prev == "_") or (next_char.isalnum() or next_char == "_")


def is_hot_path(path: Path) -> bool:
    rel = rel_path(path)
    return any(rel == root or rel.startswith(f"{root}/") for root in HOT_ROOTS)


def line_col_for_offset(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    line_start = text.rfind("\n", 0, offset) + 1
    return line, offset - line_start + 1


def source_line_for_offset(text: str, offset: int) -> str:
    line_start = text.rfind("\n", 0, offset) + 1
    line_end = text.find("\n", offset)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end].strip()


def is_test_env_write_context(rel: str, text: str, offset: int) -> bool:
    if "/tests/" in rel or rel.endswith("_test.rs") or rel.endswith("_tests.rs"):
        return True

    prefix = text[:offset]
    last_cfg_test = prefix.rfind("#[cfg(test)]")
    if last_cfg_test == -1:
        return False
    return "mod tests" in prefix[last_cfg_test:]


def parse_string_literal_argument(text: str, offset: int) -> str | None:
    i = offset
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] not in ('"', "'"):
        return None

    quote = text[i]
    i += 1
    value: list[str] = []
    while i < len(text):
        char = text[i]
        if char == "\\":
            if i + 1 >= len(text):
                return None
            value.append(text[i + 1])
            i += 2
            continue
        if char == quote:
            return "".join(value)
        value.append(char)
        i += 1
    return None


def direct_env_read_occurrence(rel: str, text: str, match: re.Match[str]) -> dict[str, Any]:
    line, column = line_col_for_offset(text, match.start())
    return {
        "path": rel,
        "line": line,
        "column": column,
        "call": match.group("call"),
        "env_name": parse_string_literal_argument(text, match.end()),
        "source": source_line_for_offset(text, match.start()),
    }


def process_env_write_occurrence(rel: str, text: str, match: re.Match[str]) -> dict[str, Any]:
    line, column = line_col_for_offset(text, match.start())
    return {
        "path": rel,
        "line": line,
        "column": column,
        "call": match.group("call"),
        "env_name": parse_string_literal_argument(text, match.end()),
        "source": source_line_for_offset(text, match.start()),
        "test_context": is_test_env_write_context(rel, text, match.start()),
    }


def classify_hot_direct_env_reads(
    occurrences: list[dict[str, Any]],
) -> dict[str, Any]:
    remaining = [
        {
            **classification,
            "remaining": int(classification.get("expected_count", 1)),
        }
        for classification in HOT_DIRECT_ENV_READ_CLASSIFICATIONS
    ]
    classified: list[dict[str, Any]] = []
    unclassified: list[dict[str, Any]] = []

    for occurrence in occurrences:
        matched_index: int | None = None
        for index, classification in enumerate(remaining):
            if classification["remaining"] <= 0:
                continue
            if occurrence["path"] != classification["path"]:
                continue
            if occurrence["call"] != classification["call"]:
                continue
            expected_env_name = classification.get("env_name")
            if expected_env_name is not None and occurrence.get("env_name") != expected_env_name:
                continue
            line_contains = classification.get("line_contains")
            if line_contains is not None and line_contains not in occurrence.get("source", ""):
                continue
            matched_index = index
            break

        if matched_index is None:
            unclassified.append(occurrence)
            continue

        classification = remaining[matched_index]
        classification["remaining"] -= 1
        classified.append(
            {
                **occurrence,
                "classification": classification["classification"],
                "read_phase": classification["read_phase"],
                "reason": classification["reason"],
            }
        )

    unused = [
        {
            key: value
            for key, value in classification.items()
            if key != "remaining"
        }
        | {"unused_count": classification["remaining"]}
        for classification in remaining
        if classification["remaining"] > 0
    ]
    return {
        "classified_count": len(classified),
        "unclassified_count": len(unclassified),
        "classified": classified,
        "unclassified": unclassified,
        "unused_classifications": unused,
    }


def classify_process_env_writes(
    occurrences: list[dict[str, Any]],
) -> dict[str, Any]:
    remaining = [
        {
            **classification,
            "remaining": int(classification.get("expected_count", 1)),
        }
        for classification in PROCESS_ENV_WRITE_CLASSIFICATIONS
    ]
    classified: list[dict[str, Any]] = []
    unclassified: list[dict[str, Any]] = []

    for occurrence in occurrences:
        if occurrence.get("test_context"):
            classified.append(
                {
                    **occurrence,
                    "classification": "test_only_env_mutation",
                    "reason": "test fixture setup/cleanup outside product runtime paths",
                }
            )
            continue

        matched_index: int | None = None
        for index, classification in enumerate(remaining):
            if classification["remaining"] <= 0:
                continue
            if occurrence["path"] != classification["path"]:
                continue
            if occurrence["call"] != classification["call"]:
                continue
            matched_index = index
            break

        if matched_index is None:
            unclassified.append(occurrence)
            continue

        classification = remaining[matched_index]
        classification["remaining"] -= 1
        classified.append(
            {
                **occurrence,
                "classification": classification["classification"],
                "reason": classification["reason"],
            }
        )

    unused = [
        {
            key: value
            for key, value in classification.items()
            if key != "remaining"
        }
        | {"unused_count": classification["remaining"]}
        for classification in remaining
        if classification["remaining"] > 0
    ]
    return {
        "classified_count": len(classified),
        "unclassified_count": len(unclassified),
        "classified": classified,
        "unclassified": unclassified,
        "unused_classifications": unused,
    }


def scan_sources(roots: tuple[str, ...]) -> dict[str, Any]:
    token_paths: dict[str, set[str]] = defaultdict(set)
    embedded_token_paths: dict[str, set[str]] = defaultdict(set)
    hot_token_paths: dict[str, set[str]] = defaultdict(set)
    direct_reads_by_file: Counter[str] = Counter()
    hot_direct_reads_by_file: Counter[str] = Counter()
    process_env_writes_by_file: Counter[str] = Counter()
    token_count_by_file: Counter[str] = Counter()
    hot_direct_occurrences: list[dict[str, Any]] = []
    process_env_write_occurrences: list[dict[str, Any]] = []

    files = iter_source_files(roots)
    for path in files:
        rel = rel_path(path)
        try:
            text = path.read_text(errors="ignore")
        except OSError as exc:
            print(f"warning: could not read {rel}: {exc}", file=sys.stderr)
            continue

        for match in FERRUM_TOKEN_RE.finditer(text):
            token = match.group(0)
            token_paths[token].add(rel)
            token_count_by_file[rel] += 1
            if is_embedded_token(text, match.start(), match.end()):
                embedded_token_paths[token].add(rel)
            if is_hot_path(path):
                hot_token_paths[token].add(rel)

        direct_read_matches = list(DIRECT_ENV_READ_RE.finditer(text))
        if direct_read_matches:
            direct_reads_by_file[rel] = len(direct_read_matches)
            if is_hot_path(path):
                hot_direct_reads_by_file[rel] = len(direct_read_matches)
                hot_direct_occurrences.extend(
                    direct_env_read_occurrence(rel, text, match)
                    for match in direct_read_matches
                )

        process_env_write_matches = list(PROCESS_ENV_WRITE_RE.finditer(text))
        if process_env_write_matches:
            process_env_writes_by_file[rel] = len(process_env_write_matches)
            process_env_write_occurrences.extend(
                process_env_write_occurrence(rel, text, match)
                for match in process_env_write_matches
            )

    unique_tokens = sorted(token_paths)
    hot_tokens = sorted(hot_token_paths)
    embedded_tokens = sorted(embedded_token_paths)
    standalone_tokens = sorted(set(unique_tokens) - set(embedded_tokens))
    hot_direct_classification = classify_hot_direct_env_reads(hot_direct_occurrences)
    process_env_write_classification = classify_process_env_writes(
        process_env_write_occurrences
    )
    test_process_env_writes = sum(
        1 for occurrence in process_env_write_occurrences if occurrence["test_context"]
    )
    non_test_process_env_writes = len(process_env_write_occurrences) - test_process_env_writes
    non_test_process_env_writes_classified = sum(
        1
        for occurrence in process_env_write_classification["classified"]
        if not occurrence.get("test_context")
    )

    return {
        "scan_roots": list(roots),
        "files_scanned": len(files),
        "unique_ferrum_tokens": len(unique_tokens),
        "unique_ferrum_env_candidates": len(standalone_tokens),
        "embedded_ferrum_tokens": embedded_tokens,
        "direct_env_reads": sum(direct_reads_by_file.values()),
        "process_env_writes": sum(process_env_writes_by_file.values()),
        "test_process_env_writes": test_process_env_writes,
        "non_test_process_env_writes": non_test_process_env_writes,
        "process_env_writes_classified": process_env_write_classification["classified_count"],
        "non_test_process_env_writes_classified": non_test_process_env_writes_classified,
        "process_env_writes_unclassified": process_env_write_classification[
            "unclassified_count"
        ],
        "non_test_process_env_writes_unclassified": process_env_write_classification[
            "unclassified_count"
        ],
        "hot_unique_ferrum_tokens": len(hot_tokens),
        "hot_direct_env_reads": sum(hot_direct_reads_by_file.values()),
        "hot_direct_env_reads_classified": hot_direct_classification["classified_count"],
        "hot_direct_env_reads_unclassified": hot_direct_classification["unclassified_count"],
        "candidate_names": standalone_tokens,
        "unique_names": unique_tokens,
        "hot_unique_names": hot_tokens,
        "name_paths": {name: sorted(paths) for name, paths in sorted(token_paths.items())},
        "top_direct_env_read_files": top_counter(direct_reads_by_file),
        "top_process_env_write_files": top_counter(process_env_writes_by_file),
        "hot_top_direct_env_read_files": top_counter(hot_direct_reads_by_file),
        "hot_direct_env_read_classification": hot_direct_classification,
        "process_env_write_classification": process_env_write_classification,
        "top_ferrum_token_files": top_counter(token_count_by_file),
    }


def top_counter(counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"path": path, "count": count}
        for path, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": rel_path(path),
            "exists": False,
            "entries": 0,
            "names": [],
            "errors": [f"registry not found: {rel_path(path)}"],
        }

    errors: list[str] = []
    names: list[str] = []
    seen: set[str] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        for field in REQUIRED_REGISTRY_FIELDS:
            if field not in fieldnames:
                errors.append(f"missing required registry column: {field}")

        for line_no, row in enumerate(reader, start=2):
            name = (row.get("name") or "").strip()
            if not name:
                errors.append(f"line {line_no}: missing name")
                continue
            if name in seen:
                errors.append(f"line {line_no}: duplicate registry entry {name}")
                continue
            seen.add(name)
            names.append(name)

            for field in REQUIRED_REGISTRY_FIELDS:
                if not (row.get(field) or "").strip():
                    errors.append(f"line {line_no}: {name} has empty {field}")

            value_type = (row.get("type") or "").strip()
            if value_type and value_type not in VALID_TYPES:
                errors.append(f"line {line_no}: {name} has invalid type {value_type!r}")

            scope = (row.get("scope") or "").strip()
            if scope and scope not in VALID_SCOPES:
                errors.append(f"line {line_no}: {name} has invalid scope {scope!r}")

            stability = (row.get("stability") or "").strip()
            if stability and stability not in VALID_STABILITY:
                errors.append(f"line {line_no}: {name} has invalid stability {stability!r}")
            if stability in SUNSET_REQUIRED_STABILITY:
                sunset = (row.get("sunset") or "").strip()
                if sunset.lower() in EMPTY_SENTINELS:
                    errors.append(
                        f"line {line_no}: {name} has {stability} stability but no sunset condition"
                    )

            read_phase = (row.get("read_phase") or "").strip()
            if read_phase and read_phase not in VALID_READ_PHASES:
                errors.append(f"line {line_no}: {name} has invalid read_phase {read_phase!r}")

    return {
        "path": rel_path(path),
        "exists": True,
        "entries": len(names),
        "names": sorted(names),
        "errors": errors,
    }


def load_name_baseline(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    names: list[str] = []
    seen: set[str] = set()
    if not path.exists():
        return {
            "path": rel_path(path),
            "exists": False,
            "entries": 0,
            "names": [],
            "errors": [f"missing baseline not found: {rel_path(path)}"],
        }

    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if not FERRUM_TOKEN_RE.fullmatch(line):
            errors.append(f"line {line_no}: invalid Ferrum env name {line!r}")
            continue
        if line in seen:
            errors.append(f"line {line_no}: duplicate baseline entry {line}")
            continue
        seen.add(line)
        names.append(line)

    return {
        "path": rel_path(path),
        "exists": True,
        "entries": len(names),
        "names": sorted(names),
        "errors": errors,
    }


def audit_product_toml_surface(path: Path) -> dict[str, Any]:
    """Keep checked-in ferrum.toml from becoming an advanced selector catalog."""
    errors: list[str] = []
    runtime_keys: list[str] = []
    ferrum_tokens: list[str] = []
    if not path.exists():
        return {
            "path": rel_path(path),
            "exists": False,
            "runtime_keys": [],
            "allowed_runtime_keys": sorted(PRODUCT_TOML_RUNTIME_KEYS),
            "ferrum_tokens": [],
            "errors": [f"product config not found: {rel_path(path)}"],
        }

    current_section: str | None = None
    text = path.read_text(errors="ignore")
    for line_no, line in enumerate(text.splitlines(), start=1):
        for token in FERRUM_TOKEN_RE.findall(line):
            ferrum_tokens.append(token)
            errors.append(
                f"{rel_path(path)}:{line_no}: checked-in product config must not mention {token}"
            )

        section_match = TOML_SECTION_RE.match(line)
        if section_match:
            current_section = section_match.group(1).strip()
            continue
        if current_section != "runtime":
            continue

        key_match = TOML_KEY_RE.match(line)
        if not key_match:
            continue
        key = key_match.group(2)
        runtime_keys.append(key)
        if key not in PRODUCT_TOML_RUNTIME_KEYS:
            errors.append(
                f"{rel_path(path)}:{line_no}: [runtime].{key} is not a product-facing sample key"
            )

    return {
        "path": rel_path(path),
        "exists": True,
        "runtime_keys": sorted(set(runtime_keys)),
        "allowed_runtime_keys": sorted(PRODUCT_TOML_RUNTIME_KEYS),
        "ferrum_tokens": sorted(set(ferrum_tokens)),
        "errors": errors,
    }


def write_registry_fixture(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "name",
        "type",
        "default",
        "owner",
        "scope",
        "stability",
        "read_phase",
        "replacement",
        "sunset",
        "notes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def registry_fixture_row(**updates: str) -> dict[str, str]:
    row = {
        "name": "FERRUM_TEST_FLAG",
        "type": "bool",
        "default": "0",
        "owner": "tests",
        "scope": "runtime",
        "stability": "default",
        "read_phase": "startup",
        "replacement": "runtime_config.test.flag",
        "sunset": "none",
        "notes": "fixture",
    }
    row.update(updates)
    return row


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        registry_path = root / "registry.tsv"

        write_registry_fixture(registry_path, [registry_fixture_row()])
        valid = load_registry(registry_path)
        assert not valid["errors"], valid["errors"]

        write_registry_fixture(registry_path, [registry_fixture_row(type="flag")])
        invalid_type = load_registry(registry_path)
        assert any("invalid type" in error for error in invalid_type["errors"]), invalid_type[
            "errors"
        ]

        write_registry_fixture(
            registry_path,
            [registry_fixture_row(stability="diagnostic", sunset="none")],
        )
        missing_sunset = load_registry(registry_path)
        assert any(
            "no sunset condition" in error for error in missing_sunset["errors"]
        ), missing_sunset["errors"]

        write_registry_fixture(registry_path, [registry_fixture_row(scope="global")])
        invalid_scope = load_registry(registry_path)
        assert any("invalid scope" in error for error in invalid_scope["errors"]), invalid_scope[
            "errors"
        ]

        product_toml = root / "ferrum.toml"
        product_toml.write_text(
            "\n".join(
                [
                    "[runtime]",
                    '# preset = "m3_qwen3_30b_a3b_int4"',
                    '# kv_dtype = "fp16"',
                    "# moe_graph = true",
                ]
            )
        )
        product_ok = audit_product_toml_surface(product_toml)
        assert not product_ok["errors"], product_ok["errors"]

        product_toml.write_text(
            "\n".join(
                [
                    "[runtime]",
                    "# fa2_source = true",
                    "# FERRUM_FA2_SOURCE=1",
                ]
            )
        )
        product_bad = audit_product_toml_surface(product_toml)
        assert any("fa2_source" in error for error in product_bad["errors"]), product_bad[
            "errors"
        ]
        assert any("FERRUM_FA2_SOURCE" in error for error in product_bad["errors"]), product_bad[
            "errors"
        ]

        product_write_text = "fn main() { std::env::set_var(\"FERRUM_TEST_FLAG\", \"1\"); }"
        product_write = [
            process_env_write_occurrence("crates/ferrum-cli/src/main.rs", product_write_text, match)
            for match in PROCESS_ENV_WRITE_RE.finditer(product_write_text)
        ]
        product_classification = classify_process_env_writes(product_write)
        assert product_classification["unclassified_count"] == 1, product_classification

        test_write_text = "\n".join(
            [
                "#[cfg(test)]",
                "mod tests {",
                "    fn env_fixture() {",
                "        std::env::set_var(\"FERRUM_TEST_FLAG\", \"1\");",
                "    }",
                "}",
            ]
        )
        test_write = [
            process_env_write_occurrence(
                "crates/ferrum-cli/src/some_module.rs", test_write_text, match
            )
            for match in PROCESS_ENV_WRITE_RE.finditer(test_write_text)
        ]
        test_classification = classify_process_env_writes(test_write)
        assert test_classification["unclassified_count"] == 0, test_classification
        assert test_classification["classified"][0]["classification"] == "test_only_env_mutation"

        bridge_write_text = "fn bridge() { std::env::set_var(&entry.key, &entry.effective_value); }"
        bridge_write = [
            process_env_write_occurrence(
                "crates/ferrum-cli/src/runtime_env.rs", bridge_write_text, match
            )
            for match in PROCESS_ENV_WRITE_RE.finditer(bridge_write_text)
        ]
        bridge_classification = classify_process_env_writes(bridge_write)
        assert bridge_classification["unclassified_count"] == 0, bridge_classification
        assert (
            bridge_classification["classified"][0]["classification"]
            == "runtime_config_compatibility_bridge"
        )

        threshold_report = {
            "direct_env_reads": 3,
            "process_env_writes": 1,
            "non_test_process_env_writes": 1,
            "hot_direct_env_reads": 0,
        }
        assert not threshold_errors(
            threshold_report,
            {
                "direct_env_reads": 3,
                "process_env_writes": 1,
                "non_test_process_env_writes": 1,
                "hot_direct_env_reads": 0,
            },
        )
        assert threshold_errors(
            threshold_report,
            {
                "direct_env_reads": 2,
                "process_env_writes": None,
                "non_test_process_env_writes": 1,
                "hot_direct_env_reads": 0,
            },
        ) == ["direct_env_reads=3 exceeds limit 2"]
        assert threshold_errors(
            threshold_report,
            {
                "direct_env_reads": 3,
                "process_env_writes": 1,
                "non_test_process_env_writes": 0,
                "hot_direct_env_reads": 0,
            },
        ) == ["non_test_process_env_writes=1 exceeds limit 0"]

    print("check_ferrum_env_registry self-test ok")


def compare_registry(
    scan: dict[str, Any], registry: dict[str, Any], ignored_names: set[str]
) -> dict[str, Any]:
    scanned = set(scan["candidate_names"]) - ignored_names
    registered = set(registry["names"])
    covered = sorted(scanned & registered)
    missing = sorted(scanned - registered)
    extra = sorted(registered - scanned)
    return {
        "coverage": len(covered),
        "scan_count": len(scanned),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "coverage_percent": round((len(covered) / len(scanned) * 100.0), 2) if scanned else 100.0,
        "missing_names": missing,
        "extra_names": extra,
        "errors": registry["errors"],
        "ignored_count": len(ignored_names),
        "ignored_names": sorted(ignored_names),
    }


def compare_missing_baseline(
    comparison: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    missing = set(comparison["missing_names"])
    baseline_names = set(baseline["names"])
    new_missing = sorted(missing - baseline_names)
    resolved_missing = sorted(baseline_names - missing)
    still_missing = sorted(missing & baseline_names)
    return {
        "path": baseline["path"],
        "entries": baseline["entries"],
        "errors": baseline["errors"],
        "new_missing_count": len(new_missing),
        "resolved_missing_count": len(resolved_missing),
        "still_missing_count": len(still_missing),
        "new_missing_names": new_missing,
        "resolved_missing_names": resolved_missing,
    }


def threshold_errors(report: dict[str, Any], limits: dict[str, int | None]) -> list[str]:
    errors = []
    for key, limit in limits.items():
        if limit is None:
            continue
        actual = report[key]
        if actual > limit:
            errors.append(f"{key}={actual} exceeds limit {limit}")
    return errors


def render_human(report: dict[str, Any]) -> str:
    lines = [
        "Ferrum env registry audit",
        f"  scan_roots: {', '.join(report['scan_roots'])}",
        f"  files_scanned: {report['files_scanned']}",
        f"  unique_ferrum_tokens: {report['unique_ferrum_tokens']}",
        f"  unique_ferrum_env_candidates: {report['unique_ferrum_env_candidates']}",
        f"  direct_env_reads: {report['direct_env_reads']}",
        f"  process_env_writes: {report['process_env_writes']}",
        f"  test_process_env_writes: {report['test_process_env_writes']}",
        f"  non_test_process_env_writes: {report['non_test_process_env_writes']}",
        f"  process_env_writes_classified: {report['process_env_writes_classified']}",
        f"  non_test_process_env_writes_classified: {report['non_test_process_env_writes_classified']}",
        f"  process_env_writes_unclassified: {report['process_env_writes_unclassified']}",
        f"  non_test_process_env_writes_unclassified: {report['non_test_process_env_writes_unclassified']}",
        f"  hot_unique_ferrum_tokens: {report['hot_unique_ferrum_tokens']}",
        f"  hot_direct_env_reads: {report['hot_direct_env_reads']}",
        f"  hot_direct_env_reads_classified: {report['hot_direct_env_reads_classified']}",
        f"  hot_direct_env_reads_unclassified: {report['hot_direct_env_reads_unclassified']}",
    ]
    if report["embedded_ferrum_tokens"]:
        lines.append(f"  embedded_ferrum_tokens: {', '.join(report['embedded_ferrum_tokens'])}")
    classification = report.get("hot_direct_env_read_classification")
    if classification and classification["unclassified"]:
        lines.append("  unclassified_hot_direct_env_reads:")
        lines.extend(
            "    - {path}:{line}: {call}({env})".format(
                path=occurrence["path"],
                line=occurrence["line"],
                call=occurrence["call"],
                env=occurrence["env_name"] or "?",
            )
            for occurrence in classification["unclassified"]
        )
    write_classification = report.get("process_env_write_classification")
    if write_classification and write_classification["unclassified"]:
        lines.append("  unclassified_process_env_writes:")
        lines.extend(
            "    - {path}:{line}: {call}({env})".format(
                path=occurrence["path"],
                line=occurrence["line"],
                call=occurrence["call"],
                env=occurrence["env_name"] or "?",
            )
            for occurrence in write_classification["unclassified"]
        )

    registry = report.get("registry")
    if registry:
        comparison = registry["comparison"]
        lines.extend(
            [
                f"  registry: {registry['path']}",
                f"  registry_entries: {registry['entries']}",
                f"  registry_coverage: {comparison['coverage']}/{comparison['scan_count']} ({comparison['coverage_percent']}%)",
                f"  registry_missing: {comparison['missing_count']}",
                f"  registry_extra: {comparison['extra_count']}",
                f"  ignored_non_env_names: {comparison['ignored_count']}",
            ]
        )
        if comparison["errors"]:
            lines.append("  registry_errors:")
            lines.extend(f"    - {error}" for error in comparison["errors"])
        missing_baseline = registry.get("missing_baseline")
        if missing_baseline:
            lines.extend(
                [
                    f"  missing_baseline: {missing_baseline['path']}",
                    f"  missing_baseline_entries: {missing_baseline['entries']}",
                    f"  new_unregistered_names: {missing_baseline['new_missing_count']}",
                    f"  resolved_baseline_names: {missing_baseline['resolved_missing_count']}",
                ]
            )
            if missing_baseline["errors"]:
                lines.append("  missing_baseline_errors:")
                lines.extend(f"    - {error}" for error in missing_baseline["errors"])
            if missing_baseline["new_missing_names"]:
                lines.append("  new_unregistered_name_list:")
                lines.extend(f"    - {name}" for name in missing_baseline["new_missing_names"])
    product = report.get("product_config_surface")
    if product:
        lines.extend(
            [
                f"  product_config: {product['path']}",
                f"  product_runtime_keys: {', '.join(product['runtime_keys']) or 'none'}",
                f"  product_ferrum_tokens: {', '.join(product['ferrum_tokens']) or 'none'}",
                f"  product_config_errors: {len(product['errors'])}",
            ]
        )
        if product["errors"]:
            lines.extend(f"    - {error}" for error in product["errors"])
    if report.get("threshold_errors"):
        lines.append("  threshold_errors:")
        lines.extend(f"    - {error}" for error in report["threshold_errors"])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="run checker self-tests and exit")
    parser.add_argument(
        "--registry",
        default="docs/runtime-env-registry.tsv",
        help="TSV registry path to compare against, relative to repo root by default",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        dest="scan_roots",
        help="Override scan root; may be passed more than once",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--fail-on-registry-gap",
        action="store_true",
        help="exit non-zero if the registry is invalid, missing scanned names, or has stale names",
    )
    parser.add_argument(
        "--missing-baseline",
        default=None,
        help=(
            "newline file of currently known unregistered FERRUM_* names; "
            f"defaults to {DEFAULT_MISSING_BASELINE} when --fail-on-new-missing is set"
        ),
    )
    parser.add_argument(
        "--ignore-names",
        default=DEFAULT_IGNORED_NAMES,
        help="newline file of scanned FERRUM_* names that are not environment knobs",
    )
    parser.add_argument(
        "--fail-on-new-missing",
        action="store_true",
        help="exit non-zero only for registry/schema errors or unregistered names not in the missing baseline",
    )
    parser.add_argument(
        "--fail-on-baseline-drift",
        action="store_true",
        help="with --missing-baseline, also fail if the baseline still lists names that are now registered or gone",
    )
    parser.add_argument(
        "--max-direct-env-reads",
        type=int,
        default=None,
        help="fail if total direct env read calls exceed this value",
    )
    parser.add_argument(
        "--max-process-env-writes",
        type=int,
        default=None,
        help="fail if total process env write calls exceed this value",
    )
    parser.add_argument(
        "--max-non-test-process-env-writes",
        type=int,
        default=None,
        help="fail if non-test process env write calls exceed this value",
    )
    parser.add_argument(
        "--max-hot-direct-env-reads",
        type=int,
        default=None,
        help="fail if hot-path direct env read calls exceed this value",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    roots = tuple(args.scan_roots) if args.scan_roots else DEFAULT_SCAN_ROOTS
    scan = scan_sources(roots)

    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = REPO_ROOT / registry_path
    registry = load_registry(registry_path)

    ignore_path = Path(args.ignore_names)
    if not ignore_path.is_absolute():
        ignore_path = REPO_ROOT / ignore_path
    ignored = load_name_baseline(ignore_path)
    ignored_names = set(ignored["names"])

    comparison = compare_registry(scan, registry, ignored_names)
    registry_report = dict(registry)
    registry_report["comparison"] = comparison
    registry_report["ignored_names"] = ignored

    missing_baseline_report = None
    if args.missing_baseline or args.fail_on_new_missing or args.fail_on_baseline_drift:
        missing_baseline_path = Path(args.missing_baseline or DEFAULT_MISSING_BASELINE)
        if not missing_baseline_path.is_absolute():
            missing_baseline_path = REPO_ROOT / missing_baseline_path
        missing_baseline = load_name_baseline(missing_baseline_path)
        missing_baseline_report = compare_missing_baseline(comparison, missing_baseline)
        registry_report["missing_baseline"] = missing_baseline_report

    scan["registry"] = registry_report
    scan["product_config_surface"] = audit_product_toml_surface(REPO_ROOT / "ferrum.toml")
    scan["threshold_errors"] = threshold_errors(
        scan,
        {
            "direct_env_reads": args.max_direct_env_reads,
            "process_env_writes": args.max_process_env_writes,
            "non_test_process_env_writes": args.max_non_test_process_env_writes,
            "hot_direct_env_reads": args.max_hot_direct_env_reads,
        },
    )

    if args.json:
        print(json.dumps(scan, indent=2, sort_keys=True))
    else:
        print(render_human(scan))

    if args.fail_on_registry_gap and (
        ignored["errors"]
        or comparison["errors"]
        or comparison["missing_count"]
        or comparison["extra_count"]
        or scan["hot_direct_env_reads_unclassified"]
        or scan["process_env_writes_unclassified"]
        or scan["product_config_surface"]["errors"]
        or scan["threshold_errors"]
    ):
        return 1
    if args.fail_on_new_missing and missing_baseline_report is not None:
        if (
            ignored["errors"]
            or comparison["errors"]
            or missing_baseline_report["errors"]
            or missing_baseline_report["new_missing_count"]
        ):
            return 1
    if args.fail_on_baseline_drift and missing_baseline_report is not None:
        if (
            missing_baseline_report["errors"]
            or missing_baseline_report["new_missing_count"]
            or missing_baseline_report["resolved_missing_count"]
        ):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
