#!/usr/bin/env python3
"""Build the Runtime vNext source and coupling inventory.

The inventory deliberately scans all of ``scripts/``.  That is a conservative
superset of the product and release scripts named by G00 and prevents a script
from escaping the baseline merely because it was moved out of
``scripts/release``.

File identity is its SHA256, never its path.  Passing a previous inventory with
``--baseline`` therefore makes moves and copies visible without reclassifying
the content solely from its new path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = ("crates", "scripts")
CLASSIFICATIONS = (
    "production",
    "test",
    "generated",
    "vendor",
    "example",
    "fixture",
)
SOURCE_DISCOVERY_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "target",
}
LOC_LANGUAGES = {
    "rust",
    "c",
    "cpp",
    "cuda",
    "cuda-header",
    "c-header",
    "cpp-header",
    "objective-c",
    "objective-cpp",
    "metal",
    "python",
    "shell",
    "makefile",
    "dockerfile",
}

LARGE_NATIVE_LOC = 10_000
LARGE_NATIVE_BYTES = 5 * 1024 * 1024
LARGE_NATIVE_TRANSLATION_UNITS = 10

NATIVE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".cuh",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".m",
    ".metal",
    ".mm",
}
TRANSLATION_UNIT_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", ".m", ".mm"}

LANGUAGE_BY_SUFFIX = {
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cu": "cuda",
    ".cuh": "cuda-header",
    ".h": "c-header",
    ".hh": "cpp-header",
    ".hpp": "cpp-header",
    ".hxx": "cpp-header",
    ".json": "json",
    ".jsonl": "jsonl",
    ".m": "objective-c",
    ".md": "markdown",
    ".metal": "metal",
    ".mm": "objective-cpp",
    ".py": "python",
    ".pyc": "python-bytecode",
    ".rs": "rust",
    ".sh": "shell",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
}
LANGUAGE_BY_NAME = {
    "Cargo.lock": "toml",
    "Cargo.toml": "toml",
    "Dockerfile": "dockerfile",
    "Makefile": "makefile",
}

FIXTURE_PARTS = {"fixture", "fixtures", "testdata", "test-data", "snapshots"}
EXAMPLE_PARTS = {"example", "examples"}
GENERATED_PARTS = {"generated", "gen", "autogen", "bindings"}
TEST_PARTS = {"test", "tests", "benches"}
VENDOR_PARTS = {
    "vendor",
    "vendored",
    "third_party",
    "third-party",
    "external",
    "upstream",
}

ARCH_TOKEN_RE = re.compile(
    r"\b(?:qwen(?:_?3(?:_?5)?|35|2)?|llama|gemma|mistral|mixtral|deepseek|"
    r"phi(?:3|4)?|glm|starcoder|smollm)[A-Za-z0-9_]*\b",
    re.IGNORECASE,
)
QWEN35_RE = re.compile(
    r"\b(?=[A-Za-z_][A-Za-z0-9_]*\b)"
    r"(?=[A-Za-z0-9_]*(?:qwen_?3_?5|qwen35))"
    r"[A-Za-z_][A-Za-z0-9_]*\b",
    re.IGNORECASE,
)
DECL_RE = re.compile(
    r"\b(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?"
    r"(fn|struct|enum|trait|type|const|static)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
FACTORY_DECL_RE = re.compile(
    r"\b(?:fn|struct|enum|trait|type)\s+([A-Za-z_][A-Za-z0-9_]*factory[A-Za-z0-9_]*)\b",
    re.IGNORECASE,
)
RUNNER_DECL_RE = re.compile(
    r"\b(?:fn|struct|enum|trait|type)\s+([A-Za-z_][A-Za-z0-9_]*runner[A-Za-z0-9_]*)\b",
    re.IGNORECASE,
)
MODEL_RUN_FUNCTION_RE = re.compile(
    r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*(?:decode_loop|prefill|decode|generate|run_batch|run_unified|runner)[A-Za-z0-9_]*)\b",
    re.IGNORECASE,
)
SCAFFOLDING_RE = re.compile(
    r"(?:setup|admission|batch|orchestrat|state_(?:transition|update)|finalize|cleanup|"
    r"backend_dispatch|decode_loop|run_unified|run_batch)",
    re.IGNORECASE,
)
BACKEND_TRAIT_RE = re.compile(r"\btrait\s+Backend\b")
FUNCTION_RE = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)")
CFG_TEST_RE = re.compile(r"#\s*\[\s*cfg\s*\([^\]]*\btest\b[^\]]*\)\s*\]")
MODULE_RE = re.compile(r"\bmod\s+[A-Za-z_][A-Za-z0-9_]*")
BACKEND_CFG_RE = re.compile(
    r"(?:#\s*\[\s*cfg(?:_attr)?\s*\([^\]]*\b(?:cuda|metal)\b[^\]]*\)\s*\]"
    r"|\bcfg!\s*\([^;\n]*(?:cuda|metal)[^;\n]*\)"
    r"|^\s*#\s*(?:if|ifdef|ifndef|elif)[^\n]*(?:CUDA|METAL))",
    re.IGNORECASE,
)
ENV_CALL_RE = re.compile(
    r"(?P<reader>std::env::(?:var|var_os)|env::(?:var|var_os)|std::getenv|getenv|"
    r"runtime_config_value|runtime_snapshot_value|ferrum_env_value|"
    r"os\.getenv|os\.environ\.get|option_env!|env!)\s*\("
    r"(?P<args>[^)]{0,240}?)"
    r"[\"'](?P<key>FERRUM_[A-Z0-9_]+)[\"']",
    re.DOTALL,
)
PYTHON_ENV_INDEX_RE = re.compile(
    r"(?P<reader>os\.environ)\s*\[\s*[\"'](?P<key>FERRUM_[A-Z0-9_]+)[\"']\s*\]"
)
SHELL_ENV_RE = re.compile(r"(?P<reader>shell-expansion)\$\{?(?P<key>FERRUM_[A-Z0-9_]+)\}?")
GENERIC_ENV_RE = re.compile(
    r"\b(?P<reader>std::env::(?:vars|vars_os))\s*\("
)
ENV_CONSTANT_RE = re.compile(
    r"\b(?:const|static)\s+(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)[^=;\n]*="
    r"\s*[\"'](?P<key>FERRUM_[A-Z0-9_]+)[\"']"
)
ENV_IDENTIFIER_CALL_RE = re.compile(
    r"(?P<reader>std::env::(?:var|var_os)|env::(?:var|var_os)|std::getenv|getenv|"
    r"runtime_config_value|runtime_snapshot_value|ferrum_env_value|"
    r"os\.getenv|os\.environ\.get|option_env!|env!)\s*\("
    r"(?:[^,()]+,\s*)?&?\s*(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)\s*\)"
)
PRODUCT_DECISION_RE = re.compile(
    r"(?:resolve|source|alias|config|preset|capabil|runtime|template|token|model|backend|dtype|device)",
    re.IGNORECASE,
)


class InventoryError(Exception):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def language_for(path: Path, data: bytes) -> str:
    if path.name in LANGUAGE_BY_NAME:
        return LANGUAGE_BY_NAME[path.name]
    language = LANGUAGE_BY_SUFFIX.get(path.suffix.lower())
    if language:
        return language
    first_line = data.splitlines()[0] if data.splitlines() else b""
    if first_line.startswith(b"#!"):
        if b"python" in first_line:
            return "python"
        if b"sh" in first_line or b"bash" in first_line or b"zsh" in first_line:
            return "shell"
    return "binary" if b"\0" in data[:8192] else "text"


def classify_path(rel: str) -> str:
    path = Path(rel)
    parts = {part.lower() for part in path.parts}
    directory_parts = {part.lower() for part in path.parts[:-1]}
    name = path.name.lower()
    if parts & FIXTURE_PARTS:
        return "fixture"
    if parts & EXAMPLE_PARTS:
        return "example"
    if path.suffix.lower() == ".pyc" or parts & GENERATED_PARTS or name.endswith(".generated.rs"):
        return "generated"
    if parts & TEST_PARTS or re.search(r"(?:^|[_\-.])tests?(?:[_\-.]|$)", name):
        return "test"
    if directory_parts & VENDOR_PARTS or any(
        part.startswith(("vllm_", "flash_attn")) for part in directory_parts
    ):
        return "vendor"
    return "production"


def strip_c_like_comments(lines: list[str]) -> list[str]:
    """Remove C/Rust comments while preserving strings and line numbers."""

    output: list[str] = []
    block_depth = 0
    in_string = False
    escaped = False
    for line in lines:
        chars: list[str] = []
        idx = 0
        while idx < len(line):
            pair = line[idx : idx + 2]
            char = line[idx]
            if block_depth:
                if pair == "/*":
                    block_depth += 1
                    idx += 2
                elif pair == "*/":
                    block_depth -= 1
                    idx += 2
                else:
                    idx += 1
                continue
            if in_string:
                chars.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                idx += 1
                continue
            if pair == "//":
                break
            if pair == "/*":
                block_depth = 1
                idx += 2
                continue
            chars.append(char)
            if char == '"':
                in_string = True
            idx += 1
        output.append("".join(chars))
    return output


def code_lines_for(language: str, text: str) -> list[str]:
    lines = text.splitlines()
    if language in {
        "rust",
        "c",
        "cpp",
        "cuda",
        "cuda-header",
        "c-header",
        "cpp-header",
        "objective-c",
        "objective-cpp",
        "metal",
    }:
        return strip_c_like_comments(lines)
    if language in {"python", "shell", "toml", "yaml", "makefile", "dockerfile"}:
        return ["" if line.lstrip().startswith("#") and not line.startswith("#!") else line for line in lines]
    return lines


def mask_strings(code: str) -> str:
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', code)


def rust_line_classifications(code_lines: list[str], base: str) -> list[str]:
    if base != "production":
        return [base] * len(code_lines)

    classes = [base] * len(code_lines)
    depth = 0
    active_test_bases: list[int] = []
    pending_test_cfg = False
    for idx, code in enumerate(code_lines):
        masked = mask_strings(code)
        active_before = bool(active_test_bases)
        if CFG_TEST_RE.search(masked):
            pending_test_cfg = True
        if active_before or pending_test_cfg:
            classes[idx] = "test"

        opens = masked.count("{")
        closes = masked.count("}")
        if pending_test_cfg and MODULE_RE.search(masked):
            if opens:
                active_test_bases.append(depth)
            pending_test_cfg = False
        elif pending_test_cfg and masked.strip() and not masked.lstrip().startswith("#"):
            # G00 only reclassifies cfg(test) modules and their descendants.
            pending_test_cfg = False

        depth += opens - closes
        while active_test_bases and depth <= active_test_bases[-1]:
            active_test_bases.pop()
    return classes


def logical_loc(
    language: str, text: str, base_classification: str
) -> tuple[int, dict[str, int], list[str], list[str]]:
    code_lines = code_lines_for(language, text)
    if language not in LOC_LANGUAGES:
        return 0, {}, code_lines, [base_classification] * len(code_lines)
    if language == "rust":
        line_classes = rust_line_classifications(code_lines, base_classification)
    else:
        line_classes = [base_classification] * len(code_lines)
    counts: Counter[str] = Counter()
    for code, classification in zip(code_lines, line_classes):
        if code.strip():
            counts[classification] += 1
    return sum(counts.values()), dict(sorted(counts.items())), code_lines, line_classes


def finding(
    category: str,
    rel: str,
    line_no: int,
    line_classification: str,
    text: str,
    **extra: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "category": category,
        "path": rel,
        "line": line_no,
        "line_classification": line_classification,
        "text": text.strip()[:300],
    }
    result.update(extra)
    return result


def backend_trait_methods(
    rel: str, original_lines: list[str], code_lines: list[str], line_classes: list[str]
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    depth = 0
    pending_base: int | None = None
    active_base: int | None = None
    for idx, code in enumerate(code_lines):
        masked = mask_strings(code)
        depth_before = depth
        if BACKEND_TRAIT_RE.search(masked):
            pending_base = depth_before
        if pending_base is not None and "{" in masked:
            active_base = pending_base
            pending_base = None
        if active_base is not None:
            for match in FUNCTION_RE.finditer(masked):
                findings.append(
                    finding(
                        "backend_trait_method",
                        rel,
                        idx + 1,
                        line_classes[idx],
                        original_lines[idx],
                        symbol=match.group(1),
                        trait="Backend",
                    )
                )
        depth += masked.count("{") - masked.count("}")
        if active_base is not None and depth <= active_base:
            active_base = None
    return findings


def offset_line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def scan_coupling(
    rel: str,
    language: str,
    text: str,
    code_lines: list[str],
    line_classes: list[str],
) -> list[dict[str, Any]]:
    if language in {"binary", "python-bytecode"}:
        return []
    original_lines = text.splitlines()
    findings: list[dict[str, Any]] = []

    if language == "rust":
        findings.extend(backend_trait_methods(rel, original_lines, code_lines, line_classes))

    model_source = rel.startswith("crates/ferrum-models/")
    product_command = rel in {
        "crates/ferrum-cli/src/commands/run.rs",
        "crates/ferrum-cli/src/commands/serve.rs",
    }
    for idx, code in enumerate(code_lines):
        if not code.strip():
            continue
        original = original_lines[idx]
        classification = line_classes[idx]
        seen_arch: set[str] = set()
        for match in ARCH_TOKEN_RE.finditer(code):
            symbol = match.group(0)
            lowered = symbol.lower()
            if lowered in seen_arch:
                continue
            seen_arch.add(lowered)
            findings.append(
                finding(
                    "architecture_named_symbol",
                    rel,
                    idx + 1,
                    classification,
                    original,
                    symbol=symbol,
                )
            )
        for match in QWEN35_RE.finditer(code):
            findings.append(
                finding(
                    "qwen35_symbol",
                    rel,
                    idx + 1,
                    classification,
                    original,
                    symbol=match.group(0),
                )
            )
        for match in DECL_RE.finditer(code):
            declaration_kind, symbol = match.groups()
            if ARCH_TOKEN_RE.search(symbol):
                findings.append(
                    finding(
                        "architecture_named_api",
                        rel,
                        idx + 1,
                        classification,
                        original,
                        symbol=symbol,
                        declaration_kind=declaration_kind,
                    )
                )
            if product_command and PRODUCT_DECISION_RE.search(symbol):
                findings.append(
                    finding(
                        "product_decision_candidate",
                        rel,
                        idx + 1,
                        classification,
                        original,
                        symbol=symbol,
                        declaration_kind=declaration_kind,
                    )
                )
            if model_source and declaration_kind == "fn" and SCAFFOLDING_RE.search(symbol):
                findings.append(
                    finding(
                        "model_scaffolding_candidate",
                        rel,
                        idx + 1,
                        classification,
                        original,
                        symbol=symbol,
                    )
                )
        for match in FACTORY_DECL_RE.finditer(code):
            symbol = match.group(1)
            findings.append(
                finding(
                    "factory_symbol",
                    rel,
                    idx + 1,
                    classification,
                    original,
                    symbol=symbol,
                )
            )
            if (
                "legacy" in symbol.lower()
                or "executorfactory" in symbol.lower()
                or model_source
                or rel == "crates/ferrum-engine/src/registry.rs"
            ):
                findings.append(
                    finding(
                        "legacy_factory_candidate",
                        rel,
                        idx + 1,
                        classification,
                        original,
                        symbol=symbol,
                    )
                )
        runner_patterns = (
            (RUNNER_DECL_RE, MODEL_RUN_FUNCTION_RE) if model_source else (RUNNER_DECL_RE,)
        )
        for regex in runner_patterns:
            for match in regex.finditer(code):
                findings.append(
                    finding(
                        "model_runner_candidate",
                        rel,
                        idx + 1,
                        classification,
                        original,
                        symbol=match.group(1),
                    )
                )
        if BACKEND_CFG_RE.search(code):
            backends = sorted(
                backend for backend in ("cuda", "metal") if re.search(backend, code, re.IGNORECASE)
            )
            findings.append(
                finding(
                    "backend_cfg",
                    rel,
                    idx + 1,
                    classification,
                    original,
                    backends=backends,
                )
            )

    # Join comment-stripped lines to support calls whose key is on the next line.
    searchable = "\n".join(code_lines)
    env_matches: list[tuple[int, str, str | None]] = []
    for regex in (ENV_CALL_RE, PYTHON_ENV_INDEX_RE, SHELL_ENV_RE):
        for match in regex.finditer(searchable):
            env_matches.append((match.start(), match.group("reader"), match.group("key")))
    concrete_offsets = {offset for offset, _reader, key in env_matches if key is not None}
    env_constants = {
        match.group("symbol"): match.group("key")
        for match in ENV_CONSTANT_RE.finditer(searchable)
    }
    for match in ENV_IDENTIFIER_CALL_RE.finditer(searchable):
        key = env_constants.get(match.group("symbol"))
        if key is not None and match.start() not in concrete_offsets:
            env_matches.append((match.start(), match.group("reader"), key))
            concrete_offsets.add(match.start())
    for match in GENERIC_ENV_RE.finditer(searchable):
        if match.start() not in concrete_offsets:
            env_matches.append((match.start(), match.group("reader"), None))
    seen_env: set[tuple[int, str, str | None]] = set()
    for offset, reader, key in sorted(
        env_matches, key=lambda item: (item[0], item[1], item[2] or "")
    ):
        line_no = offset_line(searchable, offset)
        identity = (line_no, reader, key)
        if identity in seen_env:
            continue
        seen_env.add(identity)
        findings.append(
            finding(
                "ferrum_env_read",
                rel,
                line_no,
                line_classes[line_no - 1],
                original_lines[line_no - 1],
                reader=reader,
                key=key,
            )
        )
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in findings:
        key = (
            item["category"],
            item["path"],
            item["line"],
            item.get("symbol"),
            item.get("reader"),
            item.get("key"),
        )
        unique[key] = item
    return list(unique.values())


def native_tree_key(rel: str) -> str:
    parts = Path(rel).parts
    lowered = [part.lower() for part in parts[:-1]]
    for family in ("vllm", "flash_attn", "cutlass", "llama_cpp"):
        if any(family in part for part in lowered):
            return f"upstream:{family}"
    for idx, part in enumerate(lowered):
        if part in VENDOR_PARTS:
            end = min(len(parts), idx + 2)
            return "path:" + "/".join(parts[:end])
    if len(parts) >= 3 and parts[0] == "crates":
        if lowered[2] in {"kernels", "native", "cuda", "metal", "csrc"}:
            return "path:" + "/".join(parts[:3])
        return "path:" + "/".join(parts[:2])
    if len(parts) >= 2 and parts[0] == "scripts":
        return "path:" + "/".join(parts[:2]) if Path(rel).parent != Path("scripts") else "path:scripts"
    return "path:" + str(Path(rel).parent)


def load_baseline(path: Path | None) -> tuple[dict[str, list[str]], dict[str, str]]:
    if path is None:
        return {}, {}
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise InventoryError(f"baseline not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InventoryError(f"invalid baseline JSON {path}: {exc}") from exc
    if data.get("schema_version") != 1 or not isinstance(data.get("files"), list):
        raise InventoryError(f"{path}: expected runtime inventory schema_version 1")
    by_sha: dict[str, list[str]] = defaultdict(list)
    by_path: dict[str, str] = {}
    for idx, entry in enumerate(data["files"]):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise InventoryError(f"{path}: files[{idx}] is invalid")
        sha = entry.get("sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise InventoryError(f"{path}: files[{idx}].sha256 is invalid")
        by_sha[sha].append(entry["path"])
        by_path[entry["path"]] = sha
    return {sha: sorted(paths) for sha, paths in by_sha.items()}, by_path


def iter_files(root: Path, excluded: set[Path]) -> tuple[list[Path], str]:
    git_files = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *SCAN_ROOTS,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    files_by_path: dict[Path, Path] = {}
    if git_files.returncode == 0:
        for raw in git_files.stdout.split(b"\0"):
            if not raw:
                continue
            path = root / raw.decode("utf-8", errors="surrogateescape")
            if path.is_file() and path.resolve() not in excluded:
                files_by_path[path.resolve()] = path

    for scan_root in SCAN_ROOTS:
        base = root / scan_root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            relative_parts = path.relative_to(base).parts
            if any(part in SOURCE_DISCOVERY_EXCLUDED_DIRS for part in relative_parts[:-1]):
                continue
            if path.is_file() and path.resolve() not in excluded:
                files_by_path[path.resolve()] = path
    method = (
        "git tracked/untracked plus filesystem ignored source scan"
        if git_files.returncode == 0
        else "filesystem source scan"
    )
    return (
        sorted(files_by_path.values(), key=lambda path: path.relative_to(root).as_posix()),
        method,
    )


def git_state(root: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    sha = run(["rev-parse", "HEAD"])
    tree_sha = run(["rev-parse", "HEAD^{tree}"])
    status = run(["status", "--short"])
    return {
        "sha": sha,
        "tree_sha": tree_sha,
        "dirty": bool(status) if status is not None else None,
        "status_short": status.splitlines() if status else [],
    }


def aggregate_native_trees(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in files:
        if Path(entry["path"]).suffix.lower() not in NATIVE_SUFFIXES:
            continue
        if entry["classification"] in {"test", "fixture", "example"}:
            continue
        groups[native_tree_key(entry["path"])].append(entry)

    results: list[dict[str, Any]] = []
    for tree_key, entries in sorted(groups.items()):
        source_bytes = sum(entry["size_bytes"] for entry in entries)
        production_loc = sum(
            entry["logical_loc_by_classification"].get("production", 0)
            + entry["logical_loc_by_classification"].get("vendor", 0)
            for entry in entries
        )
        translation_units = sum(
            Path(entry["path"]).suffix.lower() in TRANSLATION_UNIT_SUFFIXES for entry in entries
        )
        reasons: list[str] = []
        if production_loc >= LARGE_NATIVE_LOC:
            reasons.append("production_loc")
        if source_bytes >= LARGE_NATIVE_BYTES:
            reasons.append("source_bytes")
        if translation_units >= LARGE_NATIVE_TRANSLATION_UNITS:
            reasons.append("translation_units")
        content_hashes = sorted(entry["sha256"] for entry in entries)
        content_root_sha256 = sha256_bytes(("\n".join(content_hashes) + "\n").encode())
        results.append(
            {
                "tree_key": tree_key,
                "content_root_sha256": content_root_sha256,
                "paths": sorted(entry["path"] for entry in entries),
                "source_file_count": len(entries),
                "production_loc": production_loc,
                "source_bytes": source_bytes,
                "translation_unit_count": translation_units,
                "is_large": bool(reasons),
                "qualifying_reasons": reasons,
            }
        )
    return results


def duplicate_product_decisions(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in findings:
        if item["category"] == "product_decision_candidate":
            occurrences[item["symbol"]].append(
                {"path": item["path"], "line": item["line"]}
            )
    return [
        {"symbol": symbol, "occurrences": sorted(items, key=lambda item: (item["path"], item["line"]))}
        for symbol, items in sorted(occurrences.items())
        if len({item["path"] for item in items}) > 1
    ]


def build_inventory(
    root: Path,
    out_path: Path,
    baseline_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise InventoryError(f"root is not a directory: {root}")
    missing_roots = [scan_root for scan_root in SCAN_ROOTS if not (root / scan_root).is_dir()]
    if missing_roots:
        raise InventoryError(f"missing scan roots: {', '.join(missing_roots)}")

    baseline_by_sha, baseline_by_path = load_baseline(baseline_path)
    excluded = {out_path.resolve(), Path(str(out_path) + ".tmp").resolve()}
    paths, discovery_method = iter_files(root, excluded)
    entries: list[dict[str, Any]] = []
    all_findings: list[dict[str, Any]] = []

    for path in paths:
        rel = path.relative_to(root).as_posix()
        data = path.read_bytes()
        sha = sha256_bytes(data)
        language = language_for(path, data)
        classification = classify_path(rel)
        if language in {"binary", "python-bytecode"}:
            text = ""
            loc = 0
            loc_by_classification: dict[str, int] = {}
            code_lines: list[str] = []
            line_classes: list[str] = []
        else:
            text = data.decode("utf-8", errors="replace")
            loc, loc_by_classification, code_lines, line_classes = logical_loc(
                language, text, classification
            )
        file_findings = scan_coupling(rel, language, text, code_lines, line_classes)
        all_findings.extend(file_findings)
        baseline_paths = baseline_by_sha.get(sha, [])
        if rel in baseline_paths:
            path_status = "unchanged_identity"
        elif baseline_paths:
            path_status = "moved_or_copied_identity"
        else:
            path_status = "new_identity" if baseline_path else "uncompared"
        entries.append(
            {
                "path": rel,
                "sha256": sha,
                "content_id": f"sha256:{sha}",
                "size_bytes": len(data),
                "language": language,
                "classification": classification,
                "logical_loc": loc,
                "logical_loc_by_classification": loc_by_classification,
                "baseline_paths_with_same_identity": baseline_paths,
                "path_status": path_status,
                "coupling_finding_count": len(file_findings),
                "coupling_counts": dict(
                    sorted(Counter(item["category"] for item in file_findings).items())
                ),
            }
        )

    current_by_sha: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        current_by_sha[entry["sha256"]].append(entry["path"])
    content_identities = [
        {
            "sha256": sha,
            "content_id": f"sha256:{sha}",
            "paths": sorted(current_paths),
            "baseline_paths": baseline_by_sha.get(sha, []),
        }
        for sha, current_paths in sorted(current_by_sha.items())
    ]
    movements: list[dict[str, Any]] = []
    if baseline_path:
        for sha in sorted(set(current_by_sha) & set(baseline_by_sha)):
            old_paths = set(baseline_by_sha[sha])
            new_paths = set(current_by_sha[sha])
            removed = sorted(old_paths - new_paths)
            added = sorted(new_paths - old_paths)
            if removed and added:
                movements.append(
                    {
                        "sha256": sha,
                        "content_id": f"sha256:{sha}",
                        "from_paths": removed,
                        "to_paths": added,
                    }
                )

    native_trees = aggregate_native_trees(entries)
    classification_file_counts = Counter(entry["classification"] for entry in entries)
    loc_counts: Counter[str] = Counter()
    for entry in entries:
        loc_counts.update(entry["logical_loc_by_classification"])
    category_counts = Counter(item["category"] for item in all_findings)
    per_root_counts = Counter(entry["path"].split("/", 1)[0] for entry in entries)

    result: dict[str, Any] = {
        "schema_version": 1,
        "analyzer": {
            "path": "scripts/release/runtime_vnext_inventory.py",
            "classification_values": list(CLASSIFICATIONS),
            "loc_languages": sorted(LOC_LANGUAGES),
            "source_discovery_excluded_dirs": sorted(SOURCE_DISCOVERY_EXCLUDED_DIRS),
            "identity_key": "sha256",
            "large_native_thresholds": {
                "production_loc_gte": LARGE_NATIVE_LOC,
                "source_bytes_gte": LARGE_NATIVE_BYTES,
                "translation_units_gte": LARGE_NATIVE_TRANSLATION_UNITS,
                "qualifier": "any",
            },
        },
        "root": str(root),
        "git": git_state(root),
        "scope": {
            "scan_roots": list(SCAN_ROOTS),
            "scripts_policy": "all scripts (superset of product/release scripts)",
            "discovery_method": discovery_method,
            "discovered_file_count": len(paths),
            "inventoried_file_count": len(entries),
            "coverage_ratio": 1.0 if len(paths) == len(entries) else 0.0,
            "file_count_by_root": dict(sorted(per_root_counts.items())),
            "excluded_paths": sorted(
                path.relative_to(root).as_posix()
                for path in excluded
                if path.is_relative_to(root)
            ),
        },
        "summary": {
            "file_count": len(entries),
            "file_count_by_classification": {
                classification: classification_file_counts.get(classification, 0)
                for classification in CLASSIFICATIONS
            },
            "logical_loc": sum(entry["logical_loc"] for entry in entries),
            "logical_loc_by_classification": {
                classification: loc_counts.get(classification, 0)
                for classification in CLASSIFICATIONS
            },
            "coupling_finding_count": len(all_findings),
            "coupling_count_by_category": dict(sorted(category_counts.items())),
            "native_source_tree_count": len(native_trees),
            "large_third_party_native_source_count": sum(
                tree["is_large"] for tree in native_trees
            ),
        },
        "files": entries,
        "content_identities": content_identities,
        "move_tracking": {
            "baseline": str(baseline_path.resolve()) if baseline_path else None,
            "baseline_file_count": len(baseline_by_path),
            "movement_count": len(movements),
            "movements": movements,
        },
        "coupling": {
            "findings": sorted(
                all_findings,
                key=lambda item: (item["path"], item["line"], item["category"], item.get("symbol", "")),
            ),
            "potential_run_serve_duplicate_decisions": duplicate_product_decisions(all_findings),
        },
        "large_native_source_trees": native_trees,
    }
    if result["scope"]["coverage_ratio"] != 1.0:
        raise InventoryError("inventory coverage is not 100%")
    return result


def write_inventory(path: Path, result: dict[str, Any]) -> None:
    if path.exists() and path.is_dir():
        raise InventoryError(f"--out must be a JSON file, not a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InventoryError(f"self-test: {message}")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="runtime-vnext-inventory-") as temp:
        root = Path(temp)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        (root / ".gitignore").write_text("crates/demo/ignored-native/\n")
        rust_path = root / "crates/demo/src/lib.rs"
        rust_path.parent.mkdir(parents=True)
        rust_path.write_text(
            """pub trait Backend {
    fn alloc();
    fn qwen35_decode();
}

#[cfg(feature = \"cuda\")]
pub fn cuda_path() {}

pub fn hidden_env() {
    let _ = std::env::var(\"FERRUM_HIDDEN_SWITCH\");
}

pub struct LegacyExecutorFactory;
pub struct Qwen35ModelRunner;
pub fn qwen35_decode_loop() {}

#[cfg(test)]
mod tests {
    #[test]
    fn inline_test() {}
}

pub fn after_tests_is_production() {}
"""
        )
        moved_path = root / "crates/demo/src/moved.rs"
        moved_path.write_text("pub fn content_identity_survives_move() {}\n")
        tests = root / "crates/demo/tests/integration.rs"
        tests.parent.mkdir(parents=True)
        tests.write_text("#[test]\nfn integration() {}\n")
        example = root / "crates/demo/examples/demo.rs"
        example.parent.mkdir(parents=True)
        example.write_text("fn main() {}\n")
        generated = root / "crates/demo/generated/bindings.rs"
        generated.parent.mkdir(parents=True)
        generated.write_text("pub const GENERATED: bool = true;\n")
        fixture = root / "scripts/release/fixtures/runtime_vnext_inventory/input.json"
        fixture.parent.mkdir(parents=True)
        fixture.write_text('{"fixture": true}\n')
        markdown = root / "scripts/release/notes.md"
        markdown.write_text("\n".join(f"documentation {idx}" for idx in range(20)) + "\n")
        product_script = root / "scripts/product_smoke.sh"
        product_script.write_text("#!/bin/sh\necho ${FERRUM_PRODUCT_SMOKE}\n")
        vendor = root / "crates/demo/vendor/upstream"
        vendor.mkdir(parents=True)
        for idx in range(LARGE_NATIVE_TRANSLATION_UNITS):
            (vendor / f"kernel_{idx}.cu").write_text(f"void kernel_{idx}() {{}}\n")
        ignored_native = root / "crates/demo/ignored-native/hidden.cu"
        ignored_native.parent.mkdir(parents=True)
        ignored_native.write_text("void ignored_but_inventoried() {}\n")

        moved_sha = sha256_bytes(moved_path.read_bytes())
        baseline_path = root / "baseline.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "files": [
                        {
                            "path": "crates/demo/src/old_location.rs",
                            "sha256": moved_sha,
                        }
                    ],
                }
            )
        )
        out = root / "artifact/coupling-inventory.json"
        result = build_inventory(root, out, baseline_path)
        write_inventory(out, result)
        reparsed = json.loads(out.read_text())

        require(reparsed["scope"]["coverage_ratio"] == 1.0, "coverage must be 100%")
        require(reparsed["scope"]["file_count_by_root"] == {"crates": 16, "scripts": 3}, "scan root counts")
        by_path = {entry["path"]: entry for entry in reparsed["files"]}
        rust = by_path["crates/demo/src/lib.rs"]
        require(rust["classification"] == "production", "Rust source classification")
        require(rust["logical_loc_by_classification"].get("production", 0) > 0, "production LOC")
        require(rust["logical_loc_by_classification"].get("test", 0) > 0, "inline test LOC")
        require(by_path["crates/demo/tests/integration.rs"]["classification"] == "test", "test classification")
        require(by_path["crates/demo/examples/demo.rs"]["classification"] == "example", "example classification")
        require(by_path["crates/demo/generated/bindings.rs"]["classification"] == "generated", "generated classification")
        require(by_path["scripts/release/fixtures/runtime_vnext_inventory/input.json"]["classification"] == "fixture", "fixture classification")
        require(by_path["scripts/release/notes.md"]["logical_loc"] == 0, "Markdown must not enter LOC")
        require(by_path["crates/demo/vendor/upstream/kernel_0.cu"]["classification"] == "vendor", "vendor classification")
        require(
            "crates/demo/ignored-native/hidden.cu" in by_path,
            "ignored native source must still be inventoried",
        )
        counts = reparsed["summary"]["coupling_count_by_category"]
        for category in (
            "qwen35_symbol",
            "architecture_named_api",
            "backend_trait_method",
            "backend_cfg",
            "ferrum_env_read",
            "legacy_factory_candidate",
            "model_runner_candidate",
        ):
            require(counts.get(category, 0) > 0, f"missing {category}")
        require(reparsed["summary"]["large_third_party_native_source_count"] == 1, "large native threshold")
        require(reparsed["move_tracking"]["movement_count"] == 1, "SHA move tracking")
        require(
            by_path["crates/demo/src/moved.rs"]["path_status"] == "moved_or_copied_identity",
            "moved file identity",
        )
        require(all(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) for entry in reparsed["files"]), "file SHA256")
        require(
            all(entry["classification"] in CLASSIFICATIONS for entry in reparsed["files"]),
            "classification enum",
        )
        composite_classes = rust_line_classifications(
            [
                '#[cfg(any(test, feature = "test-utils"))]',
                "mod composite_tests {",
                "fn helper() {}",
                "}",
            ],
            "production",
        )
        require(composite_classes[2] == "test", "composite cfg(test) module classification")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="output JSON file")
    parser.add_argument("--baseline", type=Path, help="previous inventory for SHA-based move tracking")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--self-test", action="store_true", help="run synthetic analyzer tests")
    args = parser.parse_args(argv)
    if args.self_test and (args.out or args.baseline):
        parser.error("--self-test cannot be combined with --out or --baseline")
    if not args.self_test and args.out is None:
        parser.error("--out is required unless --self-test is set")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            self_test()
            print("RUNTIME VNEXT INVENTORY SELF-TEST PASS")
            return 0
        assert args.out is not None
        out = args.out if args.out.is_absolute() else args.root.resolve() / args.out
        baseline = args.baseline
        if baseline is not None and not baseline.is_absolute():
            baseline = args.root.resolve() / baseline
        result = build_inventory(args.root, out, baseline)
        write_inventory(out, result)
        print(f"RUNTIME VNEXT INVENTORY PASS: {out}")
        return 0
    except (InventoryError, OSError) as exc:
        target = args.out if args.out is not None else "self-test"
        print(f"RUNTIME VNEXT INVENTORY FAIL: {target}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
