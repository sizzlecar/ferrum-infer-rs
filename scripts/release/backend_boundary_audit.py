#!/usr/bin/env python3
"""Audit direct CUDA/Metal product-path decisions.

The goal is deliberately narrow: catch backend-specific branching that leaks
outside backend-local code, backend registration, runtime preset resolution, or
release/test lane selection. The script records every match, then fails only
for matches that are neither in an allowed path nor listed in the checked-in
allowlist.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("eq_ignore_ascii_case_cuda", re.compile(r'eq_ignore_ascii_case\("cuda"\)')),
    ("eq_ignore_ascii_case_metal", re.compile(r'eq_ignore_ascii_case\("metal"\)')),
    ("equals_cuda", re.compile(r'==\s*"cuda"')),
    ("equals_metal", re.compile(r'==\s*"metal"')),
    ("b_is_cuda_backend", re.compile(r"\bB::is_cuda_backend\s*\(")),
    ("b_is_metal_backend", re.compile(r"\bB::is_metal_backend\s*\(")),
    ("is_cuda_backend", re.compile(r"\bis_cuda_backend\s*\(")),
    ("is_metal_backend", re.compile(r"\bis_metal_backend\s*\(")),
)

SCAN_EXTENSIONS = {
    ".rs",
    ".py",
    ".sh",
    ".json",
    ".toml",
}

SCAN_ROOTS = (
    "crates",
    "scripts",
)

ALLOWED_PATH_PREFIXES = (
    "crates/ferrum-kernels/src/backend/",
    "scripts/release/",
)

ALLOWED_EXACT_PATHS = {
    # Backend registration / instantiation.
    "crates/ferrum-engine/src/registry.rs",
}

ALLOWLIST_REQUIRED_KEYS = {"path", "reason", "owner", "review_condition"}


class AuditError(Exception):
    pass


@dataclass(frozen=True)
class AllowEntry:
    path: str
    reason: str
    owner: str
    review_condition: str


def repo_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def load_allowlist(path: Path) -> dict[str, AllowEntry]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise AuditError(f"allowlist not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AuditError(f"invalid allowlist JSON {path}: {exc}") from exc
    if data.get("schema_version") != 1:
        raise AuditError(f"{path}: schema_version must be 1")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise AuditError(f"{path}: entries must be a list")
    out: dict[str, AllowEntry] = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise AuditError(f"{path}: entries[{idx}] must be an object")
        missing = sorted(ALLOWLIST_REQUIRED_KEYS - set(entry))
        if missing:
            raise AuditError(f"{path}: entries[{idx}] missing keys: {', '.join(missing)}")
        allow = AllowEntry(
            path=str(entry["path"]),
            reason=str(entry["reason"]).strip(),
            owner=str(entry["owner"]).strip(),
            review_condition=str(entry["review_condition"]).strip(),
        )
        if not allow.path or allow.path.startswith("/"):
            raise AuditError(f"{path}: entries[{idx}].path must be repo-relative")
        if not allow.reason or not allow.owner or not allow.review_condition:
            raise AuditError(f"{path}: entries[{idx}] has an empty required value")
        if allow.path in out:
            raise AuditError(f"{path}: duplicate allowlist path {allow.path}")
        out[allow.path] = allow
    return out


def should_scan(path: Path, root: Path, out_dir: Path, allowlist_path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix not in SCAN_EXTENSIONS:
        return False
    try:
        rel = repo_relative(path, root)
    except ValueError:
        return False
    if rel.startswith("target/") or "/__pycache__/" in rel:
        return False
    if path == allowlist_path or path == Path(__file__).resolve():
        return False
    try:
        path.relative_to(out_dir.resolve())
        return False
    except ValueError:
        return True


def iter_scan_files(root: Path, out_dir: Path, allowlist_path: Path) -> list[Path]:
    files: list[Path] = []
    for scan_root in SCAN_ROOTS:
        base = root / scan_root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if should_scan(path, root, out_dir, allowlist_path):
                files.append(path)
    return sorted(files)


def allowance_for(rel: str, allowlist: dict[str, AllowEntry]) -> tuple[bool, dict[str, str] | None]:
    if rel in ALLOWED_EXACT_PATHS or any(rel.startswith(prefix) for prefix in ALLOWED_PATH_PREFIXES):
        return True, {
            "kind": "allowed_path",
            "reason": "backend-local, backend registry, or release-script path",
            "owner": "repo-policy",
            "review_condition": "Review if this path starts making shared product decisions.",
        }
    entry = allowlist.get(rel)
    if entry is not None:
        return True, {
            "kind": "allowlist",
            "reason": entry.reason,
            "owner": entry.owner,
            "review_condition": entry.review_condition,
        }
    return False, None


def scan(root: Path, out_dir: Path, allowlist_path: Path) -> dict[str, Any]:
    allowlist = load_allowlist(allowlist_path)
    findings: list[dict[str, Any]] = []
    for path in iter_scan_files(root, out_dir, allowlist_path):
        rel = repo_relative(path, root)
        allowed, allowance = allowance_for(rel, allowlist)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(lines, start=1):
            matched = [name for name, pattern in PATTERNS if pattern.search(line)]
            if not matched:
                continue
            finding: dict[str, Any] = {
                "path": rel,
                "line": line_no,
                "patterns": matched,
                "text": line.strip(),
                "allowed": allowed,
            }
            if allowance is not None:
                finding["allowance"] = allowance
            findings.append(finding)
    violations = [finding for finding in findings if not finding["allowed"]]
    return {
        "schema_version": 1,
        "status": "pass" if not violations else "fail",
        "root": str(root),
        "allowlist": repo_relative(allowlist_path, root),
        "scan_roots": list(SCAN_ROOTS),
        "patterns": [name for name, _ in PATTERNS],
        "allowed_path_prefixes": list(ALLOWED_PATH_PREFIXES),
        "allowed_exact_paths": sorted(ALLOWED_EXACT_PATHS),
        "finding_count": len(findings),
        "violation_count": len(violations),
        "findings": findings,
        "violations": violations,
    }


def write_outputs(out_dir: Path, result: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "backend_boundary_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Backend Boundary Audit",
        "",
        f"- status: `{result['status']}`",
        f"- findings: {result['finding_count']}",
        f"- violations: {result['violation_count']}",
        "",
        "| path | line | allowed | patterns | reason |",
        "|---|---:|---|---|---|",
    ]
    for finding in result["findings"]:
        allowance = finding.get("allowance") or {}
        lines.append(
            "| {path} | {line} | {allowed} | {patterns} | {reason} |".format(
                path=finding["path"],
                line=finding["line"],
                allowed="yes" if finding["allowed"] else "no",
                patterns=", ".join(finding["patterns"]),
                reason=str(allowance.get("reason", "")),
            )
        )
    (out_dir / "backend_boundary_audit.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path("scripts/release/backend_boundary_allowlist.json"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out.resolve()
    allowlist_path = args.allowlist
    if not allowlist_path.is_absolute():
        allowlist_path = root / allowlist_path
    allowlist_path = allowlist_path.resolve()

    try:
        result = scan(root, out_dir, allowlist_path)
        write_outputs(out_dir, result)
    except AuditError as exc:
        print(f"BACKEND BOUNDARY AUDIT FAIL: {out_dir}: {exc}", file=sys.stderr)
        return 1

    if result["status"] != "pass":
        print(f"BACKEND BOUNDARY AUDIT FAIL: {out_dir}", file=sys.stderr)
        for violation in result["violations"]:
            print(
                f"  {violation['path']}:{violation['line']} "
                f"{','.join(violation['patterns'])}",
                file=sys.stderr,
            )
        return 1
    print(f"BACKEND BOUNDARY AUDIT PASS: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
