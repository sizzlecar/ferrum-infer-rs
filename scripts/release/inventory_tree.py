#!/usr/bin/env python3
"""Inventory crates/docs/scripts before release-gate cleanup."""
from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False).stdout


def workspace_members() -> list[str]:
    text = (ROOT / "Cargo.toml").read_text()
    m = re.search(r"members\s*=\s*\[(.*?)\]", text, re.S)
    if not m:
        return []
    body = re.sub(r"#.*", "", m.group(1))
    return sorted(re.findall(r'"([^"]+)"', body))


def category_script(path: Path) -> str:
    s = str(path)
    name = path.name.lower()
    if s.startswith("scripts/release/"):
        return "release gate"
    if "microbench" in s:
        return "microbench"
    if "fa2" in name:
        return "FA2 special gate"
    if name.startswith("m3_") or "m3" in name:
        return "M3/CUDA runner"
    if "pod" in name or "vast" in name or "cuda_build" in name:
        return "GPU setup/pod helper"
    if "bench" in name or "aggregate" in name or "compare" in name or "profile" in name:
        return "benchmark/report utility"
    if name.endswith(".sh") or name.endswith(".py") or name.endswith(".cu") or name.endswith(".cpp"):
        return "legacy/archive candidate"
    return "benchmark/report utility"


def category_doc(path: Path) -> str:
    s = str(path)
    name = path.name.lower()
    if s.startswith("docs/release/"):
        return "release docs"
    if s.startswith("docs/bench/"):
        return "benchmark reports/artifacts"
    if s.startswith("docs/status/"):
        return "status notes"
    if "archive" in s:
        return "archive candidates"
    if any(x in name for x in ["session", "debug", "postmortem", "plan"]):
        return "session/debug notes"
    return "public product docs"


def reference_count(path: Path) -> int:
    rel = str(path)
    out = run(["git", "grep", "-n", "--", rel])
    return len([line for line in out.splitlines() if line])


def grouped_files(root: str, categorizer) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = defaultdict(list)
    base = ROOT / root
    if not base.exists():
        return out
    for path in sorted(base.rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            rel = path.relative_to(ROOT)
            out[categorizer(rel)].append(rel)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    members = workspace_members()
    crate_dirs = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "crates").glob("*/Cargo.toml"))
    crate_dirs = [str(Path(p).parent) for p in crate_dirs]
    member_set = set(members)
    disk_set = set(crate_dirs)
    scripts = grouped_files("scripts", category_script)
    docs = grouped_files("docs", category_doc)
    candidates = scripts.get("legacy/archive candidate", []) + docs.get("archive candidates", [])

    lines = ["# crates/docs/scripts inventory", "", f"Generated from `{ROOT}`.", ""]
    lines += ["## Workspace crate members", ""]
    lines += [f"- `{m}`" for m in members] or ["- none"]
    lines += ["", "## Actual crate directories", ""]
    lines += [f"- `{d}`" for d in crate_dirs] or ["- none"]
    lines += ["", "## Crate difference sets", ""]
    lines += ["Workspace members missing on disk:"]
    lines += [f"- `{x}`" for x in sorted(member_set - disk_set)] or ["- none"]
    lines += ["", "Crates on disk not listed in workspace:"]
    lines += [f"- `{x}`" for x in sorted(disk_set - member_set)] or ["- none"]
    lines += ["", "## Scripts inventory", ""]
    for category in sorted(scripts):
        lines += [f"### {category}", ""]
        lines += [f"- `{p}`" for p in scripts[category]] or ["- none"]
        lines.append("")
    lines += ["## Docs inventory", ""]
    for category in sorted(docs):
        lines += [f"### {category}", ""]
        lines += [f"- `{p}`" for p in docs[category]] or ["- none"]
        lines.append("")
    lines += ["## Archive/delete candidate reference counts", ""]
    if candidates:
        for path in candidates:
            lines.append(f"- `{path}`: git-grep references = {reference_count(path)}")
    else:
        lines.append("- none")
    lines.append("")

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"INVENTORY PASS: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
