#!/usr/bin/env python3
"""Bind the existing G00a fact checkpoint as the Runtime vNext G00F DAG node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PASS_PREFIX = "FERRUM RUNTIME VNEXT G00F FACTS PASS"
UNLOCKS = ["S0A", "S1"]
DOES_NOT_PROVE = [
    "G00P",
    "G01",
    "G01B",
    "model_migration",
    "performance",
    "production_wiring",
    "release",
]


class CheckpointError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointError(message)


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"required regular file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot read {label} {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def clean_source() -> dict[str, Any]:
    status = [line for line in git_text("status", "--short").splitlines() if line]
    require(not status, f"G00F requires a clean checkout: {status}")
    return {
        "git_sha": git_text("rev-parse", "HEAD"),
        "git_tree_sha": git_text("rev-parse", "HEAD^{tree}"),
        "dirty": False,
        "status_short": [],
    }


def resolve_g00a(g00a_outer_path: Path, source: dict[str, Any]) -> dict[str, Any]:
    outer_path = g00a_outer_path.resolve()
    require(REPO_ROOT not in outer_path.parents, "G00a artifact must be outside the source tree")
    outer = read_json(outer_path, "G00a outer manifest")
    require(
        outer.get("lane") == "vnext-g00a" and outer.get("status") == "pass",
        "G00a outer manifest lane/status mismatch",
    )
    require(outer.get("git_sha") == source["git_sha"], "G00a outer manifest is stale")
    require(
        outer.get("dirty_status") == {"is_dirty": False, "status_short": []},
        "G00a outer manifest records a dirty source",
    )
    child_ref = outer.get("child_artifacts", {}).get("child_manifest")
    require(isinstance(child_ref, dict), "G00a outer manifest lacks child manifest identity")
    child_path = Path(str(child_ref.get("path", ""))).resolve()
    require(child_path.parent == outer_path.parent, "G00a outer and child manifests differ in root")
    child_digest = sha256(child_path)
    require(child_ref.get("sha256") == child_digest, "G00a child manifest SHA256 mismatch")
    child = read_json(child_path, "G00a child manifest")

    sys.path.insert(0, str(REPO_ROOT / "scripts/release"))
    import run_gate  # pylint: disable=import-outside-toplevel

    lane = run_gate.LaneCommand(
        cmd=[],
        expected_child_pass_line=child.get("pass_line"),
        child_manifest_path=child_path,
        provenance_kind="vnext-g00a",
    )
    provenance = run_gate.validate_vnext_g00a_provenance(
        lane,
        child,
        child_digest,
        verify_checkout=True,
    )
    collector = child.get("collector")
    require(isinstance(collector, dict), "G00a child collector is missing")
    require(
        collector.get("git_sha") == source["git_sha"]
        and collector.get("git_tree_sha") == source["git_tree_sha"],
        "G00a collector source is stale",
    )
    return {
        "outer_manifest": {
            "path": str(outer_path),
            "sha256": sha256(outer_path),
        },
        "child_manifest": {
            "path": str(child_path),
            "sha256": child_digest,
        },
        "artifact_index_sha256": provenance["artifact_index_sha256"],
        "model_lane_count": provenance["model_lane_count"],
        "historical_bug_counts": provenance["historical_bug_counts"],
        "facts_reused_without_copy": True,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_checkpoint(g00a_path: Path, out_dir: Path) -> str:
    source = clean_source()
    output = out_dir.resolve()
    require(REPO_ROOT not in output.parents and output != REPO_ROOT, "G00F output must be outside the source tree")
    require(not output.exists() or not any(output.iterdir()), f"G00F output is not empty: {output}")
    g00a = resolve_g00a(g00a_path, source)
    pass_line = f"{PASS_PREFIX}: {output}"

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        manifest = {
            "schema_version": 1,
            "artifact_type": "runtime_vnext_g00f_facts_manifest",
            "checkpoint_id": "G00F",
            "lane": "runtime-vnext-g00f",
            "status": "pass",
            "canonical": True,
            "artifact_dir": str(output),
            "source": source,
            "g00a": g00a,
            "unlocks": UNLOCKS,
            "does_not_prove": DOES_NOT_PROVE,
            "pass_line": pass_line,
        }
        write_json(staging / "manifest.json", manifest)
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return pass_line


def self_test() -> int:
    require(UNLOCKS == ["S0A", "S1"], "G00F unlock scope drifted")
    require(
        set(DOES_NOT_PROVE)
        == {
            "G00P",
            "G01",
            "G01B",
            "model_migration",
            "performance",
            "production_wiring",
            "release",
        },
        "G00F negative claim scope drifted",
    )
    print("FERRUM RUNTIME VNEXT G00F FACTS SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g00a", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.g00a is None or args.out is None:
        parser.error("--g00a and --out are required")
    try:
        print(build_checkpoint(args.g00a, args.out))
        return 0
    except CheckpointError as error:
        print(f"{PASS_PREFIX} FAIL: {args.out}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
