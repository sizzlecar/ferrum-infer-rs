#!/usr/bin/env python3
"""Aggregate the release-blocking R0 source, numerics, and CUDA S2 evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import runtime_vnext_g08a_checkpoint as g08a
import runtime_vnext_s2_cuda_product_contract as s2_contract


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
PASS_PREFIX = "FERRUM RUNTIME VNEXT R0 CORE CLOSURE PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT R0 CORE CLOSURE SELFTEST PASS"
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
DEPENDENCY_KEYS = ("source", "numerics", "s2")
CONTROL_PLANE_PREFIXES = ("docs/",)
CONTROL_PLANE_FILES = frozenset(
    {
        "scripts/release/runtime_vnext_r0_core_closure.py",
        "scripts/release/run_gate.py",
        "scripts/release/change_impact_rules.json",
        "scripts/release/fixtures/change_impact/planner_fixtures.json",
    }
)
S2_MULTITURN_SCENARIO = (
    "scripts/release/scenarios/runtime_vnext_s2_multiturn_concurrency_cuda.json"
)
DEPENDENCY_LOCAL_FILES = {
    "source": frozenset(
        {
            "scripts/release/runtime_vnext_s2_cuda_product_contract.py",
            "scripts/release/runtime_vnext_s2_multiturn_concurrency_checkpoint.py",
            S2_MULTITURN_SCENARIO,
        }
    ),
    "numerics": frozenset(
        {
            "scripts/release/runtime_vnext_s2_cuda_product_contract.py",
            "scripts/release/runtime_vnext_s2_multiturn_concurrency_checkpoint.py",
            S2_MULTITURN_SCENARIO,
        }
    ),
    "s2": frozenset(),
}
DOES_NOT_PROVE = [
    "R1 three-model CUDA and Metal correctness",
    "R2 performance, profile, or build acceptance",
    "R3 release assets or installed regression",
    "v0.8.0 release readiness",
]


class R0Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R0Error(message)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise R0Error(f"invalid {label} JSON {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="ascii") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    require(resolved.is_file() and not resolved.is_symlink(), f"artifact is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def validate_ref(value: Any, label: str) -> Path:
    require(
        isinstance(value, dict) and set(value) == {"path", "sha256", "size_bytes"},
        f"{label} reference fields differ",
    )
    path = Path(str(value["path"])).expanduser().resolve()
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    require(path.stat().st_size == value["size_bytes"], f"{label} size mismatch")
    require(sha256(path) == value["sha256"], f"{label} SHA256 mismatch")
    return path


def git_text(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    require(process.returncode == 0, f"git {' '.join(args)} failed: {process.stderr.strip()}")
    return process.stdout.strip()


def current_source() -> dict[str, Any]:
    status = [line for line in git_text("status", "--short").splitlines() if line]
    require(not status, f"R0 source must be clean: {status[:8]}")
    return {
        "git_sha": git_text("rev-parse", "HEAD"),
        "git_tree_sha": git_text("rev-parse", "HEAD^{tree}"),
        "dirty": False,
    }


def normalize_source(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} source is missing")
    source = {
        "git_sha": value.get("git_sha"),
        "git_tree_sha": value.get("git_tree_sha"),
        "dirty": value.get("dirty"),
    }
    require(
        GIT_SHA_RE.fullmatch(str(source["git_sha"])) is not None
        and GIT_SHA_RE.fullmatch(str(source["git_tree_sha"])) is not None
        and source["dirty"] is False,
        f"{label} source identity is invalid",
    )
    return source


def source_from_outer(path: Path, key: str) -> dict[str, Any]:
    outer = read_json(path.expanduser().resolve(), f"R0 {key} outer manifest")
    child_artifacts = outer.get("child_artifacts")
    require(isinstance(child_artifacts, dict), f"R0 {key} child provenance is missing")
    return normalize_source(child_artifacts.get("source"), f"R0 {key}")


def control_plane_only(paths: list[str], key: str) -> tuple[list[str], list[str]]:
    require(key in DEPENDENCY_LOCAL_FILES, f"unknown R0 dependency closure: {key}")
    allowed_files = CONTROL_PLANE_FILES | DEPENDENCY_LOCAL_FILES[key]
    allowed = []
    rejected = []
    for path in paths:
        if path in allowed_files or path.startswith(CONTROL_PLANE_PREFIXES):
            allowed.append(path)
        else:
            rejected.append(path)
    return allowed, rejected


def source_closure(
    source: dict[str, Any], current: dict[str, Any], key: str, label: str
) -> dict[str, Any]:
    recorded_sha = str(source["git_sha"])
    require(
        git_text("rev-parse", f"{recorded_sha}^{{tree}}") == source["git_tree_sha"],
        f"{label} recorded source tree differs from git",
    )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", recorded_sha, str(current["git_sha"])],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    require(ancestor.returncode == 0, f"{label} source is not an ancestor of current HEAD")
    changed = [
        line
        for line in git_text(
            "diff",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            f"{recorded_sha}..{current['git_sha']}",
        ).splitlines()
        if line
    ]
    allowed, rejected = control_plane_only(changed, key)
    require(
        not rejected,
        f"{label} evidence is stale after product or validator changes: {rejected[:8]}",
    )
    return {
        "from_git_sha": recorded_sha,
        "to_git_sha": current["git_sha"],
        "changed_files": allowed,
        "changed_file_count": len(allowed),
        "policy": "control-plane-only",
    }


def validate_dependencies(
    paths: dict[str, Path], current: dict[str, Any]
) -> dict[str, Any]:
    require(set(paths) == set(DEPENDENCY_KEYS), "R0 dependency path set mismatch")
    dependencies: dict[str, Any] = {}
    for key in DEPENDENCY_KEYS:
        path = paths[key].expanduser().resolve()
        source = source_from_outer(path, key)
        closure = source_closure(source, current, key, f"R0 {key}")
        validated = g08a.validate_outer(
            path,
            key,
            g08a.DEPENDENCY_SPECS[key],
            source,
            verify_checkout=False,
        )
        if key == "s2":
            child_path = Path(validated["child_manifest"]["path"])
            deep = s2_contract.verify_checkpoint_manifest(
                child_path,
                verify_checkout=False,
            )
            require(deep.get("source") == source, "R0 S2 deep source binding mismatch")
            bindings = deep.get("bindings")
            require(isinstance(bindings, dict), "R0 S2 deep bindings are missing")
            validated["summary"]["product_execution_identity"] = copy.deepcopy(
                bindings.get("product_execution_identity")
            )
        dependencies[key] = {
            "source": source,
            "source_closure": closure,
            "outer_manifest": validated["outer_manifest"],
            "child_manifest": validated["child_manifest"],
            "summary": validated["summary"],
        }
    require(
        dependencies["source"]["source"] == dependencies["numerics"]["source"],
        "R0 G08A source and numerics evidence must share one source identity",
    )
    return dependencies


def acceptance(dependencies: dict[str, Any]) -> dict[str, Any]:
    source_summary = dependencies["source"]["summary"]["summary"]
    require(source_summary.get("lifecycle_ownership_categories") == 5, "R0 lifecycle category coverage differs")
    require(source_summary.get("lifecycle_implementation_owner_count") == 1, "R0 lifecycle owner count differs")
    require(source_summary.get("legacy_source_selection_count") == 0, "R0 legacy selection is nonzero")
    require(source_summary.get("provider_file_count", 10**9) <= 8, "R0 provider file limit failed")
    require(source_summary.get("provider_glue_production_loc", 10**9) <= 1500, "R0 provider glue limit failed")

    numerics = dependencies["numerics"]["summary"]
    parity = numerics.get("token_parity")
    same_history = numerics.get("same_history")
    require(isinstance(parity, dict) and isinstance(same_history, dict), "R0 numerical summaries are missing")
    require(numerics.get("operation_state_row_count", 0) >= 27, "R0 operation/state coverage is incomplete")
    require(numerics.get("layer_checkpoint_count") == 2, "R0 layer checkpoint coverage differs")
    require(parity.get("case_count") == parity.get("prompt_token_match_count") == 20, "R0 prompt token parity is incomplete")
    require(parity.get("product_output_token_count_per_runtime") == 1280, "R0 token observation denominator differs")
    require(parity.get("exception_count") == parity.get("waiver_count") == 0, "R0 token parity contains an exception or waiver")
    require(same_history.get("case_count") == 20, "R0 same-history prompt denominator differs")
    require(same_history.get("validated_decision_count") == 1280, "R0 same-history decision denominator differs")
    require(same_history.get("ferrum_oracle_exact_count") == 1280, "R0 Ferrum/oracle exact decisions are incomplete")
    require(same_history.get("exception_count") == same_history.get("waiver_count") == 0, "R0 same-history contains an exception or waiver")

    s2 = dependencies["s2"]["summary"]
    s2_acceptance = s2.get("acceptance")
    require(
        isinstance(s2_acceptance, dict)
        and s2_acceptance
        and all(value is True for value in s2_acceptance.values()),
        "R0 CUDA S2 acceptance is incomplete",
    )
    require(s2.get("historical_case_count") == 5, "R0 historical resource case denominator differs")
    require(s2.get("historical_source_test_count") == 7, "R0 historical source test denominator differs")
    product_identity = s2.get("product_execution_identity")
    require(isinstance(product_identity, dict), "R0 CUDA product execution identity is missing")
    require(
        product_identity.get("entrypoints") == ["run", "serve"]
        and product_identity.get("same_resolved_execution_plan") is True
        and product_identity.get("same_runtime_implementation") is True
        and product_identity.get("production_legacy_selection_count") == 0,
        "R0 CUDA run/serve plan or runtime identity failed",
    )
    return {
        "source_ownership": {
            "lifecycle_categories": 5,
            "lifecycle_owner_count": 1,
            "legacy_selection_count": 0,
            "provider_file_count": source_summary["provider_file_count"],
            "provider_glue_production_loc": source_summary["provider_glue_production_loc"],
        },
        "numerics": {
            "operation_state_rows": numerics["operation_state_row_count"],
            "layer_checkpoints": numerics["layer_checkpoint_count"],
            "prompt_token_parity": "20/20",
            "same_history_decisions": "1280/1280",
            "free_run_exact_sequences_diagnostic": f"{parity.get('generated_sequence_match_count')}/20",
            "waiver_count": 0,
            "exception_count": 0,
        },
        "cuda_s2": {
            "acceptance_count": len(s2_acceptance),
            "historical_resource_cases": "5/5",
            "historical_source_tests": 7,
            "product_entrypoints": ["run", "serve"],
            "resolved_execution_plan_hash": product_identity[
                "resolved_execution_plan_hash"
            ],
            "runtime_implementation_fingerprint": product_identity[
                "runtime_implementation_fingerprint"
            ],
            "same_resolved_execution_plan": True,
            "same_runtime_implementation": True,
            "production_legacy_selection_count": 0,
        },
    }


def dependency_checkpoint(dependency: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": copy.deepcopy(dependency["source"]),
        "source_closure": copy.deepcopy(dependency["source_closure"]),
        "outer_manifest": copy.deepcopy(dependency["outer_manifest"]),
        "child_manifest": copy.deepcopy(dependency["child_manifest"]),
        "summary": copy.deepcopy(dependency["summary"]),
    }


def verify_manifest(manifest_path: Path, *, verify_checkout: bool = True) -> dict[str, Any]:
    path = manifest_path.expanduser().resolve()
    manifest = read_json(path, "R0 manifest")
    require(
        set(manifest)
        == {
            "schema_version",
            "artifact_type",
            "checkpoint_id",
            "lane",
            "status",
            "canonical",
            "artifact_dir",
            "source",
            "dependencies",
            "acceptance",
            "unlocks",
            "does_not_prove",
            "created_at",
            "pass_line",
        },
        "R0 manifest field set mismatch",
    )
    root = path.parent
    require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type") == "runtime_vnext_r0_core_closure_manifest"
        and manifest.get("checkpoint_id") == "R0"
        and manifest.get("lane") == "runtime-vnext-r0"
        and manifest.get("status") == "pass"
        and manifest.get("canonical") is True
        and Path(str(manifest.get("artifact_dir", ""))).expanduser().resolve() == root,
        "R0 manifest identity/status mismatch",
    )
    require(manifest.get("pass_line") == f"{PASS_PREFIX}: {root}", "R0 manifest PASS line mismatch")
    source = normalize_source(manifest.get("source"), "R0 current")
    if verify_checkout:
        require(source == current_source(), "R0 aggregate source is stale")
    dependencies = manifest.get("dependencies")
    require(isinstance(dependencies, dict) and set(dependencies) == set(DEPENDENCY_KEYS), "R0 dependency set mismatch")
    for key in DEPENDENCY_KEYS:
        dependency = dependencies[key]
        require(
            isinstance(dependency, dict)
            and set(dependency)
            == {"source", "source_closure", "outer_manifest", "child_manifest", "summary"},
            f"R0 {key} dependency fields differ",
        )
        dependency_source = normalize_source(dependency["source"], f"R0 {key}")
        validate_ref(dependency["outer_manifest"], f"R0 {key} outer manifest")
        validate_ref(dependency["child_manifest"], f"R0 {key} child manifest")
        expected_closure = source_closure(
            dependency_source, source, key, f"R0 {key}"
        )
        require(dependency["source_closure"] == expected_closure, f"R0 {key} source closure mismatch")
    require(manifest.get("acceptance") == acceptance(dependencies), "R0 acceptance summary mismatch")
    require(manifest.get("unlocks") == ["R1"], "R0 unlock set mismatch")
    require(manifest.get("does_not_prove") == DOES_NOT_PROVE, "R0 does_not_prove mismatch")
    return {
        "kind": "vnext-r0",
        "child_manifest": {"path": str(path), "sha256": sha256(path)},
        "source": source,
        "acceptance": copy.deepcopy(manifest["acceptance"]),
    }


def build(paths: dict[str, Path], out: Path) -> str:
    output = out.expanduser().resolve()
    require(REPO_ROOT not in output.parents and output != REPO_ROOT, "R0 output must be outside the source tree")
    require(not output.exists() or not any(output.iterdir()), f"R0 output must be absent or empty: {output}")
    source = current_source()
    dependencies = validate_dependencies(paths, source)
    accepted = acceptance(dependencies)
    staging_parent = output.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=staging_parent))
    try:
        pass_line = f"{PASS_PREFIX}: {output}"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_r0_core_closure_manifest",
            "checkpoint_id": "R0",
            "lane": "runtime-vnext-r0",
            "status": "pass",
            "canonical": True,
            "artifact_dir": str(output),
            "source": source,
            "dependencies": {
                key: dependency_checkpoint(dependencies[key]) for key in DEPENDENCY_KEYS
            },
            "acceptance": accepted,
            "unlocks": ["R1"],
            "does_not_prove": DOES_NOT_PROVE,
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "pass_line": pass_line,
        }
        write_json(staging / "manifest.json", manifest, exclusive=True)
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
        verify_manifest(output / "manifest.json", verify_checkout=True)
        return pass_line
    except BaseException:
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        if output.exists() and output.is_dir() and not any(output.iterdir()):
            output.rmdir()
        raise


def expect_reject(action: Any, marker: str) -> None:
    try:
        action()
    except R0Error:
        return
    raise R0Error(f"self-test mutation was accepted: {marker}")


def fixture_dependencies() -> dict[str, Any]:
    return {
        "source": {
            "summary": {
                "summary": {
                    "lifecycle_ownership_categories": 5,
                    "lifecycle_implementation_owner_count": 1,
                    "legacy_source_selection_count": 0,
                    "provider_file_count": 1,
                    "provider_glue_production_loc": 568,
                }
            }
        },
        "numerics": {
            "summary": {
                "operation_state_row_count": 27,
                "layer_checkpoint_count": 2,
                "token_parity": {
                    "case_count": 20,
                    "prompt_token_match_count": 20,
                    "product_output_token_count_per_runtime": 1280,
                    "generated_sequence_match_count": 19,
                    "exception_count": 0,
                    "waiver_count": 0,
                },
                "same_history": {
                    "case_count": 20,
                    "validated_decision_count": 1280,
                    "ferrum_oracle_exact_count": 1280,
                    "exception_count": 0,
                    "waiver_count": 0,
                },
            }
        },
        "s2": {
            "summary": {
                "acceptance": copy.deepcopy(s2_contract.ACCEPTANCE),
                "historical_case_count": 5,
                "historical_source_test_count": 7,
                "product_execution_identity": {
                    "entrypoints": ["run", "serve"],
                    "resolved_execution_plan_hash": "1" * 64,
                    "runtime_implementation_fingerprint": "2" * 64,
                    "same_resolved_execution_plan": True,
                    "same_runtime_implementation": True,
                    "production_legacy_selection_count": 0,
                },
            }
        },
    }


def self_test() -> int:
    allowed, rejected = control_plane_only(
        [
            "docs/goals/r0.md",
            "scripts/release/runtime_vnext_r0_core_closure.py",
            "scripts/release/change_impact_rules.json",
            "scripts/release/runtime_vnext_s2_cuda_product_contract.py",
            S2_MULTITURN_SCENARIO,
        ],
        "numerics",
    )
    require(len(allowed) == 5 and not rejected, "R0 control-plane closure rejected allowed paths")
    _, rejected = control_plane_only([S2_MULTITURN_SCENARIO], "s2")
    require(
        rejected == [S2_MULTITURN_SCENARIO],
        "R0 S2 closure accepted a changed product scenario",
    )
    _, rejected = control_plane_only(["crates/ferrum-engine/src/lib.rs"], "numerics")
    require(rejected == ["crates/ferrum-engine/src/lib.rs"], "R0 product change was not rejected")
    dependencies = fixture_dependencies()
    accepted = acceptance(dependencies)
    require(
        accepted["numerics"]["same_history_decisions"] == "1280/1280"
        and accepted["cuda_s2"]["historical_resource_cases"] == "5/5",
        "R0 fixture acceptance differs",
    )
    bad = copy.deepcopy(dependencies)
    bad["source"]["summary"]["summary"]["lifecycle_implementation_owner_count"] = 2
    expect_reject(lambda: acceptance(bad), "multiple lifecycle owners")
    bad = copy.deepcopy(dependencies)
    bad["numerics"]["summary"]["same_history"]["ferrum_oracle_exact_count"] = 1279
    expect_reject(lambda: acceptance(bad), "numerical decision miss")
    bad = copy.deepcopy(dependencies)
    bad["s2"]["summary"]["acceptance"]["run_and_serve_product_paths_covered"] = False
    expect_reject(lambda: acceptance(bad), "S2 product path miss")
    source = {
        "git_sha": git_text("rev-parse", "HEAD"),
        "git_tree_sha": git_text("rev-parse", "HEAD^{tree}"),
        "dirty": False,
    }
    with tempfile.TemporaryDirectory(prefix="ferrum-r0-selftest-") as temporary:
        root = Path(temporary).resolve()
        sealed_dependencies = fixture_dependencies()
        for key in DEPENDENCY_KEYS:
            outer = root / f"{key}.outer.json"
            child = root / f"{key}.child.json"
            write_json(outer, {"key": key, "kind": "outer"})
            write_json(child, {"key": key, "kind": "child"})
            sealed_dependencies[key].update(
                {
                    "source": source,
                    "source_closure": source_closure(source, source, key, key),
                    "outer_manifest": file_ref(outer),
                    "child_manifest": file_ref(child),
                }
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_r0_core_closure_manifest",
            "checkpoint_id": "R0",
            "lane": "runtime-vnext-r0",
            "status": "pass",
            "canonical": True,
            "artifact_dir": str(root),
            "source": source,
            "dependencies": sealed_dependencies,
            "acceptance": acceptance(sealed_dependencies),
            "unlocks": ["R1"],
            "does_not_prove": DOES_NOT_PROVE,
            "created_at": "2026-08-06T00:00:00+00:00",
            "pass_line": f"{PASS_PREFIX}: {root}",
        }
        write_json(root / "manifest.json", manifest)
        verified = verify_manifest(root / "manifest.json", verify_checkout=False)
        require(verified.get("kind") == "vnext-r0", "R0 manifest fixture failed")
        manifest["acceptance"]["cuda_s2"]["production_legacy_selection_count"] = 1
        write_json(root / "manifest.json", manifest)
        expect_reject(
            lambda: verify_manifest(root / "manifest.json", verify_checkout=False),
            "manifest acceptance mutation",
        )
    print(SELFTEST_PASS_LINE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g08a-source", type=Path)
    parser.add_argument("--g08a-numerics", type=Path)
    parser.add_argument("--s2", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            return self_test()
        required = {
            "--g08a-source": args.g08a_source,
            "--g08a-numerics": args.g08a_numerics,
            "--s2": args.s2,
            "--out": args.out,
        }
        missing = [flag for flag, value in required.items() if value is None]
        require(not missing, "missing required arguments: " + ", ".join(missing))
        assert args.g08a_source is not None
        assert args.g08a_numerics is not None
        assert args.s2 is not None
        assert args.out is not None
        pass_line = build(
            {
                "source": args.g08a_source,
                "numerics": args.g08a_numerics,
                "s2": args.s2,
            },
            args.out,
        )
        print(pass_line)
        return 0
    except (OSError, R0Error, RuntimeError, ValueError) as error:
        print(f"FERRUM RUNTIME VNEXT R0 CORE CLOSURE FAIL: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
