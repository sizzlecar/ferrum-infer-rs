#!/usr/bin/env python3
"""Validate that the vNext CUDA release graph does not enable Candle CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtime_vnext_plan_reference import FEATURES as CUDA_RELEASE_FEATURES


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "runtime-vnext-cuda-candle-boundary"
PASS_PREFIX = "FERRUM RUNTIME VNEXT CUDA CANDLE BOUNDARY PASS"
SELFTEST_PASS = "FERRUM RUNTIME VNEXT CUDA CANDLE BOUNDARY SELFTEST PASS"
WORKSPACE_PACKAGES = (
    "ferrum-cli",
    "ferrum-engine",
    "ferrum-models",
    "ferrum-kernels",
)
CANDLE_PACKAGES = ("candle-core", "candle-nn", "candle-kernels")


class BoundaryError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BoundaryError(message)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_ref(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def package_rows(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    relevant_names = {*WORKSPACE_PACKAGES, *CANDLE_PACKAGES}
    packages = {
        package["id"]: package
        for package in metadata.get("packages", [])
        if isinstance(package, dict) and isinstance(package.get("id"), str)
    }
    rows: dict[str, dict[str, Any]] = {}
    resolve = metadata.get("resolve")
    require(isinstance(resolve, dict), "cargo metadata is missing resolve")
    nodes = resolve.get("nodes")
    require(isinstance(nodes, list), "cargo metadata resolve.nodes must be a list")
    for node in nodes:
        require(isinstance(node, dict), "cargo metadata node must be an object")
        package = packages.get(node.get("id"))
        require(package is not None, f"cargo metadata node package is missing: {node.get('id')}")
        name = package.get("name")
        require(isinstance(name, str) and name, "cargo package name must be non-empty")
        if name not in relevant_names:
            continue
        features = node.get("features")
        require(
            isinstance(features, list)
            and all(isinstance(feature, str) for feature in features),
            f"cargo metadata features must be strings for {name}",
        )
        if name in rows:
            raise BoundaryError(f"resolved graph contains duplicate package name: {name}")
        rows[name] = {
            "name": name,
            "version": package.get("version"),
            "source": package.get("source"),
            "features": sorted(features),
        }
    return rows


def validate_official_graph(metadata: dict[str, Any]) -> dict[str, Any]:
    rows = package_rows(metadata)
    for name in WORKSPACE_PACKAGES:
        require(name in rows, f"official CUDA graph is missing {name}")
        require(
            "candle-cuda-compat" not in rows[name]["features"],
            f"official CUDA graph unexpectedly enables {name}/candle-cuda-compat",
        )
    require(
        "candle-kernels" not in rows,
        "official CUDA graph must not resolve candle-kernels",
    )
    for name in ("candle-core", "candle-nn"):
        require(name in rows, f"official CUDA graph is missing base {name}")
        require(
            "cuda" not in rows[name]["features"],
            f"official CUDA graph unexpectedly enables {name}/cuda",
        )
    kernels = rows["ferrum-kernels"]["features"]
    require("cuda" in kernels, "official CUDA graph is missing ferrum-kernels/cuda")
    for feature in ("vllm-moe-marlin", "vllm-paged-attn-v2"):
        require(feature in kernels, f"official CUDA graph is missing ferrum-kernels/{feature}")
    return {name: rows[name] for name in (*WORKSPACE_PACKAGES, "candle-core", "candle-nn")}


def validate_compat_graph(metadata: dict[str, Any]) -> dict[str, Any]:
    rows = package_rows(metadata)
    require("candle-kernels" in rows, "compat CUDA graph must resolve candle-kernels")
    for name in WORKSPACE_PACKAGES:
        require(name in rows, f"compat CUDA graph is missing {name}")
        require(
            "candle-cuda-compat" in rows[name]["features"],
            f"compat CUDA graph is missing {name}/candle-cuda-compat",
        )
    for name in ("candle-core", "candle-nn"):
        require(
            name in rows and "cuda" in rows[name]["features"],
            f"compat CUDA graph is missing {name}/cuda",
        )
    return {name: rows[name] for name in (*WORKSPACE_PACKAGES, *CANDLE_PACKAGES)}


def validate_feature_declarations(
    source_root: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    packages = metadata.get("packages")
    require(isinstance(packages, list), "cargo metadata packages must be a list")
    by_name: dict[str, dict[str, Any]] = {}
    for package in packages:
        require(isinstance(package, dict), "cargo metadata package must be an object")
        name = package.get("name")
        if name not in WORKSPACE_PACKAGES:
            continue
        require(name not in by_name, f"cargo metadata contains duplicate workspace package {name}")
        by_name[name] = package

    result = {}
    for name in WORKSPACE_PACKAGES:
        package = by_name.get(name)
        require(package is not None, f"cargo metadata is missing workspace package {name}")
        manifest_path = package.get("manifest_path")
        require(
            isinstance(manifest_path, str) and manifest_path,
            f"{name} cargo metadata is missing manifest_path",
        )
        path = Path(manifest_path).resolve()
        try:
            relative_manifest = path.relative_to(source_root)
        except ValueError as error:
            raise BoundaryError(
                f"{name} manifest escapes source root: {manifest_path}"
            ) from error
        features = package.get("features")
        require(isinstance(features, dict), f"{name} manifest is missing features")
        cuda = features.get("cuda")
        compat = features.get("candle-cuda-compat")
        require(isinstance(cuda, list), f"{name}/cuda must be a feature list")
        require(
            isinstance(compat, list),
            f"{name}/candle-cuda-compat must be a feature list",
        )
        require(
            all("candle-core/cuda" not in item and "candle-nn/cuda" not in item for item in cuda),
            f"{name}/cuda directly enables Candle CUDA",
        )
        require(
            any("candle-cuda-compat" in item or "/cuda" in item for item in compat),
            f"{name}/candle-cuda-compat does not forward a CUDA compatibility feature",
        )
        result[name] = {
            "manifest": relative_manifest.as_posix(),
            "cuda": cuda,
            "candle_cuda_compat": compat,
        }
    return result


def run_metadata(source_root: Path, features: list[str], out: Path) -> dict[str, Any]:
    feature_arg = ",".join(f"ferrum-cli/{feature}" for feature in features)
    command = [
        "cargo",
        "metadata",
        "--format-version",
        "1",
        "--locked",
        "--no-default-features",
        "--features",
        feature_arg,
    ]
    completed = subprocess.run(
        command,
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    write_json(
        out / "command.json",
        {"argv": command, "cwd": str(source_root), "returncode": completed.returncode},
    )
    require(
        completed.returncode == 0,
        f"cargo metadata failed for features {features}: {completed.stderr.strip()}",
    )
    try:
        metadata = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise BoundaryError(f"cargo metadata returned invalid JSON: {error}") from error
    require(isinstance(metadata, dict), "cargo metadata root must be an object")
    write_json(out / "metadata.json", metadata)
    return metadata


def git_value(source_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_gate(source_root: Path, out: Path, allow_dirty: bool) -> None:
    source_root = source_root.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dirty = git_value(source_root, "status", "--short")
    require(allow_dirty or not dirty, "source worktree is dirty")
    official = run_metadata(source_root, list(CUDA_RELEASE_FEATURES), out / "official")
    compat = run_metadata(
        source_root,
        [*CUDA_RELEASE_FEATURES, "candle-cuda-compat"],
        out / "compat",
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "pass",
        "captured_at": now_iso(),
        "source_git_sha": git_value(source_root, "rev-parse", "HEAD"),
        "source_tree_sha": git_value(source_root, "write-tree"),
        "source_dirty": bool(dirty),
        "official_features": list(CUDA_RELEASE_FEATURES),
        "official_graph": validate_official_graph(official),
        "compat_features": [*CUDA_RELEASE_FEATURES, "candle-cuda-compat"],
        "compat_graph": validate_compat_graph(compat),
        "feature_declarations": validate_feature_declarations(source_root, official),
    }
    report_path = out / "report.json"
    write_json(report_path, report)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "pass",
        "source_git_sha": report["source_git_sha"],
        "source_tree_sha": report["source_tree_sha"],
        "source_dirty": report["source_dirty"],
        "report": file_ref(report_path, out),
        "official_metadata": file_ref(out / "official/metadata.json", out),
        "compat_metadata": file_ref(out / "compat/metadata.json", out),
        "pass_line": f"{PASS_PREFIX}: {out}",
    }
    write_json(out / "manifest.json", manifest)
    print(manifest["pass_line"])


def fake_metadata(*, compat: bool) -> dict[str, Any]:
    packages = []
    nodes = []
    names = [*WORKSPACE_PACKAGES, "candle-core", "candle-nn"]
    if compat:
        names.append("candle-kernels")
    for index, name in enumerate(names):
        package_id = f"{name} 1.0.0 (path+file:///fixture/{index})"
        declared_features: dict[str, list[str]] = {}
        if name in WORKSPACE_PACKAGES:
            declared_features = {
                "cuda": ["dep:cudarc"],
                "candle-cuda-compat": ["cuda", "candle-core/cuda"],
            }
        packages.append(
            {
                "id": package_id,
                "name": name,
                "version": "1.0.0",
                "source": None,
                "manifest_path": f"/fixture/crates/{name}/Cargo.toml",
                "features": declared_features,
            }
        )
        features = ["default"]
        if name in WORKSPACE_PACKAGES:
            features.append("cuda")
            if compat:
                features.append("candle-cuda-compat")
        if name == "ferrum-kernels":
            features.extend(["vllm-moe-marlin", "vllm-paged-attn-v2"])
        if compat and name in ("candle-core", "candle-nn"):
            features.append("cuda")
        nodes.append({"id": package_id, "features": features})
    return {"packages": packages, "resolve": {"nodes": nodes}}


def self_test() -> None:
    official = fake_metadata(compat=False)
    validate_official_graph(official)
    validate_compat_graph(fake_metadata(compat=True))
    validate_feature_declarations(Path("/fixture"), official)
    bad = fake_metadata(compat=False)
    candle_core = next(
        node for node in bad["resolve"]["nodes"] if node["id"].startswith("candle-core ")
    )
    candle_core["features"].append("cuda")
    try:
        validate_official_graph(bad)
    except BoundaryError as error:
        require("candle-core/cuda" in str(error), f"unexpected mutation error: {error}")
    else:
        raise AssertionError("official Candle CUDA mutation unexpectedly passed")
    bad_compat = fake_metadata(compat=True)
    bad_compat["packages"] = [
        package
        for package in bad_compat["packages"]
        if package["name"] != "candle-kernels"
    ]
    bad_compat["resolve"]["nodes"] = [
        node
        for node in bad_compat["resolve"]["nodes"]
        if not node["id"].startswith("candle-kernels ")
    ]
    try:
        validate_compat_graph(bad_compat)
    except BoundaryError as error:
        require("candle-kernels" in str(error), f"unexpected compat mutation error: {error}")
    else:
        raise AssertionError("missing compat Candle kernels unexpectedly passed")
    bad_declaration = fake_metadata(compat=False)
    kernels_package = next(
        package
        for package in bad_declaration["packages"]
        if package["name"] == "ferrum-kernels"
    )
    kernels_package["features"]["cuda"].append("candle-core/cuda")
    try:
        validate_feature_declarations(Path("/fixture"), bad_declaration)
    except BoundaryError as error:
        require(
            "directly enables Candle CUDA" in str(error),
            f"unexpected declaration mutation error: {error}",
        )
    else:
        raise AssertionError("Candle CUDA feature declaration mutation unexpectedly passed")
    print(SELFTEST_PASS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        require(args.out is not None, "--out is required unless --self-test is used")
        run_gate(args.source_root, args.out, args.allow_dirty)
        return 0
    except (BoundaryError, OSError, subprocess.SubprocessError) as error:
        print(f"FERRUM RUNTIME VNEXT CUDA CANDLE BOUNDARY REJECT: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
