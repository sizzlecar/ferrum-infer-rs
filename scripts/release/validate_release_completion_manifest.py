#!/usr/bin/env python3
"""Validate the final v0.8.0 release-completion manifest.

Live GitHub/crates.io/Homebrew reads belong to their canonical release lanes.
This validator fail-closes over those lane manifests, their shared release/tag
identity, the promoted asset identities, and the complete workspace crate set.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import re
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "version",
    "git_sha",
    "git_tree_sha",
    "dirty_status",
    "tag",
    "release_id",
    "github_release_url",
    "github_release",
    "release_assets",
    "staged_assets_manifest",
    "unit_source_gate_artifact",
    "metal_tarball_gate_artifact",
    "cuda_tarball_gate_artifact",
    "homebrew_metal_gate_artifact",
    "homebrew_cuda_fetch_gate_artifact",
    "workflow_policy_gate_artifact",
    "g10a_gate_artifact",
    "g08_rc_gate_artifact",
    "g09_rc_gate_artifact",
    "metal_three_model_gate_artifact",
    "cuda_three_model_gate_artifact",
    "published_assets_gate_artifact",
    "crates_io_gate_artifact",
    "prepromotion_gate_artifact",
    "g10b_gate_artifact",
    "release_summary_artifact",
    "promotion_receipt",
    "cargo_workspace_crates",
}

EXPECTED_CRATES = {
    "ferrum-bench-core",
    "ferrum-cli",
    "ferrum-engine",
    "ferrum-interfaces",
    "ferrum-kernels",
    "ferrum-kv",
    "ferrum-models",
    "ferrum-native-ops",
    "ferrum-native-ops-builder",
    "ferrum-quantization",
    "ferrum-sampler",
    "ferrum-scheduler",
    "ferrum-server",
    "ferrum-testkit",
    "ferrum-tokenizer",
    "ferrum-types",
}

EXPECTED_ASSETS = {
    "ferrum-linux-x86_64.tar.gz",
    "ferrum-linux-x86_64-cuda-sm89.tar.gz",
    "ferrum-macos-aarch64.tar.gz",
}

GATE_FIELDS = {
    "unit_source_gate_artifact": ("unit", "FERRUM GATE unit PASS: "),
    "metal_tarball_gate_artifact": ("metal-tarball", "METAL TARBALL GATE PASS: "),
    "cuda_tarball_gate_artifact": ("cuda-tarball", "CUDA TARBALL GATE PASS: "),
    "homebrew_metal_gate_artifact": ("homebrew-metal", "HOMEBREW METAL GATE PASS: "),
    "homebrew_cuda_fetch_gate_artifact": ("homebrew-cuda-fetch", "HOMEBREW CUDA FETCH GATE PASS: "),
    "workflow_policy_gate_artifact": (None, "FERRUM RELEASE WORKFLOW POLICY PASS: "),
    "g10a_gate_artifact": ("vnext-g10a", "FERRUM GATE vnext-g10a PASS: "),
    "g08_rc_gate_artifact": ("vnext-g08-rc", "FERRUM GATE vnext-g08-rc PASS: "),
    "g09_rc_gate_artifact": ("vnext-g09-rc", "FERRUM GATE vnext-g09-rc PASS: "),
    "metal_three_model_gate_artifact": (
        "runtime-vnext-metal-three-model",
        "FERRUM GATE runtime-vnext-metal-three-model PASS: ",
    ),
    "cuda_three_model_gate_artifact": (
        "runtime-vnext-cuda-three-model",
        "FERRUM GATE runtime-vnext-cuda-three-model PASS: ",
    ),
    "published_assets_gate_artifact": (
        "runtime-vnext-published-assets",
        "FERRUM GATE runtime-vnext-published-assets PASS: ",
    ),
    "crates_io_gate_artifact": (None, "FERRUM CRATES IO V0.8.0 PASS: "),
    "prepromotion_gate_artifact": (
        "runtime-vnext-prepromotion",
        "FERRUM GATE runtime-vnext-prepromotion PASS: ",
    ),
    "g10b_gate_artifact": ("vnext-g10b", "FERRUM GATE vnext-g10b PASS: "),
}

STRICT_GOAL_FIELDS = {
    "g10a_gate_artifact": "vnext-g10a",
    "g08_rc_gate_artifact": "vnext-g08-rc",
    "g09_rc_gate_artifact": "vnext-g09-rc",
    "metal_three_model_gate_artifact": "runtime-vnext-metal-three-model",
    "cuda_three_model_gate_artifact": "runtime-vnext-cuda-three-model",
    "published_assets_gate_artifact": "runtime-vnext-published-assets",
    "prepromotion_gate_artifact": "runtime-vnext-prepromotion",
    "g10b_gate_artifact": "vnext-g10b",
}

ASSET_BACKENDS = {
    "cpu": "ferrum-linux-x86_64.tar.gz",
    "cuda": "ferrum-linux-x86_64-cuda-sm89.tar.gz",
    "metal": "ferrum-macos-aarch64.tar.gz",
}


class ValidationError(Exception):
    pass


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {path}: {exc}") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_ref(where: str, raw: Any, root: Path) -> tuple[dict[str, Any], Path]:
    if not isinstance(raw, dict) or set(raw) != {"path", "sha256", "size_bytes"}:
        raise ValidationError(f"{where} must be a path/SHA256/size reference")
    raw_path = require_non_empty_string(f"{where}.path", raw.get("path"))
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    if path.is_symlink():
        raise ValidationError(f"{where} must not be a symlink: {path}")
    path = path.resolve()
    if not path.is_file():
        raise ValidationError(f"{where} is not a regular file: {path}")
    validate_sha256(f"{where}.sha256", raw.get("sha256"))
    size = raw.get("size_bytes")
    if type(size) is not int or size < 0 or path.stat().st_size != size:
        raise ValidationError(f"{where}.size_bytes differs")
    if file_sha256(path) != raw["sha256"]:
        raise ValidationError(f"{where}.sha256 differs")
    return copy.deepcopy(raw), path


def same_ref_sha(where: str, raw: Any, expected_sha256: str) -> None:
    if not isinstance(raw, dict) or raw.get("sha256") != expected_sha256:
        raise ValidationError(f"{where} does not bind the canonical artifact")


def require_non_empty_string(where: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{where} must be a non-empty string")
    return value


def validate_sha256(where: str, value: Any) -> None:
    text = require_non_empty_string(where, value)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", text):
        raise ValidationError(f"{where} must be a 64-character SHA256 hex digest")


def resolve_artifact(where: str, value: Any, manifest_dir: Path) -> Path:
    path = Path(require_non_empty_string(where, value))
    if not path.is_absolute():
        path = manifest_dir / path
    if path.is_symlink():
        raise ValidationError(f"{where} must not be a symlink: {path}")
    path = path.resolve()
    if not path.exists():
        raise ValidationError(f"{where} does not exist: {path}")
    return path


def validate_gate_artifact(
    where: str,
    value: Any,
    manifest_dir: Path,
    *,
    expected_lane: str | None,
    expected_pass_prefix: str,
) -> str:
    path = resolve_artifact(where, value, manifest_dir)
    if path.is_dir():
        candidates = [path / "gate.manifest.json", path / "gate.json"]
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not path.is_file():
        raise ValidationError(f"{where} must identify a gate manifest file: {path}")
    data = load_json(path)
    if not isinstance(data, dict) or data.get("status") != "pass":
        raise ValidationError(f"{where} is not a PASS manifest: {path}")
    if expected_lane is not None and data.get("lane") != expected_lane:
        raise ValidationError(f"{where} lane differs: {data.get('lane')!r}")
    pass_lines = [data.get("pass_line"), data.get("child_pass_line")]
    if not any(
        isinstance(line, str) and line.startswith(expected_pass_prefix)
        for line in pass_lines
    ):
        raise ValidationError(f"{where} required PASS line is absent: {path}")
    return str(path)


def validate_goal_outer_child(
    where: str,
    outer_path: Path,
    *,
    expected_lane: str,
) -> dict[str, Any]:
    """Authenticate a canonical outer to its adjacent strict child manifest."""

    if outer_path.name != "gate.manifest.json":
        raise ValidationError(f"{where} must identify canonical gate.manifest.json")
    outer = load_json(outer_path)
    root = outer_path.parent.resolve()
    if (
        not isinstance(outer, dict)
        or outer.get("schema_version") != 1
        or outer.get("status") != "pass"
        or outer.get("lane") != expected_lane
        or outer.get("child_returncode") != 0
        or outer.get("artifact_dir") != str(root)
        or outer.get("pass_line") != f"FERRUM GATE {expected_lane} PASS: {root}"
    ):
        raise ValidationError(f"{where} canonical outer identity/status differs")
    child_artifacts = outer.get("child_artifacts")
    if not isinstance(child_artifacts, dict):
        raise ValidationError(f"{where} lacks canonical child_artifacts")
    if child_artifacts.get("kind") != expected_lane:
        raise ValidationError(f"{where} child kind differs")
    child_ref = child_artifacts.get("child_manifest")
    if not isinstance(child_ref, dict):
        raise ValidationError(f"{where} lacks child manifest reference")
    child_path = root / "manifest.json"
    if child_path.is_symlink() or not child_path.is_file():
        raise ValidationError(f"{where} adjacent child manifest is missing")
    validate_sha256(f"{where}.child.sha256", child_ref.get("sha256"))
    if (
        type(child_ref.get("size_bytes")) is not int
        or child_ref["size_bytes"] != child_path.stat().st_size
        or child_ref["sha256"] != file_sha256(child_path)
    ):
        raise ValidationError(f"{where} outer-to-child byte binding differs")
    child = load_json(child_path)
    if (
        not isinstance(child, dict)
        or child.get("lane") != expected_lane
        or child.get("status") != "pass"
        or child.get("pass_line") != outer.get("child_pass_line")
    ):
        raise ValidationError(f"{where} child identity/PASS differs")
    try:
        import runtime_vnext_goal_gate as goal_gate

        if expected_lane == "runtime-vnext-prepromotion":
            # The completion validator independently replays the immutable
            # prepromotion identity/closure below.  Do not make final local
            # completion depend on a second live crates.io validator call.
            verified = validate_prepromotion_child(child_path)
        else:
            verified = goal_gate.verify_goal_manifest(
                child_path,
                expected_lane=expected_lane,
                verify_checkout=False,
            )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValidationError(f"{where} strict child validation failed: {exc}") from exc
    source = verified.get("source")
    if expected_lane == "runtime-vnext-prepromotion":
        recorded_source = child_artifacts.get("source")
        if (
            not isinstance(recorded_source, dict)
            or set(recorded_source) != {"git_sha", "git_tree_sha", "dirty"}
            or recorded_source.get("git_sha") != child.get("release_candidate_sha")
            or recorded_source.get("dirty") is not False
            or not isinstance(recorded_source.get("git_tree_sha"), str)
            or re.fullmatch(r"[0-9a-f]{40}", recorded_source["git_tree_sha"])
            is None
        ):
            raise ValidationError(f"{where} prepromotion source binding differs")
        source = copy.deepcopy(recorded_source)
        verified["source"] = copy.deepcopy(source)
    if (
        not isinstance(source, dict)
        or set(source) != {"git_sha", "git_tree_sha", "dirty"}
        or source.get("dirty") is not False
        or child_artifacts.get("source") != source
    ):
        raise ValidationError(f"{where} child source binding differs")
    return {
        "outer_path": outer_path,
        "outer": outer,
        "child_path": child_path,
        "child_sha256": child_ref["sha256"],
        "child": child,
        "verified": verified,
        "source": source,
    }


def validate_prepromotion_child(path: Path) -> dict[str, Any]:
    value = load_json(path)
    root = path.parent.resolve()
    fields = {
        "schema_version",
        "artifact_type",
        "status",
        "lane",
        "version",
        "canonical",
        "artifact_dir",
        "manifest_id",
        "release_candidate_sha",
        "pass_line",
        "prepromotion_pass_line",
        "release",
        "consumption",
        "dependencies",
        "created_at",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValidationError("prepromotion child fields differ")
    rc_sha = value.get("release_candidate_sha")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != "runtime_vnext_prepromotion_manifest"
        or value.get("status") != "pass"
        or value.get("lane") != "runtime-vnext-prepromotion"
        or value.get("version") != "0.8.0"
        or value.get("canonical") is not True
        or value.get("artifact_dir") != str(root)
        or value.get("pass_line") != f"FERRUM V0.8.0 PREPROMOTION PASS: {root}"
        or value.get("prepromotion_pass_line") != value.get("pass_line")
        or not isinstance(rc_sha, str)
        or re.fullmatch(r"[0-9a-f]{40}", rc_sha) is None
    ):
        raise ValidationError("prepromotion child identity/status differs")
    release = value.get("release")
    if (
        not isinstance(release, dict)
        or set(release)
        != {"id", "tag_name", "tag_sha", "draft", "prerelease", "asset_set_sha256"}
        or not isinstance(release.get("id"), str)
        or not release["id"]
        or release.get("tag_name") != "v0.8.0"
        or release.get("tag_sha") != rc_sha
        or release.get("draft") is not False
        or release.get("prerelease") is not True
    ):
        raise ValidationError("prepromotion release identity differs")
    validate_sha256("prepromotion release asset_set_sha256", release.get("asset_set_sha256"))
    consumption = value.get("consumption")
    if (
        not isinstance(consumption, dict)
        or set(consumption)
        != {"state", "release_id", "token", "consumed_at", "consumed_by"}
        or consumption.get("state") != "unconsumed"
        or consumption.get("release_id") != release["id"]
        or re.fullmatch(r"[A-Za-z0-9._-]{32,}", str(consumption.get("token", "")))
        is None
        or consumption.get("consumed_at") is not None
        or consumption.get("consumed_by") is not None
    ):
        raise ValidationError("prepromotion consumption identity differs")
    dependencies = value.get("dependencies")
    expected_dependencies = {
        "published_assets",
        "crates_io",
        "homebrew_metal",
        "homebrew_cuda_fetch",
        "workflow_policy",
    }
    prefixes = {
        "published_assets": "FERRUM RUNTIME VNEXT PUBLISHED ASSETS PASS: ",
        "crates_io": "FERRUM CRATES IO V0.8.0 PASS: ",
        "homebrew_metal": "HOMEBREW METAL GATE PASS: ",
        "homebrew_cuda_fetch": "HOMEBREW CUDA FETCH GATE PASS: ",
        "workflow_policy": "FERRUM RELEASE WORKFLOW POLICY PASS: ",
    }
    if not isinstance(dependencies, dict) or set(dependencies) != expected_dependencies:
        raise ValidationError("prepromotion dependency denominator differs")
    dependency_paths: dict[str, Path] = {}
    for name in sorted(expected_dependencies):
        row = dependencies.get(name)
        if (
            not isinstance(row, dict)
            or set(row) != {"status", "pass_line", "manifest"}
            or row.get("status") != "pass"
            or not str(row.get("pass_line", "")).startswith(prefixes[name])
        ):
            raise ValidationError(f"prepromotion {name} dependency differs")
        _, dependency_paths[name] = require_ref(
            f"prepromotion {name} manifest", row.get("manifest"), root
        )
    identity_payload = {
        "schema_version": 1,
        "lane": "runtime-vnext-prepromotion",
        "release_candidate_sha": rc_sha,
        "release": release,
        "consumption": consumption,
        "dependencies": dependencies,
    }
    if value.get("manifest_id") != canonical_json_sha256(identity_payload):
        raise ValidationError("prepromotion manifest_id does not bind immutable payload")
    return {
        "kind": "runtime-vnext-prepromotion",
        "path": path,
        "manifest": value,
        "source": {
            "git_sha": rc_sha,
            # Filled and cross-checked from the canonical published child by
            # validate_manifest; this placeholder is never trusted alone.
            "git_tree_sha": None,
            "dirty": False,
        },
        "dependencies": dependency_paths,
    }


def validate_release_assets(data: dict[str, Any]) -> None:
    assets = data["release_assets"]
    if not isinstance(assets, list) or not assets:
        raise ValidationError("release_assets must be a non-empty list")
    names: set[str] = set()
    ids: set[int] = set()
    for idx, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise ValidationError(f"release_assets[{idx}] must be an object")
        name = require_non_empty_string(f"release_assets[{idx}].name", asset.get("name"))
        if name in names:
            raise ValidationError(f"duplicate release asset name: {name}")
        names.add(name)
        asset_id = asset.get("id")
        if not isinstance(asset_id, int) or isinstance(asset_id, bool) or asset_id <= 0:
            raise ValidationError(f"release_assets[{idx}].id must be a positive integer")
        if asset_id in ids:
            raise ValidationError(f"duplicate release asset id: {asset_id}")
        ids.add(asset_id)
        size = asset.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValidationError(f"release_assets[{idx}].size_bytes must be positive")
        validate_sha256(f"release_assets[{idx}].sha256", asset.get("sha256"))
        validate_sha256(
            f"release_assets[{idx}].staged_sha256", asset.get("staged_sha256")
        )
        validate_sha256(
            f"release_assets[{idx}].binary_sha256", asset.get("binary_sha256")
        )
        if asset["sha256"] != asset["staged_sha256"]:
            raise ValidationError(f"release_assets[{idx}] differs from staged bytes")
    if names != EXPECTED_ASSETS:
        raise ValidationError(
            f"release asset set differs: missing={sorted(EXPECTED_ASSETS - names)} "
            f"extra={sorted(names - EXPECTED_ASSETS)}"
        )


def validate_crates(data: dict[str, Any]) -> None:
    crates = data["cargo_workspace_crates"]
    if not isinstance(crates, list) or not crates:
        raise ValidationError("cargo_workspace_crates must be a non-empty list")
    names: set[str] = set()
    for idx, crate in enumerate(crates):
        if not isinstance(crate, dict):
            raise ValidationError(f"cargo_workspace_crates[{idx}] must be an object")
        name = require_non_empty_string(f"cargo_workspace_crates[{idx}].name", crate.get("name"))
        if name in names:
            raise ValidationError(f"duplicate cargo workspace crate: {name}")
        names.add(name)
        version = require_non_empty_string(
            f"cargo_workspace_crates[{idx}].version", crate.get("version")
        )
        if version != "0.8.0":
            raise ValidationError(f"cargo_workspace_crates[{idx}].version must be 0.8.0")
        visible = crate.get("crates_io_visible")
        if visible is not True:
            raise ValidationError(
                f"cargo_workspace_crates[{idx}].crates_io_visible must be true"
            )
    if names != EXPECTED_CRATES:
        raise ValidationError(
            f"cargo workspace crate set differs: missing={sorted(EXPECTED_CRATES - names)} "
            f"extra={sorted(names - EXPECTED_CRATES)}"
        )


def validate_github_release(data: dict[str, Any]) -> None:
    release = data["github_release"]
    if not isinstance(release, dict):
        raise ValidationError("github_release must be an object")
    required = {
        "id",
        "tag",
        "target_git_sha",
        "draft",
        "prerelease",
        "published_at",
        "url",
    }
    missing = sorted(required - set(release))
    if missing:
        raise ValidationError(f"github_release missing fields: {', '.join(missing)}")
    if release.get("id") != data["release_id"]:
        raise ValidationError("github_release.id differs from release_id")
    if release.get("tag") != "v0.8.0" or release.get("tag") != data["tag"]:
        raise ValidationError("github_release tag differs")
    if release.get("target_git_sha") != data["git_sha"]:
        raise ValidationError("GitHub release target differs from release candidate")
    if release.get("draft") is not False or release.get("prerelease") is not False:
        raise ValidationError("GitHub release is not promoted and public")
    if release.get("url") != data["github_release_url"]:
        raise ValidationError("GitHub release URL differs")
    require_non_empty_string("github_release.published_at", release.get("published_at"))


def validate_manifest(path: Path, out_dir: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValidationError("manifest must be an object")
    missing = sorted(REQUIRED_FIELDS - set(data))
    if missing:
        raise ValidationError(f"manifest missing fields: {', '.join(missing)}")
    if data.get("version") != "0.8.0":
        raise ValidationError("version must be 0.8.0")
    git_sha = require_non_empty_string("git_sha", data["git_sha"])
    git_tree_sha = require_non_empty_string("git_tree_sha", data["git_tree_sha"])
    if not re.fullmatch(r"[0-9a-f]{40}", git_sha):
        raise ValidationError("git_sha must be a 40-character lowercase Git SHA")
    if not re.fullmatch(r"[0-9a-f]{40}", git_tree_sha):
        raise ValidationError("git_tree_sha must be a 40-character lowercase Git tree SHA")
    dirty = data["dirty_status"]
    if not isinstance(dirty, dict) or "is_dirty" not in dirty:
        raise ValidationError("dirty_status must include is_dirty")
    if dirty.get("is_dirty") is not False or dirty.get("status_short") not in ([], None):
        raise ValidationError("release source must be clean")
    if data.get("tag") != "v0.8.0":
        raise ValidationError("tag must be v0.8.0")
    release_id = data.get("release_id")
    if not isinstance(release_id, int) or isinstance(release_id, bool) or release_id <= 0:
        raise ValidationError("release_id must be a positive integer")
    require_non_empty_string("github_release_url", data["github_release_url"])
    validate_github_release(data)
    validate_release_assets(data)
    manifest_dir = path.parent
    artifacts: dict[str, str] = {}
    artifact_paths: dict[str, Path] = {}
    for field, contract in GATE_FIELDS.items():
        rendered = validate_gate_artifact(
            field,
            data[field],
            manifest_dir,
            expected_lane=contract[0],
            expected_pass_prefix=contract[1],
        )
        artifacts[field] = rendered
        artifact_paths[field] = Path(rendered)

    goals = {
        field: validate_goal_outer_child(
            field,
            artifact_paths[field],
            expected_lane=lane,
        )
        for field, lane in STRICT_GOAL_FIELDS.items()
    }
    expected_source = {
        "git_sha": git_sha,
        "git_tree_sha": git_tree_sha,
        "dirty": False,
    }
    for field, goal in goals.items():
        if goal["source"] != expected_source:
            raise ValidationError(f"{field} release-candidate source differs")
        outer = goal["outer"]
        if outer.get("git_sha") != git_sha:
            raise ValidationError(f"{field} outer git SHA differs")
        outer_dirty = outer.get("dirty_status")
        if not isinstance(outer_dirty, dict) or outer_dirty.get("is_dirty") is not False:
            raise ValidationError(f"{field} outer source is dirty")
    for field, contract in GATE_FIELDS.items():
        if field in STRICT_GOAL_FIELDS or field == "crates_io_gate_artifact":
            continue
        document = load_json(artifact_paths[field])
        if not isinstance(document, dict):
            raise ValidationError(f"{field} manifest is not an object")
        if contract[0] is not None:
            dirty_status = document.get("dirty_status")
            if (
                document.get("git_sha") != git_sha
                or not isinstance(dirty_status, dict)
                or dirty_status.get("is_dirty") is not False
            ):
                raise ValidationError(f"{field} source identity differs")
        elif field == "workflow_policy_gate_artifact" and (
            document.get("git_sha") != git_sha
            or document.get("git_tree") != git_tree_sha
            or document.get("dirty") is not False
        ):
            raise ValidationError("workflow policy source identity differs")

    staged = resolve_artifact(
        "staged_assets_manifest", data["staged_assets_manifest"], manifest_dir
    )
    summary = resolve_artifact(
        "release_summary_artifact", data["release_summary_artifact"], manifest_dir
    )
    promotion = resolve_artifact("promotion_receipt", data["promotion_receipt"], manifest_dir)
    for label, artifact in (
        ("staged_assets_manifest", staged),
        ("release_summary_artifact", summary),
        ("promotion_receipt", promotion),
    ):
        if not artifact.is_file():
            raise ValidationError(f"{label} must be a file: {artifact}")
    try:
        import runtime_vnext_goal_gate as goal_gate

        staged_goal = goal_gate.validate_staged_assets_manifest(staged)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValidationError(f"staged_assets_manifest strict validation failed: {exc}") from exc
    if staged_goal.get("release_candidate") != expected_source:
        raise ValidationError("staged assets release-candidate source differs")

    g10a = goals["g10a_gate_artifact"]
    same_ref_sha(
        "G10A staged_assets input",
        g10a["child"].get("inputs", {}).get("staged_assets"),
        file_sha256(staged),
    )
    published = goals["published_assets_gate_artifact"]
    prepromotion = goals["prepromotion_gate_artifact"]
    g10b = goals["g10b_gate_artifact"]
    published_release = published["child"].get("release")
    prepromotion_release = prepromotion["child"].get("release")
    final_release = g10b["child"].get("release")
    if not all(isinstance(item, dict) for item in (published_release, prepromotion_release, final_release)):
        raise ValidationError("release DAG release identities are missing")
    release_id_text = str(release_id)
    asset_set_sha256 = published_release.get("asset_set_sha256")
    if (
        published_release.get("id") != release_id_text
        or published_release.get("tag_name") != "v0.8.0"
        or published_release.get("tag_sha") != git_sha
        or published_release.get("draft") is not False
        or published_release.get("prerelease") is not True
        or prepromotion_release.get("id") != release_id_text
        or prepromotion_release.get("tag_sha") != git_sha
        or prepromotion_release.get("prerelease") is not True
        or prepromotion_release.get("asset_set_sha256") != asset_set_sha256
        or final_release.get("id") != release_id_text
        or final_release.get("tag_name") != "v0.8.0"
        or final_release.get("tag_sha") != git_sha
        or final_release.get("draft") is not False
        or final_release.get("prerelease") is not False
        or final_release.get("asset_set_sha256") != asset_set_sha256
        or final_release.get("html_url") != data["github_release_url"]
        or published_release.get("html_url") != data["github_release_url"]
        or final_release.get("published_at") != data["github_release"].get("published_at")
    ):
        raise ValidationError("published/prepromotion/G10B release identity differs")
    validate_sha256("release DAG asset_set_sha256", asset_set_sha256)

    pre_dependencies = prepromotion["verified"]["dependencies"]
    dependency_fields = {
        "crates_io": "crates_io_gate_artifact",
        "homebrew_metal": "homebrew_metal_gate_artifact",
        "homebrew_cuda_fetch": "homebrew_cuda_fetch_gate_artifact",
        "workflow_policy": "workflow_policy_gate_artifact",
    }
    for dependency, field in dependency_fields.items():
        if file_sha256(pre_dependencies[dependency]) != file_sha256(artifact_paths[field]):
            raise ValidationError(f"prepromotion {dependency} differs from completion input")
    if file_sha256(pre_dependencies["published_assets"]) != published["child_sha256"]:
        raise ValidationError("prepromotion published-assets dependency differs")

    g10b_inputs = g10b["child"].get("inputs")
    if not isinstance(g10b_inputs, dict):
        raise ValidationError("G10B inputs are missing")
    for key, field in {
        "g10a": "g10a_gate_artifact",
        "g08_rc": "g08_rc_gate_artifact",
        "g09_rc": "g09_rc_gate_artifact",
        "published_assets": "published_assets_gate_artifact",
        "prepromotion": "prepromotion_gate_artifact",
    }.items():
        same_ref_sha(f"G10B {key}", g10b_inputs.get(key), goals[field]["child_sha256"])

    summary_data = load_json(summary)
    if (
        not isinstance(summary_data, dict)
        or summary_data.get("status") != "pass"
        or summary_data.get("release_candidate") != expected_source
        or summary_data.get("release") != final_release
        or summary_data.get("asset_set_sha256") != asset_set_sha256
    ):
        raise ValidationError("release_summary_artifact is not PASS")
    promotion_data = load_json(promotion)
    prepromotion_data = prepromotion["child"]
    promotion_fields = {
        "schema_version",
        "state",
        "release_id",
        "tag",
        "release_candidate_sha",
        "prepromotion_manifest_sha256",
        "prepromotion_manifest_id",
        "consumption_token",
        "workflow_run_id",
        "workflow_run_attempt",
        "consumed_at",
        "consumed_by",
        "promotion",
        "asset_ids",
    }
    promotion_state = (
        promotion_data.get("promotion")
        if isinstance(promotion_data, dict)
        else None
    )
    if (
        not isinstance(promotion_data, dict)
        or set(promotion_data) != promotion_fields
        or promotion_data.get("schema_version") != 1
        or promotion_data.get("state") != "consumed"
        or str(promotion_data.get("release_id")) != str(release_id)
        or promotion_data.get("tag") != "v0.8.0"
        or promotion_data.get("release_candidate_sha") != git_sha
        or promotion_data.get("prepromotion_manifest_id")
        != prepromotion_data.get("manifest_id")
        or promotion_data.get("prepromotion_manifest_sha256")
        != prepromotion["child_sha256"]
        or promotion_data.get("consumption_token")
        != prepromotion_data.get("consumption", {}).get("token")
        or type(promotion_data.get("workflow_run_id")) is not int
        or promotion_data["workflow_run_id"] <= 0
        or type(promotion_data.get("workflow_run_attempt")) is not int
        or promotion_data["workflow_run_attempt"] <= 0
        or promotion_data.get("consumed_by") != "release-promote.yml"
        or not isinstance(promotion_data.get("consumed_at"), str)
        or not promotion_data["consumed_at"]
        or not isinstance(promotion_state, dict)
        or promotion_state.get("state") != "complete"
        or not isinstance(promotion_state.get("promoted_at"), str)
        or not promotion_state["promoted_at"]
        or not isinstance(promotion_data.get("asset_ids"), list)
        or any(
            not isinstance(asset_id, int)
            or isinstance(asset_id, bool)
            or asset_id <= 0
            for asset_id in promotion_data["asset_ids"]
        )
        or len(set(promotion_data["asset_ids"]))
        != len(promotion_data["asset_ids"])
        or not set(asset["id"] for asset in data["release_assets"]).issubset(
            promotion_data["asset_ids"]
        )
    ):
        raise ValidationError("promotion_receipt contract differs")
    same_ref_sha("G10B promotion receipt", g10b_inputs.get("promotion_receipt"), file_sha256(promotion))

    published_assets = published["child"].get("assets")
    if not isinstance(published_assets, dict):
        raise ValidationError("published-assets primary asset identities are missing")
    completion_by_name = {asset["name"]: asset for asset in data["release_assets"]}
    for backend, name in ASSET_BACKENDS.items():
        staged_row = staged_goal["assets"].get(backend)
        published_row = published_assets.get(backend)
        completion_row = completion_by_name[name]
        if not isinstance(staged_row, dict) or not isinstance(published_row, dict):
            raise ValidationError(f"{backend} release asset identity is missing")
        expected_tarball_sha = staged_row.get("tarball", {}).get("sha256")
        expected_binary_sha = staged_row.get("binary", {}).get("sha256")
        if (
            published_row.get("name") != name
            or published_row.get("digest") != f"sha256:{expected_tarball_sha}"
            or published_row.get("tarball_sha256") != expected_tarball_sha
            or published_row.get("binary_sha256") != expected_binary_sha
            or published_row.get("size") != staged_row.get("tarball", {}).get("size_bytes")
            or completion_row
            != {
                "id": published_row.get("id"),
                "name": name,
                "size_bytes": published_row.get("size"),
                "sha256": expected_tarball_sha,
                "staged_sha256": expected_tarball_sha,
                "binary_sha256": expected_binary_sha,
            }
        ):
            raise ValidationError(f"{backend} staged/published/completion asset identity differs")

    validate_crates(data)
    result = {
        "schema_version": 1,
        "status": "pass",
        "manifest": str(path),
        "tag": data["tag"],
        "release_id": release_id,
        "release_assets": data["release_assets"],
        "artifacts": artifacts,
        "staged_assets_manifest": str(staged),
        "release_summary_artifact": str(summary),
        "promotion_receipt": str(promotion),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "release_completion_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def _write_fixture_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def make_selftest_manifest(path: Path, *, artifact_root: Path | None = None) -> None:
    """Create a fully linked local fixture for completion/run_gate self-tests."""

    root = path.parent.resolve()
    artifact_root = (artifact_root or (root / "artifacts")).resolve()
    source = {"git_sha": "1" * 40, "git_tree_sha": "2" * 40, "dirty": False}
    rc_tag = "v0.8.0-rc.1"
    release_id = 12345
    release_id_text = str(release_id)
    release_url = "https://example.invalid/releases/v0.8.0"
    asset_set_digest = "3" * 64

    staged_root = artifact_root / "staged-assets"
    staged_assets: dict[str, Any] = {}
    for index, backend in enumerate(("cpu", "metal", "cuda"), start=1):
        backend_root = staged_root / "assets" / backend
        backend_root.mkdir(parents=True)

        def relative_ref(file_path: Path) -> dict[str, Any]:
            return {
                "path": file_path.relative_to(staged_root).as_posix(),
                "sha256": file_sha256(file_path),
                "size_bytes": file_path.stat().st_size,
            }

        payload = f"fixture-{backend}-binary\n".encode("utf-8")
        tarball = backend_root / f"{backend}.tar.gz"
        info = tarfile.TarInfo("ferrum")
        info.size = len(payload)
        info.mode = 0o755
        with tarfile.open(tarball, "w:gz") as archive:
            archive.addfile(info, io.BytesIO(payload))
        checksum = backend_root / f"{backend}.tar.gz.sha256"
        checksum.write_text(f"{file_sha256(tarball)}  {tarball.name}\n", encoding="utf-8")
        version_manifest = backend_root / "version.json"
        _write_fixture_json(version_manifest, {"version": "0.8.0"})
        workflow_run_id = 1001 if backend in {"cpu", "metal"} else 1002
        workflow_path = (
            ".github/workflows/release-cuda.yml"
            if backend == "cuda"
            else ".github/workflows/release.yml"
        )
        asset_base = {
            "cpu": "ferrum-linux-x86_64",
            "metal": "ferrum-macos-aarch64",
            "cuda": "ferrum-linux-x86_64-cuda-sm89",
        }[backend]
        artifact_archive = backend_root / "github-artifact.zip"
        artifact_archive.write_bytes(f"fixture-{backend}-artifact\n".encode("utf-8"))
        artifact = {
            "id": 2000 + index,
            "name": f"{asset_base}-v0.8.0-rc-{source['git_sha']}",
            "digest": f"sha256:{file_sha256(artifact_archive)}",
        }
        artifact_manifest = backend_root / "artifact.json"
        _write_fixture_json(
            artifact_manifest,
            {
                "schema_version": 1,
                "artifact_type": "runtime_vnext_github_workflow_artifact_manifest",
                "status": "pass",
                "repository": "sizzlecar/ferrum-infer-rs",
                "release_candidate": source,
                "release_candidate_tag": rc_tag,
                "publish_release": False,
                "artifact": artifact,
                "archive": relative_ref(artifact_archive),
                "workflow_run_id": workflow_run_id,
                "workflow_run": {
                    "id": workflow_run_id,
                    "attempt": 1,
                    "path": workflow_path,
                    "event": "workflow_dispatch",
                    "head_sha": source["git_sha"],
                    "status": "completed",
                    "conclusion": "success",
                },
                "workflow_inputs": {
                    "release_candidate_sha": source["git_sha"],
                    "release_candidate_tag": rc_tag,
                    "staging_label": "v0.8.0-rc",
                    "publish_release": False,
                },
            },
        )
        binary = {
            "archive_path": "ferrum",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        dependency = backend_root / "dependency-abi.json"
        _write_fixture_json(
            dependency,
            {
                "release_candidate": source,
                "release_candidate_tag": rc_tag,
                "binary_sha256": binary["sha256"],
                "tarball_sha256": file_sha256(tarball),
            },
        )

        row = {
            "backend": backend,
            "workflow_run_id": workflow_run_id,
            "artifact": artifact,
            "artifact_manifest": relative_ref(artifact_manifest),
            "tarball": relative_ref(tarball),
            "sha256_file": relative_ref(checksum),
            "version_manifest": relative_ref(version_manifest),
            "dependency_abi_manifest": relative_ref(dependency),
            "binary": binary,
        }
        if backend == "cuda":
            row["target_sm"] = "89"
        staged_assets[backend] = row
    staged_path = staged_root / "manifest.json"
    _write_fixture_json(
        staged_path,
        {
            "schema_version": 1,
            "artifact_type": "runtime_vnext_staged_assets_manifest",
            "status": "pass",
            "canonical": True,
            "version": "0.8.0",
            "publish_release": False,
            "release_candidate": source,
            "release_candidate_tag": rc_tag,
            "artifact_dir": str(staged_root),
            "assets": staged_assets,
            "created_at": "2026-08-14T00:00:00+00:00",
            "pass_line": f"FERRUM RUNTIME VNEXT STAGED ASSETS PASS: {staged_root}",
        },
    )

    artifact_types = {
        "vnext-g10a": "runtime_vnext_g10a_release_freeze_manifest",
        "vnext-g08-rc": "runtime_vnext_g08_rc_manifest",
        "vnext-g09-rc": "runtime_vnext_g09_rc_manifest",
        "runtime-vnext-metal-three-model": "runtime_vnext_three_model_metal_source_manifest",
        "runtime-vnext-cuda-three-model": "runtime_vnext_three_model_cuda_source_manifest",
        "runtime-vnext-published-assets": "runtime_vnext_published_assets_manifest",
        "vnext-g10b": "runtime_vnext_g10b_published_release_manifest",
        "vnext-g10": "runtime_vnext_g10_release_manifest",
    }
    child_prefixes = {
        "vnext-g10a": "FERRUM RUNTIME VNEXT G10A RELEASE FREEZE PASS",
        "vnext-g08-rc": "FERRUM RUNTIME VNEXT G08 RELEASE CANDIDATE CORRECTNESS PASS",
        "vnext-g09-rc": "FERRUM RUNTIME VNEXT G09 RELEASE CANDIDATE PERFORMANCE PASS",
        "runtime-vnext-metal-three-model": "FERRUM RUNTIME VNEXT THREE MODEL METAL SOURCE PASS",
        "runtime-vnext-cuda-three-model": "FERRUM RUNTIME VNEXT THREE MODEL CUDA SOURCE PASS",
        "runtime-vnext-published-assets": "FERRUM RUNTIME VNEXT PUBLISHED ASSETS PASS",
        "runtime-vnext-prepromotion": "FERRUM V0.8.0 PREPROMOTION PASS",
        "vnext-g10b": "FERRUM RUNTIME VNEXT G10B PUBLISHED RELEASE PASS",
        "vnext-g10": "FERRUM RUNTIME VNEXT G10 V0.8.0 RELEASE PASS",
    }
    goal_paths: dict[str, Path] = {}

    def write_goal(lane: str, inputs: dict[str, Any], extra: dict[str, Any]) -> Path:
        lane_root = artifact_root / lane
        lane_root.mkdir(parents=True, exist_ok=True)
        child_path = lane_root / "manifest.json"
        child = {
            "schema_version": 1,
            "artifact_type": artifact_types[lane],
            "lane": lane,
            "status": "pass",
            "canonical": True,
            "version": "0.8.0",
            "release_candidate": source,
            "artifact_dir": str(lane_root),
            "inputs": inputs,
            "acceptance": {"failure_count": 0},
            "created_at": "2026-08-14T00:00:00+00:00",
            "pass_line": f"{child_prefixes[lane]}: {lane_root}",
            "additional_pass_lines": (
                [f"FERRUM V0.8.0 THREE MODEL METAL CUDA RELEASE PASS: {lane_root}"]
                if lane == "runtime-vnext-published-assets"
                else []
            ),
            **extra,
        }
        _write_fixture_json(child_path, child)
        child_ref = _fixture_ref(child_path)
        outer = {
            "schema_version": 1,
            "lane": lane,
            "status": "pass",
            "child_returncode": 0,
            "artifact_dir": str(lane_root),
            "git_sha": source["git_sha"],
            "dirty_status": {"is_dirty": False, "status_short": []},
            "pass_line": f"FERRUM GATE {lane} PASS: {lane_root}",
            "child_pass_line": child["pass_line"],
            "child_artifacts": {
                "kind": lane,
                "child_manifest": child_ref,
                "source": source,
            },
        }
        _write_fixture_json(lane_root / "gate.manifest.json", outer)
        goal_paths[lane] = child_path
        return child_path

    g10a_path = write_goal(
        "vnext-g10a",
        {"staged_assets": _fixture_ref(staged_path)},
        {"source_closure": {}, "release_candidate_tag": rc_tag},
    )
    lane_rows: dict[str, Any] = {}
    for lane_key, (model_key, backend) in {
        "m1_cuda": ("m1-qwen35-4b", "cuda"),
        "m1_metal": ("m1-qwen35-4b", "metal"),
        "m2_cuda": ("m2-qwen35-35b-a3b", "cuda"),
        "m2_metal": ("m2-qwen35-35b-a3b", "metal"),
        "m3_cuda": ("m3-qwen3-30b-a3b", "cuda"),
        "m3_metal": ("m3-qwen3-30b-a3b", "metal"),
    }.items():
        lane_rows[lane_key] = {
            "model_key": model_key,
            "backend": backend,
            "source": source,
            "binary_sha256": staged_assets[backend]["binary"]["sha256"],
            "tarball_sha256": staged_assets[backend]["tarball"]["sha256"],
        }
    llama_rows = {
        backend: {
            "model_key": "llama31-8b-compat",
            "backend": backend,
            "source": source,
            "binary_sha256": staged_assets[backend]["binary"]["sha256"],
            "tarball_sha256": staged_assets[backend]["tarball"]["sha256"],
            "entrypoints": ["run", "serve"],
            "correctness_status": "pass",
            "performance_status": "pass",
            "full_matrix_claim": False,
        }
        for backend in ("metal", "cuda")
    }
    g08_path = write_goal(
        "vnext-g08-rc",
        {"g10a": _fixture_ref(g10a_path)},
        {
            "staged_assets": _fixture_ref(staged_path),
            "lanes": lane_rows,
            "llama_dense_supplemental": llama_rows,
        },
    )
    g09_path = write_goal(
        "vnext-g09-rc",
        {"g10a": _fixture_ref(g10a_path), "g08_rc": _fixture_ref(g08_path)},
        {
            "staged_assets": _fixture_ref(staged_path),
            "lanes": lane_rows,
            "llama_dense_supplemental": llama_rows,
            "correctness": {"status": "pass"},
        },
    )
    metal_path = write_goal(
        "runtime-vnext-metal-three-model",
        {"g10a": _fixture_ref(g10a_path), "g08_rc": _fixture_ref(g08_path), "g09_rc": _fixture_ref(g09_path)},
        {"backend": "metal", "lanes": {f"m{i}_metal": {} for i in (1, 2, 3)}},
    )
    cuda_path = write_goal(
        "runtime-vnext-cuda-three-model",
        {"g10a": _fixture_ref(g10a_path), "g08_rc": _fixture_ref(g08_path), "g09_rc": _fixture_ref(g09_path)},
        {"backend": "cuda", "lanes": {f"m{i}_cuda": {} for i in (1, 2, 3)}},
    )
    published_assets: dict[str, Any] = {}
    primary_assets: list[dict[str, Any]] = []
    for index, (backend, name) in enumerate(ASSET_BACKENDS.items(), start=1):
        staged_row = staged_assets[backend]
        asset_id = 100 + index
        published_assets[backend] = {
            "id": asset_id,
            "name": name,
            "size": staged_row["tarball"]["size_bytes"],
            "digest": f"sha256:{staged_row['tarball']['sha256']}",
            "tarball_sha256": staged_row["tarball"]["sha256"],
            "binary_sha256": staged_row["binary"]["sha256"],
            "workflow_run_id": staged_row["workflow_run_id"],
            "staged_artifact_id": staged_row["artifact"]["id"],
        }
        primary_assets.append(
            {
                "id": asset_id,
                "name": name,
                "size_bytes": staged_row["tarball"]["size_bytes"],
                "sha256": staged_row["tarball"]["sha256"],
                "staged_sha256": staged_row["tarball"]["sha256"],
                "binary_sha256": staged_row["binary"]["sha256"],
            }
        )
    prerelease = {
        "id": release_id_text,
        "html_url": release_url,
        "tag_name": "v0.8.0",
        "tag_sha": source["git_sha"],
        "release_candidate_tag": rc_tag,
        "draft": False,
        "prerelease": True,
        "published_at": "2026-08-14T00:00:00+00:00",
        "asset_set_sha256": asset_set_digest,
        "asset_count": 18,
    }
    published_path = write_goal(
        "runtime-vnext-published-assets",
        {
            "g10a": _fixture_ref(g10a_path),
            "g08_rc": _fixture_ref(g08_path),
            "g09_rc": _fixture_ref(g09_path),
            "staged_assets": _fixture_ref(staged_path),
        },
        {"release": prerelease, "assets": published_assets, "lanes": lane_rows},
    )

    def write_non_goal(field: str, lane: str | None, child_prefix: str) -> Path:
        field_root = artifact_root / field
        field_root.mkdir(parents=True, exist_ok=True)
        manifest_path = field_root / "gate.manifest.json"
        if lane is None:
            document = {
                "schema_version": 1,
                "status": "pass",
                "lane": "runtime-vnext-release-workflow-policy",
                "git_sha": source["git_sha"],
                "git_tree": source["git_tree_sha"],
                "dirty": False,
                "pass_line": f"{child_prefix}: {field_root}",
            }
        else:
            document = {
                "schema_version": 1,
                "status": "pass",
                "lane": lane,
                "artifact_dir": str(field_root),
                "git_sha": source["git_sha"],
                "dirty_status": {"is_dirty": False, "status_short": []},
                "pass_line": f"FERRUM GATE {lane} PASS: {field_root}",
                "child_pass_line": f"{child_prefix}: {field_root}",
            }
        _write_fixture_json(manifest_path, document)
        return manifest_path

    non_goal_specs = {
        "unit_source_gate_artifact": ("unit", "FERRUM GATE unit PASS"),
        "metal_tarball_gate_artifact": ("metal-tarball", "METAL TARBALL GATE PASS"),
        "cuda_tarball_gate_artifact": ("cuda-tarball", "CUDA TARBALL GATE PASS"),
        "homebrew_metal_gate_artifact": ("homebrew-metal", "HOMEBREW METAL GATE PASS"),
        "homebrew_cuda_fetch_gate_artifact": ("homebrew-cuda-fetch", "HOMEBREW CUDA FETCH GATE PASS"),
        "workflow_policy_gate_artifact": (None, "FERRUM RELEASE WORKFLOW POLICY PASS"),
    }
    non_goal_paths = {
        field: write_non_goal(field, lane, prefix)
        for field, (lane, prefix) in non_goal_specs.items()
    }
    crates_path = artifact_root / "crates_io_gate_artifact" / "crates-io.manifest.json"
    _write_fixture_json(
        crates_path,
        {
            "schema_version": 1,
            "status": "pass",
            "lane": "runtime-vnext-crates-io",
            "pass_line": f"FERRUM CRATES IO V0.8.0 PASS: {crates_path.parent}",
        },
    )
    dependencies = {
        "published_assets": {
            "status": "pass",
            "pass_line": f"FERRUM RUNTIME VNEXT PUBLISHED ASSETS PASS: {published_path.parent}",
            "manifest": _fixture_ref(published_path),
        },
        "crates_io": {
            "status": "pass",
            "pass_line": f"FERRUM CRATES IO V0.8.0 PASS: {crates_path.parent}",
            "manifest": _fixture_ref(crates_path),
        },
        "homebrew_metal": {
            "status": "pass",
            "pass_line": f"HOMEBREW METAL GATE PASS: {non_goal_paths['homebrew_metal_gate_artifact'].parent}",
            "manifest": _fixture_ref(non_goal_paths["homebrew_metal_gate_artifact"]),
        },
        "homebrew_cuda_fetch": {
            "status": "pass",
            "pass_line": f"HOMEBREW CUDA FETCH GATE PASS: {non_goal_paths['homebrew_cuda_fetch_gate_artifact'].parent}",
            "manifest": _fixture_ref(non_goal_paths["homebrew_cuda_fetch_gate_artifact"]),
        },
        "workflow_policy": {
            "status": "pass",
            "pass_line": f"FERRUM RELEASE WORKFLOW POLICY PASS: {non_goal_paths['workflow_policy_gate_artifact'].parent}",
            "manifest": _fixture_ref(non_goal_paths["workflow_policy_gate_artifact"]),
        },
    }
    pre_root = artifact_root / "runtime-vnext-prepromotion"
    pre_root.mkdir(parents=True)
    consumption = {
        "state": "unconsumed",
        "release_id": release_id_text,
        "token": "selftest-consumption-token-0000000000000000",
        "consumed_at": None,
        "consumed_by": None,
    }
    pre_release = {
        "id": release_id_text,
        "tag_name": "v0.8.0",
        "tag_sha": source["git_sha"],
        "draft": False,
        "prerelease": True,
        "asset_set_sha256": asset_set_digest,
    }
    identity_payload = {
        "schema_version": 1,
        "lane": "runtime-vnext-prepromotion",
        "release_candidate_sha": source["git_sha"],
        "release": pre_release,
        "consumption": consumption,
        "dependencies": dependencies,
    }
    pre_child = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_prepromotion_manifest",
        "status": "pass",
        "lane": "runtime-vnext-prepromotion",
        "version": "0.8.0",
        "canonical": True,
        "artifact_dir": str(pre_root),
        "manifest_id": canonical_json_sha256(identity_payload),
        "release_candidate_sha": source["git_sha"],
        "pass_line": f"FERRUM V0.8.0 PREPROMOTION PASS: {pre_root}",
        "prepromotion_pass_line": f"FERRUM V0.8.0 PREPROMOTION PASS: {pre_root}",
        "release": pre_release,
        "consumption": consumption,
        "dependencies": dependencies,
        "created_at": "2026-08-14T00:00:00+00:00",
    }
    pre_path = pre_root / "manifest.json"
    _write_fixture_json(pre_path, pre_child)
    _write_fixture_json(
        pre_root / "gate.manifest.json",
        {
            "schema_version": 1,
            "lane": "runtime-vnext-prepromotion",
            "status": "pass",
            "child_returncode": 0,
            "artifact_dir": str(pre_root),
            "git_sha": source["git_sha"],
            "dirty_status": {"is_dirty": False, "status_short": []},
            "pass_line": f"FERRUM GATE runtime-vnext-prepromotion PASS: {pre_root}",
            "child_pass_line": pre_child["pass_line"],
            "child_artifacts": {
                "kind": "runtime-vnext-prepromotion",
                "child_manifest": _fixture_ref(pre_path),
                "source": source,
            },
        },
    )
    promotion_path = artifact_root / "promotion" / "promotion-consumption.json"
    all_asset_ids = [row["id"] for row in primary_assets] + list(range(201, 216))
    _write_fixture_json(
        promotion_path,
        {
            "schema_version": 1,
            "state": "consumed",
            "release_id": release_id_text,
            "tag": "v0.8.0",
            "release_candidate_sha": source["git_sha"],
            "prepromotion_manifest_sha256": file_sha256(pre_path),
            "prepromotion_manifest_id": pre_child["manifest_id"],
            "consumption_token": consumption["token"],
            "workflow_run_id": 9001,
            "workflow_run_attempt": 1,
            "consumed_at": "2026-08-14T00:01:00+00:00",
            "consumed_by": "release-promote.yml",
            "promotion": {
                "state": "complete",
                "promoted_at": "2026-08-14T00:02:00+00:00",
            },
            "asset_ids": all_asset_ids,
        },
    )
    final_release = dict(prerelease)
    final_release["prerelease"] = False
    final_release["published_at"] = "2026-08-14T00:02:00+00:00"
    g10b_path = write_goal(
        "vnext-g10b",
        {
            "g10a": _fixture_ref(g10a_path),
            "g08_rc": _fixture_ref(g08_path),
            "g09_rc": _fixture_ref(g09_path),
            "published_assets": _fixture_ref(published_path),
            "prepromotion": _fixture_ref(pre_path),
            "promotion_receipt": _fixture_ref(promotion_path),
        },
        {
            "release": final_release,
            "promotion": {
                "state": "complete",
                "receipt": _fixture_ref(promotion_path),
                "prepromotion_manifest_id": pre_child["manifest_id"],
                "workflow_run_id": 9001,
            },
        },
    )
    write_goal(
        "vnext-g10",
        {
            "g10a": _fixture_ref(g10a_path),
            "g08_rc": _fixture_ref(g08_path),
            "g09_rc": _fixture_ref(g09_path),
            "g10b": _fixture_ref(g10b_path),
        },
        {"release": final_release},
    )
    summary_path = artifact_root / "release-summary" / "g0_release_summary.json"
    _write_fixture_json(
        summary_path,
        {
            "schema_version": 1,
            "status": "pass",
            "release_candidate": source,
            "release": final_release,
            "asset_set_sha256": asset_set_digest,
            "gates": [],
        },
    )
    field_paths = {
        **{field: str(manifest) for field, manifest in non_goal_paths.items()},
        "crates_io_gate_artifact": str(crates_path),
        "g10a_gate_artifact": str(g10a_path.parent / "gate.manifest.json"),
        "g08_rc_gate_artifact": str(g08_path.parent / "gate.manifest.json"),
        "g09_rc_gate_artifact": str(g09_path.parent / "gate.manifest.json"),
        "metal_three_model_gate_artifact": str(metal_path.parent / "gate.manifest.json"),
        "cuda_three_model_gate_artifact": str(cuda_path.parent / "gate.manifest.json"),
        "published_assets_gate_artifact": str(published_path.parent / "gate.manifest.json"),
        "prepromotion_gate_artifact": str(pre_path.parent / "gate.manifest.json"),
        "g10b_gate_artifact": str(g10b_path.parent / "gate.manifest.json"),
    }
    _write_fixture_json(
        path,
        {
            "version": "0.8.0",
            "git_sha": source["git_sha"],
            "git_tree_sha": source["git_tree_sha"],
            "dirty_status": {"is_dirty": False, "status_short": []},
            "tag": "v0.8.0",
            "release_id": release_id,
            "github_release_url": release_url,
            "github_release": {
                "id": release_id,
                "tag": "v0.8.0",
                "target_git_sha": source["git_sha"],
                "draft": False,
                "prerelease": False,
                "published_at": final_release["published_at"],
                "url": release_url,
            },
            "release_assets": primary_assets,
            "staged_assets_manifest": str(staged_path),
            "release_summary_artifact": str(summary_path),
            "promotion_receipt": str(promotion_path),
            "cargo_workspace_crates": [
                {"name": name, "version": "0.8.0", "crates_io_visible": True}
                for name in sorted(EXPECTED_CRATES)
            ],
            **field_paths,
        },
    )


def self_test() -> int:
    def expect_reject(label: str, mutate: Any) -> None:
        with tempfile.TemporaryDirectory(prefix=f"ferrum-completion-{label}-") as temp:
            root = Path(temp)
            manifest = root / "completion.json"
            make_selftest_manifest(manifest)
            mutate(manifest)
            try:
                validate_manifest(manifest, root / "out")
            except ValidationError:
                return
            raise ValidationError(f"negative fixture {label} unexpectedly passed")

    with tempfile.TemporaryDirectory(prefix="ferrum-completion-pass-") as temp:
        root = Path(temp)
        manifest = root / "completion.json"
        make_selftest_manifest(manifest)
        fixture = load_json(manifest)
        if any(
            field in fixture
            for field in (
                "metal_source_gate_artifact",
                "cuda_full_source_gate_artifact",
                "cuda_dense_source_gate_artifact",
            )
        ):
            raise ValidationError(
                "completion fixture unexpectedly requires legacy full accelerator gates"
            )
        validate_manifest(manifest, root / "out")

    def mutate_asset(manifest: Path) -> None:
        value = load_json(manifest)
        value["release_assets"][0]["sha256"] = "f" * 64
        value["release_assets"][0]["staged_sha256"] = "f" * 64
        _write_fixture_json(manifest, value)

    def mutate_receipt(manifest: Path) -> None:
        value = load_json(manifest)
        receipt = Path(value["promotion_receipt"])
        document = load_json(receipt)
        document["prepromotion_manifest_sha256"] = "f" * 64
        _write_fixture_json(receipt, document)
        g10b_outer = Path(value["g10b_gate_artifact"])
        g10b_child = g10b_outer.parent / "manifest.json"
        g10b = load_json(g10b_child)
        g10b["inputs"]["promotion_receipt"] = _fixture_ref(receipt)
        g10b["promotion"]["receipt"] = _fixture_ref(receipt)
        _write_fixture_json(g10b_child, g10b)
        outer = load_json(g10b_outer)
        outer["child_artifacts"]["child_manifest"] = _fixture_ref(g10b_child)
        _write_fixture_json(g10b_outer, outer)

    def mutate_summary(manifest: Path) -> None:
        value = load_json(manifest)
        summary = Path(value["release_summary_artifact"])
        document = load_json(summary)
        document["release"]["prerelease"] = True
        _write_fixture_json(summary, document)

    def mutate_goal_child(manifest: Path) -> None:
        value = load_json(manifest)
        outer_path = Path(value["published_assets_gate_artifact"])
        child = outer_path.parent / "manifest.json"
        document = load_json(child)
        document["release_candidate"]["git_sha"] = "f" * 40
        _write_fixture_json(child, document)

    def mutate_missing_sampled(manifest: Path) -> None:
        value = load_json(manifest)
        value["g08_rc_gate_artifact"] = str(manifest.parent / "missing-g08-rc.json")
        _write_fixture_json(manifest, value)

    def mutate_prepromotion_as_sampled(manifest: Path) -> None:
        value = load_json(manifest)
        value["g08_rc_gate_artifact"] = value["prepromotion_gate_artifact"]
        _write_fixture_json(manifest, value)

    expect_reject("self-reported-asset", mutate_asset)
    expect_reject("promotion-prepromotion-sha", mutate_receipt)
    expect_reject("prepromotion-summary", mutate_summary)
    expect_reject("outer-child-byte-binding", mutate_goal_child)
    expect_reject("missing-sampled-correctness", mutate_missing_sampled)
    expect_reject("prepromotion-is-not-sampled", mutate_prepromotion_as_sampled)
    print("FERRUM RELEASE COMPLETION SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.manifest is None or args.out is None:
        parser.error("--manifest and --out are required unless --self-test is used")
    try:
        validate_manifest(args.manifest, args.out)
    except ValidationError as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "release_completion_gate.json").write_text(
            json.dumps({"status": "fail", "error": str(exc)}, indent=2) + "\n"
        )
        print(f"FERRUM RELEASE COMPLETION FAIL: {args.out}: {exc}", file=sys.stderr)
        return 1
    print(f"FERRUM RELEASE COMPLETION PASS: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
