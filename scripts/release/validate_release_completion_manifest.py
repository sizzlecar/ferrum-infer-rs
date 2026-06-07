#!/usr/bin/env python3
"""Validate the release completion manifest shape.

This is the local contract check used by `run_gate.py release-complete`. Remote
publication checks can be layered on top of the same manifest once release
credentials and network access are available in the release lane.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "git_sha",
    "dirty_status",
    "tag",
    "github_release_url",
    "release_assets",
    "metal_source_gate_artifact",
    "cuda_full_source_gate_artifact",
    "cuda_dense_source_gate_artifact",
    "metal_tarball_gate_artifact",
    "cuda_tarball_gate_artifact",
    "homebrew_metal_gate_artifact",
    "homebrew_cuda_fetch_gate_artifact",
    "cargo_workspace_crates",
}


class ValidationError(Exception):
    pass


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {path}: {exc}") from exc


def require_non_empty_string(where: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{where} must be a non-empty string")
    return value


def validate_sha256(where: str, value: Any) -> None:
    text = require_non_empty_string(where, value)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", text):
        raise ValidationError(f"{where} must be a 64-character SHA256 hex digest")


def validate_gate_artifact(where: str, value: Any, manifest_dir: Path) -> str:
    path = Path(require_non_empty_string(where, value))
    if not path.is_absolute():
        path = manifest_dir / path
    if not path.exists():
        raise ValidationError(f"{where} does not exist: {path}")
    return str(path)


def validate_release_assets(data: dict[str, Any]) -> None:
    assets = data["release_assets"]
    if not isinstance(assets, list) or not assets:
        raise ValidationError("release_assets must be a non-empty list")
    for idx, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise ValidationError(f"release_assets[{idx}] must be an object")
        require_non_empty_string(f"release_assets[{idx}].name", asset.get("name"))
        validate_sha256(f"release_assets[{idx}].sha256", asset.get("sha256"))


def validate_crates(data: dict[str, Any]) -> None:
    crates = data["cargo_workspace_crates"]
    if not isinstance(crates, list) or not crates:
        raise ValidationError("cargo_workspace_crates must be a non-empty list")
    for idx, crate in enumerate(crates):
        if not isinstance(crate, dict):
            raise ValidationError(f"cargo_workspace_crates[{idx}] must be an object")
        require_non_empty_string(f"cargo_workspace_crates[{idx}].name", crate.get("name"))
        require_non_empty_string(f"cargo_workspace_crates[{idx}].version", crate.get("version"))
        visible = crate.get("crates_io_visible")
        if visible is not True:
            raise ValidationError(
                f"cargo_workspace_crates[{idx}].crates_io_visible must be true"
            )


def validate_manifest(path: Path, out_dir: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValidationError("manifest must be an object")
    missing = sorted(REQUIRED_FIELDS - set(data))
    if missing:
        raise ValidationError(f"manifest missing fields: {', '.join(missing)}")
    require_non_empty_string("git_sha", data["git_sha"])
    dirty = data["dirty_status"]
    if not isinstance(dirty, dict) or "is_dirty" not in dirty:
        raise ValidationError("dirty_status must include is_dirty")
    require_non_empty_string("tag", data["tag"])
    require_non_empty_string("github_release_url", data["github_release_url"])
    validate_release_assets(data)
    manifest_dir = path.parent
    gate_fields = [
        "metal_source_gate_artifact",
        "cuda_full_source_gate_artifact",
        "cuda_dense_source_gate_artifact",
        "metal_tarball_gate_artifact",
        "cuda_tarball_gate_artifact",
        "homebrew_metal_gate_artifact",
        "homebrew_cuda_fetch_gate_artifact",
    ]
    artifacts = {
        field: validate_gate_artifact(field, data[field], manifest_dir) for field in gate_fields
    }
    validate_crates(data)
    result = {
        "schema_version": 1,
        "status": "pass",
        "manifest": str(path),
        "tag": data["tag"],
        "artifacts": artifacts,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "release_completion_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
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
