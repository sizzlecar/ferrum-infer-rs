#!/usr/bin/env python3
"""Aggregate G0 release gate artifacts.

The historical Runtime vNext 0.8.0 contract remains the default for release
roots other than ``0.8.4``.  Ferrum 0.8.4 has a deliberately smaller, scoped
profile that validates the current source, accelerator, tarball, and Homebrew
gates and binds every selected outer manifest to one clean candidate SHA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


LEGACY_REQUIRED_GROUPS = {
    "unit": [
        "unit/unit.gate.json",
        "unit/gate.json",
        "unit.gate.json",
        "source/unit.gate.json",
        "source-unit/unit.gate.json",
    ],
    "metal-tarball": ["metal-tarball/gate.json"],
    "cuda-tarball": ["cuda-tarball/gate.json"],
    "homebrew-metal": ["homebrew-metal/gate.json"],
    "homebrew-cuda-fetch": ["homebrew-cuda-fetch/gate.json"],
    "vnext-g08-rc": [
        "vnext-g08-rc/gate.manifest.json",
        "runtime-vnext-final/vnext-g08-rc/gate.manifest.json",
    ],
    "vnext-g09-rc": [
        "vnext-g09-rc/gate.manifest.json",
        "runtime-vnext-final/vnext-g09-rc/gate.manifest.json",
    ],
    "runtime-vnext-metal-three-model": [
        "runtime-vnext-metal-three-model/gate.manifest.json",
        "runtime-vnext-final/runtime-vnext-metal-three-model/gate.manifest.json",
    ],
    "runtime-vnext-cuda-three-model": [
        "runtime-vnext-cuda-three-model/gate.manifest.json",
        "runtime-vnext-final/runtime-vnext-cuda-three-model/gate.manifest.json",
    ],
    "runtime-vnext-published-assets": [
        "runtime-vnext-published-assets/gate.manifest.json",
        "runtime-vnext-final/runtime-vnext-published-assets/gate.manifest.json",
    ],
    "runtime-vnext-prepromotion": [
        "runtime-vnext-prepromotion/gate.manifest.json",
        "runtime-vnext-final/runtime-vnext-prepromotion/gate.manifest.json",
    ],
    "vnext-g10b": [
        "vnext-g10b/gate.manifest.json",
        "runtime-vnext-final/vnext-g10b/gate.manifest.json",
    ],
    "vnext-g10": [
        "vnext-g10/gate.manifest.json",
        "runtime-vnext-final/vnext-g10/gate.manifest.json",
    ],
}
LEGACY_EXPECTED_VNEXT_LANES = {
    "vnext-g08-rc",
    "vnext-g09-rc",
    "runtime-vnext-metal-three-model",
    "runtime-vnext-cuda-three-model",
    "runtime-vnext-published-assets",
    "runtime-vnext-prepromotion",
    "vnext-g10b",
    "vnext-g10",
}
LEGACY_OPTIONAL = [
    "g0_cuda4090_smoke.gate.json",
    "source-cuda-smoke/g0_cuda4090_smoke.gate.json",
    "cuda-smoke/g0_cuda4090_smoke.gate.json",
    "metal/metal.gate.json",
    "metal/gate.json",
    "metal.gate.json",
    "source/metal.gate.json",
    "source-metal/metal.gate.json",
    "g0_cuda4090_full.gate.json",
    "source/g0_cuda4090_full.gate.json",
    "source-cuda-full/g0_cuda4090_full.gate.json",
    "cuda-full/g0_cuda4090_full.gate.json",
    "g0_cuda4090_llama_dense.gate.json",
    "source/g0_cuda4090_llama_dense.gate.json",
    "source-cuda-llama-dense/g0_cuda4090_llama_dense.gate.json",
    "source-cuda-llama-dense/gate.json",
    "cuda-llama-dense/g0_cuda4090_llama_dense.gate.json",
    "cuda-llama-dense/gate.json",
]


@dataclass(frozen=True)
class GateSpec:
    name: str
    lane: str
    directories: tuple[str, ...]
    child_gate: str
    child_identity_field: str
    child_identity: str
    child_pass_prefix: str


REQUIRED_GATES = (
    GateSpec(
        name="unit",
        lane="unit",
        directories=("source-unit", "unit"),
        child_gate="unit.gate.json",
        child_identity_field="lane",
        child_identity="unit",
        child_pass_prefix="G0 SOURCE unit PASS: ",
    ),
    GateSpec(
        name="metal-source",
        lane="metal",
        directories=("source-metal", "metal"),
        child_gate="metal.gate.json",
        child_identity_field="lane",
        child_identity="metal",
        child_pass_prefix="G0 SOURCE metal PASS: ",
    ),
    GateSpec(
        name="cuda-qwen-full",
        lane="cuda-full",
        directories=("source-cuda-full", "cuda-full"),
        child_gate="g0_cuda4090_full.gate.json",
        child_identity_field="lane",
        child_identity="g0_cuda4090_full",
        child_pass_prefix="G0 SOURCE g0_cuda4090_full PASS: ",
    ),
    GateSpec(
        name="cuda-llama-dense",
        lane="cuda-llama-dense",
        directories=("source-cuda-llama-dense", "cuda-llama-dense"),
        child_gate="g0_cuda4090_llama_dense.gate.json",
        child_identity_field="lane",
        child_identity="g0_cuda4090_llama_dense",
        child_pass_prefix="G0 SOURCE g0_cuda4090_llama_dense PASS: ",
    ),
    GateSpec(
        name="metal-tarball",
        lane="metal-tarball",
        directories=("metal-tarball",),
        child_gate="gate.json",
        child_identity_field="mode",
        child_identity="metal-tarball",
        child_pass_prefix="METAL TARBALL GATE PASS: ",
    ),
    GateSpec(
        name="cuda-tarball",
        lane="cuda-tarball",
        directories=("cuda-tarball",),
        child_gate="gate.json",
        child_identity_field="mode",
        child_identity="cuda-tarball",
        child_pass_prefix="CUDA TARBALL GATE PASS: ",
    ),
    GateSpec(
        name="homebrew-metal",
        lane="homebrew-metal",
        directories=("homebrew-metal",),
        child_gate="gate.json",
        child_identity_field="mode",
        child_identity="homebrew-metal",
        child_pass_prefix="HOMEBREW METAL GATE PASS: ",
    ),
    GateSpec(
        name="homebrew-cuda-fetch",
        lane="homebrew-cuda-fetch",
        directories=("homebrew-cuda-fetch",),
        child_gate="gate.json",
        child_identity_field="mode",
        child_identity="homebrew-cuda-fetch",
        child_pass_prefix="HOMEBREW CUDA FETCH GATE PASS: ",
    ),
)

OPTIONAL_GATES = (
    GateSpec(
        name="cuda-smoke",
        lane="cuda-smoke",
        directories=("source-cuda-smoke", "cuda-smoke"),
        child_gate="g0_cuda4090_smoke.gate.json",
        child_identity_field="lane",
        child_identity="g0_cuda4090_smoke",
        child_pass_prefix="G0 SOURCE g0_cuda4090_smoke PASS: ",
    ),
)


def load_json_object(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.is_file():
        return None, f"missing {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"invalid json {path}: {exc}"
    if not isinstance(data, dict):
        return None, f"gate is not a JSON object {path}"
    return data, ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def legacy_load_gate(
    path: Path,
    *,
    expected_lane: str | None = None,
) -> tuple[bool, str]:
    """Validate the historical 0.8.0 summary input without schema changes."""
    gate, error = load_json_object(path)
    if gate is None:
        return False, error
    if gate.get("status") != "pass":
        return False, f"gate not pass {path}: {gate}"
    if expected_lane is not None:
        expected_prefix = f"FERRUM GATE {expected_lane} PASS: "
        if gate.get("lane") != expected_lane:
            return False, f"gate lane differs {path}: {gate.get('lane')!r}"
        pass_line = gate.get("pass_line")
        if not isinstance(pass_line, str) or not pass_line.startswith(expected_prefix):
            return False, f"gate pass line differs {path}: {pass_line!r}"
        artifact_dir = gate.get("artifact_dir")
        if (
            not isinstance(artifact_dir, str)
            or pass_line != expected_prefix + artifact_dir
        ):
            return False, f"gate artifact binding differs {path}"
    return True, ""


def validate_clean_candidate(
    manifest: dict[str, Any],
    manifest_path: Path,
) -> tuple[str | None, str]:
    candidate_sha = manifest.get("git_sha")
    if (
        not isinstance(candidate_sha, str)
        or re.fullmatch(r"[0-9a-f]{40}", candidate_sha) is None
    ):
        return None, f"gate candidate git SHA missing or invalid {manifest_path}"
    if manifest.get("dirty_status") != {"is_dirty": False, "status_short": []}:
        return None, f"gate candidate checkout is dirty {manifest_path}"
    return candidate_sha, ""


def validate_child_source_candidate(
    child: dict[str, Any],
    child_path: Path,
    candidate_sha: str,
) -> str:
    """Bind child provenance when the delegated gate records source metadata."""
    source = child.get("source")
    if source is None:
        return ""
    if not isinstance(source, dict):
        return f"child gate source is not an object {child_path}"
    if "git_sha" not in source:
        return ""
    if source.get("git_sha") != candidate_sha:
        return f"child gate candidate differs {child_path}"
    dirty = source.get("dirty_status")
    if dirty is not None and dirty != {"is_dirty": False, "status_short": []}:
        return f"child gate source is dirty {child_path}"
    if source.get("dirty") not in (None, False):
        return f"child gate source is dirty {child_path}"
    return ""


def validate_gate(
    artifact_dir: Path,
    spec: GateSpec,
) -> tuple[bool, str, tuple[Path, Path, str] | None]:
    manifest_path = artifact_dir / "gate.manifest.json"
    manifest, error = load_json_object(manifest_path)
    if manifest is None:
        return False, error, None

    if manifest.get("status") != "pass":
        return False, f"gate not pass {manifest_path}", None
    if manifest.get("lane") != spec.lane:
        return (
            False,
            f"gate lane differs {manifest_path}: {manifest.get('lane')!r}",
            None,
        )
    if manifest.get("child_returncode") != 0:
        return (
            False,
            f"gate child return code differs {manifest_path}: "
            f"{manifest.get('child_returncode')!r}",
            None,
        )

    candidate_sha, error = validate_clean_candidate(manifest, manifest_path)
    if candidate_sha is None:
        return False, error, None

    recorded_artifact_dir = manifest.get("artifact_dir")
    if not isinstance(recorded_artifact_dir, str) or not recorded_artifact_dir:
        return False, f"gate artifact_dir missing {manifest_path}", None

    expected_outer_pass = f"FERRUM GATE {spec.lane} PASS: {recorded_artifact_dir}"
    if manifest.get("pass_line") != expected_outer_pass:
        return False, f"gate pass line differs {manifest_path}", None
    expected_child_pass = spec.child_pass_prefix + recorded_artifact_dir
    if manifest.get("child_pass_line") != expected_child_pass:
        return False, f"gate child pass line differs {manifest_path}", None

    child_path = artifact_dir / spec.child_gate
    child, error = load_json_object(child_path)
    if child is None:
        return False, error, None
    if child.get("status") != "pass":
        return False, f"child gate not pass {child_path}", None
    if child.get(spec.child_identity_field) != spec.child_identity:
        return (
            False,
            f"child gate {spec.child_identity_field} differs {child_path}: "
            f"{child.get(spec.child_identity_field)!r}",
            None,
        )
    child_artifacts = manifest.get("child_artifacts")
    if not isinstance(child_artifacts, dict):
        return False, f"gate child_artifacts missing {manifest_path}", None
    binding = child_artifacts.get("child_manifest")
    if not isinstance(binding, dict):
        return False, f"gate child manifest binding missing {manifest_path}", None
    if set(binding) != {"path", "sha256", "size_bytes"}:
        return False, f"gate child manifest binding fields differ {manifest_path}", None
    recorded_child = binding.get("path")
    if not isinstance(recorded_child, str) or not recorded_child or "\\" in recorded_child:
        return False, f"gate child manifest recorded path differs {manifest_path}", None
    recorded_root = PurePosixPath(recorded_artifact_dir)
    expected_recorded_child = recorded_root / spec.child_gate
    if PurePosixPath(recorded_child) != expected_recorded_child:
        return False, f"gate child manifest recorded path differs {manifest_path}", None
    child_sha = binding.get("sha256")
    if not isinstance(child_sha, str) or re.fullmatch(r"[0-9a-f]{64}", child_sha) is None:
        return False, f"gate child manifest SHA256 is invalid {manifest_path}", None
    if binding.get("size_bytes") != child_path.stat().st_size:
        return False, f"gate child manifest size differs {child_path}", None
    if child_sha != sha256_file(child_path):
        return False, f"gate child manifest SHA256 differs {child_path}", None
    child_source_error = validate_child_source_candidate(
        child,
        child_path,
        candidate_sha,
    )
    if child_source_error:
        return False, child_source_error, None
    return True, "", (manifest_path, child_path, candidate_sha)


def select_gate(
    root: Path,
    spec: GateSpec,
    *,
    required: bool,
) -> tuple[tuple[Path, Path, str] | None, str]:
    candidates = [root / directory for directory in spec.directories]
    selected = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "gate.manifest.json").exists()
        ),
        None,
    )
    if selected is None:
        if not required:
            attached_child = next(
                (
                    candidate / spec.child_gate
                    for candidate in candidates
                    if (candidate / spec.child_gate).exists()
                ),
                None,
            )
            if attached_child is None:
                return None, ""
            return None, f"{spec.name}: missing outer manifest for {attached_child}"
        expected = " OR ".join(
            str(path / "gate.manifest.json") for path in candidates
        )
        return None, f"{spec.name}: missing {expected}"

    ok, error, evidence = validate_gate(selected, spec)
    if not ok:
        return None, f"{spec.name}: {error}"
    return evidence, ""


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def v084_main(root: Path) -> int:
    errors: list[str] = []
    gates: list[str] = []
    candidate_shas: set[str] = set()
    for spec in REQUIRED_GATES:
        evidence, error = select_gate(root, spec, required=True)
        if error:
            errors.append(error)
            continue
        assert evidence is not None
        manifest_path, _, candidate_sha = evidence
        gates.append(relative_to_root(manifest_path, root))
        candidate_shas.add(candidate_sha)

    for spec in OPTIONAL_GATES:
        evidence, error = select_gate(root, spec, required=False)
        if error:
            errors.append(error)
            continue
        if evidence is not None:
            manifest_path, _, candidate_sha = evidence
            gates.append(relative_to_root(manifest_path, root))
            candidate_shas.add(candidate_sha)

    if len(candidate_shas) != 1:
        errors.append(
            "required gates do not bind one clean candidate git SHA: "
            + ", ".join(sorted(candidate_shas))
        )

    if errors:
        for error in errors:
            print(f"G0 RELEASE FAIL: {error}", file=sys.stderr)
        return 1

    candidate_sha = next(iter(candidate_shas))
    pass_line = f"G0 RELEASE PASS: {root}"
    summary = {
        "schema_version": 1,
        "status": "pass",
        "artifact_dir": str(root),
        "release_candidate_sha": candidate_sha,
        "pass_line": pass_line,
        "gates": gates,
    }
    (root / "g0_release_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(pass_line)
    return 0


def legacy_main(root: Path) -> int:
    """Preserve the strict Runtime vNext 0.8.0 release-summary contract."""
    errors: list[str] = []
    gates: list[str] = []
    selected: dict[str, Path] = {}
    for name, alternatives in LEGACY_REQUIRED_GROUPS.items():
        group_errors = []
        for relative in alternatives:
            ok, message = legacy_load_gate(
                root / relative,
                expected_lane=(
                    name if name in LEGACY_EXPECTED_VNEXT_LANES else None
                ),
            )
            if ok:
                gates.append(relative)
                selected[name] = (root / relative).resolve()
                break
            group_errors.append(message)
        else:
            errors.append(f"{name}: " + " OR ".join(group_errors))
    for relative in LEGACY_OPTIONAL:
        path = root / relative
        if path.exists():
            ok, message = legacy_load_gate(path)
            if ok:
                gates.append(relative)
            else:
                errors.append(message)
    if errors:
        for error in errors:
            print(f"G0 RELEASE FAIL: {error}", file=sys.stderr)
        return 1

    try:
        from validate_release_completion_manifest import (
            ValidationError,
            validate_goal_outer_child,
        )

        strict = {
            lane: validate_goal_outer_child(
                lane,
                selected[lane],
                expected_lane=lane,
            )
            for lane in LEGACY_EXPECTED_VNEXT_LANES
        }
        sources = {
            json.dumps(row["source"], sort_keys=True) for row in strict.values()
        }
        if len(sources) != 1:
            raise ValidationError(
                "vNext final summary release-candidate sources differ"
            )
        source = strict["vnext-g10b"]["source"]
        published_release = strict["runtime-vnext-published-assets"]["child"].get(
            "release"
        )
        prepromotion_release = strict["runtime-vnext-prepromotion"]["child"].get(
            "release"
        )
        g10b_release = strict["vnext-g10b"]["child"].get("release")
        g10_release = strict["vnext-g10"]["child"].get("release")
        if not all(
            isinstance(item, dict)
            for item in (
                published_release,
                prepromotion_release,
                g10b_release,
                g10_release,
            )
        ):
            raise ValidationError("vNext final summary release identities are missing")
        release_id = published_release.get("id")
        asset_set = published_release.get("asset_set_sha256")
        if (
            published_release.get("tag_name") != "v0.8.0"
            or published_release.get("tag_sha") != source["git_sha"]
            or published_release.get("draft") is not False
            or published_release.get("prerelease") is not True
            or prepromotion_release.get("id") != release_id
            or prepromotion_release.get("tag_sha") != source["git_sha"]
            or prepromotion_release.get("prerelease") is not True
            or prepromotion_release.get("asset_set_sha256") != asset_set
            or g10b_release.get("id") != release_id
            or g10b_release.get("tag_sha") != source["git_sha"]
            or g10b_release.get("draft") is not False
            or g10b_release.get("prerelease") is not False
            or g10b_release.get("asset_set_sha256") != asset_set
            or g10_release != g10b_release
        ):
            raise ValidationError(
                "published/prepromotion/G10B/G10 release identity differs or "
                "promotion is incomplete"
            )
    except Exception as exc:
        print(
            f"G0 RELEASE FAIL: strict final-promotion binding: {exc}",
            file=sys.stderr,
        )
        return 1

    summary = {
        "schema_version": 1,
        "status": "pass",
        "release_candidate": source,
        "release": g10_release,
        "asset_set_sha256": asset_set,
        "g10b_child_sha256": strict["vnext-g10b"]["child_sha256"],
        "g10_child_sha256": strict["vnext-g10"]["child_sha256"],
        "gates": gates,
    }
    (root / "g0_release_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"G0 RELEASE PASS: {root}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--profile",
        choices=("auto", "legacy", "v084"),
        default="auto",
        help="auto selects v084 only for a release root named 0.8.4",
    )
    args = parser.parse_args()
    profile = args.profile
    if profile == "auto":
        profile = "v084" if args.root.name == "0.8.4" else "legacy"
    return v084_main(args.root) if profile == "v084" else legacy_main(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
