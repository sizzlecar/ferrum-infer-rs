#!/usr/bin/env python3
"""Aggregate G0 release gate artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_GROUPS = {
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
EXPECTED_VNEXT_LANES = {
    "vnext-g08-rc",
    "vnext-g09-rc",
    "runtime-vnext-metal-three-model",
    "runtime-vnext-cuda-three-model",
    "runtime-vnext-published-assets",
    "runtime-vnext-prepromotion",
    "vnext-g10b",
    "vnext-g10",
}
OPTIONAL = [
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


def load_gate(path: Path, *, expected_lane: str | None = None) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing {path}"
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return False, f"invalid json {path}: {e}"
    if data.get("status") != "pass":
        return False, f"gate not pass {path}: {data}"
    if expected_lane is not None:
        expected_prefix = f"FERRUM GATE {expected_lane} PASS: "
        if data.get("lane") != expected_lane:
            return False, f"gate lane differs {path}: {data.get('lane')!r}"
        pass_line = data.get("pass_line")
        if not isinstance(pass_line, str) or not pass_line.startswith(expected_prefix):
            return False, f"gate pass line differs {path}: {pass_line!r}"
        artifact_dir = data.get("artifact_dir")
        if not isinstance(artifact_dir, str) or pass_line != expected_prefix + artifact_dir:
            return False, f"gate artifact binding differs {path}"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    root = args.root
    errors: list[str] = []
    gates: list[str] = []
    selected: dict[str, Path] = {}
    for name, alternatives in REQUIRED_GROUPS.items():
        group_errors = []
        for rel in alternatives:
            ok, msg = load_gate(
                root / rel,
                expected_lane=name if name in EXPECTED_VNEXT_LANES else None,
            )
            if ok:
                gates.append(rel)
                selected[name] = (root / rel).resolve()
                break
            group_errors.append(msg)
        else:
            errors.append(f"{name}: " + " OR ".join(group_errors))
    for rel in OPTIONAL:
        path = root / rel
        if path.exists():
            ok, msg = load_gate(path)
            if ok:
                gates.append(rel)
            else:
                errors.append(msg)
    if errors:
        for err in errors:
            print(f"G0 RELEASE FAIL: {err}", file=sys.stderr)
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
            for lane in EXPECTED_VNEXT_LANES
        }
        sources = {json.dumps(row["source"], sort_keys=True) for row in strict.values()}
        if len(sources) != 1:
            raise ValidationError("vNext final summary release-candidate sources differ")
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
                "published/prepromotion/G10B/G10 release identity differs or promotion is incomplete"
            )
    except Exception as exc:
        print(f"G0 RELEASE FAIL: strict final-promotion binding: {exc}", file=sys.stderr)
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
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"G0 RELEASE PASS: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
