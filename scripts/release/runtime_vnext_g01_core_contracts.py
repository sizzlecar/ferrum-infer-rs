#!/usr/bin/env python3
"""Build and verify the aggregate G01 core-contract checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_gate  # noqa: E402
import runtime_vnext_g01b_checkpoint as g01b  # noqa: E402


PASS_PREFIX = "FERRUM RUNTIME VNEXT G01 CORE CONTRACTS PASS"
SELFTEST_PASS = "FERRUM RUNTIME VNEXT G01 CORE CONTRACTS SELFTEST PASS"
MODEL_ID = "Qwen/Qwen3.5-4B"
INPUT_SPECS = {
    "g01a": {
        "lane": "vnext-g01a",
        "artifact_type": "runtime_vnext_g01a_contract_split_manifest",
        "child_kind": "vnext-g01a-s0a",
        "child_relative": "g01a-contract-split/manifest.json",
    },
    "g01b": {
        "lane": "vnext-g01b",
        "artifact_type": "runtime_vnext_g01b_production_reference_contract_manifest",
        "child_kind": "vnext-g01b",
        "child_relative": "g01b-reference-contract/manifest.json",
    },
}
EVIDENCE_SPECS = {
    "adr": {
        "input": "g01a",
        "source_path": "adr.md",
        "path": "adr.md",
    },
    "contract_map": {
        "input": "g01a",
        "source_path": "contract-map.json",
        "path": "contract-map.json",
    },
    "product": {
        "input": "g01b",
        "source_ref": "product",
        "path": "qwen35-4b-cuda-production.json",
    },
    "extension_drills": {
        "input": "g01b",
        "source_ref": "extension_drills",
        "path": "extension-drills.json",
    },
    "plan_snapshots": {
        "input": "g01b",
        "source_ref": "plan_snapshots",
        "path": "plan-snapshots/summary.json",
    },
    "overhead": {
        "input": "g01b",
        "source_ref": "overhead",
        "path": "overhead.json",
    },
}
ACCEPTANCE = {
    "g01a_current_and_byte_bound": True,
    "g01b_current_and_byte_bound": True,
    "g01b_consumes_exact_g01a": True,
    "g01a_g01b_consume_exact_g00f": True,
    "contract_source_scope_current": True,
    "production_source_scope_current": True,
    "qwen35_4b_model_identity_current": True,
    "s1_input_provenance_preserved": True,
    "selected_evidence_byte_bound": True,
}
DOES_NOT_PROVE = [
    "G02",
    "G03",
    "G04",
    "G05",
    "G06",
    "G07",
    "G08",
    "G09",
    "G10",
    "full_model_migration",
    "performance",
    "release",
]


class AggregateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregateError(message)


def require_object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be non-empty")
    return value


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {item}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise AggregateError(f"invalid {label}: {path}: {error}") from error
    return require_object(value, label)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"cannot hash non-regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def git(*args: str) -> str:
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
    status = [line for line in git("status", "--short").splitlines() if line.strip()]
    require(not status, f"G01 requires a clean checkout: {status}")
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_tree_sha": git("rev-parse", "HEAD^{tree}"),
        "dirty": False,
        "status_short": [],
    }


def safe_artifact_path(root: Path, relative: Any, label: str) -> Path:
    text = require_string(relative, label)
    path = Path(text)
    require(not path.is_absolute() and ".." not in path.parts, f"{label} is unsafe")
    resolved = (root / path).resolve()
    require(root.resolve() in resolved.parents, f"{label} escapes the artifact root")
    require(resolved.is_file() and not resolved.is_symlink(), f"{label} is missing: {resolved}")
    return resolved


def relocated_child_path(
    outer_path: Path,
    artifact_dir: str,
    recorded: str,
    key: str,
) -> Path:
    recorded_root = Path(artifact_dir)
    recorded_path = Path(recorded)
    require(
        recorded_root.is_absolute() and recorded_path.is_absolute(),
        f"{key} input paths must be absolute",
    )
    try:
        relative = recorded_path.relative_to(recorded_root)
    except ValueError as error:
        raise AggregateError(f"{key} child path escapes its recorded artifact root") from error
    require(
        relative.as_posix() == INPUT_SPECS[key]["child_relative"],
        f"{key} child logical path mismatch",
    )
    local_root = outer_path.parent.resolve()
    resolved = (local_root / relative).resolve()
    require(local_root in resolved.parents, f"{key} relocated child escapes its artifact root")
    return resolved


def child_source(child: dict[str, Any], label: str) -> dict[str, Any]:
    source = require_object(child.get("source"), f"{label} source")
    require(
        source.get("dirty") is False
        and source.get("status_short") == []
        and g01b.GIT_SHA_RE.fullmatch(str(source.get("git_sha"))) is not None
        and g01b.GIT_SHA_RE.fullmatch(str(source.get("git_tree_sha"))) is not None,
        f"{label} source identity/clean state mismatch",
    )
    return source


def validate_outer_child_pair(
    key: str,
    outer: dict[str, Any],
    child: dict[str, Any],
    child_digest: str,
) -> dict[str, Any]:
    spec = INPUT_SPECS[key]
    lane = spec["lane"]
    require(outer.get("schema_version") == 1, f"{key} outer schema mismatch")
    require(
        outer.get("lane") == lane and outer.get("status") == "pass",
        f"{key} outer identity/status mismatch",
    )
    dirty = require_object(outer.get("dirty_status"), f"{key} outer dirty status")
    require(
        dirty.get("is_dirty") is False and dirty.get("status_short") == [],
        f"{key} outer source was dirty",
    )
    artifact_dir = require_string(outer.get("artifact_dir"), f"{key} artifact_dir")
    require(
        outer.get("pass_line") == f"FERRUM GATE {lane} PASS: {artifact_dir}",
        f"{key} outer PASS line mismatch",
    )
    artifacts = require_object(outer.get("child_artifacts"), f"{key} child artifacts")
    require(artifacts.get("kind") == spec["child_kind"], f"{key} child kind mismatch")
    child_ref = require_object(artifacts.get("child_manifest"), f"{key} child ref")
    require_string(child_ref.get("path"), f"{key} child ref path")
    require(
        child_ref.get("sha256") == child_digest,
        f"{key} outer/child digest mismatch",
    )
    require(
        child.get("artifact_type") == spec["artifact_type"]
        and child.get("status") == "pass",
        f"{key} child identity/status mismatch",
    )
    child_pass = require_string(child.get("pass_line"), f"{key} child PASS line")
    require(outer.get("child_pass_line") == child_pass, f"{key} outer/child PASS mismatch")
    source = child_source(child, key)
    require(outer.get("git_sha") == source["git_sha"], f"{key} outer/child source mismatch")
    return source


def load_input(path: Path, key: str, *, verify_checkout: bool) -> dict[str, Any]:
    outer_path = path.expanduser().resolve()
    outer = read_json(outer_path, f"{key} outer manifest")
    artifact_dir = require_string(outer.get("artifact_dir"), f"{key} artifact_dir")
    artifacts = require_object(outer.get("child_artifacts"), f"{key} child artifacts")
    child_ref = require_object(artifacts.get("child_manifest"), f"{key} child ref")
    child_recorded = require_string(child_ref.get("path"), f"{key} child path")
    child_path = relocated_child_path(
        outer_path,
        artifact_dir,
        child_recorded,
        key,
    )
    child_digest = sha256(child_path)
    child = read_json(child_path, f"{key} child manifest")
    source = validate_outer_child_pair(key, outer, child, child_digest)
    if key == "g01a":
        command = run_gate.LaneCommand(
            cmd=[],
            child_manifest_path=child_path,
            provenance_kind="vnext-g01a-s0a",
        )
        summary = run_gate.validate_vnext_g01a_s0a_provenance(
            command,
            child,
            child_digest,
            verify_checkout=verify_checkout,
        )
    else:
        summary = g01b.verify_checkpoint_manifest(
            child_path,
            verify_checkout=verify_checkout,
        )
    return {
        "outer_path": outer_path,
        "outer": outer,
        "outer_sha256": sha256(outer_path),
        "child_path": child_path,
        "child": child,
        "child_sha256": child_digest,
        "source": source,
        "summary": summary,
    }


def manifest_file_ref(value: Any, label: str) -> dict[str, Any]:
    ref = require_object(value, label)
    require_string(ref.get("path"), f"{label}.path")
    digest = require_string(ref.get("sha256"), f"{label}.sha256")
    require(g01b.SHA256_RE.fullmatch(digest) is not None, f"{label}.sha256 is invalid")
    return ref


def input_reference(item: dict[str, Any], root: Path, key: str) -> dict[str, Any]:
    directory = root / "inputs" / key
    directory.mkdir(parents=True, exist_ok=False)
    outer_copy = directory / "gate.manifest.json"
    child_copy = directory / "manifest.json"
    shutil.copyfile(item["outer_path"], outer_copy)
    shutil.copyfile(item["child_path"], child_copy)
    require(sha256(outer_copy) == item["outer_sha256"], f"{key} outer copy changed")
    require(sha256(child_copy) == item["child_sha256"], f"{key} child copy changed")
    return {
        "lane": INPUT_SPECS[key]["lane"],
        "source": item["source"],
        "outer_manifest": {
            "path": outer_copy.relative_to(root).as_posix(),
            "sha256": item["outer_sha256"],
        },
        "child_manifest": {
            "path": child_copy.relative_to(root).as_posix(),
            "sha256": item["child_sha256"],
        },
    }


def g01a_index(child: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = child.get("artifact_index")
    require(isinstance(rows, list), "G01A artifact index is missing")
    index = {}
    for ordinal, value in enumerate(rows):
        row = require_object(value, f"G01A artifact index row {ordinal}")
        path = require_string(row.get("path"), f"G01A artifact index path {ordinal}")
        require(path not in index, f"G01A artifact index duplicates {path}")
        index[path] = row
    return index


def evidence_source(
    inputs: dict[str, dict[str, Any]],
    name: str,
) -> tuple[Path, str, str]:
    spec = EVIDENCE_SPECS[name]
    item = inputs[spec["input"]]
    child_root = item["child_path"].parent
    if spec["input"] == "g01a":
        source_path = spec["source_path"]
        row = require_object(g01a_index(item["child"]).get(source_path), f"G01A {name} index")
        digest = require_string(row.get("sha256"), f"G01A {name} SHA256")
    else:
        refs = require_object(item["child"].get("evidence"), "G01B evidence")
        ref = manifest_file_ref(refs.get(spec["source_ref"]), f"G01B {name} ref")
        source_path = ref["path"]
        digest = ref["sha256"]
    path = safe_artifact_path(child_root, source_path, f"{name} source")
    require(sha256(path) == digest, f"{name} source digest mismatch")
    return path, source_path, digest


def copy_evidence(
    root: Path,
    inputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    refs = {}
    for name, spec in EVIDENCE_SPECS.items():
        source, source_path, digest = evidence_source(inputs, name)
        destination = root / spec["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        require(sha256(destination) == digest, f"{name} copied evidence changed")
        refs[name] = {
            "path": spec["path"],
            "sha256": digest,
            "source_input": spec["input"],
            "source_path": source_path,
        }
    return refs


def s1_model_inputs(g01b_child: dict[str, Any]) -> dict[str, Any]:
    inputs = require_object(g01b_child.get("inputs"), "G01B inputs")
    rows = {}
    for key in ("s1", "s1_capacity", "s1_decode_capacity"):
        ref = require_object(inputs.get(key), f"G01B input {key}")
        rows[key] = {
            "validator_git_sha": require_string(
                ref.get("validator_git_sha"), f"G01B {key} validator SHA"
            ),
            "source_git_sha": require_string(
                ref.get("source_git_sha"), f"G01B {key} source SHA"
            ),
            "outer_manifest_sha256": manifest_file_ref(
                ref.get("outer_manifest"), f"G01B {key} outer ref"
            )["sha256"],
            "child_manifest_sha256": manifest_file_ref(
                ref.get("child_manifest"), f"G01B {key} child ref"
            )["sha256"],
        }
    return rows


def cross_bindings(
    g01a_child: dict[str, Any],
    g01a_outer_sha: str,
    g01a_child_sha: str,
    g01b_child: dict[str, Any],
    source: dict[str, Any],
    product: dict[str, Any],
) -> dict[str, Any]:
    for key, child in (("g01a", g01a_child), ("g01b", g01b_child)):
        child_identity = child_source(child, key)
        require(child_identity == source, f"{key} source differs from G01 source")

    g01b_inputs = require_object(g01b_child.get("inputs"), "G01B inputs")
    bound_g01a = require_object(g01b_inputs.get("g01a"), "G01B G01A binding")
    require(
        manifest_file_ref(bound_g01a.get("outer_manifest"), "G01B G01A outer")["sha256"]
        == g01a_outer_sha
        and manifest_file_ref(bound_g01a.get("child_manifest"), "G01B G01A child")["sha256"]
        == g01a_child_sha,
        "G01B does not consume the supplied G01A bytes",
    )

    g01a_g00f = require_object(g01a_child.get("g00f"), "G01A G00F binding")
    g01b_g00f = require_object(g01b_inputs.get("g00f"), "G01B G00F binding")
    g00f_outer = manifest_file_ref(g01a_g00f.get("outer_manifest"), "G01A G00F outer")
    g00f_child = manifest_file_ref(g01a_g00f.get("child_manifest"), "G01A G00F child")
    require(
        g00f_outer["sha256"]
        == manifest_file_ref(g01b_g00f.get("outer_manifest"), "G01B G00F outer")["sha256"]
        and g00f_child["sha256"]
        == manifest_file_ref(g01b_g00f.get("child_manifest"), "G01B G00F child")["sha256"],
        "G01A and G01B consume different G00F bytes",
    )

    contract = g01b.contract_scope(source["git_sha"])
    production = g01b.production_scope(source["git_sha"])
    contract_summary = {"file_count": contract["file_count"], "sha256": contract["sha256"]}
    production_summary = {
        "file_count": production["file_count"],
        "sha256": production["sha256"],
    }
    require(
        g01b_child.get("contract_source_scope") == contract_summary,
        "G01B contract source scope is stale",
    )
    require(
        g01b_child.get("production_source_scope") == production_summary,
        "G01B production source scope is stale",
    )
    model = require_object(g01b_child.get("model"), "G01B model")
    require(
        model.get("model_id") == MODEL_ID and product.get("model") == model,
        "G01B product/model identity mismatch",
    )
    return {
        "g00f": {
            "outer_manifest_sha256": g00f_outer["sha256"],
            "child_manifest_sha256": g00f_child["sha256"],
            "source": require_object(g01a_g00f.get("source"), "G01A G00F source"),
        },
        "contract_source_scope": contract_summary,
        "production_source_scope": production_summary,
        "model": model,
        "model_inputs": s1_model_inputs(g01b_child),
    }


def artifact_index(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"G01 artifact contains symlink: {path}")
        if not path.is_file() or path == root / "manifest.json":
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def copied_inputs(root: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    refs = require_object(manifest.get("children"), "G01 children")
    require(set(refs) == set(INPUT_SPECS), "G01 child matrix mismatch")
    result = {}
    for key, spec in INPUT_SPECS.items():
        ref = require_object(refs.get(key), f"G01 child {key}")
        require(
            set(ref) == {"lane", "source", "outer_manifest", "child_manifest"},
            f"G01 child {key} field set mismatch",
        )
        require(ref.get("lane") == spec["lane"], f"G01 child {key} lane mismatch")
        outer_ref = manifest_file_ref(ref.get("outer_manifest"), f"G01 {key} outer ref")
        child_ref = manifest_file_ref(ref.get("child_manifest"), f"G01 {key} child ref")
        require(
            set(outer_ref) == {"path", "sha256"}
            and set(child_ref) == {"path", "sha256"}
            and outer_ref["path"] == f"inputs/{key}/gate.manifest.json"
            and child_ref["path"] == f"inputs/{key}/manifest.json",
            f"G01 child {key} manifest reference mismatch",
        )
        outer_path = safe_artifact_path(root, outer_ref["path"], f"G01 {key} outer")
        child_path = safe_artifact_path(root, child_ref["path"], f"G01 {key} child")
        require(sha256(outer_path) == outer_ref["sha256"], f"G01 {key} outer digest mismatch")
        require(sha256(child_path) == child_ref["sha256"], f"G01 {key} child digest mismatch")
        outer = read_json(outer_path, f"G01 copied {key} outer")
        child = read_json(child_path, f"G01 copied {key} child")
        source = validate_outer_child_pair(key, outer, child, child_ref["sha256"])
        require(ref.get("source") == source, f"G01 copied {key} source binding mismatch")
        result[key] = {
            "outer": outer,
            "outer_sha256": outer_ref["sha256"],
            "child": child,
            "child_sha256": child_ref["sha256"],
            "source": source,
        }
    return result


def copied_evidence(
    root: Path,
    manifest: dict[str, Any],
    inputs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    refs = require_object(manifest.get("evidence"), "G01 evidence")
    require(set(refs) == set(EVIDENCE_SPECS), "G01 evidence matrix mismatch")
    documents = {}
    for name, spec in EVIDENCE_SPECS.items():
        ref = manifest_file_ref(refs.get(name), f"G01 evidence {name}")
        require(
            set(ref) == {"path", "sha256", "source_input", "source_path"}
            and ref.get("path") == spec["path"]
            and ref.get("source_input") == spec["input"],
            f"G01 evidence {name} identity mismatch",
        )
        path = safe_artifact_path(root, ref["path"], f"G01 evidence {name}")
        require(sha256(path) == ref["sha256"], f"G01 evidence {name} digest mismatch")
        if spec["input"] == "g01a":
            row = require_object(
                g01a_index(inputs["g01a"]["child"]).get(spec["source_path"]),
                f"G01 evidence {name} G01A source",
            )
            expected_path = spec["source_path"]
            expected_digest = row.get("sha256")
        else:
            source_refs = require_object(
                inputs["g01b"]["child"].get("evidence"), "copied G01B evidence"
            )
            source_ref = manifest_file_ref(
                source_refs.get(spec["source_ref"]), f"copied G01B {name} ref"
            )
            expected_path = source_ref["path"]
            expected_digest = source_ref["sha256"]
        require(
            ref.get("source_path") == expected_path
            and ref["sha256"] == expected_digest,
            f"G01 evidence {name} source binding mismatch",
        )
        if path.suffix == ".json":
            documents[name] = read_json(path, f"G01 evidence {name}")
    return refs, documents


def verify_checkpoint_manifest(
    manifest_path: Path,
    *,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    path = manifest_path.expanduser().resolve()
    manifest = read_json(path, "G01 manifest")
    root = path.parent
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
            "output_root",
            "source",
            "children",
            "bindings",
            "evidence",
            "acceptance",
            "artifact_count",
            "artifact_index_sha256",
            "artifact_index",
            "unlocks",
            "does_not_prove",
            "started_at",
            "finished_at",
            "duration_seconds",
            "pass_line",
        },
        "G01 manifest field set mismatch",
    )
    require(
        manifest.get("schema_version") == 1
        and manifest.get("artifact_type") == "runtime_vnext_g01_core_contracts_manifest"
        and manifest.get("checkpoint_id") == "G01"
        and manifest.get("lane") == "runtime-vnext-g01"
        and manifest.get("status") == "pass"
        and manifest.get("canonical") is True,
        "G01 manifest identity/status mismatch",
    )
    output = Path(require_string(manifest.get("output_root"), "G01 output_root")).resolve()
    require(
        Path(require_string(manifest.get("artifact_dir"), "G01 artifact_dir")).resolve()
        == root
        and root == output / "g01-contracts",
        "G01 output layout mismatch",
    )
    require(manifest.get("pass_line") == f"{PASS_PREFIX}: {output}", "G01 PASS line mismatch")
    source = child_source({"source": manifest.get("source")}, "G01")
    if verify_checkout:
        require(clean_source() == source, "G01 source is stale against current checkout")

    inputs = copied_inputs(root, manifest)
    for key in INPUT_SPECS:
        require(inputs[key]["source"] == source, f"G01 {key} source is stale")
    _, documents = copied_evidence(root, manifest, inputs)
    product = require_object(documents.get("product"), "G01 copied product")
    bindings = cross_bindings(
        inputs["g01a"]["child"],
        inputs["g01a"]["outer_sha256"],
        inputs["g01a"]["child_sha256"],
        inputs["g01b"]["child"],
        source,
        product,
    )
    require(manifest.get("bindings") == bindings, "G01 binding summary mismatch")
    require(manifest.get("acceptance") == ACCEPTANCE, "G01 acceptance mismatch")
    require(manifest.get("unlocks") == ["S2"], "G01 unlocks mismatch")
    require(manifest.get("does_not_prove") == DOES_NOT_PROVE, "G01 does_not_prove mismatch")
    rows = artifact_index(root)
    require(manifest.get("artifact_count") == len(rows), "G01 artifact count mismatch")
    require(manifest.get("artifact_index") == rows, "G01 artifact index mismatch")
    require(
        manifest.get("artifact_index_sha256") == canonical_sha256(rows),
        "G01 artifact index digest mismatch",
    )
    return {
        "kind": "vnext-g01",
        "child_manifest": {
            "path": str(path),
            "sha256": sha256(path),
            "artifact_count": len(rows),
        },
        "source": source,
        "bindings": bindings,
    }


def build_checkpoint(g01a_path: Path, g01b_path: Path, output_root: Path) -> str:
    source = clean_source()
    output = output_root.expanduser().resolve()
    require(
        REPO_ROOT not in output.parents and output != REPO_ROOT,
        "G01 output must be outside the source tree",
    )
    root = output / "g01-contracts"
    require(not root.exists(), f"G01 output already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    started_at = iso_now()
    started = time.monotonic()
    try:
        inputs = {
            "g01a": load_input(g01a_path, "g01a", verify_checkout=True),
            "g01b": load_input(g01b_path, "g01b", verify_checkout=True),
        }
        for key in INPUT_SPECS:
            require(inputs[key]["source"] == source, f"{key} is stale against current source")
        children = {
            key: input_reference(inputs[key], root, key) for key in INPUT_SPECS
        }
        evidence = copy_evidence(root, inputs)
        product = read_json(root / EVIDENCE_SPECS["product"]["path"], "G01 product")
        bindings = cross_bindings(
            inputs["g01a"]["child"],
            inputs["g01a"]["outer_sha256"],
            inputs["g01a"]["child_sha256"],
            inputs["g01b"]["child"],
            source,
            product,
        )
        rows = artifact_index(root)
        pass_line = f"{PASS_PREFIX}: {output}"
        manifest = {
            "schema_version": 1,
            "artifact_type": "runtime_vnext_g01_core_contracts_manifest",
            "checkpoint_id": "G01",
            "lane": "runtime-vnext-g01",
            "status": "pass",
            "canonical": True,
            "artifact_dir": str(root),
            "output_root": str(output),
            "source": source,
            "children": children,
            "bindings": bindings,
            "evidence": evidence,
            "acceptance": ACCEPTANCE,
            "artifact_count": len(rows),
            "artifact_index_sha256": canonical_sha256(rows),
            "artifact_index": rows,
            "unlocks": ["S2"],
            "does_not_prove": DOES_NOT_PROVE,
            "started_at": started_at,
            "finished_at": iso_now(),
            "duration_seconds": time.monotonic() - started,
            "pass_line": pass_line,
        }
        write_json(root / "manifest.json", manifest)
        verify_checkpoint_manifest(root / "manifest.json", verify_checkout=True)
        return pass_line
    except Exception as error:
        write_json(
            root / "failure.json",
            {
                "schema_version": 1,
                "artifact_type": "runtime_vnext_g01_failure",
                "source": source,
                "started_at": started_at,
                "finished_at": iso_now(),
                "duration_seconds": time.monotonic() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def self_test() -> int:
    source_sha = "1" * 40
    source_tree = "2" * 40
    child_digest = "3" * 64
    source = {
        "git_sha": source_sha,
        "git_tree_sha": source_tree,
        "dirty": False,
        "status_short": [],
    }
    for key, spec in INPUT_SPECS.items():
        artifact_dir = f"/tmp/{key}"
        child_pass = f"{key} CHILD PASS"
        child = {
            "artifact_type": spec["artifact_type"],
            "status": "pass",
            "source": source,
            "pass_line": child_pass,
        }
        outer = {
            "schema_version": 1,
            "lane": spec["lane"],
            "status": "pass",
            "git_sha": source_sha,
            "dirty_status": {"is_dirty": False, "status_short": []},
            "artifact_dir": artifact_dir,
            "pass_line": f"FERRUM GATE {spec['lane']} PASS: {artifact_dir}",
            "child_pass_line": child_pass,
            "child_artifacts": {
                "kind": spec["child_kind"],
                "child_manifest": {
                    "path": f"{artifact_dir}/manifest.json",
                    "sha256": child_digest,
                },
            },
        }
        require(
            validate_outer_child_pair(key, outer, child, child_digest) == source,
            f"{key} valid outer/child pair failed",
        )
        forged = json.loads(json.dumps(outer))
        forged["dirty_status"] = {"is_dirty": True, "status_short": [" M source"]}
        try:
            validate_outer_child_pair(key, forged, child, child_digest)
            raise AssertionError(f"{key} accepted a dirty outer")
        except AggregateError as error:
            require("outer source was dirty" in str(error), f"{key} dirty rejection drifted")
        forged = json.loads(json.dumps(outer))
        forged["child_artifacts"]["child_manifest"]["sha256"] = "4" * 64
        try:
            validate_outer_child_pair(key, forged, child, child_digest)
            raise AssertionError(f"{key} accepted a forged child digest")
        except AggregateError as error:
            require("outer/child digest mismatch" in str(error), f"{key} digest rejection drifted")
    with tempfile.TemporaryDirectory(prefix="ferrum-g01-selftest-") as temporary:
        root = Path(temporary)
        duplicate = root / "duplicate.json"
        duplicate.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
        try:
            read_json(duplicate, "duplicate fixture")
            raise AssertionError("G01 strict JSON accepted a duplicate key")
        except AggregateError as error:
            require("duplicate JSON key" in str(error), "G01 duplicate-key rejection drifted")
        evidence = root / "evidence.json"
        evidence.write_text("{}\n", encoding="utf-8")
        require(
            safe_artifact_path(root, "evidence.json", "evidence")
            == evidence.resolve(),
            "G01 safe artifact path drifted",
        )
        try:
            safe_artifact_path(root, "../evidence.json", "escaped evidence")
            raise AssertionError("G01 accepted an escaping artifact path")
        except AggregateError as error:
            require("unsafe" in str(error), "G01 path escape rejection drifted")
        outer = root / "g01a" / "gate.manifest.json"
        expected_child = root / "g01a" / "g01a-contract-split" / "manifest.json"
        expected_child.parent.mkdir(parents=True)
        expected_child.write_text("{}\n", encoding="utf-8")
        outer.parent.mkdir(parents=True, exist_ok=True)
        require(
            relocated_child_path(
                outer,
                "/recorded/g01a",
                "/recorded/g01a/g01a-contract-split/manifest.json",
                "g01a",
            )
            == expected_child.resolve(),
            "G01 child relocation drifted",
        )
        try:
            relocated_child_path(
                outer,
                "/recorded/g01a",
                "/recorded/escaped/manifest.json",
                "g01a",
            )
            raise AssertionError("G01 accepted an escaping child path")
        except AggregateError as error:
            require("escapes" in str(error), "G01 child path rejection drifted")
    print(SELFTEST_PASS)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g01a", type=Path)
    parser.add_argument("--g01b", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        try:
            return self_test()
        except (AggregateError, OSError, ValueError) as error:
            print(f"{SELFTEST_PASS} FAIL: {error}", file=sys.stderr)
            return 1
    missing = [
        name
        for name, value in (("--g01a", args.g01a), ("--g01b", args.g01b), ("--out", args.out))
        if value is None
    ]
    if missing:
        parser.error("required arguments: " + ", ".join(missing))
    try:
        print(build_checkpoint(args.g01a, args.g01b, args.out))
        return 0
    except (AggregateError, OSError, ValueError, RuntimeError) as error:
        print(f"{PASS_PREFIX} FAIL: {args.out}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
