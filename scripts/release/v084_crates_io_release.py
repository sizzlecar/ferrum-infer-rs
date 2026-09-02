#!/usr/bin/env python3
"""Package, publish, and verify the exact Ferrum 0.8.4 workspace crates.

``prepublish`` is publication-free.  It requires a clean release-candidate
checkout and annotated ``v0.8.4-rc.N`` tag, runs ``cargo package --locked`` for
the exact 16-crate roster, and freezes the archives plus their source/content
bindings in a versioned manifest.

``publish`` consumes those immutable archives.  It probes crates.io before
every possible upload, publishes strictly in deterministic dependency order,
waits for both API and sparse-index visibility, and never automatically
retransmits an upload with an ambiguous outcome.  A resumable JSON receipt is
persisted after each state transition.  The final gate also performs a clean
registry resolution and a clean ``cargo install`` followed by ``ferrum
--version`` and ``--help``.

The command runner, artifact-reference primitives, and output redaction are
reused from ``runtime_vnext_crates_io_release.py``.  No legacy 0.8.0 constant,
manifest validator, topology rule, or publication state machine is reused.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

try:
    import runtime_vnext_crates_io_release as legacy
except ImportError:  # pragma: no cover - supports import from the repository root
    from scripts.release import runtime_vnext_crates_io_release as legacy


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
VERSION = "0.8.4"
FINAL_TAG = "v0.8.4"
RC_TAG_RE = re.compile(r"^v0\.8\.4-rc\.[1-9][0-9]*$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CRATES_IO_API = "https://crates.io/api/v1"
CRATES_IO_SPARSE = "https://index.crates.io"
USER_AGENT = "ferrum-v084-crates-io-release/0.8.4"
PREPUBLISH_PASS_PREFIX = "FERRUM CRATES IO V0.8.4 PREPUBLISH PASS"
PUBLISH_PASS_PREFIX = "FERRUM CRATES IO V0.8.4 PASS"
SELFTEST_PASS_LINE = "FERRUM CRATES IO V0.8.4 SELFTEST PASS"

EXPECTED_CRATES = frozenset(
    {
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
)

# Explicit aliases make the version-independent legacy surface easy to audit.
artifact_ref = legacy.artifact_ref
external_artifact_ref = legacy.external_artifact_ref
validate_artifact_ref = legacy.validate_artifact_ref
validate_external_artifact_ref = legacy.validate_external_artifact_ref
sha256_file = legacy.sha256_file
sha256_bytes = legacy.sha256_bytes
canonical_json = legacy.canonical_json
atomic_write_bytes = legacy.atomic_write_bytes
write_json = legacy.write_json
run_logged_command = legacy.run_logged_command
run_captured_command = legacy.run_captured_command
base_cargo_environment = legacy.base_cargo_environment
sanitize_text = legacy.sanitize_text
safe_argv = legacy.safe_argv


class ReleaseError(RuntimeError):
    """A frozen package or publication invariant was not satisfied."""


class ExternalReleaseError(ReleaseError):
    """A bounded external publication/visibility operation failed."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def exact_fields(value: Any, fields: Iterable[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    expected = set(fields)
    actual = set(value)
    require(
        actual == expected,
        f"{label} fields differ: missing={sorted(expected - actual)} "
        f"extra={sorted(actual - expected)}",
    )
    return value


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"cannot read {label} JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def manifest_identity(value: dict[str, Any], fields: Iterable[str]) -> str:
    return sha256_bytes(canonical_json({field: value[field] for field in fields}))


def resolve_manifest_path(path: Path, names: tuple[str, ...]) -> Path:
    candidate = path.expanduser().resolve()
    if candidate.is_dir():
        for name in names:
            nested = candidate / name
            if nested.is_file() and not nested.is_symlink():
                return nested
        raise ReleaseError(f"no supported manifest under {candidate}: {names}")
    require(candidate.is_file() and not candidate.is_symlink(), f"manifest is missing or a symlink: {candidate}")
    return candidate


def git_bytes(repo: Path, *arguments: str) -> bytes:
    process = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(
        process.returncode == 0,
        f"git {' '.join(arguments)} failed: "
        f"{sanitize_text(process.stderr.decode('utf-8', errors='replace').strip())}",
    )
    return process.stdout


def git_text(repo: Path, *arguments: str) -> str:
    return git_bytes(repo, *arguments).decode("utf-8", errors="strict").strip()


def validate_annotated_rc_tag(repo: Path, tag: str, expected_commit: str) -> dict[str, str]:
    require(RC_TAG_RE.fullmatch(tag) is not None, "release candidate tag must match v0.8.4-rc.N")
    ref = f"refs/tags/{tag}"
    require(git_text(repo, "cat-file", "-t", ref) == "tag", f"{tag} must be an annotated tag")
    object_sha = git_text(repo, "rev-parse", ref)
    peeled = git_text(repo, "rev-parse", f"{ref}^{{commit}}")
    require(GIT_SHA_RE.fullmatch(object_sha) is not None, f"{tag} object SHA is invalid")
    require(peeled == expected_commit, f"{tag} peels to a different release candidate")
    return {"name": tag, "object_sha": object_sha, "peeled_commit_sha": peeled}


def clean_release_candidate(repo: Path, *, expected_sha: str, tag: str) -> dict[str, Any]:
    repo = repo.expanduser().resolve()
    require((repo / "Cargo.toml").is_file(), f"invalid repository root: {repo}")
    require(GIT_SHA_RE.fullmatch(expected_sha) is not None, "release candidate SHA must be 40 lowercase hex characters")
    observed_sha = git_text(repo, "rev-parse", "HEAD")
    observed_tree = git_text(repo, "rev-parse", "HEAD^{tree}")
    dirty = [line for line in git_text(repo, "status", "--short", "--untracked-files=all").splitlines() if line]
    require(not dirty, f"release candidate checkout is dirty: {dirty[:12]}")
    require(observed_sha == expected_sha, "checkout HEAD differs from the requested release candidate SHA")
    return {
        "git_sha": observed_sha,
        "git_tree_sha": observed_tree,
        "dirty": False,
        "tag": validate_annotated_rc_tag(repo, tag, observed_sha),
    }


def validate_candidate(value: Any, label: str) -> dict[str, Any]:
    row = exact_fields(value, {"git_sha", "git_tree_sha", "dirty", "tag"}, label)
    require(isinstance(row["git_sha"], str) and GIT_SHA_RE.fullmatch(row["git_sha"]), f"{label}.git_sha is invalid")
    require(isinstance(row["git_tree_sha"], str) and GIT_SHA_RE.fullmatch(row["git_tree_sha"]), f"{label}.git_tree_sha is invalid")
    require(row["dirty"] is False, f"{label}.dirty must be false")
    tag = exact_fields(row["tag"], {"name", "object_sha", "peeled_commit_sha"}, f"{label}.tag")
    require(isinstance(tag["name"], str) and RC_TAG_RE.fullmatch(tag["name"]), f"{label}.tag name differs")
    require(isinstance(tag["object_sha"], str) and GIT_SHA_RE.fullmatch(tag["object_sha"]), f"{label}.tag object SHA is invalid")
    require(tag["peeled_commit_sha"] == row["git_sha"], f"{label}.tag peeled commit differs")
    return row


def is_publishable(package: dict[str, Any]) -> bool:
    publish = package.get("publish")
    return publish is None or (isinstance(publish, list) and "crates-io" in publish)


def package_map_from_metadata(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    packages = metadata.get("packages")
    members = metadata.get("workspace_members")
    require(isinstance(packages, list) and isinstance(members, list), "cargo metadata workspace fields are invalid")
    by_id = {
        row.get("id"): row
        for row in packages
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    require(set(members) <= set(by_id), "cargo metadata omits a workspace member")
    selected: dict[str, dict[str, Any]] = {}
    for member in members:
        package = by_id[member]
        name = package.get("name")
        if isinstance(name, str) and name.startswith("ferrum-") and is_publishable(package):
            require(name not in selected, f"duplicate publishable package: {name}")
            selected[name] = package
    require(
        set(selected) == set(EXPECTED_CRATES),
        "publishable Ferrum roster differs: "
        f"missing={sorted(set(EXPECTED_CRATES) - set(selected))} "
        f"extra={sorted(set(selected) - set(EXPECTED_CRATES))}",
    )
    for name, package in selected.items():
        require(package.get("version") == VERSION, f"{name} must be version {VERSION}")
    return selected


def internal_dependency_graph(packages: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    names = set(packages)
    graph: dict[str, list[dict[str, str]]] = {}
    for name, package in packages.items():
        dependencies = package.get("dependencies")
        require(isinstance(dependencies, list), f"{name}.dependencies must be a list")
        rows: list[dict[str, str]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for raw in dependencies:
            require(isinstance(raw, dict), f"{name} dependency is invalid")
            dependency = raw.get("name")
            if dependency not in names:
                continue
            requirement = raw.get("req")
            kind = raw.get("kind") or "normal"
            target = raw.get("target") or ""
            require(requirement == f"^{VERSION}", f"{name} internal dependency {dependency} must require ^{VERSION}, found {requirement!r}")
            require(kind in {"normal", "build", "dev"}, f"{name}->{dependency} dependency kind differs")
            key = (str(dependency), str(kind), str(target), str(requirement))
            if key not in seen:
                rows.append({"name": str(dependency), "kind": str(kind), "target": str(target), "requirement": str(requirement)})
                seen.add(key)
        graph[name] = sorted(rows, key=lambda row: (row["name"], row["kind"], row["target"], row["requirement"]))
    return graph


def stable_topological_order(graph: dict[str, list[dict[str, str]]]) -> list[str]:
    dependencies = {name: {row["name"] for row in rows} for name, rows in graph.items()}
    for name, deps in dependencies.items():
        require(deps <= set(graph), f"{name} topology references an unknown crate")
        require(name not in deps, f"{name} has a self dependency")
    ready = sorted(name for name, deps in dependencies.items() if not deps)
    result: list[str] = []
    while ready:
        current = ready.pop(0)
        result.append(current)
        for name in sorted(dependencies):
            if current in dependencies[name]:
                dependencies[name].remove(current)
                if not dependencies[name] and name not in result and name not in ready:
                    ready.append(name)
                    ready.sort()
    remaining = {name: sorted(deps) for name, deps in dependencies.items() if deps}
    require(not remaining, f"publishable dependency graph contains a cycle: {remaining}")
    require(len(result) == len(graph), "topological order omitted a crate")
    return result


def dependency_tables(manifest: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for key in ("dependencies", "dev-dependencies", "build-dependencies"):
        value = manifest.get(key)
        if isinstance(value, dict):
            yield key, value
    targets = manifest.get("target")
    if isinstance(targets, dict):
        for target_name, target in targets.items():
            if not isinstance(target, dict):
                continue
            for key in ("dependencies", "dev-dependencies", "build-dependencies"):
                value = target.get(key)
                if isinstance(value, dict):
                    yield f"target.{target_name}.{key}", value


def inspect_crate_archive(
    archive: Path,
    *,
    expected_name: str,
    candidate_sha: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    expected_root = f"{expected_name}-{VERSION}"
    require(SHA256_RE.fullmatch(source_manifest_sha256) is not None, f"{expected_name} source manifest SHA is invalid")
    file_rows: list[dict[str, Any]] = []
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            require(bool(members), f"{archive.name} is empty")
            member_names: set[str] = set()
            payloads: dict[str, bytes] = {}
            for member in members:
                pure = PurePosixPath(member.name)
                require(not pure.is_absolute() and ".." not in pure.parts and pure.parts and pure.parts[0] == expected_root, f"unsafe/unexpected member in {archive.name}: {member.name}")
                require(member.name not in member_names, f"duplicate member in {archive.name}: {member.name}")
                require(not member.issym() and not member.islnk() and not member.isdev() and not member.isfifo(), f"unsupported member type in {archive.name}: {member.name}")
                member_names.add(member.name)
                if member.isfile():
                    stream = bundle.extractfile(member)
                    require(stream is not None, f"cannot read {member.name}")
                    payload = stream.read()
                    payloads[member.name] = payload
                    file_rows.append({"path": member.name, "size_bytes": len(payload), "mode": member.mode & 0o777, "sha256": sha256_bytes(payload)})
                else:
                    require(member.isdir(), f"unsupported member type in {archive.name}: {member.name}")
            normalized_name = f"{expected_root}/Cargo.toml"
            original_name = f"{expected_root}/Cargo.toml.orig"
            vcs_name = f"{expected_root}/.cargo_vcs_info.json"
            require({normalized_name, original_name, vcs_name} <= set(payloads), f"{archive.name} lacks Cargo.toml/Cargo.toml.orig/VCS binding")
            normalized_bytes = payloads[normalized_name]
            original_bytes = payloads[original_name]
            require(sha256_bytes(original_bytes) == source_manifest_sha256, f"{archive.name} Cargo.toml.orig differs from candidate source")
            manifest = tomllib.loads(normalized_bytes.decode("utf-8"))
            package = manifest.get("package")
            require(isinstance(package, dict) and package.get("name") == expected_name and package.get("version") == VERSION, f"{archive.name} normalized package identity differs")
            path_dependency_count = 0
            for table_name, table in dependency_tables(manifest):
                for dependency_key, specification in table.items():
                    if isinstance(specification, dict):
                        if "path" in specification:
                            path_dependency_count += 1
                        dependency_name = specification.get("package", dependency_key)
                        if dependency_name in EXPECTED_CRATES:
                            # Cargo writes a caret requirement from metadata as
                            # the bare version in normalized Cargo.toml.
                            require(specification.get("version") in {VERSION, f"^{VERSION}"}, f"{archive.name} normalized {table_name}.{dependency_key} version differs")
                    elif dependency_key in EXPECTED_CRATES:
                        require(specification in {VERSION, f"^{VERSION}"}, f"{archive.name} normalized {table_name}.{dependency_key} version differs")
            require(path_dependency_count == 0, f"{archive.name} retains normalized path dependencies")
            vcs = json.loads(payloads[vcs_name].decode("utf-8"))
            git = vcs.get("git") if isinstance(vcs, dict) else None
            require(isinstance(git, dict) and git.get("sha1") == candidate_sha and git.get("dirty") in (None, False), f"{archive.name} VCS source binding differs")
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError, tarfile.TarError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseError(f"cannot inspect {archive}: {exc}") from exc
    file_rows.sort(key=lambda row: row["path"])
    return {
        "root": expected_root,
        "member_count": len(members),
        "file_count": len(file_rows),
        "content_sha256": sha256_bytes(canonical_json(file_rows)),
        "normalized_manifest_sha256": sha256_bytes(normalized_bytes),
        "original_manifest_sha256": sha256_bytes(original_bytes),
        "vcs_git_sha": candidate_sha,
        "path_dependency_count": 0,
    }


def validate_command_receipt(value: Any, *, root: Path, label: str) -> dict[str, Any]:
    try:
        return legacy.validate_command_receipt_evidence(value, root=root, label=label)
    except legacy.ReleaseError as exc:
        raise ReleaseError(str(exc)) from exc


def validate_exact_command(
    value: Any,
    *,
    root: Path,
    label: str,
    expected_argv: list[str],
) -> dict[str, Any]:
    command = validate_command_receipt(value, root=root, label=label)
    require(command.get("argv") == expected_argv, f"{label} argv differs")
    return command


def cargo_metadata(repo: Path, *, out: Path, artifact_root: Path, timeout_seconds: int) -> dict[str, Any]:
    receipt, stdout = run_captured_command(
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
        cwd=repo,
        env=base_cargo_environment(),
        stdout_path=out / "cargo-metadata.json",
        stderr_path=out / "cargo-metadata.stderr.log",
        receipt_path=out / "cargo-metadata.command.json",
        artifact_root=artifact_root,
        expected_seconds=30,
        deadline_seconds=min(timeout_seconds, 600),
    )
    require(receipt.get("exit_code") == 0 and receipt.get("timed_out") is False, "cargo metadata --locked failed")
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ReleaseError(f"cargo metadata output is invalid: {exc}") from exc
    require(isinstance(value, dict), "cargo metadata output must be an object")
    return value


def cargo_version(repo: Path, *, out: Path, artifact_root: Path) -> str:
    receipt, stdout = run_captured_command(
        ["cargo", "--version"],
        cwd=repo,
        env=base_cargo_environment(),
        stdout_path=out / "cargo-version.stdout",
        stderr_path=out / "cargo-version.stderr",
        receipt_path=out / "cargo-version.command.json",
        artifact_root=artifact_root,
        expected_seconds=5,
        deadline_seconds=30,
    )
    require(receipt.get("exit_code") == 0 and receipt.get("timed_out") is False, "cargo --version failed")
    version = stdout.strip()
    require(version.startswith("cargo "), "cargo --version output differs")
    return version


def package_workspace(
    *,
    repo: Path,
    target: Path,
    command_root: Path,
    artifact_root: Path,
    timeout_seconds: int,
) -> tuple[dict[str, Path], dict[str, Any]]:
    # Cargo's workspace form is required before the first 0.8.4 crate exists
    # on crates.io: individual packaging otherwise rejects an unpublished
    # internal dependency even with --no-verify.
    argv = ["cargo", "package", "--workspace", "--locked", "--no-verify", "--target-dir", str(target)]
    receipt = run_logged_command(
        argv,
        cwd=repo,
        env=base_cargo_environment(),
        log_path=command_root / "workspace.package.log",
        receipt_path=command_root / "workspace.package.command.json",
        artifact_root=artifact_root,
        expected_seconds=600,
        deadline_seconds=min(timeout_seconds, 3600),
    )
    require(receipt.get("exit_code") == 0 and receipt.get("timed_out") is False, "cargo package --workspace --locked failed")
    archives = {name: target / "package" / f"{name}-{VERSION}.crate" for name in EXPECTED_CRATES}
    for name, archive in archives.items():
        require(archive.is_file() and not archive.is_symlink(), f"cargo package omitted {archive}")
    actual = {path.name for path in (target / "package").glob("ferrum-*.crate")}
    expected = {path.name for path in archives.values()}
    require(actual == expected, f"cargo package workspace archive denominator differs: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
    return archives, receipt


PREPUBLISH_FIELDS = {
    "schema_version",
    "artifact_type",
    "status",
    "mode",
    "version",
    "canonical",
    "release_candidate",
    "cargo",
    "topology",
    "packages",
    "created_at",
    "credential_policy",
    "does_not_prove",
    "manifest_id",
    "pass_line",
}


def validate_prepublish_manifest(
    path: Path,
    *,
    repo: Path | None = None,
    recorded_root: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    manifest_path = resolve_manifest_path(path, ("prepublish.manifest.json", "gate.manifest.json"))
    root = manifest_path.parent.resolve()
    value = exact_fields(read_json(manifest_path, "prepublish manifest"), PREPUBLISH_FIELDS, "prepublish manifest")
    require(value["schema_version"] == SCHEMA_VERSION and value["artifact_type"] == "ferrum_v084_crates_io_prepublish_manifest", "prepublish schema/type differs")
    require(value["status"] == "pass" and value["mode"] == "prepublish" and value["version"] == VERSION and value["canonical"] is True, "prepublish status/mode/version differs")
    candidate = validate_candidate(value["release_candidate"], "prepublish.release_candidate")
    cargo = exact_fields(value["cargo"], {"version", "metadata", "metadata_command", "version_command", "worker_bounds"}, "prepublish.cargo")
    require(isinstance(cargo["version"], str) and cargo["version"].startswith("cargo "), "prepublish Cargo version differs")
    require(cargo["worker_bounds"] == {"cargo_build_jobs": 2, "rust_test_threads": 8}, "prepublish worker bounds differ")
    metadata_path = validate_artifact_ref(cargo["metadata"], root=root, label="prepublish cargo metadata", nonempty=True)
    validate_command_receipt(cargo["metadata_command"], root=root, label="prepublish cargo metadata command")
    validate_command_receipt(cargo["version_command"], root=root, label="prepublish cargo version command")
    metadata = read_json(metadata_path, "prepublish cargo metadata")
    packages = package_map_from_metadata(metadata)
    graph = internal_dependency_graph(packages)
    order = stable_topological_order(graph)
    topology = exact_fields(value["topology"], {"algorithm", "crate_count", "graph", "graph_sha256", "order"}, "prepublish.topology")
    require(topology["algorithm"] == "kahn-lexicographic-v1" and topology["crate_count"] == len(EXPECTED_CRATES), "prepublish topology identity differs")
    require(topology["graph"] == graph and topology["graph_sha256"] == sha256_bytes(canonical_json(graph)), "prepublish topology graph differs")
    require(topology["order"] == order, "prepublish topology order is not canonical")
    rows = value["packages"]
    require(isinstance(rows, list) and len(rows) == len(EXPECTED_CRATES), "prepublish package row count differs")
    for position, (name, raw) in enumerate(zip(order, rows), 1):
        row = exact_fields(
            raw,
            {
                "position", "name", "version", "manifest_path",
                "source_manifest_sha256", "source_manifest_git_blob_sha",
                "internal_dependencies", "archive", "archive_inspection",
                "package_command",
            },
            f"prepublish.packages[{position - 1}]",
        )
        require(row["position"] == position and row["name"] == name and row["version"] == VERSION, f"prepublish package identity/order differs at {position}")
        require(row["internal_dependencies"] == graph[name], f"{name} dependency binding differs")
        require(isinstance(row["manifest_path"], str) and not PurePosixPath(row["manifest_path"]).is_absolute() and ".." not in PurePosixPath(row["manifest_path"]).parts, f"{name} manifest path is unsafe")
        require(isinstance(row["source_manifest_sha256"], str) and SHA256_RE.fullmatch(row["source_manifest_sha256"]), f"{name} source manifest SHA differs")
        require(isinstance(row["source_manifest_git_blob_sha"], str) and GIT_SHA_RE.fullmatch(row["source_manifest_git_blob_sha"]), f"{name} source manifest git blob differs")
        archive = validate_artifact_ref(row["archive"], root=root, label=f"{name} immutable archive", nonempty=True)
        observed = inspect_crate_archive(archive, expected_name=name, candidate_sha=candidate["git_sha"], source_manifest_sha256=row["source_manifest_sha256"])
        require(observed == row["archive_inspection"], f"{name} archive inspection changed")
        command = validate_command_receipt(row["package_command"], root=root, label=f"{name} package command")
        argv = command.get("argv")
        require(isinstance(argv, list) and argv[:5] == ["cargo", "package", "--workspace", "--locked", "--no-verify"] and "--target-dir" in argv, f"{name} was not produced by cargo package --workspace --locked")
        if repo is not None:
            checkout = repo.resolve()
            manifest = (checkout / row["manifest_path"]).resolve()
            require(manifest.is_relative_to(checkout) and manifest.is_file() and not manifest.is_symlink(), f"{name} source manifest is missing")
            source_bytes = manifest.read_bytes()
            require(sha256_bytes(source_bytes) == row["source_manifest_sha256"], f"{name} source manifest bytes changed")
            git_blob = git_text(checkout, "rev-parse", f"{candidate['git_sha']}:{row['manifest_path']}")
            require(git_blob == row["source_manifest_git_blob_sha"], f"{name} source manifest git blob changed")
            require(git_bytes(checkout, "show", f"{candidate['git_sha']}:{row['manifest_path']}") == source_bytes, f"{name} source manifest does not match candidate commit")
            metadata_manifest = Path(str(packages[name].get("manifest_path", ""))).resolve()
            require(metadata_manifest == manifest, f"{name} cargo metadata manifest path differs")
    require(value["credential_policy"] == {"secret_values_recorded": False, "token_cli_arguments": False}, "prepublish credential policy differs")
    expected_id = manifest_identity(value, ("schema_version", "artifact_type", "version", "release_candidate", "cargo", "topology", "packages"))
    require(value["manifest_id"] == expected_id, "prepublish manifest identity differs")
    pass_root = root if recorded_root is None else recorded_root.expanduser().resolve()
    require(value["pass_line"] == f"{PREPUBLISH_PASS_PREFIX}: {pass_root}", "prepublish exact PASS line differs")
    return value, manifest_path


def install_manifest_pair(
    *,
    root: Path,
    primary_name: str,
    alias_name: str,
    value: dict[str, Any],
    validator: Callable[[Path], Any],
    allow_existing_identical: bool = False,
) -> Path:
    primary = root / primary_name
    alias = root / alias_name
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")
    if primary.exists() or alias.exists():
        require(allow_existing_identical, "refusing to overwrite a canonical PASS manifest")
        for path in (primary, alias):
            if path.exists():
                require(path.is_file() and not path.is_symlink(), f"canonical manifest path is invalid: {path}")
                require(path.read_bytes() == payload, f"canonical manifest bytes differ: {path}")
                validator(path)
        if not primary.exists():
            atomic_write_bytes(primary, payload)
            validator(primary)
        if not alias.exists():
            atomic_write_bytes(alias, payload)
            validator(alias)
        return primary
    candidate = root / f".{primary_name}.candidate-{os.getpid()}"
    try:
        atomic_write_bytes(candidate, payload)
        validator(candidate)
        os.replace(candidate, primary)
        atomic_write_bytes(alias, payload)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise
    return primary


def create_prepublish(args: argparse.Namespace) -> Path:
    repo = args.repo.expanduser().resolve()
    out = args.out.expanduser().resolve()
    require(not out.exists(), f"refusing to overwrite prepublish output: {out}")
    require(not out.is_relative_to(repo), "prepublish output must be outside the release checkout")
    candidate = clean_release_candidate(repo, expected_sha=args.release_candidate_sha, tag=args.release_candidate_tag)
    out.mkdir(parents=True)
    metadata_root = out / "metadata"
    metadata = cargo_metadata(repo, out=metadata_root, artifact_root=out, timeout_seconds=args.command_timeout_seconds)
    cargo_text = cargo_version(repo, out=metadata_root, artifact_root=out)
    packages = package_map_from_metadata(metadata)
    graph = internal_dependency_graph(packages)
    order = stable_topological_order(graph)
    target = out / "work" / "package-target"
    command_root = out / "package-commands"
    archive_root = out / "archives"
    archive_root.mkdir()
    workspace_archives, _ = package_workspace(repo=repo, target=target, command_root=command_root, artifact_root=out, timeout_seconds=args.command_timeout_seconds)
    rows: list[dict[str, Any]] = []
    for position, name in enumerate(order, 1):
        source_archive = workspace_archives[name]
        archive = archive_root / f"{name}-{VERSION}.crate"
        require(not archive.exists(), f"duplicate package archive: {archive}")
        shutil.copyfile(source_archive, archive)
        manifest = Path(str(packages[name]["manifest_path"])).resolve()
        require(manifest.is_relative_to(repo) and manifest.is_file() and not manifest.is_symlink(), f"{name} manifest is outside the candidate checkout")
        manifest_rel = manifest.relative_to(repo).as_posix()
        source_bytes = manifest.read_bytes()
        source_sha = sha256_bytes(source_bytes)
        git_blob = git_text(repo, "rev-parse", f"{candidate['git_sha']}:{manifest_rel}")
        require(git_bytes(repo, "show", f"{candidate['git_sha']}:{manifest_rel}") == source_bytes, f"{name} source manifest differs from the candidate commit")
        inspection = inspect_crate_archive(archive, expected_name=name, candidate_sha=candidate["git_sha"], source_manifest_sha256=source_sha)
        rows.append(
            {
                "position": position,
                "name": name,
                "version": VERSION,
                "manifest_path": manifest_rel,
                "source_manifest_sha256": source_sha,
                "source_manifest_git_blob_sha": git_blob,
                "internal_dependencies": graph[name],
                "archive": artifact_ref(archive, root=out),
                "archive_inspection": inspection,
                "package_command": artifact_ref(command_root / "workspace.package.command.json", root=out),
            }
        )
    clean_release_candidate(repo, expected_sha=candidate["git_sha"], tag=args.release_candidate_tag)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ferrum_v084_crates_io_prepublish_manifest",
        "status": "pass",
        "mode": "prepublish",
        "version": VERSION,
        "canonical": True,
        "release_candidate": candidate,
        "cargo": {
            "version": cargo_text,
            "metadata": artifact_ref(metadata_root / "cargo-metadata.json", root=out),
            "metadata_command": artifact_ref(metadata_root / "cargo-metadata.command.json", root=out),
            "version_command": artifact_ref(metadata_root / "cargo-version.command.json", root=out),
            "worker_bounds": {"cargo_build_jobs": 2, "rust_test_threads": 8},
        },
        "topology": {
            "algorithm": "kahn-lexicographic-v1",
            "crate_count": len(order),
            "graph": graph,
            "graph_sha256": sha256_bytes(canonical_json(graph)),
            "order": order,
        },
        "packages": rows,
        "created_at": iso_now(),
        "credential_policy": {"secret_values_recorded": False, "token_cli_arguments": False},
        "does_not_prove": ["any package is visible on crates.io", "cargo install from crates.io succeeds", "the GitHub release is final"],
        "manifest_id": "",
        "pass_line": f"{PREPUBLISH_PASS_PREFIX}: {out}",
    }
    manifest["manifest_id"] = manifest_identity(manifest, ("schema_version", "artifact_type", "version", "release_candidate", "cargo", "topology", "packages"))
    install_manifest_pair(root=out, primary_name="prepublish.manifest.json", alias_name="gate.manifest.json", value=manifest, validator=lambda path: validate_prepublish_manifest(path, repo=repo))
    assert_no_secret_output(out)
    print(manifest["pass_line"])
    return out


def index_prefix(name: str) -> str:
    lowered = name.lower()
    if len(lowered) == 1:
        return "1"
    if len(lowered) == 2:
        return "2"
    if len(lowered) == 3:
        return f"3/{lowered[0]}"
    return f"{lowered[:2]}/{lowered[2:4]}"


def http_json(url: str, label: str) -> tuple[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return "missing", None
        category = "auth" if exc.code in {401, 403} else "network"
        raise ExternalReleaseError(category, f"{label} returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ExternalReleaseError("network", f"{label} request failed: {sanitize_text(str(exc))}") from exc
    try:
        return "visible", json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalReleaseError("index", f"{label} returned invalid JSON") from exc


def http_json_lines(url: str, label: str) -> tuple[str, list[dict[str, Any]] | None]:
    state, payload = http_json(url, label)
    if state == "missing":
        return state, None
    # Sparse-index endpoints are JSONL, not a single JSON document.  The
    # production helper below is replaced instead of http_json for that path.
    require(isinstance(payload, list), f"{label} JSONL adapter returned invalid rows")
    return state, payload


def fetch_sparse_index(url: str, label: str) -> tuple[str, list[dict[str, Any]] | None]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return "missing", None
        raise ExternalReleaseError("network", f"{label} returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeDecodeError) as exc:
        raise ExternalReleaseError("network", f"{label} request failed: {sanitize_text(str(exc))}") from exc
    rows: list[dict[str, Any]] = []
    try:
        for line in text.splitlines():
            if line.strip():
                row = json.loads(line)
                require(isinstance(row, dict), f"{label} row is not an object")
                rows.append(row)
    except json.JSONDecodeError as exc:
        raise ExternalReleaseError("index", f"{label} returned invalid JSONL") from exc
    return "visible", rows


def probe_crates_io(
    name: str,
    checksum: str,
    *,
    api_reader: Callable[[str, str], tuple[str, Any]] = http_json,
    index_reader: Callable[[str, str], tuple[str, list[dict[str, Any]] | None]] = fetch_sparse_index,
) -> dict[str, Any]:
    require(SHA256_RE.fullmatch(checksum) is not None, f"{name} expected checksum is invalid")
    api_state, api = api_reader(f"{CRATES_IO_API}/crates/{urllib.parse.quote(name)}/{VERSION}", f"crates.io API {name} {VERSION}")
    index_state, index = index_reader(f"{CRATES_IO_SPARSE}/{index_prefix(name)}/{name.lower()}", f"crates.io sparse index {name}")
    api_checksum = None
    if api_state == "visible":
        require(isinstance(api, dict) and isinstance(api.get("version"), dict), f"crates.io API payload for {name} is invalid")
        version = api["version"]
        require(version.get("num") == VERSION, f"crates.io API version for {name} differs")
        api_checksum = version.get("checksum")
        require(api_checksum == checksum, f"published {name} API checksum differs from the frozen archive")
    index_checksum = None
    if index_state == "visible":
        require(isinstance(index, list), f"crates.io sparse index for {name} is invalid")
        matches = [row for row in index if isinstance(row, dict) and row.get("vers") == VERSION]
        if not matches:
            index_state = "missing"
        else:
            require(len(matches) == 1, f"crates.io sparse index has duplicate {name} {VERSION} rows")
            index_checksum = matches[0].get("cksum")
            require(index_checksum == checksum, f"published {name} index checksum differs from the frozen archive")
    return {
        "visible": api_state == "visible" and index_state == "visible",
        "partial": api_state == "visible" or index_state == "visible",
        "api": {"state": api_state, "checksum": api_checksum},
        "index": {"state": index_state, "checksum": index_checksum},
        "observed_at": iso_now(),
    }


def poll_crates_io(name: str, checksum: str, *, timeout_seconds: int, interval_seconds: int, probe_fn: Callable[[str, str], dict[str, Any]] = probe_crates_io) -> dict[str, Any]:
    require(timeout_seconds > 0, "visibility timeout must be positive")
    require(1 <= interval_seconds <= 60, "visibility interval must be 1..60 seconds")
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] | None = None
    while True:
        last = probe_fn(name, checksum)
        if last["visible"]:
            return last
        if time.monotonic() >= deadline:
            raise ExternalReleaseError("visibility", f"{name} {VERSION} did not become visible in API and index before the deadline; last={last}")
        time.sleep(min(interval_seconds, max(0.1, deadline - time.monotonic())))


def publish_environment() -> dict[str, str]:
    environment = base_cargo_environment()
    if not environment.get("CARGO_REGISTRY_TOKEN"):
        for alias in ("CRATES_IO_TOKEN", "CARGO_TOKEN"):
            if environment.get(alias):
                environment["CARGO_REGISTRY_TOKEN"] = environment[alias]
                break
    return environment


def classify_publish_failure(log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    ambiguous_markers = (
        "timeout", "timed out", "connection", "network", "dns", "tls", "ssl",
        "already exists", "already uploaded", "previously uploaded",
    )
    # Mixed proxy/server output does not prove that an upload was rejected
    # before acceptance, so ambiguous delivery signals always win over auth.
    if any(marker in text for marker in ambiguous_markers):
        return "ambiguous"
    if any(marker in text for marker in ("unauthorized", "forbidden", "authentication failed", "invalid token", "no token found", "403")):
        return "auth"
    return "command"


def next_stage_attempt(stage_root: Path) -> Path:
    """Create the next immutable attempt directory for a resumable final stage."""
    stage_root.mkdir(parents=True, exist_ok=True)
    numbers = []
    for child in stage_root.iterdir():
        match = re.fullmatch(r"attempt-([1-9][0-9]*)", child.name)
        if match is not None:
            require(child.is_dir() and not child.is_symlink(), f"invalid stage attempt path: {child}")
            numbers.append(int(match.group(1)))
    attempt = stage_root / f"attempt-{max(numbers, default=0) + 1}"
    attempt.mkdir()
    return attempt


def evidence_tree_identity(root: Path) -> dict[str, Any]:
    root = root.resolve()
    require(root.is_dir() and not root.is_symlink(), f"evidence tree root is invalid: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"evidence tree contains a symlink: {path}")
        if path.is_dir():
            continue
        require(path.is_file(), f"evidence tree contains an unsupported entry: {path}")
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    require(files, f"evidence tree is empty: {root}")
    return {
        "file_count": len(files),
        "total_size_bytes": sum(row["size_bytes"] for row in files),
        "sha256": sha256_bytes(canonical_json(files)),
    }


def portable_prepublish_binding(
    *,
    prepublish: dict[str, Any],
    copied_manifest: Path,
    artifact_root: Path,
    original_manifest: Path,
) -> dict[str, Any]:
    artifact_root = artifact_root.resolve()
    copied_manifest = copied_manifest.resolve()
    copied_root = copied_manifest.parent
    require(copied_root.is_relative_to(artifact_root), "copied prepublish evidence escapes the publish artifact")
    tree = evidence_tree_identity(copied_root)
    return {
        "manifest": artifact_ref(copied_manifest, root=artifact_root),
        "manifest_id": prepublish["manifest_id"],
        "tree": {
            "path": copied_root.relative_to(artifact_root).as_posix(),
            **tree,
        },
        "original_path": str(original_manifest.expanduser().resolve()),
    }


def copy_prepublish_evidence(
    *,
    prepublish: dict[str, Any],
    prepublish_path: Path,
    out: Path,
) -> tuple[dict[str, Any], Path]:
    source_manifest = prepublish_path.resolve()
    source_root = source_manifest.parent
    source_identity = evidence_tree_identity(source_root)
    destination = out.resolve() / "prepublish-evidence"
    require(not destination.exists(), f"copied prepublish evidence already exists: {destination}")
    shutil.copytree(source_root, destination, symlinks=False)
    copied_manifest = destination / source_manifest.relative_to(source_root)
    require(
        evidence_tree_identity(destination) == source_identity,
        "copied prepublish evidence tree differs from the source tree",
    )
    copied, _ = validate_prepublish_manifest(
        copied_manifest,
        recorded_root=source_root,
    )
    require(copied == prepublish, "copied prepublish manifest differs from the source manifest")
    binding = portable_prepublish_binding(
        prepublish=copied,
        copied_manifest=copied_manifest,
        artifact_root=out,
        original_manifest=source_manifest,
    )
    return binding, copied_manifest


def validate_portable_prepublish_binding(
    value: Any,
    *,
    artifact_root: Path,
    repo: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    binding = exact_fields(
        value,
        {"manifest", "manifest_id", "tree", "original_path"},
        "portable prepublish binding",
    )
    original_text = binding["original_path"]
    require(
        isinstance(original_text, str) and original_text and Path(original_text).is_absolute(),
        "portable prepublish original path is not absolute provenance",
    )
    tree = exact_fields(
        binding["tree"],
        {"path", "file_count", "total_size_bytes", "sha256"},
        "portable prepublish tree binding",
    )
    tree_text = tree["path"]
    require(isinstance(tree_text, str) and tree_text, "portable prepublish tree path is missing")
    tree_pure = PurePosixPath(tree_text)
    require(not tree_pure.is_absolute() and ".." not in tree_pure.parts and "\\" not in tree_text, "portable prepublish tree path is unsafe")
    root = artifact_root.resolve()
    copied_root = root.joinpath(*tree_pure.parts).resolve()
    require(copied_root.is_relative_to(root) and copied_root.is_dir() and not copied_root.is_symlink(), "portable prepublish tree is missing")
    manifest_path = validate_artifact_ref(
        binding["manifest"],
        root=root,
        label="portable prepublish manifest",
        nonempty=True,
    )
    require(manifest_path.parent.resolve() == copied_root, "portable prepublish manifest is outside its bound tree")
    observed_tree = evidence_tree_identity(copied_root)
    require(
        tree["file_count"] == observed_tree["file_count"]
        and tree["total_size_bytes"] == observed_tree["total_size_bytes"]
        and tree["sha256"] == observed_tree["sha256"],
        "portable prepublish evidence tree identity differs",
    )
    original_root = Path(original_text).parent
    prepublish, _ = validate_prepublish_manifest(
        manifest_path,
        repo=repo,
        recorded_root=original_root,
    )
    require(binding["manifest_id"] == prepublish["manifest_id"], "portable prepublish manifest identity differs")
    return prepublish, manifest_path


RECEIPT_FIELDS = {
    "schema_version", "artifact_type", "status", "version", "prepublish",
    "release_candidate", "publish_order", "packages", "clean_resolution",
    "install", "credential_policy", "started_at", "updated_at",
}
RECEIPT_PACKAGE_FIELDS = {
    "position", "name", "version", "archive_sha256", "state",
    "disposition", "dry_run_attempts", "publish_attempts",
    "visibility_observations", "visibility", "last_error",
}


def new_publish_receipt(
    prepublish: dict[str, Any],
    prepublish_path: Path,
    *,
    prepublish_binding: dict[str, Any] | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    if prepublish_binding is None:
        binding_root = prepublish_path.parent if artifact_root is None else artifact_root
        prepublish_binding = portable_prepublish_binding(
            prepublish=prepublish,
            copied_manifest=prepublish_path,
            artifact_root=binding_root,
            original_manifest=prepublish_path,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ferrum_v084_crates_io_publish_resume_receipt",
        "status": "in_progress",
        "version": VERSION,
        "prepublish": copy.deepcopy(prepublish_binding),
        "release_candidate": copy.deepcopy(prepublish["release_candidate"]),
        "publish_order": copy.deepcopy(prepublish["topology"]["order"]),
        "packages": [
            {
                "position": row["position"], "name": row["name"], "version": VERSION,
                "archive_sha256": row["archive"]["sha256"], "state": "pending",
                "disposition": None, "dry_run_attempts": [], "publish_attempts": [],
                "visibility_observations": [], "visibility": None, "last_error": None,
            }
            for row in prepublish["packages"]
        ],
        "clean_resolution": {"state": "pending", "result": None},
        "install": {"state": "pending", "result": None},
        "credential_policy": {"source": "existing-cargo-config-or-environment", "secret_values_recorded": False, "token_cli_arguments": False},
        "started_at": iso_now(),
        "updated_at": iso_now(),
    }


def validate_publish_receipt(
    receipt: dict[str, Any],
    *,
    prepublish: dict[str, Any],
    prepublish_path: Path,
    artifact_root: Path | None = None,
    recorded_artifact_root: Path | None = None,
) -> None:
    exact_fields(receipt, RECEIPT_FIELDS, "publish resume receipt")
    require(receipt["schema_version"] == SCHEMA_VERSION and receipt["artifact_type"] == "ferrum_v084_crates_io_publish_resume_receipt" and receipt["version"] == VERSION, "publish resume receipt identity differs")
    require(receipt["status"] in {"in_progress", "pass"}, "publish resume receipt status differs")
    binding_root = prepublish_path.parent if artifact_root is None else artifact_root
    bound_prepublish, bound_path = validate_portable_prepublish_binding(
        receipt["prepublish"],
        artifact_root=binding_root,
    )
    require(
        bound_path.resolve() == prepublish_path.resolve()
        and bound_prepublish == prepublish,
        "publish receipt prepublish bytes changed",
    )
    require(receipt["release_candidate"] == prepublish["release_candidate"] and receipt["publish_order"] == prepublish["topology"]["order"], "publish receipt candidate/order differs")
    rows = receipt["packages"]
    require(isinstance(rows, list) and len(rows) == len(EXPECTED_CRATES), "publish receipt package count differs")
    for package, row in zip(prepublish["packages"], rows):
        exact_fields(row, RECEIPT_PACKAGE_FIELDS, f"publish receipt package {package['name']}")
        require(row["position"] == package["position"] and row["name"] == package["name"] and row["version"] == VERSION and row["archive_sha256"] == package["archive"]["sha256"], f"publish receipt package binding differs: {package['name']}")
        require(row["state"] in {"pending", "retryable-auth", "publish-started", "awaiting-visibility", "visible"}, f"publish receipt state differs: {package['name']}")
        require(isinstance(row["dry_run_attempts"], list) and len(row["dry_run_attempts"]) <= 3, f"{package['name']} dry-run attempt bound differs")
        require(isinstance(row["publish_attempts"], list) and len(row["publish_attempts"]) <= 3, f"{package['name']} upload attempt bound differs")
        if row["state"] == "retryable-auth":
            error = exact_fields(row["last_error"], {"category", "stage", "recoverable", "at"}, f"{package['name']} retryable auth error")
            require(
                error["category"] == "auth"
                and error["stage"] == "publish"
                and error["recoverable"] is True
                and 1 <= len(row["publish_attempts"]) <= 3,
                f"{package['name']} retryable auth evidence differs",
            )
        if artifact_root is not None:
            command_root = artifact_root.resolve() if recorded_artifact_root is None else recorded_artifact_root.expanduser().resolve()
            target = str((command_root / "work" / "publish-target").resolve())
            for index, command_ref in enumerate(row["dry_run_attempts"], 1):
                validate_exact_command(
                    command_ref,
                    root=artifact_root,
                    label=f"{package['name']} dry-run attempt {index}",
                    expected_argv=[
                        "cargo", "publish", "--dry-run", "--locked", "-p",
                        package["name"], "--target-dir", target,
                    ],
                )
            for index, command_ref in enumerate(row["publish_attempts"], 1):
                validate_exact_command(
                    command_ref,
                    root=artifact_root,
                    label=f"{package['name']} publish attempt {index}",
                    expected_argv=[
                        "cargo", "publish", "--locked", "-p", package["name"],
                        "--target-dir", target,
                    ],
                )
            if row["state"] == "visible":
                visibility = row["visibility"]
                require(isinstance(visibility, dict) and visibility.get("visible") is True, f"{package['name']} visible state lacks visibility evidence")
                require(visibility.get("api", {}).get("checksum") == package["archive"]["sha256"] and visibility.get("index", {}).get("checksum") == package["archive"]["sha256"], f"{package['name']} receipt visibility checksum differs")
    require(receipt["credential_policy"] == {"source": "existing-cargo-config-or-environment", "secret_values_recorded": False, "token_cli_arguments": False}, "publish credential policy differs")


def persist_receipt(out: Path, receipt: dict[str, Any]) -> None:
    receipt["updated_at"] = iso_now()
    write_json(out / "publish.resume.json", receipt)


def command_pass(receipt: dict[str, Any]) -> bool:
    return receipt.get("exit_code") == 0 and receipt.get("timed_out") is False


def publish_one_crate(
    *,
    repo: Path,
    out: Path,
    prepublish_root: Path,
    package: dict[str, Any],
    row: dict[str, Any],
    receipt: dict[str, Any],
    poll_timeout: int,
    poll_interval: int,
    command_timeout: int,
    probe_fn: Callable[[str, str], dict[str, Any]] = probe_crates_io,
    poll_fn: Callable[..., dict[str, Any]] = poll_crates_io,
    runner: Callable[..., dict[str, Any]] = run_logged_command,
    persist_fn: Callable[[Path, dict[str, Any]], None] = persist_receipt,
    candidate_check: Callable[[], Any] | None = None,
) -> None:
    name = package["name"]
    checksum = package["archive"]["sha256"]
    archive = validate_artifact_ref(package["archive"], root=prepublish_root, label=f"{name} frozen archive", nonempty=True)
    require(sha256_file(archive) == checksum == row["archive_sha256"], f"{name} frozen archive changed")
    observed = probe_fn(name, checksum)
    row["visibility_observations"].append(copy.deepcopy(observed))
    if observed["visible"]:
        row["state"] = "visible"
        row["disposition"] = row["disposition"] or "already-visible-exact-not-retransmitted"
        row["visibility"] = observed
        row["last_error"] = None
        persist_fn(out, receipt)
        return
    if observed["partial"] or row["state"] in {"publish-started", "awaiting-visibility"}:
        row["state"] = "awaiting-visibility"
        persist_fn(out, receipt)
        try:
            visible = poll_fn(name, checksum, timeout_seconds=poll_timeout, interval_seconds=poll_interval)
        except ExternalReleaseError as exc:
            row["last_error"] = {"category": exc.category, "stage": "visibility", "at": iso_now()}
            persist_fn(out, receipt)
            raise
        row["visibility_observations"].append(copy.deepcopy(visible))
        row["visibility"] = visible
        row["state"] = "visible"
        row["disposition"] = row["disposition"] or "published-prior-attempt"
        row["last_error"] = None
        persist_fn(out, receipt)
        return
    if row["state"] == "retryable-auth":
        error = row.get("last_error")
        require(
            isinstance(error, dict)
            and error.get("category") == "auth"
            and error.get("stage") == "publish"
            and error.get("recoverable") is True
            and 1 <= len(row["publish_attempts"]) < 3,
            f"{name} lacks evidence for a recoverable authentication retry",
        )
    else:
        require(row["state"] == "pending" and not row["publish_attempts"], f"{name} cannot start a second upload")
    recovering_auth = row["state"] == "retryable-auth"
    attempt_number = len(row["dry_run_attempts"]) + 1
    require(attempt_number <= 3, f"{name} exceeded the bounded dry-run retry count")
    attempt_root = out / "publish-commands" / name / f"attempt-{attempt_number}"
    target = out / "work" / "publish-target"
    generated = target / "package" / f"{name}-{VERSION}.crate"
    if generated.exists():
        generated.unlink()
    dry_argv = ["cargo", "publish", "--dry-run", "--locked", "-p", name, "--target-dir", str(target)]
    dry_receipt_path = attempt_root / "dry-run.command.json"
    dry_log = attempt_root / "dry-run.log"
    dry = runner(
        dry_argv,
        cwd=repo,
        env=publish_environment(),
        log_path=dry_log,
        receipt_path=dry_receipt_path,
        artifact_root=out,
        expected_seconds=300,
        deadline_seconds=min(command_timeout, 1800),
    )
    row["dry_run_attempts"].append(artifact_ref(dry_receipt_path, root=out))
    if not command_pass(dry):
        if not recovering_auth:
            row["last_error"] = {"category": classify_publish_failure(dry_log), "stage": "dry-run", "at": iso_now()}
        persist_fn(out, receipt)
        raise ReleaseError(f"cargo publish --dry-run --locked failed for {name}")
    require(generated.is_file() and not generated.is_symlink() and sha256_file(generated) == checksum, f"cargo publish dry-run regenerated different bytes for {name}")
    observed = probe_fn(name, checksum)
    row["visibility_observations"].append(copy.deepcopy(observed))
    if observed["visible"] or observed["partial"]:
        row["state"] = "visible" if observed["visible"] else "awaiting-visibility"
        row["disposition"] = "already-visible-exact-not-retransmitted"
        persist_fn(out, receipt)
        if observed["visible"]:
            row["visibility"] = observed
            return
        visible = poll_fn(name, checksum, timeout_seconds=poll_timeout, interval_seconds=poll_interval)
        row["visibility"] = visible
        row["state"] = "visible"
        persist_fn(out, receipt)
        return
    if candidate_check is not None:
        candidate_check()
    row["state"] = "publish-started"
    row["disposition"] = "published-by-this-producer"
    persist_fn(out, receipt)
    publish_argv = ["cargo", "publish", "--locked", "-p", name, "--target-dir", str(target)]
    publish_receipt_path = attempt_root / "publish.command.json"
    publish_log = attempt_root / "publish.log"
    published = runner(
        publish_argv,
        cwd=repo,
        env=publish_environment(),
        log_path=publish_log,
        receipt_path=publish_receipt_path,
        artifact_root=out,
        expected_seconds=300,
        deadline_seconds=min(command_timeout, 1800),
    )
    row["publish_attempts"].append(artifact_ref(publish_receipt_path, root=out))
    require(generated.is_file() and sha256_file(generated) == checksum, f"cargo publish regenerated different bytes for {name}")
    row["state"] = "awaiting-visibility"
    persist_fn(out, receipt)
    if not command_pass(published):
        category = classify_publish_failure(publish_log)
        if category == "auth" and published.get("timed_out") is False:
            row["state"] = "retryable-auth"
            row["disposition"] = "not-published-auth-rejected"
            row["last_error"] = {
                "category": "auth",
                "stage": "publish",
                "recoverable": True,
                "at": iso_now(),
            }
            persist_fn(out, receipt)
            raise ExternalReleaseError(
                "auth",
                f"cargo publish authentication was rejected before acceptance for {name}; fix credentials and resume",
            )
        row["last_error"] = {
            "category": category,
            "stage": "publish",
            "recoverable": False,
            "at": iso_now(),
        }
        persist_fn(out, receipt)
        raise ExternalReleaseError("ambiguous-upload", f"cargo publish outcome is ambiguous for {name}; resume will only probe/poll and never retransmit")
    visible = poll_fn(name, checksum, timeout_seconds=poll_timeout, interval_seconds=poll_interval)
    row["visibility_observations"].append(copy.deepcopy(visible))
    row["visibility"] = visible
    row["state"] = "visible"
    row["last_error"] = None
    persist_fn(out, receipt)


def clean_registry_resolution(
    *,
    out: Path,
    receipt: dict[str, Any],
    timeout_seconds: int,
    logged_runner: Callable[..., dict[str, Any]] = run_logged_command,
    captured_runner: Callable[..., tuple[dict[str, Any], str]] = run_captured_command,
) -> dict[str, Any]:
    state = receipt["clean_resolution"]
    if state["state"] == "pass":
        return state["result"]
    require(state["state"] == "pending", "clean-resolution receipt state differs")
    root = next_stage_attempt(out / "clean-resolution")
    dependencies = "\n".join(f'"{name}" = "={VERSION}"' for name in sorted(EXPECTED_CRATES))
    atomic_write_bytes(root / "Cargo.toml", ("[package]\nname = \"ferrum-v084-resolution-check\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[dependencies]\n" + dependencies + "\n").encode())
    atomic_write_bytes(root / "src" / "lib.rs", b"pub fn registry_resolution_check() {}\n")
    cargo_home = root / "cargo-home"
    environment = base_cargo_environment(cargo_home=cargo_home)
    environment["CARGO_TARGET_DIR"] = str((root / "target").resolve())
    commands = root / "commands"
    lock = logged_runner(
        ["cargo", "generate-lockfile"], cwd=root, env=environment,
        log_path=commands / "generate-lockfile.log", receipt_path=commands / "generate-lockfile.command.json",
        artifact_root=out, expected_seconds=120, deadline_seconds=min(timeout_seconds, 1200),
    )
    require(command_pass(lock), "clean crates.io resolution lock failed")
    metadata_receipt, stdout = captured_runner(
        ["cargo", "metadata", "--locked", "--format-version", "1"], cwd=root, env=environment,
        stdout_path=root / "metadata.json", stderr_path=commands / "metadata.stderr.log",
        receipt_path=commands / "metadata.command.json", artifact_root=out,
        expected_seconds=60, deadline_seconds=min(timeout_seconds, 900),
    )
    require(command_pass(metadata_receipt), "clean crates.io metadata failed")
    try:
        metadata = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ReleaseError(f"clean crates.io metadata is invalid: {exc}") from exc
    selected = {
        row.get("name"): row
        for row in metadata.get("packages", [])
        if isinstance(row, dict) and row.get("name") in EXPECTED_CRATES
    }
    require(set(selected) == set(EXPECTED_CRATES), "clean crates.io resolution omitted a Ferrum crate")
    for name, row in selected.items():
        source = row.get("source")
        require(row.get("version") == VERSION and isinstance(source, str) and source.startswith("registry+") and "crates.io" in source, f"clean resolution for {name} did not use crates.io {VERSION}")
    result = {
        "status": "pass",
        "cargo_lock": artifact_ref(root / "Cargo.lock", root=out),
        "metadata": artifact_ref(root / "metadata.json", root=out),
        "resolved": [{"name": name, "version": VERSION, "source": selected[name]["source"]} for name in sorted(EXPECTED_CRATES)],
        "commands": {"generate_lockfile": artifact_ref(commands / "generate-lockfile.command.json", root=out), "metadata": artifact_ref(commands / "metadata.command.json", root=out)},
    }
    state["state"] = "pass"
    state["result"] = result
    persist_receipt(out, receipt)
    return result


def clean_install(
    *,
    out: Path,
    receipt: dict[str, Any],
    timeout_seconds: int,
    logged_runner: Callable[..., dict[str, Any]] = run_logged_command,
    captured_runner: Callable[..., tuple[dict[str, Any], str]] = run_captured_command,
) -> dict[str, Any]:
    state = receipt["install"]
    if state["state"] == "pass":
        return state["result"]
    require(state["state"] == "pending", "clean-install receipt state differs")
    root = next_stage_attempt(out / "clean-install")
    cargo_home = root / "cargo-home"
    environment = base_cargo_environment(cargo_home=cargo_home)
    install_root = root / "root"
    target = root / "target"
    commands = root / "commands"
    argv = ["cargo", "install", "ferrum-cli", "--version", VERSION, "--locked", "--root", str(install_root), "--target-dir", str(target)]
    installed = logged_runner(
        argv, cwd=root, env=environment,
        log_path=commands / "cargo-install.log", receipt_path=commands / "cargo-install.command.json",
        artifact_root=out, expected_seconds=1200, deadline_seconds=timeout_seconds,
    )
    require(command_pass(installed), "clean cargo install ferrum-cli 0.8.4 failed")
    binary = install_root / "bin" / "ferrum"
    require(binary.is_file() and not binary.is_symlink(), "clean cargo install omitted ferrum")
    version_receipt, version_stdout = captured_runner(
        [str(binary), "--version"], cwd=root, env=environment,
        stdout_path=commands / "ferrum-version.stdout", stderr_path=commands / "ferrum-version.stderr",
        receipt_path=commands / "ferrum-version.command.json", artifact_root=out,
        expected_seconds=10, deadline_seconds=60,
    )
    require(command_pass(version_receipt) and re.fullmatch(rf"ferrum(?:-cli)?\s+{re.escape(VERSION)}(?:\s+.*)?", version_stdout.strip()) is not None, "installed ferrum --version differs")
    help_receipt, help_stdout = captured_runner(
        [str(binary), "--help"], cwd=root, env=environment,
        stdout_path=commands / "ferrum-help.stdout", stderr_path=commands / "ferrum-help.stderr",
        receipt_path=commands / "ferrum-help.command.json", artifact_root=out,
        expected_seconds=10, deadline_seconds=60,
    )
    require(command_pass(help_receipt) and help_stdout.strip() and "usage" in help_stdout.lower(), "installed ferrum --help differs")
    result = {
        "status": "pass", "command": argv,
        "binary": artifact_ref(binary, root=out), "binary_sha256": sha256_file(binary),
        "version_stdout": artifact_ref(commands / "ferrum-version.stdout", root=out),
        "help_stdout": artifact_ref(commands / "ferrum-help.stdout", root=out),
        "commands": {"install": artifact_ref(commands / "cargo-install.command.json", root=out), "version": artifact_ref(commands / "ferrum-version.command.json", root=out), "help": artifact_ref(commands / "ferrum-help.command.json", root=out)},
    }
    state["state"] = "pass"
    state["result"] = result
    persist_receipt(out, receipt)
    return result


def validate_clean_resolution(value: Any, *, root: Path) -> None:
    row = exact_fields(value, {"status", "cargo_lock", "metadata", "resolved", "commands"}, "clean registry resolution")
    require(row["status"] == "pass", "clean registry resolution did not PASS")
    validate_artifact_ref(row["cargo_lock"], root=root, label="clean resolution Cargo.lock", nonempty=True)
    metadata_path = validate_artifact_ref(row["metadata"], root=root, label="clean resolution metadata", nonempty=True)
    metadata = read_json(metadata_path, "clean resolution metadata")
    resolved = row["resolved"]
    require(isinstance(resolved, list) and {item.get("name") for item in resolved if isinstance(item, dict)} == set(EXPECTED_CRATES), "clean resolution roster differs")
    for item in resolved:
        require(item.get("version") == VERSION and isinstance(item.get("source"), str) and "crates.io" in item["source"], "clean resolution source/version differs")
    metadata_rows = {(item.get("name"), item.get("version")) for item in metadata.get("packages", []) if isinstance(item, dict)}
    require(all((name, VERSION) in metadata_rows for name in EXPECTED_CRATES), "clean metadata roster/version differs")
    commands = exact_fields(row["commands"], {"generate_lockfile", "metadata"}, "clean resolution commands")
    validate_exact_command(
        commands["generate_lockfile"],
        root=root,
        label="clean resolution generate-lockfile command",
        expected_argv=["cargo", "generate-lockfile"],
    )
    validate_exact_command(
        commands["metadata"],
        root=root,
        label="clean resolution metadata command",
        expected_argv=["cargo", "metadata", "--locked", "--format-version", "1"],
    )


def validate_install(value: Any, *, root: Path, recorded_root: Path | None = None) -> None:
    row = exact_fields(value, {"status", "command", "binary", "binary_sha256", "version_stdout", "help_stdout", "commands"}, "clean install")
    require(row["status"] == "pass", "clean install did not PASS")
    binary = validate_artifact_ref(row["binary"], root=root, label="clean installed ferrum", nonempty=True)
    require(row["binary_sha256"] == sha256_file(binary), "clean installed binary SHA differs")
    version_path = validate_artifact_ref(row["version_stdout"], root=root, label="clean installed version stdout", nonempty=True)
    help_path = validate_artifact_ref(row["help_stdout"], root=root, label="clean installed help stdout", nonempty=True)
    require(re.fullmatch(rf"ferrum(?:-cli)?\s+{re.escape(VERSION)}(?:\s+.*)?", version_path.read_text().strip()) is not None, "clean installed version stdout differs")
    help_text = help_path.read_text(encoding="utf-8", errors="strict")
    require("usage" in help_text.lower() and "ferrum" in help_text.lower(), "clean installed help stdout differs")
    install_root = binary.parent.parent.resolve()
    require(binary == install_root / "bin" / "ferrum", "clean installed binary is not <root>/bin/ferrum")
    command_artifact_root = root.resolve() if recorded_root is None else recorded_root.expanduser().resolve()
    binary_ref_path = PurePosixPath(row["binary"]["path"])
    recorded_binary = command_artifact_root.joinpath(*binary_ref_path.parts).resolve()
    recorded_install_root = recorded_binary.parent.parent
    require(recorded_binary == recorded_install_root / "bin" / "ferrum", "recorded clean install binary is not <root>/bin/ferrum")
    target = recorded_install_root.parent / "target"
    expected_install_argv = [
        "cargo", "install", "ferrum-cli", "--version", VERSION, "--locked",
        "--root", str(recorded_install_root), "--target-dir", str(target),
    ]
    require(
        row["command"] == expected_install_argv,
        f"clean install manifest argv differs: observed={row['command']!r} expected={expected_install_argv!r}",
    )
    commands = exact_fields(row["commands"], {"install", "version", "help"}, "clean install commands")
    validate_exact_command(
        commands["install"],
        root=root,
        label="clean install command",
        expected_argv=expected_install_argv,
    )
    validate_exact_command(
        commands["version"],
        root=root,
        label="clean install version command",
        expected_argv=[str(recorded_binary), "--version"],
    )
    validate_exact_command(
        commands["help"],
        root=root,
        label="clean install help command",
        expected_argv=[str(recorded_binary), "--help"],
    )


PUBLISH_FIELDS = {
    "schema_version", "artifact_type", "status", "lane", "version", "canonical",
    "release_candidate", "prepublish", "publish_order", "packages",
    "cargo_workspace_crates", "clean_resolution", "install", "resume_receipt",
    "created_at", "credential_policy", "artifact_dir", "manifest_id", "pass_line",
}


def validate_publish_manifest(
    path: Path,
    *,
    repo: Path | None = None,
    require_receipt_pass: bool = True,
) -> dict[str, Any]:
    manifest_path = resolve_manifest_path(path, ("crates-io.manifest.json", "gate.manifest.json"))
    root = manifest_path.parent.resolve()
    value = exact_fields(read_json(manifest_path, "publish manifest"), PUBLISH_FIELDS, "publish manifest")
    require(value["schema_version"] == SCHEMA_VERSION and value["artifact_type"] == "ferrum_v084_crates_io_publish_manifest", "publish schema/type differs")
    require(value["status"] == "pass" and value["lane"] == "runtime-vnext-crates-io" and value["version"] == VERSION and value["canonical"] is True, "publish status/lane/version differs")
    artifact_dir = value["artifact_dir"]
    require(
        isinstance(artifact_dir, str) and artifact_dir and Path(artifact_dir).is_absolute(),
        "publish artifact_dir must be an absolute recorded path",
    )
    recorded_root = Path(artifact_dir).expanduser().resolve()
    candidate = validate_candidate(value["release_candidate"], "publish.release_candidate")
    prepublish, prepublish_path = validate_portable_prepublish_binding(
        value["prepublish"],
        artifact_root=root,
        repo=repo,
    )
    require(candidate == prepublish["release_candidate"], "publish/prepublish identity differs")
    order = prepublish["topology"]["order"]
    require(value["publish_order"] == order, "publish order differs from prepublish topology")
    rows = value["packages"]
    require(isinstance(rows, list) and len(rows) == len(EXPECTED_CRATES), "publish package rows differ")
    for position, (name, package, row) in enumerate(zip(order, prepublish["packages"], rows), 1):
        exact_fields(row, {"position", "name", "version", "archive_sha256", "crates_io_visible", "disposition", "api_checksum", "index_checksum"}, f"publish package {name}")
        require(row["position"] == position and row["name"] == name and row["version"] == VERSION and row["archive_sha256"] == package["archive"]["sha256"], f"publish package binding differs: {name}")
        require(row["crates_io_visible"] is True and row["api_checksum"] == row["index_checksum"] == row["archive_sha256"], f"publish visibility/checksum differs: {name}")
        require(isinstance(row["disposition"], str) and row["disposition"], f"publish disposition is absent: {name}")
    workspace = value["cargo_workspace_crates"]
    require(isinstance(workspace, list) and len(workspace) == len(EXPECTED_CRATES) and {item.get("name") for item in workspace if isinstance(item, dict)} == set(EXPECTED_CRATES), "published workspace roster differs")
    require(all(item.get("version") == VERSION and item.get("crates_io_visible") is True for item in workspace), "published workspace version/visibility differs")
    validate_clean_resolution(value["clean_resolution"], root=root)
    validate_install(value["install"], root=root, recorded_root=recorded_root)
    receipt_path = validate_artifact_ref(value["resume_receipt"], root=root, label="publish resume receipt", nonempty=True)
    receipt = read_json(receipt_path, "publish resume receipt")
    validate_publish_receipt(
        receipt,
        prepublish=prepublish,
        prepublish_path=prepublish_path,
        artifact_root=root,
        recorded_artifact_root=recorded_root,
    )
    require(receipt["prepublish"] == value["prepublish"], "publish/receipt portable prepublish binding differs")
    require(
        receipt["clean_resolution"] == {"state": "pass", "result": value["clean_resolution"]}
        and receipt["install"] == {"state": "pass", "result": value["install"]},
        "publish receipt final-stage results differ from the manifest",
    )
    expected_receipt_status = "pass" if require_receipt_pass else "in_progress"
    require(
        receipt["status"] == expected_receipt_status
        and all(item["state"] == "visible" for item in receipt["packages"]),
        "publish resume receipt is incomplete",
    )
    require(value["credential_policy"] == {"source": "existing-cargo-config-or-environment", "secret_values_recorded": False, "token_cli_arguments": False}, "publish credential policy differs")
    expected_id = manifest_identity(value, ("schema_version", "artifact_type", "version", "release_candidate", "prepublish", "publish_order", "packages", "clean_resolution", "install", "artifact_dir"))
    require(value["manifest_id"] == expected_id, "publish manifest identity differs")
    require(value["pass_line"] == f"{PUBLISH_PASS_PREFIX}: {artifact_dir}", "publish exact PASS line differs")
    return value


def has_unredacted_secret_marker(text: str) -> bool:
    for pattern in (
        r"(?i)(?:CARGO_(?:REGISTRY_TOKEN|REGISTRIES_[A-Z0-9_]+_TOKEN|TOKEN)|CRATES_IO_TOKEN)\s*=\s*(?!<redacted>)[^\s]+",
        r"(?i)authorization\s*:\s*bearer\s+(?!<redacted>)[^\s]+",
        r"(?i)\btoken\s*[=:]\s*(?!<redacted>)[\"']?[^\s\"']+",
    ):
        if re.search(pattern, text):
            return True
    return False


def assert_no_secret_output(root: Path) -> None:
    secret_values = {
        value
        for key, value in os.environ.items()
        if value and len(value) >= 8 and re.search(r"(?i)(?:token|secret|password|api[_-]?key)", key)
    }
    text_suffixes = {".json", ".log", ".stdout", ".stderr", ".txt"}
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        require(not has_unredacted_secret_marker(text), f"unredacted credential marker found in artifact {path}")
        require(not any(secret in text for secret in secret_values), f"credential value found in artifact {path}")


def publish_release(args: argparse.Namespace) -> Path:
    repo = args.repo.expanduser().resolve()
    out = args.out.expanduser().resolve()
    require(not out.is_relative_to(repo), "publish output must be outside the release checkout")
    receipt_path = out / "publish.resume.json"
    if out.exists():
        require(args.resume, f"publish output exists; use --resume: {out}")
        require(receipt_path.is_file(), f"publish resume receipt is missing: {receipt_path}")
        receipt = read_json(receipt_path, "publish resume receipt")
        prepublish, prepublish_path = validate_portable_prepublish_binding(
            receipt.get("prepublish"),
            artifact_root=out,
            repo=repo,
        )
    else:
        require(not args.resume, "--resume requires an existing publish output")
        source_prepublish, source_prepublish_path = validate_prepublish_manifest(
            args.prepublish,
            repo=repo,
        )
        out.mkdir(parents=True)
        binding, prepublish_path = copy_prepublish_evidence(
            prepublish=source_prepublish,
            prepublish_path=source_prepublish_path,
            out=out,
        )
        prepublish, _ = validate_portable_prepublish_binding(
            binding,
            artifact_root=out,
            repo=repo,
        )
        receipt = new_publish_receipt(
            prepublish,
            prepublish_path,
            prepublish_binding=binding,
            artifact_root=out,
        )
        persist_receipt(out, receipt)
    candidate = clean_release_candidate(repo, expected_sha=prepublish["release_candidate"]["git_sha"], tag=prepublish["release_candidate"]["tag"]["name"])
    require(candidate == prepublish["release_candidate"], "current release candidate/tag differs from prepublish")
    validate_publish_receipt(receipt, prepublish=prepublish, prepublish_path=prepublish_path, artifact_root=out)
    completed = out / "crates-io.manifest.json"
    alias = out / "gate.manifest.json"
    existing_manifest = completed if completed.is_file() else alias if alias.is_file() else None
    if receipt["status"] == "pass" and existing_manifest is not None:
        manifest = validate_publish_manifest(existing_manifest, repo=repo)
        install_manifest_pair(
            root=out,
            primary_name="crates-io.manifest.json",
            alias_name="gate.manifest.json",
            value=manifest,
            validator=lambda path: validate_publish_manifest(path, repo=repo),
            allow_existing_identical=True,
        )
        clean_release_candidate(repo, expected_sha=candidate["git_sha"], tag=candidate["tag"]["name"])
        print(manifest["pass_line"])
        return out
    if receipt["status"] == "pass":
        # Compatibility recovery for receipts written by the original 0.8.4
        # implementation before its final manifest was installed.
        receipt["status"] = "in_progress"
        persist_receipt(out, receipt)
    if receipt["status"] == "in_progress" and existing_manifest is not None:
        manifest = validate_publish_manifest(
            existing_manifest,
            repo=repo,
            require_receipt_pass=False,
        )
        install_manifest_pair(
            root=out,
            primary_name="crates-io.manifest.json",
            alias_name="gate.manifest.json",
            value=manifest,
            validator=lambda path: validate_publish_manifest(
                path,
                repo=repo,
                require_receipt_pass=False,
            ),
            allow_existing_identical=True,
        )
        receipt["status"] = "pass"
        persist_receipt(out, receipt)
        manifest = validate_publish_manifest(completed, repo=repo)
        clean_release_candidate(repo, expected_sha=candidate["git_sha"], tag=candidate["tag"]["name"])
        print(manifest["pass_line"])
        return out
    require(receipt["status"] == "in_progress", "publish receipt state differs")
    packages_by_name = {row["name"]: row for row in prepublish["packages"]}
    rows_by_name = {row["name"]: row for row in receipt["packages"]}

    def candidate_check() -> dict[str, Any]:
        return clean_release_candidate(repo, expected_sha=candidate["git_sha"], tag=candidate["tag"]["name"])

    for name in prepublish["topology"]["order"]:
        publish_one_crate(
            repo=repo,
            out=out,
            prepublish_root=prepublish_path.parent,
            package=packages_by_name[name],
            row=rows_by_name[name],
            receipt=receipt,
            poll_timeout=args.visibility_timeout_seconds,
            poll_interval=args.visibility_interval_seconds,
            command_timeout=args.command_timeout_seconds,
            candidate_check=candidate_check,
        )
        require(rows_by_name[name]["state"] == "visible", f"{name} did not reach visible state")
    resolution = clean_registry_resolution(out=out, receipt=receipt, timeout_seconds=args.command_timeout_seconds)
    install = clean_install(out=out, receipt=receipt, timeout_seconds=args.command_timeout_seconds)
    final_rows: list[dict[str, Any]] = []
    for package in prepublish["packages"]:
        name = package["name"]
        checksum = package["archive"]["sha256"]
        visibility = probe_crates_io(name, checksum)
        require(visibility["visible"], f"final API/index visibility re-read failed for {name}")
        row = rows_by_name[name]
        final_rows.append({"position": package["position"], "name": name, "version": VERSION, "archive_sha256": checksum, "crates_io_visible": True, "disposition": row["disposition"], "api_checksum": visibility["api"]["checksum"], "index_checksum": visibility["index"]["checksum"]})
    candidate_check()
    assert_no_secret_output(out)
    binding = copy.deepcopy(receipt["prepublish"])
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ferrum_v084_crates_io_publish_manifest",
        "status": "pass",
        "lane": "runtime-vnext-crates-io",
        "version": VERSION,
        "canonical": True,
        "release_candidate": candidate,
        "prepublish": binding,
        "publish_order": prepublish["topology"]["order"],
        "packages": final_rows,
        "cargo_workspace_crates": [{"name": row["name"], "version": VERSION, "crates_io_visible": True} for row in final_rows],
        "clean_resolution": resolution,
        "install": install,
        "resume_receipt": artifact_ref(receipt_path, root=out),
        "created_at": iso_now(),
        "credential_policy": {"source": "existing-cargo-config-or-environment", "secret_values_recorded": False, "token_cli_arguments": False},
        "artifact_dir": str(out),
        "manifest_id": "",
        "pass_line": f"{PUBLISH_PASS_PREFIX}: {out}",
    }
    manifest["manifest_id"] = manifest_identity(manifest, ("schema_version", "artifact_type", "version", "release_candidate", "prepublish", "publish_order", "packages", "clean_resolution", "install", "artifact_dir"))
    install_manifest_pair(
        root=out,
        primary_name="crates-io.manifest.json",
        alias_name="gate.manifest.json",
        value=manifest,
        validator=lambda path: validate_publish_manifest(
            path,
            repo=repo,
            require_receipt_pass=False,
        ),
    )
    receipt["status"] = "pass"
    persist_receipt(out, receipt)
    validate_publish_manifest(completed, repo=repo)
    assert_no_secret_output(out)
    print(manifest["pass_line"])
    return out


def make_fixture_archive(path: Path, *, name: str, candidate_sha: str, source_manifest: bytes) -> None:
    root = f"{name}-{VERSION}"
    normalized = source_manifest
    files = {
        f"{root}/Cargo.toml": normalized,
        f"{root}/Cargo.toml.orig": source_manifest,
        f"{root}/.cargo_vcs_info.json": (json.dumps({"git": {"sha1": candidate_sha, "dirty": False}}) + "\n").encode(),
        f"{root}/src/lib.rs": b"pub fn fixture() -> u32 { 804 }\n",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as bundle:
        for filename, payload in sorted(files.items()):
            info = tarfile.TarInfo(filename)
            info.size = len(payload)
            info.mode = 0o644
            bundle.addfile(info, io.BytesIO(payload))


def fixture_metadata(root: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    names = sorted(EXPECTED_CRATES)
    manifests: dict[str, bytes] = {}
    packages = []
    for index, name in enumerate(names):
        dependency = names[index - 1] if index else None
        dependency_text = "" if dependency is None else f'\n[dependencies]\n"{dependency}" = "^{VERSION}"\n'
        manifest = (f'[package]\nname = "{name}"\nversion = "{VERSION}"\nedition = "2021"\nlicense = "MIT"\n' + dependency_text).encode()
        manifests[name] = manifest
        manifest_path = root / "source" / name / "Cargo.toml"
        atomic_write_bytes(manifest_path, manifest)
        dependencies = []
        if dependency is not None:
            dependencies.append({"name": dependency, "req": f"^{VERSION}", "kind": None, "target": None})
        packages.append({"id": f"fixture#{name}", "name": name, "version": VERSION, "publish": None, "manifest_path": str(manifest_path), "dependencies": dependencies})
    return {"packages": packages, "workspace_members": [row["id"] for row in packages]}, manifests


def fixture_command_receipt(root: Path, *, name: str, argv: list[str]) -> dict[str, Any]:
    command_root = root / "commands"
    log = command_root / f"{name}.log"
    receipt_path = command_root / f"{name}.json"
    atomic_write_bytes(log, b"offline fixture command PASS\n")
    receipt = {"schema_version": 1, "argv": argv, "exit_code": 0, "timed_out": False, "credential_values_recorded": False, "log": artifact_ref(log, root=root)}
    write_json(receipt_path, receipt)
    return artifact_ref(receipt_path, root=root)


def build_fixture_prepublish(root: Path) -> Path:
    candidate_sha = "a" * 40
    metadata, manifests = fixture_metadata(root)
    metadata_path = root / "metadata.json"
    write_json(metadata_path, metadata)
    packages = package_map_from_metadata(metadata)
    graph = internal_dependency_graph(packages)
    order = stable_topological_order(graph)
    metadata_command = fixture_command_receipt(root, name="metadata", argv=["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"])
    version_command = fixture_command_receipt(root, name="version", argv=["cargo", "--version"])
    rows = []
    for position, name in enumerate(order, 1):
        archive = root / "archives" / f"{name}-{VERSION}.crate"
        make_fixture_archive(archive, name=name, candidate_sha=candidate_sha, source_manifest=manifests[name])
        package_command = fixture_command_receipt(root, name=f"package-{name}", argv=["cargo", "package", "--workspace", "--locked", "--no-verify", "--target-dir", "/offline/target"])
        source_sha = sha256_bytes(manifests[name])
        rows.append({"position": position, "name": name, "version": VERSION, "manifest_path": f"source/{name}/Cargo.toml", "source_manifest_sha256": source_sha, "source_manifest_git_blob_sha": hashlib.sha1(manifests[name]).hexdigest(), "internal_dependencies": graph[name], "archive": artifact_ref(archive, root=root), "archive_inspection": inspect_crate_archive(archive, expected_name=name, candidate_sha=candidate_sha, source_manifest_sha256=source_sha), "package_command": package_command})
    candidate = {"git_sha": candidate_sha, "git_tree_sha": "b" * 40, "dirty": False, "tag": {"name": "v0.8.4-rc.1", "object_sha": "c" * 40, "peeled_commit_sha": candidate_sha}}
    manifest: dict[str, Any] = {
        "schema_version": 1, "artifact_type": "ferrum_v084_crates_io_prepublish_manifest", "status": "pass", "mode": "prepublish", "version": VERSION, "canonical": True,
        "release_candidate": candidate,
        "cargo": {"version": "cargo 1.91.0", "metadata": artifact_ref(metadata_path, root=root), "metadata_command": metadata_command, "version_command": version_command, "worker_bounds": {"cargo_build_jobs": 2, "rust_test_threads": 8}},
        "topology": {"algorithm": "kahn-lexicographic-v1", "crate_count": len(order), "graph": graph, "graph_sha256": sha256_bytes(canonical_json(graph)), "order": order},
        "packages": rows, "created_at": iso_now(), "credential_policy": {"secret_values_recorded": False, "token_cli_arguments": False},
        "does_not_prove": ["offline fixture"], "manifest_id": "", "pass_line": f"{PREPUBLISH_PASS_PREFIX}: {root.resolve()}",
    }
    manifest["manifest_id"] = manifest_identity(manifest, ("schema_version", "artifact_type", "version", "release_candidate", "cargo", "topology", "packages"))
    path = root / "prepublish.manifest.json"
    write_json(path, manifest)
    return path


def expect_failure(label: str, operation: Callable[[], Any], contains: str) -> None:
    try:
        operation()
    except (ReleaseError, legacy.ReleaseError) as exc:
        require(contains in str(exc), f"{label} failed for the wrong reason: {exc}")
    else:
        raise ReleaseError(f"negative self-test unexpectedly passed: {label}")


def run_state_machine_selftest(root: Path, prepublish: dict[str, Any], prepublish_path: Path) -> None:
    package = prepublish["packages"][0]
    row = new_publish_receipt(prepublish, prepublish_path)["packages"][0]
    visible = {"visible": True, "partial": True, "api": {"state": "visible", "checksum": package["archive"]["sha256"]}, "index": {"state": "visible", "checksum": package["archive"]["sha256"]}, "observed_at": iso_now()}

    def forbidden_runner(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ReleaseError("idempotent path attempted cargo publish")

    publish_one_crate(repo=root, out=root / "idempotent", prepublish_root=prepublish_path.parent, package=package, row=row, receipt={"fixture": True}, poll_timeout=1, poll_interval=1, command_timeout=1200, probe_fn=lambda name, checksum: visible, runner=forbidden_runner, persist_fn=lambda out, receipt: None)
    require(row["state"] == "visible" and not row["publish_attempts"], "already-visible exact package was retransmitted")

    packages = prepublish["packages"][:2]
    receipt = {"fixture": True}
    rows = [copy.deepcopy(new_publish_receipt(prepublish, prepublish_path)["packages"][index]) for index in range(2)]
    remote = {package["name"]: False for package in packages}
    commands: list[tuple[str, str]] = []
    archive_by_name = {package["name"]: validate_artifact_ref(package["archive"], root=prepublish_path.parent, label="fixture archive", nonempty=True) for package in packages}

    def probe(name: str, checksum: str) -> dict[str, Any]:
        if remote[name]:
            return {**visible, "api": {"state": "visible", "checksum": checksum}, "index": {"state": "visible", "checksum": checksum}}
        return {"visible": False, "partial": False, "api": {"state": "missing", "checksum": None}, "index": {"state": "missing", "checksum": None}, "observed_at": iso_now()}

    def poll(name: str, checksum: str, **kwargs: Any) -> dict[str, Any]:
        require(remote[name], f"offline fixture polled before fake publication: {name}")
        return probe(name, checksum)

    def runner(argv: list[str], **kwargs: Any) -> dict[str, Any]:
        name = argv[argv.index("-p") + 1]
        stage = "dry-run" if "--dry-run" in argv else "publish"
        commands.append((stage, name))
        target = Path(argv[argv.index("--target-dir") + 1])
        generated = target / "package" / f"{name}-{VERSION}.crate"
        generated.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive_by_name[name], generated)
        if stage == "publish":
            remote[name] = True
        log_path = kwargs["log_path"]
        receipt_path = kwargs["receipt_path"]
        atomic_write_bytes(log_path, b"offline fake cargo command PASS\n")
        command_receipt = {"schema_version": 1, "argv": argv, "exit_code": 0, "timed_out": False, "credential_values_recorded": False, "log": artifact_ref(log_path, root=kwargs["artifact_root"])}
        write_json(receipt_path, command_receipt)
        return command_receipt

    out = root / "publish-state"
    out.mkdir()
    for package, state_row in zip(packages, rows):
        publish_one_crate(repo=root, out=out, prepublish_root=prepublish_path.parent, package=package, row=state_row, receipt=receipt, poll_timeout=1, poll_interval=1, command_timeout=1200, probe_fn=probe, poll_fn=poll, runner=runner, persist_fn=lambda out, receipt: None)
    expected = [(stage, package["name"]) for package in packages for stage in ("dry-run", "publish")]
    require(commands == expected, f"offline publish order was not serialized: {commands}")

    package = prepublish["packages"][0]
    auth_receipt = new_publish_receipt(prepublish, prepublish_path)
    auth_row = auth_receipt["packages"][0]
    auth_out = root / "auth-resume"
    auth_out.mkdir()
    auth_remote = False
    auth_publish_attempts = 0

    def auth_probe(name: str, checksum: str) -> dict[str, Any]:
        if auth_remote:
            return {**visible, "api": {"state": "visible", "checksum": checksum}, "index": {"state": "visible", "checksum": checksum}}
        return {"visible": False, "partial": False, "api": {"state": "missing", "checksum": None}, "index": {"state": "missing", "checksum": None}, "observed_at": iso_now()}

    def auth_runner(argv: list[str], **kwargs: Any) -> dict[str, Any]:
        nonlocal auth_publish_attempts, auth_remote
        stage = "dry-run" if "--dry-run" in argv else "publish"
        target = Path(argv[argv.index("--target-dir") + 1])
        generated = target / "package" / f"{package['name']}-{VERSION}.crate"
        generated.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive_by_name[package["name"]], generated)
        exit_code = 0
        log = b"offline fake cargo command PASS\n"
        if stage == "publish":
            auth_publish_attempts += 1
            if auth_publish_attempts == 1:
                exit_code = 1
                log = b"error: unauthorized: invalid token\n"
            else:
                auth_remote = True
        atomic_write_bytes(kwargs["log_path"], log)
        command_receipt = {
            "schema_version": 1,
            "argv": argv,
            "exit_code": exit_code,
            "timed_out": False,
            "credential_values_recorded": False,
            "log": artifact_ref(kwargs["log_path"], root=kwargs["artifact_root"]),
        }
        write_json(kwargs["receipt_path"], command_receipt)
        return command_receipt

    try:
        publish_one_crate(
            repo=root, out=auth_out, prepublish_root=prepublish_path.parent,
            package=package, row=auth_row, receipt=auth_receipt,
            poll_timeout=1, poll_interval=1, command_timeout=1200,
            probe_fn=auth_probe, poll_fn=lambda name, checksum, **kwargs: auth_probe(name, checksum),
            runner=auth_runner, persist_fn=lambda out, receipt: None,
        )
        raise ReleaseError("authentication rejection unexpectedly passed")
    except ExternalReleaseError as exc:
        require(exc.category == "auth", f"authentication rejection category differs: {exc.category}")
    validate_publish_receipt(auth_receipt, prepublish=prepublish, prepublish_path=prepublish_path)
    require(auth_row["state"] == "retryable-auth", "authentication rejection was not resumable")
    publish_one_crate(
        repo=root, out=auth_out, prepublish_root=prepublish_path.parent,
        package=package, row=auth_row, receipt=auth_receipt,
        poll_timeout=1, poll_interval=1, command_timeout=1200,
        probe_fn=auth_probe, poll_fn=lambda name, checksum, **kwargs: auth_probe(name, checksum),
        runner=auth_runner, persist_fn=lambda out, receipt: None,
    )
    require(
        auth_row["state"] == "visible"
        and auth_publish_attempts == 2
        and len(auth_row["publish_attempts"]) == 2,
        "authentication resume did not perform exactly one evidence-backed retry",
    )

    network_receipt = new_publish_receipt(prepublish, prepublish_path)
    network_row = network_receipt["packages"][0]
    network_out = root / "network-resume"
    network_out.mkdir()
    network_runner_calls = 0

    classifier_root = root / "classifier"
    auth_only_log = classifier_root / "auth-only.log"
    mixed_log = classifier_root / "mixed.log"
    already_exists_log = classifier_root / "already-exists.log"
    atomic_write_bytes(auth_only_log, b"error: authentication failed: invalid token\n")
    atomic_write_bytes(mixed_log, b"error: unauthorized (403) after TLS connection timeout\n")
    atomic_write_bytes(already_exists_log, b"error: crate version already exists\n")
    require(classify_publish_failure(auth_only_log) == "auth", "terminal auth-only failure was not retryable")
    require(classify_publish_failure(mixed_log) == "ambiguous", "mixed auth/network failure was not ambiguous")
    require(classify_publish_failure(already_exists_log) == "ambiguous", "already-exists outcome was not ambiguous")

    def missing_probe(name: str, checksum: str) -> dict[str, Any]:
        return {"visible": False, "partial": False, "api": {"state": "missing", "checksum": None}, "index": {"state": "missing", "checksum": None}, "observed_at": iso_now()}

    def network_runner(argv: list[str], **kwargs: Any) -> dict[str, Any]:
        nonlocal network_runner_calls
        network_runner_calls += 1
        target = Path(argv[argv.index("--target-dir") + 1])
        generated = target / "package" / f"{package['name']}-{VERSION}.crate"
        generated.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive_by_name[package["name"]], generated)
        publish = "--dry-run" not in argv
        atomic_write_bytes(
            kwargs["log_path"],
            b"error: unauthorized (403) after TLS connection timeout\n" if publish else b"offline dry-run PASS\n",
        )
        value = {
            "schema_version": 1,
            "argv": argv,
            "exit_code": 1 if publish else 0,
            "timed_out": False,
            "credential_values_recorded": False,
            "log": artifact_ref(kwargs["log_path"], root=kwargs["artifact_root"]),
        }
        write_json(kwargs["receipt_path"], value)
        return value

    try:
        publish_one_crate(
            repo=root, out=network_out, prepublish_root=prepublish_path.parent,
            package=package, row=network_row, receipt=network_receipt,
            poll_timeout=1, poll_interval=1, command_timeout=1200,
            probe_fn=missing_probe, runner=network_runner,
            persist_fn=lambda out, receipt: None,
        )
        raise ReleaseError("ambiguous network upload unexpectedly passed")
    except ExternalReleaseError as exc:
        require(exc.category == "ambiguous-upload", f"network failure category differs: {exc.category}")
    first_network_calls = network_runner_calls
    try:
        publish_one_crate(
            repo=root, out=network_out, prepublish_root=prepublish_path.parent,
            package=package, row=network_row, receipt=network_receipt,
            poll_timeout=1, poll_interval=1, command_timeout=1200,
            probe_fn=missing_probe,
            poll_fn=lambda name, checksum, **kwargs: (_ for _ in ()).throw(
                ExternalReleaseError("visibility", "offline visibility remains absent")
            ),
            runner=network_runner, persist_fn=lambda out, receipt: None,
        )
        raise ReleaseError("ambiguous network resume unexpectedly passed")
    except ExternalReleaseError as exc:
        require(exc.category == "visibility", f"network resume category differs: {exc.category}")
    require(
        network_runner_calls == first_network_calls
        and len(network_row["publish_attempts"]) == 1,
        "ambiguous network outcome was retransmitted",
    )


def run_final_stage_resume_selftest(root: Path, prepublish: dict[str, Any], prepublish_path: Path) -> None:
    receipt = new_publish_receipt(prepublish, prepublish_path)
    out = root / "final-stage-resume"
    out.mkdir()
    resolution_calls = 0

    def command_receipt(argv: list[str], kwargs: dict[str, Any], *, exit_code: int, text: bytes) -> dict[str, Any]:
        atomic_write_bytes(kwargs["log_path"], text)
        value = {
            "schema_version": 1,
            "argv": argv,
            "exit_code": exit_code,
            "timed_out": False,
            "credential_values_recorded": False,
            "log": artifact_ref(kwargs["log_path"], root=kwargs["artifact_root"]),
        }
        write_json(kwargs["receipt_path"], value)
        return value

    def resolution_logged(argv: list[str], **kwargs: Any) -> dict[str, Any]:
        nonlocal resolution_calls
        resolution_calls += 1
        if resolution_calls == 1:
            return command_receipt(argv, kwargs, exit_code=1, text=b"temporary registry failure\n")
        atomic_write_bytes(Path(kwargs["cwd"]) / "Cargo.lock", b"# offline lock fixture\n")
        return command_receipt(argv, kwargs, exit_code=0, text=b"offline lock PASS\n")

    def resolution_captured(argv: list[str], **kwargs: Any) -> tuple[dict[str, Any], str]:
        stdout = json.dumps(
            {
                "packages": [
                    {"name": name, "version": VERSION, "source": "registry+https://github.com/rust-lang/crates.io-index"}
                    for name in sorted(EXPECTED_CRATES)
                ]
            }
        )
        atomic_write_bytes(kwargs["stdout_path"], stdout.encode())
        atomic_write_bytes(kwargs["stderr_path"], b"")
        value = command_receipt(argv, {**kwargs, "log_path": kwargs["stderr_path"]}, exit_code=0, text=b"")
        return value, stdout

    expect_failure(
        "clean resolution first attempt",
        lambda: clean_registry_resolution(
            out=out, receipt=receipt, timeout_seconds=1200,
            logged_runner=resolution_logged, captured_runner=resolution_captured,
        ),
        "clean crates.io resolution lock failed",
    )
    resolution = clean_registry_resolution(
        out=out, receipt=receipt, timeout_seconds=1200,
        logged_runner=resolution_logged, captured_runner=resolution_captured,
    )
    require(
        resolution["status"] == "pass"
        and (out / "clean-resolution/attempt-1").is_dir()
        and (out / "clean-resolution/attempt-2").is_dir(),
        "clean resolution did not retain and resume immutable attempts",
    )

    install_calls = 0

    def install_logged(argv: list[str], **kwargs: Any) -> dict[str, Any]:
        nonlocal install_calls
        install_calls += 1
        if install_calls == 1:
            return command_receipt(argv, kwargs, exit_code=1, text=b"temporary install failure\n")
        install_root = Path(argv[argv.index("--root") + 1])
        binary = install_root / "bin/ferrum"
        atomic_write_bytes(binary, b"#!/bin/sh\nexit 0\n")
        return command_receipt(argv, kwargs, exit_code=0, text=b"offline install PASS\n")

    def install_captured(argv: list[str], **kwargs: Any) -> tuple[dict[str, Any], str]:
        stdout = f"ferrum {VERSION}\n" if argv[-1] == "--version" else "Usage: ferrum [COMMAND]\n"
        atomic_write_bytes(kwargs["stdout_path"], stdout.encode())
        atomic_write_bytes(kwargs["stderr_path"], b"")
        value = command_receipt(argv, {**kwargs, "log_path": kwargs["stderr_path"]}, exit_code=0, text=b"")
        return value, stdout

    expect_failure(
        "clean install first attempt",
        lambda: clean_install(
            out=out, receipt=receipt, timeout_seconds=1200,
            logged_runner=install_logged, captured_runner=install_captured,
        ),
        "clean cargo install ferrum-cli 0.8.4 failed",
    )
    install = clean_install(
        out=out, receipt=receipt, timeout_seconds=1200,
        logged_runner=install_logged, captured_runner=install_captured,
    )
    require(
        install["status"] == "pass"
        and (out / "clean-install/attempt-1").is_dir()
        and (out / "clean-install/attempt-2").is_dir(),
        "clean install did not retain and resume immutable attempts",
    )


def run_portable_publish_manifest_selftest(root: Path) -> None:
    source_root = root / "portable-source"
    source_manifest = build_fixture_prepublish(source_root)
    source_prepublish, _ = validate_prepublish_manifest(source_manifest)
    out = root / "portable-publish"
    out.mkdir()
    binding, copied_manifest = copy_prepublish_evidence(
        prepublish=source_prepublish,
        prepublish_path=source_manifest,
        out=out,
    )
    prepublish, _ = validate_portable_prepublish_binding(binding, artifact_root=out)
    receipt = new_publish_receipt(
        prepublish,
        copied_manifest,
        prepublish_binding=binding,
        artifact_root=out,
    )

    for package, row in zip(prepublish["packages"], receipt["packages"]):
        checksum = package["archive"]["sha256"]
        visible = {
            "visible": True,
            "partial": True,
            "api": {"state": "visible", "checksum": checksum},
            "index": {"state": "visible", "checksum": checksum},
            "observed_at": iso_now(),
        }
        row["state"] = "visible"
        row["disposition"] = "already-visible-exact-not-retransmitted"
        row["visibility_observations"] = [copy.deepcopy(visible)]
        row["visibility"] = visible

    def command_receipt(argv: list[str], kwargs: dict[str, Any], *, text: bytes = b"offline command PASS\n") -> dict[str, Any]:
        atomic_write_bytes(kwargs["log_path"], text)
        value = {
            "schema_version": 1,
            "argv": argv,
            "exit_code": 0,
            "timed_out": False,
            "credential_values_recorded": False,
            "log": artifact_ref(kwargs["log_path"], root=kwargs["artifact_root"]),
        }
        write_json(kwargs["receipt_path"], value)
        return value

    def resolution_logged(argv: list[str], **kwargs: Any) -> dict[str, Any]:
        atomic_write_bytes(Path(kwargs["cwd"]) / "Cargo.lock", b"# portable offline lock fixture\n")
        return command_receipt(argv, kwargs)

    def resolution_captured(argv: list[str], **kwargs: Any) -> tuple[dict[str, Any], str]:
        stdout = json.dumps(
            {
                "packages": [
                    {"name": name, "version": VERSION, "source": "registry+https://github.com/rust-lang/crates.io-index"}
                    for name in sorted(EXPECTED_CRATES)
                ]
            }
        )
        atomic_write_bytes(kwargs["stdout_path"], stdout.encode())
        return command_receipt(argv, {**kwargs, "log_path": kwargs["stderr_path"]}, text=b""), stdout

    resolution = clean_registry_resolution(
        out=out,
        receipt=receipt,
        timeout_seconds=1200,
        logged_runner=resolution_logged,
        captured_runner=resolution_captured,
    )

    def install_logged(argv: list[str], **kwargs: Any) -> dict[str, Any]:
        install_root = Path(argv[argv.index("--root") + 1])
        atomic_write_bytes(install_root / "bin" / "ferrum", b"#!/bin/sh\nexit 0\n")
        return command_receipt(argv, kwargs)

    def install_captured(argv: list[str], **kwargs: Any) -> tuple[dict[str, Any], str]:
        stdout = f"ferrum {VERSION}\n" if argv[-1] == "--version" else "Usage: ferrum [COMMAND]\n"
        atomic_write_bytes(kwargs["stdout_path"], stdout.encode())
        return command_receipt(argv, {**kwargs, "log_path": kwargs["stderr_path"]}, text=b""), stdout

    install = clean_install(
        out=out,
        receipt=receipt,
        timeout_seconds=1200,
        logged_runner=install_logged,
        captured_runner=install_captured,
    )

    bad_resolution = copy.deepcopy(resolution)
    bad_resolution["commands"]["generate_lockfile"] = fixture_command_receipt(
        out,
        name="bad-resolution-lock",
        argv=["cargo", "generate-lockfile", "--offline"],
    )
    expect_failure(
        "resolution exact generate-lockfile argv",
        lambda: validate_clean_resolution(bad_resolution, root=out),
        "argv differs",
    )
    bad_metadata = copy.deepcopy(resolution)
    bad_metadata["commands"]["metadata"] = fixture_command_receipt(
        out,
        name="bad-resolution-metadata",
        argv=["cargo", "metadata", "--format-version", "1"],
    )
    expect_failure(
        "resolution exact metadata argv",
        lambda: validate_clean_resolution(bad_metadata, root=out),
        "argv differs",
    )

    bad_install = copy.deepcopy(install)
    bad_install["commands"]["install"] = fixture_command_receipt(
        out,
        name="bad-install",
        argv=["cargo", "install", "ferrum-cli", "--version", VERSION, "--locked"],
    )
    expect_failure(
        "install exact root and target argv",
        lambda: validate_install(bad_install, root=out),
        "argv differs",
    )
    for stage, suffix in (("version", "--help"), ("help", "--version")):
        bad_probe = copy.deepcopy(install)
        binary = validate_artifact_ref(install["binary"], root=out, label="portable installed binary", nonempty=True)
        bad_probe["commands"][stage] = fixture_command_receipt(
            out,
            name=f"bad-install-{stage}",
            argv=[str(binary), suffix],
        )
        expect_failure(
            f"install exact {stage} argv",
            lambda value=bad_probe: validate_install(value, root=out),
            "argv differs",
        )

    target = str((out / "work" / "publish-target").resolve())
    for stage, argv in (
        ("dry-run", ["cargo", "publish", "--dry-run", "--locked", "-p", "wrong-crate", "--target-dir", target]),
        ("publish", ["cargo", "publish", "--locked", "-p", "wrong-crate", "--target-dir", target]),
    ):
        bad_receipt = copy.deepcopy(receipt)
        attempt = fixture_command_receipt(out, name=f"bad-{stage}", argv=argv)
        bad_receipt["packages"][0][f"{stage.replace('-', '_')}_attempts"] = [attempt]
        expect_failure(
            f"publish exact {stage} crate argv",
            lambda value=bad_receipt: validate_publish_receipt(
                value,
                prepublish=prepublish,
                prepublish_path=copied_manifest,
                artifact_root=out,
            ),
            "argv differs",
        )

    receipt["status"] = "pass"
    persist_receipt(out, receipt)
    final_rows = []
    for package, row in zip(prepublish["packages"], receipt["packages"]):
        checksum = package["archive"]["sha256"]
        final_rows.append(
            {
                "position": package["position"],
                "name": package["name"],
                "version": VERSION,
                "archive_sha256": checksum,
                "crates_io_visible": True,
                "disposition": row["disposition"],
                "api_checksum": checksum,
                "index_checksum": checksum,
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ferrum_v084_crates_io_publish_manifest",
        "status": "pass",
        "lane": "runtime-vnext-crates-io",
        "version": VERSION,
        "canonical": True,
        "release_candidate": copy.deepcopy(prepublish["release_candidate"]),
        "prepublish": copy.deepcopy(binding),
        "publish_order": copy.deepcopy(prepublish["topology"]["order"]),
        "packages": final_rows,
        "cargo_workspace_crates": [
            {"name": row["name"], "version": VERSION, "crates_io_visible": True}
            for row in final_rows
        ],
        "clean_resolution": resolution,
        "install": install,
        "resume_receipt": artifact_ref(out / "publish.resume.json", root=out),
        "created_at": iso_now(),
        "credential_policy": {
            "source": "existing-cargo-config-or-environment",
            "secret_values_recorded": False,
            "token_cli_arguments": False,
        },
        "artifact_dir": str(out.resolve()),
        "manifest_id": "",
        "pass_line": f"{PUBLISH_PASS_PREFIX}: {out.resolve()}",
    }
    manifest["manifest_id"] = manifest_identity(
        manifest,
        (
            "schema_version", "artifact_type", "version", "release_candidate",
            "prepublish", "publish_order", "packages", "clean_resolution",
            "install", "artifact_dir",
        ),
    )
    write_json(out / "crates-io.manifest.json", manifest)
    validate_publish_manifest(out / "crates-io.manifest.json")

    moved = root / "portable-publish-moved"
    shutil.move(str(out), moved)
    shutil.rmtree(source_root)
    validate_publish_manifest(moved / "crates-io.manifest.json")
    copied_archive = next((moved / binding["tree"]["path"]).glob("archives/*.crate"))
    atomic_write_bytes(copied_archive, copied_archive.read_bytes() + b"tamper")
    expect_failure(
        "portable copied evidence tamper",
        lambda: validate_publish_manifest(moved / "crates-io.manifest.json"),
        "portable prepublish evidence tree identity differs",
    )


def run_selftest() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-v084-crates-selftest-") as temporary:
        root = Path(temporary).resolve()
        prepublish_path = build_fixture_prepublish(root / "prepublish-source")
        prepublish, _ = validate_prepublish_manifest(prepublish_path)
        run_state_machine_selftest(root, prepublish, prepublish_path)
        run_final_stage_resume_selftest(root, prepublish, prepublish_path)
        run_portable_publish_manifest_selftest(root)

        bad_graph = copy.deepcopy(prepublish["topology"]["graph"])
        first, second = prepublish["topology"]["order"][:2]
        bad_graph[first] = [{"name": second, "kind": "normal", "target": "", "requirement": f"^{VERSION}"}]
        expect_failure("dependency cycle", lambda: stable_topological_order(bad_graph), "cycle")

        bad_packages = {"ferrum-types": {"dependencies": [{"name": "ferrum-types", "req": "*", "kind": None, "target": None}]}}
        expect_failure("internal dependency version", lambda: internal_dependency_graph(bad_packages), f"must require ^{VERSION}")

        archive_ref = prepublish["packages"][0]["archive"]
        archive_path = prepublish_path.parent / archive_ref["path"]
        original = archive_path.read_bytes()
        archive_path.write_bytes(original + b"tamper")
        expect_failure("immutable archive", lambda: validate_prepublish_manifest(prepublish_path), "size changed")
        archive_path.write_bytes(original)

        source_sha = prepublish["packages"][0]["source_manifest_sha256"]
        expect_failure("candidate source binding", lambda: inspect_crate_archive(archive_path, expected_name=prepublish["packages"][0]["name"], candidate_sha="d" * 40, source_manifest_sha256=source_sha), "VCS source binding differs")

        checksum = prepublish["packages"][0]["archive"]["sha256"]
        api = lambda url, label: ("visible", {"version": {"num": VERSION, "checksum": checksum}})
        index = lambda url, label: ("visible", [{"vers": VERSION, "cksum": checksum}])
        require(probe_crates_io("ferrum-types", checksum, api_reader=api, index_reader=index)["visible"], "offline visibility positive fixture failed")
        bad_api = lambda url, label: ("visible", {"version": {"num": VERSION, "checksum": "e" * 64}})
        expect_failure("published checksum mismatch", lambda: probe_crates_io("ferrum-types", checksum, api_reader=bad_api, index_reader=index), "API checksum differs")

        require("super-secret" not in sanitize_text("CARGO_REGISTRY_TOKEN=super-secret"), "secret sanitizer leaked a token")
        require(has_unredacted_secret_marker("CARGO_REGISTRY_TOKEN=super-secret"), "secret marker negative fixture was not detected")
        require(not has_unredacted_secret_marker("CARGO_REGISTRY_TOKEN=<redacted>"), "redacted secret marker was rejected")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run offline positive and negative fixtures")
    subparsers = parser.add_subparsers(dest="mode")
    prepublish = subparsers.add_parser("prepublish", help="freeze exact cargo package archives without publication")
    prepublish.add_argument("--repo", type=Path, default=REPO_ROOT)
    prepublish.add_argument("--out", type=Path, required=True)
    prepublish.add_argument("--release-candidate-sha", required=True)
    prepublish.add_argument("--release-candidate-tag", required=True)
    prepublish.add_argument("--command-timeout-seconds", type=int, default=7200)
    publish = subparsers.add_parser("publish", help="publish frozen archives serially and verify crates.io")
    publish.add_argument("--repo", type=Path, default=REPO_ROOT)
    publish.add_argument("--prepublish", type=Path, required=True)
    publish.add_argument("--out", type=Path, required=True)
    publish.add_argument("--resume", action="store_true")
    publish.add_argument("--visibility-timeout-seconds", type=int, default=900)
    publish.add_argument("--visibility-interval-seconds", type=int, default=15)
    publish.add_argument("--command-timeout-seconds", type=int, default=7200)
    args = parser.parse_args(argv)
    if args.self_test:
        require(args.mode is None, "--self-test cannot be combined with a release mode")
    else:
        require(args.mode is not None, "select prepublish, publish, or --self-test")
        require(args.command_timeout_seconds >= 1200, "command timeout must be at least 1200 seconds")
        if args.mode == "publish":
            require(args.visibility_timeout_seconds >= 60, "visibility timeout must be at least 60 seconds")
            require(1 <= args.visibility_interval_seconds <= 60, "visibility interval must be 1..60 seconds")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.self_test:
            run_selftest()
            print(SELFTEST_PASS_LINE)
        elif args.mode == "prepublish":
            create_prepublish(args)
        elif args.mode == "publish":
            publish_release(args)
        else:
            raise ReleaseError(f"unsupported mode: {args.mode}")
        return 0
    except (ReleaseError, legacy.ReleaseError, subprocess.SubprocessError, OSError) as exc:
        print(f"FERRUM CRATES IO V0.8.4 FAIL: {sanitize_text(str(exc))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
