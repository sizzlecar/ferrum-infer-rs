#!/usr/bin/env python3
"""Package, preflight, publish, and verify Ferrum v0.8.0 on crates.io.

The ``prepublish`` (alias ``package``) mode is deliberately network-independent
with respect to unpublished Ferrum crates.  It packages the clean release
candidate, publishes the exact archives into a git-backed ephemeral Cargo
registry, and builds/tests the extracted package contents against that registry.

The ``publish`` mode consumes the immutable prepublish archive references.  It
serializes ``cargo publish --dry-run --locked`` and ``cargo publish --locked``,
waits for crates.io API/index/clean-resolution visibility before advancing, and
finishes with a clean locked install plus ``ferrum --version``/``--help``.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import re
import selectors
import shutil
import signal
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


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
VERSION = "0.8.0"
RELEASE_CANDIDATE_TAG = "v0.8.0-rc.1"
FINAL_TAG = "v0.8.0"
REGISTRY_NAME = "ferrum-prepublish"
CRATES_IO_INDEX = "https://github.com/rust-lang/crates.io-index"
CRATES_IO_SPARSE = "https://index.crates.io"
CRATES_IO_API = "https://crates.io/api/v1"
USER_AGENT = "ferrum-runtime-vnext-crates-io-release/0.8.0"
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

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

RELEASE_GATE_LANES = {
    "g10a": ("vnext-g10a", "FERRUM RUNTIME VNEXT G10A RELEASE FREEZE PASS"),
    "g08_rc": ("vnext-g08-rc", "FERRUM RUNTIME VNEXT G08 RELEASE CANDIDATE CORRECTNESS PASS"),
    "g09_rc": ("vnext-g09-rc", "FERRUM RUNTIME VNEXT G09 RELEASE CANDIDATE PERFORMANCE PASS"),
    "published_assets": (
        "runtime-vnext-published-assets",
        "FERRUM RUNTIME VNEXT PUBLISHED ASSETS PASS",
    ),
}
MAX_SAFE_RETRANSMISSIONS = 0

PREPUBLISH_PASS_PREFIX = "FERRUM CRATES IO V0.8.0 PREPUBLISH PASS"
PASS_PREFIX = "FERRUM CRATES IO V0.8.0 PASS"
SELFTEST_PASS_LINE = "FERRUM CRATES IO V0.8.0 SELFTEST PASS"


class ReleaseError(RuntimeError):
    """The crates.io release contract was not satisfied."""


class ExternalReleaseError(ReleaseError):
    """A retryable authentication, index, or network boundary failed."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
            "ascii"
        ),
    )


def install_validated_json_manifests(
    *,
    root: Path,
    primary_name: str,
    alias_names: Iterable[str],
    value: dict[str, Any],
    validator: Callable[[Path], Any],
) -> Path:
    """Validate a non-canonical candidate before exposing any PASS manifest."""

    primary = root / primary_name
    aliases = [root / name for name in alias_names]
    require(not primary.exists(), f"canonical manifest already exists: {primary}")
    for alias in aliases:
        require(not alias.exists(), f"canonical manifest alias already exists: {alias}")
    candidate = root / f".{primary_name}.candidate-{os.getpid()}"
    require(not candidate.exists(), f"manifest candidate already exists: {candidate}")
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")
    try:
        atomic_write_bytes(candidate, payload)
        validator(candidate)
        os.replace(candidate, primary)
        for alias in aliases:
            atomic_write_bytes(alias, payload)
    except BaseException:
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
        raise
    return primary


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseError(f"cannot read {label} JSON {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


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


def artifact_ref(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    base = root.resolve()
    require(resolved.is_relative_to(base), f"artifact escapes root: {resolved}")
    require(
        resolved.is_file() and not resolved.is_symlink(),
        f"artifact is not a regular file: {resolved}",
    )
    return {
        "path": resolved.relative_to(base).as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def external_artifact_ref(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    require(
        resolved.is_file() and not resolved.is_symlink(),
        f"external artifact is not a regular file: {resolved}",
    )
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def validate_external_artifact_ref(value: Any, label: str) -> Path:
    row = exact_fields(value, {"path", "sha256", "size_bytes"}, label)
    path = Path(str(row["path"])).expanduser()
    require(path.is_absolute(), f"{label}.path must be absolute")
    path = path.resolve()
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    require(
        type(row["size_bytes"]) is int and row["size_bytes"] == path.stat().st_size,
        f"{label} size changed",
    )
    require(
        isinstance(row["sha256"], str)
        and SHA256_RE.fullmatch(row["sha256"])
        and sha256_file(path) == row["sha256"],
        f"{label} SHA256 changed",
    )
    return path


def validate_artifact_ref(
    value: Any, *, root: Path, label: str, nonempty: bool = False
) -> Path:
    row = exact_fields(value, {"path", "sha256", "size_bytes"}, label)
    relative = PurePosixPath(str(row["path"]))
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        f"{label}.path is unsafe",
    )
    path = (root / relative.as_posix()).resolve()
    require(path.is_relative_to(root.resolve()), f"{label}.path escapes artifact root")
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    size = row["size_bytes"]
    require(type(size) is int and size >= int(nonempty), f"{label}.size_bytes is invalid")
    require(path.stat().st_size == size, f"{label} size changed")
    digest = row["sha256"]
    require(isinstance(digest, str) and SHA256_RE.fullmatch(digest), f"{label} SHA256 is invalid")
    require(sha256_file(path) == digest, f"{label} SHA256 changed")
    return path


def sanitize_text(value: str) -> str:
    patterns = (
        re.compile(
            r"(?i)((?:CARGO_(?:REGISTRY_TOKEN|REGISTRIES_[A-Z0-9_]+_TOKEN|TOKEN)|CRATES_IO_TOKEN)\s*=\s*)\S+"
        ),
        re.compile(r"(?i)(authorization\s*:\s*bearer\s+)\S+"),
        re.compile(r"(?i)(\btoken\s*[=:]\s*)[\"']?[^\s\"']+"),
    )
    redacted = value
    for pattern in patterns:
        redacted = pattern.sub(r"\1<redacted>", redacted)
    return redacted


def safe_argv(argv: list[str]) -> None:
    lowered = [item.lower() for item in argv]
    require("--token" not in lowered, "token-bearing Cargo arguments are forbidden")
    for item in argv:
        require(
            not re.search(
                r"(?i)(?:cargo_(?:registry_token|registries_.+_token|token)|crates_io_token)=",
                item,
            ),
            "token-bearing environment assignments are forbidden in command arguments",
        )


def command_ref(path: Path, root: Path) -> dict[str, Any]:
    return artifact_ref(path, root=root)


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5)


def run_logged_command(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    receipt_path: Path,
    artifact_root: Path,
    expected_seconds: int,
    deadline_seconds: int,
) -> dict[str, Any]:
    """Run one bounded process and stream sanitized line output to an artifact."""
    safe_argv(argv)
    require(expected_seconds > 0, "expected command duration must be positive")
    require(
        deadline_seconds >= expected_seconds,
        "command deadline must be at least its expected duration",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        "START "
        f"command={json.dumps(argv)} expected_seconds={expected_seconds} "
        f"hard_deadline_seconds={deadline_seconds} progress_log={log_path}"
    )
    started_wall = time.monotonic()
    started_at = iso_now()
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    require(process.stdout is not None, "command stdout pipe was not created")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    pending = b""
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        while True:
            elapsed = time.monotonic() - started_wall
            if elapsed > deadline_seconds:
                timed_out = True
                terminate_process_group(process)
            events = selector.select(timeout=0.5)
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 65536)
                if chunk:
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        log.write(
                            sanitize_text(line.decode("utf-8", errors="replace")) + "\n"
                        )
                        log.flush()
                else:
                    selector.unregister(key.fileobj)
            if process.poll() is not None and not selector.get_map():
                break
            if timed_out and process.poll() is not None:
                # Drain the closed pipe on the next selector iteration.
                continue
        if pending:
            log.write(sanitize_text(pending.decode("utf-8", errors="replace")))
        log.flush()
        os.fsync(log.fileno())
    ended_at = iso_now()
    duration = time.monotonic() - started_wall
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "argv": argv,
        "cwd": str(cwd.resolve()),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": round(duration, 6),
        "expected_seconds": expected_seconds,
        "hard_deadline_seconds": deadline_seconds,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "worker_bounds": {"cargo_build_jobs": 2, "rust_test_threads": 8},
        "credential_values_recorded": False,
        "log": command_ref(log_path, artifact_root),
        "log_tail": sanitize_text(log_path.read_text(encoding="utf-8", errors="replace")[-4000:]),
    }
    write_json(receipt_path, receipt)
    print(
        f"END command={argv[0]} exit_code={process.returncode} "
        f"duration_seconds={duration:.3f} receipt={receipt_path}"
    )
    return receipt


def run_captured_command(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    receipt_path: Path,
    artifact_root: Path,
    expected_seconds: int,
    deadline_seconds: int,
) -> tuple[dict[str, Any], str]:
    """Run a bounded short command whose stdout is machine-readable."""
    safe_argv(argv)
    print(
        "START "
        f"command={json.dumps(argv)} expected_seconds={expected_seconds} "
        f"hard_deadline_seconds={deadline_seconds} progress_log={stderr_path}"
    )
    started = time.monotonic()
    started_at = iso_now()
    timed_out = False
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=deadline_seconds,
        )
        exit_code = completed.returncode
        stdout = sanitize_text(completed.stdout.decode("utf-8", errors="replace"))
        stderr = sanitize_text(completed.stderr.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired as error:
        timed_out = True
        exit_code = -signal.SIGKILL
        stdout = sanitize_text((error.stdout or b"").decode("utf-8", errors="replace"))
        stderr = sanitize_text((error.stderr or b"").decode("utf-8", errors="replace"))
    atomic_write_bytes(stdout_path, stdout.encode("utf-8"))
    atomic_write_bytes(stderr_path, stderr.encode("utf-8"))
    duration = time.monotonic() - started
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "argv": argv,
        "cwd": str(cwd.resolve()),
        "started_at": started_at,
        "ended_at": iso_now(),
        "duration_seconds": round(duration, 6),
        "expected_seconds": expected_seconds,
        "hard_deadline_seconds": deadline_seconds,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "worker_bounds": {"cargo_build_jobs": 2, "rust_test_threads": 8},
        "credential_values_recorded": False,
        "stdout": command_ref(stdout_path, artifact_root),
        "stderr": command_ref(stderr_path, artifact_root),
    }
    write_json(receipt_path, receipt)
    print(
        f"END command={argv[0]} exit_code={exit_code} "
        f"duration_seconds={duration:.3f} receipt={receipt_path}"
    )
    return receipt, stdout


def require_command_pass(receipt: dict[str, Any], label: str) -> None:
    require(
        receipt.get("exit_code") == 0 and receipt.get("timed_out") is False,
        f"{label} failed; see {receipt}",
    )


def base_cargo_environment(*, cargo_home: Path | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CARGO_BUILD_JOBS"] = "2"
    environment["RUST_TEST_THREADS"] = "8"
    environment["CARGO_TERM_COLOR"] = "never"
    environment.pop("CARGO_TARGET_DIR", None)
    if cargo_home is not None:
        cargo_home.mkdir(parents=True, exist_ok=True)
        environment["CARGO_HOME"] = str(cargo_home.resolve())
        for key in (
            "CARGO_REGISTRY_TOKEN",
            "CARGO_REGISTRIES_CRATES_IO_TOKEN",
            "CRATES_IO_TOKEN",
            "CARGO_TOKEN",
        ):
            environment.pop(key, None)
    return environment


def publish_cargo_environment() -> dict[str, str]:
    """Use only pre-existing Cargo configuration/environment credentials."""
    environment = base_cargo_environment()
    if not environment.get("CARGO_REGISTRY_TOKEN"):
        for alias in ("CRATES_IO_TOKEN", "CARGO_TOKEN"):
            if environment.get(alias):
                # Cargo recognizes CARGO_REGISTRY_TOKEN for crates.io.  The value
                # remains only in this child environment and is never serialized.
                environment["CARGO_REGISTRY_TOKEN"] = environment[alias]
                break
    for alias in ("CRATES_IO_TOKEN", "CARGO_TOKEN"):
        environment.pop(alias, None)
    return environment


def git_output(repo: Path, *argv: str) -> str:
    process = subprocess.run(
        ["git", *argv],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(
        process.returncode == 0,
        f"git {' '.join(argv)} failed: {sanitize_text(process.stderr.strip())}",
    )
    return process.stdout.strip()


def validate_origin_main_ancestry(repo: Path, release_candidate_sha: str) -> str:
    origin_main = git_output(repo, "rev-parse", "refs/remotes/origin/main^{commit}")
    process = subprocess.run(
        ["git", "merge-base", "--is-ancestor", release_candidate_sha, origin_main],
        cwd=repo,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(
        process.returncode == 0,
        "release candidate is not an ancestor of refs/remotes/origin/main: "
        f"{sanitize_text(process.stderr.decode('utf-8', errors='replace').strip())}",
    )
    return origin_main


def validate_annotated_tag(repo: Path, tag: str, expected_commit: str) -> dict[str, str]:
    require(tag in {RELEASE_CANDIDATE_TAG, FINAL_TAG}, f"unexpected release tag: {tag}")
    ref = f"refs/tags/{tag}"
    object_type = git_output(repo, "cat-file", "-t", ref)
    require(object_type == "tag", f"{tag} must be an annotated tag, found {object_type!r}")
    object_sha = git_output(repo, "rev-parse", ref)
    peeled = git_output(repo, "rev-parse", f"{ref}^{{commit}}")
    require(peeled == expected_commit, f"{tag} peeled commit differs from release candidate")
    return {"name": tag, "object_sha": object_sha, "peeled_commit_sha": peeled}


def clean_release_candidate(
    repo: Path,
    *,
    expected_sha: str,
    expected_tree: str | None,
    tag: str,
) -> dict[str, Any]:
    repo = repo.resolve()
    require((repo / "Cargo.toml").is_file(), f"repository root is invalid: {repo}")
    require(SHA1_RE.fullmatch(expected_sha) is not None, "release candidate SHA is invalid")
    observed_sha = git_output(repo, "rev-parse", "HEAD")
    observed_tree = git_output(repo, "rev-parse", "HEAD^{tree}")
    dirty_lines = [line for line in git_output(repo, "status", "--short").splitlines() if line]
    require(not dirty_lines, f"release candidate checkout is dirty: {dirty_lines[:12]}")
    require(observed_sha == expected_sha, "checkout HEAD differs from release candidate SHA")
    if expected_tree is not None:
        require(SHA1_RE.fullmatch(expected_tree) is not None, "release candidate tree SHA is invalid")
        require(observed_tree == expected_tree, "checkout tree differs from release candidate tree")
    return {
        "git_sha": observed_sha,
        "git_tree_sha": observed_tree,
        "dirty": False,
        "tag": validate_annotated_tag(repo, tag, observed_sha),
    }


def resolve_manifest_path(path: Path, names: tuple[str, ...]) -> Path:
    candidate = path.expanduser().resolve()
    if candidate.is_dir():
        for name in names:
            nested = candidate / name
            if nested.is_file():
                return nested
        raise ReleaseError(f"no supported manifest found under {candidate}: {names}")
    require(candidate.is_file(), f"manifest does not exist: {candidate}")
    return candidate


def verify_authoritative_goal_manifest(
    path: Path, *, expected_lane: str
) -> dict[str, Any]:
    """Delegate canonical Runtime vNext DAG validation to its source of truth."""

    try:
        import runtime_vnext_goal_gate as goal_gate
    except ImportError as error:
        raise ReleaseError(
            f"cannot load authoritative Runtime vNext goal validator: {error}"
        ) from error
    try:
        verified = goal_gate.verify_goal_manifest(
            path, expected_lane=expected_lane
        )
    except goal_gate.GoalGateError as error:
        raise ReleaseError(
            f"{expected_lane} authoritative validation failed: {sanitize_text(str(error))}"
        ) from error
    require(
        isinstance(verified, dict)
        and Path(verified.get("path", "")).resolve() == path.resolve(),
        f"{expected_lane} authoritative validator returned a different manifest",
    )
    return verified


def validate_goal_gate(path: Path, *, key: str) -> dict[str, Any]:
    require(key in RELEASE_GATE_LANES, f"unsupported release gate key: {key}")
    lane, prefix = RELEASE_GATE_LANES[key]
    manifest_path = resolve_manifest_path(path, ("manifest.json", "gate.manifest.json"))
    verified = verify_authoritative_goal_manifest(
        manifest_path, expected_lane=lane
    )
    value = verified["manifest"]
    require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("lane") == lane
        and value.get("status") == "pass"
        and value.get("canonical") is True
        and value.get("version") == VERSION,
        f"{key} release gate identity/status differs",
    )
    require(
        isinstance(value.get("pass_line"), str)
        and value["pass_line"].startswith(f"{prefix}: "),
        f"{key} release gate PASS line differs",
    )
    source = exact_fields(
        verified.get("source"),
        {"git_sha", "git_tree_sha", "dirty"},
        f"{key}.release_candidate",
    )
    require(
        isinstance(source["git_sha"], str)
        and SHA1_RE.fullmatch(source["git_sha"])
        and isinstance(source["git_tree_sha"], str)
        and SHA1_RE.fullmatch(source["git_tree_sha"])
        and source["dirty"] is False,
        f"{key} release candidate differs",
    )
    tag = value.get("release_candidate_tag")
    if key == "published_assets":
        release = value.get("release")
        require(isinstance(release, dict), "published-assets release is missing")
        require(
            release.get("tag_name") == FINAL_TAG
            and release.get("tag_sha") == source["git_sha"]
            and release.get("draft") is False
            and release.get("prerelease") is True,
            "published-assets prerelease identity differs",
        )
        tag = release.get("release_candidate_tag")
    require(tag in (None, RELEASE_CANDIDATE_TAG), f"{key} RC tag differs")
    if key == "g10a":
        require(tag == RELEASE_CANDIDATE_TAG, "G10A RC tag differs")
    return {
        "key": key,
        "lane": lane,
        "path": manifest_path,
        "ref": external_artifact_ref(manifest_path),
        "manifest": value,
        "source": copy.deepcopy(source),
        "release_candidate_tag": tag or RELEASE_CANDIDATE_TAG,
    }


def validate_release_gate_bundle(
    *, g10a: Path, g08_rc: Path, g09_rc: Path, published_assets: Path
) -> dict[str, Any]:
    gates = {
        "g10a": validate_goal_gate(g10a, key="g10a"),
        "g08_rc": validate_goal_gate(g08_rc, key="g08_rc"),
        "g09_rc": validate_goal_gate(g09_rc, key="g09_rc"),
        "published_assets": validate_goal_gate(
            published_assets, key="published_assets"
        ),
    }
    candidate = gates["g10a"]["source"]
    require(
        all(gate["source"] == candidate for gate in gates.values()),
        "G10A/G08-RC/G09-RC/published-assets release candidate differs",
    )
    require(
        all(
            gate["release_candidate_tag"] == RELEASE_CANDIDATE_TAG
            for gate in gates.values()
        ),
        "release gate RC tags differ",
    )
    expected_refs = {key: gate["ref"]["sha256"] for key, gate in gates.items()}
    link_expectations = {
        "g08_rc": {"g10a": expected_refs["g10a"]},
        "g09_rc": {
            "g10a": expected_refs["g10a"],
            "g08_rc": expected_refs["g08_rc"],
        },
        "published_assets": {
            "g10a": expected_refs["g10a"],
            "g08_rc": expected_refs["g08_rc"],
            "g09_rc": expected_refs["g09_rc"],
        },
    }
    for gate_key, expected in link_expectations.items():
        inputs = gates[gate_key]["manifest"].get("inputs")
        require(isinstance(inputs, dict), f"{gate_key} inputs are missing")
        for input_key, digest in expected.items():
            ref = inputs.get(input_key)
            require(
                isinstance(ref, dict) and ref.get("sha256") == digest,
                f"{gate_key} does not bind the exact {input_key} gate",
            )
    return {
        "release_candidate": candidate,
        "release_candidate_tag": RELEASE_CANDIDATE_TAG,
        "refs": {key: gate["ref"] for key, gate in gates.items()},
    }


def release_candidate_from_g10a(path: Path) -> tuple[str, str | None]:
    gate = validate_goal_gate(path, key="g10a")
    candidate = gate["source"]
    return candidate["git_sha"], candidate["git_tree_sha"]


def cargo_metadata(
    repo: Path,
    *,
    out: Path,
    artifact_root: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    receipt, stdout = run_captured_command(
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
        cwd=repo,
        env=base_cargo_environment(),
        stdout_path=out / "cargo-metadata.json",
        stderr_path=out / "cargo-metadata.stderr.log",
        receipt_path=out / "cargo-metadata.command.json",
        artifact_root=artifact_root,
        expected_seconds=30,
        deadline_seconds=min(timeout_seconds, 300),
    )
    require_command_pass(receipt, "cargo metadata")
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ReleaseError(f"cargo metadata did not emit JSON: {error}") from error
    require(isinstance(value, dict), "cargo metadata output must be an object")
    return value


def is_crates_io_publishable(package: dict[str, Any]) -> bool:
    publish = package.get("publish")
    return publish is None or (
        isinstance(publish, list) and "crates-io" in publish
    )


def package_map_from_metadata(
    metadata: dict[str, Any], *, expected_roster: frozenset[str] = EXPECTED_CRATES
) -> dict[str, dict[str, Any]]:
    packages = metadata.get("packages")
    members = metadata.get("workspace_members")
    require(isinstance(packages, list), "cargo metadata packages must be a list")
    require(isinstance(members, list), "cargo metadata workspace_members must be a list")
    by_id = {
        row.get("id"): row for row in packages if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    require(set(members) <= set(by_id), "cargo metadata workspace member is missing")
    selected: dict[str, dict[str, Any]] = {}
    for member in members:
        package = by_id[member]
        name = package.get("name")
        if isinstance(name, str) and name.startswith("ferrum-") and is_crates_io_publishable(package):
            require(name not in selected, f"duplicate publishable package name: {name}")
            selected[name] = package
    require(
        set(selected) == set(expected_roster),
        "publishable Ferrum crate roster differs: "
        f"missing={sorted(set(expected_roster) - set(selected))} "
        f"extra={sorted(set(selected) - set(expected_roster))}",
    )
    for name, package in selected.items():
        require(package.get("version") == VERSION, f"{name} version must be {VERSION}")
    return selected


def internal_dependency_graph(
    packages: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    names = set(packages)
    graph: dict[str, list[dict[str, str]]] = {}
    for name, package in packages.items():
        dependencies = package.get("dependencies")
        require(isinstance(dependencies, list), f"{name}.dependencies must be a list")
        rows: list[dict[str, str]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for dependency in dependencies:
            require(isinstance(dependency, dict), f"{name} dependency must be an object")
            dependency_name = dependency.get("name")
            if dependency_name not in names:
                continue
            requirement = dependency.get("req")
            kind = dependency.get("kind") or "normal"
            target = dependency.get("target") or ""
            require(
                requirement == f"^{VERSION}",
                f"{name} internal dependency {dependency_name} must require ^{VERSION}, found {requirement!r}",
            )
            require(kind in {"normal", "build", "dev"}, f"{name}->{dependency_name} kind is invalid")
            key = (str(dependency_name), str(kind), str(target), str(requirement))
            if key not in seen:
                rows.append(
                    {
                        "name": str(dependency_name),
                        "kind": str(kind),
                        "target": str(target),
                        "requirement": str(requirement),
                    }
                )
                seen.add(key)
        graph[name] = sorted(rows, key=lambda row: tuple(row.values()))
    return graph


def stable_topological_order(graph: dict[str, list[dict[str, str]]]) -> list[str]:
    dependencies = {
        name: {row["name"] for row in rows} for name, rows in graph.items()
    }
    for name, deps in dependencies.items():
        require(deps <= set(graph), f"{name} has dependency outside topology: {sorted(deps - set(graph))}")
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
    require(not remaining, f"publishable crate dependency graph contains a cycle: {remaining}")
    require(len(result) == len(graph), "topological order omitted a crate")
    return result


def index_prefix(name: str) -> str:
    lowered = name.lower()
    length = len(lowered)
    if length == 1:
        return "1"
    if length == 2:
        return "2"
    if length == 3:
        return f"3/{lowered[0]}"
    return f"{lowered[:2]}/{lowered[2:4]}"


def index_relative_path(name: str) -> Path:
    return Path(index_prefix(name)) / name.lower()


def metadata_dependency_to_index(
    dependency: dict[str, Any], *, internal_names: set[str]
) -> dict[str, Any]:
    original_name = dependency.get("name")
    rename = dependency.get("rename")
    require(isinstance(original_name, str), "metadata dependency name is invalid")
    require(rename is None or isinstance(rename, str), f"{original_name} rename is invalid")
    registry = dependency.get("registry")
    if original_name in internal_names:
        registry = None
    elif registry is None:
        registry = CRATES_IO_INDEX
    return {
        "name": rename or original_name,
        "req": dependency.get("req"),
        "features": dependency.get("features") or [],
        "optional": dependency.get("optional") is True,
        "default_features": dependency.get("uses_default_features") is not False,
        "target": dependency.get("target"),
        "kind": dependency.get("kind") or "normal",
        "registry": registry,
        "package": original_name if rename else None,
    }


def index_entry(
    package: dict[str, Any], *, checksum: str, internal_names: set[str]
) -> dict[str, Any]:
    dependencies = package.get("dependencies")
    require(isinstance(dependencies, list), "package dependencies are invalid")
    features = package.get("features") or {}
    require(isinstance(features, dict), "package features are invalid")
    entry: dict[str, Any] = {
        "name": package["name"],
        "vers": package["version"],
        "deps": [
            metadata_dependency_to_index(row, internal_names=internal_names)
            for row in dependencies
        ],
        "cksum": checksum,
        "features": features,
        "yanked": False,
        "links": package.get("links"),
        "v": 2,
    }
    if package.get("rust_version") is not None:
        entry["rust_version"] = package["rust_version"]
    return entry


def git_index_command(index: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=index,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(
        process.returncode == 0,
        f"local registry git {' '.join(arguments)} failed: {sanitize_text(process.stderr.strip())}",
    )
    return process.stdout.strip()


def initialize_registry(index: Path, package_root: Path) -> None:
    index.mkdir(parents=True, exist_ok=False)
    git_index_command(index, "init", "--quiet")
    git_index_command(index, "config", "user.name", "Ferrum Release Gate")
    git_index_command(index, "config", "user.email", "release-gate@invalid.example")
    config = {
        "dl": package_root.resolve().as_uri() + "/{crate}-{version}.crate",
    }
    write_json(index / "config.json", config)
    git_index_command(index, "add", "config.json")
    git_index_command(index, "commit", "--quiet", "-m", "initialize ephemeral registry")


def publish_to_local_registry(
    index: Path,
    package: dict[str, Any],
    archive: Path,
    *,
    internal_names: set[str],
) -> tuple[dict[str, Any], str]:
    checksum = sha256_file(archive)
    entry = index_entry(package, checksum=checksum, internal_names=internal_names)
    relative = index_relative_path(package["name"])
    entry_path = index / relative
    require(not entry_path.exists(), f"duplicate local registry entry: {relative}")
    atomic_write_bytes(entry_path, canonical_json(entry) + b"\n")
    git_index_command(index, "add", relative.as_posix())
    git_index_command(
        index,
        "commit",
        "--quiet",
        "-m",
        f"publish {package['name']} {package['version']}",
    )
    return entry, git_index_command(index, "rev-parse", "HEAD")


def dependency_tables(manifest: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for key in ("dependencies", "dev-dependencies", "build-dependencies"):
        value = manifest.get(key)
        if isinstance(value, dict):
            yield key, value
    target = manifest.get("target")
    if isinstance(target, dict):
        for target_name, target_value in target.items():
            if not isinstance(target_value, dict):
                continue
            for key in ("dependencies", "dev-dependencies", "build-dependencies"):
                value = target_value.get(key)
                if isinstance(value, dict):
                    yield f"target.{target_name}.{key}", value


def inspect_crate_archive(
    archive: Path,
    *,
    expected_name: str,
    expected_version: str,
    expected_git_sha: str | None,
) -> dict[str, Any]:
    expected_root = f"{expected_name}-{expected_version}"
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            members = bundle.getmembers()
            names = {member.name for member in members}
            for member in members:
                path = PurePosixPath(member.name)
                require(
                    not path.is_absolute() and ".." not in path.parts,
                    f"unsafe member in {archive.name}: {member.name}",
                )
                require(
                    path.parts and path.parts[0] == expected_root,
                    f"unexpected package root in {archive.name}: {member.name}",
                )
                require(
                    not member.issym()
                    and not member.islnk()
                    and not member.isdev()
                    and not member.isfifo(),
                    f"unsupported archive member type: {member.name}",
                )
            cargo_name = f"{expected_root}/Cargo.toml"
            require(cargo_name in names, f"{archive.name} lacks normalized Cargo.toml")
            cargo_stream = bundle.extractfile(cargo_name)
            require(cargo_stream is not None, f"cannot read {cargo_name}")
            manifest_bytes = cargo_stream.read()
            manifest = tomllib.loads(manifest_bytes.decode("utf-8"))
            package = manifest.get("package")
            require(isinstance(package, dict), f"{archive.name} package table is missing")
            require(package.get("name") == expected_name, f"{archive.name} package name differs")
            require(package.get("version") == expected_version, f"{archive.name} version differs")
            for table_name, table in dependency_tables(manifest):
                for dependency_name, specification in table.items():
                    if isinstance(specification, dict):
                        require(
                            "path" not in specification,
                            f"{archive.name} normalized {table_name}.{dependency_name} retains a path dependency",
                        )
            vcs_name = f"{expected_root}/.cargo_vcs_info.json"
            require(vcs_name in names, f"{archive.name} lacks .cargo_vcs_info.json")
            vcs_stream = bundle.extractfile(vcs_name)
            require(vcs_stream is not None, f"cannot read {vcs_name}")
            vcs = json.loads(vcs_stream.read().decode("utf-8"))
            require(isinstance(vcs, dict), f"{archive.name} VCS info is invalid")
            git = vcs.get("git")
            require(isinstance(git, dict), f"{archive.name} VCS git info is missing")
            if expected_git_sha is not None:
                require(git.get("sha1") == expected_git_sha, f"{archive.name} VCS SHA differs")
                require(git.get("dirty") in (None, False), f"{archive.name} VCS state is dirty")
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError, tarfile.TarError, tomllib.TOMLDecodeError) as error:
        raise ReleaseError(f"cannot inspect crate archive {archive}: {error}") from error
    return {
        "root": expected_root,
        "member_count": len(members),
        "normalized_manifest_sha256": sha256_bytes(manifest_bytes),
        "vcs_git_sha": git.get("sha1"),
        "path_dependency_count": 0,
    }


def extract_crate_archive(archive: Path, destination: Path) -> Path:
    require(not destination.exists(), f"extraction destination already exists: {destination}")
    destination.mkdir(parents=True)
    roots: set[str] = set()
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            for member in bundle.getmembers():
                relative = PurePosixPath(member.name)
                require(
                    not relative.is_absolute() and ".." not in relative.parts,
                    f"unsafe archive member: {member.name}",
                )
                require(relative.parts, f"empty archive member: {member.name}")
                roots.add(relative.parts[0])
                target = (destination / relative.as_posix()).resolve()
                require(target.is_relative_to(destination.resolve()), "archive extraction escaped destination")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stream = bundle.extractfile(member)
                    require(stream is not None, f"cannot read archive member: {member.name}")
                    with target.open("wb") as handle:
                        shutil.copyfileobj(stream, handle)
                    os.chmod(target, member.mode & 0o777)
                else:
                    raise ReleaseError(f"unsupported archive member type: {member.name}")
    except (OSError, EOFError, tarfile.TarError) as error:
        raise ReleaseError(f"cannot extract crate archive {archive}: {error}") from error
    require(len(roots) == 1, f"crate archive has multiple roots: {sorted(roots)}")
    root = destination / next(iter(roots))
    require((root / "Cargo.toml").is_file(), "extracted crate lacks Cargo.toml")
    return root


def local_registry_config(index: Path, internal_names: Iterable[str]) -> str:
    lines = [
        f"[registries.{REGISTRY_NAME}]",
        f'index = "{index.resolve().as_uri()}"',
        "",
        "[patch.crates-io]",
    ]
    for name in sorted(internal_names):
        lines.append(
            f'"{name}" = {{ version = "={VERSION}", registry = "{REGISTRY_NAME}" }}'
        )
    lines.append("")
    return "\n".join(lines)


def validate_registry_resolution(
    metadata: dict[str, Any],
    *,
    root_name: str,
    internal_names: set[str],
    expected_registry_url: str | None,
    original_repo: Path | None,
) -> list[dict[str, str]]:
    packages = metadata.get("packages")
    require(isinstance(packages, list), "resolved cargo metadata packages must be a list")
    rows: list[dict[str, str]] = []
    for package in packages:
        if not isinstance(package, dict) or package.get("name") not in internal_names:
            continue
        name = str(package["name"])
        version = package.get("version")
        source = package.get("source")
        require(version == VERSION, f"resolved {name} version differs")
        if name == root_name and source is None:
            source_kind = "extracted-package-root"
        else:
            require(
                isinstance(source, str)
                and source.startswith("registry+")
                and (expected_registry_url is None or expected_registry_url in source),
                f"resolved internal crate {name} did not come from ephemeral registry: {source!r}",
            )
            source_kind = "ephemeral-registry"
        manifest_path = Path(str(package.get("manifest_path", ""))).resolve()
        if original_repo is not None:
            require(
                not manifest_path.is_relative_to(original_repo.resolve()),
                f"resolved {name} through release workspace path: {manifest_path}",
            )
        rows.append({"name": name, "version": str(version), "source": source_kind})
    require(any(row["name"] == root_name for row in rows), f"resolved graph omits {root_name}")
    return sorted(rows, key=lambda row: row["name"])


def validate_extracted_package(
    *,
    archive: Path,
    package_name: str,
    internal_names: set[str],
    index: Path,
    validation_root: Path,
    cargo_home: Path,
    target_dir: Path,
    artifact_root: Path,
    original_repo: Path | None,
    timeout_seconds: int,
    offline: bool,
    cargo_config_text: str | None = None,
    expected_registry_url: str | None = None,
) -> dict[str, Any]:
    extracted_parent = validation_root / "extracted"
    package_root = extract_crate_archive(archive, extracted_parent)
    cargo_config = package_root / ".cargo" / "config.toml"
    atomic_write_bytes(
        cargo_config,
        (cargo_config_text or local_registry_config(index, internal_names)).encode("utf-8"),
    )
    lockfile = package_root / "Cargo.lock"
    if lockfile.exists():
        lockfile.unlink()
    env = base_cargo_environment(cargo_home=cargo_home)
    env["CARGO_TARGET_DIR"] = str(target_dir.resolve())
    common = ["--offline"] if offline else []
    commands = validation_root / "commands"
    lock_receipt = run_logged_command(
        ["cargo", "generate-lockfile", *common],
        cwd=package_root,
        env=env,
        log_path=commands / "generate-lockfile.log",
        receipt_path=commands / "generate-lockfile.command.json",
        artifact_root=artifact_root,
        expected_seconds=60,
        deadline_seconds=min(timeout_seconds, 900),
    )
    require_command_pass(lock_receipt, f"{package_name} clean lock resolution")
    metadata_receipt, metadata_stdout = run_captured_command(
        ["cargo", "metadata", "--locked", "--format-version", "1", *common],
        cwd=package_root,
        env=env,
        stdout_path=validation_root / "resolved-metadata.json",
        stderr_path=commands / "metadata.stderr.log",
        receipt_path=commands / "metadata.command.json",
        artifact_root=artifact_root,
        expected_seconds=30,
        deadline_seconds=min(timeout_seconds, 600),
    )
    require_command_pass(metadata_receipt, f"{package_name} clean cargo metadata")
    try:
        resolved = json.loads(metadata_stdout)
    except json.JSONDecodeError as error:
        raise ReleaseError(f"{package_name} resolved metadata is invalid: {error}") from error
    registry_url = expected_registry_url if cargo_config_text is not None else index.resolve().as_uri()
    resolved_internal = validate_registry_resolution(
        resolved,
        root_name=package_name,
        internal_names=internal_names,
        expected_registry_url=registry_url,
        original_repo=original_repo,
    )
    build_receipt = run_logged_command(
        ["cargo", "build", "--locked", *common],
        cwd=package_root,
        env=env,
        log_path=commands / "build.log",
        receipt_path=commands / "build.command.json",
        artifact_root=artifact_root,
        expected_seconds=900,
        deadline_seconds=timeout_seconds,
    )
    require_command_pass(build_receipt, f"{package_name} packaged build")
    test_receipt = run_logged_command(
        ["cargo", "test", "--locked", "--all-targets", *common],
        cwd=package_root,
        env=env,
        log_path=commands / "test.log",
        receipt_path=commands / "test.command.json",
        artifact_root=artifact_root,
        expected_seconds=900,
        deadline_seconds=timeout_seconds,
    )
    require_command_pass(test_receipt, f"{package_name} packaged test")
    return {
        "status": "pass",
        "package_root": package_root.relative_to(artifact_root).as_posix(),
        "cargo_lock": artifact_ref(lockfile, root=artifact_root),
        "cargo_config": artifact_ref(cargo_config, root=artifact_root),
        "resolved_metadata": artifact_ref(
            validation_root / "resolved-metadata.json", root=artifact_root
        ),
        "resolved_internal_crates": resolved_internal,
        "commands": {
            "generate_lockfile": artifact_ref(
                commands / "generate-lockfile.command.json", root=artifact_root
            ),
            "metadata": artifact_ref(commands / "metadata.command.json", root=artifact_root),
            "build": artifact_ref(commands / "build.command.json", root=artifact_root),
            "test": artifact_ref(commands / "test.command.json", root=artifact_root),
        },
        "workspace_path_dependency_count": 0,
    }


def package_archive(
    *,
    repo: Path,
    name: str,
    package_target: Path,
    log_root: Path,
    artifact_root: Path,
    timeout_seconds: int,
) -> tuple[Path, dict[str, Any]]:
    receipt = run_logged_command(
        [
            "cargo",
            "package",
            "--locked",
            "--no-verify",
            "-p",
            name,
            "--target-dir",
            str(package_target),
        ],
        cwd=repo,
        env=base_cargo_environment(),
        log_path=log_root / f"{name}.package.log",
        receipt_path=log_root / f"{name}.package.command.json",
        artifact_root=artifact_root,
        expected_seconds=120,
        deadline_seconds=min(timeout_seconds, 1200),
    )
    require_command_pass(receipt, f"cargo package {name}")
    archive = package_target / "package" / f"{name}-{VERSION}.crate"
    require(archive.is_file() and not archive.is_symlink(), f"cargo package omitted {archive}")
    return archive, receipt


def manifest_identity(value: dict[str, Any], fields: Iterable[str]) -> str:
    return sha256_bytes(canonical_json({field: value[field] for field in fields}))


PREPUBLISH_FIELDS = {
    "schema_version",
    "artifact_type",
    "status",
    "mode",
    "version",
    "canonical",
    "release_candidate",
    "g10a",
    "cargo",
    "topology",
    "registry",
    "packages",
    "created_at",
    "does_not_prove",
    "manifest_id",
    "pass_line",
}


def validate_release_candidate_object(value: Any, label: str) -> dict[str, Any]:
    row = exact_fields(value, {"git_sha", "git_tree_sha", "dirty", "tag"}, label)
    require(isinstance(row["git_sha"], str) and SHA1_RE.fullmatch(row["git_sha"]), f"{label}.git_sha invalid")
    require(isinstance(row["git_tree_sha"], str) and SHA1_RE.fullmatch(row["git_tree_sha"]), f"{label}.git_tree_sha invalid")
    require(row["dirty"] is False, f"{label}.dirty must be false")
    tag = exact_fields(row["tag"], {"name", "object_sha", "peeled_commit_sha"}, f"{label}.tag")
    require(tag["name"] == RELEASE_CANDIDATE_TAG, f"{label}.tag name differs")
    require(isinstance(tag["object_sha"], str) and SHA1_RE.fullmatch(tag["object_sha"]), f"{label}.tag object SHA invalid")
    require(tag["peeled_commit_sha"] == row["git_sha"], f"{label}.tag peeled commit differs")
    return row


def validate_prepublish_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = resolve_manifest_path(
        path, ("prepublish.manifest.json", "gate.manifest.json")
    )
    root = manifest_path.parent.resolve()
    value = exact_fields(read_json(manifest_path, "prepublish manifest"), PREPUBLISH_FIELDS, "prepublish manifest")
    require(value["schema_version"] == SCHEMA_VERSION, "prepublish schema version differs")
    require(value["artifact_type"] == "runtime_vnext_crates_io_prepublish_manifest", "prepublish artifact type differs")
    require(value["status"] == "pass" and value["mode"] == "prepublish", "prepublish status/mode differs")
    require(value["version"] == VERSION and value["canonical"] is True, "prepublish version/canonical differs")
    candidate = validate_release_candidate_object(value["release_candidate"], "prepublish.release_candidate")
    g10a_path = validate_external_artifact_ref(value["g10a"], "prepublish G10A")
    g10a = validate_goal_gate(g10a_path, key="g10a")
    require(g10a["source"] == {key: candidate[key] for key in ("git_sha", "git_tree_sha", "dirty")}, "prepublish G10A release candidate differs")
    cargo = exact_fields(value["cargo"], {"version", "metadata", "worker_bounds"}, "prepublish.cargo")
    require(isinstance(cargo["version"], str) and cargo["version"].startswith("cargo "), "prepublish Cargo version invalid")
    require(cargo["worker_bounds"] == {"cargo_build_jobs": 2, "rust_test_threads": 8}, "prepublish worker bounds differ")
    validate_artifact_ref(cargo["metadata"], root=root, label="prepublish cargo metadata", nonempty=True)
    topology = exact_fields(value["topology"], {"algorithm", "crate_count", "graph", "graph_sha256", "order"}, "prepublish.topology")
    require(topology["algorithm"] == "kahn-lexicographic-v1", "prepublish topology algorithm differs")
    require(topology["crate_count"] == len(EXPECTED_CRATES), "prepublish crate count differs")
    require(isinstance(topology["graph"], dict), "prepublish topology graph invalid")
    require(sha256_bytes(canonical_json(topology["graph"])) == topology["graph_sha256"], "prepublish topology graph SHA differs")
    order = topology["order"]
    require(isinstance(order, list) and set(order) == set(EXPECTED_CRATES) and len(order) == len(EXPECTED_CRATES), "prepublish topology order differs")
    require(stable_topological_order(topology["graph"]) == order, "prepublish topology is not canonical")
    registry = exact_fields(value["registry"], {"kind", "name", "index_path", "index_url", "head_commit", "config"}, "prepublish.registry")
    require(registry["kind"] == "git-backed-ephemeral-cargo-registry", "prepublish registry kind differs")
    require(registry["name"] == REGISTRY_NAME, "prepublish registry name differs")
    index_path = (root / str(registry["index_path"])).resolve()
    require(index_path.is_relative_to(root) and (index_path / ".git").is_dir(), "prepublish registry index is missing")
    require(registry["index_url"] == index_path.as_uri(), "prepublish registry URL differs")
    require(git_index_command(index_path, "status", "--porcelain") == "", "prepublish registry index is dirty")
    require(git_index_command(index_path, "rev-parse", "HEAD") == registry["head_commit"], "prepublish registry head changed")
    validate_artifact_ref(registry["config"], root=root, label="prepublish registry config", nonempty=True)
    packages = value["packages"]
    require(isinstance(packages, list) and len(packages) == len(EXPECTED_CRATES), "prepublish packages differ")
    observed_names: list[str] = []
    for position, package in enumerate(packages, 1):
        row = exact_fields(
            package,
            {
                "position",
                "name",
                "version",
                "manifest_path",
                "manifest_sha256",
                "internal_dependencies",
                "package",
                "archive_inspection",
                "index",
                "package_command",
                "validation",
            },
            f"prepublish.packages[{position - 1}]",
        )
        name = row["name"]
        require(row["position"] == position, f"prepublish package position {position} differs")
        require(name == order[position - 1], f"prepublish package order differs at {position}")
        require(name in EXPECTED_CRATES and row["version"] == VERSION, f"prepublish package identity differs at {position}")
        observed_names.append(name)
        archive = validate_artifact_ref(row["package"], root=root, label=f"{name} archive", nonempty=True)
        inspect_crate_archive(
            archive,
            expected_name=name,
            expected_version=VERSION,
            expected_git_sha=candidate["git_sha"],
        )
        validate_artifact_ref(row["index"]["entry"], root=root, label=f"{name} index entry", nonempty=True)
        require(isinstance(row["index"].get("commit"), str) and SHA1_RE.fullmatch(row["index"]["commit"]), f"{name} index commit invalid")
        validate_artifact_ref(row["package_command"], root=root, label=f"{name} package command", nonempty=True)
        validation = row["validation"]
        require(isinstance(validation, dict) and validation.get("status") == "pass", f"{name} package validation is not PASS")
        require(validation.get("workspace_path_dependency_count") == 0, f"{name} validation used workspace paths")
    require(observed_names == order, "prepublish package roster/order differs")
    identity = manifest_identity(
        value,
        ("schema_version", "artifact_type", "version", "release_candidate", "g10a", "topology", "registry", "packages"),
    )
    require(value["manifest_id"] == identity, "prepublish manifest identity differs")
    require(value["pass_line"] == f"{PREPUBLISH_PASS_PREFIX}: {root}", "prepublish PASS line differs")
    return value, manifest_path


def create_prepublish(args: argparse.Namespace) -> Path:
    repo = args.repo.expanduser().resolve()
    out = args.out.expanduser().resolve()
    require(not out.exists(), f"refusing to overwrite prepublish artifact: {out}")
    require(not out.is_relative_to(repo), "prepublish artifact must be outside the release checkout")
    g10a = validate_goal_gate(args.g10a, key="g10a")
    expected_sha = g10a["source"]["git_sha"]
    expected_tree = g10a["source"]["git_tree_sha"]
    if args.release_candidate_sha is not None:
        require(args.release_candidate_sha == expected_sha, "explicit release candidate SHA differs from G10A")
    if args.release_candidate_tree_sha is not None:
        require(args.release_candidate_tree_sha == expected_tree, "explicit release candidate tree differs from G10A")
    require(args.release_candidate_tag == RELEASE_CANDIDATE_TAG, f"prepublish tag must be {RELEASE_CANDIDATE_TAG}")
    candidate = clean_release_candidate(
        repo,
        expected_sha=expected_sha,
        expected_tree=expected_tree,
        tag=args.release_candidate_tag,
    )
    validate_origin_main_ancestry(repo, candidate["git_sha"])
    out.mkdir(parents=True)
    metadata_root = out / "metadata"
    metadata = cargo_metadata(
        repo,
        out=metadata_root,
        artifact_root=out,
        timeout_seconds=args.command_timeout_seconds,
    )
    packages = package_map_from_metadata(metadata)
    graph = internal_dependency_graph(packages)
    order = stable_topological_order(graph)
    package_root = out / "packages"
    package_root.mkdir()
    index = out / "local-registry" / "index"
    initialize_registry(index, package_root)
    package_target = out / "work" / "package-target"
    package_logs = out / "package-commands"
    packaged: dict[str, dict[str, Any]] = {}
    internal_names = set(packages)
    for position, name in enumerate(order, 1):
        source_archive, package_receipt = package_archive(
            repo=repo,
            name=name,
            package_target=package_target,
            log_root=package_logs,
            artifact_root=out,
            timeout_seconds=args.command_timeout_seconds,
        )
        destination = package_root / f"{name}-{VERSION}.crate"
        shutil.copyfile(source_archive, destination)
        inspection = inspect_crate_archive(
            destination,
            expected_name=name,
            expected_version=VERSION,
            expected_git_sha=candidate["git_sha"],
        )
        entry, index_commit = publish_to_local_registry(
            index,
            packages[name],
            destination,
            internal_names=internal_names,
        )
        entry_path = index / index_relative_path(name)
        manifest_path = Path(packages[name]["manifest_path"]).resolve()
        require(manifest_path.is_relative_to(repo), f"{name} manifest is outside release checkout")
        packaged[name] = {
            "position": position,
            "name": name,
            "version": VERSION,
            "manifest_path": manifest_path.relative_to(repo).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "internal_dependencies": copy.deepcopy(graph[name]),
            "package": artifact_ref(destination, root=out),
            "archive_inspection": inspection,
            "index": {
                "entry": artifact_ref(entry_path, root=out),
                "entry_sha256": sha256_bytes(canonical_json(entry)),
                "commit": index_commit,
            },
            "package_command": artifact_ref(
                package_logs / f"{name}.package.command.json", root=out
            ),
            "validation": None,
        }
    validation_cargo_home = out / "work" / "validation-cargo-home"
    validation_target = out / "work" / "validation-target"
    for name in order:
        packaged[name]["validation"] = validate_extracted_package(
            archive=package_root / f"{name}-{VERSION}.crate",
            package_name=name,
            internal_names=internal_names,
            index=index,
            validation_root=out / "validation" / name,
            cargo_home=validation_cargo_home,
            target_dir=validation_target,
            artifact_root=out,
            original_repo=repo,
            timeout_seconds=args.command_timeout_seconds,
            offline=False,
        )
    cargo_version = subprocess.run(
        ["cargo", "--version"], text=True, capture_output=True, check=True
    ).stdout.strip()
    topology = {
        "algorithm": "kahn-lexicographic-v1",
        "crate_count": len(order),
        "graph": graph,
        "graph_sha256": sha256_bytes(canonical_json(graph)),
        "order": order,
    }
    registry = {
        "kind": "git-backed-ephemeral-cargo-registry",
        "name": REGISTRY_NAME,
        "index_path": index.relative_to(out).as_posix(),
        "index_url": index.resolve().as_uri(),
        "head_commit": git_index_command(index, "rev-parse", "HEAD"),
        "config": artifact_ref(index / "config.json", root=out),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_crates_io_prepublish_manifest",
        "status": "pass",
        "mode": "prepublish",
        "version": VERSION,
        "canonical": True,
        "release_candidate": candidate,
        "g10a": g10a["ref"],
        "cargo": {
            "version": cargo_version,
            "metadata": artifact_ref(metadata_root / "cargo-metadata.json", root=out),
            "worker_bounds": {"cargo_build_jobs": 2, "rust_test_threads": 8},
        },
        "topology": topology,
        "registry": registry,
        "packages": [packaged[name] for name in order],
        "created_at": iso_now(),
        "does_not_prove": [
            "any crate is visible on crates.io",
            "cargo install from crates.io succeeds",
            "the GitHub release is promoted",
            "R3 or v0.8.0 release completion",
        ],
        "manifest_id": "",
        "pass_line": f"{PREPUBLISH_PASS_PREFIX}: {out}",
    }
    manifest["manifest_id"] = manifest_identity(
        manifest,
        ("schema_version", "artifact_type", "version", "release_candidate", "g10a", "topology", "registry", "packages"),
    )
    # Packaging and validation must not have changed tracked source state.
    clean_release_candidate(
        repo,
        expected_sha=candidate["git_sha"],
        expected_tree=candidate["git_tree_sha"],
        tag=RELEASE_CANDIDATE_TAG,
    )
    validate_origin_main_ancestry(repo, candidate["git_sha"])
    install_validated_json_manifests(
        root=out,
        primary_name="prepublish.manifest.json",
        alias_names=("gate.manifest.json",),
        value=manifest,
        validator=lambda path: validate_prepublish_manifest(path),
    )
    print(manifest["pass_line"])
    return out


def http_json(url: str, label: str) -> tuple[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Cache-Control": "no-cache"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return "missing", None
        category = "auth" if error.code in {401, 403} else "network"
        raise ExternalReleaseError(category, f"{label} returned HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ExternalReleaseError("network", f"{label} request failed: {sanitize_text(str(error))}") from error
    try:
        return "visible", json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExternalReleaseError("index", f"{label} returned invalid JSON") from error


def http_json_lines(url: str, label: str) -> tuple[str, list[dict[str, Any]] | None]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Cache-Control": "no-cache"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return "missing", None
        category = "auth" if error.code in {401, 403} else "network"
        raise ExternalReleaseError(category, f"{label} returned HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeDecodeError) as error:
        raise ExternalReleaseError("network", f"{label} request failed: {sanitize_text(str(error))}") from error
    rows: list[dict[str, Any]] = []
    try:
        for line in text.splitlines():
            if line.strip():
                value = json.loads(line)
                require(isinstance(value, dict), f"{label} row is not an object")
                rows.append(value)
    except json.JSONDecodeError as error:
        raise ExternalReleaseError("index", f"{label} returned invalid JSON lines") from error
    return "visible", rows


def crates_io_index_url(name: str) -> str:
    return f"{CRATES_IO_SPARSE}/{index_prefix(name)}/{name.lower()}"


def probe_crates_io(name: str, version: str, expected_checksum: str) -> dict[str, Any]:
    api_state, api = http_json(
        f"{CRATES_IO_API}/crates/{urllib.parse.quote(name)}/{urllib.parse.quote(version)}",
        f"crates.io API {name} {version}",
    )
    index_state, index_payload = http_json_lines(
        crates_io_index_url(name), f"crates.io sparse index {name}"
    )
    api_checksum: str | None = None
    if api_state == "visible":
        require(isinstance(api, dict), f"crates.io API payload for {name} is invalid")
        version_row = api.get("version")
        require(isinstance(version_row, dict), f"crates.io API version row for {name} is missing")
        require(version_row.get("num") == version, f"crates.io API version for {name} differs")
        api_checksum = version_row.get("checksum")
        require(api_checksum == expected_checksum, f"published {name} package SHA256 differs from prepublish")
    index_checksum: str | None = None
    if index_state == "visible":
        require(isinstance(index_payload, list), f"sparse index payload for {name} is invalid")
        rows = [row for row in index_payload if isinstance(row, dict) and row.get("vers") == version]
        if not rows:
            index_state = "missing"
        else:
            require(len(rows) == 1, f"sparse index has duplicate {name} {version} rows")
            index_checksum = rows[0].get("cksum")
            require(index_checksum == expected_checksum, f"crates.io index checksum for {name} differs")
    visible = api_state == "visible" and index_state == "visible"
    partial = api_state == "visible" or index_state == "visible"
    return {
        "visible": visible,
        "partial": partial,
        "api": {"state": api_state, "checksum": api_checksum},
        "index": {"state": index_state, "checksum": index_checksum},
        "observed_at": iso_now(),
    }


def poll_crates_io(
    name: str,
    version: str,
    checksum: str,
    *,
    timeout_seconds: int,
    interval_seconds: int,
) -> dict[str, Any]:
    require(timeout_seconds > 0, "visibility timeout must be positive")
    require(1 <= interval_seconds <= 60, "visibility poll interval must be between 1 and 60 seconds")
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] | None = None
    while True:
        last = probe_crates_io(name, version, checksum)
        if last["visible"]:
            return last
        if time.monotonic() >= deadline:
            raise ExternalReleaseError(
                "index",
                f"{name} {version} did not become visible in API and index before deadline; last={last}",
            )
        time.sleep(min(interval_seconds, max(0.1, deadline - time.monotonic())))


def clean_crates_io_resolution(
    *,
    name: str,
    work_root: Path,
    artifact_root: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    work_root.mkdir(parents=True, exist_ok=False)
    manifest = (
        "[package]\n"
        f'name = "verify-{name}"\n'
        'version = "0.0.0"\n'
        'edition = "2021"\n\n'
        "[dependencies]\n"
        f'"{name}" = "={VERSION}"\n'
    )
    atomic_write_bytes(work_root / "Cargo.toml", manifest.encode("utf-8"))
    source = work_root / "src" / "lib.rs"
    atomic_write_bytes(source, b"pub fn clean_resolution_probe() {}\n")
    env = base_cargo_environment(cargo_home=work_root / "cargo-home")
    env["CARGO_TARGET_DIR"] = str((work_root / "target").resolve())
    command_root = work_root / "commands"
    lock = run_logged_command(
        ["cargo", "generate-lockfile"],
        cwd=work_root,
        env=env,
        log_path=command_root / "generate-lockfile.log",
        receipt_path=command_root / "generate-lockfile.command.json",
        artifact_root=artifact_root,
        expected_seconds=60,
        deadline_seconds=min(timeout_seconds, 900),
    )
    require_command_pass(lock, f"clean crates.io resolution lock for {name}")
    receipt, stdout = run_captured_command(
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        cwd=work_root,
        env=env,
        stdout_path=work_root / "metadata.json",
        stderr_path=command_root / "metadata.stderr.log",
        receipt_path=command_root / "metadata.command.json",
        artifact_root=artifact_root,
        expected_seconds=30,
        deadline_seconds=min(timeout_seconds, 600),
    )
    require_command_pass(receipt, f"clean crates.io metadata for {name}")
    try:
        metadata = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ReleaseError(f"clean crates.io metadata for {name} is invalid: {error}") from error
    rows = [
        row
        for row in metadata.get("packages", [])
        if isinstance(row, dict) and row.get("name") == name and row.get("version") == VERSION
    ]
    require(len(rows) == 1, f"clean crates.io resolution did not select {name} {VERSION}")
    source_url = rows[0].get("source")
    require(
        isinstance(source_url, str)
        and source_url.startswith("registry+")
        and "crates.io" in source_url,
        f"clean resolution for {name} did not use crates.io: {source_url!r}",
    )
    return {
        "status": "pass",
        "source": source_url,
        "cargo_lock": artifact_ref(work_root / "Cargo.lock", root=artifact_root),
        "metadata": artifact_ref(work_root / "metadata.json", root=artifact_root),
        "commands": {
            "generate_lockfile": artifact_ref(command_root / "generate-lockfile.command.json", root=artifact_root),
            "metadata": artifact_ref(command_root / "metadata.command.json", root=artifact_root),
        },
    }


def classify_publish_failure(log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    if any(marker in text for marker in ("unauthorized", "forbidden", "authentication", "invalid token", "no token found", "403")):
        return "auth"
    if any(marker in text for marker in ("already exists", "already uploaded", "previously uploaded")):
        return "already-exists"
    if any(marker in text for marker in ("timed out", "timeout", "connection", "dns", "tls", "ssl", "network")):
        return "network"
    if "index" in text:
        return "index"
    return "command"


PUBLISH_RECEIPT_FIELDS = {
    "schema_version",
    "artifact_type",
    "status",
    "version",
    "prepublish",
    "release_candidate",
    "release_gates",
    "publish_order",
    "packages",
    "install",
    "started_at",
    "updated_at",
}
PUBLISH_RECEIPT_PACKAGE_FIELDS = {
    "position",
    "name",
    "version",
    "package_sha256",
    "state",
    "disposition",
    "dry_run_attempts",
    "publish_attempts",
    "visibility",
    "visibility_observations",
    "safe_retransmit_count",
    "clean_resolution",
    "last_error",
}


def new_publish_receipt(
    *,
    prepublish: dict[str, Any],
    prepublish_path: Path,
    prepublish_sha: str,
    release_gates: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_crates_io_publish_resume_receipt",
        "status": "in_progress",
        "version": VERSION,
        "prepublish": {
            "path": str(prepublish_path.resolve()),
            "sha256": prepublish_sha,
            "manifest_id": prepublish["manifest_id"],
        },
        "release_candidate": copy.deepcopy(prepublish["release_candidate"]),
        "release_gates": copy.deepcopy(release_gates["refs"]),
        "publish_order": copy.deepcopy(prepublish["topology"]["order"]),
        "packages": [
            {
                "position": row["position"],
                "name": row["name"],
                "version": VERSION,
                "package_sha256": row["package"]["sha256"],
                "state": "pending",
                "disposition": None,
                "dry_run_attempts": [],
                "publish_attempts": [],
                "visibility": None,
                "visibility_observations": [],
                "safe_retransmit_count": 0,
                "clean_resolution": None,
                "last_error": None,
            }
            for row in prepublish["packages"]
        ],
        "install": {"state": "pending", "attempts": [], "result": None},
        "started_at": iso_now(),
        "updated_at": iso_now(),
    }


def validate_publish_receipt_binding(
    receipt: dict[str, Any],
    *,
    prepublish: dict[str, Any],
    prepublish_path: Path,
    release_gates: dict[str, Any],
) -> None:
    exact_fields(receipt, PUBLISH_RECEIPT_FIELDS, "publish resume receipt")
    require(receipt.get("schema_version") == SCHEMA_VERSION, "resume receipt schema differs")
    require(receipt.get("artifact_type") == "runtime_vnext_crates_io_publish_resume_receipt", "resume receipt type differs")
    require(receipt.get("version") == VERSION, "resume receipt version differs")
    require(
        receipt.get("status") in {"in_progress", "pass"},
        "resume receipt status differs",
    )
    binding = exact_fields(
        receipt.get("prepublish"),
        {"path", "sha256", "manifest_id"},
        "resume receipt prepublish binding",
    )
    require(binding.get("path") == str(prepublish_path.resolve()), "resume receipt prepublish path differs")
    require(binding.get("sha256") == sha256_file(prepublish_path), "resume receipt prepublish SHA changed")
    require(binding.get("manifest_id") == prepublish["manifest_id"], "resume receipt manifest id differs")
    require(receipt.get("release_candidate") == prepublish["release_candidate"], "resume receipt RC differs")
    require(receipt.get("release_gates") == release_gates["refs"], "resume receipt release gates differ")
    require(receipt.get("publish_order") == prepublish["topology"]["order"], "resume receipt order differs")
    rows = receipt.get("packages")
    require(isinstance(rows, list) and len(rows) == len(prepublish["packages"]), "resume package rows differ")
    for expected, observed in zip(prepublish["packages"], rows):
        exact_fields(
            observed,
            PUBLISH_RECEIPT_PACKAGE_FIELDS,
            f"resume package {expected['name']}",
        )
        require(
            observed.get("position") == expected["position"]
            and observed.get("name") == expected["name"]
            and observed.get("version") == VERSION,
            "resume package identity differs",
        )
        require(observed.get("package_sha256") == expected["package"]["sha256"], "resume package SHA differs")
        require(
            isinstance(observed.get("visibility_observations"), list)
            and type(observed.get("safe_retransmit_count")) is int
            and observed["safe_retransmit_count"] == MAX_SAFE_RETRANSMISSIONS,
            "resume package ambiguity state differs",
        )
        require(
            observed.get("state")
            in {"pending", "publish-started", "awaiting-visibility", "visible"},
            "resume package state differs",
        )
    exact_fields(
        receipt.get("install"),
        {"state", "attempts", "result"},
        "resume install state",
    )
    require(
        receipt["install"].get("state") in {"pending", "pass"}
        and isinstance(receipt["install"].get("attempts"), list),
        "resume install state differs",
    )


def persist_publish_receipt(out: Path, receipt: dict[str, Any]) -> None:
    receipt["updated_at"] = iso_now()
    write_json(out / "publish.resume.json", receipt)


def settle_ambiguous_upload(
    name: str,
    version: str,
    checksum: str,
    *,
    poll_timeout: int,
    poll_interval: int,
) -> dict[str, Any]:
    """Poll a prior upload to visibility without ever authorizing retransmission."""

    require(poll_timeout > 0, "ambiguity poll timeout must be positive")
    require(
        1 <= poll_interval <= 60,
        "ambiguity poll interval must be between 1 and 60 seconds",
    )
    observations: list[dict[str, Any]] = []
    deadline = time.monotonic() + poll_timeout
    while True:
        observed = probe_crates_io(name, version, checksum)
        observations.append(observed)
        if observed["visible"]:
            return {
                "state": "visible",
                "visibility": observed,
                "observations": observations,
            }
        require(
            observed["partial"]
            or (
                observed.get("api", {}).get("state") == "missing"
                and observed.get("index", {}).get("state") == "missing"
            ),
            f"{name} ambiguity observation is neither partial nor a complete 404",
        )
        now = time.monotonic()
        if now >= deadline:
            return {
                "state": "ambiguous-unresolved",
                "visibility": None,
                "observations": observations,
            }
        time.sleep(min(poll_interval, max(0.1, deadline - now)))


def publish_one_crate(
    *,
    repo: Path,
    out: Path,
    package: dict[str, Any],
    row: dict[str, Any],
    receipt: dict[str, Any],
    poll_timeout: int,
    poll_interval: int,
    command_timeout: int,
) -> None:
    name = package["name"]
    checksum = package["package"]["sha256"]
    # Every entry/re-entry checks immutable local bytes before any remote action.
    prepublish_root = Path(receipt["prepublish"]["path"]).parent
    archive = validate_artifact_ref(package["package"], root=prepublish_root, label=f"{name} immutable package", nonempty=True)
    require(sha256_file(archive) == checksum, f"{name} immutable package SHA changed")
    try:
        observed = probe_crates_io(name, VERSION, checksum)
    except ExternalReleaseError as error:
        row["last_error"] = {"category": error.category, "message": str(error), "at": iso_now()}
        persist_publish_receipt(out, receipt)
        raise
    if observed["visible"]:
        row["state"] = "visible"
        row["disposition"] = row["disposition"] or "already-visible-not-retransmitted"
        row["visibility"] = observed
    elif observed["partial"]:
        row["state"] = "awaiting-visibility"
        persist_publish_receipt(out, receipt)
        row["visibility"] = poll_crates_io(
            name, VERSION, checksum, timeout_seconds=poll_timeout, interval_seconds=poll_interval
        )
        row["state"] = "visible"
        row["disposition"] = row["disposition"] or "published-prior-attempt"
    elif row["state"] in {"publish-started", "awaiting-visibility"}:
        settlement = settle_ambiguous_upload(
            name,
            VERSION,
            checksum,
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
        )
        row["visibility_observations"].extend(settlement["observations"])
        if settlement["state"] == "visible":
            row["visibility"] = settlement["visibility"]
            row["state"] = "visible"
            row["disposition"] = row["disposition"] or "published-prior-attempt"
        else:
            row["state"] = "awaiting-visibility"
            row["disposition"] = (
                row["disposition"] or "ambiguous-upload-not-retransmitted"
            )
            row["last_error"] = {
                "category": "ambiguous-upload-unresolved",
                "stage": "ambiguity-poll",
                "at": iso_now(),
            }
            persist_publish_receipt(out, receipt)
            raise ExternalReleaseError(
                "ambiguous",
                f"{name} upload remains unresolved after the visibility deadline; "
                "automatic retransmission is forbidden and crates.io must be checked manually",
            )
        persist_publish_receipt(out, receipt)
    if row["state"] == "visible":
        resolution_parent = out / "resolution" / name
        resolution_attempt = 1 + len(list(resolution_parent.glob("attempt-*")))
        row["clean_resolution"] = clean_crates_io_resolution(
            name=name,
            work_root=out / "resolution" / name / f"attempt-{resolution_attempt}",
            artifact_root=out,
            timeout_seconds=command_timeout,
        )
        row["last_error"] = None
        persist_publish_receipt(out, receipt)
        return
    attempt = len(row["dry_run_attempts"]) + 1
    command_root = out / "publish-commands" / name / f"attempt-{attempt}"
    target = out / "work" / "publish-target"
    environment = publish_cargo_environment()
    dry_run = run_logged_command(
        ["cargo", "publish", "--dry-run", "--locked", "-p", name, "--target-dir", str(target)],
        cwd=repo,
        env=environment,
        log_path=command_root / "dry-run.log",
        receipt_path=command_root / "dry-run.command.json",
        artifact_root=out,
        expected_seconds=300,
        deadline_seconds=min(command_timeout, 1800),
    )
    row["dry_run_attempts"].append(artifact_ref(command_root / "dry-run.command.json", root=out))
    if dry_run["exit_code"] != 0 or dry_run["timed_out"]:
        category = classify_publish_failure(command_root / "dry-run.log")
        row["last_error"] = {"category": category, "stage": "dry-run", "at": iso_now()}
        persist_publish_receipt(out, receipt)
        if category in {"auth", "network", "index"}:
            raise ExternalReleaseError(category, f"cargo publish dry-run failed for {name}")
        raise ReleaseError(f"cargo publish dry-run failed for {name}")
    generated = target / "package" / f"{name}-{VERSION}.crate"
    require(generated.is_file(), f"cargo publish dry-run did not generate {generated}")
    require(sha256_file(generated) == checksum, f"cargo publish dry-run package SHA changed for {name}")
    # Close the race with another release process before invoking upload.
    observed = probe_crates_io(name, VERSION, checksum)
    if observed["visible"] or observed["partial"]:
        row["state"] = "awaiting-visibility" if observed["partial"] else "visible"
        row["disposition"] = "already-visible-not-retransmitted"
        persist_publish_receipt(out, receipt)
        return publish_one_crate(
            repo=repo, out=out, package=package, row=row, receipt=receipt,
            poll_timeout=poll_timeout, poll_interval=poll_interval,
            command_timeout=command_timeout,
        )
    row["state"] = "publish-started"
    row["disposition"] = row["disposition"] or "published-by-this-producer"
    persist_publish_receipt(out, receipt)
    publish = run_logged_command(
        ["cargo", "publish", "--locked", "-p", name, "--target-dir", str(target)],
        cwd=repo,
        env=environment,
        log_path=command_root / "publish.log",
        receipt_path=command_root / "publish.command.json",
        artifact_root=out,
        expected_seconds=300,
        deadline_seconds=min(command_timeout, 1800),
    )
    row["publish_attempts"].append(artifact_ref(command_root / "publish.command.json", root=out))
    category = None if publish["exit_code"] == 0 and not publish["timed_out"] else classify_publish_failure(command_root / "publish.log")
    if category == "auth":
        # crates.io rejected authentication before accepting bytes; a later
        # authenticated resume still probes API/index before another upload.
        row["state"] = "pending"
        row["last_error"] = {"category": category, "stage": "publish", "at": iso_now()}
        persist_publish_receipt(out, receipt)
        raise ExternalReleaseError(category, f"cargo publish authentication failed for {name}")
    if category not in (None, "already-exists"):
        # Network errors after POST are ambiguous.  Never retransmit them:
        # resume only polls for the possibly accepted immutable package.
        row["state"] = "awaiting-visibility"
        row["last_error"] = {"category": category, "stage": "publish", "at": iso_now()}
        persist_publish_receipt(out, receipt)
        raise ExternalReleaseError(category or "network", f"cargo publish outcome is ambiguous for {name}; resume will poll without retransmission")
    require(sha256_file(generated) == checksum, f"cargo publish package SHA changed for {name}")
    row["state"] = "awaiting-visibility"
    persist_publish_receipt(out, receipt)
    row["visibility"] = poll_crates_io(
        name, VERSION, checksum, timeout_seconds=poll_timeout, interval_seconds=poll_interval
    )
    row["state"] = "visible"
    resolution_parent = out / "resolution" / name
    resolution_attempt = 1 + len(list(resolution_parent.glob("attempt-*")))
    row["clean_resolution"] = clean_crates_io_resolution(
        name=name,
        work_root=resolution_parent / f"attempt-{resolution_attempt}",
        artifact_root=out,
        timeout_seconds=command_timeout,
    )
    row["last_error"] = None
    persist_publish_receipt(out, receipt)


def clean_install(
    *, out: Path, receipt: dict[str, Any], timeout_seconds: int
) -> dict[str, Any]:
    install = receipt["install"]
    if install["state"] == "pass":
        return install["result"]
    attempt_number = len(install["attempts"]) + 1
    attempt = out / "install" / f"attempt-{attempt_number}"
    command_root = attempt / "commands"
    environment = base_cargo_environment(cargo_home=attempt / "cargo-home")
    target = attempt / "target"
    root = attempt / "root"
    command = [
        "cargo", "install", "ferrum-cli", "--version", VERSION, "--locked",
        "--root", str(root), "--target-dir", str(target),
    ]
    result = run_logged_command(
        command,
        cwd=attempt,
        env=environment,
        log_path=command_root / "cargo-install.log",
        receipt_path=command_root / "cargo-install.command.json",
        artifact_root=out,
        expected_seconds=1200,
        deadline_seconds=timeout_seconds,
    )
    install["attempts"].append(artifact_ref(command_root / "cargo-install.command.json", root=out))
    persist_publish_receipt(out, receipt)
    require_command_pass(result, "clean cargo install ferrum-cli 0.8.0")
    binary = root / "bin" / "ferrum"
    require(binary.is_file() and not binary.is_symlink(), "clean cargo install omitted ferrum binary")
    version_receipt, version_stdout = run_captured_command(
        [str(binary), "--version"], cwd=attempt, env=environment,
        stdout_path=command_root / "ferrum-version.stdout",
        stderr_path=command_root / "ferrum-version.stderr",
        receipt_path=command_root / "ferrum-version.command.json",
        artifact_root=out, expected_seconds=10, deadline_seconds=60,
    )
    require_command_pass(version_receipt, "installed ferrum --version")
    require(
        re.fullmatch(rf"ferrum(?:-cli)?\s+{re.escape(VERSION)}(?:\s+.*)?", version_stdout.strip())
        is not None,
        "installed ferrum --version does not report exact version 0.8.0",
    )
    help_receipt, help_stdout = run_captured_command(
        [str(binary), "--help"], cwd=attempt, env=environment,
        stdout_path=command_root / "ferrum-help.stdout",
        stderr_path=command_root / "ferrum-help.stderr",
        receipt_path=command_root / "ferrum-help.command.json",
        artifact_root=out, expected_seconds=10, deadline_seconds=60,
    )
    require_command_pass(help_receipt, "installed ferrum --help")
    require(help_stdout.strip(), "installed ferrum --help is empty")
    value = {
        "status": "pass",
        "command": command[:6],
        "binary": artifact_ref(binary, root=out),
        "binary_sha256": sha256_file(binary),
        "version_stdout": artifact_ref(command_root / "ferrum-version.stdout", root=out),
        "help_stdout": artifact_ref(command_root / "ferrum-help.stdout", root=out),
        "commands": {
            "install": artifact_ref(command_root / "cargo-install.command.json", root=out),
            "version": artifact_ref(command_root / "ferrum-version.command.json", root=out),
            "help": artifact_ref(command_root / "ferrum-help.command.json", root=out),
        },
    }
    install["state"] = "pass"
    install["result"] = value
    persist_publish_receipt(out, receipt)
    return value


def validate_command_receipt_evidence(
    value: Any, *, root: Path, label: str
) -> dict[str, Any]:
    receipt_path = validate_artifact_ref(value, root=root, label=label, nonempty=True)
    receipt = read_json(receipt_path, label)
    require(
        receipt.get("schema_version") == SCHEMA_VERSION
        and receipt.get("exit_code") == 0
        and receipt.get("timed_out") is False
        and receipt.get("credential_values_recorded") is False,
        f"{label} command did not PASS",
    )
    for key in ("log", "stdout", "stderr"):
        if key in receipt:
            validate_artifact_ref(
                receipt[key], root=root, label=f"{label}.{key}", nonempty=False
            )
    return receipt


def validate_clean_resolution_evidence(
    value: Any, *, root: Path, name: str
) -> None:
    row = exact_fields(
        value,
        {"status", "source", "cargo_lock", "metadata", "commands"},
        f"{name} clean resolution",
    )
    require(
        row["status"] == "pass"
        and isinstance(row["source"], str)
        and row["source"].startswith("registry+")
        and "crates.io" in row["source"],
        f"{name} clean resolution source/status differs",
    )
    validate_artifact_ref(
        row["cargo_lock"], root=root, label=f"{name} clean Cargo.lock", nonempty=True
    )
    metadata_path = validate_artifact_ref(
        row["metadata"], root=root, label=f"{name} clean metadata", nonempty=True
    )
    metadata = read_json(metadata_path, f"{name} clean metadata")
    matches = [
        package
        for package in metadata.get("packages", [])
        if isinstance(package, dict)
        and package.get("name") == name
        and package.get("version") == VERSION
    ]
    require(
        len(matches) == 1
        and isinstance(matches[0].get("source"), str)
        and "crates.io" in matches[0]["source"],
        f"{name} clean metadata does not resolve crates.io {VERSION}",
    )
    commands = exact_fields(
        row["commands"], {"generate_lockfile", "metadata"}, f"{name} clean commands"
    )
    validate_command_receipt_evidence(
        commands["generate_lockfile"],
        root=root,
        label=f"{name} clean generate-lockfile command",
    )
    validate_command_receipt_evidence(
        commands["metadata"], root=root, label=f"{name} clean metadata command"
    )


def validate_install_evidence(value: Any, *, root: Path) -> None:
    install = exact_fields(
        value,
        {
            "status",
            "command",
            "binary",
            "binary_sha256",
            "version_stdout",
            "help_stdout",
            "commands",
        },
        "clean install",
    )
    require(
        install["status"] == "pass"
        and install["command"]
        == ["cargo", "install", "ferrum-cli", "--version", VERSION, "--locked"],
        "clean install identity/status differs",
    )
    binary = validate_artifact_ref(
        install["binary"], root=root, label="installed ferrum binary", nonempty=True
    )
    require(
        install["binary_sha256"] == sha256_file(binary),
        "installed ferrum binary SHA256 differs",
    )
    version_path = validate_artifact_ref(
        install["version_stdout"],
        root=root,
        label="installed ferrum version stdout",
        nonempty=True,
    )
    version_text = version_path.read_text(encoding="utf-8", errors="strict").strip()
    require(
        re.fullmatch(rf"ferrum(?:-cli)?\s+{re.escape(VERSION)}(?:\s+.*)?", version_text)
        is not None,
        "installed ferrum version stdout differs",
    )
    help_path = validate_artifact_ref(
        install["help_stdout"],
        root=root,
        label="installed ferrum help stdout",
        nonempty=True,
    )
    help_text = help_path.read_text(encoding="utf-8", errors="strict")
    require(
        "Usage:" in help_text and "ferrum" in help_text.lower(),
        "installed ferrum help stdout differs",
    )
    commands = exact_fields(
        install["commands"], {"install", "version", "help"}, "clean install commands"
    )
    for key in ("install", "version", "help"):
        validate_command_receipt_evidence(
            commands[key], root=root, label=f"clean install {key} command"
        )


PUBLISH_MANIFEST_FIELDS = {
    "schema_version", "artifact_type", "status", "lane", "version", "canonical",
    "release_candidate", "release", "release_gates", "prepublish", "publish_order", "packages",
    "cargo_workspace_crates", "install", "resume_receipt", "created_at",
    "credential_policy", "manifest_id", "pass_line",
}


def validate_prepublish_publish_binding(
    *,
    candidate: dict[str, Any],
    gate_refs: dict[str, Any],
    prepublish: dict[str, Any],
) -> None:
    require(
        prepublish.get("release_candidate") == candidate,
        "publish prepublish release candidate differs",
    )
    require(
        prepublish.get("g10a") == gate_refs.get("g10a"),
        "publish prepublish G10A binding differs",
    )


def validate_publish_manifest(path: Path) -> dict[str, Any]:
    manifest_path = resolve_manifest_path(path, ("crates-io.manifest.json", "gate.manifest.json"))
    root = manifest_path.parent
    value = exact_fields(read_json(manifest_path, "crates.io publish manifest"), PUBLISH_MANIFEST_FIELDS, "crates.io publish manifest")
    require(value["schema_version"] == SCHEMA_VERSION, "publish schema version differs")
    require(value["artifact_type"] == "runtime_vnext_crates_io_release_manifest", "publish artifact type differs")
    require(value["status"] == "pass" and value["lane"] == "runtime-vnext-crates-io", "publish status/lane differs")
    require(value["version"] == VERSION and value["canonical"] is True, "publish version/canonical differs")
    candidate = validate_release_candidate_object(value["release_candidate"], "publish.release_candidate")
    release = exact_fields(value["release"], {"final_tag", "release_candidate_tag"}, "publish.release")
    final_tag = exact_fields(release["final_tag"], {"name", "object_sha", "peeled_commit_sha"}, "publish.release.final_tag")
    require(
        final_tag["name"] == FINAL_TAG
        and isinstance(final_tag["object_sha"], str)
        and SHA1_RE.fullmatch(final_tag["object_sha"]) is not None
        and final_tag["peeled_commit_sha"] == candidate["git_sha"],
        "final annotated tag binding differs",
    )
    require(release["release_candidate_tag"] == candidate["tag"], "RC tag binding differs")
    gate_refs = exact_fields(
        value["release_gates"], set(RELEASE_GATE_LANES), "publish.release_gates"
    )
    gate_paths = {
        key: validate_external_artifact_ref(ref, f"publish {key} gate")
        for key, ref in gate_refs.items()
    }
    gates = validate_release_gate_bundle(
        g10a=gate_paths["g10a"],
        g08_rc=gate_paths["g08_rc"],
        g09_rc=gate_paths["g09_rc"],
        published_assets=gate_paths["published_assets"],
    )
    require(gates["refs"] == gate_refs, "publish release gate refs differ")
    require(
        gates["release_candidate"]
        == {key: candidate[key] for key in ("git_sha", "git_tree_sha", "dirty")},
        "publish release gates differ from release candidate",
    )
    prepublish = exact_fields(value["prepublish"], {"path", "sha256", "manifest_id"}, "publish.prepublish")
    prepublish_path = Path(prepublish["path"])
    require(
        prepublish_path.is_absolute()
        and prepublish_path.is_file()
        and isinstance(prepublish["sha256"], str)
        and SHA256_RE.fullmatch(prepublish["sha256"]) is not None
        and sha256_file(prepublish_path) == prepublish["sha256"],
        "publish prepublish ref changed",
    )
    prepublish_value, _ = validate_prepublish_manifest(prepublish_path)
    require(prepublish_value["manifest_id"] == prepublish["manifest_id"], "publish prepublish id differs")
    validate_prepublish_publish_binding(
        candidate=candidate,
        gate_refs=gate_refs,
        prepublish=prepublish_value,
    )
    require(value["publish_order"] == prepublish_value["topology"]["order"], "publish order differs")
    packages = value["packages"]
    require(isinstance(packages, list) and len(packages) == len(EXPECTED_CRATES), "published package rows differ")
    for position, (expected, raw_row) in enumerate(
        zip(value["publish_order"], packages), 1
    ):
        row = exact_fields(
            raw_row,
            {
                "position",
                "name",
                "version",
                "package_sha256",
                "crates_io_visible",
                "disposition",
                "api_checksum",
                "index_checksum",
                "clean_resolution",
            },
            f"published package {expected}",
        )
        require(row["position"] == position, f"{expected} publish position differs")
        require(row.get("name") == expected and row.get("version") == VERSION, "published package identity differs")
        require(row.get("crates_io_visible") is True, f"{expected} is not marked visible")
        require(row.get("package_sha256") == next(item for item in prepublish_value["packages"] if item["name"] == expected)["package"]["sha256"], f"{expected} package SHA binding differs")
        require(
            row["api_checksum"]
            == row["index_checksum"]
            == row["package_sha256"]
            and isinstance(row["disposition"], str)
            and row["disposition"],
            f"{expected} visibility checksum/disposition differs",
        )
        validate_clean_resolution_evidence(
            row.get("clean_resolution"), root=root, name=expected
        )
    workspace = value["cargo_workspace_crates"]
    require(isinstance(workspace, list) and {row.get("name") for row in workspace} == set(EXPECTED_CRATES), "published workspace roster differs")
    require(all(row.get("version") == VERSION and row.get("crates_io_visible") is True for row in workspace), "published workspace visibility differs")
    validate_install_evidence(value["install"], root=root)
    resume_path = validate_artifact_ref(
        value["resume_receipt"], root=root, label="publish resume receipt", nonempty=True
    )
    resume = read_json(resume_path, "publish resume receipt")
    validate_publish_receipt_binding(
        resume,
        prepublish=prepublish_value,
        prepublish_path=prepublish_path,
        release_gates=gates,
    )
    require(
        resume.get("status") == "pass"
        and resume.get("release_gates") == gate_refs
        and isinstance(resume.get("packages"), list)
        and len(resume["packages"]) == len(EXPECTED_CRATES)
        and all(row.get("state") == "visible" for row in resume["packages"]),
        "publish resume receipt is not a complete PASS",
    )
    require(value["credential_policy"] == {"source": "existing-cargo-config-or-environment", "secret_values_recorded": False, "api_me_probe_used": False}, "credential policy differs")
    identity = manifest_identity(value, ("schema_version", "artifact_type", "version", "release_candidate", "release", "release_gates", "prepublish", "publish_order", "packages", "install"))
    require(value["manifest_id"] == identity, "publish manifest id differs")
    require(value["pass_line"] == f"{PASS_PREFIX}: {root.resolve()}", "publish PASS line differs")
    return value


def publish_release(args: argparse.Namespace) -> Path:
    prepublish, prepublish_path = validate_prepublish_manifest(args.prepublish)
    release_gates = validate_release_gate_bundle(
        g10a=args.g10a,
        g08_rc=args.g08_rc,
        g09_rc=args.g09_rc,
        published_assets=args.published_assets,
    )
    require(
        prepublish["g10a"]["sha256"] == release_gates["refs"]["g10a"]["sha256"],
        "publish G10A differs from prepublish G10A",
    )
    require(
        release_gates["release_candidate"]
        == {
            key: prepublish["release_candidate"][key]
            for key in ("git_sha", "git_tree_sha", "dirty")
        },
        "publish release gates differ from prepublish release candidate",
    )
    repo = args.repo.expanduser().resolve()
    out = args.out.expanduser().resolve()
    require(not out.is_relative_to(repo), "publish artifact must be outside release checkout")
    candidate = clean_release_candidate(
        repo,
        expected_sha=prepublish["release_candidate"]["git_sha"],
        expected_tree=prepublish["release_candidate"]["git_tree_sha"],
        tag=RELEASE_CANDIDATE_TAG,
    )
    require(candidate == prepublish["release_candidate"], "current RC/tag identity differs from prepublish")
    validate_origin_main_ancestry(repo, candidate["git_sha"])
    require(args.final_tag == FINAL_TAG, f"final tag must be {FINAL_TAG}")
    final_tag = validate_annotated_tag(repo, args.final_tag, candidate["git_sha"])
    receipt_path = out / "publish.resume.json"
    if out.exists():
        require(args.resume, f"publish artifact already exists; pass --resume: {out}")
        require(receipt_path.is_file(), f"publish resume receipt is missing: {receipt_path}")
        receipt = read_json(receipt_path, "publish resume receipt")
    else:
        require(not args.resume, "--resume requires an existing publish artifact")
        out.mkdir(parents=True)
        receipt = new_publish_receipt(
            prepublish=prepublish,
            prepublish_path=prepublish_path,
            prepublish_sha=sha256_file(prepublish_path),
            release_gates=release_gates,
        )
        persist_publish_receipt(out, receipt)
    validate_publish_receipt_binding(
        receipt,
        prepublish=prepublish,
        prepublish_path=prepublish_path,
        release_gates=release_gates,
    )
    completed_manifest = out / "crates-io.manifest.json"
    if receipt.get("status") == "pass" and completed_manifest.is_file():
        completed = validate_publish_manifest(completed_manifest)
        clean_release_candidate(
            repo,
            expected_sha=candidate["git_sha"],
            expected_tree=candidate["git_tree_sha"],
            tag=RELEASE_CANDIDATE_TAG,
        )
        validate_annotated_tag(repo, FINAL_TAG, candidate["git_sha"])
        validate_origin_main_ancestry(repo, candidate["git_sha"])
        print(completed["pass_line"])
        return out
    if receipt.get("status") == "pass":
        receipt["status"] = "in_progress"
        persist_publish_receipt(out, receipt)
    for package, row in zip(prepublish["packages"], receipt["packages"]):
        publish_one_crate(
            repo=repo, out=out, package=package, row=row, receipt=receipt,
            poll_timeout=args.poll_timeout_seconds,
            poll_interval=args.poll_interval_seconds,
            command_timeout=args.command_timeout_seconds,
        )
        require(row["state"] == "visible", f"{row['name']} did not reach visible state")
    install = clean_install(out=out, receipt=receipt, timeout_seconds=args.command_timeout_seconds)
    # Final live re-read.  A cached receipt is never enough for the PASS artifact.
    package_rows: list[dict[str, Any]] = []
    for package, row in zip(prepublish["packages"], receipt["packages"]):
        visibility = probe_crates_io(package["name"], VERSION, package["package"]["sha256"])
        require(visibility["visible"], f"final visibility re-read failed for {package['name']}")
        package_rows.append(
            {
                "position": package["position"],
                "name": package["name"],
                "version": VERSION,
                "package_sha256": package["package"]["sha256"],
                "crates_io_visible": True,
                "disposition": row["disposition"],
                "api_checksum": visibility["api"]["checksum"],
                "index_checksum": visibility["index"]["checksum"],
                "clean_resolution": row["clean_resolution"],
            }
        )
    clean_release_candidate(
        repo,
        expected_sha=candidate["git_sha"],
        expected_tree=candidate["git_tree_sha"],
        tag=RELEASE_CANDIDATE_TAG,
    )
    validate_annotated_tag(repo, FINAL_TAG, candidate["git_sha"])
    validate_origin_main_ancestry(repo, candidate["git_sha"])
    final_gates = validate_release_gate_bundle(
        g10a=args.g10a,
        g08_rc=args.g08_rc,
        g09_rc=args.g09_rc,
        published_assets=args.published_assets,
    )
    require(final_gates["refs"] == release_gates["refs"], "release gates changed during publish")
    receipt["status"] = "pass"
    persist_publish_receipt(out, receipt)
    prepublish_binding = {
        "path": str(prepublish_path.resolve()),
        "sha256": sha256_file(prepublish_path),
        "manifest_id": prepublish["manifest_id"],
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_crates_io_release_manifest",
        "status": "pass",
        "lane": "runtime-vnext-crates-io",
        "version": VERSION,
        "canonical": True,
        "release_candidate": candidate,
        "release": {"release_candidate_tag": candidate["tag"], "final_tag": final_tag},
        "release_gates": copy.deepcopy(release_gates["refs"]),
        "prepublish": prepublish_binding,
        "publish_order": prepublish["topology"]["order"],
        "packages": package_rows,
        "cargo_workspace_crates": [
            {"name": row["name"], "version": VERSION, "crates_io_visible": True}
            for row in package_rows
        ],
        "install": install,
        "resume_receipt": artifact_ref(receipt_path, root=out),
        "created_at": iso_now(),
        "credential_policy": {
            "source": "existing-cargo-config-or-environment",
            "secret_values_recorded": False,
            "api_me_probe_used": False,
        },
        "manifest_id": "",
        "pass_line": f"{PASS_PREFIX}: {out}",
    }
    manifest["manifest_id"] = manifest_identity(
        manifest,
        ("schema_version", "artifact_type", "version", "release_candidate", "release", "release_gates", "prepublish", "publish_order", "packages", "install"),
    )
    install_validated_json_manifests(
        root=out,
        primary_name="crates-io.manifest.json",
        alias_names=("gate.manifest.json",),
        value=manifest,
        validator=validate_publish_manifest,
    )
    print(manifest["pass_line"])
    return out


def fixture_package_metadata(
    name: str, dependencies: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "name": name,
        "version": VERSION,
        "dependencies": dependencies,
        "features": {},
        "links": None,
        "rust_version": None,
    }


def make_fixture_crate(
    path: Path,
    *,
    name: str,
    dependency: str | None,
    git_sha: str,
) -> None:
    root = f"{name}-{VERSION}"
    dependency_text = ""
    source_text = "pub fn value() -> u32 { 7 }\n"
    test_text = "#[test]\nfn packaged_test() { assert_eq!(value(), 7); }\n"
    if dependency is not None:
        dependency_text = f'\n[dependencies]\n"{dependency}" = "^{VERSION}"\n'
        source_text = f"pub fn value() -> u32 {{ {dependency.replace('-', '_')}::value() }}\n"
        test_text = "#[test]\nfn packaged_test() { assert_eq!(value(), 7); }\n"
    manifest = (
        "[package]\n"
        f'name = "{name}"\n'
        f'version = "{VERSION}"\n'
        'edition = "2021"\n'
        'license = "MIT"\n'
        f"{dependency_text}"
    ).encode("utf-8")
    vcs = canonical_json({"git": {"sha1": git_sha, "dirty": False}}) + b"\n"
    files = {
        f"{root}/Cargo.toml": manifest,
        f"{root}/.cargo_vcs_info.json": vcs,
        f"{root}/src/lib.rs": (source_text + test_text).encode("utf-8"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w:gz") as archive:
        for member_name, payload in sorted(files.items()):
            info = tarfile.TarInfo(member_name)
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, fileobj=__import__("io").BytesIO(payload))


def make_local_registry_source(
    destination: Path,
    *,
    index: Path,
    archives: Iterable[Path],
    names: Iterable[str],
) -> str:
    destination.mkdir(parents=True, exist_ok=False)
    local_index = destination / "index"
    for name in names:
        source = index / index_relative_path(name)
        target = local_index / index_relative_path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for archive in archives:
        shutil.copyfile(archive, destination / archive.name)
    return (
        "[source.crates-io]\n"
        'replace-with = "ferrum-selftest-local"\n\n'
        "[source.ferrum-selftest-local]\n"
        f'local-registry = "{destination.resolve()}"\n'
    )


def expect_failure(label: str, callback: Callable[[], Any], marker: str | None = None) -> None:
    try:
        callback()
    except ReleaseError as error:
        if marker is not None:
            require(marker in str(error), f"negative fixture {label} failed for wrong reason: {error}")
        return
    raise ReleaseError(f"negative fixture {label} unexpectedly passed")


def run_publish_state_machine_selftest(
    root: Path, *, archive: Path, package_name: str
) -> None:
    """Exercise the real publish state machine without any network endpoint."""
    prepublish_path = root / "fixture-prepublish.json"
    write_json(prepublish_path, {"fixture": True})
    package_ref = artifact_ref(archive, root=root)
    prepublish = {
        "manifest_id": "b" * 64,
        "release_candidate": {
            "git_sha": "a" * 40,
            "git_tree_sha": "c" * 40,
            "dirty": False,
            "tag": {
                "name": RELEASE_CANDIDATE_TAG,
                "object_sha": "d" * 40,
                "peeled_commit_sha": "a" * 40,
            },
        },
        "topology": {"order": [package_name]},
        "packages": [
            {"position": 1, "name": package_name, "package": package_ref}
        ],
    }
    gate_refs = {
        key: {
            "path": str(root / f"{key}.json"),
            "sha256": "1" * 64,
            "size_bytes": 1,
        }
        for key in RELEASE_GATE_LANES
    }
    state = {"remote_visible": False, "publish_calls": 0}

    def visibility(visible: bool) -> dict[str, Any]:
        return {
            "visible": visible,
            "partial": False,
            "api": {
                "state": "visible" if visible else "missing",
                "checksum": package_ref["sha256"] if visible else None,
            },
            "index": {
                "state": "visible" if visible else "missing",
                "checksum": package_ref["sha256"] if visible else None,
            },
            "observed_at": iso_now(),
        }

    def fake_probe(name: str, version: str, checksum: str) -> dict[str, Any]:
        require(
            name == package_name
            and version == VERSION
            and checksum == package_ref["sha256"],
            "state-machine probe identity differs",
        )
        return visibility(state["remote_visible"])

    def fake_poll(
        name: str,
        version: str,
        checksum: str,
        *,
        timeout_seconds: int,
        interval_seconds: int,
    ) -> dict[str, Any]:
        require(state["remote_visible"], "state-machine poll ran before upload")
        return fake_probe(name, version, checksum)

    def fake_command(
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
        receipt_path: Path,
        artifact_root: Path,
        expected_seconds: int,
        deadline_seconds: int,
    ) -> dict[str, Any]:
        del cwd, env, expected_seconds, deadline_seconds
        log_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(log_path, b"offline state-machine fixture\n")
        target = Path(argv[argv.index("--target-dir") + 1])
        generated = target / "package" / f"{package_name}-{VERSION}.crate"
        generated.parent.mkdir(parents=True, exist_ok=True)
        if "--dry-run" in argv:
            shutil.copyfile(archive, generated)
        else:
            require(generated.is_file(), "state-machine publish lacks dry-run archive")
            state["publish_calls"] += 1
            state["remote_visible"] = True
        command_receipt = {
            "schema_version": SCHEMA_VERSION,
            "argv": argv,
            "exit_code": 0,
            "timed_out": False,
            "credential_values_recorded": False,
            "log": artifact_ref(log_path, root=artifact_root),
        }
        write_json(receipt_path, command_receipt)
        return command_receipt

    def fake_resolution(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"status": "pass", "fixture": True}

    originals = {
        "probe_crates_io": globals()["probe_crates_io"],
        "poll_crates_io": globals()["poll_crates_io"],
        "run_logged_command": globals()["run_logged_command"],
        "clean_crates_io_resolution": globals()["clean_crates_io_resolution"],
        "sleep": time.sleep,
        "monotonic": time.monotonic,
    }
    clock = {"now": 0.0}

    def fake_sleep(seconds: float) -> None:
        clock["now"] += seconds

    try:
        globals()["probe_crates_io"] = fake_probe
        globals()["poll_crates_io"] = fake_poll
        globals()["run_logged_command"] = fake_command
        globals()["clean_crates_io_resolution"] = fake_resolution
        time.sleep = fake_sleep
        time.monotonic = lambda: clock["now"]

        ambiguous_out = root / "publish-state-machine-ambiguous"
        ambiguous_out.mkdir()
        ambiguous = new_publish_receipt(
            prepublish=prepublish,
            prepublish_path=prepublish_path,
            prepublish_sha=sha256_file(prepublish_path),
            release_gates={"refs": gate_refs},
        )
        ambiguous_row = ambiguous["packages"][0]
        ambiguous_row["state"] = "publish-started"
        expect_failure(
            "ambiguous-upload-never-retransmitted",
            lambda: publish_one_crate(
                repo=root,
                out=ambiguous_out,
                package=prepublish["packages"][0],
                row=ambiguous_row,
                receipt=ambiguous,
                poll_timeout=3,
                poll_interval=1,
                command_timeout=1200,
            ),
            "automatic retransmission is forbidden",
        )
        require(
            ambiguous_row["state"] == "awaiting-visibility"
            and ambiguous_row["safe_retransmit_count"] == 0
            and len(ambiguous_row["visibility_observations"]) >= 3
            and state["publish_calls"] == 0,
            "ambiguous crash/resume attempted an upload retransmission",
        )

        out = root / "publish-state-machine-normal"
        out.mkdir()
        receipt = new_publish_receipt(
            prepublish=prepublish,
            prepublish_path=prepublish_path,
            prepublish_sha=sha256_file(prepublish_path),
            release_gates={"refs": gate_refs},
        )
        row = receipt["packages"][0]
        publish_one_crate(
            repo=root,
            out=out,
            package=prepublish["packages"][0],
            row=row,
            receipt=receipt,
            poll_timeout=3,
            poll_interval=1,
            command_timeout=1200,
        )
        require(
            row["state"] == "visible"
            and row["safe_retransmit_count"] == 0
            and state["publish_calls"] == 1,
            "normal publish did not converge exactly once",
        )
        publish_one_crate(
            repo=root,
            out=out,
            package=prepublish["packages"][0],
            row=row,
            receipt=receipt,
            poll_timeout=3,
            poll_interval=1,
            command_timeout=1200,
        )
        require(state["publish_calls"] == 1, "visible resume retransmitted a crate")
    finally:
        globals()["probe_crates_io"] = originals["probe_crates_io"]
        globals()["poll_crates_io"] = originals["poll_crates_io"]
        globals()["run_logged_command"] = originals["run_logged_command"]
        globals()["clean_crates_io_resolution"] = originals[
            "clean_crates_io_resolution"
        ]
        time.sleep = originals["sleep"]
        time.monotonic = originals["monotonic"]


def run_selftest() -> None:
    # This fixture is intentionally dependency-closed and every Cargo command
    # uses --offline.  It cannot read or write any real registry endpoint.
    with tempfile.TemporaryDirectory(prefix="ferrum-crates-io-selftest-") as temporary:
        root = Path(temporary)
        packages_root = root / "packages"
        packages_root.mkdir()
        index = root / "index"
        initialize_registry(index, packages_root)
        git_sha = "a" * 40
        leaf = "ferrum-fixture-leaf"
        app = "ferrum-fixture-app"
        leaf_archive = packages_root / f"{leaf}-{VERSION}.crate"
        app_archive = packages_root / f"{app}-{VERSION}.crate"
        make_fixture_crate(leaf_archive, name=leaf, dependency=None, git_sha=git_sha)
        make_fixture_crate(app_archive, name=app, dependency=leaf, git_sha=git_sha)
        dep = {
            "name": leaf,
            "req": f"^{VERSION}",
            "features": [],
            "optional": False,
            "uses_default_features": True,
            "target": None,
            "kind": None,
            "registry": None,
            "rename": None,
        }
        leaf_meta = fixture_package_metadata(leaf, [])
        app_meta = fixture_package_metadata(app, [dep])
        publish_to_local_registry(index, leaf_meta, leaf_archive, internal_names={leaf, app})
        publish_to_local_registry(index, app_meta, app_archive, internal_names={leaf, app})
        local_source_config = make_local_registry_source(
            root / "local-registry-source",
            index=index,
            archives=(leaf_archive, app_archive),
            names=(leaf, app),
        )
        inspect_crate_archive(
            app_archive,
            expected_name=app,
            expected_version=VERSION,
            expected_git_sha=git_sha,
        )
        validation = validate_extracted_package(
            archive=app_archive,
            package_name=app,
            internal_names={leaf, app},
            index=index,
            validation_root=root / "validation" / app,
            cargo_home=root / "cargo-home",
            target_dir=root / "target",
            artifact_root=root,
            original_repo=None,
            timeout_seconds=900,
            offline=True,
            cargo_config_text=local_source_config,
            expected_registry_url=None,
        )
        require(validation["status"] == "pass", "offline registry fixture did not pass")
        require(
            {row["name"] for row in validation["resolved_internal_crates"]} == {leaf, app},
            "offline registry fixture did not resolve both packaged crates",
        )

        graph = {
            leaf: [],
            app: [{"name": leaf, "kind": "normal", "target": "", "requirement": f"^{VERSION}"}],
        }
        require(stable_topological_order(graph) == [leaf, app], "stable topology fixture differs")
        cycle = copy.deepcopy(graph)
        cycle[leaf] = [{"name": app, "kind": "normal", "target": "", "requirement": f"^{VERSION}"}]
        expect_failure("dependency-cycle", lambda: stable_topological_order(cycle), "cycle")

        bad_metadata = {
            "packages": [
                {"id": "one", "name": "ferrum-one", "version": VERSION, "publish": None, "dependencies": []}
            ],
            "workspace_members": ["one"],
        }
        expect_failure(
            "incomplete-publishable-roster",
            lambda: package_map_from_metadata(bad_metadata),
            "roster differs",
        )
        wrong_req_packages = {
            leaf: fixture_package_metadata(leaf, []),
            app: fixture_package_metadata(app, [{**dep, "req": "*"}]),
        }
        expect_failure(
            "unversioned-internal-dependency",
            lambda: internal_dependency_graph(wrong_req_packages),
            f"must require ^{VERSION}",
        )

        ref = artifact_ref(app_archive, root=root)
        original = app_archive.read_bytes()
        with app_archive.open("ab") as handle:
            handle.write(b"tamper")
        expect_failure(
            "immutable-package-sha",
            lambda: validate_artifact_ref(ref, root=root, label="fixture package"),
            "changed",
        )
        atomic_write_bytes(app_archive, original)

        fake_metadata = {
            "packages": [
                {
                    "name": app,
                    "version": VERSION,
                    "source": None,
                    "manifest_path": str(root / "workspace" / "Cargo.toml"),
                },
                {
                    "name": leaf,
                    "version": VERSION,
                    "source": None,
                    "manifest_path": str(root / "workspace" / "leaf" / "Cargo.toml"),
                },
            ]
        }
        expect_failure(
            "workspace-path-dependency",
            lambda: validate_registry_resolution(
                fake_metadata,
                root_name=app,
                internal_names={leaf, app},
                expected_registry_url=index.as_uri(),
                original_repo=root / "workspace",
            ),
            "release workspace path",
        )
        secret = "ferrum-hostile-token-value"
        for token_name in (
            "CARGO_REGISTRY_TOKEN",
            "CARGO_REGISTRIES_CRATES_IO_TOKEN",
            "CRATES_IO_TOKEN",
            "CARGO_TOKEN",
        ):
            require(
                secret not in sanitize_text(f"{token_name}={secret}"),
                f"credential sanitizer leaked {token_name}",
            )
            expect_failure(
                f"argv-{token_name.lower()}",
                lambda token_name=token_name: safe_argv(
                    ["cargo", "publish", f"{token_name}={secret}"]
                ),
                "token-bearing",
            )
        require(
            secret
            not in sanitize_text(f"Authorization: Bearer {secret}"),
            "credential sanitizer leaked a bearer token",
        )
        credential_keys = (
            "CARGO_REGISTRY_TOKEN",
            "CARGO_REGISTRIES_CRATES_IO_TOKEN",
            "CRATES_IO_TOKEN",
            "CARGO_TOKEN",
        )
        saved_credentials = {key: os.environ.get(key) for key in credential_keys}
        try:
            for key in credential_keys:
                os.environ.pop(key, None)
            os.environ["CRATES_IO_TOKEN"] = secret
            environment = publish_cargo_environment()
            require(
                environment.get("CARGO_REGISTRY_TOKEN") == secret
                and "CRATES_IO_TOKEN" not in environment
                and "CARGO_TOKEN" not in environment,
                "publish credential aliases were not normalized and removed",
            )
        finally:
            for key, old_value in saved_credentials.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

        shallow_gate = root / "shallow-g10a.json"
        write_json(
            shallow_gate,
            {
                "schema_version": SCHEMA_VERSION,
                "lane": "vnext-g10a",
                "status": "pass",
                "canonical": True,
                "version": VERSION,
                "pass_line": f"{RELEASE_GATE_LANES['g10a'][1]}: {root}",
                "release_candidate": {
                    "git_sha": git_sha,
                    "git_tree_sha": "c" * 40,
                    "dirty": False,
                },
                "release_candidate_tag": RELEASE_CANDIDATE_TAG,
            },
        )
        expect_failure(
            "shallow-goal-gate",
            lambda: validate_goal_gate(shallow_gate, key="g10a"),
            "authoritative validation failed",
        )

        rejected_root = root / "rejected-pass-install"
        rejected_root.mkdir()
        expect_failure(
            "canonical-pass-before-validation",
            lambda: install_validated_json_manifests(
                root=rejected_root,
                primary_name="manifest.json",
                alias_names=("gate.manifest.json",),
                value={"status": "pass"},
                validator=lambda _path: require(False, "hostile candidate rejected"),
            ),
            "hostile candidate rejected",
        )
        require(
            not (rejected_root / "manifest.json").exists()
            and not (rejected_root / "gate.manifest.json").exists(),
            "rejected candidate exposed a canonical PASS manifest",
        )

        accepted_root = root / "accepted-pass-install"
        accepted_root.mkdir()
        install_validated_json_manifests(
            root=accepted_root,
            primary_name="manifest.json",
            alias_names=("gate.manifest.json",),
            value={"status": "pass"},
            validator=lambda path: require(
                read_json(path, "candidate").get("status") == "pass",
                "candidate status differs",
            ),
        )
        require(
            (accepted_root / "manifest.json").read_bytes()
            == (accepted_root / "gate.manifest.json").read_bytes(),
            "validated canonical manifest aliases differ",
        )

        release_candidate = {
            "git_sha": git_sha,
            "git_tree_sha": "c" * 40,
            "dirty": False,
            "tag": {
                "name": RELEASE_CANDIDATE_TAG,
                "object_sha": "d" * 40,
                "peeled_commit_sha": git_sha,
            },
        }
        g10a_ref = {
            "path": str(root / "g10a.json"),
            "sha256": "1" * 64,
            "size_bytes": 1,
        }
        prepublish_fixture = {
            "manifest_id": "b" * 64,
            "release_candidate": release_candidate,
            "g10a": g10a_ref,
            "topology": {"order": [app]},
            "packages": [
                {
                    "position": 1,
                    "name": app,
                    "package": {"sha256": "e" * 64},
                }
            ],
        }
        validate_prepublish_publish_binding(
            candidate=release_candidate,
            gate_refs={"g10a": g10a_ref},
            prepublish=prepublish_fixture,
        )
        wrong_candidate = copy.deepcopy(release_candidate)
        wrong_candidate["git_sha"] = "9" * 40
        wrong_candidate["tag"]["peeled_commit_sha"] = "9" * 40
        expect_failure(
            "cross-source-prepublish",
            lambda: validate_prepublish_publish_binding(
                candidate=wrong_candidate,
                gate_refs={"g10a": g10a_ref},
                prepublish=prepublish_fixture,
            ),
            "release candidate differs",
        )

        receipt_prepublish_path = root / "receipt-prepublish.json"
        write_json(receipt_prepublish_path, {"fixture": True})
        gate_refs = {
            key: {
                "path": str(root / f"{key}.json"),
                "sha256": "1" * 64,
                "size_bytes": 1,
            }
            for key in RELEASE_GATE_LANES
        }
        resume = new_publish_receipt(
            prepublish=prepublish_fixture,
            prepublish_path=receipt_prepublish_path,
            prepublish_sha=sha256_file(receipt_prepublish_path),
            release_gates={"refs": gate_refs},
        )
        validate_publish_receipt_binding(
            resume,
            prepublish=prepublish_fixture,
            prepublish_path=receipt_prepublish_path,
            release_gates={"refs": gate_refs},
        )
        tampered_resume = copy.deepcopy(resume)
        tampered_resume["release_candidate"] = wrong_candidate
        expect_failure(
            "cross-source-resume",
            lambda: validate_publish_receipt_binding(
                tampered_resume,
                prepublish=prepublish_fixture,
                prepublish_path=receipt_prepublish_path,
                release_gates={"refs": gate_refs},
            ),
            "resume receipt RC differs",
        )
        retransmit_resume = copy.deepcopy(resume)
        retransmit_resume["packages"][0]["safe_retransmit_count"] = 1
        expect_failure(
            "receipt-automatic-retransmission",
            lambda: validate_publish_receipt_binding(
                retransmit_resume,
                prepublish=prepublish_fixture,
                prepublish_path=receipt_prepublish_path,
                release_gates={"refs": gate_refs},
            ),
            "ambiguity state differs",
        )
        run_publish_state_machine_selftest(
            root, archive=app_archive, package_name=app
        )
    print(SELFTEST_PASS_LINE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run offline positive/negative fixtures")
    subparsers = parser.add_subparsers(dest="mode")
    prepublish = subparsers.add_parser(
        "prepublish", aliases=["package"], help="create and validate immutable .crate packages"
    )
    prepublish.add_argument("--repo", type=Path, default=REPO_ROOT)
    prepublish.add_argument("--out", type=Path, required=True)
    prepublish.add_argument("--g10a", type=Path, required=True)
    prepublish.add_argument("--release-candidate-sha")
    prepublish.add_argument("--release-candidate-tree-sha")
    prepublish.add_argument("--release-candidate-tag", required=True)
    prepublish.add_argument("--command-timeout-seconds", type=int, default=7200)
    publish = subparsers.add_parser("publish", help="publish immutable prepublish packages serially")
    publish.add_argument("--repo", type=Path, default=REPO_ROOT)
    publish.add_argument("--prepublish", type=Path, required=True)
    publish.add_argument("--out", type=Path, required=True)
    publish.add_argument("--g10a", type=Path, required=True)
    publish.add_argument("--g08-rc", type=Path, required=True)
    publish.add_argument("--g09-rc", type=Path, required=True)
    publish.add_argument("--published-assets", type=Path, required=True)
    publish.add_argument("--final-tag", required=True)
    publish.add_argument("--resume", action="store_true")
    publish.add_argument("--poll-timeout-seconds", type=int, default=900)
    publish.add_argument("--poll-interval-seconds", type=int, default=15)
    publish.add_argument("--command-timeout-seconds", type=int, default=7200)
    args = parser.parse_args(argv)
    if args.self_test:
        require(args.mode is None, "--self-test cannot be combined with a release mode")
    else:
        require(args.mode is not None, "select prepublish/package, publish, or --self-test")
        require(args.command_timeout_seconds >= 1200, "command timeout must be at least 1200 seconds")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.self_test:
            run_selftest()
        elif args.mode in {"prepublish", "package"}:
            create_prepublish(args)
        elif args.mode == "publish":
            publish_release(args)
        else:
            raise ReleaseError(f"unsupported mode: {args.mode}")
        return 0
    except (ReleaseError, subprocess.SubprocessError, OSError) as error:
        print(f"ERROR: {sanitize_text(str(error))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
