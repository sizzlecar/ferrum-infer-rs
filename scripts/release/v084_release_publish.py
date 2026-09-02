#!/usr/bin/env python3
"""Safe promotion, Homebrew publication, and final evidence assembly for 0.8.4.

All externally mutating operations are opt-in.  ``promote`` requires
``--execute`` before it can PATCH GitHub, and ``homebrew`` requires
``--publish`` before it can change or push a tap checkout.  Credentials are
read only by the underlying ``gh``/``git`` processes from their existing
environment; they are never accepted as command-line options or persisted.

The collectors deliberately finish by invoking the read-only authoritative
validators in :mod:`v084_release_goal_gate`.  A terminal release PASS line is
therefore never emitted merely because a network or git command succeeded.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable
from urllib.parse import urlparse


RELEASE_DIR = Path(__file__).resolve().parent
if str(RELEASE_DIR) not in sys.path:
    sys.path.insert(0, str(RELEASE_DIR))

import v084_release_goal_gate as goal_gate  # noqa: E402


VERSION = "0.8.4"
TAG = "v0.8.4"
REPOSITORY = "sizzlecar/ferrum-infer-rs"
TAP_REPOSITORY = "sizzlecar/homebrew-ferrum"
TAP_BRANCH = "main"
SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TOKEN_NAMES = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "HOMEBREW_GITHUB_API_TOKEN",
    "CARGO_REGISTRY_TOKEN",
    "CRATES_IO_TOKEN",
)
FORMULA_PATHS = ("Formula/ferrum.rb", "Formula/ferrum-cuda.rb")
ASSET_NAMES = {
    "cpu": "ferrum-linux-x86_64.tar.gz",
    "metal": "ferrum-macos-aarch64.tar.gz",
    "cuda": "ferrum-linux-x86_64-cuda-sm89.tar.gz",
}
PROMOTION_PASS_PREFIX = "FERRUM 0.8.4 PROMOTION PASS"
PROMOTION_DRY_RUN_PREFIX = "FERRUM 0.8.4 PROMOTION DRY RUN PASS"
HOMEBREW_DRY_RUN_PREFIX = "FERRUM 0.8.4 HOMEBREW DRY RUN PASS"
HOMEBREW_PUBLISH_PREFIX = "FERRUM 0.8.4 HOMEBREW PUBLISH PASS"
SELFTEST_PASS_LINE = "FERRUM 0.8.4 RELEASE PUBLISH SELFTEST PASS"
OFFICIAL_FINAL_OUT = (RELEASE_DIR.parent.parent / goal_gate.FINAL_ARTIFACT_DIR).resolve()


class PublishError(RuntimeError):
    """A fail-closed publication or evidence-assembly check failed."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise PublishError(message)


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(value: dt.datetime | None = None) -> str:
    return (value or now()).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def pretty_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), f"refusing to replace symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, pretty_json(value))


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PublishError(f"cannot read {label} {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def safe_regular(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    require(expanded.exists(), f"{label} is missing: {expanded}")
    require(expanded.is_file() and not expanded.is_symlink(), f"{label} is not a regular non-symlink file: {expanded}")
    return expanded.resolve()


def reject_symlink_chain(path: Path, root: Path, label: str) -> None:
    """Reject a symlink in any component below ``root`` (not only the leaf)."""

    root = root.resolve()
    require(path.resolve().is_relative_to(root), f"{label} escapes its root")
    relative = path.absolute().relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), f"{label} traverses a symlink: {cursor}")


def artifact_ref(path: Path, root: Path) -> dict[str, Any]:
    path = safe_regular(path, "artifact")
    root = root.resolve()
    require(path.is_relative_to(root), f"artifact escapes output root: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def resolve_ref(value: Any, root: Path, label: str) -> Path:
    require(isinstance(value, dict) and set(value) == {"path", "size_bytes", "sha256"}, f"{label} reference fields differ")
    text = value.get("path")
    require(isinstance(text, str) and text, f"{label} reference path is empty")
    pure = PurePosixPath(text)
    require(not pure.is_absolute() and "\\" not in text and ".." not in pure.parts, f"{label} reference path is unsafe")
    root = root.resolve()
    candidate = root.joinpath(*pure.parts)
    reject_symlink_chain(candidate, root, label)
    path = candidate.resolve()
    require(path.is_relative_to(root) and path.is_file() and not path.is_symlink(), f"{label} reference is missing or escapes")
    require(type(value.get("size_bytes")) is int and value["size_bytes"] == path.stat().st_size, f"{label} reference size differs")
    require(isinstance(value.get("sha256"), str) and SHA256_RE.fullmatch(value["sha256"]) is not None and value["sha256"] == sha256_file(path), f"{label} reference SHA256 differs")
    return path


def secret_values() -> tuple[str, ...]:
    return tuple(value for name in TOKEN_NAMES if len((value := os.environ.get(name, ""))) >= 8)


def sanitize(value: str) -> str:
    result = value
    for secret in secret_values():
        result = result.replace(secret, "<redacted>")
    result = re.sub(r"(?i)(authorization\s*:\s*bearer\s+)\S+", r"\1<redacted>", result)
    result = re.sub(r"(?i)(https://)[^/@\s]+:[^/@\s]+(@github\.com)", r"\1<redacted>\2", result)
    return result


def assert_no_secrets(root: Path) -> None:
    needles = tuple(value.encode("utf-8") for value in secret_values())
    for path in root.rglob("*"):
        require(not path.is_symlink(), f"artifact tree contains symlink: {path}")
        if not path.is_file():
            continue
        payload = path.read_bytes()
        require(not any(needle in payload for needle in needles), f"credential value was persisted in {path}")
        if path.suffix.lower() in {".json", ".txt", ".log", ".stdout", ".stderr", ".diff", ".rb"}:
            text = payload.decode("utf-8", errors="replace")
            require(re.search(r"(?i)authorization\s*:\s*bearer\s+(?!<redacted>)\S+", text) is None, f"bearer credential marker persisted in {path}")


def ensure_new_or_directory(out: Path) -> Path:
    expanded = out.expanduser()
    if expanded.exists():
        require(expanded.is_dir() and not expanded.is_symlink(), f"output is not a regular directory: {expanded}")
    else:
        expanded.mkdir(parents=True)
    return expanded.resolve()


def discover_portable_refs(value: Any, owner: Path, root: Path) -> list[Path]:
    """Return existing, byte-bound relative references from one JSON value.

    Some evidence schemas also use ``path/size_bytes/sha256`` for recorded
    *remote* absolute identities.  Those are intentionally not treated as
    portable references here; the authoritative validator checks them as
    provenance rather than trying to dereference them locally.
    """

    found: list[Path] = []
    if isinstance(value, dict):
        if set(value) == {"path", "size_bytes", "sha256"}:
            raw = value.get("path")
            if isinstance(raw, str) and raw:
                pure = PurePosixPath(raw)
                if not pure.is_absolute() and "\\" not in raw and ".." not in pure.parts:
                    # Release manifests conventionally anchor every nested
                    # ref at the top manifest root.  A few standalone command
                    # receipts anchor at their own directory, so accept that
                    # only when byte identity makes the resolution unique.
                    candidates: set[Path] = set()
                    anchor = owner.parent
                    while True:
                        candidates.add(anchor.joinpath(*pure.parts))
                        if anchor == root:
                            break
                        require(anchor.is_relative_to(root), "portable JSON owner escapes closure root")
                        anchor = anchor.parent
                    matches: list[Path] = []
                    for candidate in candidates:
                        if not candidate.exists():
                            continue
                        reject_symlink_chain(candidate, root, "portable closure reference")
                        resolved = candidate.resolve()
                        if (
                            resolved.is_file()
                            and not resolved.is_symlink()
                            and type(value.get("size_bytes")) is int
                            and value["size_bytes"] == resolved.stat().st_size
                            and isinstance(value.get("sha256"), str)
                            and value["sha256"] == sha256_file(resolved)
                        ):
                            matches.append(resolved)
                    matches = list(dict.fromkeys(matches))
                    require(len(matches) <= 1, f"portable closure reference is ambiguous: {raw}")
                    found.extend(matches)
        for child in value.values():
            found.extend(discover_portable_refs(child, owner, root))
    elif isinstance(value, list):
        for child in value:
            found.extend(discover_portable_refs(child, owner, root))
    return found


def copy_portable_closure(source_manifest: Path, destination: Path) -> Path:
    """Copy a manifest and every reachable relative byte-bound reference."""

    source_manifest = safe_regular(source_manifest, "portable manifest")
    source_root = source_manifest.parent.resolve()
    destination = destination.resolve()
    require(not destination.exists(), f"portable destination already exists: {destination}")
    require(not destination.is_relative_to(source_root), "portable destination may not be inside its source root")
    destination.mkdir(parents=True)
    queue = [source_manifest]
    seen: set[Path] = set()
    while queue:
        source = queue.pop()
        if source in seen:
            continue
        seen.add(source)
        require(source.is_relative_to(source_root) and source.is_file() and not source.is_symlink(), f"portable source escapes or is not regular: {source}")
        relative = source.relative_to(source_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), f"portable closure path collision: {target}")
        shutil.copyfile(source, target, follow_symlinks=False)
        require(sha256_file(target) == sha256_file(source), f"portable copy differs: {relative}")
        if source.suffix.lower() == ".json":
            try:
                document = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise PublishError(f"cannot read portable closure JSON {source}: {error}") from error
            queue.extend(discover_portable_refs(document, source, source_root))
    return destination / source_manifest.name


def copy_tree_strict(source_root: Path, destination: Path) -> None:
    source_root = source_root.resolve()
    require(source_root.is_dir() and not source_root.is_symlink(), f"tree source is invalid: {source_root}")
    destination = destination.resolve()
    require(not destination.exists(), f"tree destination exists: {destination}")
    require(not destination.is_relative_to(source_root), "tree destination may not be inside source")
    for path in source_root.rglob("*"):
        require(not path.is_symlink(), f"tree source contains symlink: {path}")
        require(path.is_dir() or path.is_file(), f"tree source contains unsupported entry: {path}")
    shutil.copytree(source_root, destination, symlinks=False)
    for path in destination.rglob("*"):
        require(not path.is_symlink(), f"copied tree contains symlink: {path}")


def install_validated_manifest(candidate: Path, final: Path, validator: Callable[[Path], Any]) -> Any:
    require(candidate.parent == final.parent, "candidate/final manifest roots differ")
    result = validator(candidate)
    payload = candidate.read_bytes()
    if final.exists():
        require(final.is_file() and not final.is_symlink() and final.read_bytes() == payload, f"canonical manifest differs: {final}")
    else:
        atomic_write(final, payload)
    candidate.unlink(missing_ok=True)
    return validator(final)


def validate_patch_body(value: Any) -> bytes:
    require(isinstance(value, dict) and set(value) == {"prerelease"} and value["prerelease"] is False, "GitHub promotion PATCH body must contain only prerelease:false")
    payload = canonical_json(value)
    require(payload == b'{"prerelease":false}', "GitHub promotion PATCH bytes differ")
    return payload


def promotion_target_identity(
    value: Any, *, prerelease: bool, where: str
) -> dict[str, Any]:
    try:
        return goal_gate.promotion_target_identity(
            value, prerelease=prerelease, where=where
        )
    except Exception as error:
        raise PublishError(f"{where} target identity differs: {error}") from error


def validate_mutation_attempt(
    path: Path, *, release_id: int, endpoint: str
) -> dict[str, Any]:
    value = read_json(path, "promotion mutation attempt")
    fields = {
        "schema_version",
        "artifact_type",
        "status",
        "method",
        "endpoint",
        "body",
        "body_sha256",
        "release_id",
        "attempted_at",
        "confirmed_at",
        "confirmation",
        "ambiguous_outcome_recovered",
    }
    require(set(value) == fields, "promotion mutation attempt fields differ")
    body = {"prerelease": False}
    require(
        value["schema_version"] == SCHEMA_VERSION
        and value["artifact_type"]
        == "ferrum_v084_github_promotion_mutation_receipt"
        and value["method"] == "PATCH"
        and value["endpoint"] == endpoint
        and value["body"] == body
        and value["body_sha256"] == sha256_bytes(validate_patch_body(body))
        and value["release_id"] == release_id,
        "promotion mutation attempt immutable identity differs",
    )
    require(
        isinstance(value["attempted_at"], str) and value["attempted_at"],
        "promotion mutation attempt timestamp is absent",
    )
    if value["status"] == "attempted":
        require(
            value["confirmed_at"] is None
            and value["confirmation"] == "pending"
            and value["ambiguous_outcome_recovered"] is False,
            "unconfirmed promotion mutation attempt state differs",
        )
    elif value["status"] == "confirmed":
        try:
            goal_gate.validate_promotion_mutation_receipt(
                value,
                release_id=release_id,
                where="promotion mutation receipt",
            )
        except Exception as error:
            raise PublishError(
                f"confirmed promotion mutation receipt is invalid: {error}"
            ) from error
    else:
        raise PublishError("promotion mutation attempt status differs")
    return value


def persist_mutation_attempt(path: Path, *, release_id: int, endpoint: str) -> None:
    require(not path.exists(), "promotion mutation attempt already exists")
    body = {"prerelease": False}
    write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "ferrum_v084_github_promotion_mutation_receipt",
            "status": "attempted",
            "method": "PATCH",
            "endpoint": endpoint,
            "body": body,
            "body_sha256": sha256_bytes(validate_patch_body(body)),
            "release_id": release_id,
            "attempted_at": iso(),
            "confirmed_at": None,
            "confirmation": "pending",
            "ambiguous_outcome_recovered": False,
        },
    )
    validate_mutation_attempt(path, release_id=release_id, endpoint=endpoint)


def confirm_mutation_attempt(
    path: Path, *, release_id: int, endpoint: str, confirmation: str
) -> dict[str, Any]:
    require(
        confirmation
        in {"patch-response", "saved-patch-response", "live-state-recovery"},
        "promotion mutation confirmation differs",
    )
    value = validate_mutation_attempt(path, release_id=release_id, endpoint=endpoint)
    if value["status"] == "confirmed":
        compatible = value["confirmation"] == confirmation or {
            value["confirmation"],
            confirmation,
        } <= {"patch-response", "saved-patch-response"}
        require(compatible, "saved promotion mutation confirmation differs")
        return value
    value["status"] = "confirmed"
    value["confirmed_at"] = iso()
    value["confirmation"] = confirmation
    value["ambiguous_outcome_recovered"] = confirmation == "live-state-recovery"
    write_json(path, value)
    return validate_mutation_attempt(path, release_id=release_id, endpoint=endpoint)


Transport = Callable[[str, str, dict[str, Any] | None, Path], dict[str, Any]]


def gh_transport(method: str, endpoint: str, body: dict[str, Any] | None, receipt_path: Path) -> dict[str, Any]:
    require(method in {"GET", "PATCH"}, "unsupported GitHub REST method")
    argv = ["gh", "api", endpoint, "--hostname", "github.com", "--method", method]
    payload: bytes | None = None
    if body is not None:
        require(method == "PATCH", "request body is allowed only for PATCH")
        payload = validate_patch_body(body)
        argv.extend(["--input", "-"])
    require(not any(secret in "\0".join(argv) for secret in secret_values()), "credential value is forbidden in GitHub argv")
    started = now()
    timeout_seconds = 60
    try:
        process = subprocess.run(
            argv,
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=RELEASE_DIR.parent.parent,
            timeout=timeout_seconds,
            check=False,
        )
        timed_out = False
    except subprocess.TimeoutExpired as error:
        process = subprocess.CompletedProcess(argv, 124, error.stdout or b"", error.stderr or b"")
        timed_out = True
    finished = now()
    stderr = sanitize(process.stderr.decode("utf-8", errors="replace"))
    receipt = {
        "schema_version": 1,
        "kind": "github-rest-command",
        "method": method,
        "endpoint": endpoint,
        "argv": argv,
        "body": copy.deepcopy(body),
        "body_sha256": sha256_bytes(payload or b""),
        "credentials_in_argv": False,
        "environment_recorded": False,
        "started_at": iso(started),
        "finished_at": iso(finished),
        "duration_seconds": (finished - started).total_seconds(),
        "expected_duration_seconds": 10,
        "deadline_seconds": timeout_seconds,
        "deadline_at": iso(started + dt.timedelta(seconds=timeout_seconds)),
        "progress_signal": "gh process exit and parseable JSON response bytes",
        "timed_out": timed_out,
        "returncode": process.returncode,
        "response_size_bytes": len(process.stdout),
        "response_sha256": sha256_bytes(process.stdout),
        "stderr_tail": stderr[-2000:],
    }
    write_json(receipt_path, receipt)
    require(not timed_out and process.returncode == 0, f"GitHub REST {method} {endpoint} failed: {stderr[-1000:]}")
    try:
        value = json.loads(process.stdout.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PublishError(f"GitHub REST returned invalid JSON for {endpoint}: {error}") from error
    require(isinstance(value, dict), f"GitHub REST response is not an object: {endpoint}")
    return value


def transport_and_persist(
    transport: Transport,
    method: str,
    endpoint: str,
    body: dict[str, Any] | None,
    receipt_path: Path,
    *,
    response_path: Path | None = None,
) -> dict[str, Any]:
    """Run one REST operation and atomically persist its returned snapshot.

    Keeping this wrapper immediately around the mutating transport minimizes
    the local crash window.  A process can still die after GitHub commits the
    PATCH but before the response reaches Python; ``run_promote`` has a
    separate, fail-closed recovery path for exactly that state.
    """

    value = transport(method, endpoint, body, receipt_path)
    if response_path is not None:
        write_json(response_path, value)
    return value


def copy_prerelease_for_promotion(prerelease_path: Path, out: Path) -> Path:
    copied = out / "evidence" / "prerelease"
    return copy_portable_closure(prerelease_path, copied)


def run_promote(args: argparse.Namespace, *, transport: Transport = gh_transport) -> Path | None:
    started = now()
    prerelease_path = safe_regular(args.prerelease_manifest, "prerelease manifest")
    try:
        prerelease = goal_gate.validate_prerelease_manifest(prerelease_path)
    except Exception as error:
        raise PublishError(f"prerelease deep validation failed: {error}") from error
    out = ensure_new_or_directory(args.out)
    canonical = out / "promotion.manifest.json"
    if canonical.exists():
        try:
            validated = goal_gate.validate_promotion_manifest(canonical)
        except Exception as error:
            raise PublishError(f"existing canonical promotion manifest is invalid: {error}") from error
        print(validated["pass_line"])
        return canonical

    evidence_root = out / "evidence"
    commands = out / "commands"
    github = evidence_root / "github"
    commands.mkdir(parents=True, exist_ok=True)
    github.mkdir(parents=True, exist_ok=True)
    copied_prerelease = out / "evidence" / "prerelease" / prerelease_path.name
    if not copied_prerelease.exists():
        copied_prerelease = copy_prerelease_for_promotion(prerelease_path, out)
    try:
        copied_value = goal_gate.validate_prerelease_manifest(copied_prerelease)
    except Exception as error:
        raise PublishError(f"copied prerelease closure failed validation: {error}") from error
    require(copied_value["source"] == prerelease["source"] and copied_value["release"] == prerelease["release"], "copied prerelease identity differs")

    release_id = prerelease["release"]["id"]
    release_endpoint = f"/repos/{REPOSITORY}/releases/{release_id}"
    live = transport("GET", release_endpoint, None, commands / "github-live-release.json")
    require(live.get("id") == release_id and live.get("tag_name") == TAG, "live GitHub release id/tag differs")
    before_path = github / "release-before.json"
    after_path = github / "release-after.json"
    mutation_path = github / "promotion-mutation.json"

    if not args.execute:
        plan_snapshot = github / "dry-run-live-release.json"
        write_json(plan_snapshot, live)
        finished = now()
        receipt = {
            "schema_version": 1,
            "artifact_type": "ferrum_v084_promotion_collector_receipt",
            "status": "dry-run",
            "version": VERSION,
            "started_at": iso(started),
            "finished_at": iso(finished),
            "generated_at": iso(),
            "execute": False,
            "mutation_performed": False,
            "planned_patch_body": {"prerelease": False},
            "release_id": release_id,
            "progress_signal": "saved live GitHub release snapshot",
            "deadline_seconds_per_request": 60,
        }
        write_json(out / "promotion.collector.json", receipt)
        assert_no_secrets(out)
        print(f"{PROMOTION_DRY_RUN_PREFIX}: {out}")
        return None

    patch_performed = False
    resume_recovered_patch_response = False
    if live.get("prerelease") is True:
        if before_path.exists():
            saved_before = read_json(before_path, "saved release-before")
            require(
                promotion_target_identity(
                    saved_before, prerelease=True, where="saved release-before"
                )
                == promotion_target_identity(
                    live, prerelease=True, where="live prerelease"
                ),
                "saved release-before target identity differs from live prerelease",
            )
        else:
            write_json(before_path, live)
        require(not after_path.exists(), "saved release-after exists while live release is still prerelease")
        if mutation_path.exists():
            validate_mutation_attempt(
                mutation_path, release_id=release_id, endpoint=release_endpoint
            )
            raise PublishError(
                "a promotion PATCH was already attempted but the live release is still "
                "a prerelease; refusing to issue a second PATCH"
            )
        # This durable intent is written before the network mutation.  From
        # this point onward a retry may observe/recover, but may never PATCH a
        # second time when the first request's outcome is ambiguous.
        persist_mutation_attempt(
            mutation_path, release_id=release_id, endpoint=release_endpoint
        )
        after = transport_and_persist(
            transport,
            "PATCH",
            release_endpoint,
            {"prerelease": False},
            commands / "github-promotion-patch.json",
            response_path=after_path,
        )
        require(after.get("prerelease") is False, "GitHub PATCH response did not promote release")
        confirm_mutation_attempt(
            mutation_path,
            release_id=release_id,
            endpoint=release_endpoint,
            confirmation="patch-response",
        )
        patch_performed = True
    elif live.get("prerelease") is False:
        require(before_path.is_file() and not before_path.is_symlink(), "release is already final but saved release-before is absent")
        require(
            mutation_path.is_file() and not mutation_path.is_symlink(),
            "release is already final but the durable promotion mutation attempt is absent",
        )
        validate_mutation_attempt(
            mutation_path, release_id=release_id, endpoint=release_endpoint
        )
        saved_before = read_json(before_path, "saved release-before")
        require(saved_before.get("prerelease") is True, "saved release-before is not a prerelease snapshot")
        if after_path.exists():
            require(after_path.is_file() and not after_path.is_symlink(), "saved release-after/PATCH response is not a regular file")
            after = read_json(after_path, "saved release-after")
            require(after.get("prerelease") is False, "saved promotion transition differs")
            require(
                promotion_target_identity(
                    after, prerelease=False, where="saved release-after"
                )
                == promotion_target_identity(
                    live, prerelease=False, where="live final release"
                ),
                "live final target identity differs from saved PATCH response",
            )
            confirm_mutation_attempt(
                mutation_path,
                release_id=release_id,
                endpoint=release_endpoint,
                confirmation="saved-patch-response",
            )
        else:
            # Recover only the narrowly identifiable crash state: before was
            # durably saved, the live release is now final, and nothing except
            # GitHub's promotion fields changed.  The explicit row comparison
            # keeps asset id/name/size/digest identity visible in this path.
            try:
                before_rows = goal_gate.github_asset_rows(
                    saved_before,
                    prerelease=True,
                    where="promotion crash-recovery release-before",
                )
                live_rows = goal_gate.github_asset_rows(
                    live,
                    prerelease=False,
                    where="promotion crash-recovery live release",
                )
            except Exception as error:
                raise PublishError(f"cannot recover promotion PATCH response identity: {error}") from error
            require(
                saved_before.get("id") == live.get("id") == release_id
                and saved_before.get("tag_name") == live.get("tag_name") == TAG,
                "cannot recover promotion PATCH response: release id/tag differs",
            )
            require(before_rows == live_rows, "cannot recover promotion PATCH response: asset id/name/size/digest set changed")
            require(
                goal_gate.asset_set_sha256(live_rows)
                == prerelease["release"]["asset_set_sha256"],
                "cannot recover promotion PATCH response: asset fingerprint differs",
            )
            after = copy.deepcopy(live)
            write_json(after_path, after)
            confirm_mutation_attempt(
                mutation_path,
                release_id=release_id,
                endpoint=release_endpoint,
                confirmation="live-state-recovery",
            )
            resume_recovered_patch_response = True
    else:
        raise PublishError("live GitHub release prerelease field is not boolean")

    observed_after = transport("GET", release_endpoint, None, commands / "github-release-after-get.json")
    require(
        promotion_target_identity(after, prerelease=False, where="PATCH after")
        == promotion_target_identity(
            observed_after, prerelease=False, where="GET after"
        ),
        "GitHub release target identity changed after promotion",
    )
    latest = transport("GET", f"/repos/{REPOSITORY}/releases/latest", None, commands / "github-latest-release.json")
    tag_ref = transport("GET", f"/repos/{REPOSITORY}/git/ref/tags/{TAG}", None, commands / "github-tag-ref.json")
    ref_object = tag_ref.get("object") if isinstance(tag_ref, dict) else None
    require(isinstance(ref_object, dict) and ref_object.get("type") == "tag" and GIT_SHA_RE.fullmatch(str(ref_object.get("sha", ""))) is not None, "final tag ref is not annotated")
    tag_object = transport("GET", f"/repos/{REPOSITORY}/git/tags/{ref_object['sha']}", None, commands / "github-tag-object.json")
    latest_path = github / "latest-release.json"
    tag_ref_path = github / "tag-ref.json"
    tag_path = github / "tag-object.json"
    write_json(latest_path, latest)
    write_json(tag_ref_path, tag_ref)
    write_json(tag_path, tag_object)

    finished = now()
    pass_line = f"{PROMOTION_PASS_PREFIX}: {out}"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ferrum_v084_promotion_manifest",
        "status": "pass",
        "version": VERSION,
        "source": copy.deepcopy(prerelease["source"]),
        "release": copy.deepcopy(prerelease["release"]),
        "evidence": {
            "prerelease_manifest": artifact_ref(copied_prerelease, out),
            "mutation_receipt": artifact_ref(mutation_path, out),
            "release_before": artifact_ref(before_path, out),
            "release_after": artifact_ref(after_path, out),
            "latest_release": artifact_ref(latest_path, out),
            "tag_ref_snapshot": artifact_ref(tag_ref_path, out),
            "tag_snapshot": artifact_ref(tag_path, out),
        },
        "artifact_dir": str(out),
        "pass_line": pass_line,
    }
    candidate = out / ".promotion.manifest.candidate.json"
    write_json(candidate, manifest)
    try:
        validated = install_validated_manifest(candidate, canonical, goal_gate.validate_promotion_manifest)
    except Exception as error:
        candidate.unlink(missing_ok=True)
        raise PublishError(f"assembled promotion manifest failed authoritative validation: {error}") from error
    receipt = {
        "schema_version": 1,
        "artifact_type": "ferrum_v084_promotion_collector_receipt",
        "status": "pass",
        "version": VERSION,
        "started_at": iso(started),
        "finished_at": iso(finished),
        "generated_at": iso(),
        "execute": True,
        "mutation_performed": patch_performed,
        "resume_without_patch": not patch_performed,
        "resume_recovered_patch_response": resume_recovered_patch_response,
        "patch_body": {"prerelease": False},
        "patch_body_sha256": sha256_bytes(validate_patch_body({"prerelease": False})),
        "manifest": artifact_ref(canonical, out),
        "progress_signal": "before/after/latest/tag snapshots and validator PASS",
        "deadline_seconds_per_request": 60,
    }
    write_json(out / "promotion.collector.json", receipt)
    assert_no_secrets(out)
    require(validated["pass_line"] == pass_line, "authoritative promotion PASS line differs")
    print(pass_line)
    return canonical


def promotion_assets(promotion_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    promotion_path = safe_regular(promotion_path, "promotion manifest")
    try:
        promotion = goal_gate.validate_promotion_manifest(promotion_path)
    except Exception as error:
        raise PublishError(f"promotion deep validation failed: {error}") from error
    raw = read_json(promotion_path, "promotion manifest")
    prerelease_path = resolve_ref(raw["evidence"]["prerelease_manifest"], promotion_path.parent, "promotion prerelease")
    prerelease = goal_gate.validate_prerelease_manifest(prerelease_path)
    assets: dict[str, dict[str, str]] = {}
    for backend, name in ASSET_NAMES.items():
        digest = prerelease["packages"][backend]["asset_sha256"]
        require(SHA256_RE.fullmatch(digest) is not None, f"{backend} release asset SHA differs")
        assets[backend] = {
            "name": name,
            "sha256": digest,
            "url": f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{name}",
        }
    return promotion, assets


def render_formulae(assets: dict[str, dict[str, str]]) -> dict[str, str]:
    metal = f'''class Ferrum < Formula
  desc "Production-grade LLM inference in Rust for Apple Silicon and Linux CPU"
  homepage "https://github.com/{REPOSITORY}"
  version "{VERSION}"
  license "MIT"

  on_macos do
    on_arm do
      url "{assets["metal"]["url"]}"
      sha256 "{assets["metal"]["sha256"]}"
    end
  end

  on_linux do
    on_intel do
      url "{assets["cpu"]["url"]}"
      sha256 "{assets["cpu"]["sha256"]}"
    end
  end

  conflicts_with "ferrum-cuda", because: "both install the ferrum binary"

  def install
    bin.install "ferrum"
    doc.install "README.md"
  end

  test do
    assert_match "ferrum #{{version}}", shell_output("#{{bin}}/ferrum --version")
    assert_match "serve", shell_output("#{{bin}}/ferrum serve --help")
  end
end
'''
    cuda = f'''class FerrumCuda < Formula
  desc "Production-grade LLM inference in Rust with NVIDIA CUDA sm89 support"
  homepage "https://github.com/{REPOSITORY}"
  url "{assets["cuda"]["url"]}"
  version "{VERSION}"
  sha256 "{assets["cuda"]["sha256"]}"
  license "MIT"

  depends_on :linux

  conflicts_with "ferrum", because: "both install the ferrum binary"

  def install
    bin.install "ferrum"
    doc.install "README.md"
    doc.install "CUDA-BUILD.txt"
  end

  def caveats
    <<~EOS
      ferrum-cuda is the Linux x86_64 CUDA sm89 build. It requires an NVIDIA
      driver plus CUDA 12 runtime libraries such as libcudart, cublas, curand,
      and libcuda on the target host.
    EOS
  end

  test do
    assert_path_exists bin/"ferrum"
  end
end
'''
    result = {"Formula/ferrum.rb": metal, "Formula/ferrum-cuda.rb": cuda}
    validate_formulae(result, assets)
    return result


def validate_formulae(formulae: dict[str, str], assets: dict[str, dict[str, str]]) -> None:
    require(set(formulae) == set(FORMULA_PATHS), "formula path denominator differs")
    expected = {
        "Formula/ferrum.rb": [assets["metal"]["url"], assets["metal"]["sha256"], assets["cpu"]["url"], assets["cpu"]["sha256"], 'bin.install "ferrum"', 'doc.install "README.md"'],
        "Formula/ferrum-cuda.rb": [assets["cuda"]["url"], assets["cuda"]["sha256"], 'bin.install "ferrum"', 'doc.install "README.md"', 'doc.install "CUDA-BUILD.txt"'],
    }
    for name, text in formulae.items():
        require(text.endswith("\n") and f'version "{VERSION}"' in text, f"{name} version/newline differs")
        require("0.7.7" not in text and "0.8.0" not in text and "0.8.3" not in text, f"{name} retained a prior version")
        for marker in expected[name]:
            require(text.count(marker) == 1, f"{name} canonical marker differs: {marker}")
        sha_rows = re.findall(r'^\s*sha256 "([0-9a-f]{64})"$', text, re.MULTILINE)
        require(len(sha_rows) == (2 if name.endswith("ferrum.rb") else 1), f"{name} checksum denominator differs")


def normalize_origin(value: str, *, allow_local: bool = False) -> str:
    text = value.strip()
    if allow_local and Path(text).exists():
        return TAP_REPOSITORY
    if text.startswith("git@github.com:"):
        path = text.removeprefix("git@github.com:")
    else:
        parsed = urlparse(text)
        require(parsed.scheme == "https" and parsed.hostname == "github.com" and parsed.username is None and parsed.password is None and not parsed.query and not parsed.fragment, "tap origin must be credential-free github.com HTTPS or SSH")
        path = parsed.path.lstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    require(path == TAP_REPOSITORY, f"tap origin repository differs: {path}")
    return path


def run_git(
    repo: Path,
    argv: list[str],
    *,
    receipts: list[dict[str, Any]],
    check: bool = True,
    timeout_seconds: int = 120,
) -> subprocess.CompletedProcess[str]:
    command = ["git", *argv]
    require(not any(secret in "\0".join(command) for secret in secret_values()), "credential value is forbidden in git argv")
    started = now()
    try:
        process = subprocess.run(command, cwd=repo, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_seconds, check=False)
        timed_out = False
    except subprocess.TimeoutExpired as error:
        process = subprocess.CompletedProcess(command, 124, stdout=error.stdout or "", stderr=error.stderr or "")
        timed_out = True
    finished = now()
    stdout = sanitize(process.stdout)
    stderr = sanitize(process.stderr)
    receipts.append({
        "argv": command,
        "cwd": str(repo.resolve()),
        "started_at": iso(started),
        "finished_at": iso(finished),
        "duration_seconds": (finished - started).total_seconds(),
        "expected_duration_seconds": 5 if argv[:1] != ["push"] else 30,
        "deadline_seconds": timeout_seconds,
        "deadline_at": iso(started + dt.timedelta(seconds=timeout_seconds)),
        "progress_signal": "git process exit and stdout/stderr bytes",
        "returncode": process.returncode,
        "timed_out": timed_out,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "environment_recorded": False,
        "credentials_in_argv": False,
    })
    if check:
        require(not timed_out and process.returncode == 0, f"git command failed: {command}: {stderr[-1000:]}")
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def git_text(repo: Path, argv: list[str], receipts: list[dict[str, Any]]) -> str:
    return run_git(repo, argv, receipts=receipts).stdout.strip()


def remote_main(repo: Path, receipts: list[dict[str, Any]]) -> str:
    output = git_text(repo, ["ls-remote", "--exit-code", "origin", f"refs/heads/{TAP_BRANCH}"], receipts)
    rows = [line.split() for line in output.splitlines() if line.strip()]
    require(len(rows) == 1 and len(rows[0]) == 2 and rows[0][1] == f"refs/heads/{TAP_BRANCH}" and GIT_SHA_RE.fullmatch(rows[0][0]) is not None, "tap remote main identity differs")
    return rows[0][0]


def changed_paths(repo: Path, receipts: list[dict[str, Any]], *extra: str) -> set[str]:
    output = git_text(repo, ["diff", "--name-only", *extra], receipts)
    return {line for line in output.splitlines() if line}


def formula_diff(repo: Path, formulae: dict[str, str]) -> str:
    chunks: list[str] = []
    for relative in FORMULA_PATHS:
        path = repo / relative
        require(path.is_file() and not path.is_symlink(), f"tap formula is missing/non-regular: {relative}")
        old = path.read_text(encoding="utf-8")
        chunks.extend(difflib.unified_diff(old.splitlines(keepends=True), formulae[relative].splitlines(keepends=True), fromfile=f"a/{relative}", tofile=f"b/{relative}"))
    return "".join(chunks)


def tap_state(repo: Path, receipts: list[dict[str, Any]], *, allow_local_origin: bool = False) -> dict[str, str]:
    repo = repo.expanduser().resolve()
    require(repo.is_dir() and not repo.is_symlink(), f"tap checkout is invalid: {repo}")
    require(git_text(repo, ["rev-parse", "--is-inside-work-tree"], receipts) == "true", "tap is not a git worktree")
    branch = git_text(repo, ["branch", "--show-current"], receipts)
    require(branch == TAP_BRANCH, "tap checkout branch must be main")
    origin = git_text(repo, ["remote", "get-url", "origin"], receipts)
    normalize_origin(origin, allow_local=allow_local_origin)
    status = git_text(repo, ["status", "--porcelain=v1", "--untracked-files=all"], receipts)
    require(status == "", f"tap checkout contains dirty or untracked files: {status}")
    head = git_text(repo, ["rev-parse", "HEAD"], receipts)
    tracking = git_text(repo, ["rev-parse", "refs/remotes/origin/main"], receipts)
    remote = remote_main(repo, receipts)
    return {"branch": branch, "origin": TAP_REPOSITORY, "head": head, "origin_main": tracking, "remote_main": remote}


def run_homebrew(args: argparse.Namespace, *, allow_local_origin: bool = False) -> Path:
    started = now()
    promotion, assets = promotion_assets(args.promotion_manifest)
    out = ensure_new_or_directory(args.out)
    tap = args.tap.expanduser().resolve()
    require(not out.is_relative_to(tap) and not tap.is_relative_to(out), "Homebrew output and tap checkout must be disjoint")
    receipts: list[dict[str, Any]] = []
    formulae = render_formulae(assets)
    state = tap_state(tap, receipts, allow_local_origin=allow_local_origin)
    proposal_root = out / "proposal"
    for relative, text in formulae.items():
        path = proposal_root / relative
        if path.exists():
            require(path.is_file() and not path.is_symlink() and path.read_text(encoding="utf-8") == text, f"existing proposal differs: {path}")
        else:
            atomic_write(path, text.encode("utf-8"))
    diff = formula_diff(tap, formulae)
    atomic_write(out / "formula.diff", diff.encode("utf-8"))
    proposal_hashes = {relative: sha256_bytes(text.encode("utf-8")) for relative, text in formulae.items()}
    current_exact = all((tap / relative).read_text(encoding="utf-8") == formulae[relative] for relative in FORMULA_PATHS)
    if not args.publish:
        require(
            state["head"] == state["origin_main"] == state["remote_main"],
            "Homebrew dry-run tap baseline must have HEAD == origin/main == remote main",
        )
    if state["origin_main"] != state["remote_main"]:
        # A successful push followed by a local fetch interruption is the one
        # safe stale-tracking state: local HEAD already equals remote main and
        # both formula bytes are exact.  Dry-runs never repair tracking refs.
        require(
            args.publish and state["head"] == state["remote_main"] and current_exact,
            "tap origin/main tracking ref differs from remote main",
        )
    mutation_performed = False
    commit_sha: str | None = None

    if args.publish:
        remote_before = state["remote_main"]
        if state["head"] == remote_before:
            if not current_exact:
                for relative, text in formulae.items():
                    atomic_write(tap / relative, text.encode("utf-8"))
                status_output = run_git(
                    tap,
                    ["status", "--porcelain=v1", "--untracked-files=all"],
                    receipts=receipts,
                ).stdout
                status_paths = {
                    line[3:]
                    for line in status_output.splitlines()
                    if line
                }
                require(status_paths == set(FORMULA_PATHS), f"tap mutation touched unexpected paths: {sorted(status_paths)}")
                run_git(tap, ["add", "--", *FORMULA_PATHS], receipts=receipts)
                staged = changed_paths(tap, receipts, "--cached")
                require(staged == status_paths, f"staged tap paths differ: {sorted(staged)}")
                run_git(tap, ["commit", "-m", f"ferrum {VERSION}", "--", *FORMULA_PATHS], receipts=receipts)
                mutation_performed = True
            else:
                # Exact formula bytes at the remote head means this invocation
                # is idempotent and must not create an empty commit.
                commit_sha = state["head"]
        else:
            # A prior invocation may have committed locally and failed before
            # push.  Only the exact one-parent/two-formula proposal can resume.
            require(current_exact, "tap local head diverged from remote without exact proposed formula bytes")
            parents = git_text(tap, ["rev-list", "--parents", "-n", "1", "HEAD"], receipts).split()
            require(len(parents) == 2 and parents[1] == remote_before, "tap local divergence is not a single resumable release commit")
            commit_paths = set(git_text(tap, ["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"], receipts).splitlines())
            require(commit_paths == set(FORMULA_PATHS), "resumable tap commit changed unexpected paths")

        head_after_commit = git_text(tap, ["rev-parse", "HEAD"], receipts)
        current_remote = remote_main(tap, receipts)
        if current_remote != head_after_commit:
            require(current_remote == remote_before, "tap remote main drifted before push")
            run_git(tap, ["push", "origin", "HEAD:refs/heads/main"], receipts=receipts, timeout_seconds=300)
            mutation_performed = True
        remote_after = remote_main(tap, receipts)
        require(remote_after == head_after_commit, "tap push did not install exact commit")
        run_git(tap, ["fetch", "--no-tags", "origin", "main"], receipts=receipts, timeout_seconds=300)
        require(git_text(tap, ["rev-parse", "refs/remotes/origin/main"], receipts) == head_after_commit, "tap fetched origin/main differs after push")
        for relative, text in formulae.items():
            remote_bytes = run_git(tap, ["show", f"refs/remotes/origin/main:{relative}"], receipts=receipts).stdout.encode("utf-8")
            require(remote_bytes == text.encode("utf-8"), f"remote tap formula bytes differ after push: {relative}")
        require(git_text(tap, ["status", "--porcelain=v1", "--untracked-files=all"], receipts) == "", "tap is dirty after publication")
        commit_sha = head_after_commit

    finished = now()
    status = "pass" if args.publish else "dry-run"
    prefix = HOMEBREW_PUBLISH_PREFIX if args.publish else HOMEBREW_DRY_RUN_PREFIX
    pass_line = f"{prefix}: {out}"
    receipt_path = out / "git.commands.json"
    write_json(receipt_path, {"schema_version": 1, "commands": receipts})
    manifest = {
        "schema_version": 1,
        "artifact_type": "ferrum_v084_homebrew_publish_manifest",
        "status": status,
        "version": VERSION,
        "source": copy.deepcopy(promotion["source"]),
        "release": copy.deepcopy(promotion["release"]),
        "assets": assets,
        "tap": {
            "repository": TAP_REPOSITORY,
            "branch": TAP_BRANCH,
            "checkout": str(tap),
            "initial": state,
            "published_commit": commit_sha,
        },
        "proposal": {
            "formula_sha256": proposal_hashes,
            "diff": artifact_ref(out / "formula.diff", out),
            "formulae": {relative: artifact_ref(proposal_root / relative, out) for relative in FORMULA_PATHS},
        },
        "publication": {
            "requested": bool(args.publish),
            "mutation_performed": mutation_performed,
            "allowed_paths": list(FORMULA_PATHS),
            "force_push": False,
            "commands": artifact_ref(receipt_path, out),
        },
        "started_at": iso(started),
        "finished_at": iso(finished),
        "generated_at": iso(),
        "artifact_dir": str(out),
        "pass_line": pass_line,
    }
    manifest_path = out / "homebrew.manifest.json"
    write_json(manifest_path, manifest)
    assert_no_secrets(out)
    print(pass_line)
    return manifest_path


PAIR_SPECS = {
    "unit": ("unit", "unit.gate.json", "G0 SOURCE unit PASS: "),
    "metal": ("metal", "metal.gate.json", "G0 SOURCE metal PASS: "),
    "cuda_full": ("cuda-full", "g0_cuda4090_full.gate.json", "G0 SOURCE g0_cuda4090_full PASS: "),
    "cuda_llama_dense": ("cuda-llama-dense", "g0_cuda4090_llama_dense.gate.json", "G0 SOURCE g0_cuda4090_llama_dense PASS: "),
    "metal_tarball": ("metal-tarball", "gate.json", "METAL TARBALL GATE PASS: "),
    "cuda_tarball": ("cuda-tarball", "gate.json", "CUDA TARBALL GATE PASS: "),
    "homebrew_metal": ("homebrew-metal", "gate.json", "HOMEBREW METAL GATE PASS: "),
    "homebrew_cuda_fetch": ("homebrew-cuda-fetch", "gate.json", "HOMEBREW CUDA FETCH GATE PASS: "),
}


def input_pair(args: argparse.Namespace, key: str) -> tuple[Path, Path]:
    prefix = {
        "unit": "unit",
        "metal": "metal_source",
        "cuda_full": "cuda_full",
        "cuda_llama_dense": "cuda_llama_dense",
        "metal_tarball": "metal_tarball",
        "cuda_tarball": "cuda_tarball",
        "homebrew_metal": "homebrew_metal",
        "homebrew_cuda_fetch": "homebrew_cuda_fetch",
    }[key]
    return safe_regular(getattr(args, f"{prefix}_outer"), f"{key} outer"), safe_regular(getattr(args, f"{prefix}_child"), f"{key} child")


def copy_gate_pair(outer: Path, child: Path, destination: Path) -> tuple[Path, Path]:
    require(outer.parent == child.parent, "gate outer/child must share an artifact directory")
    copied_outer = copy_portable_closure(outer, destination)
    copied_child = destination / child.name
    if not copied_child.exists():
        shutil.copyfile(child, copied_child, follow_symlinks=False)
    require(copied_child.is_file() and not copied_child.is_symlink() and sha256_file(copied_child) == sha256_file(child), "copied gate child differs")
    return copied_outer, copied_child


def run_assemble_final(args: argparse.Namespace) -> Path:
    started = now()
    out = args.out.expanduser().resolve()
    require(
        out == OFFICIAL_FINAL_OUT,
        f"final output must be the canonical release artifact directory: {OFFICIAL_FINAL_OUT}",
    )
    out = ensure_new_or_directory(out)
    managed = out / "final-evidence"
    require(
        not managed.exists(),
        f"refusing to overwrite managed final evidence: {managed}",
    )
    prerelease_input = safe_regular(args.prerelease_manifest, "final prerelease manifest")
    promotion_input = safe_regular(args.promotion_manifest, "final promotion manifest")
    try:
        prerelease = goal_gate.validate_prerelease_manifest(prerelease_input)
        promotion = goal_gate.validate_promotion_manifest(promotion_input)
    except Exception as error:
        raise PublishError(f"final prerelease/promotion validation failed: {error}") from error
    require(prerelease["source"] == promotion["source"] and prerelease["release"] == promotion["release"], "final prerelease/promotion identity differs")
    source_sha = prerelease["source"]["git_sha"]

    portable = managed / "portable"
    copied_prerelease = copy_portable_closure(prerelease_input, portable / "prerelease")
    copied_promotion = copy_portable_closure(promotion_input, portable / "promotion")
    copied_workflow = copy_portable_closure(safe_regular(args.workflow_policy_manifest, "workflow policy manifest"), portable / "workflow-policy")
    copied_native = copy_portable_closure(safe_regular(args.native_set_manifest, "native set manifest"), portable / "native-set")
    crates_input = safe_regular(args.crates_manifest, "crates.io publish manifest")
    copy_tree_strict(crates_input.parent, portable / "crates-io")
    copied_crates = portable / "crates-io" / crates_input.name

    summary_input = safe_regular(args.g0_summary, "G0 release summary")
    summary = read_json(summary_input, "G0 release summary")
    require(summary.get("release_candidate_sha") == source_sha and summary.get("status") == "pass", "G0 summary candidate/status differs")
    gate_rows = summary.get("gates")
    require(isinstance(gate_rows, list) and len(gate_rows) == len(PAIR_SPECS), "G0 summary must name exactly the eight required 0.8.4 gates")
    summary_root = summary_input.parent.resolve()
    summary_paths: dict[Path, str] = {}
    for raw in gate_rows:
        require(isinstance(raw, str) and raw, "G0 summary gate path is empty")
        pure = PurePosixPath(raw)
        require(not pure.is_absolute() and "\\" not in raw and ".." not in pure.parts, f"unsafe G0 summary gate path: {raw}")
        resolved = summary_root.joinpath(*pure.parts).resolve()
        require(resolved.is_relative_to(summary_root) and resolved.is_file() and not resolved.is_symlink(), f"G0 summary gate is missing: {raw}")
        require(resolved not in summary_paths, f"duplicate G0 summary gate: {raw}")
        summary_paths[resolved] = raw

    g0_root = managed / "g0"
    g0_root.mkdir()
    evidence_pairs: dict[str, dict[str, Any]] = {}
    for key, (lane, child_name, child_prefix) in PAIR_SPECS.items():
        outer, child = input_pair(args, key)
        require(outer in summary_paths, f"G0 summary does not bind supplied {key} outer manifest")
        try:
            goal_gate.validate_outer_child_gate(
                {"outer": goal_gate.make_ref(outer, outer.parent), "child": goal_gate.make_ref(child, outer.parent)},
                root=outer.parent,
                g0_gate_paths=None,
                lane=lane,
                child_filename=child_name,
                child_pass_prefix=child_prefix,
                source_sha=source_sha,
                where=f"assemble-final {key}",
            )
        except Exception as error:
            raise PublishError(f"{key} outer/child validation failed: {error}") from error
        recorded = PurePosixPath(summary_paths[outer])
        destination = g0_root.joinpath(*recorded.parent.parts)
        copied_outer, copied_child = copy_gate_pair(outer, child, destination)
        evidence_pairs[key] = {
            "outer": artifact_ref(copied_outer, managed),
            "child": artifact_ref(copied_child, managed),
        }

    copied_summary = g0_root / summary_input.name
    shutil.copyfile(summary_input, copied_summary, follow_symlinks=False)
    require(sha256_file(copied_summary) == sha256_file(summary_input), "copied G0 summary differs")
    try:
        goal_gate.validate_g0_summary(summary, path=copied_summary, source_sha=source_sha)
    except Exception as error:
        raise PublishError(f"copied G0 summary failed validation: {error}") from error

    final_evidence = {
        "prerelease_manifest": artifact_ref(copied_prerelease, managed),
        "promotion_manifest": artifact_ref(copied_promotion, managed),
        "metal_tarball": evidence_pairs["metal_tarball"],
        "cuda_tarball": evidence_pairs["cuda_tarball"],
        "crates_io": artifact_ref(copied_crates, managed),
        "homebrew_metal": evidence_pairs["homebrew_metal"],
        "homebrew_cuda_fetch": evidence_pairs["homebrew_cuda_fetch"],
        "workflow_policy": artifact_ref(copied_workflow, managed),
        "native_operator_set": artifact_ref(copied_native, managed),
        "g0_summary": artifact_ref(copied_summary, managed),
        "g0_gates": {
            "unit": evidence_pairs["unit"],
            "metal": evidence_pairs["metal"],
            "cuda_full": evidence_pairs["cuda_full"],
            "cuda_llama_dense": evidence_pairs["cuda_llama_dense"],
        },
    }
    final = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ferrum_v084_final_manifest",
        "status": "pass",
        "version": VERSION,
        "source": copy.deepcopy(prerelease["source"]),
        "release": copy.deepcopy(prerelease["release"]),
        "evidence": final_evidence,
        "artifact_dir": goal_gate.FINAL_ARTIFACT_DIR,
        "pass_line": f"FERRUM 0.8.4 RELEASE PASS: {goal_gate.FINAL_ARTIFACT_DIR}",
    }
    candidate = managed / ".final.manifest.candidate.json"
    canonical = managed / "final.manifest.json"
    write_json(candidate, final)
    try:
        validated = install_validated_manifest(candidate, canonical, goal_gate.validate_final_manifest)
    except Exception as error:
        candidate.unlink(missing_ok=True)
        raise PublishError(f"assembled final manifest failed authoritative validation: {error}") from error
    finished = now()
    receipt = {
        "schema_version": 1,
        "artifact_type": "ferrum_v084_final_collector_receipt",
        "status": "pass",
        "version": VERSION,
        "source": copy.deepcopy(prerelease["source"]),
        "release": copy.deepcopy(prerelease["release"]),
        "started_at": iso(started),
        "finished_at": iso(finished),
        "generated_at": iso(),
        "manifest": artifact_ref(canonical, managed),
        "input_sha256": {
            "prerelease": sha256_file(prerelease_input),
            "promotion": sha256_file(promotion_input),
            "crates_io": sha256_file(crates_input),
            "workflow_policy": sha256_file(safe_regular(args.workflow_policy_manifest, "workflow policy manifest")),
            "native_operator_set": sha256_file(safe_regular(args.native_set_manifest, "native set manifest")),
            "g0_summary": sha256_file(summary_input),
        },
        "progress_signal": "portable evidence closure and authoritative final validator PASS",
        "deadline_seconds": 1800,
    }
    write_json(managed / "final.collector.json", receipt)
    assert_no_secrets(managed)
    require(validated["pass_line"] == final["pass_line"], "authoritative final PASS line differs")
    print(final["pass_line"])
    return canonical


def expect_failure(call: Callable[[], Any], contains: str) -> None:
    try:
        call()
    except Exception as error:
        require(contains.lower() in str(error).lower(), f"negative self-test failed for wrong reason: {error}")
    else:
        raise PublishError(f"negative self-test unexpectedly passed: {contains}")


class FakeGitHub:
    def __init__(
        self,
        routes: dict[str, dict[str, Any]],
        *,
        crash_after_patch: bool = False,
        crash_without_visible_change: bool = False,
    ) -> None:
        self.routes = copy.deepcopy(routes)
        self.current = copy.deepcopy(routes["before"])
        self.patch_count = 0
        self.crash_after_patch = crash_after_patch
        self.crash_without_visible_change = crash_without_visible_change

    def __call__(self, method: str, endpoint: str, body: dict[str, Any] | None, receipt_path: Path) -> dict[str, Any]:
        started = iso()
        if method == "PATCH":
            validate_patch_body(body)
            self.patch_count += 1
            if not (self.crash_after_patch and self.crash_without_visible_change):
                self.current = copy.deepcopy(self.routes["after"])
            result = self.current
        elif endpoint.endswith("/releases/latest"):
            result = self.routes["after"]
        elif "/git/ref/tags/" in endpoint:
            result = self.routes["tag_ref"]
        elif "/git/tags/" in endpoint:
            result = self.routes["tag"]
        else:
            result = self.current
        write_json(receipt_path, {
            "schema_version": 1, "kind": "offline-fake-github", "method": method,
            "endpoint": endpoint, "body": body, "started_at": started,
            "finished_at": iso(), "deadline_seconds": 60,
            "progress_signal": "offline fixture response", "returncode": 0,
            "credentials_in_argv": False,
        })
        if method == "PATCH" and self.crash_after_patch:
            raise PublishError("simulated crash after remote PATCH before response persistence")
        return copy.deepcopy(result)


def git_fixture(argv: list[str], cwd: Path) -> str:
    process = subprocess.run(["git", *argv], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    require(process.returncode == 0, f"self-test git failed: {argv}: {process.stderr}")
    return process.stdout.strip()


def build_tap_fixture(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=False)
    bare = root / "tap.git"
    seed = root / "seed"
    checkout = root / "checkout"
    git_fixture(["init", "--bare", str(bare)], root)
    git_fixture(["init", "-b", "main", str(seed)], root)
    git_fixture(["config", "user.name", "Ferrum Selftest"], seed)
    git_fixture(["config", "user.email", "selftest@example.invalid"], seed)
    (seed / "Formula").mkdir()
    atomic_write(seed / "Formula/ferrum.rb", b'class Ferrum < Formula\n  version "0.8.3"\nend\n')
    atomic_write(seed / "Formula/ferrum-cuda.rb", b'class FerrumCuda < Formula\n  version "0.8.3"\nend\n')
    git_fixture(["add", "Formula"], seed)
    git_fixture(["commit", "-m", "seed"], seed)
    git_fixture(["remote", "add", "origin", str(bare)], seed)
    git_fixture(["push", "-u", "origin", "main"], seed)
    git_fixture(["symbolic-ref", "HEAD", "refs/heads/main"], bare)
    git_fixture(["clone", str(bare), str(checkout)], root)
    git_fixture(["config", "user.name", "Ferrum Selftest"], checkout)
    git_fixture(["config", "user.email", "selftest@example.invalid"], checkout)
    return checkout, bare


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-v084-release-publish-") as temporary:
        root = Path(temporary)
        prerelease_path, fixture_promotion, final_path = goal_gate.build_selftest_fixture(root / "goal-fixture")
        promotion_raw = read_json(fixture_promotion, "fixture promotion")
        routes = {
            "before": read_json(resolve_ref(promotion_raw["evidence"]["release_before"], fixture_promotion.parent, "fixture before"), "fixture before"),
            "after": read_json(resolve_ref(promotion_raw["evidence"]["release_after"], fixture_promotion.parent, "fixture after"), "fixture after"),
            "tag_ref": read_json(resolve_ref(promotion_raw["evidence"]["tag_ref_snapshot"], fixture_promotion.parent, "fixture tag ref"), "fixture tag ref"),
            "tag": read_json(resolve_ref(promotion_raw["evidence"]["tag_snapshot"], fixture_promotion.parent, "fixture tag"), "fixture tag"),
        }

        crash_fake = FakeGitHub(routes, crash_after_patch=True)
        crash_out = root / "promotion-crash-resume"
        crash_args = argparse.Namespace(
            prerelease_manifest=prerelease_path,
            out=crash_out,
            execute=True,
        )
        expect_failure(
            lambda: run_promote(crash_args, transport=crash_fake),
            "simulated crash after remote PATCH",
        )
        require(crash_fake.patch_count == 1, "crash fixture did not perform exactly one remote PATCH")
        require((crash_out / "evidence/github/release-before.json").is_file(), "crash fixture did not persist release-before")
        require(not (crash_out / "evidence/github/release-after.json").exists(), "crash fixture unexpectedly persisted release-after")
        crash_fake.crash_after_patch = False
        recovered = run_promote(crash_args, transport=crash_fake)
        require(recovered is not None and crash_fake.patch_count == 1, "crash recovery repeated GitHub PATCH")
        recovery_receipt = read_json(crash_out / "promotion.collector.json", "crash recovery receipt")
        require(
            recovery_receipt.get("resume_recovered_patch_response") is True
            and recovery_receipt.get("resume_without_patch") is True,
            "crash recovery receipt does not identify recovered PATCH response",
        )

        # An ambiguous request outcome may still read back as prerelease.  A
        # durable attempt exists, so the retry must fail closed without ever
        # issuing a second PATCH.
        ambiguous_fake = FakeGitHub(
            routes,
            crash_after_patch=True,
            crash_without_visible_change=True,
        )
        ambiguous_out = root / "promotion-ambiguous-still-prerelease"
        ambiguous_args = argparse.Namespace(
            prerelease_manifest=prerelease_path,
            out=ambiguous_out,
            execute=True,
        )
        expect_failure(
            lambda: run_promote(ambiguous_args, transport=ambiguous_fake),
            "simulated crash after remote PATCH",
        )
        require(
            ambiguous_fake.patch_count == 1,
            "ambiguous fixture did not perform exactly one PATCH attempt",
        )
        ambiguous_fake.crash_after_patch = False
        expect_failure(
            lambda: run_promote(ambiguous_args, transport=ambiguous_fake),
            "refusing to issue a second PATCH",
        )
        require(
            ambiguous_fake.patch_count == 1,
            "ambiguous promotion recovery repeated GitHub PATCH",
        )

        missing_before = FakeGitHub(routes)
        missing_before.current = copy.deepcopy(routes["after"])
        expect_failure(
            lambda: run_promote(
                argparse.Namespace(
                    prerelease_manifest=prerelease_path,
                    out=root / "promotion-missing-before",
                    execute=True,
                ),
                transport=missing_before,
            ),
            "release-before is absent",
        )
        require(missing_before.patch_count == 0, "missing-before recovery attempted PATCH")

        mismatch_fake = FakeGitHub(routes, crash_after_patch=True)
        mismatch_out = root / "promotion-crash-mismatch"
        mismatch_args = argparse.Namespace(
            prerelease_manifest=prerelease_path,
            out=mismatch_out,
            execute=True,
        )
        expect_failure(
            lambda: run_promote(mismatch_args, transport=mismatch_fake),
            "simulated crash after remote PATCH",
        )
        mismatch_fake.crash_after_patch = False
        mismatch_fake.current["assets"][0]["size"] += 1
        expect_failure(
            lambda: run_promote(mismatch_args, transport=mismatch_fake),
            "asset",
        )
        require(mismatch_fake.patch_count == 1, "identity-mismatch recovery repeated PATCH")

        fake = FakeGitHub(routes)
        promotion_out = root / "promotion-out"
        promote_args = argparse.Namespace(prerelease_manifest=prerelease_path, out=promotion_out, execute=True)
        promoted = run_promote(promote_args, transport=fake)
        require(promoted is not None and fake.patch_count == 1, "promotion self-test did not perform exactly one PATCH")
        portable_promotion = copy_portable_closure(
            promoted, root / "promotion-portable-closure"
        )
        goal_gate.validate_promotion_manifest(portable_promotion)
        portable_raw = read_json(portable_promotion, "portable promotion")
        portable_mutation = resolve_ref(
            portable_raw["evidence"]["mutation_receipt"],
            portable_promotion.parent,
            "portable promotion mutation receipt",
        )
        require(
            portable_mutation.is_file(),
            "portable promotion closure omitted the mutation receipt",
        )
        # Exercise resume after the remote is already final.  Remove only the
        # canonical manifest; saved before/after identity must prevent PATCH.
        promoted.unlink()
        promoted = run_promote(promote_args, transport=fake)
        require(promoted is not None and fake.patch_count == 1, "promotion resume repeated PATCH")
        expect_failure(lambda: validate_patch_body({"prerelease": False, "name": "expanded"}), "only prerelease:false")

        mutable_routes = copy.deepcopy(routes)
        mutable_routes["after"]["name"] = "display name changed concurrently"
        mutable_routes["after"]["assets"][0]["download_count"] = 99
        mutable_fake = FakeGitHub(mutable_routes)
        mutable_promoted = run_promote(
            argparse.Namespace(
                prerelease_manifest=prerelease_path,
                out=root / "promotion-mutable-service-fields",
                execute=True,
            ),
            transport=mutable_fake,
        )
        require(
            mutable_promoted is not None and mutable_fake.patch_count == 1,
            "mutable GitHub display/counter fields blocked target identity promotion",
        )

        changed_routes = copy.deepcopy(routes)
        changed_routes["after"]["assets"][0]["size"] += 1
        changed_fake = FakeGitHub(changed_routes)
        expect_failure(
            lambda: run_promote(argparse.Namespace(prerelease_manifest=prerelease_path, out=root / "promotion-asset-change", execute=True), transport=changed_fake),
            "target identity changed",
        )

        checkout, bare = build_tap_fixture(root / "tap-fixture")
        diverged_checkout = root / "tap-diverged"
        git_fixture(["clone", str(bare), str(diverged_checkout)], root)
        git_fixture(["config", "user.name", "Ferrum Selftest"], diverged_checkout)
        git_fixture(["config", "user.email", "selftest@example.invalid"], diverged_checkout)
        _, fixture_assets = promotion_assets(promoted)
        diverged_formulae = render_formulae(fixture_assets)
        for relative, text in diverged_formulae.items():
            atomic_write(diverged_checkout / relative, text.encode("utf-8"))
        git_fixture(["add", *FORMULA_PATHS], diverged_checkout)
        git_fixture(["commit", "-m", f"ferrum {VERSION}"], diverged_checkout)
        expect_failure(
            lambda: run_homebrew(
                argparse.Namespace(
                    promotion_manifest=promoted,
                    tap=diverged_checkout,
                    out=root / "homebrew-diverged-dry",
                    publish=False,
                ),
                allow_local_origin=True,
            ),
            "HEAD == origin/main == remote main",
        )
        homebrew_out = root / "homebrew-dry"
        run_homebrew(argparse.Namespace(promotion_manifest=promoted, tap=checkout, out=homebrew_out, publish=False), allow_local_origin=True)
        require(git_fixture(["status", "--porcelain=v1", "--untracked-files=all"], checkout) == "", "Homebrew dry run changed tap")
        atomic_write(checkout / "unexpected.txt", b"dirty\n")
        expect_failure(
            lambda: run_homebrew(argparse.Namespace(promotion_manifest=promoted, tap=checkout, out=root / "homebrew-dirty", publish=False), allow_local_origin=True),
            "dirty",
        )
        (checkout / "unexpected.txt").unlink()
        run_homebrew(argparse.Namespace(promotion_manifest=promoted, tap=checkout, out=root / "homebrew-publish", publish=True), allow_local_origin=True)

        # Remote drift is detected against a clean checkout whose origin/main
        # tracking ref has not silently been rewritten by the collector.
        drift_checkout = root / "drift-checkout"
        git_fixture(["clone", str(bare), str(drift_checkout)], root)
        second = root / "second"
        git_fixture(["clone", str(bare), str(second)], root)
        git_fixture(["config", "user.name", "Ferrum Selftest"], second)
        git_fixture(["config", "user.email", "selftest@example.invalid"], second)
        atomic_write(second / "README.md", b"drift\n")
        git_fixture(["add", "README.md"], second)
        git_fixture(["commit", "-m", "remote drift"], second)
        git_fixture(["push", "origin", "main"], second)
        expect_failure(
            lambda: run_homebrew(argparse.Namespace(promotion_manifest=promoted, tap=drift_checkout, out=root / "homebrew-drift", publish=False), allow_local_origin=True),
            "dry-run tap baseline",
        )

        final_raw = read_json(final_path, "fixture final")
        evidence = final_raw["evidence"]
        kwargs: dict[str, Any] = {
            "prerelease_manifest": resolve_ref(evidence["prerelease_manifest"], final_path.parent, "final prerelease"),
            "promotion_manifest": resolve_ref(evidence["promotion_manifest"], final_path.parent, "final promotion"),
            "crates_manifest": resolve_ref(evidence["crates_io"], final_path.parent, "final crates"),
            "workflow_policy_manifest": resolve_ref(evidence["workflow_policy"], final_path.parent, "final workflow"),
            "native_set_manifest": resolve_ref(evidence["native_operator_set"], final_path.parent, "final native"),
            "g0_summary": resolve_ref(evidence["g0_summary"], final_path.parent, "final summary"),
            "out": root / "assembled-final",
        }
        mapping = {
            "unit": ("g0_gates", "unit", "unit"),
            "metal_source": ("g0_gates", "metal", "metal"),
            "cuda_full": ("g0_gates", "cuda_full", "cuda_full"),
            "cuda_llama_dense": ("g0_gates", "cuda_llama_dense", "cuda_llama_dense"),
            "metal_tarball": (None, "metal_tarball", "metal_tarball"),
            "cuda_tarball": (None, "cuda_tarball", "cuda_tarball"),
            "homebrew_metal": (None, "homebrew_metal", "homebrew_metal"),
            "homebrew_cuda_fetch": (None, "homebrew_cuda_fetch", "homebrew_cuda_fetch"),
        }
        for prefix, (group, key, _unused) in mapping.items():
            pair = evidence[group][key] if group else evidence[key]
            kwargs[f"{prefix}_outer"] = resolve_ref(pair["outer"], final_path.parent, f"{prefix} outer")
            kwargs[f"{prefix}_child"] = resolve_ref(pair["child"], final_path.parent, f"{prefix} child")
        expect_failure(
            lambda: run_assemble_final(argparse.Namespace(**kwargs)),
            "canonical release artifact directory",
        )
        require(
            not kwargs["out"].exists(),
            "noncanonical final assembly created a fixture artifact directory",
        )
        # The goal fixture remains the internal deep-validator receipt.  The
        # publisher self-test must never emit a production RELEASE PASS from a
        # temporary directory.
        goal_gate.validate_final_manifest(final_path)
        assembled_raw = read_json(final_path, "final validator fixture")
        child = resolve_ref(
            assembled_raw["evidence"]["metal_tarball"]["child"],
            final_path.parent,
            "final validator fixture child",
        )
        atomic_write(child, child.read_bytes() + b"\n")
        expect_failure(lambda: goal_gate.validate_final_manifest(final_path), "changed")


def add_pair_arguments(parser: argparse.ArgumentParser) -> None:
    for prefix in (
        "unit", "metal-source", "cuda-full", "cuda-llama-dense",
        "metal-tarball", "cuda-tarball", "homebrew-metal", "homebrew-cuda-fetch",
    ):
        parser.add_argument(f"--{prefix}-outer", required=True, type=Path)
        parser.add_argument(f"--{prefix}-child", required=True, type=Path)


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run hermetic fake-GitHub/bare-tap tests")
    subparsers = parser.add_subparsers(dest="mode")

    promote = subparsers.add_parser("promote", help="dry-run or explicitly promote the validated GitHub prerelease")
    promote.add_argument("--prerelease-manifest", required=True, type=Path)
    promote.add_argument("--out", required=True, type=Path)
    promote.add_argument("--execute", action="store_true", help="perform the sole allowed GitHub PATCH")

    homebrew = subparsers.add_parser("homebrew", help="dry-run or explicitly publish the 0.8.4 tap formulae")
    homebrew.add_argument("--promotion-manifest", required=True, type=Path)
    homebrew.add_argument("--tap", required=True, type=Path)
    homebrew.add_argument("--out", required=True, type=Path)
    homebrew.add_argument("--publish", action="store_true", help="commit and push exact formula bytes")

    final = subparsers.add_parser("assemble-final", help="assemble the portable final 0.8.4 evidence closure")
    final.add_argument("--prerelease-manifest", required=True, type=Path)
    final.add_argument("--promotion-manifest", required=True, type=Path)
    final.add_argument("--crates-manifest", required=True, type=Path)
    final.add_argument("--workflow-policy-manifest", required=True, type=Path)
    final.add_argument("--native-set-manifest", required=True, type=Path)
    final.add_argument("--g0-summary", required=True, type=Path)
    final.add_argument("--out", required=True, type=Path)
    add_pair_arguments(final)

    args = parser.parse_args()
    if args.self_test:
        require(args.mode is None, "--self-test cannot be combined with a subcommand")
    else:
        require(args.mode is not None, "choose a subcommand or --self-test")
    return args


def main() -> int:
    try:
        args = build_parser()
        if args.self_test:
            self_test()
            print(SELFTEST_PASS_LINE)
        elif args.mode == "promote":
            run_promote(args)
        elif args.mode == "homebrew":
            run_homebrew(args)
        else:
            run_assemble_final(args)
        return 0
    except (PublishError, goal_gate.ValidationError, OSError, subprocess.SubprocessError) as error:
        print(f"FERRUM 0.8.4 RELEASE PUBLISH FAIL: {sanitize(str(error))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
