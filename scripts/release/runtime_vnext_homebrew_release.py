#!/usr/bin/env python3
"""Safely prepare or explicitly publish Ferrum v0.8.0 Homebrew formulae.

The default mode is a non-mutating dry run.  It validates the canonical
Runtime vNext release DAG and the exact v0.7.7 tap base, then writes proposed
formulae and a manifest under ``--out``.  Only ``--publish`` may modify the tap,
create a commit, or push it.  The publish path is idempotent across a commit or
push boundary and never force-pushes.
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


VERSION = "0.8.0"
TAG = "v0.8.0"
RC_TAG = "v0.8.0-rc.1"
SCHEMA_VERSION = 1
LANE = "runtime-vnext-homebrew-release"
TAP_REPOSITORY = "sizzlecar/homebrew-ferrum"
TAP_BRANCH = "main"
TAP_BASE_HEAD = "f1201ba97fd125fd66762afa1cf25183c7d81bdb"
TAP_ORIGINS = {
    "https://github.com/sizzlecar/homebrew-ferrum.git",
    "git@github.com:sizzlecar/homebrew-ferrum.git",
}
BASE_FORMULA_BLOBS = {
    "Formula/ferrum.rb": "48ccc0c87516d7307cece78f97ba68d7f84de826",
    "Formula/ferrum-cuda.rb": "3a51718ae40a68651acf93386b1f87aec3a33548",
}
FORMULA_PATHS = tuple(BASE_FORMULA_BLOBS)
ASSET_NAMES = {
    "cpu": "ferrum-linux-x86_64.tar.gz",
    "metal": "ferrum-macos-aarch64.tar.gz",
    "cuda": "ferrum-linux-x86_64-cuda-sm89.tar.gz",
}
RELEASE_GATE_LANES = {
    "g10a": "vnext-g10a",
    "g08_rc": "vnext-g08-rc",
    "g09_rc": "vnext-g09-rc",
    "published_assets": "runtime-vnext-published-assets",
}
DRY_RUN_PASS_PREFIX = "FERRUM HOMEBREW V0.8.0 DRY RUN PASS"
PUBLISH_PASS_PREFIX = "FERRUM HOMEBREW V0.8.0 PUBLISH PASS"
SELFTEST_PASS_LINE = "FERRUM HOMEBREW V0.8.0 SELFTEST PASS"
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOKEN_ENV_NAMES = (
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "HOMEBREW_GITHUB_API_TOKEN",
)


BASE_METAL_FORMULA = '''class Ferrum < Formula
  desc "Production-grade LLM inference in Rust for Apple Silicon and Linux CPU"
  homepage "https://github.com/sizzlecar/ferrum-infer-rs"
  version "0.7.7"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v0.7.7/ferrum-macos-aarch64.tar.gz"
      sha256 "e685342eb9d3050c1d4117ef71ecd20418f91bd731c8f304269b4ec36936247b"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v0.7.7/ferrum-linux-x86_64.tar.gz"
      sha256 "b98e22701a3a5b6d79ce4489ba1fcc1201a2925f24b9c87e35bc1678fb459715"
    end
  end

  conflicts_with "ferrum-cuda", because: "both install the ferrum binary"

  def install
    bin.install "ferrum"
    doc.install "README.md"
  end

  test do
    assert_match "ferrum #{version}", shell_output("#{bin}/ferrum --version")
    assert_match "serve", shell_output("#{bin}/ferrum serve --help")
  end
end
'''

BASE_CUDA_FORMULA = '''class FerrumCuda < Formula
  desc "Production-grade LLM inference in Rust with NVIDIA CUDA sm89 support"
  homepage "https://github.com/sizzlecar/ferrum-infer-rs"
  url "https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v0.7.7/ferrum-linux-x86_64-cuda-sm89.tar.gz"
  version "0.7.7"
  sha256 "6397de942d1c767383982186642da3990df49ea1cd6f604434a6cc9d6f7f912b"
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

BASE_FORMULAE = {
    "Formula/ferrum.rb": BASE_METAL_FORMULA,
    "Formula/ferrum-cuda.rb": BASE_CUDA_FORMULA,
}


class HomebrewReleaseError(RuntimeError):
    """The Homebrew release contract was not satisfied."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise HomebrewReleaseError(message)


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
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
            "ascii"
        ),
    )


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HomebrewReleaseError(f"cannot read {label} {path}: {error}") from error
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


def sanitize_text(value: str) -> str:
    patterns = (
        re.compile(
            r"(?i)((?:GITHUB|GH|HOMEBREW_GITHUB_API)_TOKEN\s*=\s*)\S+"
        ),
        re.compile(r"(?i)(authorization\s*:\s*bearer\s+)\S+"),
        re.compile(r"(?i)(https://[^/@\s]+:)[^/@\s]+(@github\.com)"),
    )
    result = value
    for pattern in patterns:
        result = pattern.sub(r"\1<redacted>\2" if pattern.groups == 2 else r"\1<redacted>", result)
    return result


def secret_values() -> tuple[str, ...]:
    return tuple(
        value
        for name in TOKEN_ENV_NAMES
        if len((value := os.environ.get(name, ""))) >= 8
    )


def assert_no_secrets(root: Path, secrets: Iterable[str]) -> None:
    values = tuple(value.encode("utf-8") for value in secrets if value)
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            payload = path.read_bytes()
            require(
                not any(secret in payload for secret in values),
                f"credential value was persisted in {path}",
            )


def artifact_ref(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    require(
        resolved.is_file() and not resolved.is_symlink(),
        f"artifact is not a regular file: {resolved}",
    )
    if root is None:
        rendered_path = str(resolved)
    else:
        base = root.resolve()
        require(resolved.is_relative_to(base), f"artifact escapes root: {resolved}")
        rendered_path = resolved.relative_to(base).as_posix()
    return {
        "path": rendered_path,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def validate_artifact_ref(
    value: Any, *, root: Path | None, label: str, nonempty: bool = True
) -> Path:
    row = exact_fields(value, {"path", "sha256", "size_bytes"}, label)
    raw_path = PurePosixPath(str(row["path"]))
    if root is None:
        path = Path(str(row["path"])).expanduser()
        require(path.is_absolute(), f"{label}.path must be absolute")
    else:
        require(
            not raw_path.is_absolute() and ".." not in raw_path.parts,
            f"{label}.path is unsafe",
        )
        path = root / raw_path.as_posix()
    path = path.resolve()
    require(path.is_file() and not path.is_symlink(), f"{label} is missing")
    require(
        type(row["size_bytes"]) is int
        and row["size_bytes"] == path.stat().st_size
        and row["size_bytes"] >= int(nonempty),
        f"{label} size differs",
    )
    require(
        isinstance(row["sha256"], str)
        and SHA256_RE.fullmatch(row["sha256"]) is not None
        and sha256_file(path) == row["sha256"],
        f"{label} SHA256 differs",
    )
    return path


def goal_manifest_path(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "manifest.json"
    require(candidate.is_file(), f"goal manifest is missing: {candidate}")
    return candidate


def verify_goal(path: Path, *, lane: str) -> dict[str, Any]:
    manifest_path = goal_manifest_path(path)
    try:
        import runtime_vnext_goal_gate as goal_gate
    except ImportError as error:
        raise HomebrewReleaseError(
            f"cannot import authoritative goal validator: {error}"
        ) from error
    try:
        verified = goal_gate.verify_goal_manifest(
            manifest_path, expected_lane=lane
        )
    except goal_gate.GoalGateError as error:
        raise HomebrewReleaseError(
            f"{lane} authoritative validation failed: {sanitize_text(str(error))}"
        ) from error
    require(
        isinstance(verified, dict)
        and Path(verified.get("path", "")).resolve() == manifest_path,
        f"{lane} validator returned a different manifest",
    )
    verified = copy.deepcopy(verified)
    verified["ref"] = artifact_ref(manifest_path)
    return verified


def ref_sha(value: Any, label: str) -> str:
    require(isinstance(value, dict), f"{label} ref is missing")
    digest = value.get("sha256")
    require(
        isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
        f"{label} ref SHA256 differs",
    )
    return digest


def bind_verified_gate_bundle(gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    require(set(gates) == set(RELEASE_GATE_LANES), "release gate denominator differs")
    sources = [gate.get("source") for gate in gates.values()]
    require(all(source == sources[0] for source in sources), "release gate sources differ")
    source = exact_fields(
        sources[0], {"git_sha", "git_tree_sha", "dirty"}, "release candidate"
    )
    require(
        SHA1_RE.fullmatch(str(source["git_sha"])) is not None
        and SHA1_RE.fullmatch(str(source["git_tree_sha"])) is not None
        and source["dirty"] is False,
        "release candidate identity differs",
    )
    g10a = gates["g10a"]
    require(
        g10a["manifest"].get("release_candidate_tag") == RC_TAG,
        "G10A release-candidate tag differs",
    )
    refs = {key: gate["ref"] for key, gate in gates.items()}
    expected_links = {
        "g08_rc": {"g10a": ref_sha(refs["g10a"], "G10A")},
        "g09_rc": {
            "g10a": ref_sha(refs["g10a"], "G10A"),
            "g08_rc": ref_sha(refs["g08_rc"], "G08-RC"),
        },
        "published_assets": {
            "g10a": ref_sha(refs["g10a"], "G10A"),
            "g08_rc": ref_sha(refs["g08_rc"], "G08-RC"),
            "g09_rc": ref_sha(refs["g09_rc"], "G09-RC"),
        },
    }
    for gate_key, links in expected_links.items():
        inputs = gates[gate_key]["manifest"].get("inputs")
        require(isinstance(inputs, dict), f"{gate_key} inputs are missing")
        for input_key, expected_sha in links.items():
            require(
                ref_sha(inputs.get(input_key), f"{gate_key}.{input_key}")
                == expected_sha,
                f"{gate_key} does not bind exact {input_key}",
            )
    published = gates["published_assets"]["manifest"]
    release = exact_fields(
        published.get("release"),
        {
            "id",
            "html_url",
            "tag_name",
            "tag_sha",
            "release_candidate_tag",
            "draft",
            "prerelease",
            "published_at",
            "asset_set_sha256",
            "asset_count",
        },
        "published release",
    )
    require(
        release["tag_name"] == TAG
        and release["tag_sha"] == source["git_sha"]
        and release["release_candidate_tag"] == RC_TAG
        and release["draft"] is False
        and release["prerelease"] is True,
        "published release identity/state differs",
    )
    assets = published.get("assets")
    require(isinstance(assets, dict) and set(assets) == set(ASSET_NAMES), "asset set differs")
    normalized_assets: dict[str, Any] = {}
    for backend, name in ASSET_NAMES.items():
        row = assets[backend]
        require(isinstance(row, dict), f"published {backend} asset is missing")
        checksum = row.get("tarball_sha256")
        require(
            row.get("name") == name
            and isinstance(checksum, str)
            and SHA256_RE.fullmatch(checksum) is not None
            and row.get("digest") == f"sha256:{checksum}"
            and type(row.get("size")) is int
            and row["size"] > 0,
            f"published {backend} asset identity differs",
        )
        normalized_assets[backend] = {
            "name": name,
            "sha256": checksum,
            "size_bytes": row["size"],
            "url": (
                "https://github.com/sizzlecar/ferrum-infer-rs/releases/"
                f"download/{TAG}/{name}"
            ),
        }
    return {
        "source": copy.deepcopy(source),
        "release": copy.deepcopy(release),
        "refs": copy.deepcopy(refs),
        "assets": normalized_assets,
    }


def validate_gate_bundle(args: argparse.Namespace) -> dict[str, Any]:
    gates = {
        "g10a": verify_goal(args.g10a, lane=RELEASE_GATE_LANES["g10a"]),
        "g08_rc": verify_goal(args.g08_rc, lane=RELEASE_GATE_LANES["g08_rc"]),
        "g09_rc": verify_goal(args.g09_rc, lane=RELEASE_GATE_LANES["g09_rc"]),
        "published_assets": verify_goal(
            args.published_assets,
            lane=RELEASE_GATE_LANES["published_assets"],
        ),
    }
    return bind_verified_gate_bundle(gates)


def render_formulae(assets: dict[str, Any]) -> dict[str, str]:
    for backend in ASSET_NAMES:
        require(backend in assets, f"missing {backend} asset")
        require(
            SHA256_RE.fullmatch(str(assets[backend].get("sha256", ""))) is not None,
            f"{backend} asset SHA256 differs",
        )
    metal = BASE_METAL_FORMULA
    metal = metal.replace('version "0.7.7"', f'version "{VERSION}"', 1)
    metal = metal.replace("/v0.7.7/ferrum-macos-aarch64.tar.gz", f"/{TAG}/ferrum-macos-aarch64.tar.gz", 1)
    metal = metal.replace(
        'sha256 "e685342eb9d3050c1d4117ef71ecd20418f91bd731c8f304269b4ec36936247b"',
        f'sha256 "{assets["metal"]["sha256"]}"',
        1,
    )
    metal = metal.replace("/v0.7.7/ferrum-linux-x86_64.tar.gz", f"/{TAG}/ferrum-linux-x86_64.tar.gz", 1)
    metal = metal.replace(
        'sha256 "b98e22701a3a5b6d79ce4489ba1fcc1201a2925f24b9c87e35bc1678fb459715"',
        f'sha256 "{assets["cpu"]["sha256"]}"',
        1,
    )
    cuda = BASE_CUDA_FORMULA
    cuda = cuda.replace("/v0.7.7/ferrum-linux-x86_64-cuda-sm89.tar.gz", f"/{TAG}/ferrum-linux-x86_64-cuda-sm89.tar.gz", 1)
    cuda = cuda.replace('version "0.7.7"', f'version "{VERSION}"', 1)
    cuda = cuda.replace(
        'sha256 "6397de942d1c767383982186642da3990df49ea1cd6f604434a6cc9d6f7f912b"',
        f'sha256 "{assets["cuda"]["sha256"]}"',
        1,
    )
    result = {"Formula/ferrum.rb": metal, "Formula/ferrum-cuda.rb": cuda}
    validate_formulae(result, assets)
    return result


def validate_formulae(formulae: dict[str, str], assets: dict[str, Any]) -> None:
    require(set(formulae) == set(FORMULA_PATHS), "formula path denominator differs")
    expected_markers = {
        "Formula/ferrum.rb": (
            "class Ferrum < Formula",
            f'version "{VERSION}"',
            f'/{TAG}/{ASSET_NAMES["metal"]}',
            f'sha256 "{assets["metal"]["sha256"]}"',
            f'/{TAG}/{ASSET_NAMES["cpu"]}',
            f'sha256 "{assets["cpu"]["sha256"]}"',
        ),
        "Formula/ferrum-cuda.rb": (
            "class FerrumCuda < Formula",
            f'version "{VERSION}"',
            f'/{TAG}/{ASSET_NAMES["cuda"]}',
            f'sha256 "{assets["cuda"]["sha256"]}"',
        ),
    }
    for path, markers in expected_markers.items():
        text = formulae[path]
        require(text.endswith("\n"), f"{path} lacks final newline")
        require("0.7.7" not in text, f"{path} retained the old version")
        for marker in markers:
            require(text.count(marker) == 1, f"{path} marker differs: {marker}")
        sha_lines = re.findall(r'^\s*sha256 "([0-9a-f]{64})"$', text, re.M)
        expected_count = 2 if path.endswith("ferrum.rb") else 1
        require(len(sha_lines) == expected_count, f"{path} SHA denominator differs")


def run_command(
    argv: list[str],
    *,
    cwd: Path,
    recorder: list[dict[str, Any]] | None = None,
    check: bool = True,
    timeout_seconds: int = 300,
) -> subprocess.CompletedProcess[str]:
    rendered_argv = "\0".join(argv)
    require(
        not any(
            re.search(r"(?i)(?:github|gh|homebrew_github_api)_token=", item)
            for item in argv
        )
        and not any(secret in rendered_argv for secret in secret_values()),
        "token-bearing command argument is forbidden",
    )
    process = subprocess.run(
        argv,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    stdout = sanitize_text(process.stdout)
    stderr = sanitize_text(process.stderr)
    if recorder is not None:
        recorder.append(
            {
                "argv": [sanitize_text(item) for item in argv],
                "cwd": sanitize_text(str(cwd.resolve())),
                "returncode": process.returncode,
                "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr[-2000:],
                "environment_recorded": False,
            }
        )
    if check and process.returncode != 0:
        raise HomebrewReleaseError(
            f"command failed ({process.returncode}): "
            f"{sanitize_text(str(argv))}: {stderr[-1000:]}"
        )
    return subprocess.CompletedProcess(
        argv, process.returncode, stdout=stdout, stderr=stderr
    )


def git_output(repo: Path, *arguments: str) -> str:
    return run_command(["git", *arguments], cwd=repo).stdout.strip()


def validate_tap_base(
    repo: Path,
    *,
    expected_head: str,
    expected_blobs: dict[str, str],
    allowed_origins: set[str],
) -> dict[str, Any]:
    repo = repo.expanduser().resolve()
    require((repo / ".git").exists(), f"tap checkout is not a git repository: {repo}")
    require(git_output(repo, "branch", "--show-current") == TAP_BRANCH, "tap branch must be main")
    origin = git_output(repo, "remote", "get-url", "origin")
    require(origin in allowed_origins, f"unexpected tap origin: {origin}")
    require(git_output(repo, "status", "--porcelain") == "", "tap checkout is dirty")
    head = git_output(repo, "rev-parse", "HEAD")
    require(head == expected_head, "tap HEAD differs from the frozen base")
    for relative, expected_blob in expected_blobs.items():
        path = repo / relative
        require(path.is_file() and not path.is_symlink(), f"tap formula is missing: {relative}")
        require(path.read_text(encoding="utf-8") == BASE_FORMULAE[relative], f"tap {relative} differs from v0.7.7 base")
        require(git_output(repo, "hash-object", relative) == expected_blob, f"tap {relative} blob differs")
    return {"repo": repo, "origin": origin, "head": head}


def formula_diff(formulae: dict[str, str]) -> str:
    chunks: list[str] = []
    for path in FORMULA_PATHS:
        chunks.extend(
            difflib.unified_diff(
                BASE_FORMULAE[path].splitlines(keepends=True),
                formulae[path].splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
    result = "".join(chunks)
    require(result, "formula diff is empty")
    return result


def write_proposal(out: Path, formulae: dict[str, str]) -> dict[str, Any]:
    for relative, text in formulae.items():
        atomic_write_text(out / "proposal" / relative, text)
    atomic_write_text(out / "proposal.diff", formula_diff(formulae))
    return {
        "formulae": {
            relative: artifact_ref(out / "proposal" / relative, root=out)
            for relative in FORMULA_PATHS
        },
        "diff": artifact_ref(out / "proposal.diff", root=out),
    }


def validate_worktree_formulae(repo: Path, formulae: dict[str, str]) -> None:
    for relative, expected in formulae.items():
        require(
            (repo / relative).read_text(encoding="utf-8") == expected,
            f"tap worktree {relative} differs from proposal",
        )


def proposal_commit_is_valid(
    repo: Path,
    *,
    commit: str,
    base_head: str,
    formulae: dict[str, str],
) -> bool:
    if SHA1_RE.fullmatch(commit) is None:
        return False
    if git_output(repo, "rev-parse", f"{commit}^") != base_head:
        return False
    if git_output(repo, "log", "-1", "--format=%s", commit) != f"ferrum {VERSION}":
        return False
    changed = set(
        filter(
            None,
            git_output(
                repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit
            ).splitlines(),
        )
    )
    if changed != set(FORMULA_PATHS):
        return False
    for relative, expected in formulae.items():
        observed = run_command(
            ["git", "show", f"{commit}:{relative}"], cwd=repo
        ).stdout
        if observed != expected:
            return False
    return True


def publish_to_tap(
    repo: Path,
    *,
    formulae: dict[str, str],
    base_head: str,
    base_blobs: dict[str, str],
    allowed_origins: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repo = repo.expanduser().resolve()
    commands: list[dict[str, Any]] = []
    require((repo / ".git").exists(), "tap checkout is not a git repository")
    require(git_output(repo, "branch", "--show-current") == TAP_BRANCH, "tap branch must be main")
    origin = git_output(repo, "remote", "get-url", "origin")
    require(origin in allowed_origins, f"unexpected tap origin: {origin}")
    run_command(["git", "fetch", "origin", TAP_BRANCH], cwd=repo, recorder=commands)
    remote_head = git_output(repo, "rev-parse", f"refs/remotes/origin/{TAP_BRANCH}")
    head = git_output(repo, "rev-parse", "HEAD")
    status = git_output(repo, "status", "--porcelain")

    if remote_head != base_head:
        require(
            proposal_commit_is_valid(
                repo, commit=remote_head, base_head=base_head, formulae=formulae
            ),
            "remote main moved to an unrecognized commit",
        )
        if head == base_head and not status:
            run_command(
                ["git", "merge", "--ff-only", f"refs/remotes/origin/{TAP_BRANCH}"],
                cwd=repo,
                recorder=commands,
            )
            head = git_output(repo, "rev-parse", "HEAD")
            status = git_output(repo, "status", "--porcelain")
        require(head == remote_head and not status, "local tap is not at published remote main")
        validate_worktree_formulae(repo, formulae)
        return (
            {
                "base_head": base_head,
                "commit": remote_head,
                "remote_head": remote_head,
                "remote_verified": True,
                "disposition": "already-published-not-repushed",
            },
            commands,
        )

    if head == base_head:
        if status:
            dirty_paths = {
                line[3:] for line in status.splitlines() if len(line) >= 4
            }
            require(dirty_paths == set(FORMULA_PATHS), "tap has unrelated dirty files")
            validate_worktree_formulae(repo, formulae)
        else:
            validate_tap_base(
                repo,
                expected_head=base_head,
                expected_blobs=base_blobs,
                allowed_origins=allowed_origins,
            )
            for relative, text in formulae.items():
                atomic_write_text(repo / relative, text)
        validate_worktree_formulae(repo, formulae)
        for relative in FORMULA_PATHS:
            run_command(["ruby", "-c", relative], cwd=repo, recorder=commands)
        run_command(["git", "diff", "--check"], cwd=repo, recorder=commands)
        run_command(["git", "add", "--", *FORMULA_PATHS], cwd=repo, recorder=commands)
        run_command(
            ["git", "commit", "-m", f"ferrum {VERSION}"],
            cwd=repo,
            recorder=commands,
        )
        head = git_output(repo, "rev-parse", "HEAD")
    else:
        require(not status, "tap proposal commit worktree is dirty")
    require(
        proposal_commit_is_valid(
            repo, commit=head, base_head=base_head, formulae=formulae
        ),
        "local proposal commit differs",
    )
    run_command(
        ["git", "push", "origin", f"HEAD:refs/heads/{TAP_BRANCH}"],
        cwd=repo,
        recorder=commands,
    )
    run_command(["git", "fetch", "origin", TAP_BRANCH], cwd=repo, recorder=commands)
    remote_head = git_output(repo, "rev-parse", f"refs/remotes/origin/{TAP_BRANCH}")
    require(remote_head == head, "remote main did not reach the proposal commit")
    require(git_output(repo, "status", "--porcelain") == "", "tap became dirty after push")
    return (
        {
            "base_head": base_head,
            "commit": head,
            "remote_head": remote_head,
            "remote_verified": True,
            "disposition": "committed-and-pushed",
        },
        commands,
    )


MANIFEST_FIELDS = {
    "schema_version",
    "artifact_type",
    "status",
    "lane",
    "mode",
    "version",
    "canonical",
    "release_candidate",
    "release",
    "release_gates",
    "assets",
    "tap",
    "proposal",
    "commands",
    "credential_policy",
    "created_at",
    "manifest_id",
    "pass_line",
}


def manifest_identity(value: dict[str, Any]) -> str:
    fields = (
        "schema_version",
        "artifact_type",
        "lane",
        "mode",
        "version",
        "release_candidate",
        "release",
        "release_gates",
        "assets",
        "tap",
        "proposal",
        "commands",
        "credential_policy",
        "created_at",
    )
    return sha256_bytes(canonical_json({key: value[key] for key in fields}))


def validate_manifest(
    path: Path,
    *,
    expected_base_head: str = TAP_BASE_HEAD,
    verify_release_gates: bool = True,
) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    root = manifest_path.parent
    value = exact_fields(read_json(manifest_path, "Homebrew manifest"), MANIFEST_FIELDS, "Homebrew manifest")
    mode = value.get("mode")
    require(mode in {"dry-run", "publish"}, "Homebrew manifest mode differs")
    prefix = PUBLISH_PASS_PREFIX if mode == "publish" else DRY_RUN_PASS_PREFIX
    require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("artifact_type") == "runtime_vnext_homebrew_release_manifest"
        and value.get("status") == "pass"
        and value.get("lane") == LANE
        and value.get("version") == VERSION
        and value.get("canonical") is True
        and value.get("pass_line") == f"{prefix}: {root}",
        "Homebrew manifest identity/status differs",
    )
    try:
        created_at = dt.datetime.fromisoformat(str(value.get("created_at")))
    except ValueError as error:
        raise HomebrewReleaseError("Homebrew manifest created_at differs") from error
    require(
        created_at.tzinfo is not None,
        "Homebrew manifest created_at must include a timezone",
    )
    source = exact_fields(
        value.get("release_candidate"),
        {"git_sha", "git_tree_sha", "dirty"},
        "Homebrew release candidate",
    )
    require(
        SHA1_RE.fullmatch(str(source["git_sha"])) is not None
        and SHA1_RE.fullmatch(str(source["git_tree_sha"])) is not None
        and source["dirty"] is False,
        "Homebrew release candidate differs",
    )
    release = exact_fields(
        value.get("release"),
        {
            "id",
            "html_url",
            "tag_name",
            "tag_sha",
            "release_candidate_tag",
            "draft",
            "prerelease",
            "published_at",
            "asset_set_sha256",
            "asset_count",
        },
        "Homebrew published release",
    )
    require(
        release["tag_name"] == TAG
        and release["tag_sha"] == source["git_sha"]
        and release["release_candidate_tag"] == RC_TAG
        and release["draft"] is False
        and release["prerelease"] is True
        and isinstance(release["asset_set_sha256"], str)
        and SHA256_RE.fullmatch(release["asset_set_sha256"]) is not None
        and type(release["asset_count"]) is int
        and release["asset_count"] > 0,
        "Homebrew published release identity/state differs",
    )
    gates = exact_fields(value.get("release_gates"), set(RELEASE_GATE_LANES), "Homebrew release gates")
    gate_paths: dict[str, Path] = {}
    for key, ref in gates.items():
        gate_paths[key] = validate_artifact_ref(
            ref, root=None, label=f"Homebrew {key} gate"
        )
    assets = exact_fields(value.get("assets"), set(ASSET_NAMES), "Homebrew assets")
    for backend, name in ASSET_NAMES.items():
        row = exact_fields(assets[backend], {"name", "sha256", "size_bytes", "url"}, f"Homebrew {backend} asset")
        require(
            row["name"] == name
            and SHA256_RE.fullmatch(str(row["sha256"])) is not None
            and type(row["size_bytes"]) is int
            and row["size_bytes"] > 0
            and row["url"]
            == (
                "https://github.com/sizzlecar/ferrum-infer-rs/releases/"
                f"download/{TAG}/{name}"
            ),
            f"Homebrew {backend} asset differs",
        )
    proposal = exact_fields(value.get("proposal"), {"formulae", "diff"}, "Homebrew proposal")
    formulae: dict[str, str] = {}
    formula_refs = exact_fields(proposal["formulae"], set(FORMULA_PATHS), "Homebrew formula refs")
    for relative, ref in formula_refs.items():
        formula_path = validate_artifact_ref(ref, root=root, label=f"Homebrew {relative}")
        formulae[relative] = formula_path.read_text(encoding="utf-8")
    diff_path = validate_artifact_ref(
        proposal["diff"], root=root, label="Homebrew proposal diff"
    )
    validate_formulae(formulae, assets)
    require(
        formulae == render_formulae(assets),
        "Homebrew formulae differ from the frozen v0.7.7-to-v0.8.0 transform",
    )
    require(
        diff_path.read_text(encoding="utf-8") == formula_diff(formulae),
        "Homebrew proposal diff differs from the exact formulae",
    )
    tap = exact_fields(
        value.get("tap"),
        {"repository", "branch", "base_head", "commit", "remote_head", "remote_verified", "disposition"},
        "Homebrew tap",
    )
    require(
        tap["repository"] == TAP_REPOSITORY
        and tap["branch"] == TAP_BRANCH
        and tap["base_head"] == expected_base_head,
        "Homebrew tap identity differs",
    )
    if mode == "publish":
        require(
            SHA1_RE.fullmatch(str(tap["commit"])) is not None
            and tap["remote_head"] == tap["commit"]
            and tap["remote_verified"] is True
            and tap["disposition"]
            in {"committed-and-pushed", "already-published-not-repushed"},
            "Homebrew published tap state differs",
        )
    else:
        require(
            tap["commit"] is None
            and tap["remote_head"] is None
            and tap["remote_verified"] is False
            and tap["disposition"] == "dry-run-no-mutation",
            "Homebrew dry-run tap state differs",
        )
    require(isinstance(value.get("commands"), list), "Homebrew commands must be a list")
    if mode == "dry-run":
        require(value["commands"] == [], "Homebrew dry-run recorded mutating commands")
    for command in value["commands"]:
        exact_fields(
            command,
            {"argv", "cwd", "returncode", "stdout_tail", "stderr_tail", "environment_recorded"},
            "Homebrew command receipt",
        )
        require(
            isinstance(command["argv"], list)
            and command["argv"]
            and all(isinstance(item, str) and item for item in command["argv"])
            and isinstance(command["cwd"], str)
            and Path(command["cwd"]).is_absolute()
            and command["returncode"] == 0
            and isinstance(command["stdout_tail"], str)
            and isinstance(command["stderr_tail"], str)
            and command["environment_recorded"] is False
            and sanitize_text(json.dumps(command, sort_keys=True))
            == json.dumps(command, sort_keys=True),
            "Homebrew command did not pass",
        )
    require(
        value.get("credential_policy")
        == {"source": "existing-git-credential-helper", "secret_values_recorded": False},
        "Homebrew credential policy differs",
    )
    if verify_release_gates:
        verified = {
            key: verify_goal(gate_paths[key], lane=lane)
            for key, lane in RELEASE_GATE_LANES.items()
        }
        rebound = bind_verified_gate_bundle(verified)
        require(
            value["release_candidate"] == rebound["source"]
            and value["release"] == rebound["release"]
            and value["release_gates"] == rebound["refs"]
            and value["assets"] == rebound["assets"],
            "Homebrew manifest is stale against authoritative release gates",
        )
    require(value.get("manifest_id") == manifest_identity(value), "Homebrew manifest id differs")
    return value


def install_manifest(
    out: Path,
    manifest: dict[str, Any],
    *,
    expected_base_head: str = TAP_BASE_HEAD,
    verify_release_gates: bool = True,
) -> Path:
    canonical = out / "manifest.json"
    alias = out / "gate.manifest.json"
    require(not canonical.exists() and not alias.exists(), "canonical Homebrew manifest already exists")
    candidate = out / f".manifest.candidate-{os.getpid()}.json"
    try:
        write_json(candidate, manifest)
        validate_manifest(
            candidate,
            expected_base_head=expected_base_head,
            verify_release_gates=verify_release_gates,
        )
        os.replace(candidate, canonical)
        atomic_write_bytes(alias, canonical.read_bytes())
    except BaseException:
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
        raise
    return canonical


def build_manifest(
    *,
    out: Path,
    mode: str,
    bundle: dict[str, Any],
    proposal: dict[str, Any],
    tap: dict[str, Any],
    commands: list[dict[str, Any]],
) -> dict[str, Any]:
    out = out.expanduser().resolve()
    prefix = PUBLISH_PASS_PREFIX if mode == "publish" else DRY_RUN_PASS_PREFIX
    value = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_homebrew_release_manifest",
        "status": "pass",
        "lane": LANE,
        "mode": mode,
        "version": VERSION,
        "canonical": True,
        "release_candidate": copy.deepcopy(bundle["source"]),
        "release": copy.deepcopy(bundle["release"]),
        "release_gates": copy.deepcopy(bundle["refs"]),
        "assets": copy.deepcopy(bundle["assets"]),
        "tap": {"repository": TAP_REPOSITORY, "branch": TAP_BRANCH, **copy.deepcopy(tap)},
        "proposal": copy.deepcopy(proposal),
        "commands": copy.deepcopy(commands),
        "credential_policy": {
            "source": "existing-git-credential-helper",
            "secret_values_recorded": False,
        },
        "created_at": iso_now(),
        "manifest_id": "",
        "pass_line": f"{prefix}: {out}",
    }
    value["manifest_id"] = manifest_identity(value)
    return value


def run_release(args: argparse.Namespace) -> Path:
    out = args.out.expanduser().resolve()
    require(not out.exists(), f"refusing to overwrite Homebrew artifact: {out}")
    out.mkdir(parents=True)
    secrets = secret_values()
    try:
        bundle = validate_gate_bundle(args)
        formulae = render_formulae(bundle["assets"])
        proposal = write_proposal(out, formulae)
        if args.publish:
            tap, commands = publish_to_tap(
                args.tap_repo,
                formulae=formulae,
                base_head=TAP_BASE_HEAD,
                base_blobs=BASE_FORMULA_BLOBS,
                allowed_origins=TAP_ORIGINS,
            )
            mode = "publish"
        else:
            base = validate_tap_base(
                args.tap_repo,
                expected_head=TAP_BASE_HEAD,
                expected_blobs=BASE_FORMULA_BLOBS,
                allowed_origins=TAP_ORIGINS,
            )
            tap = {
                "base_head": base["head"],
                "commit": None,
                "remote_head": None,
                "remote_verified": False,
                "disposition": "dry-run-no-mutation",
            }
            commands = []
            mode = "dry-run"
        manifest = build_manifest(
            out=out,
            mode=mode,
            bundle=bundle,
            proposal=proposal,
            tap=tap,
            commands=commands,
        )
        install_manifest(out, manifest)
        assert_no_secrets(out, secrets)
        print(manifest["pass_line"])
        return out
    except BaseException:
        assert_no_secrets(out, secrets)
        raise


def expect_failure(
    label: str, callback: Callable[[], Any], marker: str | None = None
) -> None:
    try:
        callback()
    except HomebrewReleaseError as error:
        if marker is not None:
            require(marker in str(error), f"{label} failed for wrong reason: {error}")
        return
    raise HomebrewReleaseError(f"{label} unexpectedly passed")


def fixture_gate_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = {"git_sha": "a" * 40, "git_tree_sha": "b" * 40, "dirty": False}
    refs: dict[str, Any] = {}
    for index, key in enumerate(RELEASE_GATE_LANES, 1):
        path = root / f"{key}.json"
        write_json(path, {"fixture": key})
        refs[key] = artifact_ref(path)
        require(refs[key]["sha256"] != str(index) * 64, "fixture digest collision")
    assets = {
        backend: {
            "name": name,
            "tarball_sha256": str(index) * 64,
            "digest": f"sha256:{str(index) * 64}",
            "size": index,
        }
        for index, (backend, name) in enumerate(ASSET_NAMES.items(), 1)
    }
    gates = {
        "g10a": {
            "source": source,
            "ref": refs["g10a"],
            "manifest": {"release_candidate_tag": RC_TAG, "inputs": {}},
        },
        "g08_rc": {
            "source": source,
            "ref": refs["g08_rc"],
            "manifest": {"inputs": {"g10a": refs["g10a"]}},
        },
        "g09_rc": {
            "source": source,
            "ref": refs["g09_rc"],
            "manifest": {
                "inputs": {"g10a": refs["g10a"], "g08_rc": refs["g08_rc"]}
            },
        },
        "published_assets": {
            "source": source,
            "ref": refs["published_assets"],
            "manifest": {
                "inputs": {
                    "g10a": refs["g10a"],
                    "g08_rc": refs["g08_rc"],
                    "g09_rc": refs["g09_rc"],
                },
                "release": {
                    "id": "1",
                    "html_url": "https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/v0.8.0",
                    "tag_name": TAG,
                    "tag_sha": source["git_sha"],
                    "release_candidate_tag": RC_TAG,
                    "draft": False,
                    "prerelease": True,
                    "published_at": iso_now(),
                    "asset_set_sha256": "f" * 64,
                    "asset_count": 18,
                },
                "assets": assets,
            },
        },
    }
    return gates, refs


def initialize_tap_fixture(root: Path) -> tuple[Path, Path, str, dict[str, str]]:
    remote = root / "remote.git"
    tap = root / "tap"
    run_command(["git", "init", "--bare", str(remote)], cwd=root)
    run_command(["git", "init", "-b", TAP_BRANCH, str(tap)], cwd=root)
    run_command(["git", "config", "user.name", "Ferrum Selftest"], cwd=tap)
    run_command(["git", "config", "user.email", "selftest@example.invalid"], cwd=tap)
    for relative, text in BASE_FORMULAE.items():
        atomic_write_text(tap / relative, text)
    atomic_write_text(tap / "README.md", "# fixture tap\n")
    run_command(["git", "add", "."], cwd=tap)
    run_command(["git", "commit", "-m", "ferrum 0.7.7"], cwd=tap)
    run_command(["git", "remote", "add", "origin", str(remote)], cwd=tap)
    run_command(["git", "push", "-u", "origin", TAP_BRANCH], cwd=tap)
    base_head = git_output(tap, "rev-parse", "HEAD")
    blobs = {relative: git_output(tap, "hash-object", relative) for relative in FORMULA_PATHS}
    return tap, remote, base_head, blobs


def run_selftest() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-homebrew-release-selftest-") as temporary:
        root = Path(temporary)
        gates, _ = fixture_gate_bundle(root)
        bundle = bind_verified_gate_bundle(gates)
        formulae = render_formulae(bundle["assets"])
        require("0.8.0" in formulae["Formula/ferrum.rb"], "formula renderer omitted version")

        tampered = copy.deepcopy(gates)
        tampered["published_assets"]["manifest"]["inputs"]["g09_rc"] = copy.deepcopy(
            tampered["published_assets"]["manifest"]["inputs"]["g09_rc"]
        )
        tampered["published_assets"]["manifest"]["inputs"]["g09_rc"]["sha256"] = "0" * 64
        expect_failure(
            "tampered gate chain",
            lambda: bind_verified_gate_bundle(tampered),
            "does not bind exact g09_rc",
        )
        tampered = copy.deepcopy(gates)
        tampered["published_assets"]["manifest"]["assets"]["cuda"]["digest"] = "sha256:" + "0" * 64
        expect_failure(
            "tampered CUDA asset",
            lambda: bind_verified_gate_bundle(tampered),
            "cuda asset identity differs",
        )
        bad_formulae = copy.deepcopy(formulae)
        bad_formulae["Formula/ferrum-cuda.rb"] = bad_formulae[
            "Formula/ferrum-cuda.rb"
        ].replace(bundle["assets"]["cuda"]["sha256"], "0" * 64)
        expect_failure(
            "tampered CUDA formula",
            lambda: validate_formulae(bad_formulae, bundle["assets"]),
            "marker differs",
        )

        token = "hostile-homebrew-token-value"
        for name in TOKEN_ENV_NAMES:
            require(token not in sanitize_text(f"{name}={token}"), f"sanitizer leaked {name}")
        require(
            token
            not in sanitize_text(f"https://x-access-token:{token}@github.com/repo"),
            "sanitizer leaked URL credentials",
        )

        tap, remote, base_head, blobs = initialize_tap_fixture(root)
        base_status = git_output(tap, "status", "--porcelain")
        validate_tap_base(
            tap,
            expected_head=base_head,
            expected_blobs=blobs,
            allowed_origins={str(remote)},
        )
        require(
            git_output(tap, "rev-parse", "HEAD") == base_head
            and git_output(tap, "status", "--porcelain") == base_status == "",
            "dry-run base validation mutated the tap",
        )

        publish_state, commands = publish_to_tap(
            tap,
            formulae=formulae,
            base_head=base_head,
            base_blobs=blobs,
            allowed_origins={str(remote)},
        )
        require(
            publish_state["disposition"] == "committed-and-pushed"
            and publish_state["commit"]
            == git_output(tap, "rev-parse", f"refs/remotes/origin/{TAP_BRANCH}"),
            "fixture publish did not reach the bare remote",
        )
        repeated, repeated_commands = publish_to_tap(
            tap,
            formulae=formulae,
            base_head=base_head,
            base_blobs=blobs,
            allowed_origins={str(remote)},
        )
        require(
            repeated["disposition"] == "already-published-not-repushed"
            and not any("push" in command["argv"] for command in repeated_commands),
            "fixture resume repushed an already published formula",
        )

        out = root / "artifact"
        out.mkdir()
        proposal = write_proposal(out, formulae)
        manifest = build_manifest(
            out=out,
            mode="publish",
            bundle=bundle,
            proposal=proposal,
            tap=publish_state,
            commands=commands,
        )
        hostile_path = out / "hostile.json"

        def hostile_manifest(
            label: str, mutate: Callable[[dict[str, Any]], None], marker: str
        ) -> None:
            hostile = copy.deepcopy(manifest)
            mutate(hostile)
            hostile["manifest_id"] = manifest_identity(hostile)
            try:
                write_json(hostile_path, hostile)
                expect_failure(
                    label,
                    lambda: validate_manifest(
                        hostile_path,
                        expected_base_head=base_head,
                        verify_release_gates=False,
                    ),
                    marker,
                )
            finally:
                hostile_path.unlink(missing_ok=True)

        hostile_manifest(
            "tampered tap base",
            lambda value: value["tap"].update({"base_head": "0" * 40}),
            "tap identity differs",
        )
        hostile_manifest(
            "tampered manifest asset SHA",
            lambda value: value["assets"]["cuda"].update({"sha256": "0" * 64}),
            "marker differs",
        )
        hostile_manifest(
            "tampered manifest gate ref",
            lambda value: value["release_gates"]["g09_rc"].update(
                {"sha256": "0" * 64}
            ),
            "SHA256 differs",
        )
        hostile_manifest(
            "token-bearing command receipt",
            lambda value: value["commands"][0].update(
                {"stdout_tail": f"GITHUB_TOKEN={token}"}
            ),
            "command did not pass",
        )
        install_manifest(
            out,
            manifest,
            expected_base_head=base_head,
            verify_release_gates=False,
        )
        validate_manifest(
            out / "manifest.json",
            expected_base_head=base_head,
            verify_release_gates=False,
        )
        saved = {name: os.environ.get(name) for name in TOKEN_ENV_NAMES}
        try:
            os.environ["GITHUB_TOKEN"] = token
            assert_no_secrets(out, secret_values())
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    print(SELFTEST_PASS_LINE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--publish", action="store_true", help="commit and push the tap; default is dry-run")
    parser.add_argument("--tap-repo", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--g10a", type=Path)
    parser.add_argument("--g08-rc", type=Path)
    parser.add_argument("--g09-rc", type=Path)
    parser.add_argument("--published-assets", type=Path)
    args = parser.parse_args(argv)
    if args.self_test:
        require(not args.publish, "--self-test cannot publish")
        for name in ("tap_repo", "out", "g10a", "g08_rc", "g09_rc", "published_assets"):
            require(getattr(args, name) is None, f"--self-test cannot use --{name.replace('_', '-')}")
    else:
        for name in ("tap_repo", "out", "g10a", "g08_rc", "g09_rc", "published_assets"):
            require(getattr(args, name) is not None, f"--{name.replace('_', '-')} is required")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.self_test:
            run_selftest()
        else:
            run_release(args)
        return 0
    except (HomebrewReleaseError, OSError, subprocess.SubprocessError) as error:
        print(f"ERROR: {sanitize_text(str(error))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
