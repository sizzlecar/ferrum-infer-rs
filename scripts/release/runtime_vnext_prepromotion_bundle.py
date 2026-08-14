#!/usr/bin/env python3
"""Pack or verify the immutable two-file v0.8.0 prepromotion bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


OUTER_NAME = "gate.manifest.json"
CHILD_NAME = "manifest.json"
EXPECTED_NAMES = (OUTER_NAME, CHILD_NAME)
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OUTER_PASS_PREFIX = "FERRUM GATE runtime-vnext-prepromotion PASS:"
CHILD_PASS_PREFIX = "FERRUM V0.8.0 PREPROMOTION PASS:"
PACK_PASS_PREFIX = "FERRUM PREPROMOTION BUNDLE PACK PASS"
VERIFY_PASS_PREFIX = "FERRUM PREPROMOTION BUNDLE VERIFY PASS"


class BundleError(RuntimeError):
    """The prepromotion bundle is malformed or does not match its identity."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> Path:
    candidate = path.expanduser()
    require(
        candidate.is_file() and not candidate.is_symlink(),
        f"{label} must be a regular non-symlink file: {candidate}",
    )
    return candidate.resolve()


def json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"{label} is not valid UTF-8 JSON: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def validate_pair(
    outer_bytes: bytes,
    child_bytes: bytes,
    *,
    expected_child_sha256: str | None = None,
) -> dict[str, Any]:
    outer = json_object(outer_bytes, OUTER_NAME)
    child = json_object(child_bytes, CHILD_NAME)
    child_sha256 = sha256_bytes(child_bytes)
    outer_sha256 = sha256_bytes(outer_bytes)
    if expected_child_sha256 is not None:
        require(
            SHA256_RE.fullmatch(expected_child_sha256) is not None,
            "expected child SHA256 must be lowercase hexadecimal",
        )
        require(
            child_sha256 == expected_child_sha256,
            "prepromotion child manifest SHA256 mismatch",
        )

    require(outer.get("schema_version") == 1, "outer schema_version must be 1")
    require(
        outer.get("lane") == "runtime-vnext-prepromotion",
        "outer lane is not runtime-vnext-prepromotion",
    )
    require(
        outer.get("status") == "pass" and outer.get("child_returncode") == 0,
        "outer gate did not pass",
    )
    require(
        isinstance(outer.get("pass_line"), str)
        and outer["pass_line"].startswith(OUTER_PASS_PREFIX),
        "outer canonical PASS line is missing",
    )
    require(
        child.get("schema_version") == 1
        and child.get("lane") == "runtime-vnext-prepromotion"
        and child.get("status") == "pass",
        "child prepromotion identity/status differs",
    )
    require(
        isinstance(child.get("pass_line"), str)
        and child["pass_line"].startswith(CHILD_PASS_PREFIX),
        "child prepromotion PASS line is missing",
    )
    require(
        outer.get("child_pass_line") == child.get("pass_line"),
        "outer child PASS line does not bind the child",
    )
    artifacts = outer.get("child_artifacts")
    require(isinstance(artifacts, dict), "outer child_artifacts must be an object")
    child_ref = artifacts.get("child_manifest")
    require(isinstance(child_ref, dict), "outer child manifest reference is missing")
    require(
        child_ref.get("sha256") == child_sha256,
        "outer child manifest SHA256 does not bind the child",
    )
    if "size_bytes" in child_ref:
        require(
            child_ref.get("size_bytes") == len(child_bytes),
            "outer child manifest size does not bind the child",
        )
    for key in (
        "manifest_id",
        "release_candidate_sha",
        "prepromotion_pass_line",
        "release",
        "consumption",
        "dependencies",
    ):
        if key in outer:
            require(outer[key] == child.get(key), f"outer {key} differs from child")
    return {
        "outer_sha256": outer_sha256,
        "child_sha256": child_sha256,
        "release_candidate_sha": child.get("release_candidate_sha"),
        "manifest_id": child.get("manifest_id"),
    }


def write_receipt(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    destination = path.expanduser().resolve()
    require(not destination.exists(), f"receipt already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def bundle_receipt(
    *,
    mode: str,
    archive: Path,
    pair: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "pass",
        "mode": mode,
        "archive_name": archive.name,
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "files": list(EXPECTED_NAMES),
        **pair,
    }


def pack(args: argparse.Namespace) -> int:
    outer = regular_file(args.outer, "outer manifest")
    child = regular_file(args.child, "child manifest")
    archive = args.archive.expanduser().resolve()
    require(outer != child, "outer and child manifests must be distinct files")
    require(not archive.exists(), f"archive already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    outer_bytes = outer.read_bytes()
    child_bytes = child.read_bytes()
    require(len(outer_bytes) <= MAX_MEMBER_BYTES, "outer manifest is too large")
    require(len(child_bytes) <= MAX_MEMBER_BYTES, "child manifest is too large")
    pair = validate_pair(outer_bytes, child_bytes)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{archive.name}.",
            suffix=".tmp",
            dir=archive.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        with zipfile.ZipFile(temporary_name, "w", compression=zipfile.ZIP_STORED) as bundle:
            for name, payload in (
                (OUTER_NAME, outer_bytes),
                (CHILD_NAME, child_bytes),
            ):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                bundle.writestr(info, payload)
        temporary = Path(temporary_name)
        require(temporary.stat().st_size <= MAX_ARCHIVE_BYTES, "bundle archive is too large")
        os.replace(temporary, archive)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    receipt = bundle_receipt(mode="pack", archive=archive, pair=pair)
    write_receipt(args.receipt, receipt)
    print(f"{PACK_PASS_PREFIX}: {archive}")
    print(json.dumps(receipt, sort_keys=True))
    return 0


def verified_members(archive: Path) -> tuple[bytes, bytes]:
    require(archive.stat().st_size <= MAX_ARCHIVE_BYTES, "bundle archive is too large")
    try:
        with zipfile.ZipFile(archive) as bundle:
            infos = bundle.infolist()
            require(
                len(infos) == len(EXPECTED_NAMES)
                and sorted(info.filename for info in infos) == sorted(EXPECTED_NAMES),
                "bundle must contain exactly gate.manifest.json and manifest.json",
            )
            values: dict[str, bytes] = {}
            for info in infos:
                require(not info.is_dir(), f"bundle member is unexpectedly a directory: {info.filename}")
                require(info.flag_bits & 0x1 == 0, f"bundle member is encrypted: {info.filename}")
                require(
                    info.compress_type == zipfile.ZIP_STORED,
                    f"bundle member uses a non-canonical compression method: {info.filename}",
                )
                mode = info.external_attr >> 16
                require(
                    stat.S_IFMT(mode) == stat.S_IFREG,
                    f"bundle member is not a regular file: {info.filename}",
                )
                require(
                    0 <= info.file_size <= MAX_MEMBER_BYTES,
                    f"bundle member is too large: {info.filename}",
                )
                values[info.filename] = bundle.read(info)
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise BundleError(f"cannot read prepromotion bundle: {error}") from error
    return values[OUTER_NAME], values[CHILD_NAME]


def verify(args: argparse.Namespace) -> int:
    archive = regular_file(args.archive, "bundle archive")
    require(
        SHA256_RE.fullmatch(args.expected_archive_sha256) is not None,
        "expected archive SHA256 must be lowercase hexadecimal",
    )
    require(
        sha256_file(archive) == args.expected_archive_sha256,
        "prepromotion bundle archive SHA256 mismatch",
    )
    outer_bytes, child_bytes = verified_members(archive)
    pair = validate_pair(
        outer_bytes,
        child_bytes,
        expected_child_sha256=args.expected_child_sha256,
    )
    output = args.out.expanduser().resolve()
    require(not output.exists(), f"verification output must be fresh: {output}")
    output.mkdir(parents=True)
    (output / OUTER_NAME).write_bytes(outer_bytes)
    (output / CHILD_NAME).write_bytes(child_bytes)
    receipt = bundle_receipt(mode="verify", archive=archive, pair=pair)
    write_receipt(args.receipt, receipt)
    print(f"{VERIFY_PASS_PREFIX}: {output}")
    print(json.dumps(receipt, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    pack_parser = subparsers.add_parser("pack")
    pack_parser.add_argument("--outer", type=Path, required=True)
    pack_parser.add_argument("--child", type=Path, required=True)
    pack_parser.add_argument("--archive", type=Path, required=True)
    pack_parser.add_argument("--receipt", type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--archive", type=Path, required=True)
    verify_parser.add_argument("--expected-archive-sha256", required=True)
    verify_parser.add_argument("--expected-child-sha256", required=True)
    verify_parser.add_argument("--out", type=Path, required=True)
    verify_parser.add_argument("--receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return pack(args) if args.mode == "pack" else verify(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BundleError as error:
        print(f"FERRUM PREPROMOTION BUNDLE FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
