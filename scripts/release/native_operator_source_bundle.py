#!/usr/bin/env python3
"""Create, verify, fetch, and extract deterministic native source bundles."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SCHEMA_VERSION = 1
BUNDLE_PREFIX = "ferrum-native-cuda-v1"
ARCHIVE_FORMAT = "tar.gz"
MAX_MEMBERS = 1_000
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
SELFTEST_PASS_LINE = "FERRUM NATIVE SOURCE BUNDLE SELFTEST PASS"


class BundleError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def read_json(path: Path, label: str) -> Any:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"cannot read {label} {path}: {error}") from error


def write_json_create_new(path: Path, value: Any) -> None:
    require(not path.exists(), f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="ascii") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def relative_source_path(raw: Any, label: str) -> str:
    require(isinstance(raw, str) and raw, f"{label} must be a non-empty string")
    path = PurePosixPath(raw)
    require(
        not path.is_absolute()
        and ".." not in path.parts
        and "." not in path.parts
        and path.as_posix() == raw,
        f"{label} must be a normalized relative POSIX path",
    )
    return raw


def definition_members(definition_root: Path) -> list[str]:
    definition_paths = sorted(definition_root.glob("*.json"))
    require(definition_paths, f"no source definitions found under {definition_root}")
    members: set[str] = set()
    operators: set[str] = set()
    for definition_path in definition_paths:
        definition = read_json(definition_path, "native source definition")
        require(isinstance(definition, dict), f"definition must be an object: {definition_path}")
        require(
            definition.get("schema_version") == 3,
            f"definition schema mismatch: {definition_path}",
        )
        operator = definition.get("operator")
        require(
            isinstance(operator, str)
            and operator.startswith("ferrum.cuda.")
            and operator not in operators,
            f"definition operator is invalid or duplicated: {definition_path}",
        )
        operators.add(operator)
        for field in ("translation_units", "headers"):
            rows = definition.get(field)
            require(isinstance(rows, list), f"{definition_path}:{field} must be an array")
            for index, raw in enumerate(rows):
                members.add(
                    relative_source_path(
                        raw,
                        f"{definition_path.name}.{field}[{index}]",
                    )
                )
    require(
        1 <= len(members) <= MAX_MEMBERS,
        f"source bundle member count is outside [1,{MAX_MEMBERS}]",
    )
    return sorted(members)


def inventory_members(source_root: Path, members: Iterable[str]) -> list[dict[str, Any]]:
    canonical_root = source_root.resolve()
    inventory = []
    total_bytes = 0
    for relative in members:
        path = source_root / relative
        require(path.is_file() and not path.is_symlink(), f"source member is missing: {path}")
        resolved = path.resolve()
        require(
            resolved.is_relative_to(canonical_root),
            f"source member escapes source root: {relative}",
        )
        size = path.stat().st_size
        require(
            0 < size <= MAX_MEMBER_BYTES,
            f"source member size is invalid: {relative}: {size}",
        )
        total_bytes += size
        require(total_bytes <= MAX_TOTAL_BYTES, "source bundle exceeds total size bound")
        inventory.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "size_bytes": size,
            }
        )
    return inventory


def create_archive(source_root: Path, members: list[dict[str, Any]], archive: Path) -> None:
    require(not archive.exists(), f"archive output already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw,
                mtime=0,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.USTAR_FORMAT,
                ) as tar:
                    for row in members:
                        payload = (source_root / row["path"]).read_bytes()
                        info = tarfile.TarInfo(row["path"])
                        info.size = len(payload)
                        info.mode = 0o644
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = 0
                        with tempfile.SpooledTemporaryFile(max_size=MAX_MEMBER_BYTES) as body:
                            body.write(payload)
                            body.seek(0)
                            tar.addfile(info, body)
            raw.flush()
            os.fsync(raw.fileno())
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)


def validate_manifest(raw: Any) -> dict[str, Any]:
    require(isinstance(raw, dict), "bundle manifest must be an object")
    require(
        set(raw)
        == {
            "schema_version",
            "bundle_id",
            "source_layout",
            "member_set_sha256",
            "members",
            "archive",
            "distribution",
        },
        "bundle manifest shape mismatch",
    )
    require(raw.get("schema_version") == SCHEMA_VERSION, "bundle manifest schema mismatch")
    members = raw.get("members")
    require(
        isinstance(members, list) and 1 <= len(members) <= MAX_MEMBERS,
        "bundle manifest member set is invalid",
    )
    paths = []
    total_bytes = 0
    for index, row in enumerate(members):
        require(
            isinstance(row, dict)
            and set(row) == {"path", "sha256", "size_bytes"},
            f"bundle member {index} shape mismatch",
        )
        paths.append(relative_source_path(row.get("path"), f"members[{index}].path"))
        require(
            isinstance(row.get("sha256"), str)
            and len(row["sha256"]) == 64
            and all(character in "0123456789abcdef" for character in row["sha256"]),
            f"bundle member {index} SHA256 is invalid",
        )
        size = row.get("size_bytes")
        require(
            isinstance(size, int)
            and not isinstance(size, bool)
            and 0 < size <= MAX_MEMBER_BYTES,
            f"bundle member {index} size is invalid",
        )
        total_bytes += size
    require(paths == sorted(set(paths)), "bundle members must be sorted and unique")
    require(total_bytes <= MAX_TOTAL_BYTES, "bundle member bytes exceed the total bound")
    member_set_sha = hashlib.sha256(canonical_json_bytes(members)).hexdigest()
    require(
        raw.get("member_set_sha256") == member_set_sha,
        "bundle member-set SHA256 mismatch",
    )
    require(
        raw.get("bundle_id") == f"{BUNDLE_PREFIX}+sha256.{member_set_sha}",
        "bundle id mismatch",
    )
    require(
        raw.get("source_layout") == "ferrum-kernels-relative-v1",
        "bundle source layout mismatch",
    )
    archive = raw.get("archive")
    require(
        isinstance(archive, dict)
        and set(archive) == {"file_name", "format", "sha256", "size_bytes"}
        and archive.get("format") == ARCHIVE_FORMAT
        and isinstance(archive.get("file_name"), str)
        and PurePosixPath(archive["file_name"]).name == archive["file_name"]
        and isinstance(archive.get("sha256"), str)
        and len(archive["sha256"]) == 64
        and isinstance(archive.get("size_bytes"), int)
        and archive["size_bytes"] > 0,
        "bundle archive identity is invalid",
    )
    distribution = raw.get("distribution")
    require(
        isinstance(distribution, dict)
        and set(distribution) == {"kind", "repository", "tag", "asset_name"}
        and distribution.get("kind") == "github-release-asset"
        and distribution.get("asset_name") == archive["file_name"]
        and all(
            isinstance(distribution.get(field), str) and distribution[field]
            for field in ("repository", "tag")
        ),
        "bundle distribution identity is invalid",
    )
    return raw


def verify_archive(manifest: dict[str, Any], archive: Path) -> None:
    archive_identity = manifest["archive"]
    require(
        archive.is_file()
        and not archive.is_symlink()
        and archive.stat().st_size == archive_identity["size_bytes"]
        and sha256(archive) == archive_identity["sha256"],
        f"bundle archive identity mismatch: {archive}",
    )
    expected = {row["path"]: row for row in manifest["members"]}
    observed: dict[str, dict[str, Any]] = {}
    with tarfile.open(archive, mode="r:gz") as tar:
        members = tar.getmembers()
        require(len(members) == len(expected), "bundle archive member count mismatch")
        for index, member in enumerate(members):
            path = relative_source_path(member.name, f"archive.members[{index}]")
            require(
                member.isfile()
                and not member.issym()
                and not member.islnk()
                and member.uid == 0
                and member.gid == 0
                and member.mtime == 0
                and member.mode == 0o644,
                f"bundle archive member metadata is invalid: {path}",
            )
            expected_row = expected.get(path)
            require(expected_row is not None and path not in observed, f"unexpected member: {path}")
            extracted = tar.extractfile(member)
            require(extracted is not None, f"cannot read bundle member: {path}")
            digest = hashlib.sha256()
            size = 0
            while chunk := extracted.read(1024 * 1024):
                size += len(chunk)
                require(size <= MAX_MEMBER_BYTES, f"bundle member exceeds size bound: {path}")
                digest.update(chunk)
            row = {"path": path, "sha256": digest.hexdigest(), "size_bytes": size}
            require(row == expected_row, f"bundle member identity mismatch: {path}")
            observed[path] = row
    require(list(observed) == list(expected), "bundle archive order differs from manifest")


def extract_archive(manifest: dict[str, Any], archive: Path, out: Path) -> None:
    verify_archive(manifest, archive)
    require(not out.exists(), f"extract output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            for member in tar.getmembers():
                destination = temporary / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                require(source is not None, f"cannot extract bundle member: {member.name}")
                with destination.open("xb") as handle:
                    shutil.copyfileobj(source, handle, length=1024 * 1024)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(destination, 0o644)
        observed = inventory_members(temporary, [row["path"] for row in manifest["members"]])
        require(observed == manifest["members"], "extracted bundle inventory mismatch")
        temporary.replace(out)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_materialized_tree(manifest: dict[str, Any], root: Path) -> None:
    require(root.is_dir() and not root.is_symlink(), f"materialized source root is missing: {root}")
    expected_paths = [row["path"] for row in manifest["members"]]
    observed_paths = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"materialized source tree contains a symlink: {path}")
        if path.is_file():
            observed_paths.append(path.relative_to(root).as_posix())
        else:
            require(path.is_dir(), f"materialized source tree contains a special file: {path}")
    require(
        observed_paths == expected_paths,
        "materialized source tree members differ from the bundle manifest",
    )
    require(
        inventory_members(root, expected_paths) == manifest["members"],
        "materialized source tree identity differs from the bundle manifest",
    )


def cached_archive(manifest: dict[str, Any], cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / manifest["archive"]["file_name"]
    if not archive.exists():
        distribution = manifest["distribution"]
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                distribution["tag"],
                "--repo",
                distribution["repository"],
                "--pattern",
                distribution["asset_name"],
                "--dir",
                str(cache),
                "--clobber",
            ],
            check=True,
        )
    return archive


def create(args: argparse.Namespace) -> None:
    source_root = args.source_root.expanduser().resolve()
    definition_root = args.definition_root.expanduser().resolve()
    archive = args.archive.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    members = inventory_members(source_root, definition_members(definition_root))
    member_set_sha = hashlib.sha256(canonical_json_bytes(members)).hexdigest()
    create_archive(source_root, members, archive)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": f"{BUNDLE_PREFIX}+sha256.{member_set_sha}",
        "source_layout": "ferrum-kernels-relative-v1",
        "member_set_sha256": member_set_sha,
        "members": members,
        "archive": {
            "file_name": archive.name,
            "format": ARCHIVE_FORMAT,
            "sha256": sha256(archive),
            "size_bytes": archive.stat().st_size,
        },
        "distribution": {
            "kind": "github-release-asset",
            "repository": args.github_repository,
            "tag": args.github_tag,
            "asset_name": archive.name,
        },
    }
    validate_manifest(manifest)
    verify_archive(manifest, archive)
    write_json_create_new(manifest_path, manifest)
    print(f"FERRUM NATIVE SOURCE BUNDLE READY: {manifest_path}")


def materialize(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = validate_manifest(read_json(manifest_path, "source bundle manifest"))
    if args.archive is not None:
        archive = args.archive.expanduser().resolve()
    else:
        archive = cached_archive(manifest, args.cache.expanduser().resolve())
    out = args.out.expanduser().resolve()
    extract_archive(manifest, archive, out)
    verify_materialized_tree(manifest, out)
    print(f"FERRUM NATIVE SOURCE BUNDLE MATERIALIZED: {out}")


def ensure_materialized(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = validate_manifest(read_json(manifest_path, "source bundle manifest"))
    out = args.out.expanduser().resolve()
    if out.exists():
        verify_materialized_tree(manifest, out)
    else:
        archive = cached_archive(manifest, args.cache.expanduser().resolve())
        extract_archive(manifest, archive, out)
        verify_materialized_tree(manifest, out)
    print(f"FERRUM NATIVE SOURCE BUNDLE READY: {out}")


def verify(args: argparse.Namespace) -> None:
    manifest = validate_manifest(
        read_json(args.manifest.expanduser().resolve(), "source bundle manifest")
    )
    verify_archive(manifest, args.archive.expanduser().resolve())
    print(f"FERRUM NATIVE SOURCE BUNDLE VERIFIED: {args.archive.expanduser().resolve()}")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-native-source-bundle-") as raw:
        root = Path(raw)
        source = root / "source"
        definitions = root / "definitions"
        source.mkdir()
        definitions.mkdir()
        (source / "kernels").mkdir()
        (source / "kernels/a.cu").write_text("extern \"C\" __global__ void a() {}\n")
        (source / "kernels/a.cuh").write_text("#pragma once\n")
        write_json_create_new(
            definitions / "a.json",
            {
                "schema_version": 3,
                "operator": "ferrum.cuda.fixture",
                "translation_units": ["kernels/a.cu"],
                "headers": ["kernels/a.cuh"],
            },
        )
        archive = root / "fixture.tar.gz"
        manifest_path = root / "fixture.json"
        args = argparse.Namespace(
            source_root=source,
            definition_root=definitions,
            archive=archive,
            manifest=manifest_path,
            github_repository="owner/repo",
            github_tag="fixture",
        )
        create(args)
        manifest = validate_manifest(read_json(manifest_path, "fixture manifest"))
        verify_archive(manifest, archive)
        extracted = root / "extracted"
        extract_archive(manifest, archive, extracted)
        verify_materialized_tree(manifest, extracted)
        require(
            (extracted / "kernels/a.cu").read_bytes()
            == (source / "kernels/a.cu").read_bytes(),
            "materialized fixture differs",
        )
        ensure_materialized(
            argparse.Namespace(
                manifest=manifest_path,
                cache=root / "cache",
                out=extracted,
            )
        )
        (extracted / "unexpected.cu").write_text("// unexpected\n", encoding="ascii")
        try:
            verify_materialized_tree(manifest, extracted)
        except BundleError:
            pass
        else:
            raise BundleError("materialized source tree accepted an unexpected member")
        tampered = root / "tampered.tar.gz"
        shutil.copy2(archive, tampered)
        with tampered.open("ab") as handle:
            handle.write(b"tamper")
        try:
            verify_archive(manifest, tampered)
        except BundleError:
            pass
        else:
            raise BundleError("tampered archive was accepted")
    print(SELFTEST_PASS_LINE)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subcommands = result.add_subparsers(dest="command", required=True)
    create_parser = subcommands.add_parser("create")
    create_parser.add_argument("--source-root", type=Path, required=True)
    create_parser.add_argument("--definition-root", type=Path, required=True)
    create_parser.add_argument("--archive", type=Path, required=True)
    create_parser.add_argument("--manifest", type=Path, required=True)
    create_parser.add_argument("--github-repository", required=True)
    create_parser.add_argument("--github-tag", required=True)
    create_parser.set_defaults(action=create)
    verify_parser = subcommands.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--archive", type=Path, required=True)
    verify_parser.set_defaults(action=verify)
    materialize_parser = subcommands.add_parser("materialize")
    materialize_parser.add_argument("--manifest", type=Path, required=True)
    materialize_parser.add_argument("--archive", type=Path)
    materialize_parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/ferrum/native-sources")
    materialize_parser.add_argument("--out", type=Path, required=True)
    materialize_parser.set_defaults(action=materialize)
    ensure_parser = subcommands.add_parser("ensure")
    ensure_parser.add_argument("--manifest", type=Path, required=True)
    ensure_parser.add_argument(
        "--cache",
        type=Path,
        default=Path.home() / ".cache/ferrum/native-sources",
    )
    ensure_parser.add_argument("--out", type=Path, required=True)
    ensure_parser.set_defaults(action=ensure_materialized)
    selftest_parser = subcommands.add_parser("self-test")
    selftest_parser.set_defaults(action=lambda _args: self_test())
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        args.action(args)
        return 0
    except (BundleError, OSError, subprocess.SubprocessError, tarfile.TarError) as error:
        print(f"FERRUM NATIVE SOURCE BUNDLE REJECT: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
