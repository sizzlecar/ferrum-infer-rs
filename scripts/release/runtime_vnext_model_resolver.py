#!/usr/bin/env python3
"""Resolve the runtime-vNext model catalog to immutable Hugging Face files.

The resolver deliberately does not download model weights. Hugging Face LFS OIDs
are content SHA256 digests, so only bounded metadata files are fetched. Some
safetensors index files are stored in LFS; those are downloaded only after path,
size, and content-SHA checks prove that they are metadata rather than weights.
"""

from __future__ import annotations

import argparse
import base64
import copy
import datetime as dt
import fnmatch
import hashlib
import http.client
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "scripts/release/configs/runtime_vnext_models.json"
HF_BASE = "https://huggingface.co"
OUTPUT_NAME = "model-resolution.json"
MAX_METADATA_BYTES = 32 * 1024 * 1024
LFS_METADATA_SUFFIXES = (".safetensors.index.json",)
MAX_TREE_PAGES = 100
PROVENANCE_BODY_KINDS = {"model-info", "repo-tree"}
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFETENSORS_SHARD_RE = re.compile(
    r"-(\d{5,6})-of-(\d{5,6})\.safetensors$"
)
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MOVING_REVISIONS = {"main", "master", "latest", "head", "trunk"}
DEFAULT_SEMANTIC_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
]
DEFAULT_TOKENIZER_FILES = ["tokenizer_config.json", "tokenizer.json"]
OPTIONAL_TOKENIZER_FILES = [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
]


class ResolutionError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResolutionError(message)


def require_object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return value


def require_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be a non-empty string")
    return value


def reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def strict_json_loads(payload: str | bytes) -> Any:
    return json.loads(
        payload,
        object_pairs_hook=unique_json_object,
        parse_constant=reject_json_constant,
    )


def require_repo(value: Any, label: str) -> str:
    repo = require_string(value, label)
    require(REPO_RE.fullmatch(repo) is not None, f"{label} is not an owner/name Hugging Face repo")
    return repo


def normalize_sha256(value: Any, label: str) -> str:
    digest = require_string(value, label).lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    require(SHA256_RE.fullmatch(digest) is not None, f"{label} must be a SHA256 digest")
    return digest


def catalog_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def quote_repo(repo: str) -> str:
    return urllib.parse.quote(repo, safe="/")


def model_api_url(repo: str, revision: str | None) -> str:
    base = f"{HF_BASE}/api/models/{quote_repo(repo)}"
    if revision is None:
        return base
    return f"{base}/revision/{urllib.parse.quote(revision, safe='')}"


def tree_api_url(repo: str, revision: str) -> str:
    query = urllib.parse.urlencode({"recursive": "true", "expand": "true"})
    return f"{HF_BASE}/api/models/{quote_repo(repo)}/tree/{revision}?{query}"


def resolve_file_url(repo: str, revision: str, path: str) -> str:
    return (
        f"{HF_BASE}/{quote_repo(repo)}/resolve/{revision}/"
        f"{urllib.parse.quote(path, safe='/')}"
    )


def next_link(value: str | None, current_url: str) -> str | None:
    if not value:
        return None
    for part in value.split(","):
        match = re.match(r'\s*<([^>]+)>\s*;\s*rel="?next"?', part, re.IGNORECASE)
        if match:
            return urllib.parse.urljoin(current_url, match.group(1))
    return None


def validate_metadata_content_length(value: str | None, url: str) -> None:
    if value is None:
        return
    try:
        size = int(value)
    except ValueError as exc:
        raise ResolutionError(f"invalid metadata Content-Length for {url}: {value!r}") from exc
    require(size >= 0, f"invalid metadata Content-Length for {url}: {value!r}")
    require(
        size <= MAX_METADATA_BYTES,
        f"metadata response exceeds download limit ({MAX_METADATA_BYTES} bytes): {url}",
    )


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict[str, str]
    body: bytes


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Avoid forwarding a Hugging Face bearer token to signed CDN hosts."""

    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        old_host = urllib.parse.urlsplit(req.full_url).netloc
        new_host = urllib.parse.urlsplit(newurl).netloc
        if old_host != new_host:
            redirected.remove_header("Authorization")
            redirected.unredirected_hdrs.pop("Authorization", None)
        return redirected


class Transport:
    provenance = "abstract"

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def fetch(self, url: str, kind: str) -> Response:
        raise NotImplementedError

    def fetch_json(self, url: str, kind: str) -> tuple[Any, Response]:
        response = self.fetch(url, kind)
        try:
            return strict_json_loads(response.body), response
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ResolutionError(f"{kind} returned invalid JSON from {url}: {exc}") from exc

    def record(self, url: str, kind: str, response: Response) -> None:
        row = {
            "method": "GET",
            "kind": kind,
            "url": url,
            "status": response.status,
            "response_bytes": len(response.body),
            "response_sha256": hashlib.sha256(response.body).hexdigest(),
        }
        if kind in PROVENANCE_BODY_KINDS:
            row["response_body_base64"] = base64.b64encode(response.body).decode("ascii")
        self.requests.append(row)


class NetworkTransport(Transport):
    provenance = "network_huggingface_https"

    def __init__(self, *, timeout: float = 45.0, retries: int = 2) -> None:
        super().__init__()
        self.timeout = timeout
        self.retries = retries
        self.token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.opener = urllib.request.build_opener(SafeRedirectHandler())

    def fetch(self, url: str, kind: str) -> Response:
        require(url.startswith(f"{HF_BASE}/"), f"refusing non-Hugging-Face request URL: {url}")
        headers = {
            "Accept": "application/json" if kind != "metadata-file" else "application/octet-stream",
            "Accept-Encoding": "identity",
            "User-Agent": "ferrum-runtime-vnext-model-resolver/1",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(url, headers=headers, method="GET")
            try:
                with self.opener.open(request, timeout=self.timeout) as handle:
                    response_headers = {
                        key.lower(): value for key, value in handle.headers.items()
                    }
                    if kind == "metadata-file":
                        validate_metadata_content_length(
                            response_headers.get("content-length"),
                            url,
                        )
                        body = handle.read(MAX_METADATA_BYTES + 1)
                        require(
                            len(body) <= MAX_METADATA_BYTES,
                            f"metadata response exceeds download limit ({MAX_METADATA_BYTES} bytes): {url}",
                        )
                    else:
                        body = handle.read()
                    response = Response(
                        status=int(handle.status),
                        headers=response_headers,
                        body=body,
                    )
                require(200 <= response.status < 300, f"HTTP {response.status} for {kind}: {url}")
                self.record(url, kind, response)
                return response
            except urllib.error.HTTPError as exc:
                last_error = exc
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self.retries:
                    raise ResolutionError(f"HTTP {exc.code} for {kind}: {url}") from exc
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                http.client.IncompleteRead,
            ) as exc:
                last_error = exc
                if attempt == self.retries:
                    raise ResolutionError(f"network error for {kind}: {url}: {type(exc).__name__}") from exc
            time.sleep(0.5 * (2**attempt))
        raise ResolutionError(f"request failed for {kind}: {url}: {type(last_error).__name__}")


class FixtureTransport(Transport):
    provenance = "internal_selftest_fixture"

    def __init__(self, fixture_path: Path) -> None:
        super().__init__()
        source = fixture_path / "responses.json" if fixture_path.is_dir() else fixture_path
        try:
            data = strict_json_loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ResolutionError(f"invalid offline fixture {source}: {exc}") from exc
        require_object(data, "offline fixture")
        require(data.get("schema_version") == 1, "offline fixture schema_version must be 1")
        self.responses = require_object(data.get("responses"), "offline fixture.responses")

    def fetch(self, url: str, kind: str) -> Response:
        raw = self.responses.get(url)
        require(raw is not None, f"offline fixture has no response for {url}")
        item = require_object(raw, f"offline fixture response {url}")
        status = item.get("status", 200)
        require(isinstance(status, int) and not isinstance(status, bool), f"fixture status for {url} must be an integer")
        headers_raw = require_object(item.get("headers", {}), f"fixture headers for {url}")
        headers = {str(key).lower(): str(value) for key, value in headers_raw.items()}
        body_fields = [key for key in ("json", "text", "base64") if key in item]
        require(len(body_fields) == 1, f"fixture response {url} must have exactly one body field")
        field = body_fields[0]
        if field == "json":
            body = json.dumps(
                item[field],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        elif field == "text":
            require(isinstance(item[field], str), f"fixture text body for {url} must be a string")
            body = item[field].encode("utf-8")
        else:
            require(isinstance(item[field], str), f"fixture base64 body for {url} must be a string")
            try:
                body = base64.b64decode(item[field], validate=True)
            except ValueError as exc:
                raise ResolutionError(f"invalid fixture base64 body for {url}") from exc
        if kind == "metadata-file":
            validate_metadata_content_length(headers.get("content-length"), url)
            require(
                len(body) <= MAX_METADATA_BYTES,
                f"metadata response exceeds download limit ({MAX_METADATA_BYTES} bytes): {url}",
            )
        response = Response(status=status, headers=headers, body=body)
        self.record(url, kind, response)
        require(200 <= status < 300, f"HTTP {status} for {kind}: {url}")
        return response


@dataclass
class TreeFile:
    path: str
    size: int
    oid: str
    lfs_oid: str | None
    lfs_oid_redacted: bool = False


@dataclass
class Snapshot:
    repo: str
    revision: str
    requested_revision: dict[str, Any]
    model_url: str
    tree_urls: list[str]
    tree: dict[str, TreeFile]
    license_id: str | None
    gated: bool | str | None


class Resolver:
    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        self.snapshot_cache: dict[tuple[str, str, str | None], Snapshot] = {}
        self.file_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.content_cache: dict[tuple[str, str, str], bytes] = {}

    def revision_request(self, raw: Any, label: str) -> tuple[str, str | None]:
        revision = require_object(raw, label)
        status = revision.get("status")
        value = revision.get("value")
        require(status in {"pinned", "resolution_required"}, f"{label}.status is unsupported")
        if status == "pinned":
            pinned = require_string(value, f"{label}.value").lower()
            require(pinned not in MOVING_REVISIONS, f"{label}.value must not be a moving revision")
            require(GIT_SHA_RE.fullmatch(pinned) is not None, f"{label}.value must be a full immutable commit")
            return status, pinned
        require(value is None, f"{label}.value must be null until resolution")
        return status, None

    def snapshot(
        self,
        repo_raw: Any,
        revision_raw: Any,
        label: str,
        *,
        forced_revision: str | None = None,
    ) -> Snapshot:
        repo = require_repo(repo_raw, f"{label}.repo")
        if forced_revision is None:
            status, requested = self.revision_request(revision_raw, f"{label}.revision")
        else:
            require(GIT_SHA_RE.fullmatch(forced_revision) is not None, f"{label} forced revision is not immutable")
            status, requested = "same_as_weight_revision", forced_revision
        cache_key = (repo, status, requested)
        cached = self.snapshot_cache.get(cache_key)
        if cached is not None:
            return cached

        api_requested = requested if status in {"pinned", "same_as_weight_revision"} else None
        info_url = model_api_url(repo, api_requested)
        info_raw, _ = self.transport.fetch_json(info_url, "model-info")
        info = require_object(info_raw, f"model info for {repo}")
        resolved = require_string(info.get("sha"), f"model info {repo}.sha").lower()
        require(GIT_SHA_RE.fullmatch(resolved) is not None, f"model info {repo}.sha is not a full immutable commit")
        if requested is not None:
            require(resolved == requested, f"{repo} resolved to {resolved}, expected pinned {requested}")

        tree, tree_urls = self.load_tree(repo, resolved)
        card_data = info.get("cardData")
        license_id: str | None = None
        if isinstance(card_data, dict):
            raw_license = card_data.get("license")
            if isinstance(raw_license, str) and raw_license.strip():
                license_id = raw_license.strip()
            elif isinstance(raw_license, list) and all(isinstance(item, str) for item in raw_license):
                license_id = ",".join(raw_license)
        snapshot = Snapshot(
            repo=repo,
            revision=resolved,
            requested_revision={"status": status, "value": requested},
            model_url=info_url,
            tree_urls=tree_urls,
            tree=tree,
            license_id=license_id,
            gated=info.get("gated") if isinstance(info.get("gated"), (bool, str)) else None,
        )
        self.snapshot_cache[cache_key] = snapshot
        return snapshot

    def load_tree(self, repo: str, revision: str) -> tuple[dict[str, TreeFile], list[str]]:
        url: str | None = tree_api_url(repo, revision)
        seen_urls: set[str] = set()
        files: dict[str, TreeFile] = {}
        request_urls: list[str] = []
        while url is not None:
            require(url not in seen_urls, f"tree pagination cycle for {repo}@{revision}")
            require(len(seen_urls) < MAX_TREE_PAGES, f"tree pagination exceeded {MAX_TREE_PAGES} pages for {repo}@{revision}")
            seen_urls.add(url)
            request_urls.append(url)
            raw, response = self.transport.fetch_json(url, "repo-tree")
            for index, entry_raw in enumerate(require_list(raw, f"tree {repo}@{revision}")):
                entry = require_object(entry_raw, f"tree {repo}@{revision}[{index}]")
                if entry.get("type") not in {"file", None}:
                    continue
                path = require_string(entry.get("path"), f"tree {repo}@{revision}[{index}].path")
                require(not path.startswith("/") and ".." not in Path(path).parts, f"unsafe tree path {path!r}")
                require(path not in files, f"duplicate tree path {path!r} for {repo}@{revision}")
                size = entry.get("size")
                require(isinstance(size, int) and not isinstance(size, bool) and size >= 0, f"invalid size for {repo}/{path}")
                oid = require_string(entry.get("oid"), f"tree {repo}/{path}.oid")
                lfs_oid: str | None = None
                lfs_oid_redacted = False
                if entry.get("lfs") is not None:
                    lfs = require_object(entry.get("lfs"), f"tree {repo}/{path}.lfs")
                    raw_lfs_oid = require_string(lfs.get("oid"), f"tree {repo}/{path}.lfs.oid")
                    if set(raw_lfs_oid) == {"*"}:
                        lfs_oid_redacted = True
                    else:
                        lfs_oid = normalize_sha256(raw_lfs_oid, f"tree {repo}/{path}.lfs.oid")
                    lfs_size = lfs.get("size")
                    require(
                        isinstance(lfs_size, int) and not isinstance(lfs_size, bool) and lfs_size > 0,
                        f"invalid LFS size for {repo}/{path}",
                    )
                    require(lfs_size == size, f"tree/LFS size mismatch for {repo}/{path}")
                files[path] = TreeFile(
                    path=path,
                    size=size,
                    oid=oid,
                    lfs_oid=lfs_oid,
                    lfs_oid_redacted=lfs_oid_redacted,
                )
            url = next_link(response.headers.get("link"), url)
        require(files, f"empty repository tree for {repo}@{revision}")
        return files, request_urls

    def locked_file(self, snapshot: Snapshot, path: str) -> dict[str, Any]:
        cache_key = (snapshot.repo, snapshot.revision, path)
        cached = self.file_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)
        entry = snapshot.tree.get(path)
        require(entry is not None, f"missing file {snapshot.repo}@{snapshot.revision}/{path}")
        require(entry.size > 0, f"selected file has invalid zero size: {snapshot.repo}/{path}")
        row: dict[str, Any] = {
            "path": path,
            "size_bytes": entry.size,
            "git_oid": entry.oid,
        }
        require(
            not entry.lfs_oid_redacted,
            f"selected LFS SHA256 is redacted for gated file {snapshot.repo}/{path}; provide HF_TOKEN",
        )
        if entry.lfs_oid is not None:
            row.update(
                {
                    "sha256": entry.lfs_oid,
                    "sha256_source": "hugging_face_lfs_oid",
                    "lfs_oid": entry.lfs_oid,
                }
            )
        else:
            require(
                entry.size <= MAX_METADATA_BYTES,
                f"non-LFS selected file exceeds metadata limit ({MAX_METADATA_BYTES} bytes): {snapshot.repo}/{path}",
            )
            url = resolve_file_url(snapshot.repo, snapshot.revision, path)
            response = self.transport.fetch(url, "metadata-file")
            require(
                len(response.body) == entry.size,
                f"download/tree size mismatch for {snapshot.repo}/{path}: {len(response.body)} != {entry.size}",
            )
            row.update(
                {
                    "sha256": hashlib.sha256(response.body).hexdigest(),
                    "sha256_source": "downloaded_content",
                    "content_request_url": url,
                }
            )
            self.content_cache[cache_key] = response.body
        self.file_cache[cache_key] = row
        return copy.deepcopy(row)

    def file_content(self, snapshot: Snapshot, path: str, label: str) -> bytes:
        cache_key = (snapshot.repo, snapshot.revision, path)
        self.locked_file(snapshot, path)
        content = self.content_cache.get(cache_key)
        require(content is not None, f"{label} must be a downloaded non-LFS metadata file")
        return content

    def safetensors_index_content(self, snapshot: Snapshot, path: str, label: str) -> bytes:
        require(
            any(path.endswith(suffix) for suffix in LFS_METADATA_SUFFIXES),
            f"{label} is not an allowed safetensors index metadata path",
        )
        cache_key = (snapshot.repo, snapshot.revision, path)
        row = self.locked_file(snapshot, path)
        content = self.content_cache.get(cache_key)
        if content is not None:
            return content
        entry = snapshot.tree[path]
        require(entry.lfs_oid is not None, f"{label} metadata content is unavailable")
        require(
            entry.size <= MAX_METADATA_BYTES,
            f"LFS metadata exceeds metadata limit ({MAX_METADATA_BYTES} bytes): {snapshot.repo}/{path}",
        )
        url = resolve_file_url(snapshot.repo, snapshot.revision, path)
        response = self.transport.fetch(url, "metadata-file")
        require(
            len(response.body) == entry.size,
            f"download/tree size mismatch for {snapshot.repo}/{path}: {len(response.body)} != {entry.size}",
        )
        digest = hashlib.sha256(response.body).hexdigest()
        require(
            digest == entry.lfs_oid,
            f"downloaded LFS metadata SHA256 differs from tree LFS OID: {snapshot.repo}/{path}",
        )
        self.content_cache[cache_key] = response.body
        row["content_request_url"] = url
        row["lfs_metadata_downloaded"] = True
        self.file_cache[cache_key] = row
        return response.body

    def validate_expected_files(
        self,
        files: list[dict[str, Any]],
        expected_raw: Any,
        label: str,
    ) -> None:
        expected = require_object(expected_raw, label)
        actual = {str(row["path"]): str(row["sha256"]) for row in files}
        require(set(expected) == set(actual), f"{label} paths must exactly match the locked files")
        for path, digest_raw in expected.items():
            digest = normalize_sha256(digest_raw, f"{label}.{path}")
            require(actual[path] == digest, f"{label} content SHA256 mismatch for {path}")

    def json_string_at_pointer(
        self,
        content: bytes,
        pointer: str,
        label: str,
    ) -> str:
        require(pointer.startswith("/"), f"{label}.json_pointer must be an absolute JSON pointer")
        try:
            value: Any = strict_json_loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ResolutionError(f"{label} container is not valid UTF-8 JSON: {exc}") from exc
        for raw_part in pointer[1:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            require(isinstance(value, dict) and part in value, f"{label} JSON pointer {pointer!r} is absent")
            value = value[part]
        require(isinstance(value, str) and bool(value), f"{label} JSON pointer {pointer!r} must select a non-empty string")
        return value

    def select_catalog_files(self, snapshot: Snapshot, specs_raw: Any, label: str) -> list[dict[str, Any]]:
        specs = require_list(specs_raw, f"{label}.files")
        selected: set[str] = set()
        conditional: list[tuple[str, dict[str, Any]]] = []
        for index, spec_raw in enumerate(specs):
            spec = require_object(spec_raw, f"{label}.files[{index}]")
            has_path = "path" in spec
            has_glob = "glob" in spec
            require(has_path != has_glob, f"{label}.files[{index}] needs exactly one of path or glob")
            if has_path:
                expression = require_string(spec.get("path"), f"{label}.files[{index}].path")
                require(not expression.startswith("/") and ".." not in Path(expression).parts, f"unsafe catalog path {expression!r}")
                matches = [expression] if expression in snapshot.tree else []
            else:
                expression = require_string(spec.get("glob"), f"{label}.files[{index}].glob")
                require(not expression.startswith("/") and ".." not in Path(expression).parts, f"unsafe catalog glob {expression!r}")
                matches = sorted(path for path in snapshot.tree if fnmatch.fnmatchcase(path, expression))
            if spec.get("required") is True:
                require(matches, f"{label} required file selection {expression!r} matched nothing")
            conditional_only = spec.get("required_if_sharded") is True
            if conditional_only:
                require(has_path, f"{label}.files[{index}] required_if_sharded must use path")
                require(
                    spec.get("required") is not True,
                    f"{label}.files[{index}] must not combine required and required_if_sharded",
                )
                conditional.append((expression, spec))
            expected_sha256 = spec.get("expected_sha256")
            if expected_sha256 is not None:
                require(has_path, f"{label}.files[{index}].expected_sha256 requires an exact path selector")
                require(spec.get("required") is True, f"{label}.files[{index}].expected_sha256 requires required=true")
                expected_sha256 = normalize_sha256(
                    expected_sha256,
                    f"{label}.files[{index}].expected_sha256",
                )
            for path in matches:
                expected_size = spec.get("expected_size_bytes")
                if expected_size is not None:
                    require(
                        isinstance(expected_size, int) and not isinstance(expected_size, bool) and expected_size > 0,
                        f"{label}.files[{index}].expected_size_bytes must be positive",
                    )
                    require(snapshot.tree[path].size == expected_size, f"{label} expected size mismatch for {path}")
                if expected_sha256 is not None:
                    entry = snapshot.tree[path]
                    require(
                        not entry.lfs_oid_redacted and entry.lfs_oid is not None,
                        f"{label} expected SHA256 for {path} requires a visible Hugging Face LFS OID",
                    )
                    require(
                        entry.lfs_oid == expected_sha256,
                        f"{label} expected SHA256 mismatch for {path}",
                    )
                if not conditional_only:
                    selected.add(path)
        safetensors = [path for path in selected if path.endswith(".safetensors")]
        is_sharded = len(safetensors) > 1 or any(
            SAFETENSORS_SHARD_RE.search(path) for path in safetensors
        )
        if is_sharded:
            for path, _ in conditional:
                require(path in snapshot.tree, f"{label} sharded model is missing required {path}")
                selected.add(path)
            index_paths = sorted(
                path
                for path, _ in conditional
                if path.endswith(".safetensors.index.json")
            )
            require(
                len(index_paths) == 1,
                f"{label} sharded safetensors must declare exactly one required index",
            )
            self.validate_safetensors_shards(
                snapshot,
                index_paths[0],
                set(safetensors),
                label,
            )
        else:
            require(
                not any(path in selected for path, _ in conditional),
                f"{label} unsharded model selected a conditional index",
            )
        require(selected, f"{label} selected no files")
        return [self.locked_file(snapshot, path) for path in sorted(selected)]

    def validate_safetensors_shards(
        self,
        snapshot: Snapshot,
        index_path: str,
        selected_shards: set[str],
        label: str,
    ) -> None:
        content = self.safetensors_index_content(
            snapshot,
            index_path,
            f"{label} safetensors index",
        )
        try:
            document = strict_json_loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ResolutionError(f"{label} safetensors index is invalid JSON: {exc}") from exc
        index = require_object(document, f"{label} safetensors index")
        weight_map = require_object(index.get("weight_map"), f"{label} safetensors index.weight_map")
        require(weight_map, f"{label} safetensors index.weight_map must not be empty")
        expected_shards: set[str] = set()
        for tensor, raw_path in weight_map.items():
            require_string(tensor, f"{label} safetensors index tensor")
            shard = require_string(raw_path, f"{label} safetensors index shard")
            require(
                not shard.startswith("/")
                and ".." not in Path(shard).parts
                and shard.endswith(".safetensors"),
                f"{label} safetensors index contains an unsafe shard path",
            )
            expected_shards.add(shard)
        require(
            expected_shards == selected_shards,
            f"{label} safetensors index shard set differs from selected weight shards",
        )
        numbered: list[tuple[int, int, int, int, str]] = []
        for shard in sorted(expected_shards):
            match = SAFETENSORS_SHARD_RE.search(shard)
            require(match is not None, f"{label} sharded safetensors path lacks canonical numbering: {shard}")
            numbered.append(
                (
                    int(match.group(1)),
                    int(match.group(2)),
                    len(match.group(1)),
                    len(match.group(2)),
                    shard,
                )
            )
        require(
            len({number_width for _, _, number_width, _, _ in numbered}) == 1,
            f"{label} sharded safetensors number width differs",
        )
        require(
            len({total_width for _, _, _, total_width, _ in numbered}) == 1,
            f"{label} sharded safetensors total width differs",
        )
        totals = {total for _, total, _, _, _ in numbered}
        require(len(totals) == 1, f"{label} safetensors shards disagree on total count")
        total = totals.pop()
        require(total == len(numbered), f"{label} safetensors shard count differs from numbered total")
        require(
            {number for number, _, _, _, _ in numbered} == set(range(1, total + 1)),
            f"{label} safetensors shard numbering is incomplete",
        )

    def exact_files(
        self,
        snapshot: Snapshot,
        required_paths: list[str],
        label: str,
        *,
        optional_paths: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        selected: set[str] = set()
        for path in required_paths:
            require(isinstance(path, str) and path, f"{label} contains an invalid required path")
            require(path in snapshot.tree, f"{label} missing required {path}")
            selected.add(path)
        for path in optional_paths or []:
            if path in snapshot.tree:
                selected.add(path)
        return [self.locked_file(snapshot, path) for path in sorted(selected)]

    def license_files(self, snapshot: Snapshot) -> list[dict[str, Any]]:
        selected = []
        for path in sorted(snapshot.tree):
            basename = Path(path).name.lower()
            if basename in {"license", "license.md", "license.txt", "copying", "notice", "notice.txt"}:
                entry = snapshot.tree[path]
                if entry.lfs_oid is not None or entry.size <= MAX_METADATA_BYTES:
                    selected.append(self.locked_file(snapshot, path))
        return selected

    def source_json(self, snapshot: Snapshot, files: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "repo": snapshot.repo,
            "requested_revision": snapshot.requested_revision,
            "revision": snapshot.revision,
            "model_request_url": snapshot.model_url,
            "tree_request_urls": snapshot.tree_urls,
            "files": files,
            "license": {
                "hugging_face_id": snapshot.license_id,
                "files": self.license_files(snapshot),
            },
            "gated": snapshot.gated,
        }

    def lane(self, lane_raw: Any, index: int) -> dict[str, Any]:
        lane = require_object(lane_raw, f"catalog.models[{index}]")
        lane_id = require_string(lane.get("id"), f"catalog.models[{index}].id")
        weight = self.snapshot(lane.get("repo"), lane.get("revision"), f"catalog lane {lane_id}")
        weight_files = self.select_catalog_files(weight, lane.get("files"), f"catalog lane {lane_id}")

        reference = require_object(lane.get("reference"), f"catalog lane {lane_id}.reference")
        semantic_repo = require_repo(reference.get("semantic_repo"), f"catalog lane {lane_id}.reference.semantic_repo")
        semantic_rule = require_object(
            reference.get("semantic_revision"),
            f"catalog lane {lane_id}.reference.semantic_revision",
        )
        if semantic_rule.get("status") == "same_as_weight_revision":
            require(semantic_repo == weight.repo, f"catalog lane {lane_id} same_as_weight_revision uses another repo")
            semantic = weight
        else:
            semantic = self.snapshot(
                semantic_repo,
                semantic_rule,
                f"catalog lane {lane_id}.semantic_source",
            )
        semantic_required_raw = reference.get("required_semantic_files", DEFAULT_SEMANTIC_FILES)
        semantic_required = [
            require_string(path, f"catalog lane {lane_id}.required_semantic_files")
            for path in require_list(semantic_required_raw, f"catalog lane {lane_id}.required_semantic_files")
        ]
        semantic_files = self.exact_files(
            semantic,
            semantic_required,
            f"catalog lane {lane_id}.semantic_source",
        )
        self.validate_expected_files(
            semantic_files,
            reference.get("semantic_file_sha256"),
            f"catalog lane {lane_id}.reference.semantic_file_sha256",
        )

        result: dict[str, Any] = {
            "catalog_lane_id": lane_id,
            "model_id": require_string(lane.get("model_id"), f"catalog lane {lane_id}.model_id"),
            "backend": require_string(lane.get("backend"), f"catalog lane {lane_id}.backend"),
            "format": require_string(lane.get("format"), f"catalog lane {lane_id}.format"),
            "weight_source": self.source_json(weight, weight_files),
            "semantic_source": self.source_json(semantic, semantic_files),
        }
        generation_rule = require_object(
            reference.get("generation_config_source"),
            f"catalog lane {lane_id}.reference.generation_config_source",
        )
        require(
            generation_rule.get("source") == "semantic_source",
            f"catalog lane {lane_id}.generation_config_source.source must be semantic_source",
        )
        generation_path = require_string(
            generation_rule.get("path"),
            f"catalog lane {lane_id}.generation_config_source.path",
        )
        generation_policy = generation_rule.get("policy")
        require(
            generation_policy in {"required", "absent"},
            f"catalog lane {lane_id}.generation_config_source.policy must be required or absent",
        )
        semantic_by_path = {row["path"]: row for row in semantic_files}
        if generation_policy == "required":
            generation_file = semantic_by_path.get(generation_path)
            require(
                generation_file is not None,
                f"catalog lane {lane_id} required generation config is not locked",
            )
            expected_generation_sha = normalize_sha256(
                generation_rule.get("content_sha256"),
                f"catalog lane {lane_id}.generation_config_source.content_sha256",
            )
            require(
                generation_file["sha256"] == expected_generation_sha,
                f"catalog lane {lane_id} generation config content SHA256 mismatch",
            )
            result["generation_config"] = {
                "source": "semantic_source",
                "repo": semantic.repo,
                "revision": semantic.revision,
                "path": generation_path,
                "policy": "required",
                "present": True,
                "file": copy.deepcopy(generation_file),
            }
        else:
            require(
                generation_path not in semantic.tree,
                f"catalog lane {lane_id} generation config declared absent but exists at the pinned revision",
            )
            require(
                "content_sha256" not in generation_rule,
                f"catalog lane {lane_id} absent generation config must not declare content_sha256",
            )
            result["generation_config"] = {
                "source": "semantic_source",
                "repo": semantic.repo,
                "revision": semantic.revision,
                "path": generation_path,
                "policy": "absent",
                "present": False,
            }
        if reference.get("official_upstream") is not None:
            upstream_rule = require_object(
                reference.get("official_upstream"),
                f"catalog lane {lane_id}.reference.official_upstream",
            )
            upstream = self.snapshot(
                upstream_rule.get("repo"),
                upstream_rule.get("revision"),
                f"catalog lane {lane_id}.official_upstream",
            )
            match_files = [
                require_string(path, f"catalog lane {lane_id}.official_upstream.blob_oid_match_files")
                for path in require_list(
                    upstream_rule.get("blob_oid_match_files"),
                    f"catalog lane {lane_id}.official_upstream.blob_oid_match_files",
                )
            ]
            require(match_files, f"catalog lane {lane_id}.official_upstream must match at least one file")
            expected_git_oids = require_object(
                upstream_rule.get("expected_git_oids"),
                f"catalog lane {lane_id}.official_upstream.expected_git_oids",
            )
            expected_content_sha256 = require_object(
                upstream_rule.get("expected_content_sha256"),
                f"catalog lane {lane_id}.official_upstream.expected_content_sha256",
            )
            expected_size_bytes = require_object(
                upstream_rule.get("expected_size_bytes"),
                f"catalog lane {lane_id}.official_upstream.expected_size_bytes",
            )
            expected_paths = set(match_files)
            require(
                set(expected_git_oids) == expected_paths
                and set(expected_content_sha256) == expected_paths
                and set(expected_size_bytes) == expected_paths,
                f"catalog lane {lane_id}.official_upstream expected match maps must cover exactly blob_oid_match_files",
            )
            require(
                upstream_rule.get("required_gated") is True,
                f"catalog lane {lane_id}.official_upstream.required_gated must be true",
            )
            require(
                upstream.gated not in {None, False},
                f"catalog lane {lane_id} official upstream is expected to be gated",
            )
            matches: list[dict[str, Any]] = []
            for path in match_files:
                mirror_file = semantic_by_path.get(path)
                require(mirror_file is not None, f"catalog lane {lane_id} mirror did not lock {path}")
                official_file = upstream.tree.get(path)
                require(official_file is not None, f"catalog lane {lane_id} official upstream missing {path}")
                require(
                    mirror_file.get("git_oid") == official_file.oid,
                    f"catalog lane {lane_id} mirror Git blob differs from official upstream for {path}",
                )
                require(
                    mirror_file.get("size_bytes") == official_file.size,
                    f"catalog lane {lane_id} mirror size differs from official upstream for {path}",
                )
                expected_oid = require_string(
                    expected_git_oids.get(path),
                    f"catalog lane {lane_id}.official_upstream.expected_git_oids.{path}",
                )
                require(
                    re.fullmatch(r"[0-9a-f]{40}", expected_oid) is not None,
                    f"catalog lane {lane_id}.official_upstream expected Git OID is invalid for {path}",
                )
                expected_sha = normalize_sha256(
                    expected_content_sha256.get(path),
                    f"catalog lane {lane_id}.official_upstream.expected_content_sha256.{path}",
                )
                expected_size = expected_size_bytes.get(path)
                require(
                    isinstance(expected_size, int) and not isinstance(expected_size, bool) and expected_size > 0,
                    f"catalog lane {lane_id}.official_upstream expected size is invalid for {path}",
                )
                require(
                    official_file.oid == expected_oid
                    and mirror_file["sha256"] == expected_sha
                    and official_file.size == expected_size,
                    f"catalog lane {lane_id} official upstream frozen evidence mismatch for {path}",
                )
                matches.append(
                    {
                        "path": path,
                        "git_oid": official_file.oid,
                        "size_bytes": official_file.size,
                        "content_sha256": mirror_file["sha256"],
                    }
                )
            result["official_upstream"] = {
                "repo": upstream.repo,
                "revision": upstream.revision,
                "mirror_repo": semantic.repo,
                "mirror_revision": semantic.revision,
                "model_request_url": upstream.model_url,
                "tree_request_urls": upstream.tree_urls,
                "gated": upstream.gated,
                "verification_method": "mirror_content_sha256_and_official_git_blob_oid",
                "mirror_blob_oid_matches": matches,
                "access_note": require_string(
                    upstream_rule.get("access_note"),
                    f"catalog lane {lane_id}.official_upstream.access_note",
                ),
            }
        if reference.get("tokenizer_repo") is not None:
            tokenizer_rule = require_object(
                reference.get("tokenizer_revision"),
                f"catalog lane {lane_id}.reference.tokenizer_revision",
            )
            tokenizer = self.snapshot(
                reference.get("tokenizer_repo"),
                tokenizer_rule,
                f"catalog lane {lane_id}.tokenizer_source",
            )
            tokenizer_required = [
                require_string(path, f"catalog lane {lane_id}.required_tokenizer_files")
                for path in require_list(
                    reference.get("required_tokenizer_files"),
                    f"catalog lane {lane_id}.required_tokenizer_files",
                )
            ]
            tokenizer_files = self.exact_files(
                tokenizer,
                tokenizer_required,
                f"catalog lane {lane_id}.tokenizer_source",
            )
            self.validate_expected_files(
                tokenizer_files,
                reference.get("tokenizer_file_sha256"),
                f"catalog lane {lane_id}.reference.tokenizer_file_sha256",
            )
            result["tokenizer_source"] = self.source_json(tokenizer, tokenizer_files)
        chat_rule = require_object(
            reference.get("chat_template_source"),
            f"catalog lane {lane_id}.reference.chat_template_source",
        )
        chat_source_name = chat_rule.get("source")
        require(
            chat_source_name in {"semantic_source", "tokenizer_source"},
            f"catalog lane {lane_id}.chat_template_source.source is invalid",
        )
        chat_snapshot = semantic
        chat_files = semantic_files
        if chat_source_name == "tokenizer_source":
            require(
                reference.get("tokenizer_repo") is not None,
                f"catalog lane {lane_id} chat template selects a missing tokenizer_source",
            )
            chat_snapshot = tokenizer
            chat_files = tokenizer_files
        chat_path = require_string(
            chat_rule.get("path"),
            f"catalog lane {lane_id}.chat_template_source.path",
        )
        chat_file = next((row for row in chat_files if row["path"] == chat_path), None)
        require(chat_file is not None, f"catalog lane {lane_id} chat template container is not locked")
        expected_container_sha = normalize_sha256(
            chat_rule.get("container_sha256"),
            f"catalog lane {lane_id}.chat_template_source.container_sha256",
        )
        require(
            chat_file["sha256"] == expected_container_sha,
            f"catalog lane {lane_id} chat template container SHA256 mismatch",
        )
        pointer = require_string(
            chat_rule.get("json_pointer"),
            f"catalog lane {lane_id}.chat_template_source.json_pointer",
        )
        template = self.json_string_at_pointer(
            self.file_content(chat_snapshot, chat_path, f"catalog lane {lane_id}.chat_template_source"),
            pointer,
            f"catalog lane {lane_id}.chat_template_source",
        )
        template_bytes = template.encode("utf-8")
        template_sha = hashlib.sha256(template_bytes).hexdigest()
        expected_template_sha = normalize_sha256(
            chat_rule.get("content_sha256"),
            f"catalog lane {lane_id}.chat_template_source.content_sha256",
        )
        require(
            template_sha == expected_template_sha,
            f"catalog lane {lane_id} chat template content SHA256 mismatch",
        )
        result["chat_template"] = {
            "source": chat_source_name,
            "repo": chat_snapshot.repo,
            "revision": chat_snapshot.revision,
            "path": chat_path,
            "json_pointer": pointer,
            "container_sha256": chat_file["sha256"],
            "content_sha256": template_sha,
            "content_bytes": len(template_bytes),
        }
        return result


def git_identity() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        proc = subprocess.run(
            args,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"

    status = run(["git", "status", "--short"])
    return {
        "git_sha": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def resolve_catalog(catalog_path: Path, transport: Transport) -> dict[str, Any]:
    try:
        catalog = strict_json_loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ResolutionError(f"invalid model catalog {catalog_path}: {exc}") from exc
    require_object(catalog, "catalog")
    require(catalog.get("schema_version") == 1, "catalog schema_version must be 1")
    catalog_id = require_string(catalog.get("catalog_id"), "catalog.catalog_id")
    lanes_raw = require_list(catalog.get("models"), "catalog.models")
    require(lanes_raw, "catalog.models must not be empty")
    resolver = Resolver(transport)
    lanes = [resolver.lane(raw, index) for index, raw in enumerate(lanes_raw)]
    lane_ids = [lane["catalog_lane_id"] for lane in lanes]
    require(len(set(lane_ids)) == len(lane_ids), "catalog lane ids must be unique")
    expected_metadata_urls: set[str] = set()
    for lane in lanes:
        for source_name in ("weight_source", "semantic_source", "tokenizer_source"):
            source = lane.get(source_name)
            if source is None:
                continue
            rows = [*source["files"], *source["license"]["files"]]
            for row in rows:
                content_url = row.get("content_request_url")
                if content_url is not None:
                    expected_metadata_urls.add(str(content_url))
    actual_metadata_urls = {
        str(request["url"])
        for request in transport.requests
        if request.get("kind") == "metadata-file"
    }
    require(
        actual_metadata_urls == expected_metadata_urls,
        "metadata request provenance differs from the exact selected metadata file set",
    )
    require(
        all(
            request.get("response_bytes", MAX_METADATA_BYTES + 1) <= MAX_METADATA_BYTES
            for request in transport.requests
            if request.get("kind") == "metadata-file"
        ),
        "metadata request provenance exceeds the download limit",
    )
    return {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_model_resolution",
        "generated_at": utc_now(),
        "source": git_identity(),
        "catalog_id": catalog_id,
        "catalog_path": str(catalog_path.relative_to(ROOT)) if catalog_path.is_relative_to(ROOT) else str(catalog_path),
        "catalog_sha256": catalog_sha256(catalog_path),
        "resolver": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": catalog_sha256(Path(__file__).resolve()),
        },
        "policy": {
            "revision": "full_hugging_face_commit",
            "large_weight_downloaded": False,
            "lfs_sha256_source": "Hugging Face tree lfs.oid",
            "non_lfs_max_download_bytes": MAX_METADATA_BYTES,
            "lfs_metadata_download": {
                "allowed_suffixes": list(LFS_METADATA_SUFFIXES),
                "max_bytes": MAX_METADATA_BYTES,
                "selector_requirement": "weight_source_exact_path_required_if_sharded",
                "sha256_must_match_lfs_oid": True,
            },
            "raw_response_body_kinds": sorted(PROVENANCE_BODY_KINDS),
            "transport": transport.provenance,
        },
        "lanes": lanes,
        "requests": transport.requests,
    }


def write_output(out_arg: Path, data: dict[str, Any]) -> tuple[Path, Path]:
    if out_arg.suffix.lower() == ".json":
        output = out_arg if out_arg.is_absolute() else ROOT / out_arg
        pass_path = output
    else:
        pass_path = out_arg if out_arg.is_absolute() else ROOT / out_arg
        output = pass_path / OUTPUT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output, pass_path


def tree_entry(path: str, body: bytes, *, lfs_oid: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": "file",
        "path": path,
        "size": len(body),
        "oid": hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest(),
    }
    if lfs_oid is not None:
        entry["lfs"] = {"oid": lfs_oid, "size": len(body), "pointerSize": 130}
    return entry


def selftest_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    rev_weight = "1" * 40
    rev_gguf = "2" * 40
    rev_semantic = "3" * 40
    rev_tokenizer = "4" * 40
    rev_official = "5" * 40
    config = b'{"model_type":"qwen"}\n'
    generation = b'{"temperature":0.7}\n'
    tokenizer_config = b'{"chat_template":"test"}\n'
    tokenizer = b'{"model":{}}\n'
    index = (
        json.dumps(
            {
                "weight_map": {
                    "layer.0.weight": "model-00001-of-000002.safetensors",
                    "layer.1.weight": "model-00002-of-000002.safetensors",
                }
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    quant = b'{"bits":4}\n'
    license_body = b"Apache License 2.0\n"
    shard_a = b"a" * 17
    shard_b = b"b" * 19
    gguf = b"GGUF" + b"x" * 20

    catalog = {
        "schema_version": 1,
        "catalog_id": "selftest",
        "models": [
            {
                "id": "DENSE-CUDA",
                "model_id": "dense",
                "backend": "cuda",
                "repo": "test/dense",
                "revision": {"status": "resolution_required", "value": None},
                "files": [
                    {"path": "config.json", "required": True},
                    {"glob": "*.safetensors", "required": True},
                    {"path": "model.safetensors.index.json", "required_if_sharded": True},
                ],
                "format": "safetensors",
                "reference": {
                    "semantic_repo": "test/dense",
                    "semantic_revision": {"status": "same_as_weight_revision"},
                    "required_semantic_files": [
                        "config.json",
                        "generation_config.json",
                        "tokenizer_config.json",
                        "tokenizer.json",
                    ],
                    "semantic_file_sha256": {
                        "config.json": hashlib.sha256(config).hexdigest(),
                        "generation_config.json": hashlib.sha256(generation).hexdigest(),
                        "tokenizer_config.json": hashlib.sha256(tokenizer_config).hexdigest(),
                        "tokenizer.json": hashlib.sha256(tokenizer).hexdigest(),
                    },
                    "generation_config_source": {
                        "source": "semantic_source",
                        "path": "generation_config.json",
                        "policy": "required",
                        "content_sha256": hashlib.sha256(generation).hexdigest(),
                    },
                    "chat_template_source": {
                        "source": "semantic_source",
                        "path": "tokenizer_config.json",
                        "json_pointer": "/chat_template",
                        "container_sha256": hashlib.sha256(tokenizer_config).hexdigest(),
                        "content_sha256": hashlib.sha256(b"test").hexdigest(),
                    },
                },
            },
            {
                "id": "GGUF-METAL",
                "model_id": "gguf",
                "backend": "metal",
                "repo": "test/gguf",
                "revision": {"status": "pinned", "value": rev_gguf},
                "files": [
                    {
                        "path": "model.gguf",
                        "required": True,
                        "expected_size_bytes": len(gguf),
                        "expected_sha256": hashlib.sha256(gguf).hexdigest(),
                    }
                ],
                "format": "gguf",
                "reference": {
                    "semantic_repo": "test/semantic",
                    "semantic_revision": {"status": "resolution_required", "value": None},
                    "required_semantic_files": [
                        "config.json",
                        "generation_config.json",
                        "tokenizer_config.json",
                        "tokenizer.json",
                    ],
                    "semantic_file_sha256": {
                        "config.json": hashlib.sha256(config).hexdigest(),
                        "generation_config.json": hashlib.sha256(generation).hexdigest(),
                        "tokenizer_config.json": hashlib.sha256(tokenizer_config).hexdigest(),
                        "tokenizer.json": hashlib.sha256(tokenizer).hexdigest(),
                    },
                    "generation_config_source": {
                        "source": "semantic_source",
                        "path": "generation_config.json",
                        "policy": "required",
                        "content_sha256": hashlib.sha256(generation).hexdigest(),
                    },
                    "official_upstream": {
                        "repo": "test/official",
                        "revision": {"status": "pinned", "value": rev_official},
                        "required_gated": True,
                        "blob_oid_match_files": [
                            "config.json",
                            "generation_config.json",
                            "tokenizer.json",
                        ],
                        "expected_git_oids": {
                            path: tree_entry(path, body)["oid"]
                            for path, body in (
                                ("config.json", config),
                                ("generation_config.json", generation),
                                ("tokenizer.json", tokenizer),
                            )
                        },
                        "expected_content_sha256": {
                            path: hashlib.sha256(body).hexdigest()
                            for path, body in (
                                ("config.json", config),
                                ("generation_config.json", generation),
                                ("tokenizer.json", tokenizer),
                            )
                        },
                        "expected_size_bytes": {
                            "config.json": len(config),
                            "generation_config.json": len(generation),
                            "tokenizer.json": len(tokenizer),
                        },
                        "access_note": "selftest official repo is gated",
                    },
                    "tokenizer_repo": "test/tokenizer",
                    "tokenizer_revision": {"status": "resolution_required", "value": None},
                    "required_tokenizer_files": ["tokenizer_config.json", "tokenizer.json"],
                    "tokenizer_file_sha256": {
                        "tokenizer_config.json": hashlib.sha256(tokenizer_config).hexdigest(),
                        "tokenizer.json": hashlib.sha256(tokenizer).hexdigest(),
                    },
                    "chat_template_source": {
                        "source": "tokenizer_source",
                        "path": "tokenizer_config.json",
                        "json_pointer": "/chat_template",
                        "container_sha256": hashlib.sha256(tokenizer_config).hexdigest(),
                        "content_sha256": hashlib.sha256(b"test").hexdigest(),
                    },
                },
            },
        ],
    }

    dense_tree = [
        tree_entry("config.json", config),
        tree_entry("generation_config.json", generation),
        tree_entry("tokenizer_config.json", tokenizer_config),
        tree_entry("tokenizer.json", tokenizer),
        tree_entry("model-00001-of-000002.safetensors", shard_a, lfs_oid=hashlib.sha256(shard_a).hexdigest()),
        tree_entry("model-00002-of-000002.safetensors", shard_b, lfs_oid=hashlib.sha256(shard_b).hexdigest()),
        tree_entry(
            "model.safetensors.index.json",
            index,
            lfs_oid=hashlib.sha256(index).hexdigest(),
        ),
        tree_entry("LICENSE", license_body),
    ]
    semantic_tree = [
        tree_entry("config.json", config),
        tree_entry("generation_config.json", generation),
        tree_entry("tokenizer_config.json", tokenizer_config),
        tree_entry("tokenizer.json", tokenizer),
        tree_entry("LICENSE.txt", license_body),
    ]
    tokenizer_tree = [
        tree_entry("config.json", config),
        tree_entry("generation_config.json", generation),
        tree_entry("tokenizer_config.json", tokenizer_config),
        tree_entry("tokenizer.json", tokenizer),
    ]
    responses: dict[str, Any] = {}

    def add_repo(repo: str, revision: str, requested: str | None, tree: list[dict[str, Any]], license_id: str) -> None:
        responses[model_api_url(repo, requested)] = {
            "status": 200,
            "json": {"sha": revision, "cardData": {"license": license_id}, "gated": False},
        }
        responses[tree_api_url(repo, revision)] = {"status": 200, "json": tree}

    add_repo("test/dense", rev_weight, None, dense_tree, "apache-2.0")
    add_repo("test/gguf", rev_gguf, rev_gguf, [tree_entry("model.gguf", gguf, lfs_oid=hashlib.sha256(gguf).hexdigest())], "apache-2.0")
    add_repo("test/semantic", rev_semantic, None, semantic_tree, "apache-2.0")
    add_repo("test/tokenizer", rev_tokenizer, None, tokenizer_tree, "apache-2.0")
    official_tree = [
        tree_entry("config.json", config),
        tree_entry("generation_config.json", generation),
        tree_entry("tokenizer.json", tokenizer),
    ]
    responses[model_api_url("test/official", rev_official)] = {
        "status": 200,
        "json": {"sha": rev_official, "cardData": {"license": "llama3.1"}, "gated": "manual"},
    }
    responses[tree_api_url("test/official", rev_official)] = {
        "status": 200,
        "json": official_tree,
    }

    contents = {
        ("test/dense", rev_weight, "config.json"): config,
        ("test/dense", rev_weight, "generation_config.json"): generation,
        ("test/dense", rev_weight, "tokenizer_config.json"): tokenizer_config,
        ("test/dense", rev_weight, "tokenizer.json"): tokenizer,
        ("test/dense", rev_weight, "model.safetensors.index.json"): index,
        ("test/dense", rev_weight, "LICENSE"): license_body,
        ("test/semantic", rev_semantic, "config.json"): config,
        ("test/semantic", rev_semantic, "generation_config.json"): generation,
        ("test/semantic", rev_semantic, "tokenizer_config.json"): tokenizer_config,
        ("test/semantic", rev_semantic, "tokenizer.json"): tokenizer,
        ("test/semantic", rev_semantic, "LICENSE.txt"): license_body,
        ("test/tokenizer", rev_tokenizer, "config.json"): config,
        ("test/tokenizer", rev_tokenizer, "generation_config.json"): generation,
        ("test/tokenizer", rev_tokenizer, "tokenizer_config.json"): tokenizer_config,
        ("test/tokenizer", rev_tokenizer, "tokenizer.json"): tokenizer,
    }
    for (repo, revision, path), body in contents.items():
        responses[resolve_file_url(repo, revision, path)] = {
            "status": 200,
            "base64": base64.b64encode(body).decode("ascii"),
        }
    return catalog, {"schema_version": 1, "responses": responses}


def expect_failure(catalog: dict[str, Any], fixture: dict[str, Any], needle: str, root: Path) -> None:
    catalog_path = root / f"negative-{hashlib.sha256(needle.encode()).hexdigest()[:8]}.catalog.json"
    fixture_path = root / f"negative-{hashlib.sha256(needle.encode()).hexdigest()[:8]}.fixture.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    try:
        resolve_catalog(catalog_path, FixtureTransport(fixture_path))
    except ResolutionError as exc:
        require(needle in str(exc), f"selftest expected {needle!r}, got {exc!r}")
    else:
        raise ResolutionError(f"selftest negative case unexpectedly passed: {needle}")


def run_selftest() -> None:
    for payload in ('{"fact":1,"fact":2}', '{"fact":NaN}'):
        try:
            strict_json_loads(payload)
        except ValueError:
            pass
        else:
            raise ResolutionError(f"strict JSON selftest unexpectedly accepted {payload}")
    for content_length in (str(MAX_METADATA_BYTES + 1), "-1", "not-a-size"):
        try:
            validate_metadata_content_length(content_length, "https://huggingface.co/selftest")
        except ResolutionError:
            pass
        else:
            raise ResolutionError(
                f"metadata Content-Length selftest unexpectedly accepted {content_length!r}"
            )
    catalog, fixture = selftest_fixture()
    with tempfile.TemporaryDirectory(prefix="runtime-vnext-model-resolver-") as temporary:
        root = Path(temporary)
        catalog_path = root / "catalog.json"
        fixture_path = root / "fixture.json"
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
        fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
        result = resolve_catalog(catalog_path, FixtureTransport(fixture_path))
        require(result["policy"]["transport"] == "internal_selftest_fixture", "selftest transport provenance mismatch")
        require(
            result["policy"]["raw_response_body_kinds"] == ["model-info", "repo-tree"],
            "selftest raw response body policy mismatch",
        )
        for request in result["requests"]:
            has_body = "response_body_base64" in request
            require(
                has_body == (request["kind"] in PROVENANCE_BODY_KINDS),
                "selftest request body provenance mismatch",
            )
        require(len(result["lanes"]) == 2, "selftest lane count mismatch")
        dense = result["lanes"][0]
        weight_files = dense["weight_source"]["files"]
        require(len(weight_files) == 4, "selftest weight file count mismatch")
        require(
            sum(row["sha256_source"] == "hugging_face_lfs_oid" for row in weight_files) == 3,
            "selftest did not preserve LFS hashes",
        )
        index_file = next(row for row in weight_files if row["path"] == "model.safetensors.index.json")
        require(
            index_file.get("lfs_metadata_downloaded") is True
            and index_file.get("content_request_url")
            == resolve_file_url("test/dense", "1" * 40, "model.safetensors.index.json"),
            "selftest did not bind the downloaded LFS safetensors index",
        )
        require(
            dense["semantic_source"]["revision"] == dense["weight_source"]["revision"],
            "selftest same_as_weight_revision mismatch",
        )
        gguf_lane = result["lanes"][1]
        require("tokenizer_source" in gguf_lane, "selftest tokenizer source missing")
        require(
            gguf_lane["weight_source"]["files"] == [
                {
                    "path": "model.gguf",
                    "size_bytes": 24,
                    "sha256": catalog["models"][1]["files"][0]["expected_sha256"],
                    "sha256_source": "hugging_face_lfs_oid",
                    "git_oid": tree_entry("model.gguf", b"GGUF" + b"x" * 20)["oid"],
                    "lfs_oid": catalog["models"][1]["files"][0]["expected_sha256"],
                }
            ],
            "selftest catalog expected SHA256 did not bind the resolved GGUF LFS identity",
        )
        require(
            dense["chat_template"]["content_sha256"] == hashlib.sha256(b"test").hexdigest(),
            "selftest chat template content binding mismatch",
        )
        require(
            dense["generation_config"]["present"] is True,
            "selftest generation config binding mismatch",
        )
        require(
            len(gguf_lane["official_upstream"]["mirror_blob_oid_matches"]) == 3,
            "selftest official upstream match matrix mismatch",
        )

        moving = copy.deepcopy(catalog)
        moving["models"][1]["revision"]["value"] = "main"
        expect_failure(moving, fixture, "must not be a moving revision", root)

        invalid_index_selector = copy.deepcopy(catalog)
        invalid_index_selector["models"][0]["files"][2] = {
            "glob": "*.safetensors.index.json",
            "required_if_sharded": True,
        }
        invalid_selector_catalog = root / "invalid-index-selector.catalog.json"
        invalid_selector_fixture = root / "invalid-index-selector.fixture.json"
        invalid_selector_catalog.write_text(json.dumps(invalid_index_selector), encoding="utf-8")
        invalid_selector_fixture.write_text(json.dumps(fixture), encoding="utf-8")
        invalid_selector_transport = FixtureTransport(invalid_selector_fixture)
        try:
            resolve_catalog(invalid_selector_catalog, invalid_selector_transport)
        except ResolutionError as exc:
            require(
                "required_if_sharded must use path" in str(exc),
                f"invalid index selector rejected for unexpected reason: {exc}",
            )
        else:
            raise ResolutionError("glob safetensors index selector unexpectedly passed")
        require(
            all(request.get("kind") != "metadata-file" for request in invalid_selector_transport.requests),
            "invalid index selector fetched metadata before rejection",
        )

        missing = copy.deepcopy(fixture)
        dense_tree_url = tree_api_url("test/dense", "1" * 40)
        missing["responses"][dense_tree_url]["json"] = [
            row for row in missing["responses"][dense_tree_url]["json"] if row["path"] != "config.json"
        ]
        expect_failure(catalog, missing, "required file selection 'config.json' matched nothing", root)

        bad_lfs = copy.deepcopy(fixture)
        bad_lfs["responses"][dense_tree_url]["json"][4]["lfs"]["oid"] = "not-a-digest"
        expect_failure(catalog, bad_lfs, "must be a SHA256 digest", root)

        bad_size = copy.deepcopy(fixture)
        bad_size["responses"][dense_tree_url]["json"][4]["lfs"]["size"] += 1
        expect_failure(catalog, bad_size, "tree/LFS size mismatch", root)

        oversized_lfs_index = copy.deepcopy(fixture)
        oversized_index_entry = next(
            row
            for row in oversized_lfs_index["responses"][dense_tree_url]["json"]
            if row["path"] == "model.safetensors.index.json"
        )
        oversized_index_entry["size"] = MAX_METADATA_BYTES + 1
        oversized_index_entry["lfs"]["size"] = MAX_METADATA_BYTES + 1
        expect_failure(
            catalog,
            oversized_lfs_index,
            "LFS metadata exceeds metadata limit",
            root,
        )

        mismatched_lfs_index = copy.deepcopy(fixture)
        index_url = resolve_file_url("test/dense", "1" * 40, "model.safetensors.index.json")
        valid_index_size = len(
            base64.b64decode(fixture["responses"][index_url]["base64"], validate=True)
        )
        mismatched_lfs_index["responses"][index_url]["base64"] = base64.b64encode(
            b"x" * valid_index_size
        ).decode("ascii")
        expect_failure(
            catalog,
            mismatched_lfs_index,
            "downloaded LFS metadata SHA256 differs from tree LFS OID",
            root,
        )

        bad_expected_sha = copy.deepcopy(catalog)
        bad_expected_sha["models"][1]["files"][0]["expected_sha256"] = "0" * 64
        expect_failure(bad_expected_sha, fixture, "expected SHA256 mismatch for model.gguf", root)

        missing_expected_lfs = copy.deepcopy(fixture)
        gguf_tree_url = tree_api_url("test/gguf", "2" * 40)
        missing_expected_lfs["responses"][gguf_tree_url]["json"][0].pop("lfs")
        expect_failure(
            catalog,
            missing_expected_lfs,
            "expected SHA256 for model.gguf requires a visible Hugging Face LFS OID",
            root,
        )

        redacted_expected_lfs = copy.deepcopy(fixture)
        redacted_expected_lfs["responses"][gguf_tree_url]["json"][0]["lfs"]["oid"] = "*" * 64
        expect_failure(
            catalog,
            redacted_expected_lfs,
            "expected SHA256 for model.gguf requires a visible Hugging Face LFS OID",
            root,
        )

        ambiguous_expected_sha = copy.deepcopy(catalog)
        ambiguous_expected_sha["models"][0]["files"][1]["expected_sha256"] = "a" * 64
        expect_failure(
            ambiguous_expected_sha,
            fixture,
            "expected_sha256 requires an exact path selector",
            root,
        )

        missing_shard = copy.deepcopy(fixture)
        missing_shard["responses"][dense_tree_url]["json"] = [
            row
            for row in missing_shard["responses"][dense_tree_url]["json"]
            if row["path"] != "model-00002-of-000002.safetensors"
        ]
        expect_failure(
            catalog,
            missing_shard,
            "safetensors index shard set differs from selected weight shards",
            root,
        )

        mixed_total_width = copy.deepcopy(fixture)
        mixed_index_body = (
            json.dumps(
                {
                    "weight_map": {
                        "layer.0.weight": "model-00001-of-000002.safetensors",
                        "layer.1.weight": "model-00002-of-00002.safetensors",
                    }
                },
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        mixed_total_width["responses"][dense_tree_url]["json"] = [
            {**row, "path": "model-00002-of-00002.safetensors"}
            if row["path"] == "model-00002-of-000002.safetensors"
            else tree_entry(
                "model.safetensors.index.json",
                mixed_index_body,
                lfs_oid=hashlib.sha256(mixed_index_body).hexdigest(),
            )
            if row["path"] == "model.safetensors.index.json"
            else row
            for row in mixed_total_width["responses"][dense_tree_url]["json"]
        ]
        mixed_total_width["responses"][
            resolve_file_url("test/dense", "1" * 40, "model.safetensors.index.json")
        ] = {
            "status": 200,
            "base64": base64.b64encode(mixed_index_body).decode("ascii"),
        }
        expect_failure(
            catalog,
            mixed_total_width,
            "sharded safetensors total width differs",
            root,
        )

        mixed_number_width = copy.deepcopy(fixture)
        mixed_number_index_body = (
            json.dumps(
                {
                    "weight_map": {
                        "layer.0.weight": "model-00001-of-000002.safetensors",
                        "layer.1.weight": "model-000002-of-000002.safetensors",
                    }
                },
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        mixed_number_width["responses"][dense_tree_url]["json"] = [
            {**row, "path": "model-000002-of-000002.safetensors"}
            if row["path"] == "model-00002-of-000002.safetensors"
            else tree_entry(
                "model.safetensors.index.json",
                mixed_number_index_body,
                lfs_oid=hashlib.sha256(mixed_number_index_body).hexdigest(),
            )
            if row["path"] == "model.safetensors.index.json"
            else row
            for row in mixed_number_width["responses"][dense_tree_url]["json"]
        ]
        mixed_number_width["responses"][
            resolve_file_url("test/dense", "1" * 40, "model.safetensors.index.json")
        ] = {
            "status": 200,
            "base64": base64.b64encode(mixed_number_index_body).decode("ascii"),
        }
        expect_failure(
            catalog,
            mixed_number_width,
            "sharded safetensors number width differs",
            root,
        )

        unsharded = copy.deepcopy(fixture)
        unsharded["responses"][dense_tree_url]["json"] = [
            {**row, "path": "model.safetensors"}
            if row["path"] == "model-00001-of-000002.safetensors"
            else row
            for row in unsharded["responses"][dense_tree_url]["json"]
            if row["path"] != "model-00002-of-000002.safetensors"
        ]
        unsharded_path = root / "unsharded.fixture.json"
        unsharded_path.write_text(json.dumps(unsharded), encoding="utf-8")
        unsharded_result = resolve_catalog(catalog_path, FixtureTransport(unsharded_path))
        unsharded_weight_paths = {
            row["path"] for row in unsharded_result["lanes"][0]["weight_source"]["files"]
        }
        require(
            "model.safetensors" in unsharded_weight_paths
            and "model.safetensors.index.json" not in unsharded_weight_paths,
            "unsharded selftest selected a conditional safetensors index",
        )
        require(
            all(
                request.get("url")
                != resolve_file_url("test/dense", "1" * 40, "model.safetensors.index.json")
                for request in unsharded_result["requests"]
            ),
            "unsharded selftest downloaded a conditional safetensors index",
        )

        incomplete_index = copy.deepcopy(fixture)
        incomplete_index_body = (
            json.dumps(
                {"weight_map": {"layer.0.weight": "model-00001-of-000002.safetensors"}},
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        incomplete_index["responses"][dense_tree_url]["json"] = [
            tree_entry(row["path"], incomplete_index_body)
            if row["path"] == "model.safetensors.index.json"
            else row
            for row in incomplete_index["responses"][dense_tree_url]["json"]
        ]
        incomplete_index["responses"][
            resolve_file_url("test/dense", "1" * 40, "model.safetensors.index.json")
        ] = {
            "status": 200,
            "base64": base64.b64encode(incomplete_index_body).decode("ascii"),
        }
        expect_failure(
            catalog,
            incomplete_index,
            "safetensors index shard set differs from selected weight shards",
            root,
        )

        http_error = copy.deepcopy(fixture)
        http_error["responses"][model_api_url("test/dense", None)]["status"] = 503
        expect_failure(catalog, http_error, "HTTP 503", root)

        bad_semantic_sha = copy.deepcopy(catalog)
        bad_semantic_sha["models"][0]["reference"]["semantic_file_sha256"]["config.json"] = "0" * 64
        expect_failure(bad_semantic_sha, fixture, "content SHA256 mismatch for config.json", root)

        bad_generation_policy = copy.deepcopy(catalog)
        bad_generation_policy["models"][0]["reference"]["generation_config_source"] = {
            "source": "semantic_source",
            "path": "generation_config.json",
            "policy": "absent",
        }
        expect_failure(
            bad_generation_policy,
            fixture,
            "generation config declared absent but exists",
            root,
        )

        bad_template_sha = copy.deepcopy(catalog)
        bad_template_sha["models"][0]["reference"]["chat_template_source"]["content_sha256"] = "0" * 64
        expect_failure(bad_template_sha, fixture, "chat template content SHA256 mismatch", root)

        bad_official_oid = copy.deepcopy(catalog)
        bad_official_oid["models"][1]["reference"]["official_upstream"]["expected_git_oids"][
            "config.json"
        ] = "0" * 40
        expect_failure(bad_official_oid, fixture, "official upstream frozen evidence mismatch", root)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="artifact directory or JSON output path")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args(argv)
    if not args.self_test and args.out is None:
        parser.error("--out is required unless --self-test is used")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.retries < 0 or args.retries > 5:
        parser.error("--retries must be between 0 and 5")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.self_test:
            run_selftest()
            print("RUNTIME VNEXT MODEL RESOLUTION SELFTEST PASS")
            return 0
        transport: Transport = NetworkTransport(timeout=args.timeout, retries=args.retries)
        result = resolve_catalog(CATALOG_PATH, transport)
        _, pass_path = write_output(args.out, result)
        print(f"RUNTIME VNEXT MODEL RESOLUTION PASS: {pass_path}")
        return 0
    except ResolutionError as exc:
        print(f"RUNTIME VNEXT MODEL RESOLUTION FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
