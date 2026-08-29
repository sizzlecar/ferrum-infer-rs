#!/usr/bin/env python3
"""Build a metadata-only source lock for the fixed Qwen3.8 block-FP8 checkpoint.

The tool consumes the immutable output of ``runtime_vnext_model_resolver.py``.
It fetches the already-locked config and safetensors index as bounded metadata,
then reads only each shard's eight-byte safetensors prefix and JSON header with
strict HTTP Range requests. Tensor payload bytes are never requested.

The result is diagnostic input for A1 M0. It is deliberately not a final
``model-lock.json`` receipt: execution and quality approval identities remain
null until their respective implementation and CUDA evidence exist.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_NAME = "block-fp8-source-header-lock.json"

CHECKPOINT_ID = "qwen38-27b-fp8"
CHECKPOINT_REPO = "Qwen/Qwen3.8-27B-FP8"
CHECKPOINT_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
CHECKPOINT_LICENSE = "apache-2.0"
CHECKPOINT_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
CHECKPOINT_MODEL_TYPE = "qwen3_5"
CHECKPOINT_FORMAT = "safetensors_fp8_e4m3_dynamic_block_128x128"
INDEX_PATH = "model.safetensors.index.json"
CONFIG_PATH = "config.json"

EXPECTED_SHARD_COUNT = 66
EXPECTED_TENSOR_COUNT = 1606
EXPECTED_PARTITION_COUNTS = {
    "execution_eligible_text": 1251,
    "typed_nonexecuted_visual": 333,
    "typed_nonexecuted_mtp": 22,
    "unknown": 0,
}
EXPECTED_DTYPE_COUNTS = {"BF16": 1199, "F8_E4M3": 407}
EXPECTED_PAIR_COUNTS = {
    "execution_eligible_text": 400,
    "typed_nonexecuted_visual": 0,
    "typed_nonexecuted_mtp": 7,
}
EXPECTED_OPERATION_COUNTS = {
    "operation.gated_delta_recurrent_attention": 144,
    "operation.causal_paged_attention": 64,
    "operation.dense_swiglu": 192,
    "operation.last_token_dense_linear": 0,
}

MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_TENSORS_PER_SHARD = 10_000
MAX_TOTAL_TENSORS = 100_000
MAX_TENSOR_RANK = 8
MAX_DIMENSION = 1 << 31
MAX_SHARD_WORKERS = 4
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CONTENT_RANGE_RE = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")
SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_.\-/]+$")

DTYPE_BYTES = {
    "BF16": 2,
    "F8_E4M3": 1,
}

OP_GATED_DELTA = "operation.gated_delta_recurrent_attention"
OP_CAUSAL_ATTENTION = "operation.causal_paged_attention"
OP_DENSE_SWIGLU = "operation.dense_swiglu"
OP_LOGITS = "operation.last_token_dense_linear"


class SourceLockError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceLockError(message)


def reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def strict_json_loads(payload: str | bytes, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=unique_json_object,
            parse_constant=reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SourceLockError(f"{label} is not strict JSON: {exc}") from exc


def as_object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def as_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return value


def as_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be a non-empty string")
    return value


def as_positive_int(value: Any, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} must be a positive integer",
    )
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def stable_file_url(path: str) -> str:
    require_safe_repo_path(path, "checkpoint file path")
    repo = urllib.parse.quote(CHECKPOINT_REPO, safe="/")
    revision = urllib.parse.quote(CHECKPOINT_REVISION, safe="")
    encoded_path = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{revision}/{encoded_path}"


def require_safe_repo_path(path: str, label: str) -> None:
    require(bool(path) and SAFE_PATH_RE.fullmatch(path) is not None, f"{label} is unsafe")
    parsed = Path(path)
    require(not parsed.is_absolute() and ".." not in parsed.parts, f"{label} is unsafe")


def safe_redirect_host(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or parsed.username or parsed.password:
        return False
    if parsed.port not in {None, 443}:
        return False
    return (
        host == "huggingface.co"
        or host.endswith(".huggingface.co")
        or host.endswith(".hf.co")
    )


class SafeHttpsRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow only HTTPS redirects to Hugging Face-controlled hosts."""

    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        if not safe_redirect_host(newurl):
            raise SourceLockError("refusing unsafe checkpoint download redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
            redirected.unredirected_hdrs.pop("Authorization", None)
            redirected.remove_header("Cookie")
            redirected.unredirected_hdrs.pop("Cookie", None)
        return redirected


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes


class Transport:
    provenance = "abstract"

    def fetch_metadata(self, path: str, expected_size: int, expected_sha256: str) -> bytes:
        raise NotImplementedError

    def fetch_range(self, path: str, start: int, end: int, expected_total: int) -> HttpResponse:
        raise NotImplementedError


class NetworkTransport(Transport):
    provenance = "huggingface_https_range_without_credentials"

    def __init__(self, *, timeout_seconds: float, retries: int) -> None:
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def _open_once(self, request: urllib.request.Request) -> Any:
        opener = urllib.request.build_opener(SafeHttpsRedirectHandler())
        return opener.open(request, timeout=self.timeout_seconds)

    def _request(self, request: urllib.request.Request, label: str, consume: Any) -> Any:
        """Retry the complete open/read transaction, not only TLS setup."""

        last_kind = "unknown"
        for attempt in range(self.retries + 1):
            try:
                # One opener per request keeps redirect-handler state isolated
                # when the fixed, bounded shard pool performs Range reads.
                with self._open_once(request) as handle:
                    return consume(handle)
            except urllib.error.HTTPError as exc:
                last_kind = f"HTTP {exc.code}"
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self.retries:
                    raise SourceLockError(f"{label} failed with HTTP {exc.code}") from exc
            except SourceLockError:
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_kind = type(exc).__name__
                if attempt == self.retries:
                    raise SourceLockError(f"{label} failed with {last_kind}") from exc
            time.sleep(0.5 * (2**attempt))
        raise SourceLockError(f"{label} failed with {last_kind}")

    def fetch_metadata(self, path: str, expected_size: int, expected_sha256: str) -> bytes:
        require(expected_size <= MAX_METADATA_BYTES, f"{path} exceeds metadata byte limit")
        request = urllib.request.Request(
            stable_file_url(path),
            headers={
                "Accept": "application/octet-stream",
                "Accept-Encoding": "identity",
                "User-Agent": "ferrum-vnext-block-fp8-source-lock/1",
            },
            method="GET",
        )
        def consume(handle: Any) -> bytes:
            status = int(handle.status)
            require(status == 200, f"metadata request for {path} returned HTTP {status}")
            content_encoding = handle.headers.get("Content-Encoding")
            require(
                content_encoding in {None, "identity"},
                f"metadata request for {path} used content encoding",
            )
            content_length = handle.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError as exc:
                    raise SourceLockError(f"metadata request for {path} has invalid Content-Length") from exc
                require(declared_length == expected_size, f"metadata size differs for {path}")
            return handle.read(MAX_METADATA_BYTES + 1)

        body = self._request(request, f"metadata request for {path}", consume)
        require(len(body) <= MAX_METADATA_BYTES, f"metadata response exceeds limit for {path}")
        require(len(body) == expected_size, f"metadata size differs for {path}")
        require(sha256_bytes(body) == expected_sha256, f"metadata SHA256 differs for {path}")
        return body

    def fetch_range(self, path: str, start: int, end: int, expected_total: int) -> HttpResponse:
        require(0 <= start <= end, f"invalid byte range for {path}")
        requested = end - start + 1
        require(requested <= MAX_HEADER_BYTES, f"range response limit exceeded for {path}")
        request = urllib.request.Request(
            stable_file_url(path),
            headers={
                "Accept": "application/octet-stream",
                "Accept-Encoding": "identity",
                "Range": f"bytes={start}-{end}",
                "User-Agent": "ferrum-vnext-block-fp8-source-lock/1",
            },
            method="GET",
        )
        def consume(handle: Any) -> HttpResponse:
            status = int(handle.status)
            headers = {key.lower(): value.strip() for key, value in handle.headers.items()}
            require(status == 206, f"range request for {path} returned HTTP {status}, not 206")
            require(
                headers.get("content-encoding") in {None, "identity"},
                f"range request for {path} used content encoding",
            )
            validate_content_range(
                headers.get("content-range"),
                start,
                end,
                expected_total,
                path,
            )
            content_length = headers.get("content-length")
            require(content_length is not None, f"range request for {path} lacks Content-Length")
            try:
                declared_length = int(content_length)
            except ValueError as exc:
                raise SourceLockError(f"range request for {path} has invalid Content-Length") from exc
            require(declared_length == requested, f"range Content-Length differs for {path}")
            body = handle.read(requested + 1)
            require(len(body) == requested, f"range body length differs for {path}")
            return HttpResponse(status=status, headers=headers, body=body)

        return self._request(request, f"range request for {path}", consume)


def validate_content_range(
    value: str | None,
    expected_start: int,
    expected_end: int,
    expected_total: int,
    path: str,
) -> None:
    require(value is not None, f"range response for {path} lacks Content-Range")
    match = CONTENT_RANGE_RE.fullmatch(value)
    require(match is not None, f"range response for {path} has invalid Content-Range")
    start, end, total = (int(part) for part in match.groups())
    require(
        (start, end, total) == (expected_start, expected_end, expected_total),
        f"range response for {path} has mismatched Content-Range",
    )


@dataclass(frozen=True)
class ExpectedProfile:
    shard_count: int
    tensor_count: int
    partition_counts: dict[str, int]
    dtype_counts: dict[str, int]
    pair_counts: dict[str, int]
    operation_counts: dict[str, int]


PRODUCTION_PROFILE = ExpectedProfile(
    shard_count=EXPECTED_SHARD_COUNT,
    tensor_count=EXPECTED_TENSOR_COUNT,
    partition_counts=EXPECTED_PARTITION_COUNTS,
    dtype_counts=EXPECTED_DTYPE_COUNTS,
    pair_counts=EXPECTED_PAIR_COUNTS,
    operation_counts=EXPECTED_OPERATION_COUNTS,
)


@dataclass(frozen=True)
class ResolutionInput:
    document: dict[str, Any]
    lane: dict[str, Any]
    files: dict[str, dict[str, Any]]
    shards: list[dict[str, Any]]
    resolution_path: str
    resolution_sha256: str


def validate_locked_file(row_raw: Any, label: str) -> dict[str, Any]:
    row = as_object(row_raw, label)
    path = as_string(row.get("path"), f"{label}.path")
    require_safe_repo_path(path, f"{label}.path")
    size = as_positive_int(row.get("size_bytes"), f"{label}.size_bytes")
    digest = as_string(row.get("sha256"), f"{label}.sha256").lower()
    require(SHA256_RE.fullmatch(digest) is not None, f"{label}.sha256 is invalid")
    result = copy.deepcopy(row)
    result["path"] = path
    result["size_bytes"] = size
    result["sha256"] = digest
    return result


def load_resolution(path: Path, profile: ExpectedProfile = PRODUCTION_PROFILE) -> ResolutionInput:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SourceLockError(f"cannot read source resolution: {path}") from exc
    document = as_object(strict_json_loads(raw, "source resolution"), "source resolution")
    require(document.get("schema_version") == 1, "source resolution schema_version must be 1")
    require(
        document.get("artifact_type") == "runtime_vnext_model_resolution",
        "source resolution artifact_type differs",
    )
    lanes = as_list(document.get("lanes"), "source resolution.lanes")
    require(len(lanes) == 1, "source resolution must contain exactly one lane")
    lane = as_object(lanes[0], "source resolution.lanes[0]")
    require(lane.get("catalog_lane_id") == CHECKPOINT_ID, "source resolution lane id differs")
    require(lane.get("model_id") == CHECKPOINT_ID, "source resolution model id differs")
    require(lane.get("backend") == "cuda", "source resolution backend differs")
    require(lane.get("format") == CHECKPOINT_FORMAT, "source resolution format differs")
    require(
        lane.get("safetensors_shard_naming") == "index_authoritative",
        "source resolution must use index-authoritative shard naming",
    )
    source = as_object(lane.get("weight_source"), "source resolution weight_source")
    require(source.get("repo") == CHECKPOINT_REPO, "source resolution repo differs")
    require(source.get("revision") == CHECKPOINT_REVISION, "source resolution revision differs")
    require(GIT_SHA_RE.fullmatch(CHECKPOINT_REVISION) is not None, "checkpoint revision is not full")
    require(source.get("gated") in {False, None}, "fixed A1 checkpoint must remain public")
    license_lock = as_object(source.get("license"), "source resolution license")
    license_id = as_string(
        license_lock.get("hugging_face_id"),
        "source resolution license.hugging_face_id",
    ).lower()
    require(license_id == CHECKPOINT_LICENSE, "source resolution license differs")

    rows = as_list(source.get("files"), "source resolution weight_source.files")
    files: dict[str, dict[str, Any]] = {}
    for index, row_raw in enumerate(rows):
        row = validate_locked_file(row_raw, f"source resolution file[{index}]")
        require(row["path"] not in files, f"duplicate locked file {row['path']}")
        content_url = row.get("content_request_url")
        if content_url is not None:
            require(content_url == stable_file_url(row["path"]), f"stable URL differs for {row['path']}")
        files[row["path"]] = row
    require(CONFIG_PATH in files, "source resolution lacks config.json")
    require(INDEX_PATH in files, "source resolution lacks safetensors index")

    shards = sorted(
        (row for row in files.values() if row["path"].endswith(".safetensors")),
        key=lambda row: row["path"],
    )
    require(len(shards) == profile.shard_count, "source resolution shard count differs")
    for shard in shards:
        require(shard.get("sha256_source") == "hugging_face_lfs_oid", f"{shard['path']} is not LFS-locked")
        lfs_oid = as_string(shard.get("lfs_oid"), f"{shard['path']}.lfs_oid").lower()
        require(lfs_oid == shard["sha256"], f"{shard['path']} LFS OID differs from SHA256")

    display_path = str(path)
    try:
        display_path = str(path.resolve().relative_to(ROOT))
    except ValueError:
        pass
    return ResolutionInput(
        document=document,
        lane=lane,
        files=files,
        shards=shards,
        resolution_path=display_path,
        resolution_sha256=sha256_bytes(raw),
    )


def metadata_file(transport: Transport, row: dict[str, Any], label: str) -> bytes:
    require(
        row["size_bytes"] <= MAX_METADATA_BYTES,
        f"locked {label} exceeds metadata byte limit",
    )
    return transport.fetch_metadata(row["path"], row["size_bytes"], row["sha256"])


def parse_recipe(config_body: bytes) -> dict[str, Any]:
    config = as_object(strict_json_loads(config_body, CONFIG_PATH), CONFIG_PATH)
    architectures = as_list(config.get("architectures"), "config.architectures")
    require(architectures == [CHECKPOINT_ARCHITECTURE], "config architecture differs")
    require(config.get("model_type") == CHECKPOINT_MODEL_TYPE, "config model_type differs")
    require(config.get("language_model_only") is False, "config language_model_only differs")
    quant = as_object(config.get("quantization_config"), "config.quantization_config")
    require(set(quant) == {
        "activation_scheme",
        "fmt",
        "modules_to_not_convert",
        "quant_method",
        "weight_block_size",
    }, "config quantization recipe fields differ")
    require(quant.get("quant_method") == "fp8", "quant_method must be fp8")
    require(quant.get("fmt") == "e4m3", "FP8 format must be e4m3")
    require(quant.get("activation_scheme") == "dynamic", "activation scheme must be dynamic")
    require(quant.get("weight_block_size") == [128, 128], "weight block size must be [128,128]")
    exclusions_raw = as_list(
        quant.get("modules_to_not_convert"),
        "config.quantization_config.modules_to_not_convert",
    )
    exclusions = [
        as_string(value, f"config.quantization_config.modules_to_not_convert[{index}]")
        for index, value in enumerate(exclusions_raw)
    ]
    require(len(exclusions) == len(set(exclusions)), "quantization exclusions contain duplicates")
    normalized_exclusions = sorted(exclusions)
    exclusions_digest = canonical_sha256({"modules_to_not_convert": normalized_exclusions})
    return {
        "quant_method": "fp8",
        "source_dtype": "F8_E4M3",
        "format": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": exclusions,
        "modules_to_not_convert_count": len(exclusions),
        "modules_to_not_convert_digest": exclusions_digest,
    }


def parse_index(index_body: bytes, expected_shards: set[str]) -> dict[str, str]:
    index = as_object(strict_json_loads(index_body, INDEX_PATH), INDEX_PATH)
    require(set(index).issubset({"metadata", "weight_map"}), "safetensors index has unknown fields")
    metadata = as_object(index.get("metadata"), "safetensors index.metadata")
    require(not metadata, "fixed A1 safetensors index metadata must remain empty")
    weight_map_raw = as_object(index.get("weight_map"), "safetensors index.weight_map")
    require(weight_map_raw, "safetensors index weight_map is empty")
    weight_map: dict[str, str] = {}
    for tensor_name, shard_raw in weight_map_raw.items():
        require(isinstance(tensor_name, str) and bool(tensor_name), "index tensor name is invalid")
        shard = as_string(shard_raw, f"index shard for {tensor_name}")
        require_safe_repo_path(shard, f"index shard for {tensor_name}")
        require(shard.endswith(".safetensors"), f"index shard for {tensor_name} is not safetensors")
        weight_map[tensor_name] = shard
    require(set(weight_map.values()) == expected_shards, "index shard set differs from resolver lock")
    return weight_map


def tensor_storage_bytes(dtype: str, shape: list[int], label: str) -> int:
    require(dtype in DTYPE_BYTES, f"{label} uses unsupported dtype {dtype!r}")
    require(len(shape) <= MAX_TENSOR_RANK, f"{label} rank exceeds limit")
    elements = 1
    for index, dimension in enumerate(shape):
        require(
            isinstance(dimension, int)
            and not isinstance(dimension, bool)
            and 0 <= dimension <= MAX_DIMENSION,
            f"{label}.shape[{index}] is invalid",
        )
        elements *= dimension
    return elements * DTYPE_BYTES[dtype]


def read_shard_header(
    transport: Transport,
    shard: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = shard["path"]
    total_size = shard["size_bytes"]
    prefix = transport.fetch_range(path, 0, 7, total_size)
    require(len(prefix.body) == 8, f"safetensors prefix length differs for {path}")
    header_bytes = struct.unpack("<Q", prefix.body)[0]
    require(header_bytes > 0, f"safetensors header is empty for {path}")
    require(header_bytes % 8 == 0, f"safetensors header is not 8-byte aligned for {path}")
    require(header_bytes <= MAX_HEADER_BYTES, f"safetensors header exceeds limit for {path}")
    require(8 + header_bytes < total_size, f"safetensors header consumes shard for {path}")
    header_response = transport.fetch_range(path, 8, 8 + header_bytes - 1, total_size)
    raw_header = header_response.body
    require(raw_header.startswith(b"{"), f"safetensors header does not start with JSON for {path}")
    header = as_object(strict_json_loads(raw_header, f"safetensors header {path}"), f"header {path}")
    metadata = header.pop("__metadata__", None)
    if metadata is not None:
        metadata = as_object(metadata, f"header {path}.__metadata__")
        require(
            all(isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()),
            f"header metadata is invalid for {path}",
        )
    require(len(header) <= MAX_TENSORS_PER_SHARD, f"tensor count exceeds shard limit for {path}")

    payload_bytes = total_size - 8 - header_bytes
    rows: list[dict[str, Any]] = []
    intervals: list[tuple[int, int, str]] = []
    for tensor_name, tensor_raw in header.items():
        require(isinstance(tensor_name, str) and bool(tensor_name), f"invalid tensor name in {path}")
        tensor = as_object(tensor_raw, f"tensor {tensor_name}")
        require(set(tensor) == {"dtype", "shape", "data_offsets"}, f"tensor metadata fields differ for {tensor_name}")
        dtype = as_string(tensor.get("dtype"), f"{tensor_name}.dtype")
        shape_raw = as_list(tensor.get("shape"), f"{tensor_name}.shape")
        shape = list(shape_raw)
        storage_bytes = tensor_storage_bytes(dtype, shape, tensor_name)
        offsets = as_list(tensor.get("data_offsets"), f"{tensor_name}.data_offsets")
        require(len(offsets) == 2, f"{tensor_name}.data_offsets must contain two values")
        start, end = offsets
        require(
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start <= end <= payload_bytes,
            f"{tensor_name}.data_offsets are invalid",
        )
        require(end - start == storage_bytes, f"{tensor_name} dtype/shape/data_offsets disagree")
        rows.append(
            {
                "name": tensor_name,
                "shard": path,
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, end],
                "storage_bytes": storage_bytes,
            }
        )
        intervals.append((start, end, tensor_name))

    intervals.sort()
    cursor = 0
    for start, end, tensor_name in intervals:
        require(start == cursor, f"safetensors data has a hole or overlap before {tensor_name}")
        cursor = end
    require(cursor == payload_bytes, f"safetensors data does not cover the payload for {path}")
    rows.sort(key=lambda row: row["name"])
    return rows, {
        "path": path,
        "locked_size_bytes": total_size,
        "locked_lfs_sha256": shard["sha256"],
        "prefix_bytes_requested": 8,
        "header_bytes_requested": header_bytes,
        "payload_bytes_requested": 0,
        "payload_storage_bytes": payload_bytes,
        "header_sha256": sha256_bytes(raw_header),
        "tensor_count": len(rows),
    }


def classify_tensor(name: str) -> tuple[str, str]:
    if name.startswith("model.visual."):
        return "typed_nonexecuted_visual", "vision"
    if name.startswith("mtp."):
        return "typed_nonexecuted_mtp", "mtp"
    if name.startswith("model.language_model.") or name == "lm_head.weight":
        return "execution_eligible_text", "text"
    return "unknown", "unknown"


def operation_for_quant_weight(name: str) -> str | None:
    if re.fullmatch(
        r"model\.language_model\.layers\.[0-9]+\.linear_attn\."
        r"(?:in_proj_qkv|in_proj_z|out_proj)\.weight",
        name,
    ):
        return OP_GATED_DELTA
    if re.fullmatch(
        r"model\.language_model\.layers\.[0-9]+\.self_attn\."
        r"(?:q_proj|k_proj|v_proj|o_proj)\.weight",
        name,
    ):
        return OP_CAUSAL_ATTENTION
    if re.fullmatch(
        r"model\.language_model\.layers\.[0-9]+\.mlp\."
        r"(?:gate_proj|up_proj|down_proj)\.weight",
        name,
    ):
        return OP_DENSE_SWIGLU
    if name == "lm_head.weight":
        return OP_LOGITS
    return None


def enrich_and_validate_tensors(
    rows: list[dict[str, Any]],
    recipe: dict[str, Any],
    profile: ExpectedProfile,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    require(len(rows) == profile.tensor_count, "full tensor count differs from fixed profile")
    by_name = {row["name"]: row for row in rows}
    require(len(by_name) == len(rows), "tensor names are duplicated across shards")
    exclusions = set(recipe["modules_to_not_convert"])

    partition_names: dict[str, list[str]] = {
        "execution_eligible_text": [],
        "typed_nonexecuted_visual": [],
        "typed_nonexecuted_mtp": [],
        "unknown": [],
    }
    partition_bytes = {key: 0 for key in partition_names}
    dtype_counts: dict[str, int] = {}
    dtype_storage: dict[str, int] = {}
    pairs: list[dict[str, Any]] = []
    scale_names: set[str] = set()
    operation_counts = {operation: 0 for operation in profile.operation_counts}

    for row in rows:
        partition, component = classify_tensor(row["name"])
        row["partition"] = partition
        row["component"] = component
        partition_names[partition].append(row["name"])
        partition_bytes[partition] += row["storage_bytes"]
        dtype_counts[row["dtype"]] = dtype_counts.get(row["dtype"], 0) + 1
        dtype_storage[row["dtype"]] = dtype_storage.get(row["dtype"], 0) + row["storage_bytes"]

    require(not partition_names["unknown"], "unknown tensor namespace is not allowed")
    actual_partition_counts = {key: len(value) for key, value in partition_names.items()}
    require(actual_partition_counts == profile.partition_counts, "tensor partition counts differ")
    require(dtype_counts == profile.dtype_counts, "tensor dtype counts differ")

    for row in rows:
        name = row["name"]
        if not name.endswith(".weight_scale_inv"):
            continue
        scale_names.add(name)
        value_name = name.removesuffix("_scale_inv")
        value = by_name.get(value_name)
        require(value is not None, f"orphan FP8 sidecar {name}")
        require(value["dtype"] == "F8_E4M3", f"sidecar {name} does not pair with F8_E4M3")
        require(value_name.endswith(".weight"), f"sidecar {name} does not pair with a weight")
        require(row["dtype"] == "BF16", f"sidecar {name} must be BF16")
        require(len(value["shape"]) == 2, f"FP8 weight {value_name} must be rank two")
        n, k = value["shape"]
        expected_scale_shape = [math.ceil(n / 128), math.ceil(k / 128)]
        require(row["shape"] == expected_scale_shape, f"bad block scale shape for {name}")
        require(row["shard"] == value["shard"], f"FP8 value/scale pair crosses shards for {value_name}")
        require(row["partition"] == value["partition"], f"FP8 value/scale partition differs for {value_name}")
        module_name = value_name.removesuffix(".weight")
        require(module_name not in exclusions, f"quantized weight contradicts exclusion {module_name}")
        operation_id = None
        if value["partition"] == "execution_eligible_text":
            operation_id = operation_for_quant_weight(value_name)
            require(operation_id is not None, f"text FP8 weight has no operation mapping: {value_name}")
            require(operation_id in operation_counts, f"unexpected operation mapping for {value_name}")
            operation_counts[operation_id] += 1
        pairs.append(
            {
                "value_tensor": value_name,
                "scale_tensor": name,
                "shard": value["shard"],
                "value_dtype": value["dtype"],
                "value_shape": value["shape"],
                "scale_dtype": row["dtype"],
                "scale_shape": row["shape"],
                "block_shape": [128, 128],
                "partition": value["partition"],
                "component": value["component"],
                "operation_id": operation_id,
            }
        )

    for row in rows:
        if row["dtype"] == "F8_E4M3" and row["name"].endswith(".weight"):
            expected_scale = f"{row['name']}_scale_inv"
            require(expected_scale in scale_names, f"FP8 weight lacks scale sidecar: {row['name']}")
        elif row["dtype"] == "F8_E4M3":
            raise SourceLockError(f"F8_E4M3 tensor is not a weight: {row['name']}")

    pairs.sort(key=lambda row: row["value_tensor"])
    pair_counts = {
        partition: sum(1 for pair in pairs if pair["partition"] == partition)
        for partition in profile.pair_counts
    }
    require(pair_counts == profile.pair_counts, "FP8 pair partition counts differ")
    require(operation_counts == profile.operation_counts, "quantized operation counts differ")

    partition = {
        key: {
            "tensor_count": len(partition_names[key]),
            "storage_bytes": partition_bytes[key],
            "tensor_names": sorted(partition_names[key]),
        }
        for key in partition_names
    }
    stats = {
        "tensor_count": len(rows),
        "storage_bytes": sum(row["storage_bytes"] for row in rows),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "dtype_storage_bytes": dict(sorted(dtype_storage.items())),
        "partition": partition,
        "fp8_pair_count": len(pairs),
        "fp8_pair_counts_by_partition": pair_counts,
        "quantized_operation_counts": operation_counts,
    }
    return rows, pairs, stats


def locked_file_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": row["path"],
        "size_bytes": row["size_bytes"],
        "sha256": row["sha256"],
        "sha256_source": row.get("sha256_source"),
        "lfs_oid": row.get("lfs_oid"),
        "git_oid": row.get("git_oid"),
    }


def build_lock(
    resolution: ResolutionInput,
    transport: Transport,
    *,
    profile: ExpectedProfile = PRODUCTION_PROFILE,
    emit_progress: bool = True,
    shard_workers: int = 1,
) -> dict[str, Any]:
    require(
        1 <= shard_workers <= MAX_SHARD_WORKERS,
        f"shard_workers must be between one and {MAX_SHARD_WORKERS}",
    )
    config_body = metadata_file(transport, resolution.files[CONFIG_PATH], CONFIG_PATH)
    index_body = metadata_file(transport, resolution.files[INDEX_PATH], INDEX_PATH)
    recipe = parse_recipe(config_body)
    weight_map = parse_index(index_body, {row["path"] for row in resolution.shards})

    def scan_shard(
        shard: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        shard_rows, header_lock = read_shard_header(transport, shard)
        expected_names = sorted(name for name, path in weight_map.items() if path == shard["path"])
        actual_names = [row["name"] for row in shard_rows]
        require(actual_names == expected_names, f"header tensor set differs from index for {shard['path']}")
        return shard_rows, header_lock

    completed: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
    if shard_workers == 1:
        for shard in resolution.shards:
            completed.append(scan_shard(shard))
            if emit_progress:
                header_lock = completed[-1][1]
                print(
                    "VNEXT BLOCK FP8 SOURCE LOCK PROGRESS: "
                    f"shard={len(completed)}/{len(resolution.shards)} "
                    f"path={header_lock['path']} "
                    f"header_bytes={header_lock['header_bytes_requested']} "
                    f"tensors={header_lock['tensor_count']} payload_bytes_requested=0",
                    file=sys.stderr,
                    flush=True,
                )
    else:
        futures: dict[
            Future[tuple[list[dict[str, Any]], dict[str, Any]]], str
        ] = {}
        with ThreadPoolExecutor(
            max_workers=shard_workers,
            thread_name_prefix="fp8-header",
        ) as executor:
            try:
                for shard in resolution.shards:
                    futures[executor.submit(scan_shard, shard)] = shard["path"]
                for future in as_completed(futures):
                    result = future.result()
                    completed.append(result)
                    if emit_progress:
                        header_lock = result[1]
                        print(
                            "VNEXT BLOCK FP8 SOURCE LOCK PROGRESS: "
                            f"shard={len(completed)}/{len(resolution.shards)} "
                            f"path={header_lock['path']} "
                            f"header_bytes={header_lock['header_bytes_requested']} "
                            f"tensors={header_lock['tensor_count']} payload_bytes_requested=0",
                            file=sys.stderr,
                            flush=True,
                        )
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    require(len(completed) == len(resolution.shards), "not all shard headers completed")
    completed.sort(key=lambda result: result[1]["path"])
    tensors: list[dict[str, Any]] = []
    shard_headers: list[dict[str, Any]] = []
    for shard_rows, header_lock in completed:
        tensors.extend(shard_rows)
        shard_headers.append(header_lock)
        require(len(tensors) <= MAX_TOTAL_TENSORS, "aggregate tensor count exceeds limit")

    tensors.sort(key=lambda row: row["name"])
    require([row["name"] for row in tensors] == sorted(weight_map), "aggregate header tensor set differs from index")
    tensors, pairs, stats = enrich_and_validate_tensors(tensors, recipe, profile)

    files = [locked_file_identity(row) for row in sorted(resolution.files.values(), key=lambda row: row["path"])]
    checkpoint_content_document = {
        "repository": CHECKPOINT_REPO,
        "revision": CHECKPOINT_REVISION,
        "files": files,
    }
    checkpoint_content_digest = canonical_sha256(checkpoint_content_document)
    source_schema_document = {
        "recipe": {
            "quant_method": recipe["quant_method"],
            "source_dtype": recipe["source_dtype"],
            "format": recipe["format"],
            "activation_scheme": recipe["activation_scheme"],
            "weight_block_size": recipe["weight_block_size"],
            "modules_to_not_convert_digest": recipe["modules_to_not_convert_digest"],
        },
        "tensors": tensors,
        "fp8_pairs": pairs,
    }
    source_schema_fingerprint = canonical_sha256(source_schema_document)

    semantic_source = as_object(resolution.lane.get("semantic_source"), "semantic_source")
    chat_template = as_object(resolution.lane.get("chat_template"), "chat_template")
    return {
        "schema_version": 1,
        "artifact_type": "vnext_block_fp8_source_header_lock_diagnostic",
        "generated_at": utc_now(),
        "checkpoint": {
            "id": CHECKPOINT_ID,
            "repository": CHECKPOINT_REPO,
            "revision": CHECKPOINT_REVISION,
            "license": CHECKPOINT_LICENSE,
            "architecture": CHECKPOINT_ARCHITECTURE,
            "model_type": CHECKPOINT_MODEL_TYPE,
            "language_model_only": False,
            "backend": "cuda",
            "format": CHECKPOINT_FORMAT,
        },
        "input_resolution": {
            "path": resolution.resolution_path,
            "sha256": resolution.resolution_sha256,
            "artifact_type": resolution.document["artifact_type"],
            "catalog_id": resolution.document.get("catalog_id"),
            "resolver": resolution.document.get("resolver"),
        },
        "transport_policy": {
            "provenance": transport.provenance,
            "credentials_used": False,
            "shard_workers": shard_workers,
            "shard_workers_hard_max": MAX_SHARD_WORKERS,
            "metadata_max_bytes": MAX_METADATA_BYTES,
            "safetensors_header_max_bytes": MAX_HEADER_BYTES,
            "per_shard_requests": ["bytes=0-7", "bytes=8-(8+header_length-1)"],
            "strict_content_range": True,
            "locked_total_size_required": True,
            "tensor_payload_bytes_requested": 0,
            "signed_redirect_urls_recorded": False,
        },
        "content_lock": {
            "files": files,
            "file_count": len(files),
            "checkpoint_content_digest": checkpoint_content_digest,
            "digest_semantics": "canonical repository/revision/path/size/content identities from resolver",
        },
        "semantic_lock": {
            "repository": semantic_source.get("repo"),
            "revision": semantic_source.get("revision"),
            "chat_template_content_sha256": chat_template.get("content_sha256"),
            "chat_template_container_sha256": chat_template.get("container_sha256"),
        },
        "source_recipe": recipe,
        "index_lock": {
            "path": INDEX_PATH,
            "sha256": resolution.files[INDEX_PATH]["sha256"],
            "index_authoritative": True,
            "shard_count": len(resolution.shards),
            "tensor_count": len(weight_map),
        },
        "shard_headers": shard_headers,
        "tensors": tensors,
        "fp8_value_scale_pairs": pairs,
        "inventory": stats,
        "fingerprints": {
            "checkpoint_content_digest": checkpoint_content_digest,
            "source_schema_fingerprint": source_schema_fingerprint,
            "execution_contract_fingerprint": None,
            "quality_vector_digest": None,
        },
        "execution_mapping": {
            "scope": "execution-eligible text FP8 value/scale pairs only",
            "operation_counts": stats["quantized_operation_counts"],
            "catalog_coverage": "not_evaluated_by_metadata_tool",
            "execution_contract_fingerprint": None,
        },
        "diagnostic_limitations": [
            "This artifact is not model-lock.json and is not a checkpoint terminal receipt.",
            "No tensor payload byte was downloaded or numerically validated.",
            "Backend catalog/provider coverage is not established by metadata.",
            "Execution contract and quality-vector identities remain null.",
        ],
    }


class FixtureTransport(Transport):
    provenance = "internal_no_network_selftest_fixture"

    def __init__(
        self,
        metadata: dict[str, bytes],
        shards: dict[str, bytes],
    ) -> None:
        self.metadata = metadata
        self.shards = shards

    def fetch_metadata(self, path: str, expected_size: int, expected_sha256: str) -> bytes:
        require(path in self.metadata, f"fixture lacks metadata {path}")
        body = self.metadata[path]
        require(len(body) == expected_size, f"fixture metadata size differs for {path}")
        require(sha256_bytes(body) == expected_sha256, f"fixture metadata SHA differs for {path}")
        return body

    def fetch_range(self, path: str, start: int, end: int, expected_total: int) -> HttpResponse:
        require(path in self.shards, f"fixture lacks shard {path}")
        body = self.shards[path]
        require(len(body) == expected_total, f"fixture shard total differs for {path}")
        require(0 <= start <= end < len(body), f"fixture range is invalid for {path}")
        part = body[start : end + 1]
        return HttpResponse(
            status=206,
            headers={
                "content-range": f"bytes {start}-{end}/{len(body)}",
                "content-length": str(len(part)),
            },
            body=part,
        )


def make_safetensors_fixture(tensors: dict[str, tuple[str, list[int]]]) -> bytes:
    header: dict[str, Any] = {}
    payload = bytearray()
    for name in sorted(tensors):
        dtype, shape = tensors[name]
        size = tensor_storage_bytes(dtype, shape, f"fixture {name}")
        start = len(payload)
        payload.extend(b"\0" * size)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    raw_header = canonical_json_bytes(header)
    raw_header += b" " * ((-len(raw_header)) % 8)
    return struct.pack("<Q", len(raw_header)) + raw_header + bytes(payload)


def selftest_world(case: str) -> tuple[ResolutionInput, FixtureTransport, ExpectedProfile]:
    require(case in {"good", "bad_scale", "unknown_namespace", "orphan_sidecar"}, "bad fixture case")
    config = canonical_json_bytes(
        {
            "architectures": [CHECKPOINT_ARCHITECTURE],
            "language_model_only": False,
            "model_type": CHECKPOINT_MODEL_TYPE,
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "modules_to_not_convert": ["lm_head"],
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            },
        }
    )
    shard_tensors: dict[str, dict[str, tuple[str, list[int]]]] = {
        "layers-0.safetensors": {
            "lm_head.weight": ("BF16", [8, 8]),
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": (
                "F8_E4M3",
                [129, 257],
            ),
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv": (
                "BF16",
                [2, 3],
            ),
            "model.language_model.layers.0.mlp.gate_proj.weight": ("F8_E4M3", [128, 256]),
            "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv": ("BF16", [1, 2]),
            "mtp.layers.0.mlp.down_proj.weight": ("F8_E4M3", [128, 128]),
            "mtp.layers.0.mlp.down_proj.weight_scale_inv": ("BF16", [1, 1]),
            "mtp.norm.weight": ("BF16", [8]),
        }
    }
    if case == "bad_scale":
        shard_tensors["layers-0.safetensors"][
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv"
        ] = ("BF16", [1, 3])
    elif case == "unknown_namespace":
        shard_tensors["layers-0.safetensors"]["mystery.weight"] = ("BF16", [1])
    elif case == "orphan_sidecar":
        shard_tensors["layers-0.safetensors"]["mtp.orphan.weight_scale_inv"] = (
            "BF16",
            [1, 1],
        )

    shards = {path: make_safetensors_fixture(tensors) for path, tensors in shard_tensors.items()}
    weight_map = {
        name: path
        for path, tensors in shard_tensors.items()
        for name in tensors
    }
    index = canonical_json_bytes({"metadata": {}, "weight_map": weight_map})
    metadata = {CONFIG_PATH: config, INDEX_PATH: index}
    file_rows = []
    for path, body in metadata.items():
        file_rows.append(
            {
                "path": path,
                "size_bytes": len(body),
                "sha256": sha256_bytes(body),
                "sha256_source": "downloaded_content",
                "content_request_url": stable_file_url(path),
            }
        )
    for index_number, (path, body) in enumerate(shards.items(), start=1):
        digest = hashlib.sha256(f"fixture-{case}-{index_number}".encode()).hexdigest()
        file_rows.append(
            {
                "path": path,
                "size_bytes": len(body),
                "sha256": digest,
                "lfs_oid": digest,
                "sha256_source": "hugging_face_lfs_oid",
            }
        )
    resolution_doc = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_model_resolution",
        "catalog_id": "selftest",
        "resolver": {"path": "fixture", "sha256": "f" * 64},
        "lanes": [
            {
                "catalog_lane_id": CHECKPOINT_ID,
                "model_id": CHECKPOINT_ID,
                "backend": "cuda",
                "format": CHECKPOINT_FORMAT,
                "safetensors_shard_naming": "index_authoritative",
                "weight_source": {
                    "repo": CHECKPOINT_REPO,
                    "revision": CHECKPOINT_REVISION,
                    "gated": False,
                    "license": {"hugging_face_id": CHECKPOINT_LICENSE, "files": []},
                    "files": file_rows,
                },
                "semantic_source": {"repo": CHECKPOINT_REPO, "revision": CHECKPOINT_REVISION},
                "chat_template": {
                    "content_sha256": "a" * 64,
                    "container_sha256": "b" * 64,
                },
            }
        ],
    }
    resolution_raw = canonical_json_bytes(resolution_doc)
    resolution = ResolutionInput(
        document=resolution_doc,
        lane=resolution_doc["lanes"][0],
        files={row["path"]: row for row in file_rows},
        shards=[row for row in file_rows if row["path"].endswith(".safetensors")],
        resolution_path="internal-selftest-fixture.json",
        resolution_sha256=sha256_bytes(resolution_raw),
    )
    partition_counts = {
        "execution_eligible_text": 5,
        "typed_nonexecuted_visual": 0,
        "typed_nonexecuted_mtp": 3,
        "unknown": 0,
    }
    dtype_counts = {"BF16": 5, "F8_E4M3": 3}
    if case == "unknown_namespace":
        partition_counts["unknown"] = 1
        dtype_counts["BF16"] += 1
    elif case == "orphan_sidecar":
        partition_counts["typed_nonexecuted_mtp"] += 1
        dtype_counts["BF16"] += 1
    profile = ExpectedProfile(
        shard_count=1,
        tensor_count=sum(len(tensors) for tensors in shard_tensors.values()),
        partition_counts=partition_counts,
        dtype_counts=dtype_counts,
        pair_counts={
            "execution_eligible_text": 2,
            "typed_nonexecuted_visual": 0,
            "typed_nonexecuted_mtp": 1,
        },
        operation_counts={
            OP_GATED_DELTA: 1,
            OP_CAUSAL_ATTENTION: 0,
            OP_DENSE_SWIGLU: 1,
            OP_LOGITS: 0,
        },
    )
    return resolution, FixtureTransport(metadata, shards), profile


def expect_reject(case: str, message_fragment: str) -> None:
    resolution, transport, profile = selftest_world(case)
    try:
        build_lock(resolution, transport, profile=profile, emit_progress=False)
    except SourceLockError as exc:
        require(message_fragment in str(exc), f"{case} rejected for the wrong reason: {exc}")
        return
    raise SourceLockError(f"{case} fixture was accepted")


def run_selftest() -> None:
    class FakeRangeHandle:
        def __init__(self, *, fail_read: bool) -> None:
            self.status = 206
            self.headers = {
                "Content-Range": "bytes 0-7/1024",
                "Content-Length": "8",
            }
            self.fail_read = fail_read

        def __enter__(self) -> "FakeRangeHandle":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            if self.fail_read:
                raise TimeoutError("synthetic body-read timeout")
            return b"12345678"

    class ReadTimeoutThenSuccessTransport(NetworkTransport):
        def __init__(self) -> None:
            super().__init__(timeout_seconds=1.0, retries=1)
            self.attempts = 0

        def _open_once(self, _request: urllib.request.Request) -> FakeRangeHandle:
            self.attempts += 1
            return FakeRangeHandle(fail_read=self.attempts == 1)

    retry_transport = ReadTimeoutThenSuccessTransport()
    retried = retry_transport.fetch_range("layers-0.safetensors", 0, 7, 1024)
    require(retried.body == b"12345678", "body-read retry returned wrong bytes")
    require(retry_transport.attempts == 2, "body-read timeout was not retried exactly once")

    validate_content_range("bytes 0-7/1024", 0, 7, 1024, "fixture.safetensors")
    try:
        validate_content_range("bytes 0-7/2048", 0, 7, 1024, "fixture.safetensors")
    except SourceLockError as exc:
        require("mismatched Content-Range" in str(exc), "bad total rejected for wrong reason")
    else:
        raise SourceLockError("mismatched Content-Range total was accepted")
    resolution, transport, profile = selftest_world("good")
    lock = build_lock(resolution, transport, profile=profile, emit_progress=False)
    require(lock["inventory"]["tensor_count"] == 8, "good fixture tensor count differs")
    require(lock["inventory"]["fp8_pair_count"] == 3, "good fixture pair count differs")
    require(lock["inventory"]["partition"]["unknown"]["tensor_count"] == 0, "good fixture has unknown tensors")
    require(lock["fingerprints"]["execution_contract_fingerprint"] is None, "execution digest must be null")
    require(lock["fingerprints"]["quality_vector_digest"] is None, "quality digest must be null")
    parallel_lock = build_lock(
        resolution,
        transport,
        profile=profile,
        emit_progress=False,
        shard_workers=MAX_SHARD_WORKERS,
    )
    require(
        parallel_lock["fingerprints"] == lock["fingerprints"],
        "bounded parallel header scan changed source identities",
    )
    expect_reject("bad_scale", "bad block scale shape")
    expect_reject("unknown_namespace", "unknown tensor namespace")
    expect_reject("orphan_sidecar", "orphan FP8 sidecar")
    print("VNEXT BLOCK FP8 SOURCE LOCK SELF-TEST PASS")


def write_output(out_arg: Path, document: dict[str, Any]) -> Path:
    output = out_arg if out_arg.suffix.lower() == ".json" else out_arg / OUTPUT_NAME
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=Path, help="runtime_vnext_model_resolver.py output")
    parser.add_argument("--out", type=Path, help="output JSON path or artifact directory")
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--shard-workers", type=int, default=MAX_SHARD_WORKERS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        require(args.resolution is None and args.out is None, "--self-test cannot be combined with live inputs")
    else:
        require(args.resolution is not None, "--resolution is required")
        require(args.out is not None, "--out is required")
        require(args.timeout_seconds > 0, "--timeout-seconds must be positive")
        require(0 <= args.retries <= 5, "--retries must be between zero and five")
        require(
            1 <= args.shard_workers <= MAX_SHARD_WORKERS,
            f"--shard-workers must be between one and {MAX_SHARD_WORKERS}",
        )
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.self_test:
            run_selftest()
            return 0
        resolution_path = args.resolution
        if not resolution_path.is_absolute():
            resolution_path = ROOT / resolution_path
        resolution = load_resolution(resolution_path)
        transport = NetworkTransport(
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
        )
        document = build_lock(
            resolution,
            transport,
            shard_workers=args.shard_workers,
        )
        output = write_output(args.out, document)
        print(f"VNEXT BLOCK FP8 SOURCE HEADER LOCK DIAGNOSTIC: {output}")
        return 0
    except SourceLockError as exc:
        print(f"VNEXT BLOCK FP8 SOURCE LOCK ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
