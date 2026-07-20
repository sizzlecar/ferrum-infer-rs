#!/usr/bin/env python3
"""Validate the checked-in Runtime vNext numerical tolerance catalog."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_REPO_PATH = "scripts/release/configs/runtime_vnext_numerical_tolerances.json"
DEFAULT_CATALOG = REPO_ROOT / CATALOG_REPO_PATH

SELF_TEST_PASS = "RUNTIME VNEXT NUMERICAL TOLERANCE SELF-TEST PASS"
WORKTREE_VALID = "RUNTIME VNEXT NUMERICAL TOLERANCE WORKTREE VALID"
FOUNDATION_PASS = "RUNTIME VNEXT NUMERICAL TOLERANCE FOUNDATION PASS"
COMPLETE_PASS = "RUNTIME VNEXT NUMERICAL TOLERANCE COMPLETE PASS"

CATALOG_SCHEMA_VERSION = 1
G08A_PROFILE = "g08a_m1_metal.v1"
MINIMUM_COSINE = 0.999
MAXIMUM_RELATIVE_L2 = 0.01

ROOT_FIELDS = frozenset({"schema_version", "coverage", "rows"})
COVERAGE_FIELDS = frozenset(
    {"profile", "status", "scope", "missing_required_coverage"}
)
ROW_FIELDS = frozenset(
    {
        "tolerance_id",
        "coverage_markers",
        "backend",
        "model_scope",
        "operation_id",
        "operation_schema_version",
        "checkpoint_kind",
        "checkpoint_name",
        "dtype",
        "quant_format",
        "shape_domain",
        "oracle_identity",
        "oracle_precision",
        "bounds",
        "invariants",
        "basis",
        "source_commit",
        "owner",
        "review_commit",
        "row_fingerprint",
    }
)
SHAPE_DOMAIN_FIELDS = frozenset({"fixture_id", "dimensions", "semantics"})
BOUNDS_FIELDS = frozenset({"cosine_min", "relative_l2_max", "max_abs_max"})
INVARIANT_FIELDS = frozenset({"exact_shape", "exact_dtype", "max_nan", "max_inf"})
BASIS_FIELDS = frozenset({"kind", "source_path", "test_name", "assertion"})

G08A_REQUIRED_COVERAGE = frozenset(
    {
        "checkpoint.full_model",
        "checkpoint.full_vocab_logits",
        "layer.full_attention",
        "layer.linear_attention",
        "operation.causal_paged_attention@2.0.fixed_page_split",
        "operation.dense_linear@1.0",
        "operation.dense_swiglu@1.0",
        "operation.gated_delta_recurrent_attention@4.0.log_rate_grouped",
        "operation.gated_delta_recurrent_attention@4.0.negative_rate_interleaved",
        "operation.last_token_dense_linear@1.0",
        "operation.residual_add@1.0",
        "operation.rms_norm@1.0",
        "operation.token_embedding@1.0",
        "quant_format.gguf_q4_k_m",
        "state.causal_attention.kv_state",
        "state.gated_delta.conv_state.log_rate_grouped",
        "state.gated_delta.conv_state.negative_rate_interleaved",
        "state.gated_delta.delta_state.log_rate_grouped",
        "state.gated_delta.delta_state.negative_rate_interleaved",
    }
)
G08A_SCOPE = (
    "Exact Qwen3.5-4B Metal operation/state/layer/model/logit fixtures; "
    "token sequence equality remains a separate exact gate"
)

# Coverage is awarded only to reviewed, exact oracle descriptors. An oracle name
# that merely looks like a CPU/Transformers reference is not trusted.
TRUSTED_ORACLE_REGISTRY: dict[str, dict[str, str]] = {
    "cpu.fp32.rust.causal_attention_reference": {
        "oracle_precision": "fp32",
        "source_commit": "ecaeb5087ad45a5148d917fdab63d83cb046d678",
        "basis_kind": "checked_in_conformance_test",
        "source_path": (
            "crates/ferrum-kernels/src/backend/metal/vnext_ops/"
            "causal_attention_tests.rs"
        ),
        "test_name": (
            "fixed_page_attention_matches_cpu_and_preserves_split_decode_state_on_real_metal"
        ),
    },
    "cpu.fp32.rust.gated_delta_reference": {
        "oracle_precision": "fp32",
        "source_commit": "ecaeb5087ad45a5148d917fdab63d83cb046d678",
        "basis_kind": "checked_in_conformance_test",
        "source_path": (
            "crates/ferrum-kernels/src/backend/metal/vnext_ops/"
            "gated_delta_attention_tests.rs"
        ),
        "test_name": (
            "recurrent_core_matches_cpu_and_preserves_split_decode_state_on_real_metal"
        ),
    },
    "cpu.fp32.rust.gated_delta_conv_state_reference": {
        "oracle_precision": "fp32",
        "source_commit": "ecaeb5087ad45a5148d917fdab63d83cb046d678",
        "basis_kind": "checked_in_conformance_test",
        "source_path": (
            "crates/ferrum-kernels/src/backend/metal/vnext_ops/"
            "gated_delta_attention_tests.rs"
        ),
        "test_name": (
            "recurrent_core_matches_cpu_and_preserves_split_decode_state_on_real_metal"
        ),
    },
    "cpu.fp32.rust.gated_delta_delta_state_reference": {
        "oracle_precision": "fp32",
        "source_commit": "ecaeb5087ad45a5148d917fdab63d83cb046d678",
        "basis_kind": "checked_in_conformance_test",
        "source_path": (
            "crates/ferrum-kernels/src/backend/metal/vnext_ops/"
            "gated_delta_attention_tests.rs"
        ),
        "test_name": (
            "recurrent_core_matches_cpu_and_preserves_split_decode_state_on_real_metal"
        ),
    },
}


def _gated_delta_shape_domain(
    checkpoint_name: str, decay: str, mapping: str
) -> dict[str, Any]:
    dimensions = {
        "key_dim": 128,
        "key_heads": 16,
        "tokens": 4,
        "value_dim": 128,
        "value_heads": 32,
    }
    if checkpoint_name == "output":
        dimensions["conv_kernel"] = 4
        fixture_prefix = "gated_delta"
    elif checkpoint_name == "conv_state":
        dimensions.update(
            {"conv_kernel": 4, "conv_state_width": 3, "qkv_features": 8192}
        )
        fixture_prefix = "gated_delta.conv_state"
    elif checkpoint_name == "delta_state":
        fixture_prefix = "gated_delta.delta_state"
    else:
        raise AssertionError(f"unsupported checked-in GDN fixture: {checkpoint_name}")
    fixture_decay = "log_rate" if decay == "log_rate" else "negative_rate"
    fixture_mapping = "grouped" if mapping == "grouped_by_key_head" else "interleaved"
    return {
        "fixture_id": f"{fixture_prefix}.{fixture_decay}.{fixture_mapping}.split_decode",
        "dimensions": dimensions,
        "semantics": {
            "decay_parameterization": decay,
            "split_segments": [3, 1],
            "value_head_mapping": mapping,
        },
    }


def _coverage_selector(
    *,
    model_scope: str,
    operation_id: str,
    operation_schema_version: str,
    checkpoint_kind: str,
    checkpoint_name: str,
    dtype: str,
    quant_format: str,
    shape_domain: dict[str, Any],
    oracle_identity: str,
) -> dict[str, Any]:
    return {
        "backend": "metal",
        "model_scope": model_scope,
        "operation_id": operation_id,
        "operation_schema_version": operation_schema_version,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_name": checkpoint_name,
        "dtype": dtype,
        "quant_format": quant_format,
        "shape_domain": shape_domain,
        "oracle_identity": oracle_identity,
    }


# Only finalized coverage has a rule. Missing G08A markers intentionally have no
# rule until their exact fixture and independent oracle are reviewed.
G08A_COVERAGE_RULES: dict[str, dict[str, Any]] = {
    "operation.causal_paged_attention@2.0.fixed_page_split": _coverage_selector(
        model_scope="operation_contract",
        operation_id="operation.causal_paged_attention",
        operation_schema_version="2.0",
        checkpoint_kind="operation_output",
        checkpoint_name="output",
        dtype="fp16",
        quant_format="none",
        shape_domain={
            "fixture_id": "causal_attention.fixed_page.split_decode",
            "dimensions": {
                "head_dim": 32,
                "key_value_heads": 1,
                "query_heads": 2,
                "rope_dim": 16,
                "tokens": 2,
            },
            "semantics": {"fixed_page_kv": True, "split_segments": [1, 1]},
        },
        oracle_identity="cpu.fp32.rust.causal_attention_reference",
    ),
}

for _decay, _mapping, _suffix in (
    ("log_rate", "grouped_by_key_head", "log_rate_grouped"),
    ("negative_rate", "interleaved_by_key_head", "negative_rate_interleaved"),
):
    G08A_COVERAGE_RULES[
        f"operation.gated_delta_recurrent_attention@4.0.{_suffix}"
    ] = _coverage_selector(
        model_scope="qwen3.5-4b",
        operation_id="operation.gated_delta_recurrent_attention",
        operation_schema_version="4.0",
        checkpoint_kind="operation_output",
        checkpoint_name="output",
        dtype="fp16",
        quant_format="none",
        shape_domain=_gated_delta_shape_domain("output", _decay, _mapping),
        oracle_identity="cpu.fp32.rust.gated_delta_reference",
    )
    G08A_COVERAGE_RULES[f"state.gated_delta.conv_state.{_suffix}"] = (
        _coverage_selector(
            model_scope="qwen3.5-4b",
            operation_id="operation.gated_delta_recurrent_attention",
            operation_schema_version="4.0",
            checkpoint_kind="state",
            checkpoint_name="conv_state",
            dtype="fp16",
            quant_format="none",
            shape_domain=_gated_delta_shape_domain("conv_state", _decay, _mapping),
            oracle_identity="cpu.fp32.rust.gated_delta_conv_state_reference",
        )
    )
    G08A_COVERAGE_RULES[f"state.gated_delta.delta_state.{_suffix}"] = (
        _coverage_selector(
            model_scope="qwen3.5-4b",
            operation_id="operation.gated_delta_recurrent_attention",
            operation_schema_version="4.0",
            checkpoint_kind="state",
            checkpoint_name="delta_state",
            dtype="fp32",
            quant_format="none",
            shape_domain=_gated_delta_shape_domain("delta_state", _decay, _mapping),
            oracle_identity="cpu.fp32.rust.gated_delta_delta_state_reference",
        )
    )

IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._@-]{0,191}$")
OPERATION_ID_RE = re.compile(r"^operation\.[a-z0-9][a-z0-9._-]{0,182}$")
CONTRACT_VERSION_RE = re.compile(r"^[1-9][0-9]*\.[0-9]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

BACKENDS = frozenset({"cpu", "cuda", "metal", "shared"})
CHECKPOINT_KINDS = frozenset(
    {"operation_output", "state", "layer_output", "full_model", "full_vocab_logits"}
)
DTYPES = frozenset({"fp16", "bf16", "fp32"})
OWNERS = frozenset({"runtime-vnext-g03"})
BASIS_KINDS = frozenset({"checked_in_conformance_test"})


class CatalogError(RuntimeError):
    """The tolerance catalog is malformed or violates the numerical policy."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CatalogError(message)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CatalogError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise CatalogError(f"non-finite JSON constant is forbidden: {value}")


def load_catalog_bytes(payload: bytes) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CatalogError(f"catalog is not UTF-8: {error}") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except CatalogError:
        raise
    except json.JSONDecodeError as error:
        raise CatalogError(f"catalog is not valid JSON: {error}") from error


def load_worktree_catalog() -> Any:
    try:
        return load_catalog_bytes(DEFAULT_CATALOG.read_bytes())
    except OSError as error:
        raise CatalogError(f"cannot read canonical catalog {DEFAULT_CATALOG}: {error}") from error


def _git(args: list[str], *, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise CatalogError(f"git {' '.join(args)} failed: {detail}")
    if text:
        return result.stdout.decode("utf-8", errors="strict").strip()
    return result.stdout


def load_catalog_from_git(
    revision: str, expected_blob_sha: str | None
) -> tuple[Any, dict[str, str]]:
    commit = _git(["rev-parse", f"{revision}^{{commit}}"])
    _require(isinstance(commit, str) and GIT_SHA_RE.fullmatch(commit) is not None,
             "catalog revision did not resolve to a full commit SHA")
    blob = _git(["rev-parse", f"{commit}:{CATALOG_REPO_PATH}"])
    _require(isinstance(blob, str) and GIT_SHA_RE.fullmatch(blob) is not None,
             "canonical catalog did not resolve to a Git blob SHA")
    if expected_blob_sha is not None:
        _require(GIT_SHA_RE.fullmatch(expected_blob_sha) is not None,
                 "--expected-blob-sha must be a 40-character Git object id")
        _require(blob == expected_blob_sha,
                 f"catalog Git blob mismatch: expected {expected_blob_sha}, resolved {blob}")
    payload = _git(["cat-file", "blob", blob], text=False)
    _require(isinstance(payload, bytes), "git cat-file returned non-bytes payload")
    return load_catalog_bytes(payload), {"commit": commit, "git_blob_sha": blob}


def canonical_json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CatalogError(f"value is not canonical JSON: {error}") from error
    return hashlib.sha256(payload).hexdigest()


def row_fingerprint(row: dict[str, Any]) -> str:
    material = {key: value for key, value in row.items() if key != "row_fingerprint"}
    return canonical_json_sha256(material)


def _require_exact_fields(
    value: Any, fields: frozenset[str], label: str
) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    _require(not fields - actual, f"{label} is missing fields: {sorted(fields - actual)}")
    _require(not actual - fields, f"{label} has unknown fields: {sorted(actual - fields)}")
    return value


def _require_text(value: Any, label: str, maximum_length: int = 512) -> str:
    _require(isinstance(value, str), f"{label} must be a string")
    _require(value == value.strip() and bool(value), f"{label} must be non-empty and trimmed")
    _require(len(value) <= maximum_length, f"{label} exceeds {maximum_length} characters")
    return value


def _require_identifier(value: Any, label: str) -> str:
    text = _require_text(value, label, 192)
    _require(IDENTIFIER_RE.fullmatch(text) is not None,
             f"{label} is not a canonical identifier")
    return text


def _require_number(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{label} must be numeric")
    number = float(value)
    _require(math.isfinite(number), f"{label} must be finite")
    return number


def _require_commit(value: Any, label: str) -> str:
    commit = _require_text(value, label, 40)
    _require(GIT_SHA_RE.fullmatch(commit) is not None, f"{label} is not a full Git SHA")
    return commit


def _require_repo_path(value: Any, label: str) -> str:
    text = _require_text(value, label, 512)
    path = PurePosixPath(text)
    _require(not path.is_absolute() and ".." not in path.parts and text == path.as_posix(),
             f"{label} must be a canonical repository-relative path")
    return text


def _validate_semantic_value(value: Any, label: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        _require(value >= 0, f"{label} integer must be non-negative")
        return
    if isinstance(value, float):
        _require(math.isfinite(value), f"{label} float must be finite")
        return
    if isinstance(value, str):
        identifier = _require_identifier(value, label)
        _require(identifier not in {"any", "all"} and "+" not in identifier and ".." not in identifier,
                 f"{label} must select one exact semantic value")
        return
    if isinstance(value, list):
        _require(bool(value), f"{label} list must not be empty")
        for index, item in enumerate(value):
            _require(not isinstance(item, (list, dict)), f"{label}[{index}] must be scalar")
            _validate_semantic_value(item, f"{label}[{index}]")
        return
    raise CatalogError(f"{label} has unsupported exact-fixture value type")


def _validate_shape_domain(value: Any, label: str) -> dict[str, Any]:
    domain = _require_exact_fields(value, SHAPE_DOMAIN_FIELDS, label)
    _require_identifier(domain["fixture_id"], f"{label}.fixture_id")
    dimensions = domain["dimensions"]
    _require(isinstance(dimensions, dict) and bool(dimensions),
             f"{label}.dimensions must be a non-empty object")
    for key, dimension in dimensions.items():
        _require_identifier(key, f"{label}.dimensions key")
        _require(isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0,
                 f"{label}.dimensions.{key} must be a positive integer")
    semantics = domain["semantics"]
    _require(isinstance(semantics, dict) and bool(semantics),
             f"{label}.semantics must be a non-empty object")
    for key, semantic in semantics.items():
        _require_identifier(key, f"{label}.semantics key")
        _validate_semantic_value(semantic, f"{label}.semantics.{key}")
    return domain


def _validate_bounds(value: Any, label: str) -> dict[str, Any]:
    bounds = _require_exact_fields(value, BOUNDS_FIELDS, label)
    cosine = _require_number(bounds["cosine_min"], f"{label}.cosine_min")
    relative_l2 = _require_number(bounds["relative_l2_max"], f"{label}.relative_l2_max")
    max_abs = _require_number(bounds["max_abs_max"], f"{label}.max_abs_max")
    _require(MINIMUM_COSINE <= cosine <= 1.0,
             f"{label}.cosine_min widens the MODEL_MATRIX minimum {MINIMUM_COSINE}")
    _require(0.0 <= relative_l2 <= MAXIMUM_RELATIVE_L2,
             f"{label}.relative_l2_max widens the MODEL_MATRIX maximum {MAXIMUM_RELATIVE_L2}")
    _require(max_abs >= 0.0, f"{label}.max_abs_max must be non-negative")
    return bounds


def _validate_invariants(value: Any, label: str) -> dict[str, Any]:
    invariants = _require_exact_fields(value, INVARIANT_FIELDS, label)
    _require(invariants["exact_shape"] is True, f"{label}.exact_shape must be true")
    _require(invariants["exact_dtype"] is True, f"{label}.exact_dtype must be true")
    for key in ("max_nan", "max_inf"):
        _require(invariants[key] == 0 and not isinstance(invariants[key], bool),
                 f"{label}.{key} must be integer zero")
    return invariants


def _validate_basis(value: Any, label: str) -> dict[str, Any]:
    basis = _require_exact_fields(value, BASIS_FIELDS, label)
    kind = _require_identifier(basis["kind"], f"{label}.kind")
    _require(kind in BASIS_KINDS, f"{label}.kind is unsupported")
    _require_repo_path(basis["source_path"], f"{label}.source_path")
    _require_identifier(basis["test_name"], f"{label}.test_name")
    _require_text(basis["assertion"], f"{label}.assertion", 1024)
    return basis


def _has_trusted_oracle(row: dict[str, Any]) -> bool:
    expected = TRUSTED_ORACLE_REGISTRY.get(row["oracle_identity"])
    if expected is None:
        return False
    basis = row["basis"]
    actual = {
        "oracle_precision": row["oracle_precision"],
        "source_commit": row["source_commit"],
        "basis_kind": basis["kind"],
        "source_path": basis["source_path"],
        "test_name": basis["test_name"],
    }
    return actual == expected


def _coverage_selector_material(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": row["backend"],
        "model_scope": row["model_scope"],
        "operation_id": row["operation_id"],
        "operation_schema_version": row["operation_schema_version"],
        "checkpoint_kind": row["checkpoint_kind"],
        "checkpoint_name": row["checkpoint_name"],
        "dtype": row["dtype"],
        "quant_format": row["quant_format"],
        "shape_domain": row["shape_domain"],
        "oracle_identity": row["oracle_identity"],
    }


def _derived_coverage_markers(row: dict[str, Any]) -> list[str]:
    if not _has_trusted_oracle(row):
        return []
    selector = _coverage_selector_material(row)
    return sorted(
        marker
        for marker, expected_selector in G08A_COVERAGE_RULES.items()
        if selector == expected_selector
    )


def _validate_row(
    value: Any, index: int, *, enforce_current_coverage: bool = True
) -> dict[str, Any]:
    label = f"rows[{index}]"
    row = _require_exact_fields(value, ROW_FIELDS, label)
    tolerance_id = _require_identifier(row["tolerance_id"], f"{label}.tolerance_id")
    markers = row["coverage_markers"]
    _require(isinstance(markers, list) and bool(markers),
             f"{label}.coverage_markers must be a non-empty list")
    normalized_markers = [
        _require_identifier(marker, f"{label}.coverage_markers[{marker_index}]")
        for marker_index, marker in enumerate(markers)
    ]
    _require(normalized_markers == sorted(set(normalized_markers)),
             f"{label}.coverage_markers must be unique and sorted")
    backend = _require_identifier(row["backend"], f"{label}.backend")
    _require(backend in BACKENDS, f"{label}.backend is unsupported")
    _require_identifier(row["model_scope"], f"{label}.model_scope")
    operation_id = _require_text(row["operation_id"], f"{label}.operation_id", 192)
    _require(OPERATION_ID_RE.fullmatch(operation_id) is not None,
             f"{label}.operation_id is not canonical")
    operation_version = _require_text(
        row["operation_schema_version"], f"{label}.operation_schema_version", 32
    )
    _require(CONTRACT_VERSION_RE.fullmatch(operation_version) is not None,
             f"{label}.operation_schema_version is not canonical")
    checkpoint_kind = _require_identifier(row["checkpoint_kind"], f"{label}.checkpoint_kind")
    _require(checkpoint_kind in CHECKPOINT_KINDS,
             f"{label}.checkpoint_kind is unsupported")
    _require_identifier(row["checkpoint_name"], f"{label}.checkpoint_name")
    dtype = _require_identifier(row["dtype"], f"{label}.dtype")
    _require(dtype in DTYPES, f"{label}.dtype is unsupported")
    quant_format = _require_identifier(row["quant_format"], f"{label}.quant_format")
    shape_domain = _validate_shape_domain(row["shape_domain"], f"{label}.shape_domain")
    oracle_identity = _require_identifier(row["oracle_identity"], f"{label}.oracle_identity")
    _require_identifier(row["oracle_precision"], f"{label}.oracle_precision")
    _validate_bounds(row["bounds"], f"{label}.bounds")
    _validate_invariants(row["invariants"], f"{label}.invariants")
    _validate_basis(row["basis"], f"{label}.basis")
    _require_commit(row["source_commit"], f"{label}.source_commit")
    owner = _require_identifier(row["owner"], f"{label}.owner")
    _require(owner in OWNERS, f"{label}.owner is not an approved catalog owner")
    _require_commit(row["review_commit"], f"{label}.review_commit")
    fingerprint = _require_text(row["row_fingerprint"], f"{label}.row_fingerprint", 64)
    _require(SHA256_RE.fullmatch(fingerprint) is not None,
             f"{label}.row_fingerprint is not SHA-256")
    expected = row_fingerprint(row)
    _require(fingerprint == expected,
             f"{label}.row_fingerprint mismatch: expected {expected}")
    derived_markers = _derived_coverage_markers(row)
    if enforce_current_coverage:
        _require(
            normalized_markers == derived_markers,
            f"{label}.coverage_markers do not match the typed row selector: "
            f"expected {derived_markers}, declared {normalized_markers}",
        )

    identity = canonical_json_sha256(
        {
            "backend": backend,
            "model_scope": row["model_scope"],
            "operation_id": operation_id,
            "operation_schema_version": operation_version,
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_name": row["checkpoint_name"],
            "dtype": dtype,
            "quant_format": quant_format,
            "shape_domain": shape_domain,
            "oracle_identity": oracle_identity,
        }
    )
    return {
        "identity": identity,
        "markers": normalized_markers,
        "tolerance_id": tolerance_id,
    }


def validate_catalog_document(value: Any, *, require_complete: bool = False) -> dict[str, Any]:
    unknown_rules = sorted(set(G08A_COVERAGE_RULES) - G08A_REQUIRED_COVERAGE)
    _require(
        not unknown_rules,
        f"coverage rules are outside the reviewed G08A profile: {unknown_rules}",
    )
    unregistered_rule_oracles = sorted(
        {
            selector["oracle_identity"]
            for selector in G08A_COVERAGE_RULES.values()
            if selector["oracle_identity"] not in TRUSTED_ORACLE_REGISTRY
        }
    )
    _require(
        not unregistered_rule_oracles,
        f"coverage rules use unregistered oracles: {unregistered_rule_oracles}",
    )
    document = _require_exact_fields(value, ROOT_FIELDS, "catalog")
    schema_version = document["schema_version"]
    _require(isinstance(schema_version, int) and not isinstance(schema_version, bool),
             "catalog.schema_version must be an integer")
    _require(schema_version == CATALOG_SCHEMA_VERSION,
             f"catalog.schema_version must be {CATALOG_SCHEMA_VERSION}")

    coverage = _require_exact_fields(document["coverage"], COVERAGE_FIELDS, "coverage")
    profile = _require_identifier(coverage["profile"], "coverage.profile")
    _require(profile == G08A_PROFILE, f"coverage.profile must be {G08A_PROFILE}")
    scope = _require_text(coverage["scope"], "coverage.scope", 512)
    _require(scope == G08A_SCOPE, "coverage.scope differs from the reviewed profile boundary")

    rows = document["rows"]
    _require(isinstance(rows, list) and bool(rows), "catalog.rows must be a non-empty list")
    tolerance_ids: list[str] = []
    identities: set[str] = set()
    actual_markers: set[str] = set()
    for index, row in enumerate(rows):
        validated = _validate_row(row, index)
        tolerance_id = validated["tolerance_id"]
        _require(tolerance_id not in tolerance_ids, f"duplicate tolerance id: {tolerance_id}")
        _require(validated["identity"] not in identities,
                 f"ambiguous duplicate exact fixture: {tolerance_id}")
        tolerance_ids.append(tolerance_id)
        identities.add(validated["identity"])
        actual_markers.update(validated["markers"])
    _require(tolerance_ids == sorted(tolerance_ids),
             "catalog rows must be sorted by tolerance_id")

    missing = sorted(G08A_REQUIRED_COVERAGE - actual_markers)
    declared_missing = coverage["missing_required_coverage"]
    _require(isinstance(declared_missing, list),
             "coverage.missing_required_coverage must be a list")
    declared = [
        _require_identifier(marker, f"coverage.missing_required_coverage[{index}]")
        for index, marker in enumerate(declared_missing)
    ]
    _require(declared == sorted(set(declared)),
             "coverage.missing_required_coverage must be unique and sorted")
    _require(declared == missing,
             f"coverage gap set is stale: expected {missing}, declared {declared}")
    expected_status = "complete" if not missing else "foundation_only"
    status = _require_identifier(coverage["status"], "coverage.status")
    _require(status == expected_status,
             f"coverage.status must be {expected_status} for the computed gap set")
    if require_complete:
        _require(not missing,
                 f"complete catalog required; missing coverage: {missing}")

    return {
        "coverage_profile": profile,
        "coverage_status": status,
        "missing_required_coverage": missing,
        "row_count": len(rows),
        "tolerance_ids": tolerance_ids,
    }


def _git_commit_exists(commit: str, label: str) -> None:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    _require(result.returncode == 0,
             f"{label} is not a commit in this repository: {commit}")


def _require_ancestor(ancestor: str, descendant: str, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    _require(result.returncode == 0, f"{label}: {ancestor} is not an ancestor of {descendant}")


def validate_catalog_provenance(document: dict[str, Any], catalog_commit: str) -> None:
    _git_commit_exists(catalog_commit, "catalog commit")
    for index, raw in enumerate(document["rows"]):
        row = raw
        label = f"rows[{index}]"
        source_commit = row["source_commit"]
        review_commit = row["review_commit"]
        _git_commit_exists(source_commit, f"{label}.source_commit")
        _git_commit_exists(review_commit, f"{label}.review_commit")
        _require_ancestor(source_commit, review_commit, f"{label} source/review lineage")
        _require_ancestor(review_commit, catalog_commit, f"{label} review/catalog lineage")
        basis = row["basis"]
        source_path = basis["source_path"]
        payload = _git(["show", f"{source_commit}:{source_path}"], text=False)
        _require(isinstance(payload, bytes), f"{label}.basis source is not a Git blob")
        test_name = basis["test_name"].encode("utf-8")
        _require(test_name in payload,
                 f"{label}.basis test is absent at source commit: {basis['test_name']}")


def _selector_material(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "backend",
            "model_scope",
            "operation_id",
            "operation_schema_version",
            "checkpoint_kind",
            "checkpoint_name",
            "dtype",
            "quant_format",
            "shape_domain",
            "oracle_identity",
            "oracle_precision",
        )
    }


def validate_no_widening_documents(
    current: dict[str, Any], previous: dict[str, Any]
) -> None:
    current_rows = {row["tolerance_id"]: row for row in current["rows"]}
    previous_rows = {row["tolerance_id"]: row for row in previous["rows"]}
    removed = sorted(set(previous_rows) - set(current_rows))
    _require(not removed, f"catalog removed or renamed reviewed tolerance rows: {removed}")
    for tolerance_id, previous_row in previous_rows.items():
        current_row = current_rows[tolerance_id]
        _require(
            _selector_material(current_row) == _selector_material(previous_row),
            f"reviewed tolerance selector changed in place: {tolerance_id}",
        )
        old_bounds = previous_row["bounds"]
        new_bounds = current_row["bounds"]
        _require(
            new_bounds["cosine_min"] >= old_bounds["cosine_min"],
            f"post-hoc cosine widening is forbidden: {tolerance_id}",
        )
        _require(
            new_bounds["relative_l2_max"] <= old_bounds["relative_l2_max"],
            f"post-hoc relative-L2 widening is forbidden: {tolerance_id}",
        )
        _require(
            new_bounds["max_abs_max"] <= old_bounds["max_abs_max"],
            f"post-hoc absolute widening is forbidden: {tolerance_id}",
        )
        _require(
            current_row["invariants"] == previous_row["invariants"],
            f"reviewed invariants changed in place: {tolerance_id}",
        )


def _catalog_history_commits(revision: str) -> list[str]:
    result = _git(
        ["log", "--first-parent", "--format=%H", revision, "--", CATALOG_REPO_PATH]
    )
    _require(isinstance(result, str), "git log returned non-text output")
    if not result:
        return []
    commits = result.splitlines()
    _require(
        all(GIT_SHA_RE.fullmatch(commit) is not None for commit in commits),
        "catalog history did not resolve to full commit SHAs",
    )
    _require(len(commits) == len(set(commits)), "catalog history contains duplicates")
    return commits


def _first_parent(commit: str) -> str | None:
    lineage = _git(["rev-list", "--parents", "-n", "1", commit])
    _require(isinstance(lineage, str), "git rev-list returned non-text output")
    parts = lineage.split()
    _require(bool(parts) and parts[0] == commit, "catalog commit lineage is malformed")
    return parts[1] if len(parts) > 1 else None


def _validate_historical_catalog_document(document: Any) -> None:
    catalog = _require_exact_fields(document, ROOT_FIELDS, "historical catalog")
    _require(
        catalog["schema_version"] == CATALOG_SCHEMA_VERSION,
        f"historical catalog.schema_version must be {CATALOG_SCHEMA_VERSION}",
    )
    _require_exact_fields(catalog["coverage"], COVERAGE_FIELDS, "historical coverage")
    rows = catalog["rows"]
    _require(
        isinstance(rows, list) and bool(rows),
        "historical catalog.rows must be a non-empty list",
    )
    tolerance_ids: list[str] = []
    identities: set[str] = set()
    for index, row in enumerate(rows):
        validated = _validate_row(row, index, enforce_current_coverage=False)
        tolerance_id = validated["tolerance_id"]
        _require(
            tolerance_id not in tolerance_ids,
            f"historical duplicate tolerance id: {tolerance_id}",
        )
        _require(
            validated["identity"] not in identities,
            f"historical ambiguous duplicate exact fixture: {tolerance_id}",
        )
        tolerance_ids.append(tolerance_id)
        identities.add(validated["identity"])
    _require(
        tolerance_ids == sorted(tolerance_ids),
        "historical catalog rows must be sorted by tolerance_id",
    )


def validate_no_widening_document_history(
    current: dict[str, Any], history: list[dict[str, Any]]
) -> None:
    for previous in history:
        validate_no_widening_documents(current, previous)


def validate_no_widening_from_revision(
    current: dict[str, Any], baseline_revision: str | None
) -> list[str]:
    if baseline_revision is None:
        return []
    commits = _catalog_history_commits(baseline_revision)
    for commit in commits:
        previous, _ = load_catalog_from_git(commit, None)
        try:
            _validate_historical_catalog_document(previous)
            validate_no_widening_documents(current, previous)
        except CatalogError as error:
            raise CatalogError(
                f"catalog violates historical version {commit}: {error}"
            ) from error
    return commits


def _self_test() -> None:
    document = _self_test_document()
    summary = validate_catalog_document(document)
    _require(summary["coverage_status"] == "foundation_only",
             "self-test fixture must remain incomplete")

    _expect_rejected(
        "foundation complete gate",
        lambda: validate_catalog_document(document, require_complete=True),
        "complete catalog required",
    )

    tampered = copy.deepcopy(document)
    tampered["rows"][0]["bounds"]["max_abs_max"] = 0.007
    _expect_rejected(
        "tampered fingerprint",
        lambda: validate_catalog_document(tampered),
        "row_fingerprint mismatch",
    )

    duplicate_id = copy.deepcopy(document)
    duplicate_id["rows"].append(copy.deepcopy(duplicate_id["rows"][0]))
    _expect_rejected(
        "duplicate tolerance id",
        lambda: validate_catalog_document(duplicate_id),
        "duplicate tolerance id",
    )

    ambiguous = copy.deepcopy(document)
    duplicate = copy.deepcopy(ambiguous["rows"][0])
    duplicate["tolerance_id"] = "runtime-vnext.test.duplicate"
    duplicate["row_fingerprint"] = row_fingerprint(duplicate)
    ambiguous["rows"].append(duplicate)
    ambiguous["rows"].sort(key=lambda row: row["tolerance_id"])
    _expect_rejected(
        "ambiguous exact fixture",
        lambda: validate_catalog_document(ambiguous),
        "ambiguous duplicate exact fixture",
    )

    widened = copy.deepcopy(document)
    widened["rows"][0]["bounds"]["relative_l2_max"] = 0.011
    widened["rows"][0]["row_fingerprint"] = row_fingerprint(widened["rows"][0])
    _expect_rejected(
        "policy widening",
        lambda: validate_catalog_document(widened),
        "widens the MODEL_MATRIX maximum",
    )

    false_complete = copy.deepcopy(document)
    false_complete["coverage"]["status"] = "complete"
    _expect_rejected(
        "false complete coverage",
        lambda: validate_catalog_document(false_complete),
        "coverage.status must be foundation_only",
    )

    forged_markers = copy.deepcopy(document)
    forged_markers["rows"][0]["coverage_markers"] = sorted(G08A_REQUIRED_COVERAGE)
    forged_markers["rows"][0]["row_fingerprint"] = row_fingerprint(
        forged_markers["rows"][0]
    )
    forged_markers["coverage"]["missing_required_coverage"] = []
    forged_markers["coverage"]["status"] = "complete"
    _expect_rejected(
        "forged complete coverage markers",
        lambda: validate_catalog_document(forged_markers, require_complete=True),
        "do not match the typed row selector",
    )

    untrusted_oracle = copy.deepcopy(document)
    untrusted_oracle["rows"][0]["oracle_precision"] = "fp16"
    untrusted_oracle["rows"][0]["row_fingerprint"] = row_fingerprint(
        untrusted_oracle["rows"][0]
    )
    _expect_rejected(
        "untrusted coverage oracle",
        lambda: validate_catalog_document(untrusted_oracle),
        "do not match the typed row selector",
    )

    forged_oracle_prefix = copy.deepcopy(document)
    forged_oracle_prefix["rows"][0]["oracle_identity"] = (
        "cpu.fp32.rust.unregistered_reference"
    )
    forged_oracle_prefix["rows"][0]["row_fingerprint"] = row_fingerprint(
        forged_oracle_prefix["rows"][0]
    )
    _expect_rejected(
        "unregistered trusted-looking oracle",
        lambda: validate_catalog_document(forged_oracle_prefix),
        "do not match the typed row selector",
    )

    for field, replacement in (
        ("backend", "cuda"),
        ("model_scope", "qwen3.5-4b"),
        ("operation_id", "operation.dense_linear"),
        ("quant_format", "gguf_q4_k_m"),
    ):
        wrong_selector = copy.deepcopy(document)
        wrong_selector["rows"][0][field] = replacement
        wrong_selector["rows"][0]["row_fingerprint"] = row_fingerprint(
            wrong_selector["rows"][0]
        )
        _expect_rejected(
            f"coverage selector {field}",
            lambda candidate=wrong_selector: validate_catalog_document(candidate),
            "do not match the typed row selector",
        )

    wrong_fixture = copy.deepcopy(document)
    wrong_fixture["rows"][0]["shape_domain"]["fixture_id"] = (
        "causal_attention.fixed_page.other"
    )
    wrong_fixture["rows"][0]["row_fingerprint"] = row_fingerprint(
        wrong_fixture["rows"][0]
    )
    _expect_rejected(
        "coverage selector fixture",
        lambda: validate_catalog_document(wrong_fixture),
        "do not match the typed row selector",
    )

    wrong_oracle_descriptor = copy.deepcopy(document)
    wrong_oracle_descriptor["rows"][0]["basis"]["test_name"] = "forged_test"
    wrong_oracle_descriptor["rows"][0]["row_fingerprint"] = row_fingerprint(
        wrong_oracle_descriptor["rows"][0]
    )
    _expect_rejected(
        "oracle descriptor mutation",
        lambda: validate_catalog_document(wrong_oracle_descriptor),
        "do not match the typed row selector",
    )

    wrong_oracle_implementation = copy.deepcopy(document)
    wrong_oracle_implementation["rows"][0]["source_commit"] = "d" * 40
    wrong_oracle_implementation["rows"][0]["row_fingerprint"] = row_fingerprint(
        wrong_oracle_implementation["rows"][0]
    )
    _expect_rejected(
        "oracle implementation commit mutation",
        lambda: validate_catalog_document(wrong_oracle_implementation),
        "do not match the typed row selector",
    )

    stale_gaps = copy.deepcopy(document)
    stale_gaps["coverage"]["missing_required_coverage"] = []
    _expect_rejected(
        "stale coverage gaps",
        lambda: validate_catalog_document(stale_gaps),
        "coverage gap set is stale",
    )

    narrowed = copy.deepcopy(document)
    narrowed["rows"][0]["bounds"]["max_abs_max"] = 0.005
    narrowed["rows"][0]["row_fingerprint"] = row_fingerprint(narrowed["rows"][0])
    validate_no_widening_documents(narrowed, document)

    history_widened = copy.deepcopy(document)
    history_widened["rows"][0]["bounds"]["max_abs_max"] = 0.007
    history_widened["rows"][0]["row_fingerprint"] = row_fingerprint(history_widened["rows"][0])
    _expect_rejected(
        "history absolute widening",
        lambda: validate_no_widening_documents(history_widened, document),
        "post-hoc absolute widening",
    )

    strict_oldest = copy.deepcopy(document)
    strict_oldest["rows"][0]["bounds"]["max_abs_max"] = 0.005
    strict_oldest["rows"][0]["row_fingerprint"] = row_fingerprint(
        strict_oldest["rows"][0]
    )
    relaxed_nearest = copy.deepcopy(document)
    validate_no_widening_documents(document, relaxed_nearest)
    _expect_rejected(
        "all-history widening after skipped nearest gate",
        lambda: validate_no_widening_document_history(
            document, [relaxed_nearest, strict_oldest]
        ),
        "post-hoc absolute widening",
    )

    state_catalog = load_worktree_catalog()
    removed_state_id = (
        "runtime-vnext.metal.gated-delta.v4.state.conv.fp16.none.log-rate-grouped"
    )
    state_catalog["rows"] = [
        row for row in state_catalog["rows"] if row["tolerance_id"] != removed_state_id
    ]
    expected_state_gap = "state.gated_delta.conv_state.log_rate_grouped"
    state_catalog["coverage"]["missing_required_coverage"] = sorted(
        set(state_catalog["coverage"]["missing_required_coverage"])
        | {expected_state_gap}
    )
    state_summary = validate_catalog_document(state_catalog)
    _require(
        expected_state_gap in state_summary["missing_required_coverage"],
        "removing one GDN state semantic fixture must create its own coverage gap",
    )
    _require(
        "state.gated_delta.conv_state.negative_rate_interleaved"
        not in state_summary["missing_required_coverage"],
        "one GDN state semantic fixture must not cover or erase the other",
    )

    removed_row = copy.deepcopy(document)
    removed_row["rows"] = []
    _expect_rejected(
        "history row removal",
        lambda: validate_no_widening_documents(removed_row, document),
        "removed or renamed",
    )

    union_selector = copy.deepcopy(document)
    union_selector["rows"][0]["shape_domain"]["semantics"]["mode"] = "one+two"
    union_selector["rows"][0]["row_fingerprint"] = row_fingerprint(union_selector["rows"][0])
    _expect_rejected(
        "semantic union selector",
        lambda: validate_catalog_document(union_selector),
        "canonical identifier",
    )

    bad_owner = copy.deepcopy(document)
    bad_owner["rows"][0]["owner"] = "x"
    bad_owner["rows"][0]["row_fingerprint"] = row_fingerprint(bad_owner["rows"][0])
    _expect_rejected(
        "unowned row",
        lambda: validate_catalog_document(bad_owner),
        "not an approved catalog owner",
    )

    unknown = copy.deepcopy(document)
    unknown["rows"][0]["artifact_override"] = True
    _expect_rejected(
        "artifact-local override",
        lambda: validate_catalog_document(unknown),
        "unknown fields",
    )

    missing = copy.deepcopy(document)
    del missing["rows"][0]["basis"]
    _expect_rejected(
        "missing basis", lambda: validate_catalog_document(missing), "missing fields"
    )

    non_finite = copy.deepcopy(document)
    non_finite["rows"][0]["bounds"]["cosine_min"] = math.nan
    _expect_rejected(
        "non-finite bound", lambda: validate_catalog_document(non_finite), "must be finite"
    )

    duplicate_key = b'{"schema_version":1,"schema_version":1}'
    _expect_rejected(
        "duplicate JSON key", lambda: load_catalog_bytes(duplicate_key), "duplicate JSON key"
    )
    print(SELF_TEST_PASS)


def _self_test_document() -> dict[str, Any]:
    row: dict[str, Any] = {
        "tolerance_id": "runtime-vnext.metal.causal-attention.v2.operation.fp16.none.fixture",
        "coverage_markers": [
            "operation.causal_paged_attention@2.0.fixed_page_split"
        ],
        "backend": "metal",
        "model_scope": "operation_contract",
        "operation_id": "operation.causal_paged_attention",
        "operation_schema_version": "2.0",
        "checkpoint_kind": "operation_output",
        "checkpoint_name": "output",
        "dtype": "fp16",
        "quant_format": "none",
        "shape_domain": {
            "fixture_id": "causal_attention.fixed_page.split_decode",
            "dimensions": {
                "head_dim": 32,
                "key_value_heads": 1,
                "query_heads": 2,
                "rope_dim": 16,
                "tokens": 2,
            },
            "semantics": {"fixed_page_kv": True, "split_segments": [1, 1]},
        },
        "oracle_identity": "cpu.fp32.rust.causal_attention_reference",
        "oracle_precision": "fp32",
        "bounds": {"cosine_min": 0.999, "relative_l2_max": 0.01, "max_abs_max": 0.006},
        "invariants": {"exact_shape": True, "exact_dtype": True, "max_nan": 0, "max_inf": 0},
        "basis": {
            "kind": "checked_in_conformance_test",
            "source_path": "crates/ferrum-kernels/src/backend/metal/vnext_ops/causal_attention_tests.rs",
            "test_name": "fixed_page_attention_matches_cpu_and_preserves_split_decode_state_on_real_metal",
            "assertion": "Real Metal operation output is checked against the CPU FP32 oracle with max_abs <= 0.006.",
        },
        "source_commit": "ecaeb5087ad45a5148d917fdab63d83cb046d678",
        "owner": "runtime-vnext-g03",
        "review_commit": "f" * 40,
    }
    row["row_fingerprint"] = row_fingerprint(row)
    missing = sorted(
        G08A_REQUIRED_COVERAGE
        - set(row["coverage_markers"])
    )
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "coverage": {
            "profile": G08A_PROFILE,
            "status": "foundation_only",
            "scope": G08A_SCOPE,
            "missing_required_coverage": missing,
        },
        "rows": [row],
    }


def _expect_rejected(label: str, action: Callable[[], Any], marker: str) -> None:
    try:
        action()
    except CatalogError as error:
        if marker not in str(error):
            raise AssertionError(f"{label} failed for the wrong reason: {error}") from error
        return
    raise AssertionError(f"{label} was accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--working-tree",
        action="store_true",
        help="authoring-only validation of the canonical working-tree path; never prints PASS",
    )
    source.add_argument(
        "--git-revision",
        default="HEAD",
        help="revision whose canonical catalog blob is validated (default: HEAD)",
    )
    parser.add_argument(
        "--expected-blob-sha",
        help="required artifact-provided blob OID; must match the canonical path at --git-revision",
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            _self_test()
            return 0
        if args.working_tree:
            _require(args.expected_blob_sha is None,
                     "--expected-blob-sha is forbidden with --working-tree")
            _require(not args.require_complete,
                     "--require-complete is forbidden with --working-tree")
            document = load_worktree_catalog()
            summary = validate_catalog_document(document)
            head = _git(["rev-parse", "HEAD^{commit}"])
            _require(isinstance(head, str), "HEAD did not resolve to text")
            validate_catalog_provenance(document, head)
            compared = validate_no_widening_from_revision(document, head)
            summary["compared_catalog_commit"] = compared[0] if compared else None
            summary["compared_catalog_commits"] = compared
            print(f"{WORKTREE_VALID}: {DEFAULT_CATALOG} {json.dumps(summary, sort_keys=True)}")
            return 0

        document, binding = load_catalog_from_git(args.git_revision, args.expected_blob_sha)
        summary = validate_catalog_document(document, require_complete=args.require_complete)
        validate_catalog_provenance(document, binding["commit"])
        compared = validate_no_widening_from_revision(
            document, _first_parent(binding["commit"])
        )
        summary["compared_catalog_commit"] = compared[0] if compared else None
        summary["compared_catalog_commits"] = compared
        summary["catalog_commit"] = binding["commit"]
        summary["catalog_git_blob_sha"] = binding["git_blob_sha"]
        pass_line = COMPLETE_PASS if summary["coverage_status"] == "complete" else FOUNDATION_PASS
        print(f"{pass_line}: {json.dumps(summary, sort_keys=True)}")
        return 0
    except (CatalogError, AssertionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
