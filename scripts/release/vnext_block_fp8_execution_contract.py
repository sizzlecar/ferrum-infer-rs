#!/usr/bin/env python3
"""Validate the A1 Qwen3.8 block-FP8 execution/coverage contract lock.

This is a static, diagnostic lock. It intentionally records missing source to
execution materialization, provider coverage, and compiler quality-approval
authority. It does not execute CUDA, approve an approximate transformation, or
produce an M0/model-adoption PASS receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    ROOT
    / "scripts/release/configs/vnext_model_adoption"
    / "qwen38_27b_fp8_execution_contract.json"
)

CHECKPOINT = {
    "id": "qwen38-27b-fp8",
    "repository": "Qwen/Qwen3.8-27B-FP8",
    "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
}
SOURCE_FORMAT_ID = "quantization.safetensors.fp8-e4m3-block-grid-inverse-scale"
QUALITY_VECTOR_DIGEST = "4c8b44a6a6e2ca803f6a3916b033a50a8a007cb2452a0e9246ed6c7f3cacbb51"
MATERIALIZER_ID = "weight-materializer.cuda.block-fp8-to-marlin-fp8-w8a16"
MATERIALIZER_VERSION = {"major": 1, "minor": 0}
EXECUTION_WEIGHT_FORMAT_ID = "weight-format.execution.cuda.marlin-fp8-w8a16-mixed"
EXECUTION_WEIGHT_LAYOUT_ID = "weight-layout.execution.cuda.marlin-fp8-w8a16-mixed"
EXECUTION_QUANTIZATION_FORMAT_ID = "quantization.marlin.fp8-e4m3fn-channelwise"
MARLIN_CAPABILITY_ID = "capability.kernel.cuda.marlin.fp8-w8a16"

EXPECTED_EXECUTION_CONTRACT_FINGERPRINT = (
    "882bc49ca312875a12a5290319f6c8294386a5960c2065cbda3f3dff2d55598e"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ROOT_KEYS = {
    "schema_version",
    "contract_id",
    "status",
    "checkpoint",
    "source_contract",
    "operation_coverage",
    "planned_materializer",
    "quality_approval",
    "execution_contract_fingerprint",
}
CHECKPOINT_KEYS = {"id", "repository", "revision"}
SOURCE_KEYS = {
    "format_id",
    "values_dtype",
    "scales_dtype",
    "scale_kind",
    "block_shape",
    "execution_pair_count",
}
COVERAGE_KEYS = {
    "operation_id",
    "operation_version",
    "source_fp8_pair_count",
    "count_derivation",
    "existing_execution_provider_acceptance",
    "current_static_coverage",
    "missing_boundaries",
}
VERSION_KEYS = {"major", "minor"}
MATERIALIZER_KEYS = {
    "status",
    "id",
    "version",
    "implementation_fingerprint",
    "fidelity",
    "fidelity_enum",
    "source_format_id",
    "execution_weight_format_id",
    "execution_weight_layout_id",
    "execution_quantization_format_id",
    "required_capability_id",
    "conversion_semantics",
    "transient_policy",
}
QUALITY_KEYS = {
    "status",
    "witness_schema_id",
    "quality_vector_digest",
    "current_witness",
    "compiler_selection_authorized",
    "required_witness_fields",
    "selection_requirements",
}
WITNESS_FIELDS_KEYS = {
    "approval_authority",
    "candidate",
    "checkpoint",
    "materializer",
    "source",
    "execution_schema",
    "contract",
    "numeric_evidence",
    "decision",
}
SELECTION_KEYS = {
    "all_bound_identities_must_match",
    "implementation_fingerprints_required",
    "source_schema_fingerprint_required",
    "execution_schema_fingerprint_required",
    "execution_contract_fingerprint_required",
    "checked_in_quality_vector_digest_required",
    "required_case_count",
    "completed_case_count",
    "relative_l2_max",
    "nan_count_max",
    "inf_count_max",
    "artifact_sha256_required",
    "global_boolean_or_environment_override_forbidden",
}

EXPECTED_COVERAGE = [
    {
        "operation_id": "operation.gated_delta_recurrent_attention",
        "operation_version": {"major": 6, "minor": 0},
        "source_fp8_pair_count": 144,
        "count_derivation": "48 linear-attention layers x 3 FP8 projection pairs",
        "existing_execution_provider_acceptance": True,
        "current_static_coverage": False,
        "missing_boundaries": [
            "source-block-fp8-to-existing-marlin-fp8-materializer",
            "typed-compiler-quality-approval-authority",
        ],
    },
    {
        "operation_id": "operation.causal_paged_attention",
        "operation_version": {"major": 2, "minor": 0},
        "source_fp8_pair_count": 64,
        "count_derivation": "16 full-attention layers x 4 FP8 projection pairs",
        "existing_execution_provider_acceptance": False,
        "current_static_coverage": False,
        "missing_boundaries": [
            "causal-paged-attention-provider-existing-marlin-fp8-acceptance",
            "source-block-fp8-to-existing-marlin-fp8-materializer",
            "typed-compiler-quality-approval-authority",
        ],
    },
    {
        "operation_id": "operation.dense_swiglu",
        "operation_version": {"major": 1, "minor": 0},
        "source_fp8_pair_count": 192,
        "count_derivation": "64 decoder layers x 3 FP8 projection pairs",
        "existing_execution_provider_acceptance": False,
        "current_static_coverage": False,
        "missing_boundaries": [
            "dense-swiglu-provider-existing-marlin-fp8-acceptance",
            "source-block-fp8-to-existing-marlin-fp8-materializer",
            "typed-compiler-quality-approval-authority",
        ],
    },
    {
        "operation_id": "operation.last_token_dense_linear",
        "operation_version": {"major": 1, "minor": 0},
        "source_fp8_pair_count": 0,
        "count_derivation": (
            "lm_head is an official dense exclusion; no source FP8 pair enters logits"
        ),
        "existing_execution_provider_acceptance": True,
        "current_static_coverage": True,
        "missing_boundaries": [],
    },
]

EXPECTED_WITNESS_FIELDS = {
    "approval_authority": ["id", "version", "implementation_fingerprint"],
    "candidate": ["git_sha", "dirty", "binary_sha256"],
    "checkpoint": ["id", "revision"],
    "materializer": ["id", "version", "implementation_fingerprint", "fidelity"],
    "source": ["format_id", "source_schema_fingerprint"],
    "execution_schema": [
        "weight_format_id",
        "weight_layout_id",
        "quantization_format_id",
        "fingerprint",
    ],
    "contract": ["execution_contract_fingerprint", "quality_vector_digest"],
    "numeric_evidence": [
        "artifact_sha256",
        "required_case_count",
        "completed_case_count",
        "relative_l2_max_observed",
        "nan_count",
        "inf_count",
    ],
    "decision": ["approved", "approved_at_utc"],
}


class ContractError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_json_object,
            parse_constant=reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ContractError(f"cannot load strict JSON contract {path}: {error}") from error
    require(isinstance(value, dict), "execution contract root must be an object")
    return value


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    require(
        actual == expected,
        f"{label} keys differ: missing={sorted(expected - actual)} "
        f"extra={sorted(actual - expected)}",
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


def contract_fingerprint(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "execution_contract_fingerprint"}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def require_int(value: Any, expected: int, label: str) -> None:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value == expected,
        f"{label} must be {expected}",
    )


def validate_version(value: Any, expected: dict[str, int], label: str) -> None:
    version = exact_keys(value, VERSION_KEYS, label)
    require_int(version["major"], expected["major"], f"{label}.major")
    require_int(version["minor"], expected["minor"], f"{label}.minor")


def validate_coverage(value: Any) -> None:
    require(isinstance(value, list), "operation_coverage must be a list")
    require(len(value) == 4, "operation_coverage must contain exactly four rows")
    for index, (actual, expected) in enumerate(zip(value, EXPECTED_COVERAGE, strict=True)):
        row = exact_keys(actual, COVERAGE_KEYS, f"operation_coverage[{index}]")
        require(row == expected, f"operation_coverage[{index}] drifted")

    pair_total = sum(row["source_fp8_pair_count"] for row in value)
    require(pair_total == 400, "operation coverage must account for exactly 400 FP8 pairs")
    coverage = [row["current_static_coverage"] for row in value]
    require(
        coverage == [False, False, False, True],
        "current static coverage must remain false/false/false/true",
    )
    for index, row in enumerate(value):
        missing = row["missing_boundaries"]
        require(isinstance(missing, list), f"operation_coverage[{index}].missing_boundaries must be a list")
        require(len(missing) == len(set(missing)), f"operation_coverage[{index}] has duplicate missing boundaries")
        if row["current_static_coverage"]:
            require(not missing, f"operation_coverage[{index}] cannot be covered with missing boundaries")
        else:
            require(bool(missing), f"operation_coverage[{index}] must name its missing boundaries")


def validate_materializer(value: Any) -> None:
    materializer = exact_keys(value, MATERIALIZER_KEYS, "planned_materializer")
    require(materializer["status"] == "not_implemented", "planned materializer must remain not_implemented")
    require(materializer["id"] == MATERIALIZER_ID, "planned materializer id drifted")
    validate_version(materializer["version"], MATERIALIZER_VERSION, "planned_materializer.version")
    require(materializer["implementation_fingerprint"] is None, "unimplemented materializer cannot claim an implementation fingerprint")
    require(materializer["fidelity"] == "approximate", "planned materializer fidelity must be approximate")
    require(
        materializer["fidelity_enum"] == "WeightMaterializationFidelity::Approximate",
        "planned materializer fidelity enum drifted",
    )
    require(materializer["source_format_id"] == SOURCE_FORMAT_ID, "planned materializer source format drifted")
    require(
        materializer["execution_weight_format_id"] == EXECUTION_WEIGHT_FORMAT_ID,
        "planned execution weight format drifted",
    )
    require(
        materializer["execution_weight_layout_id"] == EXECUTION_WEIGHT_LAYOUT_ID,
        "planned execution weight layout drifted",
    )
    require(
        materializer["execution_quantization_format_id"]
        == EXECUTION_QUANTIZATION_FORMAT_ID,
        "planned execution quantization format drifted",
    )
    require(
        materializer["required_capability_id"] == MARLIN_CAPABILITY_ID,
        "planned Marlin capability drifted",
    )
    require(
        materializer["conversion_semantics"]
        == "cold-path decode of source E4M3 values with BF16 128x128 inverse scales followed by channelwise E4M3FN quantization and existing tiled Marlin packing",
        "planned conversion semantics drifted",
    )
    require(
        materializer["transient_policy"]
        == "one bounded source component group at a time; no persistent dense checkpoint copy",
        "planned transient policy drifted",
    )


def validate_quality_approval(value: Any) -> None:
    quality = exact_keys(value, QUALITY_KEYS, "quality_approval")
    require(quality["status"] == "authority_not_implemented", "quality authority must remain not implemented")
    require(
        quality["witness_schema_id"] == "quality-approval.weight-materializer.numeric.v1",
        "quality witness schema id drifted",
    )
    require(quality["quality_vector_digest"] == QUALITY_VECTOR_DIGEST, "quality vector digest drifted")
    require(quality["current_witness"] is None, "diagnostic contract cannot contain an approval witness")
    require(
        quality["compiler_selection_authorized"] is False,
        "diagnostic contract cannot authorize compiler selection",
    )
    witness_fields = exact_keys(
        quality["required_witness_fields"],
        WITNESS_FIELDS_KEYS,
        "quality_approval.required_witness_fields",
    )
    require(witness_fields == EXPECTED_WITNESS_FIELDS, "typed quality witness fields drifted")
    selection = exact_keys(
        quality["selection_requirements"],
        SELECTION_KEYS,
        "quality_approval.selection_requirements",
    )
    expected_selection = {
        "all_bound_identities_must_match": True,
        "implementation_fingerprints_required": True,
        "source_schema_fingerprint_required": True,
        "execution_schema_fingerprint_required": True,
        "execution_contract_fingerprint_required": True,
        "checked_in_quality_vector_digest_required": True,
        "required_case_count": 4,
        "completed_case_count": 4,
        "relative_l2_max": 0.05,
        "nan_count_max": 0,
        "inf_count_max": 0,
        "artifact_sha256_required": True,
        "global_boolean_or_environment_override_forbidden": True,
    }
    require(selection == expected_selection, "quality selection requirements drifted")
    require(
        isinstance(selection["relative_l2_max"], (int, float))
        and not isinstance(selection["relative_l2_max"], bool)
        and math.isfinite(selection["relative_l2_max"]),
        "relative L2 threshold must be finite",
    )


def validate_contract(document: dict[str, Any]) -> dict[str, Any]:
    root = exact_keys(document, ROOT_KEYS, "execution contract")
    require_int(root["schema_version"], 1, "schema_version")
    require(
        root["contract_id"] == "qwen38-27b-fp8-a1-execution-coverage-v1",
        "contract id drifted",
    )
    require(
        root["status"] == "diagnostic_not_implemented",
        "execution contract must remain diagnostic_not_implemented",
    )

    checkpoint = exact_keys(root["checkpoint"], CHECKPOINT_KEYS, "checkpoint")
    require(checkpoint == CHECKPOINT, "checkpoint identity drifted")

    source = exact_keys(root["source_contract"], SOURCE_KEYS, "source_contract")
    require(source["format_id"] == SOURCE_FORMAT_ID, "checked source format id drifted")
    require(source["values_dtype"] == "F8_E4M3", "source values dtype drifted")
    require(source["scales_dtype"] == "BF16", "source scales dtype drifted")
    require(source["scale_kind"] == "inverse_scale", "source scale kind drifted")
    require(source["block_shape"] == [128, 128], "source block shape drifted")
    require_int(source["execution_pair_count"], 400, "source execution pair count")

    validate_coverage(root["operation_coverage"])
    validate_materializer(root["planned_materializer"])
    validate_quality_approval(root["quality_approval"])

    fingerprint = root["execution_contract_fingerprint"]
    require(
        isinstance(fingerprint, str) and SHA256_RE.fullmatch(fingerprint) is not None,
        "execution contract fingerprint must be lowercase SHA256",
    )
    computed = contract_fingerprint(root)
    require(fingerprint == computed, "execution contract fingerprint does not match canonical contract")
    require(
        fingerprint == EXPECTED_EXECUTION_CONTRACT_FINGERPRINT,
        "execution contract fingerprint drifted from the checked lock",
    )
    return {
        "checkpoint_id": checkpoint["id"],
        "source_fp8_pair_count": source["execution_pair_count"],
        "static_coverage": [row["current_static_coverage"] for row in root["operation_coverage"]],
        "materializer_status": root["planned_materializer"]["status"],
        "quality_approval_status": root["quality_approval"]["status"],
        "compiler_selection_authorized": False,
        "execution_contract_fingerprint": fingerprint,
    }


def re_fingerprint(document: dict[str, Any]) -> None:
    document["execution_contract_fingerprint"] = contract_fingerprint(document)


def expect_rejected(label: str, mutate: Callable[[dict[str, Any]], None], base: dict[str, Any]) -> None:
    candidate = copy.deepcopy(base)
    mutate(candidate)
    try:
        validate_contract(candidate)
    except ContractError:
        return
    raise ContractError(f"self-test mutation was accepted: {label}")


def run_self_test(document: dict[str, Any]) -> None:
    validate_contract(document)

    def drift_count(value: dict[str, Any]) -> None:
        value["operation_coverage"][0]["source_fp8_pair_count"] = 145
        re_fingerprint(value)

    def forge_coverage(value: dict[str, Any]) -> None:
        value["operation_coverage"][0]["current_static_coverage"] = True
        value["operation_coverage"][0]["missing_boundaries"] = []
        re_fingerprint(value)

    def forge_approval(value: dict[str, Any]) -> None:
        value["quality_approval"]["status"] = "approved"
        value["quality_approval"]["compiler_selection_authorized"] = True
        value["quality_approval"]["current_witness"] = {"approved": True}
        re_fingerprint(value)

    def drift_materializer(value: dict[str, Any]) -> None:
        value["planned_materializer"]["id"] = "weight-materializer.cuda.forged"
        re_fingerprint(value)

    def drift_quality_vector(value: dict[str, Any]) -> None:
        value["quality_approval"]["quality_vector_digest"] = "f" * 64
        re_fingerprint(value)

    def stale_fingerprint(value: dict[str, Any]) -> None:
        value["source_contract"]["execution_pair_count"] = 399

    expect_rejected("operation pair count drift", drift_count, document)
    expect_rejected("false static coverage", forge_coverage, document)
    expect_rejected("forged quality approval", forge_approval, document)
    expect_rejected("planned materializer identity drift", drift_materializer, document)
    expect_rejected("quality vector drift", drift_quality_vector, document)
    expect_rejected("stale canonical fingerprint", stale_fingerprint, document)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--print-computed-fingerprint",
        action="store_true",
        help="print the canonical fingerprint before checked-lock comparison",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        document = load_json(args.contract)
        if args.print_computed_fingerprint:
            print(contract_fingerprint(document))
            return 0
        summary = validate_contract(document)
        if args.self_test:
            run_self_test(document)
            print("VNEXT BLOCK FP8 EXECUTION CONTRACT SELF-TEST PASS")
        else:
            print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
            print(f"VNEXT BLOCK FP8 EXECUTION CONTRACT PASS: {args.contract}")
        return 0
    except ContractError as error:
        print(f"VNEXT BLOCK FP8 EXECUTION CONTRACT REJECT: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
