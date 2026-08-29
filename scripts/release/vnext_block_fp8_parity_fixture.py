#!/usr/bin/env python3
"""Validate and materialize the A1 block-FP8 small-tensor quality vector.

The checked-in fixture locks exactly two weight shapes crossed with exactly two
activation batch sizes.  It is intentionally a CPU-only fixture tool: it
generates source E4M3 bytes, BF16 inverse-scale grids, F16 activations, and the
decoded-source matmul reference.  A CUDA runner may consume the materialized
bytes later, but this tool never probes or allocates a GPU and never upgrades
the checked-in ``not_run`` state into M3 evidence.

Generation is single-process and worker-free.  Independent size limits are
checked before any tensor buffer is allocated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = (
    ROOT
    / "scripts/release/configs/vnext_model_adoption"
    / "qwen38_27b_fp8_m3_quality_vector.json"
)

CHECKPOINT_ID = "qwen38-27b-fp8"
CHECKPOINT_REPOSITORY = "Qwen/Qwen3.8-27B-FP8"
CHECKPOINT_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
FIXTURE_ID = "qwen38-27b-fp8-block-fp8-small-tensor-v1"

BLOCK_SHAPE = (128, 128)
WEIGHT_SHAPES = ((256, 128), (256, 256))
ACTIVATION_BATCHES = (1, 4)
EXPECTED_CASE_COUNT = 4

# Source-level containment.  These bounds are independent of model size,
# machine memory, batch size supplied by a caller, or CUDA device properties.
MAX_CASES = 4
MAX_WEIGHT_ELEMENTS = 256 * 256
MAX_ACTIVATION_ELEMENTS = 4 * 256
GENERATOR_WORKERS = 1

MASK64 = (1 << 64) - 1
LCG_MULTIPLIER = 6364136223846793005
LCG_INCREMENT = 1442695040888963407
ROOT_SEED = 0x5147454E33384650
SHAPE_SEED_XOR = 0x9E3779B97F4A7C15
WEIGHT_SEED_XOR = 0x5745494748545F31
SCALE_SEED_XOR = 0x5343414C455F5F31
ACTIVATION_SEED_XOR = 0x4143544956415445

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FixtureError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_json_object,
            parse_constant=reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FixtureError(f"cannot load strict JSON fixture {path}: {exc}") from exc
    require(isinstance(value, dict), "fixture root must be an object")
    return value


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    require(
        actual == expected,
        f"{label} keys differ: missing={sorted(expected - actual)} "
        f"extra={sorted(actual - expected)}",
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def f32(value: float) -> float:
    """Round a Python float to IEEE-754 binary32."""

    return struct.unpack("<f", struct.pack("<f", value))[0]


def decode_e4m3(byte: int) -> float:
    """Decode NVIDIA/safetensors finite E4M3 semantics from one source byte."""

    require(0 <= byte <= 0xFF, "E4M3 value must fit in one byte")
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 0x0F
    mantissa = byte & 0x07
    if exponent == 0:
        return sign * float(mantissa) * (2.0**-9)
    if exponent == 0x0F and mantissa == 0x07:
        return math.nan
    return sign * (1.0 + float(mantissa) / 8.0) * (2.0 ** (exponent - 7))


def decode_bf16_le(payload: bytes) -> list[float]:
    require(len(payload) % 2 == 0, "BF16 payload length must be even")
    return [
        struct.unpack("<f", struct.pack("<I", bits << 16))[0]
        for (bits,) in struct.iter_unpack("<H", payload)
    ]


def decode_f16_le(payload: bytes) -> list[float]:
    require(len(payload) % 2 == 0, "F16 payload length must be even")
    return [value for (value,) in struct.iter_unpack("<e", payload)]


class Lcg64:
    def __init__(self, seed: int) -> None:
        require(0 <= seed <= MASK64, "LCG seed must fit in u64")
        self.state = seed

    def next_u64(self) -> int:
        self.state = (
            self.state * LCG_MULTIPLIER + LCG_INCREMENT
        ) & MASK64
        return self.state


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    weight_shape: tuple[int, int]
    activation_batch: int


@dataclass(frozen=True)
class GeneratedCase:
    spec: CaseSpec
    scale_shape: tuple[int, int]
    values: bytes
    scales: bytes
    activations: bytes
    reference: bytes

    def digest_row(self) -> dict[str, Any]:
        return {
            "case_id": self.spec.case_id,
            "weight_shape": list(self.spec.weight_shape),
            "scale_shape": list(self.scale_shape),
            "activation_batch": self.spec.activation_batch,
            "values_f8e4m3_sha256": sha256_bytes(self.values),
            "scales_bf16le_sha256": sha256_bytes(self.scales),
            "activations_f16le_sha256": sha256_bytes(self.activations),
            "reference_f32le_sha256": sha256_bytes(self.reference),
        }


def derive_shape_stream_seed(shape_index: int, stream_xor: int) -> int:
    require(0 <= shape_index < len(WEIGHT_SHAPES), "shape index is out of range")
    mixed = ROOT_SEED ^ (((shape_index + 1) * SHAPE_SEED_XOR) & MASK64) ^ stream_xor
    return Lcg64(mixed).next_u64()


def generate_values(shape_index: int, element_count: int) -> bytes:
    require(
        0 < element_count <= MAX_WEIGHT_ELEMENTS,
        "weight allocation exceeds the independent small-tensor bound",
    )
    rng = Lcg64(derive_shape_stream_seed(shape_index, WEIGHT_SEED_XOR))
    values = bytearray(element_count)
    for index in range(element_count):
        word = rng.next_u64()
        if (word & 0x0F) == 0:
            values[index] = 0
            continue
        magnitude = 0x20 + ((word >> 9) & 0x1F)
        sign = (word >> 31) & 0x80
        values[index] = magnitude | sign
    return bytes(values)


def generate_scales(shape_index: int, element_count: int) -> bytes:
    require(0 < element_count <= 4, "scale grid exceeds the fixed fixture bound")
    rng = Lcg64(derive_shape_stream_seed(shape_index, SCALE_SEED_XOR))
    payload = bytearray()
    for _ in range(element_count):
        # Exact positive BF16 dyadics in [2^-8, 2^-7].
        bf16_bits = 0x3B80 + 0x20 * (rng.next_u64() % 5)
        payload.extend(struct.pack("<H", bf16_bits))
    return bytes(payload)


def generate_activations(shape_index: int, batch: int, input_features: int) -> bytes:
    element_count = batch * input_features
    require(
        0 < element_count <= MAX_ACTIVATION_ELEMENTS,
        "activation allocation exceeds the independent small-tensor bound",
    )
    rng = Lcg64(derive_shape_stream_seed(shape_index, ACTIVATION_SEED_XOR))
    payload = bytearray()
    for _ in range(element_count):
        # signed/64 is exactly representable in F16, so generation does not
        # depend on a host-language floating-point rounding convention.
        signed = int(rng.next_u64() % 129) - 64
        payload.extend(struct.pack("<e", signed / 64.0))
    return bytes(payload)


def source_decoded_reference(
    values: bytes,
    scales_payload: bytes,
    activations_payload: bytes,
    output_features: int,
    input_features: int,
    batch: int,
) -> bytes:
    require(len(values) == output_features * input_features, "FP8 value length mismatch")
    scale_rows = (output_features + BLOCK_SHAPE[0] - 1) // BLOCK_SHAPE[0]
    scale_columns = (input_features + BLOCK_SHAPE[1] - 1) // BLOCK_SHAPE[1]
    scales = decode_bf16_le(scales_payload)
    require(len(scales) == scale_rows * scale_columns, "inverse-scale grid length mismatch")
    activations = decode_f16_le(activations_payload)
    require(len(activations) == batch * input_features, "activation length mismatch")

    decoded_values = [decode_e4m3(byte) for byte in values]
    require(all(math.isfinite(value) for value in decoded_values), "generated E4M3 contains NaN")
    require(
        all(math.isfinite(value) and value > 0.0 for value in scales),
        "invalid BF16 inverse scale",
    )

    output = bytearray()
    for batch_index in range(batch):
        activation_offset = batch_index * input_features
        for output_index in range(output_features):
            weight_offset = output_index * input_features
            scale_row = output_index // BLOCK_SHAPE[0]
            accumulator = 0.0
            for input_index in range(input_features):
                scale_column = input_index // BLOCK_SHAPE[1]
                inverse_scale = scales[scale_row * scale_columns + scale_column]
                decoded_weight = f32(
                    decoded_values[weight_offset + input_index] * inverse_scale
                )
                product = f32(
                    activations[activation_offset + input_index] * decoded_weight
                )
                accumulator = f32(accumulator + product)
            output.extend(struct.pack("<f", accumulator))
    return bytes(output)


def expected_case_specs() -> list[CaseSpec]:
    cases = [
        CaseSpec(
            case_id=f"weight-{output_features}x{input_features}-batch-{batch}",
            weight_shape=(output_features, input_features),
            activation_batch=batch,
        )
        for output_features, input_features in WEIGHT_SHAPES
        for batch in ACTIVATION_BATCHES
    ]
    require(len(cases) == EXPECTED_CASE_COUNT <= MAX_CASES, "fixture case bound changed")
    return cases


def generate_case(spec: CaseSpec) -> GeneratedCase:
    shape_index = WEIGHT_SHAPES.index(spec.weight_shape)
    output_features, input_features = spec.weight_shape
    weight_elements = output_features * input_features
    scale_shape = (
        (output_features + BLOCK_SHAPE[0] - 1) // BLOCK_SHAPE[0],
        (input_features + BLOCK_SHAPE[1] - 1) // BLOCK_SHAPE[1],
    )
    values = generate_values(shape_index, weight_elements)
    scales = generate_scales(shape_index, scale_shape[0] * scale_shape[1])
    activations = generate_activations(
        shape_index, spec.activation_batch, input_features
    )
    reference = source_decoded_reference(
        values,
        scales,
        activations,
        output_features,
        input_features,
        spec.activation_batch,
    )
    return GeneratedCase(spec, scale_shape, values, scales, activations, reference)


def generate_all_cases() -> list[GeneratedCase]:
    # Deliberately sequential: this small fixture never derives worker count
    # from tensor size, batch, memory, or caller input.
    require(GENERATOR_WORKERS == 1, "fixture generator must remain single-worker")
    return [generate_case(spec) for spec in expected_case_specs()]


def quality_digest_payload(
    fixture: dict[str, Any], generated_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": fixture["schema_version"],
        "fixture_id": fixture["fixture_id"],
        "checkpoint": fixture["checkpoint"],
        "generator": fixture["generator"],
        "source_contract": fixture["source_contract"],
        "activation_contract": fixture["activation_contract"],
        "reference_contract": fixture["reference_contract"],
        "weight_shapes": fixture["weight_shapes"],
        "activation_batches": fixture["activation_batches"],
        "cases": generated_rows,
    }


def validate_fixed_contract(fixture: dict[str, Any]) -> None:
    exact_keys(
        fixture,
        {
            "schema_version",
            "fixture_id",
            "checkpoint",
            "generator",
            "source_contract",
            "activation_contract",
            "reference_contract",
            "weight_shapes",
            "activation_batches",
            "cases",
            "quality_vector_digest",
            "cuda_execution",
        },
        "fixture",
    )
    require(fixture["schema_version"] == 1, "fixture schema_version must be 1")
    require(fixture["fixture_id"] == FIXTURE_ID, "fixture_id drifted")

    checkpoint = fixture["checkpoint"]
    require(isinstance(checkpoint, dict), "checkpoint must be an object")
    exact_keys(checkpoint, {"id", "repository", "revision"}, "checkpoint")
    require(checkpoint["id"] == CHECKPOINT_ID, "checkpoint id drifted")
    require(checkpoint["repository"] == CHECKPOINT_REPOSITORY, "checkpoint repository drifted")
    require(checkpoint["revision"] == CHECKPOINT_REVISION, "checkpoint revision drifted")

    generator = fixture["generator"]
    require(isinstance(generator, dict), "generator must be an object")
    exact_keys(
        generator,
        {
            "algorithm",
            "root_seed_hex",
            "multiplier_hex",
            "increment_hex",
            "stream_partitioning",
            "worker_count",
        },
        "generator",
    )
    require(generator["algorithm"] == "lcg64-v1", "generator algorithm drifted")
    require(generator["root_seed_hex"] == f"0x{ROOT_SEED:016x}", "root seed drifted")
    require(
        generator["multiplier_hex"] == f"0x{LCG_MULTIPLIER:016x}",
        "LCG multiplier drifted",
    )
    require(
        generator["increment_hex"] == f"0x{LCG_INCREMENT:016x}",
        "LCG increment drifted",
    )
    require(
        generator["stream_partitioning"]
        == "shape-index xor fixed stream-domain, then one lcg64 step",
        "generator stream partitioning drifted",
    )
    require(
        generator["worker_count"] == GENERATOR_WORKERS,
        "worker count must remain independently bounded at one",
    )

    source = fixture["source_contract"]
    require(isinstance(source, dict), "source_contract must be an object")
    exact_keys(
        source,
        {
            "values_dtype",
            "values_layout",
            "scales_dtype",
            "scales_layout",
            "scale_kind",
            "block_shape",
            "decode_formula",
        },
        "source_contract",
    )
    require(source["values_dtype"] == "F8_E4M3", "source values dtype drifted")
    require(source["values_layout"] == "row-major [N,K]", "source values layout drifted")
    require(source["scales_dtype"] == "BF16", "source scales dtype drifted")
    require(
        source["scales_layout"] == "row-major [ceil(N/128),ceil(K/128)]",
        "source scale layout drifted",
    )
    require(source["scale_kind"] == "inverse_scale", "source scale kind drifted")
    require(source["block_shape"] == list(BLOCK_SHAPE), "source block shape drifted")
    require(
        source["decode_formula"]
        == "W[n,k]=decode_finite_e4m3(values[n,k])*decode_bf16(scale_inv[n//128,k//128])",
        "source decode formula drifted",
    )

    activation = fixture["activation_contract"]
    require(isinstance(activation, dict), "activation_contract must be an object")
    exact_keys(activation, {"dtype", "layout", "generator_values"}, "activation_contract")
    require(activation["dtype"] == "F16", "activation dtype drifted")
    require(activation["layout"] == "row-major [B,K]", "activation layout drifted")
    require(
        activation["generator_values"]
        == "signed integer in [-64,64] divided by 64",
        "activation generator drifted",
    )

    reference = fixture["reference_contract"]
    require(isinstance(reference, dict), "reference_contract must be an object")
    exact_keys(
        reference,
        {
            "semantics",
            "accumulator_dtype",
            "output_dtype",
            "relative_l2_formula",
            "relative_l2_max",
            "nan_count_max",
            "inf_count_max",
        },
        "reference_contract",
    )
    require(
        reference["semantics"]
        == (
            "activation matmul transpose(source values decoded with locked BF16 "
            "inverse scales and 128x128 layout)"
        ),
        "reference semantics drifted",
    )
    require(reference["accumulator_dtype"] == "F32 step-rounded", "reference accumulator drifted")
    require(reference["output_dtype"] == "F32 little-endian", "reference output dtype drifted")
    require(
        reference["relative_l2_formula"]
        == "norm(actual-reference)_2/max(norm(reference)_2,1e-6)",
        "relative L2 formula drifted",
    )
    require(reference["relative_l2_max"] == 0.05, "relative L2 threshold drifted")
    require(reference["nan_count_max"] == 0, "NaN threshold drifted")
    require(reference["inf_count_max"] == 0, "Inf threshold drifted")

    require(
        fixture["weight_shapes"] == [list(shape) for shape in WEIGHT_SHAPES],
        "weight shapes drifted",
    )
    require(fixture["activation_batches"] == list(ACTIVATION_BATCHES), "activation batches drifted")

    cuda = fixture["cuda_execution"]
    require(isinstance(cuda, dict), "cuda_execution must be an object")
    exact_keys(
        cuda,
        {"status", "required_case_count", "completed_case_count", "reason", "artifact"},
        "cuda_execution",
    )
    require(cuda["status"] == "not_run", "checked-in fixture must not claim CUDA execution")
    require(cuda["required_case_count"] == EXPECTED_CASE_COUNT, "CUDA case denominator drifted")
    require(
        cuda["completed_case_count"] == 0,
        "not_run fixture cannot contain completed CUDA cases",
    )
    require(isinstance(cuda["reason"], str) and cuda["reason"], "CUDA not_run reason is required")
    require(cuda["artifact"] is None, "not_run fixture cannot reference a CUDA artifact")


def validate_fixture(fixture: dict[str, Any]) -> tuple[list[GeneratedCase], str]:
    validate_fixed_contract(fixture)
    generated = generate_all_cases()
    generated_rows = [case.digest_row() for case in generated]
    checked_rows = fixture["cases"]
    require(isinstance(checked_rows, list), "cases must be an array")
    require(len(checked_rows) == EXPECTED_CASE_COUNT, "fixture requires exactly four cases")
    require(checked_rows == generated_rows, "generated case metadata or payload digest drifted")

    cells = {
        (tuple(row["weight_shape"]), row["activation_batch"])
        for row in checked_rows
    }
    expected_cells = {
        (shape, batch) for shape in WEIGHT_SHAPES for batch in ACTIVATION_BATCHES
    }
    require(
        cells == expected_cells,
        "cases are not the exact 2 weight shapes x 2 batches matrix",
    )
    require(
        len({row["case_id"] for row in checked_rows}) == EXPECTED_CASE_COUNT,
        "case ids must be unique",
    )

    quality_digest = sha256_bytes(
        canonical_json_bytes(quality_digest_payload(fixture, generated_rows))
    )
    checked_digest = fixture["quality_vector_digest"]
    require(
        isinstance(checked_digest, str) and SHA256_RE.fullmatch(checked_digest) is not None,
        "quality_vector_digest must be lowercase SHA256",
    )
    require(checked_digest == quality_digest, "quality_vector_digest drifted")
    return generated, quality_digest


def materialize_case(case: GeneratedCase, out_dir: Path, quality_digest: str) -> None:
    if out_dir.exists():
        require(out_dir.is_dir(), f"materialization output is not a directory: {out_dir}")
        require(not any(out_dir.iterdir()), f"materialization output must be empty: {out_dir}")
    else:
        out_dir.mkdir(parents=True)

    files = {
        "values.f8e4m3.bin": case.values,
        "scales.bf16le.bin": case.scales,
        "activations.f16le.bin": case.activations,
        "reference.f32le.bin": case.reference,
    }
    for name, payload in files.items():
        (out_dir / name).write_bytes(payload)
    receipt = {
        "schema_version": 1,
        "fixture_id": FIXTURE_ID,
        "quality_vector_digest": quality_digest,
        "case": case.digest_row(),
        "cuda_execution": {
            "status": "not_run",
            "relative_l2": None,
            "nan_count": None,
            "inf_count": None,
        },
        "files": [
            {"path": name, "size_bytes": len(payload), "sha256": sha256_bytes(payload)}
            for name, payload in files.items()
        ],
        "terminal_line": f"VNEXT BLOCK FP8 PARITY CASE NOT_RUN: {case.spec.case_id}",
    }
    (out_dir / "case.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(receipt["terminal_line"])


def measure_actual_output(
    case: GeneratedCase, actual_payload: bytes, quality_digest: str
) -> dict[str, Any]:
    require(
        len(actual_payload) == len(case.reference),
        f"CUDA output byte length {len(actual_payload)} differs from "
        f"reference {len(case.reference)}",
    )
    actual = [value for (value,) in struct.iter_unpack("<f", actual_payload)]
    reference = [value for (value,) in struct.iter_unpack("<f", case.reference)]
    nan_count = sum(math.isnan(value) for value in actual)
    inf_count = sum(math.isinf(value) for value in actual)

    relative_l2: float | None = None
    if nan_count == 0 and inf_count == 0:
        error_squared = sum(
            (observed - expected) ** 2
            for observed, expected in zip(actual, reference, strict=True)
        )
        reference_squared = sum(expected**2 for expected in reference)
        relative_l2 = math.sqrt(error_squared) / max(
            math.sqrt(reference_squared), 1.0e-6
        )
        require(math.isfinite(relative_l2), "relative L2 calculation was non-finite")

    passed = (
        relative_l2 is not None
        and relative_l2 <= 0.05
        and nan_count == 0
        and inf_count == 0
    )
    status = "pass" if passed else "reject"
    terminal_line = (
        f"VNEXT BLOCK FP8 PARITY CASE {status.upper()}: {case.spec.case_id}"
    )
    return {
        "schema_version": 1,
        "fixture_id": FIXTURE_ID,
        "quality_vector_digest": quality_digest,
        "case_id": case.spec.case_id,
        "weight_shape": list(case.spec.weight_shape),
        "activation_batch": case.spec.activation_batch,
        "reference_semantics": (
            "source E4M3 values multiplied by BF16 inverse scales on the "
            "128x128 grid, then activation matmul"
        ),
        "relative_l2": relative_l2,
        "relative_l2_max": 0.05,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "status": status,
        "terminal_line": terminal_line,
    }


def self_test(fixture_path: Path) -> None:
    fixture = load_json(fixture_path)
    generated, quality_digest = validate_fixture(fixture)
    require(len(generated) == EXPECTED_CASE_COUNT, "self-test case count mismatch")

    exact = measure_actual_output(generated[0], generated[0].reference, quality_digest)
    require(exact["status"] == "pass", "exact reference output must pass parity")
    require(exact["relative_l2"] == 0.0, "exact reference relative L2 must be zero")
    zero_output = bytes(len(generated[0].reference))
    drift = measure_actual_output(generated[0], zero_output, quality_digest)
    require(drift["status"] == "reject", "zero CUDA output must fail parity")
    require(drift["relative_l2"] == 1.0, "zero output relative L2 must be one")
    nan_output = bytearray(generated[0].reference)
    nan_output[:4] = struct.pack("<f", math.nan)
    non_finite = measure_actual_output(generated[0], bytes(nan_output), quality_digest)
    require(non_finite["status"] == "reject", "NaN CUDA output must fail parity")
    require(non_finite["nan_count"] == 1, "NaN count must be exact")
    require(non_finite["relative_l2"] is None, "non-finite output has no numeric relative L2")

    anchors = {
        0x00: 0.0,
        0x01: 2.0**-9,
        0x38: 1.0,
        0x3F: 1.875,
        0x7E: 448.0,
    }
    for byte, expected in anchors.items():
        require(decode_e4m3(byte) == expected, f"E4M3 anchor 0x{byte:02x} drifted")
    require(math.isnan(decode_e4m3(0x7F)), "E4M3 NaN anchor drifted")

    mutations: list[tuple[str, dict[str, Any]]] = []
    missing_case = copy.deepcopy(fixture)
    missing_case["cases"].pop()
    mutations.append(("incomplete matrix", missing_case))
    bad_block = copy.deepcopy(fixture)
    bad_block["source_contract"]["block_shape"] = [128, 64]
    mutations.append(("block shape drift", bad_block))
    bad_digest = copy.deepcopy(fixture)
    bad_digest["cases"][0]["values_f8e4m3_sha256"] = "0" * 64
    mutations.append(("payload digest drift", bad_digest))
    false_cuda_claim = copy.deepcopy(fixture)
    false_cuda_claim["cuda_execution"]["status"] = "pass"
    mutations.append(("false CUDA claim", false_cuda_claim))

    for label, mutation in mutations:
        try:
            validate_fixture(mutation)
        except FixtureError:
            continue
        raise FixtureError(f"self-test mutation was accepted: {label}")
    print("VNEXT BLOCK FP8 PARITY FIXTURE SELF-TEST PASS")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--print-derived",
        action="store_true",
        help="print generated case digest rows and the quality-vector digest",
    )
    parser.add_argument("--materialize-case", metavar="CASE_ID")
    parser.add_argument(
        "--measure-actual",
        metavar="CASE_ID",
        help="measure a CUDA runner's raw little-endian F32 output for one case",
    )
    parser.add_argument("--actual-output", type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    fixture_path = args.fixture.resolve()
    if args.self_test:
        self_test(fixture_path)
        return 0

    fixture = load_json(fixture_path)
    generated, quality_digest = validate_fixture(fixture)
    if args.print_derived:
        print(
            json.dumps(
                {
                    "cases": [case.digest_row() for case in generated],
                    "quality_vector_digest": quality_digest,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    if args.materialize_case is not None:
        require(args.measure_actual is None, "choose only one case action")
        require(args.out is not None, "--materialize-case requires --out")
        require(args.actual_output is None, "--actual-output is invalid with --materialize-case")
        matching = [case for case in generated if case.spec.case_id == args.materialize_case]
        require(len(matching) == 1, f"unknown fixture case: {args.materialize_case}")
        materialize_case(matching[0], args.out.resolve(), quality_digest)
        return 0
    if args.measure_actual is not None:
        require(args.actual_output is not None, "--measure-actual requires --actual-output")
        require(args.out is None, "--out is invalid with --measure-actual")
        matching = [case for case in generated if case.spec.case_id == args.measure_actual]
        require(len(matching) == 1, f"unknown fixture case: {args.measure_actual}")
        try:
            actual_payload = args.actual_output.read_bytes()
        except OSError as exc:
            raise FixtureError(f"cannot read CUDA output {args.actual_output}: {exc}") from exc
        receipt = measure_actual_output(matching[0], actual_payload, quality_digest)
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
        print(receipt["terminal_line"])
        return 0 if receipt["status"] == "pass" else 1
    require(args.actual_output is None, "--actual-output requires --measure-actual")
    require(args.out is None, "--out is only valid with --materialize-case")
    print(f"VNEXT BLOCK FP8 PARITY FIXTURE PASS: {fixture_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except FixtureError as exc:
        print(f"VNEXT BLOCK FP8 PARITY FIXTURE REJECT: {exc}", file=sys.stderr)
        raise SystemExit(1)
