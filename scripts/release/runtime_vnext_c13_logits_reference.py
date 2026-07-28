#!/usr/bin/env python3
"""Compare a Ferrum C13 checkpoint with a provenance-bound vLLM raw-logits reference."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
REFERENCE_COLLECTOR_ID = "ferrum.runtime-vnext.c13-vllm-raw-logits.v1"
COMPARATOR_ID = "ferrum.runtime-vnext.c13-logits-comparison.v1"
KEEP_LINE = "FERRUM C13 LOGITS REFERENCE KEEP"
REJECT_LINE = "FERRUM C13 LOGITS REFERENCE REJECT"
INCONCLUSIVE_LINE = "FERRUM C13 LOGITS REFERENCE INCONCLUSIVE"
SELF_TEST_LINE = "FERRUM C13 LOGITS REFERENCE SELF-TEST PASS"
THRESHOLDS = {
    "keep_centered_cosine_min": 0.99,
    "keep_affine_relative_l2_max": 0.15,
    "keep_top20_overlap_min": 0.80,
    "keep_nucleus_jaccard_min": 0.50,
    "reject_centered_cosine_below": 0.90,
    "reject_affine_relative_l2_above": 0.35,
    "reject_top20_overlap_below": 0.50,
}


class ComparisonError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_json_bytes(value))


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonError(f"cannot read JSON object {path}: {error}") from error
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def full_sha(path: Path, label: str, length: int) -> str:
    try:
        value = path.read_text().strip().split()[0]
    except (OSError, IndexError) as error:
        raise ComparisonError(f"cannot read {label} from {path}: {error}") from error
    require(
        len(value) == length and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a {length}-character lowercase hex value",
    )
    return value


def read_values(path: Path, element_type: str, count: int) -> list[float]:
    widths = {"f16": 2, "f32le": 4}
    require(element_type in widths, f"unsupported element type {element_type!r}")
    payload = path.read_bytes()
    require(
        len(payload) == count * widths[element_type],
        f"{path} byte length does not match {count} {element_type} values",
    )
    if element_type == "f16":
        return [value[0] for value in struct.iter_unpack("<e", payload)]
    values = array.array("f")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def finite_vector(values: list[float], label: str) -> None:
    require(bool(values), f"{label} is empty")
    require(all(math.isfinite(value) for value in values), f"{label} contains NaN or infinity")


def ranked_token_ids(values: list[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda token_id: (-values[token_id], token_id))[:count]


def argmax(values: list[float]) -> int:
    return ranked_token_ids(values, 1)[0]


def dot(left: list[float], right: list[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


def norm(values: list[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def cosine(left: list[float], right: list[float]) -> float:
    denominator = norm(left) * norm(right)
    require(denominator > 0.0, "cosine requires non-zero vectors")
    return dot(left, right) / denominator


def centered(values: list[float]) -> list[float]:
    mean = math.fsum(values) / len(values)
    return [value - mean for value in values]


def distribution_metrics(ferrum: list[float], reference: list[float]) -> dict[str, float]:
    differences = [left - right for left, right in zip(ferrum, reference)]
    reference_norm = norm(reference)
    ferrum_centered = centered(ferrum)
    reference_centered = centered(reference)
    centered_reference_norm = norm(reference_centered)
    centered_ferrum_norm = norm(ferrum_centered)
    require(reference_norm > 0.0, "reference logits norm is zero")
    require(
        centered_reference_norm > 0.0 and centered_ferrum_norm > 0.0,
        "centered logits norm is zero",
    )

    variance = dot(reference_centered, reference_centered)
    scale = dot(reference_centered, ferrum_centered) / variance
    reference_mean = math.fsum(reference) / len(reference)
    ferrum_mean = math.fsum(ferrum) / len(ferrum)
    offset = ferrum_mean - scale * reference_mean
    residual = [
        observed - (scale * expected + offset)
        for observed, expected in zip(ferrum, reference)
    ]

    return {
        "cosine": cosine(ferrum, reference),
        "centered_cosine": cosine(ferrum_centered, reference_centered),
        "relative_l2": norm(differences) / reference_norm,
        "centered_relative_l2": norm(
            [left - right for left, right in zip(ferrum_centered, reference_centered)]
        )
        / centered_reference_norm,
        "affine_scale": scale,
        "affine_offset": offset,
        "affine_relative_l2": norm(residual) / centered_ferrum_norm,
        "max_abs": max(abs(value) for value in differences),
        "mean_abs": math.fsum(abs(value) for value in differences) / len(differences),
        "ferrum_mean": ferrum_mean,
        "reference_mean": reference_mean,
    }


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = math.fsum(weights)
    require(total > 0.0 and math.isfinite(total), "softmax normalization is invalid")
    return [weight / total for weight in weights]


def jensen_shannon(ferrum: list[float], reference: list[float]) -> float:
    left = softmax(ferrum)
    right = softmax(reference)
    total = 0.0
    for left_probability, right_probability in zip(left, right):
        midpoint = (left_probability + right_probability) / 2.0
        if left_probability > 0.0:
            total += 0.5 * left_probability * math.log(left_probability / midpoint)
        if right_probability > 0.0:
            total += 0.5 * right_probability * math.log(right_probability / midpoint)
    return total


def nucleus(values: list[float], sampling: dict[str, Any]) -> list[dict[str, Any]]:
    temperature = float(sampling["temperature"])
    top_k = int(sampling["top_k"])
    top_p = float(sampling["top_p"])
    require(temperature > 0.0, "C13 reference requires non-zero temperature")
    require(top_k > 0, "C13 reference requires a bounded top-k")
    require(0.0 < top_p <= 1.0, "C13 reference top-p is invalid")
    ranked = ranked_token_ids(values, min(top_k, len(values)))
    scaled = [values[token_id] / temperature for token_id in ranked]
    probabilities = softmax(scaled)
    retained: list[dict[str, Any]] = []
    cumulative = 0.0
    for rank, (token_id, probability) in enumerate(
        zip(ranked, probabilities), start=1
    ):
        cumulative += probability
        retained.append(
            {
                "token_id": token_id,
                "rank": rank,
                "probability_within_top_k": probability,
                "cumulative_probability": cumulative,
            }
        )
        if cumulative >= top_p:
            break
    return retained


def top_summary(values: list[float], count: int = 20) -> list[dict[str, Any]]:
    return [
        {"token_id": token_id, "rank": rank, "logit": values[token_id]}
        for rank, token_id in enumerate(ranked_token_ids(values, count), start=1)
    ]


def set_overlap(left: list[int], right: list[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    require(bool(left_set) and bool(right_set), "set overlap requires non-empty sets")
    return len(left_set & right_set) / min(len(left_set), len(right_set))


def jaccard(left: list[int], right: list[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    require(bool(union), "Jaccard requires a non-empty union")
    return len(left_set & right_set) / len(union)


def select_ferrum_output(wave: dict[str, Any]) -> dict[str, Any]:
    outputs = wave.get("product_outputs")
    require(isinstance(outputs, list), "Ferrum wave.product_outputs must be a list")
    matches = [
        output
        for output in outputs
        if isinstance(output, dict)
        and output.get("output_mode") == "full-logits"
        and output.get("participant_index") == 0
    ]
    require(len(matches) == 1, "Ferrum wave must contain one participant-0 full-logits output")
    return matches[0]


def classify(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    strong = (
        metrics["argmax_equal"]
        and metrics["top20_overlap"] >= THRESHOLDS["keep_top20_overlap_min"]
        and metrics["nucleus_jaccard"] >= THRESHOLDS["keep_nucleus_jaccard_min"]
        and metrics["distribution"]["centered_cosine"]
        >= THRESHOLDS["keep_centered_cosine_min"]
        and metrics["distribution"]["affine_relative_l2"]
        <= THRESHOLDS["keep_affine_relative_l2_max"]
    )
    if strong:
        reasons.append("raw logits retain the same dominant token and local C13 candidate set")
        reasons.append("global centered/affine residual metrics meet reference thresholds")
        return "KEEP_REFERENCE_ALIGNMENT", reasons

    divergent = (
        metrics["top20_overlap"] < THRESHOLDS["reject_top20_overlap_below"]
        or metrics["distribution"]["centered_cosine"]
        < THRESHOLDS["reject_centered_cosine_below"]
        or metrics["distribution"]["affine_relative_l2"]
        > THRESHOLDS["reject_affine_relative_l2_above"]
    )
    if divergent:
        if metrics["top20_overlap"] < THRESHOLDS["reject_top20_overlap_below"]:
            reasons.append("top-20 candidate overlap is below the divergence threshold")
        if (
            metrics["distribution"]["centered_cosine"]
            < THRESHOLDS["reject_centered_cosine_below"]
        ):
            reasons.append("centered cosine is below the divergence threshold")
        if (
            metrics["distribution"]["affine_relative_l2"]
            > THRESHOLDS["reject_affine_relative_l2_above"]
        ):
            reasons.append("affine residual is above the divergence threshold")
        return "REJECT_MODEL_EXECUTION_DIVERGENCE", reasons

    reasons.append("reference metrics fall between alignment and divergence thresholds")
    if not metrics["argmax_equal"]:
        reasons.append("argmax differs, but the wider distribution is not decisively divergent")
    return "INCONCLUSIVE_NUMERICAL_DRIFT", reasons


def compare(args: argparse.Namespace) -> tuple[Path, str]:
    wave_path = args.ferrum_wave.resolve()
    reference_path = args.vllm_reference.resolve()
    build_dir = args.ferrum_build_dir.resolve()
    out_dir = args.out.resolve()

    wave = load_object(wave_path)
    reference = load_object(reference_path)
    require(
        reference.get("schema_version") == SCHEMA_VERSION,
        "vLLM reference schema version mismatch",
    )
    require(
        reference.get("collector_id") == REFERENCE_COLLECTOR_ID,
        "vLLM reference collector identity mismatch",
    )
    require(reference.get("status") == "pass", "vLLM reference status is not pass")
    reference_vllm = reference.get("vllm")
    require(isinstance(reference_vllm, dict), "reference.vllm must be an object")
    require(
        reference_vllm.get("dirty") is False
        and reference_vllm.get("git_status") == "",
        "vLLM reference source must be clean",
    )
    ferrum_output = select_ferrum_output(wave)

    model = reference.get("model")
    prompt = reference.get("prompt")
    reference_output = reference.get("output")
    sampling = reference.get("sampling")
    require(isinstance(model, dict), "reference.model must be an object")
    require(isinstance(prompt, dict), "reference.prompt must be an object")
    require(isinstance(reference_output, dict), "reference.output must be an object")
    require(isinstance(sampling, dict), "reference.sampling must be an object")
    require(
        wave.get("model_id") == model.get("model_id"),
        "Ferrum/vLLM model IDs differ",
    )

    token_span = ferrum_output.get("token_span")
    require(isinstance(token_span, dict), "Ferrum output token_span must be an object")
    require(
        token_span.get("fingerprint") == prompt.get("token_span_fingerprint"),
        "Ferrum/vLLM token-span fingerprints differ",
    )
    require(
        token_span.get("full_input_tokens") == prompt.get("token_count"),
        "Ferrum/vLLM prompt token counts differ",
    )

    ferrum_layout = ferrum_output.get("output_layout")
    reference_layout = reference_output.get("raw_logits")
    require(isinstance(ferrum_layout, dict), "Ferrum output_layout must be an object")
    require(isinstance(reference_layout, dict), "reference raw_logits must be an object")
    count = int(ferrum_layout.get("element_count", -1))
    require(count > 0, "Ferrum logits count must be positive")
    require(
        count == reference_layout.get("element_count"),
        "Ferrum/vLLM vocabulary sizes differ",
    )
    require(ferrum_layout.get("element_type") == "f16", "Ferrum logits must be f16")
    require(reference_layout.get("element_type") == "f32le", "vLLM logits must be f32le")

    ferrum_raw = wave_path.parent / str(ferrum_output.get("raw_file"))
    reference_raw = reference_path.parent / str(reference_layout.get("file"))
    require(ferrum_raw.is_file(), f"Ferrum raw logits are missing: {ferrum_raw}")
    require(reference_raw.is_file(), f"vLLM raw logits are missing: {reference_raw}")
    require(
        sha256_file(ferrum_raw) == ferrum_output.get("raw_sha256"),
        "Ferrum raw-logits SHA256 mismatch",
    )
    require(
        sha256_file(reference_raw) == reference_layout.get("sha256"),
        "vLLM raw-logits SHA256 mismatch",
    )

    ferrum = read_values(ferrum_raw, "f16", count)
    vllm = read_values(reference_raw, "f32le", count)
    finite_vector(ferrum, "Ferrum logits")
    finite_vector(vllm, "vLLM logits")

    ferrum_top20_ids = ranked_token_ids(ferrum, 20)
    vllm_top20_ids = ranked_token_ids(vllm, 20)
    ferrum_nucleus = nucleus(ferrum, sampling)
    vllm_nucleus = nucleus(vllm, sampling)
    ferrum_nucleus_ids = [entry["token_id"] for entry in ferrum_nucleus]
    vllm_nucleus_ids = [entry["token_id"] for entry in vllm_nucleus]
    metrics = {
        "element_count": count,
        "ferrum_argmax": argmax(ferrum),
        "vllm_argmax": argmax(vllm),
        "argmax_equal": argmax(ferrum) == argmax(vllm),
        "top20_overlap": set_overlap(ferrum_top20_ids, vllm_top20_ids),
        "nucleus_jaccard": jaccard(ferrum_nucleus_ids, vllm_nucleus_ids),
        "jensen_shannon": jensen_shannon(ferrum, vllm),
        "distribution": distribution_metrics(ferrum, vllm),
    }
    decision, reasons = classify(metrics)

    ferrum_git_sha = full_sha(build_dir / "git.sha", "Ferrum git SHA", 40)
    ferrum_binary_sha256 = full_sha(
        build_dir / "binary.sha256", "Ferrum binary SHA256", 64
    )
    git_status_path = build_dir / "git.status"
    require(git_status_path.is_file(), "Ferrum build git.status is missing")
    ferrum_git_status = git_status_path.read_text().strip()
    require(not ferrum_git_status, "Ferrum checkpoint source must be clean")

    result = {
        "schema_version": SCHEMA_VERSION,
        "comparator_id": COMPARATOR_ID,
        "decision": decision,
        "reasons": reasons,
        "thresholds": THRESHOLDS,
        "identity": {
            "model_id": wave["model_id"],
            "model_revision": model.get("model_revision"),
            "tokenizer_revision": model.get("tokenizer_revision"),
            "token_count": prompt["token_count"],
            "token_span_fingerprint": prompt["token_span_fingerprint"],
        },
        "ferrum": {
            "git_sha": ferrum_git_sha,
            "git_status": ferrum_git_status,
            "dirty": bool(ferrum_git_status),
            "binary_sha256": ferrum_binary_sha256,
            "wave": str(wave_path),
            "raw_logits": str(ferrum_raw),
            "raw_logits_sha256": ferrum_output["raw_sha256"],
            "top20": top_summary(ferrum),
            "c13_nucleus": ferrum_nucleus,
        },
        "vllm": {
            **reference_vllm,
            "reference": str(reference_path),
            "raw_logits": str(reference_raw),
            "raw_logits_sha256": reference_layout["sha256"],
            "top20": top_summary(vllm),
            "c13_nucleus": vllm_nucleus,
        },
        "sampling": sampling,
        "metrics": metrics,
    }
    result_path = out_dir / "comparison.json"
    atomic_write_json(result_path, result)
    return result_path, decision


def synthetic_fixture(root: Path, reference_values: list[float], ferrum_values: list[float]) -> tuple[Path, Path, Path]:
    model_id = "synthetic/c13"
    fingerprint = "a" * 64
    checkpoint = root / "checkpoint"
    build = root / "build"
    reference_dir = root / "reference"
    checkpoint.mkdir(parents=True)
    build.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    ferrum_path = checkpoint / "logits.bin"
    atomic_write(ferrum_path, b"".join(struct.pack("<e", value) for value in ferrum_values))
    ferrum_sha = sha256_file(ferrum_path)
    wave = {
        "model_id": model_id,
        "product_outputs": [
            {
                "output_mode": "full-logits",
                "participant_index": 0,
                "token_span": {
                    "fingerprint": fingerprint,
                    "full_input_tokens": 3,
                },
                "output_layout": {
                    "element_type": "f16",
                    "element_count": len(ferrum_values),
                },
                "raw_file": ferrum_path.name,
                "raw_sha256": ferrum_sha,
            }
        ],
    }
    wave_path = checkpoint / "wave.json"
    atomic_write_json(wave_path, wave)
    raw_reference = array.array("f", reference_values)
    if sys.byteorder != "little":
        raw_reference.byteswap()
    reference_raw_path = reference_dir / "raw-logits.f32le"
    atomic_write(reference_raw_path, raw_reference.tobytes())
    reference = {
        "schema_version": SCHEMA_VERSION,
        "collector_id": REFERENCE_COLLECTOR_ID,
        "status": "pass",
        "model": {
            "model_id": model_id,
            "model_revision": "1" * 40,
            "tokenizer_revision": "2" * 40,
        },
        "prompt": {
            "token_count": 3,
            "token_span_fingerprint": fingerprint,
        },
        "output": {
            "raw_logits": {
                "file": reference_raw_path.name,
                "element_type": "f32le",
                "element_count": len(reference_values),
                "sha256": sha256_file(reference_raw_path),
            }
        },
        "sampling": {
            "temperature": 1.0,
            "top_k": min(4, len(reference_values)),
            "top_p": 0.95,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "seed": 9271,
        },
        "vllm": {
            "git_sha": "3" * 40,
            "git_status": "",
            "dirty": False,
        },
    }
    reference_path = reference_dir / "reference.json"
    atomic_write_json(reference_path, reference)
    atomic_write(build / "git.sha", ("4" * 40 + "\n").encode())
    atomic_write(build / "git.status", b"")
    atomic_write(build / "binary.sha256", ("5" * 64 + "\n").encode())
    return wave_path, reference_path, build


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        reference_values = [7.0, 5.0, 3.0, 1.0, -1.0, -3.0]
        aligned_values = [7.01, 4.99, 3.01, 1.0, -1.0, -3.01]
        wave, reference, build = synthetic_fixture(
            root / "aligned", reference_values, aligned_values
        )
        args = argparse.Namespace(
            ferrum_wave=wave,
            vllm_reference=reference,
            ferrum_build_dir=build,
            out=root / "aligned-result",
        )
        _, decision = compare(args)
        require(decision == "KEEP_REFERENCE_ALIGNMENT", "aligned fixture was not kept")

        divergent_values = [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]
        wave, reference, build = synthetic_fixture(
            root / "divergent", reference_values, divergent_values
        )
        args = argparse.Namespace(
            ferrum_wave=wave,
            vllm_reference=reference,
            ferrum_build_dir=build,
            out=root / "divergent-result",
        )
        _, decision = compare(args)
        require(
            decision == "REJECT_MODEL_EXECUTION_DIVERGENCE",
            "divergent fixture was not rejected",
        )

        reference_document = load_object(reference)
        reference_document["prompt"]["token_span_fingerprint"] = "b" * 64
        atomic_write_json(reference, reference_document)
        try:
            compare(args)
        except ComparisonError as error:
            require("fingerprints differ" in str(error), "wrong provenance rejection")
        else:
            raise ComparisonError("mismatched prompt fingerprint was accepted")
    print(f"{SELF_TEST_LINE}: synthetic")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--self-test", action="store_true")
    result.add_argument("--ferrum-wave", type=Path)
    result.add_argument("--ferrum-build-dir", type=Path)
    result.add_argument("--vllm-reference", type=Path)
    result.add_argument("--out", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        missing = [
            name
            for name in ("ferrum_wave", "ferrum_build_dir", "vllm_reference", "out")
            if getattr(args, name) is None
        ]
        require(not missing, "missing required arguments: " + ", ".join(missing))
        result_path, decision = compare(args)
        if decision == "KEEP_REFERENCE_ALIGNMENT":
            print(f"{KEEP_LINE}: {result_path.parent}")
            return 0
        if decision == "REJECT_MODEL_EXECUTION_DIVERGENCE":
            print(f"{REJECT_LINE}: {result_path.parent}")
            return 2
        print(f"{INCONCLUSIVE_LINE}: {result_path.parent}")
        return 3
    except (ComparisonError, OSError, ValueError) as error:
        print(f"{REJECT_LINE}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
