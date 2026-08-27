#!/usr/bin/env python3
"""Read-only final gate for the fixed Qwen3.8 CUDA adoption goal."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


PASS_PREFIX = "QWEN38 CUDA ADOPTION GOAL PASS"
REPOSITORY = "cyankiwi/Qwen3.8-27B-AWQ-INT4"
REVISION = "63768c10df38c0395e12ef49edac1bd539eaeeea"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_ARTIFACTS = {
    "model_lock": "source/model-lock.json",
    "contract_tests": "source/contract-tests.json",
    "unit_gate": "source/unit/gate.manifest.json",
    "panda_host": "panda-pad/host.json",
    "panda_build": "panda-pad/build/receipt.json",
    "panda_smoke": "panda-pad/smoke/validation.json",
    "qwen38_host": "qwen38-4090/host.json",
    "qwen38_build": "qwen38-4090/build/receipt.json",
    "qwen38_correctness": "qwen38-4090/correctness/validation.json",
    "qwen38_usability": "qwen38-4090/usability/validation.json",
    "qwen38_bench_report": "qwen38-4090/usability/bench-report.json",
}
EXPECTED_SHARDS = {
    "model-00001-of-00005.safetensors": (2542796928, "54d83c1d36631de231876217a8e0c2483eccee8746369a482b79442bdfc5d958"),
    "model-00002-of-00005.safetensors": (4967650936, "64be5fc2f66a3e5679ba229261a7a0d8112b06f6f560c750a62ca9457f90006c"),
    "model-00003-of-00005.safetensors": (4996718528, "7b90d6c7059d615a560cd4d2e766d328210605041061681550d80f380a8b529b"),
    "model-00004-of-00005.safetensors": (4976405864, "03b2624ec788780a2915003cd2871c29c87dfb6f2a8d189ef3918662d6a1ed56"),
    "model-00005-of-00005.safetensors": (3534428672, "eb5ea1fbef28b13ac89158924ee7cfe7c9f111c79ae177b290c0abd45c38925c"),
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValidationError(f"missing artifact: {path}") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON artifact {path}: {error}") from error
    require(isinstance(value, dict), f"artifact must be a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_pass(value: Any) -> bool:
    return isinstance(value, str) and value.lower() == "pass"


def require_candidate(data: dict[str, Any], git_sha: str, label: str) -> None:
    require(data.get("git_sha") == git_sha, f"{label} git_sha differs from candidate")
    require(data.get("dirty_status") == [], f"{label} must record a clean worktree")


def require_checkpoint(data: dict[str, Any], label: str) -> None:
    checkpoint = data.get("checkpoint")
    require(isinstance(checkpoint, dict), f"{label} checkpoint is missing")
    require(checkpoint.get("repository") == REPOSITORY, f"{label} repository differs")
    require(checkpoint.get("revision") == REVISION, f"{label} revision differs")


def validate_model_lock(lock: dict[str, Any]) -> None:
    require(lock.get("decision") == "KEEP", "model lock decision must be KEEP")
    require_checkpoint(lock, "model lock")
    scope = lock.get("scope", {})
    require(scope.get("backend") == "cuda", "model lock backend must be cuda")
    require(scope.get("device") == "exactly-one-rtx-4090", "model lock device differs")
    require(scope.get("modality") == "text-only", "model lock modality must be text-only")
    shards = lock.get("shards")
    require(isinstance(shards, list) and len(shards) == 5, "model lock must contain five shards")
    observed = {
        row.get("name"): (row.get("size_bytes"), row.get("sha256"))
        for row in shards
        if isinstance(row, dict)
    }
    require(observed == EXPECTED_SHARDS, "model shard size/SHA lock differs")
    architecture = lock.get("architecture", {})
    require(architecture.get("text_model_type") == "qwen3_5_text", "text architecture differs")
    require(architecture.get("num_hidden_layers") == 64, "text layer count differs")
    quant = lock.get("quantization_contract", {})
    weights = quant.get("weights", {})
    require(quant.get("quant_method") == "compressed-tensors", "quant method differs")
    require(quant.get("format") == "pack-quantized", "quant format differs")
    require(weights.get("num_bits") == 4, "weight bits must be 4")
    require(weights.get("group_size") == 32, "weight group size must be 32")
    require(weights.get("symmetric") is False, "weights must be asymmetric")
    require(quant.get("input_activations") is None, "activation quantization is not allowed")
    require(quant.get("mixed_dense_and_quantized") is True, "mixed dense/quantized lock is missing")


def validate_contract(contract: dict[str, Any], git_sha: str) -> None:
    require(is_pass(contract.get("status")), "source contract tests did not pass")
    require_candidate(contract, git_sha, "source contract tests")
    require_checkpoint(contract, "source contract tests")
    fmt = contract.get("format_contract", {})
    require(fmt.get("recognized") == fmt.get("expected") == 1, "format recognition is incomplete")
    require(fmt.get("bits") == 4 and fmt.get("group_size") == 32, "format bits/group differ")
    require(fmt.get("symmetric") is False, "format must be asymmetric")
    require(fmt.get("activation_quantization") is False, "activation quantization must be absent")
    rejected = contract.get("typed_rejections", {})
    invalid = rejected.get("invalid_config_cases", {})
    require(invalid.get("passed") == invalid.get("expected") == 3, "typed config rejects are incomplete")
    require(rejected.get("missing_sidecar", {}).get("passed") == 1, "missing sidecar reject failed")
    parity = contract.get("cuda_parity", {})
    require(parity.get("passed") == parity.get("expected") == 4, "CUDA parity fixtures are incomplete")
    require(float(parity.get("max_relative_error", 1.0)) < 0.05, "CUDA parity exceeds 5%")
    require(parity.get("non_finite_count") == 0, "CUDA parity contains NaN/Inf")
    template = contract.get("chat_template", {})
    require(template.get("golden_cases_passed") == template.get("golden_cases_expected") == 4, "template goldens are incomplete")


def validate_unit(unit: dict[str, Any], git_sha: str) -> None:
    require(is_pass(unit.get("status")), "unit gate did not pass")
    require(unit.get("lane") == "unit", "unit gate lane differs")
    require(unit.get("git_sha") == git_sha, "unit gate git_sha differs")
    dirty = unit.get("dirty_status", {})
    require(dirty.get("is_dirty") is False and dirty.get("status_short") == [], "unit gate worktree is dirty")
    artifact_dir = unit.get("artifact_dir")
    require(unit.get("pass_line") == f"FERRUM GATE unit PASS: {artifact_dir}", "unit gate PASS line differs")
    require(unit.get("child_pass_line") == f"G0 SOURCE unit PASS: {artifact_dir}", "unit child PASS line differs")


def validate_panda(host: dict[str, Any], build: dict[str, Any], smoke: dict[str, Any], git_sha: str) -> None:
    require_candidate(host, git_sha, "panda host")
    require("RTX 4050" in str(host.get("gpu_raw", "")), "panda host is not the RTX 4050 lane")
    require("release 12.6" in str(host.get("nvcc_raw", "")), "panda CUDA toolkit differs")
    require(is_pass(build.get("status")), "panda CUDA build did not pass")
    require_candidate(build, git_sha, "panda build")
    require(build.get("exit_code") == 0, "panda CUDA build exit code differs")
    require(build.get("build_configuration", {}).get("native_operator_count") == 4, "panda native operator set is incomplete")
    binary_sha = build.get("binary", {}).get("sha256")
    require(isinstance(binary_sha, str) and SHA256_RE.fullmatch(binary_sha), "panda binary SHA is missing")
    require(is_pass(smoke.get("status")), "panda product smoke did not pass")
    require(smoke.get("git_sha") == git_sha, "panda smoke git_sha differs")
    require(smoke.get("binary_sha256") == binary_sha, "panda smoke used another binary")
    require(smoke.get("error_scan_count") == 0, "panda smoke error scan is non-zero")
    q35 = smoke.get("qwen35", {})
    require(q35.get("model") == "Qwen/Qwen3.5-0.8B", "panda same-architecture model differs")
    require(q35.get("revision") == "2fc06364715b967f1860aea9cf38778875588b17", "panda same-architecture revision differs")
    require(q35.get("run_single_nonempty") == 1 and q35.get("run_two_turn_nonempty") == 2, "panda run coverage is incomplete")
    require(q35.get("serve_nonstream", {}).get("http_status") == 200, "panda non-stream serve failed")
    stream = q35.get("serve_stream", {})
    require(stream.get("http_status") == 200 and stream.get("done_count") == 1 and stream.get("completion_tokens", 0) > 0, "panda stream serve failed")
    sentinel = smoke.get("qwen3_sentinel", {})
    require(sentinel.get("model") == "qwen3:0.6b", "panda sentinel differs")
    require(sentinel.get("run_single_nonempty") == 1, "panda sentinel run failed")
    require(sentinel.get("serve_nonstream", {}).get("http_status") == 200, "panda sentinel serve failed")


def validate_4090(host: dict[str, Any], build: dict[str, Any], correctness: dict[str, Any], git_sha: str) -> str:
    require(is_pass(host.get("status")), "4090 host receipt did not pass")
    require_candidate(host, git_sha, "4090 host")
    require(host.get("gpu_count") == 1 and "RTX 4090" in str(host.get("gpu_name", "")), "G3 requires exactly one RTX 4090")
    require(str(host.get("compute_capability")) in {"8.9", "89", "sm_89"}, "4090 compute capability differs")
    require(is_pass(build.get("status")), "4090 CUDA build did not pass")
    require_candidate(build, git_sha, "4090 build")
    require(build.get("exit_code") == 0, "4090 CUDA build exit code differs")
    binary_sha = build.get("binary", {}).get("sha256")
    require(isinstance(binary_sha, str) and SHA256_RE.fullmatch(binary_sha), "4090 binary SHA is missing")
    require(is_pass(correctness.get("status")), "4090 correctness did not pass")
    require_candidate(correctness, git_sha, "4090 correctness")
    require_checkpoint(correctness, "4090 correctness")
    require(correctness.get("binary_sha256") == binary_sha, "4090 correctness used another binary")
    require(correctness.get("error_scan_count") == 0, "4090 correctness error scan is non-zero")
    run = correctness.get("run", {})
    known = run.get("known_answer", {})
    require(known.get("passed") is True and "paris" in str(known.get("content", "")).lower(), "known-answer run failed")
    two_turn = run.get("two_turn", {})
    require(two_turn.get("passed") is True and two_turn.get("assistant_turns", 0) >= 2 and two_turn.get("recalled_marker") is True, "two-turn run failed")
    serve = correctness.get("serve", {})
    nonstream = serve.get("nonstream", {})
    require(nonstream.get("passed") is True and nonstream.get("http_status") == 200 and nonstream.get("completion_tokens", 0) > 0, "non-stream serve failed")
    stream = serve.get("stream", {})
    require(stream.get("passed") is True and stream.get("http_status") == 200 and stream.get("done_count") == 1 and stream.get("completion_tokens", 0) > 0, "stream serve failed")
    tool = serve.get("required_tool", {})
    require(tool.get("passed") is True and tool.get("http_status") == 200 and tool.get("tool_call_count", 0) > 0, "required-tool serve failed")
    schema = serve.get("strict_schema", {})
    require(schema.get("passed") is True and schema.get("http_status") == 200 and schema.get("schema_valid") is True, "strict-schema serve failed")
    return binary_sha


def validate_usability(validation: dict[str, Any], report: dict[str, Any], git_sha: str, binary_sha: str, report_path: Path) -> None:
    require(is_pass(validation.get("status")), "4090 usability validation did not pass")
    require_candidate(validation, git_sha, "4090 usability")
    require_checkpoint(validation, "4090 usability")
    require(validation.get("binary_sha256") == binary_sha, "4090 usability used another binary")
    require(validation.get("bench_report_sha256") == sha256(report_path), "bench report SHA differs")
    require(report.get("backend") == "cuda" and report.get("concurrency") == 1, "usability lane must be CUDA c=1")
    require(report.get("n_repeats") == 1 and report.get("n_requests_per_run") == 3, "usability lane must contain one three-request run")
    require(report.get("warmup_requests") == 0, "usability lane must not add hidden warmups")
    repeats = report.get("repeat_metrics")
    require(isinstance(repeats, list) and len(repeats) == 1, "usability report repeat evidence differs")
    row = repeats[0]
    require(row.get("expected_requests") == row.get("completed_requests") == 3, "usability completed request count differs")
    require(row.get("errored_requests") == 0, "usability has failed requests")
    require(row.get("output_token_count_source") == "usage", "usability must use server usage token counts")
    require(float(row.get("output_throughput_tps", 0.0)) >= 5.0, "usability output throughput is below 5 tok/s")
    require(float(row.get("ttft_ms", {}).get("p50", 30001.0)) <= 30000.0, "usability p50 TTFT exceeds 30 seconds")
    quality = row.get("quality_issues", {})
    require(isinstance(quality, dict) and all(value == 0 for value in quality.values()), "usability quality issue count is non-zero")


def validate(root: Path) -> None:
    manifest = load_json(root / "goal.manifest.json")
    require(manifest.get("schema_version") == 1, "goal manifest schema_version must be 1")
    require(manifest.get("goal") == "qwen38-cuda-adoption", "goal manifest id differs")
    require(is_pass(manifest.get("status")), "goal manifest status is not pass")
    git_sha = manifest.get("git_sha")
    require(isinstance(git_sha, str) and GIT_SHA_RE.fullmatch(git_sha), "candidate git SHA is invalid")
    require(manifest.get("dirty_status") == [], "goal manifest must record a clean worktree")
    require_checkpoint(manifest, "goal manifest")
    scope = manifest.get("scope", {})
    require(scope == {"backend": "cuda", "device": "exactly-one-rtx-4090", "modality": "text-only"}, "goal scope differs")
    entries = manifest.get("artifacts")
    require(isinstance(entries, dict) and set(entries) == set(REQUIRED_ARTIFACTS), "goal artifact index differs")
    loaded: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for key, relative in REQUIRED_ARTIFACTS.items():
        entry = entries.get(key)
        require(isinstance(entry, dict) and entry.get("path") == relative, f"artifact path differs: {key}")
        digest = entry.get("sha256")
        require(isinstance(digest, str) and SHA256_RE.fullmatch(digest), f"artifact SHA is invalid: {key}")
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"artifact file is missing or symlinked: {relative}")
        require(sha256(path) == digest, f"artifact SHA mismatch: {relative}")
        loaded[key] = load_json(path)
        paths[key] = path
    validate_model_lock(loaded["model_lock"])
    validate_contract(loaded["contract_tests"], git_sha)
    validate_unit(loaded["unit_gate"], git_sha)
    validate_panda(loaded["panda_host"], loaded["panda_build"], loaded["panda_smoke"], git_sha)
    binary_sha = validate_4090(loaded["qwen38_host"], loaded["qwen38_build"], loaded["qwen38_correctness"], git_sha)
    validate_usability(loaded["qwen38_usability"], loaded["qwen38_bench_report"], git_sha, binary_sha, paths["qwen38_bench_report"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    root = args.out_dir.expanduser().resolve()
    try:
        require(root.is_dir(), f"artifact directory is missing: {root}")
        validate(root)
    except (OSError, ValueError, TypeError, ValidationError) as error:
        print(f"QWEN38 CUDA ADOPTION GOAL REJECT: {error}", file=sys.stderr)
        return 1
    print(f"{PASS_PREFIX}: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
