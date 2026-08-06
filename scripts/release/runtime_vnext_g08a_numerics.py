#!/usr/bin/env python3
"""Aggregate G08A Metal op/layer/model/logit/token numerical evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import runtime_vnext_g08a_metal_op_numerics as op_numerics
import runtime_vnext_numerical_tolerances as tolerances
import runtime_vnext_qwen35_full_attention_gate as full_attention_gate
import runtime_vnext_qwen35_layer_reference_gate as linear_attention_gate
import runtime_vnext_qwen35_model_reference_gate as model_reference_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
MODEL_LOCK = CONFIG_DIR / "runtime_vnext_g08a_m1_metal.models.lock.json"
PROMPT_CORPUS = CONFIG_DIR / "runtime_vnext_g08a_m1_token_parity_prompts.json"
PASS_PREFIX = "FERRUM RUNTIME VNEXT G08A NUMERICS PASS"
FAIL_PREFIX = "FERRUM RUNTIME VNEXT G08A NUMERICS FAIL"
SELFTEST_PASS = "FERRUM RUNTIME VNEXT G08A NUMERICS SELFTEST PASS"
MODEL_KEY = "m1-qwen35-4b"
BACKEND = "metal"
TOKEN_COUNT = 64
PROMPT_COUNT = 20
NEAR_TIE_MARGIN = 1.0e-3
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
LINEAR_TOLERANCE_ID = linear_attention_gate.TOLERANCE_ID
FULL_ATTENTION_TOLERANCE_ID = full_attention_gate.TOLERANCE_ID
FULL_MODEL_TOLERANCE_ID = model_reference_gate.FULL_MODEL_TOLERANCE_ID
LOGITS_TOLERANCE_ID = model_reference_gate.LOGITS_TOLERANCE_ID
TOKEN_CASE_FIELDS = frozenset(
    {
        "prompt_id",
        "prompt_sha256",
        "rendered_prompt_sha256",
        "ferrum_prompt_token_ids",
        "reference_prompt_token_ids",
        "prompt_token_ids_sha256",
        "ferrum_generated_token_ids",
        "reference_generated_token_ids",
        "generated_token_ids_sha256",
        "reference_top2_margins",
        "near_tie_diagnostics",
        "status",
    }
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read JSON {path}: {error}") from error


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def token_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(b"".join(struct.pack("<I", token) for token in token_ids)).hexdigest()


def exact_object(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == fields, f"{label} fields differ: {sorted(set(value) ^ fields)}")
    return value


def finite(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def git_text(*args: str) -> str:
    process = subprocess.run(
        ["git", "-c", "core.preloadindex=false", "-c", "index.threads=1", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(process.returncode == 0, process.stderr.strip() or "git command failed")
    return process.stdout.strip()


def current_source_identity(*, require_clean: bool = False) -> tuple[str, str, str, dict[str, Any]]:
    catalog, provenance = tolerances.load_catalog_from_git("HEAD", None)
    summary = tolerances.validate_catalog_document(catalog, require_complete=True)
    tolerances.validate_catalog_provenance(catalog, provenance["commit"])
    tolerances.validate_no_widening_from_revision(catalog, provenance["commit"])
    source_tree_sha = git_text("rev-parse", "HEAD^{tree}")
    require(GIT_SHA_RE.fullmatch(source_tree_sha) is not None, "source tree SHA is invalid")
    if require_clean:
        require(not git_text("status", "--short"), "numerics gate requires a clean checkout")
    return (
        provenance["commit"],
        source_tree_sha,
        provenance["git_blob_sha"],
        {**summary, "catalog": catalog},
    )


def validate_reference_path(
    gate: dict[str, Any],
    gate_path: Path,
    label: str,
    source_git_sha: str,
) -> tuple[Path, dict[str, Any]]:
    reference = gate.get("reference_artifact")
    require(isinstance(reference, str), f"{label} reference artifact is missing")
    artifact_dir = Path(reference).expanduser().resolve()
    report_path = artifact_dir / "report.json"
    require(report_path.is_file(), f"{label} reference report is unavailable")
    require(
        gate.get("reference_report_sha256") == sha256_file(report_path),
        f"{label} reference report SHA differs",
    )
    require(gate_path.is_file(), f"{label} gate file is unavailable")
    report = read_json(report_path)
    require(isinstance(report, dict), f"{label} reference report must be an object")
    actual = report.get("actual")
    require(isinstance(actual, dict), f"{label} reference report actual capture is missing")
    require(
        actual.get("git_sha") == source_git_sha,
        f"{label} actual capture source SHA is stale",
    )
    return artifact_dir, report


def validate_child_gate(
    path: Path,
    *,
    label: str,
    source_git_sha: str,
    catalog_blob: str,
    raw_validator: Callable[[Path, dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    gate = read_json(path)
    require(isinstance(gate, dict), f"{label} gate must be an object")
    require(gate.get("schema_version") == 1 and gate.get("status") == "pass", f"{label} is not PASS schema v1")
    require(gate.get("catalog_commit") == source_git_sha, f"{label} source SHA is stale")
    require(gate.get("catalog_git_blob_sha") == catalog_blob, f"{label} catalog blob is stale")
    compared = gate.get("compared_catalog_commits")
    require(isinstance(compared, list) and source_git_sha in compared, f"{label} widening history is incomplete")
    artifact_dir, report = validate_reference_path(gate, path, label, source_git_sha)
    try:
        raw_validation = raw_validator(artifact_dir, report)
    except linear_attention_gate.GateError as error:
        raise GateError(f"{label} raw reference validation failed: {error}") from error
    require(isinstance(raw_validation, dict), f"{label} raw validator result is invalid")
    for key, expected in raw_validation.items():
        require(gate.get(key) == expected, f"{label} canonical validation field differs: {key}")
    return gate


def model_lock_identity() -> dict[str, Any]:
    lock = read_json(MODEL_LOCK)
    model = lock["models"][0]
    lane = model["lanes"][BACKEND]
    require(model["key"] == MODEL_KEY and len(lane["files"]) == 1, "M1 Metal model lock differs")
    return {
        "models_lock_sha256": sha256_file(MODEL_LOCK),
        "model_revision": lane["revision"],
        "model_file_sha256": lane["files"][0]["sha256"],
        "semantic_revision": lane["semantic_source"]["revision"],
        "chat_template_sha256": lane["chat_template"]["content_sha256"],
    }


def validate_binary(value: Any, label: str) -> dict[str, Any]:
    binary = exact_object(value, frozenset({"path", "sha256"}), label)
    require(isinstance(binary["path"], str) and binary["path"], f"{label}.path is invalid")
    require(isinstance(binary["sha256"], str) and SHA256_RE.fullmatch(binary["sha256"]), f"{label}.sha256 is invalid")
    path = Path(binary["path"])
    require(path.is_file(), f"{label} is unavailable: {path}")
    require(sha256_file(path) == binary["sha256"], f"{label} SHA differs")
    return binary


def validate_tokens(value: Any, *, count: int | None, label: str) -> list[int]:
    require(isinstance(value, list) and bool(value), f"{label} must be a non-empty array")
    require(
        all(isinstance(token, int) and not isinstance(token, bool) and 0 <= token < 2**32 for token in value),
        f"{label} contains an invalid token",
    )
    if count is not None:
        require(len(value) == count, f"{label} must contain exactly {count} tokens")
    return value


def validate_near_ties(value: Any, margins: list[float], label: str) -> None:
    require(isinstance(value, list), f"{label} must be an array")
    expected_steps = [index for index, margin in enumerate(margins) if margin < NEAR_TIE_MARGIN]
    require([row.get("step") for row in value if isinstance(row, dict)] == expected_steps, f"{label} does not cover every near tie")
    fields = frozenset(
        {
            "step",
            "reference_margin",
            "ferrum_top2_token_ids",
            "reference_top2_token_ids",
            "ferrum_top2_logits",
            "reference_top2_logits",
            "high_precision_reference",
        }
    )
    for index, raw in enumerate(value):
        row = exact_object(raw, fields, f"{label}[{index}]")
        step = row["step"]
        require(row["reference_margin"] == margins[step], f"{label}[{index}] margin differs")
        for field in ("ferrum_top2_token_ids", "reference_top2_token_ids"):
            require(len(validate_tokens(row[field], count=2, label=f"{label}[{index}].{field}")) == 2, "top2 token count differs")
        for field in ("ferrum_top2_logits", "reference_top2_logits"):
            require(isinstance(row[field], list) and len(row[field]) == 2, f"{label}[{index}].{field} differs")
            [finite(item, f"{label}[{index}].{field}") for item in row[field]]
        high = exact_object(
            row["high_precision_reference"],
            frozenset({"kind", "logical_dtype", "top2_token_ids", "top2_logits", "margin"}),
            f"{label}[{index}].high_precision_reference",
        )
        require(high["kind"] == "cpu_fp32_dequantized_gguf" and high["logical_dtype"] == "fp32", "near-tie oracle differs")
        validate_tokens(high["top2_token_ids"], count=2, label="near-tie oracle tokens")
        require(isinstance(high["top2_logits"], list) and len(high["top2_logits"]) == 2, "near-tie oracle logits differ")
        [finite(item, "near-tie oracle logit") for item in high["top2_logits"]]
        require(finite(high["margin"], "near-tie oracle margin") >= 0.0, "near-tie oracle margin is negative")


def require_flag_value(argv: list[str], flag: str, expected: str, label: str) -> None:
    indexes = [index for index, token in enumerate(argv) if token == flag]
    require(len(indexes) == 1, f"{label} must contain {flag} exactly once")
    index = indexes[0]
    require(index + 1 < len(argv) and argv[index + 1] == expected, f"{label} {flag} differs")


def validate_token_parity(
    path: Path,
    source_git_sha: str,
    source_tree_sha: str,
) -> dict[str, Any]:
    parity = read_json(path)
    required_fields = frozenset(
        {
            "schema_version",
            "status",
            "source_git_sha",
            "source_tree_sha",
            "source_dirty",
            "model_key",
            "backend",
            "model_revision",
            "model_file_sha256",
            "semantic_revision",
            "chat_template_sha256",
            "models_lock_sha256",
            "prompt_corpus_sha256",
            "reference_kind",
            "ferrum_binary",
            "reference_source_git_sha",
            "reference_source_dirty",
            "reference_binary",
            "deterministic_config",
            "command_contract",
            "case_count",
            "passed_count",
            "exception_count",
            "waiver_count",
            "cases",
        }
    )
    parity = exact_object(parity, required_fields, "token parity")
    require(parity["schema_version"] == 2 and parity["status"] == "pass", "token parity is not PASS schema v2")
    require(parity["source_git_sha"] == source_git_sha and parity["source_dirty"] is False, "token parity source is stale or dirty")
    require(parity["source_tree_sha"] == source_tree_sha, "token parity tree SHA is stale")
    identity = model_lock_identity()
    for field, expected in identity.items():
        require(parity[field] == expected, f"token parity {field} differs")
    require(parity["model_key"] == MODEL_KEY and parity["backend"] == BACKEND, "token parity model/backend differs")
    require(parity["prompt_corpus_sha256"] == sha256_file(PROMPT_CORPUS), "token parity prompt corpus differs")
    require(parity["reference_kind"] == "same_gguf_llama_cpp_external", "token parity reference kind differs")
    validate_binary(parity["ferrum_binary"], "Ferrum binary")
    validate_binary(parity["reference_binary"], "llama.cpp binary")
    require(
        isinstance(parity["reference_source_git_sha"], str)
        and GIT_SHA_RE.fullmatch(parity["reference_source_git_sha"])
        and parity["reference_source_dirty"] is False,
        "llama.cpp source identity is invalid",
    )
    require(
        parity["deterministic_config"]
        == {
            "temperature": 0.0,
            "seed": 9271,
            "enable_thinking": False,
            "max_output_tokens": TOKEN_COUNT,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repeat_penalty": 1.0,
        },
        "token parity deterministic config differs",
    )
    command = exact_object(
        parity["command_contract"],
        frozenset(
            {
                "ferrum_entrypoint",
                "ferrum_argv_prefix",
                "reference_entrypoint",
                "reference_argv_prefix",
                "reference_http_contract",
                "ferrum_execution_count",
                "reference_execution_count",
            }
        ),
        "token parity command contract",
    )
    require(
        command["ferrum_entrypoint"] == "run"
        and command["reference_entrypoint"] == "llama-server",
        "token parity entrypoints differ",
    )
    require(command["ferrum_execution_count"] == command["reference_execution_count"] == PROMPT_COUNT, "token parity command count differs")
    for field in ("ferrum_argv_prefix", "reference_argv_prefix"):
        require(isinstance(command[field], list) and all(isinstance(item, str) and item for item in command[field]), f"{field} is invalid")
    ferrum_argv = command["ferrum_argv_prefix"]
    reference_argv = command["reference_argv_prefix"]
    require(
        Path(ferrum_argv[0]).resolve() == Path(parity["ferrum_binary"]["path"]).resolve()
        and "run" in ferrum_argv
        and "qwen3.5:4b-q4_k_m" in ferrum_argv
        and "--prompt" in ferrum_argv
        and "--request-dump-dir" in ferrum_argv
        and "--disable-thinking" in ferrum_argv,
        "Ferrum parity command is not the explicit product run path",
    )
    require(
        Path(reference_argv[0]).resolve()
        == Path(parity["reference_binary"]["path"]).resolve(),
        "llama.cpp command binary differs",
    )
    for flag, expected in {
        "--backend": "metal",
        "--max-tokens": "64",
        "--temperature": "0",
        "--seed": "9271",
        "--top-k": "0",
        "--top-p": "1",
        "--min-p": "0",
        "--presence-penalty": "0",
        "--repeat-penalty": "1",
        "--max-model-len": "1024",
        "--max-num-seqs": "1",
        "--max-num-batched-tokens": "1024",
        "--output-format": "jsonl",
    }.items():
        require_flag_value(ferrum_argv, flag, expected, "Ferrum parity command")
    for flag, expected in {
        "--model": "MODEL",
        "--host": "127.0.0.1",
        "--port": "PORT",
        "--ctx-size": "1024",
        "--parallel": "1",
        "--threads": "4",
        "--threads-batch": "4",
        "--n-gpu-layers": "99",
        "--chat-template-kwargs": '{"enable_thinking":false}',
        "--reasoning": "off",
        "--cache-ram": "0",
    }.items():
        require_flag_value(reference_argv, flag, expected, "llama.cpp parity server command")
    require(
        reference_argv.count("--jinja") == 1
        and reference_argv.count("--no-warmup") == 1,
        "llama.cpp parity server must enable Jinja and disable warmup explicitly",
    )
    reference_http = exact_object(
        command["reference_http_contract"],
        frozenset(
            {
                "apply_template_path",
                "tokenize_path",
                "completion_path",
                "apply_template_count",
                "tokenize_count",
                "completion_count",
                "tokenize_add_special",
                "tokenize_parse_special",
                "completion_prompt_kind",
                "completion_request",
            }
        ),
        "token parity reference HTTP contract",
    )
    require(
        reference_http
        == {
            "apply_template_path": "/apply-template",
            "tokenize_path": "/tokenize",
            "completion_path": "/completion",
            "apply_template_count": PROMPT_COUNT,
            "tokenize_count": PROMPT_COUNT,
            "completion_count": PROMPT_COUNT,
            "tokenize_add_special": False,
            "tokenize_parse_special": True,
            "completion_prompt_kind": "exact_token_ids",
            "completion_request": {
                "n_predict": TOKEN_COUNT,
                "temperature": 0.0,
                "seed": 9271,
                "top_k": 0,
                "top_p": 1.0,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repeat_penalty": 1.0,
                "repeat_last_n": 64,
                "samplers": ["top_k", "top_p", "min_p", "temperature"],
                "return_tokens": True,
                "n_probs": 2,
                "post_sampling_probs": False,
                "cache_prompt": False,
                "stream": False,
                "ignore_eos": False,
            },
        },
        "token parity reference HTTP behavior differs",
    )
    require(parity["case_count"] == parity["passed_count"] == PROMPT_COUNT, "token parity must pass 20/20")
    require(parity["exception_count"] == parity["waiver_count"] == 0, "token parity has an exception or waiver")

    corpus = read_json(PROMPT_CORPUS)
    prompts = corpus["prompts"]
    require(corpus["prompt_count"] == len(prompts) == PROMPT_COUNT, "checked-in prompt corpus count differs")
    expected = {prompt["id"]: canonical_sha256(prompt["messages"]) for prompt in prompts}
    cases = parity["cases"]
    require(isinstance(cases, list) and len(cases) == PROMPT_COUNT, "token parity cases must contain 20 rows")
    require([case.get("prompt_id") for case in cases if isinstance(case, dict)] == list(expected), "token parity prompt order differs")
    for index, raw in enumerate(cases):
        case = exact_object(raw, TOKEN_CASE_FIELDS, f"token parity cases[{index}]")
        prompt_id = case["prompt_id"]
        require(case["status"] == "pass" and case["prompt_sha256"] == expected[prompt_id], f"{prompt_id} prompt/status differs")
        require(isinstance(case["rendered_prompt_sha256"], str) and SHA256_RE.fullmatch(case["rendered_prompt_sha256"]), f"{prompt_id} rendered prompt SHA invalid")
        ferrum_prompt = validate_tokens(case["ferrum_prompt_token_ids"], count=None, label=f"{prompt_id} Ferrum prompt tokens")
        reference_prompt = validate_tokens(case["reference_prompt_token_ids"], count=None, label=f"{prompt_id} reference prompt tokens")
        require(ferrum_prompt == reference_prompt, f"{prompt_id} rendered prompt tokenization differs")
        require(case["prompt_token_ids_sha256"] == token_sha256(ferrum_prompt), f"{prompt_id} prompt token SHA differs")
        ferrum_output = validate_tokens(case["ferrum_generated_token_ids"], count=TOKEN_COUNT, label=f"{prompt_id} Ferrum output")
        reference_output = validate_tokens(case["reference_generated_token_ids"], count=TOKEN_COUNT, label=f"{prompt_id} reference output")
        require(ferrum_output == reference_output, f"{prompt_id} generated token sequence differs")
        require(case["generated_token_ids_sha256"] == token_sha256(ferrum_output), f"{prompt_id} generated token SHA differs")
        margins = case["reference_top2_margins"]
        require(isinstance(margins, list) and len(margins) == TOKEN_COUNT, f"{prompt_id} top2 margin count differs")
        margins = [finite(margin, f"{prompt_id} margin") for margin in margins]
        require(all(margin >= 0.0 for margin in margins), f"{prompt_id} has a negative margin")
        validate_near_ties(case["near_tie_diagnostics"], margins, f"{prompt_id} near ties")
    return {
        "case_count": PROMPT_COUNT,
        "token_count_per_case": TOKEN_COUNT,
        "matched_token_count": PROMPT_COUNT * TOKEN_COUNT,
        "exception_count": 0,
        "waiver_count": 0,
        "reference_kind": parity["reference_kind"],
        "ferrum_binary_sha256": parity["ferrum_binary"]["sha256"],
        "reference_binary_sha256": parity["reference_binary"]["sha256"],
    }


def input_receipt(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def validate_and_write(
    *,
    op_artifact: Path,
    linear_gate_path: Path,
    full_attention_gate_path: Path,
    model_gate_path: Path,
    token_parity_path: Path,
    out: Path,
    require_clean: bool = True,
    child_raw_validators: dict[
        str, Callable[[Path, dict[str, Any]], dict[str, Any]]
    ] | None = None,
) -> dict[str, Any]:
    source_git_sha, source_tree_sha, catalog_blob, catalog_summary = current_source_identity(
        require_clean=require_clean
    )
    rows = {row["tolerance_id"]: row for row in catalog_summary["catalog"]["rows"]}
    validators = child_raw_validators or {
        "linear-attention": lambda artifact_dir, report: linear_attention_gate.validate_report(
            artifact_dir, report, rows[LINEAR_TOLERANCE_ID]
        ),
        "full-attention": lambda artifact_dir, report: full_attention_gate.validate_report(
            artifact_dir, report, rows[FULL_ATTENTION_TOLERANCE_ID]
        ),
        "full-model": lambda artifact_dir, report: model_reference_gate.validate_report(
            artifact_dir,
            report,
            rows[FULL_MODEL_TOLERANCE_ID],
            rows[LOGITS_TOLERANCE_ID],
        ),
    }
    require(
        set(validators) == {"linear-attention", "full-attention", "full-model"},
        "child raw validator set differs",
    )
    op_receipt = op_numerics.validate_receipt(op_artifact, git_revision=source_git_sha)
    linear = validate_child_gate(
        linear_gate_path,
        label="linear-attention",
        source_git_sha=source_git_sha,
        catalog_blob=catalog_blob,
        raw_validator=validators["linear-attention"],
    )
    full_attention = validate_child_gate(
        full_attention_gate_path,
        label="full-attention",
        source_git_sha=source_git_sha,
        catalog_blob=catalog_blob,
        raw_validator=validators["full-attention"],
    )
    model = validate_child_gate(
        model_gate_path,
        label="full-model",
        source_git_sha=source_git_sha,
        catalog_blob=catalog_blob,
        raw_validator=validators["full-model"],
    )
    require(linear.get("tolerance_id") == LINEAR_TOLERANCE_ID and linear.get("row_fingerprint") == rows[LINEAR_TOLERANCE_ID]["row_fingerprint"], "linear-attention row binding differs")
    require(full_attention.get("tolerance_id") == FULL_ATTENTION_TOLERANCE_ID and full_attention.get("row_fingerprint") == rows[FULL_ATTENTION_TOLERANCE_ID]["row_fingerprint"], "full-attention row binding differs")
    require(
        model.get("tolerances")
        == {
            FULL_MODEL_TOLERANCE_ID: rows[FULL_MODEL_TOLERANCE_ID]["row_fingerprint"],
            LOGITS_TOLERANCE_ID: rows[LOGITS_TOLERANCE_ID]["row_fingerprint"],
        },
        "full-model/logits row binding differs",
    )
    require(model.get("catalog_coverage_status") == "complete", "model gate catalog coverage is incomplete")
    token_summary = validate_token_parity(
        token_parity_path.expanduser().resolve(),
        source_git_sha,
        source_tree_sha,
    )
    out = out.expanduser().resolve()
    require(not out.exists() or not any(out.iterdir()), "output directory is not empty")
    out.mkdir(parents=True, exist_ok=True)
    pass_line = f"{PASS_PREFIX}: {out}"
    inputs = {
        "metal_op_numerics": input_receipt(op_artifact / "metal-op-numerics.json"),
        "linear_attention": input_receipt(linear_gate_path),
        "full_attention": input_receipt(full_attention_gate_path),
        "full_model": input_receipt(model_gate_path),
        "token_parity": input_receipt(token_parity_path),
    }
    validation = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_g08a_numerics_validation",
        "status": "pass",
        "validated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "source_git_sha": source_git_sha,
        "source_tree_sha": source_tree_sha,
        "catalog_git_blob_sha": catalog_blob,
        "catalog_row_count": catalog_summary["row_count"],
        "artifact_local_tolerance_count": 0,
        "operation_state_row_count": op_receipt["summary"]["row_count"],
        "layer_checkpoint_count": 2,
        "full_model_checkpoint_count": 1,
        "full_vocabulary_logits_checkpoint_count": 1,
        "token_parity": token_summary,
        "inputs": inputs,
        "pass_line": pass_line,
    }
    validation_path = out / "validation.json"
    write_json(validation_path, validation)
    manifest = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_g08a_numerics_manifest",
        "lane": "runtime-vnext-g08a-numerics",
        "status": "pass",
        "canonical": True,
        "source_git_sha": source_git_sha,
        "source_tree_sha": source_tree_sha,
        "dirty": False,
        "artifact_dir": str(out),
        "validation": {"path": str(validation_path), "sha256": sha256_file(validation_path)},
        "summary": {
            "catalog_row_count": catalog_summary["row_count"],
            "operation_state_row_count": op_receipt["summary"]["row_count"],
            **token_summary,
        },
        "inputs": inputs,
        "does_not_prove": [
            "G08A CUDA/Metal C01-C21 model matrices",
            "G08A source ownership or legacy deletion",
            "G08A performance smoke",
            "G08A final PASS",
        ],
        "pass_line": pass_line,
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def fixture_child(path: Path, *, kind: str, source_git_sha: str, catalog_blob: str, reference: Path, rows: dict[str, Any]) -> None:
    gate: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "catalog_commit": source_git_sha,
        "catalog_git_blob_sha": catalog_blob,
        "compared_catalog_commits": [source_git_sha],
        "reference_artifact": str(reference),
        "reference_report_sha256": sha256_file(reference / "report.json"),
    }
    if kind == "linear":
        gate.update({"tolerance_id": LINEAR_TOLERANCE_ID, "row_fingerprint": rows[LINEAR_TOLERANCE_ID]["row_fingerprint"]})
    elif kind == "full_attention":
        gate.update({"tolerance_id": FULL_ATTENTION_TOLERANCE_ID, "row_fingerprint": rows[FULL_ATTENTION_TOLERANCE_ID]["row_fingerprint"]})
    else:
        gate.update({"catalog_coverage_status": "complete", "tolerances": {FULL_MODEL_TOLERANCE_ID: rows[FULL_MODEL_TOLERANCE_ID]["row_fingerprint"], LOGITS_TOLERANCE_ID: rows[LOGITS_TOLERANCE_ID]["row_fingerprint"]}})
    write_json(path, gate)


def fixture_parity(
    path: Path,
    source_git_sha: str,
    source_tree_sha: str,
    binary: Path,
) -> dict[str, Any]:
    corpus = read_json(PROMPT_CORPUS)
    cases = []
    for ordinal, prompt in enumerate(corpus["prompts"], start=1):
        prompt_tokens = [1000 + ordinal, 2000 + ordinal]
        output_tokens = list(range(ordinal * 100, ordinal * 100 + TOKEN_COUNT))
        cases.append(
            {
                "prompt_id": prompt["id"],
                "prompt_sha256": canonical_sha256(prompt["messages"]),
                "rendered_prompt_sha256": hashlib.sha256(f"rendered-{ordinal}".encode()).hexdigest(),
                "ferrum_prompt_token_ids": list(prompt_tokens),
                "reference_prompt_token_ids": list(prompt_tokens),
                "prompt_token_ids_sha256": token_sha256(prompt_tokens),
                "ferrum_generated_token_ids": list(output_tokens),
                "reference_generated_token_ids": list(output_tokens),
                "generated_token_ids_sha256": token_sha256(output_tokens),
                "reference_top2_margins": [0.5] * TOKEN_COUNT,
                "near_tie_diagnostics": [],
                "status": "pass",
            }
        )
    identity = model_lock_identity()
    binary_identity = {"path": str(binary), "sha256": sha256_file(binary)}
    return {
        "schema_version": 2,
        "status": "pass",
        "source_git_sha": source_git_sha,
        "source_tree_sha": source_tree_sha,
        "source_dirty": False,
        "model_key": MODEL_KEY,
        "backend": BACKEND,
        **identity,
        "prompt_corpus_sha256": sha256_file(PROMPT_CORPUS),
        "reference_kind": "same_gguf_llama_cpp_external",
        "ferrum_binary": binary_identity,
        "reference_source_git_sha": "2" * 40,
        "reference_source_dirty": False,
        "reference_binary": binary_identity,
        "deterministic_config": {
            "temperature": 0.0,
            "seed": 9271,
            "enable_thinking": False,
            "max_output_tokens": TOKEN_COUNT,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repeat_penalty": 1.0,
        },
        "command_contract": {
            "ferrum_entrypoint": "run",
            "ferrum_argv_prefix": [
                str(binary),
                "run",
                "qwen3.5:4b-q4_k_m",
                "--backend",
                "metal",
                "--prompt",
                "PROMPT",
                "--request-dump-dir",
                "REQUEST_DUMP_DIR",
                "--disable-thinking",
                "--max-tokens",
                "64",
                "--temperature",
                "0",
                "--seed",
                "9271",
                "--top-k",
                "0",
                "--top-p",
                "1",
                "--min-p",
                "0",
                "--presence-penalty",
                "0",
                "--repeat-penalty",
                "1",
                "--max-model-len",
                "1024",
                "--max-num-seqs",
                "1",
                "--max-num-batched-tokens",
                "1024",
                "--output-format",
                "jsonl",
            ],
            "reference_entrypoint": "llama-server",
            "reference_argv_prefix": [
                str(binary),
                "--model",
                "MODEL",
                "--host",
                "127.0.0.1",
                "--port",
                "PORT",
                "--ctx-size",
                "1024",
                "--parallel",
                "1",
                "--threads",
                "4",
                "--threads-batch",
                "4",
                "--n-gpu-layers",
                "99",
                "--jinja",
                "--chat-template-kwargs",
                '{"enable_thinking":false}',
                "--reasoning",
                "off",
                "--no-warmup",
                "--cache-ram",
                "0",
            ],
            "reference_http_contract": {
                "apply_template_path": "/apply-template",
                "tokenize_path": "/tokenize",
                "completion_path": "/completion",
                "apply_template_count": PROMPT_COUNT,
                "tokenize_count": PROMPT_COUNT,
                "completion_count": PROMPT_COUNT,
                "tokenize_add_special": False,
                "tokenize_parse_special": True,
                "completion_prompt_kind": "exact_token_ids",
                "completion_request": {
                    "n_predict": TOKEN_COUNT,
                    "temperature": 0.0,
                    "seed": 9271,
                    "top_k": 0,
                    "top_p": 1.0,
                    "min_p": 0.0,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repeat_penalty": 1.0,
                    "repeat_last_n": 64,
                    "samplers": ["top_k", "top_p", "min_p", "temperature"],
                    "return_tokens": True,
                    "n_probs": 2,
                    "post_sampling_probs": False,
                    "cache_prompt": False,
                    "stream": False,
                    "ignore_eos": False,
                },
            },
            "ferrum_execution_count": PROMPT_COUNT,
            "reference_execution_count": PROMPT_COUNT,
        },
        "case_count": PROMPT_COUNT,
        "passed_count": PROMPT_COUNT,
        "exception_count": 0,
        "waiver_count": 0,
        "cases": cases,
    }


def self_test() -> None:
    source_git_sha, source_tree_sha, catalog_blob, summary = current_source_identity()
    rows = {row["tolerance_id"]: row for row in summary["catalog"]["rows"]}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        reference = root / "reference"
        reference.mkdir()
        write_json(reference / "report.json", {"actual": {"git_sha": source_git_sha}})
        linear = root / "linear.json"
        full_attention = root / "full-attention.json"
        model = root / "model.json"
        fixture_child(linear, kind="linear", source_git_sha=source_git_sha, catalog_blob=catalog_blob, reference=reference, rows=rows)
        fixture_child(full_attention, kind="full_attention", source_git_sha=source_git_sha, catalog_blob=catalog_blob, reference=reference, rows=rows)
        fixture_child(model, kind="model", source_git_sha=source_git_sha, catalog_blob=catalog_blob, reference=reference, rows=rows)
        binary = root / "binary"
        binary.write_bytes(b"fixture")
        parity_path = root / "token-parity.json"
        parity = fixture_parity(parity_path, source_git_sha, source_tree_sha, binary)
        write_json(parity_path, parity)
        token_summary = validate_token_parity(
            parity_path,
            source_git_sha,
            source_tree_sha,
        )
        require(token_summary["matched_token_count"] == 1280, "token parity fixture count differs")

        def reject(name: str, mutate: Callable[[dict[str, Any]], None], marker: str) -> None:
            candidate = copy.deepcopy(parity)
            mutate(candidate)
            write_json(parity_path, candidate)
            try:
                validate_token_parity(parity_path, source_git_sha, source_tree_sha)
            except GateError as error:
                require(marker.lower() in str(error).lower(), f"{name} rejected unexpectedly: {error}")
                return
            raise GateError(f"{name} unexpectedly passed")

        reject("token flip", lambda value: value["cases"][0]["reference_generated_token_ids"].__setitem__(0, 999), "sequence differs")
        reject("waiver", lambda value: value.update({"waiver_count": 1}), "exception or waiver")
        reject("short output", lambda value: value["cases"][0].update({"ferrum_generated_token_ids": value["cases"][0]["ferrum_generated_token_ids"][:-1]}), "exactly 64")
        write_json(parity_path, parity)

        op_root = root / "metal-ops"
        op_root.mkdir()
        op_rows = op_numerics.operation_rows(summary["catalog"])
        test_names = sorted({row["basis"]["test_name"] for row in op_rows.values()})
        op_log = "\n".join(
            [
                *(f"test module::{name} ... ok" for name in test_names),
                *(
                    op_numerics.METRICS_PREFIX
                    + json.dumps(
                        {
                            "label": row["tolerance_id"],
                            "metrics": op_numerics.fixture_metric(row),
                        },
                        sort_keys=True,
                    )
                    for row in op_rows.values()
                ),
                "test result: ok. 7 passed; 0 failed",
            ]
        )
        stdout_path = op_root / "cargo-test.stdout.log"
        stderr_path = op_root / "cargo-test.stderr.log"
        stdout_path.write_text(op_log, encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        op_receipt = {
            "schema_version": 1,
            "status": "pass",
            "source_git_sha": source_git_sha,
            "catalog_git_blob_sha": catalog_blob,
            "catalog_row_count": summary["row_count"],
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_sha256": sha256_file(stderr_path),
            "test_binary": {"path": str(binary), "sha256": sha256_file(binary)},
            "summary": op_numerics.parse_log(op_log, summary["catalog"]),
        }
        write_json(op_root / "metal-op-numerics.json", op_receipt)
        aggregate_out = root / "aggregate"
        aggregate = validate_and_write(
            op_artifact=op_root,
            linear_gate_path=linear,
            full_attention_gate_path=full_attention,
            model_gate_path=model,
            token_parity_path=parity_path,
            out=aggregate_out,
            require_clean=False,
            child_raw_validators={
                "linear-attention": lambda _artifact_dir, _report: {},
                "full-attention": lambda _artifact_dir, _report: {},
                "full-model": lambda _artifact_dir, _report: {},
            },
        )
        require(
            aggregate["summary"]["operation_state_row_count"] == 27
            and aggregate["summary"]["matched_token_count"] == 1280
            and (aggregate_out / "manifest.json").is_file(),
            "aggregate fixture output differs",
        )

        stale_report = {"actual": {"git_sha": "0" * 40}}
        write_json(reference / "report.json", stale_report)
        stale_gate = read_json(linear)
        stale_gate["reference_report_sha256"] = sha256_file(reference / "report.json")
        write_json(linear, stale_gate)
        try:
            validate_child_gate(
                linear,
                label="linear-attention",
                source_git_sha=source_git_sha,
                catalog_blob=catalog_blob,
                raw_validator=lambda _artifact_dir, _report: {},
            )
        except GateError as error:
            require("source sha is stale" in str(error).lower(), "wrong stale capture rejection")
        else:
            raise GateError("stale actual capture unexpectedly passed")

        write_json(reference / "report.json", {"actual": {"git_sha": source_git_sha}})
        forged_gate = read_json(linear)
        forged_gate["reference_report_sha256"] = sha256_file(reference / "report.json")
        write_json(linear, forged_gate)
        try:
            validate_child_gate(
                linear,
                label="linear-attention",
                source_git_sha=source_git_sha,
                catalog_blob=catalog_blob,
                raw_validator=lambda _artifact_dir, _report: {"metrics": {"forged": True}},
            )
        except GateError as error:
            require(
                "canonical validation field differs" in str(error).lower(),
                "wrong forged child-gate rejection",
            )
        else:
            raise GateError("forged child gate unexpectedly passed")
    print(SELFTEST_PASS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op-artifact", type=Path)
    parser.add_argument("--linear-attention", type=Path)
    parser.add_argument("--full-attention", type=Path)
    parser.add_argument("--full-model", type=Path)
    parser.add_argument("--token-parity", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        required = [args.op_artifact, args.linear_attention, args.full_attention, args.full_model, args.token_parity, args.out]
        require(all(path is not None for path in required), "all evidence inputs and --out are required")
        manifest = validate_and_write(
            op_artifact=args.op_artifact,
            linear_gate_path=args.linear_attention,
            full_attention_gate_path=args.full_attention,
            model_gate_path=args.full_model,
            token_parity_path=args.token_parity,
            out=args.out,
        )
        print(manifest["pass_line"])
        return 0
    except (GateError, op_numerics.EvidenceError, tolerances.CatalogError, OSError, ValueError) as error:
        print(f"{FAIL_PREFIX}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
