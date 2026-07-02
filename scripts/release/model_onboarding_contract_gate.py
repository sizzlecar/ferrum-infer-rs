#!/usr/bin/env python3
"""Validate model onboarding contracts before README/support claims.

The gate is intentionally artifact-driven: a backend support claim must cite
run, serve, profile, replay, preset, and capability artifacts produced at the
same git SHA as the contract.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "scripts/release/model_onboarding_contract.schema.json"
DEFAULT_FIXTURES = REPO_ROOT / "scripts/release/fixtures/model_onboarding_contract"
PASS_LINE = "MODEL ONBOARDING CONTRACT PASS"
SELFTEST_PASS_LINE = "MODEL ONBOARDING CONTRACT SELFTEST PASS"
SCHEMA_VERSION = 1
GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
REQUIRED_TEMPLATE_CASES = {
    "single_turn",
    "multi_turn",
    "system_message",
    "tool_injection",
    "reasoning_history",
}
KNOWN_ARCHITECTURES = {
    "deepseek",
    "deepseek_r1",
    "llama",
    "llama_dense",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "qwen3_moe",
}
FORBIDDEN_KEYS = {
    "ad_hoc_diagnostics",
    "debug_only",
    "diagnostic_fields",
    "temporary_diagnostics",
    "tmp_diagnostics",
}
TOP_LEVEL_KEYS = {
    "schema_version",
    "contract_id",
    "source_git_sha",
    "model",
    "facts",
    "runtime_preset",
    "fallback_policy",
    "capabilities",
    "backend_support",
    "performance_claims",
}
ARTIFACT_KEYS = {
    "kind",
    "path",
    "status",
    "pass_line",
    "git_sha",
    "created_at",
    "entrypoint",
    "backend",
    "impact_domains",
}
EXPECTED_FAIL_MARKERS = {
    "builtin_template_fallback_allowed.json": "fallback_policy.allow_builtin_template_fallback",
    "cuda_claim_without_cuda_artifact.json": "backend_support[0].correctness.run_artifact",
    "missing_chat_template_source.json": "facts.chat_template_source",
    "missing_run_artifact.json": "backend_support[0].correctness.run_artifact",
    "missing_serve_artifact.json": "backend_support[0].correctness.serve_artifact",
    "native_operator_claim_without_artifact.json": "native_operator_artifact",
    "performance_claim_before_correctness.json": "performance claims require correctness pass",
    "runtime_preset_missing_rejected_candidates.json": "runtime_preset.rejected_candidates",
    "schema_claim_without_schema_artifact.json": "structured_output",
    "silent_fallback.json": "silent backend fallback",
    "stale_artifact_git_sha.json": "stale artifact git_sha",
    "tool_claim_without_tool_artifact.json": "tool_calling",
    "unknown_architecture_without_design_doc.json": "model.design_doc",
}


class ContractError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ContractError(f"{path}: expected JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(args: list[str], default: str = "unknown") -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return default
    return proc.stdout.strip() or default


def as_object(value: Any, label: str, problems: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        problems.append(f"{label} must be a JSON object")
        return {}
    return value


def as_list(value: Any, label: str, problems: list[str]) -> list[Any]:
    if not isinstance(value, list):
        problems.append(f"{label} must be a JSON array")
        return []
    return value


def non_empty_string(value: Any, label: str, problems: list[str]) -> str | None:
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{label} must be a non-empty string")
        return None
    return value.strip()


def no_extra_keys(value: dict[str, Any], allowed: set[str], label: str, problems: list[str]) -> None:
    extra = sorted(set(value) - allowed)
    if extra:
        problems.append(f"{label} unexpected keys: {', '.join(extra)}")


def scan_forbidden_keys(value: Any, label: str, problems: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_label = f"{label}.{key}" if label else str(key)
            if key in FORBIDDEN_KEYS:
                problems.append(f"{key_label} is a forbidden temporary diagnostic field")
            scan_forbidden_keys(child, key_label, problems)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            scan_forbidden_keys(child, f"{label}[{index}]", problems)


def resolve_artifact_path(raw: str, contract_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    contract_relative = (contract_path.parent / path).resolve()
    if contract_relative.exists():
        return contract_relative
    return (REPO_ROOT / path).resolve()


def validate_artifact(
    value: Any,
    label: str,
    *,
    contract_path: Path,
    source_git_sha: str,
    expected_kind: str | None = None,
    expected_backend: str | None = None,
    expected_entrypoint: str | None = None,
    require_exists: bool = True,
    problems: list[str],
) -> dict[str, Any]:
    artifact = as_object(value, label, problems)
    if not artifact:
        return {}
    no_extra_keys(artifact, ARTIFACT_KEYS, label, problems)
    kind = non_empty_string(artifact.get("kind"), f"{label}.kind", problems)
    if expected_kind is not None and kind != expected_kind:
        problems.append(f"{label}.kind must be {expected_kind!r}")
    if kind is not None and ("temporary" in kind or "diagnostic" in kind):
        problems.append(f"{label}.kind must not be temporary/diagnostic evidence")
    status = artifact.get("status")
    if status != "pass":
        problems.append(f"{label}.status must be 'pass'")
    git_sha = non_empty_string(artifact.get("git_sha"), f"{label}.git_sha", problems)
    if git_sha is not None:
        if not GIT_SHA_RE.match(git_sha):
            problems.append(f"{label}.git_sha must be a 40-character git SHA")
        elif git_sha.lower() != source_git_sha.lower():
            problems.append(
                f"{label}: stale artifact git_sha {git_sha} != contract source_git_sha {source_git_sha}"
            )
    pass_line = non_empty_string(artifact.get("pass_line"), f"{label}.pass_line", problems)
    if pass_line is not None:
        upper = pass_line.upper()
        if "PASS:" not in upper:
            problems.append(f"{label}.pass_line must contain a gate PASS line")
        if "SELFTEST" in upper or "SELF-TEST" in upper:
            problems.append(f"{label}.pass_line must not be self-test evidence")
    if expected_backend is not None:
        backend = artifact.get("backend")
        if backend != expected_backend:
            problems.append(f"{label}.backend must be {expected_backend!r}")
    if expected_entrypoint is not None:
        entrypoint = artifact.get("entrypoint")
        if entrypoint != expected_entrypoint:
            problems.append(f"{label}.entrypoint must be {expected_entrypoint!r}")
    raw_path = non_empty_string(artifact.get("path"), f"{label}.path", problems)
    if raw_path is not None and require_exists:
        path = resolve_artifact_path(raw_path, contract_path)
        if not path.exists():
            problems.append(f"{label}.path does not exist: {raw_path}")
    return artifact


def validate_model(model: dict[str, Any], contract_path: Path, source_git_sha: str, problems: list[str]) -> None:
    no_extra_keys(model, {"id", "family", "architecture", "weight_format", "source", "design_doc"}, "model", problems)
    for key in ["id", "family", "architecture", "weight_format", "source"]:
        non_empty_string(model.get(key), f"model.{key}", problems)
    architecture = model.get("architecture")
    if isinstance(architecture, str) and architecture not in KNOWN_ARCHITECTURES:
        validate_artifact(
            model.get("design_doc"),
            "model.design_doc",
            contract_path=contract_path,
            source_git_sha=source_git_sha,
            expected_kind="design_doc",
            problems=problems,
        )


def validate_facts(facts: dict[str, Any], contract_path: Path, source_git_sha: str, problems: list[str]) -> None:
    allowed = {
        "tokenizer_source",
        "chat_template_source",
        "generation_config_source",
        "token_id_source",
        "eos_token_ids",
        "bos_token_id",
        "stop_tokens",
        "template_golden",
        "template_golden_cases",
    }
    no_extra_keys(facts, allowed, "facts", problems)
    for key in ["tokenizer_source", "chat_template_source", "generation_config_source"]:
        non_empty_string(facts.get(key), f"facts.{key}", problems)
    chat_template_source = facts.get("chat_template_source")
    if isinstance(chat_template_source, str) and chat_template_source.lower() in {"builtin", "chatml", "builtin_chatml"}:
        problems.append("facts.chat_template_source must cite model metadata, not builtin ChatML")
    if facts.get("token_id_source") not in {"metadata", "generation_config", "documented_config"}:
        problems.append("facts.token_id_source must be metadata, generation_config, or documented_config")
    if not isinstance(facts.get("eos_token_ids"), list):
        problems.append("facts.eos_token_ids must be an array")
    if not isinstance(facts.get("stop_tokens"), list):
        problems.append("facts.stop_tokens must be an array")
    validate_artifact(
        facts.get("template_golden"),
        "facts.template_golden",
        contract_path=contract_path,
        source_git_sha=source_git_sha,
        expected_kind="template_golden",
        problems=problems,
    )
    cases = {str(item) for item in as_list(facts.get("template_golden_cases"), "facts.template_golden_cases", problems)}
    missing_cases = sorted(REQUIRED_TEMPLATE_CASES - cases)
    if missing_cases:
        problems.append(f"facts.template_golden_cases missing: {', '.join(missing_cases)}")


def validate_runtime_preset(runtime_preset: dict[str, Any], problems: list[str]) -> None:
    no_extra_keys(runtime_preset, {"selected", "rejected_candidates"}, "runtime_preset", problems)
    non_empty_string(runtime_preset.get("selected"), "runtime_preset.selected", problems)
    rejected = as_list(runtime_preset.get("rejected_candidates"), "runtime_preset.rejected_candidates", problems)
    if not rejected:
        problems.append("runtime_preset.rejected_candidates must record at least one rejected candidate")
    for index, item in enumerate(rejected):
        non_empty_string(item, f"runtime_preset.rejected_candidates[{index}]", problems)


def validate_fallback_policy(policy: dict[str, Any], problems: list[str]) -> None:
    allowed = {
        "allow_builtin_template_fallback",
        "allow_backend_fallback",
        "allow_attention_fallback",
    }
    no_extra_keys(policy, allowed, "fallback_policy", problems)
    for key in sorted(allowed):
        if policy.get(key) is not False:
            problems.append(f"fallback_policy.{key} must be false")


def validate_capabilities(
    capabilities: dict[str, Any],
    contract_path: Path,
    source_git_sha: str,
    problems: list[str],
) -> set[str]:
    if not capabilities:
        problems.append("capabilities must not be empty")
    supported: set[str] = set()
    for name, raw in sorted(capabilities.items()):
        label = f"capabilities.{name}"
        capability = as_object(raw, label, problems)
        if not capability:
            continue
        no_extra_keys(capability, {"supported", "source", "artifact"}, label, problems)
        if not isinstance(capability.get("supported"), bool):
            problems.append(f"{label}.supported must be boolean")
            continue
        non_empty_string(capability.get("source"), f"{label}.source", problems)
        if capability["supported"]:
            supported.add(name)
            expected_kind = {
                "structured_output": "structured_schema",
                "tool_calling": "tool_call",
            }.get(name, "capability")
            validate_artifact(
                capability.get("artifact"),
                f"{label}.artifact",
                contract_path=contract_path,
                source_git_sha=source_git_sha,
                expected_kind=expected_kind,
                problems=problems,
            )
    return supported


def validate_backend_support(
    item: dict[str, Any],
    index: int,
    *,
    supported_capabilities: set[str],
    contract_path: Path,
    source_git_sha: str,
    problems: list[str],
) -> None:
    label = f"backend_support[{index}]"
    allowed = {
        "backend",
        "status",
        "hardware",
        "selection",
        "fallback",
        "native_operators",
        "correctness",
        "performance_claims",
    }
    no_extra_keys(item, allowed, label, problems)
    backend = item.get("backend")
    if backend not in {"cpu", "cuda", "metal"}:
        problems.append(f"{label}.backend must be cpu, cuda, or metal")
        return
    status = item.get("status")
    if status != "supported":
        problems.append(f"{label}.status must be 'supported' for support claims")
        return
    selection = as_object(item.get("selection"), f"{label}.selection", problems)
    no_extra_keys(
        selection,
        {"runtime_preset", "scheduler", "attention_impl", "kv_layout", "kv_dtype", "recurrent_state_max_slots"},
        f"{label}.selection",
        problems,
    )
    for key in ["runtime_preset", "scheduler", "attention_impl", "kv_layout", "kv_dtype"]:
        non_empty_string(selection.get(key), f"{label}.selection.{key}", problems)
    fallback = as_object(item.get("fallback"), f"{label}.fallback", problems)
    no_extra_keys(fallback, {"allowed", "actual_backend", "actual_attention_impl", "reason"}, f"{label}.fallback", problems)
    if fallback.get("allowed") is not False:
        problems.append(f"{label}: fallback.allowed must be false")
    if fallback.get("actual_backend") != backend:
        problems.append(f"{label}: silent backend fallback actual_backend={fallback.get('actual_backend')!r} requested={backend!r}")
    actual_attention = fallback.get("actual_attention_impl")
    expected_attention = selection.get("attention_impl")
    if actual_attention is not None and actual_attention != expected_attention:
        problems.append(
            f"{label}: silent attention fallback actual_attention_impl={actual_attention!r} "
            f"selected={expected_attention!r}"
        )
    correctness = as_object(item.get("correctness"), f"{label}.correctness", problems)
    no_extra_keys(
        correctness,
        {
            "status",
            "run_artifact",
            "serve_artifact",
            "profile_artifact",
            "replay_artifact",
            "preset_snapshot",
            "tool_call_artifact",
            "structured_output_artifact",
            "native_operator_artifact",
        },
        f"{label}.correctness",
        problems,
    )
    correctness_status = correctness.get("status")
    if correctness_status != "pass":
        problems.append(f"{label}.correctness.status must be 'pass'")
    validate_artifact(
        correctness.get("run_artifact"),
        f"{label}.correctness.run_artifact",
        contract_path=contract_path,
        source_git_sha=source_git_sha,
        expected_kind="run_smoke",
        expected_backend=backend,
        expected_entrypoint="run",
        problems=problems,
    )
    validate_artifact(
        correctness.get("serve_artifact"),
        f"{label}.correctness.serve_artifact",
        contract_path=contract_path,
        source_git_sha=source_git_sha,
        expected_kind="serve_smoke",
        expected_backend=backend,
        expected_entrypoint="serve",
        problems=problems,
    )
    validate_artifact(
        correctness.get("profile_artifact"),
        f"{label}.correctness.profile_artifact",
        contract_path=contract_path,
        source_git_sha=source_git_sha,
        expected_kind="profile",
        expected_backend=backend,
        problems=problems,
    )
    validate_artifact(
        correctness.get("replay_artifact"),
        f"{label}.correctness.replay_artifact",
        contract_path=contract_path,
        source_git_sha=source_git_sha,
        expected_kind="replay",
        expected_backend=backend,
        problems=problems,
    )
    validate_artifact(
        correctness.get("preset_snapshot"),
        f"{label}.correctness.preset_snapshot",
        contract_path=contract_path,
        source_git_sha=source_git_sha,
        expected_kind="preset_snapshot",
        expected_backend=backend,
        problems=problems,
    )
    if "tool_calling" in supported_capabilities:
        validate_artifact(
            correctness.get("tool_call_artifact"),
            f"{label}.correctness.tool_call_artifact",
            contract_path=contract_path,
            source_git_sha=source_git_sha,
            expected_kind="tool_call",
            expected_backend=backend,
            problems=problems,
        )
    if "structured_output" in supported_capabilities:
        validate_artifact(
            correctness.get("structured_output_artifact"),
            f"{label}.correctness.structured_output_artifact",
            contract_path=contract_path,
            source_git_sha=source_git_sha,
            expected_kind="structured_schema",
            expected_backend=backend,
            problems=problems,
        )
    native_operators = as_list(item.get("native_operators", []), f"{label}.native_operators", problems)
    if native_operators:
        validate_artifact(
            correctness.get("native_operator_artifact"),
            f"{label}.correctness.native_operator_artifact",
            contract_path=contract_path,
            source_git_sha=source_git_sha,
            expected_kind="native_operator",
            expected_backend=backend,
            problems=problems,
        )
    performance_claims = as_list(item.get("performance_claims", []), f"{label}.performance_claims", problems)
    if performance_claims and correctness_status != "pass":
        problems.append(f"{label}: performance claims require correctness pass")
    for claim_index, claim in enumerate(performance_claims):
        validate_artifact(
            claim,
            f"{label}.performance_claims[{claim_index}]",
            contract_path=contract_path,
            source_git_sha=source_git_sha,
            expected_kind="performance",
            expected_backend=backend,
            problems=problems,
        )


def validate_contract(contract: dict[str, Any], *, contract_path: Path) -> list[str]:
    problems: list[str] = []
    scan_forbidden_keys(contract, "", problems)
    no_extra_keys(contract, TOP_LEVEL_KEYS, "contract", problems)
    if contract.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    non_empty_string(contract.get("contract_id"), "contract_id", problems)
    source_git_sha = non_empty_string(contract.get("source_git_sha"), "source_git_sha", problems)
    if source_git_sha is not None and not GIT_SHA_RE.match(source_git_sha):
        problems.append("source_git_sha must be a 40-character git SHA")
        source_git_sha = "0" * 40
    source_git_sha = source_git_sha or "0" * 40
    validate_model(as_object(contract.get("model"), "model", problems), contract_path, source_git_sha, problems)
    validate_facts(as_object(contract.get("facts"), "facts", problems), contract_path, source_git_sha, problems)
    validate_runtime_preset(as_object(contract.get("runtime_preset"), "runtime_preset", problems), problems)
    validate_fallback_policy(as_object(contract.get("fallback_policy"), "fallback_policy", problems), problems)
    supported_capabilities = validate_capabilities(
        as_object(contract.get("capabilities"), "capabilities", problems),
        contract_path,
        source_git_sha,
        problems,
    )
    backend_support = as_list(contract.get("backend_support"), "backend_support", problems)
    if not backend_support:
        problems.append("backend_support must contain at least one supported backend")
    for index, raw_item in enumerate(backend_support):
        validate_backend_support(
            as_object(raw_item, f"backend_support[{index}]", problems),
            index,
            supported_capabilities=supported_capabilities,
            contract_path=contract_path,
            source_git_sha=source_git_sha,
            problems=problems,
        )
    for index, claim in enumerate(as_list(contract.get("performance_claims", []), "performance_claims", problems)):
        validate_artifact(
            claim,
            f"performance_claims[{index}]",
            contract_path=contract_path,
            source_git_sha=source_git_sha,
            expected_kind="performance",
            problems=problems,
        )
    return problems


def validate_contract_path(path: Path) -> dict[str, Any]:
    data = read_json(path)
    problems = validate_contract(data, contract_path=path)
    return {
        "contract": str(path),
        "contract_id": data.get("contract_id"),
        "model_id": (data.get("model") or {}).get("id") if isinstance(data.get("model"), dict) else None,
        "status": "pass" if not problems else "fail",
        "problems": problems,
    }


def load_schema() -> dict[str, Any]:
    schema = read_json(SCHEMA_PATH)
    if schema.get("title") != "Ferrum model onboarding contract":
        raise ContractError(f"{SCHEMA_PATH}: unexpected schema title")
    return schema


def run_selftest(fixtures: Path = DEFAULT_FIXTURES) -> dict[str, Any]:
    load_schema()
    pass_paths = sorted((fixtures / "pass").glob("*.json"))
    fail_paths = sorted((fixtures / "fail").glob("*.json"))
    if not pass_paths:
        raise ContractError(f"missing pass fixtures under {fixtures / 'pass'}")
    if not fail_paths:
        raise ContractError(f"missing fail fixtures under {fixtures / 'fail'}")
    missing_fail_fixtures = sorted(set(EXPECTED_FAIL_MARKERS) - {path.name for path in fail_paths})
    if missing_fail_fixtures:
        raise ContractError(f"missing expected fail fixtures: {', '.join(missing_fail_fixtures)}")

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in pass_paths:
        result = validate_contract_path(path)
        results.append(result)
        if result["status"] != "pass":
            failures.append(f"{path.name} expected pass: {result['problems']}")
    for path in fail_paths:
        result = validate_contract_path(path)
        results.append(result)
        if result["status"] != "fail":
            failures.append(f"{path.name} expected fail but passed")
            continue
        marker = EXPECTED_FAIL_MARKERS.get(path.name)
        if marker and not any(marker in problem for problem in result["problems"]):
            failures.append(f"{path.name} did not fail with marker {marker!r}: {result['problems']}")
    if failures:
        raise ContractError("\n".join(failures))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "pass_fixtures": [path.name for path in pass_paths],
        "fail_fixtures": [path.name for path in fail_paths],
        "results": results,
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    load_schema()
    if not args.contract:
        raise ContractError("at least one --contract is required")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    results = [validate_contract_path(path) for path in args.contract]
    failures = [result for result in results if result["status"] != "pass"]
    if failures:
        raise ContractError(json.dumps(failures, indent=2, sort_keys=True))
    dirty_files = git_output(["status", "--short"]).splitlines()
    pass_line = f"{PASS_LINE}: {out}"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "gate": "model_onboarding_contract",
        "contracts": results,
        "pass_line": pass_line,
    }
    write_json(out / "model_onboarding_contract_summary.json", summary)
    write_json(
        out / "gate.manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "goal": "release-regression-hardening-2026-06-28",
            "phase": "model_onboarding_contract",
            "status": "pass",
            "repo_root": str(REPO_ROOT),
            "git_sha": git_output(["rev-parse", "HEAD"]),
            "git_branch": git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
            "git_dirty": bool(dirty_files),
            "dirty_files": dirty_files,
            "command": sys.argv,
            "artifact_dir": str(out),
            "pass_line": pass_line,
            "outputs": {"summary": str(out / "model_onboarding_contract_summary.json")},
            "validation_summary": summary,
        },
    )
    (out / "pass_line.txt").write_text(pass_line + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--contract", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            summary = run_selftest(args.fixtures)
            if args.out:
                args.out.mkdir(parents=True, exist_ok=True)
                write_json(args.out / "model_onboarding_contract_selftest.json", summary)
            print(SELFTEST_PASS_LINE)
            return 0
        if args.out is None:
            raise ContractError("--out is required unless --self-test is set")
        summary = run_gate(args)
        print(summary["pass_line"])
        return 0
    except ContractError as exc:
        print(f"MODEL ONBOARDING CONTRACT FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
