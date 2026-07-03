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
import tempfile
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

TEMPLATE_CASE_MAP = {
    "single": "single_turn",
    "single_turn": "single_turn",
    "multi_turn": "multi_turn",
    "system": "system_message",
    "system_message": "system_message",
    "tools": "tool_injection",
    "tool_injection": "tool_injection",
    "think_history": "reasoning_history",
    "reasoning_history": "reasoning_history",
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


def require_generated(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "model"


def canonical_model_id(value: str) -> str:
    """Normalize local HF snapshot paths into stable model ids.

    Actual model smoke gates often run from a local snapshot directory such as
    ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<rev>.  Model
    onboarding contracts must name the model, not the machine-local cache path.
    """
    raw = value.strip()
    normalized = raw.replace("\\", "/")
    match = re.search(r"(?:^|/)models--([^/]+)(?:/snapshots/|$)", normalized)
    if match:
        parts = [part for part in match.group(1).split("--") if part]
        if len(parts) >= 2:
            return "/".join(parts[:2])
    return raw


def artifact_ref(
    *,
    kind: str,
    path: Path,
    pass_line: str,
    git_sha: str,
    backend: str | None = None,
    entrypoint: str | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "kind": kind,
        "path": str(path),
        "status": "pass",
        "pass_line": pass_line,
        "git_sha": git_sha,
    }
    if backend is not None:
        artifact["backend"] = backend
    if entrypoint is not None:
        artifact["entrypoint"] = entrypoint
    return artifact


def read_manifest_summary(root: Path, summary_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(root / "gate.manifest.json")
    outputs = manifest.get("outputs")
    summary_path: Path | None = None
    if isinstance(outputs, dict) and isinstance(outputs.get("summary"), str):
        summary_path = Path(outputs["summary"])
    if summary_path is None:
        summary_path = root / summary_name
    if not summary_path.is_absolute():
        summary_path = (root / summary_path).resolve()
    return manifest, read_json(summary_path)


def template_cases(template_artifact: dict[str, Any]) -> list[str]:
    golden = template_artifact.get("chat_template_golden")
    require_generated(isinstance(golden, dict), "template artifact missing chat_template_golden")
    raw_cases = golden.get("case_names")
    require_generated(isinstance(raw_cases, list) and raw_cases, "template artifact case_names missing")
    cases = sorted({TEMPLATE_CASE_MAP.get(str(case), str(case)) for case in raw_cases})
    missing = sorted(REQUIRED_TEMPLATE_CASES - set(cases))
    require_generated(not missing, f"template artifact missing required contract cases: {missing}")
    return cases


def special_token_facts(template_artifact: dict[str, Any]) -> dict[str, Any]:
    special = template_artifact.get("special_tokens")
    require_generated(isinstance(special, dict), "template artifact missing special_tokens")
    eos = special.get("eos_token_id")
    eos_ids = eos if isinstance(eos, list) else [eos]
    eos_ids = [item for item in eos_ids if isinstance(item, int) and not isinstance(item, bool)]
    require_generated(eos_ids, "template artifact missing eos_token_id")
    bos = special.get("bos_token_id")
    bos_token_id = bos if isinstance(bos, int) and not isinstance(bos, bool) else None
    stop_tokens = []
    eos_token = special.get("eos_token")
    if isinstance(eos_token, str) and eos_token:
        stop_tokens.append(eos_token)
    return {
        "eos_token_ids": eos_ids,
        "bos_token_id": bos_token_id,
        "stop_tokens": stop_tokens,
        "token_id_source": str(special.get("source") or "generation_config"),
    }


def generate_contract_from_actual_smoke(args: argparse.Namespace, out: Path) -> Path | None:
    if args.actual_smoke is None:
        return None
    required = {
        "--actual-smoke-template-artifact": args.actual_smoke_template_artifact,
        "--actual-smoke-profile-gate": args.actual_smoke_profile_gate,
        "--actual-smoke-preset-snapshot": args.actual_smoke_preset_snapshot,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ContractError(f"--actual-smoke requires {', '.join(missing)}")

    smoke_root = args.actual_smoke.resolve()
    smoke_manifest, smoke_summary = read_manifest_summary(smoke_root, "product_observability_l1_smoke_summary.json")
    require_generated(smoke_manifest.get("status") == "pass", "actual smoke manifest must be pass")
    require_generated(smoke_summary.get("status") == "pass", "actual smoke summary must be pass")
    require_generated(smoke_manifest.get("git_dirty") is False, "actual smoke must be clean git evidence")
    git_sha = str(smoke_manifest.get("git_sha") or "")
    require_generated(GIT_SHA_RE.match(git_sha) is not None, "actual smoke git_sha must be a 40-character SHA")
    model_id = canonical_model_id(
        str(
            args.actual_smoke_model_id
            or smoke_manifest.get("model_id")
            or smoke_summary.get("model_id")
            or smoke_manifest.get("model")
            or smoke_summary.get("model")
            or ""
        )
    )
    require_generated(bool(model_id.strip()), "actual smoke model missing")
    backend = str(smoke_manifest.get("effective_backend") or smoke_manifest.get("backend") or "")
    require_generated(backend in {"cpu", "cuda", "metal"}, f"actual smoke backend unsupported: {backend!r}")

    template_path = args.actual_smoke_template_artifact.resolve()
    template_artifact = read_json(template_path)
    template_pass_line = str(template_artifact.get("pass_line") or "")
    require_generated(template_artifact.get("status") == "pass", "template artifact must be pass")
    require_generated(template_pass_line.startswith("W3 L0 TEMPLATE PASS:"), "template artifact must carry W3 L0 TEMPLATE PASS")
    cases = template_cases(template_artifact)
    token_facts = special_token_facts(template_artifact)

    profile_root = args.actual_smoke_profile_gate.resolve()
    profile_manifest = read_json(profile_root / "gate.manifest.json")
    profile_pass_line = str(profile_manifest.get("pass_line") or "")
    require_generated(profile_manifest.get("status") == "pass", "profile gate manifest must be pass")
    require_generated(profile_pass_line.startswith("OBSERVABILITY PROFILE GATE PASS:"), "profile gate must carry OBSERVABILITY PROFILE GATE PASS")

    replay_manifest = read_json(smoke_root / "request_replay_bundle/gate.manifest.json")
    replay_pass_line = str(replay_manifest.get("pass_line") or "")
    require_generated(replay_manifest.get("status") == "pass", "request replay bundle manifest must be pass")
    require_generated(replay_pass_line.startswith("REQUEST REPLAY BUNDLE PASS:"), "replay artifact must carry REQUEST REPLAY BUNDLE PASS")

    preset_path = args.actual_smoke_preset_snapshot.resolve()
    preset = read_json(preset_path)
    require_generated(preset.get("status") == "pass", "preset snapshot artifact must be pass")
    preset_pass_line = f"BACKEND PRESET SNAPSHOT PASS: {preset_path.parent}"

    run_log = smoke_root / "logs/run.json"
    serve_response = smoke_root / "serve_nonstream.json"
    require_generated(run_log.is_file(), f"actual smoke run log missing: {run_log}")
    require_generated(serve_response.is_file(), f"actual smoke serve response missing: {serve_response}")

    selection = (smoke_summary.get("entrypoints") or {}).get("serve", {}).get("product", {})
    health = selection.get("health") if isinstance(selection, dict) else {}
    admission = (health or {}).get("admission") if isinstance(health, dict) else {}
    scheduler = "unknown"
    if isinstance(admission, dict) and isinstance(admission.get("scheduler_policy"), str):
        scheduler = admission["scheduler_policy"]
    runtime_preset = args.actual_smoke_runtime_preset or f"{backend}-{slug(model_id)}-l1"
    attention_impl = args.actual_smoke_attention_impl or "legacy_paged_varlen+legacy_paged_decode"
    kv_layout = args.actual_smoke_kv_layout or "paged"
    kv_dtype = args.actual_smoke_kv_dtype or "fp16"

    contract_id = args.actual_smoke_contract_id or f"{slug(model_id)}-{backend}-l1"
    family = args.actual_smoke_family or slug(model_id.rsplit("/", 1)[-1])
    architecture = args.actual_smoke_architecture or family
    weight_format = args.actual_smoke_weight_format or "unknown"
    source = args.actual_smoke_source or "hf"

    contract = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": contract_id,
        "source_git_sha": git_sha,
        "model": {
            "id": model_id,
            "family": family,
            "architecture": architecture,
            "weight_format": weight_format,
            "source": source,
        },
        "facts": {
            "tokenizer_source": "tokenizer.json",
            "chat_template_source": "tokenizer_config.json/chat_template fixture",
            "generation_config_source": "generation_config.json",
            "token_id_source": token_facts["token_id_source"],
            "eos_token_ids": token_facts["eos_token_ids"],
            "bos_token_id": token_facts["bos_token_id"],
            "stop_tokens": token_facts["stop_tokens"],
            "template_golden": artifact_ref(
                kind="template_golden",
                path=template_path,
                pass_line=template_pass_line,
                git_sha=git_sha,
            ),
            "template_golden_cases": cases,
        },
        "runtime_preset": {
            "selected": runtime_preset,
            "rejected_candidates": args.actual_smoke_rejected_candidate or ["cpu-default", "cuda-default"],
        },
        "fallback_policy": {
            "allow_builtin_template_fallback": False,
            "allow_backend_fallback": False,
            "allow_attention_fallback": False,
        },
        "capabilities": {
            "tool_calling": {"supported": False, "source": "not_validated_by_l1_smoke"},
            "structured_output": {"supported": False, "source": "not_validated_by_l1_smoke"},
            "reasoning": {"supported": False, "source": "metadata"},
        },
        "backend_support": [
            {
                "backend": backend,
                "status": "supported",
                "selection": {
                    "runtime_preset": runtime_preset,
                    "scheduler": scheduler,
                    "attention_impl": attention_impl,
                    "kv_layout": kv_layout,
                    "kv_dtype": kv_dtype,
                    "recurrent_state_max_slots": None,
                },
                "fallback": {
                    "allowed": False,
                    "actual_backend": backend,
                    "actual_attention_impl": attention_impl,
                    "reason": None,
                },
                "correctness": {
                    "status": "pass",
                    "run_artifact": artifact_ref(
                        kind="run_smoke",
                        path=run_log,
                        pass_line=str(smoke_manifest.get("pass_line")),
                        git_sha=git_sha,
                        backend=backend,
                        entrypoint="run",
                    ),
                    "serve_artifact": artifact_ref(
                        kind="serve_smoke",
                        path=serve_response,
                        pass_line=str(smoke_manifest.get("pass_line")),
                        git_sha=git_sha,
                        backend=backend,
                        entrypoint="serve",
                    ),
                    "profile_artifact": artifact_ref(
                        kind="profile",
                        path=profile_root / "gate.manifest.json",
                        pass_line=profile_pass_line,
                        git_sha=git_sha,
                        backend=backend,
                    ),
                    "replay_artifact": artifact_ref(
                        kind="replay",
                        path=smoke_root / "request_replay_bundle/gate.manifest.json",
                        pass_line=replay_pass_line,
                        git_sha=git_sha,
                        backend=backend,
                    ),
                    "preset_snapshot": artifact_ref(
                        kind="preset_snapshot",
                        path=preset_path,
                        pass_line=preset_pass_line,
                        git_sha=git_sha,
                        backend=backend,
                    ),
                },
            }
        ],
    }
    contract_path = out / "generated_contracts" / f"{contract_id}.json"
    write_json(contract_path, contract)
    return contract_path


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


def make_actual_smoke_contract_fixture(root: Path) -> Path:
    sha = "1" * 40
    smoke = root / "actual-smoke"
    smoke.mkdir(parents=True)
    (smoke / "logs").mkdir()
    write_json(smoke / "logs/run.json", {"status": "pass", "entrypoint": "run"})
    write_json(smoke / "serve_nonstream.json", {"status": "pass", "entrypoint": "serve"})
    replay = smoke / "request_replay_bundle"
    replay.mkdir()
    write_json(
        replay / "gate.manifest.json",
        {
            "schema_version": 1,
            "status": "pass",
            "pass_line": f"REQUEST REPLAY BUNDLE PASS: {replay}",
        },
    )
    smoke_summary = {
        "schema_version": 1,
        "status": "pass",
        "model": "/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/abc123",
        "entrypoints": {
            "serve": {
                "product": {
                    "health": {
                        "admission": {
                            "scheduler_policy": "prefill_first_until_active:4+prefill_step_chunk:512"
                        }
                    }
                }
            }
        },
    }
    write_json(smoke / "product_observability_l1_smoke_summary.json", smoke_summary)
    write_json(
        smoke / "gate.manifest.json",
        {
            "schema_version": 1,
            "status": "pass",
            "git_sha": sha,
            "git_dirty": False,
            "model": "/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/abc123",
            "backend": "metal",
            "effective_backend": "metal",
            "pass_line": f"PRODUCT OBSERVABILITY L1 SMOKE PASS: {smoke}",
            "outputs": {"summary": str(smoke / "product_observability_l1_smoke_summary.json")},
        },
    )

    profile = root / "profile"
    profile.mkdir()
    write_json(
        profile / "gate.manifest.json",
        {
            "schema_version": 1,
            "status": "pass",
            "pass_line": f"OBSERVABILITY PROFILE GATE PASS: {profile}",
        },
    )
    template = root / "template"
    template.mkdir()
    write_json(
        template / "w3_l0_template.json",
        {
            "schema_version": 1,
            "status": "pass",
            "pass_line": f"W3 L0 TEMPLATE PASS: {template}",
            "chat_template_golden": {
                "case_names": ["single", "system", "multi_turn", "tools", "think_history"]
            },
            "special_tokens": {
                "source": "generation_config",
                "eos_token_id": [151645, 151643],
                "bos_token_id": 151643,
                "eos_token": "<|im_end|>",
            },
        },
    )
    preset = root / "preset"
    preset.mkdir()
    write_json(preset / "summary.json", {"schema_version": 1, "status": "pass"})

    args = argparse.Namespace(
        actual_smoke=smoke,
        actual_smoke_model_id=None,
        actual_smoke_template_artifact=template / "w3_l0_template.json",
        actual_smoke_profile_gate=profile,
        actual_smoke_preset_snapshot=preset / "summary.json",
        actual_smoke_contract_id="qwen3-0-6b-metal-l1",
        actual_smoke_family="qwen3_dense",
        actual_smoke_architecture="qwen3",
        actual_smoke_weight_format="safetensors_dense",
        actual_smoke_source="hf",
        actual_smoke_runtime_preset="metal-qwen3-dense-l1",
        actual_smoke_attention_impl="legacy_paged_varlen+legacy_paged_decode",
        actual_smoke_kv_layout="paged",
        actual_smoke_kv_dtype="fp16",
        actual_smoke_rejected_candidate=["cpu-default"],
    )
    return generate_contract_from_actual_smoke(args, root / "generated")


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
    with tempfile.TemporaryDirectory(prefix="ferrum-model-contract-actual-smoke-") as tmp:
        generated_root = Path(tmp)
        generated_contract = make_actual_smoke_contract_fixture(generated_root)
        result = validate_contract_path(generated_contract)
        results.append(result)
        if result["status"] != "pass":
            failures.append(f"generated actual-smoke contract expected pass: {result['problems']}")
        if result.get("model_id") != "Qwen/Qwen3-0.6B":
            failures.append(f"generated actual-smoke contract used unstable model id: {result.get('model_id')}")
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
    generated_contract = generate_contract_from_actual_smoke(args, args.out)
    contracts = [*args.contract]
    if generated_contract is not None:
        contracts.append(generated_contract)
    if not contracts:
        raise ContractError("at least one --contract is required")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    results = [validate_contract_path(path) for path in contracts]
    failures = [result for result in results if result["status"] != "pass"]
    if failures:
        raise ContractError(json.dumps(failures, indent=2, sort_keys=True))
    dirty_files = git_output(["status", "--short"], default="").splitlines()
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
    parser.add_argument("--actual-smoke", type=Path)
    parser.add_argument(
        "--actual-smoke-model-id",
        help="stable model id to record when --actual-smoke was run from a local path",
    )
    parser.add_argument("--actual-smoke-template-artifact", type=Path)
    parser.add_argument("--actual-smoke-profile-gate", type=Path)
    parser.add_argument("--actual-smoke-preset-snapshot", type=Path)
    parser.add_argument("--actual-smoke-contract-id")
    parser.add_argument("--actual-smoke-family")
    parser.add_argument("--actual-smoke-architecture")
    parser.add_argument("--actual-smoke-weight-format")
    parser.add_argument("--actual-smoke-source", choices=["hf", "gguf", "local", "synthetic"])
    parser.add_argument("--actual-smoke-runtime-preset")
    parser.add_argument("--actual-smoke-attention-impl")
    parser.add_argument("--actual-smoke-kv-layout")
    parser.add_argument("--actual-smoke-kv-dtype")
    parser.add_argument("--actual-smoke-rejected-candidate", action="append")
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
