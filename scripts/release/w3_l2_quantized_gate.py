#!/usr/bin/env python3
"""Build a W3 L2 real-size quantized semantic correctness artifact.

This gate intentionally does not run model inference by itself. It validates a
known-answer report produced by a real W3 quantized product-path run and
packages it into the final release-grade L2 schema. HF metadata, fixture-only
probes, toy models, and waived lanes are rejected instead of being converted
into L2 PASS evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACT_NAME = "w3_l2_quantized.json"
PASS_LINE_PREFIX = "W3 L2 QUANTIZED PASS"
REQUIRED_ENTRYPOINTS = {
    "ferrum run": "run",
    "ferrum serve": "serve",
}
REQUIRED_CASE_ENTRYPOINTS = set(REQUIRED_ENTRYPOINTS)
FORBIDDEN_OUTPUT_PATTERNS = [
    "<unk>",
    "[PAD",
    "<pad>",
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|reserved_special_token",
    "\ufffd",
    "KV cache overflow",
    "panicked at",
    "panic:",
    "malformed UTF-8",
    "stream error",
]


class GateError(Exception):
    pass


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GateError(f"missing JSON report: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GateError(f"invalid JSON report {path}: {exc}") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GateError(f"{label} must be a JSON object")
    return value


def as_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise GateError(f"{label} must be a JSON array")
    return value


def non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GateError(f"{label} must be a non-empty string")
    return value


def bool_value(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise GateError(f"{label} must be boolean")
    return value


def int_value(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateError(f"{label} must be an integer")
    return value


def read_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise GateError(f"{label} missing: {path}") from exc


def report_section(report: dict[str, Any]) -> dict[str, Any]:
    quantized = report.get("quantized_semantics")
    if isinstance(quantized, dict):
        merged = dict(report)
        merged.update(quantized)
        return merged
    return report


def command_parts(raw: Any, label: str) -> list[str]:
    if isinstance(raw, str):
        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            raise GateError(f"{label} is not a valid shell command: {exc}") from exc
    elif isinstance(raw, list) and all(isinstance(part, str) for part in raw):
        parts = raw
    elif isinstance(raw, dict):
        if "command_line" in raw:
            return command_parts(raw["command_line"], f"{label}.command_line")
        if isinstance(raw.get("command"), str):
            return command_parts(raw["command"], f"{label}.command")
        raise GateError(f"{label} must include command_line or command")
    else:
        raise GateError(f"{label} must be a command string, string list, or command object")
    if not parts:
        raise GateError(f"{label} must not be empty")
    return parts


def ferrum_subcommands(parts: list[str], label: str) -> set[str]:
    if not any(part == "ferrum" or part.endswith("/ferrum") for part in parts):
        raise GateError(f"{label} must invoke the ferrum binary")
    for part in parts:
        if re.match(r"^FERRUM_[A-Z0-9_]+=", part):
            raise GateError(f"{label} uses hidden env override: {part.split('=', 1)[0]}")
    found = {subcommand for subcommand in REQUIRED_ENTRYPOINTS.values() if subcommand in parts}
    if not found:
        raise GateError(f"{label} must invoke ferrum run or ferrum serve")
    if len(found) > 1:
        raise GateError(f"{label} ambiguously contains both ferrum run and ferrum serve")
    return found


def validate_product_command(raw: Any, label: str) -> dict[str, Any]:
    parts = command_parts(raw, label)
    subcommand = next(iter(ferrum_subcommands(parts, label)))
    entrypoint = f"ferrum {subcommand}"
    if isinstance(raw, dict) and isinstance(raw.get("entrypoint"), str):
        declared = raw["entrypoint"].strip()
        if declared != entrypoint:
            raise GateError(f"{label}.entrypoint {declared!r} does not match command {entrypoint!r}")
    return {
        "entrypoint": entrypoint,
        "command_line": parts,
    }


def product_commands(report: dict[str, Any]) -> list[Any]:
    commands: list[Any] = []
    for key in ["commands", "product_commands", "command_log"]:
        value = report.get(key)
        if isinstance(value, list):
            commands.extend(value)
    return commands


def validate_product_commands(report: dict[str, Any]) -> list[dict[str, Any]]:
    commands = product_commands(report)
    if not commands:
        raise GateError("W3 L2 report must include real product command_line evidence")
    normalized: list[dict[str, Any]] = []
    detected: set[str] = set()
    for idx, command in enumerate(commands):
        normalized_command = validate_product_command(command, f"commands[{idx}]")
        normalized.append(normalized_command)
        detected.add(normalized_command["entrypoint"])
    missing_entrypoints = sorted(set(REQUIRED_ENTRYPOINTS) - detected)
    if missing_entrypoints:
        raise GateError(
            "W3 L2 report must include product commands for both ferrum run and "
            f"ferrum serve; missing {missing_entrypoints}"
        )
    return normalized


def case_passed(case: Any) -> bool:
    if isinstance(case, str):
        return case.lower() in {"pass", "passed", "ok"}
    if not isinstance(case, dict):
        return False
    if "passed" in case:
        return case["passed"] is True
    if "semantic_pass" in case:
        return case["semantic_pass"] is True
    if "status" in case:
        return case["status"] in {"pass", "passed", "ok"}
    return False


def assert_no_forbidden_output(label: str, text: str) -> None:
    for pattern in FORBIDDEN_OUTPUT_PATTERNS:
        if pattern in text:
            raise GateError(f"{label} contains forbidden output pattern {pattern!r}")


def case_output_text(case: dict[str, Any], label: str) -> str:
    for key in ["content", "output", "text", "response"]:
        value = case.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise GateError(f"{label} must include non-empty output content")


def case_entrypoint(case: dict[str, Any], label: str) -> str:
    entrypoint = non_empty_string(case.get("entrypoint"), f"{label}.entrypoint")
    if entrypoint not in REQUIRED_CASE_ENTRYPOINTS:
        raise GateError(
            f"{label}.entrypoint must be one of {sorted(REQUIRED_CASE_ENTRYPOINTS)}, "
            f"got {entrypoint!r}"
        )
    return entrypoint


def resolve_artifact(raw: Any, report_dir: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise GateError(f"{label}.artifact must be a non-empty string")
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [report_dir / path, report_dir.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise GateError(f"{label}.artifact missing: {raw} (checked {checked})")


def validate_known_answer_hygiene(cases: list[Any], report_dir: Path) -> dict[str, Any]:
    if not cases:
        raise GateError("known_answer_cases must be present for W3 L2 output hygiene")
    checked_artifacts = 0
    seen_entrypoints: set[str] = set()
    for idx, raw_case in enumerate(cases):
        label = f"known_answer_cases[{idx}]"
        case = as_object(raw_case, label)
        seen_entrypoints.add(case_entrypoint(case, label))
        text = case_output_text(case, label)
        assert_no_forbidden_output(f"{label}.content", text)
        artifact = resolve_artifact(case.get("artifact"), report_dir, label)
        artifact_text = read_text(artifact, f"{label}.artifact")
        assert_no_forbidden_output(f"{label}.artifact", artifact_text)
        checked_artifacts += 1
        finish_reason = case.get("finish_reason")
        if finish_reason is not None and finish_reason not in {"stop", "length"}:
            raise GateError(f"{label}.finish_reason must be stop or length, got {finish_reason!r}")
    missing_entrypoints = sorted(REQUIRED_CASE_ENTRYPOINTS - seen_entrypoints)
    if missing_entrypoints:
        raise GateError(
            "known_answer_cases must include case-level entrypoint coverage for "
            f"{missing_entrypoints}"
        )
    return {
        "known_answer_cases_checked": len(cases),
        "response_artifacts_checked": checked_artifacts,
        "case_entrypoints": sorted(seen_entrypoints),
        "content_non_empty": True,
        "forbidden_patterns_absent": True,
        "artifact_text_scanned": True,
    }


def known_answer_counts(source: dict[str, Any]) -> tuple[int, int, list[Any]]:
    for key in ["known_answer_cases", "cases", "known_answers"]:
        if key in source:
            cases = as_list(source[key], key)
            total = len(cases)
            passed = sum(1 for case in cases if case_passed(case))
            return total, passed, cases

    total = int_value(source.get("known_answer_total"), "known_answer_total")
    passed = int_value(source.get("known_answer_passed"), "known_answer_passed")
    return total, passed, []


def validate_report(
    report: dict[str, Any],
    *,
    model_id_override: str | None,
    format_override: str | None,
    report_dir: Path,
) -> dict[str, Any]:
    source = report_section(report)
    model_id = model_id_override or non_empty_string(source.get("model_id"), "model_id")
    quantized_format = format_override or non_empty_string(source.get("format"), "format")

    real_size_model = bool_value(source.get("real_size_model"), "real_size_model")
    if not real_size_model:
        raise GateError("real_size_model must be true; toy/fixture reports cannot pass W3 L2")

    waived = bool_value(source.get("waived", False), "waived")
    if waived:
        raise GateError("waived quantized lanes cannot pass W3 L2")

    semantic_pass = bool_value(source.get("semantic_pass"), "semantic_pass")
    known_answer_total, known_answer_passed, cases = known_answer_counts(source)
    if known_answer_total < 10:
        raise GateError(f"known_answer_total must be >= 10, got {known_answer_total}")
    if known_answer_passed != known_answer_total:
        raise GateError(
            f"known_answer_passed must equal known_answer_total, got "
            f"{known_answer_passed}/{known_answer_total}"
        )
    if not semantic_pass:
        raise GateError("semantic_pass must be true")

    surface = source.get("product_surface", source.get("runtime_surface", "typed_cli"))
    if surface not in {"typed_cli", "typed_config", "typed_defaults", "model_defaults"}:
        raise GateError(f"product_surface must be typed product behavior, got {surface!r}")
    hidden_env = source.get("hidden_env", [])
    if hidden_env != []:
        raise GateError("hidden_env must be empty for W3 L2 release-grade evidence")

    commands = validate_product_commands(source)

    if cases:
        failed = [case for case in cases if not case_passed(case)]
        if failed:
            raise GateError(f"known-answer cases include failures: {failed[:3]}")
    hygiene = validate_known_answer_hygiene(cases, report_dir)

    return {
        "model_id": model_id,
        "format": quantized_format,
        "product_surface": surface,
        "known_answer_total": known_answer_total,
        "known_answer_passed": known_answer_passed,
        "commands": commands,
        "output_hygiene": hygiene,
    }


def build_artifact(
    *,
    report_path: Path,
    out_dir: Path,
    model_id_override: str | None,
    format_override: str | None,
) -> dict[str, Any]:
    report = as_object(load_json(report_path), "known-answer report")
    validated = validate_report(
        report,
        model_id_override=model_id_override,
        format_override=format_override,
        report_dir=report_path.parent,
    )
    source_ref = str(report_path)
    artifact = {
        "schema_version": 1,
        "status": "pass",
        "level": "l2_quantized",
        "model_id": validated["model_id"],
        "product_surface": validated["product_surface"],
        "hidden_env": [],
        "generated_at": iso_now(),
        "pass_line": f"{PASS_LINE_PREFIX}: {out_dir}",
        "quantized_semantics": {
            "real_size_model": True,
            "waived": False,
            "semantic_pass": True,
            "known_answer_total": validated["known_answer_total"],
            "known_answer_passed": validated["known_answer_passed"],
            "format": validated["format"],
            "source_report": source_ref,
        },
        "output_hygiene": validated["output_hygiene"],
        "product_entrypoints": sorted(REQUIRED_ENTRYPOINTS),
        "commands": validated["commands"],
    }
    write_json(out_dir / ARTIFACT_NAME, artifact)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="real known-answer report JSON")
    parser.add_argument("--out", type=Path, help="artifact output directory")
    parser.add_argument("--model-id", help="override model_id recorded in artifact")
    parser.add_argument("--format", help="override quantized format recorded in artifact")
    parser.add_argument("--self-test", action="store_true", help="run synthetic script self-test")
    return parser.parse_args()


def selftest_report(root: Path) -> Path:
    cases = []
    for index in range(10):
        artifact = root / "responses" / f"known_answer_{index:02d}.json"
        write_json(artifact, {"choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}]})
        cases.append(
            {
                "id": f"known_answer_{index:02d}",
                "entrypoint": "ferrum run" if index == 0 else "ferrum serve",
                "passed": True,
                "semantic_pass": True,
                "content": "Paris",
                "finish_reason": "stop",
                "artifact": artifact.relative_to(root).as_posix(),
            }
        )
    report = {
        "model_id": "selftest/Qwen3.5-35B-A3B-GPTQ-Int4",
        "format": "hf-gptq-int4",
        "real_size_model": True,
        "waived": False,
        "semantic_pass": True,
        "known_answer_cases": cases,
        "product_surface": "typed_cli",
        "hidden_env": [],
        "commands": [
            {
                "entrypoint": "ferrum run",
                "command_line": [
                    "ferrum",
                    "run",
                    "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
                    "--backend",
                    "cuda",
                ],
            },
            {
                "entrypoint": "ferrum serve",
                "command_line": [
                    "ferrum",
                    "serve",
                    "--model",
                    "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
                    "--backend",
                    "cuda",
                ],
            },
        ],
    }
    path = root / "known_answer_report.json"
    write_json(path, report)
    return path


def run_selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="ferrum-w3-l2-quantized-") as tmp:
        root = Path(tmp)
        report = selftest_report(root)
        artifact = build_artifact(
            report_path=report,
            out_dir=root / "out",
            model_id_override=None,
            format_override=None,
        )
        if artifact["quantized_semantics"]["known_answer_total"] != 10:
            raise AssertionError("selftest artifact did not preserve known-answer total")
        if {command["entrypoint"] for command in artifact["commands"]} != set(REQUIRED_ENTRYPOINTS):
            raise AssertionError("selftest artifact did not preserve required product commands")
        if artifact["output_hygiene"]["response_artifacts_checked"] != 10:
            raise AssertionError("selftest artifact did not scan known-answer artifacts")
        if set(artifact["output_hygiene"]["case_entrypoints"]) != REQUIRED_CASE_ENTRYPOINTS:
            raise AssertionError("selftest artifact did not preserve case entrypoint coverage")

        bad = as_object(load_json(report), "selftest report")
        bad["known_answer_cases"][0]["passed"] = False
        bad_path = root / "bad_report.json"
        write_json(bad_path, bad)
        try:
            build_artifact(
                report_path=bad_path,
                out_dir=root / "bad_out",
                model_id_override=None,
                format_override=None,
            )
        except GateError as exc:
            if "known_answer_passed must equal known_answer_total" not in str(exc):
                raise AssertionError(f"unexpected failed-case error: {exc}") from exc
        else:
            raise AssertionError("bad known-answer report unexpectedly passed")

        toy = as_object(load_json(report), "selftest report")
        toy["real_size_model"] = False
        toy_path = root / "toy_report.json"
        write_json(toy_path, toy)
        try:
            build_artifact(
                report_path=toy_path,
                out_dir=root / "toy_out",
                model_id_override=None,
                format_override=None,
            )
        except GateError as exc:
            if "real_size_model must be true" not in str(exc):
                raise AssertionError(f"unexpected toy-report error: {exc}") from exc
        else:
            raise AssertionError("toy model report unexpectedly passed")

        entrypoint_only = as_object(load_json(report), "selftest report")
        entrypoint_only["commands"] = [
            {"entrypoint": "ferrum run"},
            {"entrypoint": "ferrum serve"},
        ]
        entrypoint_only_path = root / "entrypoint_only_report.json"
        write_json(entrypoint_only_path, entrypoint_only)
        try:
            build_artifact(
                report_path=entrypoint_only_path,
                out_dir=root / "entrypoint_only_out",
                model_id_override=None,
                format_override=None,
            )
        except GateError as exc:
            if "must include command_line or command" not in str(exc):
                raise AssertionError(f"unexpected entrypoint-only error: {exc}") from exc
        else:
            raise AssertionError("entrypoint-only product evidence unexpectedly passed")

        hidden_env = as_object(load_json(report), "selftest report")
        hidden_env["commands"][0]["command_line"] = [
            "FERRUM_FORCE_FAST_PATH=1",
            "ferrum",
            "run",
            "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
            "--backend",
            "cuda",
        ]
        hidden_env_path = root / "hidden_env_command_report.json"
        write_json(hidden_env_path, hidden_env)
        try:
            build_artifact(
                report_path=hidden_env_path,
                out_dir=root / "hidden_env_out",
                model_id_override=None,
                format_override=None,
            )
        except GateError as exc:
            if "hidden env override" not in str(exc):
                raise AssertionError(f"unexpected hidden-env command error: {exc}") from exc
        else:
            raise AssertionError("hidden-env product command unexpectedly passed")

        serve_only_cases = as_object(load_json(report), "selftest report")
        for case in serve_only_cases["known_answer_cases"]:
            case["entrypoint"] = "ferrum serve"
        serve_only_cases_path = root / "serve_only_cases_report.json"
        write_json(serve_only_cases_path, serve_only_cases)
        try:
            build_artifact(
                report_path=serve_only_cases_path,
                out_dir=root / "serve_only_cases_out",
                model_id_override=None,
                format_override=None,
            )
        except GateError as exc:
            if "case-level entrypoint coverage" not in str(exc):
                raise AssertionError(f"unexpected serve-only-cases error: {exc}") from exc
        else:
            raise AssertionError("serve-only known-answer cases unexpectedly passed")

        bad_output = as_object(load_json(report), "selftest report")
        bad_output["known_answer_cases"][0]["content"] = "<unk>"
        bad_output_path = root / "bad_output_report.json"
        write_json(bad_output_path, bad_output)
        try:
            build_artifact(
                report_path=bad_output_path,
                out_dir=root / "bad_output_out",
                model_id_override=None,
                format_override=None,
            )
        except GateError as exc:
            if "forbidden output pattern" not in str(exc):
                raise AssertionError(f"unexpected bad-output error: {exc}") from exc
        else:
            raise AssertionError("bad-output known-answer report unexpectedly passed")

    print("W3 L2 QUANTIZED SELFTEST PASS")
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            return run_selftest()
        if args.report is None:
            raise GateError("missing required arg: --report")
        if args.out is None:
            raise GateError("missing required arg: --out")
        args.out.mkdir(parents=True, exist_ok=True)
        artifact = build_artifact(
            report_path=args.report,
            out_dir=args.out,
            model_id_override=args.model_id,
            format_override=args.format,
        )
    except GateError as exc:
        print(f"W3 L2 QUANTIZED FAIL: {exc}", file=sys.stderr)
        return 1
    print(artifact["pass_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
