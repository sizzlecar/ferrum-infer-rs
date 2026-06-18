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
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACT_NAME = "w3_l2_quantized.json"
PASS_LINE_PREFIX = "W3 L2 QUANTIZED PASS"
REQUIRED_ENTRYPOINTS = {"ferrum run", "ferrum serve"}


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


def report_section(report: dict[str, Any]) -> dict[str, Any]:
    quantized = report.get("quantized_semantics")
    if isinstance(quantized, dict):
        merged = dict(report)
        merged.update(quantized)
        return merged
    return report


def normalize_command_text(command: Any) -> str:
    if isinstance(command, str):
        return command
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    if isinstance(command, dict):
        if isinstance(command.get("command_line"), list):
            return " ".join(str(part) for part in command["command_line"])
        if isinstance(command.get("command"), str):
            return command["command"]
        if isinstance(command.get("entrypoint"), str):
            return command["entrypoint"]
    return ""


def normalize_entrypoint(command: Any) -> str | None:
    if isinstance(command, dict) and isinstance(command.get("entrypoint"), str):
        entrypoint = command["entrypoint"].strip()
        if entrypoint in REQUIRED_ENTRYPOINTS:
            return entrypoint
    text = normalize_command_text(command)
    if not text:
        return None
    if "ferrum serve" in text or " serve " in f" {text} ":
        return "ferrum serve"
    if "ferrum run" in text or " run " in f" {text} ":
        return "ferrum run"
    return None


def product_entrypoints(report: dict[str, Any]) -> tuple[set[str], list[Any]]:
    commands: list[Any] = []
    for key in ["commands", "product_commands", "command_log"]:
        value = report.get(key)
        if isinstance(value, list):
            commands.extend(value)
    entrypoints = report.get("product_entrypoints")
    if isinstance(entrypoints, list):
        commands.extend({"entrypoint": item} for item in entrypoints)
    detected = {entrypoint for command in commands if (entrypoint := normalize_entrypoint(command))}
    return detected, commands


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

    detected_entrypoints, commands = product_entrypoints(source)
    missing_entrypoints = sorted(REQUIRED_ENTRYPOINTS - detected_entrypoints)
    if missing_entrypoints:
        raise GateError(
            "W3 L2 report must include product commands for both ferrum run and "
            f"ferrum serve; missing {missing_entrypoints}"
        )

    if cases:
        failed = [case for case in cases if not case_passed(case)]
        if failed:
            raise GateError(f"known-answer cases include failures: {failed[:3]}")

    return {
        "model_id": model_id,
        "format": quantized_format,
        "product_surface": surface,
        "known_answer_total": known_answer_total,
        "known_answer_passed": known_answer_passed,
        "commands": commands,
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
    cases = [
        {"id": f"known_answer_{index:02d}", "passed": True, "semantic_pass": True}
        for index in range(10)
    ]
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
