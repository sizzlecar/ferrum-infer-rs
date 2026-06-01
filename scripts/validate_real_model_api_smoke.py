#!/usr/bin/env python3
"""Validate artifacts emitted by scripts/m3_real_model_api_smoke.sh."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any


REQUIRED_COMMANDS = [
    "test_openai_client_chat_basic",
    "test_openai_client_chat_usage_fields",
    "test_openai_client_chat_streaming",
    "test_openai_client_tools_stream_options_include_usage",
    "test_openai_client_response_format_json_object",
    "test_openai_client_strict_json_schema_20_runs",
    "test_openai_client_multi_turn",
    "python_openai_test",
]


class ValidationError(Exception):
    pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_keys(where: str, value: dict[str, Any], required: set[str]) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValidationError(f"{where}: missing keys: {', '.join(missing)}")


def require_string(where: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{where} must be a non-empty string")
    return value


def require_nonnegative_int(where: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{where} must be a non-negative integer")
    return value


def validate_command(
    root: Path,
    index: int,
    command: Any,
    *,
    require_passed: bool,
) -> dict[str, Any]:
    if not isinstance(command, dict):
        raise ValidationError(f"commands[{index}] must be an object")
    require_keys(
        f"commands[{index}]",
        command,
        {"name", "cmd", "rc", "elapsed_ms", "started_at_utc", "finished_at_utc"},
    )
    name = require_string(f"commands[{index}].name", command["name"])
    require_string(f"commands[{index}].cmd", command["cmd"])
    require_string(f"commands[{index}].started_at_utc", command["started_at_utc"])
    require_string(f"commands[{index}].finished_at_utc", command["finished_at_utc"])
    require_nonnegative_int(f"commands[{index}].elapsed_ms", command["elapsed_ms"])
    rc = require_nonnegative_int(f"commands[{index}].rc", command["rc"])
    if require_passed and rc != 0:
        raise ValidationError(f"commands[{index}] {name!r} did not pass: rc={rc}")

    log_path = root / f"cargo-test-{name}.log"
    if not log_path.is_file():
        raise ValidationError(f"missing command log: {log_path}")

    return {"name": name, "rc": rc, "log": str(log_path)}


def validate_artifact(
    root: Path,
    *,
    require_required_commands: bool = True,
    require_passed: bool = True,
) -> dict[str, Any]:
    if not root.is_dir():
        raise ValidationError(f"artifact root is not a directory: {root}")
    manifest = root / "commands.md"
    summary_path = root / "run_summary.json"
    if not manifest.is_file():
        raise ValidationError(f"missing commands.md: {manifest}")
    if not summary_path.is_file():
        raise ValidationError(f"missing run_summary.json: {summary_path}")

    summary = load_json(summary_path)
    if not isinstance(summary, dict):
        raise ValidationError("run_summary.json root must be an object")
    require_keys("summary", summary, {"script", "commands", "all_passed"})
    if summary["script"] != "m3_real_model_api_smoke.sh":
        raise ValidationError("summary.script must be m3_real_model_api_smoke.sh")
    if not isinstance(summary["all_passed"], bool):
        raise ValidationError("summary.all_passed must be boolean")
    if require_passed and summary["all_passed"] is not True:
        raise ValidationError("summary.all_passed is not true")
    commands = summary["commands"]
    if not isinstance(commands, list) or not commands:
        raise ValidationError("summary.commands must be a non-empty list")

    rows = [
        validate_command(root, index, command, require_passed=require_passed)
        for index, command in enumerate(commands)
    ]
    names = [row["name"] for row in rows]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValidationError(f"duplicate command names: {duplicate_names}")

    missing_required = [
        name for name in REQUIRED_COMMANDS if name not in set(names)
    ]
    if require_required_commands and missing_required:
        raise ValidationError(f"missing required smoke commands: {missing_required}")

    return {
        "ok": True,
        "root": str(root),
        "command_count": len(rows),
        "required_commands_present": not missing_required,
        "all_passed": summary["all_passed"],
        "commands": names,
    }


def write_artifact(root: Path, commands: list[dict[str, Any]], *, all_passed: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "commands.md").write_text("# Real-model API smoke execution\n", encoding="utf-8")
    summary = {
        "script": "m3_real_model_api_smoke.sh",
        "commands": commands,
        "all_passed": all_passed,
    }
    (root / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    for command in commands:
        (root / f"cargo-test-{command['name']}.log").write_text(
            f"{command['name']} log\n",
            encoding="utf-8",
        )


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "ok"
        commands = [
            {
                "name": name,
                "cmd": f"cargo test {name}",
                "rc": 0,
                "elapsed_ms": 1,
                "started_at_utc": "2026-06-01T00:00:00Z",
                "finished_at_utc": "2026-06-01T00:00:01Z",
            }
            for name in REQUIRED_COMMANDS
        ]
        write_artifact(root, commands, all_passed=True)
        result = validate_artifact(root)
        assert result["command_count"] == len(REQUIRED_COMMANDS)
        assert result["required_commands_present"] is True

        missing = Path(tmp) / "missing"
        write_artifact(missing, commands[:-1], all_passed=True)
        try:
            validate_artifact(missing)
        except ValidationError as exc:
            assert "missing required smoke commands" in str(exc)
        else:
            raise AssertionError("missing required command unexpectedly passed")

        failed = Path(tmp) / "failed"
        failed_commands = [dict(command) for command in commands]
        failed_commands[0]["rc"] = 1
        write_artifact(failed, failed_commands, all_passed=False)
        try:
            validate_artifact(failed)
        except ValidationError as exc:
            assert "all_passed" in str(exc)
        else:
            raise AssertionError("failed artifact unexpectedly passed")

        partial_result = validate_artifact(
            missing,
            require_required_commands=False,
            require_passed=True,
        )
        assert partial_result["required_commands_present"] is False

    print("validate_real_model_api_smoke self-test ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", nargs="?", help="smoke artifact root")
    parser.add_argument("--json", action="store_true", help="emit JSON summary")
    parser.add_argument("--self-test", action="store_true", help="run self-tests and exit")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="do not require the full completion smoke command set",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="validate shape without requiring all commands to pass",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.artifact:
        raise SystemExit("artifact root is required unless --self-test is used")

    try:
        result = validate_artifact(
            Path(args.artifact),
            require_required_commands=not args.allow_partial,
            require_passed=not args.allow_failures,
        )
    except ValidationError as exc:
        raise SystemExit(f"validation error: {exc}") from None
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "real-model api smoke artifact ok: "
            f"commands={result['command_count']} all_passed={result['all_passed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
