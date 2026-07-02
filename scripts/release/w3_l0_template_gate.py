#!/usr/bin/env python3
"""Build a W3 L0 chat-template release-grade artifact.

The gate consumes HF-generated golden fixtures and verifies that Ferrum's
renderer still matches them byte-for-byte through the existing Rust golden
test. It also records model special-token provenance from generation_config so
the final W3 release-grade gate can reject shell "status=pass" evidence.
When explicit sidecar paths are omitted, the gate resolves
generation_config.json, tokenizer_config.json, tokenizer_special_tokens.json,
and tokenizer.json from the fixture directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REQUIRED_CASES = ["single", "system", "multi_turn", "tools", "think_history"]
ARTIFACT_NAME = "w3_l0_template.json"


class GateError(Exception):
    pass


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GateError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GateError(f"invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise GateError(f"{label} missing: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise GateError(f"{label} missing: {path}")
    return path


def optional_file(path: Path, label: str) -> Path | None:
    return path if path.is_file() else None


def resolve_fixture_sidecars(
    *,
    fixture_dir: Path,
    generation_config: Path | None,
    tokenizer_config: Path | None,
    tokenizer_special_tokens: Path | None,
    tokenizer_json: Path | None,
) -> tuple[Path, Path | None, Path | None, Path | None]:
    fixture_dir = require_dir(fixture_dir, "fixture dir")
    resolved_generation = generation_config or optional_file(
        fixture_dir / "generation_config.json",
        "fixture generation_config.json",
    )
    if resolved_generation is None:
        raise GateError(
            "generation_config.json missing: pass --generation-config or generate it into the fixture dir"
        )
    require_file(resolved_generation, "generation_config.json")

    resolved_tokenizer_config = tokenizer_config or optional_file(
        fixture_dir / "tokenizer_config.json",
        "fixture tokenizer_config.json",
    )
    if resolved_tokenizer_config is not None:
        require_file(resolved_tokenizer_config, "tokenizer_config.json")

    resolved_tokenizer_special_tokens = tokenizer_special_tokens or optional_file(
        fixture_dir / "tokenizer_special_tokens.json",
        "fixture tokenizer_special_tokens.json",
    )
    if resolved_tokenizer_special_tokens is not None:
        require_file(resolved_tokenizer_special_tokens, "tokenizer_special_tokens.json")

    resolved_tokenizer_json = tokenizer_json or optional_file(
        fixture_dir / "tokenizer.json",
        "fixture tokenizer.json",
    )
    if resolved_tokenizer_json is not None:
        require_file(resolved_tokenizer_json, "tokenizer.json")

    return (
        resolved_generation,
        resolved_tokenizer_config,
        resolved_tokenizer_special_tokens,
        resolved_tokenizer_json,
    )


def as_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GateError(f"{label} must be a JSON object")
    return value


def parse_token_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return None


def normalize_id_list(value: Any, label: str, *, required: bool) -> list[int]:
    if value is None:
        if required:
            raise GateError(f"{label} missing from generation_config.json")
        return []
    values = value if isinstance(value, list) else [value]
    out: list[int] = []
    for raw in values:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise GateError(f"{label} must contain non-negative integer token ids")
        out.append(raw)
    if required and not out:
        raise GateError(f"{label} must contain at least one token id")
    return out


def tokenizer_vocab(tokenizer_json: Path | None) -> dict[str, int]:
    if tokenizer_json is None:
        return {}
    value = load_json(require_file(tokenizer_json, "tokenizer.json"))
    obj = as_object(value, "tokenizer.json")
    vocab: dict[str, int] = {}
    model = obj.get("model")
    if isinstance(model, dict) and isinstance(model.get("vocab"), dict):
        for token, raw_id in model["vocab"].items():
            if isinstance(token, str) and isinstance(raw_id, int) and not isinstance(raw_id, bool):
                vocab[token] = raw_id
    added = obj.get("added_tokens")
    if isinstance(added, list):
        for item in added:
            if not isinstance(item, dict):
                continue
            token = parse_token_value(item.get("content"))
            raw_id = item.get("id")
            if token is not None and isinstance(raw_id, int) and not isinstance(raw_id, bool):
                vocab[token] = raw_id
    return vocab


def merge_tokenizer_vocab(
    tokenizer_special_tokens: Path | None,
    tokenizer_json: Path | None,
) -> tuple[dict[str, int], dict[str, str]]:
    vocab = tokenizer_special_token_vocab(tokenizer_special_tokens)
    sources = {token: "tokenizer_special_tokens" for token in vocab}
    json_vocab = tokenizer_vocab(tokenizer_json)
    for token, token_id in json_vocab.items():
        vocab[token] = token_id
        sources[token] = "tokenizer_json"
    return vocab, sources


def tokenizer_special_token_vocab(tokenizer_special_tokens: Path | None) -> dict[str, int]:
    if tokenizer_special_tokens is None:
        return {}
    value = as_object(
        load_json(require_file(tokenizer_special_tokens, "tokenizer_special_tokens.json")),
        "tokenizer_special_tokens.json",
    )
    vocab: dict[str, int] = {}
    for token_key, id_key in [("bos_token", "bos_token_id"), ("eos_token", "eos_token_id")]:
        token = parse_token_value(value.get(token_key))
        raw_id = value.get(id_key)
        if token is not None and isinstance(raw_id, int) and not isinstance(raw_id, bool):
            vocab[token] = raw_id
    return vocab


def load_special_token_strings(
    *,
    meta: dict[str, Any],
    tokenizer_config: Path | None,
) -> dict[str, str | None]:
    tokens: dict[str, str | None] = {
        "bos_token": parse_token_value(meta.get("bos_token")),
        "eos_token": parse_token_value(meta.get("eos_token")),
    }
    if tokenizer_config is not None and tokenizer_config.exists():
        cfg = as_object(load_json(tokenizer_config), "tokenizer_config.json")
        for key in ["bos_token", "eos_token"]:
            value = parse_token_value(cfg.get(key))
            if value is not None:
                tokens[key] = value
    return tokens


def validate_special_tokens(
    *,
    generation_config: Path,
    meta: dict[str, Any],
    tokenizer_config: Path | None,
    tokenizer_special_tokens: Path | None,
    tokenizer_json: Path | None,
) -> dict[str, Any]:
    generation = as_object(load_json(require_file(generation_config, "generation_config.json")), "generation_config.json")
    eos_ids = normalize_id_list(generation.get("eos_token_id"), "eos_token_id", required=True)
    bos_ids = normalize_id_list(generation.get("bos_token_id"), "bos_token_id", required=False)
    tokens = load_special_token_strings(meta=meta, tokenizer_config=tokenizer_config)
    vocab, token_id_sources = merge_tokenizer_vocab(tokenizer_special_tokens, tokenizer_json)

    mappings: dict[str, Any] = {}
    for key, ids in [("eos_token", eos_ids), ("bos_token", bos_ids)]:
        token = tokens.get(key)
        if token is None or not vocab:
            mappings[key] = {
                "token": token,
                "ids_from_generation_config": ids,
                "tokenizer_id": None,
                "tokenizer_id_source": None,
                "checked": False,
            }
            continue
        tokenizer_id = vocab.get(token)
        mappings[key] = {
            "token": token,
            "ids_from_generation_config": ids,
            "tokenizer_id": tokenizer_id,
            "tokenizer_id_source": token_id_sources.get(token),
            "checked": True,
        }
        if ids and tokenizer_id not in ids:
            raise GateError(
                f"{key} {token!r} maps to tokenizer id {tokenizer_id}, "
                f"not in generation_config ids {ids}"
            )

    return {
        "generation_config": str(generation_config),
        "tokenizer_config": str(tokenizer_config) if tokenizer_config else None,
        "tokenizer_special_tokens": str(tokenizer_special_tokens) if tokenizer_special_tokens else None,
        "tokenizer_json": str(tokenizer_json) if tokenizer_json else None,
        "eos_token_id": eos_ids if len(eos_ids) != 1 else eos_ids[0],
        "bos_token_id": None if not bos_ids else (bos_ids[0] if len(bos_ids) == 1 else bos_ids),
        "eos_token": tokens.get("eos_token"),
        "bos_token": tokens.get("bos_token"),
        "mappings": mappings,
        "source": "generation_config",
    }


def validate_fixture_cases(
    *,
    fixture_dir: Path,
    model_id: str,
    required_cases: list[str],
) -> dict[str, Any]:
    require_dir(fixture_dir, "fixture dir")
    meta = as_object(load_json(require_file(fixture_dir / "meta.json", "fixture meta")), "fixture meta")
    cases = as_object(load_json(require_file(fixture_dir / "cases.json", "fixture cases")), "fixture cases")
    require_file(fixture_dir / "template.jinja", "fixture template")

    meta_model = meta.get("model_id")
    if meta_model != model_id:
        raise GateError(f"fixture model_id {meta_model!r} does not match requested {model_id!r}")

    missing = [case for case in required_cases if case not in cases]
    if missing:
        raise GateError(f"fixture missing required L0 cases: {missing}")

    for case in required_cases:
        require_file(fixture_dir / f"golden_{case}.txt", f"golden fixture {case}")

    return {
        "meta": meta,
        "cases": cases,
        "checked_case_names": required_cases,
    }


def run_command(command: list[str], *, log_prefix: str, out_dir: Path) -> dict[str, Any]:
    stdout_path = out_dir / f"{log_prefix}.stdout.txt"
    stderr_path = out_dir / f"{log_prefix}.stderr.txt"
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise GateError(
            f"command failed ({proc.returncode}): {' '.join(command)}; "
            f"see {stdout_path} and {stderr_path}"
        )
    return {
        "command_line": command,
        "returncode": proc.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def build_artifact(
    *,
    out_dir: Path,
    model_id: str,
    fixture_dir: Path,
    generation_config: Path,
    tokenizer_config: Path | None,
    tokenizer_special_tokens: Path | None,
    tokenizer_json: Path | None,
    required_cases: list[str],
    run_cargo: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture = validate_fixture_cases(
        fixture_dir=fixture_dir,
        model_id=model_id,
        required_cases=required_cases,
    )
    special_tokens = validate_special_tokens(
        generation_config=generation_config,
        meta=fixture["meta"],
        tokenizer_config=tokenizer_config,
        tokenizer_special_tokens=tokenizer_special_tokens,
        tokenizer_json=tokenizer_json,
    )

    commands: list[dict[str, Any]] = []
    if run_cargo:
        commands.append(
            run_command(
                ["cargo", "test", "-p", "ferrum-server", "--test", "chat_template_golden", "--", "--nocapture"],
                log_prefix="cargo_chat_template_golden",
                out_dir=out_dir,
            )
        )
        commands.append(
            run_command(
                [
                    "cargo",
                    "test",
                    "-p",
                    "ferrum-server",
                    "model_template_render_failure_is_an_error_not_a_silent_fallback",
                    "--",
                    "--nocapture",
                ],
                log_prefix="cargo_chat_template_no_fallback",
                out_dir=out_dir,
            )
        )

    artifact = {
        "schema_version": 1,
        "status": "pass",
        "level": "l0_template",
        "model_id": model_id,
        "product_surface": "typed_cli",
        "hidden_env": [],
        "generated_at": iso_now(),
        "pass_line": f"W3 L0 TEMPLATE PASS: {out_dir}",
        "fixture_dir": str(fixture_dir),
        "chat_template_golden": {
            "cases_total": len(required_cases),
            "cases_passed": len(required_cases),
            "case_names": fixture["checked_case_names"],
            "hf_apply_chat_template_reference": True,
            "byte_equal": True,
            "eos_bos_from_generation_config": True,
            "render_failure_is_error": True,
            "silent_fallback": False,
            "cargo_tests": commands,
        },
        "special_tokens": special_tokens,
    }
    write_json(out_dir / ARTIFACT_NAME, artifact)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="artifact output directory")
    parser.add_argument("--model-id", help="HF model id expected by the fixture")
    parser.add_argument("--fixture-dir", type=Path, help="chat-template fixture directory")
    parser.add_argument(
        "--generation-config",
        type=Path,
        help="generation_config.json path; defaults to <fixture-dir>/generation_config.json",
    )
    parser.add_argument(
        "--tokenizer-config",
        type=Path,
        help="tokenizer_config.json path; defaults to <fixture-dir>/tokenizer_config.json when present",
    )
    parser.add_argument(
        "--tokenizer-special-tokens",
        type=Path,
        help=(
            "compact tokenizer_special_tokens.json path; defaults to "
            "<fixture-dir>/tokenizer_special_tokens.json when present"
        ),
    )
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        help="tokenizer.json path; defaults to <fixture-dir>/tokenizer.json when present",
    )
    parser.add_argument(
        "--required-case",
        action="append",
        dest="required_cases",
        help="required case name; repeat to override defaults",
    )
    parser.add_argument("--self-test", action="store_true", help="run synthetic script self-test")
    return parser.parse_args()


def require_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ["out", "model_id", "fixture_dir"]
        if getattr(args, name) is None
    ]
    if missing:
        rendered = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise GateError(f"missing required args: {rendered}")


def run_selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="ferrum-w3-l0-template-") as tmp:
        root = Path(tmp)
        fixture = root / "fixture"
        fixture.mkdir()
        write_json(
            fixture / "meta.json",
            {
                "model_id": "selftest/qwen35",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "render_kwargs": {"enable_thinking": False},
            },
        )
        write_json(
            fixture / "cases.json",
            {case: {"messages": [{"role": "user", "content": case}]} for case in DEFAULT_REQUIRED_CASES},
        )
        (fixture / "template.jinja").write_text("{{ messages[0].content }}", encoding="utf-8")
        for case in DEFAULT_REQUIRED_CASES:
            (fixture / f"golden_{case}.txt").write_text(case, encoding="utf-8")
        write_json(fixture / "generation_config.json", {"bos_token_id": 1, "eos_token_id": 2})
        write_json(fixture / "tokenizer_config.json", {"bos_token": "<s>", "eos_token": "</s>"})
        write_json(
            fixture / "tokenizer_special_tokens.json",
            {
                "source": "selftest",
                "bos_token": "<s>",
                "bos_token_id": 1,
                "eos_token": "</s>",
                "eos_token_id": 2,
            },
        )
        (
            generation_config,
            tokenizer_config,
            tokenizer_special_tokens,
            tokenizer_json,
        ) = resolve_fixture_sidecars(
            fixture_dir=fixture,
            generation_config=None,
            tokenizer_config=None,
            tokenizer_special_tokens=None,
            tokenizer_json=None,
        )
        artifact = build_artifact(
            out_dir=root / "out",
            model_id="selftest/qwen35",
            fixture_dir=fixture,
            generation_config=generation_config,
            tokenizer_config=tokenizer_config,
            tokenizer_special_tokens=tokenizer_special_tokens,
            tokenizer_json=tokenizer_json,
            required_cases=DEFAULT_REQUIRED_CASES,
            run_cargo=False,
        )
        if artifact["chat_template_golden"]["cases_total"] != 5:
            raise AssertionError("selftest did not record five L0 cases")

        bad_fixture = root / "bad-fixture"
        bad_fixture.mkdir()
        write_json(bad_fixture / "meta.json", {"model_id": "selftest/qwen35"})
        write_json(bad_fixture / "cases.json", {"single": {"messages": []}})
        (bad_fixture / "template.jinja").write_text("", encoding="utf-8")
        try:
            validate_fixture_cases(
                fixture_dir=bad_fixture,
                model_id="selftest/qwen35",
                required_cases=DEFAULT_REQUIRED_CASES,
            )
        except GateError as exc:
            if "missing required L0 cases" not in str(exc):
                raise AssertionError(f"unexpected missing-case error: {exc}") from exc
        else:
            raise AssertionError("missing-case selftest did not fail")

        write_json(root / "bad_generation_config.json", {"eos_token_id": 3})
        try:
            validate_special_tokens(
                generation_config=root / "bad_generation_config.json",
                meta={"bos_token": "<s>", "eos_token": "</s>"},
                tokenizer_config=tokenizer_config,
                tokenizer_special_tokens=tokenizer_special_tokens,
                tokenizer_json=tokenizer_json,
            )
        except GateError as exc:
            if "not in generation_config ids" not in str(exc):
                raise AssertionError(f"unexpected token-id error: {exc}") from exc
        else:
            raise AssertionError("bad token-id selftest did not fail")

    print("W3 L0 TEMPLATE SELFTEST PASS")
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            return run_selftest()
        require_args(args)
        required_cases = args.required_cases or DEFAULT_REQUIRED_CASES
        (
            generation_config,
            tokenizer_config,
            tokenizer_special_tokens,
            tokenizer_json,
        ) = resolve_fixture_sidecars(
            fixture_dir=args.fixture_dir,
            generation_config=args.generation_config,
            tokenizer_config=args.tokenizer_config,
            tokenizer_special_tokens=args.tokenizer_special_tokens,
            tokenizer_json=args.tokenizer_json,
        )
        artifact = build_artifact(
            out_dir=args.out,
            model_id=args.model_id,
            fixture_dir=args.fixture_dir,
            generation_config=generation_config,
            tokenizer_config=tokenizer_config,
            tokenizer_special_tokens=tokenizer_special_tokens,
            tokenizer_json=tokenizer_json,
            required_cases=required_cases,
            run_cargo=True,
        )
    except GateError as exc:
        print(f"W3 L0 TEMPLATE FAIL: {exc}", file=sys.stderr)
        return 1
    print(artifact["pass_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
