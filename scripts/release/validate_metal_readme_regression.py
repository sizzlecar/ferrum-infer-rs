#!/usr/bin/env python3
"""Hard validator for scripts/metal_readme_regression.py artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BAD_PATTERNS = [
    "panic",
    "panicked",
    "KV cache overflow",
    "failed to render model chat template",
    "command encoder",
    "failed assertion",
    "<unk>",
    "[PAD]",
]


def fail(errors: list[str]) -> int:
    for err in errors:
        print(f"METAL README GATE FAIL: {err}", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    root = args.out_dir
    summary = root / "summary.json"
    if not summary.is_file():
        return fail([f"missing {summary}"])
    data = json.loads(summary.read_text())
    errors: list[str] = []
    for model in data.get("models", []):
        key = model.get("key", "<unknown>")
        default_startup = model.get("default_startup") or {}
        if default_startup.get("passed") is not True:
            errors.append(
                f"{key}: default_startup.passed != true "
                f"({root / (key + '.default.effective_config.json')})"
            )
        max_sequences = default_startup.get("max_sequences")
        min_required = default_startup.get("min_required_max_sequences")
        max_allowed = default_startup.get("max_allowed_max_sequences")
        if not isinstance(max_sequences, int) or not isinstance(min_required, int):
            errors.append(f"{key}: default startup max sequence evidence missing")
        elif max_sequences < min_required:
            errors.append(
                f"{key}: default max_sequences {max_sequences} < required {min_required} "
                f"({root / (key + '.default.effective_config.json')})"
            )
        elif isinstance(max_allowed, int) and max_sequences > max_allowed:
            errors.append(
                f"{key}: default max_sequences {max_sequences} > allowed {max_allowed} "
                f"({root / (key + '.default.effective_config.json')})"
            )

        if model.get("server_ready") is not True:
            errors.append(f"{key}: server_ready != true ({root / (key + '.models.json')})")
        serve_startup = model.get("serve_startup") or {}
        serve_max_sequences = serve_startup.get("max_sequences")
        max_cell_concurrency = max(
            (int(cell.get("concurrency") or 0) for cell in model.get("cells", [])),
            default=0,
        )
        if serve_startup.get("passed") is not True:
            errors.append(
                f"{key}: serve_startup.passed != true ({root / (key + '.effective_config.json')})"
            )
        if not isinstance(serve_max_sequences, int) or serve_max_sequences < max_cell_concurrency:
            errors.append(
                f"{key}: benchmark serve max_sequences {serve_max_sequences!r} < "
                f"max cell concurrency {max_cell_concurrency} ({root / (key + '.effective_config.json')})"
            )
        chat = model.get("chat") or {}
        for gate in ["paris", "multiturn", "stream"]:
            if (chat.get(gate) or {}).get("passed") is not True:
                errors.append(f"{key}: chat.{gate}.passed != true ({root / (key + '.' + gate + '_verdict.txt')})")
        tool_call = model.get("tool_call") or {}
        if tool_call.get("status") != "pass":
            errors.append(
                f"{key}: tool_call.status != pass "
                f"({root / (key + '.tool-call-regression/tool_call_regression.json')})"
            )
        for gate in [
            "omitted_tool_choice",
            "explicit_auto_tool_choice",
            "required_tool_choice",
            "tool_result_fill",
        ]:
            if ((tool_call.get("checks") or {}).get(gate) or {}).get("passed") is not True:
                errors.append(
                    f"{key}: tool_call.{gate}.passed != true "
                    f"({root / (key + '.tool-call-regression/tool_call_regression.json')})"
                )
        if model.get("moe") is True:
            has_multi_seq_cell = any(
                int(cell.get("concurrency") or 0) >= 2
                for cell in model.get("cells", [])
            )
            probe = model.get("unsafe_batch_probe") or {}
            if not has_multi_seq_cell and probe.get("enabled") is not True:
                errors.append(f"{key}: unsafe_batch_probe missing or disabled")
            if probe.get("enabled") is True and probe.get("product_default") is not False:
                errors.append(f"{key}: unsafe_batch_probe must be marked non-product-default")
            startup = probe.get("startup") or {}
            max_sequences = startup.get("max_sequences")
            if probe.get("enabled") is True and (
                not isinstance(max_sequences, int) or max_sequences < 2
            ):
                errors.append(
                    f"{key}: unsafe_batch_probe did not exercise multi-sequence startup "
                    f"({root / (key + '.unsafe_batch.effective_config.json')})"
                )
            quality = probe.get("quality") or {}
            if probe.get("enabled") is True and not isinstance(quality.get("passed"), bool):
                errors.append(
                    f"{key}: unsafe_batch_probe quality result missing "
                    f"({root / (key + '.unsafe_batch.c4.quality.json')})"
                )
        if (model.get("run") or {}).get("passed") is not True:
            errors.append(f"{key}: run.passed != true ({root / (key + '.run.verdict.txt')})")
        for cell in model.get("cells", []):
            c = cell.get("concurrency")
            quality = cell.get("quality") or {}
            if quality.get("passed") is not True:
                errors.append(f"{key} c={c}: quality.passed != true ({root / f'{key}.c{c}.quality.json'})")
            for name in ["status_200", "marker_ok", "square_ok", "format_ok"]:
                if quality.get(name) != quality.get("requests"):
                    errors.append(
                        f"{key} c={c}: quality {name} {quality.get(name)!r} != "
                        f"requests {quality.get('requests')!r} ({root / f'{key}.c{c}.quality.json'})"
                    )
            if quality.get("crosstalk") != 0:
                errors.append(f"{key} c={c}: quality crosstalk != 0 ({root / f'{key}.c{c}.quality.json'})")
            if quality.get("length_finishes") != 0:
                errors.append(f"{key} c={c}: quality length_finishes != 0 ({root / f'{key}.c{c}.quality.json'})")
            if cell.get("completed") != cell.get("prompts"):
                errors.append(f"{key} c={c}: completed != prompts ({root / f'{key}.c{c}.json'})")
            if cell.get("failed") != 0:
                errors.append(f"{key} c={c}: failed != 0 ({root / f'{key}.c{c}.json'})")
            if not isinstance(cell.get("output_throughput_tok_s"), (int, float)) or cell.get("output_throughput_tok_s") <= 0:
                errors.append(f"{key} c={c}: throughput missing/non-positive ({root / f'{key}.c{c}.json'})")
            if not isinstance(cell.get("ratio_to_readme"), (int, float)) or cell.get("ratio_to_readme") < 0.90:
                errors.append(f"{key} c={c}: ratio_to_readme < 0.90 ({root / f'{key}.c{c}.json'})")
            if cell.get("not_regressed_90pct") is not True:
                errors.append(f"{key} c={c}: not_regressed_90pct != true ({root / f'{key}.c{c}.json'})")
        for suffix in ["server.stdout", "server.stderr", "run.stdout", "run.stderr", "run_text_long.stdout", "run_text_long.stderr"]:
            path = root / f"{key}.{suffix}"
            if not path.is_file():
                continue
            text = path.read_text(errors="replace")
            lower = text.lower()
            for pat in BAD_PATTERNS:
                if pat.lower() in lower:
                    errors.append(f"{key}: forbidden log pattern {pat!r} in {path}")
    if errors:
        return fail(errors)
    print(f"METAL README GATE PASS: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
