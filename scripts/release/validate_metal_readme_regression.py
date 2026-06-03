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
        if model.get("server_ready") is not True:
            errors.append(f"{key}: server_ready != true ({root / (key + '.models.json')})")
        chat = model.get("chat") or {}
        for gate in ["paris", "multiturn", "stream"]:
            if (chat.get(gate) or {}).get("passed") is not True:
                errors.append(f"{key}: chat.{gate}.passed != true ({root / (key + '.' + gate + '_verdict.txt')})")
        if (model.get("run") or {}).get("passed") is not True:
            errors.append(f"{key}: run.passed != true ({root / (key + '.run.verdict.txt')})")
        for cell in model.get("cells", []):
            c = cell.get("concurrency")
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
