#!/usr/bin/env python3
"""Validate `[cuda-build-summary]` lines emitted by ferrum-kernels/build.rs."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


SUMMARY_RE = re.compile(
    r"^(?:\[[^\]]+\]\s+)?\[cuda-build-summary\]\s+"
    r"artifact=(?P<artifact>\S+)\s+"
    r"status=(?P<status>built|cache_hit)\s+"
    r"reason=(?P<reason>\S+)\s+"
    r"elapsed_ms=(?P<elapsed_ms>[0-9]+)\s+"
    r"inputs_hash=(?P<inputs_hash>fnv1a64:[0-9a-f]{16})$"
)


class ValidationError(Exception):
    pass


def parse_summary_lines(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    malformed: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if "[cuda-build-summary]" not in line:
            continue
        line = line.strip()
        match = SUMMARY_RE.match(line)
        if not match:
            malformed.append(f"line {line_no}: {line}")
            continue
        row = match.groupdict()
        row["elapsed_ms"] = int(row["elapsed_ms"])
        row["line_no"] = line_no
        rows.append(row)
    return rows, malformed


def validate_summary(
    text: str,
    *,
    require_rows: bool = True,
    require_artifacts: list[str] | None = None,
    require_cache_hit: list[str] | None = None,
    require_built: list[str] | None = None,
) -> dict[str, Any]:
    rows, malformed = parse_summary_lines(text)
    if malformed:
        raise ValidationError("malformed summary lines:\n" + "\n".join(malformed))
    if require_rows and not rows:
        raise ValidationError("no cuda build summary rows found")

    artifacts = {row["artifact"]: row for row in rows}
    missing_artifacts = [
        artifact for artifact in (require_artifacts or []) if artifact not in artifacts
    ]
    if missing_artifacts:
        raise ValidationError(f"missing required artifacts: {missing_artifacts}")

    wrong_cache = [
        artifact
        for artifact in (require_cache_hit or [])
        if artifacts.get(artifact, {}).get("status") != "cache_hit"
    ]
    if wrong_cache:
        raise ValidationError(f"artifacts were not cache_hit: {wrong_cache}")

    wrong_built = [
        artifact
        for artifact in (require_built or [])
        if artifacts.get(artifact, {}).get("status") != "built"
    ]
    if wrong_built:
        raise ValidationError(f"artifacts were not built: {wrong_built}")

    status_counts = Counter(row["status"] for row in rows)
    return {
        "ok": True,
        "row_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "artifacts": sorted(artifacts),
    }


def run_self_test() -> None:
    fixture = "\n".join(
        [
            "[cuda-build-summary] artifact=core-ptx:kernels/paged_varlen_attention_vllm.cu status=cache_hit reason=signature-match elapsed_ms=0 inputs_hash=fnv1a64:0123456789abcdef",
            "[ferrum-kernels 0.7.3] [cuda-build-summary] artifact=vllm_marlin status=cache_hit reason=signature-match elapsed_ms=1 inputs_hash=fnv1a64:1111111111111111",
            "[cuda-build-summary] artifact=vllm_moe_marlin status=built reason=signature-changed elapsed_ms=474451 inputs_hash=fnv1a64:abcdef0123456789",
        ]
    )
    result = validate_summary(
        fixture,
        require_artifacts=[
            "core-ptx:kernels/paged_varlen_attention_vllm.cu",
            "vllm_marlin",
            "vllm_moe_marlin",
        ],
        require_cache_hit=[
            "core-ptx:kernels/paged_varlen_attention_vllm.cu",
            "vllm_marlin",
        ],
        require_built=["vllm_moe_marlin"],
    )
    assert result["row_count"] == 3
    assert result["status_counts"] == {"built": 1, "cache_hit": 2}

    try:
        validate_summary(
            fixture,
            require_cache_hit=["vllm_moe_marlin"],
        )
    except ValidationError as exc:
        assert "not cache_hit" in str(exc)
    else:
        raise AssertionError("wrong cache-hit status unexpectedly passed")

    malformed = "[cuda-build-summary] artifact=x status=done reason=r elapsed_ms=1 inputs_hash=bad"
    try:
        validate_summary(malformed)
    except ValidationError as exc:
        assert "malformed" in str(exc)
    else:
        raise AssertionError("malformed summary unexpectedly passed")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "summary.log"
        path.write_text(fixture + "\n", encoding="utf-8")
        loaded = path.read_text(encoding="utf-8")
        assert validate_summary(loaded)["row_count"] == 3

    print("validate_cuda_build_summary self-test ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", help="build log path to validate")
    parser.add_argument("--json", action="store_true", help="emit JSON summary")
    parser.add_argument("--self-test", action="store_true", help="run self-tests and exit")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="do not fail when no cuda-build-summary rows are present",
    )
    parser.add_argument(
        "--require-artifact",
        action="append",
        default=[],
        help="artifact name that must appear; may be passed multiple times",
    )
    parser.add_argument(
        "--require-cache-hit",
        action="append",
        default=[],
        help="artifact name that must appear with status=cache_hit",
    )
    parser.add_argument(
        "--require-built",
        action="append",
        default=[],
        help="artifact name that must appear with status=built",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if not args.log:
        raise SystemExit("log path is required unless --self-test is used")

    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    result = validate_summary(
        text,
        require_rows=not args.allow_empty,
        require_artifacts=args.require_artifact,
        require_cache_hit=args.require_cache_hit,
        require_built=args.require_built,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "cuda build summary ok: "
            f"rows={result['row_count']} statuses={result['status_counts']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
