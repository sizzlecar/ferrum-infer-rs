#!/usr/bin/env python3
"""Concurrent content-quality probe for a running OpenAI-compatible server."""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

FORBIDDEN_TEXT = [
    "<unk>",
    "[PAD]",
    "<|assistant|>",
    "<|tool|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|reserved_special_token_",
    "classname=",
]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def post_json(base_url: str, payload: dict[str, Any], timeout: int = 180) -> tuple[int, str]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def has_forbidden_text(text: str) -> str | None:
    for token in FORBIDDEN_TEXT:
        if token in text:
            return token
    return None


def parse_cells(raw: str) -> list[int]:
    cells = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not cells or any(cell <= 0 for cell in cells):
        raise argparse.ArgumentTypeError("expected comma-separated positive concurrency cells")
    return cells


def marker_line_ok(line: str, marker: str) -> bool:
    return line.strip().rstrip(".。!！,，;；:：") == marker


def answer_line_ok(line: str, answer: str) -> bool:
    return line.strip().rstrip(".。!！,，;；:：") == answer


def run_concurrency_quality_regression(
    base_url: str,
    model: str,
    out: Path,
    concurrency_cells: list[int],
    *,
    timeout: int = 180,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"model": model, "cells": []}
    failed: list[str] = []

    for concurrency in concurrency_cells:
        prefix = f"c{concurrency}.quality"
        nonce = f"M{concurrency:02d}"

        def call(i: int) -> dict[str, Any]:
            marker = f"K{nonce}{i:02d}"
            value = i + 1
            answer = f"S{value * value:04d}"
            payload = {
                "model": model,
                "temperature": 0,
                "max_tokens": 96,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Reply with exactly two short lines. "
                            f"Line 1 must be this code exactly: {marker}. "
                            f"Line 2 must be this checksum exactly: {answer}."
                        ),
                    }
                ],
            }
            status, body = post_json(base_url, payload, timeout=timeout)
            content = ""
            finish_reason = None
            json_ok = False
            try:
                parsed = json.loads(body)
                json_ok = True
                choice = parsed["choices"][0]
                content = strip_think(choice["message"].get("content") or "")
                finish_reason = choice.get("finish_reason")
            except Exception:
                content = body[:500]
            forbidden = has_forbidden_text(content)
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            format_ok = (
                len(lines) == 2
                and marker_line_ok(lines[0], marker)
                and answer_line_ok(lines[1], answer)
            )
            return {
                "i": i,
                "status": status,
                "json_ok": json_ok,
                "marker": marker,
                "square": answer,
                "marker_ok": marker in content,
                "square_ok": answer in content,
                "format_ok": format_ok,
                "finish_reason": finish_reason,
                "forbidden_text": forbidden,
                "content_head": content[:500],
            }

        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(call, i) for i in range(concurrency)]
            for future in as_completed(futures):
                rows.append(future.result())
        rows.sort(key=lambda row: int(row.get("i") or 0))

        markers = {row["i"]: row["marker"] for row in rows}
        crosstalk = 0
        for row in rows:
            text = str(row.get("content_head") or "")
            for other_i, marker in markers.items():
                if other_i != row["i"] and marker in text:
                    crosstalk += 1

        status_200 = sum(1 for row in rows if row.get("status") == 200)
        json_ok = sum(1 for row in rows if row.get("json_ok") is True)
        marker_ok = sum(1 for row in rows if row.get("marker_ok") is True)
        square_ok = sum(1 for row in rows if row.get("square_ok") is True)
        format_ok = sum(1 for row in rows if row.get("format_ok") is True)
        length_finishes = sum(1 for row in rows if row.get("finish_reason") == "length")
        forbidden_count = sum(1 for row in rows if row.get("forbidden_text"))
        cell = {
            "concurrency": concurrency,
            "requests": concurrency,
            "status_200": status_200,
            "json_ok": json_ok,
            "marker_ok": marker_ok,
            "square_ok": square_ok,
            "format_ok": format_ok,
            "crosstalk": crosstalk,
            "length_finishes": length_finishes,
            "forbidden_count": forbidden_count,
            "passed": (
                status_200 == concurrency
                and json_ok == concurrency
                and marker_ok == concurrency
                and square_ok == concurrency
                and format_ok == concurrency
                and crosstalk == 0
                and length_finishes == 0
                and forbidden_count == 0
            ),
            "rows": rows,
        }
        write(out / f"{prefix}.json", json.dumps(cell, ensure_ascii=False, indent=2) + "\n")
        result["cells"].append({k: v for k, v in cell.items() if k != "rows"})
        if not cell["passed"]:
            failed.append(
                f"c={concurrency} status_200={status_200}/{concurrency} "
                f"marker_ok={marker_ok}/{concurrency} square_ok={square_ok}/{concurrency} "
                f"format_ok={format_ok}/{concurrency} crosstalk={crosstalk} "
                f"length_finishes={length_finishes} forbidden={forbidden_count}"
            )

    result["status"] = "fail" if failed else "pass"
    if failed:
        result["errors"] = failed
    write(out / "concurrency_quality_regression.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if failed:
        raise RuntimeError("; ".join(failed))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency-cells", type=parse_cells, default=[1, 4, 16, 32])
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()
    try:
        run_concurrency_quality_regression(
            args.base_url,
            args.model,
            args.out,
            args.concurrency_cells,
            timeout=args.timeout,
        )
    except Exception as e:
        args.out.mkdir(parents=True, exist_ok=True)
        summary = args.out / "concurrency_quality_regression.json"
        if not summary.is_file():
            write(
                summary,
                json.dumps(
                    {"status": "fail", "model": args.model, "error": str(e)},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
        print(f"OPENAI CONCURRENCY QUALITY REGRESSION FAIL: {e}", file=sys.stderr)
        return 1
    print(f"OPENAI CONCURRENCY QUALITY REGRESSION PASS: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
