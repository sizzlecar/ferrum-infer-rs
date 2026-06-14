#!/usr/bin/env python3
"""Summarize ferrum FERRUM_LAYER_DUMP f32 .bin files.

The script is intentionally small and dependency-free so CUDA failure artifacts
can be inspected on machines without numpy.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


LAYER_RE = re.compile(r"layer_(\d+)\.bin$")


def ordered_dump_files(root: Path) -> list[Path]:
    files = []
    embed = root / "embed.bin"
    if embed.exists():
        files.append(embed)
    layers = []
    for path in root.glob("layer_*.bin"):
        match = LAYER_RE.match(path.name)
        if match:
            layers.append((int(match.group(1)), path))
    files.extend(path for _, path in sorted(layers))
    logits = root / "logits.bin"
    if logits.exists():
        files.append(logits)
    return files


def read_f32(path: Path) -> array.array:
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise ValueError(f"{path} byte length {len(raw)} is not divisible by 4")
    vals = array.array("f")
    vals.frombytes(raw)
    if sys.byteorder != "little":
        vals.byteswap()
    return vals


def with_coord(point: dict[str, Any] | None, last_dim: int | None) -> dict[str, Any] | None:
    if point is None or not last_dim:
        return point
    index = point.get("index")
    if not isinstance(index, int):
        return point
    out = dict(point)
    out["row"] = index // last_dim
    out["col"] = index % last_dim
    return out


def summarize_values(path: Path, thresholds: list[float], last_dim: int | None) -> dict[str, Any]:
    vals = read_f32(path)
    shape = None
    if last_dim and len(vals) % last_dim == 0:
        shape = [len(vals) // last_dim, last_dim]
    finite = 0
    nan = 0
    pos_inf = 0
    neg_inf = 0
    min_val = None
    max_val = None
    max_abs = 0.0
    max_abs_index = None
    sum_abs = 0.0
    first_nonfinite = None
    first_thresholds = {str(t): None for t in thresholds}

    for i, val in enumerate(vals):
        if math.isfinite(val):
            finite += 1
            av = abs(val)
            sum_abs += av
            if av > max_abs:
                max_abs = av
                max_abs_index = i
            min_val = val if min_val is None else min(min_val, val)
            max_val = val if max_val is None else max(max_val, val)
            for threshold in thresholds:
                key = str(threshold)
                if first_thresholds[key] is None and av > threshold:
                    first_thresholds[key] = {"index": i, "value": val}
        elif math.isnan(val):
            nan += 1
            if first_nonfinite is None:
                first_nonfinite = {"index": i, "kind": "nan"}
        elif val > 0:
            pos_inf += 1
            if first_nonfinite is None:
                first_nonfinite = {"index": i, "kind": "+inf"}
        else:
            neg_inf += 1
            if first_nonfinite is None:
                first_nonfinite = {"index": i, "kind": "-inf"}

    count = len(vals)
    nonfinite = nan + pos_inf + neg_inf
    return {
        "file": path.name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "count": count,
        "shape": shape,
        "finite": finite,
        "nonfinite": nonfinite,
        "nan": nan,
        "pos_inf": pos_inf,
        "neg_inf": neg_inf,
        "all_finite": nonfinite == 0,
        "all_nonfinite": nonfinite == count and count > 0,
        "min": min_val,
        "max": max_val,
        "max_abs": max_abs,
        "max_abs_index": max_abs_index,
        "max_abs_coord": with_coord({"index": max_abs_index}, last_dim)
        if max_abs_index is not None and shape
        else None,
        "mean_abs": (sum_abs / finite) if finite else None,
        "first_nonfinite": with_coord(first_nonfinite, last_dim) if shape else first_nonfinite,
        "first_abs_gt": {
            key: with_coord(value, last_dim) if shape else value
            for key, value in first_thresholds.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        default=[],
        help="absolute-value threshold to record first crossing; repeatable",
    )
    parser.add_argument(
        "--last-dim",
        type=int,
        help="annotate flat indices as [row, col] when count is divisible by this dimension",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dump_dir
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    thresholds = args.threshold or [100.0, 1000.0, 10000.0]
    files = ordered_dump_files(root)
    if not files:
        raise SystemExit(f"no dump .bin files found under {root}")
    summaries = [summarize_values(path, thresholds, args.last_dim) for path in files]

    first_nonfinite_file = next((s for s in summaries if s["nonfinite"] > 0), None)
    first_threshold_files: dict[str, dict[str, Any] | None] = {}
    for threshold in thresholds:
        key = str(threshold)
        first_threshold_files[key] = next(
            (s for s in summaries if s["first_abs_gt"][key] is not None), None
        )

    report = {
        "dump_dir": str(root),
        "thresholds": thresholds,
        "files": summaries,
        "summary": {
            "file_count": len(summaries),
            "last_dim": args.last_dim,
            "first_nonfinite_file": first_nonfinite_file["file"] if first_nonfinite_file else None,
            "first_nonfinite": first_nonfinite_file["first_nonfinite"]
            if first_nonfinite_file
            else None,
            "first_abs_gt": {
                key: {
                    "file": value["file"],
                    "point": value["first_abs_gt"][key],
                    "file_max_abs": value["max_abs"],
                }
                if value
                else None
                for key, value in first_threshold_files.items()
            },
            "last_file": summaries[-1]["file"],
            "last_file_all_nonfinite": summaries[-1]["all_nonfinite"],
            "max_abs_by_file": {s["file"]: s["max_abs"] for s in summaries},
            "nonfinite_by_file": {s["file"]: s["nonfinite"] for s in summaries},
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
