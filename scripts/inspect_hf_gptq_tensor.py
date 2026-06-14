#!/usr/bin/env python3
"""Inspect one GPTQ tensor from a remote safetensors shard via HTTP ranges.

This is intentionally dependency-free. It is for lightweight W2 diagnostics where
downloading a whole 27B checkpoint would be wasteful.
"""

from __future__ import annotations

import argparse
import array
import http.client
import json
import math
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_REPO = "circulus/gemma-3-27b-it-gptq"
DEFAULT_REVISION = "70d89a3a6b401b5f56558cb5d4c0f1fd158980b2"
GPTQ_SUFFIXES = ("qweight", "scales", "qzeros", "g_idx")


class Fetcher:
    def __init__(self, repo: str, revision: str, timeout: float, retries: int) -> None:
        self.repo = repo
        self.revision = revision
        self.timeout = timeout
        self.retries = retries
        self.bytes_read = 0
        self.requests = 0

    def url(self, path: str) -> str:
        repo = urllib.parse.quote(self.repo, safe="/")
        revision = urllib.parse.quote(self.revision, safe="")
        parts = [urllib.parse.quote(part, safe="") for part in path.split("/")]
        return f"https://huggingface.co/{repo}/resolve/{revision}/{'/'.join(parts)}"

    def request(self, path: str, byte_range: tuple[int, int] | None = None) -> bytes:
        url = self.url(path)
        headers = {"User-Agent": "ferrum-gptq-inspector/1.0"}
        if byte_range is not None:
            start, end = byte_range
            headers["Range"] = f"bytes={start}-{end}"
        last_error: BaseException | None = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = resp.read()
                self.requests += 1
                self.bytes_read += len(data)
                return data
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                OSError,
                http.client.HTTPException,
            ) as exc:
                last_error = exc
                if attempt == self.retries:
                    break
                time.sleep(0.75 * (attempt + 1))
        raise RuntimeError(f"failed to fetch {path} range={byte_range}: {last_error}")


def load_remote_json(fetcher: Fetcher, path: str) -> dict[str, Any]:
    return json.loads(fetcher.request(path).decode("utf-8"))


def locate_shard(fetcher: Fetcher, prefix: str, explicit_file: str | None) -> str:
    if explicit_file:
        return explicit_file
    index = load_remote_json(fetcher, "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("model.safetensors.index.json has no weight_map")
    shards = {weight_map.get(f"{prefix}.{suffix}") for suffix in GPTQ_SUFFIXES}
    if len(shards) != 1 or None in shards:
        raise RuntimeError(f"GPTQ tensors for {prefix} are not all in one shard: {shards}")
    shard = next(iter(shards))
    if not isinstance(shard, str):
        raise RuntimeError(f"invalid shard entry for {prefix}: {shard!r}")
    return shard


def load_safetensors_header(fetcher: Fetcher, shard: str) -> tuple[dict[str, Any], int]:
    first = fetcher.request(shard, (0, 7))
    if len(first) != 8:
        raise RuntimeError(f"could not read safetensors header length from {shard}")
    header_len = struct.unpack("<Q", first)[0]
    raw = fetcher.request(shard, (8, 8 + header_len - 1))
    if len(raw) != header_len:
        raise RuntimeError(f"short safetensors header: got {len(raw)} expected {header_len}")
    return json.loads(raw.decode("utf-8")), 8 + header_len


def tensor_meta(header: dict[str, Any], name: str) -> dict[str, Any]:
    meta = header.get(name)
    if not isinstance(meta, dict):
        raise RuntimeError(f"tensor {name} not found in safetensors header")
    for key in ("dtype", "shape", "data_offsets"):
        if key not in meta:
            raise RuntimeError(f"tensor {name} missing {key}")
    return meta


def tensor_bytes(
    fetcher: Fetcher,
    shard: str,
    data_base: int,
    meta: dict[str, Any],
    chunk_bytes: int,
) -> bytes:
    start, end = meta["data_offsets"]
    abs_start = data_base + int(start)
    abs_end = data_base + int(end) - 1
    total = abs_end - abs_start + 1
    if chunk_bytes <= 0 or total <= chunk_bytes:
        return fetcher.request(shard, (abs_start, abs_end))
    chunks = []
    pos = abs_start
    while pos <= abs_end:
        chunk_end = min(abs_end, pos + chunk_bytes - 1)
        chunks.append(fetcher.request(shard, (pos, chunk_end)))
        pos = chunk_end + 1
    return b"".join(chunks)


def decode_i32(raw: bytes) -> array.array:
    vals = array.array("i")
    vals.frombytes(raw)
    if sys.byteorder != "little":
        vals.byteswap()
    return vals


def bf16_to_f32(v: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(v) << 16))[0]


def decode_float(raw: bytes, dtype: str) -> list[float]:
    if dtype == "F16":
        return [x[0] for x in struct.iter_unpack("<e", raw)]
    if dtype == "BF16":
        words = array.array("H")
        words.frombytes(raw)
        if sys.byteorder != "little":
            words.byteswap()
        return [bf16_to_f32(x) for x in words]
    if dtype == "F32":
        vals = array.array("f")
        vals.frombytes(raw)
        if sys.byteorder != "little":
            vals.byteswap()
        return list(vals)
    raise RuntimeError(f"unsupported float dtype {dtype}")


def shape_product(shape: list[int]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def ensure_len(name: str, actual: int, shape: list[int]) -> None:
    expected = shape_product(shape)
    if actual != expected:
        raise RuntimeError(f"{name} length {actual} does not match shape {shape} ({expected})")


def qzero_hist(qzeros: array.array) -> dict[str, Any]:
    hist = [0] * 16
    first_bad: dict[str, int] | None = None
    for word_index, raw in enumerate(qzeros):
        word = int(raw) & 0xFFFF_FFFF
        for lane in range(8):
            code = (word >> (lane * 4)) & 0xF
            hist[code] += 1
            if code != 7 and first_bad is None:
                first_bad = {"word_index": word_index, "lane": lane, "code": code}
    total = sum(hist)
    return {
        "histogram": hist,
        "total_nibbles": total,
        "code7_count": hist[7],
        "all_code7": hist[7] == total,
        "first_non_code7": first_bad,
    }


def g_idx_stats(g_idx: array.array, groups: int, group_size: int) -> dict[str, Any]:
    counts = [0] * groups
    bad_preview = []
    for i, raw in enumerate(g_idx):
        group = int(raw)
        if 0 <= group < groups:
            counts[group] += 1
        elif len(bad_preview) < 8:
            bad_preview.append({"index": i, "group": group})
    unbalanced = [
        {"group": group, "count": count}
        for group, count in enumerate(counts)
        if count != group_size
    ]
    sequential = all(int(v) == i // group_size for i, v in enumerate(g_idx))
    return {
        "groups": groups,
        "group_size": group_size,
        "min_count": min(counts) if counts else 0,
        "max_count": max(counts) if counts else 0,
        "balanced_full_groups": not bad_preview and not unbalanced,
        "sequential_non_desc_act": sequential,
        "bad_group_preview": bad_preview,
        "unbalanced_preview": unbalanced[:8],
        "unbalanced_count": len(unbalanced),
    }


def float_stats(vals: list[float]) -> dict[str, Any]:
    finite = 0
    nonfinite = 0
    max_abs = 0.0
    min_val = None
    max_val = None
    for val in vals:
        if math.isfinite(val):
            finite += 1
            max_abs = max(max_abs, abs(val))
            min_val = val if min_val is None else min(min_val, val)
            max_val = val if max_val is None else max(max_val, val)
        else:
            nonfinite += 1
    return {
        "count": len(vals),
        "finite": finite,
        "nonfinite": nonfinite,
        "all_finite": nonfinite == 0,
        "max_abs": max_abs,
        "min": min_val,
        "max": max_val,
    }


def probe_input(kind: str, k: int, scale: float) -> float:
    if kind == "sin":
        return math.sin(k * 0.0041) * scale
    if kind == "alt":
        return (1.0 if (k % 2 == 0) else -1.0) * scale
    raise RuntimeError(f"unknown probe input {kind}")


def inspect_dequant(
    qweight: array.array,
    scales: list[float],
    qzeros: array.array,
    g_idx: array.array,
    k: int,
    n: int,
    groups: int,
    sample_cols: int,
    matmul_scales: list[float],
    matmul_input: str,
) -> dict[str, Any]:
    if sample_cols <= 0 or sample_cols >= n:
        cols = list(range(n))
        sampled = False
    else:
        if sample_cols == 1:
            cols = [0]
        else:
            cols = sorted({round(i * (n - 1) / (sample_cols - 1)) for i in range(sample_cols)})
        sampled = len(cols) != n

    outputs = [[0.0] * len(cols) for _ in matmul_scales]
    finite = 0
    nonfinite = 0
    max_abs = 0.0
    min_val = None
    max_val = None
    first_nonfinite = None

    for ki in range(k):
        row_base = (ki // 8) * n
        q_shift = (ki % 8) * 4
        group = int(g_idx[ki])
        if group < 0 or group >= groups:
            raise RuntimeError(f"g_idx[{ki}]={group} outside [0,{groups})")
        qzero_base = group * (n // 8)
        scale_base = group * n
        probe_values = [probe_input(matmul_input, ki, scale) for scale in matmul_scales]
        for out_col_index, col in enumerate(cols):
            packed = int(qweight[row_base + col]) & 0xFFFF_FFFF
            q = (packed >> q_shift) & 0xF
            zpacked = int(qzeros[qzero_base + (col // 8)]) & 0xFFFF_FFFF
            zero = ((zpacked >> ((col % 8) * 4)) & 0xF) + 1
            w = scales[scale_base + col] * (q - zero)
            if math.isfinite(w):
                finite += 1
                max_abs = max(max_abs, abs(w))
                min_val = w if min_val is None else min(min_val, w)
                max_val = w if max_val is None else max(max_val, w)
            else:
                nonfinite += 1
                if first_nonfinite is None:
                    first_nonfinite = {"k": ki, "col": col, "value": repr(w)}
            for probe_index, probe_value in enumerate(probe_values):
                outputs[probe_index][out_col_index] += probe_value * w

    probes = []
    for scale, out in zip(matmul_scales, outputs):
        probes.append(
            {
                "input": matmul_input,
                "input_scale": scale,
                "sampled_cols": sampled,
                "output_cols": len(cols),
                "output_max_abs": max((abs(v) for v in out), default=0.0),
                "output_min": min(out) if out else None,
                "output_max": max(out) if out else None,
                "output_all_finite": all(math.isfinite(v) for v in out),
            }
        )

    return {
        "sampled_cols": sampled,
        "cols_checked": len(cols),
        "cols_total": n,
        "weights_checked": finite + nonfinite,
        "finite": finite,
        "nonfinite": nonfinite,
        "all_finite": nonfinite == 0,
        "max_abs": max_abs,
        "min": min_val,
        "max": max_val,
        "first_nonfinite": first_nonfinite,
        "matmul_probes": probes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--file", help="safetensors shard; default resolves through index")
    parser.add_argument("--tensor-prefix", required=True, help="e.g. model.layers.0.self_attn.o_proj")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--sample-cols",
        type=int,
        default=0,
        help="0 means all output columns; otherwise use evenly spaced sampled columns",
    )
    parser.add_argument(
        "--range-chunk-mib",
        type=int,
        default=8,
        help="split large tensor HTTP range reads into chunks; 0 disables chunking",
    )
    parser.add_argument(
        "--matmul-input",
        choices=("sin", "alt"),
        default="sin",
        help="deterministic probe input pattern",
    )
    parser.add_argument(
        "--matmul-input-scale",
        action="append",
        type=float,
        default=[],
        help="can be repeated; defaults to 1.0 and 17.7",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fetcher = Fetcher(args.repo, args.revision, args.timeout, args.retries)
    range_chunk_bytes = args.range_chunk_mib * 1024 * 1024
    shard = locate_shard(fetcher, args.tensor_prefix, args.file)
    header, data_base = load_safetensors_header(fetcher, shard)

    metas = {
        suffix: tensor_meta(header, f"{args.tensor_prefix}.{suffix}")
        for suffix in GPTQ_SUFFIXES
    }
    qweight = decode_i32(
        tensor_bytes(fetcher, shard, data_base, metas["qweight"], range_chunk_bytes)
    )
    qzeros = decode_i32(
        tensor_bytes(fetcher, shard, data_base, metas["qzeros"], range_chunk_bytes)
    )
    g_idx = decode_i32(
        tensor_bytes(fetcher, shard, data_base, metas["g_idx"], range_chunk_bytes)
    )
    scales = decode_float(
        tensor_bytes(fetcher, shard, data_base, metas["scales"], range_chunk_bytes),
        metas["scales"]["dtype"],
    )

    shapes = {suffix: list(map(int, meta["shape"])) for suffix, meta in metas.items()}
    ensure_len("qweight", len(qweight), shapes["qweight"])
    ensure_len("qzeros", len(qzeros), shapes["qzeros"])
    ensure_len("g_idx", len(g_idx), shapes["g_idx"])
    ensure_len("scales", len(scales), shapes["scales"])

    if len(shapes["qweight"]) != 2 or len(shapes["qzeros"]) != 2 or len(shapes["scales"]) != 2:
        raise RuntimeError(f"unexpected GPTQ shapes: {shapes}")
    k = len(g_idx)
    q_rows, n = shapes["qweight"]
    groups, scale_n = shapes["scales"]
    qzero_groups, qzero_words = shapes["qzeros"]
    if q_rows * 8 != k:
        raise RuntimeError(f"qweight rows {q_rows} do not pack K={k}")
    if scale_n != n or qzero_groups != groups or qzero_words * 8 != n:
        raise RuntimeError(f"inconsistent GPTQ shapes: {shapes}")
    if k % groups != 0:
        raise RuntimeError(f"K={k} is not divisible by groups={groups}")
    group_size = k // groups

    matmul_scales = args.matmul_input_scale or [1.0, 17.7]
    report = {
        "repo": args.repo,
        "revision": args.revision,
        "shard": shard,
        "tensor_prefix": args.tensor_prefix,
        "shapes": shapes,
        "dtypes": {suffix: metas[suffix]["dtype"] for suffix in GPTQ_SUFFIXES},
        "derived": {
            "k": k,
            "n": n,
            "groups": groups,
            "group_size": group_size,
        },
        "g_idx": g_idx_stats(g_idx, groups, group_size),
        "qzeros": qzero_hist(qzeros),
        "scales": float_stats(scales),
        "dequant": inspect_dequant(
            qweight,
            scales,
            qzeros,
            g_idx,
            k,
            n,
            groups,
            args.sample_cols,
            matmul_scales,
            args.matmul_input,
        ),
        "fetch": {
            "requests": fetcher.requests,
            "range_payload_bytes_read": fetcher.bytes_read,
            "range_chunk_mib": args.range_chunk_mib,
        },
    }
    report["summary"] = {
        "tensor_prefix": args.tensor_prefix,
        "k": k,
        "n": n,
        "group_size": group_size,
        "g_idx_balanced_full_groups": report["g_idx"]["balanced_full_groups"],
        "g_idx_sequential_non_desc_act": report["g_idx"]["sequential_non_desc_act"],
        "qzeros_all_code7": report["qzeros"]["all_code7"],
        "scales_all_finite": report["scales"]["all_finite"],
        "scales_max_abs": report["scales"]["max_abs"],
        "dequant_sampled_cols": report["dequant"]["sampled_cols"],
        "dequant_cols_checked": report["dequant"]["cols_checked"],
        "dequant_weight_all_finite": report["dequant"]["all_finite"],
        "dequant_weight_max_abs": report["dequant"]["max_abs"],
        "matmul_probe_max_abs": [
            probe["output_max_abs"] for probe in report["dequant"]["matmul_probes"]
        ],
        "range_payload_bytes_read": report["fetch"]["range_payload_bytes_read"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
