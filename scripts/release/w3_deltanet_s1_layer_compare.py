#!/usr/bin/env python3
"""W3-S1 Gated DeltaNet single-layer dump comparator.

This script defines the W3-S1 single-layer correctness contract.  It compares a
reference dump against a Ferrum dump for:

* DeltaNet q/k/v/beta projections and recurrent delta-rule output.
* Router logits, top-k indices, and top-k weights.
* Routed expert output, shared expert output, merge semantics, and final layer
  output.

Self-test mode writes a deterministic reference dump and a synthetic Ferrum dump
with the same tensors, then compares them.  That proves the S1 gate contract and
schema, but it is not final W3-S1 evidence for a real model implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS_SELFTEST = "W3 DELTANET S1 LAYER COMPARE SELFTEST PASS"
PASS_COMPARE = "W3 DELTANET S1 LAYER COMPARE PASS"
MANIFEST_NAME = "w3_deltanet_s1_layer_compare_manifest.json"
DUMP_MANIFEST_NAME = "w3_deltanet_s1_dump_manifest.json"

FLOAT_TENSORS = [
    "input",
    "delta_q",
    "delta_k",
    "delta_v",
    "delta_beta",
    "delta_core",
    "delta_gate",
    "delta_output",
    "router_logits",
    "router_topk_weights",
    "routed_expert_output",
    "shared_expert_output",
    "moe_output",
    "layer_output",
]

INT_TENSORS = ["router_topk_indices"]


class CompareError(Exception):
    pass


@dataclass(frozen=True)
class Shape:
    tokens: int
    hidden_dim: int
    heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    experts: int
    top_k: int
    expert_hidden_dim: int

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if value <= 0:
                raise CompareError(f"{name} must be positive, got {value}")
        if self.top_k > self.experts:
            raise CompareError("top_k cannot exceed experts")
        if self.value_heads % self.heads != 0:
            raise CompareError("value_heads must be divisible by heads")

    @property
    def qk_dim(self) -> int:
        return self.heads * self.key_dim

    @property
    def value_total_dim(self) -> int:
        return self.value_heads * self.value_dim

    @property
    def qk_repeat_factor(self) -> int:
        return self.value_heads // self.heads

    def key_head_for_value_head(self, value_head: int) -> int:
        return value_head // self.qk_repeat_factor


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def command_line() -> list[str]:
    return [sys.executable, *sys.argv]


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def git_output(args: list[str], *, default: str = "unknown") -> str:
    try:
        proc = run_command(["git", *args])
    except OSError:
        return default
    if proc.returncode != 0:
        return default
    return proc.stdout.strip()


def git_summary() -> dict[str, Any]:
    status = git_output(["status", "--short", "--untracked-files=no"], default="")
    tracked_lines = [line for line in status.splitlines() if line.strip()]
    untracked = git_output(["ls-files", "--others", "--exclude-standard"], default="")
    untracked_lines = [line for line in untracked.splitlines() if line.strip()]
    return {
        "sha": git_output(["rev-parse", "HEAD"]),
        "is_dirty": bool(tracked_lines or untracked_lines),
        "tracked_status_short": tracked_lines,
        "untracked_count": len(untracked_lines),
        "untracked_sample": untracked_lines[:20],
    }


def lcg(seed: int) -> Any:
    state = seed & 0xFFFFFFFF
    while True:
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        yield state


def rand_float(gen: Any, scale: float) -> float:
    raw = next(gen)
    centered = (raw / 0xFFFFFFFF) * 2.0 - 1.0
    return centered * scale


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def silu(x: float) -> float:
    return x * sigmoid(x)


def softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    denom = sum(exps)
    return [value / denom for value in exps]


def matmul_token_major(x: list[float], weights: list[float], tokens: int, in_dim: int, out_dim: int) -> list[float]:
    out = [0.0] * (tokens * out_dim)
    for t in range(tokens):
        for o in range(out_dim):
            acc = 0.0
            for i in range(in_dim):
                acc += x[t * in_dim + i] * weights[i * out_dim + o]
            out[t * out_dim + o] = acc
    return out


def tensor_shapes(shape: Shape) -> dict[str, list[int]]:
    return {
        "input": [shape.tokens, shape.hidden_dim],
        "delta_q": [shape.tokens, shape.heads, shape.key_dim],
        "delta_k": [shape.tokens, shape.heads, shape.key_dim],
        "delta_v": [shape.tokens, shape.value_heads, shape.value_dim],
        "delta_beta": [shape.tokens, shape.value_heads],
        "delta_core": [shape.tokens, shape.value_heads, shape.value_dim],
        "delta_gate": [shape.tokens, shape.value_total_dim],
        "delta_output": [shape.tokens, shape.hidden_dim],
        "router_logits": [shape.tokens, shape.experts],
        "router_topk_indices": [shape.tokens, shape.top_k],
        "router_topk_weights": [shape.tokens, shape.top_k],
        "routed_expert_output": [shape.tokens, shape.hidden_dim],
        "shared_expert_output": [shape.tokens, shape.hidden_dim],
        "moe_output": [shape.tokens, shape.hidden_dim],
        "layer_output": [shape.tokens, shape.hidden_dim],
    }


def count_from_shape(dims: list[int]) -> int:
    count = 1
    for dim in dims:
        count *= dim
    return count


def make_weights(shape: Shape, seed: int) -> dict[str, Any]:
    gen = lcg(seed)

    def floats(count: int, scale: float) -> list[float]:
        return [rand_float(gen, scale) for _ in range(count)]

    expert_weights = []
    for _ in range(shape.experts):
        expert_weights.append(
            {
                "gate": floats(shape.hidden_dim * shape.expert_hidden_dim, 0.10),
                "up": floats(shape.hidden_dim * shape.expert_hidden_dim, 0.10),
                "down": floats(shape.expert_hidden_dim * shape.hidden_dim, 0.10),
            }
        )
    return {
        "input": floats(shape.tokens * shape.hidden_dim, 0.30),
        "w_q": floats(shape.hidden_dim * shape.qk_dim, 0.12),
        "w_k": floats(shape.hidden_dim * shape.qk_dim, 0.12),
        "w_v": floats(shape.hidden_dim * shape.value_total_dim, 0.12),
        "w_beta": floats(shape.hidden_dim * shape.value_heads, 0.10),
        "w_delta_gate": floats(shape.hidden_dim * shape.value_total_dim, 0.10),
        "w_o": floats(shape.value_total_dim * shape.hidden_dim, 0.10),
        "w_router": floats(shape.hidden_dim * shape.experts, 0.10),
        "experts": expert_weights,
        "shared_gate": floats(shape.hidden_dim * shape.expert_hidden_dim, 0.10),
        "shared_up": floats(shape.hidden_dim * shape.expert_hidden_dim, 0.10),
        "shared_down": floats(shape.expert_hidden_dim * shape.hidden_dim, 0.10),
    }


def qk_index(shape: Shape, t: int, h: int, kk: int) -> int:
    return (t * shape.heads + h) * shape.key_dim + kk


def value_index(shape: Shape, t: int, value_head: int, vv: int) -> int:
    return (t * shape.value_heads + value_head) * shape.value_dim + vv


def delta_rule(shape: Shape, q: list[float], k: list[float], v: list[float], beta: list[float]) -> list[float]:
    out = [0.0] * (shape.tokens * shape.value_total_dim)
    for value_head in range(shape.value_heads):
        key_head = shape.key_head_for_value_head(value_head)
        state = [0.0] * (shape.key_dim * shape.value_dim)
        for t in range(shape.tokens):
            bt = beta[t * shape.value_heads + value_head]
            for vv in range(shape.value_dim):
                pred = 0.0
                for kk in range(shape.key_dim):
                    pred += k[qk_index(shape, t, key_head, kk)] * state[kk * shape.value_dim + vv]
                delta = bt * (v[value_index(shape, t, value_head, vv)] - pred)
                for kk in range(shape.key_dim):
                    state[kk * shape.value_dim + vv] += k[qk_index(shape, t, key_head, kk)] * delta
            for vv in range(shape.value_dim):
                acc = 0.0
                for kk in range(shape.key_dim):
                    acc += q[qk_index(shape, t, key_head, kk)] * state[kk * shape.value_dim + vv]
                out[value_index(shape, t, value_head, vv)] = acc
    return out


def expert_mlp(
    token: list[float],
    gate: list[float],
    up: list[float],
    down: list[float],
    hidden_dim: int,
    expert_hidden_dim: int,
) -> list[float]:
    gate_vec = [0.0] * expert_hidden_dim
    up_vec = [0.0] * expert_hidden_dim
    for j in range(expert_hidden_dim):
        gate_acc = 0.0
        up_acc = 0.0
        for i in range(hidden_dim):
            gate_acc += token[i] * gate[i * expert_hidden_dim + j]
            up_acc += token[i] * up[i * expert_hidden_dim + j]
        gate_vec[j] = silu(gate_acc)
        up_vec[j] = up_acc
    fused = [gate_vec[j] * up_vec[j] for j in range(expert_hidden_dim)]
    out = [0.0] * hidden_dim
    for o in range(hidden_dim):
        acc = 0.0
        for j in range(expert_hidden_dim):
            acc += fused[j] * down[j * hidden_dim + o]
        out[o] = acc
    return out


def compute_reference(shape: Shape, seed: int) -> dict[str, Any]:
    weights = make_weights(shape, seed)
    x = weights["input"]
    q = matmul_token_major(x, weights["w_q"], shape.tokens, shape.hidden_dim, shape.qk_dim)
    k = matmul_token_major(x, weights["w_k"], shape.tokens, shape.hidden_dim, shape.qk_dim)
    v = matmul_token_major(
        x,
        weights["w_v"],
        shape.tokens,
        shape.hidden_dim,
        shape.value_total_dim,
    )
    beta_raw = matmul_token_major(
        x,
        weights["w_beta"],
        shape.tokens,
        shape.hidden_dim,
        shape.value_heads,
    )
    beta = [sigmoid(value) for value in beta_raw]
    delta_core = delta_rule(shape, q, k, v, beta)
    delta_gate_raw = matmul_token_major(
        x,
        weights["w_delta_gate"],
        shape.tokens,
        shape.hidden_dim,
        shape.value_total_dim,
    )
    delta_gate = [sigmoid(value) for value in delta_gate_raw]
    gated_delta = [a * b for a, b in zip(delta_core, delta_gate)]
    delta_output = matmul_token_major(
        gated_delta,
        weights["w_o"],
        shape.tokens,
        shape.value_total_dim,
        shape.hidden_dim,
    )
    router_logits = matmul_token_major(
        x,
        weights["w_router"],
        shape.tokens,
        shape.hidden_dim,
        shape.experts,
    )
    router_topk_indices: list[int] = []
    router_topk_weights: list[float] = []
    routed_expert_output = [0.0] * (shape.tokens * shape.hidden_dim)
    shared_expert_output = [0.0] * (shape.tokens * shape.hidden_dim)
    for t in range(shape.tokens):
        token = x[t * shape.hidden_dim : (t + 1) * shape.hidden_dim]
        logits = router_logits[t * shape.experts : (t + 1) * shape.experts]
        ranked = sorted(range(shape.experts), key=lambda idx: (-logits[idx], idx))
        selected = ranked[: shape.top_k]
        weights_topk = softmax([logits[idx] for idx in selected])
        router_topk_indices.extend(selected)
        router_topk_weights.extend(weights_topk)
        routed = [0.0] * shape.hidden_dim
        for expert_id, weight in zip(selected, weights_topk):
            expert = weights["experts"][expert_id]
            expert_out = expert_mlp(
                token,
                expert["gate"],
                expert["up"],
                expert["down"],
                shape.hidden_dim,
                shape.expert_hidden_dim,
            )
            for i in range(shape.hidden_dim):
                routed[i] += weight * expert_out[i]
        shared = expert_mlp(
            token,
            weights["shared_gate"],
            weights["shared_up"],
            weights["shared_down"],
            shape.hidden_dim,
            shape.expert_hidden_dim,
        )
        for i in range(shape.hidden_dim):
            routed_expert_output[t * shape.hidden_dim + i] = routed[i]
            shared_expert_output[t * shape.hidden_dim + i] = shared[i]
    moe_output = [
        routed_expert_output[i] + shared_expert_output[i]
        for i in range(shape.tokens * shape.hidden_dim)
    ]
    layer_output = [
        x[i] + delta_output[i] + moe_output[i]
        for i in range(shape.tokens * shape.hidden_dim)
    ]
    return {
        "float_tensors": {
            "input": x,
            "delta_q": q,
            "delta_k": k,
            "delta_v": v,
            "delta_beta": beta,
            "delta_core": delta_core,
            "delta_gate": delta_gate,
            "delta_output": delta_output,
            "router_logits": router_logits,
            "router_topk_weights": router_topk_weights,
            "routed_expert_output": routed_expert_output,
            "shared_expert_output": shared_expert_output,
            "moe_output": moe_output,
            "layer_output": layer_output,
        },
        "int_tensors": {
            "router_topk_indices": router_topk_indices,
        },
    }


def write_f32(path: Path, values: list[float]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def read_f32(path: Path) -> list[float]:
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise CompareError(f"{path} size is not a float32 multiple")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CompareError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CompareError(f"invalid JSON in {path}: {exc}") from exc


def write_dump(path: Path, shape: Shape, seed: int, tensors: dict[str, Any], *, producer: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    shapes = tensor_shapes(shape)
    for name in FLOAT_TENSORS:
        values = tensors["float_tensors"][name]
        expected = count_from_shape(shapes[name])
        if len(values) != expected:
            raise CompareError(f"{name} length {len(values)} != expected {expected}")
        write_f32(path / f"{name}.bin", values)
    for name in INT_TENSORS:
        values = tensors["int_tensors"][name]
        expected = count_from_shape(shapes[name])
        if len(values) != expected:
            raise CompareError(f"{name} length {len(values)} != expected {expected}")
        write_json(path / f"{name}.json", values)
    write_json(
        path / DUMP_MANIFEST_NAME,
        {
            "schema_version": 1,
            "producer": producer,
            "created_at": iso_now(),
            "shape": shape.__dict__,
            "seed": seed,
            "float_tensors": {name: f"{name}.bin" for name in FLOAT_TENSORS},
            "int_tensors": {name: f"{name}.json" for name in INT_TENSORS},
            "tensor_shapes": shapes,
            "layout": "token-major contiguous float32 unless noted",
            "semantics": {
                "delta_rule": "S_t = S_{t-1} + beta_t * k_t^T * (v_t - k_t @ S_{t-1}); core_t = q_t @ S_t; q/k heads repeat onto value heads when value_heads > heads",
                "delta_state_layout": "[value_heads, value_dim, key_dim]",
                "delta_output": "matmul(delta_core * sigmoid(x @ w_delta_gate), w_o)",
                "router": "stable top-k by descending logit then ascending expert id; weights are softmax over selected logits",
                "expert": "down(silu(x @ gate) * (x @ up))",
                "moe_merge": "routed_topk_expert_sum + shared_expert_output",
                "layer_output": "input + delta_output + moe_output",
            },
        },
    )


def read_dump(path: Path) -> dict[str, Any]:
    manifest = load_json(path / DUMP_MANIFEST_NAME)
    shapes = manifest.get("tensor_shapes")
    if not isinstance(shapes, dict):
        raise CompareError(f"{path} dump manifest missing tensor_shapes")
    float_tensors = {}
    for name in FLOAT_TENSORS:
        rel = manifest.get("float_tensors", {}).get(name, f"{name}.bin")
        values = read_f32(path / rel)
        dims = shapes.get(name)
        if not isinstance(dims, list):
            raise CompareError(f"{path} dump manifest missing shape for {name}")
        expected = count_from_shape([int(dim) for dim in dims])
        if len(values) != expected:
            raise CompareError(f"{path}/{rel} length {len(values)} != manifest count {expected}")
        float_tensors[name] = values
    int_tensors = {}
    for name in INT_TENSORS:
        rel = manifest.get("int_tensors", {}).get(name, f"{name}.json")
        values = load_json(path / rel)
        if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
            raise CompareError(f"{path}/{rel} must be a JSON integer list")
        dims = shapes.get(name)
        if not isinstance(dims, list):
            raise CompareError(f"{path} dump manifest missing shape for {name}")
        expected = count_from_shape([int(dim) for dim in dims])
        if len(values) != expected:
            raise CompareError(f"{path}/{rel} length {len(values)} != manifest count {expected}")
        int_tensors[name] = values
    return {
        "manifest": manifest,
        "float_tensors": float_tensors,
        "int_tensors": int_tensors,
    }


def error_stats(expected: list[float], actual: list[float]) -> dict[str, float]:
    if len(expected) != len(actual):
        raise CompareError(f"length mismatch: expected {len(expected)}, got {len(actual)}")
    max_abs = 0.0
    max_rel = 0.0
    sum_abs = 0.0
    rmse_acc = 0.0
    for exp, got in zip(expected, actual):
        diff = abs(exp - got)
        max_abs = max(max_abs, diff)
        denom = max(abs(exp), 1e-12)
        max_rel = max(max_rel, diff / denom)
        sum_abs += diff
        rmse_acc += diff * diff
    n = max(len(expected), 1)
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mean_abs": sum_abs / n,
        "rmse": math.sqrt(rmse_acc / n),
    }


def compare_dumps(reference_dir: Path, ferrum_dir: Path, *, atol: float) -> dict[str, Any]:
    reference = read_dump(reference_dir)
    ferrum = read_dump(ferrum_dir)
    ref_shapes = reference["manifest"].get("tensor_shapes")
    ferrum_shapes = ferrum["manifest"].get("tensor_shapes")
    if ref_shapes != ferrum_shapes:
        raise CompareError("reference and Ferrum tensor_shapes differ")
    comparisons = {}
    failures = []
    for name in FLOAT_TENSORS:
        stats = error_stats(reference["float_tensors"][name], ferrum["float_tensors"][name])
        passed = stats["max_abs"] <= atol
        comparisons[name] = {
            **stats,
            "atol": atol,
            "status": "pass" if passed else "fail",
        }
        if not passed:
            failures.append(f"{name} max_abs {stats['max_abs']} > {atol}")
    for name in INT_TENSORS:
        expected = reference["int_tensors"][name]
        actual = ferrum["int_tensors"][name]
        mismatches = [
            idx
            for idx, (exp, got) in enumerate(zip(expected, actual))
            if exp != got
        ]
        comparisons[name] = {
            "mismatches": len(mismatches),
            "first_mismatch": mismatches[0] if mismatches else None,
            "status": "pass" if not mismatches else "fail",
        }
        if mismatches:
            failures.append(f"{name} has {len(mismatches)} mismatches")
    if failures:
        raise CompareError("; ".join(failures))
    return {
        "reference_manifest": reference["manifest"],
        "ferrum_manifest": ferrum["manifest"],
        "comparisons": comparisons,
    }


def shape_from_args(args: argparse.Namespace) -> Shape:
    value_heads = args.value_heads if args.value_heads is not None else args.heads
    shape = Shape(
        tokens=args.tokens,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        value_heads=value_heads,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        experts=args.experts,
        top_k=args.top_k,
        expert_hidden_dim=args.expert_hidden_dim,
    )
    shape.validate()
    return shape


def run_selftest(args: argparse.Namespace) -> int:
    shape = shape_from_args(args)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = out_dir / "reference_dump"
    ferrum_dir = out_dir / "synthetic_ferrum_dump"
    if reference_dir.exists():
        shutil.rmtree(reference_dir)
    if ferrum_dir.exists():
        shutil.rmtree(ferrum_dir)
    tensors = compute_reference(shape, args.seed)
    write_dump(reference_dir, shape, args.seed, tensors, producer="internal-python-reference")
    write_dump(ferrum_dir, shape, args.seed, tensors, producer="synthetic-ferrum-selftest")
    result = compare_dumps(reference_dir, ferrum_dir, atol=args.atol)
    pass_line = f"{PASS_SELFTEST}: {out_dir}"
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "mode": "self-test",
        "pass_line": pass_line,
        "created_at": iso_now(),
        "command_line": command_line(),
        "git": git_summary(),
        "shape": shape.__dict__,
        "seed": args.seed,
        "tolerance": {"max_abs": args.atol},
        "reference_dump": str(reference_dir),
        "ferrum_dump": str(ferrum_dir),
        "checks": {
            "delta_rule": "pass",
            "deltanet_layer": "pass",
            "router_topk": "pass",
            "expert_layout": "pass",
            "shared_expert_merge": "pass",
        },
        "comparisons": result["comparisons"],
        "note": "self-test validates the S1 dump contract only; it is not real-model W3-S1 evidence",
    }
    write_json(out_dir / MANIFEST_NAME, manifest)
    print(pass_line)
    return 0


def run_compare(args: argparse.Namespace) -> int:
    if args.reference_dump is None or args.ferrum_dump is None:
        raise CompareError("--compare requires --reference-dump and --ferrum-dump")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = Path(args.reference_dump).resolve()
    ferrum_dir = Path(args.ferrum_dump).resolve()
    result = compare_dumps(reference_dir, ferrum_dir, atol=args.atol)
    pass_line = f"{PASS_COMPARE}: {out_dir}"
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "mode": "compare",
        "pass_line": pass_line,
        "created_at": iso_now(),
        "command_line": command_line(),
        "git": git_summary(),
        "tolerance": {"max_abs": args.atol},
        "reference_dump": str(reference_dir),
        "ferrum_dump": str(ferrum_dir),
        "reference_manifest": result["reference_manifest"],
        "ferrum_manifest": result["ferrum_manifest"],
        "checks": {
            "delta_rule": "pass",
            "deltanet_layer": "pass",
            "router_topk": "pass",
            "expert_layout": "pass",
            "shared_expert_merge": "pass",
        },
        "comparisons": result["comparisons"],
        "note": "real W3-S1 evidence only if reference_dump is from official/HF reference and ferrum_dump is product-generated",
    }
    write_json(out_dir / MANIFEST_NAME, manifest)
    print(pass_line)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--compare", action="store_true")
    parser.add_argument("--out", default="target/w3_deltanet_s1_layer_selftest")
    parser.add_argument("--reference-dump")
    parser.add_argument("--ferrum-dump")
    parser.add_argument("--tokens", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--value-heads", type=int)
    parser.add_argument("--key-dim", type=int, default=3)
    parser.add_argument("--value-dim", type=int, default=4)
    parser.add_argument("--experts", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-hidden-dim", type=int, default=6)
    parser.add_argument("--seed", type=int, default=9271)
    parser.add_argument("--atol", type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            return run_selftest(args)
        return run_compare(args)
    except CompareError as exc:
        print(f"W3 DELTANET S1 LAYER COMPARE FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
