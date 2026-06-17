#!/usr/bin/env python3
"""W3 official/HF config probe for Gated DeltaNet reference selection.

This is a metadata-only gate. It downloads Hugging Face `config.json` files,
extracts the nested `text_config`, and validates the fields needed before W3-S1
can move from deterministic single-layer evidence to official/HF layer dumps.
It intentionally does not download model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS_LINE = "W3 HF CONFIG PROBE PASS"
MANIFEST_NAME = "w3_hf_config_probe_manifest.json"


class ProbeError(Exception):
    pass


@dataclass(frozen=True)
class ModelProbe:
    role: str
    model_id: str
    expected_model_type: str
    expect_moe: bool
    min_linear_layers: int
    min_full_attention_layers: int

    @property
    def source_url(self) -> str:
        return f"https://huggingface.co/{self.model_id}/raw/main/config.json"


DEFAULT_MODELS = [
    ModelProbe(
        role="dense_min_reference",
        model_id="Qwen/Qwen3.5-0.8B",
        expected_model_type="qwen3_5_text",
        expect_moe=False,
        min_linear_layers=1,
        min_full_attention_layers=1,
    ),
    ModelProbe(
        role="dense_4b_reference",
        model_id="Qwen/Qwen3.5-4B",
        expected_model_type="qwen3_5_text",
        expect_moe=False,
        min_linear_layers=1,
        min_full_attention_layers=1,
    ),
    ModelProbe(
        role="moe_shared_expert_reference",
        model_id="Qwen/Qwen3.6-35B-A3B",
        expected_model_type="qwen3_5_moe_text",
        expect_moe=True,
        min_linear_layers=1,
        min_full_attention_layers=1,
    ),
]


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def command_line() -> list[str]:
    return [sys.executable, *sys.argv]


def run_git(args: list[str]) -> str:
    import subprocess

    proc = subprocess.run(
        ["git", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def git_summary() -> dict[str, Any]:
    tracked = [
        line
        for line in run_git(["status", "--short", "--untracked-files=no"]).splitlines()
        if line.strip()
    ]
    untracked = [
        line
        for line in run_git(["ls-files", "--others", "--exclude-standard"]).splitlines()
        if line.strip()
    ]
    return {
        "sha": run_git(["rev-parse", "HEAD"]),
        "is_dirty": bool(tracked or untracked),
        "tracked_status_short": tracked,
        "untracked_count": len(untracked),
        "untracked_sample": untracked[:20],
    }


def fetch_json(url: str) -> tuple[dict[str, Any], str]:
    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            raw = response.read()
    except Exception as exc:  # pragma: no cover - exact network errors vary
        raise ProbeError(f"failed to fetch {url}: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()
    except json.JSONDecodeError as exc:
        raise ProbeError(f"invalid JSON from {url}: {exc}") from exc


def nested_text_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise ProbeError("config.text_config must be an object")
    return text_config


def require_int(cfg: dict[str, Any], key: str, problems: list[str]) -> int | None:
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        problems.append(f"{key} must be a positive integer, got {value!r}")
        return None
    return value


def summarize_layer_types(cfg: dict[str, Any], expected_len: int | None, problems: list[str]) -> dict[str, Any]:
    layer_types = cfg.get("layer_types")
    if not isinstance(layer_types, list) or not all(isinstance(item, str) for item in layer_types):
        problems.append("layer_types must be a string list")
        return {"count": 0, "linear_attention": 0, "full_attention": 0, "prefix": []}
    if expected_len is not None and len(layer_types) != expected_len:
        problems.append(f"layer_types length {len(layer_types)} != num_hidden_layers {expected_len}")
    return {
        "count": len(layer_types),
        "linear_attention": layer_types.count("linear_attention"),
        "full_attention": layer_types.count("full_attention"),
        "prefix": layer_types[:12],
    }


def validate_model(probe: ModelProbe, raw_config: dict[str, Any], raw_sha256: str) -> dict[str, Any]:
    cfg = nested_text_config(raw_config)
    problems: list[str] = []
    model_type = cfg.get("model_type")
    if model_type != probe.expected_model_type:
        problems.append(
            f"text_config.model_type {model_type!r} != expected {probe.expected_model_type!r}"
        )
    hidden_size = require_int(cfg, "hidden_size", problems)
    num_hidden_layers = require_int(cfg, "num_hidden_layers", problems)
    linear_key_head_dim = require_int(cfg, "linear_key_head_dim", problems)
    linear_value_head_dim = require_int(cfg, "linear_value_head_dim", problems)
    linear_num_key_heads = require_int(cfg, "linear_num_key_heads", problems)
    linear_num_value_heads = require_int(cfg, "linear_num_value_heads", problems)
    linear_conv_kernel_dim = require_int(cfg, "linear_conv_kernel_dim", problems)
    layer_summary = summarize_layer_types(cfg, num_hidden_layers, problems)
    if layer_summary["linear_attention"] < probe.min_linear_layers:
        problems.append("layer_types must include linear_attention layers")
    if layer_summary["full_attention"] < probe.min_full_attention_layers:
        problems.append("layer_types must include full_attention layers")

    moe_fields: dict[str, Any] = {}
    if probe.expect_moe:
        for key in ["num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size"]:
            moe_fields[key] = require_int(cfg, key, problems)
        if moe_fields.get("num_experts") is not None and moe_fields["num_experts"] < 2:
            problems.append("num_experts must be >= 2")
        if (
            moe_fields.get("num_experts_per_tok") is not None
            and moe_fields.get("num_experts") is not None
            and moe_fields["num_experts_per_tok"] > moe_fields["num_experts"]
        ):
            problems.append("num_experts_per_tok cannot exceed num_experts")
    else:
        for key in ["num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size"]:
            if cfg.get(key) is not None:
                problems.append(f"dense model should not define {key}")

    if problems:
        raise ProbeError(f"{probe.model_id}: " + "; ".join(problems))

    return {
        "role": probe.role,
        "model_id": probe.model_id,
        "source_url": probe.source_url,
        "raw_config_sha256": raw_sha256,
        "top_level_model_type": raw_config.get("model_type"),
        "text_model_type": model_type,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "layer_types": layer_summary,
        "linear_attention": {
            "linear_num_key_heads": linear_num_key_heads,
            "linear_num_value_heads": linear_num_value_heads,
            "linear_key_head_dim": linear_key_head_dim,
            "linear_value_head_dim": linear_value_head_dim,
            "linear_conv_kernel_dim": linear_conv_kernel_dim,
        },
        "moe": moe_fields if probe.expect_moe else None,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_probe(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for probe in DEFAULT_MODELS:
        raw_config, raw_sha256 = fetch_json(probe.source_url)
        write_json(out_dir / f"{probe.role}.config.json", raw_config)
        results.append(validate_model(probe, raw_config, raw_sha256))
    pass_line = f"{PASS_LINE}: {out_dir}"
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "pass_line": pass_line,
        "created_at": iso_now(),
        "command_line": command_line(),
        "git": git_summary(),
        "models": results,
        "note": "metadata-only probe; no model weights were downloaded",
    }
    write_json(out_dir / MANIFEST_NAME, manifest)
    print(pass_line)
    return 0


def run_self_test(args: argparse.Namespace) -> int:
    dense = {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
        },
    }
    moe = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 4,
            "shared_expert_intermediate_size": 4,
        },
    }
    validate_model(DEFAULT_MODELS[0], dense, "a" * 64)
    validate_model(DEFAULT_MODELS[2], moe, "b" * 64)
    bad = json.loads(json.dumps(moe))
    del bad["text_config"]["shared_expert_intermediate_size"]
    try:
        validate_model(DEFAULT_MODELS[2], bad, "c" * 64)
    except ProbeError as exc:
        if "shared_expert_intermediate_size" not in str(exc):
            raise
    else:
        raise ProbeError("self-test expected missing shared expert to fail")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pass_line = f"W3 HF CONFIG PROBE SELFTEST PASS: {out_dir}"
    write_json(
        out_dir / MANIFEST_NAME,
        {
            "schema_version": 1,
            "status": "pass",
            "pass_line": pass_line,
            "created_at": iso_now(),
            "command_line": command_line(),
        },
    )
    print(pass_line)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/w3_hf_config_probe")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            return run_self_test(args)
        return run_probe(args)
    except ProbeError as exc:
        print(f"W3 HF CONFIG PROBE FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
