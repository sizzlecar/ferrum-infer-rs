#!/usr/bin/env python3
"""Probe real Qwen3.5 GPTQ weight metadata without downloading model shards.

The W3 goal requires real safetensors-index evidence before changing loader
assumptions. This script fetches only small Hugging Face metadata files, or
reads local copies, then checks whether Ferrum's Qwen3.5 manifest can resolve
the real checkpoint's dense/GPTQ tensor names.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
DEFAULT_REVISION = "3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b"
PASS_LINE_PREFIX = "W3 QWEN35 WEIGHT INDEX PROBE PASS"
SELF_TEST_PASS_LINE = "W3 QWEN35 WEIGHT INDEX PROBE SELF-TEST PASS"
SUMMARY_NAME = "w3_qwen35_weight_index_probe.json"
PREFIX_CANDIDATES = ("model.language_model", "model")
GPTQ_SUFFIXES = ("qweight", "scales", "qzeros")
LINEAR_GPTQ_ROLES = {
    "lm_head",
    "linear_attn_qkv",
    "linear_attn_z",
    "linear_attn_b",
    "linear_attn_a",
    "linear_attn_out",
    "self_attn_q",
    "self_attn_k",
    "self_attn_v",
    "self_attn_o",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
    "moe_router",
    "moe_shared_expert_gate",
    "moe_shared_expert_gate_proj",
    "moe_shared_expert_up_proj",
    "moe_shared_expert_down_proj",
}


class ProbeError(Exception):
    pass


@dataclass(frozen=True)
class WeightSpec:
    role: str
    name: str
    required: bool


@dataclass(frozen=True)
class LayerManifest:
    layer_index: int
    attention: str
    mlp: str
    tensors: list[WeightSpec]


@dataclass(frozen=True)
class TextConfig:
    top_level_model_type: str | None
    text_model_type: str
    hidden_size: int
    num_hidden_layers: int
    layer_types: list[str]
    tie_word_embeddings: bool
    num_experts: int | None
    num_experts_per_tok: int | None
    moe_intermediate_size: int | None
    shared_expert_intermediate_size: int | None
    quantization_config: dict[str, Any] | None

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None


class Fetcher:
    def __init__(self, repo: str, revision: str, timeout: float, retries: int) -> None:
        self.repo = repo
        self.revision = revision
        self.timeout = timeout
        self.retries = retries
        self.bytes_read = 0
        self.requests = 0
        self.hf_token_present = bool(os.environ.get("HF_TOKEN"))

    def url(self, path: str) -> str:
        repo = urllib.parse.quote(self.repo, safe="/")
        revision = urllib.parse.quote(self.revision, safe="")
        parts = [urllib.parse.quote(part, safe="") for part in path.split("/")]
        return f"https://huggingface.co/{repo}/resolve/{revision}/{'/'.join(parts)}"

    def request(self, path: str) -> bytes:
        url = self.url(path)
        headers = {"User-Agent": "ferrum-w3-qwen35-index-probe/1.0"}
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
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
                urllib.error.HTTPError,
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
        raise ProbeError(f"failed to fetch {path}: {last_error}")


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def git_output(args: list[str], default: str = "unknown") -> str:
    import subprocess

    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else default


def git_summary() -> dict[str, Any]:
    tracked = [
        line
        for line in git_output(["status", "--short", "--untracked-files=no"], "").splitlines()
        if line.strip()
    ]
    untracked = [
        line
        for line in git_output(["ls-files", "--others", "--exclude-standard"], "").splitlines()
        if line.strip()
    ]
    return {
        "sha": git_output(["rev-parse", "HEAD"]),
        "is_dirty": bool(tracked or untracked),
        "tracked_status_short": tracked,
        "untracked_count": len(untracked),
        "untracked_sample": untracked[:20],
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProbeError(f"read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProbeError(f"parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ProbeError(f"{path} must be a JSON object")
    return data


def load_json_sources(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    fetcher: Fetcher | None = None
    source: dict[str, Any] = {
        "model_id": args.model_id,
        "revision": args.revision,
        "config_json": str(args.config_json) if args.config_json else None,
        "index_json": str(args.index_json) if args.index_json else None,
        "quantize_config_json": str(args.quantize_config_json) if args.quantize_config_json else None,
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
    }

    if args.config_json:
        config = read_json(args.config_json)
    else:
        fetcher = fetcher or Fetcher(args.model_id, args.revision, args.timeout, args.retries)
        config = json.loads(fetcher.request("config.json").decode("utf-8"))

    if args.index_json:
        index = read_json(args.index_json)
    else:
        fetcher = fetcher or Fetcher(args.model_id, args.revision, args.timeout, args.retries)
        index = json.loads(fetcher.request("model.safetensors.index.json").decode("utf-8"))

    if args.quantize_config_json:
        quantize_config = read_json(args.quantize_config_json)
    else:
        fetcher = fetcher or Fetcher(args.model_id, args.revision, args.timeout, args.retries)
        try:
            quantize_config = json.loads(fetcher.request("quantize_config.json").decode("utf-8"))
        except ProbeError:
            quantize_config = None

    if fetcher is not None:
        source["remote_requests"] = fetcher.requests
        source["remote_bytes_read"] = fetcher.bytes_read
    return config, index, quantize_config, source


def required_int(obj: dict[str, Any], key: str) -> int:
    value = obj.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ProbeError(f"{key} must be a positive integer")
    return value


def parse_text_config(root: dict[str, Any]) -> TextConfig:
    text = root.get("text_config", root)
    if not isinstance(text, dict):
        raise ProbeError("config text_config must be an object")
    text_model_type = text.get("model_type")
    if text_model_type not in {"qwen3_5_text", "qwen3_5_moe_text"}:
        raise ProbeError(f"unsupported Qwen3.5 text model_type: {text_model_type!r}")
    layers = text.get("layer_types")
    if not isinstance(layers, list) or not layers:
        raise ProbeError("layer_types must be a non-empty array")
    layer_types = []
    for raw in layers:
        if raw in {"linear_attention", "linear"}:
            layer_types.append("linear_attention")
        elif raw in {"full_attention", "full"}:
            layer_types.append("full_attention")
        else:
            raise ProbeError(f"unsupported layer type {raw!r}")
    num_hidden_layers = required_int(text, "num_hidden_layers")
    if len(layer_types) != num_hidden_layers:
        raise ProbeError(
            f"layer_types length {len(layer_types)} does not match num_hidden_layers {num_hidden_layers}"
        )
    tie_word_embeddings = text.get(
        "tie_word_embeddings",
        root.get("tie_word_embeddings", False),
    )
    if not isinstance(tie_word_embeddings, bool):
        raise ProbeError("tie_word_embeddings must be a boolean when present")

    is_moe = text_model_type == "qwen3_5_moe_text"
    return TextConfig(
        top_level_model_type=root.get("model_type") if isinstance(root.get("model_type"), str) else None,
        text_model_type=text_model_type,
        hidden_size=required_int(text, "hidden_size"),
        num_hidden_layers=num_hidden_layers,
        layer_types=layer_types,
        tie_word_embeddings=tie_word_embeddings,
        num_experts=required_int(text, "num_experts") if is_moe else None,
        num_experts_per_tok=required_int(text, "num_experts_per_tok") if is_moe else None,
        moe_intermediate_size=required_int(text, "moe_intermediate_size") if is_moe else None,
        shared_expert_intermediate_size=required_int(text, "shared_expert_intermediate_size") if is_moe else None,
        quantization_config=root.get("quantization_config")
        if isinstance(root.get("quantization_config"), dict)
        else None,
    )


def weight_spec(role: str, name: str, required: bool = True) -> WeightSpec:
    return WeightSpec(role=role, name=name, required=required)


def layer_specs(config: TextConfig, prefix: str) -> list[LayerManifest]:
    layers = []
    for idx, attention in enumerate(config.layer_types):
        layer_prefix = f"{prefix}.layers.{idx}"
        tensors = [
            weight_spec("input_layernorm", f"{layer_prefix}.input_layernorm.weight"),
            weight_spec("post_attention_layernorm", f"{layer_prefix}.post_attention_layernorm.weight"),
        ]
        if attention == "linear_attention":
            tensors.extend(
                weight_spec(role, f"{layer_prefix}.{suffix}")
                for role, suffix in [
                    ("linear_attn_qkv", "linear_attn.in_proj_qkv.weight"),
                    ("linear_attn_z", "linear_attn.in_proj_z.weight"),
                    ("linear_attn_b", "linear_attn.in_proj_b.weight"),
                    ("linear_attn_a", "linear_attn.in_proj_a.weight"),
                    ("linear_attn_conv", "linear_attn.conv1d.weight"),
                    ("linear_attn_a_log", "linear_attn.A_log"),
                    ("linear_attn_dt_bias", "linear_attn.dt_bias"),
                    ("linear_attn_norm", "linear_attn.norm.weight"),
                    ("linear_attn_out", "linear_attn.out_proj.weight"),
                ]
            )
        else:
            tensors.extend(
                weight_spec(role, f"{layer_prefix}.{suffix}")
                for role, suffix in [
                    ("self_attn_q", "self_attn.q_proj.weight"),
                    ("self_attn_k", "self_attn.k_proj.weight"),
                    ("self_attn_v", "self_attn.v_proj.weight"),
                    ("self_attn_o", "self_attn.o_proj.weight"),
                    ("self_attn_q_norm", "self_attn.q_norm.weight"),
                    ("self_attn_k_norm", "self_attn.k_norm.weight"),
                ]
            )
        if config.is_moe:
            tensors.extend(
                [
                    weight_spec("moe_router", f"{layer_prefix}.mlp.gate.weight"),
                    weight_spec(
                        "moe_shared_expert_gate",
                        f"{layer_prefix}.mlp.shared_expert_gate.weight",
                    ),
                    weight_spec(
                        "moe_shared_expert_gate_proj",
                        f"{layer_prefix}.mlp.shared_expert.gate_proj.weight",
                    ),
                    weight_spec(
                        "moe_shared_expert_up_proj",
                        f"{layer_prefix}.mlp.shared_expert.up_proj.weight",
                    ),
                    weight_spec(
                        "moe_shared_expert_down_proj",
                        f"{layer_prefix}.mlp.shared_expert.down_proj.weight",
                    ),
                    weight_spec(
                        "moe_fused_gate_up_proj",
                        f"{layer_prefix}.mlp.experts.gate_up_proj",
                        False,
                    ),
                    weight_spec(
                        "moe_fused_down_proj",
                        f"{layer_prefix}.mlp.experts.down_proj",
                        False,
                    ),
                    weight_spec(
                        "moe_per_expert_gate_proj_qweight",
                        f"{layer_prefix}.mlp.experts.*.gate_proj.qweight",
                        False,
                    ),
                    weight_spec(
                        "moe_per_expert_up_proj_qweight",
                        f"{layer_prefix}.mlp.experts.*.up_proj.qweight",
                        False,
                    ),
                    weight_spec(
                        "moe_per_expert_down_proj_qweight",
                        f"{layer_prefix}.mlp.experts.*.down_proj.qweight",
                        False,
                    ),
                ]
            )
            mlp = "sparse_moe_shared_expert"
        else:
            tensors.extend(
                weight_spec(role, f"{layer_prefix}.{suffix}")
                for role, suffix in [
                    ("mlp_gate", "mlp.gate_proj.weight"),
                    ("mlp_up", "mlp.up_proj.weight"),
                    ("mlp_down", "mlp.down_proj.weight"),
                ]
            )
            mlp = "dense"
        layers.append(LayerManifest(layer_index=idx, attention=attention, mlp=mlp, tensors=tensors))
    return layers


def manifest(config: TextConfig, prefix: str) -> tuple[list[WeightSpec], list[LayerManifest]]:
    globals_ = [
        weight_spec("embed_tokens", f"{prefix}.embed_tokens.weight"),
        weight_spec("final_norm", f"{prefix}.norm.weight"),
        weight_spec("lm_head", f"{prefix}.lm_head.weight", not config.tie_word_embeddings),
    ]
    return globals_, layer_specs(config, prefix)


def qweight_for_dense(names: set[str], dense_name: str) -> str | None:
    module = dense_name.removesuffix(".weight")
    qweight = f"{module}.qweight"
    if all(f"{module}.{suffix}" in names for suffix in GPTQ_SUFFIXES):
        return qweight
    return None


def wildcard_matches(names: set[str], pattern: str) -> list[str]:
    escaped = re.escape(pattern).replace("\\*", r"[^.]+")
    regex = re.compile(f"^{escaped}$")
    return sorted(name for name in names if regex.match(name))


def matching_names(names: set[str], spec: WeightSpec) -> list[str]:
    if "*" in spec.name:
        return wildcard_matches(names, spec.name)
    candidates = [spec.name]
    if spec.name.endswith(".lm_head.weight"):
        candidates.append("lm_head.weight")
    for candidate in candidates:
        if candidate in names:
            return [candidate]
        if spec.role in LINEAR_GPTQ_ROLES:
            qweight = qweight_for_dense(names, candidate)
            if qweight:
                return [qweight]
    return []


def resolution_kind(declared: str, resolved: str) -> str:
    if resolved.endswith(".qweight"):
        return "gptq_qweight"
    if declared.endswith(".lm_head.weight") and resolved == "lm_head.weight":
        return "top_level_lm_head_weight"
    if resolved.endswith(".weight"):
        return "dense_weight"
    return "non_linear_tensor"


def validate_prefix(config: TextConfig, names: set[str], prefix: str) -> dict[str, Any]:
    globals_, layers = manifest(config, prefix)
    missing_required = []
    present_required = []
    present_optional = []
    missing_optional = []
    resolved_samples = []
    required_kind_counts: dict[str, int] = {}
    optional_kind_counts: dict[str, int] = {}
    for spec in globals_ + [tensor for layer in layers for tensor in layer.tensors]:
        matches = matching_names(names, spec)
        if spec.required and matches:
            present_required.append(matches[0])
            kind = resolution_kind(spec.name, matches[0])
            required_kind_counts[kind] = required_kind_counts.get(kind, 0) + 1
        elif spec.required:
            missing_required.append(spec.name)
        elif matches:
            present_optional.extend(matches)
            for match in matches:
                kind = resolution_kind(spec.name, match)
                optional_kind_counts[kind] = optional_kind_counts.get(kind, 0) + 1
        else:
            missing_optional.append(spec.name)
        if matches and len(resolved_samples) < 24:
            resolved_samples.append(
                {
                    "role": spec.role,
                    "declared": spec.name,
                    "resolved": matches[0],
                    "required": spec.required,
                }
            )
    return {
        "prefix": prefix,
        "missing_required": missing_required,
        "present_required_count": len(present_required),
        "present_optional_count": len(present_optional),
        "missing_optional_count": len(missing_optional),
        "required_resolution_kind_counts": required_kind_counts,
        "optional_resolution_kind_counts": optional_kind_counts,
        "resolved_samples": resolved_samples,
    }


def select_prefix(config: TextConfig, names: set[str]) -> dict[str, Any]:
    candidates = [validate_prefix(config, names, prefix) for prefix in PREFIX_CANDIDATES]
    selected = next((item for item in candidates if not item["missing_required"]), None)
    if selected is None:
        selected = min(candidates, key=lambda item: len(item["missing_required"]))
    return {
        "selected_prefix": selected["prefix"],
        "selected_missing_required": selected["missing_required"],
        "candidates": [
            {
                "prefix": item["prefix"],
                "missing_required_count": len(item["missing_required"]),
                "present_required_count": item["present_required_count"],
                "present_optional_count": item["present_optional_count"],
                "required_resolution_kind_counts": item["required_resolution_kind_counts"],
                "optional_resolution_kind_counts": item["optional_resolution_kind_counts"],
                "missing_required_sample": item["missing_required"][:20],
                "resolved_samples": item["resolved_samples"],
            }
            for item in candidates
        ],
    }


def quantization_summary(config: TextConfig, quantize_config: dict[str, Any] | None) -> dict[str, Any]:
    source = "quantize_config.json" if quantize_config is not None else "config.quantization_config"
    raw = quantize_config if quantize_config is not None else config.quantization_config
    if raw is None:
        return {"source": None, "pass": False, "problems": ["missing GPTQ quantization config"]}
    method = str(raw.get("quant_method", raw.get("method", ""))).lower()
    summary = {
        "source": source,
        "quant_method": method,
        "bits": raw.get("bits"),
        "group_size": raw.get("group_size"),
        "desc_act": raw.get("desc_act"),
        "sym": raw.get("sym"),
        "pass": True,
        "problems": [],
    }
    expected = {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
    }
    for key, expected_value in expected.items():
        if summary[key] != expected_value:
            summary["pass"] = False
            summary["problems"].append(
                f"{key}={summary[key]!r} does not match expected {expected_value!r}"
            )
    return summary


def expert_name(prefix: str, layer: int, expert: int, proj: str, suffix: str) -> str:
    return f"{prefix}.layers.{layer}.mlp.experts.{expert}.{proj}.{suffix}"


def expert_coverage(config: TextConfig, names: set[str], prefix: str) -> dict[str, Any]:
    if not config.is_moe:
        return {"mode": "dense", "pass": True, "layers": []}
    assert config.num_experts is not None
    layers = []
    g_idx_modes: dict[str, int] = {}
    for layer in range(config.num_hidden_layers):
        fused_gate_up = f"{prefix}.layers.{layer}.mlp.experts.gate_up_proj" in names
        fused_down = f"{prefix}.layers.{layer}.mlp.experts.down_proj" in names
        missing = []
        layer_missing_count = 0
        g_idx_count = 0
        for expert in range(config.num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for suffix in GPTQ_SUFFIXES:
                    name = expert_name(prefix, layer, expert, proj, suffix)
                    if name not in names:
                        layer_missing_count += 1
                        if len(missing) < 24:
                            missing.append(name)
                if expert_name(prefix, layer, expert, proj, "g_idx") in names:
                    g_idx_count += 1
        if g_idx_count == 0:
            g_idx_mode = "none"
        elif g_idx_count == config.num_experts * 3:
            g_idx_mode = "all"
        else:
            g_idx_mode = "partial"
        g_idx_modes[g_idx_mode] = g_idx_modes.get(g_idx_mode, 0) + 1
        per_expert_gptq_complete = layer_missing_count == 0
        fused_dense_complete = fused_gate_up and fused_down
        layers.append(
            {
                "layer_index": layer,
                "per_expert_gptq_complete": per_expert_gptq_complete,
                "fused_dense_complete": fused_dense_complete,
                "g_idx_mode": g_idx_mode,
                "missing_count": layer_missing_count,
                "missing_sample": missing[:12],
                "supported_by_loader": per_expert_gptq_complete or fused_dense_complete,
            }
        )
    unsupported = [layer for layer in layers if not layer["supported_by_loader"]]
    return {
        "mode": "sparse_moe_shared_expert",
        "num_experts": config.num_experts,
        "layers_checked": config.num_hidden_layers,
        "per_expert_gptq_tensors_checked": config.num_hidden_layers * config.num_experts * 3 * 3,
        "g_idx_modes": g_idx_modes,
        "unsupported_layers": [layer["layer_index"] for layer in unsupported],
        "unsupported_layer_sample": unsupported[:4],
        "pass": not unsupported,
        "layers_sample": layers[:8],
    }


def index_summary(index: dict[str, Any]) -> tuple[set[str], dict[str, Any]]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ProbeError("model.safetensors.index.json missing weight_map")
    names = {name for name in weight_map.keys() if isinstance(name, str)}
    shards = {shard for shard in weight_map.values() if isinstance(shard, str)}
    return names, {
        "tensor_count": len(names),
        "shard_count": len(shards),
        "metadata": index.get("metadata") if isinstance(index.get("metadata"), dict) else {},
        "shard_sample": sorted(shards)[:8],
    }


def run_probe(
    *,
    config_json: dict[str, Any],
    index_json: dict[str, Any],
    quantize_config_json: dict[str, Any] | None,
    source: dict[str, Any],
) -> dict[str, Any]:
    config = parse_text_config(config_json)
    names, index_meta = index_summary(index_json)
    prefix_validation = select_prefix(config, names)
    selected_prefix = prefix_validation["selected_prefix"]
    quant = quantization_summary(config, quantize_config_json)
    experts = expert_coverage(config, names, selected_prefix)
    problems = []
    if prefix_validation["selected_missing_required"]:
        problems.append(
            f"missing {len(prefix_validation['selected_missing_required'])} required manifest tensors"
        )
    if not quant["pass"]:
        problems.extend(quant["problems"])
    if not experts["pass"]:
        problems.append(
            f"unsupported sparse MoE expert layout on layers {experts['unsupported_layers'][:8]}"
        )
    status = "pass" if not problems else "fail"
    return {
        "status": status,
        "generated_at": iso_now(),
        "goal_doc": "docs/goals/model-coverage-2026-06-12/W3_QWEN35_RELEASE_GRADE_GOAL.md",
        "command_line": [sys.executable, *sys.argv],
        "git": git_summary(),
        "source": source,
        "config_summary": {
            "top_level_model_type": config.top_level_model_type,
            "text_model_type": config.text_model_type,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "linear_attention_layers": config.layer_types.count("linear_attention"),
            "full_attention_layers": config.layer_types.count("full_attention"),
            "layer_pattern_sample": config.layer_types[:12],
            "tie_word_embeddings": config.tie_word_embeddings,
            "num_experts": config.num_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "moe_intermediate_size": config.moe_intermediate_size,
            "shared_expert_intermediate_size": config.shared_expert_intermediate_size,
        },
        "quantization": quant,
        "index_summary": index_meta,
        "prefix_validation": prefix_validation,
        "expert_coverage": experts,
        "problems": problems,
    }


def write_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SUMMARY_NAME
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_triplet(names: set[str], module: str) -> None:
    for suffix in GPTQ_SUFFIXES:
        names.add(f"{module}.{suffix}")


def synthetic_config() -> dict[str, Any]:
    return {
        "model_type": "qwen3_5_moe",
        "tie_word_embeddings": False,
        "quantization_config": {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
            "sym": True,
        },
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "head_dim": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
        },
    }


def synthetic_index(remove_name: str | None = None) -> dict[str, Any]:
    config = parse_text_config(synthetic_config())
    names = {"model.embed_tokens.weight", "model.norm.weight"}
    add_triplet(names, "model.lm_head")
    globals_, layers = manifest(config, "model")
    for spec in globals_ + [tensor for layer in layers for tensor in layer.tensors]:
        if spec.role in LINEAR_GPTQ_ROLES:
            add_triplet(names, spec.name.removesuffix(".weight"))
        elif spec.required:
            names.add(spec.name)
    for layer in range(config.num_hidden_layers):
        for expert in range(config.num_experts or 0):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                add_triplet(names, f"model.layers.{layer}.mlp.experts.{expert}.{proj}")
    if remove_name is not None:
        names.discard(remove_name)
    return {
        "metadata": {"total_size": "0"},
        "weight_map": {name: "model-00001-of-00001.safetensors" for name in sorted(names)},
    }


def run_self_test() -> None:
    source = {
        "model_id": "synthetic",
        "revision": "synthetic",
        "hf_token_present": False,
    }
    ok = run_probe(
        config_json=synthetic_config(),
        index_json=synthetic_index(),
        quantize_config_json=None,
        source=source,
    )
    if ok["status"] != "pass":
        raise ProbeError(f"synthetic positive case failed: {ok['problems']}")
    missing = run_probe(
        config_json=synthetic_config(),
        index_json=synthetic_index(
            "model.layers.0.mlp.shared_expert_gate.qzeros",
        ),
        quantize_config_json=None,
        source=source,
    )
    if missing["status"] != "fail":
        raise ProbeError("synthetic incomplete GPTQ triplet unexpectedly passed")
    if not any("missing" in problem for problem in missing["problems"]):
        raise ProbeError(f"synthetic failure did not report missing tensors: {missing['problems']}")
    print(SELF_TEST_PASS_LINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--index-json", type=Path)
    parser.add_argument("--quantize-config-json", type=Path)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            run_self_test()
            return 0
        if args.out is None:
            raise ProbeError("--out is required unless --self-test is used")
        config, index, quantize_config, source = load_json_sources(args)
        summary = run_probe(
            config_json=config,
            index_json=index,
            quantize_config_json=quantize_config,
            source=source,
        )
        write_summary(args.out, summary)
        if summary["status"] != "pass":
            print(f"W3 QWEN35 WEIGHT INDEX PROBE FAIL: {args.out}")
            for problem in summary["problems"]:
                print(f"- {problem}")
            return 1
        pass_line = f"{PASS_LINE_PREFIX}: {args.out}"
        summary["pass_line"] = pass_line
        write_summary(args.out, summary)
        print(pass_line)
        return 0
    except ProbeError as exc:
        if not args.self_test and args.out is not None:
            failure = {
                "status": "fail",
                "generated_at": iso_now(),
                "command_line": [sys.executable, *sys.argv],
                "git": git_summary(),
                "source": {
                    "model_id": args.model_id,
                    "revision": args.revision,
                    "hf_token_present": bool(os.environ.get("HF_TOKEN")),
                },
                "problems": [str(exc)],
            }
            write_summary(args.out, failure)
            print(f"W3 QWEN35 WEIGHT INDEX PROBE FAIL: {args.out}")
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
