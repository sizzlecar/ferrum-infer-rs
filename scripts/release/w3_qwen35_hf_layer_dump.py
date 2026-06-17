#!/usr/bin/env python3
"""W3 Qwen3.5/Qwen3.6 HF layer dump harness.

This script is the official/HF side of the W3-S1 correctness gate.  It has
three modes:

* `--self-test` validates the manifest/schema without torch or transformers.
* `--contract` validates a saved HF config plus the expected Transformers source
  hook contract, without downloading weights.
* `--dump` loads a real HF model with torch/transformers and writes a first
  linear-attention layer dump.

The dump mode is intentionally explicit and evidence-oriented: it records the
selected layer, module names, dependency versions, prompt, tensor shapes, and the
PASS line.  It is not a Ferrum-vs-HF comparator by itself; the corresponding
Ferrum dump still has to be generated and compared before W3-S1 can be called
real model evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS_SELFTEST = "W3 QWEN35 HF LAYER DUMP SELFTEST PASS"
PASS_CONTRACT = "W3 QWEN35 HF LAYER CONTRACT PASS"
PASS_DUMP = "W3 QWEN35 HF LAYER DUMP PASS"
MANIFEST_NAME = "w3_qwen35_hf_layer_dump_manifest.json"

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_PROMPT = "Paris is the capital of"
DEFAULT_DENSE_SOURCE_URL = (
    "https://raw.githubusercontent.com/huggingface/transformers/main/"
    "src/transformers/models/qwen3_5/modeling_qwen3_5.py"
)
DEFAULT_MOE_SOURCE_URL = (
    "https://raw.githubusercontent.com/huggingface/transformers/main/"
    "src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py"
)

COMMON_SOURCE_NEEDLES = [
    "self.in_proj_qkv = nn.Linear",
    "self.in_proj_z = nn.Linear",
    "self.in_proj_b = nn.Linear",
    "self.in_proj_a = nn.Linear",
    "self.out_proj = nn.Linear",
    "self.chunk_gated_delta_rule",
    "core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule",
    "core_attn_out = self.norm(core_attn_out, z)",
    "output = self.out_proj(core_attn_out)",
]

DENSE_SOURCE_NEEDLES = [
    "class Qwen3_5GatedDeltaNet",
    "self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)",
    "class Qwen3_5DecoderLayer",
    "self.mlp = Qwen3_5MLP(config, config.intermediate_size)",
]

MOE_SOURCE_NEEDLES = [
    "class Qwen3_5MoeGatedDeltaNet",
    "self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)",
    "class Qwen3_5MoeSparseMoeBlock",
    "self.shared_expert = Qwen3_5MoeMLP",
    "shared_expert_output = F.sigmoid(self.shared_expert_gate",
]

LAYER_DUMP_TENSORS = [
    "layer_input",
    "input_norm",
    "mixed_qkv_raw",
    "z_raw",
    "b_raw",
    "a_raw",
    "mixed_qkv_conv",
    "delta_q",
    "delta_k",
    "delta_v",
    "delta_beta",
    "delta_g",
    "delta_core",
    "delta_norm",
    "delta_output",
    "residual_after_mixer",
    "post_attention_norm",
    "mlp_output",
    "layer_output",
]

MOE_DUMP_TENSORS = [
    "router_logits",
    "router_topk_weights",
    "router_topk_indices",
    "routed_expert_output",
    "shared_expert_output",
    "moe_output",
]


class DumpError(Exception):
    pass


@dataclass(frozen=True)
class ConfigSummary:
    model_id: str
    text_model_type: str
    hidden_size: int
    num_hidden_layers: int
    layer_types: list[str]
    first_linear_layer: int
    linear_attention: dict[str, int]
    moe: dict[str, int] | None

    @property
    def is_moe(self) -> bool:
        return self.text_model_type == "qwen3_5_moe_text"


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
    tracked = [
        line
        for line in git_output(["status", "--short", "--untracked-files=no"], default="").splitlines()
        if line.strip()
    ]
    untracked = [
        line
        for line in git_output(["ls-files", "--others", "--exclude-standard"], default="").splitlines()
        if line.strip()
    ]
    return {
        "sha": git_output(["rev-parse", "HEAD"]),
        "is_dirty": bool(tracked or untracked),
        "tracked_status_short": tracked,
        "untracked_count": len(untracked),
        "untracked_sample": untracked[:20],
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DumpError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DumpError(f"invalid JSON in {path}: {exc}") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(data: str) -> str:
    return sha256_bytes(data.encode("utf-8"))


def fetch_text(url: str, *, timeout: int = 45) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except Exception as exc:  # pragma: no cover - exact network errors vary
        raise DumpError(f"failed to fetch {url}: {exc}") from exc


def load_source(source_path: str | None, source_url: str) -> tuple[str, dict[str, Any]]:
    if source_path:
        path = Path(source_path)
        text = path.read_text(encoding="utf-8")
        return text, {
            "kind": "file",
            "path": str(path),
            "sha256": sha256_text(text),
        }
    text = fetch_text(source_url)
    return text, {
        "kind": "url",
        "url": source_url,
        "sha256": sha256_text(text),
    }


def nested_text_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise DumpError("config.text_config must be an object")
    return text_config


def required_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DumpError(f"{key} must be a positive integer, got {value!r}")
    return value


def summarize_config(raw_config: dict[str, Any], *, model_id: str) -> ConfigSummary:
    cfg = nested_text_config(raw_config)
    text_model_type = cfg.get("model_type")
    if text_model_type not in {"qwen3_5_text", "qwen3_5_moe_text"}:
        raise DumpError(f"unsupported text_config.model_type {text_model_type!r}")
    hidden_size = required_int(cfg, "hidden_size")
    num_hidden_layers = required_int(cfg, "num_hidden_layers")
    layer_types = cfg.get("layer_types")
    if not isinstance(layer_types, list) or not all(isinstance(item, str) for item in layer_types):
        raise DumpError("layer_types must be a string list")
    if len(layer_types) != num_hidden_layers:
        raise DumpError(
            f"layer_types length {len(layer_types)} != num_hidden_layers {num_hidden_layers}"
        )
    try:
        first_linear = layer_types.index("linear_attention")
    except ValueError as exc:
        raise DumpError("layer_types must include at least one linear_attention layer") from exc
    linear_attention = {
        "linear_num_key_heads": required_int(cfg, "linear_num_key_heads"),
        "linear_num_value_heads": required_int(cfg, "linear_num_value_heads"),
        "linear_key_head_dim": required_int(cfg, "linear_key_head_dim"),
        "linear_value_head_dim": required_int(cfg, "linear_value_head_dim"),
        "linear_conv_kernel_dim": required_int(cfg, "linear_conv_kernel_dim"),
    }
    moe = None
    if text_model_type == "qwen3_5_moe_text":
        moe = {
            "num_experts": required_int(cfg, "num_experts"),
            "num_experts_per_tok": required_int(cfg, "num_experts_per_tok"),
            "moe_intermediate_size": required_int(cfg, "moe_intermediate_size"),
            "shared_expert_intermediate_size": required_int(
                cfg,
                "shared_expert_intermediate_size",
            ),
        }
        if moe["num_experts_per_tok"] > moe["num_experts"]:
            raise DumpError("num_experts_per_tok cannot exceed num_experts")
    return ConfigSummary(
        model_id=model_id,
        text_model_type=text_model_type,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        layer_types=layer_types,
        first_linear_layer=first_linear,
        linear_attention=linear_attention,
        moe=moe,
    )


def validate_source_contract(summary: ConfigSummary, source_text: str) -> dict[str, Any]:
    needles = COMMON_SOURCE_NEEDLES + (MOE_SOURCE_NEEDLES if summary.is_moe else DENSE_SOURCE_NEEDLES)
    missing = [needle for needle in needles if needle not in source_text]
    if missing:
        raise DumpError("Transformers source missing required W3 hooks: " + ", ".join(missing))
    hook_lines = {}
    for needle in needles:
        idx = source_text.find(needle)
        hook_lines[needle] = source_text[:idx].count("\n") + 1
    return {
        "status": "pass",
        "text_model_type": summary.text_model_type,
        "selected_layer_idx": summary.first_linear_layer,
        "required_hooks": hook_lines,
        "source_sha256": sha256_text(source_text),
        "note": "source contract only; no model weights were loaded",
    }


def tensor_to_f32_list(tensor: Any) -> list[float]:
    arr = tensor.detach().float().cpu().contiguous().view(-1)
    return [float(value) for value in arr.tolist()]


def write_f32(path: Path, values: list[float]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def save_tensor(path: Path, name: str, tensor: Any) -> dict[str, Any]:
    values = tensor_to_f32_list(tensor)
    rel = f"{name}.bin"
    write_f32(path / rel, values)
    return {
        "file": rel,
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "numel": len(values),
        "sha256": sha256_bytes((path / rel).read_bytes()),
    }


def find_text_layers(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    for module in model.modules():
        if hasattr(module, "layers") and len(getattr(module, "layers")) > 0:
            first = getattr(module, "layers")[0]
            if hasattr(first, "layer_type"):
                return getattr(module, "layers")
    raise DumpError("could not locate Qwen3.5 text decoder layers on HF model")


def get_module_output_tensors(module: Any, args: tuple[Any, ...], output: Any) -> Any:
    del module, args
    return output[0] if isinstance(output, tuple) else output


def register_forward_capture(module: Any, name: str, captures: dict[str, Any]) -> Any:
    def hook(mod: Any, hook_args: tuple[Any, ...], hook_output: Any) -> None:
        captures[name] = get_module_output_tensors(mod, hook_args, hook_output).detach()

    return module.register_forward_hook(hook)


def patch_deltanet_for_capture(deltanet: Any, captures: dict[str, Any]) -> list[tuple[str, Any, str, Any]]:
    patches = []

    original_chunk = deltanet.chunk_gated_delta_rule

    def chunk_wrapper(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        captures["delta_q"] = query.detach()
        captures["delta_k"] = key.detach()
        captures["delta_v"] = value.detach()
        captures["delta_g"] = kwargs.get("g").detach()
        captures["delta_beta"] = kwargs.get("beta").detach()
        output = original_chunk(query, key, value, *args, **kwargs)
        captures["delta_core"] = output[0].detach() if isinstance(output, tuple) else output.detach()
        return output

    deltanet.chunk_gated_delta_rule = chunk_wrapper
    patches.append(("attr", deltanet, "chunk_gated_delta_rule", original_chunk))
    return patches


def restore_patches(patches: list[tuple[str, Any, str, Any]]) -> None:
    for kind, obj, name, original in reversed(patches):
        if kind == "attr":
            setattr(obj, name, original)


def run_real_dump(args: argparse.Namespace) -> int:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on external environment
        raise DumpError(
            "dump mode requires torch and transformers; use --self-test or --contract locally"
        ) from exc

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_id = args.model_id
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    config_dict = config.to_dict()
    summary = summarize_config(config_dict, model_id=model_id)
    if summary.is_moe and not args.allow_moe:
        raise DumpError("MoE dump requires --allow-moe because the artifact is much larger")

    dtype = getattr(torch, args.torch_dtype)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    layers = find_text_layers(model)
    layer_idx = summary.first_linear_layer
    layer = layers[layer_idx]
    if getattr(layer, "layer_type", None) != "linear_attention":
        raise DumpError(f"selected layer {layer_idx} is not linear_attention")
    if not hasattr(layer, "linear_attn"):
        raise DumpError(f"selected layer {layer_idx} has no linear_attn module")
    deltanet = layer.linear_attn

    captures: dict[str, Any] = {}
    handles = []
    patches = []
    try:
        def capture_layer_input(_module: Any, hook_args: tuple[Any, ...]) -> None:
            captures["layer_input"] = hook_args[0].detach()

        handles.append(
            layer.register_forward_pre_hook(capture_layer_input)
        )
        handles.append(register_forward_capture(layer.input_layernorm, "input_norm", captures))
        handles.append(register_forward_capture(deltanet.in_proj_qkv, "mixed_qkv_raw", captures))
        handles.append(register_forward_capture(deltanet.in_proj_z, "z_raw", captures))
        handles.append(register_forward_capture(deltanet.in_proj_b, "b_raw", captures))
        handles.append(register_forward_capture(deltanet.in_proj_a, "a_raw", captures))
        handles.append(register_forward_capture(deltanet.norm, "delta_norm", captures))
        handles.append(register_forward_capture(deltanet.out_proj, "delta_output", captures))
        handles.append(register_forward_capture(layer.post_attention_layernorm, "post_attention_norm", captures))
        handles.append(register_forward_capture(layer.mlp, "mlp_output", captures))
        handles.append(register_forward_capture(layer, "layer_output", captures))
        patches = patch_deltanet_for_capture(deltanet, captures)

        prompt = args.prompt
        tokenized = tokenizer(prompt, return_tensors="pt")
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        with torch.no_grad():
            _ = model(**tokenized, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        restore_patches(patches)

    if "layer_input" in captures and "delta_output" in captures:
        captures["residual_after_mixer"] = captures["layer_input"] + captures["delta_output"]
    if "mixed_qkv_raw" in captures:
        if getattr(deltanet, "activation", "silu") != "silu":
            raise DumpError(f"unsupported DeltaNet conv activation {deltanet.activation!r}")
        mixed_qkv = captures["mixed_qkv_raw"].transpose(1, 2)
        conv_raw = F.conv1d(
            mixed_qkv,
            deltanet.conv1d.weight,
            bias=deltanet.conv1d.bias,
            padding=deltanet.conv_kernel_size - 1,
            groups=deltanet.conv_dim,
        )
        conv_raw = conv_raw[:, :, : mixed_qkv.shape[-1]]
        captures["mixed_qkv_conv"] = F.silu(conv_raw).transpose(1, 2).detach()

    missing = [name for name in LAYER_DUMP_TENSORS if name not in captures]
    if missing:
        raise DumpError("real dump missing captured tensors: " + ", ".join(missing))

    tensors_dir = out_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    tensor_manifest = {
        name: save_tensor(tensors_dir, name, captures[name])
        for name in LAYER_DUMP_TENSORS
    }
    pass_line = f"{PASS_DUMP}: {out_dir}"
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "mode": "dump",
        "pass_line": pass_line,
        "created_at": iso_now(),
        "command_line": command_line(),
        "git": git_summary(),
        "model_id": model_id,
        "prompt": prompt,
        "prompt_input_ids": tokenized["input_ids"].detach().cpu().tolist(),
        "selected_layer_idx": layer_idx,
        "selected_layer_type": getattr(layer, "layer_type", None),
        "text_model_type": summary.text_model_type,
        "config_summary": summary.__dict__,
        "dependencies": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "runtime": {
            "device": str(device),
            "torch_dtype": args.torch_dtype,
            "trust_remote_code": args.trust_remote_code,
            "HF_HOME": os.environ.get("HF_HOME"),
        },
        "tensor_dir": str(tensors_dir),
        "tensors": tensor_manifest,
        "note": "HF reference dump only; Ferrum dump and comparator are still required for W3-S1",
    }
    write_json(out_dir / MANIFEST_NAME, manifest)
    print(pass_line)
    return 0


def run_contract(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_config = read_json(Path(args.config))
    summary = summarize_config(raw_config, model_id=args.model_id)
    default_url = DEFAULT_MOE_SOURCE_URL if summary.is_moe else DEFAULT_DENSE_SOURCE_URL
    source_text, source_meta = load_source(args.source_file, args.source_url or default_url)
    contract = validate_source_contract(summary, source_text)
    pass_line = f"{PASS_CONTRACT}: {out_dir}"
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "mode": "contract",
        "pass_line": pass_line,
        "created_at": iso_now(),
        "command_line": command_line(),
        "git": git_summary(),
        "model_id": args.model_id,
        "config_summary": summary.__dict__,
        "source": source_meta,
        "contract": contract,
        "note": "config/source contract only; no model weights were downloaded",
    }
    write_json(out_dir / MANIFEST_NAME, manifest)
    print(pass_line)
    return 0


def run_self_test(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fake_config = {
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
    fake_source = "\n".join(COMMON_SOURCE_NEEDLES + DENSE_SOURCE_NEEDLES)
    summary = summarize_config(fake_config, model_id="self-test/qwen3.5")
    contract = validate_source_contract(summary, fake_source)
    tensors_dir = out_dir / "synthetic_tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    tensor_manifest = {}
    for idx, name in enumerate(LAYER_DUMP_TENSORS):
        values = [float(idx), float(idx) + 0.25, float(idx) + 0.5, float(idx) + 0.75]
        rel = f"{name}.bin"
        write_f32(tensors_dir / rel, values)
        tensor_manifest[name] = {
            "file": rel,
            "shape": [1, 4],
            "dtype": "float32",
            "numel": len(values),
            "sha256": sha256_bytes((tensors_dir / rel).read_bytes()),
        }
    pass_line = f"{PASS_SELFTEST}: {out_dir}"
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "mode": "self-test",
        "pass_line": pass_line,
        "created_at": iso_now(),
        "command_line": command_line(),
        "git": git_summary(),
        "config_summary": summary.__dict__,
        "contract": contract,
        "tensor_dir": str(tensors_dir),
        "tensors": tensor_manifest,
        "note": "self-test validates schema and source-hook contract only",
    }
    write_json(out_dir / MANIFEST_NAME, manifest)
    print(pass_line)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--contract", action="store_true")
    mode.add_argument("--dump", action="store_true")
    parser.add_argument("--out", default="target/w3_qwen35_hf_layer_dump")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--config", help="saved HF config.json path for --contract")
    parser.add_argument("--source-file", help="Transformers modeling source file for --contract")
    parser.add_argument("--source-url", help="Transformers modeling source URL for --contract")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-dtype", default="float32")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-moe", action="store_true")
    args = parser.parse_args(argv)
    if args.contract and not args.config:
        parser.error("--contract requires --config")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            return run_self_test(args)
        if args.contract:
            return run_contract(args)
        return run_real_dump(args)
    except DumpError as exc:
        print(f"W3 QWEN35 HF LAYER DUMP FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
