#!/usr/bin/env python3
"""Generate Python reference outputs for Phase E CUDA parity tests.

Run this ONCE on any machine with torch + HF transformers (GPU or CPU is
fine; fp32 on CPU produces the most deterministic reference). Saves a
.npz with the expected prefill logits / decode logits that the Rust
CudaBackend path should reproduce.

Usage:
    python scripts/phase-e-gen-baseline.py Qwen/Qwen3-0.6B \
        --prompt '你好,欢迎使用' \
        --max-new 5 \
        --out /tmp/qwen3-06b-baseline.npz

Output fields (in the npz):
    input_ids            — tokenized prompt
    prefill_logits_last  — [vocab]  logits of the last prompt token
    decode_logits[i]     — [vocab]  logits at each generated position
    generated_tokens     — argmax-greedy tokens chosen at each step
    meta.model_id, meta.dtype, meta.device

The Rust parity test compares its own logits against these, using the
same argmax-chained token sequence.
"""

import argparse
import pathlib
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_id", help="HF model id, e.g. Qwen/Qwen3-0.6B")
    p.add_argument("--prompt", default="你好,欢迎使用", help="UTF-8 prompt")
    p.add_argument("--max-new", type=int, default=5, help="How many decode steps to capture")
    p.add_argument("--out", required=True, help="Output .npz file")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        sys.exit(f"missing python deps — `pip install torch transformers`: {e}")

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    print(f"Loading {args.model_id} (dtype={args.dtype}, device={args.device})...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    # ── Tokenize ─────────────────────────────────────────────────────────
    encoded = tok(args.prompt, return_tensors="pt").to(args.device)
    input_ids = encoded["input_ids"][0].cpu().numpy()
    print(f"Tokenized: {len(input_ids)} tokens")

    # ── Prefill ──────────────────────────────────────────────────────────
    with torch.no_grad():
        out = model(**encoded, use_cache=True)
    prefill_logits = out.logits[0, -1, :].float().cpu().numpy()  # [vocab]
    past = out.past_key_values
    next_tok = int(np.argmax(prefill_logits))
    print(f"Prefill: argmax={next_tok}, logit={prefill_logits[next_tok]:.4f}")

    # ── Decode (greedy, max_new steps) ───────────────────────────────────
    decode_logits = []
    generated = [next_tok]
    pos = len(input_ids)
    for step in range(args.max_new):
        inp = torch.tensor([[next_tok]], device=args.device)
        with torch.no_grad():
            out = model(inp, past_key_values=past, use_cache=True)
        logits = out.logits[0, -1, :].float().cpu().numpy()
        decode_logits.append(logits)
        past = out.past_key_values
        next_tok = int(np.argmax(logits))
        generated.append(next_tok)
        print(f"decode[{step}] pos={pos} argmax={next_tok} logit={logits[next_tok]:.4f}")
        pos += 1

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        input_ids=input_ids.astype(np.int32),
        prefill_logits_last=prefill_logits.astype(np.float32),
        decode_logits=np.stack(decode_logits).astype(np.float32),
        generated_tokens=np.array(generated, dtype=np.int32),
        meta_model_id=np.array(args.model_id),
        meta_dtype=np.array(args.dtype),
        meta_device=np.array(args.device),
    )
    print(f"Saved: {out_path}  ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
