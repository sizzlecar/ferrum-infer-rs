#!/usr/bin/env python3
"""Gemma 3 L1 reference dumps from HF transformers (W2 new-family gate).

Two modes:

  dump   — one fixed prompt: per-layer hidden states + final logits, saved
           as f32 little-endian .bin matching ferrum's FERRUM_LAYER_DUMP
           layout (embed.bin, layer_NN.bin, logits.bin) plus tokens.json.
  greedy — N short prompts: greedy continuations (max_new_tokens) saved to
           greedy.json for byte-equality comparison with ferrum run.

Run (CPU, bf16 — matches ferrum's BF16 lane):
  uv run --with "transformers==4.51.3" --with torch --with accelerate \
    python scripts/gemma3_l1_reference.py dump /tmp/gemma3_hf_dump
  uv run --with "transformers==4.51.3" --with torch --with accelerate \
    python scripts/gemma3_l1_reference.py greedy /tmp/gemma3_hf_dump
"""

import json
import struct
import sys
from pathlib import Path

MODEL_ID = "unsloth/gemma-3-1b-it"

DUMP_PROMPT = [{"role": "user", "content": "Why is the sky blue?"}]

GREEDY_PROMPTS = [
    "Why is the sky blue?",
    "Write a haiku about rust.",
    "What is 17 * 23?",
    "用一句话解释什么是量子纠缠。",
    "def fib(n):",
    "List three prime numbers.",
    "What is the capital of Iceland?",
    "Translate 'good morning' to French.",
    "Name a mammal that can fly.",
    "What year did the first moon landing happen?",
    "Give one synonym for 'happy'.",
    "What does HTTP stand for?",
    "今天天气怎么样?",
    "Sort these numbers: 3, 1, 2",
    "What is the boiling point of water in Celsius?",
    "Who wrote Romeo and Juliet?",
    "Spell 'necessary'.",
    "What is the square root of 144?",
    "Name the largest ocean.",
    "How many legs does a spider have?",
]


def write_f32(path: Path, tensor) -> None:
    import torch

    flat = tensor.detach().to(torch.float32).reshape(-1).tolist()
    path.write_bytes(struct.pack(f"<{len(flat)}f", *flat))


def main() -> None:
    mode, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model.eval()

    if mode == "dump":
        ids = tok.apply_chat_template(
            DUMP_PROMPT, add_generation_prompt=True, return_tensors="pt"
        )
        (out_dir / "tokens.json").write_text(json.dumps(ids[0].tolist()))
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        # hidden_states[0] = scaled embeddings; [i+1] = after layer i
        # (pre-final-norm). Matches ferrum's residual dump points.
        hs = out.hidden_states
        write_f32(out_dir / "embed.bin", hs[0][0])
        for i, h in enumerate(hs[1:]):
            write_f32(out_dir / f"layer_{i:02}.bin", h[0])
        write_f32(out_dir / "logits.bin", out.logits[0, -1])
        print(f"dumped {len(hs) - 1} layers + embed + logits to {out_dir}")
    elif mode == "greedy":
        results = []
        for prompt in GREEDY_PROMPTS:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=32, do_sample=False, temperature=None, top_p=None, top_k=None
                )
            completion = tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
            results.append({"prompt": prompt, "completion": completion})
            print(f"  {prompt[:40]!r} -> {completion[:60]!r}")
        (out_dir / "greedy.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=1)
        )
        print(f"wrote {len(results)} greedy continuations to {out_dir}/greedy.json")
    else:
        raise SystemExit(f"unknown mode {mode!r} (use: dump | greedy)")


if __name__ == "__main__":
    main()
