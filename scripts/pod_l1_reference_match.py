#!/usr/bin/env python3
"""L1 representative check: ferrum BF16 greedy vs HF transformers, N=20.

Two phases because a 24GB card can't hold the model twice:

  python3 scripts/pod_l1_reference_match.py gen --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --out /workspace/w1/l1_refs.json
  # free the GPU, start `ferrum serve`, then:
  python3 scripts/pod_l1_reference_match.py verify --refs /workspace/w1/l1_refs.json --base-url http://127.0.0.1:8000 --alias deepseek-r1:8b --out /workspace/w1/l1_report.json

Comparison: the generated text is split at '</think>' (reasoning lstrips
trailing/leading newlines the same way ferrum's splitter does) and both
halves must be byte-equal. GOAL L1 criterion: 20/20.
"""

import argparse
import json
import sys
import urllib.request

PROMPTS = [
    "What is 2+3? Answer with just the number.",
    "Name the capital of Japan.",
    "Write a haiku about the sea.",
    "Translate 'good morning' into French.",
    "What is the chemical symbol for gold?",
    "List three prime numbers greater than 10.",
    "What year did the Apollo 11 moon landing happen?",
    "Spell 'necessary'.",
    "What is 12 squared?",
    "Give one synonym for 'fast'.",
    "What language is primarily spoken in Brazil?",
    "How many continents are there?",
    "What is the boiling point of water in Celsius?",
    "Name the largest planet in the solar system.",
    "What does CPU stand for?",
    "Write the first 5 Fibonacci numbers.",
    "What is the square root of 81?",
    "Name a programming language created by Guido van Rossum.",
    "What gas do plants absorb from the air?",
    "How many minutes are in two hours?",
]
MAX_NEW_TOKENS = 96


def split_think(text: str):
    if "</think>" in text:
        reasoning, _, content = text.partition("</think>")
        reasoning = reasoning.split("<think>")[-1].strip("\n")
        return reasoning, content.lstrip("\n")
    return "", text


def cmd_gen(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    refs = []
    for p in PROMPTS:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        out = model.generate(
            ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tok.eos_token_id,
        )
        gen = out[0][ids.shape[1] :]
        text = tok.decode(gen, skip_special_tokens=True)
        refs.append({"prompt": p, "text": text})
        print(f"[gen] {p[:40]!r} -> {len(text)} chars", flush=True)
    json.dump(refs, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"[gen] wrote {len(refs)} refs to {args.out}")


def cmd_verify(args):
    refs = json.load(open(args.refs))
    results, n_pass = [], 0
    for r in refs:
        req = urllib.request.Request(
            f"{args.base_url}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": args.alias,
                    "temperature": 0,
                    "max_tokens": MAX_NEW_TOKENS,
                    "messages": [{"role": "user", "content": r["prompt"]}],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            msg = json.load(resp)["choices"][0]["message"]
        f_reasoning = (msg.get("reasoning") or "").strip("\n")
        f_content = (msg.get("content") or "").lstrip("\n")
        h_reasoning, h_content = split_think(r["text"])
        ok = (f_reasoning == h_reasoning) and (f_content == h_content)
        n_pass += ok
        results.append(
            {
                "prompt": r["prompt"],
                "pass": ok,
                "hf_reasoning": h_reasoning,
                "hf_content": h_content,
                "ferrum_reasoning": f_reasoning,
                "ferrum_content": f_content,
            }
        )
        print(f"[verify] {'ok  ' if ok else 'DIFF'} {r['prompt'][:40]!r}", flush=True)
    json.dump(
        {"passed": n_pass, "total": len(refs), "results": results},
        open(args.out, "w"),
        ensure_ascii=False,
        indent=1,
    )
    print(f"L1_REFERENCE_MATCH {n_pass}/{len(refs)}")
    sys.exit(0 if n_pass == len(refs) else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_gen)
    v = sub.add_parser("verify")
    v.add_argument("--refs", required=True)
    v.add_argument("--base-url", required=True)
    v.add_argument("--alias", required=True)
    v.add_argument("--out", required=True)
    v.set_defaults(fn=cmd_verify)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
