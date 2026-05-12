#!/usr/bin/env python3
"""Generate a shared-prefix prompts.jsonl in the same format as the
existing apples bench prompts.jsonl. Drop-in replacement that lets us
run `vllm bench serve` (the apples client) on a workload where
prefix caching matters.

128 prompts, each = SHARED_PREFIX (~200 tokens) + UNIQUE suffix (~5 tokens).
"""
import json
import sys

SHARED_PREFIX = (
    "You are a precise, helpful technical assistant. "
    "When answering questions you must follow these rules carefully:\n\n"
    "1. Be accurate. If you don't know something, say so directly.\n"
    "2. Be concise. Prefer short, direct answers over verbose ones.\n"
    "3. Show reasoning for non-trivial claims, but keep it to one or two "
    "sentences when possible.\n"
    "4. For factual queries about programming, math, or systems, cite the "
    "relevant concept by name (e.g., \"via the GIL\", \"by memoization\").\n"
    "5. Avoid filler phrases like \"Great question!\" or \"I'd be happy to "
    "help with that.\" Just answer.\n"
    "6. If a question is ambiguous, name the ambiguity in one sentence "
    "and pick the most likely interpretation.\n\n"
    "User question: "
)

QUESTIONS = [
    "What is GIL?", "Define amortized cost.", "Why quicksort O(n log n)?",
    "What does ECC RAM mean?", "Explain TLB shootdown.", "What's CRDT?",
    "Define MVCC.", "What is Raft?", "Explain Bloom filter.",
    "What is a skip list?", "What's pre-order traversal?", "Explain CAS.",
    "What is a B+ tree?", "Define cache line.", "What's NUMA?",
    "Explain VMA.", "What is the kernel page table?", "Define eviction policy.",
    "What's TCP slow start?", "Explain NAT punching.", "What is QUIC?",
    "Define WAL.", "What's HSTS?", "Explain CRC32.",
    "What's a Merkle tree?", "Define ABA problem.", "What's a memory fence?",
    "Explain RCU.", "What's a futex?", "Define copy-on-write.",
    "What's mmap used for?", "Explain epoll.",
]
prompts = [SHARED_PREFIX + q for q in (QUESTIONS * 4)[:128]]

out_path = sys.argv[1] if len(sys.argv) > 1 else "shared_prefix_prompts.jsonl"
with open(out_path, "w") as f:
    for p in prompts:
        f.write(json.dumps({"prompt": p}) + "\n")
print(f"wrote {len(prompts)} prompts to {out_path}")
print(f"shared prefix chars: {len(SHARED_PREFIX)}")
