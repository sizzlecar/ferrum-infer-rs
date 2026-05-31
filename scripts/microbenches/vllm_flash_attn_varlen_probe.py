#!/usr/bin/env python3
"""
Probe vLLM FlashAttention varlen paged-attention speed on Qwen3-MoE shapes.

This is intentionally outside the Ferrum build. Run it inside a vLLM 0.20.2
environment on the GPU pod to estimate the upside of replacing Ferrum's
simple vLLM-layout varlen attention bridge with a real FA-style kernel.

Example:
  /workspace/vllm-venv/bin/python scripts/microbenches/vllm_flash_attn_varlen_probe.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func


@dataclass(frozen=True)
class Case:
    name: str
    q_lens: tuple[int, ...]
    pos_offsets: tuple[int, ...]


CASES = (
    Case("prefill_4x256", (256, 256, 256, 256), (0, 0, 0, 0)),
    Case(
        "mixed_3x256_4x1",
        (256, 256, 256, 1, 1, 1, 1),
        (0, 0, 0, 256, 256, 256, 256),
    ),
    Case("prefill_4x512", (512, 512, 512, 512), (0, 0, 0, 0)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--fa-version", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_block_table(final_lens: list[int], block_size: int, device: str) -> torch.Tensor:
    blocks_per_seq = [(length + block_size - 1) // block_size for length in final_lens]
    max_blocks = max(blocks_per_seq)
    block_table = torch.full(
        (len(final_lens), max_blocks),
        -1,
        device=device,
        dtype=torch.int32,
    )
    block_id = 0
    for row, num_blocks in enumerate(blocks_per_seq):
        block_table[row, :num_blocks] = torch.arange(
            block_id,
            block_id + num_blocks,
            device=device,
            dtype=torch.int32,
        )
        block_id += num_blocks
    return block_table


def bench_case(case: Case, args: argparse.Namespace) -> None:
    q_lens = list(case.q_lens)
    pos_offsets = list(case.pos_offsets)
    final_lens = [pos + q_len for pos, q_len in zip(pos_offsets, q_lens)]
    total_q = sum(q_lens)
    max_q = max(q_lens)
    max_k = max(final_lens)

    block_table = build_block_table(final_lens, args.block_size, args.device)
    num_blocks = int((block_table >= 0).sum().item())
    cu = [0]
    for q_len in q_lens:
        cu.append(cu[-1] + q_len)
    cu_seqlens_q = torch.tensor(cu, device=args.device, dtype=torch.int32)
    seqused_k = torch.tensor(final_lens, device=args.device, dtype=torch.int32)

    torch.manual_seed(0)
    q = torch.randn(
        (total_q, args.num_heads, args.head_dim),
        device=args.device,
        dtype=torch.float16,
    )
    k = torch.randn(
        (num_blocks, args.block_size, args.num_kv_heads, args.head_dim),
        device=args.device,
        dtype=torch.float16,
    )
    v = torch.randn_like(k)
    out = torch.empty_like(q)

    kwargs = dict(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_k,
        softmax_scale=1.0 / math.sqrt(args.head_dim),
        causal=True,
        block_table=block_table,
        fa_version=args.fa_version,
    )
    flash_attn_varlen_func(**kwargs)
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        flash_attn_varlen_func(**kwargs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        flash_attn_varlen_func(**kwargs)
    end.record()
    torch.cuda.synchronize()

    elapsed_us = start.elapsed_time(end) * 1000.0 / args.iters
    print(
        f"{case.name}: vllm_fa{args.fa_version}_paged={elapsed_us:.2f} us "
        f"total_q={total_q} max_q={max_q} max_k={max_k} blocks={num_blocks}"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")
    for case in CASES:
        bench_case(case, args)


if __name__ == "__main__":
    main()
