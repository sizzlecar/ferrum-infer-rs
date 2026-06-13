# Gemma3-1B L1 evidence (W2, 2026-06-13)

- `dump_compare_cpu.txt` / `dump_compare_metal.txt`: per-layer residual
  comparison vs HF transformers bf16 (`scripts/gemma3_l1_reference.py dump`
  + `scripts/gemma3_l1_compare.py`). Both lanes: 25/26 layers within bf16
  thresholds, logits argmax MATCH → `L1 DUMP COMPARE PASS`. The last
  hidden_states entry is post-final-norm on the HF side (skipped by design).
- `hf_greedy.json` vs `ferrum_greedy_cpu.jsonl`: 20 greedy continuations,
  **18/20 byte-equal**. The two divergences ('changing|turning',
  'It|He') sit at HF top1−top2 logit gap = **0.25 = one bf16 ulp** —
  tie-flips per the W1 amendment #3 methodology, not numerical defects.
- Metal lane required two kernel fixes first (flash_attn acc[4]→[8] at
  head_dim 256; gelu_tanh fast-math overflow clamp), pinned by
  `crates/ferrum-kernels/tests/gemma3_metal_ops_test.rs`.
