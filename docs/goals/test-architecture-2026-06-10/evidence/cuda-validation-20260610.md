# CUDA validation evidence (pod 40361123, RTX 4090)

## Op-parity (CUDA column)
`cargo test -p ferrum-testkit --features cuda --test op_diff`: 15/15 pass,
CPU-vs-CUDA NMSE ~1e-7. 12 conformance ops now CPU+Metal+CUDA verified.

## Dense multi-turn CUDA chat (stage-3 decoupling validation)
`ferrum run Qwen/Qwen3-0.6B` piped 5 turns, --max-tokens 40, on real CUDA:
- All 5 turns answered coherently, MULTITURN_EXIT=0, no crash.
- Validates the stage-3 `supports_native_unified_decode` decoupling on real
  CUDA inference: the engine routes the unified path correctly on CUDA.
- ~100-114 tok/s decode.

hb-09 (Qwen3-MoE turn-3 paged_varlen_attn panic) is MoE-specific; the
30B-A3B model is downloading to run that verify-live probe + hb-10.

## hb-09 verify-live: PASS (open bug fixed)
6-turn MoE CUDA chat (Qwen3-30B-A3B-GPTQ-Int4), --max-tokens 32: all 6 turns
coherent (通义千问 intro, joke, 北京 attractions), HB09_EXIT=0. The turn-3
`paged_varlen_attn CUDA_ERROR_INVALID_VALUE` panic from TESTING-GAPS #1
(build 241dbc0) does NOT reproduce on current main — fixed since. Also
re-validates the stage-3 decoupling on MoE CUDA inference.

## hb-10: fix confirmed (vllm-moe-marlin coherent)
FERRUM_VLLM_MOE=1, Qwen3-30B-A3B: "北京是中国的首都，是政治、文化和国际交往中心..."
— coherent, no garbage. The marlin packing fix (049b3a42) is present.

## hb-11 verify-live: PASS (kv boundary)
6000-token prompt on Qwen3-0.6B generates (16 tokens, 61s) with NO
paged kv shared-memory INVALID_VALUE crash.
