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
