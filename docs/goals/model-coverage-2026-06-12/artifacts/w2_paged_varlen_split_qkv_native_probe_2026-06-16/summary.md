# W2 paged-varlen split-QKV native probe summary

- Scope: diagnostic native CUDA correctness probe only. This did not run
  `ferrum run`, `ferrum serve`, or any performance benchmark.
- Source: `7dc711ef817af737903098f14c068852c04d7dbf`.
- Instance: Vast `40826362`, 1x RTX 4090, reused cache-retained CUDA machine.
- CUDA environment: driver `565.77`, runtime-reported CUDA `12.7`, `nvcc`
  `12.4.131`.
- Command:
  `bash scripts/microbenches/build_and_run_paged_varlen_split_qkv_correctness.sh`.
- Result: rc `0`, stdout contains
  `VERDICT: paged varlen split-qkv correctness PASS`.
- Key rows:
  - `qk_mode=1 sliding_window=0`: q/k/v err all `0`, attention err
    `0.00012147`.
  - `qk_mode=1 sliding_window=3`: q/k/v err all `0`, attention err
    `0.00012141`.
  - `qk_mode=2 sliding_window=3`: q/k/v err all `0`, attention err
    `0.00011945`.
  - `qk_mode=3 sliding_window=3`: q/k/v err all `0`, attention err
    `0.00011978`.
  - `qk_mode=1 semantic_delta_full_vs_window=0.06742159`.
- Interpretation: the standalone native CUDA chain
  `split_qkv_norm_rope_into_paged_cache_varlen_f16` ->
  `paged_varlen_attn_f16` matches the CPU reference for synthetic nonzero
  historical KV, current varlen writes, QK-norm/RoPE modes, and sliding-window
  attention. The remaining Gemma3 paged-unified empty-output failure is more
  likely above this pair of kernels or dependent on real model/product state.
- Cleanup: Vast stop response `success=True`; final poll recorded
  `cur_state=stopped actual_status=exited`.
