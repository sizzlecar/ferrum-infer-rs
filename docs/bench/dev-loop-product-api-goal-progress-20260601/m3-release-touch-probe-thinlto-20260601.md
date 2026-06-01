# Milestone A Thin-LTO Release Touch Probe - 2026-06-01

Purpose: verify that a narrow CUDA-kernel touch no longer triggers a broad
release rebuild/link tail above the Milestone A target after the release profile
change to thin LTO.

Remote artifact:

- `/workspace/m3-release-touch-probe-thinlto-20260601-20260601_064127`

Command:

```bash
cd /workspace/ferrum-fa2-native-restore-git-ac3dfab
python3 scripts/m3_cuda_build_boundary_probe.py \
  --iterations 5 \
  --out /workspace/m3-release-touch-probe-thinlto-20260601-20260601_064127 \
  --fail-on-limit
```

Manifest:

- `/workspace/m3-release-touch-probe-thinlto-20260601-20260601_064127/build_boundary_manifest.json`
- schema: `1`
- git head: `73a26af898371f202dcdc89bfb74a1437a8e4be1`
- package: `ferrum-cli`
- features: `cuda,marlin,vllm-paged-attn-v2,vllm-moe-marlin,fa2-source`
- mutation: `touch`
- kernel: `crates/ferrum-kernels/kernels/paged_varlen_attention_vllm.cu`

Timing:

| run | elapsed_sec |
|---:|---:|
| 1 | 34.454 |
| 2 | 33.469 |
| 3 | 33.056 |
| 4 | 33.164 |
| 5 | 32.502 |

Result:

- `p50_sec_nearest_rank=33.164`
- `p95_sec_nearest_rank=34.454`
- limits: `p50<=75.0`, `p95<=90.0`
- `limits_pass=true`

CUDA summary validation:

- every run exited `0`
- every run reported `39` CUDA build summary rows
- every run reported `status_counts={"cache_hit": 39}`
- required cache-hit artifacts:
  `core-ptx:kernels/paged_varlen_attention_vllm.cu`, `marlin`,
  `vllm_marlin`, `vllm_moe_marlin`, `vllm_paged_attn`, `fa2_source`

Conclusion:

- Milestone A release-boundary timing is no longer blocked by the earlier
  `231s/234s` release touch result.
- The CUDA artifact cache-hit behavior and the release touch p50/p95 timing
  target are both proven on the restored RTX 4090 pod for this checkpoint.
- This does not close the full product/API goal by itself; Milestone I still
  needs a publishable same-pod all-cell packet, and Milestones E/F/G still have
  product-readiness evidence gaps.
