# W2 Marlin workspace-zero native CUDA A/B

- Status: diagnostic PASS, not release-grade evidence.
- Lane: W2 Gemma3 Marlin workspace-zero native A/B.
- Old cached Vast instance `40826362`: destroyed after repeated `resources_unavailable`.
- New Vast instance `41171821`: 1x RTX 4090, driver `550.127.08`, CUDA `12.4`, nvcc `12.4.131`, destroyed after artifact collection.
- Native probe source: clean `38dd65c40f5d2a1feef2fdda8291101c4e1fde90`.
- Native probe binary SHA256: `b69b949e1f68a001f02573144a9f415bab3efe10222c49c125a85306e98d6cf0`.
- Native probe rc: `0`.
- PASS line: `VERDICT: gemma3 tail MLP chain native CUDA probe complete`.
- CUDA feature retry test rc: `0`.

## Key Results

| m | product_ws_zero chain_event_us | prezero diagnostic chain_event_us | delta_us |
|---:|---:|---:|---:|
| 1 | 209.992 | 206.763 | 3.229 |
| 10 | 215.520 | 210.817 | 4.703 |
| 16 | 218.139 | 213.571 | 4.568 |
| 23 | 228.435 | 225.094 | 3.341 |
| 32 | 231.747 | 228.928 | 2.819 |

At the product-critical m16 shape, pre-zeroing the Marlin workspace saves about
`4.568us` per layer. Across 62 layers this is about `0.283ms` per decode step,
which is too small to explain the W2 c16/c32 gap versus vLLM.

## Source Validation

The first CUDA feature test exposed two test-only build issues in the
`38dd65c4` checkpoint: a missing `MarlinProfileBucketStats` import and an
ambiguous empty `from_env_vars([])` call. After syncing the local test fix to
the GPU host, the CUDA feature test passed:

```text
test backend::cuda::marlin::tests::marlin_workspace_zeroing_follows_runtime_config ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 25 filtered out
```

## Interpretation

The existing dense Marlin workspace-zero diagnostic switch is now wired and
build-tested on CUDA, but skipping dense workspace zeroing is not a high-return
W2 performance lever. The next W2 work should stay focused on the already
identified higher-impact areas: prefill/TTFT scheduling and Gemma3 decode
integration around tail MLP projection scheduling.

No `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
