# W2 Gemma3 Marlin cache-policy product-default native CUDA probe

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_cache_policy_default_native_probe_2026-06-16`
- Lane: W2 Marlin cache-policy product-default native probe
- Remote git HEAD: `c76bfcfa2b00a73a816e6d44bbd999a621b12a49`
- Probe rc: `0`
- Legacy plain binary SHA256: `b0ee9ba92b2a3ab74c382273ea2fc82763277671b436581b5fc47e0d9b896e00`
- Product default binary SHA256: `82edfb8e6561f87eef067d3ea7fe5327b54f3cc9450d6c42cf63fe72963aec66`
- PASS line: `VERDICT: gemma3 Marlin cache-policy native CUDA probe complete`
- Vast cleanup: instance `40826362` confirmed `stopped/exited`

## Key Rows

Product-style workspace-zero chain timing:

| m | legacy_plain chain_event_us | product_default chain_event_us | delta | legacy_plain down_us | product_default down_us |
|---:|---:|---:|---:|---:|---:|
| 1 | 209.607 | 202.714 | -3.3% | 70.797 | 66.830 |
| 10 | 214.821 | 208.771 | -2.8% | 70.906 | 68.152 |
| 16 | 215.344 | 211.791 | -1.6% | 70.496 | 68.852 |
| 23 | 224.792 | 221.724 | -1.4% | 75.040 | 73.970 |
| 32 | 227.980 | 225.103 | -1.3% | 75.653 | 75.414 |

Kernel-only workspace-prezero diagnostic timing:

| m | legacy_plain chain_event_us | product_default chain_event_us | delta | legacy_plain down_us | product_default down_us |
|---:|---:|---:|---:|---:|---:|
| 1 | 207.375 | 199.864 | -3.6% | 68.584 | 64.812 |
| 10 | 210.664 | 205.838 | -2.3% | 68.493 | 66.077 |
| 16 | 213.098 | 208.816 | -2.0% | 68.574 | 66.695 |
| 23 | 222.267 | 218.506 | -1.7% | 72.764 | 71.882 |
| 32 | 225.525 | 222.195 | -1.5% | 73.469 | 73.419 |

## Interpretation

After `c76bfcfa`, the product-default Marlin kernel path matches the previously
validated evict-first variant. The default path improves product-shaped tail-MLP
chain timing over the legacy plain `cp.async.cg` path across all tested rows,
with the main release-relevant m16/m32 rows improving about 1-2%.

This validates the default-path source change at native CUDA scope. It is still
diagnostic evidence, not endpoint release evidence: `ferrum run` and
`ferrum serve` correctness must pass on a product binary before any endpoint
performance claim, and the final W2 validator has not produced
`MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.
