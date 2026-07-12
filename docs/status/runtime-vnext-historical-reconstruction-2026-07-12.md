# Runtime vNext Historical Failure Reconstruction Record

Date: 2026-07-12

This record closes source-evidence gaps in the Runtime vNext H01-H15 catalog. It does not claim
that a newly generated replay log is an original production log. Each row classifies its evidence
as one of:

- `fix_diff_reconstruction`: the failure state is reconstructed from a reviewed fix commit and its
  parent, plus any retained status or benchmark record.
- `regression_contract_mutation`: the original log is absent, but the introducing regression-test
  commit documents the exact mutation and expected failure.
- `historical_artifact`: a checked-in report already contains the observed failure state.

The content-addressed G00 receipts separately bind the frozen replay input, mutation, executed
command, non-zero return code, exact failure signature, and failure log. This document is source
provenance for those receipts, not a substitute for executing them.

## Reviewed Reconstructions

| Case | Evidence classification | Reviewed source | Frozen failure state |
|---|---|---|---|
| H02.1 | fix_diff_reconstruction | `69c7d05e6446067c81f9b0a8ef1d60e4c26d5f8f` and parent `318a5ddcc595d268d7014070b9547e314d8f3aae` | Chat reaches KV exhaustion before completing round four instead of completing five rounds. |
| H02.2 | fix_diff_reconstruction | `7fb268d47d4f58ea152a977643c77d96c7254f3b` and parent `42396872bd851b376b6b8111bca0c27d44b3aa20` | `ferrum run <gguf>` remains running beyond the bounded terminal deadline. |
| H02.3 | fix_diff_reconstruction | `2d5571d2a14b8d089a9327dc23b7fc7a2b049ea2`, corrective follow-up `c89b3ded77335ff1c82d90da03c047b4027e90a8`, and the retained Metal summary | Product admission proceeds below the two-GiB physical-headroom floor and swap grows during the run. |
| H07.1 | fix_diff_reconstruction | `db7e529322eba6a6e9aae13a10207ddd5ff28f8e` and `docs/bench/v0.2-cuda/status-2026-05-05-full-cuda-graph.md` | Actual batch sizes 16 and 32 collide on the same padded graph key. |
| H07.2 | fix_diff_reconstruction | `e4236366a631d9af14c5007c406222e694660e24` and the same graph report | `release()` invalidates a captured graph while a replay slot still refers to it. |
| H08.1 | fix_diff_reconstruction | `4701cc8445c7091d329c045f4ac7b595cef94349` and `docs/goals/test-architecture-2026-06-10/patches/hb-08.patch` | Scratch growth from 32 to 64 leaves the new paged-batch slots uninitialized. |
| H10.1 | fix_diff_reconstruction | `86633c2df3d657ae0ddd0ac421f8fb6ae674cc3d` and `docs/goals/model-coverage-2026-06-12/STATUS.md` | CUDA `scale_inplace` routes through a host default and changes an f16 residual to f32. |
| H12.2 | fix_diff_reconstruction | `2203a9cd9c0b1244705c756c6c204235638c8477`, `7476295db6f4c5510d83328e450440b2aec1d8cf`, and the model-coverage status | Capacity-deferred work is readmitted in the same capacity epoch with no progress. |
| H12.4 | fix_diff_reconstruction | `b8116dc498e7fad50263d0dc580daf38194cb74e`, `16e8ee6ba2d4a9ecf3243f45052fbb0c17225682`, and the model-coverage status | Capacity defer retains an existing KV handle and resource closure remains incomplete. |
| H13.1 | regression_contract_mutation | `9d81ed9f95c1a5fb8f683b5db80e9b39f5b49abc` mutation 15 in the commit message | A successful SSE stream contains zero `[DONE]` events. |
| H13.2 | regression_contract_mutation | `9d81ed9f95c1a5fb8f683b5db80e9b39f5b49abc` plus the exact-count contract in `vllm_migration_compat.rs` | A successful SSE stream contains two `[DONE]` events. |
| H13.3 | regression_contract_mutation | `36e3dc2b787fd57198ef174356dee470be555a3e` and `docs/status/openai-api-compat-2026-05-30.md` | An `include_usage` stream reaches `[DONE]` without a final usage chunk. |
| H13.4 | regression_contract_mutation | `36e3dc2b787fd57198ef174356dee470be555a3e` plus the exact-count contract in `vllm_migration_compat.rs` | An `include_usage` stream contains two usage chunks. |
| H14.1 | historical_artifact | `eacfb09d94c162757a172a12f331cefa0e6d5016` and `docs/goals/model-coverage-2026-06-12/w1_matrix.json` | Qwen3-Coder CUDA selects EOS at position zero and returns zero output tokens. This remains an historical open failure, not a claimed fix. |
| H15.2 | fix_diff_reconstruction | `d57ff68ad3b0ea47d3156c753f58594cead010f8`, `0a57ff9686111ebd34aee13849f24e32f5b7f164`, and the model-coverage status | Fallback leaves recurrent or KV ownership attached to request A and request B can observe the stale owner. |

## Receipt Contract

Every production receipt for the rows above must use this record with evidence role
`historical_failure`, include every commit and historical artifact listed by the reviewed catalog,
and execute the corresponding replay from
`scripts/release/configs/runtime_vnext_historical_replays.json`. A passing current regression test,
an unexecuted JSON document, or a hand-written failure log is insufficient.
