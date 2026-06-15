# W2 paged-KV unified smoke diagnosis

- artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_paged_kv_unified_cuda_smoke_2026-06-16`
- source commit tested: `103c7013e849b198cabaa7ad47cd45063bf21e6d`
- build: PASS (`build.rc=0`)
- serve: ready poll `61`
- chat correctness: FAIL, content was empty, `completion_tokens=1`, `finish_reason=stop`
- bench: not run; stopped at correctness failure
- key finding: `[unified-decode] ... fallback=false fallback_reason=none`; `paged_kv_required` was removed, but paged unified output is wrong
- server evidence: `split_qkv_norm_rope_paged` and `paged_varlen_attention` both executed before the empty response
- cleanup: Vast `40826362` stopped and `actual_status=exited`
- release-grade: no `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`; failure artifact only
