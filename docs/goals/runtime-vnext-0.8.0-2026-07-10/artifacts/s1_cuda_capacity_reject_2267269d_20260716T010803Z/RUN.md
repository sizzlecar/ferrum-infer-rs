# S1 CUDA capacity diagnostic: REJECT

- Lane: `vnext-s1-cuda-capacity`
- Source: `2267269d5392c1a8149e1849d2ef384c92569492` (clean)
- Binary SHA256: `80f03fc858e2da9d1595f2a3e6e1ca9bc6254e31236266d81f0fb6a9cbbb7858`
- Hardware: 1x RTX 4090, 23028 MiB, driver 595.45.04
- Model: `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Result: `REJECT prefill_first_wait_for_release_decode_livelock`

The product `ferrum run` entrypoint passed with `Paris`, two completion tokens, and
484.728301 ms generation time. Target warmup A/C also passed with 128/16 output
tokens, one usage object, and one `[DONE]` each. The exact calibrated runtime budget
was 8,474,209,616 bytes.

The pressure run then formed a scheduler cycle. A and C each completed their 131-node
vNext prefill frame, while B entered typed `WaitForRelease` without a device
submission. The scheduler snapshot remained at one waiting request, two active decode
requests, and two decode-queue entries. Because the product command set
`--scheduler-prefill-first-until-active 4`, fill-first suppressed both active decodes
while it tried to reach four active requests. B could not become eligible until A or C
decoded and advanced the release epoch, so no member of the cycle could progress.

Over 250.303642 seconds the engine emitted 117,432
`vnext.prefill_admission_skipped_unchanged` events but only 12 admission probes. The
target trace reached 226,722,786 bytes and all three pressure client files remained
incomplete. The run was stopped at the diagnostic threshold and produced no capacity
PASS line.

This is not CUDA OOM, a kernel failure, an executor lane/fence lock, or staged B
overwriting active sequence state. B reported `execution_authority_retained=false` and
never emitted `request_accepted`, `frame_started`, or `operation_submitted`. The
stalled interval also contained no decode workspace events, proving the engine never
re-entered the model executor. The defect is the scheduler's fill-first progress
policy: a throughput preference treated an unchanged typed capacity wait as runnable
prefill work and stopped the decode work required to satisfy that wait.

The next candidate restores decode scheduling whenever fill-first produces no runnable
prefill in the current iteration. The next paid run must stop and remain REJECT if A/C
make no token progress for 30 seconds after B defers, unchanged skips exceed 512, the
target trace exceeds 16 MiB, any pressure stream lacks exactly one usage or `[DONE]`,
or the canonical capacity validator does not print its exact PASS line.

The canonical external artifact is
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-s1/vnext_capacity_2267269d_20260716T010803Z.tar.gz`
(`964604c78495af1dea600bae26c7b88745708b550bde605e8c82dd55eebab958`,
4,850,651 bytes). Vast instance `45022787` was stopped and confirmed
`cur_state=stopped`, `actual_status=exited` after the artifact was copied and verified.
