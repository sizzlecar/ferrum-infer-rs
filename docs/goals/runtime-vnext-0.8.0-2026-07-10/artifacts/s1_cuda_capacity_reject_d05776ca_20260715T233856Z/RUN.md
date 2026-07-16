# S1 CUDA capacity diagnostic: REJECT

- Lane: `vnext-s1-cuda-capacity`
- Source: `d05776caf963bc2a193b05e4ff9164823428517d` (clean)
- Binary SHA256: `0a155dcb1b540cb18505d9f110baffcfd8d6dd8b8bb5d00dbf31a0fee5695586`
- Hardware: 1x RTX 4090, 23028 MiB, driver 595.45.04
- Model: `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Result: `REJECT plan_budget_pressure_terminated_instead_of_waiting`

The product `ferrum run` entrypoint passed with `Paris`, two completion tokens, and
480.376942 ms generation time. Calibration and target replay used identical A/C
prompt hashes and 32-token prompt lengths. Warmup A/C and pressure A/C completed
their required 128/16 output tokens, so the earlier workload-mismatch defect did
not recur.

Pressure B reached a real transient capacity boundary while A remained active. Its
physical backing deferral observed release/capacity epochs 308/841 and requested
448 bytes from a pool with 192 bytes free. The exact plan budget was already fully
claimed at 8,474,209,616 bytes (8,411,511,808 static plus 62,697,808 dynamic).
Maintenance converted that valid live pressure into a terminal backend error nine
microseconds later, before any B device submission. HTTP streaming still returned
one usage object and one `[DONE]`, but with zero completion tokens and
`finish_reason=error`; A and later C completed normally.

This is not CUDA OOM, allocator exhaustion, model loading, or a kernel failure. It
is a typed admission/state-transition defect: recoverable plan-budget pressure must
preserve B's staged authority and fairness ticket, wait without probing or
submitting at unchanged epochs, and retry after A releases reusable backing.

The canonical external artifact is
`/Users/chejinxuan/ferrum-bench/artifacts/runtime-vnext-s1/vnext_capacity_d05776ca_20260715T233856Z.tar.gz`
(`81c754c78d4fcf96ea31fe94cfde11eea8331826b0d847064739f26cb4da06c6`, 210,379 bytes).
The repository retains only this conclusion and compact machine-readable summary.
