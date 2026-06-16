# W2 active-decode prefill budget c16 diagnostic

This is diagnostic evidence only and did not produce `MODEL_RELEASE_GRADE_W2 PASS`.

- Status: `diagnostic_pass`
- Ferrum completed/errors/source: [100, 100, 100] / [0, 0, 0] / usage
- Ferrum output TPS mean/LCB: 339.927 / 333.110
- Same-pod vLLM reference TPS mean/LCB: 500.670 / 478.395
- Ferrum LCB / vLLM LCB: 0.6963
- Ferrum p95 ITL: 26.637 ms
- Same-pod vLLM reference p95 ITL: 33.070 ms
- Ferrum p95 ITL / vLLM p95 ITL: 0.8055
- Delta vs previous Ferrum LCB: -81.482 tok/s
- Delta vs previous Ferrum p95 ITL: -26.183 ms
- c16 throughput diagnostic pass: False
- c16 p95 ITL diagnostic pass: True
