# W2 active-decode prefill budget c16 diagnostic

This is diagnostic evidence only and did not produce `MODEL_RELEASE_GRADE_W2 PASS`.

- Status: `diagnostic_pass`
- Ferrum completed/errors/source: [100, 100, 100] / [0, 0, 0] / usage
- Ferrum output TPS mean/LCB: 463.405 / 460.553
- Same-pod vLLM reference TPS mean/LCB: 500.670 / 478.395
- Ferrum LCB / vLLM LCB: 0.9627
- Ferrum p95 ITL: 29.247 ms
- Same-pod vLLM reference p95 ITL: 33.070 ms
- Ferrum p95 ITL / vLLM p95 ITL: 0.8844
- Delta vs previous Ferrum LCB: 127.443 tok/s
- Delta vs previous Ferrum p95 ITL: 2.610 ms
- c16 throughput diagnostic pass: True
- c16 p95 ITL diagnostic pass: True
