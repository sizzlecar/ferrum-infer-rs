# W2 same-pod c16 vLLM/Ferrum ShareGPT diagnostic

- Status: diagnostic_pass
- vLLM output TPS mean/LCB: 500.670 / 478.395
- Ferrum output TPS mean/LCB: 422.345 / 414.592
- Ferrum LCB / vLLM LCB: 0.8666
- vLLM p95 ITL: 33.070 ms
- Ferrum p95 ITL: 52.819 ms
- Ferrum p95 ITL / vLLM p95 ITL: 1.5972
- c16 throughput gate diagnostic: True
- c16 p95 ITL gate diagnostic: False
- vLLM completed/errors/source: [100, 100, 100] / [0, 0, 0] / usage
- Ferrum completed/errors/source: [100, 100, 100] / [0, 0, 0] / usage
