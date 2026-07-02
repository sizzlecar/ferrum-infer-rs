# W2 Ferrum-first diagnostic summary

This is diagnostic evidence only. It does not replace the W2 release-grade gate.

- Status: `diagnostic_pass`
- Remote git SHA: `8181a99d7d6f9167f31dccd4767b31304a3c4e50`
- Remote git status: `clean`
- Ferrum bench exit code: `0`
- Server max-num-seqs: `32`
- Server kv-capacity: `512`

| c | Ferrum mean output tok/s | Ferrum LCB output tok/s | p95 ITL ms | completed | errors | clean |
|---:|---:|---:|---:|---|---|---|
| 1 | 46.076 | 46.066 | 21.347 | [100, 100, 100] | [0, 0, 0] | True |
| 4 | 157.744 | 157.091 | 24.205 | [100, 100, 100] | [0, 0, 0] | True |
| 16 | 448.504 | 434.308 | 35.745 | [100, 100, 100] | [0, 0, 0] | True |
| 32 | 515.058 | 512.760 | 35.902 | [100, 100, 100] | [0, 0, 0] | True |
