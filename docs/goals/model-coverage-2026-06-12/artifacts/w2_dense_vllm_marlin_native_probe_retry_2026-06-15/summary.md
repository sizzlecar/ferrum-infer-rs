# W2 dense vLLM Marlin native probe retry

Status: PASS

Key m=16 timings (us):

| shape | Ferrum hot | Ferrum weight-cycle | vLLM dense Marlin |
|---|---:|---:|---:|
| qkv | 17.005 | 30.278 | 18.354 |
| gate_up | 133.715 | 133.985 | 136.581 |
| down | 30.356 | 68.651 | 36.277 |

Product profile batch used: 16

Conclusion: vLLM dense Marlin does not explain the main c16 gap. Gate_up is effectively tied with Ferrum hot kernel. Down matches a weight-cycle/cache-residency issue more than a kernel-selection issue.
