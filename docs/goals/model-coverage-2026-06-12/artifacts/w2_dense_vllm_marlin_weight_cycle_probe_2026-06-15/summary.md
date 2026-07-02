# W2 dense vLLM Marlin weight-cycle probe

Status: PASS

m=16 timings (us):

| shape | Ferrum hot | Ferrum weight-cycle | vLLM hot | vLLM weight-cycle |
|---|---:|---:|---:|---:|
| qkv | 17.005 | 30.278 | 18.315 | 30.950 |
| gate_up | 133.715 | 133.985 | 136.988 | 137.524 |
| down | 30.356 | 68.651 | 36.027 | 69.268 |

Conclusion: vLLM dense Marlin does not remove the down projection weight-cycle penalty. Dense kernel selection is not the main remaining c16 bottleneck.
