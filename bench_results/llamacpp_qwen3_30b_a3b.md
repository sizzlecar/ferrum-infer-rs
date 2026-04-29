| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | BLAS,MTL   |       8 |           pp512 |        596.58 ± 3.89 |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | BLAS,MTL   |       8 |           tg128 |         44.52 ± 6.80 |

build: 19821178b (8960)
