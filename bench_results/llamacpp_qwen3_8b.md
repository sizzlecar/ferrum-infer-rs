| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | BLAS,MTL   |       8 |           pp512 |       346.52 ± 25.26 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | BLAS,MTL   |       8 |           tg128 |         27.94 ± 0.90 |

build: 19821178b (8960)
