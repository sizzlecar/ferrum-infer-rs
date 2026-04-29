| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | BLAS,MTL   |       8 |           pp512 |       335.13 ± 24.86 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | BLAS,MTL   |       8 |           tg128 |         29.20 ± 0.44 |

build: 19821178b (8960)
