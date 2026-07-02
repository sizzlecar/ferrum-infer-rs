# W3 Qwen35 Token-Row CUDA Diagnostic GPU Contract

- Lane: W3 Qwen3.5 GPTQ-Int4 CUDA build + correctness smoke + c32 diagnostic after token-row/full-attention and final-validator changes.
- Hardware: existing Vast instance 41422823, 1x RTX 4090.
- Expected runtime/cost: 30-60 minutes if model/build cache is usable; existing instance price from Vast API is 0.662962962962963 USD/hour, so expected cost is about 0.33-0.66 USD before stop.
- Stop condition: stop after correctness smoke and c32 diagnostic artifacts are copied back, or after first CUDA build/model-load/runtime failure with logs copied back; do not launch repeated full sweeps.
- Correctness gate: product-path `ferrum run` smoke and `ferrum serve` non-stream + streaming usage smoke for Qwen/Qwen3.5-35B-A3B-GPTQ-Int4.
- Performance command: diagnostic c32 `ferrum bench-serve --fail-on-error --seed 9271`; no release performance claim without L0-L5 PASS and release-shape `--require-ci --n-repeats 3`.
