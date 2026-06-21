# W3 Qwen35 Prefill-Step Diagnostic Lane

- Lane: W3 Qwen3.5 GPTQ scheduler prefill-step diagnostic.
- Hardware: exact 1x RTX 4090 Vast instance.
- Expected runtime/cost: 30-60 minutes; selected available offer $0.29555555555555557/hr, expected compute $0.15-$0.30 plus storage overhead.
- Stop condition: copy back smoke+c32 diagnostic artifacts, or stop immediately after start/SSH/build/smoke failure with logs copied back when possible.
- Correctness gate: `ferrum serve` non-stream chat smoke and stream chat with `stream_options.include_usage=true`.
- Performance command: c32 64x1 `ferrum bench-serve --fail-on-error --seed 9271` diagnostic, not release evidence.
- Release status: diagnostic only; no W3 completion claim without `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>`.
