# W2 CUDA Graph Segment Native Probe

- Date: 2026-06-16
- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_cuda_graph_segment_native_probe_2026-06-16`
- Scope: native CUDA microbench only; no Ferrum product build, no model load, no release gate.
- vLLM source referenced locally: `/Users/chejinxuan/py_ws/vllm`.
- Hardware: existing Vast instance `40826362`, 1x RTX 4090, driver 565.77, nvcc 12.4.131.
- Binary SHA256: `9614573b5df34e77e971e57cc3a43f0b2154368912c4fe9d022eb5fa2cdd2a9b`

## Command

```bash
nvcc -O3 -arch=sm_89 -std=c++17 cuda_graph_segment_probe.cu -lcuda -lcudart -o cuda_graph_segment_probe && ./cuda_graph_segment_probe --segment-layers=1 --timed-iters=60 --warmup-iters=6
```

## Result

The probe returned 0 and printed:

```text
VERDICT: CUDA graph segment probe complete
```

Key metrics:

- eager: `1.108744 ms/step`
- monolithic graph replay: `0.795237 ms/step`, instantiate `1.963643 ms`
- segmented graph replay: `0.880079 ms/step` across `62` graphs, instantiate total `2.662781 ms`
- segmented replay overhead vs monolithic: `1.194735x`

## Interpretation

This rules out a too-simple explanation: a Gemma3-like launch count alone does not make CUDA graph instantiation fail on the RTX 4090. Both the monolithic simple graph and 62 segmented simple graphs instantiate and replay cleanly.

That makes the prior Ferrum `--unified-graph` `CUDA_ERROR_OUT_OF_MEMORY` more likely to come from the real captured contents: Marlin/attention/final logits resource usage, graph memory pool interaction, or capturing too much of the final norm/lm_head/logit packing region in one graph.

The vLLM-aligned next step is not another runtime knob sweep. It is to change the Ferrum diagnostic graph path toward segmented/breakable capture around attention or layer groups, with persistent buffers and final logits kept out of the largest captured region, then run a correctness smoke before any performance measurement.

## Cleanup

Vast cleanup confirmed `cur_state=stopped`, `actual_status=exited`, `intended_status=stopped`.

No release PASS line was produced; this is diagnostic evidence only.
