# FastH3 time-table precomputation

This Rust example executes the same managed time-conditioning component used
by model composition. It reads the original dense checkpoint, promotes the
time MLP parameters through the existing Host `WeightBacking` source, and
uploads only one block's weights at a time. It does not run the transformer or
produce video. Generated tables are experimental until the target numerical
profile and downstream consumer have been qualified.

Run with `cargo run -p ferrum-models --example precompute_fast_h3_time --features cuda -- --config /outside/repo/time.json`.
The CUDA build requires the repository's pinned native operator-set lock.
Keep configuration, outputs and reports outside the repository.

Minimal configuration (all paths must be real local paths on the CUDA host):

```json
{
  "model_dir": "/models/FastH3-Omni-NVFP4-QAD",
  "output_dir": "/evidence/time-block0",
  "first_block": 0,
  "block_count": 1,
  "iterations": 2
}
```

The output directory must not exist; its parent must exist. A complete table
set uses `block_count: 50` for the current checkpoint. Each block receives the
union of all video/audio/reference times in the checkpoint's PDD schedule;
references use `max(video_time, .999)` and audio references use `1`. The output
contains F32 time embeddings and BF16 `[time, 18 * hidden]` modulation rows.
Rows contain video/text/audio modalities, each with six modulation fields.

Defaults are declared in `arguments.rs`: device 0; 2 GiB usable device limit;
256 MiB Host payload limit; 256 MiB bootstrap allowance; 16 MiB stream allowance
and upload staging; 1 MiB per metadata file; 64 GiB source payload; 4096 tensors;
128 PDD steps. Every field is an explicit JSON option and unknown fields fail.
These limits are admission limits, not measured peak usage. Source mappings
and metadata are outside the Host payload account.

Every output carries configuration, source component SHA-256 values, plan and
provider fingerprints, time values and routing. `manifest.json` is published
after all selected components close successfully and their Host outputs are
released. An interrupted directory without a manifest is incomplete. Component
failures retain the original native error and owner chain until process exit;
they do not claim successful retirement.

Validation currently checks finite output, same-runtime repeatability, no
execution-time source rereads, and retained outputs surviving component close.
The sinusoid uses CPU libm 0.2.15; CUDA uses pedantic F32 time MLP and libdevice
SiLU followed by BF16 AdaLN. These are explicit implementation identities,
not proof of Torch bitwise equality or generated-video quality. Reported wall
times include upload/readback and are not serving SLO or throughput evidence.
