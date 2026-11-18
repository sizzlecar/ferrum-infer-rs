# FastH3 video generation

Ferrum runs `FastVideo/FastH3-Omni-NVFP4-QAD` through both
`ferrum run` and `ferrum serve`. Runtime qualification currently covers Linux
CUDA on an RTX 5090 under WSL2. The pipeline uses the checkpoint's eight
evaluations, NVFP4 weights, 36-layer quality video decoder and generated audio.
It requires at least one image or video reference.

## Setup

Use a build with `cuda-fast-h3` and its pinned native operator set. On the
RTX 5090, use CUDA 12.8 and cuDNN 9.20.0.48 **for CUDA 12**. MP4 output uses
FFmpeg with libx264 and AAC encoding. Video references also need ffprobe.
Follow the [shared GPU runtime setup](image-generation.md#p-h3-setup).
On Ubuntu, `sudo apt-get install --no-install-recommends ffmpeg` supplies
the encoder and probe tool. An existing compatible installation can be reused.
On a minimal Ubuntu 24.04.5 x86_64 image, installing FFmpeg and its distribution
dependencies downloaded 133 MB; APT estimated an additional 457 MB on disk.
This cost is in addition to the shared GPU runtime and depends on what is
already installed. Ferrum's archive and the model weights are separate.
The tested model revision is `f01e6a23ac6066aa9eb622cf94b6259442e70ec2`.
Keep the complete model directory, including the reference transformer,
packed NVFP4 weights, text/vision encoder, video/audio VAEs, schedulers,
processor files and tokenizer.

Copy the [RTX 5090 Ref2VA preset](../crates/ferrum-cli/examples/config/fast-h3-rtx5090-ref2va.json)
and set `model_directory` to the installed checkpoint and `output_directory`
to an absolute spool and results directory. Relative model paths resolve from
the process working directory. Both commands accept this file through
`--video-config`.

The [expanded configuration](../crates/ferrum-cli/examples/config/fast-h3-video-cuda.json)
is the source of this preset's compute, memory, serving and encoding settings.
Use the expanded form for custom paths, capacity limits or encoding options.
The CUDA 12.8 configuration reproduced the 124-frame release sample byte for
byte on the 5090.

Both commands find FFmpeg on the system PATH and encode a single test frame
with stereo audio before loading the model. This checks the installed build's
H.264/AAC encoders and fragmented MP4 output. An absolute `output.ffmpeg`
selects a different encoder. Reference video decoding uses the same FFmpeg by
default; set `input.ffmpeg` to select a separate build. Ferrum looks for
ffprobe beside the input FFmpeg first, then on PATH. Set `input.ffprobe` to
override that choice. PNG and WAVE references work without ffprobe.

This preset explicitly declares 29 GiB of managed device capacity, 4 GiB of
managed Host buffers, 32768 graph nodes/command records, automatic FFN row
partitioning and up to two active denoising requests. These are separate
limits: the Host buffer allowance is not a process RAM limit. The long-reference
qualification observed about 39.3 GiB server RSS. Device capacity and private
allocator reservations are not measurements of whole-device peak VRAM.

In the expanded form, omitting
`maximum_plan_nodes` selects 16384, which is insufficient for the
qualified 15-second long-reference graph. Keep the explicit node and command
limits when using that workload. A larger metadata limit does not increase the
GPU byte budget.

## CLI

```sh
ferrum run --video-config fast-h3-rtx5090-ref2va.json \
  --reference image=reference.png \
  --prompt 'The person in <Picture 1> turns toward the lake. The camera moves slowly closer.' \
  --width 832 --height 480 --num-frames 124 --fps 24 --seed 1101 \
  --output clip.mp4

ferrum run --video-config fast-h3-rtx5090-ref2va.json \
  --reference video=reference.mp4 \
  --prompt 'Use <Video 1> for the scene and camera movement. Continue the movement smoothly.' \
  --width 832 --height 480 --num-frames 362 --fps 24 --seed 1301 \
  --output continuation.mp4

ferrum serve --video-config fast-h3-rtx5090-ref2va.json --host 127.0.0.1 --port 8000
```

Repeat `--reference image=...`, `video=...` or `audio=...` in the intended order.
Prompts can refer to `<Picture 1>`, `<Video 1>` and `<Audio 1>` using each
modality's order. The image-only `--image` form remains accepted; use one
reference representation per request. Existing output files are not overwritten.

Output is H.264/AAC MP4 at 24 FPS. Canvas dimensions must be multiples of 32;
the requested frame count must already satisfy `5 + 17*n`. Common tested
counts are 124, 243 and 362, approximately 5, 10 and 15 seconds respectively.
The actual encoded duration follows those frame counts.

Images are static RGB/RGBA8 PNGs. Standalone audio references are 32 kHz mono
or stereo PCM8/16/24/32 or IEEE-F32 WAVE. Video references are decoded by the
configured FFmpeg/ffprobe; MP4, WebM, QuickTime and Matroska MIME types are
accepted by HTTP. The model selects up to the first 15 seconds and limits the
reference to the requested output interval. A video's supported soundtrack is
resampled for conditioning. The declared reference count limits do not certify
that every combination fits the configured memory budget.

## HTTP jobs

The `ferrum-video-jobs-v1` protocol is asynchronous. Submit JSON to
`POST /v1/videos` with the required fields below. `data_url` contains the actual
base64-encoded file; the server does not fetch external URLs.

```json
{
  "model": "FastH3-Omni-NVFP4-QAD",
  "prompt": "The person in <Picture 1> turns toward the lake.",
  "seed": 1101,
  "width": 832,
  "height": 480,
  "num_frames": 124,
  "fps": 24,
  "references": [
    {"type": "image", "data_url": "data:image/png;base64,<encoded file>"}
  ]
}
```

Use `type: "video"` with `data:video/mp4;base64,...`, or `type: "audio"`
with `data:audio/wav;base64,...` for the other reference kinds.

Poll `GET /v1/videos/{id}` until `completed`, `failed` or `cancelled`.
`in_progress` includes queuing, generation and encoding. Download a completed
artifact from `GET /v1/videos/{id}/content`. `DELETE /v1/videos/{id}` requests
cancellation and cleanup; cancellation is observed at safe execution boundaries.
`output.maximum_jobs` (default 16) limits uploads and running jobs. A completed
job releases both its active slot and its encoder output session. Results use
separate durable storage under `output.directory/jobs-v1`: by default at most
128 terminal jobs (`output.maximum_retained_jobs`) and 8 GiB of MP4s
(`output.maximum_retained_bytes`). The encoder's temporary spool independently
allows 2 GiB (`output.maximum_spool_bytes`); these two byte limits are additive,
with bounded job metadata in addition.

`output.retention_seconds` defaults to 86400 (24 hours), measured from the
returned `finished_at` timestamp, including failed/cancelled records. Expiry is
reclaimed on the next job admission, lookup, completion or server startup.
Capacity pressure may evict the oldest safe terminal records sooner. DELETE
also reclaims results; downloads already holding a result can finish, and their
files remain charged until the reader closes. If those readers occupy all
retained storage, new results can fail persistence instead of exceeding the
configured disk bound. Unreaped encoder or panic owners are never evicted by
TTL: the service preserves their active admission and reports incomplete
shutdown. Unknown files are preserved and cause startup to reject that store.

Completed records and MP4s survive an orderly restart with the same directory.
An unfinished job recovered after a restart reports `failed` with an interruption
error; GPU computation is not resumed. The store rejects concurrent processes,
corrupt committed records, and limits smaller than its existing stored data.
Keep the directory on persistent local storage. Existing anonymous temporary
spools from older processes are not adopted or treated as disposable user data.
`ferrum run --output` continues to publish its requested destination directly;
HTTP retention settings apply to `ferrum serve`.

Uploads have a 30-second idle deadline
(`serving.upload_idle_timeout_seconds`) and a 300-second total deadline
(`serving.upload_timeout_seconds`). Exceeding either returns HTTP 408 and
releases ingress/Host allocations. These deadlines cover receiving the body,
not generation or encoding time.
Authentication uses the ordinary `serve --api-key` configuration.

## Batching and qualified limits

Compatible requests can join and leave the shared denoising loop at step
boundaries. `serving.max_num_seqs` is an upper bound, not a guaranteed batch
size. Memory and predicted work determine whether a second request joins;
requests can wait and complete serially. `/health` exposes active, pending and
preparing counts. A full ingress queue or active-job limit still requires the
client to wait before submitting more work.

The current candidate has completed a 15-second video reference to a 362-frame
480p output in both CLI and HTTP, with identical MP4 bytes and complete audio/video
decode. Two accepted long-reference requests producing 243 frames each also
completed serially, leaving the worker and queue empty. That functional run
took about 636 seconds for the 362-frame HTTP request and 421/868 seconds for
the two concurrent requests. It missed the predeclared 600-second latency target;
it establishes completion for these inputs, not a FastVideo speed advantage.

Earlier qualified workloads include 480p image references at 124/243/362 frames,
short video references, image/audio combinations, and one 1344×768 image-reference
124-frame output. The 768p result does not establish long-reference, concurrency
or comparative-performance support at that resolution.

The measured 480p five-second pipeline completed three paired repetitions at
C1 and C2 against FastVideo on the same 5090. Both used the same checkpoint,
NVFP4, eight evaluations, 36-layer decoder, generated audio, 29 GiB capacity
and H.264/AAC output settings. Each cell contained two timed requests after
a separate-reference warmup; all 24 timed outputs were valid.

| Measurement | Ferrum, range across runs | FastVideo, range across runs |
| --- | --- | --- |
| C1 median completion latency | 55.02–55.88 s | 80.39–83.18 s |
| C2 completed clips per minute | 1.043–1.050 | 0.714–0.740 |

The geometric mean of paired Ferrum/FastVideo ratios was 0.677 for C1 latency
(95% interval 0.651–0.703) and 1.437 for C2 throughput (1.367–1.511).
All six Ferrum cells met the predeclared 120-second completion-latency target;
the largest empirical P99 was 115.02 seconds. These are small engineering
samples, not production-tail certification. C2 did not improve Ferrum's own
C1 throughput on this workload, despite using real shared forwards.

GPU queries missed samples on both implementations, so their observed memory
peaks are lower bounds. The subsequent native-allocation recovery change
passed a real CUDA pool-capacity test and CLI/HTTP regressions, with identical
video artifacts to the previously qualified pipeline. The
[goal and evidence record](goal-fast-h3-video-5090.zh.md#9-当前进度与下一步)
identifies the frozen experiment, versions, source snapshots and raw reports.
