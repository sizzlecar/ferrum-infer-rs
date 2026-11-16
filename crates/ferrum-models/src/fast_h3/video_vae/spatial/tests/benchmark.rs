//! Reusable CPU diagnostic for the original 480p spatial stitching workload.
//! Source construction is outside timing; no inference or quality claim.
use super::*;

#[test]
#[ignore = "CPU performance diagnostic; use an optimized profile on the inference host"]
fn spatial_480p_host_blend_benchmark() {
    let budget = 768 << 20;
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            budget,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: budget,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let config = config();
    let plan = FastH3VideoSpatialPlan::new(
        &config,
        [1, 2, 37, 30, 52],
        FastH3VideoSpatialPolicy::default(),
        budget,
    )
    .unwrap();
    let temporal = FastH3VideoDecodePlan::new(&config, 37, 124).unwrap();
    let frames = temporal.chunk(0).unwrap().raw_frames;
    let raw = (0..plan.tile_count())
        .map(|tile| {
            let shape = [1, 3, frames, 256, 256];
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::Output, extent(shape, 2).unwrap())
                .unwrap();
            for (i, word) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
                let value = (tile as f32 - 5.) / 4. + (i % 401) as f32 / 128.;
                word.copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
            }
            FastH3VideoRawClip {
                shape,
                bytes: bytes.freeze(),
            }
        })
        .collect::<Vec<_>>();
    let mut seconds = Vec::new();
    for _ in 0..3 {
        let started = std::time::Instant::now();
        for _ in 0..temporal.chunk_count() {
            let mut canvas = plan.begin_canvas(frames, &host).unwrap();
            for (tile, raw) in raw.iter().enumerate() {
                canvas.push(tile, raw).unwrap();
            }
            let output = canvas.finish().unwrap();
            std::hint::black_box(output.bytes().bytes());
        }
        seconds.push(started.elapsed().as_secs_f64());
    }
    eprintln!(
        "{}",
        serde_json::json!({
            "scope": "CPU spatial canvas only, original 480p tile/overlap geometry",
            "tiles_per_chunk": plan.tile_count(),
            "raw_frames": frames,
            "temporal_chunks": temporal.chunk_count(),
            "repetitions": seconds.len(),
            "seconds": seconds,
        })
    );
    drop(raw);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
