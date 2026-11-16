use super::super::encoder::tests as fixture;
use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;

fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> T {
    T::try_from(s.into()).unwrap()
}

#[test]
fn reference_trim_preserves_full_qwen_source_and_pads_from_the_last_valid_vae_frame() {
    let mut metadata = serde_json::to_value(fixture::architecture()).unwrap();
    metadata["temporal_downsample_factors"] = serde_json::json!([2, 1]);
    let a = FastH3VideoVaeConfig::parse(&serde_json::to_vec(&metadata).unwrap(), 4096).unwrap();
    let (_dir, archive) = fixture::archive(&fixture::original_weights());
    let prepare = |policy| {
        PreparedFastH3TiledVideoEncoder::prepare(
            &a,
            archive.clone(),
            FastH3VideoEncoderGeometry {
                batch: 1,
                frames: 38,
                height: 8,
                width: 8,
            },
            policy,
            FastH3ImageSpatialPolicy::WholeImageDiagnostic,
            fixture::limits(),
            FastH3VideoEncoderProfile::Reference,
        )
        .unwrap()
    };
    let all = prepare(FastH3VideoTemporalPolicy::AllFrames);
    let reference = prepare(FastH3VideoTemporalPolicy::ReferenceClipTrim);
    assert_eq!(reference.input_spec(), all.input_spec());
    assert_eq!(reference.output_spec().dimensions(), [1, 4, 15, 4, 4]);
    assert_eq!(all.output_spec().dimensions(), [1, 4, 24, 4, 4]);
    let host = fixture::host();
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 3 * 38 * 64)
        .unwrap();
    for c in 0..3 {
        for t in 0..38 {
            bytes.as_mut()[(c * 38 + t) * 64..(c * 38 + t + 1) * 64].fill((c * 50 + t) as u8);
        }
    }
    let input = ManagedHostTensor::new(
        id("value.video.trim.pixels"),
        reference.input.clone(),
        bytes.freeze(),
    )
    .unwrap();
    let mut request = reference
        .begin(&input, id("request.video.trim"), &host)
        .unwrap();
    request.next_clip = 1;
    let tile = request.tile_input(0).unwrap();
    for c in 0..3 {
        for t in 0..17 {
            let expected = (((c * 50 + (17 + t).min(25)) as f32 * (1f64 / 255.) as f32)
                - [0.485f32, 0.456, 0.406][c])
                / [0.229f32, 0.224, 0.225][c];
            for cell in tile.bytes().bytes()[(c * 17 + t) * 64 * 4..(c * 17 + t + 1) * 64 * 4]
                .chunks_exact(4)
            {
                assert_eq!(cell, expected.to_le_bytes());
            }
        }
    }
    drop((request, input, tile));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn video_tiles_preserve_channels_time_seams_padding_and_resume_after_pressure() {
    let mut metadata = serde_json::to_value(fixture::architecture()).unwrap();
    metadata["temporal_downsample_factors"] = serde_json::json!([2, 1]);
    let a = FastH3VideoVaeConfig::parse(&serde_json::to_vec(&metadata).unwrap(), 4096).unwrap();
    let weights = fixture::original_weights();
    let (_dir, archive) = fixture::archive(&weights);
    let encoder = PreparedFastH3VideoEncoder::prepare(
        &a,
        archive,
        FastH3VideoEncoderGeometry {
            batch: 2,
            frames: 17,
            height: 12,
            width: 12,
        },
        fixture::limits(),
        FastH3VideoEncoderProfile::Reference,
    )
    .unwrap();
    // Scaled four-tile grid exercises the public baseline's identical raw
    // neighbour rule with genuine full causal CNN output at every tile.
    let prepared = PreparedFastH3TiledVideoEncoder {
        fingerprint: encoder.family().fingerprint().unwrap(),
        encoder,
        spatial: PH3ImageVaeTilePlan::new(
            20,
            20,
            2,
            PH3ImageVaeSpatialPolicy::Tiled {
                tile_pixels: 12,
                minimum_overlap_pixels: 4,
                max_tiles: 4,
            },
        )
        .unwrap(),
        temporal: FastH3VideoEncodePlan::new(&a, 22, 15).unwrap(),
        input: spec(vec![2, 3, 22, 20, 20], ElementType::U8).unwrap(),
        output: spec(vec![2, 4, 15, 10, 10], ElementType::F32).unwrap(),
    };
    let host = fixture::host();
    let c = ReferenceVNextComposition::create(id("device.video_tiles.reference")).unwrap();
    let d = c.runtime().descriptor();
    let compiled = CompiledTensorComponent::compile(
        prepared.encoder.family().clone(),
        c.registry(),
        c.catalog(),
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        c.weight_materializers(),
        prepared.encoder.compile_options(),
    )
    .unwrap();
    let mut session = compiled
        .initialize(
            c.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(prepared.encoder.weight_source()),
            ComponentInitializationConfig {
                run_id: id("run.video_tiles.init"),
                request_id: id("request.video_tiles.init"),
                transaction_id: id("transaction.video_tiles.init"),
                static_policy: StaticInitializationPolicy::new(1 << 20, 128).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let mut bytes = host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            prepared.input.minimum_storage_bytes().unwrap() as usize,
        )
        .unwrap();
    for (i, v) in bytes.as_mut().iter_mut().enumerate() {
        *v = ((i * 7 + i / 400) % 251) as u8;
    }
    let input = ManagedHostTensor::new(
        id("value.video.reference.pixels"),
        prepared.input.clone(),
        bytes.freeze(),
    )
    .unwrap();
    assert!(prepared
        .begin(&input, id("request.video_tiles.foreign"), &fixture::host())
        .is_err());
    let baseline_usage = host.usage().unwrap().committed_bytes();
    assert!(prepared
        .begin(&input, id("request.video_tiles.partial"), &host)
        .unwrap()
        .finish()
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), baseline_usage);
    let mut request = prepared
        .begin(&input, id("request.video_tiles.execute"), &host)
        .unwrap();
    // Occupy the remaining shared quota. Releasing the reservation must permit
    // retry without losing progress or submitting an extra native tile.
    // Exhaust Output's remaining class capacity to refuse the first canvas.
    let available = host.limits().class_bytes(HostMemoryClass::Output)
        - host
            .usage()
            .unwrap()
            .class_committed_bytes(HostMemoryClass::Output);
    let pressure = host.reserve(HostMemoryClass::Output, available).unwrap();
    assert!(matches!(
        request.advance(&mut session, id("run.video_tiles.pressure")),
        Err(FastH3VideoMomentError::Preparation(_))
    ));
    assert_eq!(request.executed_tiles(), 0);
    drop(pressure);
    for tile in 0..8 {
        assert_eq!(
            request
                .advance(&mut session, id("run.video_tiles.execute"))
                .unwrap(),
            tile == 7
        );
        assert_eq!(request.executed_tiles(), tile + 1);
        assert_eq!(request.completed_clips(), (tile + 1) / 4);
    }
    let moments = request.finish().unwrap();
    assert_eq!(moments.tensor(), prepared.output_spec());
    let expected = independent(&weights, input.bytes().bytes());
    for (i, (word, e)) in moments
        .bytes()
        .bytes()
        .chunks_exact(4)
        .zip(expected)
        .enumerate()
    {
        let actual = f32::from_le_bytes(word.try_into().unwrap());
        assert!(
            actual.is_finite() && (f64::from(actual) - e).abs() < 1e-4 * (1. + e.abs()),
            "video stitched moment {i}: {actual} vs {e}"
        );
    }
    assert!(session.close().is_ok());
    drop((input, prepared, compiled, c));
    assert!(host.owns(moments.bytes()));
    drop(moments);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

fn independent(weights: &fixture::Weights, pixels: &[u8]) -> Vec<f64> {
    let mut result = vec![0.; 2 * 4 * 15 * 10 * 10];
    for clip in 0..2 {
        let mut raw = Vec::new();
        for tile in 0..4 {
            let mut input = Vec::new();
            for batch in 0..2 {
                for channel in 0..3 {
                    for t in 0..17 {
                        for y in 0..12 {
                            for x in 0..12 {
                                let frame = (clip * 17 + t).min(21);
                                let index = ((((batch * 3 + channel) * 22 + frame) * 20
                                    + tile / 2 * 8
                                    + y)
                                    * 20)
                                    + tile % 2 * 8
                                    + x;
                                input.push(
                                    ((f64::from(pixels[index]) / 255.
                                        - [0.485, 0.456, 0.406][channel])
                                        / [0.229, 0.224, 0.225][channel])
                                        as f32,
                                );
                            }
                        }
                    }
                }
            }
            raw.push(fixture::oracle::encode_with_final_downsample(
                weights,
                &input,
                FastH3VideoEncoderGeometry {
                    batch: 2,
                    frames: 17,
                    height: 12,
                    width: 12,
                },
                false,
            ));
        }
        let mut canvas = vec![0.; 2 * 4 * 9 * 10 * 10];
        for tile in 0..4 {
            let mut work = raw[tile].clone();
            let row = tile / 2;
            let col = tile % 2;
            for plane in 0..2 * 4 * 9 {
                if row > 0 {
                    for y in 0..2 {
                        for x in 0..6 {
                            let index = plane * 36 + y * 6 + x;
                            let a = raw[tile - 2][plane * 36 + (4 + y) * 6 + x];
                            work[index] = a * (1. - y as f64 / 2.) + work[index] * (y as f64 / 2.);
                        }
                    }
                }
                if col > 0 {
                    for y in 0..6 {
                        for x in 0..2 {
                            let index = plane * 36 + y * 6 + x;
                            let a = raw[tile - 1][plane * 36 + y * 6 + 4 + x];
                            work[index] = a * (1. - x as f64 / 2.) + work[index] * (x as f64 / 2.);
                        }
                    }
                }
                for y in 0..if row == 0 { 4 } else { 6 } {
                    for x in 0..if col == 0 { 4 } else { 6 } {
                        canvas[(plane * 10 + row * 4 + y) * 10 + col * 4 + x] =
                            work[plane * 36 + y * 6 + x];
                    }
                }
            }
        }
        let keep = if clip == 0 { 9 } else { 6 };
        for bc in 0..8 {
            for t in 0..keep {
                let from = (bc * 9 + t) * 100;
                let to = (bc * 15 + clip * 9 + t) * 100;
                result[to..to + 100].copy_from_slice(&canvas[from..from + 100]);
            }
        }
    }
    result
}
