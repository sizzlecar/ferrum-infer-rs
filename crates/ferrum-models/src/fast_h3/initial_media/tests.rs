use super::*;
use crate::fast_h3::{
    FastH3ManagedContinuation, FastH3PddPlan, FastH3PddSchedule, FastH3PredictionView,
    FastH3VelocityInput,
};
use std::sync::Arc;

fn media_host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                output_bytes: bytes,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}

fn latent(host: &HostMemoryAccount, dimensions: Vec<u64>, base: f32) -> ManagedHostTensor {
    let tensor = ResolvedTensorSpec::new(
        dimensions,
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let mut bytes = host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            tensor.minimum_storage_bytes().unwrap() as usize,
        )
        .unwrap();
    for (index, chunk) in bytes.as_mut().chunks_exact_mut(4).enumerate() {
        chunk.copy_from_slice(&(base + index as f32 / 128.0).to_le_bytes());
    }
    ManagedHostTensor::new(
        ProgramValueId::new("value.test.reference").unwrap(),
        tensor,
        bytes.freeze(),
    )
    .unwrap()
}

fn media_ids() -> [ProgramValueId; 2] {
    [
        ProgramValueId::new("value.test.video").unwrap(),
        ProgramValueId::new("value.test.audio").unwrap(),
    ]
}

#[test]
fn ordered_reference_inputs_keep_modality_order_audio_clean_and_target_rng_after_visuals() {
    let host = media_host(1 << 20);
    let image = latent(&host, vec![1, 24, 1, 2, 4], 1.0);
    let clip = latent(&host, vec![1, 24, 7, 4, 2], 10.0);
    let sound = latent(&host, vec![2, 32, 3], 20.0);
    let soundtrack = latent(&host, vec![2, 32, 2], 30.0);
    let target = FastH3VideoGeometry::new(64, 64, 5).unwrap();
    let geometry = [
        FastH3Reference::Audio {
            latents_per_channel: 3,
        },
        FastH3Reference::Image {
            latent_height: 2,
            latent_width: 4,
        },
        FastH3Reference::Video {
            latent_frames: 7,
            latent_height: 4,
            latent_width: 2,
            audio_latents_per_channel: Some(2),
        },
    ];
    let layout = FastH3PackedLayout::ref2va(3, &geometry, &target, 128).unwrap();
    let references = [
        FastH3ReferenceLatents::Audio(&sound),
        FastH3ReferenceLatents::Image(&image),
        FastH3ReferenceLatents::Video {
            video: &clip,
            audio: Some(&soundtrack),
        },
    ];
    let before = host.usage().unwrap().allocated_bytes();
    let [video, audio] =
        prepare_fast_h3_reference_media(314, &target, &layout, &references, media_ids(), &host)
            .unwrap();

    // Independently index (stereo,time,feature) rows from the original
    // (stereo,feature,time) encoder output; audio references are not augmented.
    let mut row = 0;
    for (samples, base) in [(3, 20.0_f32), (2, 30.0)] {
        for stereo in 0..2 {
            for t in 0..samples {
                for channel in 0..32 {
                    let source = (stereo * 32 + channel) * samples + t;
                    let destination = ((row + stereo * samples + t) * 32 + channel) * 4;
                    assert_eq!(
                        read(&audio.bytes().bytes()[destination..destination + 4]),
                        base + source as f32 / 128.0
                    );
                }
            }
        }
        row += samples * 2;
    }
    assert_eq!(row, layout.target_audio_rows().start);

    // The existing RNG is independently qualified. Here only draw order and
    // tensor boundaries are checked, with direct indexing instead of packing.
    let mut image_noise = vec![0; image.bytes().len()];
    let mut clip_noise = vec![0; clip.bytes().len()];
    let mut target_noise = vec![0; target.latent_shape().byte_len::<f32>().unwrap()];
    let mut audio_noise = vec![0; (layout.target_audio_rows().end - row) * 32 * 4];
    fill_noise_sequence(
        314,
        &mut [
            &mut image_noise,
            &mut clip_noise,
            &mut target_noise,
            &mut audio_noise,
        ],
        &mut [0; SCRATCH_BYTES],
    )
    .unwrap();
    let mut offset = 0;
    for (frames, height, width, base, noise) in
        [(1, 2, 4, 1.0_f32, image_noise), (7, 4, 2, 10.0, clip_noise)]
    {
        for t in 0..frames {
            for py in 0..height / 2 {
                for px in 0..width / 2 {
                    for c in 0..24 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let dense =
                                    ((c * frames + t) * height + py * 2 + dy) * width + px * 2 + dx;
                                let packed = ((t * (height / 2) + py) * (width / 2) + px) * 96
                                    + c * 4
                                    + dy * 2
                                    + dx;
                                let expected = 0.999_f32 * (base + dense as f32 / 128.0)
                                    + (1.0_f32 - 0.999_f32)
                                        * read(&noise[dense * 4..dense * 4 + 4]);
                                assert_eq!(
                                    read(
                                        &video.bytes().bytes()
                                            [offset + packed * 4..offset + packed * 4 + 4]
                                    )
                                    .to_bits(),
                                    expected.to_bits()
                                );
                            }
                        }
                    }
                }
            }
        }
        offset += frames * height * width * 24 * 4;
    }
    assert_eq!(offset, layout.target_video_rows().start * 96 * 4);
    assert_eq!(&audio.bytes().bytes()[row * 32 * 4..], audio_noise);
    for t in 0..2 {
        for py in 0..2 {
            for px in 0..2 {
                for c in 0..24 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let dense = ((c * 2 + t) * 4 + py * 2 + dy) * 4 + px * 2 + dx;
                            let packed = ((t * 2 + py) * 2 + px) * 96 + c * 4 + dy * 2 + dx;
                            assert_eq!(
                                &video.bytes().bytes()
                                    [offset + packed * 4..offset + packed * 4 + 4],
                                &target_noise[dense * 4..dense * 4 + 4]
                            );
                        }
                    }
                }
            }
        }
    }
    drop((video, audio));
    assert_eq!(host.usage().unwrap().allocated_bytes(), before);
    let wrong_order = [
        geometry[1].clone(),
        geometry[0].clone(),
        geometry[2].clone(),
    ];
    let wrong = FastH3PackedLayout::ref2va(3, &wrong_order, &target, 128).unwrap();
    assert!(
        prepare_fast_h3_reference_media(314, &target, &wrong, &references, media_ids(), &host)
            .is_err()
    );
    assert_eq!(host.usage().unwrap().allocated_bytes(), before);
    drop((image, clip, sound, soundtrack));
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn reference_media_rejects_foreign_audio_wrong_kind_and_capacity_without_retaining_grants() {
    let host = media_host(4096);
    let foreign = media_host(4096);
    let image = latent(&host, vec![1, 24, 1, 2, 2], 1.0);
    let sound = latent(&foreign, vec![2, 32, 1], 2.0);
    let target = FastH3VideoGeometry::new(64, 64, 5).unwrap();
    let geometry = [
        FastH3Reference::Image {
            latent_height: 2,
            latent_width: 2,
        },
        FastH3Reference::Audio {
            latents_per_channel: 1,
        },
    ];
    let layout = FastH3PackedLayout::ref2va(3, &geometry, &target, 128).unwrap();
    let before = host.usage().unwrap().allocated_bytes();
    assert!(prepare_fast_h3_reference_media(
        1,
        &target,
        &layout,
        &[
            FastH3ReferenceLatents::Image(&image),
            FastH3ReferenceLatents::Audio(&sound)
        ],
        media_ids(),
        &host
    )
    .is_err());
    let image_layout = FastH3PackedLayout::ref2va(3, &geometry[..1], &target, 128).unwrap();
    // Complete input is valid but outputs+scratch cannot fit this real account.
    assert!(prepare_fast_h3_reference_media(
        1,
        &target,
        &image_layout,
        &[FastH3ReferenceLatents::Image(&image)],
        media_ids(),
        &host
    )
    .is_err());
    assert_eq!(host.usage().unwrap().allocated_bytes(), before);
    assert_eq!(host.usage().unwrap().reserved_bytes(), 0);
    let large = media_host(1 << 20);
    let clip = latent(&large, vec![1, 24, 7, 2, 2], 1.0);
    assert!(prepare_fast_h3_reference_media(
        1,
        &target,
        &image_layout,
        &[FastH3ReferenceLatents::Image(&clip)],
        media_ids(),
        &large
    )
    .is_err());
    drop((image, sound, clip));
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(foreign.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(large.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn seeded_keyframe_inputs_pack_real_coordinates_and_freeze_augmented_prefix() {
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                output_bytes: 1 << 20,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let mut owner = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 24 * 4 * 4 * 4)
        .unwrap();
    for (i, dst) in owner.as_mut().chunks_exact_mut(4).enumerate() {
        dst.copy_from_slice(&((i as f32 - 100.0) / 32.0).to_le_bytes());
    }
    let reference = ManagedHostTensor::new(
        ProgramValueId::new("value.test.normalized_keyframe").unwrap(),
        ResolvedTensorSpec::new(
            vec![1, 24, 1, 4, 4],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        owner.freeze(),
    )
    .unwrap();
    let target = FastH3VideoGeometry::new(64, 64, 5).unwrap();
    let layout = FastH3PackedLayout::ref2va(
        3,
        &[FastH3Reference::Image {
            latent_height: 4,
            latent_width: 4,
        }],
        &target,
        64,
    )
    .unwrap();
    let [video, audio] = prepare_fast_h3_keyframe_media(
        314,
        &target,
        &layout,
        &reference,
        [
            ProgramValueId::new("value.test.video").unwrap(),
            ProgramValueId::new("value.test.audio").unwrap(),
        ],
        &host,
    )
    .unwrap();
    // Independent indexing: row=(patch_y,patch_x), column=(channel,dy,dx).
    // RNG values use the existing shared implementation; this verifies sequence
    // use and packing, not a new independent RNG-parity assertion.
    let mut ref_noise = vec![0; 24 * 4 * 4 * 4];
    let mut video_noise = vec![0; target.latent_shape().byte_len::<f32>().unwrap()];
    let mut audio_noise = vec![0; audio.bytes().len()];
    fill_noise_sequence(
        314,
        &mut [&mut ref_noise, &mut video_noise, &mut audio_noise],
        &mut [0; SCRATCH_BYTES],
    )
    .unwrap();
    for py in 0..2 {
        for px in 0..2 {
            for c in 0..24 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let dense = c * 16 + (py * 2 + dy) * 4 + px * 2 + dx;
                        let packed = (py * 2 + px) * 96 + c * 4 + dy * 2 + dx;
                        let expected = 0.999_f32
                            * read(&reference.bytes().bytes()[dense * 4..dense * 4 + 4])
                            + (1.0_f32 - 0.999_f32) * read(&ref_noise[dense * 4..dense * 4 + 4]);
                        assert_eq!(
                            read(&video.bytes().bytes()[packed * 4..packed * 4 + 4]).to_bits(),
                            expected.to_bits()
                        );
                    }
                }
            }
        }
    }
    let prefix_bytes = layout.target_video_rows().start * 96 * 4;
    for t in 0..2 {
        for py in 0..2 {
            for px in 0..2 {
                for c in 0..24 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let dense = ((c * 2 + t) * 4 + py * 2 + dy) * 4 + px * 2 + dx;
                            let packed = ((t * 2 + py) * 2 + px) * 96 + c * 4 + dy * 2 + dx;
                            assert_eq!(
                                &video.bytes().bytes()
                                    [prefix_bytes + packed * 4..prefix_bytes + packed * 4 + 4],
                                &video_noise[dense * 4..dense * 4 + 4]
                            );
                        }
                    }
                }
            }
        }
    }
    assert_eq!(audio.bytes().bytes(), audio_noise);
    let plan = Arc::new(
        FastH3PddPlan::from_schedule(
            FastH3PddSchedule::new(4, vec![0, 2, 4], 1., 1., 4).unwrap(),
            96,
            32,
            4,
        )
        .unwrap(),
    );
    let mut continuation = FastH3ManagedContinuation::initialize(
        RequestIdentity::new("request.test.seeded").unwrap(),
        plan,
        &layout,
        host.clone(),
        &video,
        &audio,
    )
    .unwrap();
    assert_eq!(
        &continuation.video().bytes().bytes()[..prefix_bytes],
        &video.bytes().bytes()[..prefix_bytes]
    );
    let velocities = [
        ResolvedTensorSpec::new(
            vec![layout.target_video_rows().end as u64, 96],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        ResolvedTensorSpec::new(
            vec![layout.target_audio_rows().end as u64, 32],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
    ];
    let output_video = host
        .allocate_zeroed(
            HostMemoryClass::Output,
            velocities[0].minimum_storage_bytes().unwrap() as usize,
        )
        .unwrap()
        .freeze();
    let output_audio = host
        .allocate_zeroed(
            HostMemoryClass::Output,
            velocities[1].minimum_storage_bytes().unwrap() as usize,
        )
        .unwrap()
        .freeze();
    let request = continuation.request().clone();
    continuation
        .prepare_advance(FastH3PredictionView::new(
            &request,
            0,
            FastH3VelocityInput::new(&velocities[0], &output_video),
            FastH3VelocityInput::new(&velocities[1], &output_audio),
        ))
        .unwrap()
        .commit();
    assert_eq!(
        &continuation.video().bytes().bytes()[..prefix_bytes],
        &video.bytes().bytes()[..prefix_bytes]
    );
    drop((output_video, output_audio));
    // Failed preparation must not consume a shared RNG state or retain grants.
    let before = host.usage().unwrap().allocated_bytes();
    let wrong = FastH3PackedLayout::ref2va(
        3,
        &[FastH3Reference::Image {
            latent_height: 2,
            latent_width: 2,
        }],
        &target,
        64,
    )
    .unwrap();
    assert!(prepare_fast_h3_keyframe_media(
        314,
        &target,
        &wrong,
        &reference,
        [
            ProgramValueId::new("value.test.v").unwrap(),
            ProgramValueId::new("value.test.a").unwrap()
        ],
        &host
    )
    .is_err());
    assert_eq!(host.usage().unwrap().allocated_bytes(), before);
    drop((continuation, video, audio, reference));
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}
