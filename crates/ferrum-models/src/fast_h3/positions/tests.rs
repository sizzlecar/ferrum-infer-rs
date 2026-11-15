use super::*;
use crate::fast_h3::{FastH3Checkpoint, FastH3MetadataLimits, FastH3SegmentKind};
use ferrum_runtime::{HostMemoryAccount, HostMemoryClassLimits, HostMemoryLimits, ProgramValueId};
use half::bf16;

fn plan() -> FastH3PositionPlan {
    FastH3PositionPlan::ref2va(
        3,
        &[
            FastH3Reference::Image {
                latent_height: 4,
                latent_width: 4,
            },
            FastH3Reference::Audio {
                latents_per_channel: 2,
            },
            FastH3Reference::Video {
                latent_frames: 3,
                latent_height: 4,
                latent_width: 4,
                audio_latents_per_channel: Some(2),
            },
        ],
        &FastH3VideoGeometry::new(32, 32, 5).unwrap(),
        FastH3PositionLimits {
            max_sequence_rows: 45,
            max_regions: 7,
        },
    )
    .unwrap()
}

#[test]
fn mixed_references_share_clocks_and_use_the_correct_audio_width() {
    let plan = plan();
    let positions: Vec<_> = plan.positions().collect();
    assert_eq!(positions.len(), plan.layout().sequence_rows());
    assert_eq!(positions[..3], [[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]);
    assert_eq!(
        positions[3..7],
        [[3., 0., 0.], [3., 0., 16.], [3., 16., 0.], [3., 16., 16.]]
    );
    // Standalone audio uses the target's single width position; video audio
    // uses the reference's two width positions. Both are channel-major.
    assert_eq!(
        positions[7..11],
        [[4., 0., 0.], [5., 0., 0.], [4., 0., 0.], [5., 0., 0.]]
    );
    assert_eq!(
        positions[11..15],
        [[6., 0., 0.], [7., 0., 0.], [6., 0., 16.], [7., 0., 16.]]
    );
    for (frame, expected_time) in [6., 6. + 5. / 3., 6. + 25. / 3.].into_iter().enumerate() {
        for (cell, [h, w]) in [[0., 0.], [0., 16.], [16., 0.], [16., 16.]]
            .into_iter()
            .enumerate()
        {
            let [time, actual_h, actual_w] = positions[15 + frame * 4 + cell];
            assert!((time - expected_time).abs() < 2e-14);
            assert_eq!([actual_h, actual_w], [h, w]);
        }
    }
    // Video span is 15, so video wins max(audio span 2, video span 15).
    for channel in 0..2 {
        for time in 0..8 {
            assert_eq!(
                positions[27 + channel * 8 + time],
                [21. + time as f64, 0., 0.]
            );
        }
    }
    assert_eq!(positions[43], [21., 0., 0.]);
    assert!((positions[44][0] - (21. + 5. / 3.)).abs() < 2e-14);
    assert_eq!(
        plan.layout().segments()[3].kind,
        FastH3SegmentKind::ReferenceAudio
    );
    let mut iter = plan.positions();
    assert_eq!(iter.len(), 45);
    assert_eq!(iter.nth(44), Some(positions[44]));
    assert_eq!(iter.len(), 0);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);
}

#[test]
fn video_clock_uses_audio_when_longer_and_rectangular_grids_stay_centered() {
    let plan = FastH3PositionPlan::ref2va(
        1,
        &[FastH3Reference::Video {
            latent_frames: 1,
            latent_height: 2,
            latent_width: 8,
            audio_latents_per_channel: Some(4),
        }],
        &FastH3VideoGeometry::new(32, 32, 5).unwrap(),
        FastH3PositionLimits {
            max_sequence_rows: 64,
            max_regions: 5,
        },
    )
    .unwrap();
    let positions: Vec<_> = plan.positions().collect();
    // sqrt(2*8)=4: h=8 and w=-16,0,16,32 (endpoint excluded).
    assert_eq!(positions[1], [1., 0., -16.]);
    assert_eq!(positions[5], [1., 0., 32.]);
    assert_eq!(
        positions[9..13],
        [[1., 8., -16.], [1., 8., 0.], [1., 8., 16.], [1., 8., 32.]]
    );
    assert_eq!(positions[13], [5., 0., 0.]);
}

#[test]
fn temporal_grid_repeats_the_vae_frame_pattern_and_advances_silent_video() {
    let plan = FastH3PositionPlan::ref2va(
        3,
        &[FastH3Reference::Video {
            latent_frames: 37,
            latent_height: 2,
            latent_width: 2,
            audio_latents_per_channel: None,
        }],
        &FastH3VideoGeometry::new(32, 32, 5).unwrap(),
        FastH3PositionLimits {
            max_sequence_rows: 58,
            max_regions: 4,
        },
    )
    .unwrap();
    let positions: Vec<_> = plan.positions().collect();
    for frame in 0..37 {
        // Independent integer frame counts for full five-latent VAE cycles
        // and their within-cycle prefixes; production sums F64 span values.
        let original_frames = frame / 5 * 17 + [0, 1, 5, 9, 13][frame % 5];
        let expected = 3. + (original_frames * 5) as f64 / 3.;
        assert!((positions[3 + frame][0] - expected).abs() < 5e-13);
        assert_eq!(positions[3 + frame][1..], [0., 0.]);
    }
    assert!((positions[40][0] - (3. + 620. / 3.)).abs() < 5e-13);
    assert_eq!(positions[40], positions[56]);
}

fn config() -> FastH3Checkpoint {
    FastH3Checkpoint::parse(
        include_bytes!("../fixtures/inference.json"),
        include_bytes!("../fixtures/transformer.json"),
        include_bytes!("../fixtures/video_scheduler.json"),
        include_bytes!("../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 256,
        },
    )
    .unwrap()
}
fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn id(s: &str) -> ProgramValueId {
    ProgramValueId::new(s).unwrap()
}

#[test]
fn managed_rotary_matches_independent_trig_and_releases_the_original_account() {
    let plan = plan();
    let checkpoint = config();
    let config = checkpoint.transformer();
    let bytes = plan.layout().sequence_rows() * config.rotary_dim() * 4;
    let account = host(bytes);
    let (cos, sin) = plan
        .managed_bf16_frequencies(config, &account, id("value.cos"), id("value.sin"))
        .unwrap();
    assert_eq!(cos.tensor().dimensions(), [45, 96]);
    assert_eq!(account.usage().unwrap().committed_bytes(), bytes);
    let read = |tensor: &ferrum_runtime::ManagedHostTensor, index: usize| {
        u16::from_le_bytes(
            tensor.bytes().bytes()[2 * index..2 * index + 2]
                .try_into()
                .unwrap(),
        )
    };
    for (row, position) in plan.positions().enumerate() {
        for (axis, coordinate) in position.into_iter().enumerate() {
            for f in 0..16 {
                // Independent F64 analytic inverse/trig. Allow one BF16 ULP
                // for the declared production F32 inverse/angle/trig steps;
                // this is not a Torch GPU equivalence oracle.
                let angle = coordinate * libm::pow(10000., -(f as f64) / 16.);
                for (tensor, expected) in [(&cos, libm::cos(angle)), (&sin, libm::sin(angle))] {
                    let index = row * 96 + axis * 16 + f;
                    let actual = read(tensor, index);
                    let expected = bf16::from_f32(expected as f32).to_bits();
                    assert!(
                        actual.abs_diff(expected) <= 1,
                        "row={row} axis={axis} f={f}"
                    );
                    assert_eq!(actual, read(tensor, index + 48));
                }
            }
        }
    }
    let [view_cos, view_sin] = plan
        .single_request_bf16_rotary_inputs(config, [cos.clone(), sin.clone()])
        .unwrap();
    for (view, original) in [(&view_cos, &cos), (&view_sin, &sin)] {
        assert_eq!(view.tensor().dimensions(), [1, 45, 96]);
        assert_eq!(view.value_id(), original.value_id());
        assert!(view.bytes().same_region(original.bytes()));
        assert!(account.owns(view.bytes()));
    }
    assert_eq!(account.usage().unwrap().committed_bytes(), bytes);
    drop((cos, sin, view_cos, view_sin));
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    // First allocation fits; failure of the second must release it too.
    let short = host(bytes - 1);
    assert!(plan
        .managed_bf16_frequencies(config, &short, id("value.cos"), id("value.sin"))
        .is_err());
    assert_eq!(short.usage().unwrap().committed_bytes(), 0);
    assert!(plan
        .managed_bf16_frequencies(config, &account, id("value.same"), id("value.same"))
        .is_err());
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn coordinate_metadata_rejects_capacity_and_geometry_before_expansion() {
    let target = FastH3VideoGeometry::new(32, 32, 5).unwrap();
    let image = FastH3Reference::Image {
        latent_height: 2,
        latent_width: 2,
    };
    for limits in [
        FastH3PositionLimits {
            max_sequence_rows: 19,
            max_regions: 4,
        },
        FastH3PositionLimits {
            max_sequence_rows: 20,
            max_regions: 3,
        },
    ] {
        assert!(
            FastH3PositionPlan::ref2va(1, std::slice::from_ref(&image), &target, limits).is_err()
        );
    }
    let limits = FastH3PositionLimits {
        max_sequence_rows: 64,
        max_regions: 5,
    };
    assert!(FastH3PositionPlan::ref2va(1, &[], &target, limits).is_err());
    for reference in [
        FastH3Reference::Image {
            latent_height: 3,
            latent_width: 2,
        },
        FastH3Reference::Image {
            latent_height: usize::MAX - 1,
            latent_width: 4,
        },
        FastH3Reference::Audio {
            latents_per_channel: 0,
        },
        FastH3Reference::Video {
            latent_frames: 0,
            latent_height: 2,
            latent_width: 2,
            audio_latents_per_channel: None,
        },
    ] {
        assert!(FastH3PositionPlan::ref2va(1, &[reference], &target, limits).is_err());
    }
}
