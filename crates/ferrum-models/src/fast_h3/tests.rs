use super::*;

// Parsing fixtures are the four metadata owners from FastH3 NVFP4 revision
// f01e6a23ac6066aa9eb622cf94b6259442e70ec2. They contain no weights.
const INFERENCE: &[u8] = include_bytes!("fixtures/inference.json");
const TRANSFORMER: &[u8] = include_bytes!("fixtures/transformer.json");
const VIDEO_SCHEDULER: &[u8] = include_bytes!("fixtures/video_scheduler.json");
const AUDIO_SCHEDULER: &[u8] = include_bytes!("fixtures/audio_scheduler.json");

fn limits() -> FastH3MetadataLimits {
    FastH3MetadataLimits {
        max_file_bytes: 4096,
        max_pdd_steps: 256,
    }
}

fn checkpoint() -> FastH3Checkpoint {
    FastH3Checkpoint::parse(
        INFERENCE,
        TRANSFORMER,
        VIDEO_SCHEDULER,
        AUDIO_SCHEDULER,
        limits(),
    )
    .unwrap()
}

#[test]
fn checkpoint_owners_define_widened_heads_and_modality_clocks() {
    let checkpoint = checkpoint();
    let model = checkpoint.transformer();
    assert_eq!(model.video_head_width(), 96);
    assert_eq!(model.audio_head_width(), 32);
    assert_eq!(model.video_projection_rows(), 3072);
    assert_eq!(model.audio_projection_rows(), 1024);
    assert_eq!(
        (checkpoint.video_shift(), checkpoint.audio_shift()),
        (12.0, 3.0)
    );
    assert_eq!(checkpoint.evaluation_count(), 8);
}

#[test]
fn metadata_rejects_duplicate_unknown_and_inconsistent_contracts() {
    let duplicate =
        String::from_utf8(INFERENCE.to_vec())
            .unwrap()
            .replacen('{', "{\"pdd_steps\":32,", 1);
    assert!(FastH3Checkpoint::parse(
        duplicate.as_bytes(),
        TRANSFORMER,
        VIDEO_SCHEDULER,
        AUDIO_SCHEDULER,
        limits()
    )
    .is_err());
    for (field, value) in [
        ("unknown_sampler", serde_json::json!(true)),
        ("pdd_steps", serde_json::json!(16)),
        ("num_inference_steps", serde_json::json!(9)),
        ("pdd_step_indices", serde_json::json!([0, 8, 8, 32])),
        ("video_scheduler_shift", serde_json::json!(3.0)),
        ("guidance_scale", serde_json::json!(6.0)),
        ("grid_max_t", serde_json::json!(1.0)),
        ("vsa_ref_keep_rate", serde_json::json!(1.0)),
        ("pdd_steps", serde_json::json!(true)),
    ] {
        let mut inference: serde_json::Value = serde_json::from_slice(INFERENCE).unwrap();
        inference[field] = value;
        let bytes = serde_json::to_vec(&inference).unwrap();
        assert!(
            FastH3Checkpoint::parse(
                &bytes,
                TRANSFORMER,
                VIDEO_SCHEDULER,
                AUDIO_SCHEDULER,
                limits()
            )
            .is_err(),
            "{field}"
        );
    }
    let mut bounded = limits();
    bounded.max_pdd_steps = 16;
    assert!(FastH3Checkpoint::parse(
        INFERENCE,
        TRANSFORMER,
        VIDEO_SCHEDULER,
        AUDIO_SCHEDULER,
        bounded
    )
    .is_err());
    bounded = limits();
    bounded.max_file_bytes = 32;
    assert!(FastH3Checkpoint::parse(
        INFERENCE,
        TRANSFORMER,
        VIDEO_SCHEDULER,
        AUDIO_SCHEDULER,
        bounded
    )
    .is_err());
}

#[test]
fn pdd_coefficients_cover_the_fine_grid_and_keep_audio_separate() {
    let plan = FastH3PddPlan::new(&checkpoint()).unwrap();
    assert_eq!(plan.initial_video_noise_scale(), 0.999_f32);
    assert_eq!(plan.initial_audio_noise_scale(), 0.999_f32);
    assert_eq!(plan.steps()[0].heads, 0..4);
    assert_eq!(
        plan.steps()[0].video.next_sigma(),
        (0.999_f64 * 84.0 / 85.0) as f32
    );
    assert_eq!(
        plan.steps()[0].audio.next_sigma(),
        (0.999_f64 * 21.0 / 22.0) as f32
    );
    assert_eq!(plan.steps().last().unwrap().video.next_sigma(), 0.0);
    for audio in [false, true] {
        let mut total = 0.0;
        for step in plan.steps() {
            let modality = if audio { &step.audio } else { &step.video };
            let normalized: f32 = modality.normalized_head_weights().iter().sum();
            assert!((normalized - 1.0).abs() < 2e-7);
            total += modality.integration_weights().iter().sum::<f64>();
            assert_eq!(modality.timestep(), 1.0_f32 - modality.sigma());
        }
        // Integrating a constant fine-head prediction spans the full .999
        // noise interval, regardless of partition or modality shift.
        assert!((total + FAST_H3_GRID_MAX_T).abs() < 1e-14);
    }
    assert_ne!(
        plan.steps()[0].video.normalized_head_weights(),
        plan.steps()[0].audio.normalized_head_weights()
    );
}

#[test]
fn video_geometry_exposes_padding_and_checks_capacity_arithmetic() {
    let geometry = FastH3VideoGeometry::new(256, 256, 120).unwrap();
    assert_eq!(
        (
            geometry.requested_frames(),
            geometry.aligned_frames(),
            geometry.latent_frames()
        ),
        (120, 124, 37)
    );
    assert_eq!(geometry.audio_latents_per_channel(), 207);
    assert_eq!(geometry.video_rows(), 37 * 8 * 8);
    let longer = FastH3VideoGeometry::new(256, 256, 240).unwrap();
    assert_eq!(
        (
            longer.aligned_frames(),
            longer.latent_frames(),
            longer.audio_latents_per_channel()
        ),
        (243, 72, 405)
    );
    assert!(FastH3VideoGeometry::new(255, 256, 120).is_err());
    assert!(FastH3VideoGeometry::new(256, 256, 0).is_err());
    assert!(FastH3VideoGeometry::new(256, 256, usize::MAX).is_err());
    assert!(FastH3VideoGeometry::new(usize::MAX - 31, usize::MAX - 31, 120).is_err());
}

#[test]
fn ordered_references_remain_outside_target_updates() {
    let geometry = FastH3VideoGeometry::new(64, 64, 5).unwrap();
    let references = [
        FastH3Reference::Image {
            latent_height: 4,
            latent_width: 4,
        },
        FastH3Reference::Video {
            latent_frames: 2,
            latent_height: 4,
            latent_width: 4,
            audio_latents_per_channel: Some(3),
        },
        FastH3Reference::Audio {
            latents_per_channel: 2,
        },
    ];
    let layout = FastH3PackedLayout::ref2va(3, &references, &geometry, 100).unwrap();
    let actual: Vec<_> = layout
        .segments()
        .iter()
        .map(|segment| (segment.kind, segment.packed_rows.clone()))
        .collect();
    use FastH3SegmentKind::*;
    assert_eq!(
        actual,
        vec![
            (Text, 0..3),
            (ReferenceImage, 3..7),
            (ReferenceAudio, 7..13),
            (ReferenceVideo, 13..21),
            (ReferenceAudio, 21..25),
            (TargetAudio, 25..41),
            (TargetVideo, 41..49)
        ]
    );
    assert_eq!(layout.target_video_rows(), 12..20);
    assert_eq!(layout.target_audio_rows(), 10..26);
    assert_eq!(layout.sequence_rows(), 49);
    assert!(FastH3PackedLayout::ref2va(3, &references, &geometry, 48).is_err());
    assert!(FastH3PackedLayout::ref2va(3, &[], &geometry, 100).is_err());
    assert!(FastH3PackedLayout::ref2va(
        3,
        &[FastH3Reference::Audio {
            latents_per_channel: usize::MAX
        }],
        &geometry,
        usize::MAX
    )
    .is_err());
}
