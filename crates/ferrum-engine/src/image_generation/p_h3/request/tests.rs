use super::*;
use crate::image_generation::*;
use ferrum_models::p_h3_image::{PH3ImageSamplingLimits, PH3ImageSamplingSchedule};
use ferrum_runtime::StaticInitializationPolicy;

fn transformer() -> PH3ImageTransformerConfig {
    PH3ImageTransformerConfig::from_hf_json(
        &serde_json::to_vec(&serde_json::json!({
            "_class_name":"MiniMaxH3Transformer3DModel", "attention_head_dim":8,
            "audio_in_channels":4,"ffn_dim":16,"final_norm_eps":1e-5,"freq_dim":8,
            "hidden_size":8,"in_channels":4,"norm_eps":1e-5,"num_attention_heads":1,
            "num_layers":1,"num_refiner_layers":2,"patch_size":[1,2,2],"qk_norm_eps":1e-5,
            "rope_freq_dim":1,"rope_theta":10000.0,"text_dim":8,"time_embed_dim":4,
            "time_embed_hidden_dim":8
        }))
        .unwrap(),
    )
    .unwrap()
}
fn config() -> ImagePipelineConfig {
    ImagePipelineConfig {
        device_capacity_bytes: 1 << 30,
        maximum_user_tokens: 512,
        maximum_width: 4096,
        maximum_height: 4096,
        maximum_batch_size: 2,
        weight_arena_target_bytes: None,
        decoder_batch_policy: Default::default(),
        edit_prefix_policy: Default::default(),
        plan_cache: Default::default(),
        residency: Default::default(),
        dynamic_residency: Default::default(),
        static_initialization: StaticInitializationPolicy::new(256, 8).unwrap(),
        executor: Default::default(),
    }
}
fn request(width: u32, height: u32) -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "a blue cup".into(),
        width,
        height,
        seed: 42,
        n: 1,
    }
}
fn member(name: &str, width: u32, height: u32, reference: bool) -> RequestGeometry {
    RequestGeometry::new(
        &request(width, height),
        RequestIdentity::new(name).unwrap(),
        &config(),
        16,
        &transformer(),
        vec![PH3ImageModality::Text; 3],
        reference.then(|| PH3ImageReferenceImage::new(2, 2).unwrap()),
    )
    .unwrap()
}
fn schedule() -> PH3ImageSamplingSchedule {
    let index = serde_json::to_vec(&serde_json::json!({
        "_class_name":"PH3ImageModularPipeline","_blocks_class_name":"PH3ImageBlocks",
        "scheduler":["diffusers","MiniMaxH3Scheduler",{}],
        "audio_scheduler":["diffusers","MiniMaxH3Scheduler",{}],
        "sample_sigmas":[1.0,0.75,0.25,0.0]
    }))
    .unwrap();
    let scheduler =
        br#"{"_class_name":"MiniMaxH3Scheduler","shift":1.0,"shift_provided_sigmas":false}"#;
    PH3ImageSamplingSchedule::parse(
        &index,
        scheduler,
        scheduler,
        PH3ImageSamplingLimits {
            max_metadata_bytes: 4096,
            max_evaluations: 32,
        },
    )
    .unwrap()
}

#[test]
fn mixed_progress_and_removal_keep_request_local_time_and_rotary_coordinates() {
    let transformer = transformer();
    let a = member("request.a", 64, 32, true);
    let b = member("request.b", 32, 64, false);
    let batch = BatchGeometry::new(&transformer, vec![&a, &b], 2, 4096).unwrap();
    let schedule = schedule();
    let steps = schedule.steps();
    let times = batch
        .times(&[(&a.identity, steps[2]), (&b.identity, steps[0])], 0.999)
        .unwrap();
    assert_eq!(times, [0.75, 0.75, 0.999, 0.0, 0.0]);
    assert_eq!(batch.indices[0], [0, 0, 0, 2, 1, 1, 1, 1, 0, 0]);
    assert_eq!(batch.indices[1], [3, 3, 3, 4, 4, 4, 4, 3, 3]);
    assert_eq!(batch.positions[0], [0.0, 0.0, 0.0]);
    assert_eq!(batch.positions[10], [0.0, 0.0, 0.0]);
    let remaining = BatchGeometry::new(&transformer, vec![&b], 2, 4096).unwrap();
    assert_eq!(remaining.positions, batch.positions[10..]);
    assert_eq!(
        remaining.times(&[(&b.identity, steps[0])], 0.999).unwrap(),
        [0.0, 0.0]
    );
    assert_eq!(remaining.indices[0], [0, 0, 0, 1, 1, 1, 1, 0, 0]);
    assert!(batch
        .times(&[(&b.identity, steps[0]), (&a.identity, steps[2])], 0.999)
        .is_err());
    assert!(batch.times(&[(&a.identity, steps[2])], 0.999).is_err());
}

#[test]
fn reference_time_rounds_python_scalar_only_at_curve_input_boundary() {
    let transformer = transformer();
    let a = member("request.edit", 32, 32, true);
    let b = member("request.generate", 32, 32, false);
    let batch = BatchGeometry::new(&transformer, vec![&a, &b], 2, 4096).unwrap();
    let schedule = schedule();
    let step = schedule.steps()[2];
    let steps = [(&a.identity, step), (&b.identity, step)];
    let policy = super::super::edit_noise::PH3ImageEditNoisePolicy::default();
    assert_eq!(policy.augmentation().to_bits(), 0x3fef_f7ce_d916_872b);
    for (augmentation, expected) in [
        (policy.augmentation(), 0x3f7f_be77),
        (0.0, 0x3f40_0000),
        (f64::from_bits(1.0_f64.to_bits() - 1), 0x3f80_0000),
    ] {
        let times = batch.times(&steps, augmentation).unwrap();
        assert_eq!(times[2].to_bits(), expected);
        assert_eq!([times[0], times[1], times[3], times[4]], [0.75; 4]);
    }
    for invalid in [
        f64::NAN,
        f64::INFINITY,
        -f64::from_bits(1),
        f64::from_bits(1.0_f64.to_bits() + 1),
    ] {
        assert!(batch.times(&steps, invalid).is_err());
    }
}

#[test]
fn geometry_rejects_incomplete_patches_and_actual_metadata_capacity_before_allocation() {
    let transformer = transformer();
    let a = member("request.a", 2048, 2048, false);
    let b = member("request.b", 4096, 2048, false);
    assert_eq!(a.image_sample.dimensions(), [1, 4, 1, 128, 128]);
    assert_eq!(b.image_sample.dimensions(), [1, 4, 1, 128, 256]);
    assert_eq!(a.audio_sample.dimensions(), [1, 4, 2, 2]);
    let exact =
        (a.layout(&transformer).unwrap().rows() + b.layout(&transformer).unwrap().rows()) * 32;
    assert!(BatchGeometry::new(&transformer, vec![&a, &b], 2, exact).is_ok());
    assert!(BatchGeometry::new(&transformer, vec![&a, &b], 2, exact - 1).is_err());
    assert!(BatchGeometry::new(&transformer, vec![&a, &a], 2, exact).is_err());
    assert!(BatchGeometry::new(&transformer, vec![&a, &b], 1, exact).is_err());
    for (width, height) in [(0, 32), (48, 32), (33, 32), (4128, 32)] {
        assert!(RequestGeometry::new(
            &request(width, height),
            RequestIdentity::new("request.bad").unwrap(),
            &config(),
            16,
            &transformer,
            vec![PH3ImageModality::Text],
            None
        )
        .is_err());
    }
}
