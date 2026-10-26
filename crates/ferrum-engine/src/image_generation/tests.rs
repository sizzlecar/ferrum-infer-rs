use super::*;

mod adapter;
mod batch_order;
mod decoder;
mod directory;
mod fixture;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
mod ownership;
mod plan_cache;
mod prefix;
mod preflight;
mod residency;

fn config() -> ImagePipelineConfig {
    ImagePipelineConfig {
        device_capacity_bytes: 4096,
        maximum_user_tokens: 512,
        maximum_width: 1024,
        maximum_height: 1024,
        maximum_batch_size: ImagePipelineConfig::DEFAULT_MAXIMUM_BATCH_SIZE,
        decoder_batch_policy: ImageDecoderBatchPolicy::default(),
        edit_prefix_policy: ImageEditPrefixPolicy::default(),
        plan_cache: ImagePlanCachePolicy::default(),
        residency: ImageResidencyPolicy::default(),
        dynamic_residency: ImageDynamicResidencyPolicy::default(),
        static_initialization: StaticInitializationPolicy::new(256, 8).unwrap(),
        executor: TensorExecutorConfig::default(),
    }
}

fn request() -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "a blue teapot".into(),
        width: 512,
        height: 768,
        seed: 42,
        n: 1,
    }
}

#[test]
fn compatible_request_batches_keep_independent_prompts_and_seeds_and_reject_mixed_shapes() {
    let mut config = config();
    config.maximum_batch_size = 2;
    let a = request();
    let mut b = a.clone();
    b.prompt = "a different yellow cup".into();
    b.seed = 42 ^ (1_u64 << 40);
    assert_eq!(
        prepare::batch_dimensions(&[a.clone(), b.clone()], &config, 16).unwrap(),
        (48, 32)
    );
    assert!(prepare::batch_dimensions(&[], &config, 16).is_err());
    b.width = 1024;
    assert!(prepare::batch_dimensions(&[a.clone(), b.clone()], &config, 16).is_err());
    b.width = a.width;
    b.n = 2;
    assert!(prepare::batch_dimensions(&[a.clone(), b], &config, 16).is_err());
    assert!(prepare::batch_dimensions(
        &vec![a; STRUCTURED_ATTENTION_BF16_MAX_BATCH + 1],
        &config,
        16
    )
    .is_err());
}

#[test]
fn request_dimensions_are_exact_and_unsupported_controls_fail_before_device_work() {
    let request = request();
    assert_eq!(request.latent_dimensions(&config(), 16).unwrap(), (48, 32));
    for (width, height, n) in [
        (0, 512, 1),
        (513, 512, 1),
        (16, 512, 1),
        (1056, 512, 1),
        (512, 512, 2),
        (512, 512, 0),
    ] {
        assert!(ImageGenerationRequest {
            width,
            height,
            n,
            ..request.clone()
        }
        .latent_dimensions(&config(), 16)
        .is_err());
    }
    assert!(request.latent_dimensions(&config(), 0).is_err());
    let mut raw = serde_json::to_value(&request).unwrap();
    for name in [
        "negative_prompt",
        "guidance_scale",
        "steps",
        "image",
        "batch",
        "frames",
    ] {
        raw[name] = serde_json::json!(1);
        assert!(
            serde_json::from_value::<ImageGenerationRequest>(raw.clone()).is_err(),
            "{name}"
        );
        raw.as_object_mut().unwrap().remove(name);
    }
    let mut oversized = request;
    oversized.prompt = "x".repeat(1024 * 1024 + 1);
    assert!(oversized.latent_dimensions(&config(), 16).is_err());
}

#[test]
fn limits_are_explicit_and_pixel_bounds_do_not_authorize_invalid_latent_geometry() {
    let mut limits = config();
    for batch in [0, STRUCTURED_ATTENTION_BF16_MAX_BATCH + 1] {
        limits.maximum_batch_size = batch;
        assert!(limits.validate().is_err());
    }
    limits.maximum_batch_size = 1;
    assert!(prepare::batch_dimensions(&[request(), request()], &limits, 16).is_err());
    limits.maximum_batch_size = STRUCTURED_ATTENTION_BF16_MAX_BATCH;
    assert!(
        prepare::batch_dimensions(&vec![request(); limits.maximum_batch_size], &limits, 16).is_ok()
    );
    for tokens in [0, 8193] {
        limits.maximum_user_tokens = tokens;
        assert!(limits.validate().is_err());
    }
    limits.maximum_user_tokens = 512;
    limits.device_capacity_bytes = 0;
    assert!(limits.validate().is_err());
    limits.device_capacity_bytes = 4096;
    limits.maximum_width = u32::MAX;
    limits.maximum_height = u32::MAX;
    let too_wide = ImageGenerationRequest {
        width: 2050 * 16,
        ..request()
    };
    assert!(too_wide.latent_dimensions(&limits, 16).is_err());
    // A different validated decoder scale is respected; no hardcoded /16.
    assert_eq!(request().latent_dimensions(&limits, 8).unwrap(), (96, 64));
}
