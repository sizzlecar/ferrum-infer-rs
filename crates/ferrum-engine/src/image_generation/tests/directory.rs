use super::*;
use ferrum_models::pipeline_source::PipelineSourceLimits;
use serde_json::{json, Value};
use std::{fs, path::Path};

fn json_file(root: &Path, relative: &str, value: &Value) {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}

fn installation() -> (tempfile::TempDir, [u64; 3]) {
    // Reuse actual tiny schemas and payloads; the directory API itself does
    // not construct this fixture's Reference runtime or any device context.
    let (pipeline, directory) = super::adapter::pipeline(1);
    let expected = pipeline.assets.receipt().required_weight_bytes;
    drop(pipeline);
    let root = directory.path();
    fs::rename(root.join("text"), root.join("text_encoder")).unwrap();
    fs::rename(root.join("dit"), root.join("transformer")).unwrap();
    for component in ["transformer", "vae"] {
        fs::rename(
            root.join(component).join("model.safetensors"),
            root.join(component)
                .join("diffusion_pytorch_model.safetensors"),
        )
        .unwrap();
    }
    json_file(
        root,
        "model_index.json",
        &json!({
            "_class_name":"QwenImage21Pipeline",
            "processor":["transformers","Qwen3VLProcessor"],
            "text_encoder":["transformers","Qwen3VLForConditionalGeneration"],
            "transformer":["diffusers","QwenImage21Transformer2DModel"],
            "vae":["diffusers","AutoencoderKLQwenImage21"],
            "scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"],
            "sample_sigmas":[1.0,0.5]
        }),
    );
    json_file(root, "text_encoder/config.json", &fixture::text_metadata());
    json_file(
        root,
        "transformer/config.json",
        &fixture::transformer_metadata(),
    );
    json_file(
        root,
        "vae/config.json",
        &json!({
            "_class_name":"AutoencoderKLQwenImage21","is_residual":true,"patch_size":null,
            "dropout":0.0,"out_channels":4,"attn_scales":[],"z_dim":64,"base_dim":1,
            "dim_mult":[1,2,4,8,8],"num_res_blocks":2,"temperal_downsample":[false,true,true,true],
            "latents_mean":vec![0.;64],"latents_std":vec![1.;64]
        }),
    );
    json_file(
        root,
        "scheduler/scheduler_config.json",
        &json!({
            "_class_name":"FlowMatchEulerDiscreteScheduler","_diffusers_version":"fixture",
            "base_image_seq_len":256,"base_shift":0.5,"invert_sigmas":false,"max_image_seq_len":8192,
            "max_shift":0.9,"num_train_timesteps":1000,"shift":1.0,"shift_terminal":null,
            "stochastic_sampling":false,"time_shift_type":"exponential","use_beta_sigmas":false,
            "use_dynamic_shifting":false,"use_exponential_sigmas":false,"use_karras_sigmas":false
        }),
    );
    json_file(
        root,
        "processor/processor_config.json",
        &json!({"processor_class":"Qwen3VLProcessor"}),
    );
    let (tokenizer, config, template) = super::adapter::tokenizer();
    for (name, bytes) in [
        ("tokenizer.json", tokenizer),
        ("tokenizer_config.json", config),
        ("chat_template.jinja", template),
    ] {
        fs::write(root.join("processor").join(name), bytes).unwrap();
    }
    (directory, expected)
}

#[test]
fn complete_root_uses_declared_roles_and_standard_archives() {
    let (directory, expected) = installation();
    let assets =
        QwenImage21Assets::from_directory(directory.path(), PipelineSourceLimits::default())
            .unwrap();
    assert_eq!(assets.receipt().required_weight_bytes, expected);
    assert!(assets
        .receipt()
        .metadata_sha256
        .contains_key("package/processor/processor_config.json"));
    assert_eq!(assets.spatial_scale().unwrap(), 16);
    assert_eq!(assets.schedule.evaluation_count(), 2);
    drop(assets);
    // The declaration selects the architecture even after an arbitrary move.
    let parent = tempfile::tempdir().unwrap();
    let renamed = parent.path().join("unrelated-public-alias");
    fs::rename(directory.path(), &renamed).unwrap();
    assert_eq!(
        QwenImage21Assets::from_directory(&renamed, PipelineSourceLimits::default())
            .unwrap()
            .receipt()
            .required_weight_bytes,
        expected
    );
}

#[test]
fn output_requirements_follow_decoder_batch_geometry_without_payload_allocation() {
    let (pipeline, _directory) = super::adapter::pipeline(1);
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            1,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 1,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let mut limits = ImagePipelineLimits {
        maximum_user_tokens: 1024,
        maximum_width: 2048,
        maximum_height: 2048,
        maximum_batch_size: STRUCTURED_ATTENTION_BF16_MAX_BATCH,
        decoder_batch_policy: ImageDecoderBatchPolicy::SingleImage,
        edit_prefix_policy: ImageEditPrefixPolicy::Recompute,
        plan_cache: ImagePlanCachePolicy::LastPerStage,
        residency: ImageResidencyPolicy::Cold,
        dynamic_residency: ImageDynamicResidencyPolicy::Release,
        static_initialization: StaticInitializationPolicy::new(256, 8).unwrap(),
        executor: TensorExecutorConfig::default(),
    };
    let batch = pipeline.assets.output_requirements(&limits, &host).unwrap();
    let decoder = pipeline.assets.decoder_batch(1, 128, 128).unwrap();
    let frame = decoder
        .output_tensor()
        .unwrap()
        .minimum_storage_bytes()
        .unwrap() as usize;
    assert_eq!(batch.raw_batch_bytes, frame * limits.maximum_batch_size);
    assert!(batch.execution_output_bytes >= batch.raw_batch_bytes);
    limits.maximum_batch_size = 1;
    let single = pipeline.assets.output_requirements(&limits, &host).unwrap();
    assert_eq!(single.raw_batch_bytes, frame);
    assert_eq!(
        single.execution_output_bytes,
        batch.execution_output_bytes / batch.batch_size
    );
    limits.maximum_width = 2047;
    assert!(pipeline.assets.output_requirements(&limits, &host).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn absent_or_invalid_metadata_fails_before_opening_weight_payloads() {
    let (directory, _) = installation();
    let transformer = directory.path().join("transformer");
    fs::remove_file(transformer.join("config.json")).unwrap();
    fs::write(
        transformer.join("diffusion_pytorch_model.safetensors"),
        b"not an archive",
    )
    .unwrap();
    let error =
        QwenImage21Assets::from_directory(directory.path(), PipelineSourceLimits::default())
            .err()
            .unwrap()
            .to_string();
    assert!(error.contains("transformer/config.json"), "{error}");
    assert!(!transformer.join("config.json").exists());
    json_file(
        directory.path(),
        "transformer/config.json",
        &json!({"_class_name":"UnknownTransformer"}),
    );
    let error =
        QwenImage21Assets::from_directory(directory.path(), PipelineSourceLimits::default())
            .err()
            .unwrap()
            .to_string();
    assert!(error.contains("_class_name"), "{error}");
}

#[cfg(unix)]
#[test]
fn automatic_root_keeps_the_metadata_symlink_boundary() {
    let (directory, _) = installation();
    let outside = tempfile::tempdir().unwrap();
    let target = outside.path().join("config.json");
    fs::rename(directory.path().join("transformer/config.json"), &target).unwrap();
    std::os::unix::fs::symlink(&target, directory.path().join("transformer/config.json")).unwrap();
    let error =
        QwenImage21Assets::from_directory(directory.path(), PipelineSourceLimits::default())
            .err()
            .unwrap()
            .to_string();
    assert!(error.contains("outside its source root"), "{error}");
}
