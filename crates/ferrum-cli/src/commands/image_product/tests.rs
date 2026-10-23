use super::{configuration::*, *};
use clap::{CommandFactory, Parser, Subcommand};
use serde_json::{json, Value};
use std::{fs, path::PathBuf};

fn configuration() -> Value {
    json!({
        "model":"local-picture", "execution_profile":"experimental_serial_cold",
        "metadata": {"model_index":"model_index.json", "text_encoder":"text/config.json",
            "transformer":"transformer/config.json", "vae":"vae/config.json",
            "scheduler":"scheduler/config.json", "tokenizer":"tokenizer/tokenizer.json",
            "tokenizer_config":"tokenizer/tokenizer_config.json", "chat_template":"tokenizer/chat_template.jinja"},
        "weights": {"text_encoder":{"kind":"index","directory":"text","file":"weights.index.json"},
            "transformer":{"kind":"index","directory":"transformer","file":"diffusion.index.json"},
            "vae":{"kind":"single","directory":"vae","file":"diffusion.safetensors"}},
        "backend":{"kind":"cuda", "ordinal":0, "usable_memory_bytes":1024,
            "bootstrap_memory_bytes":64,"stream_memory_bytes":64,"maximum_streams":"auto",
            "maximum_in_flight_per_stream":1,"maximum_commands_per_submission":1024,
            "opaque_memory_profile":"unqualified-test",
            "cudnn":{"library_path":"lib/libcudnn.so.9","opaque_memory_bytes":64,
                "workspace_limit_bytes":64,"opaque_memory_profile":"unqualified-test"},
            "host":{"total_bytes":4096,"input_preprocess_bytes":2048,
                "transfer_staging_bytes":1024,"output_bytes":2048,"weight_backing_bytes":0},
            "pipeline":{"maximum_user_tokens":128,
                "maximum_width":512,"maximum_height":512,"maximum_staging_bytes":512,
                "maximum_initialization_commands":32,"maximum_backing_maintenance_passes":8}}
    })
}

#[test]
fn stream_capacity_auto_and_explicit_limits_are_shared_by_run_and_serve() {
    use ferrum_engine::image_generation::{ImageStreamLimit, ImageStreamLimitSource};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("image.json");
    let load = |wire: &Value| {
        fs::write(&path, serde_json::to_vec(wire).unwrap()).unwrap();
        ImageProductConfig::load(&path)
    };
    for serving in [false, true] {
        let mut wire = configuration();
        if serving {
            wire["serving"] = json!({"maximum_pending":2,"maximum_prompt_bytes":4096,
                "max_num_seqs":1,"request_batch_max_wait_ms":0,"default_width":512,
                "default_height":512,"default_seed":0,"rgba_scratch_bytes":16384,"output":{}});
        }
        wire["backend"]
            .as_object_mut()
            .unwrap()
            .remove("maximum_streams");
        let ImageBackendConfig::Cuda(omitted) = load(&wire).unwrap().backend;
        assert_eq!(omitted.maximum_streams, ImageStreamLimit::Auto);
        let automatic = omitted.stream_plan().unwrap();
        assert_eq!(automatic.source(), ImageStreamLimitSource::Auto);
        assert_eq!(automatic.required_streams(), 2);
        assert_eq!(automatic.resolved_streams(), 2);
        wire["backend"]["maximum_streams"] = json!("auto");
        let ImageBackendConfig::Cuda(named) = load(&wire).unwrap().backend;
        assert_eq!(named.stream_plan().unwrap(), automatic);
        for limit in [2, 4] {
            wire["backend"]["maximum_streams"] = json!(limit);
            let ImageBackendConfig::Cuda(explicit) = load(&wire).unwrap().backend;
            let plan = explicit.stream_plan().unwrap();
            assert_eq!(plan.source(), ImageStreamLimitSource::Explicit);
            assert_eq!(plan.required_streams(), automatic.required_streams());
            assert_eq!(plan.resolved_streams(), limit);
            #[cfg(feature = "cuda")]
            assert_eq!(explicit.resolved().unwrap().stream_plan().unwrap(), plan);
        }
        for invalid in [
            json!(0),
            json!(1),
            json!(-1),
            json!(1.5),
            json!(null),
            json!("automatic"),
        ] {
            wire["backend"]["maximum_streams"] = invalid;
            assert!(load(&wire).is_err());
        }
    }
}

#[test]
fn fp8_precision_is_shared_by_run_and_serve_without_changing_residency_or_attention() {
    use ferrum_engine::image_generation::{
        ImageCudaAttentionPolicy, ImageCudaPrecisionPolicy, ImageResidencyPolicy,
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("image.json");
    let load = |wire: &Value| {
        fs::write(&path, serde_json::to_vec(wire).unwrap()).unwrap();
        ImageProductConfig::load(&path)
    };
    let mut wire = configuration();
    let ImageBackendConfig::Cuda(original) = load(&wire).unwrap().backend;
    assert_eq!(original.precision_policy, ImageCudaPrecisionPolicy::Bf16);
    wire["backend"]["precision_policy"] = json!("fp8_linear_experimental_v1");
    assert!(load(&wire).is_err());
    wire["backend"]["host"]["weight_backing_bytes"] = json!(1024);
    for serving in [false, true] {
        wire["serving"] = if serving {
            json!({"maximum_pending":2,"maximum_prompt_bytes":4096,"max_num_seqs":1,
                "request_batch_max_wait_ms":0,"default_width":512,"default_height":512,
                "default_seed":0,"rgba_scratch_bytes":16384,"output":{}})
        } else {
            Value::Null
        };
        let ImageBackendConfig::Cuda(config) = load(&wire).unwrap().backend;
        assert_eq!(
            config.precision_policy,
            ImageCudaPrecisionPolicy::Fp8LinearExperimentalV1
        );
        assert_eq!(
            config.attention_policy,
            ImageCudaAttentionPolicy::Sequential
        );
        assert_eq!(config.pipeline.residency, ImageResidencyPolicy::Cold);
    }
    for invalid in [json!("omni"), json!("fp8"), json!(null), json!({})] {
        wire["backend"]["precision_policy"] = invalid;
        assert!(load(&wire).is_err());
    }
}

#[test]
fn fp8_host_cache_is_explicit_budgeted_and_shared_by_run_and_serve() {
    use ferrum_engine::image_generation::{ImageFp8HostCacheBudgets, ImageFp8HostCachePolicy};
    use std::num::NonZeroU64;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("image.json");
    let load = |wire: &Value| {
        fs::write(&path, serde_json::to_vec(wire).unwrap()).unwrap();
        ImageProductConfig::load(&path)
    };
    let mut wire = configuration();
    let ImageBackendConfig::Cuda(defaults) = load(&wire).unwrap().backend;
    assert_eq!(defaults.fp8_host_cache, ImageFp8HostCachePolicy::Off);
    let cache = json!({"stage_budgets":{"maximum_bytes_by_stage":{"text_encoder":1024}}});
    wire["backend"]["fp8_host_cache"] = cache.clone();
    assert!(load(&wire).is_err()); // BF16 has no converted FP8 payload.
    wire["backend"]["precision_policy"] = json!("fp8_linear_experimental_v1");
    wire["backend"]["host"] = json!({"total_bytes":5120,"input_preprocess_bytes":1024,
        "transfer_staging_bytes":1024,"output_bytes":2048,"weight_backing_bytes":1024});
    let expected = ImageFp8HostCachePolicy::StageBudgets {
        maximum_bytes_by_stage: ImageFp8HostCacheBudgets {
            text_encoder: NonZeroU64::new(1024),
            ..ImageFp8HostCacheBudgets::default()
        },
    };
    for serving in [false, true] {
        wire["serving"] = if serving {
            json!({"maximum_pending":2,"maximum_prompt_bytes":4096,"max_num_seqs":1,
                "request_batch_max_wait_ms":0,"default_width":512,"default_height":512,
                "default_seed":0,"rgba_scratch_bytes":16384,"output":{}})
        } else {
            Value::Null
        };
        wire["backend"]["fp8_host_cache"] = cache.clone();
        let ImageBackendConfig::Cuda(config) = load(&wire).unwrap().backend;
        assert_eq!(config.fp8_host_cache, expected);
        #[cfg(feature = "cuda")]
        assert_eq!(config.resolved().unwrap().fp8_host_cache, expected);
        wire["backend"]["host"]["total_bytes"] = json!(5119);
        assert!(load(&wire).is_err());
        wire["backend"]["host"]["total_bytes"] = json!(5120);
        wire["backend"]["host"]["weight_backing_bytes"] = json!(1023);
        assert!(load(&wire).is_err());
        wire["backend"]["host"]["weight_backing_bytes"] = json!(1024);
        for stages in [
            json!({}),
            json!({"text_encoder":0}),
            json!({"decoder":1024}),
            json!({"vision_encoder":1024}),
            json!({"text_encoder":1024,"unknown":1}),
        ] {
            wire["backend"]["fp8_host_cache"] =
                json!({"stage_budgets":{"maximum_bytes_by_stage":stages}});
            assert!(load(&wire).is_err());
        }
    }
}

#[test]
fn image_configuration_is_explicit_strict_and_resolves_from_its_directory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("image.json");
    fs::write(&path, serde_json::to_vec(&configuration()).unwrap()).unwrap();
    let config = ImageProductConfig::load(&path).unwrap();
    let base = fs::canonicalize(dir.path()).unwrap();
    assert_eq!(
        config.metadata.transformer,
        base.join("transformer/config.json")
    );
    let ImageBackendConfig::Cuda(cuda) = config.backend;
    assert_eq!(cuda.pipeline.maximum_batch_size, 1);
    assert_eq!(
        cuda.resolved_pipeline().unwrap().decoder_batch_policy,
        ferrum_engine::image_generation::ImageDecoderBatchPolicy::SingleImage
    );
    assert_eq!(cuda.cudnn.library_path, base.join("lib/libcudnn.so.9"));
    assert!(cuda.cudnn.dependency_libraries.is_empty());
    assert!(
        matches!(config.weights.vae, WeightSource::Single{directory,file}
        if directory == base.join("vae") && file == "diffusion.safetensors")
    );
    for (key, value) in [
        ("maximum_batch_size", json!(8)),
        ("execution_profile", json!("concurrent")),
    ] {
        let mut invalid = configuration();
        invalid[key] = value;
        fs::write(&path, serde_json::to_vec(&invalid).unwrap()).unwrap();
        assert!(ImageProductConfig::load(&path).is_err());
    }
    let mut invalid = configuration();
    invalid["backend"]["kind"] = json!("cpu");
    assert!(serde_json::from_value::<ImageProductConfig>(invalid).is_err());
    let original = serde_json::to_string(&configuration()).unwrap();
    let duplicate = format!("{{\"model\":\"duplicate\",{}", &original[1..]);
    assert!(serde_json::from_str::<ImageProductConfig>(&duplicate).is_err());
    fs::write(&path, b"[]").unwrap();
    assert!(ImageProductConfig::load(&path).is_err());
}

#[test]
fn optional_edit_processor_is_shared_by_run_and_serve_and_resolves_exactly() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("image.json");
    let base = fs::canonicalize(directory.path()).unwrap();
    let load = |wire: &Value| {
        fs::write(&path, serde_json::to_vec(wire).unwrap()).unwrap();
        ImageProductConfig::load(&path)
    };
    let mut wire = configuration();
    assert!(load(&wire).unwrap().metadata.processor.is_none());
    wire["metadata"]["processor"] = Value::Null;
    assert!(load(&wire).unwrap().metadata.processor.is_none());
    for serving in [false, true] {
        wire["serving"] = if serving {
            json!({"maximum_pending":2,"maximum_prompt_bytes":4096,"max_num_seqs":1,
                "request_batch_max_wait_ms":0,"default_width":512,"default_height":512,
                "default_seed":0,"rgba_scratch_bytes":16384,"output":{}})
        } else {
            Value::Null
        };
        wire["metadata"]["processor"] = json!("processor/processor_config.json");
        let config = load(&wire).unwrap();
        assert_eq!(
            config.metadata.processor.as_deref(),
            Some(base.join("processor/processor_config.json").as_path())
        );
        assert_eq!(
            config.execution_profile,
            ImageExecutionProfile::ExperimentalSerialCold
        );
        let absolute = base.join("explicit-processor.json");
        wire["metadata"]["processor"] = json!(absolute);
        assert_eq!(load(&wire).unwrap().metadata.processor, Some(absolute));
    }
    for invalid in [json!(""), json!(true), json!([]), json!({"path":"p.json"})] {
        wire["metadata"]["processor"] = invalid;
        assert!(load(&wire).is_err());
    }
    let mut metadata = configuration()["metadata"].clone();
    metadata["processor_config"] = json!("p.json");
    assert!(serde_json::from_value::<MetadataPaths>(metadata).is_err());
    let original = serde_json::to_string(&configuration()["metadata"]).unwrap();
    let duplicate = format!(
        "{{\"processor\":null,\"processor\":\"p.json\",{}",
        &original[1..]
    );
    assert!(serde_json::from_str::<MetadataPaths>(&duplicate).is_err());
}

#[test]
fn requested_edit_processor_is_bounded_and_cannot_fall_back_before_weights() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("image.json");
    let mut wire = configuration();
    wire["metadata"]["processor"] = json!("processor.json");
    fs::write(&path, serde_json::to_vec(&wire).unwrap()).unwrap();
    fs::write(
        directory.path().join("model_index.json"),
        serde_json::to_vec(&json!({"_class_name":"QwenImage21Pipeline",
            "processor":["transformers","Qwen3VLProcessor"],
            "text_encoder":["transformers","Qwen3VLForConditionalGeneration"],
            "transformer":["diffusers","QwenImage21Transformer2DModel"],
            "vae":["diffusers","AutoencoderKLQwenImage21"],
            "scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"]}))
        .unwrap(),
    )
    .unwrap();
    let config = ImageProductConfig::load(&path).unwrap();
    let error = config.prepare_assets().err().unwrap();
    assert!(error.to_string().contains("processor.json"), "{error:#}");
    let processor = directory.path().join("processor.json");
    let file = fs::File::create(&processor).unwrap();
    file.set_len(4 * 1024 * 1024 + 1).unwrap();
    drop(file);
    let error = config.prepare_assets().err().unwrap();
    assert!(
        error.to_string().contains("exceeds metadata limit"),
        "{error:#}"
    );
    fs::write(processor, []).unwrap();
    let error = config.prepare_assets().err().unwrap();
    assert!(
        error.to_string().contains("empty or oversized"),
        "{error:#}"
    );
    // Omission keeps the old path: the next missing required generation
    // metadata is now observed instead of probing a processor or any weights.
    wire["metadata"]
        .as_object_mut()
        .unwrap()
        .remove("processor");
    fs::write(&path, serde_json::to_vec(&wire).unwrap()).unwrap();
    let error = ImageProductConfig::load(&path)
        .unwrap()
        .prepare_assets()
        .err()
        .unwrap();
    assert!(error.to_string().contains("config.json"), "{error:#}");
    assert!(!error.to_string().contains("processor.json"), "{error:#}");
}

#[test]
fn explicit_cudnn_dependencies_keep_declared_order_without_loading_native_code() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("image.json");
    let mut wire = configuration();
    wire["backend"]["cudnn"]["dependency_libraries"] =
        json!(["lib/libnvrtc.so.13", "lib/libnvJitLink.so.13"]);
    fs::write(&path, serde_json::to_vec(&wire).unwrap()).unwrap();
    let config = ImageProductConfig::load(&path).unwrap();
    let ImageBackendConfig::Cuda(cuda) = config.backend;
    let base = fs::canonicalize(directory.path()).unwrap();
    assert_eq!(
        cuda.cudnn.dependency_libraries,
        vec![
            base.join("lib/libnvrtc.so.13"),
            base.join("lib/libnvJitLink.so.13")
        ]
    );
    #[cfg(feature = "cuda")]
    assert_eq!(
        cuda.resolved().unwrap().cudnn.dependency_libraries,
        cuda.cudnn.dependency_libraries
    );
    wire["backend"]["cudnn"]["dependency_libraries"] = json!(vec!["lib/missing.so"; 17]);
    fs::write(&path, serde_json::to_vec(&wire).unwrap()).unwrap();
    assert!(ImageProductConfig::load(&path).is_err());
}

#[test]
fn one_device_budget_drives_plan_resolution_and_obsolete_override_is_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("image.json");
    let mut wire = configuration();
    let load = |wire: &Value| {
        fs::write(&path, serde_json::to_vec(wire).unwrap()).unwrap();
        ImageProductConfig::load(&path)
    };
    // Both product entrypoints use this source loader. No assets or device
    // are opened; these nonzero opaque declarations must not be subtracted
    // from the internal plan's copy of the one account ceiling.
    for serving in [false, true] {
        if serving {
            wire["serving"] = json!({"maximum_pending":2,"maximum_prompt_bytes":4096,
                "max_num_seqs":1,"request_batch_max_wait_ms":0,"default_width":512,
                "default_height":512,"default_seed":0,"rgba_scratch_bytes":16384,"output":{}});
        }
        for budget in [1024, 30 * 1024 * 1024 * 1024_u64] {
            wire["backend"]["usable_memory_bytes"] = json!(budget);
            let config = load(&wire).unwrap();
            let ImageBackendConfig::Cuda(cuda) = config.backend;
            assert_eq!(
                cuda.resolved_pipeline().unwrap().device_capacity_bytes,
                budget
            );
            #[cfg(feature = "cuda")]
            {
                // Constructing typed settings still performs no CUDA calls.
                let resolved = cuda.resolved().unwrap();
                let ferrum_engine::image_generation::cuda::ImageCudaMemoryPolicy::ExplicitTotal(
                    capacity,
                ) = resolved.memory
                else {
                    panic!("explicit config must keep explicit policy");
                };
                assert_eq!(capacity.get(), budget);
                assert_eq!(
                    resolved
                        .pipeline
                        .with_device_capacity(capacity)
                        .device_capacity_bytes,
                    budget
                );
            }
        }
    }
    wire["backend"]["usable_memory_bytes"] = json!(0);
    assert!(load(&wire)
        .unwrap_err()
        .to_string()
        .contains("usable_memory_bytes"));
    wire["backend"]["usable_memory_bytes"] = json!(1024);
    for obsolete in [json!(1024), json!(512), json!(null)] {
        wire["backend"]["pipeline"]["device_capacity_bytes"] = obsolete;
        let error = load(&wire).unwrap_err().to_string();
        assert!(
            error.contains("unknown field") && error.contains("device_capacity_bytes"),
            "{error}"
        );
    }
}

#[path = "directory_tests.rs"]
mod directory_tests;

#[test]
fn experimental_batch_profile_connects_service_limit_to_real_pipeline_capacity() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("image.json");
    let mut wire = configuration();
    wire["serving"] = json!({"maximum_pending":4,"maximum_prompt_bytes":4096,"max_num_seqs":2,
        "request_batch_max_wait_ms":25,"default_width":512,"default_height":512,
        "default_seed":0,"rgba_scratch_bytes":16384,"output":{}});
    let load = |value: &Value| {
        fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
        ImageProductConfig::load(&path)
    };
    assert!(load(&wire).is_err()); // Missing cap remains one, including for serve.
    wire["backend"]["pipeline"]["maximum_batch_size"] = json!(2);
    assert!(load(&wire).is_err()); // Serial is not silently upgraded.
    wire["execution_profile"] = json!("experimental_batched_cold");
    let config = load(&wire).unwrap();
    assert_eq!(
        config.execution_profile.as_str(),
        "experimental_batched_cold"
    );
    let ImageBackendConfig::Cuda(cuda) = &config.backend;
    assert_eq!(cuda.resolved_pipeline().unwrap().maximum_batch_size, 2);
    assert_eq!(config.serving.as_ref().unwrap().max_num_seqs, 2);
    wire["serving"]["max_num_seqs"] = json!(3);
    assert!(load(&wire).is_err());
    wire["serving"]["max_num_seqs"] = json!(2);
    wire["serving"]["maximum_prompt_bytes"] = json!(512 * 1024 + 1);
    assert!(load(&wire).is_err());
    wire["serving"]["maximum_prompt_bytes"] = json!(512 * 1024);
    assert!(load(&wire).is_ok());
    wire["serving"]["maximum_prompt_bytes"] = json!(4096);
    for cap in [0, ferrum_runtime::STRUCTURED_ATTENTION_BF16_MAX_BATCH + 1] {
        wire["backend"]["pipeline"]["maximum_batch_size"] = json!(cap);
        assert!(load(&wire).is_err());
    }
    wire["backend"]["pipeline"]["maximum_batch_size"] =
        json!(ferrum_runtime::STRUCTURED_ATTENTION_BF16_MAX_BATCH);
    assert!(load(&wire).is_ok());
}

#[test]
fn edit_prefix_policy_is_explicit_and_shared_by_run_and_serve_configuration() {
    use ferrum_engine::image_generation::ImageEditPrefixPolicy;
    let mut wire = configuration();
    let config: ImageProductConfig = serde_json::from_value(wire.clone()).unwrap();
    let ImageBackendConfig::Cuda(cuda) = config.backend;
    assert_eq!(
        cuda.resolved_pipeline().unwrap().edit_prefix_policy,
        ImageEditPrefixPolicy::Recompute
    );
    for (name, expected) in [
        ("recompute", ImageEditPrefixPolicy::Recompute),
        ("request_cache", ImageEditPrefixPolicy::RequestCache),
    ] {
        wire["backend"]["pipeline"]["edit_prefix_policy"] = json!(name);
        let config: ImageProductConfig = serde_json::from_value(wire.clone()).unwrap();
        let ImageBackendConfig::Cuda(cuda) = config.backend;
        assert_eq!(
            cuda.resolved_pipeline().unwrap().edit_prefix_policy,
            expected
        );
    }
    for invalid in [json!(null), json!(true), json!("automatic")] {
        wire["backend"]["pipeline"]["edit_prefix_policy"] = invalid;
        assert!(serde_json::from_value::<ImageProductConfig>(wire.clone()).is_err());
    }
}

#[test]
fn decoder_batch_policy_is_explicit_independent_and_strict() {
    use ferrum_engine::image_generation::ImageDecoderBatchPolicy;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("image.json");
    let mut wire = configuration();
    wire["execution_profile"] = json!("experimental_batched_cold");
    wire["backend"]["pipeline"]["maximum_batch_size"] = json!(8);
    for (name, expected) in [
        ("single_image", ImageDecoderBatchPolicy::SingleImage),
        ("full_batch", ImageDecoderBatchPolicy::FullBatch),
    ] {
        wire["backend"]["pipeline"]["decoder_batch_policy"] = json!(name);
        fs::write(&path, serde_json::to_vec(&wire).unwrap()).unwrap();
        let config = ImageProductConfig::load(&path).unwrap();
        let ImageBackendConfig::Cuda(cuda) = config.backend;
        let resolved = cuda.resolved_pipeline().unwrap();
        assert_eq!(resolved.maximum_batch_size, 8);
        assert_eq!(resolved.decoder_batch_policy, expected);
    }
    for invalid in [
        json!(2),
        json!(null),
        json!("automatic"),
        json!({"single_image": {"max": 4}}),
    ] {
        wire["backend"]["pipeline"]["decoder_batch_policy"] = invalid;
        assert!(serde_json::from_value::<ImageProductConfig>(wire.clone()).is_err());
    }
}

#[test]
fn decoder_backing_retention_is_bounded_and_shared_by_run_and_serve() {
    use ferrum_engine::image_generation::ImageDynamicResidencyPolicy;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("image.json");
    let load = |wire: &Value| {
        fs::write(&path, serde_json::to_vec(wire).unwrap()).unwrap();
        ImageProductConfig::load(&path)
    };
    let mut wire = configuration();
    let ImageBackendConfig::Cuda(cuda) = load(&wire).unwrap().backend;
    assert_eq!(
        cuda.resolved_pipeline().unwrap().dynamic_residency,
        ImageDynamicResidencyPolicy::Release
    );
    wire["backend"]["pipeline"]["dynamic_residency"] =
        json!({"retain_decoder":{"maximum_bytes":1024}});
    assert!(load(&wire).is_err());
    wire["execution_profile"] = json!("experimental_resident_v1");
    wire["backend"]["pipeline"]["residency"] = json!("all_model_components");
    for serving in [false, true] {
        wire["serving"] = Value::Null;
        if serving {
            wire["serving"] = json!({"maximum_pending":2,"maximum_prompt_bytes":4096,
                "max_num_seqs":1,"request_batch_max_wait_ms":0,"default_width":512,
                "default_height":512,"default_seed":0,"rgba_scratch_bytes":16384,"output":{}});
        }
        let ImageBackendConfig::Cuda(cuda) = load(&wire).unwrap().backend;
        assert_eq!(
            cuda.resolved_pipeline().unwrap().dynamic_residency,
            ImageDynamicResidencyPolicy::RetainDecoder {
                maximum_bytes: 1024
            }
        );
        #[cfg(feature = "cuda")]
        assert_eq!(
            cuda.resolved().unwrap().pipeline.dynamic_residency,
            ImageDynamicResidencyPolicy::RetainDecoder {
                maximum_bytes: 1024
            }
        );
    }
    for invalid in [
        json!({"retain_decoder":{"maximum_bytes":0}}),
        json!({"retain_decoder":{"maximum_bytes":-1}}),
        json!({"retain_decoder":{"maximum_bytes":1024,"unaccounted":true}}),
        json!("automatic"),
    ] {
        wire["backend"]["pipeline"]["dynamic_residency"] = invalid;
        assert!(load(&wire).is_err());
    }
}

#[test]
fn per_stage_dynamic_budgets_are_shared_by_run_and_serve_and_reject_ineligible_stages() {
    use ferrum_engine::image_generation::{
        ImageDynamicResidencyPolicy, ImageStageRetentionBudgets,
    };
    use std::num::NonZeroU64;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("image.json");
    let load = |wire: &Value| {
        fs::write(&path, serde_json::to_vec(wire).unwrap()).unwrap();
        ImageProductConfig::load(&path)
    };
    let mut wire = configuration();
    let selected = json!({"retain_stage_budgets":{"maximum_bytes_by_stage":{
        "text_encoder":2048,"vision_encoder":3072,"decoder":4096
    }}});
    wire["backend"]["pipeline"]["dynamic_residency"] = selected.clone();
    assert!(load(&wire).is_err());
    wire["execution_profile"] = json!("experimental_resident_v1");
    wire["backend"]["pipeline"]["residency"] = json!("denoiser_and_decoder");
    assert!(load(&wire).is_err());
    wire["backend"]["pipeline"]["residency"] = json!("all_model_components");
    let expected = ImageDynamicResidencyPolicy::RetainStageBudgets {
        maximum_bytes_by_stage: ImageStageRetentionBudgets {
            text_encoder: NonZeroU64::new(2048),
            vision_encoder: NonZeroU64::new(3072),
            decoder: NonZeroU64::new(4096),
            ..ImageStageRetentionBudgets::default()
        },
    };
    for serving in [false, true] {
        wire["serving"] = Value::Null;
        if serving {
            wire["serving"] = json!({"maximum_pending":2,"maximum_prompt_bytes":4096,
                "max_num_seqs":1,"request_batch_max_wait_ms":0,"default_width":512,
                "default_height":512,"default_seed":0,"rgba_scratch_bytes":16384,"output":{}});
        }
        wire["backend"]["pipeline"]["dynamic_residency"] = selected.clone();
        wire["backend"]["pipeline"]["residency"] = json!("all_model_components");
        let ImageBackendConfig::Cuda(cuda) = load(&wire).unwrap().backend;
        assert_eq!(
            cuda.resolved_pipeline().unwrap().dynamic_residency,
            expected
        );
        #[cfg(feature = "cuda")]
        assert_eq!(
            cuda.resolved().unwrap().pipeline.dynamic_residency,
            expected
        );
        for invalid in [
            json!({}),
            json!({"decoder":0}),
            json!({"decoder":-1}),
            json!({"noise":1024}),
            json!({"decoder":1024,"unknown_stage":1}),
        ] {
            wire["backend"]["pipeline"]["dynamic_residency"] = json!({
                "retain_stage_budgets":{"maximum_bytes_by_stage":invalid}
            });
            assert!(load(&wire).is_err());
        }
        // Noise is an explicit AllStages opt-in; old AllModelComponents rejects it.
        wire["backend"]["pipeline"]["residency"] = json!("all_stages");
        wire["backend"]["pipeline"]["dynamic_residency"] = json!({
            "retain_stage_budgets":{"maximum_bytes_by_stage":{"noise":1024}}
        });
        let ImageBackendConfig::Cuda(cuda) = load(&wire).unwrap().backend;
        assert_eq!(
            cuda.resolved_pipeline().unwrap().residency,
            ferrum_engine::image_generation::ImageResidencyPolicy::AllStages
        );
        assert_eq!(
            serde_json::to_value(cuda.resolved_pipeline().unwrap().dynamic_residency).unwrap(),
            wire["backend"]["pipeline"]["dynamic_residency"]
        );
    }
}

#[test]
fn residency_is_explicit_and_shared_by_run_and_serve_configuration() {
    use ferrum_engine::image_generation::ImageResidencyPolicy;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("image.json");
    let load = |wire: &Value| {
        fs::write(&path, serde_json::to_vec(wire).unwrap()).unwrap();
        ImageProductConfig::load(&path)
    };
    let mut wire = configuration();
    let ImageBackendConfig::Cuda(cuda) = load(&wire).unwrap().backend;
    assert_eq!(
        cuda.resolved_pipeline().unwrap().residency,
        ImageResidencyPolicy::Cold
    );
    for (name, policy) in [
        (
            "denoiser_and_decoder",
            ImageResidencyPolicy::DenoiserAndDecoder,
        ),
        (
            "all_model_components",
            ImageResidencyPolicy::AllModelComponents,
        ),
        ("all_stages", ImageResidencyPolicy::AllStages),
    ] {
        wire["execution_profile"] = json!("experimental_serial_cold");
        wire["backend"]["pipeline"]["residency"] = json!(name);
        assert!(load(&wire).is_err()); // A cold execution identity must remain truthful.
        wire["execution_profile"] = json!("experimental_resident_v1");
        for serving in [false, true] {
            wire["serving"] = Value::Null;
            if serving {
                wire["serving"] = json!({"maximum_pending":2,"maximum_prompt_bytes":4096,
                "max_num_seqs":1,"request_batch_max_wait_ms":0,"default_width":512,
                "default_height":512,"default_seed":0,"rgba_scratch_bytes":16384,"output":{}});
            }
            let config = load(&wire).unwrap();
            assert_eq!(
                config.execution_profile.as_str(),
                "experimental_resident_v1"
            );
            let ImageBackendConfig::Cuda(cuda) = config.backend;
            assert_eq!(cuda.resolved_pipeline().unwrap().residency, policy);
            #[cfg(feature = "cuda")]
            assert_eq!(cuda.resolved().unwrap().pipeline.residency, policy);
        }
    }
    wire["backend"]["pipeline"]
        .as_object_mut()
        .unwrap()
        .remove("residency");
    assert!(load(&wire).is_err()); // Default cold cannot silently become resident.
    for bad in [json!(null), json!(true), json!("all"), json!({"cold":{}})] {
        wire["backend"]["pipeline"]["residency"] = bad;
        assert!(load(&wire).is_err());
    }
}

#[test]
fn bounded_metadata_and_declared_architecture_fail_before_missing_weight_sources() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("small.json");
    fs::write(&path, b"{}").unwrap();
    assert_eq!(source::read_bounded(&path, 2).unwrap(), b"{}");
    assert!(source::read_bounded(&path, 1).is_err());
    fs::write(&path, []).unwrap();
    assert!(source::read_bounded(&path, 2).is_err());

    let config_path = dir.path().join("image.json");
    fs::write(&config_path, serde_json::to_vec(&configuration()).unwrap()).unwrap();
    fs::write(
        dir.path().join("model_index.json"),
        br#"{"_class_name":"UnsupportedPipeline"}"#,
    )
    .unwrap();
    let error = ImageProductConfig::load(&config_path)
        .unwrap()
        .prepare_assets()
        .err()
        .unwrap();
    assert!(
        !error.to_string().contains("open "),
        "architecture must fail before opening other assets: {error}"
    );
}

#[test]
fn exact_weight_source_never_falls_back_to_another_filename() {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    let dir = tempfile::tempdir().unwrap();
    let data = [0u8, 0];
    serialize_to_file(
        [("w", TensorView::new(Dtype::BF16, vec![1], &data).unwrap())],
        &None,
        &dir.path().join("chosen.safetensors"),
    )
    .unwrap();
    let source = WeightSource::Single {
        directory: dir.path().into(),
        file: "chosen.safetensors".into(),
    };
    assert!(source.open().is_ok());
    for file in [
        "missing.safetensors",
        "../chosen.safetensors",
        "dir/chosen.safetensors",
    ] {
        assert!(WeightSource::Single {
            directory: dir.path().into(),
            file: file.into()
        }
        .open()
        .is_err());
    }
    assert!(WeightSource::Index {
        directory: dir.path().into(),
        file: "missing.index.json".into()
    }
    .open()
    .is_err());
}

#[derive(Parser)]
struct TestCli {
    #[command(subcommand)]
    command: TestCommand,
}
#[derive(Subcommand)]
enum TestCommand {
    Run(crate::commands::run::RunCliCommand),
    Serve(crate::commands::serve::ServeCliCommand),
}
fn validate(arguments: &[&str]) -> anyhow::Result<()> {
    let definition = TestCli::command();
    let matches = definition.clone().try_get_matches_from(arguments)?;
    validate_cli_matches(&definition, &matches)
}

#[test]
fn run_image_controls_reject_explicit_text_defaults_and_keep_text_commands() {
    assert!(validate(&[
        "ferrum",
        "run",
        "--image-config",
        "image.json",
        "--image",
        "input.png",
        "--prompt",
        "edit",
        "--output",
        "out.png"
    ])
    .is_ok());
    assert!(validate(&[
        "ferrum",
        "run",
        "text-model",
        "--image",
        "input.png",
        "--prompt",
        "edit"
    ])
    .is_err());
    assert!(validate(&[
        "ferrum",
        "serve",
        "--image-config",
        "image.json",
        "--image",
        "input.png"
    ])
    .is_err());
    assert!(validate(&[
        "ferrum",
        "run",
        "--image-config",
        "image.json",
        "--prompt",
        "a cat",
        "--output",
        "image.png"
    ])
    .is_ok());
    assert!(validate(&[
        "ferrum",
        "run",
        "text-model",
        "--prompt",
        "hello",
        "--max-tokens",
        "4096"
    ])
    .is_ok());
    for extra in [
        vec!["--max-tokens", "4096"],
        vec!["--backend", "auto"],
        vec!["text-model"],
        vec!["--temperature", "0.0"],
    ] {
        let mut args = vec![
            "ferrum",
            "run",
            "--image-config",
            "image.json",
            "--prompt",
            "a cat",
            "--output",
            "image.png",
        ];
        args.extend(extra);
        assert!(validate(&args).is_err(), "{args:?}");
    }
    assert!(validate(&[
        "ferrum",
        "run",
        "--image-config",
        "image.json",
        "--output",
        "image.png"
    ])
    .is_err());
    assert!(validate(&[
        "ferrum",
        "run",
        "--image-config",
        "image.json",
        "--prompt",
        "a cat"
    ])
    .is_err());
    assert!(validate(&["ferrum", "run", "text-model", "--width", "512"]).is_err());
}

#[cfg(not(feature = "cuda"))]
#[test]
fn non_cuda_image_run_rejects_before_file_or_asset_io() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("image.png");
    let error = execute_run(
        &PathBuf::from("missing.json"),
        "cat".into(),
        512,
        512,
        0,
        &output,
    )
    .unwrap_err();
    assert!(matches!(error, ImageProductError::Unsupported(_)));
    assert!(!output.exists());
}

#[test]
fn image_serve_rejects_text_controls_and_requires_bounded_capacity() {
    assert!(validate(&[
        "ferrum",
        "serve",
        "--image-config",
        "image.json",
        "--host",
        "127.0.0.1",
        "--port",
        "8090"
    ])
    .is_ok());
    assert!(validate(&[
        "ferrum",
        "serve",
        "--image-config",
        "image.json",
        "--backend",
        "auto"
    ])
    .is_err());
    assert!(validate(&[
        "ferrum",
        "serve",
        "--image-config",
        "image.json",
        "--prefix-rendezvous-max-wait-ms",
        "1"
    ])
    .is_err());
    assert!(validate(&["ferrum", "serve", "--model", "text-model"]).is_ok());
    let mut wire = json!({"maximum_pending":2,"maximum_prompt_bytes":4096,"max_num_seqs":1,
        "request_batch_max_wait_ms":0,"default_width":512,"default_height":512,
        "default_seed":0,"rgba_scratch_bytes":16384,"output":{}});
    serde_json::from_value::<ServingConfig>(wire.clone())
        .unwrap()
        .validate()
        .unwrap();
    wire["max_num_seqs"] = json!(ferrum_runtime::STRUCTURED_ATTENTION_BF16_MAX_BATCH + 1);
    assert!(serde_json::from_value::<ServingConfig>(wire.clone())
        .unwrap()
        .validate()
        .is_err());
    wire["max_num_seqs"] = json!(1);
    wire["maximum_pending"] = json!(0);
    assert!(serde_json::from_value::<ServingConfig>(wire)
        .unwrap()
        .validate()
        .is_err());
}

#[test]
fn bounded_plan_cache_configuration_is_strict_and_shared_by_run_and_serve() {
    use ferrum_engine::image_generation::ImagePlanCachePolicy;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("image.json");
    let load = |wire: &Value| {
        fs::write(&path, serde_json::to_vec(wire).unwrap()).unwrap();
        ImageProductConfig::load(&path)
    };
    let mut wire = configuration();
    let ImageBackendConfig::Cuda(default) = load(&wire).unwrap().backend;
    assert_eq!(
        default.resolved_pipeline().unwrap().plan_cache,
        ImagePlanCachePolicy::LastPerStage
    );
    let policy = ImagePlanCachePolicy::MultiShape {
        maximum_entries_per_stage: 2,
        maximum_encoded_metadata_bytes: 65536,
    };
    for serving in [false, true] {
        wire["serving"] = if serving {
            json!({"maximum_pending":2,"maximum_prompt_bytes":4096,"max_num_seqs":1,
                "request_batch_max_wait_ms":0,"default_width":512,"default_height":512,
                "default_seed":0,"rgba_scratch_bytes":16384,"output":{}})
        } else {
            Value::Null
        };
        for selected in [
            ImagePlanCachePolicy::Disabled,
            ImagePlanCachePolicy::LastPerStage,
            policy,
        ] {
            wire["backend"]["pipeline"]["plan_cache"] = serde_json::to_value(selected).unwrap();
            let ImageBackendConfig::Cuda(config) = load(&wire).unwrap().backend;
            assert_eq!(config.resolved_pipeline().unwrap().plan_cache, selected);
        }
    }
    for invalid in [
        json!("multi_shape"),
        json!({"multi_shape":{"maximum_entries_per_stage":2}}),
        json!({"multi_shape":{"maximum_encoded_metadata_bytes":65536}}),
        json!({"multi_shape":{"maximum_entries_per_stage":0,"maximum_encoded_metadata_bytes":65536}}),
        json!({"multi_shape":{"maximum_entries_per_stage":2,"maximum_encoded_metadata_bytes":0}}),
        json!({"multi_shape":{"maximum_entries_per_stage":2,"maximum_metadata_bytes":65536}}),
        json!({"multi_shape":{"maximum_entries_per_stage":2,"maximum_encoded_metadata_bytes":65536,"unused":1}}),
        json!({"multi_shape":{"maximum_entries_per_stage":usize::MAX,"maximum_encoded_metadata_bytes":65536}}),
    ] {
        wire["backend"]["pipeline"]["plan_cache"] = invalid;
        assert!(load(&wire).is_err());
    }
}
