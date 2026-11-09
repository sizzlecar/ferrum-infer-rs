use super::*;
use serde_json::{json, Value};

#[path = "tests/diagnostics.rs"]
mod diagnostics;

#[path = "tests/source_backing.rs"]
mod source_backing;

#[test]
fn main_vision_reuse_is_explicit_and_uses_the_shared_asset_configuration() {
    let root = tempfile::tempdir().unwrap();
    assert_eq!(
        load(root.path(), &enabled_document())
            .unwrap()
            .main_vision_reuse,
        PH3ImageMainVisionReuse::Off
    );
    for (wire_policy, expected) in [
        ("off", PH3ImageMainVisionReuse::Off),
        (
            "exact_image_within_cohort",
            PH3ImageMainVisionReuse::ExactImageWithinCohort,
        ),
    ] {
        let mut wire = enabled_document();
        wire["main_vision_reuse"] = json!(wire_policy);
        let parsed = load(root.path(), &wire).unwrap();
        assert_eq!(parsed.main_vision_reuse, expected);
        let PH3ImageBackendConfig::Cuda(backend) = parsed.backend;
        assert_eq!(backend.pipeline.maximum_batch_size, 1);
    }
    for bad in [
        json!(null),
        json!(true),
        json!("same_prompt"),
        json!({"off":true}),
    ] {
        let mut wire = enabled_document();
        wire["main_vision_reuse"] = bad;
        assert!(load(root.path(), &wire).is_err());
    }
    let mut disabled = document();
    disabled["main_vision_reuse"] = json!("exact_image_within_cohort");
    assert!(load(root.path(), &disabled).is_err());
}

#[test]
fn cpu_initial_noise_default_and_explicit_policies_keep_reference_policy_and_capacity() {
    let directory = tempfile::tempdir().unwrap();
    let mut wire = document();
    let default = load(directory.path(), &wire).unwrap();
    assert_eq!(
        default.initial_noise,
        PH3ImageInitialNoisePolicy::CpuMt19937ImageAudioV1
    );
    wire["initial_noise"] = json!("cpu_mt19937_image_audio_v1");
    let parsed = load(directory.path(), &wire).unwrap();
    assert_eq!(
        parsed.initial_noise,
        PH3ImageInitialNoisePolicy::CpuMt19937ImageAudioV1
    );
    assert_eq!(parsed.initial_noise, default.initial_noise);
    let mut native_wire = wire.clone();
    native_wire["initial_noise"] = json!("native_cuda_philox_joint_image_audio_v1");
    let native = load(directory.path(), &native_wire).unwrap();
    assert_eq!(
        native.initial_noise,
        PH3ImageInitialNoisePolicy::NativeCudaPhiloxJointImageAudioV1
    );
    assert_eq!(
        wire["backend"]["usable_memory_bytes"],
        document()["backend"]["usable_memory_bytes"]
    );
    assert_eq!(
        wire["backend"]["pipeline"],
        document()["backend"]["pipeline"]
    );
    #[cfg(feature = "cuda")]
    for config in [&default, &parsed, &native] {
        let PH3ImageBackendConfig::Cuda(backend) = &config.backend;
        std::fs::create_dir_all(backend.cudnn.library_path.parent().unwrap()).unwrap();
        std::fs::write(&backend.cudnn.library_path, b"configuration-only fixture").unwrap();
        let resolved = super::super::factory::PreparedBackend::from_p_h3_config(config).unwrap();
        let super::super::factory::PreparedBackend::PH3Cuda(resolved) = resolved else {
            panic!("declared H3 backend")
        };
        assert_eq!(resolved.options.initial_noise, config.initial_noise);
        assert_eq!(
            resolved.options.edit_noise,
            ferrum_engine::image_generation::p_h3::PH3ImageEditNoisePolicy::default()
        );
        assert_eq!(
            resolved.options.noise_profile.as_str(),
            ferrum_kernels::backend::cuda::vnext_ops::CUDA_PHILOX_NORMAL_F32_NUMERICAL_PROFILE_ID
        );
        assert_eq!(
            resolved.pipeline.maximum_batch_size,
            backend.pipeline.maximum_batch_size
        );
    }
    for invalid in [
        json!(null),
        json!(false),
        json!(1),
        json!("torch"),
        json!({"cpu_mt19937_image_audio_v1":true}),
    ] {
        wire["initial_noise"] = invalid;
        assert!(load(directory.path(), &wire).is_err());
    }
}

#[test]
fn native_allocation_reuse_is_explicit_and_charged_inside_the_original_ceiling() {
    let directory = tempfile::tempdir().unwrap();
    let mut wire = document();
    let PH3ImageBackendConfig::Cuda(default) = load(directory.path(), &wire).unwrap().backend;
    assert_eq!(default.native_allocation_reuse, None);
    let policy = json!({"maximum_idle_bytes":4294967296_u64,
        "maximum_entry_bytes":2147483648_u64,"maximum_entries":16});
    wire["backend"]["native_allocation_reuse"] = policy.clone();
    let parsed = load(directory.path(), &wire).unwrap();
    let PH3ImageBackendConfig::Cuda(backend) = &parsed.backend;
    assert_eq!(
        backend
            .native_allocation_reuse
            .unwrap()
            .maximum_idle_bytes
            .get(),
        4 << 30
    );
    assert!(!directory.path().join("comfy").exists());
    #[cfg(feature = "cuda")]
    {
        std::fs::create_dir_all(backend.cudnn.library_path.parent().unwrap()).unwrap();
        std::fs::write(&backend.cudnn.library_path, b"configuration-only fixture").unwrap();
        let resolved = backend
            .resolved(parsed.vae_spatial_policy, parsed.vae_decoder_tile_batch)
            .unwrap();
        assert_eq!(
            resolved.native_allocation_reuse.maximum_idle_bytes(),
            4 << 30
        );
    }
    for key in [
        "maximum_idle_bytes",
        "maximum_entry_bytes",
        "maximum_entries",
    ] {
        for bad in [json!(0), json!(-1), json!(1.5), json!("16")] {
            wire["backend"]["native_allocation_reuse"] = policy.clone();
            wire["backend"]["native_allocation_reuse"][key] = bad;
            assert!(load(directory.path(), &wire).is_err());
        }
        wire["backend"]["native_allocation_reuse"] = policy.clone();
        wire["backend"]["native_allocation_reuse"]
            .as_object_mut()
            .unwrap()
            .remove(key);
        assert!(load(directory.path(), &wire).is_err());
    }
    wire["backend"]["native_allocation_reuse"] = policy.clone();
    wire["backend"]["native_allocation_reuse"]["maximum_entry_bytes"] = json!(4294967297_u64);
    assert!(load(directory.path(), &wire).is_err());
    wire["backend"]["native_allocation_reuse"] = policy.clone();
    wire["backend"]["native_allocation_reuse"]["unknown"] = json!(1);
    assert!(load(directory.path(), &wire).is_err());
    wire["backend"]["native_allocation_reuse"] = policy;
    wire["backend"]["usable_memory_bytes"] = json!(4_u64 << 30);
    assert!(
        load(directory.path(), &wire).is_err(),
        "cache reservation cannot be added outside the original total"
    );
    wire["backend"]
        .as_object_mut()
        .unwrap()
        .remove("native_allocation_reuse");
    load(directory.path(), &wire).unwrap();
}

#[test]
fn native_lane_reuse_is_opt_in_and_bounded_by_resolved_streams() {
    let directory = tempfile::tempdir().unwrap();
    let mut wire = enabled_document();
    let default = load(directory.path(), &wire).unwrap();
    let PH3ImageBackendConfig::Cuda(default_backend) = default.backend;
    assert_eq!(default_backend.native_lane_reuse_maximum_idle_lanes, None);
    wire["backend"]["maximum_streams"] = json!(3);
    for maximum in [1, 3] {
        wire["backend"]["native_lane_reuse_maximum_idle_lanes"] = json!(maximum);
        let config = load(directory.path(), &wire).unwrap();
        let PH3ImageBackendConfig::Cuda(backend) = &config.backend;
        assert_eq!(
            backend.native_lane_reuse_maximum_idle_lanes.unwrap().get(),
            maximum
        );
        #[cfg(feature = "cuda")]
        {
            std::fs::create_dir_all(backend.cudnn.library_path.parent().unwrap()).unwrap();
            std::fs::write(&backend.cudnn.library_path, b"configuration-only fixture").unwrap();
            let resolved = backend
                .resolved(config.vae_spatial_policy, config.vae_decoder_tile_batch)
                .unwrap();
            assert_eq!(
                resolved.native_lane_reuse,
                ferrum_kernels::backend::cuda::vnext_runtime::CudaNativeLaneReuse::Bounded {
                    maximum_idle_lanes: std::num::NonZeroUsize::new(maximum).unwrap()
                }
            );
        }
    }
    for invalid in [json!(0), json!(-1), json!(1.5), json!("1"), json!(4)] {
        wire["backend"]["native_lane_reuse_maximum_idle_lanes"] = invalid;
        assert!(load(directory.path(), &wire).is_err());
    }
    wire["backend"]
        .as_object_mut()
        .unwrap()
        .remove("maximum_streams");
    assert!(load(directory.path(), &wire).is_err());
    wire["backend"]["native_lane_reuse_maximum_idle_lanes"] = json!(3);
    load(directory.path(), &wire).unwrap();
    let mut tensor_only = document();
    tensor_only["backend"]["native_lane_reuse_maximum_idle_lanes"] = json!(1);
    load(directory.path(), &tensor_only).unwrap();
    assert!(!directory.path().join("weights").exists());
}

#[test]
fn bounded_weight_arenas_reach_shared_expansion_options_without_loading_weights() {
    let directory = tempfile::tempdir().unwrap();
    for target in [None, Some(536_870_912_u64)] {
        let mut wire = enabled_document();
        if let Some(bytes) = target {
            wire["prompt_expansion"]["weight_arena_target_bytes"] = json!(bytes);
        }
        let config = load(directory.path(), &wire).unwrap();
        let PH3ImagePromptExpansionConfig::Enabled(expansion) = config.prompt_expansion else {
            panic!("enabled fixture")
        };
        let options = expansion.options(0).unwrap();
        assert_eq!(
            options
                .engine
                .memory
                .weight_arena_target_bytes
                .map(|v| v.get()),
            target
        );
    }
    for invalid in [json!(0), json!(-1), json!(1.5), json!("536870912")] {
        let mut wire = enabled_document();
        wire["prompt_expansion"]["weight_arena_target_bytes"] = invalid;
        assert!(load(directory.path(), &wire).is_err());
    }
    let mut disabled = document();
    disabled["prompt_expansion"]["weight_arena_target_bytes"] = json!(536_870_912);
    assert!(load(directory.path(), &disabled).is_err());
    assert!(!directory.path().join("weights").exists());
}

fn document() -> Value {
    json!({
        "architecture":"p_h3_image","model":"public-alias","preset":"comfy_w4a8_f32_v1",
        "prompt_expansion":{"mode":"disabled"},
        "metadata_directory":"metadata","archives":{"text_encoder":"comfy/text.safetensors",
            "transformer":"comfy/dit.safetensors","vae":"comfy/vae.safetensors"},
        "backend":{"kind":"cuda","ordinal":0,"usable_memory_bytes":34359738368_u64,
            "bootstrap_memory_bytes":268435456,"stream_memory_bytes":16777216,
            "maximum_in_flight_per_stream":1,"maximum_commands_per_submission":4096,
            "opaque_memory_profile":"declared-h3-runtime-v1",
            "cudnn":{"library_path":"native/libcudnn.so.9","dependency_libraries":[],
                "opaque_memory_bytes":67108864,"workspace_limit_bytes":16777216,
                "opaque_memory_profile":"declared-h3-cudnn-f32-v1"},
            "host":{"total_bytes":1073741824,"input_preprocess_bytes":536870912,
                "output_bytes":268435456,"transfer_staging_bytes":16777216,"weight_backing_bytes":67108864},
            "pipeline":{"maximum_user_tokens":1024,"maximum_width":4096,"maximum_height":4096,
                "maximum_batch_size":1,"maximum_staging_bytes":16777216,
                "maximum_initialization_commands":64,"maximum_backing_maintenance_passes":8}}
    })
}
fn load(root: &Path, value: &Value) -> Result<PH3ImageProductConfig> {
    let path = root.join("pipeline.json");
    std::fs::write(&path, serde_json::to_vec(value)?)?;
    PH3ImageProductConfig::load(&path)
}

#[test]
fn spatial_decoder_batch_is_explicit_and_independent_of_request_capacity() {
    let directory = tempfile::tempdir().unwrap();
    let mut wire = document();
    let default = load(directory.path(), &wire).unwrap();
    assert_eq!(default.vae_decoder_tile_batch.maximum_tiles(), 1);
    wire["vae_decoder_tile_batch"] = json!(4);
    let config = load(directory.path(), &wire).unwrap();
    assert_eq!(config.vae_decoder_tile_batch.maximum_tiles(), 4);
    let PH3ImageBackendConfig::Cuda(backend) = &config.backend;
    assert_eq!(backend.pipeline.maximum_batch_size, 1);
    assert_eq!(config.preset, default.preset);
    assert_eq!(config.vae_spatial_policy, default.vae_spatial_policy);
    #[cfg(feature = "cuda")]
    {
        // Configuration resolution only discovers a regular path. It must
        // carry the policy without loading a native library or a CUDA driver.
        std::fs::create_dir_all(backend.cudnn.library_path.parent().unwrap()).unwrap();
        std::fs::write(&backend.cudnn.library_path, b"configuration-only fixture").unwrap();
        let resolved = backend
            .resolved(config.vae_spatial_policy, config.vae_decoder_tile_batch)
            .unwrap();
        assert_eq!(resolved.options.vae_decoder_tile_batch.maximum_tiles(), 4);
    }
    for invalid in [json!(0), json!(5), json!(-1), json!(1.5), json!("4")] {
        wire["vae_decoder_tile_batch"] = invalid;
        assert!(load(directory.path(), &wire).is_err());
    }
}

#[test]
fn explicit_sdpa_wire_accounts_for_native_plan_allowance_before_cuda() {
    let directory = tempfile::tempdir().unwrap();
    let mut wire = document();
    let config = load(directory.path(), &wire).unwrap();
    let PH3ImageBackendConfig::Cuda(backend) = config.backend;
    assert_eq!(
        backend.attention_policy,
        ferrum_engine::image_generation::ImageCudaAttentionPolicy::Sequential
    );
    wire["backend"]["attention_policy"] = json!("cudnn_sdpa_experimental_v1");
    let config = load(directory.path(), &wire).unwrap();
    let PH3ImageBackendConfig::Cuda(backend) = config.backend;
    assert_eq!(
        backend.attention_policy,
        ferrum_engine::image_generation::ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1
    );
    let streams = backend
        .maximum_streams
        .resolve(ImageStreamRequirements::for_serial_pipeline(
            backend
                .pipeline
                .resolved_limits()
                .unwrap()
                .static_initialization,
            backend.pipeline.resolved_limits().unwrap().executor,
        ))
        .unwrap()
        .resolved_streams() as u64;
    let old_opaque = backend.bootstrap_memory_bytes
        + backend.stream_memory_bytes * streams
        + backend.cudnn.opaque_memory_bytes;
    wire["backend"]["usable_memory_bytes"] =
        json!(old_opaque + backend.cudnn.workspace_limit_bytes + 1);
    assert!(load(directory.path(), &wire).is_err());
    wire["backend"]["attention_policy"] = json!("sequential");
    load(directory.path(), &wire).unwrap();
    wire["backend"]["attention_policy"] = json!("automatic");
    assert!(load(directory.path(), &wire).is_err());
}

#[test]
fn continuous_trace_is_explicit_bounded_and_resolved_without_opening_a_sink_during_parse() {
    let directory = tempfile::tempdir().unwrap();
    assert!(load(directory.path(), &document())
        .unwrap()
        .continuous_trace
        .is_none());
    let mut wire = document();
    wire["continuous_trace"] = json!({"path":"steps.jsonl", "maximum_bytes":4096});
    let config = load(directory.path(), &wire).unwrap();
    let trace = config.continuous_trace.unwrap();
    assert_eq!(
        trace.path,
        directory.path().canonicalize().unwrap().join("steps.jsonl")
    );
    assert_eq!(trace.maximum_bytes, 4096);
    assert!(
        !trace.path.exists(),
        "metadata parsing must not create a diagnostic artifact"
    );
    for invalid in [
        json!({"path":"", "maximum_bytes":4096}),
        json!({"path":"steps.jsonl", "maximum_bytes":0}),
        json!({"path":"steps.jsonl", "maximum_bytes":4096, "append":true}),
        json!({"path":"steps.jsonl"}),
    ] {
        wire["continuous_trace"] = invalid;
        assert!(load(directory.path(), &wire).is_err());
    }
    assert!(!trace.path.exists());
}

#[cfg(feature = "cuda")]
#[test]
fn image_profile_rejects_continuous_trace_before_preopening_its_path_or_loading_assets() {
    let directory = tempfile::tempdir().unwrap();
    let mut wire = document();
    wire["continuous_trace"] = json!({"path":"steps.jsonl", "maximum_bytes":4096});
    load(directory.path(), &wire).unwrap();
    let source =
        super::super::ImageProductSource::ExplicitConfig(directory.path().join("pipeline.json"));
    let error = source
        .prepare_for_profile()
        .err()
        .expect("profile has no trace consumer");
    assert!(format!("{error:#}").contains("not image profile"));
    assert!(!directory.path().join("steps.jsonl").exists());
    assert!(!directory.path().join("comfy").exists());
}
#[test]
fn explicit_architecture_and_precision_are_independent_of_the_public_alias() {
    let root = tempfile::tempdir().unwrap();
    let base = root.path().canonicalize().unwrap();
    let config = load(root.path(), &document()).unwrap();
    let dispatched =
        super::super::document::ImageProductDocument::load(&root.path().join("pipeline.json"))
            .unwrap();
    let super::super::document::ImageProductDocument::PH3Image(dispatched) = dispatched else {
        panic!("the declared architecture must select the H3 product schema");
    };
    assert_eq!(dispatched.preset, config.preset);
    assert_eq!(dispatched.archives.transformer, config.archives.transformer);
    assert_eq!(config.preset, PH3ImageProductPreset::ComfyW4a8F32V1);
    assert_eq!(config.preset.identity(), "comfy_w4a8_f32_v1");
    assert_eq!(config.metadata_directory, base.join("metadata"));
    assert_eq!(
        config.archives.text_encoder,
        base.join("comfy/text.safetensors")
    );
    assert_eq!(
        config.vae_spatial_policy,
        PH3ImageVaeSpatialPolicy::comfy_default(4096)
    );
    let PH3ImageBackendConfig::Cuda(backend) = config.backend;
    assert_eq!(
        backend.cudnn.library_path,
        base.join("native/libcudnn.so.9")
    );
    // Parsing/resolution never opens a driver/library or source payload.
    assert!(!root.path().join("native").exists());
    assert!(!root.path().join("comfy").exists());
    let mut renamed = document();
    renamed["model"] = json!("Qwen2.1 public serving alias");
    let renamed = load(root.path(), &renamed).unwrap();
    assert_eq!(renamed.architecture, PH3ImageProductArchitecture::PH3Image);
    assert_eq!(renamed.preset, PH3ImageProductPreset::ComfyW4a8F32V1);
    assert_eq!(renamed.archives.vae, base.join("comfy/vae.safetensors"));
}

#[test]
fn undeclared_backends_precisions_and_missing_real_host_allowances_are_rejected() {
    let root = tempfile::tempdir().unwrap();
    let mut wrong = document();
    wrong["backend"]["kind"] = json!("metal");
    assert!(load(root.path(), &wrong).is_err());
    let mut wrong = document();
    wrong["preset"] = json!("fp8_linear_experimental_v1");
    assert!(load(root.path(), &wrong).is_err());
    let mut wrong = document();
    wrong["backend"]["precision_policy"] = json!("bf16");
    assert!(load(root.path(), &wrong).is_err());
    let mut wrong = document();
    wrong["backend"]["host"]["weight_backing_bytes"] = json!(0);
    assert!(format!("{:#}", load(root.path(), &wrong).err().unwrap()).contains("source backing"));
    let mut wrong = document();
    wrong["backend"]["pipeline"]["edit_prefix_policy"] = json!("request_cache");
    assert!(format!("{:#}", load(root.path(), &wrong).err().unwrap())
        .contains("recomputed bidirectional"));
    let mut wrong = document();
    wrong["archives"]["text_encoder"] = json!("");
    assert!(format!("{:#}", load(root.path(), &wrong).err().unwrap()).contains("nonempty"));
    let mut wrong = document();
    wrong["backend"]["maximum_streams"] = json!(1);
    assert!(load(root.path(), &wrong).is_err());
}

#[test]
fn bounded_document_and_missing_modular_metadata_fail_before_mapping_weights() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("oversized.json");
    std::fs::write(&path, vec![b' '; (1 << 20) + 1]).unwrap();
    assert!(
        format!("{:#}", PH3ImageProductConfig::load(&path).err().unwrap())
            .contains("metadata limit")
    );
    std::fs::create_dir(root.path().join("metadata")).unwrap();
    std::fs::write(root.path().join("metadata/model_index.json"), b"{}").unwrap();
    let config = load(root.path(), &document()).unwrap();
    let error = format!("{:#}", config.prepare_assets().err().unwrap());
    assert!(error.contains("modular_model_index.json"), "{error}");
    assert!(!error.contains("open H3 text_encoder archive"), "{error}");
}

#[test]
fn read_limits_and_spatial_policy_are_checked_typed_configuration() {
    let root = tempfile::tempdir().unwrap();
    let mut wrong = document();
    wrong["metadata_limits"] =
        json!({"configuration_bytes":0,"tokenizer_bytes":1024,"total_bytes":4096});
    assert!(load(root.path(), &wrong).is_err());
    let mut wrong = document();
    wrong["vae_spatial_policy"] =
        json!({"kind":"tiled","tile_pixels":256,"minimum_overlap_pixels":256,"max_tiles":16});
    assert!(load(root.path(), &wrong).is_err());
    let mut whole = document();
    whole["vae_spatial_policy"] = json!({"kind":"whole_image"});
    assert_eq!(
        load(root.path(), &whole).unwrap().vae_spatial_policy,
        PH3ImageVaeSpatialPolicy::WholeImage
    );
}

#[test]
fn serving_uses_the_same_explicit_host_and_pipeline_limits() {
    let root = tempfile::tempdir().unwrap();
    let mut wire = document();
    wire["backend"]["pipeline"]["maximum_batch_size"] = json!(2);
    wire["serving"] = json!({
        "batching":"auto","maximum_pending":4,"maximum_prompt_bytes":4096,
        "max_num_seqs":2,"request_batch_max_wait_ms":0,"default_width":512,
        "default_height":512,"default_seed":7,"rgba_scratch_bytes":16384,"output":{}
    });
    let loaded = load(root.path(), &wire).unwrap();
    let serving = loaded.serving.unwrap();
    assert_eq!(serving.max_num_seqs, 2);
    assert_eq!(serving.maximum_pending_bytes.unwrap().get(), 536870912 / 2);
    wire["serving"]["max_num_seqs"] = json!(3);
    assert!(format!("{:#}", load(root.path(), &wire).err().unwrap()).contains("batch capacity"));
    wire["serving"]["max_num_seqs"] = json!(2);
    wire["serving"]["request_batch_max_wait_ms"] = json!(1);
    assert!(load(root.path(), &wire).is_err());
}

fn enabled_document() -> Value {
    let mut wire = document();
    wire["prompt_expansion"] = json!({"mode":"enabled", "semantic_directory":"upstream/prompt_upsampler",
        "tokenizer_directory":"upstream/prompt_upsampler_processor", "bf16_archive":"weights/official-bf16.safetensors",
        "processor_config":"upstream/prompt_upsampler_processor/processor_config.json"});
    wire["backend"]["prompt_expansion_causal"] = json!({"module_memory_bytes":268435456,
        "opaque_memory_profile":"declared-native-dense-f16-causal-unqualified-v1"});
    wire
}

fn processor_document() -> Value {
    json!({"processor_class":"Qwen3VLProcessor", "image_processor":{
        "image_processor_type":"Qwen2VLImageProcessor", "do_convert_rgb":true,
        "do_normalize":true,"do_rescale":true,"do_resize":true,"resample":3,
        "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"rescale_factor":1.0/255.0,
        "patch_size":2,"temporal_patch_size":2,"merge_size":2,
        "size":{"shortest_edge":16,"longest_edge":256}}})
}

#[test]
fn prompt_expansion_vision_reuse_is_explicit_and_shared_configuration_preserves_limits() {
    use super::prompt_expansion::PH3ImagePromptExpansionEnabledConfig;
    use ferrum_engine::image_generation::p_h3::PH3ImagePromptExpansionVisionReuse as Policy;

    let root = tempfile::tempdir().unwrap();
    let default = load(root.path(), &enabled_document()).unwrap();
    let PH3ImagePromptExpansionConfig::Enabled(default_expansion) = &default.prompt_expansion
    else {
        panic!("enabled fixture")
    };
    assert_eq!(default_expansion.vision_reuse, Policy::Off);
    assert_eq!(
        PH3ImagePromptExpansionEnabledConfig::default().vision_reuse,
        Policy::Off
    );
    for (wire_policy, expected) in [
        ("off", Policy::Off),
        ("exact_image_within_cohort", Policy::ExactImageWithinCohort),
    ] {
        let mut wire = enabled_document();
        wire["prompt_expansion"]["vision_reuse"] = json!(wire_policy);
        let parsed = load(root.path(), &wire).unwrap();
        let PH3ImagePromptExpansionConfig::Enabled(expansion) = &parsed.prompt_expansion else {
            panic!("enabled fixture")
        };
        assert_eq!(expansion.vision_reuse, expected);
        assert_eq!(
            expansion.max_context_tokens,
            default_expansion.max_context_tokens
        );
        assert_eq!(expansion.max_new_tokens, default_expansion.max_new_tokens);
        assert_eq!(
            expansion.prefill_chunk_tokens,
            default_expansion.prefill_chunk_tokens
        );
        assert_eq!(
            expansion.vision_limits.max_patch_bytes,
            default_expansion.vision_limits.max_patch_bytes
        );
        assert_eq!(expansion.source_backing, default_expansion.source_backing);
        assert_eq!(
            expansion.source_conversion_workers,
            default_expansion.source_conversion_workers
        );
    }
    for invalid in [
        json!("automatic"),
        json!(null),
        json!(true),
        json!(1),
        json!({"exact_image_within_cohort":true}),
    ] {
        let mut wire = enabled_document();
        wire["prompt_expansion"]["vision_reuse"] = invalid;
        assert!(load(root.path(), &wire).is_err());
    }
    for policy in ["off", "exact_image_within_cohort"] {
        let mut disabled = document();
        disabled["prompt_expansion"]["vision_reuse"] = json!(policy);
        assert!(load(root.path(), &disabled).is_err());
    }
    assert!(
        !root.path().join("weights").exists(),
        "configuration parsing must not open source weights"
    );
}

#[test]
fn prompt_expansion_mtp_is_opt_in_and_preserves_request_limits() {
    use ferrum_engine::image_generation::p_h3::PH3ImagePromptExpansionMtp as Policy;
    let root = tempfile::tempdir().unwrap();
    let base = enabled_document();
    let baseline = load(root.path(), &base).unwrap();
    let PH3ImagePromptExpansionConfig::Enabled(default) = baseline.prompt_expansion else {
        panic!("enabled fixture")
    };
    assert_eq!(default.mtp, Policy::Off);
    let mut wire = base;
    wire["prompt_expansion"]["mtp"] = json!("one_draft");
    let configured = load(root.path(), &wire).unwrap();
    let PH3ImagePromptExpansionConfig::Enabled(configured) = configured.prompt_expansion else {
        panic!("enabled fixture")
    };
    assert_eq!(configured.mtp, Policy::OneDraft);
    assert_eq!(configured.max_context_tokens, default.max_context_tokens);
    assert_eq!(configured.max_new_tokens, default.max_new_tokens);
    assert_eq!(
        configured.prefill_chunk_tokens,
        default.prefill_chunk_tokens
    );
    wire["prompt_expansion"]["mtp"] = json!("unknown");
    assert!(load(root.path(), &wire).is_err());
}

#[test]
fn prompt_expansion_requires_its_explicit_processor_and_rejects_invalid_vision_limits() {
    let root = tempfile::tempdir().unwrap();
    let mut wire = enabled_document();
    wire["prompt_expansion"]
        .as_object_mut()
        .unwrap()
        .remove("processor_config");
    assert!(format!("{:#}", load(root.path(), &wire).err().unwrap()).contains("processor_config"));
    for limits in [
        json!({"maximum_encoded_bytes":0}),
        json!({"maximum_pixels":0}),
        json!({"maximum_width":0}),
        json!({"max_patch_rows":0}),
        json!({"max_patch_bytes":0}),
        json!({"max_plan_nodes":0}),
        json!({"max_stored_bytes":0}),
        json!({"hidden_resize_policy":true}),
    ] {
        let mut wire = enabled_document();
        wire["prompt_expansion"]["vision_limits"] = limits;
        assert!(load(root.path(), &wire).is_err());
    }
    let mut wire = enabled_document();
    wire["prompt_expansion"]["metadata_limits"] = json!({"max_processor_bytes":0});
    assert!(load(root.path(), &wire).is_err());
}

#[test]
fn original_processor_load_is_bounded_and_keeps_its_declared_source_identity() {
    use sha2::{Digest, Sha256};
    let root = tempfile::tempdir().unwrap();
    let mut config = load(root.path(), &enabled_document()).unwrap();
    let PH3ImagePromptExpansionConfig::Enabled(expansion) = &mut config.prompt_expansion else {
        unreachable!()
    };
    let source = serde_json::to_vec(&processor_document()).unwrap();
    std::fs::create_dir_all(expansion.processor_config.parent().unwrap()).unwrap();
    std::fs::write(&expansion.processor_config, &source).unwrap();
    expansion.metadata_limits.max_processor_bytes = source.len();
    let (processor, family) = expansion.prepare_processor().unwrap();
    assert_eq!(
        processor.source_sha256(),
        format!("{:x}", Sha256::digest(&source))
    );
    assert_eq!(processor.pixel_area_bounds(), (16, 256));
    assert_eq!(
        processor.image_limits().maximum_pixels(),
        expansion.vision_limits.maximum_pixels
    );
    assert_eq!(
        family.max_stored_bytes,
        expansion.vision_limits.max_stored_bytes
    );
    expansion.metadata_limits.max_processor_bytes -= 1;
    assert!(expansion.prepare_processor().is_err());
    expansion.metadata_limits.max_processor_bytes = source.len();
    std::fs::remove_file(&expansion.processor_config).unwrap();
    assert!(format!("{:#}", expansion.prepare_processor().err().unwrap()).contains("processor"));
    // Neither source weights nor the native backend is opened by this query.
    assert!(!expansion.bf16_archive.exists());
}

#[test]
fn original_processor_algorithm_is_validated_before_any_archive_or_runtime() {
    let root = tempfile::tempdir().unwrap();
    let config = load(root.path(), &enabled_document()).unwrap();
    let PH3ImagePromptExpansionConfig::Enabled(expansion) = &config.prompt_expansion else {
        unreachable!()
    };
    std::fs::create_dir_all(expansion.processor_config.parent().unwrap()).unwrap();
    let mut processor = processor_document();
    processor["image_processor"]["resample"] = json!(2);
    std::fs::write(
        &expansion.processor_config,
        serde_json::to_vec(&processor).unwrap(),
    )
    .unwrap();
    assert!(format!("{:#}", expansion.prepare_processor().err().unwrap()).contains("bicubic"));
    assert!(!expansion.bf16_archive.exists());
}

#[test]
fn prompt_expansion_defaults_enabled_and_missing_explicit_sources_fail_before_loading() {
    let root = tempfile::tempdir().unwrap();
    let mut omitted = document();
    omitted.as_object_mut().unwrap().remove("prompt_expansion");
    let typed: PH3ImageProductConfig = serde_json::from_value(omitted.clone()).unwrap();
    assert!(typed.prompt_expansion.is_enabled());
    let error = format!("{:#}", load(root.path(), &omitted).err().unwrap());
    assert!(error.contains("explicit semantic_directory"), "{error}");
    let disabled = load(root.path(), &document()).unwrap();
    assert!(!disabled.prompt_expansion.is_enabled());
    assert!(disabled
        .prompt_expansion
        .prepare_sources(0)
        .unwrap()
        .is_none());
}

#[test]
fn prompt_expansion_resolves_exact_sources_typed_defaults_and_one_causal_lane() {
    let root = tempfile::tempdir().unwrap();
    let base = root.path().canonicalize().unwrap();
    let mut wire = enabled_document();
    wire["backend"]["ordinal"] = json!(2);
    let config = load(root.path(), &wire).unwrap();
    let PH3ImagePromptExpansionConfig::Enabled(expansion) = &config.prompt_expansion else {
        panic!("explicitly enabled expansion");
    };
    assert_eq!(
        expansion.semantic_directory,
        base.join("upstream/prompt_upsampler")
    );
    assert_eq!(
        expansion.tokenizer_directory,
        base.join("upstream/prompt_upsampler_processor")
    );
    assert_eq!(
        expansion.bf16_archive,
        base.join("weights/official-bf16.safetensors")
    );
    assert_eq!(
        expansion.processor_config,
        base.join("upstream/prompt_upsampler_processor/processor_config.json")
    );
    let options = expansion.options(2).unwrap();
    assert_eq!(options.engine.backend.device, ferrum_types::Device::CUDA(2));
    assert_eq!(options.engine.runtime.max_model_len, Some(8192));
    assert_eq!(options.max_new_tokens, 1024);
    assert_eq!(options.prefill_chunk_tokens, 256);
    assert!(!options.engine.backend.enable_reusable_execution);
    wire["backend"]["maximum_streams"] = json!(2);
    assert!(load(root.path(), &wire).is_err());
    wire["backend"]["maximum_streams"] = json!(3);
    assert!(load(root.path(), &wire).is_ok());
    let error = format!("{:#}", config.prepare_assets().err().unwrap());
    assert!(
        error.contains("original prompt expansion metadata"),
        "{error}"
    );
}

#[test]
fn prompt_expansion_requires_matching_explicit_module_budget_and_positive_context_bounds() {
    let root = tempfile::tempdir().unwrap();
    let mut wire = enabled_document();
    wire["backend"]
        .as_object_mut()
        .unwrap()
        .remove("prompt_expansion_causal");
    assert!(load(root.path(), &wire).is_err());
    let mut wire = enabled_document();
    wire["prompt_expansion"] = json!({"mode":"disabled"});
    assert!(load(root.path(), &wire).is_err());
    for patch in [
        json!({"max_new_tokens":0}),
        json!({"max_context_tokens":1024}),
        json!({"prefill_chunk_tokens":0}),
        json!({"prefill_chunk_tokens":8193}),
        json!({"metadata_limits":{"max_tokenizer_bytes":0}}),
    ] {
        let mut wire = enabled_document();
        wire["prompt_expansion"]
            .as_object_mut()
            .unwrap()
            .extend(patch.as_object().unwrap().clone());
        assert!(load(root.path(), &wire).is_err());
    }
    for bytes in [0, u64::MAX] {
        let mut wire = enabled_document();
        wire["backend"]["prompt_expansion_causal"]["module_memory_bytes"] = json!(bytes);
        assert!(load(root.path(), &wire).is_err());
    }
    let mut wire = enabled_document();
    wire["prompt_expansion"]["max_context_tokens"] = json!(4096);
    wire["prompt_expansion"]["max_new_tokens"] = json!(512);
    wire["prompt_expansion"]["prefill_chunk_tokens"] = json!(128);
    let config = load(root.path(), &wire).unwrap();
    let PH3ImagePromptExpansionConfig::Enabled(expansion) = config.prompt_expansion else {
        unreachable!()
    };
    let options = expansion.options(0).unwrap();
    assert_eq!(options.engine.runtime.max_model_len, Some(4096));
    assert_eq!(options.max_new_tokens, 512);
    assert_eq!(options.prefill_chunk_tokens, 128);
}

#[test]
fn bounded_decode_replay_wire_is_explicit_and_shared_by_expansion_options() {
    use ferrum_types::{ReusableExecutionCaptureScope, ReusableExecutionPreparationMode};
    let config: PH3ImageCausalModuleConfig = serde_json::from_value(json!({
        "module_memory_bytes": 1024, "opaque_memory_profile": "fixture-causal-v1"
    }))
    .unwrap();
    config.validate().unwrap();
    assert!(config.decode_replay.is_none());
    let mut options = prompt_expansion::PH3ImagePromptExpansionEnabledConfig::default()
        .options(0)
        .unwrap();
    config.configure_expansion(&mut options);
    assert!(!options.engine.backend.enable_reusable_execution);
    assert_eq!(
        options
            .engine
            .backend
            .reusable_execution_capture
            .maximum_executables_per_lane,
        None
    );
    let configured: PH3ImageCausalModuleConfig = serde_json::from_value(json!({
        "module_memory_bytes": 1024, "opaque_memory_profile": "fixture-causal-v1",
        "decode_replay": {"maximum_executables": 32, "opaque_memory_bytes_per_stream": 4096}
    }))
    .unwrap();
    configured.validate().unwrap();
    configured.configure_expansion(&mut options);
    assert!(options.engine.backend.enable_reusable_execution);
    assert_eq!(
        options
            .engine
            .backend
            .reusable_execution_capture
            .maximum_executables_per_lane,
        Some(32)
    );
    assert_eq!(
        options.engine.backend.reusable_execution_capture.scope,
        ReusableExecutionCaptureScope::DecodeOnly
    );
    assert_eq!(
        options
            .engine
            .backend
            .reusable_execution_capture
            .preparation,
        ReusableExecutionPreparationMode::OnDemand
    );
    for limits in [
        json!({"maximum_executables":0,"opaque_memory_bytes_per_stream":4096}),
        json!({"maximum_executables":32,"opaque_memory_bytes_per_stream":0}),
        json!({"maximum_executables":32,"opaque_memory_bytes_per_stream":u64::MAX}),
    ] {
        let invalid: PH3ImageCausalModuleConfig = serde_json::from_value(json!({
            "module_memory_bytes":1024,"opaque_memory_profile":"fixture-causal-v1","decode_replay":limits
        })).unwrap();
        assert!(invalid.validate().is_err());
    }
}

#[test]
fn bounded_decode_replay_charges_each_original_stream_before_tensor_capacity() {
    let directory = tempfile::tempdir().unwrap();
    let mut wire = enabled_document();
    let config = load(directory.path(), &wire).unwrap();
    let PH3ImageBackendConfig::Cuda(backend) = &config.backend;
    let limits = backend.pipeline.resolved_limits().unwrap();
    let streams = backend
        .maximum_streams
        .resolve(
            ImageStreamRequirements::for_serial_pipeline(
                limits.static_initialization,
                limits.executor,
            )
            .with_serial_causal_phase(),
        )
        .unwrap()
        .resolved_streams() as u64;
    let base = backend.bootstrap_memory_bytes
        + streams * backend.stream_memory_bytes
        + backend.cudnn.opaque_memory_bytes
        + backend
            .prompt_expansion_causal
            .as_ref()
            .unwrap()
            .module_memory_bytes;
    let graph_bytes = 16 << 20;
    let floor = base + graph_bytes * streams + backend.cudnn.workspace_limit_bytes;
    wire["backend"]["usable_memory_bytes"] = json!(floor);
    // Old eager fits this exact policy; enabling graphs must account for the
    // whole original stream inventory, not merely one current request lane.
    load(directory.path(), &wire).unwrap();
    wire["backend"]["prompt_expansion_causal"]["decode_replay"] = json!({
        "maximum_executables": 32, "opaque_memory_bytes_per_stream": graph_bytes,
    });
    assert!(load(directory.path(), &wire).is_err());
    wire["backend"]["usable_memory_bytes"] = json!(floor + 1);
    load(directory.path(), &wire).unwrap();
    wire["backend"]["prompt_expansion_causal"]["decode_replay"]["opaque_memory_bytes_per_stream"] =
        json!(isize::MAX as u64);
    assert!(
        load(directory.path(), &wire).is_err(),
        "summed stream inventory overflow must reject before native creation"
    );
}

#[test]
fn h3_weight_arena_target_reaches_shared_pipeline_limits_for_run_and_serve() {
    let root = tempfile::tempdir().unwrap();
    for serving in [false, true] {
        let mut wire = document();
        if serving {
            wire["serving"] = json!({"maximum_pending":2,"maximum_prompt_bytes":4096,
                "max_num_seqs":1,"request_batch_max_wait_ms":0,"default_width":512,
                "default_height":512,"default_seed":0,"rgba_scratch_bytes":16384,"output":{}});
        }
        let PH3ImageBackendConfig::Cuda(default) = load(root.path(), &wire).unwrap().backend;
        assert_eq!(
            default
                .pipeline
                .resolved_limits()
                .unwrap()
                .weight_arena_target_bytes,
            None
        );
        wire["backend"]["pipeline"]["weight_arena_target_bytes"] = json!(256_u64 << 20);
        let parsed = load(root.path(), &wire).unwrap();
        let PH3ImageBackendConfig::Cuda(backend) = parsed.backend;
        let limits = backend.pipeline.resolved_limits().unwrap();
        assert_eq!(
            limits.weight_arena_target_bytes,
            std::num::NonZeroU64::new(256 << 20)
        );
        assert_eq!(
            limits
                .with_device_capacity(
                    std::num::NonZeroU64::new(backend.usable_memory_bytes).unwrap()
                )
                .weight_arena_target_bytes,
            std::num::NonZeroU64::new(256 << 20)
        );
        assert!(!root.path().join("comfy").exists());
        wire["backend"]["pipeline"]["weight_arena_target_bytes"] = json!(0);
        assert!(load(root.path(), &wire).is_err());
    }
}
