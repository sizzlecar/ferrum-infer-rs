//! Metadata/configuration tests: no CUDA context or native library is opened.
use super::*;
use crate::image_generation::{
    ImageDynamicResidencyPolicy, ImagePlanCachePolicy, ImageResidencyPolicy,
};
use ferrum_models::{
    p_h3_image::{
        PH3ImageTextConditioningConfig, PH3ImageTextConditioningFamily,
        PH3ImageTextConditioningLimits,
    },
    qwen3_vl_text::{
        Qwen3VLTextArchitecture, Qwen3VLTextComfyConfig, Qwen3VLTextComfyProjectionQuantization,
        Qwen3VLTextConfig, Qwen3VLTextKeyRun, Qwen3VLTextLayout,
    },
};

#[test]
fn native_allocation_reuse_reserves_one_fixed_budget_for_explicit_and_auto_limits() {
    let mut configured = config();
    let original = configured.opaque_bytes().unwrap();
    assert!(configured.native_allocation_reuse_receipt().is_none());
    configured.native_allocation_reuse = CudaNativeAllocationReuse::Bounded {
        maximum_idle_bytes: NonZeroU64::new(128 << 20).unwrap(),
        maximum_entry_bytes: NonZeroU64::new(64 << 20).unwrap(),
        maximum_entries: std::num::NonZeroUsize::new(2).unwrap(),
    };
    configured.validate().unwrap();
    assert_eq!(configured.opaque_bytes().unwrap(), original + (128 << 20));
    let receipt =
        serde_json::to_value(configured.native_allocation_reuse_receipt().unwrap()).unwrap();
    assert_eq!(
        receipt,
        serde_json::json!({"maximum_idle_bytes":128 << 20,
        "maximum_entry_bytes":64 << 20,"maximum_entries":2})
    );
    let floor = configured.opaque_bytes().unwrap() + configured.cudnn.workspace_limit_bytes;
    for bytes in [floor, floor + 1] {
        let maximum = NonZeroU64::new(bytes).unwrap();
        configured.memory = ImageCudaMemoryPolicy::ExplicitTotal(maximum);
        assert_eq!(configured.validate().is_ok(), bytes > floor);
        configured.memory = ImageCudaMemoryPolicy::AutoAvailable {
            utilization: ferrum_kernels::backend::cuda::CudaMemoryUtilization::new(0.9).unwrap(),
            maximum_bytes: Some(maximum),
        };
        assert_eq!(configured.validate().is_ok(), bytes > floor);
    }
    configured.native_allocation_reuse = CudaNativeAllocationReuse::Bounded {
        maximum_idle_bytes: NonZeroU64::new(u64::MAX).unwrap(),
        maximum_entry_bytes: NonZeroU64::new(1).unwrap(),
        maximum_entries: std::num::NonZeroUsize::new(1).unwrap(),
    };
    assert!(configured.opaque_bytes().is_err());
}

#[test]
fn decode_replay_core_and_backend_lane_caps_must_agree_before_cuda_setup() {
    use ferrum_types::{
        EngineConfig, ReusableExecutionCaptureScope, ReusableExecutionPreparationMode,
    };
    let mut engine = EngineConfig::default();
    engine.backend.enable_reusable_execution = false;
    let profile = CudaDenseF16CausalProfile::new(1024, "test-causal-cap-v1").unwrap();
    validate_decode_replay_limits(Some(&engine), Some(&profile)).unwrap();
    let profile = profile.with_decode_replay(64, 128 << 20).unwrap();
    assert!(validate_decode_replay_limits(Some(&engine), Some(&profile)).is_err());
    engine.backend.enable_reusable_execution = true;
    engine.backend.reusable_execution_capture.scope = ReusableExecutionCaptureScope::DecodeOnly;
    engine.backend.reusable_execution_capture.preparation =
        ReusableExecutionPreparationMode::OnDemand;
    for cap in [None, Some(0), Some(63), Some(65)] {
        engine
            .backend
            .reusable_execution_capture
            .maximum_executables_per_lane = cap;
        assert!(validate_decode_replay_limits(Some(&engine), Some(&profile)).is_err());
    }
    engine
        .backend
        .reusable_execution_capture
        .maximum_executables_per_lane = Some(64);
    validate_decode_replay_limits(Some(&engine), Some(&profile)).unwrap();
    assert_eq!(profile.decode_replay_limits(), Some((64, 128 << 20)));
    assert!(validate_decode_replay_limits(Some(&engine), None).is_err());
    engine.backend.reusable_execution_capture.preparation =
        ReusableExecutionPreparationMode::Startup;
    assert!(validate_decode_replay_limits(Some(&engine), Some(&profile)).is_err());
}

#[test]
fn native_lane_reuse_stays_within_the_original_stream_quota() {
    let mut configured = config();
    assert_eq!(configured.native_lane_reuse, CudaNativeLaneReuse::Off);
    let original_opaque = configured.opaque_bytes().unwrap();
    let streams = configured.stream_plan().unwrap().resolved_streams();
    for maximum in [1, streams] {
        configured.native_lane_reuse = CudaNativeLaneReuse::Bounded {
            maximum_idle_lanes: std::num::NonZeroUsize::new(maximum).unwrap(),
        };
        configured.validate().unwrap();
        assert_eq!(configured.opaque_bytes().unwrap(), original_opaque);
    }
    configured.native_lane_reuse = CudaNativeLaneReuse::Bounded {
        maximum_idle_lanes: std::num::NonZeroUsize::new(streams + 1).unwrap(),
    };
    assert!(configured.validate().is_err());
}

fn config() -> PH3ImageCudaConfig {
    PH3ImageCudaConfig {
        attention_policy: ImageCudaAttentionPolicy::Sequential,
        pinned_weight_upload: None,
        ordinal: 0,
        memory: ImageCudaMemoryPolicy::ExplicitTotal(NonZeroU64::new(4 << 30).unwrap()),
        bootstrap_memory_bytes: 256 << 20,
        stream_memory_bytes: 16 << 20,
        maximum_streams: ImageStreamLimit::Auto,
        native_lane_reuse: CudaNativeLaneReuse::Off,
        native_allocation_reuse: CudaNativeAllocationReuse::Off,
        maximum_in_flight_per_stream: 1,
        maximum_commands_per_submission: 4096,
        opaque_memory_profile: "declared-h3-runtime-v1".into(),
        prompt_expansion_causal: None,
        cudnn: CudaCudnnConfig {
            library_path: "/declared-native/libcudnn.so.9".into(),
            dependency_libraries: vec![],
            opaque_memory_bytes: 64 << 20,
            workspace_limit_bytes: 16 << 20,
            opaque_memory_profile: "declared-h3-cudnn-v1".into(),
        },
        convolution_workspace: ImageCudaConvolutionWorkspacePolicy::ConfiguredLimit,
        host: HostMemoryLimits::new(
            1 << 30,
            HostMemoryClassLimits {
                input_preprocess_bytes: 512 << 20,
                output_bytes: 256 << 20,
                transfer_staging_bytes: 16 << 20,
                weight_backing_bytes: 64 << 20,
            },
        )
        .unwrap(),
        pipeline: ImagePipelineLimits {
            maximum_user_tokens: 1024,
            maximum_width: 4096,
            maximum_height: 4096,
            maximum_batch_size: 1,
            weight_arena_target_bytes: None,
            decoder_batch_policy: ImageDecoderBatchPolicy::SingleImage,
            edit_prefix_policy: ImageEditPrefixPolicy::Recompute,
            plan_cache: ImagePlanCachePolicy::Disabled,
            residency: ImageResidencyPolicy::Cold,
            dynamic_residency: ImageDynamicResidencyPolicy::Release,
            static_initialization: StaticInitializationPolicy::new(16 << 20, 64).unwrap(),
            executor: Default::default(),
        },
        options: PH3ImagePipelineOptions::comfy_f32(
            profile(CUDA_PHILOX_NORMAL_F32_NUMERICAL_PROFILE_ID).unwrap(),
            4096,
        )
        .unwrap(),
    }
}
fn text_family() -> PreparedModelFamily {
    let architecture=Qwen3VLTextArchitecture::from_hf_value(&serde_json::json!({
        "architectures":["Qwen3VLForConditionalGeneration"],"model_type":"qwen3_vl",
        "text_config":{"model_type":"qwen3_vl_text","dtype":"bfloat16","attention_bias":false,"attention_dropout":0.0,
            "hidden_act":"silu","num_hidden_layers":5,"hidden_size":256,"intermediate_size":256,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":128,"vocab_size":16,
            "max_position_embeddings":128,"rms_norm_eps":0.25,"rope_theta":5000000,
            "rope_scaling":{"rope_type":"default","mrope_interleaved":true,"mrope_section":[64,0,0]}}})).unwrap();
    let text = Qwen3VLTextConfig::new(
        architecture,
        Qwen3VLTextLayout::new(1, vec![vec![Qwen3VLTextKeyRun::new(1, true)]]).unwrap(),
    )
    .unwrap()
    .with_hidden_state_indices(vec![1, 2, 3, 4, 5])
    .unwrap();
    let projections = (0..5)
        .flat_map(|layer| {
            [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ]
            .into_iter()
            .enumerate()
            .map(move |(i, suffix)| {
                (
                    format!("model.language_model.layers.{layer}.{suffix}"),
                    if i % 2 == 0 {
                        Qwen3VLTextComfyProjectionQuantization::W4A8
                    } else {
                        Qwen3VLTextComfyProjectionQuantization::W8A8
                    },
                )
            })
        })
        .collect();
    let text = PH3ImageTextConditioningConfig::new(
        Qwen3VLTextComfyConfig::new(text, projections).unwrap(),
        4,
    )
    .unwrap();
    TypedFamilyRegistration::new(
        PH3ImageTextConditioningFamily::new(PH3ImageTextConditioningLimits {
            max_metadata_bytes: 1 << 20,
            max_stored_bytes: 16 << 20,
            max_plan_nodes: 256,
        })
        .unwrap(),
    )
    .prepare_with_profile(
        &serde_json::to_value(text).unwrap(),
        &profile(PH3ImageTextConditioningFamily::CUDA_PROFILE_ID).unwrap(),
    )
    .unwrap()
}

#[test]
fn constructor_validation_checks_real_declared_capacity_without_a_driver() {
    let c = config();
    c.validate().unwrap();
    let mut small = c.clone();
    small.memory = ImageCudaMemoryPolicy::ExplicitTotal(NonZeroU64::new(1).unwrap());
    assert!(small.validate().is_err());
    let mut no_source = c.clone();
    no_source.host = HostMemoryLimits::new(
        1 << 30,
        HostMemoryClassLimits {
            input_preprocess_bytes: 512 << 20,
            output_bytes: 256 << 20,
            transfer_staging_bytes: 16 << 20,
            weight_backing_bytes: 0,
        },
    )
    .unwrap();
    assert!(no_source.validate().is_err());
    let mut wrong_noise = c.clone();
    wrong_noise.options.noise_profile = profile("different.native.noise").unwrap();
    assert!(wrong_noise.validate().is_err());
    let mut streams = c;
    streams.maximum_streams = ImageStreamLimit::Explicit(1);
    assert!(streams.validate().is_err());

    let mut reference_encoder = config();
    reference_encoder.options.vae_encoder =
        ferrum_models::p_h3_image_vae::PH3ImageVaeEncoderPreset::ReferenceF32;
    assert!(reference_encoder.validate().is_err());
}

#[test]
fn actual_text_program_operation_coverage_and_command_ceiling_are_verified_before_runtime() {
    let family = text_family();
    let mut contracts = operations::selected()
        .unwrap()
        .into_iter()
        .map(|(_, c)| c)
        .collect::<Vec<_>>();
    contracts.extend(operations::independent_contracts().unwrap());
    let commands = crate::image_generation::composition::component_command_count(&family).unwrap();
    assert!(preflight::validate(&[family.clone()], &contracts, commands).unwrap() > 0);
    assert!(preflight::validate(&[family.clone()], &contracts, commands - 1).is_err());
    contracts
        .retain(|c| c.descriptor().id.as_str() != COMFY_CONVROT_INT8_EMBEDDING_F32_OPERATION_ID);
    assert!(format!(
        "{:#}",
        preflight::validate(&[family], &contracts, commands)
            .err()
            .unwrap()
    )
    .contains(COMFY_CONVROT_INT8_EMBEDDING_F32_OPERATION_ID));
}

#[test]
fn causal_expansion_declares_additional_modules_and_three_stream_schedule() {
    let mut c = config();
    let baseline = c.opaque_bytes().unwrap();
    c.prompt_expansion_causal = Some(
        CudaDenseF16CausalProfile::new(256 << 20, "declared-unqualified-dense-f16-causal-v1")
            .unwrap(),
    );
    c.validate().unwrap();
    assert_eq!(c.stream_plan().unwrap().required_streams(), 3);
    assert_eq!(
        c.opaque_bytes().unwrap(),
        baseline + (256 << 20) + c.stream_memory_bytes
    );
    c.maximum_streams = ImageStreamLimit::Explicit(2);
    assert!(c.validate().is_err());
    c.maximum_streams = ImageStreamLimit::Explicit(3);
    assert!(c.validate().is_ok());
    let mut contracts = operations::selected()
        .unwrap()
        .into_iter()
        .map(|(_, c)| c)
        .collect::<Vec<_>>();
    contracts.extend(operations::independent_contracts().unwrap());
    contracts.extend(cuda_vnext_dense_f16_causal_contracts().unwrap());
    let ids = contracts
        .iter()
        .map(|c| c.descriptor().id.clone())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(ids.len(), contracts.len());
    for id in [
        TOKEN_EMBEDDING_OPERATION_ID,
        LAST_TOKEN_DENSE_LINEAR_OPERATION_ID,
        LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID,
        RMS_NORM_OPERATION_ID,
        DENSE_SWIGLU_OPERATION_ID,
        RESIDUAL_ADD_OPERATION_ID,
        GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
        CAUSAL_PAGED_ATTENTION_OPERATION_ID,
    ] {
        assert!(ids.iter().any(|found| found.as_str() == id));
    }
}
