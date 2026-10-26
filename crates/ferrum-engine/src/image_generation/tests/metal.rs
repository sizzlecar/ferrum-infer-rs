//! Actual Metal factory + shared tokenizer/noise/denoising/Euler/VAE pipeline.
//! Zero-weight typed assets exercise orchestration, not trained image quality.
use super::*;
use crate::image_generation::metal::{ImageMetalMemoryPolicy, QwenImage21MetalConfig};
use std::num::NonZeroU64;

mod cancellation;
mod prefix;

// Native factories share one physical device and its capacity owner. Tests that
// intentionally change its capacity must not register conflicting owners at once.
static METAL_FACTORY_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn metal_config() -> QwenImage21MetalConfig {
    QwenImage21MetalConfig {
        precision_policy: ImagePrecisionPolicy::Bf16,
        attention_policy: ImageMetalAttentionPolicy::Reference,
        memory: ImageMetalMemoryPolicy::AutoRecommended {
            utilization: 0.75,
            maximum_bytes: NonZeroU64::new(fixture::DEVICE_BYTES),
        },
        bootstrap_memory_bytes: 16 << 20,
        stream_memory_bytes: 1 << 20,
        maximum_streams: ImageStreamLimit::Auto,
        maximum_in_flight_per_stream: 1,
        maximum_commands_per_submission: 2048,
        opaque_memory_profile: "metal.image.fixture.declared.v1".into(),
        host: HostMemoryLimits::new(
            32 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16 << 20,
                output_bytes: 16 << 20,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        pipeline: ImagePipelineLimits {
            maximum_user_tokens: 512,
            maximum_width: 64,
            maximum_height: 64,
            maximum_batch_size: 2,
            decoder_batch_policy: ImageDecoderBatchPolicy::SingleImage,
            edit_prefix_policy: ImageEditPrefixPolicy::Recompute,
            plan_cache: ImagePlanCachePolicy::LastPerStage,
            residency: ImageResidencyPolicy::Cold,
            dynamic_residency: ImageDynamicResidencyPolicy::Release,
            static_initialization: StaticInitializationPolicy::new(1 << 20, 32).unwrap(),
            executor: TensorExecutorConfig::default(),
        },
    }
}
fn assets() -> (QwenImage21Assets, tempfile::TempDir) {
    let (reference, directory) = adapter::pipeline(2);
    (reference.assets, directory)
}

#[test]
fn metal_configuration_rejects_unavailable_modes_and_impossible_capacity() {
    let _guard = METAL_FACTORY_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let mut config = metal_config();
    config.validate().unwrap();
    config.precision_policy = ImagePrecisionPolicy::Fp8LinearExperimentalV1;
    assert!(config.validate().is_err());
    config.precision_policy = ImagePrecisionPolicy::Bf16;
    let required = config.stream_plan().unwrap().resolved_streams();
    config.maximum_streams = ImageStreamLimit::Explicit(required - 1);
    assert!(config.validate().is_err());
    config.maximum_streams = ImageStreamLimit::Auto;
    config.pipeline.edit_prefix_policy = ImageEditPrefixPolicy::RequestCache;
    config.validate().unwrap();
    config.pipeline.edit_prefix_policy = ImageEditPrefixPolicy::Recompute;
    let overhead = config.bootstrap_memory_bytes + config.stream_memory_bytes * required as u64;
    config.memory = ImageMetalMemoryPolicy::ExplicitTotal(NonZeroU64::new(overhead).unwrap());
    assert!(config.validate().is_err());
    config.memory = ImageMetalMemoryPolicy::ExplicitTotal(NonZeroU64::new(overhead + 1).unwrap());
    config.validate().unwrap();
    let (assets, _directory) = assets();
    assert!(config.create_pipeline(assets).is_err());
}

#[test]
fn metal_factory_generates_real_managed_batches_and_retires_cold_and_resident_stages() {
    let _guard = METAL_FACTORY_TEST.lock().unwrap_or_else(|e| e.into_inner());
    for (precision, residency) in [
        ImagePrecisionPolicy::Bf16,
        ImagePrecisionPolicy::Fp8LinearExperimentalV1,
    ]
    .into_iter()
    .flat_map(|precision| {
        [ImageResidencyPolicy::Cold, ImageResidencyPolicy::AllStages]
            .into_iter()
            .map(move |residency| (precision, residency))
    }) {
        let (assets, _directory) = assets();
        let mut config = metal_config();
        config.pipeline.residency = residency;
        config.precision_policy = precision;
        if precision.uses_fp8() {
            config.host = HostMemoryLimits::new(
                36 << 20,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 16 << 20,
                    output_bytes: 16 << 20,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 4 << 20,
                },
            )
            .unwrap();
        }
        let original_weights = assets.receipt().required_weight_bytes;
        let (mut pipeline, composition) = config.create_pipeline_with_receipt(assets).unwrap();
        assert_eq!(composition.precision_policy, precision);
        if precision.uses_fp8() {
            assert!(composition.execution_weight_bytes[0] < original_weights[0]);
            assert!(composition.execution_weight_bytes[1] < original_weights[1]);
            assert_eq!(composition.execution_weight_bytes[2], original_weights[2]);
        } else {
            assert_eq!(composition.execution_weight_bytes, original_weights);
        }
        let observation = composition.observed_memory.unwrap();
        assert_eq!(
            observation.effective_ceiling_bytes,
            composition.device_capacity_bytes
        );
        assert!(observation.effective_ceiling_bytes <= fixture::DEVICE_BYTES);
        assert_eq!(
            pipeline.runtime.tensor_capacity_bytes(),
            Some(composition.device_capacity_bytes)
        );
        assert!(!pipeline.supports_edits());
        let host = pipeline.host_account().clone();
        let request = |prompt: &str, seed| ImageGenerationRequest {
            prompt: prompt.into(),
            seed,
            width: 32,
            height: 32,
            n: 1,
        };
        let mut invalid = request("cat", 0);
        invalid.width = 33;
        assert!(pipeline.generate(invalid).is_err());
        assert!(!pipeline.is_stopped());
        let first = pipeline
            .generate_batch(vec![request("cat", 7), request("blue cat", 1 << 40)])
            .unwrap_or_else(|e| panic!("Metal batch: {e:?}"));
        let second = pipeline
            .generate_batch(vec![request("cat", 7), request("blue cat", 1 << 40)])
            .unwrap_or_else(|e| panic!("Metal repeat: {e:?}"));
        let first_memory = first[0].receipt.native_memory.unwrap();
        let second_memory = second[0].receipt.native_memory.unwrap();
        assert_eq!(
            first[0].receipt.native_memory,
            first[1].receipt.native_memory
        );
        assert_eq!(
            second[0].receipt.native_memory,
            second[1].receipt.native_memory
        );
        assert_eq!(first_memory.scope, DeviceNativeMemoryScope::DeviceHandle);
        // Cold stages have already retired when the receipt is produced. The
        // high-water observation must still include their native allocations,
        // and subsequent jobs must not reset a shared runtime's history.
        assert!(first_memory.observed_peak_allocated_bytes > 0);
        assert!(first_memory.observed_peak_allocated_bytes >= first_memory.allocated_bytes);
        assert!(
            second_memory.observed_peak_allocated_bytes
                >= first_memory.observed_peak_allocated_bytes
        );
        assert!(second_memory.samples > first_memory.samples);
        assert_eq!(first.len(), 2);
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(a.rgba.tensor().dimensions(), [1, 4, 32, 32]);
            assert_eq!(a.rgba.bytes().bytes(), b.rgba.bytes().bytes());
            assert_eq!(a.receipt.seed, b.receipt.seed);
            for stage in [
                ImageStage::TextEncoder,
                ImageStage::Noise,
                ImageStage::Denoiser,
                ImageStage::Decoder,
            ] {
                assert!(a
                    .receipt
                    .stages
                    .iter()
                    .any(|s| s.execution.plan.stage == stage));
            }
            assert!(a
                .rgba
                .bytes()
                .bytes()
                .chunks_exact(2)
                .all(
                    |b| half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap()))
                        .is_finite()
                ));
        }
        if residency == ImageResidencyPolicy::AllStages {
            assert!(second[0]
                .receipt
                .stages
                .iter()
                .all(|s| s.execution.session_reused));
        }
        drop(pipeline);
        assert!(host.usage().unwrap().committed_bytes() > 0); // Output owners remain valid.
        drop(first);
        drop(second);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        eprintln!("Metal full image pipeline {precision:?} {residency:?}: true B2 text/DiT, separate seeds, Euler, VAE, repeat and release passed");
    }
}

#[test]
fn metal_factory_checks_actual_graph_command_count_before_native_setup() {
    let (assets, _directory) = assets();
    let mut config = metal_config();
    // A count legal for static initialization but too small for the VAE graph.
    config.maximum_commands_per_submission = config
        .pipeline
        .static_initialization
        .maximum_commands_per_batch();
    let error = config.create_pipeline(assets).err().unwrap().to_string();
    assert!(error.contains("compute/upload commands"), "{error}");
}
