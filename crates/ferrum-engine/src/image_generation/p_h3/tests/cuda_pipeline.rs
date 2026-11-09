//! Complete tiny original-format checkpoints through the product CUDA factory
//! and the actual shared run/service consumers. Prompt expansion is explicitly
//! Disabled in this fixture; this does not qualify Enabled expansion, released
//! checkpoint quality, mixed-progress throughput, or global free VRAM.
//! Some tiny TE projections are zero: execution and cross-entrypoint agreement
//! do not establish complete TE numerical accuracy or prompt sensitivity.
//! The explicit SDPA target also checks a real B2 against independent B1 runs
//! and declares a fixture-local F32 pixel tolerance against sequential attention.
//!
//! Run this ignored target alone with one test thread. Like the existing VAE
//! CUDA tests, it needs an explicit trusted absolute cuDNN9 library path in
//! FERRUM_TEST_CUDNN_LIBRARY_PATH (and optional split-path dependencies in
//! FERRUM_TEST_CUDNN_DEPENDENCY_LIBRARIES), plus the ordinary pinned native set.
use super::fixture::CompleteFixture;
use crate::image_generation::{
    continuous_trace::{ImageContinuousTraceHandle, ImageContinuousTraceLimits},
    cuda::ImageCudaMemoryPolicy,
    p_h3::{cuda::PH3ImageCudaConfig, PH3ImageInitialNoisePolicy, PH3ImagePipelineOptions},
    service::{ImageGenerator, ImageService, ImageServiceConfig},
    ImageCudaAttentionPolicy, ImageCudaConvolutionWorkspacePolicy, ImageDecoderBatchPolicy,
    ImageDynamicResidencyPolicy, ImageEditPrefixPolicy, ImageEditRequest, ImageGenerationOutput,
    ImageGenerationRequest, ImagePipelineLimits, ImagePlanCachePolicy, ImageResidencyPolicy,
    ImageStage, ImageStageRetirement, ImageStreamLimit,
};
use anyhow::{Context, Result};
use ferrum_execution::TensorExecutorConfig;
use ferrum_kernels::backend::cuda::{
    vnext_ops::CUDA_PHILOX_NORMAL_F32_NUMERICAL_PROFILE_ID, CudaCudnnConfig,
};
use ferrum_models::{image_pixels::ImagePixelEncoding, p_h3_image_vae::PH3ImageVaeSpatialPolicy};
use ferrum_output::png::{write_rgba8_png, PngDimensions, PngLimits};
use ferrum_runtime::{
    DeviceRuntime, HostMemoryAccount, HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits,
    NumericalProfileId, StaticInitializationPolicy,
};
use ferrum_types::ImageBatchPolicy;
use serde_json::{json, Value};
use std::{num::NonZeroU64, path::PathBuf, sync::Arc};

mod tile_batch;

pub(in crate::image_generation::p_h3) fn config() -> Result<PH3ImageCudaConfig> {
    let mut options = PH3ImagePipelineOptions::comfy_f32(
        NumericalProfileId::new(CUDA_PHILOX_NORMAL_F32_NUMERICAL_PROFILE_ID)
            .map_err(anyhow::Error::msg)?,
        1,
    )?;
    // The tiny fixture has fewer than sixteen audio elements. Use its declared
    // native Philox profile; the product CPU MT19937 route rejects that shape.
    options.initial_noise = PH3ImageInitialNoisePolicy::NativeCudaPhiloxJointImageAudioV1;
    options.vae_spatial_policy = PH3ImageVaeSpatialPolicy::WholeImage;
    Ok(PH3ImageCudaConfig {
        attention_policy: crate::image_generation::ImageCudaAttentionPolicy::Sequential,
        ordinal: 0,
        memory: ImageCudaMemoryPolicy::ExplicitTotal(NonZeroU64::new(1 << 30).unwrap()),
        bootstrap_memory_bytes: 256 << 20,
        stream_memory_bytes: 16 << 20,
        maximum_streams: ImageStreamLimit::Auto,
        native_lane_reuse: ferrum_kernels::backend::cuda::vnext_runtime::CudaNativeLaneReuse::Off,
        native_allocation_reuse:
            ferrum_kernels::backend::cuda::vnext_runtime::CudaNativeAllocationReuse::Off,
        maximum_in_flight_per_stream: 1,
        maximum_commands_per_submission: 4096,
        opaque_memory_profile: "unqualified-tiny-h3-complete-cuda-v1".into(),
        // Explicit Disabled route: no source, causal provider or second runtime.
        prompt_expansion_causal: None,
        cudnn: CudaCudnnConfig {
            library_path: PathBuf::from(
                std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY_PATH")
                    .context("explicit trusted absolute cuDNN9 test library path required")?,
            ),
            dependency_libraries: std::env::var_os("FERRUM_TEST_CUDNN_DEPENDENCY_LIBRARIES")
                .map(|paths| std::env::split_paths(&paths).collect())
                .unwrap_or_default(),
            opaque_memory_bytes: 64 << 20,
            workspace_limit_bytes: 16 << 20,
            opaque_memory_profile: "unqualified-tiny-h3-cudnn9-fma-v1".into(),
        },
        convolution_workspace: ImageCudaConvolutionWorkspacePolicy::ConfiguredLimit,
        host: HostMemoryLimits::new(
            1 << 30,
            HostMemoryClassLimits {
                input_preprocess_bytes: 512 << 20,
                output_bytes: 256 << 20,
                transfer_staging_bytes: 64 << 20,
                weight_backing_bytes: 128 << 20,
            },
        )?,
        pipeline: ImagePipelineLimits {
            maximum_user_tokens: 512,
            maximum_width: 32,
            maximum_height: 32,
            maximum_batch_size: 1,
            weight_arena_target_bytes: None,
            decoder_batch_policy: ImageDecoderBatchPolicy::SingleImage,
            edit_prefix_policy: ImageEditPrefixPolicy::Recompute,
            plan_cache: ImagePlanCachePolicy::LastPerStage,
            residency: ImageResidencyPolicy::DenoiserAndDecoder,
            dynamic_residency: ImageDynamicResidencyPolicy::Release,
            static_initialization: StaticInitializationPolicy::new(1 << 20, 64)?,
            executor: TensorExecutorConfig::default(),
        },
        options,
    })
}

pub(super) fn request(seed: u64) -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "cat".into(),
        width: 32,
        height: 32,
        seed,
        n: 1,
    }
}

pub(super) fn edit(host: &HostMemoryAccount, seed: u64) -> Result<ImageEditRequest> {
    edit_with_pattern(host, seed, false)
}

fn edit_with_pattern(
    host: &HostMemoryAccount,
    seed: u64,
    alternate: bool,
) -> Result<ImageEditRequest> {
    // Codec/test sink memory is separate from the admitted immutable ingress
    // owner. The actual controller strictly decodes these bytes after admission.
    let mut encoded = Vec::new();
    write_rgba8_png(
        &mut encoded,
        PngDimensions::new(32, 32)?,
        PngLimits::default(),
        |writer| {
            for y in 0..32u8 {
                for x in 0..32u8 {
                    let pixel = if alternate {
                        [255 - y * 7, 255 - x * 7, (x | y) * 7, 255]
                    } else {
                        [x * 7, y * 7, (x ^ y) * 7, 255]
                    };
                    writer.write_all(&pixel)?;
                }
            }
            Ok(())
        },
    )?;
    let mut image = host.allocate_zeroed(HostMemoryClass::InputPreprocess, encoded.len())?;
    image.as_mut().copy_from_slice(&encoded);
    Ok(ImageEditRequest {
        references: Vec::new(),
        generation: request(seed),
        image: image.freeze(),
    })
}

pub(super) fn assert_output(
    output: &ImageGenerationOutput,
    host: &HostMemoryAccount,
    seed: u64,
    runtime_fingerprint: &str,
    editing: bool,
) -> Result<Vec<u8>> {
    assert_output_for_batch(output, host, seed, runtime_fingerprint, editing, 1, 0)
}

fn assert_output_for_batch(
    output: &ImageGenerationOutput,
    host: &HostMemoryAccount,
    seed: u64,
    runtime_fingerprint: &str,
    editing: bool,
    batch_size: usize,
    batch_index: usize,
) -> Result<Vec<u8>> {
    assert!(host.owns(output.rgba.bytes()));
    assert_eq!(output.rgba.bytes().class(), HostMemoryClass::Output);
    assert_eq!(output.rgba.encoding(), ImagePixelEncoding::F32UnitRgbRne);
    assert_eq!(output.rgba.tensor().dimensions(), &[1, 3, 32, 32]);
    assert_eq!(output.rgba.bytes().len(), 3 * 32 * 32 * 4);
    let receipt = &output.receipt;
    assert_eq!(
        (receipt.width, receipt.height, receipt.seed),
        (32, 32, seed)
    );
    assert_eq!(
        (receipt.batch_size, receipt.batch_index),
        (batch_size, batch_index)
    );
    assert_eq!(receipt.runtime_fingerprint, runtime_fingerprint);
    assert_eq!(receipt.steps.len(), 2);
    for (index, step) in receipt.steps.iter().enumerate() {
        assert_eq!(step.index, index);
        assert!(step.sigma.is_finite() && step.next_sigma.is_finite());
        assert!(step.sigma > step.next_sigma);
    }
    assert_eq!(receipt.steps.last().unwrap().next_sigma, 0.0);
    for stage in [
        ImageStage::TextEncoder,
        ImageStage::TextConnector,
        ImageStage::TextRefiner,
        ImageStage::Noise,
        ImageStage::Denoiser,
        ImageStage::Decoder,
    ] {
        assert!(
            receipt.stages.iter().any(|item| {
                item.execution.plan.stage == stage && item.execution.completed_executions > 0
            }),
            "missing actual completed {stage:?} execution"
        );
    }
    for stage in [ImageStage::VisionEncoder, ImageStage::VaeEncoder] {
        assert_eq!(
            receipt.stages.iter().any(|item| {
                item.execution.plan.stage == stage && item.execution.completed_executions > 0
            }),
            editing,
            "reference stages must execute only for an edit"
        );
    }
    assert!(receipt.stages.iter().any(|stage| stage.close.is_some()));
    for stage in &receipt.stages {
        if let Some(close) = &stage.close {
            assert_eq!(
                close.evidence().plan_hash().as_str(),
                stage.execution.plan.plan_hash
            );
            assert_eq!(
                close.evidence().runtime_implementation_fingerprint(),
                runtime_fingerprint
            );
            assert!(matches!(stage.retirement, ImageStageRetirement::Closed));
        }
    }
    // This validates finite unit-domain RGB through the same pixel reader used
    // by run/serve PNG sinks, rather than comparing receipt metadata alone.
    let view = output.rgba.view(host)?;
    assert_eq!(view.rgba_dimensions(), [1, 4, 32, 32]);
    let mut rgba = Vec::with_capacity(view.frame_bytes());
    let mut scratch = [0u8; 127];
    view.write_frame_rgba8(0, &mut scratch, &mut rgba)?;
    assert_eq!(rgba.len(), 32 * 32 * 4);
    assert!(rgba.chunks_exact(4).all(|pixel| pixel[3] == 255));
    Ok(rgba)
}

pub(super) fn assert_complete_trace(
    trace: &ImageContinuousTraceHandle,
    context: &Value,
    outputs: &[(&ImageGenerationOutput, bool)],
    expansion_enabled: bool,
) -> Result<()> {
    let mut bytes = Vec::new();
    let status = trace.write_jsonl(&mut bytes)?;
    assert!(status.sealed && !status.truncated);
    assert_eq!(status.dropped, 0);
    let mut records = bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(serde_json::from_slice::<Value>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let footer = records.pop().expect("actual collector footer");
    let session = json!(status.session_uuid);
    assert_eq!(footer["session_uuid"], session);
    assert_eq!(footer["footer"]["sealed"], true);
    assert_eq!(footer["footer"]["truncated"], false);
    assert_eq!(footer["footer"]["recorded"], records.len());
    assert_eq!(footer["footer"]["recorded"], status.recorded);
    assert_eq!(footer["footer"]["backing_bytes"], status.backing_bytes);
    for (index, record) in records.iter().enumerate() {
        assert_eq!(record["session_uuid"], session);
        assert_eq!(record["sequence"], index + 1);
    }
    let events = records
        .iter()
        .map(|record| &record["event"])
        .collect::<Vec<_>>();
    assert_eq!(events[0]["kind"], "header");
    assert_eq!(&events[0]["context"], context);
    assert_eq!(events[0]["prompt_expansion"].is_null(), !expansion_enabled);
    assert_eq!(
        events
            .iter()
            .filter(|event| event["kind"] == "request_initialized")
            .count(),
        outputs.len()
    );
    let mut identities = Vec::new();
    for (output, editing) in outputs {
        let receipt = &output.receipt;
        assert_eq!(
            events[0]["device"]["runtime_implementation_fingerprint"],
            receipt.runtime_fingerprint
        );
        let initialized = events
            .iter()
            .enumerate()
            .filter(|(_, event)| {
                event["kind"] == "request_initialized" && event["job"] == receipt.job
            })
            .collect::<Vec<_>>();
        assert_eq!(initialized.len(), 1);
        let (initialized_at, initialized) = initialized[0];
        let request = &initialized["request"];
        assert!(!request.as_str().expect("typed request identity").is_empty());
        assert!(
            !identities.contains(request),
            "original request identities must remain distinct"
        );
        identities.push(request.clone());
        assert_eq!(initialized["seed"], receipt.seed);
        assert_eq!(initialized["editing"], *editing);
        assert_eq!(initialized["width"], receipt.width);
        assert_eq!(initialized["height"], receipt.height);
        let committed = events
            .iter()
            .enumerate()
            .filter_map(|(position, event)| {
                if event["kind"] != "packed_step_committed" {
                    return None;
                }
                event["members"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|member| &member["request"] == request)
                    .map(|member| (position, *event, member))
            })
            .collect::<Vec<_>>();
        assert_eq!(committed.len(), receipt.steps.len());
        for ((position, event, member), step) in committed.iter().zip(&receipt.steps) {
            assert!(*position > initialized_at);
            assert_eq!(member["step_index"], step.index);
            for (field, actual) in [
                ("sigma", step.sigma),
                ("next_sigma", step.next_sigma),
                ("scheduler_timestep", step.scheduler_timestep),
            ] {
                assert_eq!(
                    (member[field].as_f64().unwrap() as f32).to_bits(),
                    actual.to_bits()
                );
            }
            let members = event["members"].as_array().unwrap();
            assert_eq!(event["packed_request_count"], members.len());
            assert_eq!(members.len(), receipt.batch_size);
            assert!(
                member["packed_row_end"].as_u64().unwrap()
                    > member["packed_row_start"].as_u64().unwrap()
            );
            assert!(!member["row_runs"].as_array().unwrap().is_empty());
            assert!(
                receipt.stages.iter().any(|stage| {
                    stage.execution.plan.stage == ImageStage::Denoiser
                        && stage.execution.completed_executions > 0
                        && event["plan_hash"] == stage.execution.plan.plan_hash
                }),
                "committed trace must bind an actually executed receipt plan"
            );
            let published = events[..*position]
                .iter()
                .rev()
                .find(|candidate| {
                    candidate["kind"] == "cohort_published"
                        && candidate["cohort_job"] == event["cohort_job"]
                })
                .expect("actual admission publication precedes its packed forward");
            assert_eq!(published["plan_hash"], event["plan_hash"]);
            assert_eq!(
                published["program_fingerprint"],
                event["program_fingerprint"]
            );
            assert_eq!(
                published["ordered_members"],
                Value::Array(members.iter().map(|m| m["request"].clone()).collect())
            );
        }
        let exited = events
            .iter()
            .enumerate()
            .filter(|(_, event)| {
                event["kind"] == "request_exited_at_fence" && &event["request"] == request
            })
            .collect::<Vec<_>>();
        assert_eq!(exited.len(), 1);
        assert_eq!(exited[0].1["reason"], "completed");
        assert_eq!(exited[0].1["completed_steps"], receipt.steps.len());
        assert!(exited[0].0 > committed.last().unwrap().0);
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires CUDA, the pinned native operator set and explicit cuDNN9 paths; expansion Disabled only"]
async fn cuda_complete_h3_expansion_disabled_generate_edit_and_service_match() -> Result<()> {
    complete_disabled_run_service(ImageCudaAttentionPolicy::Sequential).await?;
    Ok(())
}

struct ComparedPixels {
    generation: Vec<f32>,
    editing: Vec<f32>,
}
// Small independent test diagnostics, like the PNG sink vectors below. These
// copies do not replace original managed outputs or their retirement checks.
fn f32_pixels(output: &ImageGenerationOutput) -> Vec<f32> {
    output
        .rgba
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn assert_attention_profiles(output: &ImageGenerationOutput, policy: ImageCudaAttentionPolicy) {
    let (refine, steady) = match policy {
        ImageCudaAttentionPolicy::Sequential => (
            "p_h3_image.dit_forward.cuda.refine_only.shared_complete_weights.v1",
            "p_h3_image.dit_forward.cuda.refined_text.comfy_mixed.positive_bf16_velocity.v1",
        ),
        ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1 => (
            "p_h3_image.dit_forward.cuda.refine_only.comfy_mixed.cudnn_sdpa.v1",
            "p_h3_image.dit_forward.cuda.refined_text.comfy_mixed.cudnn_sdpa.v1",
        ),
    };
    for (kind, expected) in [
        (ImageStage::TextRefiner, refine),
        (ImageStage::Denoiser, steady),
    ] {
        let stages: Vec<_> = output
            .receipt
            .stages
            .iter()
            .filter(|s| s.execution.plan.stage == kind && s.execution.completed_executions > 0)
            .collect();
        assert!(!stages.is_empty(), "actual {kind:?} execution required");
        for stage in stages {
            assert_eq!(
                stage.execution.plan.profile.as_str(),
                expected,
                "actual {kind:?} must bind the selected phase/profile"
            );
        }
    }
}
// Declared only for these tiny two-step F32 RGB outputs. cuDNN's intermediate
// rounding differs from the sequential F32-online attention profile; this is
// neither a BF16-ULP nor a Torch/full-model error guarantee. Keep the measured
// maximum/RMSE visible on failure instead of silently accepting bit equality.
fn assert_fixture_attention_close(actual: &[f32], expected: &[f32], label: &str) {
    const MAX_ABSOLUTE: f64 = 1.0 / 64.0;
    const MAX_RMSE: f64 = 1.0 / 256.0;
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0_f64;
    let mut squared = 0.0_f64;
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(a.is_finite() && e.is_finite());
        let delta = f64::from(a) - f64::from(e);
        maximum = maximum.max(delta.abs());
        squared += delta * delta;
    }
    let rmse = (squared / actual.len() as f64).sqrt();
    assert!(
        maximum <= MAX_ABSOLUTE && rmse <= MAX_RMSE,
        "{label}: max_abs={maximum} (limit{MAX_ABSOLUTE}), RMSE={rmse} (limit{MAX_RMSE})"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires CUDA/cuDNN9.20 and pinned native set; explicit vendor SDPA full tiny H3 profile, B1/B2 and run/service"]
async fn cuda_complete_h3_explicit_segmented_sdpa_profiles_match_declared_fixture_tolerance(
) -> Result<()> {
    // Complete original retirement between the two real Runtime constructions.
    // No baseline pixels or provider implementation are substituted for CUDA.
    let sequential = complete_disabled_run_service(ImageCudaAttentionPolicy::Sequential).await?;
    let vendor =
        complete_disabled_run_service(ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1).await?;
    assert_fixture_attention_close(
        &vendor.generation,
        &sequential.generation,
        "SDPA versus sequential generation",
    );
    assert_fixture_attention_close(
        &vendor.editing,
        &sequential.editing,
        "SDPA versus sequential edit",
    );
    Ok(())
}

async fn complete_disabled_run_service(policy: ImageCudaAttentionPolicy) -> Result<ComparedPixels> {
    let fixture = CompleteFixture::new()?;
    let assets = fixture.assets()?;
    assert!(!assets.has_prompt_expansion());
    let mut configured = config()?;
    configured.attention_policy = policy;
    let vendor = policy == ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1;
    if vendor {
        configured.pipeline.maximum_batch_size = 2;
    }
    let (mut pipeline, composition) = configured.create_pipeline_with_receipt(assets)?;
    assert_eq!(composition.attention_policy, policy);
    assert_eq!(
        composition.attention_opaque_memory_bytes,
        policy.additional_opaque_bytes()
    );
    assert_eq!(composition.stream_plan.resolved_streams(), 2);
    assert!(composition
        .prompt_expansion_causal_module_memory_bytes
        .is_none());
    let runtime = Arc::downgrade(pipeline.runtime());
    let fingerprint = pipeline
        .runtime()
        .descriptor()
        .runtime_implementation_fingerprint
        .clone();
    let host = pipeline.host_account().clone();
    let service_config = ImageServiceConfig {
        batching: ImageBatchPolicy::Auto,
        maximum_pending: 2,
        maximum_pending_bytes: 1 << 20,
        maximum_prompt_bytes: 1024,
        max_num_seqs: 1,
        request_batch_max_wait_ms: 0,
    };
    let trace_context = json!({"entrypoint":"direct_then_service_test",
        "composition":&composition, "service_config":service_config});
    let trace = pipeline.enable_continuous_trace(
        ImageContinuousTraceLimits {
            maximum_bytes: 1 << 20,
        },
        &trace_context,
    )?;
    let reference = edit(&host, 29)?;
    let mut multiple_references = reference.clone();
    multiple_references
        .references
        .push(edit_with_pattern(&host, 29, true)?.image);
    assert_ne!(
        multiple_references.image.bytes(),
        multiple_references.references[0].bytes()
    );

    // These are the same public single-request entrypoints consumed by run.
    // Failures retain the original typed runtime/session owners, not a fake
    // successful close or a substituted test generator.
    let generated = pipeline
        .generate(request(17))
        .unwrap_or_else(|error| panic!("complete tiny CUDA generation failed: {error}"));
    let edited = pipeline
        .edit(reference.clone())
        .unwrap_or_else(|error| panic!("complete tiny CUDA edit failed: {error}"));
    let multiple_edited = pipeline
        .edit(multiple_references.clone())
        .unwrap_or_else(|error| panic!("complete tiny CUDA multi-reference edit failed: {error}"));
    let generated_rgba = assert_output(&generated, &host, 17, &fingerprint, false)?;
    let edited_rgba = assert_output(&edited, &host, 29, &fingerprint, true)?;
    let multiple_edited_rgba = assert_output(&multiple_edited, &host, 29, &fingerprint, true)?;
    assert!(
        multiple_edited
            .receipt
            .stages
            .iter()
            .filter(|stage| stage.execution.plan.stage == ImageStage::VaeEncoder)
            .map(|stage| stage.execution.completed_executions)
            .sum::<usize>()
            >= multiple_references.image_count()
    );
    assert_attention_profiles(&generated, policy);
    assert_attention_profiles(&edited, policy);
    assert_attention_profiles(&multiple_edited, policy);
    let compared = ComparedPixels {
        generation: f32_pixels(&generated),
        editing: f32_pixels(&edited),
    };
    let packed = if vendor {
        let independent = pipeline
            .generate(request(43))
            .unwrap_or_else(|error| panic!("SDPA independent second B1 failed: {error}"));
        assert_output(&independent, &host, 43, &fingerprint, false)?;
        assert_attention_profiles(&independent, policy);
        let outputs = pipeline
            .generate_batch(vec![request(17), request(43)])
            .unwrap_or_else(|error| panic!("SDPA actual B2 failed: {error}"));
        assert_eq!(outputs.len(), 2);
        for (index, seed) in [17, 43].into_iter().enumerate() {
            assert_output_for_batch(&outputs[index], &host, seed, &fingerprint, false, 2, index)?;
            assert_attention_profiles(&outputs[index], policy);
            let b1 = if index == 0 { &generated } else { &independent };
            assert_fixture_attention_close(
                &f32_pixels(&outputs[index]),
                &f32_pixels(b1),
                "SDPA packed lane versus its independent B1",
            );
        }
        Some((independent, outputs))
    } else {
        None
    };
    assert!(!pipeline.is_stopped());

    // Move the very same runtime/actor/Host/source installation into serving.
    // B1 service is deliberate: it checks entrypoint equivalence after the
    // vendor path's real B2 above. The separate trajectory target checks
    // changing membership; no HTTP concurrency is substituted for packed work.
    let service = ImageService::new(pipeline, service_config)?;
    assert!(service.host_account().same_parent(&host));
    assert!(service.supports_edits());
    assert_eq!(service.maximum_reference_images(), 8);
    let served = service.try_submit(request(17))?.wait().await?;
    let served_edit = service.try_submit_edit(reference.clone())?.wait().await?;
    let served_multiple_edit = service
        .try_submit_edit(multiple_references.clone())?
        .wait()
        .await?;
    assert_attention_profiles(&served, policy);
    assert_attention_profiles(&served_edit, policy);
    assert_attention_profiles(&served_multiple_edit, policy);
    assert_eq!(
        generated.rgba.bytes().bytes(),
        served.rgba.bytes().bytes(),
        "same-seed B1 generation must retain its actual F32 pixels across entrypoints"
    );
    assert_eq!(
        edited.rgba.bytes().bytes(),
        served_edit.rgba.bytes().bytes()
    );
    assert_eq!(
        generated_rgba,
        assert_output(&served, &host, 17, &fingerprint, false)?
    );
    assert_eq!(
        edited_rgba,
        assert_output(&served_edit, &host, 29, &fingerprint, true)?
    );
    assert_eq!(
        multiple_edited.rgba.bytes().bytes(),
        served_multiple_edit.rgba.bytes().bytes()
    );
    assert_eq!(
        multiple_edited_rgba,
        assert_output(&served_multiple_edit, &host, 29, &fingerprint, true)?
    );

    let retired = service.shutdown().await?;
    assert!(
        !retired.has_failure(),
        "original worker cleanup failed: {:?}",
        retired.failure()
    );
    let snapshot = service.snapshot();
    assert!(!snapshot.accepting);
    assert_eq!(snapshot.active_requests, 0);
    assert_eq!(snapshot.pending_requests, 0);
    assert_eq!(snapshot.preparing_requests, 0);
    assert_eq!(snapshot.executed_requests, 3);
    // Successful service retirement calls the real pipeline shutdown, which
    // rejects any unacknowledged session/snapshot close. Keep immutable canvas
    // owners alive across it to check their independent Output lifetime.
    drop(retired);
    drop(service);
    assert!(
        runtime.upgrade().is_none(),
        "original CUDA Runtime owners remain after checked shutdown"
    );
    assert!(host.usage()?.class_allocated_bytes(HostMemoryClass::Output) > 0);
    let mut traced_outputs = vec![
        (&generated, false),
        (&edited, true),
        (&served, false),
        (&served_edit, true),
        (&multiple_edited, true),
        (&served_multiple_edit, true),
    ];
    if let Some((independent, outputs)) = &packed {
        traced_outputs.push((independent, false));
        traced_outputs.extend(outputs.iter().map(|output| (output, false)));
    }
    assert_complete_trace(&trace, &trace_context, &traced_outputs, false)?;
    drop(traced_outputs);
    let trace_backing = trace.status().backing_bytes;
    drop((
        generated,
        edited,
        multiple_edited,
        served,
        served_edit,
        served_multiple_edit,
        reference,
        multiple_references,
        packed,
    ));
    assert_eq!(host.usage()?.allocated_bytes(), trace_backing);
    assert_eq!(host.usage()?.committed_bytes(), trace_backing);
    drop(trace);
    let usage = host.usage()?;
    assert_eq!(usage.allocated_bytes(), 0);
    assert_eq!(usage.committed_bytes(), 0);
    // The backend has no public live CUDA-byte counter. Checked close receipts,
    // Host refunds and final Runtime Arc retirement are not global VRAM=0.
    Ok(compared)
}
