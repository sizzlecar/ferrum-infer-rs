//! Complete tiny Enabled pipeline on one original CUDA Runtime. The causal
//! source is BF16, the native causal profile computes F16, and its original HF
//! visual producer computes the explicitly selected CUDA F32 profile. No CPU
//! inference result, injected expansion text, or replacement provider is used.
//!
//! The deterministic language fixture has zero attention/MLP projections, and
//! some main-TE projections are zero. A pass checks real composition, EOS-gated
//! continuation and ownership; it does not qualify prompt/image sensitivity,
//! upstream numerical parity, released-model quality or concurrent throughput.
//! Run this ignored target alone with one test thread and the same explicit
//! trusted cuDNN/native-set setup documented by cuda_pipeline.
use super::{
    cuda_pipeline::{assert_complete_trace, assert_output, config, edit, request},
    fixture::CompleteFixture,
};
use crate::image_generation::{
    continuous_trace::ImageContinuousTraceLimits,
    p_h3::expansion::MergedExpansionFixture,
    service::{ImageService, ImageServiceConfig},
};
use anyhow::Result;
use ferrum_kernels::backend::cuda::CudaDenseF16CausalProfile;
use ferrum_models::{
    qwen3_vl_vision::hf::Qwen3VLVisionHfFamily,
    vnext::qwen35::F16_VISUAL_PROMPT_NUMERICAL_PROFILE_ID,
};
use ferrum_runtime::{
    DeviceRuntime, HostMemoryClass, CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_F16_OPERATION_ID,
    ROW_SELECT_F16_OPERATION_ID,
};
use ferrum_types::{Device, EngineConfig, ImageBatchPolicy};
use serde_json::json;
use std::sync::Arc;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires CUDA tensor+visual-causal union, pinned native set and explicit cuDNN9 paths; full tiny Enabled pipeline"]
async fn cuda_complete_h3_expansion_enabled_generate_edit_and_service_match() -> Result<()> {
    let fixture = CompleteFixture::new()?;
    let expansion = MergedExpansionFixture::new();
    let mut engine = EngineConfig::default();
    engine.backend.device = Device::CUDA(0);
    engine.runtime.max_model_len = Some(64);
    let options = MergedExpansionFixture::native_options(engine);
    let assets = fixture.assets_with_cuda_prompt_expansion(&expansion, options)?;
    assert!(assets.has_prompt_expansion());
    assert!(assets.has_prompt_expansion_visual());
    let mut configured = config()?;
    configured.prompt_expansion_causal = Some(
        CudaDenseF16CausalProfile::new(64 << 20, "unqualified-tiny-h3-visual-causal-modules-v1")
            .map_err(anyhow::Error::msg)?
            .with_visual_prompt_v1(),
    );
    let (mut pipeline, composition) = configured.create_pipeline_with_receipt(assets)?;
    assert_eq!(composition.stream_plan.resolved_streams(), 3);
    assert_eq!(
        composition.prompt_expansion_causal_module_memory_bytes,
        Some(64 << 20)
    );
    assert_eq!(
        composition.prompt_expansion_causal_mode,
        Some(F16_VISUAL_PROMPT_NUMERICAL_PROFILE_ID)
    );
    assert_eq!(
        composition.prompt_expansion_vision_profile.as_deref(),
        Some(Qwen3VLVisionHfFamily::CUDA_PROFILE_ID)
    );
    for operation in [
        ROW_SELECT_F16_OPERATION_ID,
        CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_F16_OPERATION_ID,
    ] {
        assert!(composition
            .operation_ids
            .iter()
            .any(|id| id.as_str() == operation));
    }
    let host = pipeline.host_account().clone();
    let runtime = Arc::downgrade(pipeline.runtime());
    let fingerprint = pipeline
        .runtime()
        .descriptor()
        .runtime_implementation_fingerprint
        .clone();
    let service_config = ImageServiceConfig {
        batching: ImageBatchPolicy::Auto,
        maximum_pending: 2,
        maximum_pending_bytes: 1 << 20,
        maximum_prompt_bytes: 1024,
        max_num_seqs: 1,
        request_batch_max_wait_ms: 0,
    };
    let trace_context = json!({"entrypoint":"enabled_direct_then_service_test",
        "composition":&composition, "service_config":service_config});
    let trace = pipeline.enable_continuous_trace(
        ImageContinuousTraceLimits {
            maximum_bytes: 1 << 20,
        },
        &trace_context,
    )?;
    let reference = edit(&host, 73)?;

    // Public run consumers: generate executes native causal expansion; edit
    // first executes native HF vision, binds its real F16 rows/explicit tables,
    // then executes the visual causal profile. Both may enter the main H3
    // preparation only after the source protocol accepts actual EOS and the
    // original causal executor has retired. Expanded text is never injected by
    // this test, and the original PNG owner is retained.
    let generated = pipeline
        .generate(request(61))
        .unwrap_or_else(|error| panic!("tiny Enabled CUDA generation failed: {error}"));
    let edited = pipeline
        .edit(reference.clone())
        .unwrap_or_else(|error| panic!("tiny Enabled CUDA visual edit failed: {error}"));
    let generated_rgba = assert_output(&generated, &host, 61, &fingerprint, false)?;
    let edited_rgba = assert_output(&edited, &host, 73, &fingerprint, true)?;
    assert!(!pipeline.is_stopped());

    // Serving owns the very same CUDA Runtime, registry, image actor and Host
    // parent. It advances the original native expansion state between its
    // normal image boundaries, rather than receiving pre-expanded mock text.
    let service = ImageService::new(pipeline, service_config)?;
    assert!(service.host_account().same_parent(&host));
    assert!(service.supports_edits());
    let served = service.try_submit(request(61))?.wait().await?;
    let served_edit = service.try_submit_edit(reference.clone())?.wait().await?;
    assert_eq!(generated.rgba.bytes().bytes(), served.rgba.bytes().bytes());
    assert_eq!(
        edited.rgba.bytes().bytes(),
        served_edit.rgba.bytes().bytes()
    );
    assert_eq!(
        generated_rgba,
        assert_output(&served, &host, 61, &fingerprint, false)?
    );
    assert_eq!(
        edited_rgba,
        assert_output(&served_edit, &host, 73, &fingerprint, true)?
    );

    let retired = service.shutdown().await?;
    assert!(
        !retired.has_failure(),
        "Enabled worker cleanup failed: {:?}",
        retired.failure()
    );
    let snapshot = service.snapshot();
    assert!(!snapshot.accepting);
    assert_eq!(snapshot.active_requests, 0);
    assert_eq!(snapshot.pending_requests, 0);
    assert_eq!(snapshot.preparing_requests, 0);
    assert_eq!(snapshot.executed_requests, 2);
    drop(retired);
    drop(service);
    assert!(
        runtime.upgrade().is_none(),
        "native visual/causal/image Runtime owners remain after checked shutdown"
    );
    // Caller-held immutable canvases remain genuine Output owners after every
    // original component and causal owner has acknowledged retirement.
    assert!(host.usage()?.class_allocated_bytes(HostMemoryClass::Output) > 0);
    assert_complete_trace(
        &trace,
        &trace_context,
        &[
            (&generated, false),
            (&edited, true),
            (&served, false),
            (&served_edit, true),
        ],
        true,
    )?;
    let trace_backing = trace.status().backing_bytes;
    drop((generated, edited, served, served_edit, reference));
    assert_eq!(host.usage()?.allocated_bytes(), trace_backing);
    assert_eq!(host.usage()?.committed_bytes(), trace_backing);
    drop(trace);
    assert_eq!(host.usage()?.allocated_bytes(), 0);
    assert_eq!(host.usage()?.committed_bytes(), 0);
    // No public live CUDA-byte counter exists: these acknowledgements do not
    // assert global free VRAM, nor release unrelated process/context caches.
    Ok(())
}
