//! Exercise the exact ActiveImageStage bridge with an existing actual Stored
//! Reference DiT. This tests receipts/ownership, not a fake Reference provider
//! for the CUDA-only quantized H3 text encoder.
use super::*;
use crate::image_generation::{
    denoising::fixture_pipeline, ImageDynamicResidencyPolicy, ImageResidencyPolicy,
    ImageStageAttachment, ImageStageDeclaration, ImageStagePlanRole, PreparedImageStage,
    QwenImage21Pipeline,
};
use ferrum_kernels::backend::reference::ReferenceDeviceRuntime;
use ferrum_models::qwen_image21::{
    QwenImage21T2iLayout, QwenImage21TransformerConfig, QwenImage21TransformerFamily,
};

#[test]
fn conditioning_connector_domain_preserves_provenance_and_rejects_changed_mathematics(
) -> anyhow::Result<()> {
    use crate::image_generation::p_h3::tests::fixture::CompleteFixture;
    use serde_json::json;
    use sha2::{Digest, Sha256};

    let fixture = CompleteFixture::new()?;
    let assets = fixture.assets()?;
    let prompts = assets.tokenizer.prepare(&["cat"])?;
    let config = PH3ImageTextConditioningConfig::from_comfy_checkpoint_with_connector(
        assets.text.clone(),
        prompts.layout().clone(),
        &assets.sources.text_encoder,
        &assets.connector,
    )?;
    let selected = config.connector_config()?;
    assert_ne!(
        selected, assets.connector,
        "the real metadata retains provenance absent from the reconstructed recipe"
    );
    validate_connector_domain(&selected, &assets.connector)?;
    assert_eq!(
        assets.receipt().metadata_sha256["text_connector"],
        format!("{:x}", Sha256::digest(&fixture.connector))
    );

    let original: serde_json::Value = serde_json::from_slice(&fixture.connector)?;
    let mut provenance = original.clone();
    provenance["_diffusers_version"] = json!("another-producer");
    let provenance = PH3ImageTextConnectorConfig::from_hf_json(&serde_json::to_vec(&provenance)?)?;
    validate_connector_domain(&selected, &provenance)?;
    for (field, changed) in [
        ("hidden_state_indices", json!([0, 2, 3, 4, 5])),
        ("text_hidden_size", json!(selected.text_hidden_size() + 1)),
        ("out_channels", json!(selected.out_channels() + 1)),
    ] {
        let mut wrong = original.clone();
        wrong[field] = changed;
        let wrong = PH3ImageTextConnectorConfig::from_hf_json(&serde_json::to_vec(&wrong)?)?;
        assert!(
            validate_connector_domain(&selected, &wrong).is_err(),
            "a changed {field} must still be rejected before any execution"
        );
    }
    Ok(())
}

fn prepare(
    pipeline: &QwenImage21Pipeline<ReferenceDeviceRuntime>,
) -> (
    PreparedImageStage<ReferenceDeviceRuntime>,
    QwenImage21TransformerConfig,
) {
    let config = QwenImage21TransformerConfig::new(
        pipeline.execution.sources.assets.transformer.clone(),
        QwenImage21T2iLayout::new(2, 2, 2, vec![2]).unwrap(),
    )
    .unwrap();
    let family = TypedFamilyRegistration::new(QwenImage21TransformerFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&config).unwrap(),
            &NumericalProfileId::new(QwenImage21TransformerFamily::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let stage = pipeline
        .execution
        .compile_stage(
            ImageStageDeclaration {
                stage: ImageStage::Denoiser,
                role: ImageStagePlanRole::WholeStage,
                batch_size: 1,
                weight_owner: pipeline
                    .execution
                    .sources
                    .weight_owner(ImageStage::Denoiser),
                attachment: ImageStageAttachment::MayInitializeStored,
            },
            family,
            &config.compile_options().unwrap(),
        )
        .unwrap();
    (stage, config)
}
fn inputs(
    host: &HostMemoryAccount,
    config: &QwenImage21TransformerConfig,
) -> Vec<ManagedHostTensor> {
    config
        .input_tensor_specs()
        .unwrap()
        .into_iter()
        .map(|(id, spec)| {
            let mut bytes = host
                .allocate_zeroed(
                    HostMemoryClass::InputPreprocess,
                    spec.byte_len().unwrap() as usize,
                )
                .unwrap();
            match spec.element_type {
                ElementType::Bf16 => {
                    for (i, b) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
                        b.copy_from_slice(
                            &half::bf16::from_f32((i % 7) as f32 * 0.125 - 0.25)
                                .to_bits()
                                .to_le_bytes(),
                        );
                    }
                }
                ElementType::F32 => {
                    for b in bytes.as_mut().chunks_exact_mut(4) {
                        b.copy_from_slice(&0.5_f32.to_le_bytes());
                    }
                }
                other => panic!("unexpected fixture input {other:?}"),
            }
            ManagedHostTensor::new(
                id,
                ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout).unwrap(),
                bytes.freeze(),
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn real_component_bridge_records_each_actual_attempt_and_keeps_returned_readback_owner() {
    let (mut pipeline, _directory) = fixture_pipeline();
    pipeline.execution.config.residency = ImageResidencyPolicy::Cold;
    pipeline.execution.config.dynamic_residency = ImageDynamicResidencyPolicy::Release;
    let (prepared, config) = prepare(&pipeline);
    let mut receipts = Vec::new();
    let active = pipeline
        .execution
        .start_stage(prepared, 71, &mut receipts)
        .unwrap_or_else(|e| panic!("start: {e}"));
    let work = inputs(pipeline.execution.host_account(), &config);
    let before = pipeline.execution.runtime.snapshot().commands;
    let (active, outputs) = execute_component(&mut pipeline.execution, active, 71, work)
        .unwrap_or_else(|e| panic!("execute: {e}"));
    assert!(pipeline.execution.runtime.snapshot().commands > before);
    assert_eq!(active.receipt.completed_executions, 1);
    assert_eq!(active.receipt.execution_wall_durations_us.len(), 1);
    assert_eq!(outputs.len(), 1);
    assert!(pipeline.execution.host.owns(outputs[0].bytes()));
    pipeline
        .execution
        .finish_stage(active, &mut receipts)
        .unwrap_or_else(|e| panic!("retire: {e}"));
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].execution.completed_executions, 1);
    assert_eq!(receipts[0].execution.execution_wall_durations_us.len(), 1);
    assert_eq!(pipeline.execution.runtime.snapshot().live_allocations, 0);
    assert!(!outputs[0].bytes().bytes().is_empty());
    drop(outputs);
    assert_eq!(
        pipeline.execution.host.usage().unwrap().committed_bytes(),
        0
    );
}

#[test]
fn borrowed_observer_error_and_panic_keep_the_original_active_stage_recoverable() {
    for (text, panics) in [(false, false), (false, true), (true, false), (true, true)] {
        let (mut pipeline, _directory) = fixture_pipeline();
        pipeline.execution.config.residency = ImageResidencyPolicy::Cold;
        pipeline.execution.config.dynamic_residency = ImageDynamicResidencyPolicy::Release;
        let (prepared, config) = prepare(&pipeline);
        let mut receipts = Vec::new();
        let active = pipeline
            .execution
            .start_stage(prepared, 71, &mut receipts)
            .unwrap_or_else(|e| panic!("start: {e}"));
        let work = inputs(pipeline.execution.host_account(), &config);
        let (active, outputs) = execute_component(&mut pipeline.execution, active, 71, work)
            .unwrap_or_else(|e| panic!("execute: {e}"));
        let source = if text {
            super::super::conditioning_diagnostics::tests::text_observer_failure(panics)
        } else {
            super::super::conditioning_diagnostics::tests::observer_failure(panics)
        };
        // Both main-text and main-vision hooks use this ownership transition.
        // The actual Reference DiT tests recovery, not CUDA vision correctness.
        let failure = active.processing(source);
        let active = match failure {
            ImageGenerationFailure::Processing {
                active: Some(active),
                ..
            } => active,
            _ => panic!("observer failure lost its active owner"),
        };
        assert_eq!(active.receipt.completed_executions, 1);
        drop(outputs);
        pipeline
            .execution
            .finish_stage(active, &mut receipts)
            .unwrap_or_else(|e| panic!("retire after observer failure: {e}"));
        assert_eq!(receipts.len(), 1);
        assert_eq!(pipeline.execution.runtime.snapshot().live_allocations, 0);
        assert_eq!(
            pipeline.execution.host.usage().unwrap().committed_bytes(),
            0
        );
    }
}

#[test]
fn component_failure_keeps_original_error_and_active_receipt_inside_pipeline_owner() {
    let (mut pipeline, _directory) = fixture_pipeline();
    pipeline.execution.config.residency = ImageResidencyPolicy::Cold;
    pipeline.execution.config.dynamic_residency = ImageDynamicResidencyPolicy::Release;
    let (prepared, config) = prepare(&pipeline);
    let mut receipts = Vec::new();
    let active = pipeline
        .execution
        .start_stage(prepared, 72, &mut receipts)
        .unwrap_or_else(|e| panic!("start: {e}"));
    let mut work = inputs(pipeline.execution.host_account(), &config);
    work.pop();
    let before = pipeline.execution.runtime.snapshot();
    let failure = match execute_component(&mut pipeline.execution, active, 72, work) {
        Err(failure) => pipeline_failure(failure),
        Ok(_) => panic!("missing declared input must fail"),
    };
    let ImageGenerationFailure::Pipeline {
        source,
        stages,
        weights,
        host_scopes,
        host_backing,
    } = failure
    else {
        panic!("expected the composed failure envelope")
    };
    assert!(
        stages.is_empty()
            && weights.is_empty()
            && host_scopes.is_empty()
            && host_backing.is_empty()
    );
    let ImageGenerationFailure::Execution { source, active } = *source else {
        panic!("original execution error and active session must survive")
    };
    assert_eq!(active.receipt.completed_executions, 0);
    assert_eq!(active.receipt.execution_wall_durations_us.len(), 1);
    assert_eq!(
        pipeline.execution.runtime.snapshot().submissions,
        before.submissions
    );
    assert_eq!(
        pipeline.execution.runtime.snapshot().live_allocations,
        before.live_allocations
    );
    drop(source);
    pipeline
        .execution
        .finish_stage(active, &mut receipts)
        .unwrap_or_else(|e| panic!("retire: {e}"));
    assert_eq!(receipts[0].execution.completed_executions, 0);
    assert_eq!(receipts[0].execution.execution_wall_durations_us.len(), 1);
    assert_eq!(pipeline.execution.runtime.snapshot().live_allocations, 0);
    assert_eq!(
        pipeline.execution.host.usage().unwrap().committed_bytes(),
        0
    );
}
