//! Actual managed projections exercise policy decisions independently of image
//! dimensions, model size, backend allocation heuristics, or benchmark counts.
use super::*;
use ferrum_kernels::backend::reference::reference_vnext_operation_registry;
use ferrum_models::qwen_image21::QwenImage21TransformerPhase;
use std::sync::atomic::{AtomicU64, Ordering};

fn pipeline() -> (
    QwenImage21Pipeline<ReferenceDeviceRuntime>,
    tempfile::TempDir,
) {
    let (original, directory) = numeric_pipeline();
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let composition = composition(&format!(
        "device.image.adaptive.{}",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut pipeline = QwenImage21Pipeline::new(
        composition.runtime().clone(),
        original.host,
        reference_vnext_operation_registry(composition.runtime()).unwrap(),
        composition.catalog().clone(),
        WeightMaterializerRegistry::identity_only().unwrap(),
        Arc::try_unwrap(original.assets)
            .ok()
            .expect("unique fixture assets"),
        original.profiles,
        original.config,
    )
    .unwrap();
    Arc::get_mut(&mut pipeline.assets)
        .expect("unique fixture assets")
        .sources
        .text_encoder = pipeline.assets.sources.transformer.clone();
    Arc::get_mut(&mut pipeline.assets)
        .expect("unique fixture assets")
        .sources
        .vae = pipeline.assets.sources.transformer.clone();
    pipeline.config.residency = ImageResidencyPolicy::Adaptive;
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::Adaptive;
    (pipeline, directory)
}

fn stage(
    pipeline: &QwenImage21Pipeline<ReferenceDeviceRuntime>,
    kind: ImageStage,
    rows: u64,
) -> PreparedStage<ReferenceDeviceRuntime> {
    let mut stage = prepared(pipeline, rows);
    // The same genuine numerical graph isolates typed stage ownership.
    stage.plan.stage = kind;
    stage
}

fn demand(stage: &PreparedStage<ReferenceDeviceRuntime>) -> u64 {
    stage
        .component
        .executable()
        .execution_plan()
        .payload()
        .memory()
        .minimum_runnable_request_bytes()
}

fn execute(
    pipeline: &mut QwenImage21Pipeline<ReferenceDeviceRuntime>,
    kind: ImageStage,
    rows: u64,
    job: u64,
    receipts: &mut Vec<ImageStageReceipt>,
) -> ImageStageExecutionReceipt {
    let plan = stage(pipeline, kind, rows);
    pipeline.validate_resident_capacity(&[&plan]).unwrap();
    let active = pipeline
        .start_stage(plan, job, receipts)
        .unwrap_or_else(|e| panic!("{e}"));
    let inputs = vec![input(&pipeline.host, rows, job as usize)];
    let (active, outputs) = pipeline
        .execute_stage(active, job, inputs)
        .unwrap_or_else(|e| panic!("{e}"));
    let output = only_output(outputs, IMAGE_PROJECTION_OUTPUT_ID).unwrap();
    for (index, bytes) in output.bytes().bytes().chunks_exact(2).enumerate() {
        let row = index / 8;
        let column = index % 8;
        assert_eq!(
            u16::from_le_bytes([bytes[0], bytes[1]]),
            bf16::from_f32((job as usize + row * 64 + column + 1) as f32).to_bits()
        );
    }
    drop(output);
    let receipt = active.receipt.clone();
    pipeline
        .finish_stage(active, receipts)
        .unwrap_or_else(|e| panic!("{e}"));
    receipt
}

fn close(pipeline: &mut QwenImage21Pipeline<ReferenceDeviceRuntime>) {
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
    let capacity = DeviceCapacityOwner::open(pipeline.runtime.descriptor(), DEVICE_BYTES).unwrap();
    drop(
        capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, DEVICE_BYTES)
            .unwrap(),
    );
}

#[test]
fn adaptive_wire_is_explicit_and_fixed_policy_defaults_remain_unchanged() {
    assert_eq!(ImageResidencyPolicy::default(), ImageResidencyPolicy::Cold);
    assert_eq!(
        ImageDynamicResidencyPolicy::default(),
        ImageDynamicResidencyPolicy::Release
    );
    assert_eq!(
        serde_json::from_str::<ImageResidencyPolicy>("\"adaptive\"").unwrap(),
        ImageResidencyPolicy::Adaptive
    );
    assert_eq!(
        serde_json::from_str::<ImageDynamicResidencyPolicy>("\"adaptive\"").unwrap(),
        ImageDynamicResidencyPolicy::Adaptive
    );
    ImageDynamicResidencyPolicy::Adaptive
        .validate(ImageResidencyPolicy::Adaptive)
        .unwrap();
    assert!(ImageDynamicResidencyPolicy::Adaptive
        .validate(ImageResidencyPolicy::Cold)
        .is_err());
    assert!(ImageDynamicResidencyPolicy::Adaptive
        .validate(ImageResidencyPolicy::AllModelComponents)
        .is_err());
}

#[test]
fn allocation_pressure_reclaims_dynamic_then_lru_weights_and_stops_when_empty() {
    let (mut pipeline, _directory) = pipeline();
    let mut receipts = Vec::new();
    execute(&mut pipeline, ImageStage::Denoiser, 8, 0, &mut receipts);
    execute(&mut pipeline, ImageStage::Decoder, 1, 0, &mut receipts);
    let target = ImageStage::TextEncoder;
    assert!(pipeline
        .reclaim_after_allocation_failure(target, &mut receipts)
        .unwrap_or_else(|error| panic!("{error}")));
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 2);
    assert!(pipeline
        .reclaim_after_allocation_failure(target, &mut receipts)
        .unwrap_or_else(|error| panic!("{error}")));
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 1);
    assert_eq!(
        receipts.last().unwrap().execution.plan.stage,
        ImageStage::Denoiser
    );
    assert!(pipeline
        .reclaim_after_allocation_failure(target, &mut receipts)
        .unwrap_or_else(|error| panic!("{error}")));
    assert!(!pipeline
        .reclaim_after_allocation_failure(target, &mut receipts)
        .unwrap_or_else(|error| panic!("{error}")));
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    // Successful later work still produces exact numerical output.
    execute(&mut pipeline, target, 1, 1, &mut receipts);
    close(&mut pipeline);
}

#[test]
fn allocation_recovery_requires_retryable_failure_and_releases_transaction_owner() {
    use ferrum_runtime::test_support::resource_support::failed_runtime_allocation;
    let (pipeline, _directory) = pipeline();
    let plan = stage(&pipeline, ImageStage::TextEncoder, 1).plan;
    for retryable in [true, false] {
        let (source, probe) = failed_runtime_allocation(retryable);
        let failure = ImageGenerationFailure::Initialization {
            plan: plan.clone(),
            source: ComponentInitializationError::Commit(source),
        };
        assert!(probe.runtime_alive());
        let recovered = failure.recover_allocation();
        assert_eq!(recovered.is_ok(), retryable);
        assert_eq!(probe.runtime_alive(), !retryable);
        assert_eq!(probe.snapshot().uploads, 0);
        assert_eq!(probe.snapshot().submissions, 0);
    }
}

#[test]
fn adaptive_ample_capacity_reuses_weights_across_shapes_and_edit_stage_subsets() {
    let (mut pipeline, _directory) = pipeline();
    let mut receipts = Vec::new();
    for (job, rows) in [1, 4, 1, 1].into_iter().enumerate() {
        let kinds: &[ImageStage] = if job == 0 || job == 2 {
            &[
                ImageStage::TextEncoder,
                ImageStage::VisionEncoder,
                ImageStage::VaeEncoder,
                ImageStage::Denoiser,
                ImageStage::Decoder,
            ]
        } else {
            &[
                ImageStage::TextEncoder,
                ImageStage::Denoiser,
                ImageStage::Decoder,
            ]
        };
        for &kind in kinds {
            let receipt = execute(&mut pipeline, kind, rows, job as u64, &mut receipts);
            assert_eq!(receipt.weights_reused, job > 0);
            if job == 3 {
                assert!(receipt.session_reused);
            }
        }
    }
    assert!(!receipts.iter().any(|r| matches!(
        r.retirement,
        ImageStageRetirement::SharedWeightsClosed { .. }
    )));
    assert!(receipts.iter().any(|r| matches!(&r.retirement,
        ImageStageRetirement::ResidentBackingRetained { retention } if retention.retained_bytes()>0)));
    close(&mut pipeline);
}

#[test]
fn adaptive_reclaims_only_needed_other_pool_and_keeps_target_reuse() {
    let (mut pipeline, _directory) = pipeline();
    let mut receipts = Vec::new();
    execute(&mut pipeline, ImageStage::TextEncoder, 8, 0, &mut receipts);
    execute(&mut pipeline, ImageStage::Denoiser, 1, 0, &mut receipts);
    let target = stage(&pipeline, ImageStage::Denoiser, 1);
    pipeline.config.device_capacity_bytes = 2 * target.plan.static_bytes + demand(&target);
    let before = pipeline.runtime.snapshot();
    let offset = receipts.len();
    let receipt = execute(&mut pipeline, ImageStage::Denoiser, 1, 1, &mut receipts);
    assert!(receipt.session_reused && receipt.weights_reused);
    assert_eq!(pipeline.runtime.snapshot().allocations, before.allocations);
    assert!(receipts[offset..]
        .iter()
        .any(|r| r.execution.plan.stage == ImageStage::TextEncoder
            && r.execution.completed_executions == 0
            && matches!(&r.retirement, ImageStageRetirement::Resident { idle_release }
            if idle_release.reclaimed_bytes() > 0)));
    assert!(!receipts[offset..].iter().any(|r| matches!(
        r.retirement,
        ImageStageRetirement::SharedWeightsClosed { .. }
    )));
    close(&mut pipeline);
}

#[test]
fn adaptive_static_eviction_without_reload_observations_uses_lru() {
    let (mut pipeline, _directory) = pipeline();
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::Release;
    let mut receipts = Vec::new();
    for kind in [
        ImageStage::TextEncoder,
        ImageStage::VisionEncoder,
        ImageStage::Decoder,
    ] {
        execute(&mut pipeline, kind, 1, 0, &mut receipts);
        pipeline.resident_stages.set_weight_reload_timing(kind, 0);
    }
    let target = stage(&pipeline, ImageStage::Denoiser, 1);
    pipeline.config.device_capacity_bytes = 3 * target.plan.static_bytes + demand(&target);
    let offset = receipts.len();
    execute(&mut pipeline, ImageStage::Denoiser, 1, 1, &mut receipts);
    let evicted = receipts[offset..]
        .iter()
        .filter(|r| {
            matches!(
                r.retirement,
                ImageStageRetirement::SharedWeightsClosed { .. }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(evicted.len(), 1);
    assert_eq!(evicted[0].execution.plan.stage, ImageStage::TextEncoder);
    assert_eq!(evicted[0].execution.completed_executions, 0);
    let reused = execute(&mut pipeline, ImageStage::Decoder, 1, 2, &mut receipts);
    assert!(reused.session_reused && reused.weights_reused);
    let reloaded = execute(&mut pipeline, ImageStage::TextEncoder, 1, 3, &mut receipts);
    assert!(!reloaded.session_reused && !reloaded.weights_reused);
    close(&mut pipeline);
}

#[test]
fn adaptive_static_eviction_minimizes_reload_cost_and_preserves_live_capacity() {
    let (mut pipeline, _directory) = pipeline();
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::Release;
    let mut receipts = Vec::new();
    for (kind, reload_us) in [
        (ImageStage::TextEncoder, 90),
        (ImageStage::VisionEncoder, 10),
        (ImageStage::Decoder, 20),
    ] {
        execute(&mut pipeline, kind, 1, 0, &mut receipts);
        // Supply deterministic measured costs without a wall-clock assertion.
        // Static installations and their device-account ownership are real.
        pipeline
            .resident_stages
            .set_weight_reload_timing(kind, reload_us);
    }
    let target = stage(&pipeline, ImageStage::Denoiser, 1);
    let ceiling = 2 * target.plan.static_bytes + demand(&target);
    pipeline.config.device_capacity_bytes = ceiling;
    let offset = receipts.len();
    execute(&mut pipeline, ImageStage::Denoiser, 1, 1, &mut receipts);
    let evicted: Vec<_> = receipts[offset..]
        .iter()
        .filter(|r| {
            matches!(
                r.retirement,
                ImageStageRetirement::SharedWeightsClosed { .. }
            )
        })
        .map(|r| r.execution.plan.stage)
        .collect();
    assert_eq!(evicted, [ImageStage::VisionEncoder, ImageStage::Decoder]);
    let before = pipeline.runtime.snapshot().uploaded_bytes;
    let reused = execute(&mut pipeline, ImageStage::TextEncoder, 1, 2, &mut receipts);
    assert!(reused.session_reused && reused.weights_reused);
    assert_eq!(pipeline.runtime.snapshot().uploaded_bytes - before, 128);
    // Insufficient capacity still rejects before evicting useful installations.
    pipeline.config.device_capacity_bytes = target.plan.static_bytes + demand(&target) - 1;
    assert!(pipeline.start_stage(target, 3, &mut receipts).is_err());
    pipeline.config.device_capacity_bytes = ceiling;
    close(&mut pipeline);
}

#[test]
fn adaptive_impossible_cold_stage_rejects_before_destroying_prior_residency() {
    let (mut pipeline, _directory) = pipeline();
    let mut receipts = Vec::new();
    execute(&mut pipeline, ImageStage::TextEncoder, 1, 0, &mut receipts);
    let target = stage(&pipeline, ImageStage::Decoder, 64);
    pipeline.config.device_capacity_bytes = target.plan.static_bytes + demand(&target) - 1;
    let before = pipeline.runtime.snapshot();
    let count = receipts.len();
    assert!(pipeline.validate_resident_capacity(&[&target]).is_err());
    assert!(matches!(
        pipeline.start_stage(target, 1, &mut receipts),
        Err(ImageGenerationFailure::Preflight(_))
    ));
    assert_eq!(receipts.len(), count);
    assert_eq!(
        pipeline.runtime.snapshot().live_allocations,
        before.live_allocations
    );
    assert_eq!(
        pipeline.runtime.snapshot().uploaded_bytes,
        before.uploaded_bytes
    );
    pipeline.config.device_capacity_bytes = DEVICE_BYTES;
    assert!(execute(&mut pipeline, ImageStage::TextEncoder, 1, 2, &mut receipts).session_reused);
    close(&mut pipeline);
}

#[test]
fn adaptive_prefix_admission_keeps_both_caches_and_full_installation_in_the_bound() {
    let (mut pipeline, _directory) = pipeline();
    let config = QwenImage21TransformerConfig::new(
        transformer(),
        QwenImage21T2iLayout::new(68, 2, 2, vec![68]).unwrap(),
    )
    .unwrap()
    .with_reference_image(2, 64, 16, 16)
    .unwrap();
    let mut prefill = pipeline
        .prepare_denoiser(
            &config
                .clone()
                .with_execution_phase(QwenImage21TransformerPhase::PrefixPrefill)
                .unwrap(),
            1,
        )
        .unwrap();
    let decode = pipeline
        .prepare_denoiser(
            &config
                .with_execution_phase(QwenImage21TransformerPhase::PrefixDecode)
                .unwrap(),
            1,
        )
        .unwrap();
    let pm = prefill
        .component
        .executable()
        .execution_plan()
        .payload()
        .memory();
    let dm = decode
        .component
        .executable()
        .execution_plan()
        .payload()
        .memory();
    let joint = (pm.minimum_runnable_request_bytes()
        + dm.minimum_request_bytes()
        + dm.minimum_sequence_bytes())
    .max(dm.minimum_runnable_request_bytes());
    let required = pm.static_bytes() + joint;
    assert!(pm.static_bytes() > dm.static_bytes());
    assert!(dm.minimum_request_bytes() > 0);
    let prefill_dynamic = pm.minimum_runnable_request_bytes();
    prefill.bind_prefix_admission(&decode).unwrap();
    assert_eq!(
        prefill.admission_dynamic_bytes(prefill_dynamic).unwrap(),
        joint
    );
    pipeline.config.device_capacity_bytes = required;
    pipeline
        .validate_capacity_with_prefix(&[&prefill, &decode], Some((&prefill, &decode)))
        .unwrap();
    pipeline.config.device_capacity_bytes = required - 1;
    assert!(pipeline
        .validate_capacity_with_prefix(&[&prefill, &decode], Some((&prefill, &decode)))
        .is_err());
    let mut receipts = Vec::new();
    assert!(matches!(
        pipeline.start_stage(prefill, 0, &mut receipts),
        Err(ImageGenerationFailure::Preflight(_))
    ));
    assert!(receipts.is_empty());
    assert_eq!(pipeline.runtime.snapshot().uploaded_bytes, 0);
    close(&mut pipeline);
}

#[test]
fn adaptive_referenced_weight_eviction_returns_the_original_recovery_owner() {
    let (mut pipeline, _directory) = pipeline();
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::Release;
    let mut receipts = Vec::new();
    let prefill = stage(&pipeline, ImageStage::Denoiser, 1);
    let (active, peer_weights, temporary) = pipeline
        .start_prefix_prefill(prefill, 0, &mut receipts)
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(!temporary);
    pipeline
        .finish_stage(active, &mut receipts)
        .unwrap_or_else(|e| panic!("{e}"));
    let target = stage(&pipeline, ImageStage::Decoder, 1);
    pipeline.config.device_capacity_bytes = target.plan.static_bytes + demand(&target);
    let Err(ImageGenerationFailure::SharedWeightsReferenced { outcome, .. }) =
        pipeline.start_stage(target, 1, &mut receipts)
    else {
        panic!("external snapshot lease must prevent eviction success")
    };
    assert!(pipeline.runtime.snapshot().live_allocations > 0);
    assert!(!receipts.iter().any(|r| matches!(
        r.retirement,
        ImageStageRetirement::SharedWeightsClosed { .. }
    )));
    drop(peer_weights);
    let SharedComponentWeightsCloseOutcome::Referenced { weights } = outcome else {
        panic!("recoverable original weights")
    };
    assert!(matches!(
        weights.close(),
        Ok(SharedComponentWeightsCloseOutcome::Closed(_))
    ));
    close(&mut pipeline);
}

#[test]
fn adaptive_active_request_cannot_be_stored_as_an_idle_reusable_pool() {
    let (mut pipeline, directory) = pipeline();
    let mut receipts = Vec::new();
    // Reuse the adapter's complete tiny transformer archive. A projection has
    // no readonly Request state and cannot exercise this lifetime boundary.
    Arc::get_mut(&mut pipeline.assets)
        .expect("unique fixture assets")
        .sources
        .transformer = Arc::new(SafetensorsArchive::open(directory.path().join("dit")).unwrap());
    let config = QwenImage21TransformerConfig::new(
        transformer(),
        QwenImage21T2iLayout::new(1, 2, 2, vec![1]).unwrap(),
    )
    .unwrap();
    let prefill = pipeline
        .prepare_denoiser(
            &config
                .clone()
                .with_execution_phase(QwenImage21TransformerPhase::PrefixPrefill)
                .unwrap(),
            1,
        )
        .unwrap();
    let config = config
        .with_execution_phase(QwenImage21TransformerPhase::PrefixDecode)
        .unwrap();
    let prepared = pipeline.prepare_denoiser(&config, 1).unwrap();
    let (prefill, weights, temporary) = pipeline
        .start_prefix_prefill(prefill, 0, &mut receipts)
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(!temporary);
    let active = prepared
        .start_attached(&pipeline, 0, &weights)
        .unwrap_or_else(|e| panic!("{e}"));
    prefill.close().unwrap_or_else(|e| panic!("{e}"));
    drop(weights);
    let inputs = config
        .input_tensor_specs()
        .unwrap()
        .into_iter()
        .map(|(id, spec)| {
            let tensor =
                ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout).unwrap();
            let bytes = pipeline
                .host
                .allocate_zeroed(
                    HostMemoryClass::InputPreprocess,
                    tensor.minimum_storage_bytes().unwrap() as usize,
                )
                .unwrap()
                .freeze();
            ManagedHostTensor::new(id, tensor, bytes).unwrap()
        })
        .collect();
    let request = active
        .session
        .begin_request(
            id("run.adaptive.busy").unwrap(),
            id("request.adaptive.busy").unwrap(),
            inputs,
        )
        .unwrap();
    let Err(ImageGenerationFailure::Processing {
        active: Some(active),
        ..
    }) = pipeline.finish_stage(active, &mut receipts)
    else {
        panic!("active request must keep the actual session in a failure owner")
    };
    assert!(receipts.is_empty());
    request.close().unwrap();
    active.close().unwrap_or_else(|e| panic!("{e}"));
    close(&mut pipeline);
}
