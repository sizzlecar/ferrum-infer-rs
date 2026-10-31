//! Real Reference projections isolate residency/ledger behavior from model size.
//! Each typed slot owns its installation even when its immutable source matches.
use super::*;
use ferrum_kernels::backend::reference::reference_vnext_operation_registry;
use std::sync::atomic::{AtomicU64, Ordering};

fn pipeline() -> (
    QwenImage21Pipeline<ReferenceDeviceRuntime>,
    tempfile::TempDir,
) {
    let (original, directory) = numeric_pipeline();
    static NEXT_DEVICE: AtomicU64 = AtomicU64::new(0);
    let composition = composition(&format!(
        "device.image.selected.residency.{}",
        NEXT_DEVICE.fetch_add(1, Ordering::Relaxed)
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
    pipeline.config.residency = ImageResidencyPolicy::SelectedStages {
        stages: ImageResidentStageSet {
            vision_encoder: true,
            vae_encoder: true,
            denoiser: true,
            decoder: true,
            ..ImageResidentStageSet::default()
        },
    };
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::Release;
    (pipeline, directory)
}

fn stage(
    pipeline: &QwenImage21Pipeline<ReferenceDeviceRuntime>,
    kind: ImageStage,
    rows: u64,
) -> PreparedStage<ReferenceDeviceRuntime> {
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id(IMAGE_PROJECTION_INPUT_ID).unwrap(),
        ProgramTensorSpec {
            dimensions: vec![rows, 64],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();
    PreparedStage::compile(kind, 1, projection(), &options, pipeline).unwrap()
}

fn dynamic_bytes(stage: &PreparedStage<ReferenceDeviceRuntime>) -> u64 {
    stage
        .component
        .executable()
        .execution_plan()
        .payload()
        .memory()
        .minimum_runnable_request_bytes()
}

#[test]
fn selected_shape_replacement_rebinds_original_weights_without_upload() {
    let (pipeline, _directory) = pipeline();
    super::check_shape_replacement(pipeline);
}

#[test]
fn selected_failed_execution_returns_live_owner_until_explicit_recovery() {
    let (pipeline, _directory) = pipeline();
    super::check_failed_reused_session_owner(pipeline);
}

#[test]
fn selected_incompatible_weight_plan_preserves_original_installation() {
    let (pipeline, _directory) = pipeline();
    super::check_incompatible_weight_plan_owner(pipeline);
}

#[test]
fn selected_installations_reuse_weights_but_cold_text_reloads_and_inputs_stay_independent() {
    let (mut pipeline, _directory) = pipeline();
    let capacity = DeviceCapacityOwner::open(pipeline.runtime.descriptor(), DEVICE_BYTES).unwrap();
    let mut receipts = Vec::new();
    let mut outputs = Vec::new();
    for job in 0..2_u64 {
        let mut retained_static = 0;
        for (index, kind) in [
            ImageStage::VisionEncoder,
            ImageStage::TextEncoder,
            ImageStage::VaeEncoder,
            ImageStage::Denoiser,
            ImageStage::Decoder,
        ]
        .into_iter()
        .enumerate()
        {
            let prepared = stage(&pipeline, kind, 1);
            let selected = kind != ImageStage::TextEncoder;
            if selected {
                retained_static += prepared.plan.static_bytes;
            }
            let before = pipeline.runtime.snapshot().uploaded_bytes;
            let active = pipeline
                .start_stage(prepared, job, &mut receipts)
                .unwrap_or_else(|e| panic!("start selected fixture: {e}"));
            assert_eq!(active.receipt.session_reused, selected && job > 0);
            assert_eq!(active.receipt.weights_reused, selected && job > 0);
            let uploaded = pipeline.runtime.snapshot().uploaded_bytes - before;
            if selected && job > 0 {
                assert_eq!(uploaded, 0);
            } else {
                assert!(uploaded > 0);
            }
            let offset = job as usize * 16 + index * 2;
            let (active, result) = pipeline
                .execute_stage(active, job, vec![input(&pipeline.host, 1, offset)])
                .unwrap_or_else(|e| panic!("execute fresh input: {e}"));
            let output = only_output(result, IMAGE_PROJECTION_OUTPUT_ID).unwrap();
            for (channel, bytes) in output.bytes().bytes().chunks_exact(2).enumerate() {
                assert_eq!(
                    u16::from_le_bytes([bytes[0], bytes[1]]),
                    bf16::from_f32((offset + channel + 1) as f32).to_bits()
                );
            }
            pipeline
                .finish_stage(active, &mut receipts)
                .unwrap_or_else(|e| panic!("retire selected fixture: {e}"));
            match &receipts.last().unwrap().retirement {
                ImageStageRetirement::Resident { idle_release } if selected => {
                    assert!(idle_release.reclaimed_bytes() > 0);
                }
                ImageStageRetirement::Closed if !selected => {}
                other => panic!("unexpected retirement: {other:?}"),
            }
            outputs.push(output);
        }
        // Only selected static grants remain. Dynamic Release and cold TE must
        // return every other device byte, rather than merely report residency.
        let remaining = capacity
            .reserve(
                DeviceOpaqueMemoryKind::Bootstrap,
                DEVICE_BYTES - retained_static,
            )
            .unwrap();
        assert!(capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, 1)
            .is_err());
        drop(remaining);
    }
    let retained_output_bytes = outputs.iter().map(|o| o.bytes().len()).sum::<usize>();
    pipeline.shutdown().unwrap();
    assert!(pipeline.serial_actor.is_none());
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    drop(
        capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, DEVICE_BYTES)
            .unwrap(),
    );
    assert_eq!(
        pipeline.host.usage().unwrap().committed_bytes(),
        retained_output_bytes
    );
    assert_eq!(&outputs[0].bytes().bytes()[..2], &0x3f80_u16.to_le_bytes());
    drop(outputs);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn selected_capacity_counts_cold_text_only_during_its_active_stage() {
    let (mut pipeline, _directory) = pipeline();
    let planned = [
        stage(&pipeline, ImageStage::VisionEncoder, 1),
        stage(&pipeline, ImageStage::TextEncoder, 8),
        stage(&pipeline, ImageStage::VaeEncoder, 1),
        stage(&pipeline, ImageStage::Denoiser, 1),
        stage(&pipeline, ImageStage::Decoder, 1),
    ];
    let retained = planned
        .iter()
        .filter(|s| s.plan.stage != ImageStage::TextEncoder)
        .map(|s| s.plan.static_bytes)
        .sum::<u64>();
    let text_peak = planned[1].plan.static_bytes + dynamic_bytes(&planned[1]);
    assert!(planned
        .iter()
        .filter(|s| s.plan.stage != ImageStage::TextEncoder)
        .all(|s| dynamic_bytes(s) < text_peak));
    let required = retained + text_peak;
    let refs = planned.iter().collect::<Vec<_>>();
    let before = pipeline.runtime.snapshot();
    pipeline.config.device_capacity_bytes = required - 1;
    assert!(pipeline.validate_resident_capacity(&refs).is_err());
    pipeline.config.device_capacity_bytes = required;
    pipeline.validate_resident_capacity(&refs).unwrap();
    assert_eq!(pipeline.runtime.snapshot(), before);
    pipeline.shutdown().unwrap();
}

#[test]
fn generation_after_edit_counts_absent_vision_and_encoder_installations() {
    let (mut pipeline, _directory) = pipeline();
    let generation = [
        stage(&pipeline, ImageStage::TextEncoder, 8),
        stage(&pipeline, ImageStage::Denoiser, 1),
        stage(&pipeline, ImageStage::Decoder, 1),
    ];
    let generation_only = generation[1].plan.static_bytes
        + generation[2].plan.static_bytes
        + generation[0].plan.static_bytes
        + dynamic_bytes(&generation[0]);
    let refs = generation.iter().collect::<Vec<_>>();
    pipeline.config.device_capacity_bytes = generation_only;
    pipeline.validate_resident_capacity(&refs).unwrap();
    pipeline.config.device_capacity_bytes = DEVICE_BYTES;
    let mut receipts = Vec::new();
    let mut editing_static = 0;
    for kind in [ImageStage::VisionEncoder, ImageStage::VaeEncoder] {
        let prepared = stage(&pipeline, kind, 1);
        editing_static += prepared.plan.static_bytes;
        let active = pipeline
            .start_stage(prepared, 0, &mut receipts)
            .unwrap_or_else(|e| panic!("install editing stage: {e}"));
        let (active, output) = pipeline
            .execute_stage(active, 0, vec![input(&pipeline.host, 1, 0)])
            .unwrap_or_else(|e| panic!("execute editing stage: {e}"));
        drop(output);
        pipeline
            .finish_stage(active, &mut receipts)
            .unwrap_or_else(|e| panic!("retain editing installation: {e}"));
    }
    // These stages are absent from the next generation plan, but their actual
    // shared-weight owners are still live and consume the same device ledger.
    let capacity = DeviceCapacityOwner::open(pipeline.runtime.descriptor(), DEVICE_BYTES).unwrap();
    let remaining = capacity
        .reserve(
            DeviceOpaqueMemoryKind::Bootstrap,
            DEVICE_BYTES - editing_static,
        )
        .unwrap();
    assert!(capacity
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, 1)
        .is_err());
    drop(remaining);
    let before = pipeline.runtime.snapshot();
    pipeline.config.device_capacity_bytes = generation_only;
    assert!(pipeline.validate_resident_capacity(&refs).is_err());
    pipeline.config.device_capacity_bytes = generation_only + editing_static - 1;
    assert!(pipeline.validate_resident_capacity(&refs).is_err());
    pipeline.config.device_capacity_bytes = generation_only + editing_static;
    pipeline.validate_resident_capacity(&refs).unwrap();
    assert_eq!(pipeline.runtime.snapshot(), before);
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    drop(
        capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, DEVICE_BYTES)
            .unwrap(),
    );
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn subset_capacity_keeps_other_weight_owners_without_counting_active_snapshot_twice() {
    let (mut pipeline, _directory) = pipeline();
    pipeline.config.residency = ImageResidencyPolicy::SelectedStages {
        stages: ImageResidentStageSet {
            text_encoder: true,
            denoiser: true,
            decoder: true,
            ..ImageResidentStageSet::default()
        },
    };
    let mut receipts = Vec::new();
    let mut installed_static = 0;
    for kind in [
        ImageStage::TextEncoder,
        ImageStage::Denoiser,
        ImageStage::Decoder,
    ] {
        let prepared = stage(&pipeline, kind, 1);
        installed_static += prepared.plan.static_bytes;
        let active = pipeline
            .start_stage(prepared, 0, &mut receipts)
            .unwrap_or_else(|e| panic!("install subset fixture: {e}"));
        let (active, outputs) = pipeline
            .execute_stage(active, 0, vec![input(&pipeline.host, 1, 0)])
            .unwrap_or_else(|e| panic!("execute subset fixture: {e}"));
        drop(outputs);
        pipeline
            .finish_stage(active, &mut receipts)
            .unwrap_or_else(|e| panic!("retain subset installation: {e}"));
    }
    let decoder = stage(&pipeline, ImageStage::Decoder, 1);
    // All three physical installations remain live. Only Decoder is in the
    // requested subset, and its one installation must not be charged twice.
    let required = installed_static + dynamic_bytes(&decoder);
    let before = pipeline.runtime.snapshot();
    pipeline.config.device_capacity_bytes = required - 1;
    assert!(pipeline.validate_resident_capacity(&[&decoder]).is_err());
    pipeline.config.device_capacity_bytes = required;
    pipeline.validate_resident_capacity(&[&decoder]).unwrap();
    assert_eq!(pipeline.runtime.snapshot(), before);
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}
