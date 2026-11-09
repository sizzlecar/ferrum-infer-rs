//! The same actual Stored projection and native owners used by adaptive stage
//! tests, now admitting an independent component rather than a fake stage role.
use super::*;

fn retained_dynamic_bytes(receipts: &[ImageStageReceipt], stage: ImageStage) -> u64 {
    let receipt = receipts
        .iter()
        .rev()
        .find(|receipt| receipt.execution.plan.stage == stage)
        .expect("the actual stage has a retirement receipt");
    match &receipt.retirement {
        ImageStageRetirement::ResidentBackingRetained { retention } => retention.retained_bytes(),
        ImageStageRetirement::Resident { .. }
        | ImageStageRetirement::Closed
        | ImageStageRetirement::SharedWeightsClosed { .. } => 0,
    }
}

fn admit(
    pipeline: &mut QwenImage21Pipeline<ReferenceDeviceRuntime>,
    peak: u64,
    receipts: &mut Vec<ImageStageReceipt>,
) -> bool {
    pipeline
        .execution
        .admit_auxiliary_peak(peak, receipts)
        .unwrap_or_else(|failure| panic!("auxiliary admission: {failure}"))
}

#[test]
fn auxiliary_impossible_cold_peak_preserves_every_original_idle_owner() {
    let (mut pipeline, _directory) = pipeline();
    let mut receipts = Vec::new();
    execute(&mut pipeline, ImageStage::TextEncoder, 2, 0, &mut receipts);
    let prepared = stage(&pipeline, ImageStage::Denoiser, 1);
    let peak = prepared.plan.static_bytes + demand(&prepared);
    pipeline.execution.device_opaque_allowance_bytes = 1;
    pipeline.execution.config.device_capacity_bytes = peak;
    let before = pipeline.execution.runtime.snapshot();
    let offset = receipts.len();
    assert!(!admit(&mut pipeline, peak, &mut receipts));
    assert_eq!(receipts.len(), offset);
    assert_eq!(
        pipeline.execution.runtime.snapshot().live_allocations,
        before.live_allocations
    );
    assert!(pipeline
        .execution
        .resident_stages
        .has_weights(ImageStage::TextEncoder));
    assert!(pipeline
        .execution
        .admit_auxiliary_peak(u64::MAX, &mut receipts)
        .is_err());
    assert_eq!(receipts.len(), offset);
    pipeline.execution.device_opaque_allowance_bytes = 0;
    close(&mut pipeline);
}

#[test]
fn auxiliary_trims_only_the_needed_idle_dynamic_pool_before_any_static_owner() {
    let (mut pipeline, _directory) = pipeline();
    let mut receipts = Vec::new();
    let text = execute(&mut pipeline, ImageStage::TextEncoder, 8, 0, &mut receipts);
    let denoiser = execute(&mut pipeline, ImageStage::Denoiser, 1, 1, &mut receipts);
    let text_dynamic = retained_dynamic_bytes(&receipts, ImageStage::TextEncoder);
    let denoiser_dynamic = retained_dynamic_bytes(&receipts, ImageStage::Denoiser);
    assert!(text_dynamic > 0 && denoiser_dynamic > 0);
    let auxiliary = stage(&pipeline, ImageStage::Decoder, 2);
    let peak = auxiliary.plan.static_bytes + demand(&auxiliary);
    pipeline.execution.device_opaque_allowance_bytes = 1;
    pipeline.execution.config.device_capacity_bytes =
        text.plan.static_bytes + denoiser.plan.static_bytes + denoiser_dynamic + peak + 1;
    let before = pipeline.execution.runtime.snapshot();
    let offset = receipts.len();
    assert!(admit(&mut pipeline, peak, &mut receipts));
    assert_eq!(
        pipeline.execution.runtime.snapshot().allocations,
        before.allocations
    );
    assert_eq!(
        retained_dynamic_bytes(&receipts, ImageStage::TextEncoder),
        0
    );
    assert_eq!(
        retained_dynamic_bytes(&receipts, ImageStage::Denoiser),
        denoiser_dynamic
    );
    assert!(receipts[offset..]
        .iter()
        .any(|r| r.execution.plan.stage == ImageStage::TextEncoder
            && matches!(&r.retirement,
            ImageStageRetirement::Resident { idle_release } if idle_release.reclaimed_bytes()>0)));
    assert!(!receipts[offset..].iter().any(|r| matches!(
        r.retirement,
        ImageStageRetirement::SharedWeightsClosed { .. }
    )));
    pipeline.execution.device_opaque_allowance_bytes = 0;
    close(&mut pipeline);
}

#[test]
fn auxiliary_static_reclamation_is_lru_without_decoder_or_text_target_preference() {
    for first in [ImageStage::Decoder, ImageStage::TextEncoder] {
        let (mut pipeline, _directory) = pipeline();
        pipeline.execution.config.dynamic_residency = ImageDynamicResidencyPolicy::Release;
        let other = if first == ImageStage::Decoder {
            ImageStage::TextEncoder
        } else {
            ImageStage::Decoder
        };
        let mut receipts = Vec::new();
        let a = execute(&mut pipeline, first, 1, 0, &mut receipts);
        let b = execute(&mut pipeline, ImageStage::Denoiser, 1, 1, &mut receipts);
        let c = execute(&mut pipeline, other, 1, 2, &mut receipts);
        let auxiliary = stage(&pipeline, ImageStage::VisionEncoder, 1);
        let peak = auxiliary.plan.static_bytes + demand(&auxiliary);
        pipeline.execution.config.device_capacity_bytes =
            b.plan.static_bytes + c.plan.static_bytes + peak;
        let before = pipeline.execution.runtime.snapshot();
        let offset = receipts.len();
        assert!(admit(&mut pipeline, peak, &mut receipts));
        assert_eq!(
            pipeline.execution.runtime.snapshot().allocations,
            before.allocations
        );
        let closed = receipts[offset..]
            .iter()
            .filter(|r| {
                matches!(
                    r.retirement,
                    ImageStageRetirement::SharedWeightsClosed { .. }
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].execution.plan.stage, first);
        assert!(!pipeline.execution.resident_stages.has_weights(first));
        assert!(pipeline
            .execution
            .resident_stages
            .has_weights(ImageStage::Denoiser));
        assert!(pipeline.execution.resident_stages.has_weights(other));
        assert!(a.plan.static_bytes > 0);
        close(&mut pipeline);
    }
}

#[test]
fn auxiliary_never_reclaims_a_pinned_denoiser_even_when_its_session_is_idle() {
    let (mut pipeline, _directory) = pipeline();
    let mut receipts = Vec::new();
    let installed = execute(&mut pipeline, ImageStage::Denoiser, 1, 0, &mut receipts);
    let pin = pipeline
        .execution
        .pin_weight_owner(ImageWeightOwnerKey::Denoiser)
        .unwrap();
    let dynamic = retained_dynamic_bytes(&receipts, ImageStage::Denoiser);
    let auxiliary = stage(&pipeline, ImageStage::Decoder, 1);
    let peak = auxiliary.plan.static_bytes + demand(&auxiliary);
    pipeline.execution.config.device_capacity_bytes =
        installed.plan.static_bytes + dynamic + peak - 1;
    let before = pipeline.execution.runtime.snapshot();
    let offset = receipts.len();
    assert!(!admit(&mut pipeline, peak, &mut receipts));
    assert_eq!(receipts.len(), offset);
    assert_eq!(
        pipeline.execution.runtime.snapshot().live_allocations,
        before.live_allocations
    );
    assert_eq!(
        retained_dynamic_bytes(&receipts, ImageStage::Denoiser),
        dynamic
    );
    assert!(pipeline
        .execution
        .resident_stages
        .has_weights(ImageStage::Denoiser));
    drop(pin);
    assert!(admit(&mut pipeline, peak, &mut receipts));
    close(&mut pipeline);
}

#[test]
fn auxiliary_cannot_retire_checked_out_sessions_and_preserves_original_close_failure() {
    let (mut pipeline, _directory) = pipeline();
    let mut receipts = Vec::new();
    let prepared = stage(&pipeline, ImageStage::TextEncoder, 1);
    let peak = prepared.plan.static_bytes;
    let active = pipeline
        .start_stage(prepared, 0, &mut receipts)
        .unwrap_or_else(|failure| panic!("Stored installation: {failure}"));
    pipeline.execution.config.device_capacity_bytes = 2 * peak - 1;
    let offset = receipts.len();
    assert!(!admit(&mut pipeline, peak, &mut receipts));
    assert_eq!(receipts.len(), offset);
    pipeline
        .finish_stage(active, &mut receipts)
        .unwrap_or_else(|failure| panic!("actual idle stage: {failure}"));
    let pin = pipeline
        .execution
        .pin_weight_owner(ImageWeightOwnerKey::TextEncoder)
        .unwrap();
    let external = pin.weights().clone();
    drop(pin);
    let failure = pipeline
        .execution
        .admit_auxiliary_peak(peak, &mut receipts)
        .unwrap_err();
    let ImageGenerationFailure::SharedWeightsReferenced { stage, outcome } = failure else {
        panic!("close must retain its exact referenced weight owner");
    };
    assert_eq!(stage, ImageStage::TextEncoder);
    let SharedComponentWeightsCloseOutcome::Referenced { weights } = outcome else {
        panic!("referenced close lost its original owner");
    };
    drop(external);
    assert!(pipeline.execution.runtime.snapshot().live_allocations > 0);
    let closed = weights
        .close()
        .unwrap_or_else(|failure| panic!("original close retry: {:?}", failure.failure()));
    assert!(matches!(
        closed,
        SharedComponentWeightsCloseOutcome::Closed(_)
    ));
    close(&mut pipeline);
}
