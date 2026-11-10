//! Installation metadata shares the existing bounded cache; the real Reference
//! source/provider exercise ownership without pretending to run CUDA H3 math.
use super::*;
use ferrum_kernels::backend::reference::ReferenceDeviceRuntime;

type Pipeline = QwenImage21Pipeline<ReferenceDeviceRuntime>;
type Stage = PreparedStage<ReferenceDeviceRuntime>;
fn compile(p: &Pipeline, stage: ImageStage) -> Stage {
    p.execution
        .compile_stage(
            ImageStageDeclaration {
                stage,
                role: ImageStagePlanRole::WeightInstallation,
                batch_size: 1,
                weight_owner: p.execution.sources.weight_owner(stage),
                attachment: ImageStageAttachment::InitializeShared,
            },
            projection(),
            &projection_options(),
        )
        .unwrap()
}
fn acquire(p: &Pipeline, stage: ImageStage) -> Option<Stage> {
    Stage::cached_weight_installation(stage, &p.execution).unwrap()
}

#[test]
fn weight_installation_reuses_original_geometry_and_declaration_only_in_its_domain() {
    let (mut p, _dir) = super::super::adapter::pipeline(2);
    let original = compile(&p, ImageStage::Denoiser);
    // A different request shape compiles normally without replacing the
    // installation role or borrowing its execution geometry.
    let other = cached_projection(&p, ImageStage::Denoiser, 2);
    let hit = acquire(&p, ImageStage::Denoiser).unwrap();
    assert!(Arc::ptr_eq(&original.component, &hit.component));
    assert!(!Arc::ptr_eq(&other.component, &hit.component));
    assert_eq!(original.declaration(), hit.declaration());
    assert_eq!(hit.plan.batch_size, 1);
    assert_eq!(other.plan.batch_size, 2);
    assert_eq!(original.plan.plan_hash, hit.plan.plan_hash);
    assert_eq!(
        original.plan.family_fingerprint,
        hit.plan.family_fingerprint
    );
    assert!(hit.plan.plan_cache_hit);
    let domain = Arc::clone(&p.execution.service_preparation_identity);
    p.execution.service_preparation_identity = Arc::new(());
    assert!(acquire(&p, ImageStage::Denoiser).is_none());
    p.execution.service_preparation_identity = domain;
    p.execution.config.weight_arena_target_bytes = std::num::NonZeroU64::new(4096);
    assert!(acquire(&p, ImageStage::Denoiser).is_none());
    assert!(p
        .execution
        .plan_cache
        .get_weight_installation(
            p.execution.config.plan_cache,
            ImageStage::Denoiser,
            Some(ImageWeightOwnerKey::Decoder),
            &p.execution.service_preparation_identity,
            WeightArenaPacking::ByElementType,
        )
        .unwrap()
        .is_none());
    assert_eq!(p.execution.host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(p.execution.runtime.snapshot().submissions, 0);
}

#[test]
fn weight_installation_obeys_disabled_oversized_and_aggregate_stage_limits() {
    let (mut p, _dir) = super::super::adapter::pipeline(2);
    for policy in [ImagePlanCachePolicy::Disabled, multi_shape(2, 1)] {
        p.execution.config.plan_cache = policy;
        let original = compile(&p, ImageStage::Denoiser);
        let weak = Arc::downgrade(&original.component);
        assert!(acquire(&p, ImageStage::Denoiser).is_none());
        drop(original);
        assert!(weak.upgrade().is_none());
    }
    p.execution.config.plan_cache = multi_shape(1, 16 << 20);
    let original = compile(&p, ImageStage::Denoiser);
    let encoded = original
        .plan
        .plan_cache
        .as_ref()
        .unwrap()
        .encoded_metadata_bytes
        .unwrap();
    assert!(encoded > 0);
    let hit = acquire(&p, ImageStage::Denoiser).unwrap();
    assert_eq!(
        hit.plan.plan_cache.as_ref().unwrap().encoded_metadata_bytes,
        Some(encoded)
    );
    let weak = Arc::downgrade(&original.component);
    drop(original);
    drop(hit);
    // MultiShape's stage bound includes all roles, including installation.
    drop(cached_projection(&p, ImageStage::Denoiser, 2));
    assert!(weak.upgrade().is_none());
    assert!(acquire(&p, ImageStage::Denoiser).is_none());
}

#[test]
fn weight_installation_candidate_rollback_releases_new_owner_and_preserves_old_hit() {
    let (mut p, _dir) = super::super::adapter::pipeline(2);
    p.execution.config.plan_cache = multi_shape(3, 16 << 20);
    let original = compile(&p, ImageStage::Denoiser);
    let pending = p.execution.plan_cache.begin_candidate().unwrap();
    let hit = acquire(&p, ImageStage::Denoiser).unwrap();
    let new = compile(&p, ImageStage::Decoder);
    let weak = Arc::downgrade(&new.component);
    drop(new);
    let waiting = pending.defer().unwrap();
    assert!(weak.upgrade().is_some());
    drop(waiting);
    assert!(weak.upgrade().is_none());
    assert!(Arc::ptr_eq(
        &original.component,
        &acquire(&p, ImageStage::Denoiser).unwrap().component
    ));
    assert!(Arc::ptr_eq(&original.component, &hit.component));
    assert!(acquire(&p, ImageStage::Decoder).is_none());
    assert_eq!(p.execution.runtime.snapshot().allocations, 0);
    assert_eq!(p.execution.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn weight_installation_refuses_forward_and_keeps_original_session_for_retirement() {
    let (mut p, _dir) = super::super::adapter::pipeline(2);
    let installation = compile(&p, ImageStage::Denoiser);
    let mut receipts = vec![];
    let active = p
        .execution
        .start_stage(installation, 0, &mut receipts)
        .unwrap_or_else(|e| panic!("installation: {e}"));
    let pin = p
        .execution
        .pin_weight_owner(ImageWeightOwnerKey::Denoiser)
        .unwrap();
    pin.weights()
        .validate_for(
            active.session.executable().execution_plan(),
            &p.execution.runtime,
        )
        .unwrap();
    let submissions = p.execution.runtime.snapshot().submissions;
    let failure = active
        .execute(0, vec![])
        .err()
        .expect("installation cannot run a forward");
    let ImageGenerationFailure::Processing {
        source,
        active: Some(active),
    } = failure
    else {
        panic!("forward refusal must retain the actual initialized session");
    };
    assert!(source
        .to_string()
        .contains("installation plans cannot execute"));
    assert_eq!(active.receipt.completed_executions, 0);
    assert_eq!(p.execution.runtime.snapshot().submissions, submissions);
    drop(pin);
    p.execution
        .finish_stage(active, &mut receipts)
        .unwrap_or_else(|e| panic!("retirement: {e}"));
    p.shutdown().unwrap();
    assert_eq!(p.execution.runtime.snapshot().live_allocations, 0);
    assert_eq!(p.execution.host.usage().unwrap().committed_bytes(), 0);
}
