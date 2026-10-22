use super::fixture::*;
use super::*;

#[test]
fn stage_cache_reuses_compilation_and_misses_changed_geometry_or_options() {
    let (pipeline, _directory) = super::adapter::pipeline(2);
    let compile = |batch, options: &ProgramPlanCompileOptions| {
        PreparedStage::compile(
            ImageStage::Denoiser,
            batch,
            projection(),
            options,
            &pipeline,
        )
        .unwrap()
    };
    let options = projection_options();
    let first = compile(1, &options);
    assert!(!first.plan.plan_cache_hit);
    let repeated = compile(1, &options);
    assert!(repeated.plan.plan_cache_hit);
    assert!(Arc::ptr_eq(&first.component, &repeated.component));
    assert_eq!(first.plan.plan_hash, repeated.plan.plan_hash);
    let changed = compile(2, &options);
    assert!(!changed.plan.plan_cache_hit);
    assert!(!Arc::ptr_eq(&first.component, &changed.component));
    let replaced = compile(1, &options);
    assert!(!replaced.plan.plan_cache_hit);
    let mut explicit_retention = options.clone();
    let output = projection().program().outputs()[0].clone();
    explicit_retention.retain_completion_value(output);
    let changed_options = compile(1, &explicit_retention);
    assert!(!changed_options.plan.plan_cache_hit);
    assert!(compile(1, &explicit_retention).plan.plan_cache_hit);
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn disabled_cache_compiles_again_without_retaining_component() {
    let (mut pipeline, _directory) = super::adapter::pipeline(1);
    pipeline.config.plan_cache = ImagePlanCachePolicy::Disabled;
    let options = projection_options();
    let compile = || {
        PreparedStage::compile(ImageStage::Denoiser, 1, projection(), &options, &pipeline).unwrap()
    };
    let first = compile();
    let weak = Arc::downgrade(&first.component);
    assert!(!first.plan.plan_cache_hit);
    drop(first);
    assert!(weak.upgrade().is_none());
    assert!(!compile().plan.plan_cache_hit);
}

#[test]
fn stages_have_independent_single_entry_caches_and_eviction_releases_old_metadata() {
    let (pipeline, _directory) = super::adapter::pipeline(2);
    let options = projection_options();
    let compile = |stage, batch| {
        PreparedStage::compile(stage, batch, projection(), &options, &pipeline).unwrap()
    };
    let old = compile(ImageStage::Denoiser, 1);
    let weak = Arc::downgrade(&old.component);
    drop(old);
    assert!(weak.upgrade().is_some());
    assert!(!compile(ImageStage::Decoder, 1).plan.plan_cache_hit);
    assert!(compile(ImageStage::Denoiser, 1).plan.plan_cache_hit);
    assert!(!compile(ImageStage::Denoiser, 2).plan.plan_cache_hit);
    assert!(weak.upgrade().is_none());
    assert!(compile(ImageStage::Decoder, 1).plan.plan_cache_hit);
}

#[test]
fn equal_tensor_shape_different_key_mask_or_profile_never_reuses_a_plan() {
    let (pipeline, _directory) = super::adapter::pipeline(1);
    let family = QwenImage21TransformerFamily::new().unwrap();
    let config = |valid| {
        QwenImage21TransformerConfig::new(
            transformer(),
            QwenImage21T2iLayout::new(3, 2, 2, vec![valid]).unwrap(),
        )
        .unwrap()
    };
    let prepare = |valid, profile| {
        let config = config(valid);
        let prepared = TypedFamilyRegistration::new(family.clone())
            .prepare_with_profile(
                &serde_json::to_value(&config).unwrap(),
                &NumericalProfileId::new(profile).unwrap(),
            )
            .unwrap();
        (prepared, config.compile_options().unwrap())
    };
    let (first_family, first_options) =
        prepare(3, QwenImage21TransformerFamily::REFERENCE_PROFILE_ID);
    let first = PreparedStage::compile(
        ImageStage::Denoiser,
        1,
        first_family,
        &first_options,
        &pipeline,
    )
    .unwrap();
    let (other_family, other_options) =
        prepare(2, QwenImage21TransformerFamily::REFERENCE_PROFILE_ID);
    assert_eq!(first_options, other_options);
    let masked = PreparedStage::compile(
        ImageStage::Denoiser,
        1,
        other_family,
        &other_options,
        &pipeline,
    )
    .unwrap();
    assert!(!masked.plan.plan_cache_hit);
    assert_ne!(
        first.plan.family_fingerprint,
        masked.plan.family_fingerprint
    );
    assert_ne!(first.plan.plan_hash, masked.plan.plan_hash);
    let (cuda_family, cuda_options) = prepare(2, QwenImage21TransformerFamily::CUDA_PROFILE_ID);
    assert_ne!(
        cuda_family.fingerprint().unwrap(),
        masked.plan.family_fingerprint
    );
    assert!(pipeline
        .plan_cache
        .get(
            ImageStage::Denoiser,
            1,
            &cuda_family.fingerprint().unwrap(),
            &cuda_options
        )
        .unwrap()
        .is_none());
    let (other_pipeline, _other_directory) = super::adapter::pipeline(1);
    assert!(other_pipeline
        .plan_cache
        .get(
            ImageStage::Denoiser,
            1,
            &masked.plan.family_fingerprint,
            &other_options
        )
        .unwrap()
        .is_none());
}
