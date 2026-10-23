use super::fixture::*;
use super::*;
use crate::image_generation::plan_cache::PreparedFamilyKey;

#[test]
fn prepared_family_reuse_requires_exact_geometry_identity_and_stage_role() {
    let (pipeline, _directory) = super::adapter::pipeline(2);
    let config = QwenImage21TransformerConfig::new(
        transformer(),
        QwenImage21T2iLayout::new(3, 2, 2, vec![3]).unwrap(),
    )
    .unwrap();
    let first = pipeline.prepare_denoiser(&config, 1).unwrap();
    let family = first.component.family();
    let key = PreparedFamilyKey {
        stage: ImageStage::Denoiser,
        role: ImageStagePlanRole::WholeStage,
        batch_size: 1,
        family: family.family_id(),
        metadata: family.external_metadata_id(),
        config: family.canonical_config(),
        profile: &family.numerical_profile().id,
    };
    let lookup = |key| {
        pipeline
            .plan_cache
            .prepared_family(pipeline.config.plan_cache, key)
            .unwrap()
    };
    assert_eq!(lookup(key).as_ref(), Some(family));
    let other_family = ModelFamilyId::new("family.different").unwrap();
    let other_metadata = ExternalModelMetadataId::new("metadata.different").unwrap();
    let other_profile =
        NumericalProfileId::new(QwenImage21TransformerFamily::CUDA_PROFILE_ID).unwrap();
    let different_mask = serde_json::to_value(
        QwenImage21TransformerConfig::new(
            transformer(),
            QwenImage21T2iLayout::new(3, 2, 2, vec![2]).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    for changed in [
        PreparedFamilyKey {
            stage: ImageStage::Decoder,
            ..key
        },
        PreparedFamilyKey {
            role: ImageStagePlanRole::PrefixDecode,
            ..key
        },
        PreparedFamilyKey {
            batch_size: 2,
            ..key
        },
        PreparedFamilyKey {
            family: &other_family,
            ..key
        },
        PreparedFamilyKey {
            metadata: &other_metadata,
            ..key
        },
        PreparedFamilyKey {
            profile: &other_profile,
            ..key
        },
        PreparedFamilyKey {
            config: &different_mask,
            ..key
        },
    ] {
        assert!(lookup(changed).is_none());
    }
    let repeated = pipeline.prepare_denoiser(&config, 1).unwrap();
    assert!(repeated.plan.plan_cache_hit);
    assert!(Arc::ptr_eq(&first.component, &repeated.component));
    let receipt = repeated.plan.plan_cache.as_ref().unwrap();
    assert_eq!((receipt.entries, receipt.hits, receipt.misses), (1, 1, 1));
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
}

#[test]
fn prepared_family_reuse_still_checks_new_compile_options() {
    use ferrum_models::qwen_image21::{
        QwenImage21ImageProjectionFamily, IMAGE_PROJECTION_PROFILE_ID,
    };
    let (pipeline, _directory) = super::adapter::pipeline(1);
    let original = cached_projection(&pipeline, ImageStage::Denoiser, 1);
    let family = pipeline
        .prepare_family(
            ImageStage::Denoiser,
            ImageStagePlanRole::WholeStage,
            1,
            QwenImage21ImageProjectionFamily::new().unwrap(),
            &transformer(),
            &NumericalProfileId::new(IMAGE_PROJECTION_PROFILE_ID).unwrap(),
        )
        .unwrap();
    assert_eq!(&family, original.component.family());
    let mut options = projection_options();
    options.retain_completion_value(family.program().outputs()[0].clone());
    let changed =
        PreparedStage::compile(ImageStage::Denoiser, 1, family, &options, &pipeline).unwrap();
    assert!(!changed.plan.plan_cache_hit);
    assert!(!Arc::ptr_eq(&original.component, &changed.component));
}

#[test]
fn prepared_family_lookup_does_not_extend_lru_or_disabled_retention() {
    let (mut pipeline, _directory) = super::adapter::pipeline(2);
    pipeline.config.plan_cache = multi_shape(2, 16 * 1024 * 1024);
    let first = cached_projection(&pipeline, ImageStage::Denoiser, 1);
    let family = first.component.family().clone();
    let weak = Arc::downgrade(&first.component);
    drop(first);
    let key = PreparedFamilyKey {
        stage: ImageStage::Denoiser,
        role: ImageStagePlanRole::WholeStage,
        batch_size: 1,
        family: family.family_id(),
        metadata: family.external_metadata_id(),
        config: family.canonical_config(),
        profile: &family.numerical_profile().id,
    };
    let second = cached_projection(&pipeline, ImageStage::Denoiser, 2);
    let second_weak = Arc::downgrade(&second.component);
    drop(second);
    assert!(pipeline
        .plan_cache
        .prepared_family(pipeline.config.plan_cache, key)
        .unwrap()
        .is_some());
    // A metadata-only lookup does not acquire a plan or refresh its LRU slot.
    drop(cached_projection_with_role(
        &pipeline,
        ImageStage::Denoiser,
        ImageStagePlanRole::PrefixDecode,
        1,
    ));
    assert!(weak.upgrade().is_none());
    assert!(pipeline
        .plan_cache
        .prepared_family(pipeline.config.plan_cache, key)
        .unwrap()
        .is_none());
    let second_key = PreparedFamilyKey {
        batch_size: 2,
        ..key
    };
    assert!(pipeline
        .plan_cache
        .prepared_family(pipeline.config.plan_cache, second_key)
        .unwrap()
        .is_some());
    assert!(pipeline
        .plan_cache
        .prepared_family(ImagePlanCachePolicy::Disabled, second_key)
        .unwrap()
        .is_none());
    assert!(second_weak.upgrade().is_none());
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
}

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
            pipeline.config.plan_cache,
            ImageStage::Denoiser,
            ImageStagePlanRole::WholeStage,
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
            other_pipeline.config.plan_cache,
            ImageStage::Denoiser,
            ImageStagePlanRole::WholeStage,
            1,
            &masked.plan.family_fingerprint,
            &other_options
        )
        .unwrap()
        .is_none());
}

fn multi_shape(entries: usize, bytes: usize) -> ImagePlanCachePolicy {
    ImagePlanCachePolicy::MultiShape {
        maximum_entries_per_stage: entries,
        maximum_encoded_metadata_bytes: bytes,
    }
}

fn cached_projection(
    pipeline: &QwenImage21Pipeline<ferrum_kernels::backend::reference::ReferenceDeviceRuntime>,
    stage: ImageStage,
    batch: usize,
) -> PreparedStage<ferrum_kernels::backend::reference::ReferenceDeviceRuntime> {
    cached_projection_with_role(pipeline, stage, ImageStagePlanRole::WholeStage, batch)
}

fn cached_projection_with_role(
    pipeline: &QwenImage21Pipeline<ferrum_kernels::backend::reference::ReferenceDeviceRuntime>,
    stage: ImageStage,
    role: ImageStagePlanRole,
    batch: usize,
) -> PreparedStage<ferrum_kernels::backend::reference::ReferenceDeviceRuntime> {
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id(ferrum_models::qwen_image21::IMAGE_PROJECTION_INPUT_ID).unwrap(),
        ProgramTensorSpec {
            dimensions: vec![batch as u64, 64],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();
    PreparedStage::compile_with_role(stage, role, batch, projection(), &options, pipeline).unwrap()
}

#[test]
fn cooperating_roles_hit_on_later_rounds_and_replace_only_their_own_last_shape() {
    let (pipeline, _directory) = super::adapter::pipeline(2);
    let compile =
        |role, batch| cached_projection_with_role(&pipeline, ImageStage::Denoiser, role, batch);
    let prefill = compile(ImageStagePlanRole::PrefixPrefill, 1);
    let decode = compile(ImageStagePlanRole::PrefixDecode, 1);
    let whole = compile(ImageStagePlanRole::WholeStage, 1);
    assert!(!prefill.plan.plan_cache_hit && !decode.plan.plan_cache_hit);
    // Roles separate cooperative slots even if this injected component has the
    // same family/shape. Real plans still match their full family and options.
    assert!(!Arc::ptr_eq(&prefill.component, &decode.component));
    for _ in 0..2 {
        let p = compile(ImageStagePlanRole::PrefixPrefill, 1);
        let d = compile(ImageStagePlanRole::PrefixDecode, 1);
        assert!(p.plan.plan_cache_hit && d.plan.plan_cache_hit);
        assert!(Arc::ptr_eq(&prefill.component, &p.component));
        assert!(Arc::ptr_eq(&decode.component, &d.component));
    }
    let prefill_weak = Arc::downgrade(&prefill.component);
    drop(prefill);
    assert!(
        !compile(ImageStagePlanRole::PrefixPrefill, 2)
            .plan
            .plan_cache_hit
    );
    assert!(prefill_weak.upgrade().is_none());
    assert!(
        compile(ImageStagePlanRole::PrefixDecode, 1)
            .plan
            .plan_cache_hit
    );
    assert!(
        compile(ImageStagePlanRole::WholeStage, 1)
            .plan
            .plan_cache_hit
    );
    let decode_weak = Arc::downgrade(&decode.component);
    drop(decode);
    assert!(
        !compile(ImageStagePlanRole::PrefixDecode, 2)
            .plan
            .plan_cache_hit
    );
    assert!(decode_weak.upgrade().is_none());
    assert!(
        compile(ImageStagePlanRole::PrefixPrefill, 2)
            .plan
            .plan_cache_hit
    );
    assert!(
        compile(ImageStagePlanRole::WholeStage, 1)
            .plan
            .plan_cache_hit
    );
    assert_eq!(whole.plan.role, ImageStagePlanRole::WholeStage);
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
}

#[test]
fn multi_shape_stage_limit_counts_all_roles_together() {
    let (mut pipeline, _directory) = super::adapter::pipeline(2);
    pipeline.config.plan_cache = multi_shape(2, 16 * 1024 * 1024);
    let compile =
        |role, batch| cached_projection_with_role(&pipeline, ImageStage::Denoiser, role, batch);
    let old = compile(ImageStagePlanRole::PrefixPrefill, 1);
    let weak = Arc::downgrade(&old.component);
    drop(old);
    assert!(
        !compile(ImageStagePlanRole::PrefixDecode, 1)
            .plan
            .plan_cache_hit
    );
    let newest = compile(ImageStagePlanRole::WholeStage, 1);
    assert_eq!(newest.plan.plan_cache.as_ref().unwrap().stage_entries, 2);
    assert!(weak.upgrade().is_none());
    assert!(
        compile(ImageStagePlanRole::PrefixDecode, 1)
            .plan
            .plan_cache_hit
    );
    assert!(
        compile(ImageStagePlanRole::WholeStage, 1)
            .plan
            .plan_cache_hit
    );
    assert_eq!(
        serde_json::to_value(ImageStagePlanRole::PrefixDecode).unwrap(),
        "prefix_decode"
    );
    assert!(serde_json::from_str::<ImageStagePlanRole>("\"unknown_phase\"").is_err());
}

#[test]
fn multi_shape_revisits_hit_and_stage_limit_evicts_least_recent_entry() {
    let (mut pipeline, _directory) = super::adapter::pipeline(2);
    pipeline.config.plan_cache = multi_shape(2, 16 * 1024 * 1024);
    let a = cached_projection(&pipeline, ImageStage::Denoiser, 1);
    let a_weak = Arc::downgrade(&a.component);
    let b = cached_projection(&pipeline, ImageStage::Denoiser, 2);
    let b_weak = Arc::downgrade(&b.component);
    let bytes_before_hit = b.plan.plan_cache.as_ref().unwrap().encoded_metadata_bytes;
    let hit = cached_projection(&pipeline, ImageStage::Denoiser, 1);
    assert!(hit.plan.plan_cache_hit);
    assert!(Arc::ptr_eq(&a.component, &hit.component));
    let receipt = hit.plan.plan_cache.as_ref().unwrap();
    assert_eq!(receipt.encoded_metadata_bytes, bytes_before_hit);
    assert_eq!((receipt.hits, receipt.misses, receipt.evictions), (1, 2, 0));
    drop((a, b, hit));
    let c = cached_projection(&pipeline, ImageStage::Denoiser, 3);
    assert_eq!(c.plan.plan_cache.as_ref().unwrap().stage_entries, 2);
    assert_eq!(c.plan.plan_cache.as_ref().unwrap().evictions, 1);
    assert!(b_weak.upgrade().is_none());
    assert!(a_weak.upgrade().is_some());
    let other = cached_projection(&pipeline, ImageStage::Decoder, 1);
    assert_eq!(other.plan.plan_cache.as_ref().unwrap().entries, 3);
    assert_eq!(other.plan.plan_cache.as_ref().unwrap().stage_entries, 1);
    assert!(
        cached_projection(&pipeline, ImageStage::Denoiser, 1)
            .plan
            .plan_cache_hit
    );
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn global_encoded_budget_evicts_lru_across_stages_without_recounting_hits() {
    // Derive the byte limit from these real serialized plans, not assumptions
    // about JSON formatting or a particular provider's metadata size.
    let (mut measurement, _directory) = super::adapter::pipeline(2);
    measurement.config.plan_cache = multi_shape(4, 16 * 1024 * 1024);
    let mut previous = 0;
    let mut sizes = Vec::new();
    for stage in [ImageStage::Denoiser, ImageStage::Decoder, ImageStage::Noise] {
        let plan = cached_projection(&measurement, stage, 1);
        let total = plan
            .plan
            .plan_cache
            .as_ref()
            .unwrap()
            .encoded_metadata_bytes
            .unwrap();
        sizes.push(total - previous);
        previous = total;
    }
    sizes.sort_unstable();
    let limit = sizes[1] + sizes[2];
    let mut pipeline = measurement;
    pipeline.config.plan_cache = multi_shape(4, limit);
    let a = cached_projection(&pipeline, ImageStage::Denoiser, 1);
    let b = cached_projection(&pipeline, ImageStage::Decoder, 1);
    let b_weak = Arc::downgrade(&b.component);
    assert!(
        cached_projection(&pipeline, ImageStage::Denoiser, 1)
            .plan
            .plan_cache_hit
    );
    drop(b);
    let c = cached_projection(&pipeline, ImageStage::Noise, 1);
    let receipt = c.plan.plan_cache.as_ref().unwrap();
    assert_eq!(receipt.entries, 2);
    assert_eq!(receipt.evictions, 1);
    assert!(receipt.encoded_metadata_bytes.unwrap() <= limit);
    assert!(b_weak.upgrade().is_none());
    assert!(
        cached_projection(&pipeline, ImageStage::Denoiser, 1)
            .plan
            .plan_cache_hit
    );
    assert!(
        cached_projection(&pipeline, ImageStage::Noise, 1)
            .plan
            .plan_cache_hit
    );
    assert!(Arc::strong_count(&a.component) > 1);
}

#[test]
fn oversized_entry_is_returned_without_retention_and_does_not_displace_smaller_entry() {
    let (mut pipeline, _directory) = super::adapter::pipeline(2);
    pipeline.config.plan_cache = multi_shape(2, 16 * 1024 * 1024);
    let small = cached_projection(&pipeline, ImageStage::Denoiser, 1);
    let size = small
        .plan
        .plan_cache
        .as_ref()
        .unwrap()
        .encoded_metadata_bytes
        .unwrap();
    // Reconfigure explicitly before acquiring the entry under the smaller cap.
    pipeline.config.plan_cache = multi_shape(2, size);
    let small = cached_projection(&pipeline, ImageStage::Denoiser, 1);
    let small_weak = Arc::downgrade(&small.component);
    drop(small);
    let config = QwenImage21TransformerConfig::new(
        transformer(),
        QwenImage21T2iLayout::new(3, 2, 2, vec![3]).unwrap(),
    )
    .unwrap();
    let family = TypedFamilyRegistration::new(QwenImage21TransformerFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&config).unwrap(),
            &NumericalProfileId::new(QwenImage21TransformerFamily::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let oversized = PreparedStage::compile(
        ImageStage::Denoiser,
        1,
        family,
        &config.compile_options().unwrap(),
        &pipeline,
    )
    .unwrap();
    let receipt = oversized.plan.plan_cache.as_ref().unwrap();
    assert!(!receipt.retained);
    assert_eq!(
        (
            receipt.oversized_entries_skipped,
            receipt.evictions,
            receipt.entries
        ),
        (1, 0, 1)
    );
    let weak = Arc::downgrade(&oversized.component);
    assert!(small_weak.upgrade().is_some());
    assert!(
        cached_projection(&pipeline, ImageStage::Denoiser, 1)
            .plan
            .plan_cache_hit
    );
    drop(oversized);
    assert!(weak.upgrade().is_none());
}

#[test]
fn cache_policy_wire_preserves_strings_and_requires_both_explicit_bounds() {
    use serde_json::json;
    assert_eq!(
        ImagePlanCachePolicy::default(),
        ImagePlanCachePolicy::LastPerStage
    );
    for (policy, wire) in [
        (ImagePlanCachePolicy::Disabled, json!("disabled")),
        (ImagePlanCachePolicy::LastPerStage, json!("last_per_stage")),
        (
            multi_shape(2, 4096),
            json!({"multi_shape": {
                "maximum_entries_per_stage":2, "maximum_encoded_metadata_bytes":4096
            }}),
        ),
    ] {
        assert_eq!(serde_json::to_value(policy).unwrap(), wire);
        assert_eq!(
            serde_json::from_value::<ImagePlanCachePolicy>(wire).unwrap(),
            policy
        );
    }
    for invalid in [
        json!("multi_shape"),
        json!({"multi_shape":{}}),
        json!({"multi_shape":{"maximum_entries_per_stage":2}}),
        json!({"multi_shape":{"maximum_encoded_metadata_bytes":4096}}),
        json!({"multi_shape":{"maximum_entries_per_stage":0,"maximum_encoded_metadata_bytes":4096}}),
        json!({"multi_shape":{"maximum_entries_per_stage":2,"maximum_encoded_metadata_bytes":0}}),
        json!({"multi_shape":{"maximum_entries_per_stage":2,"maximum_encoded_metadata_bytes":4096,"unknown":1}}),
        json!({"multi_shape":{"maximum_entries_per_stage":usize::MAX,"maximum_encoded_metadata_bytes":4096}}),
        json!({"multi_shape":{"maximum_entries_per_stage":2,"maximum_encoded_metadata_bytes":usize::MAX}}),
    ] {
        assert!(serde_json::from_value::<ImagePlanCachePolicy>(invalid).is_err());
    }
    assert!(serde_json::from_str::<ImagePlanCachePolicy>(r#"{"multi_shape":{"maximum_entries_per_stage":2,"maximum_entries_per_stage":3,"maximum_encoded_metadata_bytes":4096}}"#).is_err());
    assert!(multi_shape(0, 4096).validate().is_err());
}

#[test]
fn evicted_component_still_initializes_executes_and_retires_with_output_alive() {
    use super::residency::{input, numeric_pipeline, prepared};
    use ferrum_models::qwen_image21::IMAGE_PROJECTION_OUTPUT_ID;
    let (mut pipeline, _directory) = numeric_pipeline();
    pipeline.config.residency = ImageResidencyPolicy::Cold;
    pipeline.config.plan_cache = multi_shape(1, 16 * 1024 * 1024);
    let a = prepared(&pipeline, 1);
    let weak = Arc::downgrade(&a.component);
    let b = prepared(&pipeline, 2);
    assert_eq!(b.plan.plan_cache.as_ref().unwrap().evictions, 1);
    assert!(weak.upgrade().is_some()); // A is held solely by this request.
    let mut stages = Vec::new();
    let active = pipeline
        .start_stage(a, 0, &mut stages)
        .unwrap_or_else(|_| panic!("evicted component initialization"));
    // Session owns executable/provider authority independently of cached Arc.
    assert!(weak.upgrade().is_none());
    let (active, outputs) = active
        .execute(0, vec![input(&pipeline.host, 1, 0)])
        .unwrap_or_else(|_| panic!("evicted component execution"));
    let output = only_output(outputs, IMAGE_PROJECTION_OUTPUT_ID).unwrap();
    pipeline
        .finish_stage(active, &mut stages)
        .unwrap_or_else(|_| panic!("evicted component retirement"));
    assert!(matches!(stages[0].retirement, ImageStageRetirement::Closed));
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 16);
    for (index, part) in output.bytes().bytes().chunks_exact(2).enumerate() {
        assert_eq!(
            u16::from_le_bytes([part[0], part[1]]),
            half::bf16::from_f32((index + 1) as f32).to_bits()
        );
    }
    drop(output);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
    assert!(pipeline.shutdown().unwrap().is_empty());
}
