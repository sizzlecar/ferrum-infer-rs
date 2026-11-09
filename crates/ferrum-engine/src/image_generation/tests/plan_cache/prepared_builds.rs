//! Exercise the actual shared cache with the original Stored Reference
//! projection provider. H3's CUDA-only full provider is checked separately at
//! its metadata boundary; this test does not claim a CPU H3 forward.
use super::*;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountBuilds<P> {
    inner: P,
    weights: Arc<AtomicUsize>,
}
impl<P: ModelFamilyProvider> ModelFamilyProvider for CountBuilds<P> {
    type Config = P::Config;
    fn family_id(&self) -> &ModelFamilyId {
        self.inner.family_id()
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        self.inner.external_metadata_ids()
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        self.inner.validate_config_identity(raw, config)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.inner.validated_external_metadata_id(raw, config)
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Self::Config, VNextError> {
        self.inner.parse_config(raw)
    }
    fn weights(&self, config: &Self::Config) -> Result<ModelWeights, VNextError> {
        self.weights.fetch_add(1, Ordering::SeqCst);
        self.inner.weights(config)
    }
    fn numerical_profiles(
        &self,
        config: &Self::Config,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        self.inner.numerical_profiles(config)
    }
    fn specialize_weight_schema(
        &self,
        config: &Self::Config,
        schema: &WeightSchema,
        profile: &NumericalExecutionProfile,
    ) -> Result<WeightSchema, VNextError> {
        self.inner.specialize_weight_schema(config, schema, profile)
    }
    fn semantic_program(
        &self,
        config: &Self::Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        self.inner.semantic_program(config, profile)
    }
    fn semantic_metadata(
        &self,
        config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        self.inner.semantic_metadata(config)
    }
}

#[test]
fn prepared_family_cache_hit_skips_actual_provider_weight_build_and_keeps_shape_partition() {
    use ferrum_models::qwen_image21::{
        QwenImage21ImageProjectionFamily, IMAGE_PROJECTION_INPUT_ID, IMAGE_PROJECTION_PROFILE_ID,
    };
    let (mut pipeline, _directory) = super::super::adapter::pipeline(2);
    pipeline.execution.config.plan_cache = multi_shape(4, 16 * 1024 * 1024);
    let builds = Arc::new(AtomicUsize::new(0));
    let prepare = |batch| {
        pipeline
            .execution
            .prepare_family(
                ImageStage::Denoiser,
                ImageStagePlanRole::WholeStage,
                batch,
                CountBuilds {
                    inner: QwenImage21ImageProjectionFamily::new().unwrap(),
                    weights: builds.clone(),
                },
                &transformer(),
                &NumericalProfileId::new(IMAGE_PROJECTION_PROFILE_ID).unwrap(),
            )
            .unwrap()
    };
    let first_family = prepare(1);
    let cold_builds = builds.load(Ordering::SeqCst);
    assert!(cold_builds > 0);
    let first = PreparedStage::compile(
        ImageStage::Denoiser,
        1,
        first_family.clone(),
        &projection_options(),
        &pipeline,
    )
    .unwrap();
    let warm_family = prepare(1);
    assert_eq!(builds.load(Ordering::SeqCst), cold_builds);
    assert_eq!(warm_family, first_family);
    let warm = PreparedStage::compile(
        ImageStage::Denoiser,
        1,
        warm_family,
        &projection_options(),
        &pipeline,
    )
    .unwrap();
    assert!(warm.plan.plan_cache_hit);
    assert!(Arc::ptr_eq(&first.component, &warm.component));
    let other_family = prepare(2);
    assert!(builds.load(Ordering::SeqCst) > cold_builds);
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id(IMAGE_PROJECTION_INPUT_ID).unwrap(),
        ProgramTensorSpec {
            dimensions: vec![2, 64],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();
    let other =
        PreparedStage::compile(ImageStage::Denoiser, 2, other_family, &options, &pipeline).unwrap();
    assert!(!other.plan.plan_cache_hit);
    assert!(!Arc::ptr_eq(&first.component, &other.component));
    assert_eq!(
        pipeline.execution.host.usage().unwrap().committed_bytes(),
        0
    );
    assert_eq!(pipeline.execution.runtime.snapshot().live_allocations, 0);
}
