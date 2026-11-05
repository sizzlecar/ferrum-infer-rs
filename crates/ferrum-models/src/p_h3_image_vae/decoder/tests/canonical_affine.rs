//! Minimal actual Stored canonical-format affine selection, using the real
//! post-quant convolution's exact 1x1 dense view and charged BF16 promotion.
use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
struct PostQuantFixture {
    id: ModelFamilyId,
}
const PROFILE: &str = "fixture.vae.canonical_post_quant.f32.v1";
impl PostQuantFixture {
    fn pieces(
        c: &PH3ImageVaeDecoderExecutionConfig,
    ) -> Result<(WeightSchema, ModelProgram), VNextError> {
        let id = ModelFamilyId::new("family.fixture.vae.canonical_post_quant")?;
        let (components, tensors, references) = outer::schema(c)?;
        let keep = |name: &str| name.starts_with("post_quant_conv.");
        let components: Vec<_> = components
            .into_iter()
            .filter(|p| keep(&p.external_names[0]))
            .collect();
        let wanted: BTreeSet<_> = components.iter().map(|p| p.id.clone()).collect();
        let tensors: Vec<_> = tensors
            .into_iter()
            .filter(|p| match &p.physical_layout {
                PhysicalWeightLayout::Dense { component_id } => wanted.contains(component_id),
                _ => false,
            })
            .collect();
        let wanted: BTreeSet<_> = tensors.iter().map(|p| p.id.clone()).collect();
        let refs = references
            .into_iter()
            .filter(|r| wanted.contains(&r.weight_id))
            .collect();
        let node = outer::prefix(c)?.remove(0);
        let output = node.outputs[0].clone();
        let inputs = BTreeMap::from([(
            ProgramValueId::new(Family::INPUT_ID)?,
            c.input_tensor_specs()?[&ProgramValueId::new(Family::INPUT_ID)?].clone(),
        )]);
        let graph = ModelProgram::new(
            id.clone(),
            inputs.keys().cloned().collect(),
            vec![ProgramBlock {
                id: "block.fixture.vae.canonical_post_quant".into(),
                nodes: vec![node],
            }],
            vec![],
            refs,
            vec![output],
        )?
        .with_input_tensor_specs(inputs)?;
        let schema = WeightSchema {
            format_id: WeightFormatId::new("weight-format.comfy.convrot.canonical.v1")?,
            layout_id: WeightLayoutId::new("weight-layout.fixture.vae.canonical_post_quant")?,
            version: ContractVersion::new(1, 0),
            components,
            tensors,
        };
        schema.validate(&id)?;
        schema.validate_program_references(&id, &graph)?;
        Ok((schema, graph))
    }
}
impl ModelFamilyProvider for PostQuantFixture {
    type Config = PH3ImageVaeDecoderExecutionConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.fixture.vae.canonical_post_quant")])
    }
    fn parse_config(&self, r: &serde_json::Value) -> Result<Self::Config, VNextError> {
        let c = serde_json::from_value(r.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(r, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        c.validate()?;
        if serde_json::to_value(c).unwrap() != *r {
            return Err(invalid("noncanonical fixture config"));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(r, c)?;
        Ok(id("metadata.fixture.vae.canonical_post_quant"))
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(Self::pieces(c)?.0))
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let graph = Self::pieces(c)?.1;
        let out = graph.outputs()[0].clone();
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id(PROFILE),
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: out.clone(),
                boundaries: BTreeMap::from([(out, ElementType::F32)]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(AFFINE_LINEAR_F32_OPERATION_ID),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: Some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![id(PROFILE)],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("fixture profile differs"));
        }
        Ok(Self::pieces(c)?.1)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
#[test]
fn actual_canonical_stored_affine_selects_reference_and_promotes_real_conv_view_once() {
    let c = config(2, 2);
    let (_dir, source) = archive(&originals(&c));
    let host = host();
    let checkpoint =
        PH3ImageVaeDecoderCheckpoint::new(&c, &source, host.clone(), limits()).unwrap();
    let backend =
        ReferenceVNextComposition::create(id("device.reference.vae_canonical_affine")).unwrap();
    let prepared = TypedFamilyRegistration::new(PostQuantFixture {
        id: id("family.fixture.vae.canonical_post_quant"),
    })
    .prepare_with_profile(&serde_json::to_value(&c).unwrap(), &id(PROFILE))
    .unwrap();
    let mut options =
        ProgramPlanCompileOptions::new(prepared.program().input_tensor_specs().clone()).unwrap();
    options
        .insert_tensor_spec(
            prepared.program().outputs()[0].clone(),
            config::tensor(vec![4, 24]),
        )
        .unwrap();
    let d = backend.runtime().descriptor();
    let compiled = CompiledTensorComponent::compile(
        prepared,
        backend.registry(),
        backend.catalog(),
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        backend.weight_materializers(),
        &options,
    )
    .unwrap();
    let mut session = compiled
        .initialize(
            backend.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(&checkpoint),
            ComponentInitializationConfig {
                request_id: id("request.vae_canonical.init"),
                run_id: id("run.vae_canonical.init"),
                transaction_id: id("transaction.vae_canonical.init"),
                static_policy: StaticInitializationPolicy::new(1 << 20, 64).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap_or_else(|f| panic!("canonical Stored init: {f:?}"));
    let values: Vec<_> = (0..4 * 24)
        .map(|i| ((i * 7) % 31) as f32 / 32. + 0.000123)
        .collect();
    let bytes = f32_bytes(values.iter().copied());
    let mut backing = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
        .unwrap();
    backing.as_mut().copy_from_slice(&bytes);
    let input = ManagedHostTensor::new(
        id(Family::INPUT_ID),
        ResolvedTensorSpec::new(
            vec![4, 24],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        backing.freeze(),
    )
    .unwrap();
    let outputs = session
        .execute(
            id("run.vae_canonical.execute"),
            id("request.vae_canonical.execute"),
            vec![input],
        )
        .unwrap_or_else(|f| panic!("canonical Stored execute: {f:?}"));
    assert_eq!(outputs.len(), 1);
    let actual: Vec<_> = outputs[0]
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let expected = affine(
        &values,
        &floats(&source, "post_quant_conv.weight"),
        &floats(&source, "post_quant_conv.bias"),
    );
    assert_eq!(actual, expected);
    assert!(actual.iter().any(|v| bf16::from_f32(*v).to_f32() != *v));
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|f| panic!("canonical Stored close: {:?}", f.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert!(host.owns(outputs[0].bytes()));
    drop(outputs);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(backend.runtime().snapshot().live_allocations, 0);
}
