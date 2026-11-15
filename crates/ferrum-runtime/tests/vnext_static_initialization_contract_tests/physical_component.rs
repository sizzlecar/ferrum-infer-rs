//! The original Stored source is compiled, uploaded and retained by the real
//! resource transaction. The reference backend records bytes; this test does
//! not claim NVFP4 arithmetic or GPU support.
use super::*;

struct PhysicalFamily;

fn physical_schema() -> WeightSchema {
    WeightSchema {
        format_id: id("weight-format.device-operation-composite"),
        layout_id: id("weight-layout.original-packed-source"),
        version: ContractVersion::new(1, 0),
        components: vec![WeightComponentSpec {
            id: id("weight.component.packed"),
            role: WeightComponentRole::PackedValues,
            external_names: vec!["layer._nvfp4_weight".into()],
            dimensions: vec![8, 64],
            encoding: WeightEncoding::Quantized(QuantizationSpec::nvfp4_e2m1_e4m3_128x4()),
            required: true,
        }],
        tensors: vec![WeightTensorSpec {
            id: id("weight.matrix"),
            dimensions: vec![8, 64],
            logical_element_type: ElementType::U8,
            physical_layout: PhysicalWeightLayout::PhysicalComponent(
                PhysicalWeightComponentView::new(id("weight.component.packed")),
            ),
            required: true,
        }],
    }
}

impl ModelFamilyProvider for PhysicalFamily {
    type Config = TestConfig;
    fn family_id(&self) -> &ModelFamilyId {
        TestFamily.family_id()
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        TestFamily.external_metadata_ids()
    }
    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        TestFamily.validate_config_identity(raw, config)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        TestFamily.validated_external_metadata_id(raw, config)
    }
    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        TestFamily.parse_config(raw)
    }
    fn weights(&self, _: &Self::Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(physical_schema()))
    }
    fn numerical_profiles(
        &self,
        config: &Self::Config,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        TestFamily.numerical_profiles(config)
    }
    fn semantic_metadata(
        &self,
        config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        TestFamily.semantic_metadata(config)
    }
    fn semantic_program(
        &self,
        config: &Self::Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let base = TestFamily.semantic_program(config, profile)?;
        ModelProgram::new(
            self.family_id().clone(),
            base.inputs().to_vec(),
            base.blocks().to_vec(),
            base.states().to_vec(),
            vec![WeightReference {
                weight_id: id("weight.matrix"),
                value_id: id("value.weight"),
                tensor: ProgramTensorSpec {
                    dimensions: vec![8, 64],
                    element_type: ElementType::U8,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            }],
            base.outputs().to_vec(),
        )
    }
}

fn physical_catalog() -> CapabilityCatalog {
    let base = catalog();
    let mut op = operation();
    op.inputs[1] = TensorContract::new(
        vec![
            DimensionConstraint::Exact(8),
            DimensionConstraint::Exact(64),
        ],
        BTreeSet::from([ElementType::U8]),
        vec![LayoutConstraint::Contiguous],
        TensorAccess::Read,
        AliasPolicy::NoAlias,
    )
    .unwrap()
    .with_physical_weight_component(
        WeightComponentRole::PackedValues,
        physical_schema().components[0].encoding.clone(),
    )
    .unwrap();
    let provider = OperationProviderDescriptor::new(
        id("provider.operation.device-operation"),
        op.id.clone(),
        op.fingerprint().unwrap(),
        sha('c'),
        ProviderExecutionSemantics::bitwise_eager_and_replay(),
        ContractVersion::new(1, 0),
        base.device().id.clone(),
        BTreeSet::from([id("capability.compute")]),
        BTreeSet::from([physical_schema().format_id]),
        BTreeSet::from([QuantizationSpec::nvfp4_e2m1_e4m3_128x4().format_id]),
        contiguous_storage_bindings(&op),
        "resource-estimator.device-operation",
        ContractVersion::new(1, 0),
        sha('b'),
    )
    .unwrap();
    CapabilityCatalog::new(
        base.device().clone(),
        vec![op.clone()],
        BTreeMap::from([(op.id, vec![provider])]),
        vec![EngineProviderDescriptor::new(
            id("provider.engine.device-operation"),
            ContractVersion::new(1, 0),
            sha('e'),
            base.device().id.clone(),
            BTreeSet::from([id("capability.compute")]),
        )
        .unwrap()],
    )
    .unwrap()
}

struct PackedSource;
impl WeightComponentSource for PackedSource {
    fn component<'a>(
        &'a self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        assert_eq!(component, &physical_schema().components[0]);
        WeightComponentPayload::new(
            component,
            "layer._nvfp4_weight",
            "original.safetensors",
            vec![8, 64],
            ElementType::U8,
            (0..512).map(|x| (x % 251) as u8).collect::<Vec<_>>(),
        )
    }
}

#[test]
fn physical_component_keeps_stored_source_arena_capacity_and_retirement() {
    let family = TypedFamilyRegistration::new(PhysicalFamily)
        .prepare_fixture(&json!({"width":4}))
        .unwrap();
    let catalog = physical_catalog();
    let registry = operation_registry(
        &catalog,
        Arc::new(Mutex::new(ProviderBehavior::Success)),
        Arc::new(Mutex::new(ProviderTrace::default())),
    );
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id("value.input"),
        ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();
    let compiled =
        ProgramPlanCompiler::compile(&family, &catalog, &policy(), &registry.planning(), &options)
            .unwrap();
    let plan = compiled.executable().execution_plan();
    let stored = plan.payload().execution_weights().stored().unwrap();
    assert_eq!(stored.schema(), family.stored_weight_schema().unwrap());
    assert_eq!(
        stored.source_schema_fingerprint(),
        family
            .stored_weight_schema()
            .unwrap()
            .fingerprint()
            .unwrap()
    );
    assert_eq!(
        stored.component_sources(),
        &BTreeMap::from([(
            id("weight.component.packed"),
            vec![id("weight.component.packed")]
        )])
    );
    let weights = plan
        .payload()
        .nodes()
        .iter()
        .map(|node| {
            node.values()
                .iter()
                .find(|value| value.value_id().as_str() == "value.weight")
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(weights.len(), 2);
    assert_eq!(weights[0].storage(), weights[1].storage());
    let component = &weights[0].storage().components()[0];
    assert_eq!(component.length_bytes(), 512);
    assert_eq!(component.element_type(), ElementType::U8);
    assert_eq!(
        weights[0]
            .weight()
            .unwrap()
            .physical_component_view()
            .unwrap()
            .encoding(),
        &physical_schema().components[0].encoding
    );
    let (runtime, trace) = runtime(&catalog);
    let initialized = committed_transaction(plan, runtime, "physical-view")
        .initialize_static(
            &family,
            plan,
            &PackedSource,
            StaticInitializationPolicy::new(512, 2).unwrap(),
        )
        .unwrap();
    assert_eq!(initialized.receipt().uploaded_bytes(), 512);
    assert_eq!(initialized.receipt().transformed_bytes(), 0);
    assert_eq!(
        initialized.receipt().source_files(),
        &BTreeSet::from(["original.safetensors".into()])
    );
    assert_eq!(
        trace.lock().unwrap().uploaded_payloads,
        vec![(0..512).map(|x| (x % 251) as u8).collect::<Vec<_>>()]
    );
    let resources = handoff(initialized);
    let held = resources.trusted_runtime_binding().unwrap();
    assert_eq!(
        held.evidence()
            .static_provisioning_binding()
            .unwrap()
            .plan_static_bytes(),
        plan.payload().memory().static_bytes()
    );
    // A borrowed runtime binding retains this original root; close cannot
    // retire or replace its allocation while that authority remains live.
    let resources = match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Referenced { resources, .. }) => resources,
        _ => panic!("live weight authority was retired"),
    };
    assert!(resources.trusted_runtime_binding().is_err());
    drop(held);
    close_with_expected_static_resources(
        resources,
        plan.payload().memory().static_allocations().len(),
    );
}
