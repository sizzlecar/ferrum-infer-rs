use super::*;

#[path = "../vnext_core_contract/mod.rs"]
mod core;

/// Both profiles are qualified only in this synthetic declaration. A second
/// operation ensures that a supported projection alone cannot accept a model.
struct TwoStageFamily(Family);

impl ModelFamilyProvider for TwoStageFamily {
    type Config = Config;

    fn family_id(&self) -> &ModelFamilyId {
        self.0.family_id()
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        self.0.external_metadata_ids()
    }
    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        self.0.validate_config_identity(raw, config)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.0.validated_external_metadata_id(raw, config)
    }
    fn parse_config(&self, raw: &Value) -> Result<Config, VNextError> {
        self.0.parse_config(raw)
    }
    fn weight_schema(&self, config: &Config) -> Result<WeightSchema, VNextError> {
        self.0.weight_schema(config)
    }
    fn specialize_weight_schema(
        &self,
        config: &Config,
        source: &WeightSchema,
        profile: &NumericalExecutionProfile,
    ) -> Result<WeightSchema, VNextError> {
        self.0.specialize_weight_schema(config, source, profile)
    }
    fn semantic_metadata(&self, config: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        self.0.semantic_metadata(config)
    }

    fn numerical_profiles(&self, config: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let mut profiles = self.0.numerical_profiles(config)?.profiles().to_vec();
        for profile in &mut profiles {
            profile
                .boundaries
                .insert(id("value.projected"), profile.activation_type()?);
            profile.operations.push(NumericalOperationContract {
                operation_id: tail_operation(profile),
                version: ContractVersion::new(1, 0),
                multiplication_type: Some(ElementType::F32),
                accumulation_type: Some(ElementType::F32),
            });
        }
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(2, 0),
            profiles,
            vec![id("fixture.f32"), id("fixture.f16")],
        )
    }

    fn semantic_program(
        &self,
        config: &Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let base = self.0.semantic_program(config, profile)?;
        let mut blocks = base.blocks().to_vec();
        blocks[0].nodes[0].outputs = vec![id("value.projected")];
        blocks[0].nodes.push(ProgramNode {
            id: id("node.tail"),
            operation_id: tail_operation(profile),
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs: vec![id("value.projected"), id("value.weight"), id("value.state")],
            outputs: vec![id("value.output")],
            attributes: BTreeMap::new(),
        });
        ModelProgram::new(
            self.family_id().clone(),
            base.inputs().to_vec(),
            blocks,
            base.states().to_vec(),
            base.weights().to_vec(),
            base.outputs().to_vec(),
        )
    }
}

fn tail_operation(profile: &NumericalExecutionProfile) -> OperationId {
    id(&format!("operation.fixture.tail.{}", profile.id))
}

fn registration() -> TypedFamilyRegistration<TwoStageFamily> {
    TypedFamilyRegistration::new(TwoStageFamily(Family {
        family_id: id("family.numerical-fixture"),
        program_calls: Arc::default(),
        corrupt_physical_identity: false,
    }))
}

fn catalog(profiles: &FamilyNumericalProfiles, missing: Option<&OperationId>) -> CapabilityCatalog {
    let device = core::catalog().device().clone();
    let mut operations = Vec::new();
    let mut providers = BTreeMap::new();
    for profile in profiles.profiles() {
        let dtype = profile.activation_type().unwrap();
        for contract in &profile.operations {
            if missing == Some(&contract.operation_id) {
                continue;
            }
            let mut operation = core::operation();
            operation.id = contract.operation_id.clone();
            operation.version = contract.version;
            operation.inputs = vec![
                core::tensor_contract(dtype, TensorAccess::Read, AliasPolicy::NoAlias),
                TensorContract::new(
                    vec![
                        DimensionConstraint::Exact(1),
                        DimensionConstraint::Exact(256),
                    ],
                    BTreeSet::from([dtype]),
                    vec![LayoutConstraint::Contiguous],
                    TensorAccess::Read,
                    AliasPolicy::NoAlias,
                )
                .unwrap(),
                core::tensor_contract(dtype, TensorAccess::Read, AliasPolicy::NoAlias),
            ];
            operation.outputs = vec![core::tensor_contract(
                dtype,
                TensorAccess::Write,
                AliasPolicy::NoAlias,
            )];
            let candidates = ["a", "z"]
                .into_iter()
                .map(|suffix| {
                    OperationProviderDescriptor::new(
                        id(&format!("provider.{}.{suffix}", operation.id)),
                        operation.id.clone(),
                        operation.fingerprint().unwrap(),
                        core::sha(if suffix == "a" { 'a' } else { 'b' }),
                        ProviderExecutionSemantics::bitwise_eager_and_replay(),
                        ContractVersion::new(1, 0),
                        device.id.clone(),
                        device.capabilities.clone(),
                        BTreeSet::from([id("weight-format.gguf")]),
                        BTreeSet::from([id("quantization.gguf.q4-k")]),
                        core::contiguous_storage_bindings(&operation),
                        format!("estimator.{}.{suffix}", operation.id),
                        ContractVersion::new(1, 0),
                        core::sha('e'),
                    )
                    .unwrap()
                })
                .collect();
            providers.insert(operation.id.clone(), candidates);
            operations.push(operation);
        }
    }
    CapabilityCatalog::new(
        device.clone(),
        operations,
        providers,
        vec![EngineProviderDescriptor::new(
            id("engine.numerical-fixture"),
            ContractVersion::new(1, 0),
            core::sha('8'),
            device.id,
            device.capabilities,
        )
        .unwrap()],
    )
    .unwrap()
}

fn compile(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    registry: &core::TestPlanningRegistry,
) -> Result<ProgramPlanCompilation, VNextError> {
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id("value.input"),
        tensor(family.numerical_profile().activation_type()?, vec![4]),
    )]))?;
    ProgramPlanCompiler::compile(
        family,
        catalog,
        &core::policy(65536),
        &registry.planning(),
        &options,
    )
}

#[test]
fn same_gguf_compiles_both_profiles_and_registration_order_preserves_selection_and_plan() {
    let registration = registration();
    let definition = registration.define(&json!({"width": 256})).unwrap();
    let catalog = catalog(definition.numerical_profiles(), None);
    let entries =
        || core::TestPlanningEntries::new(&catalog, 16, 16, core::EstimateBehavior::Correct);
    let forward = entries().build().unwrap();
    let mut reversed = entries();
    reversed.contracts.reverse();
    reversed.estimators.reverse();
    let reversed = reversed.build().unwrap();
    let mut compiled_profiles = Vec::new();
    for profile in definition
        .numerical_profiles()
        .candidates(&NumericalExecutionPolicy::Auto)
        .unwrap()
    {
        let family = registration.prepare(&definition, &profile.id).unwrap();
        let first = compile(&family, &catalog, &forward).unwrap();
        let second = compile(&family, &catalog, &reversed).unwrap();
        assert_eq!(
            first.executable().execution_plan().plan_hash(),
            second.executable().execution_plan().plan_hash()
        );
        assert_eq!(
            first.executable().execution_plan().to_json().unwrap(),
            second.executable().execution_plan().to_json().unwrap()
        );
        forward.registry.bind_plan(first.executable()).unwrap();
        reversed.registry.bind_plan(second.executable()).unwrap();
        // Registry authority is retained independently from a matching catalog.
        assert!(reversed.registry.bind_plan(first.executable()).is_err());
        compiled_profiles.push((family, first));
    }
    let (f32, f32_plan) = &compiled_profiles[0];
    let (f16, f16_plan) = &compiled_profiles[1];
    assert_eq!(
        f32.weight_schema().components,
        f16.weight_schema().components
    );
    assert_ne!(
        f32_plan.executable().execution_plan().plan_hash(),
        f16_plan.executable().execution_plan().plan_hash()
    );
    assert_ne!(
        f32.program().fingerprint().unwrap(),
        f16.program().fingerprint().unwrap()
    );
    let lane = ExecutionLane::create(Arc::new(core::PlanningTestRuntime)).unwrap();
    let f32_identity =
        OperationDispatch::compile_submission_wave_identity(f32_plan.executable(), &lane).unwrap();
    let f16_identity =
        OperationDispatch::compile_submission_wave_identity(f16_plan.executable(), &lane).unwrap();
    assert_ne!(f32_identity.fingerprint(), f16_identity.fingerprint());
    assert_eq!(f32_identity.lane_id(), f16_identity.lane_id());
}

#[test]
fn missing_tail_rejects_whole_candidate_and_explicit_request_cannot_fall_back() {
    let registration = registration();
    let definition = registration.define(&json!({"width": 256})).unwrap();
    let preferred = definition
        .numerical_profiles()
        .resolve(&id("fixture.f32"))
        .unwrap();
    let catalog = catalog(
        definition.numerical_profiles(),
        Some(&tail_operation(preferred)),
    );
    let registry =
        core::TestPlanningRegistry::new(&catalog, 16, 16, core::EstimateBehavior::Correct);
    let preferred_family = registration.prepare(&definition, &preferred.id).unwrap();
    let failure = compile(&preferred_family, &catalog, &registry)
        .err()
        .unwrap();
    assert!(failure
        .to_string()
        .contains(tail_operation(preferred).as_str()));

    let fallback = registration
        .prepare(&definition, &id("fixture.f16"))
        .unwrap();
    let compiled = compile(&fallback, &catalog, &registry).unwrap();
    let rejected = vec![NumericalProfileRejection {
        profile_id: preferred.id.clone(),
        stage: NumericalProfileRejectionStage::ProgramCompilation,
        reason: failure.to_string(),
    }];
    let resolve = |policy, rejected| {
        NumericalProfileResolution::from_static_plan(
            policy,
            &definition,
            &fallback,
            &catalog,
            &core::policy(65536),
            compiled.executable().execution_plan(),
            rejected,
        )
    };
    assert!(resolve(NumericalExecutionPolicy::Auto, rejected.clone()).is_ok());
    assert!(resolve(NumericalExecutionPolicy::Auto, vec![]).is_err());
    assert!(resolve(
        NumericalExecutionPolicy::Require(preferred.id.clone()),
        rejected
    )
    .is_err());
}
